from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .data import SpeechTokenCorpus
from .model import ModelArgs, TransformerLM


@dataclass
class TrainConfig:
    train_data: str = "data/train"
    val_data: str = "data/val"
    output_dir: str = "out"
    resume: str | None = None
    seed: int = 1337
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    eval_interval: int = 100
    eval_steps: int = 10
    log_interval: int = 10
    save_interval: int = 100
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bfloat16"
    compile: bool = True
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None
    vocab_size: int = 501
    hidden_dim: int | None = None
    multiple_of: int = 256
    block_size: int = 2050
    dropout: float = 0.0

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] = ()) -> "TrainConfig":
        values = json.loads(Path(path).read_text())
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        for override in overrides:
            key, raw = override.split("=", 1)
            if key not in allowed:
                raise ValueError(f"unknown override: {key}")
            values[key] = json.loads(raw)
        return cls(**values)

    def model_args(self) -> ModelArgs:
        return ModelArgs(
            dim=self.dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            multiple_of=self.multiple_of,
            block_size=self.block_size,
            dropout=self.dropout,
        )


def _distributed():
    rank = int(os.environ.get("RANK", "-1"))
    if rank < 0:
        return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return True, rank, local_rank, world_size, device


def _unwrap(model):
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    step: int,
    config: TrainConfig,
    dataset_fingerprints: dict[str, str],
    rng_by_rank: list[dict[str, Any]],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "model_args": asdict(config.model_args()),
        "train_config": asdict(config),
        "batch_provenance": _batch_provenance(config),
        "dataset_fingerprints": dataset_fingerprints,
        "rng_by_rank": rng_by_rank,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _batch_provenance(config: TrainConfig) -> dict[str, int]:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return {
        "micro_batch_sequences_per_rank": config.batch_size,
        "micro_steps_per_rank": config.gradient_accumulation_steps // world_size,
        "world_size": world_size,
        "global_sequences_per_update": config.batch_size * config.gradient_accumulation_steps,
        "context_tokens": config.block_size,
        "tokens_per_update": (
            config.batch_size * config.gradient_accumulation_steps * config.block_size
        ),
    }


def _learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    ratio = min(
        1.0,
        (step - config.warmup_steps)
        / max(1, config.max_steps - config.warmup_steps - 1),
    )
    cosine = 0.5 * (1 + math.cos(math.pi * ratio))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def _make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _load_checkpoint(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


_EXACT_RESUME_FIELDS = (
    "seed",
    "batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "eval_interval",
    "eval_steps",
    "learning_rate",
    "min_learning_rate",
    "warmup_steps",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
    "precision",
    "compile",
)


def _validate_resume_checkpoint(checkpoint: dict[str, Any], config: TrainConfig) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported or missing checkpoint format_version")
    expected_model = asdict(config.model_args())
    if checkpoint.get("model_args") != expected_model:
        raise ValueError("resume model configuration does not match")
    previous = checkpoint.get("train_config")
    if not isinstance(previous, dict):
        raise ValueError("resume checkpoint is missing its training configuration")
    mismatches = {
        name: (previous.get(name), getattr(config, name))
        for name in _EXACT_RESUME_FIELDS
        if previous.get(name) != getattr(config, name)
    }
    if mismatches:
        raise ValueError(f"resume training configuration differs: {mismatches}")


def _collect_rng(train_rng, val_rng, rank: int, world_size: int):
    local = {
        "global": _rng_state(),
        "train": train_rng.bit_generator.state,
        "val": val_rng.bit_generator.state,
    }
    if not dist.is_initialized():
        return [local]
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return gathered


@torch.inference_mode()
def _evaluate(model, corpus, config, device, rng, context, world_size):
    model.eval()
    total = torch.zeros(2, device=device)
    for _ in range(config.eval_steps):
        x, y = corpus.get_batch(config.batch_size, rng, device)
        with context:
            _unwrap(model).last_loss = None
            model(x, y)
            loss = _unwrap(model).last_loss
        total += torch.stack((loss.detach().float(), torch.ones((), device=device)))
    if dist.is_initialized():
        dist.all_reduce(total)
    model.train()
    return (total[0] / total[1]).item()


def train(config: TrainConfig) -> dict[str, float]:
    is_distributed, rank, local_rank, world_size, device = _distributed()
    master = rank == 0
    if config.gradient_accumulation_steps % world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by world size")
    accumulation = config.gradient_accumulation_steps // world_size
    seed = config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_corpus = SpeechTokenCorpus(config.train_data, config.block_size)
    val_corpus = SpeechTokenCorpus(config.val_data, config.block_size)
    data_spec = (train_corpus.vocab_size, train_corpus.eos_token, train_corpus.pad_token)
    val_spec = (val_corpus.vocab_size, val_corpus.eos_token, val_corpus.pad_token)
    if data_spec != val_spec:
        raise ValueError("training and validation speech vocabularies differ")
    if config.vocab_size != train_corpus.vocab_size:
        raise ValueError(
            f"model vocab_size={config.vocab_size} does not match data "
            f"vocab_size={train_corpus.vocab_size}"
        )
    train_rng = np.random.default_rng(seed)
    val_rng = np.random.default_rng(config.seed + 100_000 + rank)
    dataset_fingerprints = {
        "train": train_corpus.fingerprint(),
        "validation": val_corpus.fingerprint(),
    }
    model = TransformerLM(config.model_args()).to(device)
    optimizer = model.configure_optimizers(
        config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device.type
    )
    scaler = _make_scaler(device.type == "cuda" and config.precision == "float16")
    start_step = 0
    if config.resume:
        checkpoint = _load_checkpoint(config.resume)
        _validate_resume_checkpoint(checkpoint, config)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_step = int(checkpoint["step"])
        if int(checkpoint.get("world_size", 1)) != world_size:
            raise ValueError("exact resume requires the original world size")
        if checkpoint.get("dataset_fingerprints") != dataset_fingerprints:
            raise ValueError("resume dataset fingerprint does not match")
        rank_rng = checkpoint["rng_by_rank"][rank]
        _restore_rng(rank_rng["global"])
        train_rng.bit_generator.state = rank_rng["train"]
        val_rng.bit_generator.state = rank_rng["val"]

    if config.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile=True requires PyTorch 2 or newer")
        model = torch.compile(model)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.precision]
    context = nullcontext() if device.type == "cpu" or dtype == torch.float32 else torch.autocast(device.type, dtype=dtype)
    output = Path(config.output_dir)
    if master:
        output.mkdir(parents=True, exist_ok=True)
        runtime_config = asdict(config)
        runtime_config["batch_provenance"] = _batch_provenance(config)
        (output / "config.json").write_text(json.dumps(runtime_config, indent=2) + "\n")

    wandb_run = None
    if master and config.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={**asdict(config), "batch_provenance": _batch_provenance(config)},
        )

    last_loss = float("nan")
    started = time.perf_counter()
    try:
        model.train()
        for step in range(start_step, config.max_steps):
            lr = _learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            for micro_step in range(accumulation):
                if is_distributed:
                    model.require_backward_grad_sync = micro_step == accumulation - 1
                x, y = train_corpus.get_batch(config.batch_size, train_rng, device)
                with context:
                    model(x, y)
                    loss = _unwrap(model).last_loss / accumulation
                scaler.scale(loss).backward()
                last_loss = loss.item() * accumulation
            if config.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_started
            completed = step + 1
            if master and (completed % config.log_interval == 0 or completed == 1):
                raw = _unwrap(model)
                mfu = raw.estimate_mfu(config.batch_size * accumulation, elapsed)
                metrics = {"step": completed, "loss/train": last_loss, "lr": lr, "mfu/raw": mfu, "step_time": elapsed}
                print(json.dumps(metrics))
                if wandb_run:
                    wandb_run.log(metrics, step=completed)
            if completed % config.eval_interval == 0 or completed == config.max_steps:
                val_loss = _evaluate(model, val_corpus, config, device, val_rng, context, world_size)
                if master:
                    metrics = {"step": completed, "loss/val": val_loss}
                    print(json.dumps(metrics))
                    if wandb_run:
                        wandb_run.log(metrics, step=completed)
            if completed % config.save_interval == 0 or completed == config.max_steps:
                rng_by_rank = _collect_rng(train_rng, val_rng, rank, world_size)
                if master:
                    _save_checkpoint(
                        output / f"checkpoint-{completed:08d}.pt",
                        model,
                        optimizer,
                        scaler,
                        completed,
                        config,
                        dataset_fingerprints,
                        rng_by_rank,
                    )
                if is_distributed:
                    dist.barrier()
    finally:
        if wandb_run:
            wandb_run.finish()
        if is_distributed:
            dist.destroy_process_group()
    return {"loss": last_loss, "elapsed": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the speech Llama with optional DDP")
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*", help="JSON-valued key=value overrides")
    args = parser.parse_args(argv)
    train(TrainConfig.load(args.config, args.overrides))


if __name__ == "__main__":
    main()
