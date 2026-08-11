from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from slm_data import PackedShard

from .checkpoint import load_training_checkpoint
from .model import TransformerLM


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hashes(path: Path) -> dict[str, str]:
    if path.is_file():
        return {path.name: _sha256(path)}
    files = [path / "config.json", *sorted(path.glob("model*.safetensors"))]
    index = path / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
    return {file.name: _sha256(file) for file in files if file.exists()}


def _metadata(shard: PackedShard) -> list[dict]:
    with (shard.path / "samples.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def _evaluation_pair(sequence: torch.Tensor, eos: int, include_terminal_eos: bool):
    """Build the paper-compatible shifted input and target for one utterance."""
    target = sequence
    if not include_terminal_eos and len(target) and int(target[-1]) == eos:
        target = target[:-1]
    if not len(target):
        raise ValueError("evaluation sequence contains no predictable units")
    inputs = torch.cat((torch.tensor([eos], dtype=target.dtype), target[:-1]))
    return inputs, target


@torch.inference_mode()
def score_shard(
    model,
    shard: PackedShard,
    batch_size: int,
    device: torch.device,
    precision: str,
    *,
    include_terminal_eos: bool,
):
    rows = _metadata(shard)
    scores = []
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
    context = nullcontext() if device.type == "cpu" or dtype == torch.float32 else torch.autocast(device.type, dtype=dtype)
    eos = int(shard.meta["preprocessing"].get("eos_token", model.vocab_size - 1))
    pad = int(shard.meta["preprocessing"].get("pad_token", eos))
    for start in range(0, len(shard), batch_size):
        sequences = [torch.from_numpy(shard.sample(i)["audio"].astype(np.int64)) for i in range(start, min(len(shard), start + batch_size))]
        if any(len(sequence) > model.params.block_size for sequence in sequences):
            raise ValueError("evaluation sequence exceeds model block_size")
        pairs = [
            _evaluation_pair(sequence, eos, include_terminal_eos)
            for sequence in sequences
        ]
        inputs, targets = zip(*pairs)
        x = pad_sequence(inputs, batch_first=True, padding_value=pad).to(device)
        y = pad_sequence(targets, batch_first=True, padding_value=-1).to(device)
        mask = y != -1
        with context:
            model(x, y, mask=mask)
            nll = model.last_loss
        for row, value in zip(rows[start:start + len(sequences)], nll):
            scores.append({**row, "nll": float(value)})
    return scores


def summarize(task: str, scores: list[dict]) -> dict:
    result = {"mean_nll": float(np.mean([row["nll"] for row in scores])), "num_examples": len(scores)}
    if task == "loss":
        return result
    grouped: dict[str, list[dict]] = {}
    for row in scores:
        group = row.get("group_id")
        if group is None:
            sample_id = row["id"]
            if sample_id.endswith("_correct") or sample_id.endswith("_incorrect"):
                group = sample_id.rsplit("_", 1)[0]
        if group is None:
            raise ValueError("multiple-choice evaluation requires group_id metadata")
        grouped.setdefault(str(group), []).append(row)
    decisions: list[tuple[dict, float]] = []
    for group_id, options in grouped.items():
        positives = [
            row
            for row in options
            if bool(row.get("correct", row["id"].endswith("_correct")))
        ]
        negatives = [row for row in options if row not in positives]
        if len(positives) != 1 or not negatives:
            raise ValueError(
                f"group {group_id} must contain exactly one correct and at least one incorrect option"
            )
        best_negative = min(row["nll"] for row in negatives)
        if positives[0]["nll"] < best_negative:
            decision = 1.0
        elif task == "sblimp" and positives[0]["nll"] == best_negative:
            decision = 0.5
        else:
            decision = 0.0
        decisions.append((positives[0], decision))
    result["num_questions"] = len(grouped)
    if task != "sblimp":
        result["accuracy"] = sum(value for _, value in decisions) / len(decisions)
        return result

    pairs: dict[str, list[tuple[str, float]]] = {}
    for row, value in decisions:
        phenomenon = str(row.get("phenomenon", "all"))
        pair_id = str(row.get("pair_id", row["group_id"]))
        pairs.setdefault(pair_id, []).append((phenomenon, value))
    by_phenomenon: dict[str, list[float]] = {}
    for values in pairs.values():
        phenomena = {phenomenon for phenomenon, _ in values}
        if len(phenomena) != 1:
            raise ValueError("sBLIMP pair spans multiple phenomena")
        pair_score = sum(value for _, value in values) / len(values)
        by_phenomenon.setdefault(values[0][0], []).append(pair_score)
    phenomenon_scores = {
        phenomenon: sum(values) / len(values)
        for phenomenon, values in sorted(by_phenomenon.items())
    }
    result["voice_pair_accuracy"] = sum(value for _, value in decisions) / len(decisions)
    result["pair_accuracy"] = sum(map(sum, by_phenomenon.values())) / sum(
        map(len, by_phenomenon.values())
    )
    result["accuracy"] = sum(phenomenon_scores.values()) / len(phenomenon_scores)
    result["by_phenomenon"] = phenomenon_scores
    result["num_pairs"] = len(pairs)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate SLM loss or paired spoken benchmarks")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=["loss", "sblimp", "storycloze"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_file():
        resolved_checkpoint = checkpoint
        model = load_training_checkpoint(checkpoint, device)
    else:
        if checkpoint.is_dir():
            resolved_checkpoint = checkpoint
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("install huggingface-hub to load a remote model") from exc
            resolved_checkpoint = Path(snapshot_download(args.checkpoint))
        model = TransformerLM.from_pretrained(resolved_checkpoint, device=device)
    shard = PackedShard(args.data, verify_checksums=True)
    include_terminal_eos = args.task == "loss"
    scores = score_shard(
        model,
        shard,
        args.batch_size,
        device,
        args.precision,
        include_terminal_eos=include_terminal_eos,
    )
    metrics = summarize(args.task, scores)
    result = {
        "format_version": 1,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "checkpoint_files_sha256": _checkpoint_hashes(resolved_checkpoint),
        "data": str(Path(args.data).resolve()),
        "data_schema_version": shard.meta["schema_version"],
        "data_checksums": shard.meta["checksums"],
        "data_source": shard.meta.get("source"),
        "data_preprocessing": shard.meta["preprocessing"],
        "seed": 1337,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "include_terminal_eos": include_terminal_eos,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
