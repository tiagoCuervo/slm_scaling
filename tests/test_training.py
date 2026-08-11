from dataclasses import replace
from pathlib import Path

import torch

from slm.train import TrainConfig, _batch_provenance, _learning_rate, train
from tests.fixtures import create_toy_dataset


def test_cpu_toy_training_and_resume(tmp_path):
    data = tmp_path / "data"
    create_toy_dataset(data)
    config = TrainConfig(
        train_data=str(data / "train"),
        val_data=str(data / "val"),
        output_dir=str(tmp_path / "initial"),
        batch_size=2,
        max_steps=3,
        eval_interval=1,
        eval_steps=1,
        save_interval=1,
        log_interval=1,
        precision="float32",
        compile=False,
        dim=32,
        n_layers=1,
        n_heads=4,
        vocab_size=34,
        multiple_of=16,
        block_size=16,
    )
    train(config)
    checkpoint = tmp_path / "initial" / "checkpoint-00000002.pt"
    assert checkpoint.exists()
    resumed = replace(
        config,
        output_dir=str(tmp_path / "resumed"),
        resume=str(checkpoint),
    )
    train(resumed)
    resumed_checkpoint = tmp_path / "resumed" / "checkpoint-00000003.pt"
    assert resumed_checkpoint.exists()
    baseline_state = torch.load(
        tmp_path / "initial" / "checkpoint-00000003.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    resumed_state = torch.load(
        resumed_checkpoint, map_location="cpu", weights_only=False
    )["model"]
    assert baseline_state.keys() == resumed_state.keys()
    for name in baseline_state:
        torch.testing.assert_close(
            baseline_state[name], resumed_state[name], rtol=0, atol=0
        )


def test_exact_batch_provenance():
    config = TrainConfig(batch_size=26, gradient_accumulation_steps=19, block_size=2050)
    provenance = _batch_provenance(config)
    assert provenance["global_sequences_per_update"] == 494
    assert provenance["tokens_per_update"] == 1_012_700


def test_cosine_decay_reaches_floor_on_final_update():
    config = TrainConfig(
        max_steps=10,
        warmup_steps=2,
        learning_rate=5e-4,
        min_learning_rate=5e-5,
    )
    assert _learning_rate(2, config) == config.learning_rate
    assert _learning_rate(9, config) == config.min_learning_rate


def test_public_scaling_configs_preserve_exact_batch_math():
    root = Path(__file__).parents[1] / "configs" / "scaling"
    expected = {
        "20m-109b.json": (64, 131_200),
        "85m-109b.json": (128, 262_400),
        "155m-65b.json": (256, 524_800),
        "309m-30b.json": (252, 516_600),
        "823m-26b.json": (494, 1_012_700),
        "823m-82b.json": (608, 1_246_400),
    }
    for name, (sequences, tokens) in expected.items():
        config = TrainConfig.load(root / name)
        provenance = _batch_provenance(config)
        assert provenance["global_sequences_per_update"] == sequences
        assert provenance["tokens_per_update"] == tokens
