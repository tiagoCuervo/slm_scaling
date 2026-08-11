from __future__ import annotations

from pathlib import Path

import numpy as np

from slm_data.packing import PreparedSample, ShardWriter


def create_toy_dataset(output: str | Path, seed: int = 1337) -> None:
    """Create deterministic packed units for tests."""
    rng = np.random.default_rng(seed)
    output = Path(output)
    for split, count in (("train", 32), ("val", 8)):
        preprocessing = {
            "generator": "slm-data-toy-v1",
            "seed": seed,
            "eos_token": 32,
            "pad_token": 33,
            "vocab_size": 34,
            "sample_rate": 16000,
        }
        with ShardWriter(
            output / split,
            "shard-00000",
            preprocessing=preprocessing,
        ) as writer:
            if writer.already_complete:
                continue
            for index in range(count):
                length = int(rng.integers(48, 80))
                codes = rng.integers(0, 32, size=length - 1, dtype=np.int16)
                codes = np.concatenate((codes, np.asarray([32], dtype=np.int16)))
                writer.write(
                    PreparedSample(
                        id=f"toy:{split}:{index:04d}",
                        codes=codes,
                        metadata={"dataset": "toy", "split": split},
                    )
                )
