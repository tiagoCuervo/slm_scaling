from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from slm_data import PackedShard


class SpeechTokenCorpus:
    """Random-window reader over packed speech-unit shards."""

    def __init__(self, root: str | Path, block_size: int, *, verify_checksums: bool = False):
        self.root = Path(root)
        paths = sorted(path.parent for path in self.root.glob("*/meta.json"))
        if not paths and (self.root / "meta.json").exists():
            paths = [self.root]
        if not paths:
            raise FileNotFoundError(f"no canonical shards under {self.root}")
        self.shards = [PackedShard(path, verify_checksums=verify_checksums) for path in paths]
        preprocessing = [shard.meta["preprocessing"] for shard in self.shards]
        for key in ("vocab_size", "eos_token", "pad_token"):
            values = {item.get(key) for item in preprocessing}
            if None in values or len(values) != 1:
                raise ValueError(f"all SLM shards must use one declared {key}")
            setattr(self, key, int(next(iter(values))))
        self.block_size = block_size
        self.available = np.asarray(
            [max(0, len(shard.codes) - block_size) for shard in self.shards], dtype=np.int64
        )
        if not self.available.sum():
            raise ValueError(f"all shards are shorter than block_size={block_size}")
        self.probabilities = self.available / self.available.sum()

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for shard in self.shards:
            digest.update(json.dumps(shard.meta, sort_keys=True).encode())
        return digest.hexdigest()

    def get_batch(self, batch_size: int, rng: np.random.Generator, device: torch.device):
        shard_ids = rng.choice(len(self.shards), size=batch_size, p=self.probabilities)
        rows = []
        for shard_id in shard_ids:
            shard = self.shards[int(shard_id)]
            start = int(rng.integers(0, len(shard.codes) - self.block_size))
            rows.append(np.asarray(shard.codes[start:start + self.block_size + 1], dtype=np.int64))
        tokens = torch.from_numpy(np.stack(rows))
        if device.type == "cuda":
            tokens = tokens.pin_memory().to(device, non_blocking=True)
        else:
            tokens = tokens.to(device)
        return tokens[:, :-1], tokens[:, 1:]
