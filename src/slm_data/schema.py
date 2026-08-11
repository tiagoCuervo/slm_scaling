from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ManifestRecord:
    """One source utterance; transcript is used only for synthetic speech."""

    id: str
    dataset: str
    split: str
    audio: str | None
    sample_rate: int | None = None
    duration: float | None = None
    offset: float = 0.0
    transcript: str | None = None
    speaker: str | None = None
    source_revision: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id or not self.dataset or not self.split:
            raise ValueError("manifest id, dataset, and split must be non-empty")
        if self.offset < 0 or (self.duration is not None and self.duration <= 0):
            raise ValueError(f"invalid segment for {self.id}")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError(f"invalid sample rate for {self.id}")


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> str:
    """Write JSONL and a uint64 byte-offset index; return the content SHA256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    idx_tmp = path.with_name(path.name + ".idx.tmp")
    offsets: list[int] = []
    digest = hashlib.sha256()
    seen: set[str] = set()
    with tmp.open("wb") as stream:
        for record in records:
            record.validate()
            if record.id in seen:
                raise ValueError(f"duplicate manifest id: {record.id}")
            seen.add(record.id)
            offsets.append(stream.tell())
            line = (json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n").encode()
            stream.write(line)
            digest.update(line)
    np.asarray(offsets, dtype=np.uint64).tofile(idx_tmp)
    tmp.replace(path)
    idx_tmp.replace(path.with_suffix(path.suffix + ".idx"))
    return digest.hexdigest()


def read_manifest(path: str | Path, indices: Iterable[int] | None = None) -> Iterator[ManifestRecord]:
    path = Path(path)
    if indices is None:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    record = ManifestRecord(**json.loads(line))
                    record.validate()
                    yield record
        return

    offsets = np.memmap(path.with_suffix(path.suffix + ".idx"), dtype=np.uint64, mode="r")
    with path.open("rb") as stream:
        for index in indices:
            if index < 0 or index >= len(offsets):
                raise IndexError(index)
            stream.seek(int(offsets[index]))
            record = ManifestRecord(**json.loads(stream.readline()))
            record.validate()
            yield record


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    index = path.with_suffix(path.suffix + ".idx")
    if index.exists():
        digest.update(index.read_bytes())
    return digest.hexdigest()


def validate_shard(path: str | Path, verify_checksums: bool = True) -> dict[str, Any]:
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version in {path}: {meta.get('schema_version')}")
    n_samples = int(meta["num_samples"])
    lengths = np.fromfile(path / "audio.len", dtype=np.int64)
    if len(lengths) != n_samples or np.any(lengths <= 0):
        raise ValueError(f"invalid audio lengths in {path}")
    if int(meta.get("num_audio_steps", -1)) != int(lengths.sum()):
        raise ValueError(f"audio count does not match metadata in {path}")
    expected_audio = int(lengths.sum()) * np.dtype(np.int16).itemsize
    if (path / "audio.bin").stat().st_size != expected_audio:
        raise ValueError(f"audio size does not match metadata in {path}")
    codes = np.memmap(path / "audio.bin", dtype=np.int16, mode="r")
    vocab_size = meta.get("preprocessing", {}).get("vocab_size")
    if np.any(codes < 0) or (
        vocab_size is not None and np.any(codes >= int(vocab_size))
    ):
        raise ValueError(f"audio codes are outside the declared vocabulary in {path}")

    sample_rows = []
    with (path / "samples.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                sample_rows.append(json.loads(line))
    if len(sample_rows) != n_samples:
        raise ValueError(f"sample metadata count does not match in {path}")
    sample_ids = [row.get("id") for row in sample_rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != n_samples:
        raise ValueError(f"sample IDs are empty or duplicated in {path}")
    if [int(row.get("audio_steps", -1)) for row in sample_rows] != lengths.tolist():
        raise ValueError(f"sample metadata lengths do not match in {path}")

    with (path / "skips.jsonl").open() as stream:
        skip_rows = [json.loads(line) for line in stream if line.strip()]
    if len(skip_rows) != int(meta.get("num_skipped", 0)):
        raise ValueError(f"skip metadata count does not match in {path}")
    if verify_checksums:
        for name, expected in meta["checksums"].items():
            if sha256_file(path / name) != expected:
                raise ValueError(f"checksum mismatch for {path / name}")
    return meta
