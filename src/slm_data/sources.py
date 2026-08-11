from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .schema import ManifestRecord, manifest_fingerprint, read_manifest


def _stable_id(dataset: str, split: str, key: str) -> str:
    digest = hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]
    return f"{dataset}:{split}:{digest}"


def librispeech_records(
    root: str | Path,
    dataset: str,
    split: str,
    *,
    source_revision: str | None = None,
) -> Iterable[ManifestRecord]:
    root = Path(root).resolve()
    audio_paths: dict[str, Path] = {}
    for suffix in ("*.flac", "*.wav"):
        for path in sorted(root.rglob(suffix)):
            if path.stem in audio_paths:
                raise ValueError(f"duplicate LibriSpeech audio ID: {path.stem}")
            audio_paths[path.stem] = path
    for utterance, audio_path in sorted(audio_paths.items()):
        speaker = utterance.split("-", 1)[0]
        yield ManifestRecord(
            id=f"{dataset}:{split}:{utterance}",
            dataset=dataset,
            split=split,
            audio=str(audio_path),
            speaker=speaker,
            source_revision=source_revision,
        )


def fairseq_tsv_records(
    manifest: str | Path,
    dataset: str,
    split: str,
    *,
    sample_rate: int = 16000,
    source_revision: str | None = None,
) -> Iterable[ManifestRecord]:
    manifest = Path(manifest)
    lines = manifest.read_text().splitlines()
    if not lines:
        raise ValueError("Fairseq manifest is empty")
    root = Path(lines[0]).expanduser()
    if not root.is_absolute():
        root = manifest.parent / root
    root = root.resolve()
    for line in lines[1:]:
        relpath, frames = line.rsplit("\t", 1)
        if int(frames) <= 0:
            raise ValueError(f"invalid frame count for {relpath}: {frames}")
        key = f"{relpath}:{frames}"
        yield ManifestRecord(
            id=_stable_id(dataset, split, key),
            dataset=dataset,
            split=split,
            audio=str((root / relpath).resolve()),
            sample_rate=sample_rate,
            duration=int(frames) / sample_rate,
            source_revision=source_revision,
        )


def normalize_jsonl_records(
    path: str | Path,
    dataset: str | None,
    split: str | None,
    source_revision: str | None = None,
):
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            raw = json.loads(line)
            ds = dataset or raw.get("dataset")
            sp = split or raw.get("split")
            if not ds or not sp:
                raise ValueError("dataset and split must be provided in the record or command")
            audio = raw.get("audio") or raw.get("audio_path") or raw.get("path")
            rid = raw.get("id") or _stable_id(ds, sp, f"{audio}:{raw.get('offset', 0)}:{index}")
            known = {
                "id", "dataset", "split", "audio", "audio_path", "path", "sample_rate",
                "duration", "offset", "transcript", "text", "speaker", "source_revision", "words",
            }
            yield ManifestRecord(
                id=rid,
                dataset=ds,
                split=sp,
                audio=audio,
                sample_rate=raw.get("sample_rate"),
                duration=raw.get("duration"),
                offset=float(raw.get("offset", 0.0)),
                transcript=(raw.get("transcript", raw.get("text")) if audio is None else None),
                speaker=raw.get("speaker"),
                source_revision=source_revision or raw.get("source_revision"),
                extra={k: v for k, v in raw.items() if k not in known},
            )


def make_shard_plan(manifest: str | Path, num_shards: int, output: str | Path) -> dict:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    records = list(read_manifest(manifest))
    if not records:
        raise ValueError("cannot plan an empty manifest")
    if num_shards > len(records):
        raise ValueError("num_shards cannot exceed the number of manifest records")
    ranked = sorted(
        enumerate(records),
        key=lambda item: (-(item[1].duration or max(len(item[1].transcript or ""), 1)), item[1].id),
    )
    loads = [0.0] * num_shards
    shards: list[list[int]] = [[] for _ in range(num_shards)]
    for index, record in ranked:
        shard = min(range(num_shards), key=lambda i: (loads[i], i))
        shards[shard].append(index)
        loads[shard] += record.duration or max(len(record.transcript or ""), 1)
    plan = {
        "format_version": 1,
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_fingerprint(manifest),
        "num_records": len(records),
        "shards": [{"index": i, "record_indices": sorted(ids), "estimated_load": loads[i]} for i, ids in enumerate(shards)],
    }
    Path(output).write_text(json.dumps(plan, indent=2) + "\n")
    return plan
