from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .backends import HubertKMeansTokenizer
from .packing import PreparedSample, ShardWriter
from .schema import ManifestRecord, manifest_fingerprint, read_manifest, sha256_file


def tinystories_manifest_records(
    dataset_id: str,
    dataset_revision: str,
    split: str,
    *,
    max_samples: int | None = None,
) -> Iterable[ManifestRecord]:
    """Read TinyStories once and emit indexed synthesis records for balanced planning."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, revision=dataset_revision)
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for index in range(total):
        text = str(dataset[index]["text"]).strip()
        yield ManifestRecord(
            id=f"tinystories:{split}:{index:08d}",
            dataset="tinystories",
            split=split,
            audio=None,
            transcript=text,
            source_revision=dataset_revision,
            extra={"source_dataset": dataset_id, "source_index": index},
        )


def _kokoro_synthesize(pipeline, text: str, voice: str):
    chunks: list[np.ndarray] = []
    for result in pipeline(text, voice=voice):
        if result.audio is None:
            raise ValueError("Kokoro returned a segment without audio")
        audio = result.audio.detach().cpu().float().reshape(-1).numpy()
        chunks.append(audio)
    if not chunks:
        raise ValueError("Kokoro returned no audio")
    return np.concatenate(chunks).astype(np.float32)


def _resample_24k_to_16k(waveform: np.ndarray) -> np.ndarray:
    try:
        from torchaudio.functional import resample
    except ImportError as exc:
        raise ImportError("Kokoro preparation requires torchaudio for 24-to-16 kHz resampling") from exc
    return resample(torch.from_numpy(waveform).view(1, -1), 24000, 16000).view(-1).numpy()


def synthesize_tinystories(
    output_dir: str,
    *,
    shard_index: int,
    manifest: str,
    plan_path: str,
    hubert_model: str,
    hubert_revision: str,
    kmeans: str,
    hubert_layer: int = 11,
    hubert_normalize: bool | None = None,
    device: str = "cuda",
    dataset_id: str = "roneneldan/TinyStories",
    dataset_revision: str,
    kokoro_id: str = "hexgrad/Kokoro-82M",
    kokoro_revision: str,
    voice: str = "af_heart",
    language: str = "a",
    split: str = "train",
    keep_audio: bool = False,
    seed: int = 1337,
):
    plan = json.loads(Path(plan_path).read_text())
    if plan["manifest"] != str(Path(manifest).resolve()):
        raise ValueError("shard plan was created for a different manifest")
    if plan.get("manifest_sha256") != manifest_fingerprint(manifest):
        raise ValueError("manifest contents changed after the shard plan was created")
    try:
        shard = plan["shards"][shard_index]
    except IndexError as exc:
        raise ValueError("shard_index is outside the plan") from exc
    if shard["index"] != shard_index:
        raise ValueError("malformed shard plan")
    assignment_sha256 = hashlib.sha256(
        json.dumps(shard["record_indices"], separators=(",", ":")).encode()
    ).hexdigest()
    records = list(read_manifest(manifest, shard["record_indices"]))
    if any(record.source_revision != dataset_revision for record in records):
        raise ValueError("manifest and requested TinyStories revisions differ")
    if any(record.split != split for record in records):
        raise ValueError("manifest and requested TinyStories splits differ")
    from huggingface_hub import snapshot_download
    from kokoro import KModel, KPipeline

    model_path = snapshot_download(kokoro_id, revision=kokoro_revision)
    model_file = Path(model_path) / "kokoro-v1_0.pth"
    kokoro_model = KModel(
        repo_id=kokoro_id,
        config=str(Path(model_path) / "config.json"),
        model=str(model_file),
    ).to(device).eval()
    pipeline = KPipeline(lang_code=language, repo_id=kokoro_id, model=kokoro_model)
    voice_path = str(Path(model_path) / "voices" / f"{voice}.pt")
    speech_tokenizer = HubertKMeansTokenizer(
        hubert_model,
        kmeans,
        layer=hubert_layer,
        device=device,
        revision=hubert_revision,
        normalize=hubert_normalize,
    )

    preprocessing = {
        "dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "manifest_sha256": manifest_fingerprint(manifest),
        "plan_format_version": plan.get("format_version"),
        "plan_num_shards": len(plan["shards"]),
        "assignment_sha256": assignment_sha256,
        "kokoro": kokoro_id,
        "kokoro_revision": kokoro_revision,
        "voice": voice,
        "language": language,
        "synthesis_sample_rate": 24000,
        "unit_sample_rate": 16000,
        "hubert_model": hubert_model,
        "hubert_revision": hubert_revision,
        "hubert_layer": hubert_layer,
        "hubert_normalize": speech_tokenizer.normalize,
        "collapsed": True,
        "kmeans": {"name": Path(kmeans).name, "sha256": sha256_file(kmeans)},
        "vocab_size": speech_tokenizer.pad_token + 1,
        "eos_token": speech_tokenizer.eos_token,
        "pad_token": speech_tokenizer.pad_token,
        "num_assigned_records": len(records),
    }
    name = f"shard-{shard_index:05d}"
    with ShardWriter(
        output_dir,
        name,
        preprocessing=preprocessing,
        keep_audio=keep_audio,
    ) as writer:
        if writer.already_complete:
            return writer.final / "meta.json"
        for record in records:
            index = int(record.extra["source_index"])
            sample_id = f"tinystories-kokoro:{record.split}:{index:08d}"
            try:
                sample_seed = (seed + 1_000_003 * index) % 2**32
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                text = (record.transcript or "").strip()
                if not text:
                    raise ValueError("empty story")
                audio_24k = _kokoro_synthesize(pipeline, text, voice_path)
                audio_16k = _resample_24k_to_16k(audio_24k)
                codes = speech_tokenizer.encode(audio_16k)
            except Exception as exc:
                writer.skip(sample_id, type(exc).__name__, str(exc))
                continue
            writer.write(
                PreparedSample(
                    id=sample_id,
                    codes=codes,
                    waveform=audio_16k if keep_audio else None,
                    sample_rate=16000,
                    metadata={
                        "dataset": "tinystories-kokoro",
                        "split": record.split,
                        "source_dataset": record.extra["source_dataset"],
                        "source_revision": record.source_revision,
                        "source_index": index,
                    },
                )
            )
    return writer.final / "meta.json"
