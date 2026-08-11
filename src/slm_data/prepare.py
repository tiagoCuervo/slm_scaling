from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .backends import HubertKMeansTokenizer, load_audio
from .packing import PreparedSample, ShardWriter
from .schema import manifest_fingerprint, read_manifest, sha256_file


def encode_shard(
    manifest: str,
    plan_path: str,
    shard_index: int,
    output_dir: str,
    *,
    hubert_model: str,
    hubert_revision: str,
    kmeans: str,
    hubert_layer: int = 11,
    hubert_normalize: bool | None = None,
    device: str = "cuda",
    keep_audio: bool = False,
) -> dict:
    plan = json.loads(Path(plan_path).read_text())
    if plan["manifest"] != str(Path(manifest).resolve()):
        raise ValueError("shard plan was created for a different manifest")
    if plan.get("manifest_sha256") != manifest_fingerprint(manifest):
        raise ValueError("manifest contents changed after the shard plan was created")
    try:
        shard = plan["shards"][shard_index]
    except IndexError as exc:
        raise ValueError(f"shard index {shard_index} is outside the plan") from exc
    if shard["index"] != shard_index:
        raise ValueError("malformed shard plan")
    assignment_sha256 = hashlib.sha256(
        json.dumps(shard["record_indices"], separators=(",", ":")).encode()
    ).hexdigest()

    speech_tokenizer = HubertKMeansTokenizer(
        hubert_model,
        kmeans,
        layer=hubert_layer,
        device=device,
        revision=hubert_revision,
        normalize=hubert_normalize,
    )
    preprocessing = {
        "manifest_sha256": manifest_fingerprint(manifest),
        "plan_format_version": plan.get("format_version"),
        "plan_num_shards": len(plan["shards"]),
        "assignment_sha256": assignment_sha256,
        "num_assigned_records": len(shard["record_indices"]),
        "sample_rate": 16000,
        "unit_extractor": "hubert-kmeans",
        "collapsed": True,
        "hubert_model": hubert_model,
        "hubert_revision": hubert_revision,
        "hubert_layer": hubert_layer,
        "hubert_normalize": speech_tokenizer.normalize,
        "kmeans": {"name": Path(kmeans).name, "sha256": sha256_file(kmeans)},
        "vocab_size": speech_tokenizer.pad_token + 1,
        "eos_token": speech_tokenizer.eos_token,
        "pad_token": speech_tokenizer.pad_token,
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
        for record in read_manifest(manifest, shard["record_indices"]):
            try:
                if record.audio is None:
                    raise ValueError("record has no audio path")
                waveform, sample_rate = load_audio(
                    record.audio, offset=record.offset, duration=record.duration, target_rate=16000
                )
                codes = speech_tokenizer.encode(waveform)
            except Exception as exc:
                writer.skip(record.id, type(exc).__name__, str(exc))
                continue
            writer.write(
                PreparedSample(
                    id=record.id,
                    codes=codes,
                    waveform=waveform if keep_audio else None,
                    sample_rate=sample_rate,
                    metadata={
                        **record.extra,
                        "dataset": record.dataset,
                        "split": record.split,
                        "source_revision": record.source_revision,
                        "speaker": record.speaker,
                        "source_audio_name": Path(record.audio).name,
                        "source_offset": record.offset,
                        "source_duration": record.duration,
                    },
                )
            )
    return json.loads((Path(output_dir) / name / "meta.json").read_text())
