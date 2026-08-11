import numpy as np
import pytest
import torch

from slm_data import PackedShard, PreparedSample, ShardWriter, validate_shard
from slm_data.backends import collapse_codes
from slm_data.schema import ManifestRecord, manifest_fingerprint, read_manifest, write_manifest
from slm_data.sources import fairseq_tsv_records, librispeech_records, make_shard_plan
from slm_data.tinystories import _kokoro_synthesize


def test_kokoro_chunks_are_concatenated():
    class Result:
        audio = torch.zeros(2400)

    waveform = _kokoro_synthesize(lambda *args, **kwargs: [Result()], "hello!", "voice")
    assert waveform.shape == (2400,)


def test_units_are_single_stream_and_collapsed():
    codes = collapse_codes(np.asarray([3, 3, 7, 7, 3], dtype=np.int16), eos_token=8)
    assert codes.tolist() == [3, 7, 3, 8]
    with pytest.raises(ValueError, match="one unit stream"):
        collapse_codes(np.ones((4, 2), dtype=np.int16), eos_token=8)


def test_manifest_index_and_deterministic_plan(tmp_path):
    records = [
        ManifestRecord(id=f"d:train:{i}", dataset="d", split="train", audio=f"{i}.wav", duration=i + 1)
        for i in range(9)
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    assert [record.id for record in read_manifest(manifest, [8, 0, 3])] == ["d:train:8", "d:train:0", "d:train:3"]
    first = make_shard_plan(manifest, 3, tmp_path / "plan-a.json")
    second = make_shard_plan(manifest, 3, tmp_path / "plan-b.json")
    assert first == second
    assert sorted(i for shard in first["shards"] for i in shard["record_indices"]) == list(range(9))


def test_plan_rejects_empty_shards_and_detects_manifest_changes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [ManifestRecord(id="d:train:0", dataset="d", split="train", audio="0.wav")],
        manifest,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        make_shard_plan(manifest, 2, tmp_path / "too-many.json")
    plan = make_shard_plan(manifest, 1, tmp_path / "plan.json")
    assert plan["manifest_sha256"]
    manifest.write_text(manifest.read_text() + "\n")
    assert plan["manifest_sha256"] != manifest_fingerprint(manifest)


def test_units_shard(tmp_path):
    with ShardWriter(tmp_path, "shard-00000", preprocessing={"test": True}) as writer:
        for index in range(3):
            writer.write(
                PreparedSample(
                    id=str(index),
                    codes=np.arange(10, dtype=np.int16),
                )
            )
    meta = validate_shard(tmp_path / "shard-00000")
    assert meta["num_samples"] == 3
    assert PackedShard(tmp_path / "shard-00000").sample(1)["audio"].shape == (10,)


def test_completed_shard_is_idempotent(tmp_path):
    with ShardWriter(tmp_path, "part", preprocessing={}) as writer:
        writer.write(PreparedSample(id="a", codes=np.ones(4, np.int16)))
    with ShardWriter(tmp_path, "part", preprocessing={}) as writer:
        assert writer.already_complete


def test_completed_shard_rejects_different_preprocessing(tmp_path):
    with ShardWriter(
        tmp_path,
        "part",
        preprocessing={"revision": "one"},
    ) as writer:
        writer.write(PreparedSample(id="a", codes=np.ones(4, np.int16)))
    with pytest.raises(ValueError, match="different parameters"):
        with ShardWriter(
            tmp_path,
            "part",
            preprocessing={"revision": "two"},
        ):
            pass


def test_writer_rejects_multiple_codebooks(tmp_path):
    with pytest.raises(ValueError, match="invalid code shape"):
        with ShardWriter(tmp_path, "part", preprocessing={}) as writer:
            writer.write(PreparedSample(id="a", codes=np.ones((4, 2), np.int16)))


def test_interrupted_shard_is_never_published(tmp_path):
    with pytest.raises(RuntimeError):
        with ShardWriter(tmp_path, "part", preprocessing={}) as writer:
            writer.write(PreparedSample(id="a", codes=np.ones(4, np.int16)))
            raise RuntimeError("interrupted")
    assert not (tmp_path / "part").exists()


def test_empty_shard_keeps_machine_readable_skip_diagnostics(tmp_path):
    with pytest.raises(ValueError, match="diagnostics"):
        with ShardWriter(
            tmp_path,
            "part",
            preprocessing={"num_assigned_records": 1},
        ) as writer:
            writer.skip("sample", "ValueError", "bad input")
    assert not (tmp_path / "part").exists()
    assert '"id": "sample"' in (tmp_path / "part.failed.skips.jsonl").read_text()
    assert '"num_skipped": 1' in (tmp_path / "part.failed.json").read_text()


def test_checksum_mismatch_is_rejected(tmp_path):
    with ShardWriter(tmp_path, "part", preprocessing={}) as writer:
        writer.write(PreparedSample(id="a", codes=np.ones(4, np.int16)))
    with (tmp_path / "part" / "audio.bin").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="audio size"):
        validate_shard(tmp_path / "part")


def test_metadata_cannot_override_schema_fields(tmp_path):
    with pytest.raises(ValueError, match="reserved fields"):
        with ShardWriter(
            tmp_path,
            "part",
            preprocessing={},
        ) as writer:
            writer.write(
                PreparedSample(
                    id="real-id",
                    codes=np.ones(4, np.int16),
                    metadata={"id": "spoofed-id"},
                )
            )


def test_sample_metadata_count_is_validated(tmp_path):
    with ShardWriter(
        tmp_path,
        "part",
        preprocessing={"vocab_size": 2},
    ) as writer:
        writer.write(PreparedSample(id="a", codes=np.ones(4, np.int16)))
    (tmp_path / "part" / "samples.jsonl").write_text("")
    with pytest.raises(ValueError, match="metadata count"):
        validate_shard(tmp_path / "part", verify_checksums=False)


def test_librispeech_adapter_indexes_audio_once_and_rejects_duplicates(tmp_path):
    chapter = tmp_path / "1" / "2"
    chapter.mkdir(parents=True)
    (chapter / "1-2.trans.txt").write_text("1-2-3 HELLO\n1-2-4 WORLD\n")
    (chapter / "1-2-3.flac").touch()
    (chapter / "1-2-4.wav").touch()
    records = list(librispeech_records(tmp_path, "librispeech", "train"))
    assert [record.id for record in records] == [
        "librispeech:train:1-2-3",
        "librispeech:train:1-2-4",
    ]
    (chapter / "1-2-3.wav").touch()
    with pytest.raises(ValueError, match="duplicate LibriSpeech audio"):
        list(librispeech_records(tmp_path, "librispeech", "train"))


def test_fairseq_adapter_resolves_relative_root_from_manifest(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = manifest_dir / "train.tsv"
    manifest.write_text("../audio\nexample.flac\t32000\n")
    record = list(fairseq_tsv_records(manifest, "d", "train"))[0]
    assert record.audio == str((audio / "example.flac").resolve())
    assert record.duration == 2.0
