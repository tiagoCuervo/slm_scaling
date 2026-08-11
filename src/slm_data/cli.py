from __future__ import annotations

import argparse
import json

from .prepare import encode_shard
from .schema import validate_shard, write_manifest
from .sources import fairseq_tsv_records, librispeech_records, make_shard_plan, normalize_jsonl_records
from .tinystories import synthesize_tinystories, tinystories_manifest_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm-data", description="Prepare versioned speech-unit shards")
    commands = parser.add_subparsers(dest="command", required=True)

    ls = commands.add_parser("manifest-librispeech")
    ls.add_argument("--root", required=True)
    ls.add_argument("--dataset", default="librispeech")
    ls.add_argument("--split", required=True)
    ls.add_argument("--source-revision", required=True)
    ls.add_argument("--output", required=True)

    tiny_manifest = commands.add_parser("manifest-tinystories")
    tiny_manifest.add_argument("--dataset", default="roneneldan/TinyStories")
    tiny_manifest.add_argument("--dataset-revision", required=True)
    tiny_manifest.add_argument("--split", default="train")
    tiny_manifest.add_argument("--max-samples", type=int)
    tiny_manifest.add_argument("--output", required=True)

    tsv = commands.add_parser("manifest-tsv")
    tsv.add_argument("--input", required=True)
    tsv.add_argument("--dataset", required=True)
    tsv.add_argument("--split", required=True)
    tsv.add_argument("--sample-rate", type=int, default=16000)
    tsv.add_argument("--source-revision", required=True)
    tsv.add_argument("--output", required=True)

    js = commands.add_parser("manifest-jsonl")
    js.add_argument("--input", required=True)
    js.add_argument("--dataset")
    js.add_argument("--split")
    js.add_argument("--source-revision")
    js.add_argument("--output", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--num-shards", type=int, required=True)
    plan.add_argument("--output", required=True)

    enc = commands.add_parser("encode")
    enc.add_argument("--manifest", required=True)
    enc.add_argument("--plan", required=True)
    enc.add_argument("--shard-index", type=int, required=True)
    enc.add_argument("--output", required=True)
    enc.add_argument("--hubert-model", required=True, help="pinned Hugging Face model ID")
    enc.add_argument("--hubert-revision", required=True)
    enc.add_argument("--kmeans", required=True)
    enc.add_argument("--hubert-layer", type=int, default=11)
    enc.add_argument("--hubert-normalize", action=argparse.BooleanOptionalAction, default=None)
    enc.add_argument("--device", default="cuda")
    enc.add_argument("--keep-audio", action="store_true")

    tiny = commands.add_parser("tinystories")
    tiny.add_argument("--output", required=True)
    tiny.add_argument("--shard-index", type=int, required=True)
    tiny.add_argument("--manifest", required=True)
    tiny.add_argument("--plan", required=True)
    tiny.add_argument("--dataset-revision", required=True)
    tiny.add_argument("--kokoro-revision", required=True)
    tiny.add_argument("--hubert-model", required=True, help="pinned Hugging Face model ID")
    tiny.add_argument("--hubert-revision", required=True)
    tiny.add_argument("--kmeans", required=True)
    tiny.add_argument("--hubert-layer", type=int, default=11)
    tiny.add_argument("--hubert-normalize", action=argparse.BooleanOptionalAction, default=None)
    tiny.add_argument("--device", default="cuda")
    tiny.add_argument("--split", default="train")
    tiny.add_argument("--keep-audio", action="store_true")

    val = commands.add_parser("validate")
    val.add_argument("paths", nargs="+")
    val.add_argument("--skip-checksums", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "manifest-librispeech":
        records = librispeech_records(
            args.root,
            args.dataset,
            args.split,
            source_revision=args.source_revision,
        )
        digest = write_manifest(records, args.output)
        print(json.dumps({"manifest": args.output, "sha256": digest}))
    elif args.command == "manifest-tinystories":
        records = tinystories_manifest_records(
            args.dataset,
            args.dataset_revision,
            args.split,
            max_samples=args.max_samples,
        )
        digest = write_manifest(records, args.output)
        print(json.dumps({"manifest": args.output, "sha256": digest}))
    elif args.command == "manifest-tsv":
        records = fairseq_tsv_records(
            args.input,
            args.dataset,
            args.split,
            sample_rate=args.sample_rate,
            source_revision=args.source_revision,
        )
        digest = write_manifest(records, args.output)
        print(json.dumps({"manifest": args.output, "sha256": digest}))
    elif args.command == "manifest-jsonl":
        records = normalize_jsonl_records(
            args.input,
            args.dataset,
            args.split,
            source_revision=args.source_revision,
        )
        digest = write_manifest(records, args.output)
        print(json.dumps({"manifest": args.output, "sha256": digest}))
    elif args.command == "plan":
        print(json.dumps(make_shard_plan(args.manifest, args.num_shards, args.output), indent=2))
    elif args.command == "encode":
        result = encode_shard(
            args.manifest,
            args.plan,
            args.shard_index,
            args.output,
            hubert_model=args.hubert_model,
            hubert_revision=args.hubert_revision,
            kmeans=args.kmeans,
            hubert_layer=args.hubert_layer,
            hubert_normalize=args.hubert_normalize,
            device=args.device,
            keep_audio=args.keep_audio,
        )
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "tinystories":
        result = synthesize_tinystories(
            args.output,
            shard_index=args.shard_index,
            manifest=args.manifest,
            plan_path=args.plan,
            dataset_revision=args.dataset_revision,
            kokoro_revision=args.kokoro_revision,
            hubert_model=args.hubert_model,
            hubert_revision=args.hubert_revision,
            kmeans=args.kmeans,
            hubert_layer=args.hubert_layer,
            hubert_normalize=args.hubert_normalize,
            device=args.device,
            split=args.split,
            keep_audio=args.keep_audio,
        )
        print(result)
    elif args.command == "validate":
        for path in args.paths:
            print(json.dumps(validate_shard(path, verify_checksums=not args.skip_checksums), indent=2))
if __name__ == "__main__":
    main()
