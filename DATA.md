# Data preparation

`slm_data` turns public audio into consecutive-run-collapsed speech units.

## Packed schema

Each independently retryable shard is written atomically and contains:

| File | Contents |
|---|---|
| `meta.json` | schema version, source and model revisions, token IDs, counts, SHA-256 checksums |
| `audio.bin` | concatenated unit sequences `[sum_time]`, `int16` |
| `audio.len` | one unit-sequence length per sample, `int64` |
| `samples.jsonl` | stable sample IDs and source provenance |
| `skips.jsonl` | structured rejection reasons |

Codes contain one mHuBERT K-means unit stream and include terminal EOS. Source transcripts are neither required nor stored in packed shards; TinyStories text
is read only to synthesize speech.

## EMNLP 2024 mixture

Hours and unit counts are the paper statistics. They were rounded and might have changed since then. Acquire each corpus under its own terms and keep
the official split when one exists; otherwise record a deterministic split in the manifest.

| Source | Paper scale | Public source | Adapter |
|---|---:|---|---|
| LibriSpeech | 960 h / 67M | [OpenSLR 12](https://www.openslr.org/12), CC BY 4.0 | `manifest-librispeech` |
| LibriLight | 53k h / 3.74B | [LibriLight](https://github.com/facebookresearch/libri-light/tree/main/data_preparation), derived from LibriVox | `manifest-tsv` |
| Spoken Wikipedia | 1k h / 32M | [Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/), CC BY-SA 4.0 | `manifest-jsonl` |
| TED-LIUM 3 | 1.6k h / 110M | [OpenSLR 51](https://www.openslr.org/51), CC BY-NC-ND | `manifest-tsv` |
| People's Speech | 7k h / 480M | [`MLCommons/peoples_speech`](https://huggingface.co/datasets/MLCommons/peoples_speech); retain per-item licenses | `manifest-jsonl` |
| VoxPopuli English | 24k h / 1.64B | [`facebook/voxpopuli`](https://huggingface.co/datasets/facebook/voxpopuli) | `manifest-jsonl` |
| sTinyStories | 72k h / 4.82B | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories); historical audio used FastSpeech2/LJSpeech | Kokoro recipe below produces a distinct dataset |

The reported mixture totals 160k hours and 10.89B units. Natural-speech evaluation excludes
sTinyStories.

## Encode audio

Download and verify the TWIST quantizer once:

```bash
mkdir -p assets
curl -L \
  https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin \
  -o assets/mhubert-25hz-l11-kmeans500.bin
echo '03cc04a9c24fec4285e73e709c485756d8f116aa8e724eac555de6a7cf8d28ad  assets/mhubert-25hz-l11-kmeans500.bin' \
  | sha256sum --check
```

For LibriSpeech:

```bash
slm-data manifest-librispeech \
  --root /path/to/LibriSpeech/train-clean-100 --split train-clean-100 \
  --source-revision openslr-12 --output manifests/librispeech.jsonl
slm-data plan --manifest manifests/librispeech.jsonl --num-shards 64 \
  --output manifests/librispeech.plan.json
slm-data encode \
  --manifest manifests/librispeech.jsonl --plan manifests/librispeech.plan.json \
  --shard-index "$SHARD" --output data/librispeech/hubert/train \
  --hubert-model slprl/mhubert-base-25hz \
  --hubert-revision a319086e1d343190047d02b7f81133fb310c1b90 \
  --kmeans assets/mhubert-25hz-l11-kmeans500.bin --no-hubert-normalize
slm-data validate data/librispeech/hubert/train/shard-00000
```

For other corpora, normalize one JSON object per audio segment, then run the same `plan` and
`encode` commands:

```json
{"id":"source:train:0001","dataset":"source","split":"train","audio":"/data/0001.flac","offset":0.0,"duration":4.2,"sample_rate":16000,"source_revision":"release-tag"}
```

The paper used the [textlesslib TWIST](https://github.com/facebookresearch/textlesslib/tree/main/examples/twist)
25 Hz mHuBERT and layer-11 quantizer through the verified
[SLP-RL conversion](https://huggingface.co/slprl/mhubert-base-25hz), without waveform
normalization. See [THIRD_PARTY.md](THIRD_PARTY.md) for citations and licenses.

## TinyStories with Kokoro

The maintained recipe uses [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), so its
output is distinct from the paper's FastSpeech2 sTinyStories. We switched to it because of being better quality and fast enough. Each worker initializes Kokoro once,
synthesizes at 24 kHz, resamples once to 16 kHz, extracts units, and discards waveforms unless
`--keep-audio` is set.

```bash
slm-data manifest-tinystories \
  --dataset roneneldan/TinyStories \
  --dataset-revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --split train --output manifests/tinystories.jsonl
slm-data plan --manifest manifests/tinystories.jsonl --num-shards 256 \
  --output manifests/tinystories.plan.json
slm-data tinystories \
  --manifest manifests/tinystories.jsonl --plan manifests/tinystories.plan.json \
  --shard-index "$SHARD" --output data/tinystories-kokoro/hubert/train \
  --dataset-revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --kokoro-revision f3ff3571791e39611d31c381e3a41a3af07b4987 \
  --hubert-model slprl/mhubert-base-25hz \
  --hubert-revision a319086e1d343190047d02b7f81133fb310c1b90 \
  --kmeans assets/mhubert-25hz-l11-kmeans500.bin --no-hubert-normalize
```

## Evaluation data

- **sBLIMP:** encode both members of each released minimal pair and preserve its group and label.
- **sStoryCloze / tStoryCloze:** encode the released spoken candidates while preserving question,
  speaker, condition, and correctness metadata.

Source audio and restricted datasets are never redistributed. Dataset and model licenses are
independent of this repository's Apache-2.0 code license.
