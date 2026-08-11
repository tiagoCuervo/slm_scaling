# Scaling suite of Generative Speech Language Models

[![Paper](https://img.shields.io/badge/EMNLP-2024-4b44ce.svg)](https://aclanthology.org/2024.emnlp-main.21/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face-FFD21E.svg)](https://huggingface.co/collections/tiagoCuervo/gslm-scaling)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Roughly a speech version of nanoGPT: a compact Llama-style autoregressive model over HuBERT units,
plus data preparation, DDP training, evaluation, safetensors export, and KV-cached generation.

This repository accompanies [*Scaling Properties of Speech Language
Models*](https://aclanthology.org/2024.emnlp-main.21/) (EMNLP 2024). See
[DATA.md](DATA.md) for public-source preparation and the packed schema, and
[RESULTS.md](RESULTS.md) for scaling-checkpoint scores. External model and vocoder credits are
in [THIRD_PARTY.md](THIRD_PARTY.md).

## Install

Python 3.11 and PyTorch 2.4+ are supported.

```bash
conda create -n slm python=3.11 -y
conda activate slm
pip install -c requirements/constraints.txt -e '.[data,hubert,audio,notebook]'
```

Run the test suite to verify the installation:

```bash
pytest -q
```

## Prepare data

All sources use one manifest → shard plan → packed-units pipeline.

```bash
slm-data manifest-librispeech \
  --root /path/to/LibriSpeech --split train-clean-100 \
  --source-revision openslr-12 --output manifests/librispeech.jsonl
slm-data plan --manifest manifests/librispeech.jsonl --num-shards 64 \
  --output manifests/librispeech.plan.json
slm-data encode \
  --manifest manifests/librispeech.jsonl \
  --plan manifests/librispeech.plan.json --shard-index "$SHARD" \
  --hubert-model slprl/mhubert-base-25hz \
  --hubert-revision a319086e1d343190047d02b7f81133fb310c1b90 \
  --kmeans assets/mhubert-25hz-l11-kmeans500.bin --no-hubert-normalize \
  --output data/librispeech/hubert/train
slm-data validate data/librispeech/hubert/train/shard-00000
```

The paper mixture contains LibriSpeech, LibriLight, Spoken Wikipedia, TED-LIUM 3, People's
Speech, English VoxPopuli, and sTinyStories. [DATA.md](DATA.md) lists their public sources,
licenses, split rules, generic manifest adapters, TinyStories/Kokoro synthesis, single-codebook
output, and evaluation conversion.

## Run the scaling study

The supplied configurations define the paper architectures, optimizer, context, and exact
batch arithmetic. Use ordinary Python or `torchrun`; gradient accumulation is global and must be
divisible by world size.

```bash
torchrun --standalone --nproc-per-node=8 -m slm.train \
  configs/scaling/155m-65b.json
```

Each scaling-curve point is a completed run with its own final token budget, not an intermediate
checkpoint. Start from the matching configuration and override `max_steps` and `output_dir`.

| Model | Base config | Tokens/update | Curve `max_steps` | Stable A100-80 MFU |
|---:|---|---:|---|---:|
| 20M | `20m-109b.json` | 131,200 | 5,050; 10,100; 15,750; 165,950; 331,900; 497,900; 663,850; 829,800 | 31.97% |
| 85M | `85m-109b.json` | 262,400 | 5,200; 10,400; 20,800; 32,500; 165,950; 248,950; 331,900; 414,900 | 36.52% |
| 155M | `155m-65b.json` | 524,800 | 5,850; 9,400; 18,850; 37,000; 82,950; 124,450 | 41.95% |
| 309M | `309m-30b.json` | 516,600 | 5,850; 9,550; 19,100; 38,250; 58,850 | 43.60% |
| 823M | paper curve | 934,800 | 8,850; 17,650 | 62.52% |
| 823M | `823m-26b.json` | 1,012,700 | 26,000 | 50.58% |
| 823M | `823m-82b.json` | 1,246,400 | 66,050 | 62.57% |

Example curve point:

```bash
slm-train configs/scaling/155m-65b.json \
  max_steps=18850 output_dir='"out/scaling/155m-9p9b"'
```

The reported MFU is the median raw value over the longest clean post-warmup segment of a
completed one-A100-80GB PCIe run, with BF16 and compilation enabled. Parameters and optimizer state remain FP32; BF16 is autocast.

## Evaluate

`slm-eval` accepts a training checkpoint, a converted directory, or a Hugging Face model. Loss,
sBLIMP, and StoryCloze use the same packed shards.

```bash
slm-eval \
  --checkpoint tiagoCuervo/gslm-scaling-155m-65p3b \
  --data data/sblimp/hubert/dev --task sblimp \
  --precision bfloat16 --output results/slm-155m-sblimp.json
```

Expected scores for all 29 paper curve endpoints and the 26B/82B extended endpoints are in
[RESULTS.md](RESULTS.md).

## Load and generate

```python
import torch
from slm import TransformerLM

model = TransformerLM.from_pretrained("tiagoCuervo/gslm-scaling-85m-108p9b", device="cuda")
prompt = torch.tensor([[12, 91, 204]], device="cuda")
units = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=50)
```

[notebooks/generate_audio.ipynb](notebooks/generate_audio.ipynb) loads a trained or Hugging Face
model, generates units, decodes them with CodeHiFiGAN, and plays or saves the waveform. Convert a
training checkpoint with:

```bash
slm-convert out/run/checkpoint.pt out/run/native
```

## Citation

```bibtex
@inproceedings{cuervo-marxer-2024-scaling,
  title     = {Scaling Properties of Speech Language Models},
  author    = {Cuervo, Santiago and Marxer, Ricard},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year      = {2024},
  address   = {Miami, Florida, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {351--361},
  doi       = {10.18653/v1/2024.emnlp-main.21},
  url       = {https://aclanthology.org/2024.emnlp-main.21/}
}
```
