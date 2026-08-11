# Expected results

Scores use mean unit NLL per
candidate continuation without a terminal EOS. sBLIMP macro accuracy averages the 12 linguistic
phenomena; voice-pair accuracy requires both voices of a minimal pair to be correct. StoryCloze
uses 1,871 two-choice questions per task.

The scaling suite contains final checkpoints from distinct completed training budgets, not
periodic checkpoints from within one run. The selection matches the paper scaling notebook:
planned budgets above 5,000 updates, at most one corpus epoch, with duplicate runs removed.

| Model | Tokens/update | Tokens seen | sBLIMP macro | sBLIMP pair | sStoryCloze | tStoryCloze |
|---|---:|---:|---:|---:|---:|---:|
| `gslm-scaling-20m-0p7b` | 131,200 | 662,560,000 | 53.43 | 51.81 | 52.32 | 63.98 |
| `gslm-scaling-20m-1p3b` | 131,200 | 1,325,120,000 | 53.66 | 51.56 | 52.54 | 65.53 |
| `gslm-scaling-20m-2p1b` | 131,200 | 2,066,400,000 | 54.50 | 52.28 | 50.88 | 65.63 |
| `gslm-scaling-20m-21p8b` | 131,200 | 21,772,640,000 | 56.21 | 53.80 | 53.50 | 69.00 |
| `gslm-scaling-20m-43p5b` | 131,200 | 43,545,280,000 | 56.20 | 54.00 | 52.65 | 70.66 |
| `gslm-scaling-20m-65p3b` | 131,200 | 65,324,480,000 | 57.07 | 54.53 | 53.55 | 69.96 |
| `gslm-scaling-20m-87p1b` | 131,200 | 87,097,120,000 | 57.15 | 54.33 | 53.34 | 70.55 |
| `gslm-scaling-20m-108p9b` | 131,200 | 108,869,760,000 | 57.15 | 54.64 | 53.45 | 70.55 |
| `gslm-scaling-85m-1p4b` | 262,400 | 1,364,480,000 | 55.04 | 52.59 | 51.84 | 67.50 |
| `gslm-scaling-85m-2p7b` | 262,400 | 2,728,960,000 | 56.41 | 53.65 | 52.22 | 68.36 |
| `gslm-scaling-85m-5p5b` | 262,400 | 5,457,920,000 | 57.88 | 54.96 | 53.39 | 69.96 |
| `gslm-scaling-85m-8p5b` | 262,400 | 8,528,000,000 | 57.72 | 54.77 | 52.75 | 70.98 |
| `gslm-scaling-85m-43p5b` | 262,400 | 43,545,280,000 | 58.90 | 56.04 | 54.57 | 72.37 |
| `gslm-scaling-85m-65p3b` | 262,400 | 65,324,480,000 | 59.08 | 56.54 | 55.53 | 74.67 |
| `gslm-scaling-85m-87p1b` | 262,400 | 87,090,560,000 | 59.34 | 56.20 | 55.32 | 74.13 |
| `gslm-scaling-85m-108p9b` | 262,400 | 108,869,760,000 | 59.04 | 56.12 | 56.01 | 74.13 |
| `gslm-scaling-155m-3p1b` | 524,800 | 3,070,080,000 | 56.71 | 53.85 | 52.81 | 70.02 |
| `gslm-scaling-155m-4p9b` | 524,800 | 4,933,120,000 | 57.32 | 54.59 | 53.71 | 70.76 |
| `gslm-scaling-155m-9p9b` | 524,800 | 9,892,480,000 | 58.42 | 55.50 | 54.30 | 71.78 |
| `gslm-scaling-155m-19p4b` | 524,800 | 19,417,600,000 | 59.20 | 56.29 | 53.98 | 73.60 |
| `gslm-scaling-155m-43p5b` | 524,800 | 43,532,160,000 | 59.69 | 56.78 | 55.53 | 73.54 |
| `gslm-scaling-155m-65p3b` | 524,800 | 65,311,360,000 | 58.77 | 55.88 | 54.84 | 75.09 |
| `gslm-scaling-309m-3p0b` | 516,600 | 3,022,110,000 | 57.58 | 54.69 | 54.20 | 71.19 |
| `gslm-scaling-309m-4p9b` | 516,600 | 4,933,530,000 | 58.61 | 55.80 | 54.94 | 72.47 |
| `gslm-scaling-309m-9p9b` | 516,600 | 9,867,060,000 | 59.57 | 56.91 | 54.94 | 73.49 |
| `gslm-scaling-309m-19p8b` | 516,600 | 19,759,950,000 | 60.33 | 57.46 | 55.69 | 74.93 |
| `gslm-scaling-309m-30p4b` | 516,600 | 30,401,910,000 | 60.06 | 57.41 | 54.41 | 75.31 |
| `gslm-scaling-823m-8p3b` | 934,800 | 8,272,980,000 | 59.75 | 56.81 | 55.85 | 73.49 |
| `gslm-scaling-823m-16p5b` | 934,800 | 16,499,220,000 | 61.08 | 58.08 | 55.21 | 75.52 |

sBLIMP is reported on the official labelled development set (25,200 questions / 6,300 voice
pairs).

Two complete extended-budget 823M endpoints used in the paper analysis are also available:

| Model | Exact tokens/update | Tokens seen | sBLIMP macro | sBLIMP pair | sStoryCloze | tStoryCloze |
|---|---:|---:|---:|---:|---:|---:|
| `gslm-scaling-823m-26b` | 1,012,700 | 26,330,200,000 | 61.10 | 58.10 | 56.28 | 76.70 |
| `gslm-scaling-823m-82b` | 1,246,400 | 82,324,720,000 | 61.33 | 58.36 | 56.92 | 77.93 |

## Stable A100 MFU

| Model | Tokens/update | MFU |
|---:|---:|---:|
| 20M | 131,200 | 31.97% |
| 85M | 262,400 | 36.52% |
| 155M | 524,800 | 41.95% |
| 309M | 516,600 | 43.60% |
| 823M curve | 934,800 | 62.52% |
| 823M/26B | 1,012,700 | 50.58% |
| 823M/82B | 1,246,400 | 62.57% |

The values are medians of raw MFU over the longest steady post-warmup
segment of representative completed runs. Measurements use one A100 80GB PCIe, BF16
autocast, FP32 parameters and optimizer state, compilation, and the exact batch arithmetic shown above. MFU uses the PaLM/nanoGPT `6N + attention` estimate and a 312-TFLOP/s BF16 peak.
