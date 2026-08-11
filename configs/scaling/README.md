# Scaling-study endpoints

The paper describes batch sizes approximately; the table below records the exact sequence and
token math used by each configuration. `gradient_accumulation_steps` is the global accumulation count and must
be divisible by the DDP world size.

| Config | Checkpoint step | Sequences/update | Tokens/update | Tokens seen |
|---|---:|---:|---:|---:|
| `20m-109b.json` | 829,800 | 64 | 131,200 | 108,869,760,000 |
| `85m-109b.json` | 414,900 | 128 | 262,400 | 108,869,760,000 |
| `155m-65b.json` | 124,450 | 256 | 524,800 | 65,311,360,000 |
| `309m-30b.json` | 58,850 | 252 | 516,600 | 30,401,910,000 |
| `823m-26b.json` | 26,000 | 494 | 1,012,700 | 26,330,200,000 |
| `823m-82b.json` | 66,050 | 608 | 1,246,400 | 82,324,720,000 |

All use 2,050-token contexts, 25 Hz layer-11 mHuBERT units, K=500, EOS 500, and padding token
501. Point `train_data` and `val_data` at canonical shards made from the mixture in `DATA.md`.
The 823M/82B run is the extended best model; 823M/26B is the endpoint used in the compute-scaling
comparison.
