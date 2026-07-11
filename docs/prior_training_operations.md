# Independent Prior Training Operations

## Recommended 8-GPU layout

The two priors are independent DDP jobs. HOI uses physical GPUs 0-3 and HSI
uses physical GPUs 4-7. Their training batch sizes do not need to match for
composition: composition consumes denoiser outputs at inference time and has no
dependency on the batches used to estimate each prior.

Recommended main-experiment settings on four 24 GiB RTX 3090 GPUs per prior:

| Prior | Per-GPU batch | Global batch | Gradient accumulation |
| --- | ---: | ---: | ---: |
| HOI | 512 | 2048 | 1 |
| HSI | 320 | 1280 | 1 |

Use validation-selected checkpoints for composition. Record optimizer updates
and processed windows in addition to epochs, because the datasets and batches
are intentionally different. Different training batches can change score
calibration indirectly through optimization quality, so tune composition
weights on a held-out HOSI validation set rather than assuming equal weights.

## RTX 3090 batch benchmark

Each measurement used four GPUs, AMP, 30 train batches, and one validation
batch. HOI and HSI candidates ran concurrently, matching the intended server
load. Short-run throughput includes CUDA/DataLoader warm-up and is therefore a
conservative estimate for a full epoch.

| Prior | Per-GPU batch | Global batch | Train samples/s | CUDA reserved peak |
| --- | ---: | ---: | ---: | ---: |
| HOI | 128 | 512 | 249.5 | 0.85 GiB |
| HOI | 256 | 1024 | 345.3 | 1.31 GiB |
| HOI | 512 | 2048 | 535.9 | 2.15 GiB |
| HOI | 1024 | 4096 | 1120.9 | 3.90 GiB |
| HSI | 64 | 256 | 217.1 | 8.59 GiB |
| HSI | 128 | 512 | 429.8 | 10.93 GiB |
| HSI | 256 | 1024 | 611.7 | 15.79 GiB |
| HSI | 320 | 1280 | 736.3 | 18.23 GiB |

`nvidia-smi` reported about 20.1 GiB process memory for HSI batch 320, leaving
about 4.4 GiB on a 24 GiB card. HSI batch 384 was not attempted because the
projected margin was too small for a long run. HOI batch 1024 fits easily, but
it halves optimizer updates per epoch relative to batch 512. It is a
throughput-oriented ablation, not the default main experiment, until large-
batch convergence and learning-rate scaling are validated.

## ETA

The training process now prints cumulative throughput, elapsed time, and ETA:

```text
epoch=003 step=0001040 run_step=0001040/0065361 loss=... samples/s=... elapsed=... eta=...
```

The first 10-30 updates include worker and CUDA warm-up, so the ETA initially
fluctuates. It becomes useful after roughly 100 updates. Epoch summaries include
validation and expose a second ETA based on complete epoch wall time.

Conservative projections from the short benchmark, with both jobs running:

| Prior | Configuration | Approx. train time/epoch | 501-epoch order of magnitude |
| --- | --- | ---: | ---: |
| HOI | 4 x 512 | 14-17 min | 5-6 days |
| HSI | 4 x 320 | 23-50 min | 8-18 days |

The ranges are intentionally broad because 30-batch warm-up dominates the
short benchmark. Use the live ETA after the first full epoch as the operational
estimate. Since HSI has about 2.19 million train windows, it dominates total
wall time. Treat 501 as a maximum and select the best validation checkpoint;
do not assume both priors need the same epoch count to be composition-ready.

## Launch and records

Launch both formal jobs from the repository root with a unique experiment tag:

```bash
./scripts/train_independent_priors.sh prior_main_seed2027
```

Optional batch overrides are the second and third arguments:

```bash
./scripts/train_independent_priors.sh prior_ablation_bs 1024 256
```

The launcher refuses to reuse a tag whose run manifest already exists. Files
are organized as follows:

```text
results/priors/<tag>/hoi_prior/
  resolved_config.yaml
  run_manifest.json
  metrics.jsonl
  train.log
  tensorboard/
  checkpoints/

results/priors/<tag>/hsi_prior/
  ...

results/priors/logs/<tag>/
  launcher.log
  hoi_terminal.log
  hsi_terminal.log
```

`run_manifest.json` records the command, Git commit, host, CUDA/PyTorch
versions, visible GPU IDs, batch definitions, planned updates, and data
contract. `metrics.jsonl` is machine-readable step/epoch history. `train.log`
contains rank-0 progress. The two terminal logs are the complete stdout/stderr
streams, including tracebacks and NCCL/DataLoader warnings. TensorBoard remains
available for interactive monitoring.

For a paper table, archive the experiment tag, Git commit, best checkpoint
epoch/step, effective global batch, processed windows, validation metric,
runtime, and evaluation JSON. Do not compare runs only by nominal epochs.
