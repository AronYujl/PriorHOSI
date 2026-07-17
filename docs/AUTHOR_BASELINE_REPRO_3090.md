# Author baseline reproduction on 8×RTX 3090

This branch is a clean reproduction fork from `b9a158f75ab0740c91c9cfc8863a65fa381b014c`.
It does not contain the Phase 1B prior code. The only source changes are the
minimum Hydra compatibility fix for the unused `dataset.seq_len` interpolation,
8-GPU topology settings, deterministic seed 42, and a visible-GPU check.

The author configuration uses 4×A100 with a per-GPU batch of 512, hence a global
batch of 2048. This port uses 8×RTX 3090 with a per-GPU batch of 256, retaining
the same global batch without gradient accumulation. Outputs are kept in two
separate directories under `results/author_b9a_*`.

Use the verified environment and run from `code/`:

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

"$INFBAGEL_PYTHON" train_infbagel.py
```

The diffusion run writes the checkpoint consumed by consistency distillation.
After the final `epoch500` checkpoint exists, run:

```bash
"$INFBAGEL_PYTHON" train_infbagel.py \
  --config-name config_train_infbagel_cm \
  ckpt_path=/data/yujinlun/InfBaGel-release/results/author_b9a_infbagel_8x3090/checkpoints/author_b9a_infbagel_8x3090_epoch500.pth
```

The distillation command must name the diffusion checkpoint explicitly;
`config_train_infbagel_cm.yaml` intentionally keeps `ckpt_path` empty so a
stale checkpoint cannot be loaded silently.

For a one-update preflight only, add `max_optimizer_updates=1` and use a new
output name. The default is `null`, so the two commands above retain the
author's complete epoch-based schedule.
