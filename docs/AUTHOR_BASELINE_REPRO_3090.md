# Author baseline reproduction on 4×RTX 3090

This branch is a clean reproduction fork from `b9a158f75ab0740c91c9cfc8863a65fa381b014c`.
It does not contain the Phase 1B prior code. The only source changes are the
minimum Hydra compatibility fix for the unused `dataset.seq_len` interpolation,
the dead cleanup reference that otherwise aborts the first diffusion loss,
worker/preflight guards, 4-GPU topology settings, deterministic seed 42, and a
visible-GPU check. None changes the network architecture or loss formula.

The author configuration uses 4×A100 with a per-GPU batch of 512, hence a global
batch of 2048. The current candidate uses 4×RTX 3090 with a per-GPU batch of
512, retaining the same topology and global batch. This batch must pass a fresh
4-GPU memory smoke before a long run; the earlier 8-GPU/batch-256 smoke remains
historical evidence only. Outputs are kept in two separate directories under
`results/author_b9a_*`.

Use the verified environment and run from `code/`:

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3

"$INFBAGEL_PYTHON" train_infbagel.py \
  num_gpus=4 batch_size=512
```

The diffusion run writes the checkpoint consumed by consistency distillation.
After the final `epoch500` checkpoint exists, run:

```bash
"$INFBAGEL_PYTHON" train_infbagel.py \
  --config-name config_train_infbagel_cm \
  num_gpus=4 batch_size=512 \
  ckpt_path=/data/yujinlun/InfBaGel-release/results/author_b9a_infbagel_4x3090/checkpoints/author_b9a_infbagel_4x3090_epoch500.pth
```

The distillation command must name the diffusion checkpoint explicitly;
`config_train_infbagel_cm.yaml` intentionally keeps `ckpt_path` empty so a
stale checkpoint cannot be loaded silently.

For a one-update preflight only, add `max_optimizer_updates=1` and use a new
output name. The default is `null`, so the two commands above retain the
author's complete epoch-based schedule.

The source fixes make the released code runnable: removing `dataset.seq_len` is
safe because the sampler never reads it, and removing the `occ_goal`/
`occ_temp` cleanup is safe because those locals are never created or returned.
The `t` shape change is equivalent for the normal full batches. Explicit seed
42 is a repository reproducibility policy and does change the random trajectory
relative to the author's previously unseeded run; exact bitwise checkpoint
identity is not expected across RTX 3090 and A100 hardware.
