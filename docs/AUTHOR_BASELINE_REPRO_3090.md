# Author baseline reproduction on 8×RTX 3090

This branch is a clean reproduction fork from `b9a158f75ab0740c91c9cfc8863a65fa381b014c`.
It does not contain the Phase 1B prior code. The only source changes are the
minimum Hydra compatibility fix for the unused `dataset.seq_len` interpolation,
the dead cleanup reference that otherwise aborts the first diffusion loss,
worker/preflight guards, 8-GPU topology settings, deterministic seed 42, and a
visible-GPU check. None changes the network architecture or loss formula.

The author configuration uses 4×A100 with a per-GPU batch of 512, hence a global
batch of 2048. The current candidate uses 8×RTX 3090 with a per-GPU batch of
256, retaining the global batch. The attempted 4-GPU/batch-512 smoke exhausted
24GB on GPU 0 during occupancy construction; the earlier 8-GPU/batch-256 smoke
passed. Outputs are kept in two separate directories under
`results/author_b9a_*`.

The dataset keeps scene occupancy tensors on the GPU. The training code now
gives DataLoader workers a CPU-only shallow view and starts them with the
`spawn` context, so the default `num_workers=2` can prefetch without inheriting
the rank's CUDA address space. If a different host still reports CUDA IPC/OOM
during loading, override `num_workers=0`; this affects loading throughput and
memory, not the model objective.

The occupancy cache is stored as `int8`, matching the loss-time occupancy
representation. This avoids creating a second full batch-sized occupancy grid
when converting a boolean selection, reducing the peak allocation without
changing occupancy values or the loss.

Use the verified environment and run from `code/`:

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

"$INFBAGEL_PYTHON" train_infbagel.py \
  num_gpus=8 batch_size=256
```

The diffusion run writes the checkpoint consumed by consistency distillation.
After the final `epoch500` checkpoint exists, run:

```bash
"$INFBAGEL_PYTHON" train_infbagel.py \
  --config-name config_train_infbagel_cm \
  num_gpus=8 batch_size=256 \
  ckpt_path=/data/yujinlun/InfBaGel-release/results/author_b9a_infbagel_8x3090/checkpoints/author_b9a_infbagel_8x3090_epoch500.pth
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
