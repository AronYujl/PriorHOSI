# Author InfBaGel reproduction plan

This branch is the isolated author-baseline reproduction fork from
`b9a158f75ab0740c91c9cfc8863a65fa381b014c`. It is not a HOIPrior/HSIPrior
implementation branch. The reference training schedule remains the author's
FP32 objective and epoch count; performance changes are introduced only as
explicit, separately named variants.

## Phase 1B author reproduction: throughput recovery

### 2026-07-17 — mixed-precision throughput smoke

Hypothesis: on Ampere RTX 3090 GPUs, autocast FP16 with GradScaler and TF32
matmuls will reduce the per-update cost of the transformer, ViT and kinematic
matrix operations by at least 1.5x while keeping losses finite and avoiding
OOM. The default `precision: fp32` path remains unchanged for fidelity checks;
the mixed-precision path is an explicitly registered speed variant and is not
claimed bitwise-equivalent to the author run.

The gate is a new, non-overwriting 10-update diffusion smoke followed by a
1-update CM smoke from that smoke checkpoint. Both must complete with finite
losses, required checkpoints, no OOM, and measured warm throughput. A full
training run is not authorized by this smoke alone. If the speed gate is not
met, the next optimization must be preregistered separately.

