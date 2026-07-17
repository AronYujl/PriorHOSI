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

Result: the gate failed. The AMP diffusion smoke took 365.34 seconds for ten
updates versus 339.88 seconds for the latest FP32 smoke, and its reported loss
scale changed from approximately 155--168 to 482--519. The one-update AMP CM
smoke took 412.30 seconds and is dominated by startup. AMP checkpoints are not
eligible for the author-reproduction run.

### 2026-07-17 — lazy training assets and synchronized timing

Hypothesis: the smoke wall time is dominated by one-time input initialization,
because each of eight DDP ranks eagerly reads and concatenates approximately
9.2 GiB of `object_points.npy` and 9.4 GiB of per-object BPS arrays before its
DataLoader workers start. Read-only NumPy memory maps opened lazily inside each
worker, with a copied current sample, must preserve the exact training values
and random-frame distribution while eliminating redundant eager reads and
rank-local copies. Returning only fields consumed by the training step must not
change the objective.

Add opt-in CUDA-synchronized segmented timing so startup, data wait, host-to-
device transfer, loss computation, backward/DDP synchronization, and optimizer
work are measured separately. Timing synchronization is enabled only for a
non-reportable profiling smoke and is excluded from production throughput.

The gate is a new, non-overwriting ten-update FP32 diffusion smoke with two
warmup updates. It must complete with finite losses and a checkpoint, report
rank-maximum stage timings, preserve the existing effective batch of 2048, and
pass exact synthetic lazy-loader regression tests. A CM smoke and any further
optimization are selected only after the warm timing identifies the dominant
stage; this change alone does not authorize a full run.

Result: the diffusion profiling gate passed. Across eight measured warm updates,
the rank-maximum synchronized update averaged 1.0967 seconds, with 0.0009
seconds of exposed data wait, 0.0014 seconds of host-to-device transfer, 0.7676
seconds of loss computation, 0.3742 seconds of backward/DDP work, and 0.0229
seconds of optimizer work. Rank-maximum CUDA memory was 8.565 GiB allocated and
8.656 GiB reserved. The full ten-update process took 52.90 seconds; one-time
dataset initialization took 12.82 seconds and first-batch wait took 12.29
seconds. At the profiled rate the 145,791-update diffusion schedule is about
44.4 hours, before a non-profile throughput confirmation.

The CUDA IPC/driver messages emitted after the checkpoint and profile summary
are classified as shutdown-cleanup warnings, not a training failure. Add an
explicit worker/process-group teardown before the next smoke so the formal run
does not leave worker resources pending at normal completion.

### 2026-07-17 — consistency-model warm timing

Hypothesis: after the same lazy-asset fix, a ten-update FP32 CM smoke with two
warmup updates will provide a valid warm-step and memory estimate without OOM.
The gate requires finite losses, a checkpoint, rank-maximum stage timings and
peak memory, and clean process teardown. A full CM run remains unauthorized
until its measured time is combined with the diffusion estimate.

