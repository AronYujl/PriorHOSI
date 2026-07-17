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

Result: the CM compute and memory gates passed, but the clean-teardown gate did
not. Across eight measured warm updates, rank-maximum synchronized CM update
time averaged 2.5110 seconds, with 2.3226 seconds in loss computation, 0.2668
seconds in backward/DDP, and less than 0.003 seconds in exposed data wait plus
host-to-device transfer. Rank-maximum CUDA memory was 4.237 GiB allocated and
4.373 GiB reserved. The complete smoke took 70.78 seconds.

The 58,491-update CM schedule is estimated at 40.8 hours. Combined with the
44.4-hour diffusion estimate, author diffusion plus consistency distillation is
approximately 85.2 hours (3.55 days). The recurring terminal CUDA IPC warnings
remain a non-fatal nested-multiprocessing cleanup issue because they appear only
after the checkpoint and synchronized summary; do not claim the cleanup fix
succeeded. Throughput optimization is no longer the gate for reproduction.

### 2026-07-17 — four-GPU automatic author pipeline

The selected author-topology reproduction uses physical GPUs 4--7, which are
idle, share NUMA node 1, and have PIX connectivity to one another. Restore the
author's per-GPU batch of 512 on four RTX 3090s, retaining the registered global
batch of 2048, FP32 precision, learning rate, seed and epoch schedules.

Use one persistent shell pipeline that first performs non-overwriting ten-update
diffusion and CM preflights at the exact four-GPU batch. A failed process,
missing checkpoint or non-finite failure must stop the chain. If both commands
exit successfully and write their checkpoints, start the 501-epoch diffusion
run; after `epoch500.pth` is verified, pass that exact path directly to the
201-epoch consistency run. Never use the preflight diffusion checkpoint for the
full CM run.

This convenience pipeline refuses a dirty worktree and locks its commit, but is
not a reportable run because this isolated branch does not contain the mandated
`tools/experiment.py` manifest launcher. Its checkpoints may be used to test
author metrics; they must not be represented as governed final-table artifacts.

Result: superseded before launch. No four-GPU pipeline result directory or
persistent session was created. The hardware choice returned to the already
profiled eight-GPU topology to reduce wall time while preserving effective
batch 2048.

### 2026-07-17 — eight-GPU automatic author pipeline

Use all eight idle RTX 3090 GPUs with per-GPU batch 256, effective batch 2048,
FP32 precision, seed 42, learning rate 1e-4, 501 diffusion epochs and 201
consistency epochs. This changes only data-parallel topology relative to the
superseded four-GPU convenience pipeline; the training objective and processed
window budget remain fixed.

The exact eight-GPU Diffusion and CM configurations have already passed finite
loss, checkpoint, synchronized timing and memory gates in
`p1b-author-throughput-mmap-profile-r1-s42-20260717` and
`p1b-author-cm-mmap-profile-r1-s42-20260717`. Therefore the persistent full
pipeline starts Diffusion directly and, only after the process exits
successfully and the exact `epoch500.pth` exists, hashes that checkpoint and
passes it to the CM run. CM must similarly finish at `epoch200.pth`. The chain
refuses a dirty worktree, locks the launched commit, archives resolved Hydra
configs, Git/hardware snapshots, commands, stage logs and checkpoint hashes,
and never reuses an existing run directory.

The first full Diffusion epoch is the live stability interval. Require finite
losses through that interval, a successful epoch-0 model-weight checkpoint and
at least 4 GiB memory headroom on every GPU before ending active polling. The
author script's checkpoints contain model weights only, not optimizer or RNG
state; they are valid direct inputs to CM but are not strict mid-training resume
artifacts. This limitation is recorded rather than changing checkpoint
semantics immediately before reproduction.

As with the superseded convenience pipeline, this run is exploratory and not
eligible for final-table reporting because this isolated author branch lacks
`tools/experiment.py`. Its purpose is to determine whether the author metrics
can be reproduced and to produce compatible Diffusion/CM weights without
mixing outputs into the prior-development branch.
