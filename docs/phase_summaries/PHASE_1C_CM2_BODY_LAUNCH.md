# Phase1C CM2.2: body endpoint student stable launch

The stable-launch gate completed on 2026-09-11. Formal training is running in a
persistent process. Phase1C stays open; corrected R2+CG remains the quality baseline.
The user's approval covers this one student training and the fixed CM2.3 acceptance.

## Scope and implementation

CM1's existing objectives and training recipe gain one low-noise term. For sampled
timesteps19/39/59, a separate frozen eval R2 teacher follows1–3 remaining DDIM steps
from the same noisy state, history and initial scene crops. Its endpoint supervises
the student's absolute body22 FK positions, including root, on all14 future frames.
The squared xyz error is averaged over the full batch, including zero high-noise
rows. The original train-mode teacher/EMA target, dropout, hand/foot GT objective,
EMA0.95 and random stream remain as CM1. Extra teacher forward and FK use fp32 with
the registered training TF32 settings. Inference stays CM16.

The HSI objective component reuses native occupancy and DDIMSolver. Temporal crop
selection preserves batch-block ordering. A separate teacher/sampler and local RNG
scope isolate the auxiliary computation. The trainer uses its checkout's absolute
ROOT_DIR and its resume contract includes the coefficient, cutoff and teacher path.
Core and other-expert paths were unchanged.

## Calibration and resource admission

The registered first8×256 real batch completed with zero optimizer updates and
finite losses/gradients. The rotation-head25% limit selected
lambda=1.2518757249916617; the trunk10% limit was1.3739437403729367. No sweep.
The initial trunk cosine is−0.1873 against consistency and+0.1312 against the old
total objective; the calibration does not establish long-term optimization behavior.
All rank values and the scale table are in the calibration report.

The160-update full-batch measurement discarded32 warmup updates and synchronized
the remaining128:1.125550seconds/update,1819.56windows/second. All1280 gradient
records were finite and identical across ranks; required trunk/cfg gradients were
present. All128 endpoint-loss records were finite. Peak allocated/reserved memory
was6.0820/6.9551GiB; sampled headroom including external GPU0 use was9301MiB.

Preparation cost0.735828GPU-h. Benchmark-based formal training forecast146.766697
GPU-h, plus8GPU-h evaluation reservation, totals155.502525<160. The initial formal
wall rate gives a156.861GPU-h projection on the same basis. Initialization,
checkpoint IO and later contention can change elapsed cost. No teacher serial
deployment latency was measured.

## Formal run and stable prefix

Run: p1-hsi-cm2-body-train-s42-20260911.
Student/teacher/EMA target initialize from sealed R2; optimizer/RNG start fresh.
8×RTX3090, micro256, accumulation1, effective2048, seed42, Adam/lr2e-4/warmup2000,
clip1, bf16_tf32;58678 updates/120172544 windows,90 epoch cap, final epoch089 only.

The first160 updates'1280 per-rank diagnostic records exactly match the independent
benchmark, maximum difference0. The first epoch completed656 updates and wrote a
complete resume checkpoint:218 student tensors,218 EMA target tensors,102 optimizer
parameter states and8 rank RNG states. Strict student/target/optimizer loads and
resume compatibility passed. The epoch000 export equals the resume student state
exactly; all student/target tensors are finite.

Stable-prefix audit covers the first640 updates, within the2000-update warmup:
5120 gradient records,5120 cfg records and512 endpoint-loss records are finite.
Rank gradients agree, required gradients exist, norm min/median/max is
0.00725411/0.14420460/0.43926346; this prefix has no clipping. Sampled training
headroom is9301MiB. Observed wall rate1.135965seconds/update gave18.23 remaining
hours at the snapshot and completion near2026-09-12 06:17 CST. This is a startup
estimate, not a quality result or a guarantee of the later full-learning-rate phase.

## Execution ownership and artifacts

Training runs from the clean detached worktree
/data/yujinlun/InfBaGel-hsi-cm2-training at e6185a1. Data/results/kinematic assets
use local links; authority-side documentation commits do not alter execution source.
The persistent launcher finishes the manifest from that pinned worktree when the
training process exits. Keep the execution worktree while the run remains active.

- experiments/results/p1_hsi_cm2_body_calibration_s42_20260911.{md,json}.
- experiments/results/p1_hsi_cm2_body_launch_s42_20260911.{md,json}.
- results/cm2_body_setup_20260911/:resolved jobs, config comparison, resource admission,
  source-worktree record and execute_job.sh.
- results/experiments/p1-hsi-cm2-body-train-s42-20260911/:manifest.json,
  config_resolved_job.yaml, preflight.json, run.log, launcher.pid, workload.pid,
  initial_reproducibility.json, initial_resume_validation.json, stable_launch.json,
  next_entry.json and eventual exit_status/training_metrics.json.
- results/p1-hsi-cm2-body-train-s42-20260911/:all loss/gradient logs and checkpoints.

Preregistration edf0c6b; implementation985e5be; measured coefficient/execution source
e6185a1. This handoff is the stable-launch completion commit. No merge or expert tag.

Targeted66 passed; authority487 passed/3 skipped in86.94seconds; registry423 valid.
Each job was resolved through the actual Hydra trainer entry. Calibration and
benchmark exited0; calibration's CUDA-context shutdown warnings remain in its log.
There was no reportable workload failure. A provisional unused scratch link made
the new worktree appear dirty; it was removed before the formal start. Data links
and scientific inputs were preserved. Documentation-only handoff needs no repeated
model tests. Validation used canonical INFBAGEL_PYTHON with pytest component/full
authority commands and tools/experiment.py validate.

## Exact next entry

Read this summary, docs/plan/OVERVIEW.md and the approved CM2.2/2.3 section of
PHASE_1C_HSI.md. Next inspection is user-triggered or follows a separately configured
failure notification. Check manifest, exit_status and final update count first.

The fixed internal checkpoint is
results/p1-hsi-cm2-body-train-s42-20260911/checkpoints/p1-hsi-cm2-body-train-s42-20260911_epoch004.pth;
final is the same directory's epoch089 file. Execute the already-approved CM2.3:
fixed60 development U/G, final375 U/G with2271 windows each, corrected deployment
interpolation, full native physical/contact/safety and representation metrics,
FID/Diversity/MM-Dist/R@1/2/3 with fixed uncertainty, and student serial latency70.
Reuse corrected R2/CM1 controls and teacher latency. No checkpoint selection or
budget change; retain failures and uncertain metrics. Quality improvement, transfer
to generated history, semantic coupling and the external-CG gap remain unproven.
