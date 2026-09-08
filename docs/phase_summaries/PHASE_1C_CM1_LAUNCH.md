# Phase 1C-CM1.1: R2 fixed-CFG consistency training launch

Completed the stable-launch gate on 2026-09-08. Formal training is running;
quality is unmeasured, Phase 1C is open, and R2+CG remains the working baseline.

## Scope and implementation

Branch: `phase/01c-cm1`, from `phase/01c-hsi`. Preregistration: `4669d33`;
execution source: `a9319f3`. One config fragment extends the existing CM recipe.
Teacher, student and EMA target initialize from the sealed R2 final EMA; optimizer
and RNG start fresh. Fixed requested CFG is 1, including timestep499. Temporal
scene masking at499 is per sample; explicit unconditional requests retain -1.
Teacher/student/target use the same per-sample masking rule. Occupancy E1 is on.

Consistency supervision excludes object/contact channels for scene-only rows.
FK uses the existing fp32 geometry computation. The new CM guidance option maps a
clean-space increment through its actual re-noising coefficient sqrt(alpha_prev).
It defaults off for historical CM configs. The global CFG bug repair changes
historical CM behavior; use historical source when reproducing those checkpoints.
R2 batch1 diffusion retains its conditioning behavior. Core is unchanged.

Teacher and target retain the prior train-mode dropout. EMA rate0.95, trainable
modules, Adam2e-4/warmup2000, clipping1 and loss weights1/1 are retained. No new
seam loss, architecture, w sweep or geometry-distillation target was introduced.
The student is trained at one operating point; the geometric gradient remains
external at inference.

## Resource and stability evidence

8 RTX3090, microbatch256, effectivebatch2048, accum1, bf16/TF32 with fp32 geometry,
OMP_NUM_THREADS4, seed42. Budget120,172,544 windows =58,678 updates. The independent
160-update benchmark completed:32 warmup,128 measured updates, CUDA-synchronized
58.4823s /0.456893s per update,6.908GiB peak allocator reserved. Formal initialization
reproduced all1,280 initial per-rank diagnostic records exactly.

The registered3,280-update stability interval completed. All26,240 diagnostics
are finite with nonzero trunk/CFG gradients. Loss median0.00109463; post-warmup
median0.00110665, gradient median0.272657/max0.847555. Onlyupdate1462 clipped
(preclip1.051382). These are operational observations, not native quality evidence.

Epoch004 is saved for the fixed internal diagnostic. Its resume state passed the
existing compatibility check and includes student/target218 tensors each,
102 Adam parameter states and all8 RNG states. No run failed. Implementation
fixture errors and their corrections are recorded in the phase plan.

At16:26 the remaining-time estimate was7.02h, ending around23:28 CST. GPU0 acquired
an external2020MiB process at16:26:13; the minimum observed free memory remained
11175MiB. This competition can extend the estimate and is recorded in the compact.

## Verification

Canonical infbagel Python throughout. Authority suite:444 passed,3 skipped,2 new
fixture failures; correcting the aliasing fixture and rerunning the component
produced6/6 passes, for446 passing distinct checks and3 skips. Production source
was unchanged by the fixture correction. Targeted resume/guidance tests passed.
Registry validation and resolved-config preflight passed. Real execution was the
registered full-batch benchmark; no separate smoke or new hashing tooling.

## Artifacts and continuing work

Run: `p1-hsi-cm1-r2-fixed-w-s42-20260908`.
Manifest/config/preflight/log/exit status/next-entry commands:
`results/experiments/p1-hsi-cm1-r2-fixed-w-s42-20260908/`.
Checkpoints and authoritative per-rank JSONL:
`results/p1-hsi-cm1-r2-fixed-w-s42-20260908/`.
Benchmark: `results/experiments/p1-hsi-cm1-benchmark-s42-20260908/`.
Compact: `experiments/results/p1_hsi_cm1_launch_s42_20260908.json`.
Immutable input identities remain referenced through the existing R2 and benchmark
manifests. Persistent session1179173 /trainer1179174 owns the running process.

Stop continuous polling after this gate. On the next user-triggered inspection,
read this summary, OVERVIEW.md, the2026-09-08 phase-plan section and next_entry.json.
Check exit status and training metrics, retain the terminal result/failure, and
complete CM1.2: fixedepoch004 generated-history internal60 (364 windows), final
58,678-updateepoch089 full375 U/G, complete Table3 metrics and paired uncertainty,
and uncontended single-GPU latency70. Reuse R2 U/CG quality outputs; reuse matching
teacher latency if available or measure that same latency subset for acceleration.
The prepared commands are not running. No best-checkpoint selection, w tuning,
next-phase merge, release tag or model promotion has occurred.
