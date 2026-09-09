# Phase 1C CM1.4: steps, deployed representation, and FID reference

Completed 2026-09-09 on phase/01c-cm1. All six registered workloads completed;
Phase 1C remains open. Retain CM16 and R2+CG as fixed comparisons. Prioritize
teacher and deployment repair before another distillation training.

## Scope and implementation

- Fixed CM1 final epoch089, CFG1 and geometry scale1; compared 16 and 25 steps.
- One config fragment: code/config/config_sample_hsi_cm1_steps25.yaml.
- Added priors/hsi/representation_consistency.py and position_fk /
  merge_position_fk modes in the existing LINGO evaluator. Rebuilt nine arms,
  375 sequences each, from their exported SMPL-X parameters on eight GPUs.
- Matched direct-position joint indices, separating 21 nonroot body joints,
  root translation, 28-joint errors and seam/interior regions. Frozen core,
  native FID, physical metrics and interpolation arithmetic stayed unchanged.
- Preregistration commit 2d7407a; execution commit 9927abf. This completion
  commit adds the report, compact, registry entries and handoff. No merge or tag.

## Results and retained negatives

CM25 versus CM16 did not resolve primary physical deficits. U boundary jerk
156.309→155.155, delta -1.15354, CI [-2.75066,0.40060]; G 165.533→167.906,
delta +2.37333, CI [-0.94811,5.90684]. All eight primary simultaneous intervals
cross zero. Guided exterior contact declines at pointwise 95%, delta -8.92433,
CI [-16.98392,-1.67103]; the simultaneous interval crosses zero. All native
metrics, including secondary penetration improvements, are retained.

FID U 20.19524→23.01966, delta +2.82441, CI [0.80394,5.26486]; G
22.90380→23.65130, delta +0.74751, CI [-1.51791,2.83127]. Both R@3 changes
are uncertain. Generation cost grows 57.72% U / 38.66% G. CM25 serial FPS
is 83.7322 U / 13.9159 G; generation seconds/sequence 1.79427 / 10.76676,
end-to-end 1.86635 / 10.84758. Timing uses 19 episodes, 69 windows, five
warmups and 14 measured episodes on one RTX3090.

Root-relative position/FK body discrepancy, equal-weight sequence mean:

| arm | U cm | G cm |
|---|---:|---:|
| GT | 0.43238 | 0.43238 |
| R2 DDPM | 0.95248 | 1.04784 |
| R2 DDIM25 | 0.94729 | 1.01186 |
| CM16 | 1.72761 | 1.72400 |
| CM25 | 1.74250 | 1.72559 |

CM16−R2 is +0.77513 cm U, CI [0.74074,0.81255], and +0.67616 cm G,
CI [0.63492,0.71845]. R2−GT is also positive in both arms. CM25−CM16
intervals cross zero. Root translation aligns to numerical precision; the
body discrepancy also occurs within windows. All 3375 records and all eight
representation contrasts completed. Batch128 SMPL-X forward takes 6.05–6.25ms;
maximum allocated/reserved memory is 261.1/274.7 MB. Full eight-GPU workload,
including startup and merge, took 29 seconds.

CM25 U has no >5g or low-walk cases. G full375 has four >5g episodes/nine
frames; holdout355 has two episodes/six frames and no low-walk cases. Absolute
guards pass. All four IDs and the largest boundary regressions are retained.

## FID audit and limits

The cited 2.811742929 is a historical mean of 60 real-vs-real FIDs: two disjoint
245-clip draws from an encoder-training pool of 3000, draw seed42; historical
pool selection used seed7. Split-result percentiles are [1.08390,5.87252].
Original m3_noise_floor.py/json contradict the older report's 1/n-extrapolation
footnote. Current test-GT self-FID is zero. The original external reports were
left intact; the new report records the correction and exact sources.

R2 U/G FID is 38.41515/40.04968. The CG difference +1.63453 has CI
[-3.96724,7.24187]. Guided FID decomposes into mean term 20.42234 and covariance
term 19.62734; prediction covariance trace 85.10491 versus GT 138.47846. This is
a distribution description; 2.81 is not a universal floor or motion-error unit.

FID reads coarse direct positions; physical metrics use rotations+root FK.
Representation errors include existing deployment interpolation. Source also
confirms reversed endpoints in the near-identical quaternion LERP branch and
different position/rotation time grids. Their contribution remains unmeasured.
Fixed 16/25 steps isolate step count; teacher/student gaps still include learned
weights and different sampling/noise formulas. No isolated training-cause claim.

## Verification and artifacts

- Canonical infbagel environment; python -m pytest tests -q: 462 passed,
  three skipped. Thirty exact job configurations fully resolved.
- Initial test import/terminal-expectation errors were fixed before workloads;
  original logs retained. Fourteen paired reports plus semantic and walk-contact
  readouts completed. No formal workload failure; zero optimizer updates.
- Simultaneous intervals use primary_multiple_comparisons_v2.json and
  default_rng42, exactly matching paired_bootstrap.py pointwise intervals.
  Initial RandomState draft retained separately; it is not the final readout.
- Total reserved cost 2.754167 GPU-h, within 12. Six immutable manifests were
  finished from the execution source before registry/documentation changes.
- [Report](../../experiments/results/p1_hsi_cm1_steps_representation_s42_20260909.md)
  and [compact](../../experiments/results/p1_hsi_cm1_steps_representation_s42_20260909.json).
  Setup index: results/cm1_representation_setup_20260909/runs.json.
- Raw representation records: results/hsi_position_fk_s42_20260909/.
  Each run directory contains manifest, resolved configs, preflight, logs,
  timings, exit status and completion record. Sealed input/checkpoint identities
  are referenced through these manifests; no new checkpoint was trained.

## Exact next entry

Read this summary, docs/plan/OVERVIEW.md and CM1.4 in docs/plan/PHASE_1C_HSI.md.
Review one fixed-weight deployment interpolation correction and matched
GT/R2/CM16 re-evaluation, preserving the current native FID definition and old
protocol references. Use corrected teacher residual to select one GT-anchored
training target; preserve the R3 hard-rebase negative. The current authorization
completed the diagnosis; it does not start this successor or new training.
