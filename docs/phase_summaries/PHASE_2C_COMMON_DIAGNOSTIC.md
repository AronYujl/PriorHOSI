# Phase 2.3 common-correction diagnostic — 2026-09-06

**Deliverable PASS. The floor objective causes an independent native quality cost in the tested A00 bundle.** Zero-step reconstruction is the preferred next comparison anchor. Full Phase 2 and learned-mixer training remain open.

## Scope and implementation

The user approved two concrete controls on `phase/02c-common-diagnostic`:
`reconstruction` sets A00 optimizer steps to zero; `no_floor` excludes only its
floor term from the optimized sum and retains 20 Adam steps at 0.05. Both apply
decode/encode and clean feedback at 10,1,0. The default optimizer objective keeps
its exact term order; floor energy remains observable when excluded. Audit records
the floor switch and actual optimizer-gradient step count, including zero.

One config fragment inherits the closed-loop evaluator. ROOT_DIR now resolves
to the current checkout absolute root at evaluator entry. The eight resolved
configs match archived A00 except run/output paths and approved interventions.
Experts and core remain fixed: R2 final EMA + CG, P15 online + Arm B, raw gate 0,
500 diffusion steps, 499 CG applications/window, sealed Arm B, seed 42.

Same scene bins 0,22,44,66 and seven objects: 28 episodes/124 windows per new row.
Reuse the matched reference/A00 from the Phase 2.2 r1 run by reference. All
56 new episodes and eight paired analyses completed. No training or parameter
search occurred. The formal run supplies real-data and batch-1 runtime validation.

## Native results

Each row has 28 episodes. Contact and HS frames are percentages. Feet height
is the native estimated floor-height statistic in centimetres; it is an engagement
proxy. FS uses the native scale. HS/OS s_mean sums penetrating vertex depths per
frame then averages over frames; it is not mean per-vertex depth.

| Row | Complete | Contact % | FS | Feet height cm | HS frames % | HS s_mean | OS s_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 22/28 | 67.340 | 0.17468 | 4.012 | 38.534 | 3.79587 | 30.11801 |
| a00 | 18/28 | 68.510 | 0.55478 | 1.670 | 75.405 | 4.58570 | 31.42949 |
| reconstruction | 22/28 | 68.466 | 0.12928 | 3.903 | 34.203 | 4.01455 | 30.53389 |
| no_floor | 22/28 | 67.620 | 0.13113 | 3.895 | 33.926 | 4.11227 | 30.58138 |

All 15 native metrics plus completion, including negative findings, are retained
in the compact result and original per-episode artifacts. Completion means both
endpoint errors below 10 cm, with no implication of a full state-machine success.

## Paired evidence

10,000 seed-42 replicates, paired episode and four-scene means. CIs below are
nominal 95% intervals per the preregistration, not simultaneous familywise bounds.

| Contrast / metric | Delta | Episode CI | Scene CI |
|---|---:|---|---|
| no_floor-a00 / foot_sliding | -0.423645 | [-0.531491, -0.314785] | [-0.529725, -0.317565] |
| no_floor-a00 / scene_human_penetration_frame_ratio | -0.414788 | [-0.500860, -0.329897] | [-0.448319, -0.373163] |
| no_floor-a00 / completed | +0.142857 | [0.035714, 0.285714] | [0.035714, 0.250000] |
| no_floor-a00 / scene_human_penetration_s_mean | -0.473422 | [-0.883284, -0.203608] | [-0.766782, -0.180063] |
| reconstruction-reference / foot_sliding | -0.045404 | [-0.083828, -0.011767] | [-0.073666, -0.012575] |
| reconstruction-reference / scene_human_penetration_frame_ratio | -0.043308 | [-0.081677, -0.012013] | [-0.092272, -0.009529] |
| reconstruction-reference / completed | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] |
| reconstruction-reference / scene_obj_penetration_s_mean | +0.415874 | [-1.235770, 2.785095] | [-1.026988, 2.602496] |
| no_floor-reference / foot_sliding | -0.043549 | [-0.081812, -0.009801] | [-0.072909, -0.014188] |
| no_floor-reference / scene_human_penetration_frame_ratio | -0.046077 | [-0.084088, -0.015257] | [-0.099455, -0.007867] |
| no_floor-reference / scene_obj_penetration_s_mean | +0.463368 | [-1.175438, 2.919328] | [-0.997768, 2.838741] |
| no_floor-reconstruction / foot_sliding | +0.001856 | [-0.001995, 0.006061] | [-0.003052, 0.007862] |
| no_floor-reconstruction / scene_human_penetration_frame_ratio | -0.002769 | [-0.008089, 0.001478] | [-0.007200, 0.001662] |
| no_floor-reconstruction / contact_percent | -0.008458 | [-0.016982, -0.002454] | [-0.017840, -0.003215] |
| no_floor-reconstruction / scene_human_penetration_s_max | +0.974378 | [0.000928, 2.859473] | [0.043547, 2.756836] |

The primary no_floor-A00 outcomes clear negative upper CIs at both units.
Completion recovers four tasks, with positive lower CIs at both units. This is
the total causal effect of excluding floor within the remaining fixed bundle,
including changed autoregressive histories and interactions with other terms.
A separate stance-mask effect has not been measured.

The previous suspicion that reconstruction itself caused the regression is
refuted for the preregistered sliding and HS-prevalence outcomes on this pilot:
both improve under zero optimizer steps. Contact and completion protections pass
for both new rows relative to reference. Scene depth changes against reference
remain unresolved, so reduced penetrating-frame prevalence is not an established
depth reduction or a full quality-gate result.

The remaining optimizer has no demonstrated incremental sliding/HS-prevalence
benefit over zero-step reconstruction. It loses 0.8458 contact percentage points
and increases HS s_max by 0.97438, both significant at both units. That contact
loss is below the two-point budget, but adds no reason to prefer its extra work.
The HS maximum statistic is the native frame-depth sum maximum, not the deepest
individual vertex. Preserve both adverse outcomes; avoid claiming equivalence
from the remaining intervals crossing zero.

## Execution and verification

Manifest wall time: 2057 seconds (34.28 minutes), eight isolated authority RTX 3090 GPUs.
All eight jobs and eight paired analyses exited zero. Both rows have complete
28-episode/124-window coverage, 61,876 CG applications and 372 corrections each.
There are 7,440 actual Adam gradient steps in no_floor and zero in reconstruction.
All native outputs/CG/actual gradients are finite; history and contact channels
are exact. Peak recorded allocated memory is 804.43 MiB in each row. Mean correction
time is 0.06585 seconds for reconstruction and 0.57561 for no_floor.
These measurements include resident models and HSI target preparation. Concurrent
generation accounting is in the compact result; it is not isolated production
latency. Screen persistence and initial stability artifacts are retained locally.

The implementation suite passed 916 tests with four skips in 163.18 seconds.
Final completion verification passed 916 tests with four skips in 158.70 seconds;
registry validation passed with 344 rows.
Component tests cover zero-step projection, default compatibility and isolated
floor exclusion. Eight exact resolved configs and their baseline comparison passed.
No separate smoke, new tool script or repeated checkpoint transfer was introduced.
This run has no operational failure; prior failed launches and negative experiments
remain in their original records.

Verification commands:

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
```

Preregistration commit: `26aa9bb`; executed implementation: `b6e37c1`.
The completion commit contains this handoff, compact result, dated plan and
registry result. Fast-forward into `phase/02-mixer`;
tag `exp/p2c-common-diagnostic-v1` identifies the completed diagnostic.

Artifacts:

- `experiments/results/p2_mixer_common_diagnostic_s42_20260906.json`.
- `results/experiments/p2-mixer-common-diagnostic-s42-20260906/` retains manifest,
  source identity, configs, preflight, logs, native audits, eight paired reports
  and completion metrics.
- Reused controls: `results/experiments/p2-mixer-relational-rollout-r1-s42-20260906/`.

## Exact next entry

Read this handoff, the overview and latest Phase 2 plan. Keep R2+CG/P15+Arm B
fixed. Use zero-step reconstruction as the preferred comparison anchor for a
separately approved next experiment, and remove the forced 2 cm floor objective
from any new candidate common recipe. The archived A00 remains reproducible.
The original geometry and HSI factors were tested with floor active; their effects
must be re-established on the corrected anchor before transfer or training.
Useful learned HSI supervision, full scene evaluation and motion realism remain
open. The four-scene diagnostic does not establish the complete Phase 2 gate.
Close this subphase; start no further workload in its closing session.
