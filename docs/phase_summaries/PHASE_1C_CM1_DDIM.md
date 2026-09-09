# Phase 1C-CM1.3: fixed R2 DDIM teacher diagnosis

Completed on 2026-09-09 on phase/01c-cm1. All five registered workloads exited
zero. The 25-grid teacher does not reproduce the large CM1 boundary-jerk
degradation. Additional student/consistency-sampler error remains. R2+CG stays
the working quality baseline; Phase 1C remains open.

## Scope and source

Preregistration: 08d8522. Implementation and execution: 4b8d305.
One new config, config_sample_hsi_r2_ddim25.yaml, routes fixed R2 final EMA
weights through the DDIMSolver.ddim_step already used by consistency training.
No optimizer updates or new checkpoints were made. Core is unchanged.

The native sampling loop shares CFG, occupancy, history clamping and geometry
energy with DDPM. DDIM visits 499,479,...,19 with eta=0: 25 steps and 50 denoiser
calls. Guidance uses the clean Jacobian at fixed noisy state, including the
dependence of inferred epsilon on the clean prediction. It runs on 24
nonterminal transitions; the final transition is direct denoising.

## Fixed coverage and principal results

DDIM U/G each cover all 375 episodes and 2,271 windows. All outputs were finite.
The sealed DDPM500 R2 and CM1/16 U/G outputs were reused. Every arm uses CFG=1;
U/G denotes external geometry guidance off/on. Native groups remain 130 walk
and 245 interactive/reaching episodes.

| Unguided, full375 | R2 DDPM500 | R2 DDIM25 | CM1/16 |
|---|---:|---:|---:|
| Penetration ratio | 0.0301742 | 0.0306028 | 0.0328194 |
| Foot sliding | 0.296591 | 0.302345 | 0.317062 |
| Boundary jerk | 124.677 | 123.055 | 156.309 |
| Exterior contact count | 315.646 | 321.659 | 306.493 |

DDIM minus DDPM boundary jerk is -1.62275, paired 95% CI [-3.47641,0.25872].
CM1 minus DDIM is +33.25427, CI [29.97637,36.42167]. The latter comparison also
increases penetration and foot sliding and decreases exterior contact.
The six fixed primary comparisons retain their directional conclusions under
Bonferroni simultaneous intervals. A CI crossing zero is not equivalence.

Aggregate DDIM foot sliding increases modestly; walk-only foot sliding has no
detected difference, and walk surface penetration improves. CM1 worsens both
walk surface penetration and foot sliding relative to DDIM.

With guidance, DDIM versus DDPM has higher penetration (+0.00247627,
CI [0.00172885,0.00326190]) and lower exterior contact (-23.89648,
CI [-31.59779,-16.60997]), while boundary/interior jerk and goal error improve.
Relative to guided DDIM, CM1 worsens penetration, foot sliding, boundary jerk
and exterior contact; interior jerk improves and goal error is inconclusive.

DDIM FID is 33.20650 U / 36.76145 G; R@3 is 0.482143 / 0.419643. These feature
metrics consume the network's direct 28-joint position channel. Physics uses
SMPL-X reconstructed from rotations plus root. Feature improvements therefore
do not directly establish improvements in the FK body's motion. Preserve the
frozen evaluator definition and make this representation distinction explicit.

## Timing, safety and resource use

Both new single-3090 latency controls used the same 19 episodes and 69 windows
as the sealed four controls; five episodes were warmup and 14 were timed.
CUDA timing was synchronized and before/after GPU process snapshots were empty.

| DDIM | Mean generation FPS | Generation seconds/episode | End-to-end seconds/episode |
|---|---:|---:|---:|
| U | 53.9471 | 2.78486 | 2.85695 |
| G | 13.6776 | 10.93398 | 11.00942 |

Total-generation-time speedups over DDPM are 20.0670x U and 18.0359x G.
The eight-GPU quality timings remain invalid for latency claims.

DDIM U has zero >5g episodes and one low-pelvis walk. DDIM G has nine >5g
episodes / 18 frames on full375, and five episodes / eight frames on the frozen
holdout355; no low-pelvis walk. The existing holdout guards pass. All full375
failure identities are retained; improved mean jerk does not erase tail risk.

Reserved cost is 3.100833 GPU-h, below the registered 12 GPU-h cap. Including
the previous CM1 training/evaluation, the accumulated CM1 cost is 69.624638
GPU-h. No reduced GPU set or hardware substitution was used.

## Verification and retained failures

Targeted checks: 60 passed. The authority invocation gave 456 passed, three
skipped and two setup errors because INFBAGEL_PYTHON was not exported for child
processes. Exporting it and rerunning the affected module gave four passed:
the complete suite is covered by 458 passed / three skipped. Both logs remain.
Nineteen exact resolved job configs, registry validation and native sampler
counts passed. No GPU workload failed or was rerun.

Use the canonical exported INFBAGEL_PYTHON and ROOT_DIR for these commands:

- python -m pytest tests -q
- python -m pytest tests/test_training_resume.py -q
- python tools/experiment.py validate

Paired geometry/group statistics use tools/paired_bootstrap.py with 10,000
seed42 draws. FID uses the paired stored 2,000 draws; R@3 preserves 224 query
occurrences from 148 unique sequences. The same-60 early/final CM1 comparison
is an additional descriptive reuse of existing artifacts, not checkpoint
selection or a new formal gate.

## Artifacts and next entry

- Report and compact: experiments/results/p1_hsi_r2_ddim25_s42_20260909.{md,json}.
- Native Table3: results/hsi_r2_ddim25_table3_s42_20260909/.
- Full statistics, failure cases, safety and command records:
  results/cm1_ddim_setup_20260909/.
- Five manifests/configs/logs:
  results/experiments/p1-hsi-r2-ddim25-*-s42-20260909/.
- Fixed R2 and CM1 checkpoint identities are referenced by the sealed manifests;
  neither checkpoint was modified. The completion commit contains this summary,
  the report, compact, phase append and five registry completion records.

Read this summary, docs/plan/OVERVIEW.md and the CM1.3 section of
docs/plan/PHASE_1C_HSI.md. A concrete successor should distinguish fixed-CM1
16/25-step sampling and quantify direct-position/FK disagreement before choosing
a physical-retention training objective. The present comparison changes DDPM
versus DDIM and its grid; CM1 versus DDIM also changes weights and consistency
sampling. It does not isolate training dropout, EMA or GT-history effects.
This completion starts no new training, next phase, merge or release tag.
