# Phase 2.5 stance recording diagnostic — 2026-09-06

**Deliverable PASS; exact replay PASS. Sparse stance coverage and correction-time
outside-mask motion are directly observed. A repaired recipe has not been tested.**
Full Phase 2 and learned-mixer training remain open.

## Scope and implementation

User approved recording-only replays of floor-free A00/A01, 28 episodes each.
A00 reuses Phase 2.3 no_floor as sealed reference; A01 reuses Phase 2.4 a01.
Fixed R2 final EMA + CG, P15 online + Arm B, seed42, bins 0/22/44/66 and seven
objects, same 500 diffusion steps/499 CG applications, corrections 10/1/0,
20 Adam steps/.05, masks/bounds/scales and objectives. No expert/core edit.

RelationalCorrector adds an opt-in record_motion switch, default false. It saves
source masks, raw residual parameters and world FK/object states before and after
correction plus translation-only and common-translation/yaw intermediate states.
Evaluator saves final sampled windows, stitched positions/rotations/object poses,
post-interpolation SMPL joints before metric processing and evaluated floor height.
Detached copies are saved per episode and correction buffers cleared after write.
No observation draws random values or changes returned inference tensors.
One inherited config fragment enables recording. Tests compare on/off outputs,
RNG state, scalar objectives, saved masks and reconstructed states.

Archival correction: the Phase 2.3/2.4 evaluate path saved scalar audits/metrics,
not full motion, despite save_motion_params=true in resolved configs. Earlier
handoff language implying persistent complete trajectories was incorrect. Their
sealed metrics remain valid. Full trajectory retention begins in this run.

## Exact replay and runtime

Eight jobs and automated analysis all exited zero. All 56 episodes and 744
correction snapshots are present in 56 motion files (48,730,808 bytes total).
All original native per-episode metrics/completion and optimizer scalar telemetry
exactly match their sealed rows. Saved evaluated joints/floor exactly reproduce
native FS for every episode. Timing/memory are excluded from equivalence.
All native/optimizer values and saved correction states are finite; history and
contact exact; each row has 124 windows, 61,876 CG calls and 7,440 optimizer steps.

Wall time 1931 s (32m11s), eight authority RTX3090 GPUs. Each row: 5376 evaluated
frames, peak allocated memory 804.43 MiB. Generation sums A00/A01:
5429.56/5160.54 s; mean measured correction .56313/.55057 s. Correction timing
ends before optional recording; sequence generation includes correction recording,
while episode serialization occurs later. These are concurrent diagnostics,
not isolated production latency or a recorder-overhead benchmark.

The native tradeoff is exactly retained: A00/A01 contact 67.620%/68.062%,
completion 22/28 each, FS .131133/.207636, HS mean 4.112274/2.115831,
OS mean 30.581380/19.052997. Feet height 3.895115/4.184735 cm; HS penetrating
frames 33.9259%/36.0004%. HS/OS mean is the frame-average sum of penetrating
vertex depths, not per-vertex depth. Completion is both endpoint errors <10cm,
not state-machine success. Prior quality failure remains unchanged.

## Diagnostic findings

1. **Zero energy means empty masks here.** Both rows have 338/372 empty source
   masks (90.86% of correction calls), and zero nonempty stationary masks. Both
   have 19/28 episodes with exclusively empty masks; these episodes account for
   85.69% of the native net FS increase. The episode-first empty-call fraction is
   88.45%; it differs from the pooled-call fraction because episode lengths differ.
2. **The coverage rules differ.** Optimizer selects ankle/toe below world y=.08/.04m
   in both adjacent source frames. Native FS subtracts its estimated floor and
   selects only the predecessor frame, with a height-dependent weight. Its floor
   proxy is roughly 4cm in these rows. Optimization acts on coarse FK; evaluation
   acts on interpolated SMPL joints. Neither representation nor support set is
   interchangeable.
3. **Movement grows outside coverage during correction.** The table below uses
   episode-first means over each episode's corrections. The FS proxy evaluates
   future transitions with native height weighting at the final episode's floor.
   That retrospective floor is a diagnostic reference, not an online estimator.
   It is not native episode FS: grid, joints, history handling and normalization
   differ. Native contact-eligible means eligible under this retrospective rule.

| Diagnostic | A00 | A01 | Delta episode 95% CI | Delta scene 95% CI |
|---|---:|---:|---|---|
| selected_displacement_cm | 0.083891 | 0.083593 | [-0.002183, 0.002176] | [-0.002511, 0.002331] |
| outside_displacement_cm | 6.431588 | 6.569860 | [0.040003, 0.280231] | [0.070285, 0.217727] |
| fs_change_inside_source_mask | -0.001096 | -0.001088 | [-0.000825, 0.001232] | [-0.000941, 0.001275] |
| fs_change_outside_source_mask | 0.007405 | 0.124410 | [0.023795, 0.270196] | [0.041473, 0.224066] |
| new_world_mask_count | 0.039583 | 1.214385 | [0.082341, 3.101300] | [0.160714, 2.802579] |
| translation_foot_increment_temporal_cm | 0.095748 | 0.285905 | [0.099429, 0.302112] | [0.137433, 0.256814] |
| yaw_foot_increment_temporal_cm | 0.017318 | 0.031283 | [0.007201, 0.023254] | [0.009226, 0.019384] |
| articulation_foot_increment_temporal_cm | 0.032607 | 0.193901 | [0.071023, 0.277176] | [0.135275, 0.193469] |

The selected-foot displacement mean includes zero for empty masks. It does not
claim every active support remains fixed. Outside displacement also includes
swing feet, so its increase alone is not a native sliding measurement. The
outside-mask FS-change proxy supplies the height-weighted complementary evidence.
At A01's correction-grid output, 28.580 of 29.983 eligible transitions per
correction lie outside the source mask (95.32%, ratio of episode-first means).
New world-threshold support also appears: mean 1.214 transitions/correction
versus .0396 in A00, with positive delta intervals at both units.

**Stage localization.** At reverse step0 the corrected FK proxy is
.318626/.509346 (A00/A01), versus final sampled-window .316318/.506291.
Thus the difference is already present before stitching/interpolation. The
step0 outside-mask before/after increment contrast is +.118816, episode CI
[.024052,.276326], scene interval retained in the compact result. Generated-
history outside-mask contrast +.100441 has positive episode interval
[.017417,.215335]. Initial-history contrast +.152072 is unresolved at episode
unit [-.015545,.419075]; do not claim that every history stratum is significant.

Stitched coarse-FK proxies are .370916/.583446, with boundary-transition
contributions .024508/.040450. The boundary contribution increase .015942 is
7.50% of that coarse-grid net increase. These descriptive numbers do not convert
to a percentage of native SMPL sliding: interpolation scale is 3 and the joint
representation changes. Native FS .131133/.207636 is independently reproduced.
Do not attribute the remaining gap solely to interpolation or joints without an
additional matched calculation.

**Height selection matters descriptively.** Native FS contributed by predecessor
feet above the optimizer's absolute-world thresholds is .123216/.172434;
its increase .049218 is 64.34% of the native net increase. Episode interval
[.007087,.101723] is positive, but scene interval [-.000862,.099298] crosses zero.
This final-grid partition is not the actual coarse-grid source-mask partition.
Liftoff transitions excluded by a two-frame native-relative mask contribute only
.002920/.003299, with unresolved change at both units. Keep the floor-reference
and temporal-mask distinctions separate.

Translation, added yaw and added articulation all increase foot temporal
increments, with positive nominal intervals at both units. Translation's norm is
largest (.285905cm in A01), articulation next (.193901cm), yaw smaller (.031283cm).
These are fixed-order vector-change norms: they do not add, and their ranking
is not a causal ablation or a percentage attribution of native FS. Source states
also diverge across rows under closed-loop feedback.

All diagnostic intervals are exploratory nominal 95%, 10,000 seed42 paired
resamples at episode and four-scene units. No multiple-testing or cross-seed
confidence claim. Complete native/diagnostic metrics, steps 10/1/0 and history
strata are retained in compact JSON and per-correction local artifacts.

## Interpretation and next entry

The diagnostic replaces a suspicion with measured coverage: the source stance
penalty is absent in most calls, and geometry increases height-weighted foot
motion outside its support already at clean correction. This supports testing
support selection/height-reference alignment before weight tuning. It does not
establish that a mask repair alone preserves geometry's OS gain or restores
native motion quality. It provides no positive HSI training target.

Keep reconstruction as the comparison anchor, R2+CG/P15+Arm B fixed, floor
objective excluded. Read this handoff, overview, Phase2 plan and HSI design priors
in the next session. Develop one concrete mask/height-reference intervention with
matched controls and native contact/completion/sliding protections, then obtain
approval before implementation/GPU work. Do not reinstate forced 2cm floor,
retune experts or start learned-mixer training. Close only Phase2.5 here.

## Verification and artifacts

Implementation full suite: 916 passed, 4 skipped in 159.36s. Completion full suite:
916 passed, 4 skipped in 161.25s. Registry valid with 348 records; diff check
passed. All 16 native and 144 diagnostic metrics have complete paired coverage
at both units. Formal replay provides
real-data validation; no separate smoke was added. Input identities reuse sealed
manifest references; no checkpoint/data transfer was needed.

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
```

Preregistration 008ca43; implementation 5c8d2d8. Completion commit contains this
handoff, compact result, plan and registry. Fast-forward into phase/02-mixer and
tag exp/p2e-stance-recording-v1 after diagnostic gate, with quality gates open.

- Compact: experiments/results/p2_mixer_stance_recording_s42_20260906.json
- Run: results/experiments/p2-mixer-stance-recording-s42-20260906/
- Analysis includes exact-equivalence check, native and diagnostic episode/scene
  projections and paired reports, stage-records, stitching-stage and runtime audit.
- Eight resolved configs, comparison, preflight, manifest, command artifacts,
  logs, per-episode audits and 56 episode-motion files remain in the run directory.
- Reused references: Phase2.3 common-diagnostic/no_floor and Phase2.4
  floor-free-factorial/a01. All earlier failures and negative findings retained.
- No operational or equivalence failure in this run; no mid-window resume exists.
