# Phase 2.10 DP-Edit implementation — 2026-09-06

**Implementation gate PASS; HSI scene-quality benefit unresolved.** The new
post-window editor completed all four registered rows: 28 episodes, 180 windows,
with the original native evaluator. Full DP-Edit versus the identical lambda0
editor has unresolved HS/OS mean differences. This single-scene diagnostic
establishes the implementation and leaves the 469-task quality claim open.

## Scope and implementation

The user approved advancing the handoff at
`/data/yujinlun/papers/PriorHOSI_SceneEvidenceEditing_Codex_Handoff.md`.
Preregistered branch `phase/02j-scene-evidence` starts from the completed Armijo
integration. P15 online with original 500-step diffusion and Arm B guidance
provides each source. R2 final EMA stays frozen and supplies full dynamic
perception versus static-only raw x0 predictions. No expert is retrained.
Learned mixer, distillation and state-machine work remain deferred.

`code/mixer/scene_evidence.py` implements raw x0-to-epsilon conversion, shared
source/candidate noise, known-empty HSI modalities, human-only evidence, and
67-coordinate relation-space editing. The source reference is the fixed
zero-residual reconstruction. Every outer iteration queries clean candidate
geometry, detaches the teacher, and takes at most one accepted local Armijo step.
The raw HSI pair keeps the same static/text/goal/progress conditions and masks
object tokens only at the denoiser. The full object remains in world queries.
The teacher uses already-prepared HOI conditions and its own RNG; no sampling
counter or native generator is advanced by teacher queries.

The levels are 300/264/229/193/157/121/86/50; beta_ref=1, lambda=.1 versus0,
weighting alpha/sigma, one noise draw/level and no direction normalization/cap.
Initial step1, shrink.5, c1=1e-4, ten backtracks, proximal weight1 on the squared
Euclidean parameter displacement. Source residual/contact/stance-increment/
endpoint and human/object scene terms have unit weights and their existing
physical scales. Bounds remain10cm/10degree per component. Floor pulling and
old HSI displacement tracking stay off. Domain checks reject newly exterior
points or increased exterior distance against the accepted state, with zero
geometric tolerance. Every explicit trial refreshes nearest-free references.
Accepted descent refers to the frozen local surrogate; different outer-loop
surrogate values do not form a global monotone likelihood.

`HOSIComposedSampler` now exposes its raw HOI head and an optional default-off
post-window hook. Enabled modes restore the actual fixed history after native
object SO(3) projection, then preserve it exactly through editing. Disabled mode
retains native arithmetic. The evaluator's two-line change connects its existing
motion recorder to the editor. Metrics, world restoration, stitching, and history
rebase remain the original evaluator. No core/expert/tool implementation changed.
The existing composed loader still loads both checkpoints in all four rows;
disabled/reconstruct/lambda0 perform zero HSI teacher forwards. Loading this
unused checkpoint remains a startup-cost limitation of the existing interface.

## Registered run and native results

Run `p2-mixer-scene-evidence-s42-20260906`, scene bin0:
`a3df624b-0917-46e9-ac15-fab766276c72`, seven objects, canonical ordinals371–377.
Four independent GPU lanes on the eight-RTX3090 authority host. Each row contains
45 windows and uses seed42, original scene-level seeding and complete native
outputs. The subset and settings were registered before results; no setting was
selected from this test-scene diagnostic. There were no operational failures,
restarts, overwritten outputs, dropped metrics or missing pairs.

Contact is percent; FS and HS/OS s_mean retain native scales. The scene s_mean
metrics are frame-averaged sums of penetrating vertex depths.

| Row | Complete | Contact % | FS | HS s_mean | OS s_mean |
|---|---:|---:|---:|---:|---:|
| disabled HOI |7/7|75.6326|.164226|15.719302|33.120716|
| reconstruct |7/7|74.4688|.086656|18.463598|34.933854|
| lambda0 |7/7|74.5164|.087464|13.023273|26.408226|
| DP-Edit |7/7|74.5164|.087471|13.015484|26.402699|

DP-Edit minus lambda0 HS mean is -.00778916, nominal95% paired interval
[-.02588486,.00243902]; OS mean is -.00552628 [-.01899556,.00248391]. Point
reductions are .0598% and .0209%. FS delta .00000713 has interval
[-.00003201,.00004680]. Contact, completion and both scene-frame ratios agree
per episode. Object endpoint error decreases .005158cm, nominal interval
[-.014343,-.000322]; hand/body penetration loss differences are also nominally
negative but tiny. These exploratory, unadjusted seven-episode intervals give
no scene-generalization or practical HSI-transfer claim. All16 metrics and all
six pairwise contrasts are retained in the compact result.

Reconstruction itself changes quality: FS falls, contact falls1.164 percentage
points and HS/OS means rise versus disabled HOI. Keep this control visible when
interpreting the total method. Most scene-depth improvement in this diagnostic
is already present with lambda0. Four100% completion rows also provide no
measurement of recovering failed tasks.

## Signal, solver, boundaries and runtime

DP evidence RMS mean/max .036167/.380616 is finite and nonzero. Mean parameter
gradient norms in the complete row are .00003642 (HOI reference), .00011148
(weighted HSI evidence) and .06571488 (explicit terms). The last two differ by
about589x. This scale observation motivates scrutiny of effective teacher
influence; it does not alone prove the cause of the native result or authorize
changing lambda on these observed test tasks.

| Row | Changed windows | Accepted updates | Trial evaluations | Domain-rejected trials | Extra HOI/HSI forwards |
|---|---:|---:|---:|---:|---:|
| lambda0 |33/45|226/360|2278|144|720/0|
| DP-Edit |45/45|329/360|1563|196|720/720|

All accepted trials pass sufficient and strict local decrease. Every first-level
HOI reference difference is exactly zero. All saved source/reference/edit tensors
are finite; history and contact channels are exact in every edited window.
Geometric exterior-point counts and maximum exterior distances never rise from
source to final state. Domain rejection retains the previous accepted solution.
Both edited rows' consecutive-window world-history differences are at most
5.96e-7 across position, object translation and rotation representations; this
verifies that edited outputs feed the next real window. Full motions, trials,
per-term values, gradient norms/dots, and native metric records are saved.

Manifest-to-completion elapsed179s. Generation sums for disabled/reconstruct/
lambda0/DP-Edit are81.06/84.36/146.63/149.92s. Complete-row editing totals67.73s,
including15.97s teacher queries and46.35s solver work/gradient diagnostics:
1.505s editor time/window. Lambda0 editing totals63.55s. The complete method's
additional HSI query cost and smaller line-search count both enter those totals.
Peak recorded allocation is1,271,632,896 bytes (1.184GiB), below20GiB headroom.
These synchronized four-lane instrumented timings measure this workload;
sharded native latency/FPS summaries remain invalidated. No separate smoke or
performance workload was added.

## Verification and artifacts

Final implementation authority suite: **945 passed, 4 skipped, 163.99s**.
The13 new cases cover conversion/sign, source cancellation, masks, raw queries,
clean geometry/empty HSI views, frozen teacher gradients, independent cells,
quadratic local descent, nonfinite failure, geometric bounds, recording/RNG,
actual normalization, relation VJP under outer no_grad and native disabled identity.
Completion suite: **945 passed, 4 skipped, 169.67s**. Two existing HSI checkpoint
shape/loading cases lack their historical checkpoint pair; two existing P8
bootstrap cases lack their sealed evaluation outputs. Registry validation passes
with358 records; diff check passes. Completion log is `completion-tests.log` in
the run root. The seven disabled-row episodes also reproduce every shared native
metric of the sealed469-task P15 Arm B baseline exactly; the comparison is saved
as `disabled_baseline_comparison.json`.

Train/test/LINGO human normalization bounds are exactly equal, verified from the
three actual norm.npy files. Live model setup verifies canonical beta equality;
P15/R2 loading uses the existing checkpoint identity checks. Input identities are
reused from the sealed manifest references; no new hash mechanism was introduced.
The four resolved jobs differ only in output directories and registered editor
switches. Registry validation passes; frozen-core and existing native-chain tests
remain in the authority suite.

- Compact result: `experiments/results/p2_mixer_scene_evidence_s42_20260906.json`.
- Run root: `results/experiments/p2-mixer-scene-evidence-s42-20260906/`.
- Runtime source: implementation1e22698; preregistration2660f28. Manifest start
  and finish use the same implementation source.
- Run root preserves `manifest.json`, `execution_plan.json`, `machine_preflight.json`,
  four `resolved/*.yaml`, `config_comparison.json`, launch/analysis commands, logs,
  exit codes, per-episode audits/motions, all native outputs and paired analyses.
- `results/scene-evidence-development/` holds full test logs, initial resolved
  config and numeric normalization comparison. The original legacy first-pass
  suite943/4 and final implementation suite945/4 are both retained.

The executed launch command is archived and already completed; its run id is
consumed. Exact evaluator commands are `execution_plan.json`'s four `command`
arrays. Reusable configuration inspection commands, from this checkout, are:

```bash
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
cd "$ROOT_DIR/code"
"$INFBAGEL_PYTHON" test_infbagel_hosi.py --config-name config_sample_hosi_scene_edit --cfg job --resolve
"$INFBAGEL_PYTHON" test_infbagel_hosi.py --config-name config_sample_hosi_scene_edit sampler.pelvis.scene_editor.lambda_dp=0 --cfg job --resolve
```

## Exact next entry

Read this summary, OVERVIEW.md and the Phase2.10/2.11 sections of
PHASE_2_COMPOSITION.md. Retain the four controls and original benchmark metrics.
Phase2.11 on `phase/02k-scene-evidence-benchmark` is the next separately registered
session. Before its formal run, choose explicitly between testing these fixed
settings or preregistering signal-scale calibration on independent development
scenes. The latter cannot use these seven test tasks to select weights. The
observed weak HSI increment supports reviewing that decision before spending
469-task comparison work; it does not justify declaring HSI unusable.

The benchmark entry is the implemented config with `hosi_shard_count=8` and
indices0–7 (or count1 for serial), a new run id and fresh output directories,
followed by original `hosi_mode=merge_shards` and existing paired-bootstrap
projection. Archive/validate all resolved configs before tools/experiment.py
start. Keep the paper InfBaGel aggregate row separate from locally paired results.
The full469 evaluation and comparison to InfBaGel have not run in this session.
Close and integrate/tag only2.10 as `exp/p2j-scene-evidence-v1`; the full Phase2
quality gate remains open.
