# Phase 2.8 A00 optimizer diagnosis — 2026-09-06

**Deliverable PASS; causal diagnostic POSITIVE.** The frozen-source contact
round trip supplies the complete initial gradient. Adam amplifies it into the
first harmful update in all 336 departing corrections. All 372 contact-off
shadow solves retain zero residual. Sampling and native outcomes exactly replay
Phase 2.7. Retain reconstruction and the fixed experts; no recipe is promoted.

## Scope and implementation

The user approved diagnosis of A00's complete common-objective increase. Reproduce
28 A00-increment episodes: seed 42, bins 0/22/44/66, seven objects, R2 final EMA+CG
and P15 online+Arm B, 500 diffusion steps, 499 CG calls/window, corrections 10/1/0,
20 Adam steps at .05, default betas (.9,.999) and epsilon 1e-8. Preserve all bounds,
scales, source-height masks, source stance-displacement target and contact labels.
The active A00 objective is residual+contact+stance+endpoint; floor, HSI and scene
terms remain inactive.

One config fragment enables optimizer observation in the existing relational
modules. Save parameters/energies at iterates 0..20, gradients and Adam moments
for updates 1..20, initial per-term gradients and cached geometry. Each original
source also receives a 20-step contact-off shadow and one unnormalized first
proposal -.05*g. Both are diagnostic only. The original 20-step result still
feeds sampling. No expert/core/evaluator or production objective/optimizer change.

## Causal results

All values below use episode-first means unless labelled counts or maxima.
Parameter magnitudes are dimensionless before tanh and the registered scales.

| Quantity | Result |
|---|---:|
| Corrections / original updates / shadow updates |372 /7,440 /7,440|
| Initial noncontact gradients exactly zero |372/372|
| Initial contact gradient exactly equals total |372/372|
| First-step and final objective increases |336/372 each|
| Ties / decreases at final iterate |36 /0|
| Source is best of21 recorded iterates |372/372|
| Contact-off trajectories with zero parameters and gradients |372/372|
| Source contact residual RMS |4.28218e-8 m|
| Initial maximum gradient |1.11979e-7|
| First maximum parameter update |.0397644|
| First maximum articulation component |.397383 degrees|
| First human RMS movement |.539569 cm|

The other 36 cases have contact labels, but exactly zero selected contact residual,
contact energy and total gradient; every iterate remains at zero parameters.
No correction has an empty contact mask. They are measured zero-trigger cases,
not missing-contact exceptions. The complete common objective is:

| Iterate | Mean objective |
|---|---:|
|0|3.93733e-13|
|1|.0451663|
|3|.419932|
|20|.0479565|

First-step loss is contact .0449961 plus residual .0001702; stance is 1.57e-13
and endpoint 3.94e-15. Initial common translation and yaw gradients are tiny
(mean maxima 1.77e-15 and 1.38e-14); articulation carries the 1.12e-7 trigger.
At step 20 the terms are residual .00035165, contact .00440813, stance .04292649
and endpoint .00027019. Stance dominates the final cost but does not start the
departure. Across all 7,440 original updates there are 3,198 objective increases,
3,522 decreases and 720 ties. Later decreases do not recover the initial minimum.

The first update matches -.05*g/(abs(g)+1e-8) for every correction at
atol 1e-8/rtol 1e-6. This is Adam executing its formula, not an Adam implementation
fault. The selected finite-step solver, which returns iterate 20, fails at a source
already at the mathematical minimum up to floating-point error. The complete
trajectory proves more than the earlier before/after logs or finite-gradient check.

The single unnormalized -.05*g proposal has mean maximum parameter size 5.59896e-9
and objective 4.02480e-13, with 329 tiny increases, 7 decreases and 36 ties. This
isolates normalization's scale; it is not evidence for a full SGD replacement.

Object rotation orthogonality error has mean maximum 4.56255e-7. Re-evaluating the
round trip in float64 on the same cached float32 inputs gives RMS 4.47675e-8 m;
the RMS difference from float32 arithmetic is 2.99817e-8 m. Cached nonorthogonality
and subsequent arithmetic both contribute; their RMS values are not additive.
Casting the loss arithmetic retains cached errors. A full float64 geometry/sampler
was not tested.

## Paired diagnostics and native replay

Contact-off minus original, 10,000 seed 42 paired replicates, nominal 95% CIs:

| Measure | Delta |28-episode CI|4-scene CI|
|---|---:|---|---|
| Original-objective increase |-.0479565|[-.0537445,-.0416525]|[-.0538459,-.0438220]|
| Human RMS movement, cm |-.283432|[-.309795,-.254221]|[-.307348,-.264697]|
| Object RMS movement, cm |-.174962|[-.190017,-.157536]|[-.192914,-.162197]|

Each shadow is scored against its own original source. These are correction
diagnostics, not a contact-free policy's native evaluation or a motion-realism gate.
All 16 original native outcomes and all existing correction scalars/motion fields
exactly replay A00-increment; native paired deltas and CIs are zero at both units.
Completion 22/28, contact 68.0543%, FS .139963, HS frames 33.8196%, HS mean 3.940869,
OS mean 30.645320. The sealed FS cost versus reconstruction remains +.010686
(+8.27%; significant at both nominal units). No native quality benefit was tested.

The compact result retains all native metrics, diagnostic means, curves and
step 10/1/0 plus initial/generated-history strata. Minor last-digit differences
between a saved float32 total and a Python sum of its term scalars reflect summation
precision; the underlying recorded term scalars replay exactly.

## Separate archived A01 finding and limits

A read-only audit of Phase 2.7 A01 includes residual+contact+stance+endpoint and
both scene terms. Its **complete** objective increases 335/372 times, ties 9 and
decreases 28, with episode-first mean .0537254→.0813346. This establishes a broader
minimization problem; A01 was not rerun or subjected to the contact-off diagnostic.
An A00 contact-origin repair alone therefore has no demonstrated A01 quality gain.

The result diagnoses this seed 42 cohort and fixed solver. It does not identify
mesh-SDF penetration body parts, certify physical support, or validate learned
mixing. Contact-off is an attribution tool; contact preservation remains required
in a future production recipe.

## Execution, verification and artifacts

Four RTX3090 sampling lanes, one per sealed scene shard, completed all 28 episodes,
124 windows and 61,876 CG calls. Initial verification used a separate GPU 4; final
analysis used GPU 0 after sampling. GPU job wall 33m45s; manifest through analysis
35m10s. Generation sum 5,636.18s, instrumented correction sum 228.90s, shadow/archive
sum 216.21s; peak allocated 804.66 MiB. These sharded, instrumented timings describe
this run rather than isolated production latency. The 28 motion files total
327,323,430 bytes. Formal execution supplied real-data functionality and timing/
memory validation without a separate smoke or performance workload.

An initial exact-replay checker used scalar equality for a NumPy array and failed.
`analysis-original.sh` and `analysis_revision.json` preserve the program and error.
The comparator was corrected to exact array equality; all GPU jobs and the final
complete analysis exited 0, with no sampling restart or raw-result overwrite.

The original vectorized all-step moment diagnostic used float32 bias powers.
Closure reconstruction using Adam's Python-double powers has maximum absolute
parameter error 1.49e-8 and maximum error divided by the registered tolerance
.761813. Every update therefore agrees at atol 1e-8/rtol 1e-6. The original
diagnostic discrepancy remains archived and is not an unexplained optimizer error.

Implementation: **924 passed, 4 skipped in 163.62 seconds**. Completion:
**924 passed, 4 skipped in 169.11 seconds**. Registry valid with 354 records;
all native replay, active-energy reconstruction and causal gates pass. The
completion log is `completion-tests.log`.

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
git diff --check
```

- Preregistration f74d1fc; implementation and complete sampling source 4471699.
  Manifest start and finish identify the same source. This handoff, compact result,
  plan and completion registry row form the completion commit.
- Integrate by fast-forward into phase/02-mixer; immutable tag
  exp/p2h-optimizer-diagnostic-v1 records this completed diagnostic.
- Compact: `experiments/results/p2_mixer_optimizer_diagnostic_s42_20260906.json`.
- Run: `results/experiments/p2-mixer-optimizer-diagnostic-s42-20260906/`.
  Retain manifest, resolved configs, preflight, launch/logs, motions/traces,
  paired/native projections, `analysis/strata.json`, `closure_audit.json`, original
  analysis/revision, and `archived-total-objective-review.json`.
- Reuse upstream input identities through the archived manifests; their original
  pre-archive paths and current incoming locations are recorded in the execution
  plan. No unchanged checkpoint/data identities were recomputed.

## Exact next entry

Read this handoff, OVERVIEW.md and the current Phase 2 plan. Retain reconstruction,
R2+CG/P15+Arm B and the forced-floor exclusion. Review a solver for the **complete
active objective**, its source contact consistency and the nearest-free scene
energy's smoothness. Use the saved geometry/optimizer inputs for bounded analysis
instead of regenerating A00 merely to recover state. Recommend one concrete A00/A01
repair experiment, retaining contact constraints and native protection gates,
before requesting its approval. No further production change or experiment is
approved here. Close only Phase 2.8; full Phase 2 and learned training remain open.
