# Phase 2.9 complete-objective Armijo solve — 2026-09-06

**Deliverable PASS; pilot quality PASS.** All 744 Armijo corrections preserve
complete-objective monotonicity. All three adjusted primary native comparisons
and the registered protections pass. Retain Armijo as the passing pilot solver,
reconstruction as the comparison anchor and the fixed expert pair. Full Phase 2,
useful HSI supervision, motion realism and learned training remain open.

The quality tradeoff is substantial: compared with old A01-increment, both human-
scene and object-scene penetration depths worsen significantly at nominal 95%
in both statistical units. This pilot pass does not establish dominance over the
old recipe or equivalence to reconstruction on protected outcomes.

## Scope and solver

The user approved the solver repair after Phase 2.8 isolated contact round-trip
error and Adam amplification. Run A00-armijo/A01-armijo, 28 episodes each, seed 42,
scene bins 0/22/44/66, seven objects, R2 final EMA+CG and P15 online+Arm B.
Fix 500 diffusion steps, 499 CG calls/window, corrections 10/1/0, representation,
bounds, scales, weights, contact labels and source-height/stance-displacement
targets. A00 optimizes residual+contact+stance+endpoint; A01 also optimizes both
scene terms. Floor and HSI factors stay off. Contact and scene objectives are
unchanged; expert/core/native evaluator code is unchanged.

The new config selects steepest descent with Armijo backtracking. Each iteration
uses d=-g, initial step 1, shrink .5 and c1=1e-4. Accept E_trial <= E_current +
c1*alpha*(g dot d) and E_trial < E_current. Every trial re-queries its current
occupancy and nearest-free references. Stop at zero gradient, 20 failed trials,
or 20 gradient iterations; return the last accepted iterate. Search exhaustion
and budget exhaustion are recorded, not interpreted as stationarity certificates.
The existing Adam path remains the default for sealed configurations.

Each source also receives an original 20-step Adam/.05 shadow using identical
geometry, targets and complete loss. Shadows do not feed generation. Save all
iterate parameters, gradients, energies, every trial decision and accepted-state
scene references, with geometry caches and complete motion recording. One config
fragment and existing component modules/tests implement the change; no new tool
script or per-experiment test module was added.

## Native results and registered gates

FS uses the native weighted-displacement scale. Contact and HS frames below are
percentages. HS/OS mean is the native frame-average sum of penetrating vertex
depths, not a per-vertex mean. Atomic completion requires both endpoint errors
below 10 cm; it is not a new composed-task evaluation.

| Row | Complete | Contact % | FS | HS frames % | HS mean | OS mean |
|---|---:|---:|---:|---:|---:|---:|
| reconstruction |22/28|68.4657|.129277|34.2028|4.014550|30.533886|
| A00-increment |22/28|68.0543|.139963|33.8196|3.940869|30.645320|
| A01-increment |22/28|68.0939|.160854|35.9975|2.212554|21.285703|
| A00-armijo |22/28|68.4657|.129318|34.2166|4.014418|30.532469|
| A01-armijo |22/28|68.3976|.137591|34.5061|3.894199|26.504058|

All three primary comparisons use 10,000 seed-42 paired replicates and Bonferroni
98.3333% percentile intervals, with negative upper limits required at both units:

| Primary contrast | Delta | 28-episode CI | 4-scene CI |
|---|---:|---|---|
| A00-armijo − A00-increment, FS |-.0106451|[-.0178746,-.0046410]|[-.0162632,-.0050597]|
| A01-armijo − A00-armijo, OS mean |-4.028411|[-9.143940,-.651786]|[-5.942986,-1.832115]|
| A01-armijo − reconstruction, OS mean |-4.029828|[-9.147277,-.652113]|[-5.946129,-1.833127]|

A00 FS falls 7.61% against old A00 and lies .0000409 above reconstruction. Its
contact and completion match reconstruction episode by episode. It is numerically
close to reconstruction, not an exact replay: very small correction and native
outcome differences remain. Tiny secondary hand/body penetration increases and
OS-mean decreases are nominally significant at both units and are retained in
the full tables; they are not meaningful evidence of a quality intervention.

A01 OS mean improves 13.20% against reconstruction. Completion outcomes match in
all 28 episodes. Contact falls .06812 percentage points from reconstruction and
new A00, with nominal intervals crossing zero. A01 FS is .008313 (6.43%) above
reconstruction: nominal episode CI [-.007186,.033708] and scene [-.007046,.032112].
HS frame ratio rises .003034 (.3034 percentage points), with episode interval
[-.007550,.016053] and scene [-.008352,.015003]. These unresolved increases pass
the registered absence-of-significant-harm protections; they do not establish
equivalence. HS mean decreases .120350 (3.00%), significant only at scene level.
This result does not meet the wider Phase 2 human-scene composition claim.

Against old A01, OS mean rises **5.218355**, nominal episode CI [1.752984,9.618991]
and scene [3.639219,6.629081]. HS mean rises **1.681645**, episode [.027105,4.873212]
and scene [.046762,4.756846]. HS maximum rises 17.668228 and OS maximum 19.435211,
also significant at both units. Armijo retains 43.57% of old A01's OS-mean benefit
against reconstruction. A01 FS falls .023263 from old A01, but both nominal
intervals cross zero. Its object endpoint error rises .112002 cm, significant
only at scene level. The pilot gate did not require retaining the old depth gain;
this measured tradeoff must accompany the pass classification.

All 16 native outcomes, every registered nominal contrast and the solver×geometry
interaction are in the compact result. Interactions are positive at both units
for object endpoint error (+.205658 cm), HS mean (+1.608096), HS maximum (+16.348304),
OS mean (+5.331206) and OS maximum (+18.080478); FS and HS-frame interactions are
unresolved. No outcome or unfavorable comparison is omitted.

## Same-source solver evidence

All stored accepted steps satisfy Armijo and strict decrease. All final complete
objectives are no greater than their sources. The accepted-state energies
independently reconstruct at atol 1e-8/rtol 1e-6. All source masks, initial contact
labels, history and output contact channels are preserved.

| Count | A00-armijo | A01-armijo |
|---|---:|---:|
| Corrections |372|372|
| Full-objective decreases / ties / increases |109 /263 /0|247 /125 /0|
| Same-source Adam decreases / ties / increases |0 /33 /339|26 /13 /333|
| Gradient evaluations |738|4,080|
| Objective evaluations, including trials |9,079|25,560|
| Line-search trials |7,597|20,736|
| Accepted updates |371|3,875|
| Stop: iteration budget / zero gradient / search exhaustion |5 /33 /334|167 /13 /192|

A00's episode-first complete objective stays at floating-point scale,
4.29913e-13→3.01475e-13, versus .0474352 after its same-source Adam shadow. The
mean final maximum parameter magnitude is 5.46211e-8, human RMS movement
8.03911e-7 cm and object RMS movement zero. Its stance and endpoint energies stay
exactly zero. A00's 334 search exhaustions mostly express the remaining numerical
floor; they do not represent a failed scene objective because A00 has none active.

A01's complete objective falls .0535347→.0399240; Adam on the same inputs ends
at .0808130. Final Armijo terms are residual .00202380, contact .00004820, stance
.00055022, endpoint .00017017, human scene .00195871 and object scene .03517293.
All 217 corrections with positive initial scene energy lower both complete and
scene energy. Of these, 167 reach the 20-iteration budget and 50 exhaust search
after some accepted updates. The other 155 start with zero scene energy: 125
remain fixed and 30 make numerical contact improvements. All zero-update A01
solves belong to that latter stratum. This separates absence of scene pressure
from searches that stop before convergence.

Armijo minus same-source Adam complete-objective change is -.0474352 for A00,
nominal episode CI [-.0530399,-.0415146], scene [-.0508028,-.0440676]; for A01
it is -.0408890, episode [-.0457565,-.0356252], scene [-.0450966,-.0366814].
A01 human RMS movement is .202190 cm versus .870231 cm under Adam, and object
movement .221276 versus .780101 cm. Their paired differences are negative at
both units. These within-source findings explain solver behavior; native rows
follow different closed-loop source trajectories. Step 10/1/0 and initial/generated
history strata, contact/stance diagnostics and all paired motion intervals remain
in the compact result and `analysis/strata.json`.

## Scene-domain issue retained

The existing voxel query gives zero displacement to points with invalid grid
indices. Integer truncation also makes geometric out-of-grid membership differ
from query validity near the lower boundary. Both were recorded without modifying
the objective or the native evaluator.

A01 source/corrected invalid object-point fractions are 1.75168%/1.74864%, and
geometric out-of-grid fractions 1.92437%/1.91609%, episode-first. Seven newly invalid
object-point observations occur in four corrections, with no newly invalid human
point. Three are scene b1b053a9, floorlamp, window 5, steps 10/1/0: six point
observations already had zero scene residual, so crossing out adds no loss drop.
The fourth is scene 0aa05d5a, monitor, window 7, step 0: one point loses .000139799
of scene energy on becoming invalid. The cohort episode-first mean drop from all
newly invalid points is 8.32136e-7. It is a real query-domain loophole, while these
measurements do not attribute the full native gain to boundary movement.

The sparse-joint/object-point optimization objective differs from native mesh-SDF
evaluation. Lower complete loss cannot certify collision removal, support labels,
motion realism or useful HSI supervision. Full HSI composition remains open.

## Execution, verification and provenance

All eight GPU jobs and automatic analysis exited zero: 56 episodes, 248 windows,
744 solves, 123,752 CG calls, 4,818 solver gradients, 34,639 objective evaluations,
28,333 trials and 4,246 accepted updates. The 744 Adam shadows add 14,880 gradient
updates; diagnostic per-term derivative calls are additional work, not solver
iterations. Both rows retain 5,376 evaluated frames. Native projections are
complete for all 28 episodes and four scenes; saved joints reproduce every FS value.

GPU job wall 36m52s; manifest through analysis 38m28s. Generation sums are
5,375.08/5,750.01 s for A00/A01, correction sums 107.93/293.50 s and shadow/archive
sums 206.50/212.65 s. Peak allocated memory is 804.43 MiB. Eight sampling lanes
include brief initial trace verification on GPU 7; these instrumented, sharded
timings describe this execution rather than isolated production latency.
The 56 motion files total 710,829,890 bytes. The formal run provided functional
and synchronized timing/memory verification. No separate smoke or benchmark,
operational failure, sampling restart or result overwrite occurred.

Implementation suite: **932 passed, 4 skipped in 171.78 seconds**. Completion
suite: **932 passed, 4 skipped in 171.53 seconds**. Registry valid with 356 records.
All native, monotonicity, frozen-source and saved-state audits pass. The completion
log is `completion-tests.log` beside the manifest.

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
git diff --check
```

- Preregistration 68b41ed; implementation and complete run/analysis source 374725a.
  Manifest start and finish record that same source. This handoff, compact result,
  plan and registry completion form the completion commit.
- Integrate by fast-forward into phase/02-mixer and tag exp/p2i-armijo-v1 after
  completion checks. The tag seals the pilot; it does not close full Phase 2.
- Compact: `experiments/results/p2_mixer_armijo_s42_20260906.json`.
- Run: `results/experiments/p2-mixer-armijo-s42-20260906/`. Preserve manifest,
  preflight, all eight resolved configs, command/logs, full motions/traces,
  native/solver/factorial/adjusted reports, `analysis/strata.json`,
  `analysis/scene-activity-strata.json`, `analysis/runtime-audit.json`,
  `analysis/saved-native-fs-check.json`, `closure_audit.json` and test logs.
- Upstream data/checkpoint identities are reused by the execution plan's archived
  manifest references; unchanged inputs were not rehashed.

## Exact next entry

Read this handoff, OVERVIEW.md and PHASE_2_COMPOSITION.md. Retain reconstruction,
the fixed experts and Armijo as the passing pilot candidate. Begin a read-only
review of A01's scene-reference switching, grid validity and the 50 positive-scene
solves that exhaust line search after descent. Use saved states to distinguish
finite solver budget from a discontinuous or uncovered objective before proposing
one next experiment. Do not start learned training based on this pilot pass.
Any new workload or mechanism needs its own concrete approval. Close only 2.9.
