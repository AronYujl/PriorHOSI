# Phase 2.6 source-relative stance height — 2026-09-06

**Deliverable PASS; pilot quality FAIL.** Coverage expands and OS benefit remains,
but the registered sliding improvement is unresolved at episode level and HS
penetrating-frame prevalence worsens against reconstruction. No recipe promotion.

## Scope and implementation

User approved two height-only interventions: A00-height/A01-height, 28 episodes
each. Reuse Phase2.5 A00/A01 and Phase2.3 reconstruction. R2 final EMA+CG,
P15 online+Arm B, seed42, bins0/22/44/66 and seven objects; same 500 steps,
499 CG calls/window, corrections10/1/0, 20 Adam/.05, original bounds/scales,
contact labels and two-adjacent-source-frame stance rule. Floor objective remains
off; HSI factor off. No expert/core, native evaluator or training change.

One config enables source_floor. Before each optimization, the detached 16-frame
FK toes are interpolated to the native scale3 grid with the last real sample
retained and its two held duplicates omitted. Select 3D displacement <.005m,
repeat the last real velocity at the final sample, apply native DBSCAN height
rule eps=.005/min_samples3, take minimum group median including noise as the
existing evaluator does. Zero if no low-speed samples. GPU implementation uses
one-dimensional core components and input-order border assignment, verified
against the native CPU reference. Freeze the resulting floor and two-frame mask
through all 20 optimizer steps. No final episode floor is available to inference.
This aligns the rule on source FK; it does not make short-window FK identical to
full-episode interpolated SMPL evaluation.

Telemetry records source floor, sample count, stance count and absolute toe
height, plus existing correction/window/stitched/evaluated motion snapshots.
Default world-height path remains available with source_floor=false. Component
tests cover CPU/GPU native-rule agreement, no artificial endpoint support, frozen
height/mask during optimization and default/explicit-world output identity.

## Native results and gate

Contact and HS frames are percentages; feet height is the native estimated floor
proxy in cm. HS/OS s_mean is the frame-average sum of penetrating vertex depths,
not per-vertex depth. Completion requires both endpoint errors <10cm and is not
state-machine success.

| Row | Complete | Contact % | FS | Feet cm | HS frames % | HS mean | OS mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| reconstruction | 22/28 | 68.466 | 0.129277 | 3.903 | 34.203 | 4.014550 | 30.533886 |
| a00 | 22/28 | 67.620 | 0.131133 | 3.895 | 33.926 | 4.112274 | 30.581380 |
| a01 | 22/28 | 68.062 | 0.207636 | 4.185 | 36.000 | 2.115831 | 19.052997 |
| a00_height | 22/28 | 69.183 | 0.140208 | 3.925 | 36.347 | 4.562122 | 30.920008 |
| a01_height | 22/28 | 69.038 | 0.140129 | 3.529 | 38.794 | 2.060411 | 20.910681 |

Three primary paired comparisons use 10,000 seed42 replicates and Bonferroni
98.3333% percentile intervals at episode and four-scene units. Both units must
pass. Remaining comparisons are secondary nominal95%, not familywise claims.

| Primary comparison | Delta | Episode adjusted CI | Scene adjusted CI |
|---|---:|---|---|
| a01_height-a01/foot_sliding | -0.067507 | [-0.197414, 0.045276] | [-0.118651, -0.006551] |
| a01_height-a00_height/scene_obj_penetration_s_mean | -10.009327 | [-23.242818, -1.697920] | [-15.788068, -4.617620] |
| a01_height-reconstruction/scene_obj_penetration_s_mean | -9.623205 | [-21.992746, -1.781231] | [-14.632658, -5.351171] |

**Sliding primary fails as unresolved.** A01-height FS .140129 is 32.51% below
old A01 .207636, but the episode CI crosses zero even at nominal95%
[-.172895,.023369]. Scene evidence is positive for the intended improvement,
but does not satisfy the two-unit gate. Against reconstruction/new A00 the FS
differences are unresolved at both units; these are not equivalence findings.

**Object-scene primary passes.** A01-height OS mean20.910681 is 31.52% below
reconstruction30.533886 and 32.37% below new A00. Both adjusted comparisons pass
at both units. Yet it is worse than old A01 by1.857684 (+9.75%), nominal episode
CI [.367184,4.213514], scene [.825934,3.633997]. Preserve this lost portion of
geometry's benefit along with the retained benefit.

**HS prevalence protection fails.** A01-height raises HS penetrating frames
from reconstruction34.2028% to38.7938% (+4.5911 percentage points), nominal
episode CI [.2643,10.3281] points, scene [.9623,10.1065] points. Against old
A01 it also rises2.7935 points, significant at both units. Against new A00,
+2.4471 points is unresolved. HS mean-depth reduction against reconstruction
is unresolved at both units; reduced mean depth and increased frame prevalence
must not be collapsed into one penetration claim.

All contact/completion point protections pass against sealed counterparts and
reconstruction; candidate contact vs new A00 drops only .1442 percentage points.
Every episode completion outcome is identical to reconstruction, 22/28 for all
five rows. A00-height increases contact vs A00 by1.5627 points, nominally
significant at both units. However its HS prevalence +2.4209 points, HS mean
+.449848 and HS max frame-depth sum+4.577183 also worsen at both units.
Its HS maximum also worsens against reconstruction. A01-height improves HS
maximum against new A00 at both units; these secondary findings are all retained.

The secondary height×geometry interaction is -.076582 for FS (episode
[-.170654,-.002463], scene [-.132198,-.034748]) and +1.519056 for OS mean
(episode [.336795,3.158738], scene [.724647,2.401408]). These describe a changed
geometry effect under height alignment; they do not replace the failed primary
A01-height/A01 test or isolate a physical cause.

## Coverage, estimated height and limits

Empty source masks fall from338/372 (90.86%) in each sealed row to5/372 (1.34%)
in each new row. No correction lacks low-speed samples, so none uses zero-floor
fallback. Episode-first mean selected transitions grow to38.71/38.82 of56 per
correction (A00-height/A01-height); old means were1.414/1.403. Larger coverage
alone does not establish correct support labels or improved native quality.

A00-height source floors range2.3595–10.0193cm; A01-height1.6832–9.9218cm.
Episode-first source means5.5156/5.5273cm; within-episode ranges average
4.0739/4.2291cm. Mean changes between adjacent correction steps within a window
are .2729/.2543cm; consecutive-window step0 changes are1.8495/1.9015cm.
Thus across-window variation is substantial relative to the step-to-step variation,
but this alone does not prove instability is the cause of quality loss.

Mean absolute discrepancy from final native estimated floor is2.4308/2.9344cm;
for A01-height, a3df624b-0917-46e9-ac15-fab766276c72/smalltable/4 reaches an
episode mean discrepancy7.9299cm. The final native estimator is not physical
ground truth, so these are discrepancies rather than measured ground-height bias.
Mean absolute source toe heights7.7157/7.7088cm and native floor proxies
3.9254/3.5286cm are different statistics. Neither establishes that hovering has
been eliminated or newly created. Selected/outside-mask displacement averages
also change their membership substantially; do not compare their magnitudes to
old-mask means as if they covered the same transitions.

The coverage intervention works as implemented but gives a quality tradeoff.
Floor fluctuations, misclassified support, and geometry interactions remain
possible contributors; this run does not causally separate them. Retain the
full negative result instead of selecting a height threshold or weight afterward.
All 15 native metrics/completion, paired reports, floor/coverage measurements,
absolute toe heights, step/history strata and motion artifacts are preserved.

## Execution, verification and artifacts

All eight GPU jobs and automatic analyses exited zero. All 56 episodes/248 windows,
744 corrections, 14,880 optimizer steps and 123,752 CG calls completed. Native values,
optimizer telemetry, source floor and saved correction states finite; exact
history/contact; saved evaluated joints reproduce native FS for every episode.
56 motion files total 49,129,944 bytes. No operational or scientific-output corruption
occurred; the quality gate failure above is the experiment result.

Wall 2156 s (35m56s), eight authority RTX3090 GPUs. Each row 5376 frames; generation
sums 5509.39/5379.69 s, mean correction .57267/.56635s, peak allocated 804.43 MiB.
Concurrent accounting is not isolated production latency. Optional recording is
outside correction timing but included in sampling; episode serialization is
later. Formal run supplies real-data and batch1 runtime validation, with no
separate smoke workload. Initial stability passed all eight lanes.

Implementation full suite: 919 passed, 4 skipped in 160.74 s. Completion full suite: **919 passed, 4 skipped in 165.41 s**. Registry valid
with 350 records; diff check passes. Native paired coverage is complete at both
units, and all saved-native FS checks pass. Sealed input identities are reused by
manifest reference; no checkpoint/data transfer or re-hashing campaign.

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
```

Preregistration 0cef9e5; implementation d2cda7c. Completion commit contains this
handoff, compact result, plan and registry. Fast-forward into phase/02-mixer and
tag exp/p2f-stance-height-v1 after deliverable gate, with quality gate FAIL.

- Compact: experiments/results/p2_mixer_stance_height_s42_20260906.json
- Run: results/experiments/p2-mixer-stance-height-s42-20260906/
- Analysis: native/diagnostic projections and paired reports, height×geometry
  factorial, primary-adjusted, saved-native-fs-check, stage-records, runtime-audit
  and floor-temporal-variation. Estimates/masks/states retained per episode.
- Prior references: Phase2.5 stance-recording A00/A01 and Phase2.3 reconstruction.
  Prior failures and negatives remain intact.

## Exact next entry

Keep reconstruction anchor, R2+CG/P15+Arm B fixed, forced floor excluded. Neither
height recipe is promoted. Full Phase2, useful learned HSI supervision, motion
realism and learned training remain open. Read this handoff, overview and current
Phase2 plan before further work; read HSI design priors before proposing HSI work.
A later read-only review can examine the saved support changes and HS contact
locations to formulate one falsifiable next mechanism. Avoid automatic threshold,
weight or estimator smoothing sweeps: larger coverage did not establish quality.
Any new implementation/GPU intervention needs separate concrete approval.
Close only Phase2.6 in this session.
