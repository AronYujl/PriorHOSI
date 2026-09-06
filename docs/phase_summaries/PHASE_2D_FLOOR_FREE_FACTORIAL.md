# Phase 2.4 floor-free factorial — 2026-09-06

**Deliverable PASS; geometry and HSI quality gates FAIL.** Retain reconstruction
as the comparison anchor. Geometry reduces object-scene depth while increasing
sliding. The HSI target significantly gives back geometry's object-scene gain and
loses one completed task. Full Phase 2 and learned-mixer training remain open.

## Scope and implementation

User approved A01, A10, A11 with forced floor excluded; reuse Phase 2.3 no_floor
as A00 and zero-step reconstruction. One inherited config fragment selects the
cell. Runtime, expert and core code remain unchanged. Fixed experts are R2 final
EMA + CG and P15 online + Arm B; known-empty input, raw gate 0, 500 steps,
499 CG calls/window, corrections at 10/1/0, 20 Adam steps at LR .05 with existing
bounds, physical scales and masks. All twelve resolved configs match sealed
no_floor except run/output locations and cell selection.

Same seed-42 four scene bins 0/22/44/66 and seven objects per scene: 84 new
episodes/372 windows plus 56 reused episodes. Eight authority RTX 3090 lanes;
no training, parameter search or separate smoke workload. The unchanged runtime
path needs no separate performance benchmark; formal batch-1 timing is retained.

## Results

Contact and HS frames below are percentages. Native feet height is an estimated
floor-height/engagement proxy in cm. HS/OS s_mean sums penetrating vertex depths
per frame and averages across frames, not across vertices. s_max is the maximum
frame depth sum. Completion means both endpoint errors <10 cm, not state-machine
success.

| Row | Completed | Contact % | FS | Feet cm | HS frames % | HS s_mean | OS s_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| reconstruction | 22/28 | 68.466 | 0.129277 | 3.903 | 34.203 | 4.014550 | 30.533886 |
| no_floor | 22/28 | 67.620 | 0.131133 | 3.895 | 33.926 | 4.112274 | 30.581380 |
| a01 | 22/28 | 68.062 | 0.207636 | 4.185 | 36.000 | 2.115831 | 19.052997 |
| a10 | 21/28 | 67.356 | 0.145477 | 3.912 | 33.515 | 3.311988 | 31.184262 |
| a11 | 21/28 | 67.530 | 0.158015 | 3.978 | 33.627 | 3.790068 | 25.386331 |

Primary OS s_mean family: five contrasts, 10,000 paired seed-42 resamples,
Bonferroni 99% percentile intervals at each unit. Require improvement at both
units. These bootstrap intervals do not provide cross-seed uncertainty; only four
scenes limits scene inference. All remaining intervals are nominal 95% secondary
intervals, including the factorial interaction. Extra metrics in the raw adjusted
report are not members of the registered primary family.

| Contrast | Delta | Episode 99% CI | Scene 99% CI |
|---|---:|---|---|
| a01-no_floor | -11.528382 | [-26.872415, -1.794158] | [-16.512715, -6.172994] |
| a11-a01 | 6.333334 | [1.124720, 13.776336] | [4.999863, 7.570164] |
| a10-no_floor | 0.602882 | [-1.710726, 3.806756] | [-1.406165, 2.788660] |
| a01-reconstruction | -11.480888 | [-26.628310, -1.823156] | [-16.276470, -6.397046] |
| a11-reconstruction | -5.147555 | [-13.461365, 0.265116] | [-8.706306, -1.201848] |

A01 reduces OS mean depth 37.60% against reconstruction and passes the primary
OS comparisons. Completion outcomes are identical; contact drops only .4032
percentage points. However FS increases .078359 (+60.61%), nominal episode CI
[.026478,.139446], scene [.038033,.114286]. Sliding also worsens against A00 at
both units. This violates the registered protection and fails geometry promotion.
HS depth improvement against reconstruction is unresolved at episode unit
[-5.559232,.012506], despite a negative scene interval; HS frame prevalence is
unresolved at both units. Object-scene maxima improve at both units.

A11-A01 increases OS s_mean 6.333334 (+33.24%), significant even after primary
adjustment at both units; OS s_max also increases, nominal episode CI
[6.074242,40.187859], scene [10.241709,31.869665]. Sliding decreases .049621,
but the episode CI [-.111446,.006944] crosses zero; scene CI
[-.069049,-.031408] is negative. Thus the earlier significant sliding benefit
with floor active is not re-established at both units here, while its OS cost
persists. A11-reconstruction OS benefit is unresolved at adjusted episode level.
A11 loses 1/28 completion (3.5714 percentage points), exceeding the two-point
budget, and its FS worsens at scene unit [.000142,.071148]. HSI promotion fails.
Its small HS mean-depth improvement against reconstruction is nominally significant
at both units; preserve this benefit alongside the failed gate. A11 contact is
.9358 percentage points below reconstruction, below the point budget but
nominally significant at scene unit only.

A10 alone gives no adjusted OS benefit against A00 and loses the same completion.
For both HSI rows the only completion change is scene
b1b053a9-b268-4f62-a06d-b9b9325c5092 / clothesstand / test_idx 0:
object endpoint error 10.4862 cm (A10), 10.4114 cm (A11), while pelvis errors stay
below 10 cm. The completion contrast CI includes zero; the preregistered point
protection still fails. No success threshold was changed after seeing this case.

The secondary factorial HSI×geometry interaction is +5.730451 for OS mean,
episode CI [1.417802,11.579067], scene [3.001423,8.459479], and -.063965 for FS,
episode [-.130450,-.002890], scene [-.104780,-.023150]. The HSI effect depends
on geometry; averaging it into one main effect would conceal the measured cost.
These are total closed-loop effects including feedback interactions, not isolated
stance-mask or optimizer-gradient-conflict identification.

## Execution, verification and artifacts

All twelve GPU jobs and automatic analysis exited zero. Ten pairwise nominal
reports, two factorial reports and the adjusted primary report completed. Every
new row has 28 episodes, 124 windows, 61,876 finite CG calls, 372 corrections and
7,440 finite optimizer steps. Native values and all optimizer telemetry are finite;
history/contact exact throughout. No operational failure occurred. Prior negative
results and failed runs remain archived.

Manifest wall time 2439 s (40m39s), eight RTX 3090 GPUs. Each row generated 5376
frames. Generation sums A01/A10/A11: 5357.76/5297.65/5384.24 s; mean correction
.56255/.56177/.56740 s; peak allocated memory 804.43 MiB each. Concurrent sums
are not isolated production latency. Initial stability passed all eight lanes.
Persistent episode artifacts survive interruption; no mid-window resume exists.

Implementation suite: 916 passed, 4 skipped in 160.39 s. Completion suite: 916 passed, 4 skipped in 157.04 s; registry valid with
346 records. All paired metrics have complete finite coverage; diff check passed.

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
```

Preregistration: 5328f6c. Executed config implementation: c9db7b2.
Completion commit contains this handoff, compact result, plan and registry.
Integrate by fast-forward into phase/02-mixer; exp/p2d-floor-free-factorial-v1
identifies the completed diagnostic deliverable, not a passed quality gate.

- Compact metrics/CIs: experiments/results/p2_mixer_floor_free_factorial_s42_20260906.json
- Full run: results/experiments/p2-mixer-floor-free-factorial-s42-20260906/
- Reused controls: results/experiments/p2-mixer-common-diagnostic-s42-20260906/
- Sealed input provenance is referenced from execution_plan.json; existing
  lifecycle manifests retain source/config/dependency and input identities.

## Exact next entry

Read this handoff, docs/plan/OVERVIEW.md, current Phase 2 plan and
HSIPRIOR_DESIGN_PRIORS.md before proposing more HSI work. Keep R2+CG/P15+Arm B
fixed and reconstruction as anchor. Geometry's sliding cost remains after floor
removal; the tested neural increment retains an OS cost. Neither recipe is
promoted. A later read-only review can inspect saved root/object/articulated and
stance changes to localize the geometry/sliding conflict before proposing one
concrete diagnostic. The interaction alone does not identify that mechanism.
Do not tune weights, redesign HSI or start learned training from this result.
Any new workload needs a separately approved concrete proposal. Full evaluation,
motion realism and useful learned HSI composition remain open. Close only 2.4.
