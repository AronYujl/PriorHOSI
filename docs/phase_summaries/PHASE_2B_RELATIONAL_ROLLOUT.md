# Phase 2.2 relational closed-loop handoff — 2026-09-06

The five-row native pilot completed. **The deliverable gate passes and the
closed-loop quality gate fails.** Keep R2 final EMA + CG and P15 online + Arm B
as the selected experts. The tested relation correction receives no production
promotion, and learned-mixer training remains pending.

## Scope and implementation

Branch: `phase/02b-relational-rollout`; integration: `phase/02-mixer`.
The user's fixed-expert continuation followed the completed Phase 2.1 window
experiment. Phase 2.2 connects that bounded relation operator to the actual
shared chain, then evaluates all registered native metrics on generated history.

At reverse steps 10,1,0, `RelationalCorrector` updates clean motion before its
DDPM posterior and before carrying it to the next temporal scene query. It uses
the existing 67 coordinates/frame: common human/object translation and yaw,
plus 21 local joint rotations reconstructed with FK. History and contact
channels are restored exactly. Object pose comes from HOI with the common
physical transform, which the sampler audit records explicitly.

The four-cell window optimizer now also accepts a single cell for independent
rollout. The passive and active paths share one problem constructor. All four
optimization rows query R2 through the known-empty input process at the same
three steps. They use 20 Adam steps at 0.05 and the Phase 2.1 physical scales,
bounds and masks. The opt-in evaluator records each finished episode's metrics,
timing and cumulative sampler audit. The experiment adds one config fragment,
`code/config/config_sample_hosi_relational_rollout.yaml`, and uses the existing
Hydra evaluator and paired-bootstrap tool. Expert and core source is unchanged.

## Fixed experiment

Run: `p2-mixer-relational-rollout-r1-s42-20260906`, seed 42. Each row contains
the same metadata-selected scene bins 0,22,44,66 and seven objects per scene:
28 episodes/124 windows per row, 140 episodes/620 windows total.

Every row uses 500 diffusion steps, matched A* goals and posterior noise,
R2-CG human-scene guidance at all 499 nonzero steps with coefficient1 and scale
1, followed by P15's sealed Arm B last-ten-step guidance. The reference retains
HOI clean predictions with that same posterior guidance. All other rows use the
common relation/floor/stance/endpoint/residual objectives. A10 adds the retained
HSI conditional-minus-temporal-masked FK target, A01 adds human/object geometry,
and A11 adds both.

The reference and A00/A01 isolate geometry and common constraints; they provide
no evidence of a benefit from the R2 learned increment. The older 469-episode
G=0 row has a different CG state and supplies context rather than the paired
reference in this experiment.

## Native result and verdict

All rows have 28 episodes. Percent columns below are episode means. HS/OS
`s_mean` is the native evaluator's per-frame sum of penetrating vertex depths,
averaged over frames; these values are not mean per-vertex depths or the Phase
2.1 window residuals in centimetres. FS retains the native evaluator's scale.

| Row | Completed | Contact % | FS | Feet height cm | HS frames % | HS s_mean | OS s_mean | OS frames % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference: matched CG/Arm B | 22/28 | 67.340 | 0.17468 | 4.012 | 38.534 | 3.79587 | 30.11801 | 27.306 |
| A00: common constraints | 18/28 | 68.510 | 0.55478 | 1.670 | 75.405 | 4.58570 | 31.42949 | 29.803 |
| A10: add HSI increment | 19/28 | 67.633 | 0.37927 | 1.860 | 71.409 | 5.57446 | 35.17554 | 29.828 |
| A01: add geometry | 19/28 | 68.435 | 0.58421 | 1.732 | 75.379 | 2.69699 | 22.52051 | 29.718 |
| A11: add both | 18/28 | 67.909 | 0.37042 | 1.788 | 71.295 | 4.48509 | 26.32717 | 29.711 |

All 15 native metrics plus completion are retained in the row summaries and
in every paired report. The key registered contrasts are:

| Contrast / metric | Difference | Episode 95% CI | Scene 95% CI |
|---|---:|---|---|
| A01-A00 OS s_mean | -8.90899 | [-20.16104,-2.02544] | [-15.44962,-4.00424] |
| A01-reference OS s_mean | -7.59750 | [-18.34687,0.52300] | [-12.85169,-0.69572] |
| A01-reference completion | -0.10714 | [-0.21429,0] | [-0.21429,0] |
| A01-reference FS | +0.40953 | [0.30640,0.51374] | [0.30537,0.51369] |
| A01-reference HS frame fraction | +0.36846 | [0.28284,0.45687] | [0.30223,0.43468] |
| A00-reference FS | +0.38010 | [0.27575,0.48858] | [0.30338,0.45682] |
| A00-reference completion | -0.14286 | [-0.28571,-0.03571] | [-0.25000,-0.03571] |
| A11-A01 FS | -0.21379 | [-0.26534,-0.16596] | [-0.26535,-0.16946] |
| A11-A01 HS frame fraction | -0.04084 | [-0.07091,-0.00795] | [-0.07735,-0.00216] |
| A11-A01 OS s_mean | +3.80666 | [0.67010,8.07866] | [1.59575,5.64067] |

The pilot geometry gate fails. Its OS improvement against A00 passes, while
its episode CI against reference crosses zero. Its completion point loss is
10.714 percentage points, beyond the registered two-point allowance even though
the completion CI touches zero. Sliding and human-scene frame prevalence worsen
at both resampling units. Contact protection passes. The scene-only OS interval
cannot replace the failed episode criterion or the failed quality protections.

The HSI increment has a measurable tradeoff: it reduces sliding and human-scene
frame prevalence relative to A01, and increases object-scene depth. Completion
also loses one task relative to A01 (a 3.571-point loss) and four relative to
reference. Thus the result refines the earlier window-only negative; it does
not support calling every HSI effect ineffective, or promote this target into
a training objective.

## What the controlled contrasts identify

A00 already produces most of the sliding and human-scene prevalence cost.
The damage therefore belongs upstream to the common reconstruction/objective
bundle, before either tested factor is added. A01 then improves OS depth within
that damaged trajectory family. This improvement and the failure against the
matched reference are both part of the result.

The objective moves the lower toe height toward 2 cm, while its horizontal
stance loss is gated by foot-contact heights in the source HOI prediction.
That fixed mask can omit newly planted feet. Feet height falls from 4.012 to
1.670 cm in A00 while native sliding rises. This makes the floor/stance coupling
a concrete candidate for a future diagnostic. The current experiment changes
FK reconstruction and several common objectives together, so it has not isolated
the floor term, the stance mask or any individual gradient as the cause.

There are four scene units. Report episode and scene uncertainty together and
retain all rows. The intervals concern this pilot and do not establish the full
Phase 2 quality gate, general scene specificity, or a learned-mixer result.

## Execution, verification and retained failure

All 20 GPU processes and six bootstrap analyses exited zero. Every row covers
the identical 28 episode names and 124 windows. There are 309,380 finite CG
applications and 1,488 relation corrections; all correction gradients are finite
and all corrected histories/contact channels are exact. All 140 native episode
audits and all 16 numerical outcomes per episode are present and finite.

The complete authority suite on runtime source passed 914 tests, with four
skips, in 172.44 seconds. New component tests establish single-cell/four-cell
optimizer equivalence and actual posterior/next-query feedback. The test fixture
was corrected for the existing step-zero posterior coefficient and valid SO(3)
history. Twenty exact configs resolved before execution; registry validation
passed. The formal workload supplied real-data and batch-1 runtime validation.
No additional smoke workload, training or parameter sweep was introduced.
The final completion verification also passed 914 tests with four skips in
160.66 seconds; registry and compact-result/reference checks passed.

Verification commands:

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
"$INFBAGEL_PYTHON" code/test_infbagel_hosi.py --config-name config_sample_hosi_relational_rollout --cfg job --resolve
```

The run's `execution_plan.json` records every exact evaluator override and GPU
assignment. Its archived `analysis.sh` records the six executed bootstrap
commands, with 10,000 replicates, seed 42 and 28 episode/four scene units.

The authority used seven RTX 3090 GPUs, physical indices 1–7, isolated through
`CUDA_VISIBLE_DEVICES`, with four BLAS threads/process. GPU 0 was occupied by a
separate HSI evaluation at launch. This allocation remained fixed. The run
manifest spans 70 minutes 17 seconds. Mean/median/p95 correction time is
0.570/0.574/0.618 seconds; maximum recorded allocated memory is 804.43 MiB.
These correction measurements include HSI target preparation and resident model
allocation. Concurrent shard generation totals 26,539.04 seconds over 26,880
frames; these are workload accounting and not isolated production latency/FPS.

The original id, `p2-mixer-relational-rollout-s42-20260906`, failed operationally
before sampling: its nohup controller was absent on first inspection, the log
was empty, and zero shard-start records existed. The terminating signal was
unrecorded. Its manifest was marked failed and its registry row retained. The
fresh r1 id ran under a detached host-owned `screen` session. All evaluation and
analysis completed locally; there was no worker transfer/recovery to repeat.

Preregistration: `59ce660`; implementation/config/tests: `72eb67e`; retained
launcher failure and executed source: `3e17090`. The completion commit contains
this handoff, dated plan result, compact result and registry completion. Integrate
by fast-forward into `phase/02-mixer`; tag `exp/p2b-relational-rollout-v1` marks
the completed pilot and its negative quality result.

Artifacts:

- Compact result: `experiments/results/p2_mixer_relational_rollout_r1_s42_20260906.json`.
- Complete local run: `results/experiments/p2-mixer-relational-rollout-r1-s42-20260906/`.
- Its manifest, resolved configs, machine preflight, execution plan, episode/native
  outputs and `analysis/` retain all source identities, outcomes and uncertainty.
- Failed launch: `results/experiments/p2-mixer-relational-rollout-s42-20260906/`.

## Exact next entry point

Read this handoff, `docs/plan/OVERVIEW.md` and the latest Phase 2 plan sections.
Keep R2+CG / P15+Arm B fixed. Review the common FK/source/floor/stance/endpoint
correction bundle before proposing one isolated diagnostic. Preserve the matched
CG reference and A00, measure native sliding and scene/hand engagement together,
and retain the HSI increment's measured sliding-versus-object-depth tradeoff.
Useful HSI supervision and the full Phase 2 quality gate remain prerequisites
for learned-mixer training. This closing session starts no new direction.
