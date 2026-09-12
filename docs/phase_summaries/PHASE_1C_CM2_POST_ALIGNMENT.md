# Phase1C CM2.4: post-training common-state endpoint diagnosis

Completed 2026-09-12 on phase/01c-cm2-postalign. CM2 improves relative-body endpoint
retention, while the low-noise boundary-velocity error on CM1 histories regresses.
Keep corrected R2+CG as the quality baseline and the CM2.3 failed joint gate.
Phase1C remains open; this session completes the diagnosis and a successor proposal.

## Scope and implementation

One configuration inherits the existing distillation_alignment probe. Source
trajectories remain R2 DDIM25 U and CM1 CM16 U; only the frozen observation student
changes to CM2 final epoch089. The observer teacher remains frozen eval R2.
All runtime, sampler, model, core, data and interpolation code is unchanged.

The exposed B_n60 cohort supplies60 episodes/364 windows per source, including12
terminal padded windows. Timesteps499/279/59 yield2184 states. CFGw1, seed42 and
external CG off are fixed. CM1 observations reuse sealed CM2.1 artifacts.

All shared state arrays, saved conditions and both teacher predictions exactly
match CM2.1. All120 source coarse motions and every sequence's68 native fields
match. Sixteen shard records agree on R2 teacher and CM2 student identities.
Source label cm16 always means CM1; these source rollouts are not CM2 rollouts.

## Findings and uncertainty

At t59, sequence-equal endpoint errors are:

| Source | CM1 body cm | CM2 body cm | CM1 boundary velocity m/s | CM2 boundary velocity m/s |
|---|---:|---:|---:|---:|
| R2 DDIM25 |1.694874|1.540839|0.155599|0.159745|
| CM1 CM16 |1.776597|1.641616|0.143158|0.152896|

CM2−CM1 body changes are−0.154035cm, family8 CI[−0.211885,−0.094992], and
−0.134981cm,[−0.201577,−0.073197]. Both279 body contrasts also improve.
The8 primary contrasts yield4 improvements,1 regression and3 uncertain results.
The regression is CM1-source t59 boundary velocity:+0.009738m/s,
family8 CI[0.001838,0.017535]. R2-source t59 velocity is uncertain.

At t59, root and global-FK changes have pointwise intervals crossing0, while
rotation errors improve by about0.21degrees at pointwise95%. CM2 retains body
gaps of1.026/1.064cm and velocity gaps0.122/0.113m/s relative to R2's local
prediction to the same endpoint. All12 pair reports and all secondary metrics,
499 descriptions, strata/population-weighted means and worst cases are retained.

First-window boundary-velocity point estimates improve in both sources; later
generated-history estimates regress. These splits are descriptive, not a paired
history-substitution intervention. They do not identify a sole dropout/EMA,
training-history or CG cause. Inference remains limited to one training seed and
the exposed development cohort. CM2.3 full375/holdout355 acceptance is unchanged.

## Mechanism and the concrete successor proposal

The probe clamps two identical history frames. If e_k is student-minus-teacher
FK error, e_0=e_1=0, so its first boundary-velocity error is e_2/dt and its first
boundary-acceleration error is e_2/dt², with dt0.1s. Their squared losses are
first-future-frame endpoint weighting. Boundary jerk also involves e_3 and has
additional temporal structure. Native physical/safety metrics remain separate.

The registered branch selects a boundary-target proposal. The report makes its
meaning explicit: preserve the entire CM2 recipe and add a soft, low-noise
first-future-frame absolute FK target against the same frozen R2 endpoint, using
all24 joints scored by the velocity diagnostic, including root. Average over the
original batch with the existing19/39/59 HSI gate. Reuse the teacher endpoint;
retain the existing14-frame/body22 target. The first future frame stays learnable.

Proposed coefficient: one real seed42 batch, min(10% CM2-total trunk gradient
ratio,25% consistency rotation-head gradient ratio). A new student would start
from R2 with the same120172544-window/58678-update/effective2048 budget, CM16/CFG1/
CG and full acceptance. The target can still transfer poorly to generated history
or degrade later frames/contact/semantics. It is a proposal, with no successor
config, run ID, training or CG change allocated in this session.

## Verification, resource and provenance

Both GPU jobs, their merges and analysis exit0.26 targeted consistency tests
pass.18 actual Hydra job configurations resolve fully; registry validation and
coverage verification are archived. Authority487 passed/3 skipped is reused
because runtime code is unchanged. No training benchmark or extra functional
workload is required for the unchanged path. Both formal diagnostics provide
the measured native replay evidence.

Pointwise bootstrap uses tools/paired_bootstrap.py,10000 paired episode draws,
seed42. The GPU float64 family8 quantiles agree with the tool's pointwise bounds.
Total reserved cost0.815017GPU-h<4: R2-source0.439083, CM1-source0.372014,
analysis0.003920. Eight RTX3090 GPUs had recorded external inference contention;
minimum sampled headroom10830MiB. No deployment-speed claim or optimizer update.
All formal workloads succeeded.

Preregistration66326d5; configuration and execution5791583; completion seals this
summary and its report. No phase merge or expert tag.

- experiments/results/p1_hsi_cm2_post_alignment_s42_20260912.{md,json}.
- results/cm2_post_alignment_setup_20260912/:jobs, input references,12 paired
  reports/family8, full error summaries, replay audits, worst cases, verification
  and completion records, launchers and original logs.
- results/hsi_cm2_post_alignment_s42_20260912/:all2184 state observations.
- results/experiments/p1-hsi-cm2-postalign-{r2-ddim25,cm16}-s42-20260912/:
  manifests, resolved configurations, preflight and GPU resource records.

Next session reads this summary, docs/plan/OVERVIEW.md and CM2.4 in
docs/plan/PHASE_1C_HSI.md. Review the precise first-future-frame target proposal
and its transfer risk before implementing any successor. R2+CG remains baseline.
