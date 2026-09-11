# Phase1C CM2.1: fixed-state distillation endpoint diagnosis

Completed 2026-09-11 on phase/01c-cm2-align, with zero optimizer updates. Keep
corrected R2+CG as the quality baseline. Phase1C remains open.

## Scope and mechanism

The user approved a diagnosis matching teacher/student noisy states, generated
history and conditions before selecting a single-variable distillation proposal.
CM clean output approximates a trajectory endpoint; an intermediate R2 single-step
x0 has a different meaning. The named distillation_alignment probe therefore
compares both against the remaining native R2 DDIM trajectory's endpoint.

Two source trajectories, R2 DDIM25 U and fixed CM1 CM16 U, each cover the exposed
B_n60 development cohort:60 episodes/364 windows, including12 terminal padded
windows. Three common timesteps499/279/59 give2184 captured states. Both sources
retain native conditions, canonical seed42, CFGw1 and no external CG. Separate
frozen eval models and a separate sampler perform the observation; RNG restoration
preserves the original trajectory. This is not the stochastic training EMA target
or DDPM500+CG, and does not isolate dropout or training-loss causes.

One config and an HSI diagnostic component reuse the existing evaluator and
Sampler.p_sample. Core and the sampler's production arithmetic were unchanged.
The evaluator optionally attaches the probe and retains all native physics outputs.

## Findings

At t59, errors to the same teacher endpoint are:

| Source | R2 local body cm | CM body cm | R2 local boundary velocity m/s | CM boundary velocity m/s |
|---|---:|---:|---:|---:|
| R2 DDIM25 |0.51516|1.69487|0.03765|0.15560|
| CM16 |0.57799|1.77660|0.03964|0.14316|

Student-minus-local-teacher body errors at59 are+1.17971cm, simultaneous CI
[1.02732,1.36022], and+1.19860cm,[1.01662,1.43032]. All8 registered contrasts
(body and boundary velocity,279/59,two sources) have positive family8 simultaneous
intervals. The preregistered body-retention proposal branch is selected.

Root, direct-position and rotation discrepancies also exist. At59 student root
errors are1.674/1.730cm and rotation errors4.294/4.669degrees. The diagnosis does
not support a rotation-head-only cause. All errors, first/generated-history windows,
strata and B_n60 population-weighted descriptions are in the compact.

Native60 CM-minus-DDIM boundary jerk is+29.26078, pointwise CI[22.27865,36.94151];
exterior contact is−48.45269,CI[−88.68723,−13.51195]. Penetration,FS and interior
jerk changes are uncertain. Their favorable point estimates and all native metrics
remain. Both sources have zero >5g and low-pelvis-walk cases on this60 cohort.
These are development results, not a fresh full375/holdout355 acceptance.

## Verification, failures and costs

All120 native coarse trajectories exactly match sealed outputs; maximum difference0.
All2184 archived states are finite and have identical clamped histories. Continuing
the R2 native trajectory from its three captured times reaches exactly the same
endpoint, maximum difference0. Sixteen shard checkpoint records identify distinct,
consistent R2/CM weights. No model or checkpoint was updated.

Targeted19 passed; authority480 passed/3 skipped;18 fully resolved job configs.
Initial unit-test expectation for a step's third difference was corrected from100
to200m/s³; the original log remains. Two GPU jobs and merges exited0.

Six endpoint pair reports and family8 finished before native-summary aggregation
failed on undefined goal_orientation_err_rad. Recovery preserved null with0 defined
values, reused completed reports and added the seventh native pair. Failure exit1,
recovery exit0 and logs remain. An unsupported filesystem birth timestamp in the
initial cost addendum was corrected in final_completion.json; the earlier addendum
remains traceable. No GPU motion generation was repeated.

GPU workload cost0.855685GPU-h; minimum sampled headroom13524MiB. The statistics
kernel timer was not persisted before failure, so its one-GPU reservation is bounded
by25.673seconds from prior GPU completion to the failed statistics log's final write.
Total cost upper bound0.862817GPU-h<12. These are instrumented diagnostic costs;
no teacher single-GPU deployment latency was measured.

## Artifacts and source

- experiments/results/p1_hsi_cm2_alignment_s42_20260911.{md,json}.
- results/cm2_alignment_setup_20260911/final_completion.json and final_verification.json.
- results/cm2_alignment_setup_20260911/paired_statistics/:7 reports and family8.
- results/hsi_cm2_alignment_s42_20260911/:per-episode state/clean arrays and metrics.
- results/experiments/p1-hsi-cm2-alignment-*-s42-20260911/:manifests,18 configs,
  preflight,logs,completion records and memory samples.

Preregistrationc732e07; implementation/execution976b4e3; this summary and report
are sealed by the completion commit. No Phase1C merge or release tag.
Validation used the canonical exported INFBAGEL_PYTHON and ROOT_DIR:
`-m pytest tests/hsi/test_consistency.py -q`, `-m pytest tests -q`,
`tools/experiment.py validate`. Bootstrap used the existing tool,10000 draws/seed42,
60 paired episodes; GPU simultaneous quantiles agree with every primary tool CI.

## Exact next entry

Read this summary, docs/plan/OVERVIEW.md and the CM2.1 section of PHASE_1C_HSI.md.
The report contains one proposed student-training experiment: keep CM1's recipe
and add low-noise(19/39/59) full-future body22 FK retention against a frozen R2
remaining-DDIM endpoint, including root. Limit target construction to1–3 steps,
calibrate one coefficient, preserve120172544 windows and the complete acceptance.
The auxiliary eval teacher must stay independent of the original train-mode
teacher/EMA/dropout path. Inference16-step sampling and CG remain unchanged.

This proposal still requires approval; no new training was registered, implemented
or started. Loss coupling, training/generated-history transfer and limited external
CG corrections remain risks. R4's GT-body-loss negative is retained; the present
diagnosis establishes a retention deficit, not the proposed objective's efficacy.
