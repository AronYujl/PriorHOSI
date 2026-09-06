# Phase 2.12 native scene-evidence benchmark — 2026-09-07

**Benchmark complete; protections and total scene-quality gate pass, HSI-evidence and joint gates fail.** The fixed independently calibrated `lambda_dp=26` was evaluated on all 469 native tasks against lambda0, reconstruction, and the sealed native-HOI row.

## Scope and execution

The preregistered rows were `full` (lambda26), `lambda0`, and `reconstruct`; the 469-task native-HOI baseline was reused from `p2-hosi-hoi-alone-g0-p15-guided-armb-s42-20260829`. There are 1,407 new rollouts, 67 scenes, and 2,086 windows per row. All 24 shard jobs produced complete outputs; the parent campaign retained its initial post-processing failure (`st_size()` called as a function) with exit 1. The independent read-only recovery corrected that analysis typo without rerunning GPU sampling and exited 0.

The run preserved the registered seed-42 paired bootstrap (10,000 replicates), scene and episode units, Bonferroni-adjusted 98.75% primary intervals, non-inferiority protections, trajectory audits, and external paper reference. No parameter was selected on the 469-task results.

## Primary results

Means (native metric units):

| row | HS s_mean | OS s_mean | contact | completed | foot sliding |
|---|---:|---:|---:|---:|---:|
| full lambda26 | 6.3743 | 29.7664 | 0.6972 | 0.7569 | 0.1219 |
| lambda0 | 6.3764 | 29.6822 | 0.6980 | 0.7569 | 0.1208 |
| reconstruct | 7.0534 | 32.3988 | 0.6979 | 0.7569 | 0.1181 |
| native HOI | 6.9867 | 32.1154 | 0.6915 | 0.7633 | 0.1651 |

For full minus lambda0, HS delta is -0.00210 with adjusted 98.75% CI [-0.07264, 0.05822], and OS delta is +0.08413 with CI [-0.04650, 0.32127]. Both intervals include zero, so the HSI-evidence gate fails. For full minus native HOI, HS delta is -0.61235 [−1.32947, −0.12726] and OS delta is −2.34904 [−3.92452, −0.93589]; both satisfy the registered total scene-quality criterion.

Protections pass at both episode and scene units. Against lambda0, contact lower bound is −0.00164 (episode) and −0.00157 (scene), completion lower bound is 0, and FS-ratio upper bounds are 1.0347 and 1.0346. Against native HOI, contact lower bounds are positive, completion lower bounds are −0.01493, and FS-ratio upper bounds are 0.8021 and 0.8000. The joint promotion gate therefore fails because the HSI-evidence family did not pass.

## Diagnostics and interpretation

The full row completed all 469 episodes with finite tensors, exact saved-foot-sliding reconstruction, and maximum world-history error below 1e-6. It used 33,376 HOI and 33,376 HSI teacher calls, accepted 14,770 updates, and had 1.33 GiB peak allocation. Full versus lambda0 changes are tiny and mixed: lambda26 slightly lowers HS, slightly raises OS, lowers contact by 0.00079, and raises foot sliding by 0.00100 (mean FS ratio 1.0083). Against native HOI it improves both scene penetration means but loses 3 completed episodes; the completion protection still passes its −0.02 margin.

Descriptive task/scene counts, object strata, completion discordance, and ten largest absolute HS/OS contributors are retained in `analysis/descriptive-contrasts.json`. Reconstruction and paper comparisons remain descriptive. The InfBaGel paper Hybrid 1:0.5 values (success 81.45%, FS 0.15, contact 76.96%, HS 3.17, OS 12.45) are unpaired, pre-repair external context and are not a superiority claim.

## Verification and artifacts

The compact result is `experiments/results/p2_mixer_evidence_benchmark_s42_20260907.json`. The sampling manifest and logs are under `results/experiments/p2-mixer-evidence-benchmark-s42-20260907/`; the corrected read-only analysis is under `results/experiments/p2-mixer-evidence-analysis-s42-20260907/`. The preregistered configuration, calibrated lambda, sealed native baseline, all shard exits, trajectory audits and bootstrap outputs are preserved. The phase closes with quality unresolved for the isolated HSI contribution; no tuning, new mechanism, or subsequent phase is started.
