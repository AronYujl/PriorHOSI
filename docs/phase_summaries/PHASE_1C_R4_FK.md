# Phase 1C R4 FK: completed teacher acceptance

R4 completed with a failed scientific gate. Keep corrected R2+CG as the working
quality baseline. Phase1C remains open. No further training or distillation was started.

## Scope and implementation

The only added objective anchors root-relative body21 FK positions to GT direct
positions on all14 future frames. R2 position MSE, rotation L1, hand/foot FK3 and
fullbody seam0.5494500113254572 are preserved. The registered one-batch calibration
selected body weight0.24812712291771177 through the rotation-head25% limit.

The user stopped the original run after656 updates and requested a fresh random
training on GPU4–7. The fresh run completed146255 updates/299530240 windows,
effective2048=4×512×1, seed42, lr2e-4, warmup2000, cosine tail117004..146255,
EMA0.9999, bf16_tf32 and fp32 geometry. FinalLR=0. The final epoch222 export is
exactly equal to214 EMA tensors in the terminal state. The fixed epoch19 snapshot
was retained. Every recorded loss/gradient was finite; update79 had a gradient
norm91848576. Its causal role in the negative result has not been identified.

## Workloads and result

Seven GPU workloads completed: epoch19 teacher-forced352 test-valid/364 train
windows; fixed60-development-sequence/364-window DDPM500 U rollout and representation;
final U/G each375 sequences/2271 windows; one frozen Table3 readout; final body
representation. The user canceled both single-card teacher latency jobs before launch.

Final sampling uses DDPM500, CFGw1, posterior-coefficient CG for G, seed42, and
fixed_rate_endpoint_hold_v1 interpolation. R2 physical/representation controls are
corrected CM1.5 artifacts. Table3 uses frozen internal LINGO features/gallery.

| Metric | R2 U | R4 U | R2 G | R4 G |
|---|---:|---:|---:|---:|
| body21 direct-position/FK divergence cm |0.794252|0.934823|0.887031|1.035614|
| pen_ratio |0.030818|0.034900|0.021884|0.025702|
| FS |0.286728|0.310904|0.272221|0.330189|
| boundary jerk |108.04497|119.34762|142.68556|137.58610|
| exterior contact |314.68475|303.11704|365.63099|365.25473|
| FID |38.41515|47.33550|40.04968|58.21284|
| MM-Dist |9.23861|9.78619|8.90005|9.76487|
| R@3 |0.415179|0.361607|0.433036|0.397321|

The family10 primary comparisons have7 definite regressions,0 improvements and3
uncertain contrasts. Guided FID delta+18.16316 has family6 simultaneous interval
[5.64912,32.19248]. R@3 declines have intervals spanning0; the registered mean
preservation bounds nevertheless fail for both arms. All group metrics, including
U interaction endpoint improvements and G total-contact increases, are preserved.

Holdout355 low-pelvis walk counts are8(U)/14(G), exceeding the limit2. >5g counts
are2 episodes/3 frames(U),3/4(G), within limits8/38. G case045-new_loco:009708
has minimum pelvis height0.127153m. Full375 G has6 episodes/11 frames>5g.
Both arms have375 finite motions and maximum nonfinite_ratio0.

Epoch19 body divergence is61.0207cm on the exposed60-sequence development cohort,
versus GT0.01884cm. Its smaller budget and EMA maturity differ from the final model;
it was used for the registered diagnosis, not checkpoint selection or early stopping.
The full teacher-forced summary and strata-weighted development comparisons are sealed.

## Verification, operational failures and costs

Authority before GPU work:476 passed/3 skipped; targeted56 passed. No runtime code
changed during acceptance; no repeat suite was needed. All seven manifests completed
on clean runtime source4fa797d. Coverage and final EMA identity were verified.
FID sequence IDs and GT text/motion embeddings are exactly equal across R2/R4.
Primary and semantic scalar CIs agree with the paired_bootstrap outputs.
Nineteen paired reports, family10/family6 simultaneous CIs, safety cases, all28 joint
means and native Diversity/MultiModality readouts are complete. FID uses2000 paired
resamples; other scalar contrasts10000, seed42. R@3 retains224 frozen gallery query
occurrences/148 unique sequences. No cross-training-seed confidence is claimed.

The first statistics attempt failed importing a sibling module of the frozen Diversity
reader (ModuleNotFoundError: build_train_bundle). All seven GPU workloads and17 paired
reports had succeeded. The recovery added the reader directory to sys.path, reused
those17 reports, and completed the two remaining semantic pairs and summaries.
The original failure log and pipeline exit1 remain; statistics recovery exit0 and
final_completion.json are authoritative for the completed acceptance.

Evaluation60.994722GPU-h<80; formal training96.068889; user-aborted initial training
including pause0.494249; calibration/benchmark0.218889. Total R4=157.776749GPU-h.
Other inference processes on GPU0–3 are recorded in each preflight. No teacher
single-card latency measurement was made.

## Provenance and artifacts

Preregistration696e473; objective implementation90e4162; calibration fixationf1fea6d;
resource admission2510307; fresh-training source4307061; training audit/evaluation
activation1a824c1; user latency cancellation/final evaluation source4fa797d.
This summary and report are sealed by the R4 acceptance completion commit.
No Phase1C merge or release tag was made. Existing checkpoint/input identities are
reused through their manifests and evaluator provenance; no new identity wrapper.

- Report/compact: experiments/results/p1_hsi_r4_fk_acceptance_s42_20260911.{md,json}.
- Training audit: experiments/results/p1_hsi_r4_fk_training_s42_20260911.json.
- Evaluation status and recovery: results/r4_evaluation_setup_20260911/final_completion.json.
- Pair reports and all diagnostics: results/r4_evaluation_setup_20260911/paired_statistics/.
- Every run: results/experiments/p1-hsi-r4-*-s42-20260911/.
- The original training run, checkpoint and cost addendum remain in their own directories.

## Next entry

Read this summary, docs/plan/OVERVIEW.md and the R4 section of PHASE_1C_HSI.md.
The approved R4 experiment is closed with a negative result. Continue to use corrected
R2+CG as quality baseline. A new teacher experiment requires a concrete separate
approval. The evidence does not establish that every GT FK objective fails, that
changing its coefficient would recover quality, or that update79 caused the failure.
