# Phase1C CM2.3: completed student acceptance

Completed on 2026-09-12. CM2 failed the joint gate; retain corrected R2+CG as the
quality baseline. All registered workloads and readouts are complete. Phase1C
remains open; no new training, coefficient adjustment, mixer work, merge or tag.

## Scope and implementation

The approved CM2 student added one low-noise body22 endpoint objective to CM1.
At timesteps19/39/59, an independent frozen eval R2 supplied a1–3-step DDIM target
for all14 future frames, including root. The original CM1 objectives, train-mode
teacher/EMA target, dropout, EMA0.95 and16-step inference were preserved.
One gradient calibration fixed lambda1.2518757249916617.

Training completed58678 updates/120172544 windows on8×RTX3090,8×256×1,
effective2048, seed42. All469424 gradient/cfg records and46512 endpoint-loss records
are finite;40 clipping events remain. The final epoch089 student exactly matches
all218 tensors in the terminal resume model. Epoch004 was only the fixed internal
diagnostic. The EMA target was not substituted for the final student.

Acceptance used corrected interpolation, fixed CFGw1 and the original CM16 CG
mapping: internal U/G each60 episodes/364 windows; final U/G each375/2271;
frozen native Table3; position/FK for final U/G and GT; student U/G latency70;
the registered ratio/safety/speed gate. R2 and CM1 reference views reuse feature
embeddings while taking physical groups from corrected native outputs.

## Results and scientific limits

| Guided metric | R2 | CM1 | CM2 |
|---|---:|---:|---:|
| Penetration ratio |0.0218843|0.0281318|0.0293217|
| FS |0.272221|0.288150|0.291540|
| Boundary jerk |142.68556|145.74579|153.99359|
| Exterior contact |365.63099|319.96180|318.06604|
| Direct-position/FK body disagreement cm |0.887031|1.656058|1.692449|
| FID |40.04968|22.90380|17.45352|
| MM-Dist |8.90005|8.25482|7.53117|
| R@3 |0.433036|0.450893|0.473214|

Against CM1, guided penetration and boundary jerk regress with paired95% intervals
excluding0; FS and exterior-contact changes are uncertain. Body disagreement grows
in both U/G. Guided FID and MM-Dist improve; R@3 improvement is uncertain. Unguided
FID17.44221 improves at the point estimate against CM1, with an interval crossing0.
All U/G, group, internal, Diversity/MultiModality and uncertainty results remain.

Guided holdout355 has9 >5g episodes/18 frames, exceeding the8-episode limit;
two low-pelvis walks meet the limit2. Full375 has10 >5g episodes/27 frames and two
low walks. Unguided has none. All case identities and maxima are retained.

The25 guided relative quality bounds give13 PASS,5 FAIL and7 INCONCLUSIVE.
The failed bounds concern full-cohort penetration/scene penetration/exterior
contact, locomotion scene penetration, and interactive exterior contact.
Safety fails independently. Feature scores read direct positions; physics uses
the FK body, so feature gains do not establish physical retention.

Position/FK self-disagreement is also distinct from same-state teacher endpoint
error. CM2.1's common-state probe was not rerun after training; this acceptance
does not identify one optimization/dropout/target mechanism as the sole cause,
nor establish that every body-FK objective fails.

## Student timing

Both modes used physical GPU1, batch1, the fixed19 episodes/69 windows,5 warmup
episodes and14 timed episodes. IDs match the historical teacher timing exactly.
CUDA regions were synchronized, with16 denoiser calls per window.

| Mode | Mean generation FPS | Mean generation s/episode | Mean end-to-end s/episode |
|---|---:|---:|---:|
| U |94.518030|1.591602|1.655440|
| G |19.039375|7.851533|7.916283|

Both before/after snapshots show another inference process using4214MiB on GPU1.
These are measured runtimes under that competition; exclusive-GPU throughput and
a controlled speedup over the old idle-teacher timing are unestablished. The
observed guided FPS is below20. Raw historical speed ratios remain in the gate
artifact, with this interpretation limit. Physical/safety failure already prevents
promotion regardless of timing conditions.

## Verification, failures, resources and source

Nine registered GPU workloads succeeded; the initial position-FK invocation failed
before reconstruction because it omitted the existing component's GT cohort.
All8 exits1 and its manifest remain. A new r1 id supplied GT and correct per-shard
GPU visibility; all8 shards and merge exited0. GT's375 metric rows exactly match
the sealed control. Completed motion generation was never repeated.

There are37 paired reports,4 cached paired FID contrasts and the full ratio gate.
Scalar resampling uses10000 draws/seed42; FID uses2000. Frozen evaluator continuity
and GT embedding equality pass. R@3 keeps224 occurrences from148 unique sequences;
no cross-training-seed interval is claimed.49 initial configs and9 recovery configs
were archived; the three resumed job configs match actual Hydra resolution.
Runtime code was unchanged, so authority487 passed/3 skipped and targeted66 are
reused. Final registry validation is retained in the setup directory.

Initial resource stop160.017709GPU-h is preserved. The user approved161 to finish
the three remaining jobs, then explicitly approved168 as the final ceiling.
Total160.084340GPU-h<168: training156.488370, preparation0.735828,
evaluation2.860143 including the failed invocation0.005768. Remaining jobs added
0.066631GPU-h. No scientific budget, dataset, metric or checkpoint rule changed.

Training source e6185a1; main quality/representation source deef2d8; timing/gate
source072fa5b. Runtime is identical across the evaluation sources. This summary
and the complete report are sealed by the final completion commit; prior partial
reports and resource stops remain traceable.

## Artifacts and exact next entry

- experiments/results/p1_hsi_cm2_acceptance_s42_20260912.{md,json}: final report/compact.
- results/hsi_cm2_gate_s42_20260912/summary.json: raw complete gate.
- results/cm2_evaluation_setup_20260912/final_completion.json: completion index.
- Same setup:37 comparisons, reference views, failure recovery, both resource
  approvals, jobs and all completion records.
- results/hsi_cm2_table3_s42_20260912/ and hsi_cm2_position_fk_r1_s42_20260912/.
- Every run's manifest/config/preflight/log is below results/experiments/.

Read this summary, OVERVIEW.md and the CM2.3 section of PHASE_1C_HSI.md before
further work. The authorized experiment is complete with a negative joint gate.
Keep corrected R2+CG and review the measured negative before proposing a separate
experiment. This completion authorizes no automatic retry or next-phase work.
