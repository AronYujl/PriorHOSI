# Phase 1C R4 FK: implementation and resource handoff

Status: objective and fixed calibration verified; the user superseded the original queue with fresh training on GPU4–7 (amendment below).
R4.1 remains open until formal initial stability and a resumable checkpoint are demonstrated.
R4.2 physical/FID/semantic acceptance is approved and pending training artifacts.

The new objective anchors root-relative body21 FK positions to the same window's GT direct
positions on all 14 future frames. R2 position MSE, rotation L1, hand/foot FK weight3,
fullbody seam0.5494500113254572, architecture and output parameterization stay fixed.
A first-real-batch zero-update calibration selects body weight0.24812712291771177
by the registered rotation-head25% rule; trunk10% alone would give67.834511.

One from-random teacher uses logical4×512×accum1/effective2048 to preserve R2 rank/data
semantics, seed42, lr2e-4, warmup2000, cosine tail117004..146255,
299530240 processed windows, EMA0.9999, bf16_tf32 and fp32 geometry.
Each epoch writes the rolling resume state; epoch19 also saves a fixed diagnostic EMA.
The final EMA is the only candidate for final acceptance.

Authority476 passed/3 skipped; targeted56 passed. Calibration exited0 with four finite
rank readouts and no update. Full microbatch benchmark completed128 updates; all four
rank gradient sequences were identical and finite. CUDA synchronized measured96 updates
took58.9455s:0.614015s/update,3335.42windows/s, roughly24.95training hours under measured
contention, plus initialization/checkpoint IO. Peak PyTorch reserved memory12.8223GiB.

Resource admission failed: observed GPU0 total used22774MiB/free1802MiB, below required
2457.6MiB free. This sample is sufficient to refuse immediate formal launch; it is not a
continuous maximum measurement. Other users' processes remain running. The persistent queue
waits until physicalGPUs0/1/6/7 each have <=6144MiB occupied, then archives fresh hardware
preflight and creates the clean-source manifest before training. Source changes stop the
queue with a retained record. The training recipe and budget remain fixed.

Artifacts:
- Calibration table and values: experiments/results/p1_hsi_r4_fk_calibration_s42_20260909.{md,json}.
- Resources: experiments/results/p1_hsi_r4_fk_resources_s42_20260909.json.
- Job and queue status: results/r4_fk_setup_20260909/.
- Calibration/benchmark manifests: results/experiments/p1-hsi-r4-fk-{calibration,benchmark}-s42-20260909/.
- Formal run: results/experiments/p1-hsi-r4-fk-train-s42-20260909/; manifest created only at actual launch.
- Training outputs: results/p1-hsi-r4-fk-train-s42-20260909/.

Measured preparation cost0.218889GPU-h. No formal training update or new teacher quality
result is claimed at handoff. No Phase1C merge, tag or distillation has been performed.

Next session: read this file, docs/plan/OVERVIEW.md and the R4 section in
PHASE_1C_HSI.md. First inspect queue_status.json, launcher exit, training logs and
rolling checkpoint; retain any failure and never reuse its run id. After stability,
report actual throughput/ETA and let the persistent process run. At completion, execute
fixed epoch19 development diagnostics, final DDPM500 U/G full375, native Table3 and body
representation readouts with registered paired uncertainty and safety/engagement gates.
Compare against the corrected CM1.5 R2+CG and GT artifacts; all efficacy conclusions await
these measurements. Existing sealed input identities are referenced by the manifests.

## User-directed restart on GPU4–7

The original queue launched at15:51:14UTC and reached epoch0/update656. Its full rolling
checkpoint was saved, then the original job was stopped. The user explicitly cancelled
layout benchmarking and chose fresh random training rather than continuation. The old
run is sealed aborted; its checkpoint and logs remain available, including a cost addendum
for the brief checkpoint-boundary pause. This is an operational restart, not a quality failure.

Active replacement id: p1-hsi-r4-fk-train-fresh47-s42-20260909. Physical GPUs4,5,6,7;
load_state_dict=false, resume_from empty, start_epoch0, full146255 updates. Scientific
recipe, logical4×512 layout, weight and acceptance rules stay fixed. The new job, manifest,
logs and final/intermediate checkpoints use the replacement id. Its physical-layout speed
will be recorded from initial formal training; no A/B throughput comparison was run.

Inspect results/r4_fk_setup_20260909/fresh47_job.json and fresh47_status.json first.
The old queue_status.json describes the aborted job and is historical.

## Training completed; fixed evaluation entry (2026-09-11)

Fresh47 completed146255 updates/299530240 windows with exit0, finalEMA epoch222 and
fixed diagnosticEMA epoch19 preserved. Runtime cost96.068889GPU-h. The final export
exactly matches214 EMA tensors and finalLR=0. All recorded gradients/losses are finite;
the isolated update79 gradient norm91848576 is included in the training audit.
Source4307061 remains the runtime implementation. Initial stability and checkpoint
verification are in the run's initial_stability.json. R4.1 launch/training work is complete;
R4.2's physical/FID/semantic acceptance remains open.

Continue from results/r4_evaluation_setup_20260911/jobs.json and references.json.
Nine fixed workloads and58 fully resolved configs cover epoch19 teacher-forced/development
rollout/body readouts, finalU/G single-card latency/full375 quality, frozenTable3 and final
representation. Read the dated R4.2 section for exact pairings, uncertainty, input reuse
and cost. Training audit: experiments/results/p1_hsi_r4_fk_training_s42_20260911.json.
No additional training or checkpoint selection is authorized by this handoff.

## Final acceptance handoff

R4 acceptance completed with a failed gate. The authoritative final summary is
docs/phase_summaries/PHASE_1C_R4_FK.md and the report is
experiments/results/p1_hsi_r4_fk_acceptance_s42_20260911.md. Seven GPU workloads and
19 paired reports are complete; the initial statistics import failure was recovered.
The user canceled teacher single-card latency. Use final_completion.json in the
evaluation setup directory to distinguish recovered completion from the retained
first-attempt pipeline exit1. Preserve corrected R2+CG as baseline.
