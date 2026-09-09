# Phase 1C R4 FK: implementation and resource handoff

Status: objective and fixed calibration verified; formal training queued for GPU resources.
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
