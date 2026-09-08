# Phase 1C-CM1.2: fixed distillation evaluation

Completed on 2026-09-09. All registered workloads completed successfully. The
student failed the quality and guided speed gates and was not promoted. R2 final
EMA with posterior-coefficient guidance remains the working baseline. Phase 1C
remains open; no merge, release tag, next-phase work or new training was started.

## Scope, implementation and source

Branch `phase/01c-cm1`. Preregistration and training closure: `de28b3c`.
Evaluation implementation and execution source: `407ff7b`.

The previously authorized fixed-w=1 student completed 58,678 updates and
120,172,544 windows on 8×RTX3090. All 469,424 per-rank diagnostic records were
finite with required gradients; 11 updates used the existing clipping rule.
The final epoch089 student was the only formal quality checkpoint. Epoch004
(3,280 updates) was used only for the fixed internal diagnostic.

One evaluation config extends the existing Table3 config. The evaluator now
records pelvis acceleration and minimum height from its existing native FK
joints. Table3 labels derive sampler type, steps, guidance and seed from each
artifact. The HSI readout component computes paired ratios and the registered
gates. Sampling, guidance arithmetic, model weights and the encoder/gallery
protocol were unchanged during this subphase. Core was unchanged.

## Fixed coverage and results

Internal epoch004: U/G each 60 episodes and 364 windows, with the frozen B_n60
population stratum weights. Final epoch089: U/G each 375 episodes and 2,271
windows. All 870 generated episodes and 5,270 windows completed with finite
motions and 16 denoiser calls per window. R2 U/CG quality motions were reused.
Native groups are 130 walk and 245 interactive/reaching episodes.

| Guided comparison | R2+CG | CM1+CG |
|---|---:|---:|
| FID, internal encoder | 40.049678 | 22.903796 |
| R-Precision@3 | 0.433036 | 0.450893 |
| MM-Dist | 8.900054 | 8.254822 |
| Penetration ratio, full375 | 0.0214163 | 0.0274793 |
| Foot sliding, full375 | 0.276482 | 0.298000 |
| Boundary jerk, full375 | 159.083175 | 165.532907 |
| Interior jerk, full375 | 72.415245 | 65.973148 |
| Goal planar error, m | 0.0583062 | 0.0509261 |
| Contact count | 893.089049 | 902.918353 |
| Exterior contact count | 367.850760 | 320.895333 |

Guided penetration ratio increased 28.31%, with a paired ratio CI
[1.21947, 1.35266]. Exterior contact decreased 12.76%, CI [0.84281, 0.90261].
Total contact includes penetrating vertices and therefore does not establish
preserved nonpenetrating contact. Foot sliding increased 7.78%; the walk-only
increase was 10.10%, CI [1.05330, 1.15347]. Interactive maximum floor-excluded
penetration, interior jerk and semantic metrics improved. All directions are
retained in the complete report rather than summarized as uniform degradation.

Of 25 relative quality bounds, 12 passed, 6 failed and 7 were inconclusive.
Safety bounds passed on the frozen holdout355: guided output had one episode
and two frames above 5g, plus one low-pelvis walk. Full375 had four episodes and
11 frames above 5g, plus one low-pelvis walk. Those full-cohort failures remain
in the artifacts. Unguided output had no >5g frames and one low-pelvis walk.

## Matched single-GPU timing

The fixed latency70 selector chose 19 episodes and 69 windows, with five warmup
episodes and 14 timed episodes. All four methods used identical IDs and window
counts. CUDA timing was synchronized. GPU process snapshots before and after
each timing workload were empty; inspected intervals used only GPU0.

| Method | Mean generation FPS | Mean generation seconds/episode | Mean end-to-end seconds/episode |
|---|---:|---:|---:|
| R2 U, 500 steps | 2.689740 | 55.883695 | 55.978002 |
| R2+CG, 500 steps | 0.762189 | 197.204652 | 197.333330 |
| CM1 U, 16 steps | 132.063265 | 1.137630 | 1.208738 |
| CM1+CG, 16 steps | 19.258679 | 7.764710 | 7.848559 |

Speedups from total generation time are 49.1229× unguided and 25.3976× guided.
Guided FPS also failed the registered ≥20 target. No planning stage applies.
The 8-GPU quality-shard timings remain invalid for latency claims.

## Validation, failures and resource use

Component checks: 72 passed. Authority suite: 450 passed, 3 skipped. Registry
validation and fully resolved configs passed. All GPU workloads exited zero.
Geometry and ordinary differences used the existing paired-bootstrap tool;
ratios used 10,000 GPU draws, FID 2,000 paired draws, all seed42.

The first internal statistics command supplied unfiltered 375-episode teacher
payloads against the fixed 60 students. The tool rejected the unequal sets.
Teacher records were explicitly projected onto the preregistered 60 IDs and
the r1 statistics completed. The failed log is retained; motions were generated
once. No failed or negative model result was removed.

Total reserved cost, including formal training, the independent benchmark and
all evaluation workloads, was 66.5238 GPU-h, below the 160 GPU-h cap. The training
run ended on 2026-09-08 at 23:47:21 CST, after 28,103.2 seconds. It was finalized
at its original source commit before returning to the evaluation branch.

## Artifacts and next entry

- Report: `experiments/results/p1_hsi_cm1_evaluation_s42_20260909.md`.
- Compact: `experiments/results/p1_hsi_cm1_evaluation_s42_20260909.json`.
- Full bounds and timing: `results/hsi_cm1_gate_s42_20260909/summary.json`.
- Student Table3 and embeddings: `results/hsi_cm1_table3_s42_20260909/`.
- Paired differences, internal stratification and failed statistics log:
  `results/cm1_evaluation_setup_20260909/`.
- All manifests/configs/logs: `results/experiments/p1-hsi-cm1-*-s42-20260909/`.
- Student checkpoints: `results/p1-hsi-cm1-r2-fixed-w-s42-20260908/checkpoints/`.

Read this summary, `docs/plan/OVERVIEW.md`, and the 2026-09-08/09 CM1 sections in
`docs/plan/PHASE_1C_HSI.md` before further HSI work. Retain R2+CG as the quality
baseline and CM1 as an unpromoted speed/semantic tradeoff. Any successor needs
a separate concrete proposal; this result authorizes no automatic retry.

The comparison combines checkpoint and sampler changes (500-step diffusion
versus 16-step consistency). It does not isolate their causes or demonstrate
distillation of external geometry guidance into an unguided model. Uncertainty
uses episodes or the frozen R@3 query occurrences, not independent training
seeds; R@3 has 224 occurrences from 148 unique sequences. Feature metrics belong
to the frozen internal LINGO encoder and are not external-paper equivalents.
