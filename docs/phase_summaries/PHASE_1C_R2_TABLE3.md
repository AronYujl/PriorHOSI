# Phase 1C R2 native Table 3 readout

Completed 2026-09-06. The user authorized completing R2's native table and
combining locomotion, object reaching and interactive columns into one table.
External baseline methods were excluded from this task.

## Fixed scope

- Reused R2 final EMA's sealed unguided and posterior-coefficient guided motions:
  375 episodes per arm, split into 130 walk and 245 non-walk episodes.
- Reused ten geometry/reaching metrics and the frozen internal text-motion
  encoder. No training, checkpoint selection, motion generation or new evaluator.
- Reaching retains final-frame minimum planar distance over 28 joints and an
  inclusive 5-cm threshold. It is not hand-to-object 3D distance.
- FID/MM-Dist cover 245 episodes. The frozen R@3 gallery contains 224 query
  occurrences from 148 unique episodes. Its legacy CI resamples occurrences.
- GPU float64 FID uses an equivalent sample-space covariance-factor formula;
  its C-reference error was 0.000054, within the registered 0.0001 tolerance.
  R-Precision and MM-Dist reproduced the same frozen reference.

## Results

| Arm | FID | R-Precision@3 | MM-Dist |
|---|---:|---:|---:|
| R2 | 38.415148 | 0.415179 | 9.238607 |
| R2 + CG | 40.049678 | 0.433036 | 8.900054 |

All thirteen native metrics and intervals are in the compact result. These
semantic point estimates are worse than the old C placeholder operating point;
C used 16-step consistency whereas R2 uses 500-step diffusion. No additional
model-selection or promotion claim is made. No reportable workload failed.

## Verification and artifacts

- Component metric checks: 3 passed. Authority suite: 437 passed, 3 skipped.
- Registry validation and fully resolved Hydra config passed.
- Workload exit 0; GPU0 RTX3090; 1012.615 s / 0.281282 GPU-h;
  peak allocated 237,524,480 bytes. Runtime validation used this actual readout.
- `tools/paired_bootstrap.py` completed both 130/245 U-versus-CG comparisons:
  10,000 replicates, seed42. FID has 2,000 paired bootstrap replicates.
- Compact: `experiments/results/p1_hsi_r2_table3_s42_20260906.json`.
- Full embeddings/query records/intervals: `results/hsi_r2_table3_s42_20260906/`.
- Manifest: `results/experiments/p1-hsi-r2-table3-s42-20260906/manifest.json`.
- Preregistration: `0b40c12`; execution source: `167933c`.
- Paper: `/data/yujinlun/iclr2027/main.tex`, table `tab:hsi-native`, and
  `/data/yujinlun/iclr2027/docs/HSI_TABLE3_R2.md`. The table has one tabular,
  three column groups and two complete R2 rows. External rows remain unavailable.
  PDF compilation succeeded. Existing unrelated manuscript edits were retained.

## Next entry

Read this summary, `docs/plan/OVERVIEW.md` and the latest Phase 1C plan section.
Use the completed table for reporting. The Phase 1C gate remains open; this
readout does not authorize new expert training or the next phase.
