# Phase 1C R2 paired qualitative selection

Completed 2026-09-06. User authorized five action classes, four training sources per
class, and paired generated/ground-truth rendering for direct gap inspection.

## Scope and input

Twenty original non-mirror LINGO v3 train sequences, 17 scenes, 97 windows. R2 final
EMA and posterior-coefficient guidance, 500 diffusion steps, seed42 with the
existing canonical-ordinal episode rule. Source heading restores the exact
observed two-frame history; subsequent history is generated. This differs from
Table3's endpoint-facing initialization and is reported separately. No training
or repeated motion generation. Core and the other expert were unchanged.

Implementation extends the existing episode builder with explicit source IDs,
the HSI evaluator with source-heading input and paired cache/media modes, the HSI
visualization component, and the existing Blender human renderer. Both arms use
identical shape, sex, scene, light, camera and time. Flat-hand SMPL-X settings
match. World-space motion is retained. GT shares the evaluator resampling; its
terminal hold is marked in the video. The main caption for rise is standing up,
not a full departure trajectory.

## Selection and observed errors

| Case | Source | Recorded-span mean joint difference, cm |
|---|---|---:|
| navigate_01 | 037:003046 | 14.66 |
| sit_04 | 037:002985 | 23.74 |
| rise_01 | 067:006770 | 13.69 |
| lie_02 | 042:003930 | 15.20 |
| wash_04 | 047:004674 | 15.70 |

The mean joint difference includes timing and valid alternative motions; it is
not the sole quality criterion. Sit04 has clearer support visibility than Sit01.
Rise01 stands later and remains forward-leaning; Lie02 changes arm/leg placement;
Wash04 changes engagement timing. Lie03/Lie04 remain upright at the end while GT
is lying down. All20 candidates and failed actions are kept.

## Execution and failures

Sampling source: 27046c6; corrected rendering source: b8e885d. Preregistration:
9136591. Sampling and GT completed successfully once, pipeline wall918s (2.04
GPU-h reserved upper bound). Four manifests record generation, failed initial
presentation, failed encoding after complete rendering, and successful encoding.
First preview clipped highlights and hid legs behind a table; original images
were retained. Filmic/lower lighting/shared motion-context camera corrected it.
Some original scanned-room occlusions remain and are disclosed.

The newest downloaded FFmpeg required NVENC API13.1/driver610; the server's570
driver supports13.0. Encoding failed before any packet. The empty output is
retained as comparison.failed-api13_1.mp4. A 2025-08-31 FFmpeg7.1 build encoded
all20 existing image sequences successfully; the final config points to it.
No driver or Python-environment modification was needed.

## Validation and deliverables

- Component evaluator/visualization:30 passed. Authority:440 passed,3 skipped.
  The first authority invocation omitted exporting INFBAGEL_PYTHON; its two
  fixture errors were resolved by exporting it and rerunning the affected module.
- Rendering correction:2 component checks passed. Source motion pairing:all20
  same shape/frame count and initial joint difference below0.001cm.
- FFprobe decoded all20 H264 videos,2097 frames per arm,15fps,1280×572.
- All local HTML media/report links exist; five selected and twenty total cards.
- Paired bootstrap:20 sequences,10000 replicates,seed42, all shared numerical
  metrics; this selected cohort is not a population/generalization experiment.

Review root:
/data/yujinlun/iclr2027/artifacts/hsi/paired_train_s42_20260906/review/
Open selected.html for five recommendations or index.html for all20.
SELECTED_FIVE.md contains Chinese rationale/gaps; selected_five_overview.jpg
contains the paired montage. The parent directory's
HSIPrior_GT_comparison_20cases.zip (147.4MB) contains portable web/video/still
results and the selected five original generated/GT parameter files.
Full source exports and reconstruction caches remain on the authority host.

Compact: experiments/results/p1_hsi_r2_qualitative_s42_20260906.json.
Manifests: results/experiments/p1-hsi-r2-qualitative*-s42-20260906/manifest.json.

## Next entry

Read this summary, docs/plan/OVERVIEW.md and the latest Phase1C plan section.
User reviews the five paired examples before further selection or final paper
artwork. This task does not promote a model, close Phase1C's quality gate, or
start a mixer phase.
