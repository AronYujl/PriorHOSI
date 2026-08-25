# Visualization worktree

This directory documents a model-independent consumer pipeline for motion
artifacts produced by HOIPrior, HSIPrior, and PriorHOSI.

The worktree boundary is deliberate:

```text
expert branch / future integration snapshot
        -> export adapter
        -> versioned motion NPZ
        -> Linux headless renderer or Windows Blender renderer
```

The renderer must never reach back into a trainer or alter an expert checkout.
It may read a checkpoint only during the one-time export step; a renderer that
consumes an NPZ does not need model weights.

Documents:

- [`../plan/PHASE_VISUALIZATION.md`](../plan/PHASE_VISUALIZATION.md): phase,
  gates, scope, and isolation contract;
- [`MOTION_EXPORT_SCHEMA.md`](MOTION_EXPORT_SCHEMA.md): provisional NPZ
  contract and provenance rules;
- [`HEADLESS_RENDERING.md`](HEADLESS_RENDERING.md): Linux/Windows rendering
  paths and paper-figure composition;
- [`WORKER2_INFERENCE.md`](WORKER2_INFERENCE.md): the one-checkpoint worker2
  dispatch, persistent inference, verification, and automatic return workflow.

## V0 validator

The schema validator is deliberately model-independent:

```bash
INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
"$INFBAGEL_PYTHON" -m pytest -q tests/visualization
"$INFBAGEL_PYTHON" -m tools.visualization.schema path/to/motion.npz \\
  --manifest path/to/manifest.json
```

It always reads NPZ files with `allow_pickle=False`, rejects ambiguous or
non-finite arrays, and never edits a source artifact. The V0 test suite uses
synthetic fixtures only; it does not load expert checkpoints or real data.

## HOI adapter and headless rendering

The existing HOI exporter writes a trusted legacy pickle with
`human_motion.pose_pred`, `human_motion.root_trans`, and object arrays. The
read-only adapter writes a canonical NPZ. The leading legacy dimension is
ambiguous, so callers must state whether it contains independent candidates or
consecutive autoregressive windows. It must not be used on untrusted pickle
files.

For older multi-candidate files, select one trajectory explicitly:

```bash
INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
"$INFBAGEL_PYTHON" -m tools.visualization.hoi_legacy \
  /path/to/*_motion_params.pkl \
  --output /path/to/artifacts/sequence-sample0.npz \
  --manifest /path/to/artifacts/sequence-sample0.manifest.json \
  --legacy-layout samples --sample-index 0 \
  --fps 30 --coordinate-frame infbagel_y_up \
  --legacy-human-frame z_up
```

The P12 evaluator instead stores one long rollout as flattened human arrays
`[W*F,22,3]` and object arrays `[W,F,3]` / `[W,F,9]`. Preserve all windows:

```bash
"$INFBAGEL_PYTHON" -m tools.visualization.hoi_legacy \
  /path/to/p12/*_motion_params.pkl \
  --output /path/to/artifacts/sequence-motion.npz \
  --manifest /path/to/artifacts/sequence-motion.manifest.json \
  --legacy-layout autoregressive_windows \
  --fps 30 --coordinate-frame infbagel_y_up \
  --legacy-human-frame y_up
```

This layout records `window_lengths`, `window_id`, and seam indices. It is
explicit rather than shape-guessed because candidate and window layouts can
have identical array shapes. Legacy provenance unavailable from an old run is
labelled `legacy-unrecorded`; such output is a visual artifact, not scientific
result evidence.

The pre-P12 released exporter stored human pose/translation after a
`yup_to_zup` conversion, while object translations stayed in the y-up world.
The adapter therefore explicitly reverses that human-side conversion when
`--legacy-human-frame z_up` is used. Exports produced by the post-P12 corrected
code should use `--legacy-human-frame y_up` instead.

The resulting NPZ can be rendered on the Linux headless server without
Blender:

```bash
"$INFBAGEL_PYTHON" -m tools.visualization.headless \
  /path/to/artifacts/sequence-sample0.npz \
  --manifest /path/to/artifacts/sequence-sample0.manifest.json \
  --output /path/to/artifacts/sequence-sample0.png \
  --smpl-models /path/to/smpl_models \
  --object-mesh /path/to/rest_object_geo/object.ply \
  --object-rest-frame z_up --object-geometry full --max-faces 0 \
  --keyframe-count 6 --device cpu --style paper \
  --hand-pose-fallback mean
```

This renderer reconstructs SMPL-X only from exported parameters, overlays
fixed keyframes in one camera, and writes a render manifest. `paper` style uses
an orthographic, axis-free, content-cropped layout. `debug` style retains axes
and may use the object convex hull for a quick smoke.

P12 does not export articulated finger pose. `--hand-pose-fallback mean` uses
the static natural SMPL-X mean hand instead of an unnatural all-zero flat hand
and records `hand_pose_source=smplx_mean`. This improves appearance but is not
claimed as predicted hand motion. A future export may provide paired
`left_hand_pose` and `right_hand_pose` arrays with shape `[F,45]`.

The generated checkpoints, motion files, meshes, images, and videos remain
outside Git under a run-specific artifact directory.

To render a synchronized fixed-camera MP4 from the same canonical NPZ, use the
CPU/headless video module documented in `HEADLESS_RENDERING.md`. It renders one
full SMPL-X/object mesh pair per source frame and encodes H.264 with FFmpeg; it
does not rerun inference and does not require Blender. Use a fresh artifact
directory for every render identity.

For paper-facing material, floor, shadow, and anti-aliased quality, use the
separate OMOMO-style Blender/Cycles tier in the same document. It consumes the
same canonical NPZ through an immutable mesh cache and produces both a video
and a six-frame 3x2 process figure. The PyTorch3D command remains the fast
debug/preview path.
