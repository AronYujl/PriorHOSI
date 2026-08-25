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

## HOI V1 adapter and headless smoke

The existing HOI exporter writes a trusted legacy pickle with
`human_motion.pose_pred`, `human_motion.root_trans`, and object arrays. The
read-only adapter selects one candidate trajectory and writes the canonical
NPZ. It must not be used on untrusted pickle files:

```bash
INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
"$INFBAGEL_PYTHON" -m tools.visualization.hoi_legacy \
  /path/to/*_motion_params.pkl \
  --output /path/to/artifacts/sequence-sample0.npz \
  --manifest /path/to/artifacts/sequence-sample0.manifest.json \
  --sample-index 0 --fps 30 --coordinate-frame infbagel_y_up \
  --legacy-human-frame z_up
```

The current legacy files may contain three candidates: human arrays are
flattened as `[S*F,22,3]`, while object arrays are `[S,F,3]` and `[S,F,9]`.
`--sample-index` prevents those candidates from being concatenated into one
false long trajectory. Legacy provenance unavailable from the old run is
labelled `legacy-unrecorded` in the generated manifest; this is a visual smoke
artifact, not a reportable experiment result.

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
  --object-rest-frame z_up --object-geometry convex_hull \
  --keyframe-count 4 --device cpu
```

This renderer reconstructs SMPL-X only from exported parameters, overlays
fixed keyframes in one camera, and writes a render manifest. It uses the object
convex hull by default for a stable lightweight proxy; a furnished room or
Blender-quality asset is a later rendering tier.

The generated checkpoints, motion files, meshes, images, and videos remain
outside Git under a run-specific artifact directory.
