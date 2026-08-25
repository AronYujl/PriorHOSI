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
  paths and paper-figure composition.

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

The generated checkpoints, motion files, meshes, images, and videos remain
outside Git under a run-specific artifact directory.
