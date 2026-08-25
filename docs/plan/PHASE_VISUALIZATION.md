# Visualization Worktree Phase

Status: HOI V1 adapter and headless CPU smoke passed; HSI/mixer work remains pending
Date: 2026-08-25 (Asia/Shanghai)  
Branch: `visualization/renderer`  
Worktree: `/data/yujinlun/InfBaGel-visualization`

## Purpose

This phase defines an isolated consumer-side visualization pipeline for
HOIPrior, HSIPrior, and the future PriorHOSI mixer. It must not change the
training, evaluation, configuration, registry, or frozen-core files owned by
`phase/01b-hoi` or `phase/01c-hsi`.

The pipeline has two boundaries:

```text
expert/mixer checkpoint + matching inference code
        -> motion export (the only model-dependent step)
        -> versioned motion artifact
        -> renderer / keyframe compositor (model-independent)
```

The exported artifact is the hand-off. Once it exists, changing a checkpoint,
sampler, denoising schedule, or mixer implementation must not be required to
re-render a figure or video.

## Isolation contract

The visualization worktree is based on `research/state-compositional-priors`
and lives on `visualization/renderer`. It is intentionally not a worktree of
either expert branch and does not merge or cherry-pick their in-progress
commits.

The following paths are out of scope for this branch:

- `code/priors/core/`;
- `code/priors/hoi/` and `code/priors/hsi/` model/training changes;
- expert trainer/evaluator behavior and expert Hydra configs;
- expert `docs/plan/` or phase summaries;
- expert `experiments/registry.jsonl` history.

Expert results are external immutable inputs. When an adapter eventually needs
expert code, it must consume a pinned integration snapshot or an exported
artifact; it must not edit an expert checkout while that branch is training.

## Scope of this design phase

1. Define a provisional, versioned NPZ motion-export contract.
2. Specify adapters for the existing HOI export, the planned HSI export, and
   a future mixer export whose stage metadata is not yet fixed.
3. Define a headless Linux renderer that needs no Blender GUI.
4. Define an optional Windows/Blender consumer that reads the same artifact.
5. Define keyframe selection and multi-pose compositing for paper figures.
6. Establish provenance and non-overwrite rules for exported artifacts.

No checkpoint is loaded and no GPU workload is started in this design phase.
No generated motion, checkpoint, scene mesh, video, or per-sample output is
tracked in Git.

## Planned subphases and gates

### V0 — Contract and boundary (passed 2026-08-25)

Deliver the schema document, artifact naming rules, coordinate-frame rules,
and worktree boundary. The gate passed with ten tests covering valid HSI and
HOI payloads, object-stream requirements, frame/rate consistency, finite
values, schema rejection, and provenance hash mismatch. The validator imports
only NumPy and the Python standard library; it does not import either expert
package.

### V1a — HOI read-only adapter (passed 2026-08-25)

The adapter reads the existing HOI `motion_params/*.pkl` without importing the
HOI package, selects one sample from the legacy candidate dimension, and
writes a new canonical NPZ plus an explicit legacy provenance manifest. The
gate passed with synthetic multi-sample fixtures and a real 42-frame HOI
pickle; the output validated with the V0 schema and preserved pose/object
frame counts. The legacy source and generated artifacts remain outside Git.

### V1b — HSI read-only adapter (pending)

Add an adapter for the eventual HSI export into the common schema. The adapter
may normalize a legacy file, but must never overwrite the source. Gate:
round-trip validation preserves frame counts, pose arrays, and provenance once
the HSI exporter is stable.

### V2a — Headless HOI mesh smoke (passed 2026-08-25)

The CPU/Agg renderer produced a deterministic PNG from the real converted HOI
artifact on Linux without Blender or a display server. It selected frames
`[0,14,27,41]`, reconstructed SMPL-X on CPU, applied the object rest mesh, and
wrote a render manifest with source/asset/output hashes. The full furnished
scene and paper-quality camera remain a later gate. The smoke artifact root is
`/data/yujinlun/InfBaGel-visualization-artifacts/hoi-smoke-20260825` (outside
Git); the NPZ SHA256 is
`2f76c86c21d0e27eee908594249007d04e22888b9a00b478e359c28233a3dff9`, and the
PNG SHA256 is
`d20f74ba0f45c00057b79c9a2aa9b66aa21f741c16bf924d9ce62708c37f8298`.

### V2b — Headless 2D/3D rendering gate (pending)

Implement a CPU/headless renderer for trajectory figures and optional mesh
frames. Gate: it produces deterministic PNG output from a fixture on Linux
without Blender or a display server.

### V3 — Paper composition

Implement fixed-keyframe selection, alpha compositing, labels, and montage
layout. Gate: the same input artifact and render config reproduce the same
figure hash; method comparison uses the same keyframes and camera.

### V4 — Mixer/long-horizon extension

Add optional `stage_id`, `state`, `window_id`, and guard-event metadata once
PriorHOSI output semantics are fixed. Gate: an episode can be rendered as a
single trajectory while retaining stage boundaries and seam locations.

## Current inputs and assumptions

- The HOI branch currently has a legacy SMPL-X/object motion-parameter export.
- The HSI branch is expected to export the same human-parameter family, but its
  final export is not yet treated as a stable contract here.
- PriorHOSI/mixer output format is intentionally unspecified at V0; the schema
  therefore reserves optional stage and routing metadata rather than freezing
  a mixer API prematurely.
- SMPL-X model files, object rest meshes, and scene assets are code-independent
  assets, but each render manifest must record their hashes.
- `data/hosi_test` SDF/occupancy files are evaluation geometry, not by
  themselves a replacement for the furniture/room meshes seen in paper figures.

## Non-goals

- changing expert training or model selection;
- deciding the mixer architecture or state-machine protocol;
- claiming that an export artifact is a scientific evaluation result;
- reproducing the authors' unpublished figure-layout script;
- requiring Blender on the Linux authority host.

## Entry and exit points

Entry: this committed design on `visualization/renderer`.  
Exit for the current HOI subphase: the schema, HOI adapter, headless renderer,
isolation documents, synthetic tests, and real HOI smoke artifact are
committed/recorded on this worktree; expert branches remain untouched; no run
ID or GPU workload was created. HSI adapter and full scene renderer remain
pending until their inputs/assets are stable.
