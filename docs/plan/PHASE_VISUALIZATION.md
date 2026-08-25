# Visualization Worktree Phase

Status: V0 fixture-validator gate passed; no model workload allocated
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

### V1 — Read-only adapters

Add adapters that read existing HOI motion parameters and the eventual HSI
export into the common schema. The adapter may normalize legacy pickle files,
but must never overwrite the source. Gate: round-trip validation preserves
frame counts, pose arrays, object transforms, and provenance.

### V2 — Headless 2D/3D rendering

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
Exit for V0: schema, headless-rendering, isolation documents, and the
synthetic validator are committed; expert branches remain byte-for-byte
untouched; no run ID or GPU workload is created. V1 remains pending until the
HSI export is stable and a read-only HOI legacy adapter is approved.
