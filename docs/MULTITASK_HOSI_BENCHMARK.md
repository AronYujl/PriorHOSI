# Multi-task HOSI dataset benchmark

Current contract: Phase 5.6.1, clarified by the user on 2026-09-11. An episode
combines **dataset OMOMO + dataset LINGO**, in either order, with a short Kimodo
inbetween construction witness. Model inference later consumes initial state,
text, goals, geometry and timing from a separate conditions file.

Earlier 5.4–5.5 construction and actual-history diagnostics remain in their
phase summaries. The 24 generated chains in
[Phase 5H](phase_summaries/PHASE_5H_EXPANDED_HISTORY.md) measure that inference
path. Their support failures supply no membership criterion for this dataset.

## Construction

1. Read complete OMOMO test sequences with original body and object tracks.
   Place them rigidly in the associated original HOSI scenes. Check every frame,
   including the middle of the action. Derive extension goals and durations
   from the actual retained motion. Original HOSI's `start_idx + 45` reference
   describes a short window and remains separate.
2. Read unmirrored LINGO from the fixed seed-42 scene-family test split. Join
   consecutive safe annotations at contiguous raw frames in the same scene.
   Preserve each action's original text and interval. Walking, standing and
   seated furniture actions are eligible. Unknown labels, temporal gaps and
   moving props split spans.
3. Search all frames for ten-frame standing contexts. Retain 49–600 frames with
   at least 0.5 m of locomotion. Cuts may be internal to a longer recording.
   The actual retained interval determines the episode's action content.
4. Both join contexts require torso tilt ≤25°, pelvis height ≥0.7 m, knee
   flexion ≤45°, root speed ≤0.5 m/s and one foot marker within 8 cm of the
   floor at every frame. Hand contact is allowed.
5. Retarget LINGO to the OMOMO subject and preserve source rotations and height
   changes under constant grounding and yaw/XZ placement. Check all LINGO
   frames against scene, persistent object, floor and bounds. Static interaction
   requires retained seated frames with bilateral support in the target scene.
6. Keep the OMOMO object at its join transform throughout LINGO, with floor or
   upward-facing scene support. Touching at the join is acceptable. A stationary
   suspended object lacks a support witness.
7. Attempt one fixed seed-42 Kimodo bridge per pair. Preserve both ten-frame
   source contexts, including moving OMOMO object contexts. Insert only
   `bridge[10:51]`: `first_source + 41_free_frames + second_source`. Each
   source context appears once in the composed motion.

The recipe uses 50 Kimodo steps with empty text, 240 native fitting steps and
400 contact correction steps. Failed attempts stay in the ledger; the next
pair is considered. Failure describes this construction procedure's outcome and
does not prove intrinsic incompatibility. Coverage order is fixed before
interpolation outputs are available.

Explicit geometry tolerances: scene mean/max penetration ≤1/10 mm, floor
≤10 mm and native body/object penetration ≤50 mm. Report these tolerances with
the measurements. Geometric acceptance establishes no dynamics guarantee.

## Inputs and artifacts

| File | Consumer and content |
|---|---|
| `inference_tasks.json` | Model input: initial body/object context, body shape, geometry, ordered text/goals and timing |
| `original_hosi_tasks.json` | All 469 original task definitions, retained unchanged |
| `construction_manifest.json` | Source identities/intervals, placements, measurements and successful/reserve/rejected IDs |
| `source_composition.npz` | Native 30-Hz source-plus-inbetween feasibility witness, outside Git |
| `construction_result.json` | Per-attempt geometry, support, context preservation and frame accounting |
| `review.md` | Coverage table and representative scene videos |

The inference table contains no future source motion, source-motion file path or
Kimodo trajectory. Each episode includes `initial_context`, `body_identity`,
`object`, `scene_sdf`, `scene_sdf_info`, `segments`, `transition`,
`frame_count` and `duration_s`. Segments supply human goals and, for OMOMO,
object position/orientation goals. LINGO seated actions have support targets.
Goals and durations describe the selected interval, including its internal cut.

A model carries achieved state continuously across segments. LINGO's object
policy is `stationary_at_achieved_boundary_transform`; scene and task goals
remain fixed. Construction motion is a feasibility reference. Full-motion
imitation scores against it are outside this benchmark's current contract.

## Measures

`mixer.source_bridge.dataset_motion_metrics` accepts a native prediction and
an inference task and reads no source motion. It computes complete scene/body/
object/floor geometry, scene bounds, per-segment human/object goal errors, HOI
hand-contact coverage, LINGO stationary-object displacement, foot support,
near-floor foot speed with contact coverage, and frame budget.

Construction additionally records position/rotation/velocity seams, correction
magnitudes, exact source/context preservation and seated support witnesses.
Every transition frame participates in full-motion checks. Report coverage for
both directions, static-containing episodes and pure locomotion separately.
Constructed counts are distinct from future model inference success rates.

## Reproducible entry

Use `code/test_infbagel_hosi.py --config-name config_sample_hosi_dataset_benchmark`
with `multitask.stage=dataset_sources`, `dataset_bridges` or `dataset_publish`.
Bridge lanes are disjoint by candidate ordinal. Publication accepts up to 128
complete witnesses from at most 256 source candidates in fixed coverage order.

Use verified `INFBAGEL_PYTHON`, set `ROOT_DIR` to the inference checkout,
archive resolved configs and use the existing experiment manifest lifecycle.
Output roots are unique. Data/model provenance reuse existing asset references.
The fixed search recipe is in [Phase 5.6.1](plan/PHASE_5_INFERENCE.md).
