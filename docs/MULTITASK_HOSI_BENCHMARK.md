# Multi-task HOSI input format

The first input pilot is published in
[`p5_multitask_pilot_s42_20260909.json`](../experiments/tasks/p5_multitask_pilot_s42_20260909.json).
It retains all 469 original tasks and adds one three-segment episode with two
concrete transition-context references. Read the
[Phase 5.4 summary](phase_summaries/PHASE_5D_MULTITASK_BENCHMARK.md) for coverage,
the retained failed construction and the input-only scope.

The next execution entry is Phase 5.5.1, an actual handoff audit through
`config_sample_hosi_handoff_audit`. It measures the published-budget prefix and
the original cached terminal separately. Each native 16-frame stride-3 window
adds 42 new 30-Hz samples after two historical coarse samples. A budget of 9.8
seconds permits 298 observed samples including the initial 0.1-second history.
Legacy exports also contain two held padding samples; these do not establish
terminal rest. A cached motion from a longer native plan remains an original-
protocol diagnostic, including when inspecting its shorter prefix.

The handoff audit records source identity, actual body/object support and speed,
goals, and prescribed-context geometry against the achieved object transform.
Its failed prerequisites block subsequent segments with explicit reasons.
Unexecuted transitions and actions have null metrics; complete-chain success
is measured after a compatible actual-history rollout is available.

## Source-only transition membership

The extension candidate table is independent of every model output. For
`omomo_to_lingo`, the OMOMO task-reference terminal must be upright, supported,
slow and hand-released before a LINGO action is considered. For
`lingo_to_omomo`, the OMOMO initial context must keep both hands separated from
the object throughout its entry frames before LINGO can precede it.

The LINGO entry or terminal context is placed in the target HOSI scene using the
episode body identity. The complete native LINGO source interval is then checked
against target scene SDF, persistent-object SDF and bounds. A source scene is
provenance only; its coordinates never define target-scene membership. All
failures retain the direction, corpus-qualified source IDs, frame interval and
measured geometry reason. Runtime generated states are checked again, but cannot
change the source-only candidate set.

The strict released-hand reverse table remains separate from the approved
`lingo_to_omomo_grasped_entry` extension. The latter accepts an OMOMO initial
context with source hand contact at its first frame and in at least half of its
ten frames. It carries a supported initial object transform and a
`kimodo_acquire_contact` bridge contract. The bridge approaches a prescribed
static first-frame grasp. Kimodo must recover contact at its suffix while
preserving the object transform. Runtime results never change membership.

Whole source clips retain their native height profile under yaw/XZ placement and
one surface-based ground translation. Feet and seated support are checked in
addition to clearance. The three historical 5.5.2a candidates were incorrectly
raised by 0.426-0.430 m during pelvis alignment and require this correction.

Phase 5.4 extends task conditions while preserving all 469 original HOSI tasks.
The first construction recipe targets HOI -> walk -> sit in the original
67 TRUMANS scenes. LINGO supplies locomotion and static-interaction source
contexts from the fixed seed-42 test partition. A recomposed episode has task
conditions and geometry references; it has no complete frame-aligned motion GT.

## Records and source indices

`original_hosi_tasks.json` wraps each original row with its file, row number,
canonical `hosi-NNN` ID and corpus-qualified source ID. `original_task` preserves
every original field and value. This table remains the original benchmark.

`sources.json` resolves `source_dataset` plus `data_idx` to the language table,
source sequence, half-open raw frame interval, 30 Hz source rate, first/last
ten-frame contexts, body identity and semantic endpoint. The same numeric
`data_idx` in OMOMO and LINGO denotes different data. `source_terminal_frame`
belongs to the complete source sequence; the first 48-frame model window does
not define it. Source boundary measurements are input diagnostics.

For HOSI, `task_reference_frame` separately identifies the declared endpoint
sample: `language.start_idx[data_idx] + 3 * test_frames[-1]`. All 469 current rows
use sample 15, or raw offset 45. Their supplied object-goal height and planar
human-object separation match this frame. A terminal feasibility witness uses
this frame and a yaw reconstructed from the supplied human/object-goal relation.
The full source sequence end remains available as separate provenance and audit.

The first LINGO action allowlist is `walk`, `sit down on chair`, `sit down on
office chair`, `sit down on sofa`, `sit down on couch`, and `stand up from seat`.
The two hand-interaction frame flags alone cannot establish absence of held
props. Sources outside the action allowlist or too short for the native model
window retain explicit exclusion records.

## Episodes and segments

An episode fixes one target scene and one human body identity. It contains
ordered `segments`, explicit `transitions` and persistent movable objects.
The HOI subject supplies body identity throughout the first recipe. Transferring
LINGO poses to that body is part of constructing geometric boundary witnesses.
The standing and seated contexts receive separate constant vertical placements
onto the floor. Their local rotations and within-context velocities are retained;
both placement offsets are recorded. The intermediate action is generated later.

Each segment supplies these inference conditions:

| Field | Meaning |
|---|---|
| `source_dataset`, `data_idx`, `source_id` | Corpus-qualified source context and action semantics |
| `scene_name` | Target scene in which the segment executes |
| `source_scene` | Original source scene, kept separately from the target |
| `start_location`, `pelvis_goal` | World Y-up ground-plane human locations, metres |
| `object_goal` | Movable-object centre goal when applicable |
| `scene_goal`, `contact_target` | Static interaction goal and geometric support identity |
| `text`, `duration_s` | Action and nominal duration budget |
| `initialization` | Source context for the first segment; achieved history for successors |
| `entry_requirements`, `exit_requirements` | Required object support, release, posture or goal state |

The HOI segment also embeds the original task unchanged. Its extension adds a
supported, slow terminal-object requirement and a specified terminal orientation
from the source witness. These are additional task-chain conditions; they do not
retroactively change the original 469-task score.

The walk goal is the static interaction's standing entry location. The sitting
goal includes a bilateral buttock support reference checked against the target
scene. The movable object remains an obstacle throughout walking and sitting.
Straight-path construction is a first-version reachability approximation. Its
failure says that this recipe did not construct a path, not that no path exists.

## Transition and runtime state

Each edge names its action, allowed duration, incoming/outgoing context duration,
contact requirements and object persistence policy. The first recipe uses
`release_and_walk` and `align_for_sitting`. A suspended object requires a separate
placement action and is excluded from a release-only chain.

Benchmark edges describe requirements independently of a generator. The current
inference choice is Kimodo with native body/contact correction. An edge reserves
1.4 seconds between 0.3-second contexts on either side: at 30 Hz this is the
existing 61-frame bridge with 10 prefix frames, 41 free frames and 10 suffix
frames. Endpoint and transition geometry still need runtime checks.

`mixer.multitask.successor_condition` carries the actual generated history and
actual achieved movable-object transforms into a successor. It updates the
successor's start location while preserving source provenance. Constructed
source poses and goal transforms are planning conditions; they must not reset
the achieved scene state. A failed support/release/geometry requirement remains
a recorded episode failure.

All transition frames participate in continuous-chain metrics. Report ordered
completion, episode success and longest completed prefix alongside goals,
scene/object penetration, hand release, seating contact, foot sliding and
boundary position/velocity changes. A video-only bridge supports a separate
segment-level demonstration. It cannot establish continuous-chain success.

## Construction output and review

The builder writes a source catalogue, all original rows, source exclusions,
one construction audit row for each original task, geometry-accepted candidate
episodes, native endpoint witnesses and scene previews. Witnesses contain
source-derived conditions; they are not model-generated motions and do not
become full-motion evaluation GT.

Candidate membership uses source and target geometry only. A candidate support
surface receives semantic seat review from its scene preview before publication.
`geometry_accepted_pending_semantic_review` is therefore distinct from a
published task. Semantic rejection and unattempted candidate-cap exclusions
remain visible in the construction report.

For the published pilot, scene-mesh review identifies a low step. The task text
is therefore `sit down on the low step`; `source_text` preserves the original
LINGO office-chair label. Inference must encode the task's text rather than
silently reusing the source label's embedding. Each edge's
`target_context_reference` resolves to its native motion field and frame range
relative to the manifest's `artifact_root`. The first bridge targets the verified
HOI terminal reference as the walk entry; the second targets the placed static
entry. Actual object geometry is checked again at execution time.

Run the existing Hydra entry with the verified native environment and
`config_sample_hosi_multitask`. A reportable construction uses
`tools/experiment.py start`, a clean implementation checkout, the exact resolved
config and machine preflight. Reusing a run directory is unsupported; every run
has a fresh experiment ID. Source identity is reused from existing manifests.

Phase 5.4 constructs and checks the inputs. Phase 5.5 executes frozen episodes
with the current experts and Kimodo, preserving actual generated histories and
reporting complete-chain metrics. The first input release does not claim a
model success rate.
