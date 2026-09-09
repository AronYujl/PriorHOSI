# Multi-task HOSI input format

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

Run the existing Hydra entry with the verified native environment and
`config_sample_hosi_multitask`. A reportable construction uses
`tools/experiment.py start`, a clean implementation checkout, the exact resolved
config and machine preflight. Reusing a run directory is unsupported; every run
has a fresh experiment ID. Source identity is reused from existing manifests.

Phase 5.4 constructs and checks the inputs. Phase 5.5 executes frozen episodes
with the current experts and Kimodo, preserving actual generated histories and
reporting complete-chain metrics. The first input release does not claim a
model success rate.
