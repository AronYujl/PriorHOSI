# Motion-export schema (V0 design)

This is a consumer contract, not a change to either expert trainer. It is
provisional until the HSI exporter and PriorHOSI mixer semantics are settled.
The first stable file must carry `schema_version=1`; V0 documents the fields
that are required to make that version possible.

## Design principle

The export is the boundary between model-dependent inference and
model-independent rendering. A renderer must be able to reconstruct the
motion without importing a sampler, loading a checkpoint, or knowing how the
prediction was denoised.

The preferred container is one non-pickle `.npz` per sequence/episode. Strings
and small metadata are stored as scalar NumPy strings or a sidecar JSON; arrays
must load with `allow_pickle=False`. Legacy HOI pickles are read by a one-way
adapter and are never rewritten in place.

## Required fields

| Field | Shape/type | Meaning |
|---|---|---|
| `schema_version` | scalar int | Export contract version |
| `sequence_id` | scalar string | Collision-free artifact identity |
| `task_family` | scalar string | `hoi`, `hsi`, or `hosi` |
| `fps` | scalar float | Motion sampling rate represented by pose arrays |
| `coordinate_frame` | scalar string | Explicit world convention, e.g. `lingo_y_up` |
| `global_orient` | `[F,3]` float32 | SMPL-X global axis-angle |
| `body_pose` | `[F,21,3]` float32 | SMPL-X body axis-angle |
| `transl` | `[F,3]` float32 | SMPL-X translation in the declared frame |
| `betas` | `[B]` float32 | Shape parameters; B is recorded, normally 16 or 10 |
| `gender` | scalar string | SMPL-X gender/model variant |
| `global_jpos` | `[T,28,3]` float32 | Optional but recommended coarse dataset joints |

`F` is the pose/FK rate. `T` is the coarse rollout rate. If both are present,
the file must include `interp_scale` and the relation `F=T*interp_scale` (or an
explicit per-window relation). A renderer must not silently interpolate or
truncate one stream to match the other.

## Object and scene fields

HOI and HOSI files additionally require:

| Field | Shape/type | Meaning |
|---|---|---|
| `object_name` | scalar string | Rest-mesh asset key |
| `object_trans` | `[T,3]` or `[F,3]` | Object translation at declared rate |
| `object_rot_mat` | `[T,3,3]` or `[F,3,3]` | Object rotation matrices |

Scene and condition metadata should be scalar fields or a sidecar record:

```text
scene_name
scene_asset_id
caption
start_location
pelvis_goal
object_goal
```

The artifact records the rate and frame for every object stream. A renderer
may resample object poses for display only after recording the chosen rule in
its render manifest.

## Long-horizon extensions

The mixer is not yet specified. These fields are therefore optional and must
not be required for HOI/HSI readers:

```text
window_lengths    [W]
seams             [W-1]
history_frames    scalar int
window_id         [T]
stage_id          [T]
state             [T] or stage table
guard_events      structured sidecar JSON
expert_source     [T] or stage table
route_weights     [T,G]
```

The distinction is intentional: `window_id` describes model stitching,
`stage_id/state` describes the task/state machine, and `expert_source` or
`route_weights` describes mixer routing. None should be inferred from frame
numbers after export.

## Provenance sidecar

Every export directory must contain a small manifest (JSON is acceptable) with
at least:

```text
export_schema_version
source_git_commit
source_live_head_at_completion
resolved_config_sha256
checkpoint_path_and_sha256
dataset_snapshot_and_sha256
smpl_models_sha256
object_asset_manifest_sha256
scene_asset_manifest_sha256
command
working_directory
created_at
```

The manifest records provenance; the NPZ stores motion. The two are linked by
`sequence_id` and a file SHA256. A new export uses a new run/artifact
directory. Existing NPZ files are never overwritten.

## Validation rules

An adapter/reader must reject, before rendering:

- missing pose, translation, gender, beta, or coordinate-frame fields;
- non-finite values or inconsistent frame counts;
- unknown schema versions unless an explicit compatibility adapter is chosen;
- object pose without a resolvable rest-mesh asset;
- ambiguous axis/order conventions;
- pickle-only inputs when the caller requested a no-pickle render path.

Validation is read-only. It must not “repair” a source export in place; repairs
produce a new artifact with a new schema/adapter record.

## Legacy compatibility

The current HOI `motion_params/*.pkl` contains the core human and object
arrays needed for conversion. The adapter should map its keys into this schema,
record the legacy source path/hash, and preserve the original sequence name.
The planned HSI export is expected to use the same SMPL-X parameter family.
PriorHOSI may add stage/routing metadata later without changing the core human
or object fields.

