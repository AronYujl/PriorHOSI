"""Scene-only LINGO rollout evaluation and ground-truth reference metrics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as Rotation

import pytorch3d.transforms as transforms

import utils as project_utils
from priors.hsi import metrics as hsi_metrics
from priors.hsi import diagnostics as hsi_diagnostics
from priors.hsi.scene_field import SceneGeometry, default_cache_dir
from utils import SMPLX_JOINTS_28, create_smplx_model, interp_jrot, interpolate_joints, run_smplx_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "dataset"
WINDOW_FRAMES = 16
HISTORY_FRAMES = 2
DATA_STEP = 3
WINDOW_STRIDE_RAW = (WINDOW_FRAMES - HISTORY_FRAMES) * DATA_STEP
NON_WATERTIGHT_SCENES = frozenset(("031", "049-bed"))

#: Metric-payload schema version, written into every ``per_sequence_metrics.json``.
#:
#: 1  -- the 2026-08-12 metric set as first sealed.
#: 2  -- the 2026-08-17 penetration key set (``pene_pct_scene``, ``pen_value``,
#:       ``pene_samples``, ``pene_sum_{mean,max}_floorexcl`` added,
#:       ``pen_frame_ratio`` removed) **plus** the 2026-08-18 faithful
#:       ``fs_nemf`` definition (L2 horizontal magnitude, mean over the four foot
#:       joints, ``T - 1`` denominator, no pre-translation, height clamped to
#:       ``max(h, 0)`` inside the weight).
#:
#: The bump exists for the FS change specifically.  The penetration change is
#: detectable from a payload's key set alone; the FS change alters the meaning of
#: three keys that already existed and adds none, so without this field a
#: schema-2 payload would be indistinguishable from a sealed schema-1 one while
#: carrying ``fs_nemf`` values roughly 2-5x larger.  The factor is data-dependent
#: (2.19x on GT, ~5.0x on a rollout whose feet never sink below y = 0), so a
#: reader must never rescale across the boundary -- compare only equal
#: ``schema_version``, and supersede an old number by recomputing from motion.
#: 3  -- the 2026-08-18 generated-arm SMPL-X frame correction.  No key is added
#:       or removed; the *value* of every FK-derived metric on the generated arm
#:       moves, because before this bump the generated arm's FK translation was
#:       mapped with ``yup_to_zup`` and its FK output mapped back with
#:       ``zup_to_yup`` while the SMPL-X rest template stayed put.  That left the
#:       whole body rotated 90 degrees about +x relative to its own pelvis and the
#:       pelvis itself displaced by ``zup_to_yup(J_rest0) - J_rest0``
#:       (``(0, +0.3795, +0.3542)`` m for the reference betas).  Every metric that
#:       reads vertices or SMPL-X joints -- penetration, foot skate, jerk,
#:       boundary jerk, goal decomposition, contact -- is therefore incomparable
#:       across this boundary on the generated arm.  Ground-truth-arm payloads are
#:       bit-identical to schema 2: that arm always used the identity path.
#: 4  -- the 2026-08-18 representation-frame correction.  No key is added or
#:       removed and the ground-truth arm is again bit-identical to schema 3,
#:       because it never reads the representation.  Every *model* row, however,
#:       is incomparable across this boundary for a stronger reason than a metric
#:       redefinition: the training representation itself changed.
#:       ``datasets/infbagel.py`` no longer conjugates ``human_orient``/
#:       ``human_pose`` with ``zup_to_yup``, so the 132-dim rotation channel a
#:       checkpoint was fitted to no longer exists.  A checkpoint trained before
#:       this bump cannot be evaluated after it -- not "differently", but on an
#:       input distribution rotated 90 deg about +x -- so schema-3 model numbers
#:       are superseded only by a retrained model, never by a recomputation.
METRICS_SCHEMA_VERSION = 4

#: Motion-export schema version, written into every per-sequence ``.npz``.
#:
#: 1  -- the 2026-08-18 export: ``global_jpos`` at the rollout rate plus the
#:       SMPL-X parameters at the FK-input rate.  On the **generated** arm this
#:       schema records ``smplx_output_transform == "zup_to_yup"`` and a
#:       ``transl`` that was mapped with ``yup_to_zup`` before FK.  To use a
#:       schema-1 generated-arm file, leave ``global_orient``/``body_pose``
#:       untouched -- they are already in the frame SMPL-X's template lives in --
#:       replace ``transl`` with ``zup_to_yup(transl)``, and apply **no**
#:       transform to the FK output despite what the file's own
#:       ``smplx_output_transform`` field says.  That recipe reproduces the
#:       exported ``global_jpos`` pelvis to 1.2e-07 m on all 375 files; obeying
#:       the recorded field instead misses it by 0.3795 m.  Ground-truth-arm
#:       schema-1 files are already correct and need no repair.
#: 2  -- the 2026-08-18 frame correction: both arms now use the identity path, so
#:       ``smplx_output_transform`` is always ``"identity"`` and ``transl`` is the
#:       literal FK input again.  A schema-2 rebuild needs no repair step.
#: 3  -- the 2026-08-18 representation-frame correction.  The field set and the
#:       ``"identity"`` transform are unchanged; the bump exists because
#:       ``smplx_pose`` on the generated arm is now the decoded rotation channel
#:       verbatim instead of ``yup_to_zup`` of it.  Ground-truth-arm files are
#:       bit-identical to schema 2.
MOTION_EXPORT_SCHEMA_VERSION = 3

#: The frozen text-motion evaluator drops sequences shorter than this and
#: truncates the rest to a multiple of ``T2M_LENGTH_MULTIPLE``.  Both numbers are
#: properties of that evaluator, not of this runner; they live here only so the
#: export can *count* what the evaluator will discard instead of letting a silent
#: truncation read as full coverage.
T2M_MIN_FRAMES = 16
T2M_LENGTH_MULTIPLE = 4

# The legacy utility stores this as a cwd-relative constant.  This entry point is
# valid from either the repository root or code/, so resolve it once here.
project_utils.SMPL_DIR = str(REPO_ROOT / "smpl_models")


def _json_value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError("cannot serialize %s" % type(value).__name__)


def _sanitize_json(value):
    """Represent undefined metric values as JSON null for paired-bootstrap input."""
    if isinstance(value, Mapping):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _load_episodes(episode_dir: Path, limit: Optional[int] = None):
    episodes = []
    for path in sorted(episode_dir.glob("*.json")):
        scene_name = path.stem
        with path.open("r", encoding="utf-8") as handle:
            scene_episodes = json.load(handle)
        for scene_index, episode in enumerate(scene_episodes):
            if episode["scene_name"] != scene_name:
                raise ValueError(
                    "%s episode %d names scene %r" % (path, scene_index, episode["scene_name"])
                )
            if episode["object_name"] is not None:
                raise ValueError("scene-only episode has object_name=%r" % episode["object_name"])
            episodes.append((scene_name, scene_index, episode))
            if limit is not None and len(episodes) >= limit:
                return episodes
    if not episodes:
        raise ValueError("no episodes found under %s" % episode_dir)
    return episodes


def _load_episode_subset(path_value, episodes, window_counts):
    if path_value is None:
        return {
            "enabled": False,
            "canonical_ordinals": list(range(len(episodes))),
            "episode_count": len(episodes),
            "window_count": int(sum(window_counts)),
        }
    path = Path(str(path_value)).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    rows = payload.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("episode subset %s has no non-empty episodes list" % path)
    ordinals = [int(row["canonical_ordinal"]) for row in rows]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("episode subset contains duplicate canonical ordinals")
    for row, ordinal in zip(rows, ordinals):
        if not 0 <= ordinal < len(episodes):
            raise ValueError("episode subset ordinal %d is out of range" % ordinal)
        scene_name, _scene_index, episode = episodes[ordinal]
        expected_id = "%s:%06d" % (scene_name, int(episode["source_sequence_idx"]))
        if str(row["sequence_id"]) != expected_id:
            raise ValueError(
                "episode subset ordinal %d names %s, evaluator enumerates %s"
                % (ordinal, row["sequence_id"], expected_id)
            )
        if int(row["window_count"]) != int(window_counts[ordinal]):
            raise ValueError("episode subset window count mismatch for %s" % expected_id)
    window_count = int(sum(window_counts[ordinal] for ordinal in ordinals))
    declared_windows = payload.get("total_windows")
    if declared_windows is not None and int(declared_windows) != window_count:
        raise ValueError(
            "episode subset declares %d windows, enumerated %d"
            % (int(declared_windows), window_count)
        )
    return {
        "enabled": True,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "design": payload.get("design"),
        "canonical_ordinals": ordinals,
        "sequence_ids": [str(row["sequence_id"]) for row in rows],
        "episode_count": len(ordinals),
        "window_count": window_count,
    }


class GroundTruthSource:
    """Memory-mapped LINGO arrays needed to reproduce the sampling body."""

    def __init__(self, root: Path, keep_text: bool = False):
        self.root = Path(root)
        self.joints = np.load(self.root / "human_joints_aligned.npy", mmap_mode="r")
        self.orient = np.load(self.root / "human_orient.npy", mmap_mode="r")
        self.pose = np.load(self.root / "human_pose.npy", mmap_mode="r")
        self.transl = np.load(self.root / "transl_aligned.npy", mmap_mode="r")
        self.betas = np.load(self.root / "betas.npy", mmap_mode="r")
        self.sequence_start = np.load(self.root / "start_idx.npy").astype(np.int64)
        self.sequence_end = np.load(self.root / "end_idx.npy").astype(np.int64)
        with (self.root / "gender.pkl").open("rb") as handle:
            self.gender = pickle.load(handle)
        language_path = (
            self.root
            / "language_motion_dict"
            / "language_motion_dict__inter_and_loco__16.pkl"
        )
        with language_path.open("rb") as handle:
            language = pickle.load(handle)
        self.window_sequence = np.asarray(language["ori_sequence_idx"], dtype=np.int64)
        self.window_start = np.asarray(language["start_idx"], dtype=np.int64)
        # datasets.infbagel reads self.text from this same pickle, so
        # window_text[data_idx][0] is the identical object the sampling path
        # reaches as cond["raw_text"].  Retained only on request: the list holds
        # 2.28 M entries and the real arm has no other use for it.
        self.window_text = language["text"] if keep_text else None

    def caption(self, data_idx: int) -> str:
        if self.window_text is None:
            raise RuntimeError("GroundTruthSource was built without keep_text=True")
        return str(self.window_text[int(data_idx)][0])

    def episode_indices(self, data_idx: int, episode_num: int) -> Tuple[int, np.ndarray]:
        sequence_index = int(self.window_sequence[data_idx])
        start = int(self.window_start[data_idx])
        expected_start = int(self.sequence_start[sequence_index])
        if start != expected_start:
            raise ValueError(
                "episode data_idx %d starts at %d, source sequence starts at %d"
                % (data_idx, start, expected_start)
            )
        end = int(self.sequence_end[sequence_index])
        expected_windows = math.ceil((end - start - HISTORY_FRAMES * DATA_STEP) / WINDOW_STRIDE_RAW)
        if int(episode_num) != expected_windows:
            raise ValueError(
                "episode data_idx %d declares %d windows, expected %d"
                % (data_idx, episode_num, expected_windows)
            )
        offsets = np.arange(WINDOW_FRAMES, dtype=np.int64) * DATA_STEP
        indices = []
        for window_index in range(expected_windows):
            raw = start + window_index * WINDOW_STRIDE_RAW + offsets
            indices.append(np.minimum(raw, end - 1))
        return sequence_index, np.stack(indices)


def _stitch_indexed(array: np.ndarray, indices: np.ndarray) -> hsi_metrics.StitchedSequence:
    windows = [torch.from_numpy(np.asarray(array[index]).copy()) for index in indices]
    return hsi_metrics.stitch_windows(
        windows, history_frames=HISTORY_FRAMES, overlap_atol=0.0
    )


def _upsampled_stitched(
    coarse: hsi_metrics.StitchedSequence, frames: torch.Tensor, scale: int
) -> hsi_metrics.StitchedSequence:
    return hsi_metrics.StitchedSequence(
        frames=frames,
        seams=tuple(int(seam * scale) for seam in coarse.seams),
        window_lengths=tuple(int(length * scale) for length in coarse.window_lengths),
        history_frames=int(coarse.history_frames * scale),
    )


def _interpolate_local_pose(local_axis_angle: torch.Tensor, scale: int) -> torch.Tensor:
    local_matrices = transforms.axis_angle_to_matrix(local_axis_angle.reshape(-1, 22, 3))
    local_quaternions = transforms.matrix_to_quaternion(local_matrices)
    local_quaternions = interp_jrot(local_quaternions, scale).reshape(-1, 22, 4)
    return transforms.matrix_to_axis_angle(
        transforms.quaternion_to_matrix(local_quaternions)
    ).reshape(-1, 22, 3)


def _run_smplx_chunks(
    local_pose: torch.Tensor,
    translation: torch.Tensor,
    betas: torch.Tensor,
    gender: str,
    device: torch.device,
    batch_size: int,
    cache: MutableMapping[str, torch.nn.Module],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if gender not in cache:
        cache[gender] = create_smplx_model(gender, device, batch_size=1)
    vertices, joints = [], []
    for begin in range(0, int(local_pose.shape[0]), batch_size):
        end = min(begin + batch_size, int(local_pose.shape[0]))
        chunk_betas = betas[None].repeat(end - begin, 1)
        chunk_vertices, chunk_joints = run_smplx_model(
            local_pose[begin:end],
            translation[begin:end],
            chunk_betas,
            gender,
            joints_ind=SMPLX_JOINTS_28,
            smpl_model=cache[gender],
        )
        vertices.append(chunk_vertices.detach())
        joints.append(chunk_joints.detach())
    return torch.cat(vertices), torch.cat(joints)


def ground_truth_motion(
    source: GroundTruthSource,
    episode: Mapping,
    device: torch.device,
    interp_scale: int,
    smplx_batch_size: int,
    smplx_cache: MutableMapping[str, torch.nn.Module],
    export_sink: Optional[MutableMapping[str, Any]] = None,
):
    sequence_index, indices = source.episode_indices(
        int(episode["data_idx"]), int(episode["episode_num"])
    )
    orient = _stitch_indexed(source.orient, indices)
    pose = _stitch_indexed(source.pose, indices)
    translation = _stitch_indexed(source.transl, indices)
    if orient.seams != pose.seams or orient.seams != translation.seams:
        raise ValueError("ground-truth windows produced inconsistent seam locations")

    local_axis_angle = torch.cat(
        (orient.frames.reshape(-1, 1, 3), pose.frames.reshape(-1, 21, 3)), dim=1
    ).to(device=device, dtype=torch.float32)
    local_axis_angle = _interpolate_local_pose(local_axis_angle, interp_scale)
    translation_frames = interpolate_joints(
        translation.frames.to(device=device, dtype=torch.float32), scale=interp_scale
    )
    betas = torch.from_numpy(np.asarray(source.betas[sequence_index]).copy()).to(
        device=device, dtype=torch.float32
    )
    vertices, joints = _run_smplx_chunks(
        local_axis_angle,
        translation_frames,
        betas,
        str(source.gender[sequence_index]),
        device,
        smplx_batch_size,
        smplx_cache,
    )
    if export_sink is not None:
        # Handles only; the caller materializes them.  See sampled_motion.
        #
        # The real arm's global_jpos is the dataset array itself, stitched on the
        # same stride-3 indices, never ground_truth_motion's FK output.  indices
        # already carries the stride, so no re-indexing happens here.
        export_sink.update(
            {
                "joints_coarse": _stitch_indexed(source.joints, indices),
                "smplx_pose": local_axis_angle,
                "smplx_transl": translation_frames,
                "betas": betas,
                "gender": str(source.gender[sequence_index]),
                "smplx_output_transform": "identity",
                "interp_scale": int(interp_scale),
            }
        )
    return (
        _upsampled_stitched(translation, vertices, interp_scale),
        _upsampled_stitched(translation, joints, interp_scale),
        sequence_index,
    )


def compute_metric_record(
    vertices: hsi_metrics.StitchedSequence,
    joints: hsi_metrics.StitchedSequence,
    geometry: SceneGeometry,
    goal: Sequence[float],
    fps: float,
) -> Dict[str, float]:
    record: Dict[str, float] = {}
    record.update(hsi_metrics.penetration_metrics(vertices, geometry))
    record.update(hsi_metrics.engagement_metrics(vertices, geometry))
    record.update(hsi_metrics.reachability_diagnostic(vertices, geometry))
    record.update(hsi_metrics.fs_nemf(joints))
    record.update(hsi_metrics.skate_ratio(joints, fps=fps))
    record.update(hsi_metrics.goal_metrics(joints, goal, fps=fps))
    record.update(hsi_metrics.goal_error_decomposition(joints.frames[:, 0], goal))
    record.update(hsi_metrics.jerk_metrics(joints, fps=fps))
    record.update(hsi_metrics.transition_distance(joints))
    record["frame_count"] = float(len(joints))
    record["window_count"] = float(len(joints.window_lengths))
    record["finite_motion"] = float(bool(torch.isfinite(joints.frames).all()))
    return record


def _aggregate_timing(
    sequences: Sequence[Mapping[str, float]],
    warmup_sequences: int,
    fps: float,
) -> Dict[str, object]:
    excluded = min(warmup_sequences, len(sequences))
    timed = sequences[excluded:]
    result: Dict[str, object] = {
        "aits": None,
        "avg_fps": None,
        "aggregate_fps": None,
        "rtf": None,
        "total_generation_seconds": None,
        "timed_sequence_count": len(timed),
        "avg_frames_per_seq": None,
        "avg_end_to_end_episode_seconds": None,
        "warmup_sequences_required": warmup_sequences,
        "warmup_sequences_excluded": excluded,
        "protocol_complete": bool(timed),
    }
    if timed:
        generation_seconds = [item["gen_seconds"] for item in timed]
        frames = [item["frames"] for item in timed]
        episode_seconds = [item["episode_seconds"] for item in timed]
        per_sequence_fps = [
            frame_count / seconds if seconds > 0 else 0.0
            for frame_count, seconds in zip(frames, generation_seconds)
        ]
        aggregate_fps = float(sum(frames) / sum(generation_seconds))
        result.update(
            {
                "aits": float(np.mean(generation_seconds)),
                "avg_fps": float(np.mean(per_sequence_fps)),
                "aggregate_fps": aggregate_fps,
                "rtf": aggregate_fps / fps,
                "total_generation_seconds": float(sum(generation_seconds)),
                "avg_frames_per_seq": float(np.mean(frames)),
                "avg_end_to_end_episode_seconds": float(np.mean(episode_seconds)),
            }
        )
    return result


def _capture_rng_state() -> Dict[str, object]:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


@contextmanager
def _rng_rewound(state: Mapping[str, object]) -> Iterator[None]:
    post_state = _capture_rng_state()
    _restore_rng_state(state)
    try:
        yield
    finally:
        _restore_rng_state(post_state)


class _ForwardCallCounter:
    def __init__(self, model: torch.nn.Module):
        self.count = 0
        self.handle = model.register_forward_pre_hook(self._increment)

    def _increment(self, _module, _args) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


def _aggregate_by_scene(records: Mapping[str, Mapping]) -> OrderedDict:
    grouped = defaultdict(list)
    for record in records.values():
        grouped[record["scene_name"]].append(record)
    summaries = OrderedDict()
    for scene_name in sorted(grouped):
        scene_records = grouped[scene_name]
        numeric = {}
        for key in scene_records[0]:
            values = [item.get(key) for item in scene_records]
            finite = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)]
            if finite:
                numeric[key] = float(np.mean(finite))
        summaries[scene_name] = {
            "sequence_count": len(scene_records),
            "non_watertight": scene_name in NON_WATERTIGHT_SCENES,
            "metrics_mean": numeric,
        }
    return summaries


def plan_episode_shards(
    window_counts: Sequence[int], shard_count: int
) -> Tuple[Tuple[int, ...], ...]:
    """Partition canonical episode ordinals into ``shard_count`` balanced shards.

    Balance is by *window* count, not episode count: the LINGO HSI test split has
    375 episodes carrying 2271 windows with min 2 and max 55, so a round-robin by
    episode index would hand one shard several 55-window episodes and let another
    finish early.  A sharded run costs the slowest shard, so the objective is the
    maximum per-shard window total.

    Deterministic greedy longest-first bin packing: episodes sorted by descending
    window count with the canonical ordinal as tie-break, each placed into the
    currently least-loaded shard with the lowest shard index as tie-break.  Each
    shard's ordinals are returned in ascending canonical order so a shard walks
    the episodes in the same relative order a serial run does.
    """
    if int(shard_count) < 1:
        raise ValueError("shard_count must be >= 1, got %s" % shard_count)
    shard_count = int(shard_count)
    counts = [int(value) for value in window_counts]
    if shard_count > len(counts):
        raise ValueError(
            "shard_count %d exceeds episode count %d" % (shard_count, len(counts))
        )
    loads = [0] * shard_count
    bins: List[List[int]] = [[] for _ in range(shard_count)]
    order = sorted(range(len(counts)), key=lambda index: (-counts[index], index))
    for index in order:
        target = min(range(shard_count), key=lambda shard: (loads[shard], shard))
        bins[target].append(index)
        loads[target] += counts[index]
    return tuple(tuple(sorted(shard)) for shard in bins)


def select_latency_subset(
    episodes: Sequence[Tuple[str, int, Mapping]], target_windows: int
) -> Tuple[int, ...]:
    """Pick a deterministic, scene-spanning episode subset under a window budget.

    Per-window sampling cost is architecturally near-constant (the model input is
    a fixed ``[1, 16, 232]``), but the guidance/penetration branch queries a
    per-scene occupancy grid and its lazily built EDT, so a subset must span
    scenes rather than take the first N episodes of one scene.

    Rule, two phases, both deterministic:

    1. *Coverage.*  Consider scenes in ascending order of their smallest episode's
       window count (canonical scene order breaks ties) and take that smallest
       episode while it fits the budget.  Taking the cheapest items first is the
       greedy optimum for maximizing the number of distinct scenes under a size
       budget, and it makes coverage monotone in the budget -- a plain canonical
       round-robin is not: at 50 windows it covered 14 scenes but at 60 only 6,
       because the extra room let one 40-window locomotion episode in early.
    2. *Fill.*  Round-robin over the covered scenes in canonical order taking each
       scene's next-smallest unused episode while it fits, to spend the remainder.

    Smallest-first buys scene coverage per window spent.  It does bias the
    fraction of step-0 (history-from-ground-truth) windows above the full
    protocol's 375/2271 = 0.165, which is the accepted cost of coverage; and eight
    of the 26 scenes have a *minimum* episode of 33-50 windows, so no subset under
    ~394 windows can cover every scene.  Returns canonical ordinals ascending.
    """
    target = int(target_windows)
    if target < 1:
        raise ValueError("latency_target_windows must be >= 1, got %s" % target_windows)
    by_scene: "OrderedDict[str, List[int]]" = OrderedDict()
    for ordinal, (scene_name, _scene_index, _episode) in enumerate(episodes):
        by_scene.setdefault(scene_name, []).append(ordinal)

    def windows_of(ordinal: int) -> int:
        return int(episodes[ordinal][2]["episode_num"])

    for scene_ordinals in by_scene.values():
        scene_ordinals.sort(key=lambda ordinal: (windows_of(ordinal), ordinal))
    scene_order = list(by_scene)
    cursor = {scene_name: 0 for scene_name in by_scene}
    selected: List[int] = []
    total = 0

    covered: List[str] = []
    coverage_order = sorted(
        scene_order,
        key=lambda scene_name: (windows_of(by_scene[scene_name][0]), scene_order.index(scene_name)),
    )
    for scene_name in coverage_order:
        ordinal = by_scene[scene_name][0]
        if total + windows_of(ordinal) > target:
            continue
        cursor[scene_name] = 1
        selected.append(ordinal)
        total += windows_of(ordinal)
        covered.append(scene_name)

    progressed = bool(covered)
    while progressed and total < target:
        progressed = False
        for scene_name in scene_order:
            if scene_name not in cursor or cursor[scene_name] == 0 or total >= target:
                continue
            scene_ordinals = by_scene[scene_name]
            position = cursor[scene_name]
            if position >= len(scene_ordinals):
                continue
            ordinal = scene_ordinals[position]
            if total + windows_of(ordinal) > target:
                continue
            cursor[scene_name] = position + 1
            selected.append(ordinal)
            total += windows_of(ordinal)
            progressed = True
    if not selected:
        raise ValueError(
            "latency_target_windows=%d is smaller than every episode's window count"
            % target
        )
    return tuple(sorted(selected))


# Wall-clock aggregates that a contended sharded run cannot measure.  Kept
# separate from call counts (``denoiser_calls_per_window``,
# ``sampler_steps_per_window``), which are hardware-independent and stay valid.
SHARD_INVALID_TIMING_KEYS = (
    "per_window_wall_seconds",
    "total_sampling_seconds",
    "aits",
    "avg_fps",
    "aggregate_fps",
    "rtf",
    "total_generation_seconds",
    "avg_end_to_end_episode_seconds",
)
SHARD_INVALID_RECORD_TIMING_KEYS = ("sampling_seconds", "per_window_wall_seconds")
SHARD_TIMING_INVALID_REASON = (
    "shard_count>1: concurrent shards contend for host CPU, PCIe and the SDF "
    "cache, so every wall-clock aggregate is contaminated; FPS comes only from "
    "a serial latency_subset pass"
)


def _invalidate_timing(timing: MutableMapping[str, Any]) -> None:
    """Null every wall-clock aggregate and mark the block explicitly invalid."""
    for key in SHARD_INVALID_TIMING_KEYS:
        if key in timing:
            timing[key] = None
    timing["protocol_complete"] = False
    timing["timing_valid"] = False
    timing["timing_invalid_reason"] = SHARD_TIMING_INVALID_REASON


def _invalidate_scene_summary_timing(scene_summary: Mapping[str, Any]) -> None:
    """Null the per-scene means of the two per-record wall-clock fields.

    ``_aggregate_by_scene`` silently drops a key whose values are all ``None``, so
    without this the merged payload would be structurally *different* from a
    serial one rather than explicitly null.
    """
    for summary in scene_summary.values():
        for key in SHARD_INVALID_RECORD_TIMING_KEYS:
            summary["metrics_mean"][key] = None


def merge_shard_payloads(
    payloads: Sequence[Mapping[str, Any]],
    expected_episodes: int,
    expected_windows: int,
    expected_shard_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Combine shard payloads into one structurally serial-identical payload.

    Every guard raises: a silently short merge is the worst possible outcome
    because it reads as a complete result.  The per-scene aggregation is
    recomputed over the union of records, never averaged from per-shard averages.

    ``expected_shard_count`` is the operator's own statement of how many shards
    the run had.  Without it the shard count is self-declared by the payloads on
    disk, so "merge 8 shards" over a directory holding a 2-shard pair would
    succeed: the pair is internally consistent.
    """
    if not payloads:
        raise ValueError("merge_shard_payloads received no shard payloads")
    expected_episodes = int(expected_episodes)
    expected_windows = int(expected_windows)

    by_index: Dict[int, Mapping[str, Any]] = {}
    for payload in payloads:
        block = payload.get("sharding")
        if not isinstance(block, Mapping):
            raise ValueError("shard payload has no 'sharding' block; not a sharded run")
        index = int(block["shard_index"])
        if index in by_index:
            raise ValueError("two payloads both claim shard_index=%d" % index)
        by_index[index] = payload

    declared = {int(item["sharding"]["shard_count"]) for item in payloads}
    if len(declared) != 1:
        raise ValueError("shard payloads disagree on shard_count: %s" % sorted(declared))
    shard_count = declared.pop()
    if expected_shard_count is not None and shard_count != int(expected_shard_count):
        raise ValueError(
            "payloads on disk declare shard_count=%d, the merge was asked for %d"
            % (shard_count, int(expected_shard_count))
        )
    if len(payloads) != shard_count:
        raise ValueError(
            "expected %d shard payloads, received %d" % (shard_count, len(payloads))
        )
    missing = sorted(set(range(shard_count)) - set(by_index))
    if missing:
        raise ValueError(
            "shard indices %s are missing; refusing to merge %d of %d shards"
            % (missing, len(by_index), shard_count)
        )

    # Merging shards produced by different checkpoints, seeds, samplers or
    # guidance settings would fabricate a result that no run ever produced.
    agreement_keys = (
        "seed", "sample_type", "guided", "fps", "sampling_body", "model_name",
        "episode_subset", "future_occ_diagnostic",
    )
    reference = by_index[0]
    for index in range(1, shard_count):
        candidate = by_index[index]
        for key in agreement_keys:
            if candidate.get(key) != reference.get(key):
                raise ValueError(
                    "shard %d disagrees with shard 0 on %r: %r vs %r"
                    % (index, key, candidate.get(key), reference.get(key))
                )
        if candidate["checkpoint"]["checkpoint_sha256"] != reference["checkpoint"]["checkpoint_sha256"]:
            raise ValueError(
                "shard %d evaluated a different checkpoint (%s) than shard 0 (%s)"
                % (
                    index,
                    candidate["checkpoint"]["checkpoint_sha256"][:12],
                    reference["checkpoint"]["checkpoint_sha256"][:12],
                )
            )
        for key in ("canonical_episode_total", "canonical_window_total"):
            if int(candidate["sharding"][key]) != int(reference["sharding"][key]):
                raise ValueError(
                    "shard %d declares %s=%d, shard 0 declares %d"
                    % (index, key, int(candidate["sharding"][key]), int(reference["sharding"][key]))
                )

    canonical_episodes = int(reference["sharding"]["canonical_episode_total"])
    canonical_windows = int(reference["sharding"]["canonical_window_total"])
    subset = reference.get("episode_subset", {"enabled": False})
    if subset.get("enabled", False):
        expected_ordinals = set(int(value) for value in subset["canonical_ordinals"])
        if int(subset["episode_count"]) != expected_episodes:
            raise ValueError("episode subset count disagrees with merge expectation")
        if int(subset["window_count"]) != expected_windows:
            raise ValueError("episode subset window count disagrees with merge expectation")
    else:
        if canonical_episodes != expected_episodes:
            raise ValueError(
                "shards enumerate %d canonical episodes, protocol expects %d"
                % (canonical_episodes, expected_episodes)
            )
        if canonical_windows != expected_windows:
            raise ValueError(
                "shards enumerate %d canonical windows, protocol expects %d"
                % (canonical_windows, expected_windows)
            )
        expected_ordinals = set(range(expected_episodes))

    records: Dict[str, Mapping[str, Any]] = {}
    ordinals: List[int] = []
    window_total = 0
    call_counts = set()
    sampler_steps = set()
    warmup_excluded = 0
    timed_sequences = 0
    warmup_required = set()
    for index in range(shard_count):
        payload = by_index[index]
        for key, record in payload["metrics"].items():
            if key in records:
                raise ValueError(
                    "duplicate sequence key %r appears in shard %d and an earlier shard"
                    % (key, index)
                )
            records[key] = record
            ordinals.append(int(record["canonical_ordinal"]))
        timing = payload["timing"]
        window_total += int(timing["window_count"])
        call_counts.add(int(timing["denoiser_calls_per_window"]))
        sampler_steps.add(int(timing["sampler_steps_per_window"]))
        warmup_excluded += int(timing["warmup_sequences_excluded"])
        timed_sequences += int(timing["timed_sequence_count"])
        warmup_required.add(int(timing["warmup_sequences_required"]))

    if len(records) != expected_episodes:
        raise ValueError(
            "merged %d episode records, protocol expects %d"
            % (len(records), expected_episodes)
        )
    duplicate_ordinals = sorted(
        ordinal for ordinal, count in
        {value: ordinals.count(value) for value in set(ordinals)}.items() if count > 1
    )
    if duplicate_ordinals:
        raise ValueError("canonical ordinals appear in more than one shard: %s" % duplicate_ordinals[:10])
    unexpected = sorted(set(ordinals) - expected_ordinals)
    if unexpected:
        raise ValueError("canonical ordinals outside the registered cohort: %s" % unexpected[:10])
    absent = sorted(expected_ordinals - set(ordinals))
    if absent:
        raise ValueError(
            "canonical ordinals missing from the merge: %d of %d, first %s"
            % (len(absent), expected_episodes, absent[:10])
        )
    if window_total != expected_windows:
        raise ValueError(
            "merged %d windows, protocol expects %d" % (window_total, expected_windows)
        )
    if len(call_counts) != 1:
        raise ValueError("shards disagree on denoiser_calls_per_window: %s" % sorted(call_counts))
    if len(sampler_steps) != 1:
        raise ValueError("shards disagree on sampler_steps_per_window: %s" % sorted(sampler_steps))
    if len(warmup_required) != 1:
        raise ValueError("shards disagree on warmup_sequences_required: %s" % sorted(warmup_required))

    ordered = OrderedDict(
        sorted(records.items(), key=lambda item: int(item[1]["canonical_ordinal"]))
    )
    scene_summary = _aggregate_by_scene(ordered)
    _invalidate_scene_summary_timing(scene_summary)

    timing = dict(reference["timing"])
    timing.update(
        {
            "window_count": window_total,
            "denoiser_calls_per_window": sorted(call_counts)[0],
            "sampler_steps_per_window": sorted(sampler_steps)[0],
            "timed_sequence_count": timed_sequences,
            "avg_frames_per_seq": None,
            "warmup_sequences_required": sorted(warmup_required)[0],
            "warmup_sequences_excluded": warmup_excluded,
        }
    )
    _invalidate_timing(timing)

    merged = dict(reference)
    merged.update(
        {
            "sequence_count": len(ordered),
            "scene_count": len(scene_summary),
            "scene_summary": scene_summary,
            "timing": timing,
            "metrics": ordered,
            "sharding": {
                "shard_index": None,
                "shard_count": shard_count,
                "merged_shard_count": shard_count,
                "canonical_episode_total": canonical_episodes,
                "canonical_window_total": canonical_windows,
                "eligible_episode_count": int(subset.get("episode_count", canonical_episodes)),
                "eligible_window_total": int(subset.get("window_count", canonical_windows)),
                "shard_window_totals": [
                    int(by_index[index]["timing"]["window_count"]) for index in range(shard_count)
                ],
                "shard_episode_counts": [
                    int(by_index[index]["sequence_count"]) for index in range(shard_count)
                ],
                "partition_rule": reference["sharding"]["partition_rule"],
                "per_episode_seeding": reference["sharding"]["per_episode_seeding"],
                "timing_valid": False,
            },
            "latency_subset": {"enabled": False},
        }
    )
    merged.pop("output_dir", None)
    return merged


def _merge_shards(cfg: DictConfig) -> Path:
    parent = Path(
        str(cfg.merge_shard_dir) if cfg.get("merge_shard_dir", None) is not None
        else str(cfg.lingo_output_dir)
    )
    shard_count = int(cfg.shard_count)
    if shard_count < 2:
        raise ValueError("merge_shards requires shard_count >= 2, got %d" % shard_count)
    if not parent.is_dir():
        raise FileNotFoundError("shard parent directory does not exist: %s" % parent)
    paths = sorted(parent.glob("*-shard[0-9][0-9]of[0-9][0-9]/evaluation/per_sequence_metrics.json"))
    if not paths:
        raise FileNotFoundError("no shard payloads found under %s" % parent)
    payloads = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    merged = merge_shard_payloads(
        payloads,
        expected_episodes=int(cfg.merge_expected_episodes),
        expected_windows=int(cfg.merge_expected_windows),
        expected_shard_count=shard_count,
    )
    merged["merged_from"] = [str(path) for path in paths]
    output_dir = parent / (
        "%s-merged%02d" % (Path(str(cfg.ckpt_path)).stem, shard_count)
    )
    return _write_payload(output_dir, merged)


def _motion_export_record(
    *,
    joints_coarse: hsi_metrics.StitchedSequence,
    smplx_pose: torch.Tensor,
    smplx_transl: torch.Tensor,
    betas: torch.Tensor,
    gender: str,
    smplx_output_transform: str,
    interp_scale: int,
) -> Dict[str, np.ndarray]:
    """Build one sequence's export arrays.

    ``joints_coarse``
        The stitched joint sequence **at the rollout rate**, un-interpolated and
        un-FK'd: ``coarse_points`` for the generated arm and
        ``_stitch_indexed(source.joints, indices)`` for the real arm.  Both are
        the dataset's own 28-slot channel layout in LINGO y-up world metres.

        This is deliberately *not* ``metric_joints``.  ``metric_joints`` is SMPL-X
        FK indexed by ``SMPLX_JOINTS_28``, whose slots 25/27 are SMPL-X 28/43
        (``middle1``) where the frozen evaluator's training bundle carries SMPL-X
        34/49 (``ring1``) -- about 2.3 cm apart -- and it reaches the generated arm
        through an IK->FK round trip that the real arm never takes.  Exporting FK
        output would therefore make the two arms incomparable in two independent
        ways at once.

    ``smplx_pose`` / ``smplx_transl``
        Exactly the tensors handed to :func:`run_smplx_model`, at the FK-input
        rate ``interp_scale * len(joints_coarse)``.  Storing the FK *inputs* rather
        than a resampled version of them is what makes a future metric
        redefinition a CPU job: the rebuild reproduces the very vertices and
        joints the sealed metrics were computed on, with no interpolation to
        re-derive.

    ``smplx_output_transform``
        From motion-export schema 2 both arms are ``identity``: the parameters are
        the literal FK inputs and the FK output is used as-is.  The field is kept
        because schema-1 generated-arm files exist on disk carrying
        ``"zup_to_yup"``, and a reader must not obey them -- see
        :data:`MOTION_EXPORT_SCHEMA_VERSION` for the repair recipe.  ``zup_to_yup``
        stays in the accepted set only so such a file can still be *described*; no
        code path produces it any more.
    """
    if smplx_output_transform not in ("identity", "zup_to_yup"):
        raise ValueError("unknown smplx_output_transform %r" % smplx_output_transform)
    joints = joints_coarse.frames.detach().to(torch.float32).cpu().numpy()
    joints = np.ascontiguousarray(joints.reshape(-1, 28, 3))
    if not np.isfinite(joints).all():
        raise ValueError("exported global_jpos is not finite")
    coarse_length = int(joints.shape[0])
    pose = smplx_pose.detach().to(torch.float32).cpu().numpy().reshape(-1, 22, 3)
    transl = smplx_transl.detach().to(torch.float32).cpu().numpy().reshape(-1, 3)
    if pose.shape[0] != transl.shape[0]:
        raise ValueError(
            "SMPL-X pose has %d frames but transl has %d" % (pose.shape[0], transl.shape[0])
        )
    if pose.shape[0] != coarse_length * int(interp_scale):
        raise ValueError(
            "SMPL-X parameters have %d frames, expected %d coarse frames x interp %d"
            % (pose.shape[0], coarse_length, int(interp_scale))
        )
    if not np.isfinite(pose).all() or not np.isfinite(transl).all():
        raise ValueError("exported SMPL-X parameters are not finite")
    return {
        "global_jpos": joints,
        # run_smplx_model consumes pose_pred[:, :1] as global_orient and
        # pose_pred[:, 1:] as body_pose; split here, re-concatenate to rebuild.
        "global_orient": np.ascontiguousarray(pose[:, 0]),
        "body_pose": np.ascontiguousarray(pose[:, 1:]),
        "transl": np.ascontiguousarray(transl),
        "betas": np.ascontiguousarray(
            betas.detach().to(torch.float32).cpu().numpy().reshape(-1)
        ),
        "gender": np.asarray(str(gender)),
        "smplx_output_transform": np.asarray(str(smplx_output_transform)),
        "interp_scale": np.asarray(int(interp_scale), dtype=np.int32),
        # The seam structure of the coarse sequence.  Any seam-sensitive metric
        # (boundary_jerk) needs it, and _upsampled_stitched derives the fine-rate
        # seams from it by multiplying through by interp_scale.
        "window_lengths": np.asarray(joints_coarse.window_lengths, dtype=np.int32),
        "seams": np.asarray(joints_coarse.seams, dtype=np.int32),
        "history_frames": np.asarray(int(joints_coarse.history_frames), dtype=np.int32),
    }


def _motion_export_extra(
    scene_name: str, episode: Mapping, sequence_index: int
) -> Dict[str, np.ndarray]:
    """Per-sequence provenance, built identically for both arms."""
    return {
        "scene_name": np.asarray(str(scene_name)),
        "data_idx": np.asarray(int(episode["data_idx"]), dtype=np.int64),
        "source_sequence_index": np.asarray(int(sequence_index), dtype=np.int64),
        "episode_num": np.asarray(int(episode["episode_num"]), dtype=np.int32),
        "pelvis_goal": np.asarray(episode["pelvis_goal"], dtype=np.float32),
        "start_location": np.asarray(episode["start_location"], dtype=np.float32),
    }


def _motion_condition_id(episode: Mapping) -> str:
    """The LINGO language-window index, which is what selects the caption."""
    return "lingo:%08d" % int(episode["data_idx"])


def _write_motion_npz(
    motion_dir: Path,
    *,
    sequence_id: str,
    condition_id: str,
    caption: str,
    fps: float,
    record: Mapping[str, np.ndarray],
    extra: Mapping[str, np.ndarray],
) -> Tuple[str, int]:
    """Write one non-pickle NPZ and return its file name and size in bytes."""
    payload = dict(record)
    payload.update(extra)
    payload["schema_version"] = np.asarray(MOTION_EXPORT_SCHEMA_VERSION, dtype=np.int32)
    payload["sequence_id"] = np.asarray(str(sequence_id))
    payload["condition_id"] = np.asarray(str(condition_id))
    payload["caption"] = np.asarray(str(caption))
    payload["fps"] = np.asarray(float(fps), dtype=np.float32)
    for key, value in payload.items():
        if np.asarray(value).dtype == np.object_:
            raise TypeError("export field %r is object dtype; NPZ must load without pickle" % key)
    file_name = "%s.npz" % str(sequence_id).replace(":", "_")
    path = motion_dir / file_name
    # "xb" so a sanitized-name collision is a hard failure, never an overwrite.
    with path.open("xb") as handle:
        np.savez(handle, **payload)
    return file_name, int(path.stat().st_size)


def _motion_export_block(
    lengths: Mapping[str, int], fps: float, arm: str
) -> Dict[str, Any]:
    """Describe the export and count what the frozen evaluator will discard.

    The guard counts are reported, never silently applied: the export keeps every
    frame.  A sequence below ``T2M_MIN_FRAMES`` is dropped whole by the evaluator,
    and one whose length is not a multiple of ``T2M_LENGTH_MULTIPLE`` loses its
    tail.  Naming both is what stops a silent truncation from reading as full
    coverage.
    """
    below = sorted(key for key, length in lengths.items() if int(length) < T2M_MIN_FRAMES)
    truncated = {
        key: int(length) % T2M_LENGTH_MULTIPLE
        for key, length in lengths.items()
        if int(length) % T2M_LENGTH_MULTIPLE
    }
    return {
        "schema_version": MOTION_EXPORT_SCHEMA_VERSION,
        "arm": arm,
        "layout": "one non-pickle NPZ per sequence under motion/<sequence_id>.npz",
        "global_jpos": {
            "shape": "[T, 28, 3]",
            "units": "metres",
            "frame": "LINGO y-up world",
            "fps": float(fps),
            "source": (
                "coarse_points.frames (generated) / "
                "_stitch_indexed(source.joints, indices).frames (real)"
            ),
            "note": (
                "dataset channel order, no re-indexing and no SMPL-X FK; slots "
                "24-27 are SMPL-X 25/34/40/49 (index1/ring1) as in the frozen "
                "evaluator's training bundle, not SMPLX_JOINTS_28's middle1"
            ),
        },
        "smplx": {
            "fields": ["global_orient[F,3]", "body_pose[F,21,3]", "transl[F,3]", "betas[16]", "gender"],
            "frame_count": "F = T * interp_scale",
            "rebuild": (
                "pose = concatenate([global_orient[:, None], body_pose], axis=1); "
                "vertices, joints28 = run_smplx_model(pose, transl, betas, gender, "
                "joints_ind=SMPLX_JOINTS_28).  Wrap with "
                "StitchedSequence(frames, seams=seams*interp_scale, "
                "window_lengths=window_lengths*interp_scale, "
                "history_frames=history_frames*interp_scale) to recompute seam metrics."
            ),
            "note": (
                "schema 2 onward both arms share one SMPL-X parameter frame -- the "
                "frame human_joints_aligned.npy is stored in -- and "
                "smplx_output_transform is always 'identity', so the rebuild above "
                "needs no per-arm branch; a schema-1 generated-arm file instead "
                "needs transl replaced by zup_to_yup(transl) and its recorded "
                "'zup_to_yup' output transform ignored"
            ),
        },
        "t2m_min_frames": T2M_MIN_FRAMES,
        "t2m_length_multiple": T2M_LENGTH_MULTIPLE,
        "sequences_exported": len(lengths),
        "frames_exported": int(sum(int(length) for length in lengths.values())),
        "below_min_frames_count": len(below),
        "below_min_frames_sequences": below,
        "losing_frames_to_truncation_count": len(truncated),
        "frames_lost_to_truncation": int(sum(truncated.values())),
        "losing_frames_to_truncation_sequences": dict(sorted(truncated.items())),
    }


def _write_payload(
    output_dir: Path,
    payload: Mapping,
    motion_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Path:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite LINGO HSI output: %s" % output_dir)
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    # The motion export is written here rather than streamed during the rollout
    # because the "refuse to overwrite" guard above is the run's isolation
    # contract: creating output_dir early to stream into it would make this
    # function reject its own run.  The whole export is ~100 KB/sequence, so
    # holding it costs about 38 MB for a 375-episode cell.
    if motion_records is not None:
        motion_dir = output_dir / "motion"
        motion_dir.mkdir(parents=False, exist_ok=False)
        for item in motion_records:
            _write_motion_npz(motion_dir, **item)
    output_path = evaluation_dir / "per_sequence_metrics.json"
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(
            _sanitize_json(payload),
            handle,
            indent=2,
            allow_nan=False,
            default=_json_value,
        )
        handle.write("\n")
    return output_path


def _write_array_payload(
    output_dir: Path,
    payload: MutableMapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    file_name: str,
) -> Path:
    """Persist diagnostic arrays before their JSON metadata, without overwrite."""
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite LINGO HSI output: %s" % output_dir)
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    array_path = evaluation_dir / file_name
    normalized = {}
    for name, value in arrays.items():
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        if not np.isfinite(array).all():
            raise ValueError("diagnostic array %s contains non-finite values" % name)
        normalized[str(name)] = array
    with array_path.open("xb") as handle:
        np.savez_compressed(handle, **normalized)
    raw = array_path.read_bytes()
    payload["array_archive"] = {
        "path": str(array_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in normalized.items()
        },
    }
    output_path = evaluation_dir / "per_sequence_metrics.json"
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(
            _sanitize_json(payload),
            handle,
            indent=2,
            allow_nan=False,
            default=_json_value,
        )
        handle.write("\n")
    return output_path


def evaluate_ground_truth(cfg: DictConfig) -> Path:
    device = torch.device(str(cfg.device))
    export_motion = bool(cfg.get("export_motion", False))
    episodes = _load_episodes(
        Path(cfg.lingo_episode_dir),
        None if cfg.lingo_sequence_limit is None else int(cfg.lingo_sequence_limit),
    )
    source = GroundTruthSource(DATASET_ROOT, keep_text=export_motion)
    smplx_cache: Dict[str, torch.nn.Module] = {}
    geometries: Dict[str, SceneGeometry] = {}
    records = OrderedDict()
    motion_records: List[Dict[str, Any]] = []
    export_lengths: "OrderedDict[str, int]" = OrderedDict()
    export_fps = float(cfg.fps) / float(cfg.interp_s)

    for ordinal, (scene_name, scene_index, episode) in enumerate(episodes, start=1):
        if scene_name not in geometries:
            geometries[scene_name] = SceneGeometry.from_scene(
                scene_name,
                dataset_root=DATASET_ROOT,
                mesh_root=Path(cfg.lingo_mesh_root),
                cache_dir=default_cache_dir(),
            )
        export_sink: Optional[Dict[str, Any]] = {} if export_motion else None
        vertices, joints, sequence_index = ground_truth_motion(
            source,
            episode,
            device,
            int(cfg.interp_s),
            int(cfg.smplx_batch_size),
            smplx_cache,
            export_sink=export_sink,
        )
        metric = compute_metric_record(
            vertices, joints, geometries[scene_name], episode["pelvis_goal"], float(cfg.fps)
        )
        sequence_name = "%s:%06d" % (scene_name, sequence_index)
        if sequence_name in records:
            raise ValueError("duplicate sequence key %s" % sequence_name)
        if export_sink is not None:
            export_record = _motion_export_record(**export_sink)
            motion_records.append(
                {
                    "sequence_id": sequence_name,
                    "condition_id": _motion_condition_id(episode),
                    "caption": source.caption(int(episode["data_idx"])),
                    "fps": export_fps,
                    "record": export_record,
                    "extra": _motion_export_extra(scene_name, episode, sequence_index),
                }
            )
            export_lengths[sequence_name] = int(export_record["global_jpos"].shape[0])
        metric.update(
            {
                "scene_name": scene_name,
                "scene_sequence_index": scene_index,
                "source_sequence_index": sequence_index,
                "is_object": False,
                "non_watertight": scene_name in NON_WATERTIGHT_SCENES,
            }
        )
        records[sequence_name] = metric
        print(
            "GT %d/%d %s pen_ratio=%.6f depth=%.6f contact=%.3f fs=%.6f"
            % (
                ordinal,
                len(episodes),
                sequence_name,
                metric["pen_ratio"],
                metric["pen_depth_mean"],
                metric["contact_count"],
                metric["fs_nemf"],
            ),
            flush=True,
        )

    scene_summary = _aggregate_by_scene(records)
    aggregate_pen_ratio = float(np.mean([item["pen_ratio"] for item in records.values()]))
    if aggregate_pen_ratio >= 0.3:
        raise RuntimeError(
            "ground-truth pen_ratio %.6f is >= 0.3; stop for coordinate/SDF diagnosis"
            % aggregate_pen_ratio
        )
    payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "model_name": "ground_truth",
        "seed": int(cfg.seed),
        "sampling_body": "smplx_vertices_10475",
        "fps": float(cfg.fps),
        "sequence_count": len(records),
        "scene_count": len(scene_summary),
        "scene_summary": scene_summary,
        "metrics": records,
    }
    if export_motion:
        payload["motion_export"] = _motion_export_block(export_lengths, export_fps, "real")
    return _write_payload(
        Path(cfg.lingo_output_dir),
        payload,
        motion_records=motion_records if export_motion else None,
    )


def _remap_checkpoint_keys(
    state_dict: Mapping[str, Any],
) -> Tuple["OrderedDict[str, Any]", int]:
    """Normalize DataParallel and legacy hand-goal checkpoint key names."""
    remapped = OrderedDict()
    remap_count = 0
    for original_key, value in state_dict.items():
        key = original_key[len("module.") :] if original_key.startswith("module.") else original_key
        if key.startswith("embedding_hand_goal."):
            key = "embedding_scene_goal." + key[len("embedding_hand_goal.") :]
            remap_count += 1
        if key in remapped:
            raise KeyError("checkpoint remap produced duplicate key %s" % key)
        remapped[key] = value
    return remapped, remap_count


def _assert_key_sets_match(
    remapped_keys: Iterable[str], expected_keys: Iterable[str]
) -> None:
    checkpoint_keys = set(remapped_keys)
    model_keys = set(expected_keys)
    if checkpoint_keys != model_keys:
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        raise RuntimeError(
            "checkpoint key set mismatch: checkpoint=%d model=%d; missing=%d %s; "
            "unexpected=%d %s"
            % (
                len(checkpoint_keys),
                len(model_keys),
                len(missing),
                missing[:10],
                len(unexpected),
                unexpected[:10],
            )
        )


def _load_strict_checkpoint(
    cfg: DictConfig,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Instantiate and strictly load either supported checkpoint orientation."""
    model = hydra.utils.instantiate(cfg.model.infbagel)
    checkpoint_path = Path(str(cfg.ckpt_path)).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    remapped, remap_count = _remap_checkpoint_keys(checkpoint)
    _assert_key_sets_match(remapped, model.state_dict())
    model.load_state_dict(remapped, strict=True)
    model.to(str(cfg.device))
    model.eval()

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "remapped_tensor_count": remap_count,
        "embedding_orientation": "hand_goal" if remap_count > 0 else "scene_goal",
        "tensor_count": len(remapped),
    }
    return model, provenance


def _preflight_nearest_free_voxel(dataset) -> None:
    """Invoke the guidance occupancy query once, before any GPU hours are spent.

    ``hasattr`` is not enough: ``InfBaGelMixDataset.get_nearest_free_voxel``
    exists but borrows its body from ``InfBaGelDataset`` unbound, so a change to
    that body's internal dispatch raises only on the first guidance application,
    many hours into a run.  Cost is seconds, not the sub-millisecond of a bare
    query: probing an occupied voxel forces one ``LazyOccRef`` EDT for scene 0
    and parks it in that cache's four slots.  Against a multi-hour guided run
    that is free; it is not free enough to put on a per-window path.
    """
    grid = dataset.scene_grid_torch
    device = grid.device
    lower = grid[:3]
    upper = grid[3:6]
    dims = grid[6:]
    voxel_size = torch.div(upper - lower, dims)
    probes = [(lower + upper) / 2.0]
    # Probe one known-occupied voxel so the call also exercises the occ_ref
    # displacement branch, which is where the direct and materialized bodies
    # index differently.  argmax, not nonzero: the grid is 400x100x600.
    occupancy = (dataset.scene_occ[0] == 1).reshape(-1)
    if bool(occupancy.any()):
        flat_index = int(occupancy.to(torch.uint8).argmax())
        depth = int(dims[2])
        height = int(dims[1])
        voxel = torch.tensor(
            [flat_index // (height * depth), (flat_index // depth) % height, flat_index % depth],
            device=device,
            dtype=torch.float64,
        )
        probes.append(lower + (voxel + 0.5) * voxel_size)
    points = torch.stack([probe.to(torch.float32) for probe in probes])
    points = points.reshape(1, 1, len(probes), 3).to(device=device, dtype=torch.float32)
    scene_flag = torch.zeros((1,), device=device, dtype=torch.long)
    try:
        is_penetrating, nearest_free_points = dataset.get_nearest_free_voxel(points, scene_flag)
    except Exception as error:  # surface the real cause with the real cost attached
        raise RuntimeError(
            "%s.get_nearest_free_voxel is not callable on this instance (%s: %s); "
            "guided evaluation would fail on its first guidance application"
            % (type(dataset).__name__, type(error).__name__, error)
        ) from error
    if tuple(is_penetrating.shape) != (1, 1, len(probes)):
        raise RuntimeError(
            "get_nearest_free_voxel returned penetration shape %s, expected %s"
            % (tuple(is_penetrating.shape), (1, 1, len(probes)))
        )
    if tuple(nearest_free_points.shape) != (1, 1, len(probes), 3):
        raise RuntimeError(
            "get_nearest_free_voxel returned free-point shape %s, expected %s"
            % (tuple(nearest_free_points.shape), (1, 1, len(probes), 3))
        )
    if is_penetrating.dtype is not torch.bool:
        raise RuntimeError(
            "get_nearest_free_voxel penetration dtype is %s, expected torch.bool"
            % is_penetrating.dtype
        )
    if nearest_free_points.dtype is not points.dtype:
        raise RuntimeError(
            "get_nearest_free_voxel free-point dtype is %s, expected %s"
            % (nearest_free_points.dtype, points.dtype)
        )
    if not bool(torch.isfinite(nearest_free_points).all()):
        raise RuntimeError("get_nearest_free_voxel returned non-finite free points")


def _scene_only_dataset(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    if not hasattr(dataset, "get_nearest_free_voxel"):
        raise TypeError("InfBaGelMixDataset lacks get_nearest_free_voxel")
    from datasets.infbagel import LazyOccRef

    if not isinstance(dataset.scene_occ_ref, LazyOccRef):
        raise TypeError("scene-only evaluation requires LazyOccRef, found %s" % type(dataset.scene_occ_ref).__name__)
    _preflight_nearest_free_voxel(dataset)
    # test_infbagel_hosi.sample_step uses the legacy dataset spelling.
    dataset.scene_dict = dataset.unified_scene_dict
    dataset.scene_grid_np = dataset.lingo_dataset.scene_grid_np

    # The settled sampler's occ_temp path still transforms an object point cloud
    # even for is_object=False.  Supply its shape-only zero tensor but prevent it
    # from being written into occupancy; LINGO conditioning is scene-only.
    original_get_occ = dataset.get_occ_for_points

    def scene_only_occ(points, _object_points, scene_flag):
        return original_get_occ(points, None, scene_flag)

    dataset.get_occ_for_points = scene_only_occ
    return dataset


def _lingo_item(dataset, data_idx: int):
    info = dataset.lingo_dataset[int(data_idx)]
    scene_name = str(dataset.lingo_window_scene_name[int(data_idx)])
    original_flag = dataset.lingo_dataset.scene_dict[scene_name]
    info["scene_flag"] = dataset.scene_flag_mapping[("lingo", original_flag)]
    info["is_object"] = False
    info["object_name"] = "none"
    return info


def _scene_condition(cfg: DictConfig, episode: Mapping, need_scene: bool = True):
    device = str(cfg.device)
    batch = int(cfg.batch_size)
    return {
        "scene_name": str(episode["scene_name"]),
        "pelvis_goal": torch.tensor(episode["pelvis_goal"], device=device, dtype=torch.float32),
        "object_goal": torch.tensor(episode["object_goal"], device=device, dtype=torch.float32),
        "need_scene": torch.full((batch,), need_scene, device=device, dtype=torch.bool),
        "need_pelvis_dir": torch.ones((batch,), device=device, dtype=torch.bool),
        "need_object": torch.zeros((batch,), device=device, dtype=torch.bool),
        "is_loco": torch.ones((batch,), device=device, dtype=torch.bool),
        "need_pi": torch.ones((batch,), device=device, dtype=torch.bool),
        "start_location": torch.tensor(
            episode["start_location"], device=device, dtype=torch.float32
        ),
        "episode_num": int(episode["episode_num"]),
    }


def _straight_trajectory(start: Sequence[float], goal: Sequence[float]) -> np.ndarray:
    start_xz = np.asarray(start, dtype=np.float64)[[0, 2]]
    goal_xz = np.asarray(goal, dtype=np.float64)[[0, 2]]
    distance = float(np.linalg.norm(goal_xz - start_xz))
    count = max(2, int(math.ceil(distance / 0.02)) + 1)
    return np.linspace(start_xz, goal_xz, count)


def _gt_pelvis_trajectory(source, sequence_index: int) -> np.ndarray:
    """The episode's ground-truth pelvis xz path, resampled to _straight_trajectory's
    2 cm spacing so `base_step`'s arc-length arithmetic is unchanged.

    ATTRIBUTION UPPER BOUND ONLY.  A deployment has no ground-truth path, so a cell
    run with hsi_gt_trajectory=true bounds what the straight-chord approximation
    costs; it can never be a baseline.  The deployable knob is hsi_lookahead_m.
    """
    begin = int(source.sequence_start[sequence_index])
    finish = int(source.sequence_end[sequence_index])
    path = np.asarray(source.joints[begin:finish, 0, :], dtype=np.float64)[:, [0, 2]]
    if path.shape[0] < 2:
        return np.repeat(path[:1], 2, axis=0)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(arc[-1])
    if total < 1e-9:
        return np.repeat(path[:1], 2, axis=0)
    count = max(2, int(math.ceil(total / 0.02)) + 1)
    even = np.linspace(0.0, total, count)
    return np.stack(
        [np.interp(even, arc, path[:, 0]), np.interp(even, arc, path[:, 1])], axis=1
    )


def sampled_motion(
    cfg: DictConfig,
    dataset,
    sampler,
    episode: Mapping,
    source: GroundTruthSource,
    smplx_cache: MutableMapping[str, torch.nn.Module],
    call_counter: _ForwardCallCounter,
    need_scene: bool = True,
    export_sink: Optional[MutableMapping[str, Any]] = None,
):
    # Importing here keeps the GT-first path independent of model sampling while
    # reusing the established autoregressive helpers verbatim.
    from test_infbagel_hosi import get_mat, sample_step, synchronize_cuda
    from utils import transform_points

    device = torch.device(str(cfg.device))
    data_idx = int(episode["data_idx"])
    data = _lingo_item(dataset, data_idx)
    sequence_index, episode_indices = source.episode_indices(
        data_idx, int(episode["episode_num"])
    )
    source_start = int(source.sequence_start[sequence_index])

    cond = _scene_condition(cfg, episode, need_scene=need_scene)
    cond["raw_text"] = dataset.lingo_dataset.text[data_idx][0]
    cond["text_emb"] = data["text_clip_embedding"].to(device).unsqueeze(0)
    if bool(cfg.get("hsi_gt_trajectory", False)):
        trajectory = _gt_pelvis_trajectory(source, sequence_index)
    else:
        trajectory = _straight_trajectory(episode["start_location"], episode["pelvis_goal"])
    seq_name_dict = {0: str(data["seq_name"])}

    joints_norm = torch.from_numpy(data["joints"]).to(device).reshape(1, WINDOW_FRAMES, -1)
    mat = torch.from_numpy(data["mat"]).to(device).reshape(1, 4, 4)
    mat_rotation_transpose = mat[0, :3, :3].T
    object_trans = torch.from_numpy(data["object_trans"]).to(device).reshape(1, WINDOW_FRAMES, 3)
    object_rotation = torch.from_numpy(data["object_rot_mat"]).to(device).reshape(1, WINDOW_FRAMES, 3, 3)
    object_rotation_ref = torch.from_numpy(data["obj_rot_mat_ref"]).to(device).reshape(1, 3, 3)
    object_points = torch.from_numpy(data["object_points"]).to(
        device=device, dtype=torch.float32
    ).reshape(1, -1, 3)
    object_bps = data["obj_bps_data"].to(device).unsqueeze(0)
    contact = torch.from_numpy(data["contact_label"]).to(device).reshape(1, WINDOW_FRAMES, 4)
    global_rotation_6d = data["global_rot_6d"].to(device).reshape(1, WINDOW_FRAMES, 22, 6)
    rest_offsets = torch.from_numpy(data["rest_human_offsets"]).to(device, dtype=torch.float32)
    betas = torch.from_numpy(data["betas"]).to(device, dtype=torch.float32)
    gender = str(data["gender"])

    points_world = dataset.denormalize_torch(joints_norm)
    object_trans_world = dataset.denormalize_torch(object_trans, is_object=True)
    start = np.asarray(episode["start_location"], dtype=np.float64)
    goal = np.asarray(episode["pelvis_goal"], dtype=np.float64)
    theta = np.arctan2(-goal[2] + start[2], goal[0] - start[0]) + np.pi / 2.0
    desired_rotation = torch.from_numpy(Rotation.from_euler("y", theta).as_matrix()).to(
        device=device, dtype=torch.float32
    )
    points_world = points_world.reshape(1, WINDOW_FRAMES, 28, 3) @ desired_rotation.T
    points_world = points_world.reshape(1, WINDOW_FRAMES, 84)
    object_trans_world = object_trans_world @ desired_rotation.T
    object_points = object_points @ desired_rotation.T
    translation_shift = points_world[:, [0], :3] - cond["start_location"]
    translation_shift[0, 0, 1] = 0.0
    points_world = (points_world.reshape(1, WINDOW_FRAMES, 28, 3) - translation_shift[:, :, None])
    points_world = points_world.reshape(1, WINDOW_FRAMES, 84)
    object_trans_world = object_trans_world - translation_shift
    object_points = object_points - translation_shift
    global_matrices = transforms.rotation_6d_to_matrix(global_rotation_6d)
    global_matrices = desired_rotation @ global_matrices
    global_rotation_6d = transforms.matrix_to_rotation_6d(global_matrices)
    oracle_pelvis_world = torch.from_numpy(
        np.asarray(source.joints[episode_indices][..., 0, :]).copy()
    ).to(device=device, dtype=torch.float32)
    oracle_pelvis_world = oracle_pelvis_world @ desired_rotation.T
    oracle_pelvis_world = oracle_pelvis_world - translation_shift[0, 0]

    point_windows, rotation_windows = [], []
    sampling_seconds = []
    denoiser_call_counts = []
    points = points_world
    generated_object_trans = object_trans_world
    generated_object_rotation = object_rotation.reshape(1, WINDOW_FRAMES, 9)
    generated_contact = contact

    sampler.begin_hsi_future_occ_episode()
    for step in range(int(episode["episode_num"])):
        if step == 0:
            mat_step = get_mat(cfg, points_world, 0)
            rotation_input = global_rotation_6d.reshape(1, WINDOW_FRAMES, 22, 6)
            root_matrix = transforms.rotation_6d_to_matrix(rotation_input[:, 0, 0]).reshape(1, 3, 3)
            root_axis = transforms.matrix_to_axis_angle(root_matrix).cpu().numpy()
            root_euler = Rotation.from_rotvec(root_axis).as_euler("zxy")
            shift_euler = np.zeros_like(root_euler)
            shift_euler[:, 2] = -root_euler[:, 2]
            shift_rotation = Rotation.from_euler("zxy", shift_euler).as_matrix()
            shifted_global = (
                torch.from_numpy(shift_rotation).to(device, dtype=torch.float32)[:, None, None]
                @ transforms.rotation_6d_to_matrix(rotation_input)
            )
            mat_step[:, :3, :3] = torch.from_numpy(np.linalg.inv(shift_rotation)).to(
                device, dtype=torch.float32
            )
            initial = points_world.reshape(1, WINDOW_FRAMES, 28, 3)[:, 0, 0]
            mat_step[:, 0, 3] = initial[:, 0]
            mat_step[:, 2, 3] = initial[:, 2]
            fixed_joints = points_world[:, :HISTORY_FRAMES]
            fixed_joints = dataset.normalize_torch(
                transform_points(fixed_joints, torch.inverse(mat_step))
            )
            fixed_object = dataset.normalize_torch(
                transform_points(object_trans_world[:, :HISTORY_FRAMES], torch.inverse(mat_step)),
                is_object=True,
            )
            fixed_rotation = object_rotation[:, :HISTORY_FRAMES].reshape(1, HISTORY_FRAMES, 9)
            fixed_contact = contact[:, :HISTORY_FRAMES]
            fixed_global = transforms.matrix_to_rotation_6d(shifted_global).reshape(
                1, WINDOW_FRAMES, 132
            )[:, :HISTORY_FRAMES]
            fixed_points = torch.cat(
                (fixed_joints, fixed_global, fixed_object, fixed_rotation, fixed_contact), dim=-1
            )
            pi = data["pi"]
        else:
            mat_step = get_mat(cfg, points, -HISTORY_FRAMES)
            rotation_input = generated_global.reshape(1, WINDOW_FRAMES, 22, 6)
            root_matrix = transforms.rotation_6d_to_matrix(
                rotation_input[:, -HISTORY_FRAMES, 0]
            ).reshape(1, 3, 3)
            root_axis = transforms.matrix_to_axis_angle(root_matrix).cpu().numpy()
            root_euler = Rotation.from_rotvec(root_axis).as_euler("zxy")
            shift_euler = np.zeros_like(root_euler)
            shift_euler[:, 2] = -root_euler[:, 2]
            shift_rotation = Rotation.from_euler("zxy", shift_euler).as_matrix()
            shifted_global = (
                torch.from_numpy(shift_rotation).to(device, dtype=torch.float32)[:, None, None]
                @ transforms.rotation_6d_to_matrix(rotation_input)
            )
            mat_step[:, :3, :3] = torch.from_numpy(np.linalg.inv(shift_rotation)).to(
                device, dtype=torch.float32
            )
            initial = points.reshape(1, WINDOW_FRAMES, 28, 3)[:, -HISTORY_FRAMES, 0]
            mat_step[:, 0, 3] = initial[:, 0]
            mat_step[:, 2, 3] = initial[:, 2]
            fixed_joints = dataset.normalize_torch(
                transform_points(points[:, -HISTORY_FRAMES:], torch.inverse(mat_step))
            )
            fixed_object = dataset.normalize_torch(
                transform_points(
                    generated_object_trans[:, -HISTORY_FRAMES:], torch.inverse(mat_step)
                ),
                is_object=True,
            )
            fixed_rotation = generated_object_rotation[:, -HISTORY_FRAMES:]
            fixed_contact = generated_contact[:, -HISTORY_FRAMES:]
            fixed_global = transforms.matrix_to_rotation_6d(shifted_global).reshape(
                1, WINDOW_FRAMES, 132
            )[:, -HISTORY_FRAMES:]
            fixed_points = torch.cat(
                (fixed_joints, fixed_global, fixed_object, fixed_rotation, fixed_contact), dim=-1
            )

        pi = torch.tensor(
            [step * (WINDOW_FRAMES - HISTORY_FRAMES) * DATA_STEP],
            device=device,
            dtype=torch.long,
        )
        end_pi = pi + WINDOW_FRAMES * DATA_STEP
        sequence_length = torch.tensor(
            [int(episode["episode_num"]) * WINDOW_STRIDE_RAW + HISTORY_FRAMES * DATA_STEP],
            device=device,
            dtype=torch.long,
        )
        human_dict = {
            "rest_human_offsets": rest_offsets,
            "betas": betas,
            "transl": torch.from_numpy(
                np.asarray(source.transl[source_start] - source.joints[source_start, 0]).copy()
            ).to(device, dtype=torch.float32),
            "gender": gender,
        }
        call_counter.reset()
        synchronize_cuda(device)
        begin = time.perf_counter()
        oracle_local = transform_points(
            oracle_pelvis_world[step, list(hsi_diagnostics.FUTURE_OCC_OFFSETS)].unsqueeze(0),
            torch.inverse(mat_step),
        )
        sampler.set_hsi_future_occ_oracle(oracle_local)
        try:
            generated = sample_step(
                cfg,
                step,
                mat_step,
                fixed_points,
                sampler,
                cond,
                trajectory,
                pi,
                end_pi,
                sequence_length,
                object_bps,
                object_points,
                {},
                {},
                seq_name_dict,
                object_rotation_ref,
                human_dict,
                desired_rotation @ mat_rotation_transpose,
            )
        except Exception:
            sampler.abort_hsi_future_occ_episode()
            raise
        finally:
            sampler.clear_hsi_future_occ_oracle()
        synchronize_cuda(device)
        sampling_seconds.append(time.perf_counter() - begin)
        denoiser_call_counts.append(call_counter.count)

        points = generated["points_orig"].clone()
        generated_object_trans = generated["obj_trans_orig"].clone()
        generated_object_rotation = generated["object_rot_mat"].clone()
        generated_contact = generated["contact_label"].clone()
        generated_global = generated["global_rot_6d"].clone()
        point_windows.append(points[0].reshape(WINDOW_FRAMES, 28, 3).cpu())
        rotation_windows.append(generated_global[0].reshape(WINDOW_FRAMES, 22, 6).cpu())

    future_occ_center_error = sampler.finish_hsi_future_occ_episode()
    coarse_points = hsi_metrics.stitch_windows(
        point_windows, history_frames=HISTORY_FRAMES, overlap_atol=2e-4
    )
    coarse_rotation = hsi_metrics.stitch_windows(
        rotation_windows, history_frames=HISTORY_FRAMES, overlap_atol=2e-4
    )
    interpolated_points = interpolate_joints(
        coarse_points.frames.reshape(-1, 84).to(device), scale=int(cfg.interp_s)
    ).reshape(-1, 28, 3)
    global_matrices = transforms.rotation_6d_to_matrix(
        coarse_rotation.frames.to(device).reshape(-1, 22, 6)
    )
    local_matrices = dataset.quat_ik_torch(global_matrices)
    local_quaternions = transforms.matrix_to_quaternion(local_matrices)
    local_quaternions = interp_jrot(local_quaternions, int(cfg.interp_s)).reshape(-1, 22, 4)
    local_axis = transforms.matrix_to_axis_angle(
        transforms.quaternion_to_matrix(local_quaternions)
    ).reshape(-1, 22, 3)
    translation_offset = torch.from_numpy(
        np.asarray(source.transl[source_start] - source.joints[source_start, 0]).copy()
    ).to(device, dtype=torch.float32)
    # No transform, anywhere on this path -- pose, translation and FK output all
    # live in the single y-up world the dataset now serves.  The rotation channel
    # carries global rotations of the y-up SMPL template (datasets/infbagel.py
    # applies a world change of basis to the root and nothing at all to the local
    # ``human_pose``), so ``quat_ik_torch`` above already yields exactly the
    # locals SMPL-X expects.  ``translation_offset`` is ``transl - pelvis``, which
    # is ``-J0(betas)`` in that same frame, so ``pelvis + offset`` is SMPL-X's own
    # ``transl`` and the returned pelvis lands back on ``interpolated_points[:, 0]``.
    #
    # The earlier revision of this block applied ``yup_to_zup`` to the pose to
    # invert the ``zup_to_yup(human_orient)``/``zup_to_yup(human_pose)``
    # conjugation the released dataset applied to LINGO.  That conjugation is gone
    # (it was a template rotation masquerading as a world change of basis, and it
    # left the LINGO rotation channel 90 deg about +x from its own joint channel),
    # so inverting it here would now be the error.  Measured after the change:
    # this chain reproduces ``human_joints_aligned.npy`` to 0.16 mm mean at
    # ``interp_s=1``; with ``yup_to_zup`` still applied it misses by ~0.6 m.
    smpl_translation = interpolated_points[:, 0] + translation_offset
    smpl_pose = local_axis
    vertices, metric_joints = _run_smplx_chunks(
        smpl_pose,
        smpl_translation,
        betas,
        gender,
        device,
        int(cfg.smplx_batch_size),
        smplx_cache,
    )
    if export_sink is not None:
        # Handles only -- no device-to-host copies here.  episode_seconds is timed
        # around this whole call, so materializing the export inside it would
        # inflate the end-to-end latency the caller reports.  The caller converts
        # after it has stopped its clock.
        #
        # coarse_points, not metric_joints: the network's own 28-slot joint channel
        # at the rollout rate, matching the real arm's dataset array slot for slot.
        # smpl_pose/smpl_translation are the literal FK inputs a few lines above,
        # so a rebuild reproduces these exact vertices.
        export_sink.update(
            {
                "joints_coarse": coarse_points,
                "smplx_pose": smpl_pose,
                "smplx_transl": smpl_translation,
                "betas": betas,
                "gender": gender,
                "smplx_output_transform": "identity",
                "interp_scale": int(cfg.interp_s),
                # The caption this rollout was actually conditioned on.  The caller
                # re-derives it independently and refuses a mismatch, so the
                # exported caption cannot drift from cond["raw_text"].
                "caption_from_cond": str(cond["raw_text"]),
            }
        )
    return (
        _upsampled_stitched(coarse_points, vertices, int(cfg.interp_s)),
        _upsampled_stitched(coarse_points, metric_joints, int(cfg.interp_s)),
        sequence_index,
        sampling_seconds,
        denoiser_call_counts,
        future_occ_center_error,
    )


def _exact_holdout_windows(
    dataset,
    source: GroundTruthSource,
    episode_dir,
    subset_path,
    expected_selected: int,
    expected_missing: int,
):
    episodes = _load_episodes(Path(episode_dir), None)
    window_counts = [int(episode[2]["episode_num"]) for episode in episodes]
    subset = _load_episode_subset(subset_path, episodes, window_counts)
    cohort_payload = json.loads(Path(subset["path"]).read_text(encoding="utf-8"))
    cohort_rows = {
        int(row["canonical_ordinal"]): row for row in cohort_payload["episodes"]
    }

    valid_indices = sorted(int(sample_idx) for dataset_id, sample_idx in dataset.indices if dataset_id == 1)
    lookup = {}
    for data_idx in valid_indices:
        key = (int(source.window_sequence[data_idx]), int(source.window_start[data_idx]))
        lookup.setdefault(key, data_idx)

    selected = []
    missing = []
    for ordinal in subset["canonical_ordinals"]:
        _scene_name, _scene_index, episode = episodes[int(ordinal)]
        sequence_index = int(source.window_sequence[int(episode["data_idx"])])
        start = int(source.window_start[int(episode["data_idx"])])
        cohort_row = cohort_rows[int(ordinal)]
        for window_index in range(int(episode["episode_num"])):
            key = (sequence_index, start + window_index * WINDOW_STRIDE_RAW)
            if key not in lookup:
                if window_index != int(episode["episode_num"]) - 1:
                    raise ValueError(
                        "non-terminal test window has no language item for source/start %s"
                        % (key,)
                    )
                missing.append(
                    {
                        "episode_id": str(cohort_row["sequence_id"]),
                        "window_index": window_index,
                        "source_sequence_index": sequence_index,
                        "source_start": key[1],
                    }
                )
                continue
            selected.append(
                {
                    "data_idx": lookup[key],
                    "episode_id": str(cohort_row["sequence_id"]),
                    "stratum": str(cohort_row["stratum"]),
                    "window_index": window_index,
                }
            )
    expected_selected = int(expected_selected)
    expected_missing = int(expected_missing)
    if len(selected) != expected_selected or len(missing) != expected_missing:
        raise ValueError(
            "teacher-forced holdout mapped %d exact windows and %d terminal padded "
            "windows, expected %d and %d"
            % (len(selected), len(missing), expected_selected, expected_missing)
        )
    stratum_weights = {}
    for row in cohort_payload["episodes"]:
        stratum_weights[str(row["stratum"])] = float(row["stratum_weight"])
    subset = dict(subset)
    subset["exact_valid_window_count"] = len(selected)
    subset["terminal_padded_window_count"] = len(missing)
    subset["terminal_padded_windows"] = missing
    return selected, subset, stratum_weights


def _teacher_forced_holdout_windows(cfg: DictConfig, dataset, source: GroundTruthSource):
    return _exact_holdout_windows(
        dataset,
        source,
        cfg.lingo_episode_dir,
        cfg.lingo_episode_subset,
        int(cfg.teacher_forced_holdout_windows),
        int(cfg.teacher_forced_terminal_padded_windows),
    )


def _teacher_forced_train_windows(dataset, count: int, seed: int):
    valid_indices = np.asarray(
        sorted(int(sample_idx) for dataset_id, sample_idx in dataset.indices if dataset_id == 1),
        dtype=np.int64,
    )
    if int(count) > len(valid_indices):
        raise ValueError("requested more teacher-forced train windows than are available")
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(valid_indices, size=int(count), replace=False))
    return [
        {
            "data_idx": int(data_idx),
            "episode_id": "train:%07d" % int(data_idx),
            "stratum": "train",
            "window_index": 0,
        }
        for data_idx in chosen
    ]


def _teacher_forced_smplx_joints(
    representation: torch.Tensor,
    dataset,
    mat: torch.Tensor,
    translation_offset: torch.Tensor,
    betas: torch.Tensor,
    gender: str,
    smplx_cache: MutableMapping[str, torch.nn.Module],
    smplx_batch_size: int,
    frame_count: int = 4,
) -> torch.Tensor:
    frame_count = int(frame_count)
    representation = representation[:, :frame_count].float()
    global_jpos = project_utils.transform_points(
        dataset.denormalize_torch(representation[:, :, :84]), mat
    ).reshape(1, frame_count, 28, 3)
    global_rotation = transforms.rotation_6d_to_matrix(
        representation[:, :, 84:216].reshape(-1, 22, 6)
    ).reshape(1, frame_count, 22, 3, 3)
    global_rotation = mat[:, None, None, :3, :3] @ global_rotation
    local_rotation = dataset.quat_ik_torch(global_rotation.reshape(-1, 22, 3, 3))
    local_axis = transforms.matrix_to_axis_angle(local_rotation).reshape(-1, 22, 3)
    smpl_translation = global_jpos[:, :, 0].reshape(-1, 3) + translation_offset
    _, joints = _run_smplx_chunks(
        local_axis,
        smpl_translation,
        betas,
        gender,
        representation.device,
        int(smplx_batch_size),
        smplx_cache,
    )
    return joints.reshape(1, frame_count, -1, 3)


def _diagnostic_window_inputs(cfg: DictConfig, dataset, selection: Mapping[str, object]):
    from torch.utils.data._utils.collate import default_collate

    device = torch.device(str(cfg.device))
    data_idx = int(selection["data_idx"])
    batch = default_collate([_lingo_item(dataset, data_idx)])

    def tensor(name, dtype=None):
        value = batch[name]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.to(device=device, dtype=dtype)

    joints = tensor("joints", torch.float32).reshape(1, WINDOW_FRAMES, 84)
    global_rotation = tensor("global_rot_6d", torch.float32).reshape(
        1, WINDOW_FRAMES, 132
    )
    object_trans = tensor("object_trans", torch.float32).reshape(1, WINDOW_FRAMES, 3)
    object_rotation = tensor("object_rot_mat", torch.float32).reshape(
        1, WINDOW_FRAMES, 9
    )
    contact = tensor("contact_label", torch.float32).reshape(1, WINDOW_FRAMES, 4)
    x_start = torch.cat(
        (joints, global_rotation, object_trans, object_rotation, contact), dim=-1
    )
    start_idx = int(dataset.lingo_dataset.start_ind[data_idx])
    translation_offset = torch.from_numpy(
        np.asarray(
            dataset.lingo_dataset.transl[start_idx]
            - dataset.lingo_dataset.joints[start_idx, 0]
        ).copy()
    ).to(device=device, dtype=torch.float32)
    return {
        "data_idx": data_idx,
        "x_start": x_start,
        "joints": joints,
        "mat": tensor("mat", torch.float32).reshape(1, 4, 4),
        "scene_flag": tensor("scene_flag", torch.long).reshape(1),
        "text_emb": tensor("text_clip_embedding", torch.float32),
        "pelvis_goal": tensor("pelvis_goal", torch.float32).reshape(1, 3),
        "scene_goal": tensor("scene_goal", torch.float32).reshape(1, 3),
        "object_goal": tensor("object_goal", torch.float32).reshape(1, 3),
        "need_scene": tensor("need_scene", torch.bool).reshape(1),
        "need_pelvis_dir": tensor("need_pelvis_dir", torch.bool).reshape(1),
        "pi": tensor("pi", torch.long).reshape(1),
        "end_pi": tensor("end_pi", torch.long).reshape(1),
        "seq_length": tensor("seg_len", torch.long).reshape(1),
        "need_pi": tensor("need_pi", torch.bool).reshape(1),
        "is_loco": tensor("is_loco", torch.bool).reshape(1),
        "is_object": tensor("is_object", torch.bool).reshape(1),
        "object_bps": tensor("obj_bps_data", torch.float32),
        "object_points": tensor("object_points", torch.float32).reshape(1, -1, 3),
        "object_rotation_ref": tensor("obj_rot_mat_ref", torch.float32).reshape(1, 3, 3),
        "rest_offsets": tensor("rest_human_offsets", torch.float32).reshape(1, 24, 3),
        "betas": tensor("betas", torch.float32).reshape(-1),
        "gender": str(batch["gender"][0]),
        "seq_name_dict": {0: str(batch["seq_name"][0])},
        "translation_offset": translation_offset,
    }


def _diagnostic_model_args(inputs):
    return (
        inputs["text_emb"],
        inputs["pelvis_goal"],
        inputs["scene_goal"],
        inputs["is_loco"],
        inputs["need_scene"],
        inputs["need_pelvis_dir"],
        inputs["pi"],
        inputs["end_pi"],
        inputs["seq_length"],
        inputs["need_pi"],
        inputs["object_goal"],
        inputs["is_object"],
        inputs["object_bps"],
    )


def _teacher_forced_window_record(
    cfg: DictConfig,
    dataset,
    sampler,
    selection: Mapping[str, object],
    timesteps: Sequence[int],
    smplx_cache: MutableMapping[str, torch.nn.Module],
) -> Dict[str, object]:
    from torch.utils.data._utils.collate import default_collate

    device = torch.device(str(cfg.device))
    data_idx = int(selection["data_idx"])
    data = _lingo_item(dataset, data_idx)
    batch = default_collate([data])

    def tensor(name, dtype=None):
        value = batch[name]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.to(device=device, dtype=dtype)

    joints = tensor("joints", torch.float32).reshape(1, WINDOW_FRAMES, 84)
    global_rotation = tensor("global_rot_6d", torch.float32).reshape(1, WINDOW_FRAMES, 132)
    object_trans = tensor("object_trans", torch.float32).reshape(1, WINDOW_FRAMES, 3)
    object_rotation = tensor("object_rot_mat", torch.float32).reshape(1, WINDOW_FRAMES, 9)
    contact = tensor("contact_label", torch.float32).reshape(1, WINDOW_FRAMES, 4)
    x_start = torch.cat((joints, global_rotation, object_trans, object_rotation, contact), dim=-1)
    mat = tensor("mat", torch.float32).reshape(1, 4, 4)
    scene_flag = tensor("scene_flag", torch.long).reshape(1)
    text_emb = tensor("text_clip_embedding", torch.float32)
    pelvis_goal = tensor("pelvis_goal", torch.float32).reshape(1, 3)
    scene_goal = tensor("scene_goal", torch.float32).reshape(1, 3)
    object_goal = tensor("object_goal", torch.float32).reshape(1, 3)
    need_scene = tensor("need_scene", torch.bool).reshape(1)
    need_pelvis_dir = tensor("need_pelvis_dir", torch.bool).reshape(1)
    pi = tensor("pi", torch.long).reshape(1)
    end_pi = tensor("end_pi", torch.long).reshape(1)
    seq_length = tensor("seg_len", torch.long).reshape(1)
    need_pi = tensor("need_pi", torch.bool).reshape(1)
    is_loco = tensor("is_loco", torch.bool).reshape(1)
    is_object = tensor("is_object", torch.bool).reshape(1)
    object_bps = tensor("obj_bps_data", torch.float32)
    object_points = tensor("object_points", torch.float32).reshape(1, -1, 3)
    object_rotation_ref = tensor("obj_rot_mat_ref", torch.float32).reshape(1, 3, 3)
    betas = tensor("betas", torch.float32).reshape(-1)
    gender = str(batch["gender"][0])

    start_idx = int(dataset.lingo_dataset.start_ind[data_idx])
    translation_offset = torch.from_numpy(
        np.asarray(
            dataset.lingo_dataset.transl[start_idx]
            - dataset.lingo_dataset.joints[start_idx, 0]
        ).copy()
    ).to(device=device, dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed) + data_idx)
    noise = torch.randn(x_start.shape, generator=generator, dtype=torch.float32).to(device)
    noise[:, :HISTORY_FRAMES] = 0.0
    target_joints = _teacher_forced_smplx_joints(
        x_start,
        dataset,
        mat,
        translation_offset,
        betas,
        gender,
        smplx_cache,
        int(cfg.smplx_batch_size),
    )

    metrics = {}
    with torch.no_grad():
        for timestep in timesteps:
            t = torch.full((1,), int(timestep), device=device, dtype=torch.long)
            x_noisy = sampler.q_sample(x_start=x_start, t=t, noise=noise)
            x_noisy[:, :HISTORY_FRAMES] = x_start[:, :HISTORY_FRAMES]
            occ, occ_list, occ_pos = sampler._compute_occ(
                x_noisy,
                x_start,
                joints,
                mat,
                scene_flag,
                object_points,
                pelvis_goal,
                scene_goal,
                object_goal,
                is_loco,
                is_object,
                need_pelvis_dir,
                object_rotation_ref,
            )
            predicted = sampler.student_model(
                x_noisy,
                occ,
                t,
                text_emb,
                pelvis_goal,
                scene_goal,
                is_loco,
                need_scene,
                need_pelvis_dir,
                pi,
                end_pi,
                seq_length,
                need_pi,
                object_goal,
                is_object,
                object_bps,
                occ_list,
                occ_pos,
            )
            predicted_joints = _teacher_forced_smplx_joints(
                predicted,
                dataset,
                mat,
                translation_offset,
                betas,
                gender,
                smplx_cache,
                int(cfg.smplx_batch_size),
            )
            values = hsi_diagnostics.teacher_forced_boundary_metrics(
                predicted_joints,
                target_joints,
                predicted,
                x_start,
                fps=float(cfg.fps) / float(dataset.lingo_dataset.step),
            )
            metrics[str(int(timestep))] = {
                name: float(value[0].cpu()) for name, value in values.items()
            }

    return {
        "episode_id": str(selection["episode_id"]),
        "stratum": str(selection["stratum"]),
        "window_index": int(selection["window_index"]),
        "data_idx": data_idx,
        "metrics": metrics,
    }


def _evaluate_teacher_forced_cohort(
    cfg: DictConfig,
    dataset,
    model: torch.nn.Module,
    selections: Sequence[Mapping[str, object]],
    split_partition: str,
    timesteps: Sequence[int],
    smplx_cache: MutableMapping[str, torch.nn.Module],
):
    sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler.set_dataset_and_model(dataset, model)
    records = []
    for position, selection in enumerate(selections, start=1):
        records.append(
            _teacher_forced_window_record(
                cfg, dataset, sampler, selection, timesteps, smplx_cache
            )
        )
        if position % 25 == 0 or position == len(selections):
            print(
                "TEACHER_FORCED %s %d/%d" % (split_partition, position, len(selections)),
                flush=True,
            )
    return records


def evaluate_teacher_forced_boundary(cfg: DictConfig) -> Path:
    if int(cfg.batch_size) != 1:
        raise ValueError("teacher-forced boundary diagnostic requires batch_size=1")
    timesteps = tuple(int(value) for value in cfg.teacher_forced_timesteps)
    if timesteps != hsi_diagnostics.TEACHER_FORCED_TIMESTEPS:
        raise ValueError(
            "teacher_forced_timesteps must be %s" % (hsi_diagnostics.TEACHER_FORCED_TIMESTEPS,)
        )

    model, checkpoint_provenance = _load_strict_checkpoint(cfg)
    source = GroundTruthSource(DATASET_ROOT)
    smplx_cache: Dict[str, torch.nn.Module] = {}

    holdout_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    holdout_cfg.dataset.split_partition = "test"
    holdout_dataset = _scene_only_dataset(holdout_cfg)
    holdout_selection, subset, stratum_weights = _teacher_forced_holdout_windows(
        holdout_cfg, holdout_dataset, source
    )
    holdout_records = _evaluate_teacher_forced_cohort(
        holdout_cfg,
        holdout_dataset,
        model,
        holdout_selection,
        "test",
        timesteps,
        smplx_cache,
    )
    del holdout_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    train_cfg.dataset.split_partition = "train"
    train_dataset = _scene_only_dataset(train_cfg)
    train_selection = _teacher_forced_train_windows(
        train_dataset, int(cfg.teacher_forced_train_windows), int(cfg.seed)
    )
    train_records = _evaluate_teacher_forced_cohort(
        train_cfg,
        train_dataset,
        model,
        train_selection,
        "train",
        timesteps,
        smplx_cache,
    )
    del train_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = hsi_diagnostics.summarize_teacher_forced_boundary(
        holdout_records,
        train_records,
        stratum_weights,
        timesteps=timesteps,
        seed=int(cfg.seed),
        replicates=int(cfg.teacher_forced_bootstrap_replicates),
    )

    selection_bytes = json.dumps(
        [item["data_idx"] for item in train_selection], separators=(",", ":")
    ).encode("utf-8")
    payload = {
        "schema_version": 1,
        "mode": "teacher_forced_boundary",
        "checkpoint": checkpoint_provenance,
        "seed": int(cfg.seed),
        "timesteps": list(timesteps),
        "protocol": {
            "model_mode": "eval_fp32",
            "history_frames": HISTORY_FRAMES,
            "window_fps": float(cfg.fps) / float(DATA_STEP),
            "future_occ_jitter_scale": float(cfg.hsi_future_occ_jitter_scale),
            "cfg": False,
            "guidance": False,
            "rollout": False,
        },
        "holdout_selection": subset,
        "stratum_weights": stratum_weights,
        "train_selection": {
            "window_count": len(train_selection),
            "rule": "default_rng(seed).choice(sorted_valid_underlying_indices, replace=False)",
            "data_index_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        },
        "summary": summary,
        "records": {"holdout": holdout_records, "train": train_records},
    }
    output_dir = Path(cfg.lingo_output_dir) / (
        "%s-%s" % (Path(str(cfg.ckpt_path)).stem, checkpoint_provenance["checkpoint_sha256"][:12])
    )
    return _write_payload(output_dir, payload)


def _predictor_decomp_window_record(
    cfg,
    dataset,
    sampler,
    selection,
    smplx_cache,
):
    inputs = _diagnostic_window_inputs(cfg, dataset, selection)
    timestep = int(cfg.predictor_decomp_timestep)
    t = torch.full((1,), timestep, device=str(cfg.device), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(
        int(cfg.seed) + inputs["data_idx"]
    )
    noise = torch.randn(
        inputs["x_start"].shape, generator=generator, dtype=torch.float32
    ).to(str(cfg.device))
    noise[:, :HISTORY_FRAMES] = 0.0
    x_noisy = sampler.q_sample(inputs["x_start"], t, noise)
    x_noisy[:, :HISTORY_FRAMES] = inputs["x_start"][:, :HISTORY_FRAMES]
    occ, occ_list, occ_pos = sampler._compute_occ(
        x_noisy,
        inputs["x_start"],
        inputs["joints"],
        inputs["mat"],
        inputs["scene_flag"],
        inputs["object_points"],
        inputs["pelvis_goal"],
        inputs["scene_goal"],
        inputs["object_goal"],
        inputs["is_loco"],
        inputs["is_object"],
        inputs["need_pelvis_dir"],
        inputs["object_rotation_ref"],
    )
    model_args = _diagnostic_model_args(inputs)
    with torch.no_grad():
        conditional = sampler.student_model(
            x_noisy, occ, t, *model_args, occ_list, occ_pos, is_sample=True
        )
        unconditional = sampler.student_model(
            x_noisy,
            occ,
            t,
            *model_args,
            occ_list,
            occ_pos,
            is_sample=True,
            is_uncondition=True,
        )
        zero_velocity_input = x_noisy.clone()
        zero_velocity_input[:, 1, :84] = zero_velocity_input[:, 0, :84]
        zero_velocity = sampler.student_model(
            zero_velocity_input,
            occ,
            t,
            *model_args,
            occ_list,
            occ_pos,
            is_sample=True,
        )

    predictions = {
        "conditional": conditional,
        "unconditional": unconditional,
        "cfg_w0": conditional,
        "cfg_w0.5": conditional + 0.5 * (conditional - unconditional),
        "cfg_w1": conditional + (conditional - unconditional),
        "zero_velocity_history": zero_velocity,
    }
    target_joints = _teacher_forced_smplx_joints(
        inputs["x_start"],
        dataset,
        inputs["mat"],
        inputs["translation_offset"],
        inputs["betas"],
        inputs["gender"],
        smplx_cache,
        int(cfg.smplx_batch_size),
        frame_count=6,
    )
    predicted_joints = {
        name: _teacher_forced_smplx_joints(
            value,
            dataset,
            inputs["mat"],
            inputs["translation_offset"],
            inputs["betas"],
            inputs["gender"],
            smplx_cache,
            int(cfg.smplx_batch_size),
            frame_count=6,
        )
        for name, value in predictions.items()
    }
    positions = {
        name: dataset.denormalize_torch(value[:, :, :84])
        for name, value in predictions.items()
    }
    target_positions = dataset.denormalize_torch(inputs["x_start"][:, :, :84])
    values = hsi_diagnostics.predictor_decomp_metrics(
        predicted_joints,
        target_joints,
        positions,
        target_positions,
        fps=float(cfg.fps) / float(DATA_STEP),
    )
    record = {
        "episode_id": str(selection["episode_id"]),
        "stratum": str(selection["stratum"]),
        "window_index": int(selection["window_index"]),
        "data_idx": inputs["data_idx"],
        "metrics": {name: float(value[0].cpu()) for name, value in values.items()},
    }
    arrays = {
        name: value[0].detach().to(torch.float32).cpu().numpy()
        for name, value in predictions.items()
    }
    return record, arrays


def _evaluate_predictor_decomp_cohort(
    cfg, dataset, model, selections, partition, smplx_cache
):
    sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler.set_dataset_and_model(dataset, model)
    records = []
    arrays = {name: [] for name in hsi_diagnostics.PREDICTOR_DECOMP_ARMS}
    for position, selection in enumerate(selections, start=1):
        record, window_arrays = _predictor_decomp_window_record(
            cfg, dataset, sampler, selection, smplx_cache
        )
        record["array_row"] = len(records)
        records.append(record)
        for name in arrays:
            arrays[name].append(window_arrays[name])
        if position % 25 == 0 or position == len(selections):
            print("PREDICTOR_DECOMP %s %d/%d" % (partition, position, len(selections)), flush=True)
    return records, {name: np.stack(values) for name, values in arrays.items()}


def evaluate_predictor_decomp(cfg: DictConfig) -> Path:
    if int(cfg.batch_size) != 1 or int(cfg.predictor_decomp_timestep) != 498:
        raise ValueError("predictor decomposition requires batch_size=1 and timestep=498")
    model, checkpoint_provenance = _load_strict_checkpoint(cfg)
    source = GroundTruthSource(DATASET_ROOT)
    smplx_cache: Dict[str, torch.nn.Module] = {}

    holdout_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    holdout_cfg.dataset.split_partition = "test"
    holdout_dataset = _scene_only_dataset(holdout_cfg)
    holdout_selection, subset, stratum_weights = _exact_holdout_windows(
        holdout_dataset,
        source,
        cfg.lingo_episode_dir,
        cfg.lingo_episode_subset,
        int(cfg.predictor_decomp_holdout_windows),
        int(cfg.predictor_decomp_terminal_padded_windows),
    )
    holdout_records, holdout_arrays = _evaluate_predictor_decomp_cohort(
        holdout_cfg,
        holdout_dataset,
        model,
        holdout_selection,
        "test",
        smplx_cache,
    )
    del holdout_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    train_cfg.dataset.split_partition = "train"
    train_dataset = _scene_only_dataset(train_cfg)
    train_selection = _teacher_forced_train_windows(
        train_dataset, int(cfg.predictor_decomp_train_windows), int(cfg.seed)
    )
    train_records, train_arrays = _evaluate_predictor_decomp_cohort(
        train_cfg, train_dataset, model, train_selection, "train", smplx_cache
    )
    summary = hsi_diagnostics.summarize_predictor_decomp(
        holdout_records,
        train_records,
        stratum_weights,
        seed=int(cfg.seed),
        replicates=int(cfg.predictor_decomp_bootstrap_replicates),
    )
    arrays = {
        name: np.concatenate((holdout_arrays[name], train_arrays[name]), axis=0)
        for name in hsi_diagnostics.PREDICTOR_DECOMP_ARMS
    }
    payload = {
        "schema_version": 1,
        "mode": "predictor_decomp",
        "checkpoint": checkpoint_provenance,
        "seed": int(cfg.seed),
        "timestep": 498,
        "protocol": {
            "model_mode": "eval_fp32",
            "all_model_forwards_is_sample": True,
            "conditional_forwards": 2,
            "cfg_combination": "offline conditional + w*(conditional-unconditional)",
            "guidance": False,
            "rollout": False,
        },
        "holdout_selection": subset,
        "stratum_weights": stratum_weights,
        "array_row_order": {
            "holdout_rows": [record["data_idx"] for record in holdout_records],
            "train_rows": [record["data_idx"] for record in train_records],
        },
        "summary": summary,
        "records": {"holdout": holdout_records, "train": train_records},
    }
    output_dir = Path(cfg.lingo_output_dir) / (
        "%s-%s" % (Path(str(cfg.ckpt_path)).stem, checkpoint_provenance["checkpoint_sha256"][:12])
    )
    return _write_array_payload(
        output_dir, payload, arrays, file_name="predictor_decomp_xhat0.npz"
    )


def _single_window_chain_record(cfg, dataset, sampler, selection, smplx_cache):
    from test_infbagel_hosi import seed_everything

    inputs = _diagnostic_window_inputs(cfg, dataset, selection)
    seed_everything(int(cfg.seed) + inputs["data_idx"])
    human_dict = {
        "rest_human_offsets": inputs["rest_offsets"][:, None].repeat(
            1, WINDOW_FRAMES, 1, 1
        ),
        "betas": inputs["betas"],
        "transl": inputs["translation_offset"],
        "gender": inputs["gender"],
    }
    sampler.begin_p_sample_trace(int(cfg.single_window_chain_trace_timestep))
    try:
        samples, _ = sampler.p_sample_loop(
            inputs["x_start"][:, :HISTORY_FRAMES],
            inputs["mat"],
            inputs["scene_flag"],
            inputs["text_emb"],
            inputs["pelvis_goal"],
            inputs["scene_goal"],
            inputs["object_goal"],
            inputs["need_scene"],
            inputs["need_pelvis_dir"],
            inputs["pi"],
            inputs["end_pi"],
            inputs["seq_length"],
            inputs["need_pi"],
            inputs["is_loco"],
            inputs["is_object"],
            inputs["object_bps"],
            inputs["object_points"],
            inputs["object_rotation_ref"],
            {},
            {},
            inputs["seq_name_dict"],
            human_dict,
            None,
            float(cfg.guidance_weight),
        )
        trace = sampler.consume_p_sample_trace()
    except Exception:
        sampler.abort_p_sample_trace()
        raise
    final = samples[-1]
    target_joints = _teacher_forced_smplx_joints(
        inputs["x_start"],
        dataset,
        inputs["mat"],
        inputs["translation_offset"],
        inputs["betas"],
        inputs["gender"],
        smplx_cache,
        int(cfg.smplx_batch_size),
    )
    trace_joints = _teacher_forced_smplx_joints(
        trace,
        dataset,
        inputs["mat"],
        inputs["translation_offset"],
        inputs["betas"],
        inputs["gender"],
        smplx_cache,
        int(cfg.smplx_batch_size),
    )
    final_joints = _teacher_forced_smplx_joints(
        final,
        dataset,
        inputs["mat"],
        inputs["translation_offset"],
        inputs["betas"],
        inputs["gender"],
        smplx_cache,
        int(cfg.smplx_batch_size),
    )
    values = hsi_diagnostics.single_window_chain_metrics(
        trace_joints,
        final_joints,
        target_joints,
        fps=float(cfg.fps) / float(DATA_STEP),
    )
    return (
        {
            "episode_id": str(selection["episode_id"]),
            "stratum": str(selection["stratum"]),
            "window_index": int(selection["window_index"]),
            "data_idx": inputs["data_idx"],
            "metrics": {name: float(value[0].cpu()) for name, value in values.items()},
        },
        trace[0].detach().to(torch.float32).cpu().numpy(),
        final[0].detach().to(torch.float32).cpu().numpy(),
    )


def evaluate_single_window_chain(cfg: DictConfig) -> Path:
    if int(cfg.batch_size) != 1 or int(cfg.single_window_chain_trace_timestep) != 498:
        raise ValueError("single-window chain requires batch_size=1 and trace timestep=498")
    model, checkpoint_provenance = _load_strict_checkpoint(cfg)
    source = GroundTruthSource(DATASET_ROOT)
    dataset = _scene_only_dataset(cfg)
    selection, subset, stratum_weights = _exact_holdout_windows(
        dataset,
        source,
        cfg.lingo_episode_dir,
        cfg.lingo_episode_subset,
        int(cfg.single_window_chain_holdout_windows),
        int(cfg.single_window_chain_terminal_padded_windows),
    )
    sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler.set_dataset_and_model(dataset, model)
    smplx_cache: Dict[str, torch.nn.Module] = {}
    records, traces, finals = [], [], []
    for position, item in enumerate(selection, start=1):
        record, trace, final = _single_window_chain_record(
            cfg, dataset, sampler, item, smplx_cache
        )
        record["array_row"] = len(records)
        records.append(record)
        traces.append(trace)
        finals.append(final)
        if position % 10 == 0 or position == len(selection):
            print("SINGLE_WINDOW_CHAIN %d/%d" % (position, len(selection)), flush=True)
    summary = hsi_diagnostics.summarize_single_window_chain(
        records,
        stratum_weights,
        seed=int(cfg.seed),
        replicates=int(cfg.single_window_chain_bootstrap_replicates),
    )
    payload = {
        "schema_version": 1,
        "mode": "single_window_chain",
        "checkpoint": checkpoint_provenance,
        "seed": int(cfg.seed),
        "w": float(cfg.w),
        "protocol": {
            "sampler": "Sampler.p_sample_loop",
            "timesteps": int(sampler.timesteps),
            "history_frames": HISTORY_FRAMES,
            "trace_timestep": int(cfg.single_window_chain_trace_timestep),
            "future_occupancy": "Sampler._compute_occ_sample",
            "guidance": False,
        },
        "holdout_selection": subset,
        "stratum_weights": stratum_weights,
        "summary": summary,
        "records": records,
    }
    output_dir = Path(cfg.lingo_output_dir) / (
        "%s-%s-w%s"
        % (
            Path(str(cfg.ckpt_path)).stem,
            checkpoint_provenance["checkpoint_sha256"][:12],
            str(cfg.w).replace(".", "p"),
        )
    )
    return _write_array_payload(
        output_dir,
        payload,
        {
            "t498_model_output": np.stack(traces),
            "final_sample": np.stack(finals),
        },
        file_name="single_window_chain_samples.npz",
    )


def _chain_rebase_record(
    cfg,
    dataset,
    sampler,
    selection,
    c0_final,
    smplx_cache,
):
    from test_infbagel_hosi import seed_everything

    inputs = _diagnostic_window_inputs(cfg, dataset, selection)
    seed_everything(int(cfg.seed) + inputs["data_idx"])
    human_dict = {
        "rest_human_offsets": inputs["rest_offsets"][:, None].repeat(
            1, WINDOW_FRAMES, 1, 1
        ),
        "betas": inputs["betas"],
        "transl": inputs["translation_offset"],
        "gender": inputs["gender"],
    }
    if str(cfg.hsi_chain_rebase_mode) == "c2":
        sampler.set_hsi_chain_rebase_oracle(inputs["x_start"][:, 2, :84])
    try:
        samples, _ = sampler.p_sample_loop(
            inputs["x_start"][:, :HISTORY_FRAMES],
            inputs["mat"],
            inputs["scene_flag"],
            inputs["text_emb"],
            inputs["pelvis_goal"],
            inputs["scene_goal"],
            inputs["object_goal"],
            inputs["need_scene"],
            inputs["need_pelvis_dir"],
            inputs["pi"],
            inputs["end_pi"],
            inputs["seq_length"],
            inputs["need_pi"],
            inputs["is_loco"],
            inputs["is_object"],
            inputs["object_bps"],
            inputs["object_points"],
            inputs["object_rotation_ref"],
            {},
            {},
            inputs["seq_name_dict"],
            human_dict,
            None,
            float(cfg.guidance_weight),
        )
    finally:
        sampler.clear_hsi_chain_rebase_oracle()
    final = samples[-1]
    target_joints = _teacher_forced_smplx_joints(
        inputs["x_start"],
        dataset,
        inputs["mat"],
        inputs["translation_offset"],
        inputs["betas"],
        inputs["gender"],
        smplx_cache,
        int(cfg.smplx_batch_size),
        frame_count=10,
    )

    def joints(value):
        return _teacher_forced_smplx_joints(
            value,
            dataset,
            inputs["mat"],
            inputs["translation_offset"],
            inputs["betas"],
            inputs["gender"],
            smplx_cache,
            int(cfg.smplx_batch_size),
            frame_count=10,
        )

    fps = float(cfg.fps) / float(DATA_STEP)
    candidate_values = hsi_diagnostics.chain_rebase_metrics(
        joints(final), target_joints, fps=fps
    )
    c0_values = hsi_diagnostics.chain_rebase_metrics(
        joints(c0_final), target_joints, fps=fps
    )

    def record(values):
        return {
            "episode_id": str(selection["episode_id"]),
            "stratum": str(selection["stratum"]),
            "window_index": int(selection["window_index"]),
            "data_idx": inputs["data_idx"],
            "metrics": {name: float(value[0].cpu()) for name, value in values.items()},
        }

    return record(c0_values), record(candidate_values), final[0].float().cpu().numpy()


def evaluate_chain_rebase(cfg: DictConfig) -> Path:
    if int(cfg.batch_size) != 1:
        raise ValueError("D4-B chain rebase requires batch_size=1")
    arm = str(cfg.hsi_chain_rebase_mode)
    if arm not in ("c1", "c2", "c3"):
        raise ValueError("D4-B reportable arm must be c1, c2, or c3")

    c0_path = Path(str(cfg.d4_chain_c0_payload))
    c0_payload = json.loads(c0_path.read_text(encoding="utf-8"))
    c0_npz = np.load(Path(c0_payload["array_archive"]["path"]), allow_pickle=False)
    c0_source = c0_payload["records"]

    model, checkpoint_provenance = _load_strict_checkpoint(cfg)
    source = GroundTruthSource(DATASET_ROOT)
    dataset = _scene_only_dataset(cfg)
    selection, subset, stratum_weights = _exact_holdout_windows(
        dataset,
        source,
        cfg.lingo_episode_dir,
        cfg.lingo_episode_subset,
        int(cfg.d4_chain_holdout_windows),
        int(cfg.d4_chain_terminal_padded_windows),
    )
    if [int(row["data_idx"]) for row in selection] != [
        int(row["data_idx"]) for row in c0_source
    ]:
        raise ValueError("D4-B selection is not aligned with sealed D3 c0")
    if cfg.get("d4_chain_smoke_windows_per_stratum") is not None:
        limit = int(cfg.d4_chain_smoke_windows_per_stratum)
        selected, selected_c0, counts = [], [], defaultdict(int)
        for item, c0_row in zip(selection, c0_source):
            stratum = str(item["stratum"])
            if counts[stratum] < limit:
                selected.append(item)
                selected_c0.append(c0_row)
                counts[stratum] += 1
        selection, c0_source = selected, selected_c0

    sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler.set_dataset_and_model(dataset, model)
    smplx_cache: Dict[str, torch.nn.Module] = {}
    c0_records, arm_records, finals = [], [], []
    for position, (item, c0_row) in enumerate(zip(selection, c0_source), start=1):
        c0_final = torch.from_numpy(
            c0_npz["final_sample"][int(c0_row["array_row"])]
        ).to(device=str(cfg.device), dtype=torch.float32)[None]
        c0_record, arm_record, final = _chain_rebase_record(
            cfg, dataset, sampler, item, c0_final, smplx_cache
        )
        arm_record["array_row"] = len(arm_records)
        c0_records.append(c0_record)
        arm_records.append(arm_record)
        finals.append(final)
        if position % 10 == 0 or position == len(selection):
            print("CHAIN_REBASE %s %d/%d" % (arm, position, len(selection)), flush=True)

    summary = hsi_diagnostics.summarize_chain_rebase(
        c0_records,
        arm_records,
        stratum_weights,
        arm=arm,
        seed=int(cfg.seed),
        replicates=int(cfg.d4_chain_bootstrap_replicates),
    )
    payload = {
        "schema_version": 1,
        "mode": "chain_rebase",
        "arm": arm,
        "checkpoint": checkpoint_provenance,
        "seed": int(cfg.seed),
        "w": float(cfg.w),
        "protocol": {
            "sampler": "Sampler.p_sample_loop",
            "timesteps": int(sampler.timesteps),
            "history_frames": HISTORY_FRAMES,
            "rebase_location": "after_cfg_and_object_zero_before_posterior_mean",
            "rebase_base": "fixed_x_history",
            "guidance": False,
        },
        "c0_input": {
            "payload": str(c0_path),
            "array_sha256": c0_payload["array_archive"]["sha256"],
        },
        "holdout_selection": subset,
        "smoke_windows_per_stratum": (
            None
            if cfg.get("d4_chain_smoke_windows_per_stratum") is None
            else int(cfg.d4_chain_smoke_windows_per_stratum)
        ),
        "stratum_weights": stratum_weights,
        "summary": summary,
        "records": {"c0": c0_records, arm: arm_records},
    }
    output_dir = Path(cfg.lingo_output_dir) / (
        "%s-%s-%s"
        % (
            Path(str(cfg.ckpt_path)).stem,
            checkpoint_provenance["checkpoint_sha256"][:12],
            arm,
        )
    )
    return _write_array_payload(
        output_dir,
        payload,
        {"final_sample": np.stack(finals)},
        file_name="chain_rebase_%s_samples.npz" % arm,
    )


def _d4_decompose_arm(cfg, dataset, inputs, predicted, target_joints, smplx_cache):
    target = inputs["x_start"]
    root_only = target.clone()
    root_only[:, :, :84] = predicted[:, :, :84]
    pose_only = target.clone()
    pose_only[:, :, 84:216] = predicted[:, :, 84:216]

    def joints(value):
        return _teacher_forced_smplx_joints(
            value,
            dataset,
            inputs["mat"],
            inputs["translation_offset"],
            inputs["betas"],
            inputs["gender"],
            smplx_cache,
            int(cfg.smplx_batch_size),
        )

    return hsi_diagnostics.d4_offline_decomp_metrics(
        target_joints,
        joints(predicted),
        joints(root_only),
        joints(pose_only),
        dataset.denormalize_torch(predicted[:, :, :84]),
        dataset.denormalize_torch(target[:, :, :84]),
        fps=float(cfg.fps) / float(DATA_STEP),
    )


def evaluate_d4_offline_decomp(cfg: DictConfig) -> Path:
    d2_path = Path(str(cfg.d4_d2_payload))
    d3_path = Path(str(cfg.d4_d3_payload))
    d2 = json.loads(d2_path.read_text(encoding="utf-8"))
    d3 = json.loads(d3_path.read_text(encoding="utf-8"))
    d2_npz = np.load(Path(d2["array_archive"]["path"]), allow_pickle=False)
    d3_npz = np.load(Path(d3["array_archive"]["path"]), allow_pickle=False)
    holdout_source = d2["records"]["holdout"]
    train_source = d2["records"]["train"]
    d3_source = d3["records"]
    if [row["data_idx"] for row in holdout_source] != [row["data_idx"] for row in d3_source]:
        raise ValueError("D2 holdout and D3 rows are not aligned")
    if cfg.get("d4_smoke_windows_per_stratum") is not None:
        limit = int(cfg.d4_smoke_windows_per_stratum)
        selected_rows = []
        counts = defaultdict(int)
        for row in holdout_source:
            stratum = str(row["stratum"])
            if counts[stratum] < limit:
                selected_rows.append(row)
                counts[stratum] += 1
        selected_indices = [int(row["array_row"]) for row in selected_rows]
        holdout_source = selected_rows
        d3_source = [d3_source[index] for index in selected_indices]
    if cfg.get("d4_train_limit") is not None:
        train_source = train_source[: int(cfg.d4_train_limit)]

    smplx_cache: Dict[str, torch.nn.Module] = {}

    def evaluate_partition(dataset, source_records, arrays_by_arm, partition):
        records = []
        for position, source_record in enumerate(source_records, start=1):
            inputs = _diagnostic_window_inputs(cfg, dataset, source_record)
            target_joints = _teacher_forced_smplx_joints(
                inputs["x_start"],
                dataset,
                inputs["mat"],
                inputs["translation_offset"],
                inputs["betas"],
                inputs["gender"],
                smplx_cache,
                int(cfg.smplx_batch_size),
            )
            metrics = {}
            for arm, array in arrays_by_arm.items():
                predicted = torch.from_numpy(array[int(source_record["array_row"])]).to(
                    device=torch.device(str(cfg.device)), dtype=torch.float32
                )[None]
                arm_values = _d4_decompose_arm(
                    cfg, dataset, inputs, predicted, target_joints, smplx_cache
                )
                metrics.update(
                    {arm + "_" + name: float(value[0].cpu()) for name, value in arm_values.items()}
                )
            records.append(
                {
                    "episode_id": str(source_record["episode_id"]),
                    "stratum": str(source_record["stratum"]),
                    "window_index": int(source_record["window_index"]),
                    "data_idx": int(source_record["data_idx"]),
                    "metrics": metrics,
                }
            )
            if position % 25 == 0 or position == len(source_records):
                print("D4_OFFLINE %s %d/%d" % (partition, position, len(source_records)), flush=True)
        return records

    holdout_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    holdout_cfg.dataset.split_partition = "test"
    holdout_dataset = _scene_only_dataset(holdout_cfg)
    holdout_records = evaluate_partition(
        holdout_dataset,
        holdout_source,
        {
            "d2_conditional": d2_npz["conditional"],
            "d3_trace": d3_npz["t498_model_output"],
            "d3_final": d3_npz["final_sample"],
        },
        "holdout",
    )
    del holdout_dataset

    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    train_cfg.dataset.split_partition = "train"
    train_dataset = _scene_only_dataset(train_cfg)
    train_records = evaluate_partition(
        train_dataset,
        train_source,
        {"d2_conditional": d2_npz["conditional"][len(holdout_source):]},
        "train",
    )
    summary = hsi_diagnostics.summarize_d4_offline_decomp(
        holdout_records,
        train_records,
        d2["stratum_weights"],
        seed=int(cfg.seed),
        replicates=int(cfg.d4_bootstrap_replicates),
    )
    payload = {
        "schema_version": 1,
        "mode": "d4_offline_decomp",
        "seed": int(cfg.seed),
        "inputs": {
            "d2_payload": str(d2_path),
            "d2_array_sha256": d2["array_archive"]["sha256"],
            "d3_payload": str(d3_path),
            "d3_array_sha256": d3["array_archive"]["sha256"],
        },
        "stratum_weights": d2["stratum_weights"],
        "summary": summary,
        "records": {"holdout": holdout_records, "train": train_records},
    }
    output_dir = Path(str(cfg.lingo_output_dir)) / "d4-offline-decomp"
    return _write_payload(output_dir, payload)


def evaluate_model(cfg: DictConfig) -> Path:
    if int(cfg.batch_size) != 1:
        raise ValueError("LINGO HSI timing protocol requires batch_size=1")
    if str(cfg.sample_type) not in ("consistency", "diffusion"):
        raise ValueError("sample_type must be consistency or diffusion")
    from test_infbagel_hosi import seed_everything, synchronize_cuda

    seed_everything(int(cfg.seed))
    guided = bool(cfg.get("use_guidance", False))
    export_motion = bool(cfg.get("export_motion", False))
    # RDS is scoped to unguided cells: guidance_loss.apply_hsi_guidance_loss pulls
    # joints toward free voxels regardless of need_scene, so the paired
    # "null-scene" rollout is still scene-driven and its divergence from the
    # scene-conditioned rollout is confounded.  Skipping the pass also halves the
    # guided cell cost.  The unguided path is the gate column and is untouched:
    # the null pass runs inside _rng_rewound, which restores the post-pass RNG
    # state on exit, so omitting it leaves the next episode's RNG identical.
    rds_requested = bool(cfg.get("hsi_compute_rds", True))
    rds_available = not guided and rds_requested
    if cfg.get("lingo_episode_subset", None) is not None and cfg.lingo_sequence_limit is not None:
        raise ValueError("lingo_episode_subset and lingo_sequence_limit are mutually exclusive")
    episodes = _load_episodes(
        Path(cfg.lingo_episode_dir),
        None if cfg.lingo_sequence_limit is None else int(cfg.lingo_sequence_limit),
    )
    window_counts = [int(episode["episode_num"]) for _, _, episode in episodes]
    canonical_episode_total = len(episodes)
    canonical_window_total = int(sum(window_counts))
    episode_subset = _load_episode_subset(
        cfg.get("lingo_episode_subset", None), episodes, window_counts
    )
    eligible_ordinals = tuple(int(value) for value in episode_subset["canonical_ordinals"])
    shard_count = int(cfg.shard_count)
    shard_index = int(cfg.shard_index)
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            "shard_index %d out of range for shard_count %d" % (shard_index, shard_count)
        )
    latency_subset = bool(cfg.get("latency_subset", False))
    if latency_subset and shard_count != 1:
        raise ValueError(
            "latency_subset requires shard_count=1: a contended shard cannot measure latency"
        )
    sharded = shard_count > 1
    if latency_subset:
        selected = select_latency_subset(episodes, int(cfg.latency_target_windows))
        partition_rule = "latency_subset_two_phase_coverage_then_fill"
    elif sharded:
        local_counts = [window_counts[ordinal] for ordinal in eligible_ordinals]
        selected = tuple(
            eligible_ordinals[position]
            for position in plan_episode_shards(local_counts, shard_count)[shard_index]
        )
        partition_rule = (
            "episode_subset_greedy_longest_first_bin_packing_by_window_count"
            if episode_subset["enabled"]
            else "greedy_longest_first_bin_packing_by_window_count"
        )
    else:
        selected = eligible_ordinals
        partition_rule = (
            "episode_subset_serial_canonical_order"
            if episode_subset["enabled"]
            else "serial_full_enumeration"
        )
    # (canonical ordinal, scene name, scene-local index, episode).  The ordinal is
    # the episode's index in the FULL enumeration, never its index in this shard:
    # it is what makes the per-episode seed, and therefore every per-episode
    # metric, independent of how the episodes were partitioned.
    work = [(ordinal,) + episodes[ordinal] for ordinal in selected]
    shard_window_total = int(sum(window_counts[ordinal] for ordinal in selected))
    warmup_required = int(cfg.timing_warmup_sequences)
    # Warmup is canonical for the quality pass so eight shards flag the same five
    # episodes a serial run flags rather than five each.  In latency mode the
    # subset is scene-spanning rather than a canonical prefix, so warmup reverts
    # to its normal meaning: the first episodes actually executed.  For the full
    # serial enumeration the two definitions coincide exactly.
    if latency_subset:
        warmup_ordinals = frozenset(selected[:warmup_required])
    else:
        warmup_ordinals = frozenset(
            ordinal for ordinal in selected if ordinal < warmup_required
        )
    warmup_in_selection = len(warmup_ordinals)
    print(
        "SELECTION shard %d/%d episodes=%d windows=%d (canonical %d/%d) rule=%s"
        % (
            shard_index,
            shard_count,
            len(work),
            shard_window_total,
            canonical_episode_total,
            canonical_window_total,
            partition_rule,
        ),
        flush=True,
    )
    dataset = _scene_only_dataset(cfg)
    model, checkpoint_provenance = _load_strict_checkpoint(cfg)
    model_name = (
        str(cfg.model_name) if cfg.model_name is not None else Path(str(cfg.ckpt_path)).stem
    )
    # model_name is payload metadata only; per-run isolation is lingo_output_dir.
    # The shard suffix means eight shards stay isolated even when they share one
    # lingo_output_dir, instead of racing on the same directory.
    shard_suffix = "" if shard_count == 1 else "-shard%02dof%02d" % (shard_index, shard_count)
    output_dir = Path(cfg.lingo_output_dir) / (
        f"{Path(str(cfg.ckpt_path)).stem}-"
        f"{checkpoint_provenance['checkpoint_sha256'][:12]}"
        f"{shard_suffix}"
    )
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite LINGO HSI output: %s" % output_dir)
    sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler.set_dataset_and_model(dataset, model)
    call_counter = _ForwardCallCounter(model)
    source = GroundTruthSource(DATASET_ROOT)
    smplx_cache: Dict[str, torch.nn.Module] = {}
    geometries: Dict[str, SceneGeometry] = {}
    records = OrderedDict()
    motion_records: List[Dict[str, Any]] = []
    export_lengths: "OrderedDict[str, int]" = OrderedDict()
    export_fps = float(cfg.fps) / float(cfg.interp_s)
    all_window_seconds: List[float] = []
    all_window_call_counts: List[int] = []
    timing_sequences: List[Mapping[str, float]] = []

    for position, (canonical_ordinal, scene_name, scene_index, episode) in enumerate(work, start=1):
        if scene_name not in geometries:
            geometries[scene_name] = SceneGeometry.from_scene(
                scene_name,
                dataset_root=DATASET_ROOT,
                mesh_root=Path(cfg.lingo_mesh_root),
                cache_dir=default_cache_dir(),
            )
        # Re-seed per episode from the canonical ordinal.  A single run-level seed
        # would make each episode's result depend on how many RNG draws every
        # earlier episode consumed, so a shard -- which sees a different
        # subsequence -- would produce different numbers.  This is unconditional,
        # not sharding-only: every cell from here must share one regime.
        seed_everything(int(cfg.seed) + int(canonical_ordinal))
        pre_rng_state = _capture_rng_state()
        export_sink: Optional[Dict[str, Any]] = {} if export_motion else None
        episode_start = time.perf_counter()
        vertices, joints, sequence_index, window_seconds, window_call_counts, future_occ_error = (
            sampled_motion(
                cfg,
                dataset,
                sampler,
                episode,
                source,
                smplx_cache,
                call_counter,
                need_scene=True,
                export_sink=export_sink,
            )
        )
        synchronize_cuda(cfg.device)
        episode_seconds = time.perf_counter() - episode_start
        joints_null = None
        if rds_available:
            with _rng_rewound(pre_rng_state):
                _, joints_null, _, _, _, _ = sampled_motion(
                    cfg,
                    dataset,
                    sampler,
                    episode,
                    source,
                    smplx_cache,
                    call_counter,
                    need_scene=False,
                )
        scene_conditioned_finite = bool(
            torch.isfinite(vertices.frames).all() and torch.isfinite(joints.frames).all()
        )
        null_scene_finite = (
            True if joints_null is None else bool(torch.isfinite(joints_null.frames).all())
        )
        if not scene_conditioned_finite or not null_scene_finite:
            raise RuntimeError(
                "non-finite sampled motion for scene %s episode %d: "
                "scene-conditioned finite=%s, paired null-scene finite=%s"
                % (
                    scene_name,
                    scene_index,
                    scene_conditioned_finite,
                    null_scene_finite,
                )
            )
        metric = compute_metric_record(
            vertices, joints, geometries[scene_name], episode["pelvis_goal"], float(cfg.fps)
        )
        metric.update(hsi_diagnostics.future_occ_motion_diagnostics(joints, fps=float(cfg.fps)))
        metric["future_occ_center_error"] = future_occ_error
        metric.update(hsi_metrics.reaction_divergence_score(joints, joints_null)
                      if rds_available else {"rds": None, "rds_max": None})
        # Canonical, not shard-local: eight shards must flag the same five
        # episodes a serial run flags, not five each.
        excluded_as_warmup = int(canonical_ordinal) in warmup_ordinals
        metric.update(
            {
                "canonical_ordinal": int(canonical_ordinal),
                "scene_name": scene_name,
                "scene_sequence_index": scene_index,
                "source_sequence_index": sequence_index,
                "is_object": False,
                "non_watertight": scene_name in NON_WATERTIGHT_SCENES,
                "rds_available": rds_available,
                "excluded_as_warmup": excluded_as_warmup,
                "sampling_seconds": None if sharded else float(sum(window_seconds)),
                "per_window_wall_seconds": None if sharded else float(np.mean(window_seconds)),
            }
        )
        sequence_name = "%s:%06d" % (scene_name, sequence_index)
        if sequence_name in records:
            raise ValueError("duplicate sequence key %s" % sequence_name)
        records[sequence_name] = metric
        if export_sink is not None:
            # Re-derive the caption from the dataset independently of the rollout
            # and refuse a mismatch, so the exported string is demonstrably the
            # one the model was conditioned on rather than an assumed match.
            caption_from_cond = export_sink.pop("caption_from_cond")
            caption = str(dataset.lingo_dataset.text[int(episode["data_idx"])][0])
            if caption != caption_from_cond:
                raise RuntimeError(
                    "caption mismatch for %s: rollout used %r, re-derived %r"
                    % (sequence_name, caption_from_cond, caption)
                )
            export_record = _motion_export_record(**export_sink)
            motion_records.append(
                {
                    "sequence_id": sequence_name,
                    "condition_id": _motion_condition_id(episode),
                    "caption": caption,
                    "fps": export_fps,
                    "record": export_record,
                    "extra": _motion_export_extra(scene_name, episode, sequence_index),
                }
            )
            export_lengths[sequence_name] = int(export_record["global_jpos"].shape[0])
        all_window_seconds.extend(window_seconds)
        all_window_call_counts.extend(window_call_counts)
        timing_sequences.append(
            {
                "gen_seconds": float(sum(window_seconds)),
                "frames": int(joints.frames.shape[0]),
                "episode_seconds": episode_seconds,
            }
        )
        print(
            "MODEL %d/%d (canonical %d/%d) %s windows=%d sec/window=%.4f finite=1"
            % (
                position,
                len(work),
                canonical_ordinal,
                canonical_episode_total,
                sequence_name,
                len(window_seconds),
                np.mean(window_seconds),
            ),
            flush=True,
        )

    scene_summary = _aggregate_by_scene(records)
    if sharded:
        _invalidate_scene_summary_timing(scene_summary)
    distinct_call_counts = sorted(set(all_window_call_counts))
    if len(distinct_call_counts) != 1:
        raise RuntimeError(
            "denoiser calls per window varied across the run: %s" % distinct_call_counts
        )
    denoiser_calls_per_window = distinct_call_counts[0]
    sampler_steps_per_window = (
        int(sampler.cm_timesteps)
        if str(cfg.sample_type) == "consistency"
        else int(sampler.timesteps)
    )
    # cm_sample evaluates once per step; p_sample evaluates conditional and unconditional passes.
    timing = {
        "per_window_wall_seconds": float(np.mean(all_window_seconds)),
        "total_sampling_seconds": float(np.sum(all_window_seconds)),
        "window_count": len(all_window_seconds),
        "denoiser_calls_per_window": denoiser_calls_per_window,
        "sampler_steps_per_window": sampler_steps_per_window,
        "cuda_synchronized": True,
        "batch_size": 1,
    }
    # The warmup episodes are a canonical-order prefix, so within a shard -- which
    # walks its ordinals in ascending order -- they are still this list's prefix.
    timing.update(
        _aggregate_timing(
            timing_sequences,
            warmup_sequences=warmup_in_selection,
            fps=float(cfg.fps),
        )
    )
    timing["warmup_sequences_required"] = warmup_required
    if sharded:
        _invalidate_timing(timing)
    else:
        # timing_valid means "the wall-clock numbers in this block are valid
        # measurements".  An uncontended run whose every episode fell inside the
        # warmup prefix has no timed data at all, and every aggregate below is
        # null; claiming validity there would be the same trap as claiming it
        # under contention.
        timing["timing_valid"] = bool(timing["protocol_complete"])
        timing["timing_invalid_reason"] = (
            None
            if timing["timing_valid"]
            else "no timed sequences: warmup_sequences_required=%d consumed all %d "
            "selected episodes" % (warmup_required, len(work))
        )
    rds_block: Dict[str, Any] = {
        "joint_set": "smplx_joints_28",
        "null_scene_mode": "need_scene=False",
        "noise_shared": "rng_state_rewound",
        "guided": guided,
        "available": rds_available,
    }
    if not rds_available:
        rds_block["rds"] = None
        rds_block["rds_max"] = None
        rds_block["skipped_reason"] = (
            "disabled by hsi_compute_rds=false for this registered diagnostic"
            if not rds_requested
            else "guidance pulls joints toward free voxels regardless of need_scene, so a "
            "guided null-scene rollout is confounded; RDS is scoped to unguided cells"
        )
    payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "model_name": model_name,
        "checkpoint": checkpoint_provenance,
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        "sample_type": str(cfg.sample_type),
        "guided": guided,
        "future_occ_diagnostic": {
            "mode": str(cfg.get("hsi_future_occ_mode", "predicted")),
            "offsets": list(hsi_diagnostics.FUTURE_OCC_OFFSETS),
        },
        "episode_subset": episode_subset,
        "rds": rds_block,
        "sampling_body": "smplx_vertices_10475",
        "fps": float(cfg.fps),
        "sequence_count": len(records),
        "scene_count": len(scene_summary),
        "scene_summary": scene_summary,
        "timing": timing,
        "sharding": {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "canonical_episode_total": canonical_episode_total,
            "canonical_window_total": canonical_window_total,
            "eligible_episode_count": int(episode_subset["episode_count"]),
            "eligible_window_total": int(episode_subset["window_count"]),
            "shard_episode_ordinals": list(selected),
            "shard_window_total": shard_window_total,
            "partition_rule": partition_rule,
            "per_episode_seeding": "seed_everything(seed + canonical_ordinal)",
            "timing_valid": bool(timing["timing_valid"]),
        },
        "latency_subset": {
            "enabled": latency_subset,
            "target_windows": int(cfg.latency_target_windows) if latency_subset else None,
            "selected_windows": shard_window_total if latency_subset else None,
            "selection_rule": partition_rule if latency_subset else None,
        },
        "metrics": records,
    }
    if export_motion:
        payload["motion_export"] = _motion_export_block(
            export_lengths, export_fps, "generated"
        )
    return _write_payload(
        output_dir, payload, motion_records=motion_records if export_motion else None
    )


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config_sample_infbagel_lingo_hsi",
)
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("ROOT_DIR", str(REPO_ROOT))
    mode = str(cfg.lingo_hsi_mode)
    if mode == "ground_truth":
        path = evaluate_ground_truth(cfg)
    elif mode == "sample":
        path = evaluate_model(cfg)
    elif mode == "merge_shards":
        path = _merge_shards(cfg)
    elif mode == "teacher_forced_boundary":
        path = evaluate_teacher_forced_boundary(cfg)
    elif mode == "predictor_decomp":
        path = evaluate_predictor_decomp(cfg)
    elif mode == "single_window_chain":
        path = evaluate_single_window_chain(cfg)
    elif mode == "d4_offline_decomp":
        path = evaluate_d4_offline_decomp(cfg)
    elif mode == "chain_rebase":
        path = evaluate_chain_rebase(cfg)
    else:
        raise ValueError(
            "lingo_hsi_mode must be ground_truth, sample, merge_shards or "
            "teacher_forced_boundary, predictor_decomp, single_window_chain or "
            "d4_offline_decomp or chain_rebase, got %s"
            % mode
        )
    print("Wrote %s" % path)


if __name__ == "__main__":
    main()
