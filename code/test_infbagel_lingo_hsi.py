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
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as Rotation

import pytorch3d.transforms as transforms

import utils as project_utils
from priors.hsi import metrics as hsi_metrics
from priors.hsi.scene_field import SceneGeometry, default_cache_dir
from utils import SMPLX_JOINTS_28, create_smplx_model, interp_jrot, interpolate_joints, run_smplx_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "dataset"
WINDOW_FRAMES = 16
HISTORY_FRAMES = 2
DATA_STEP = 3
WINDOW_STRIDE_RAW = (WINDOW_FRAMES - HISTORY_FRAMES) * DATA_STEP
NON_WATERTIGHT_SCENES = frozenset(("031", "049-bed"))

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


class GroundTruthSource:
    """Memory-mapped LINGO arrays needed to reproduce the sampling body."""

    def __init__(self, root: Path):
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
    agreement_keys = ("seed", "sample_type", "guided", "fps", "sampling_body", "model_name")
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
    absent = sorted(set(range(expected_episodes)) - set(ordinals))
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


def _write_payload(output_dir: Path, payload: Mapping) -> Path:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite LINGO HSI output: %s" % output_dir)
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
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
    episodes = _load_episodes(
        Path(cfg.lingo_episode_dir),
        None if cfg.lingo_sequence_limit is None else int(cfg.lingo_sequence_limit),
    )
    source = GroundTruthSource(DATASET_ROOT)
    smplx_cache: Dict[str, torch.nn.Module] = {}
    geometries: Dict[str, SceneGeometry] = {}
    records = OrderedDict()

    for ordinal, (scene_name, scene_index, episode) in enumerate(episodes, start=1):
        if scene_name not in geometries:
            geometries[scene_name] = SceneGeometry.from_scene(
                scene_name,
                dataset_root=DATASET_ROOT,
                mesh_root=Path(cfg.lingo_mesh_root),
                cache_dir=default_cache_dir(),
            )
        vertices, joints, sequence_index = ground_truth_motion(
            source,
            episode,
            device,
            int(cfg.interp_s),
            int(cfg.smplx_batch_size),
            smplx_cache,
        )
        metric = compute_metric_record(
            vertices, joints, geometries[scene_name], episode["pelvis_goal"], float(cfg.fps)
        )
        sequence_name = "%s:%06d" % (scene_name, sequence_index)
        if sequence_name in records:
            raise ValueError("duplicate sequence key %s" % sequence_name)
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
        "schema_version": 1,
        "model_name": "ground_truth",
        "seed": int(cfg.seed),
        "sampling_body": "smplx_vertices_10475",
        "fps": float(cfg.fps),
        "sequence_count": len(records),
        "scene_count": len(scene_summary),
        "scene_summary": scene_summary,
        "metrics": records,
    }
    return _write_payload(Path(cfg.lingo_output_dir), payload)


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


def sampled_motion(
    cfg: DictConfig,
    dataset,
    sampler,
    episode: Mapping,
    source: GroundTruthSource,
    smplx_cache: MutableMapping[str, torch.nn.Module],
    call_counter: _ForwardCallCounter,
    need_scene: bool = True,
):
    # Importing here keeps the GT-first path independent of model sampling while
    # reusing the established autoregressive helpers verbatim.
    from test_infbagel_hosi import get_mat, sample_step, synchronize_cuda
    from utils import transform_points, yup_to_zup, zup_to_yup

    device = torch.device(str(cfg.device))
    data_idx = int(episode["data_idx"])
    data = _lingo_item(dataset, data_idx)
    sequence_index, _ = source.episode_indices(data_idx, int(episode["episode_num"]))
    source_start = int(source.sequence_start[sequence_index])

    cond = _scene_condition(cfg, episode, need_scene=need_scene)
    cond["raw_text"] = dataset.lingo_dataset.text[data_idx][0]
    cond["text_emb"] = data["text_clip_embedding"].to(device).unsqueeze(0)
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

    point_windows, rotation_windows = [], []
    sampling_seconds = []
    denoiser_call_counts = []
    points = points_world
    generated_object_trans = object_trans_world
    generated_object_rotation = object_rotation.reshape(1, WINDOW_FRAMES, 9)
    generated_contact = contact

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
    smpl_translation = yup_to_zup(interpolated_points[:, 0] + translation_offset)
    smpl_pose = yup_to_zup(local_axis)
    vertices, metric_joints = _run_smplx_chunks(
        smpl_pose,
        smpl_translation,
        betas,
        gender,
        device,
        int(cfg.smplx_batch_size),
        smplx_cache,
    )
    vertices, metric_joints = zup_to_yup(vertices), zup_to_yup(metric_joints)
    return (
        _upsampled_stitched(coarse_points, vertices, int(cfg.interp_s)),
        _upsampled_stitched(coarse_points, metric_joints, int(cfg.interp_s)),
        sequence_index,
        sampling_seconds,
        denoiser_call_counts,
    )


def evaluate_model(cfg: DictConfig) -> Path:
    if int(cfg.batch_size) != 1:
        raise ValueError("LINGO HSI timing protocol requires batch_size=1")
    if str(cfg.sample_type) not in ("consistency", "diffusion"):
        raise ValueError("sample_type must be consistency or diffusion")
    from test_infbagel_hosi import seed_everything, synchronize_cuda

    seed_everything(int(cfg.seed))
    guided = bool(cfg.get("use_guidance", False))
    # RDS is scoped to unguided cells: guidance_loss.apply_hsi_guidance_loss pulls
    # joints toward free voxels regardless of need_scene, so the paired
    # "null-scene" rollout is still scene-driven and its divergence from the
    # scene-conditioned rollout is confounded.  Skipping the pass also halves the
    # guided cell cost.  The unguided path is the gate column and is untouched:
    # the null pass runs inside _rng_rewound, which restores the post-pass RNG
    # state on exit, so omitting it leaves the next episode's RNG identical.
    rds_available = not guided
    episodes = _load_episodes(
        Path(cfg.lingo_episode_dir),
        None if cfg.lingo_sequence_limit is None else int(cfg.lingo_sequence_limit),
    )
    window_counts = [int(episode["episode_num"]) for _, _, episode in episodes]
    canonical_episode_total = len(episodes)
    canonical_window_total = int(sum(window_counts))
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
        selected = plan_episode_shards(window_counts, shard_count)[shard_index]
        partition_rule = "greedy_longest_first_bin_packing_by_window_count"
    else:
        selected = tuple(range(canonical_episode_total))
        partition_rule = "serial_full_enumeration"
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
        episode_start = time.perf_counter()
        vertices, joints, sequence_index, window_seconds, window_call_counts = (
            sampled_motion(
                cfg,
                dataset,
                sampler,
                episode,
                source,
                smplx_cache,
                call_counter,
                need_scene=True,
            )
        )
        synchronize_cuda(cfg.device)
        episode_seconds = time.perf_counter() - episode_start
        joints_null = None
        if rds_available:
            with _rng_rewound(pre_rng_state):
                _, joints_null, _, _, _ = sampled_motion(
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
            "guidance pulls joints toward free voxels regardless of need_scene, so a "
            "guided null-scene rollout is confounded; RDS is scoped to unguided cells"
        )
    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "checkpoint": checkpoint_provenance,
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        "sample_type": str(cfg.sample_type),
        "guided": guided,
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
    return _write_payload(output_dir, payload)


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
    else:
        raise ValueError(
            "lingo_hsi_mode must be ground_truth, sample or merge_shards, got %s" % mode
        )
    print("Wrote %s" % path)


if __name__ == "__main__":
    main()
