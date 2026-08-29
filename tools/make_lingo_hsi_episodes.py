#!/usr/bin/env python3
"""Build deterministic scene-only LINGO test episodes from a split partition."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT_DIR / "data" / "dataset"
DEFAULT_SPLIT_MANIFEST = (
    ROOT_DIR / "experiments" / "splits" / "lingo_scene_family_disjoint_v3_seed42.json"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "lingo_hsi_test" / "data"
DEFAULT_PARTITION = "test"
DEFAULT_PER_SCENE_CAP = 20
MAX_WINDOW_SIZE = 16
AUTO_REGRE_NUM = 2
STEP = 3
GENERATED_STRIDE = (MAX_WINDOW_SIZE - AUTO_REGRE_NUM) * STEP
HISTORY_SPAN = AUTO_REGRE_NUM * STEP
WINDOW_SPAN = MAX_WINDOW_SIZE * STEP


def create_corrected_scene_labels(
    scene_name: Sequence[str], sequence_start: np.ndarray, sequence_idx: np.ndarray,
) -> np.ndarray:
    """Apply the mixed dataset's corrected per-window LINGO scene-label rule."""
    half = len(sequence_start) // 2
    source_scene_names = [str(scene_name[int(sequence_start[i])]) for i in range(half)]
    sequence_scene_names = np.asarray(
        source_scene_names + [f"{name}_mirror" for name in source_scene_names],
        dtype=object,
    )
    return sequence_scene_names[sequence_idx]


def episode_num(sequence_length: int) -> int:
    return math.ceil((sequence_length - HISTORY_SPAN) / GENERATED_STRIDE)


def rollout_frame_indices(
    sequence_start: int, sequence_end: int, windows: int,
) -> np.ndarray:
    """Raw source frames a ``windows``-window rollout covers, ``[windows, MAX_WINDOW_SIZE]``.

    This mirrors ``GroundTruthSource.episode_indices`` in
    ``code/test_infbagel_lingo_hsi.py`` frame for frame, because that function --
    not arithmetic about window counts -- is what decides which source frame the
    goal metrics are finally scored against.  Each window samples
    ``MAX_WINDOW_SIZE`` frames at stride ``STEP`` from its own start, window
    starts advance by ``GENERATED_STRIDE``, and every index is clamped to the
    last frame of the source sequence.

    The evaluator stitches those windows with
    ``stitch_windows(..., history_frames=AUTO_REGRE_NUM)``, which keeps window 0
    whole and drops the leading ``AUTO_REGRE_NUM`` frames of every later window,
    so the terminal frame of the stitched coarse sequence is this array's last
    element.  Upsampling preserves that: ``utils.interpolate_joints`` evaluates
    at ``linspace(0, T - 1, T * scale)`` and ``utils.interp_jrot`` assigns its
    final ``scale`` samples from coarse frame ``T - 1``, so both put their last
    output sample exactly on the last coarse frame.  Hence
    ``rollout_frame_indices(...)[-1, -1]`` is the source frame that
    ``goal_metrics``'s ``last_dist`` and ``success_last_*`` are measured at.
    """
    if windows < 1:
        raise ValueError("windows must be >= 1, got %d" % windows)
    offsets = np.arange(MAX_WINDOW_SIZE, dtype=np.int64) * STEP
    starts = sequence_start + np.arange(windows, dtype=np.int64) * GENERATED_STRIDE
    return np.minimum(starts[:, None] + offsets[None, :], sequence_end - 1)


def load_inputs(
    dataset_root: Path, split_manifest: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with split_manifest.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    with (dataset_root / "scene_name.pkl").open("rb") as handle:
        scene_name = pickle.load(handle)
    with (
        dataset_root / "language_motion_dict" / "language_motion_dict__inter_and_loco__16.pkl"
    ).open("rb") as handle:
        language = pickle.load(handle)

    sequence_start = np.load(dataset_root / "start_idx.npy").astype(np.int64)
    sequence_end = np.load(dataset_root / "end_idx.npy").astype(np.int64)
    joints = np.load(dataset_root / "human_joints_aligned.npy", mmap_mode="r")
    window_scene = create_corrected_scene_labels(
        scene_name, sequence_start, np.asarray(language["ori_sequence_idx"], dtype=np.int64),
    )
    return split, language, sequence_start, sequence_end, joints, window_scene


def build_episodes(
    split: Mapping[str, Any], language: Mapping[str, Any], sequence_start: np.ndarray,
    sequence_end: np.ndarray, joints: np.ndarray, window_scene: np.ndarray,
    partition: str, per_scene_cap: int,
) -> Dict[str, List[Dict[str, Any]]]:
    sequence_idx = np.asarray(language["ori_sequence_idx"], dtype=np.int64)
    window_start = np.asarray(language["start_idx"], dtype=np.int64)
    window_end = np.asarray(language["end_idx"], dtype=np.int64)
    left_hand = np.asarray(language["left_hand_inter_frame"])
    right_hand = np.asarray(language["right_hand_inter_frame"])
    sequence_length = sequence_end - sequence_start
    selected_scenes = set(split[partition]["scenes"])
    candidates: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)

    for data_idx, source_sequence_idx in enumerate(sequence_idx):
        source_sequence_idx = int(source_sequence_idx)
        length = int(sequence_length[source_sequence_idx])
        scene = str(window_scene[data_idx])
        if (
            scene in selected_scenes
            and window_start[data_idx] == sequence_start[source_sequence_idx]
            and left_hand[data_idx] == -1
            and right_hand[data_idx] == -1
            and length > 48
        ):
            candidates[scene].append((length, data_idx))

    episodes_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for scene in split[partition]["scenes"]:
        selected = sorted(candidates[scene], key=lambda item: (-item[0], item[1]))[:per_scene_cap]
        episodes = []
        for length, data_idx in selected:
            source_sequence_idx = int(sequence_idx[data_idx])
            source_start = int(sequence_start[source_sequence_idx])
            source_end = int(sequence_end[source_sequence_idx])
            # The candidate filter already pinned window_start to the sequence
            # start; this pins the window span, so a language dict whose windows
            # stopped being MAX_WINDOW_SIZE frames at stride STEP fails loudly
            # instead of silently shifting every goal.
            if int(window_end[data_idx]) - int(window_start[data_idx]) != WINDOW_SPAN:
                raise ValueError(
                    "window %d spans %d raw frames, expected %d"
                    % (
                        data_idx,
                        int(window_end[data_idx]) - int(window_start[data_idx]),
                        WINDOW_SPAN,
                    )
                )
            windows = episode_num(length)
            rollout = rollout_frame_indices(source_start, source_end, windows)
            # The goal is where the rollout ENDS, not where its first window
            # ends.  The previous rule used this window's own end frame, which
            # made start_location and pelvis_goal the two ends of one 16-coarse-
            # frame (48 raw frame, 0.53 s) window while the episode rolls out
            # `windows` of them -- median 4, max 55.  That made the goal
            # unreachably close instead of unreachably far: 50.1% of episodes had
            # it within 10 cm of the start pose, it capped the whole
            # start-to-goal distance at 0.778 m, and because
            # test_infbagel_hosi.sample_step advances ~0.8 m along `trajectory`
            # per window, its moving-target lookahead clamped to the path
            # endpoint on every window of every episode.
            terminal_frame = int(rollout[-1, -1])
            start_location = joints[int(window_start[data_idx]), 0].copy()
            start_location[1] = 0.0
            pelvis_goal = joints[terminal_frame, 0].copy()
            pelvis_goal[1] = 0.0
            episodes.append({
                "scene_name": scene,
                "data_idx": data_idx,
                "start_location": start_location.tolist(),
                "pelvis_goal": pelvis_goal.tolist(),
                # object_goal stays a copy of pelvis_goal: these are scene-only
                # episodes, object_name is None and _scene_condition sets
                # need_object=False, which zeroes both the object-goal embedding
                # and the occupancy branch's object-goal term, so the value is
                # inert.  Keeping it equal to pelvis_goal keeps that inertness
                # verifiable rather than introducing a second goal convention.
                "object_goal": pelvis_goal.tolist(),
                "object_name": None,
                "penetration_counts": 0,
                "test_frames": [0, 1, MAX_WINDOW_SIZE - 1],
                "attempts": 0,
                "episode_num": windows,
                # Provenance for the goal, so a manifest can be audited without
                # re-deriving the stitching.  Extra keys are inert: the
                # evaluator's _load_episodes and _scene_condition read by name.
                "source_sequence_idx": source_sequence_idx,
                "pelvis_goal_frame": terminal_frame,
                "pelvis_goal_frame_clamped": bool(
                    int(rollout[-1, -1])
                    < source_start + (windows - 1) * GENERATED_STRIDE + (MAX_WINDOW_SIZE - 1) * STEP
                ),
            })
        episodes_by_scene[scene] = episodes
    return episodes_by_scene


def write_outputs(
    episodes_by_scene: Mapping[str, Sequence[Mapping[str, Any]]], output_dir: Path,
    split_manifest: Path, partition: str, per_scene_cap: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for scene, episodes in episodes_by_scene.items():
        with (output_dir / f"{scene}.json").open("w", encoding="utf-8") as handle:
            json.dump(episodes, handle)
            handle.write("\n")

    scene_counts = {
        scene: {"episode_count": len(episodes), "sequence_count": len(episodes)}
        for scene, episodes in episodes_by_scene.items()
    }
    manifest = {
        "split_manifest": str(split_manifest),
        "partition": partition,
        "per_scene_cap": per_scene_cap,
        "selection_rule": (
            "For each scene, select up to the per-scene cap of eligible sequences by descending "
            "sequence length, break ties by ascending window index, and use each sequence's start window."
        ),
        "goal_rule": (
            "pelvis_goal (and its inert object_goal copy) is the ground-truth pelvis, Y zeroed, at "
            "the terminal frame of the episode's own rollout span: rollout_frame_indices(...)[-1, -1], "
            "i.e. source frame min(sequence_start + (episode_num - 1) * 42 + 45, sequence_end - 1). "
            "That is the frame the stitched, upsampled sequence ends on, so it is the frame "
            "goal_metrics scores last_dist / success_last_* at. Superseded rule, kept here so the "
            "two episode sets are distinguishable: the start window's own end frame "
            "(window_end - STEP == sequence_start + 45), one 16-coarse-frame window from the start."
        ),
        "scenes": scene_counts,
    }
    with (output_dir.parent / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--per-scene-cap", type=int, default=DEFAULT_PER_SCENE_CAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    split, language, sequence_start, sequence_end, joints, window_scene = load_inputs(
        args.dataset_root, args.split_manifest,
    )
    episodes_by_scene = build_episodes(
        split, language, sequence_start, sequence_end, joints, window_scene,
        args.partition, args.per_scene_cap,
    )
    write_outputs(
        episodes_by_scene, args.output_dir, args.split_manifest,
        args.partition, args.per_scene_cap,
    )

    print(f"Total episodes: {sum(len(episodes) for episodes in episodes_by_scene.values())}")
    for scene, episodes in episodes_by_scene.items():
        print(f"{scene}: {len(episodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
