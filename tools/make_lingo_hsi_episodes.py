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
    generated_stride = (MAX_WINDOW_SIZE - AUTO_REGRE_NUM) * STEP
    history_span = AUTO_REGRE_NUM * STEP
    return math.ceil((sequence_length - history_span) / generated_stride)


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
            start_location = joints[int(window_start[data_idx]), 0].copy()
            start_location[1] = 0.0
            pelvis_goal = joints[int(window_end[data_idx]) - STEP, 0].copy()
            pelvis_goal[1] = 0.0
            episodes.append({
                "scene_name": scene,
                "data_idx": data_idx,
                "start_location": start_location.tolist(),
                "pelvis_goal": pelvis_goal.tolist(),
                "object_goal": pelvis_goal.tolist(),
                "object_name": None,
                "penetration_counts": 0,
                "test_frames": [0, 1, MAX_WINDOW_SIZE - 1],
                "attempts": 0,
                "episode_num": episode_num(length),
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
