#!/usr/bin/env python3
"""Build the reportable CPU-only D2-Z full-split floor/gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from priors.near_ground import (  # noqa: E402
    D2Z_FLOOR_ALGORITHM,
    D2Z_FLOOR_ALGORITHM_FILE_SHA256,
    D2Z_FOOT_JOINTS,
    D2Z_GATE_AUDIT_RUN_ID,
    D2Z_GATE_AUDIT_SCHEMA,
    D2Z_HEIGHT_THRESHOLDS_M,
    immutable_gt_near_ground_gate,
    sha256_file,
)
from priors.optimizer_reset import (  # noqa: E402
    NATIVE_SELECTION_SHA256,
    select_native_holdout,
)


RUN_ID = D2Z_GATE_AUDIT_RUN_ID
SUBPHASE = "1B-D2-Z0-gate-audit-r1"
EXPECTED_PYTHON = "/data/yujinlun/anaconda3/envs/infbagel/bin/python"
EXPECTED_SPLIT_SHA256 = "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
EXPECTED_SELECTION_COUNTS = {"7": 1096, "8": 1081, "10": 1211, "11": 1232}
EXPECTED_SELECTION_DENOMINATOR = 1344
EXPECTED_SELECTION_TOTAL_ACTIVE = 4620
EXPECTED_SELECTION_TOTAL_DENOMINATOR = 5376
EXPECTED_SELECTION_FLOOR_SUMMARY = (
    0.026023785583674908,
    0.04509613197296858,
    0.05307581648230553,
)


def _exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    return parser.parse_args()


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "seed": 42,
        "execution": {
            "host": "authority/ubuntu",
            "python": str(args.python.resolve()),
            "cpu_only": True,
            "cuda_calls": 0,
            "checkpoint_loads": 0,
            "training_updates": 0,
        },
        "split": {
            "path": str(
                (REPO / "experiments/splits/omomo_hoi_train_validation_seed42.json").resolve()
            ),
            "sha256": EXPECTED_SPLIT_SHA256,
            "partitions": ["train", "internal_validation"],
        },
        "gate": {
            "previous_frame": "immutable_gt",
            "sample_stride_source_frames": 3,
            "residual_frames_per_window": 14,
            "foot_joint_indices": list(D2Z_FOOT_JOINTS),
            "thresholds_m": {
                str(joint): threshold
                for joint, threshold in zip(D2Z_FOOT_JOINTS, D2Z_HEIGHT_THRESHOLDS_M)
            },
            "floor_source": "complete_immutable_aligned_gt_sequence_30hz",
            "floor_algorithm": D2Z_FLOOR_ALGORITHM,
            "floor_algorithm_file_sha256": D2Z_FLOOR_ALGORITHM_FILE_SHA256,
        },
        "sealed_selection": {
            "selection_sha256": NATIVE_SELECTION_SHA256,
            "sequences": 32,
            "windows": 96,
            "expected_active_counts": EXPECTED_SELECTION_COUNTS,
            "expected_denominator_per_joint": EXPECTED_SELECTION_DENOMINATOR,
            "expected_total_active": EXPECTED_SELECTION_TOTAL_ACTIVE,
            "expected_total_denominator": EXPECTED_SELECTION_TOTAL_DENOMINATOR,
            "expected_floor_min_median_max_m": list(EXPECTED_SELECTION_FLOOR_SUMMARY),
        },
        "output": str(args.output.resolve()),
        "checkpoint_selection": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "consistency_authorized": False,
    }


def _source_paths() -> Mapping[str, Path]:
    train = REPO / "data/train"
    return {
        "human_joints_aligned": train / "human_joints_aligned.npy",
        "sequence_starts": train / "start_idx.npy",
        "sequence_ends": train / "end_idx.npy",
        "language_windows": (
            train / "language_motion_dict/language_motion_dict__inter_and_loco__16.pkl"
        ),
        "sequence_names": train / "scene_name.pkl",
        "split": REPO / "experiments/splits/omomo_hoi_train_validation_seed42.json",
        "floor_algorithm_file": REPO / "code/eval_metrics.py",
        "selection_algorithm_file": REPO / "code/priors/optimizer_reset.py",
        "dataset_algorithm_file": REPO / "code/priors/data.py",
        "gate_algorithm_file": REPO / "code/priors/near_ground.py",
        "d2z_dataset_loss_file": REPO / "code/priors/d2z.py",
    }


def _partition_record(
    *,
    selected_sequences: list[int],
    indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    sequence_ids: np.ndarray,
    joints: np.ndarray,
    floors: Mapping[int, float],
) -> Dict[str, object]:
    counts = np.zeros(4, dtype=np.int64)
    occupancy = []
    ordered = []
    for raw_index in indices.tolist():
        index = int(raw_index)
        start, end = int(starts[index]), int(ends[index])
        if end - start != 48:
            raise ValueError(f"window {index} has {end-start} source frames, expected 48")
        frames = np.arange(start, end, 3)
        sequence = int(sequence_ids[index])
        gate = immutable_gt_near_ground_gate(
            np.asarray(joints[frames], dtype=np.float32),
            floors[sequence],
        )
        active_by_joint = gate.sum(axis=0, dtype=np.int64)
        counts += active_by_joint
        active = int(active_by_joint.sum())
        occupancy.append(active / gate.size)
        ordered.append({
            "sequence_index": sequence,
            "window_index": index,
            "active_by_joint": {
                str(joint): int(value)
                for joint, value in zip(D2Z_FOOT_JOINTS, active_by_joint.tolist())
            },
            "active": active,
            "denominator": int(gate.size),
        })
    denominator_per_joint = len(indices) * 14
    total_denominator = denominator_per_joint * 4
    floor_values = np.asarray([floors[value] for value in selected_sequences], dtype=np.float64)
    occupancy_array = np.asarray(occupancy, dtype=np.float64)
    return {
        "sequence_indices": selected_sequences,
        "floors_m": {str(value): float(floors[value]) for value in selected_sequences},
        "floor_min_median_max_m": [
            float(floor_values.min()),
            float(np.median(floor_values)),
            float(floor_values.max()),
        ],
        "nonfinite_floor_count": int((~np.isfinite(floor_values)).sum()),
        "windows": int(len(indices)),
        "ordered_sequence_window_ids": [
            [row["sequence_index"], row["window_index"]] for row in ordered
        ],
        "ordered_window_gate_records_sha256": _canonical_sha256(ordered),
        "active_counts": {
            str(joint): int(value)
            for joint, value in zip(D2Z_FOOT_JOINTS, counts.tolist())
        },
        "denominator_per_joint": int(denominator_per_joint),
        "total_active": int(counts.sum()),
        "total_denominator": int(total_denominator),
        "active_fraction": float(counts.sum() / total_denominator),
        "per_window_occupancy": {
            "minimum": float(occupancy_array.min()),
            "median": float(np.median(occupancy_array)),
            "maximum": float(occupancy_array.max()),
            "zero_active_windows": int((occupancy_array == 0).sum()),
            "fully_active_windows": int((occupancy_array == 1).sum()),
        },
        "nonfinite_gate_count": 0,
    }


def build_audit() -> Dict[str, object]:
    paths = _source_paths()
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if source_hashes["split"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("D2-Z split SHA-256 mismatch")
    if source_hashes["floor_algorithm_file"] != D2Z_FLOOR_ALGORITHM_FILE_SHA256:
        raise ValueError("D2-Z official floor algorithm SHA-256 mismatch")
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    if split.get("algorithm") != "omomo-sequence-sha256-seed42-v1":
        raise ValueError("D2-Z split algorithm mismatch")
    with paths["language_windows"].open("rb") as handle:
        language = pickle.load(handle)
    with paths["sequence_names"].open("rb") as handle:
        sequence_names = pickle.load(handle)
    starts = np.asarray(language["start_idx"], dtype=np.int64)
    ends = np.asarray(language["end_idx"], dtype=np.int64)
    sequence_ids = np.asarray(language["ori_sequence_idx"], dtype=np.int64)
    sequence_starts = np.load(paths["sequence_starts"], mmap_mode="r")
    sequence_ends = np.load(paths["sequence_ends"], mmap_mode="r")
    joints = np.load(paths["human_joints_aligned"], mmap_mode="r")

    # Import only after the source hash has been locked above.
    from eval_metrics import determine_floor_height_and_contacts

    all_sequences = sorted({
        int(value)
        for partition in ("train", "internal_validation")
        for value in split[partition]["sequence_indices"]
    })
    floors: Dict[int, float] = {}
    for sequence in all_sequences:
        begin, end = int(sequence_starts[sequence]), int(sequence_ends[sequence])
        value = float(determine_floor_height_and_contacts(
            np.asarray(joints[begin:end], dtype=np.float64).copy(),
            fps=30,
        ))
        if not np.isfinite(value):
            raise ValueError(f"non-finite D2-Z floor for sequence {sequence}")
        floors[sequence] = value

    partitions: Dict[str, object] = {}
    partition_indices: Dict[str, np.ndarray] = {}
    for partition in ("train", "internal_validation"):
        selected = sorted(int(value) for value in split[partition]["sequence_indices"])
        indices = np.flatnonzero(np.isin(sequence_ids, selected))
        partition_indices[partition] = indices
        partitions[partition] = _partition_record(
            selected_sequences=selected,
            indices=indices,
            starts=starts,
            ends=ends,
            sequence_ids=sequence_ids,
            joints=joints,
            floors=floors,
        )

    internal = SimpleNamespace(
        partition="internal_validation",
        indices=partition_indices["internal_validation"],
        sequence_ids=sequence_ids,
        language=language,
        scene_names=sequence_names,
    )
    selection = select_native_holdout(internal)
    if selection["sha256"] != NATIVE_SELECTION_SHA256 or selection["sequences"] != 32:
        raise ValueError("D2-Z sealed selection mismatch")
    selection_counts = np.zeros(4, dtype=np.int64)
    selection_sequences = []
    for index in selection["global_indices"]:
        sequence = int(sequence_ids[index])
        selection_sequences.append(sequence)
        frames = np.arange(int(starts[index]), int(ends[index]), 3)
        selection_counts += immutable_gt_near_ground_gate(
            np.asarray(joints[frames], dtype=np.float32),
            floors[sequence],
        ).sum(axis=0, dtype=np.int64)
    unique_selection_sequences = sorted(set(selection_sequences))
    selection_floor_values = np.asarray(
        [floors[value] for value in unique_selection_sequences], dtype=np.float64,
    )
    selection_floor_summary = (
        float(selection_floor_values.min()),
        float(np.median(selection_floor_values)),
        float(selection_floor_values.max()),
    )
    selection_checks = {
        "active_counts": {
            str(joint): int(value)
            for joint, value in zip(D2Z_FOOT_JOINTS, selection_counts.tolist())
        } == EXPECTED_SELECTION_COUNTS,
        "denominator_per_joint": len(selection["global_indices"]) * 14
        == EXPECTED_SELECTION_DENOMINATOR,
        "total_active": int(selection_counts.sum()) == EXPECTED_SELECTION_TOTAL_ACTIVE,
        "total_denominator": len(selection["global_indices"]) * 14 * 4
        == EXPECTED_SELECTION_TOTAL_DENOMINATOR,
        "floor_summary": np.allclose(
            selection_floor_summary, EXPECTED_SELECTION_FLOOR_SUMMARY, rtol=0.0, atol=1e-12,
        ),
    }
    failed = sorted(name for name, passed in selection_checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-Z sealed gate preflight mismatch: {failed}")

    result: Dict[str, object] = {
        "schema": D2Z_GATE_AUDIT_SCHEMA,
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "status": "completed",
        "seed": 42,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "python": str(Path(sys.executable).resolve()),
        "hostname": socket.gethostname(),
        "cpu_only": True,
        "floor_algorithm": D2Z_FLOOR_ALGORITHM,
        "floor_algorithm_file_sha256": D2Z_FLOOR_ALGORITHM_FILE_SHA256,
        "gate_previous_frame": "immutable_gt",
        "thresholds_m": {
            str(joint): threshold
            for joint, threshold in zip(D2Z_FOOT_JOINTS, D2Z_HEIGHT_THRESHOLDS_M)
        },
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "source_hashes": source_hashes,
        "partitions": partitions,
        "sealed_selection": {
            "selection_sha256": selection["sha256"],
            "sequences": selection["sequences"],
            "windows": len(selection["global_indices"]),
            "global_indices": selection["global_indices"],
            "active_counts": {
                str(joint): int(value)
                for joint, value in zip(D2Z_FOOT_JOINTS, selection_counts.tolist())
            },
            "denominator_per_joint": EXPECTED_SELECTION_DENOMINATOR,
            "total_active": int(selection_counts.sum()),
            "total_denominator": EXPECTED_SELECTION_TOTAL_DENOMINATOR,
            "active_fraction": float(
                selection_counts.sum() / EXPECTED_SELECTION_TOTAL_DENOMINATOR
            ),
            "floor_min_median_max_m": list(selection_floor_summary),
            "checks": selection_checks,
        },
        "checkpoint_loads": 0,
        "optimizer_created": False,
        "training_updates": 0,
        "official_test_used": False,
        "checkpoint_selection": False,
        "consistency_authorized": False,
    }
    result["payload_sha256"] = _canonical_sha256(result)
    return result


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-Z gate audit run id must be {RUN_ID}")
    if Path(sys.executable).resolve() != args.python.resolve():
        raise RuntimeError("D2-Z gate audit --python does not identify the active interpreter")
    if args.python.resolve() != Path(EXPECTED_PYTHON).resolve():
        raise RuntimeError(f"D2-Z gate audit requires {EXPECTED_PYTHON}")
    if socket.gethostname() != "ubuntu":
        raise RuntimeError("D2-Z gate audit is restricted to the authority host ubuntu")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("D2-Z gate audit requires a clean worktree")
    config = resolved_config(args)
    _exclusive_json(args.resolved_config.resolve(), config)
    preflight = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "passed": True,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_clean": True,
        "hostname": socket.gethostname(),
        "python": str(Path(sys.executable).resolve()),
        "python_matches": True,
        "cpu_only": True,
        "floor_algorithm_sha256_matches": (
            sha256_file(REPO / "code/eval_metrics.py")
            == D2Z_FLOOR_ALGORITHM_FILE_SHA256
        ),
        "split_sha256_matches": (
            sha256_file(REPO / "experiments/splits/omomo_hoi_train_validation_seed42.json")
            == EXPECTED_SPLIT_SHA256
        ),
    }
    if not all(
        value for key, value in preflight.items()
        if key.endswith("_matches") or key == "git_clean"
    ):
        raise RuntimeError("D2-Z gate audit preflight failed")
    _exclusive_json(args.preflight.resolve(), preflight)
    _exclusive_json(args.output.resolve(), build_audit())


if __name__ == "__main__":
    main()
