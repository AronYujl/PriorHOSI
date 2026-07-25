#!/usr/bin/env python3
"""Build the immutable CPU-only D2-AB train support metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "code"))

from priors.d2ab import (  # noqa: E402
    D2AB_CLEARANCE_SCALE_M,
    D2AB_METADATA_RUN_ID,
    D2AB_METADATA_SCHEMA,
    D2AB_POSITION_RANGE_X_M,
    D2AB_POSITION_RANGE_Z_M,
    D2AB_SAMPLE_INTERVAL_S,
    D2AB_SOURCE_FILES,
    D2AB_VELOCITY_SCALE_S_PER_M,
    sequence_floor_and_clearance,
    sha256_file,
)


EXPECTED_FLOOR_SUMMARY = (
    -0.004783304338343441,
    0.0353932767175138,
    0.06221588589251041,
)
EXPECTED_CLEARANCE_MEDIAN = D2AB_CLEARANCE_SCALE_M


def _exclusive_json(path: Path, value: object) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    return {
        str(q): float(np.quantile(values, q, method="linear"))
        for q in (0.0, 0.05, 0.5, 0.95, 1.0)
    }


def build(repo: Path, split_path: Path) -> Dict[str, object]:
    train = repo / "data" / "train"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("algorithm") != "omomo-sequence-sha256-seed42-v1":
        raise ValueError("unexpected HOI split algorithm")
    sequence_indices = sorted(int(value) for value in split["train"]["sequence_indices"])
    if len(sequence_indices) != 4088:
        raise ValueError(f"expected 4088 train sequences, got {len(sequence_indices)}")
    joints = np.load(train / "human_joints_aligned.npy", mmap_mode="r")
    starts = np.load(train / "start_idx.npy", mmap_mode="r")
    ends = np.load(train / "end_idx.npy", mmap_mode="r")
    norm = np.load(train / "norm.npy")
    floors: Dict[str, float] = {}
    positive_values: List[np.ndarray] = []
    for sequence in sequence_indices:
        floor, clearance = sequence_floor_and_clearance(
            joints,
            int(starts[sequence]),
            int(ends[sequence]),
        )
        floors[str(sequence)] = floor
        positive = clearance[clearance > 0.0]
        if positive.size:
            positive_values.append(positive)
    pooled_positive = np.concatenate(positive_values)
    floor_values = np.asarray(list(floors.values()), dtype=np.float64)
    floor_summary = tuple(float(np.quantile(floor_values, q, method="linear")) for q in (0.0, 0.5, 1.0))
    clearance_median = float(np.quantile(pooled_positive, 0.5, method="linear"))
    if not np.allclose(floor_summary, EXPECTED_FLOOR_SUMMARY, rtol=0.0, atol=1e-15):
        raise ValueError(f"floor summary drifted: {floor_summary}")
    if not np.isclose(clearance_median, EXPECTED_CLEARANCE_MEDIAN, rtol=0.0, atol=1e-15):
        raise ValueError(f"clearance median drifted: {clearance_median}")
    source_files = {}
    for name in D2AB_SOURCE_FILES:
        path = train / name
        source_files[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": D2AB_METADATA_SCHEMA,
        "run_id": D2AB_METADATA_RUN_ID,
        "status": "completed",
        "seed": 42,
        "partition": "train",
        "split": {
            "path": str(split_path.resolve()),
            "sha256": sha256_file(split_path),
            "algorithm": split["algorithm"],
            "sequence_count": len(sequence_indices),
            "sequence_indices": sequence_indices,
        },
        "source_files": source_files,
        "constants": {
            "clearance_scale_m": D2AB_CLEARANCE_SCALE_M,
            "position_range_x_m": D2AB_POSITION_RANGE_X_M,
            "position_range_z_m": D2AB_POSITION_RANGE_Z_M,
            "velocity_scale_s_per_m": D2AB_VELOCITY_SCALE_S_PER_M,
            "sample_interval_s": D2AB_SAMPLE_INTERVAL_S,
            "floor_quantile": 0.05,
            "floor_quantile_method": "linear",
            "toe_joint_indices": [10, 11],
            "foot_joint_indices": [7, 8, 10, 11],
        },
        "floors_m": floors,
        "statistics": {
            "floor_min_m": floor_summary[0],
            "floor_median_m": floor_summary[1],
            "floor_max_m": floor_summary[2],
            "strictly_positive_clearance_count": int(pooled_positive.size),
            "strictly_positive_clearance_quantiles_m": _quantiles(pooled_positive),
            "strictly_positive_clearance_median_m": clearance_median,
            "nonfinite_floor_count": 0,
            "nonfinite_clearance_count": 0,
        },
        "official_test_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument(
        "--split",
        type=Path,
        default=REPO / "experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.repo_root.resolve(), args.split.resolve())
    _exclusive_json(args.output, value)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "schema": value["schema"],
        "sequence_count": value["split"]["sequence_count"],
        "strictly_positive_clearance_count": value["statistics"]["strictly_positive_clearance_count"],
        "clearance_scale_m": value["constants"]["clearance_scale_m"],
        "sha256": sha256_file(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
