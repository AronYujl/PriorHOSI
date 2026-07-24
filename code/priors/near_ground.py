"""Fail-closed D2-Z immutable-GT near-ground gate contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


D2Z_GATE_AUDIT_SCHEMA = "d2z-immutable-gt-near-ground-gate-audit-v1"
D2Z_GATE_AUDIT_RUN_ID = "p1-hoi-d2z-gate-audit-s42-20260724"
D2Z_FLOOR_ALGORITHM = "code/eval_metrics.py::determine_floor_height_and_contacts"
D2Z_FLOOR_ALGORITHM_FILE_SHA256 = (
    "445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547"
)
D2Z_FOOT_JOINTS = (7, 8, 10, 11)
D2Z_HEIGHT_THRESHOLDS_M = (0.08, 0.08, 0.04, 0.04)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_gt_near_ground_gate(
    sampled_aligned_joints: np.ndarray,
    floor_height_m: float,
) -> np.ndarray:
    """Return the 14x4 gate for residuals at frames 2..15.

    Each residual is gated by its immutable GT previous sampled frame (1..14).
    """
    joints = np.asarray(sampled_aligned_joints)
    if joints.shape != (16, 24, 3):
        raise ValueError(f"D2-Z gate expects sampled aligned joints [16,24,3], got {joints.shape}")
    floor = float(floor_height_m)
    if not math.isfinite(floor):
        raise ValueError("D2-Z floor height must be finite")
    if not np.isfinite(joints).all():
        raise ValueError("D2-Z sampled aligned joints contain non-finite values")
    previous_heights = joints[1:-1, D2Z_FOOT_JOINTS, 1]
    thresholds = np.asarray(D2Z_HEIGHT_THRESHOLDS_M, dtype=previous_heights.dtype)
    gate = previous_heights - floor < thresholds.reshape(1, 4)
    if gate.shape != (14, 4):
        raise AssertionError(f"unexpected D2-Z gate shape: {gate.shape}")
    return np.asarray(gate, dtype=np.bool_)


def _expected_thresholds() -> Dict[str, float]:
    return {
        str(joint): threshold
        for joint, threshold in zip(D2Z_FOOT_JOINTS, D2Z_HEIGHT_THRESHOLDS_M)
    }


def load_gate_audit_floors(
    audit_path: str,
    expected_sha256: str,
    *,
    partition: str,
    expected_sequence_indices: Iterable[int],
    expected_split_sha256: str,
) -> Mapping[int, float]:
    """Validate a complete audit and return its partition's immutable floor map."""
    path = Path(audit_path).resolve()
    configured_sha256 = str(expected_sha256)
    if len(configured_sha256) != 64:
        raise ValueError("D2-Z gate audit SHA-256 must contain exactly 64 hex characters")
    try:
        int(configured_sha256, 16)
    except ValueError as error:
        raise ValueError("D2-Z gate audit SHA-256 is not hexadecimal") from error
    actual_sha256 = sha256_file(path)
    if actual_sha256 != configured_sha256:
        raise ValueError(
            f"D2-Z gate audit SHA-256 mismatch: {actual_sha256} != {configured_sha256}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "schema": value.get("schema") == D2Z_GATE_AUDIT_SCHEMA,
        "run_id": value.get("run_id") == D2Z_GATE_AUDIT_RUN_ID,
        "seed": value.get("seed") == 42,
        "floor_algorithm": value.get("floor_algorithm") == D2Z_FLOOR_ALGORITHM,
        "floor_algorithm_file_sha256": (
            value.get("floor_algorithm_file_sha256") == D2Z_FLOOR_ALGORITHM_FILE_SHA256
        ),
        "gate_previous_frame": value.get("gate_previous_frame") == "immutable_gt",
        "thresholds_m": value.get("thresholds_m") == _expected_thresholds(),
        "split_sha256": value.get("split_sha256") == str(expected_split_sha256),
    }
    failed = sorted(name for name, passed in exact.items() if not passed)
    if failed:
        raise ValueError(f"D2-Z gate audit contract mismatch: {failed}")
    partitions = value.get("partitions")
    if not isinstance(partitions, dict) or partition not in partitions:
        raise ValueError(f"D2-Z gate audit is missing partition {partition!r}")
    record = partitions[partition]
    if not isinstance(record, dict) or not isinstance(record.get("floors_m"), dict):
        raise ValueError(f"D2-Z gate audit partition {partition!r} has no floor map")
    floor_map: Dict[int, float] = {}
    for raw_sequence, raw_floor in record["floors_m"].items():
        sequence = int(raw_sequence)
        floor = float(raw_floor)
        if str(sequence) != str(raw_sequence) or not math.isfinite(floor):
            raise ValueError(f"invalid D2-Z floor entry {raw_sequence!r}: {raw_floor!r}")
        floor_map[sequence] = floor
    expected = set(int(value) for value in expected_sequence_indices)
    if set(floor_map) != expected:
        missing = sorted(expected - set(floor_map))
        extra = sorted(set(floor_map) - expected)
        raise ValueError(
            f"D2-Z gate audit sequence coverage mismatch for {partition}: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if record.get("sequence_indices") != sorted(expected):
        raise ValueError(f"D2-Z gate audit ordered sequence indices mismatch for {partition}")
    if int(record.get("nonfinite_floor_count", -1)) != 0:
        raise ValueError(f"D2-Z gate audit reports non-finite floors for {partition}")
    return floor_map
