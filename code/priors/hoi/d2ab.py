"""D2-AB predicted-state support/no-slip metadata, dataset and loss."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch

from .data import PriorWindowDataset
from .losses import (
    D2X_FOOT_XZ_VELOCITY_SLOTS,
    _fk_positions,
    _velocity_residuals,
    hoi_training_losses,
)
from ..core.representation import REPRESENTATION


D2AB_METADATA_SCHEMA = "d2ab-predicted-support-no-slip-train-metadata-v1"
D2AB_METADATA_RUN_ID = "p1-hoi-d2ab-support-metadata-s42-20260725"
D2AB_FOOT_JOINTS: Tuple[int, ...] = (7, 8, 10, 11)
D2AB_LEFT_PAIR: Tuple[int, ...] = (7, 10)
D2AB_RIGHT_PAIR: Tuple[int, ...] = (8, 11)
D2AB_PAIR_INDEX = (0, 1, 0, 1)
D2AB_CLEARANCE_SCALE_M = 0.03925712490454316
D2AB_POSITION_RANGE_X_M = 6.658331632614136
D2AB_POSITION_RANGE_Z_M = 6.975271224975586
D2AB_VELOCITY_SCALE_S_PER_M = 0.029363068377844033
D2AB_SAMPLE_INTERVAL_S = 0.1
D2AB_SOURCE_FILES = (
    "human_joints_aligned.npy",
    "start_idx.npy",
    "end_idx.npy",
    "norm.npy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear_quantile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("D2-AB quantile input must be finite and non-empty")
    try:
        return float(np.quantile(values, quantile, method="linear"))
    except TypeError:  # pragma: no cover - compatibility with older NumPy
        return float(np.quantile(values, quantile, interpolation="linear"))


def sequence_floor_and_clearance(
    joints: np.ndarray,
    start: int,
    end: int,
) -> Tuple[float, np.ndarray]:
    """Return the toe-5% floor and four-foot clearances for one raw sequence."""
    if joints.ndim != 3 or joints.shape[1:] != (28, 3):
        raise ValueError(f"expected raw aligned joints [T,28,3], got {joints.shape}")
    if not (0 <= int(start) < int(end) <= joints.shape[0]):
        raise ValueError(f"invalid sequence bounds: {start}, {end}, {joints.shape[0]}")
    toe_y = np.asarray(joints[int(start):int(end), (10, 11), 1], dtype=np.float64)
    floor = _linear_quantile(toe_y.reshape(-1), 0.05)
    foot_y = np.asarray(
        joints[int(start):int(end), D2AB_FOOT_JOINTS, 1],
        dtype=np.float64,
    )
    clearance = (foot_y - floor).reshape(-1)
    if not np.isfinite(clearance).all():
        raise ValueError("D2-AB clearance contains non-finite values")
    return floor, clearance


def _expected_split_sha256(split_path: Path) -> str:
    return sha256_file(split_path.resolve())


def validate_metadata(
    path: Path,
    expected_sha256: str,
    *,
    split_path: Path,
    expected_train_sequence_indices: Iterable[int],
) -> Mapping[str, object]:
    """Validate the immutable train-support metadata and return its payload."""
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"D2-AB support metadata is missing: {path}")
    actual = sha256_file(path)
    if actual != str(expected_sha256):
        raise ValueError(
            f"D2-AB support metadata SHA-256 mismatch: {actual} != {expected_sha256}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": value.get("schema") == D2AB_METADATA_SCHEMA,
        "run_id": value.get("run_id") == D2AB_METADATA_RUN_ID,
        "seed": value.get("seed") == 42,
        "partition": value.get("partition") == "train",
        "split_sha256": (
            value.get("split", {}).get("sha256")
            == _expected_split_sha256(split_path)
        ),
        "clearance_scale_m": (
            float(value.get("constants", {}).get("clearance_scale_m", float("nan")))
            == D2AB_CLEARANCE_SCALE_M
        ),
        "position_range_x_m": (
            float(value.get("constants", {}).get("position_range_x_m", float("nan")))
            == D2AB_POSITION_RANGE_X_M
        ),
        "position_range_z_m": (
            float(value.get("constants", {}).get("position_range_z_m", float("nan")))
            == D2AB_POSITION_RANGE_Z_M
        ),
        "velocity_scale_s_per_m": (
            float(value.get("constants", {}).get("velocity_scale_s_per_m", float("nan")))
            == D2AB_VELOCITY_SCALE_S_PER_M
        ),
        "sample_interval_s": (
            float(value.get("constants", {}).get("sample_interval_s", float("nan")))
            == D2AB_SAMPLE_INTERVAL_S
        ),
    }
    failed = sorted(key for key, passed in expected.items() if not passed)
    if failed:
        raise ValueError(f"D2-AB support metadata contract mismatch: {failed}")
    expected_sequences = sorted(int(item) for item in expected_train_sequence_indices)
    floors = value.get("floors_m")
    if not isinstance(floors, dict):
        raise ValueError("D2-AB support metadata has no floor map")
    floor_keys = sorted(int(key) for key in floors)
    if floor_keys != expected_sequences:
        raise ValueError(
            "D2-AB support metadata train sequence coverage mismatch: "
            f"{len(floor_keys)} != {len(expected_sequences)}"
        )
    if any(str(key) != str(int(key)) for key in floors):
        raise ValueError("D2-AB floor keys must be canonical decimal sequence indices")
    if any(not math.isfinite(float(floors[str(key)])) for key in expected_sequences):
        raise ValueError("D2-AB floor map contains non-finite values")
    return value


def load_train_floor_map(
    path: Path,
    expected_sha256: str,
    *,
    split_path: Path,
    expected_train_sequence_indices: Iterable[int],
) -> Dict[int, float]:
    payload = validate_metadata(
        path,
        expected_sha256,
        split_path=split_path,
        expected_train_sequence_indices=expected_train_sequence_indices,
    )
    return {
        int(key): float(value)
        for key, value in payload["floors_m"].items()
    }


def compute_partition_floor_map(
    joints: np.ndarray,
    sequence_starts: np.ndarray,
    sequence_ends: np.ndarray,
    sequence_indices: Iterable[int],
) -> Dict[int, float]:
    """Compute held-out floors with the same formula, without using test data."""
    result: Dict[int, float] = {}
    for raw_sequence in sorted(int(item) for item in sequence_indices):
        floor, _ = sequence_floor_and_clearance(
            joints,
            int(sequence_starts[raw_sequence]),
            int(sequence_ends[raw_sequence]),
        )
        result[raw_sequence] = floor
    return result


class D2ABPriorWindowDataset(PriorWindowDataset):
    """Add the detached per-sequence floor required by the D2-AB loss."""

    def __init__(
        self,
        repo_root: str,
        expert: str,
        partition: str = "train",
        limit: int = 0,
        split_manifest: Optional[str] = None,
        *,
        support_metadata_path: str,
        support_metadata_sha256: str,
    ) -> None:
        if expert != "hoi":
            raise ValueError("D2-AB dataset is restricted to HOI")
        if split_manifest in (None, "", False):
            raise ValueError("D2-AB dataset requires the locked HOI split manifest")
        repo = Path(repo_root).resolve()
        split_path = Path(str(split_manifest))
        if not split_path.is_absolute():
            split_path = repo / split_path
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("algorithm") != "omomo-sequence-sha256-seed42-v1":
            raise ValueError("D2-AB dataset split algorithm mismatch")
        expected_sequences = sorted(
            int(value) for value in split[partition]["sequence_indices"]
        )
        super().__init__(
            repo_root,
            expert,
            partition=partition,
            limit=limit,
            split_manifest=str(split_path),
        )
        metadata_path = Path(str(support_metadata_path))
        if not metadata_path.is_absolute():
            metadata_path = repo / metadata_path
        self.support_metadata_path = metadata_path.resolve()
        self.support_metadata_sha256 = str(support_metadata_sha256)
        if partition == "train":
            self.d2ab_floor_by_sequence = load_train_floor_map(
                self.support_metadata_path,
                self.support_metadata_sha256,
                split_path=split_path,
                expected_train_sequence_indices=expected_sequences,
            )
        elif partition == "internal_validation":
            self.d2ab_floor_by_sequence = compute_partition_floor_map(
                self.joints,
                self.seq_starts,
                self.seq_ends,
                expected_sequences,
            )
        else:
            raise ValueError("D2-AB supports train and internal_validation only")

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        index = int(self.indices[item])
        sequence = int(self.sequence_ids[index])
        floor = self.d2ab_floor_by_sequence.get(sequence)
        if floor is None or not math.isfinite(float(floor)):
            raise ValueError(f"missing finite D2-AB floor for sequence {sequence}")
        result["d2ab_floor_m"] = torch.tensor(float(floor), dtype=torch.float32)
        return result


def _d2ab_physical_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    predicted_fk_positions: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    floor_m: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Construct predicted/GT physical foot velocities and predicted support."""
    if prediction.shape != target.shape or prediction.shape[-1] != REPRESENTATION.dimension:
        raise ValueError("D2-AB expects matching [B,16,232] prediction and target")
    if predicted_fk_positions.shape[:2] != prediction.shape[:2] or predicted_fk_positions.shape[-2:] != (24, 3):
        raise ValueError(
            f"D2-AB FK positions shape mismatch: {tuple(predicted_fk_positions.shape)}"
        )
    if floor_m.shape != (prediction.shape[0],):
        raise ValueError(
            f"D2-AB floor shape {tuple(floor_m.shape)} != {(prediction.shape[0],)}"
        )
    if floor_m.requires_grad:
        raise ValueError("D2-AB floor metadata must be detached")
    scale = (position_maximum - position_minimum).reshape(1, 1, 1, 3)
    minimum = position_minimum.reshape(1, 1, 1, 3)
    target_clean = target.detach()
    target_positions = (
        (target_clean[..., :84].reshape(*target.shape[:2], 28, 3) + 1.0) * scale / 2.0
        + minimum
    )
    foot_target = target_positions[:, REPRESENTATION.history_frames:, D2AB_FOOT_JOINTS]
    foot_target_previous = target_positions[
        :, REPRESENTATION.history_frames - 1:-1, D2AB_FOOT_JOINTS
    ]
    foot_predicted = predicted_fk_positions[:, REPRESENTATION.history_frames:, D2AB_FOOT_JOINTS]
    foot_predicted_previous = torch.cat(
        (
            target_positions[
                :, REPRESENTATION.history_frames - 1:REPRESENTATION.history_frames,
                D2AB_FOOT_JOINTS,
            ],
            predicted_fk_positions[
                :, REPRESENTATION.history_frames:-1, D2AB_FOOT_JOINTS
            ],
        ),
        dim=1,
    )
    predicted_velocity = (
        foot_predicted[..., (0, 2)] - foot_predicted_previous[..., (0, 2)]
    ) / D2AB_SAMPLE_INTERVAL_S
    target_velocity = (
        foot_target[..., (0, 2)] - foot_target_previous[..., (0, 2)]
    ) / D2AB_SAMPLE_INTERVAL_S
    floor = floor_m.to(
        device=predicted_fk_positions.device,
        dtype=predicted_fk_positions.dtype,
    ).reshape(-1, 1, 1, 1)
    # The pair support is a log-mean-exp soft minimum over ankle/toe heights.
    previous_foot_y = foot_predicted_previous[..., 1][..., (0, 2, 1, 3)]
    previous_foot_y = previous_foot_y.reshape(
        prediction.shape[0],
        prediction.shape[1] - REPRESENTATION.history_frames,
        2,
        2,
    )
    pair_distance = -D2AB_CLEARANCE_SCALE_M * (
        torch.logsumexp(
            -(previous_foot_y - floor) / D2AB_CLEARANCE_SCALE_M,
            dim=-1,
        ) - math.log(2.0)
    )
    support_pair = torch.sigmoid(-pair_distance / D2AB_CLEARANCE_SCALE_M)
    support_by_joint = support_pair[..., torch.as_tensor(
        D2AB_PAIR_INDEX, device=support_pair.device, dtype=torch.long,
    )]
    return {
        "target_positions": target_positions,
        "predicted_foot_previous": foot_predicted_previous,
        "predicted_velocity": predicted_velocity,
        "target_velocity": target_velocity,
        "pair_distance": pair_distance,
        "support_pair": support_pair,
        "support_by_joint": support_by_joint,
    }


def d2ab_velocity_loss(
    predicted_residual: torch.Tensor,
    target_residual: torch.Tensor,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    support_by_joint: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace exactly the eight routed element errors with the D2-AB residual."""
    if predicted_residual.shape != target_residual.shape or predicted_residual.shape[-1] != 87:
        raise ValueError("D2-AB normalized residuals must have matching [...,87] shape")
    expected = predicted_residual.shape[:-1] + (4, 2)
    if predicted_velocity.shape != expected or target_velocity.shape != expected:
        raise ValueError(
            f"D2-AB physical velocity shape mismatch: "
            f"{tuple(predicted_velocity.shape)}/{tuple(target_velocity.shape)} != {expected}"
        )
    if support_by_joint.shape != predicted_residual.shape[:-1] + (4,):
        raise ValueError("D2-AB support-by-joint shape mismatch")
    if torch.is_grad_enabled() and not support_by_joint.requires_grad:
        # Predicted support must remain differentiable; this catches accidental detach.
        raise ValueError("D2-AB predicted support must require gradients")
    physical_residual = predicted_velocity - (
        1.0 - support_by_joint.unsqueeze(-1)
    ) * target_velocity
    routed_error = D2AB_VELOCITY_SCALE_S_PER_M * physical_residual
    normalized_error = predicted_residual - target_residual
    replaced = normalized_error.clone()
    replaced[..., list(D2X_FOOT_XZ_VELOCITY_SLOTS)] = routed_error.reshape(
        *routed_error.shape[:-2], 8,
    )
    return replaced.square().mean(), physical_residual


def d2ab_hoi_training_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    goals: torch.Tensor,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    terminal_window: torch.Tensor,
    rest_object_points: torch.Tensor,
    world_to_local_rotation: torch.Tensor,
    object_rotation_reference: torch.Tensor,
    floor_m: torch.Tensor,
    *,
    fk_weight: float,
    object_surface_weight: float,
    velocity_weight: float,
    goal_weight: float,
) -> Dict[str, torch.Tensor]:
    """Keep every D2-X loss and replace only its eight routed velocity errors."""
    losses = hoi_training_losses(
        prediction,
        target,
        goals,
        rest_human_offsets,
        parents_24,
        position_minimum,
        position_maximum,
        object_minimum,
        object_maximum,
        terminal_window,
        rest_object_points,
        world_to_local_rotation,
        object_rotation_reference,
        fk_weight=fk_weight,
        object_surface_weight=object_surface_weight,
        velocity_weight=velocity_weight,
        goal_weight=goal_weight,
        fk_foot_temporal_routing=True,
        routed_foot_residual_multiplier=1.0,
    )
    scale = (position_maximum - position_minimum).reshape(1, 1, 1, 3)
    minimum = position_minimum.reshape(1, 1, 1, 3)
    predicted_positions = (
        (prediction[..., :84].reshape(*prediction.shape[:2], 28, 3) + 1.0)
        * scale / 2.0
        + minimum
    )
    # Use the same PyTorch3D conversion as the locked base loss.
    from pytorch3d import transforms

    predicted_rotation = transforms.rotation_6d_to_matrix(
        prediction[..., 84:216].reshape(*prediction.shape[:2], 22, 6)
    )
    predicted_fk = _fk_positions(
        predicted_positions[..., 0, :],
        predicted_rotation,
        rest_human_offsets,
        parents_24,
    )
    predicted_residual, target_residual = _velocity_residuals(
        prediction,
        target,
        predicted_fk,
        position_minimum,
        position_maximum,
        fk_foot_temporal_routing=True,
    )
    terms = _d2ab_physical_terms(
        prediction,
        target,
        predicted_fk,
        position_minimum,
        position_maximum,
        floor_m,
    )
    losses["velocity"], physical_residual = d2ab_velocity_loss(
        predicted_residual,
        target_residual,
        terms["predicted_velocity"],
        terms["target_velocity"],
        terms["support_by_joint"],
    )
    losses["total"] = (
        losses["reconstruction"]
        + float(fk_weight) * losses["fk"]
        + float(object_surface_weight) * losses["object_surface"]
        + float(velocity_weight) * losses["velocity"]
        + float(goal_weight) * losses["object_goal"]
    )
    losses["d2ab_support_pair"] = terms["support_pair"]
    losses["d2ab_support_by_joint"] = terms["support_by_joint"]
    losses["d2ab_pair_distance"] = terms["pair_distance"]
    losses["d2ab_physical_residual"] = physical_residual
    losses["d2ab_predicted_velocity"] = terms["predicted_velocity"]
    losses["d2ab_target_velocity"] = terms["target_velocity"]
    return losses
