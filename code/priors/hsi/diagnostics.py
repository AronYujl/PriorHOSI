"""Registered causal diagnostics for HSIPrior inference."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from priors.hsi.metrics import StitchedSequence

FUTURE_OCC_OFFSETS: Tuple[int, ...] = (5, 10, 15)
FUTURE_OCC_MODES: Tuple[str, ...] = (
    "predicted",
    "gt_crop",
    "gt_coordinate",
    "gt_both",
)
TEACHER_FORCED_TIMESTEPS: Tuple[int, ...] = (498, 250, 50)


def validate_future_occ_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in FUTURE_OCC_MODES:
        raise ValueError(
            "hsi_future_occ_mode must be one of %s, got %r"
            % (", ".join(FUTURE_OCC_MODES), mode)
        )
    return mode


def select_future_occ_centers(
    predicted_local: torch.Tensor,
    oracle_local: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return independent crop-query and coordinate centers for the 2x2."""
    mode = validate_future_occ_mode(mode)
    expected = (int(predicted_local.shape[0]), len(FUTURE_OCC_OFFSETS), 3)
    if tuple(predicted_local.shape) != expected:
        raise ValueError("predicted future centers must have shape %s" % (expected,))
    if tuple(oracle_local.shape) != expected:
        raise ValueError("oracle future centers must have shape %s" % (expected,))
    if not bool(torch.isfinite(oracle_local).all()):
        raise ValueError("oracle future centers contain non-finite values")
    gt_crop = mode in ("gt_crop", "gt_both")
    gt_coordinate = mode in ("gt_coordinate", "gt_both")
    return (
        oracle_local if gt_crop else predicted_local,
        oracle_local if gt_coordinate else predicted_local,
    )


class FutureOccCenterTelemetry:
    """GPU-side accumulator for predicted-to-GT center errors."""

    def __init__(self, timesteps: int, device):
        self.timesteps = int(timesteps)
        shape = (self.timesteps, len(FUTURE_OCC_OFFSETS))
        self.count = torch.zeros(shape, dtype=torch.int64, device=device)
        self.total = torch.zeros(shape, dtype=torch.float64, device=device)
        self.maximum = torch.zeros(shape, dtype=torch.float64, device=device)

    def record(
        self,
        timestep: int,
        predicted_local: torch.Tensor,
        oracle_local: torch.Tensor,
    ) -> None:
        timestep = int(timestep)
        if not 0 <= timestep < self.timesteps:
            raise ValueError("diffusion timestep %d outside [0,%d)" % (timestep, self.timesteps))
        error = torch.linalg.vector_norm(
            predicted_local.to(torch.float64) - oracle_local.to(torch.float64), dim=-1
        )
        self.count[timestep] += int(error.shape[0])
        self.total[timestep] += error.sum(dim=0)
        self.maximum[timestep] = torch.maximum(self.maximum[timestep], error.max(dim=0).values)

    def report(self) -> Dict[str, object]:
        count = self.count.cpu()
        total = self.total.cpu()
        maximum = self.maximum.cpu()
        by_timestep = {}
        for timestep in range(self.timesteps):
            rows = []
            for position, offset in enumerate(FUTURE_OCC_OFFSETS):
                n = int(count[timestep, position])
                rows.append(
                    {
                        "offset": offset,
                        "count": n,
                        "mean_l2_m": None if n == 0 else float(total[timestep, position] / n),
                        "max_l2_m": None if n == 0 else float(maximum[timestep, position]),
                    }
                )
            by_timestep[str(timestep)] = rows
        return {"offsets": list(FUTURE_OCC_OFFSETS), "by_timestep": by_timestep}


def future_occ_motion_diagnostics(
    joints: StitchedSequence,
    *,
    fps: float,
    root_joint: int = 0,
) -> Dict[str, float]:
    """Per-episode FK acceleration and pelvis displacement diagnostics."""
    if not isinstance(joints, StitchedSequence):
        raise TypeError("future_occ_motion_diagnostics needs a StitchedSequence")
    if not fps > 0:
        raise ValueError("fps must be positive")
    positions = joints.frames
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("joints must have shape [T,J,3]")
    if not 0 <= int(root_joint) < int(positions.shape[1]):
        raise ValueError("root_joint out of range")
    acceleration = (positions[2:] - 2.0 * positions[1:-1] + positions[:-2]) * float(fps) ** 2
    magnitude = torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=-1)

    first_start = int(joints.history_frames)
    first_centres = [k for k in (first_start, first_start + 1) if 1 <= k < len(positions) - 1]
    seam_centres = [k for seam in joints.seams for k in (seam, seam + 1) if 1 <= k < len(positions) - 1]

    def mean_at(centres: Sequence[int]) -> float:
        if not centres:
            return float("nan")
        return float(magnitude[[k - 1 for k in centres]].mean())

    pelvis = positions[:, int(root_joint)]
    step = torch.linalg.vector_norm(pelvis[1:] - pelvis[:-1], dim=-1)
    return {
        "first_window_first2_fk_acc_mps2": mean_at(first_centres),
        "seam_first2_fk_acc_mps2": mean_at(seam_centres),
        "all_window_first2_fk_acc_mps2": mean_at(first_centres + seam_centres),
        "pelvis_path_length_m": float(step.sum()) if step.numel() else 0.0,
        "pelvis_net_displacement_m": float(torch.linalg.vector_norm(pelvis[-1] - pelvis[0])),
    }


def teacher_forced_boundary_metrics(
    predicted_joints: torch.Tensor,
    target_joints: torch.Tensor,
    predicted_repr: torch.Tensor,
    target_repr: torch.Tensor,
    *,
    fps: float,
) -> Dict[str, torch.Tensor]:
    """Per-sample boundary metrics for one teacher-forced x0 prediction."""
    if tuple(predicted_joints.shape) != tuple(target_joints.shape):
        raise ValueError("predicted and target joints must have the same shape")
    if predicted_joints.ndim != 4 or predicted_joints.shape[1] < 4:
        raise ValueError("joints must have shape [B,T,J,3] with T >= 4")
    if tuple(predicted_repr.shape) != tuple(target_repr.shape):
        raise ValueError("predicted and target representations must have the same shape")
    if predicted_repr.ndim != 3 or predicted_repr.shape[1] < 4 or predicted_repr.shape[2] < 216:
        raise ValueError("representations must have shape [B,T,D] with T >= 4 and D >= 216")
    if not fps > 0:
        raise ValueError("fps must be positive")

    scale = float(fps) ** 2

    def acceleration(sequence: torch.Tensor) -> torch.Tensor:
        return (sequence[:, 2:4] - 2.0 * sequence[:, 1:3] + sequence[:, :2]) * scale

    def reduce_acceleration(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value, dim=-1).mean(dim=(1, 2))

    clamped = torch.cat((target_joints[:, :2], predicted_joints[:, 2:4]), dim=1)
    gt_acc = acceleration(target_joints[:, :4])
    clamped_acc = acceleration(clamped)
    internal_acc = acceleration(predicted_joints[:, :4])

    history_position = (
        predicted_repr[:, :2, :84] - target_repr[:, :2, :84]
    ).reshape(predicted_repr.shape[0], 2, 28, 3)
    history_rotation = predicted_repr[:, :2, 84:216] - target_repr[:, :2, 84:216]
    joint_error = torch.linalg.vector_norm(
        predicted_joints - target_joints, dim=-1
    )

    return {
        "gt_first2_fk_acc_mps2": reduce_acceleration(gt_acc),
        "clamped_first2_fk_acc_mps2": reduce_acceleration(clamped_acc),
        "internal_first2_fk_acc_mps2": reduce_acceleration(internal_acc),
        "clamped_a1_fk_acc_mps2": reduce_acceleration(clamped_acc[:, :1]),
        "clamped_a2_fk_acc_mps2": reduce_acceleration(clamped_acc[:, 1:2]),
        "pelvis_frame2_error_m": joint_error[:, 2, 0],
        "pelvis_frame3_error_m": joint_error[:, 3, 0],
        "history_pelvis_error_m": joint_error[:, :2, 0].mean(dim=1),
        "history_fk_joint_error_m": joint_error[:, :2].mean(dim=(1, 2)),
        "history_position_error_m": torch.linalg.vector_norm(
            history_position, dim=-1
        ).mean(dim=(1, 2)),
        "history_rotation_channel_mae": history_rotation.abs().mean(dim=(1, 2)),
    }


def summarize_teacher_forced_boundary(
    holdout_records: Sequence[Mapping[str, object]],
    train_records: Sequence[Mapping[str, object]],
    stratum_weights: Mapping[str, float],
    *,
    timesteps: Sequence[int] = TEACHER_FORCED_TIMESTEPS,
    seed: int = 42,
    replicates: int = 10000,
) -> Dict[str, object]:
    """Summarize the frozen D1 cohorts and apply the registered decisions."""
    timesteps = tuple(int(value) for value in timesteps)
    if not holdout_records or not train_records:
        raise ValueError("both teacher-forced cohorts must be non-empty")
    if int(replicates) < 1:
        raise ValueError("replicates must be positive")

    by_episode = defaultdict(list)
    episode_stratum = {}
    for record in holdout_records:
        episode_id = str(record["episode_id"])
        stratum = str(record["stratum"])
        by_episode[episode_id].append(record)
        episode_stratum[episode_id] = stratum

    strata = defaultdict(list)
    for episode_id in sorted(by_episode):
        strata[episode_stratum[episode_id]].append(episode_id)
    if set(strata) != set(stratum_weights):
        raise ValueError("holdout strata do not match the registered weights")

    metric_names = tuple(
        sorted(
            str(name)
            for name in holdout_records[0]["metrics"][str(timesteps[0])]
        )
    )

    def record_vector(record, timestep):
        metrics = record["metrics"][str(timestep)]
        return np.asarray([float(metrics[name]) for name in metric_names], dtype=np.float64)

    episode_vectors = {}
    for episode_id, records in by_episode.items():
        episode_vectors[episode_id] = {
            timestep: np.stack([record_vector(record, timestep) for record in records]).mean(axis=0)
            for timestep in timesteps
        }

    weights = {str(key): float(value) for key, value in stratum_weights.items()}

    def weighted_holdout(selected, timestep):
        total = np.zeros(len(metric_names), dtype=np.float64)
        for stratum, episode_ids in selected.items():
            total += weights[stratum] * np.stack(
                [episode_vectors[episode_id][timestep] for episode_id in episode_ids]
            ).mean(axis=0)
        return total

    selected_full = {stratum: tuple(ids) for stratum, ids in strata.items()}
    holdout_point = {
        timestep: weighted_holdout(selected_full, timestep) for timestep in timesteps
    }

    rng = np.random.default_rng(int(seed))
    digest = hashlib.sha256()
    holdout_boot = {
        timestep: np.empty((int(replicates), len(metric_names)), dtype=np.float64)
        for timestep in timesteps
    }
    ordered_strata = tuple(sorted(strata))
    for replicate in range(int(replicates)):
        selected = {}
        for stratum in ordered_strata:
            episode_ids = strata[stratum]
            indices = rng.integers(0, len(episode_ids), size=len(episode_ids), dtype=np.int64)
            digest.update(indices.tobytes())
            selected[stratum] = [episode_ids[index] for index in indices]
        for timestep in timesteps:
            holdout_boot[timestep][replicate] = weighted_holdout(selected, timestep)

    train_matrix = {
        timestep: np.stack([record_vector(record, timestep) for record in train_records])
        for timestep in timesteps
    }
    train_point = {timestep: values.mean(axis=0) for timestep, values in train_matrix.items()}
    train_boot = {
        timestep: np.empty((int(replicates), len(metric_names)), dtype=np.float64)
        for timestep in timesteps
    }
    for replicate in range(int(replicates)):
        indices = rng.integers(0, len(train_records), size=len(train_records), dtype=np.int64)
        for timestep in timesteps:
            train_boot[timestep][replicate] = train_matrix[timestep][indices].mean(axis=0)

    def metric_summary(point, boot):
        return {
            name: {
                "mean": float(point[index]),
                "ci": [
                    float(np.quantile(boot[:, index], 0.025)),
                    float(np.quantile(boot[:, index], 0.975)),
                ],
            }
            for index, name in enumerate(metric_names)
        }

    def derived(values):
        index = {name: position for position, name in enumerate(metric_names)}
        gt = values[..., index["gt_first2_fk_acc_mps2"]]
        clamped = values[..., index["clamped_first2_fk_acc_mps2"]]
        internal = values[..., index["internal_first2_fk_acc_mps2"]]
        return {
            "clamped_over_gt": clamped / gt,
            "internal_over_gt": internal / gt,
            "history_clamp_closure": (clamped - internal) / (clamped - gt),
        }

    holdout_summary = {}
    train_summary = {}
    derived_holdout = {}
    for timestep in timesteps:
        holdout_summary[str(timestep)] = metric_summary(
            holdout_point[timestep], holdout_boot[timestep]
        )
        train_summary[str(timestep)] = metric_summary(train_point[timestep], train_boot[timestep])
        point_derived = derived(holdout_point[timestep])
        boot_derived = derived(holdout_boot[timestep])
        derived_holdout[str(timestep)] = {
            name: {
                "value": float(point_derived[name]),
                "ci": [
                    float(np.quantile(boot_derived[name], 0.025)),
                    float(np.quantile(boot_derived[name], 0.975)),
                ],
            }
            for name in point_derived
        }

    primary = derived_holdout["498"]
    clamped_ratio = primary["clamped_over_gt"]["value"]
    if clamped_ratio >= 2.5:
        j1 = "SUPPORTED"
    elif clamped_ratio <= 1.3:
        j1 = "DEPRIORITIZED"
    else:
        j1 = "INCONCLUSIVE"

    internal_ratio = primary["internal_over_gt"]["value"]
    closure = primary["history_clamp_closure"]
    if internal_ratio <= 1.5 and closure["value"] >= 0.50 and closure["ci"][0] > 0.0:
        j2 = "SUPPORTED"
    elif closure["ci"][1] < 0.25:
        j2 = "DEPRIORITIZED"
    else:
        j2 = "INCONCLUSIVE"

    pelvis_index = metric_names.index("pelvis_frame2_error_m")
    generalization_ratio = float(
        train_point[498][pelvis_index] / holdout_point[498][pelvis_index]
    )
    j3 = "HOLDOUT_AT_LEAST_30_PERCENT_HIGHER" if generalization_ratio < 0.77 else "NOT_ESTABLISHED"

    clamped_a1 = np.asarray(
        [float(record["metrics"]["498"]["clamped_a1_fk_acc_mps2"]) for record in holdout_records]
    )
    clamped_a2 = np.asarray(
        [float(record["metrics"]["498"]["clamped_a2_fk_acc_mps2"]) for record in holdout_records]
    )

    return {
        "timesteps": list(timesteps),
        "metric_names": list(metric_names),
        "holdout": {
            "window_count": len(holdout_records),
            "episode_count": len(by_episode),
            "by_timestep": holdout_summary,
            "derived": derived_holdout,
            "a1_a2_window_correlation_t498": float(np.corrcoef(clamped_a1, clamped_a2)[0, 1]),
        },
        "train": {
            "window_count": len(train_records),
            "by_timestep": train_summary,
        },
        "bootstrap": {
            "seed": int(seed),
            "replicates": int(replicates),
            "holdout_resample_index_sha256": digest.hexdigest(),
        },
        "decisions": {
            "j1_single_forward_seam": j1,
            "j2_history_clamp_mechanism": j2,
            "j3_generalization": j3,
            "train_over_holdout_pelvis_frame2_error_t498": generalization_ratio,
            "r3_history_supervision_may_be_proposed": j1 == "SUPPORTED" and j2 == "SUPPORTED",
        },
    }
