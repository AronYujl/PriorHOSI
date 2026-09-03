"""Registered causal diagnostics for HSIPrior inference."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from priors.hsi.metrics import StitchedSequence

FUTURE_OCC_OFFSETS: Tuple[int, ...] = (5, 10, 15)
FUTURE_OCC_MODES: Tuple[str, ...] = (
    "predicted",
    "gt_crop",
    "gt_coordinate",
    "gt_both",
)


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
