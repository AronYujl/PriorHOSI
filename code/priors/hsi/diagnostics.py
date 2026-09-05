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
PREDICTOR_DECOMP_ARMS: Tuple[str, ...] = (
    "conditional",
    "unconditional",
    "cfg_w0",
    "cfg_w0.5",
    "cfg_w1",
    "zero_velocity_history",
)


def rebase_numerics_batch(cfg, sampler, model, batch, noise, timestep, precision, zero_bias):
    """Measure a fixed denoiser through production p_losses, without updates."""
    from models.infbagel import rebase_model_output
    from utils import transform_points
    import torch.nn.functional as F

    x_start = torch.cat((
        batch["joints"], batch["global_rot_6d"].flatten(2),
        batch["object_trans"], batch["object_rot_mat"].flatten(2), batch["contact_label"],
    ), dim=-1)
    mask = torch.zeros_like(x_start, dtype=torch.bool)
    mask[:, :2] = True
    t = torch.full((len(x_start),), int(timestep), device=x_start.device, dtype=torch.long)
    saved_bias = model.out.bias.detach().clone()
    captured = {}

    def head_hook(module, inputs, output):
        captured["hidden"] = inputs[0].detach().float()
        if precision == "head_fp32":
            saved_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.autocast(device_type=x_start.device.type, enabled=False):
                output = F.linear(inputs[0].float(), module.weight, module.bias)
            torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        return output

    def model_hook(_module, _inputs, output):
        if precision == "rebase_fp32":
            output = output.float()
        captured["raw"] = output
        return output

    with torch.no_grad():
        if zero_bias:
            model.out.bias[:216] = 0.0
    hooks = (model.out.register_forward_hook(head_hook), model.register_forward_hook(model_hook))
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    saved_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    torch.backends.cudnn.allow_tf32 = precision != "fp32"
    try:
        with torch.autocast(
            device_type=x_start.device.type, dtype=torch.bfloat16, enabled=precision != "fp32"
        ):
            loss = sampler.p_losses(
                x_start, batch["joints"], batch["mat"], batch["scene_flag"], mask, t,
                batch["text_clip_embedding"], batch["pelvis_goal"], batch["scene_goal"],
                batch["object_goal"], batch["need_scene"], batch["need_pelvis_dir"],
                batch["pi"], batch["end_pi"], batch["seg_len"], batch["need_pi"],
                batch["is_loco"], batch["is_object"], batch["obj_bps_data"],
                batch["obj_rot_mat_ref"], batch["rest_pose_obj_nn_pts"],
                batch["transformed_obj_verts"], batch["rest_human_offsets"],
                batch["object_points"], noise=noise.clone(),
            )
            total = loss["loss"] + float(cfg.loss_w_fk) * loss["loss_fk"]
        raw = captured["raw"]
        bias_grad, weight_grad, output_grad = torch.autograd.grad(
            total, (model.out.bias, model.out.weight, raw)
        )
        with torch.no_grad(), torch.autocast(device_type=x_start.device.type, enabled=False):
            prediction = rebase_model_output(raw.detach(), x_start, sampler.hsi_chain_rebase_mode).float()
            per_window = {
                "jpos_mse": (prediction[:, 2:, :84] - x_start[:, 2:, :84]).square().mean((1, 2)),
                "jrot_l1": (prediction[:, 2:, 84:216] - x_start[:, 2:, 84:216]).abs().mean((1, 2)),
                "raw_jpos_rms": raw[:, 2:, :84].float().square().mean((1, 2)).sqrt(),
                "raw_jrot_rms": raw[:, 2:, 84:216].float().square().mean((1, 2)).sqrt(),
                "prediction_finite_fraction": torch.isfinite(prediction).float().mean((1, 2)),
                "anchor_error_rms": (
                    prediction[:, 2, :216] - (2 * x_start[:, 1, :216] - x_start[:, 0, :216])
                ).square().mean(1).sqrt(),
            }
            _, predicted_joints = sampler._compute_human_joints(
                prediction, batch["joints"], batch["mat"], batch["rest_human_offsets"]
            )
            _, target_joints = sampler._compute_human_joints(
                x_start, batch["joints"], batch["mat"], batch["rest_human_offsets"]
            )
            gt_positions = transform_points(
                sampler.dataset.denormalize_torch(batch["joints"]), batch["mat"]
            ).reshape(len(x_start), 16, 28, 3)
            fk = (predicted_joints[:, 2:, [20, 21, 22, 23]] - gt_positions[:, 2:, [20, 21, 25, 27]]).square().mean((1, 2, 3))
            fk += (predicted_joints[:, 2:, [7, 8, 10, 11]] - gt_positions[:, 2:, [7, 8, 10, 11]]).square().mean((1, 2, 3))
            seam = torch.cat((target_joints[:, :2], predicted_joints[:, 2:4]), dim=1)
            seam_acc = seam[:, 2:] - 2 * seam[:, 1:-1] + seam[:, :-2]
            target_acc = target_joints[:, 2:4] - 2 * target_joints[:, 1:3] + target_joints[:, :2]
            per_window["fk"] = fk
            per_window["fullbody_seam"] = (seam_acc - target_acc).square().mean((1, 2, 3))
            per_window["base"] = per_window["jpos_mse"] + per_window["jrot_l1"]
            per_window["total"] = (
                per_window["base"] + float(cfg.loss_w_fk) * fk
                + sampler.fullbody_seam_loss_weight * per_window["fullbody_seam"]
            )
            g = bias_grad[:216].float()
            batch_metrics = {
                "production_total": float(total),
                "production_jpos_mse": float(loss["loss_jpos"]),
                "production_jrot_l1": float(loss["loss_jrot"]),
                "production_fk": float(loss["loss_fk"]),
                "production_fullbody_seam": float(loss["loss_fullbody_seam"]),
                "readout_total_abs_error": float((per_window["total"].mean() - total).abs()),
                "bias_gradient_rms": float(g.square().mean().sqrt()),
                "position_bias_gradient_rms": float(g[:84].square().mean().sqrt()),
                "rotation_bias_gradient_rms": float(g[84:].square().mean().sqrt()),
                "weight_gradient_rms": float(weight_grad[:216].float().square().mean().sqrt()),
                "output_gradient_sum_rms": float(output_grad[:, :, :216].float().sum((0, 1)).square().mean().sqrt()),
                "first_future_output_gradient_rms": float(output_grad[:, 2, :216].float().square().mean().sqrt()),
                "gradient_finite_fraction": float(torch.isfinite(g).float().mean()),
                "raw_dtype": str(raw.dtype),
            }
        values = {name: value.cpu().tolist() for name, value in per_window.items()}
        return values, batch_metrics, prediction.detach(), captured["hidden"]
    finally:
        for hook in hooks:
            hook.remove()
        with torch.no_grad():
            model.out.bias.copy_(saved_bias)
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        torch.backends.cudnn.allow_tf32 = saved_cudnn_tf32


def summarize_rebase_numerics(records, batch_records):
    """Separate episode readouts from batch-reduced output-head gradients."""
    episodes = defaultdict(lambda: defaultdict(list))
    for row in records:
        for name, value in row["metrics"].items():
            episodes[row["cell"]][row["episode_id"]].append((name, value))
    episode_metrics = {}
    for cell, rows in episodes.items():
        episode_metrics[cell] = {}
        for episode, values in rows.items():
            columns = defaultdict(list)
            for name, value in values:
                columns[name].append(value)
            episode_metrics[cell][episode] = {name: float(np.mean(value)) for name, value in columns.items()}
    cells = {}
    for cell, rows in episode_metrics.items():
        cells[cell] = {name: float(np.mean([row[name] for row in rows.values()])) for name in next(iter(rows.values()))}
        gradients = [r["bias_gradient_rms"] for r in batch_records if r["cell"] == cell]
        cells[cell]["batch_bias_gradient_rms_mean"] = float(np.mean(gradients))
    decisions = {}
    for timestep in (0, 50, 250, 498):
        arms = {p: cells[f"t{timestep}_{p}_original"] for p in ("bf16", "rebase_fp32", "head_fp32", "fp32")}
        a, h, f = arms["bf16"], arms["head_fp32"], arms["fp32"]
        decisions[str(timestep)] = {
            "forward_bias_invariance_broken": bool(a["bias_intervention_output_rms"] > max(1e-3, 10 * f["bias_intervention_output_rms"])),
            "backward_bias_leakage": bool(a["batch_bias_gradient_rms_mean"] > max(1e-7, 100 * f["batch_bias_gradient_rms_mean"])),
            "head_fp32_localizes_both": bool(
                h["bias_intervention_output_rms"] <= .01 * a["bias_intervention_output_rms"]
                and h["batch_bias_gradient_rms_mean"] <= .01 * a["batch_bias_gradient_rms_mean"]
            ),
        }
    return {"cells": cells, "decisions": decisions}, episode_metrics


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


def _first_two_fk_acceleration(
    predicted_joints: torch.Tensor,
    target_joints: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    clamped = torch.cat((target_joints[:, :2], predicted_joints[:, 2:4]), dim=1)
    acceleration = (clamped[:, 2:4] - 2.0 * clamped[:, 1:3] + clamped[:, :2])
    acceleration = acceleration * float(fps) ** 2
    return torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=(1, 2))


def predictor_decomp_metrics(
    predicted_joints: Mapping[str, torch.Tensor],
    target_joints: torch.Tensor,
    predicted_positions_m: Mapping[str, torch.Tensor],
    target_positions_m: torch.Tensor,
    *,
    fps: float,
) -> Dict[str, torch.Tensor]:
    """Per-window D2 metrics for the frozen predictor decomposition."""
    if tuple(sorted(predicted_joints)) != tuple(sorted(PREDICTOR_DECOMP_ARMS)):
        raise ValueError("predictor-decomp joint arms do not match the frozen contract")
    if tuple(sorted(predicted_positions_m)) != tuple(sorted(PREDICTOR_DECOMP_ARMS)):
        raise ValueError("predictor-decomp position arms do not match the frozen contract")

    result = {}
    gt_acc = _first_two_fk_acceleration(target_joints, target_joints, fps)
    result["gt_first2_fk_acc_mps2"] = gt_acc
    for arm in PREDICTOR_DECOMP_ARMS:
        joints = predicted_joints[arm]
        result["%s_first2_fk_acc_mps2" % arm] = _first_two_fk_acceleration(
            joints, target_joints, fps
        )

    history_velocity = target_joints[:, 1] - target_joints[:, 0]
    for frame in range(2, 6):
        multiplier = float(frame)
        constant_position = target_joints[:, 1]
        constant_velocity = target_joints[:, 0] + multiplier * history_velocity
        conditional_error = torch.linalg.vector_norm(
            predicted_joints["conditional"][:, frame] - target_joints[:, frame], dim=-1
        ).mean(dim=1)
        constant_position_error = torch.linalg.vector_norm(
            constant_position - target_joints[:, frame], dim=-1
        ).mean(dim=1)
        constant_velocity_error = torch.linalg.vector_norm(
            constant_velocity - target_joints[:, frame], dim=-1
        ).mean(dim=1)
        result["conditional_frame%d_fk_error_m" % frame] = conditional_error
        result["constant_position_frame%d_fk_error_m" % frame] = constant_position_error
        result["constant_velocity_frame%d_fk_error_m" % frame] = constant_velocity_error

    velocity = target_positions_m[:, 1] - target_positions_m[:, 0]
    response = (
        predicted_positions_m["zero_velocity_history"][:, 2]
        - predicted_positions_m["conditional"][:, 2]
    )
    direction = -2.0 * velocity
    denominator = direction.square().sum(dim=1)
    numerator = (response * direction).sum(dim=1)
    result["history_velocity_gain"] = torch.where(
        denominator > 0.0,
        numerator / denominator,
        torch.full_like(denominator, float("nan")),
    )
    return result


def single_window_chain_metrics(
    trace_joints: torch.Tensor,
    final_joints: torch.Tensor,
    target_joints: torch.Tensor,
    *,
    fps: float,
) -> Dict[str, torch.Tensor]:
    """Per-window D3 acceleration terms for the production reverse chain."""
    return {
        "gt_first2_fk_acc_mps2": _first_two_fk_acceleration(
            target_joints, target_joints, fps
        ),
        "trace_t498_first2_fk_acc_mps2": _first_two_fk_acceleration(
            trace_joints, target_joints, fps
        ),
        "final_first2_fk_acc_mps2": _first_two_fk_acceleration(
            final_joints, target_joints, fps
        ),
    }


def chain_rebase_metrics(
    final_joints: torch.Tensor,
    target_joints: torch.Tensor,
    *,
    fps: float,
) -> Dict[str, torch.Tensor]:
    """Per-window D4-B seam, FK-error, and internal-motion terms."""
    clamped = final_joints.clone()
    clamped[:, :2] = target_joints[:, :2]

    def acceleration(value, centre):
        term = value[:, centre + 1] - 2.0 * value[:, centre] + value[:, centre - 1]
        return torch.linalg.vector_norm(term * float(fps) ** 2, dim=-1).mean(dim=1)

    result = {}
    for label, centre in (("a1", 1), ("a2", 2)):
        result["gt_%s_fk_acc_mps2" % label] = acceleration(target_joints, centre)
        result["final_%s_fk_acc_mps2" % label] = acceleration(clamped, centre)
    result["gt_first2_fk_acc_mps2"] = 0.5 * (
        result["gt_a1_fk_acc_mps2"] + result["gt_a2_fk_acc_mps2"]
    )
    result["final_first2_fk_acc_mps2"] = 0.5 * (
        result["final_a1_fk_acc_mps2"] + result["final_a2_fk_acc_mps2"]
    )

    frame_errors = []
    for frame in range(2, 7):
        error = torch.linalg.vector_norm(
            clamped[:, frame] - target_joints[:, frame], dim=-1
        ).mean(dim=1)
        result["final_frame%d_fk_error_m" % frame] = error
        if frame >= 3:
            frame_errors.append(error)
    result["final_frame3_6_fk_error_m"] = torch.stack(frame_errors).mean(dim=0)

    gt_internal = []
    final_internal = []
    for frame in range(3, 9):
        gt_value = acceleration(target_joints, frame)
        final_value = acceleration(clamped, frame)
        result["gt_internal_frame%d_fk_acc_mps2" % frame] = gt_value
        result["final_internal_frame%d_fk_acc_mps2" % frame] = final_value
        gt_internal.append(gt_value)
        final_internal.append(final_value)
    result["gt_internal_frame3_8_fk_acc_mps2"] = torch.stack(gt_internal).mean(dim=0)
    result["final_internal_frame3_8_fk_acc_mps2"] = torch.stack(final_internal).mean(dim=0)

    def third_difference(value):
        term = value[:, 3] - 3.0 * value[:, 2] + 3.0 * value[:, 1] - value[:, 0]
        return torch.linalg.vector_norm(term * float(fps) ** 3, dim=-1).mean(dim=1)

    result["gt_cross_seam_third_difference_mps3"] = third_difference(target_joints)
    result["final_cross_seam_third_difference_mps3"] = third_difference(clamped)
    return result


def chain_rebase_rollout_telemetry(
    joints: StitchedSequence,
    rotations_6d: StitchedSequence,
    *,
    fps: float,
) -> Dict[str, float]:
    """D5 coarse-rollout seam and raw-6D diagnostics."""
    if joints.seams != rotations_6d.seams or len(joints) != len(rotations_6d):
        raise ValueError("joint and rotation rollout seams must match")

    positions = joints.frames
    a1, a2, third = [], [], []
    for seam in joints.seams:
        a1_term = positions[seam] - 2.0 * positions[seam - 1] + positions[seam - 2]
        a2_term = positions[seam + 1] - 2.0 * positions[seam] + positions[seam - 1]
        third_term = (
            positions[seam + 1]
            - 3.0 * positions[seam]
            + 3.0 * positions[seam - 1]
            - positions[seam - 2]
        )
        a1.append(torch.linalg.vector_norm(a1_term * float(fps) ** 2, dim=-1).mean())
        a2.append(torch.linalg.vector_norm(a2_term * float(fps) ** 2, dim=-1).mean())
        third.append(
            torch.linalg.vector_norm(third_term * float(fps) ** 3, dim=-1).mean()
        )

    rotation = rotations_6d.frames.reshape(-1, 6).to(torch.float64)
    first, second = rotation[:, :3], rotation[:, 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1)
    second_norm = torch.linalg.vector_norm(second, dim=-1)
    first_unit = first / first_norm.clamp_min(1e-12)[:, None]
    second_unit = second / second_norm.clamp_min(1e-12)[:, None]
    cosine = (first_unit * second_unit).sum(dim=-1).abs()

    projected_second = second - (first_unit * second).sum(dim=-1)[:, None] * first_unit
    second_orthogonal = projected_second / torch.linalg.vector_norm(
        projected_second, dim=-1
    ).clamp_min(1e-12)[:, None]
    third_orthogonal = torch.linalg.cross(first_unit, second_orthogonal, dim=-1)
    matrix = torch.stack((first_unit, second_orthogonal, third_orthogonal), dim=-2)
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    orthogonality = (matrix @ matrix.transpose(-1, -2) - identity).abs()

    def mean(values):
        return float(torch.stack(values).mean()) if values else float("nan")

    return {
        "coarse_seam_a1_fk_acc_mps2": mean(a1),
        "coarse_seam_a2_fk_acc_mps2": mean(a2),
        "coarse_cross_seam_third_difference_mps3": mean(third),
        "coarse_seam_count": float(len(joints.seams)),
        "rotation6d_first_axis_norm_mae": float((first_norm - 1.0).abs().mean()),
        "rotation6d_second_axis_norm_mae": float((second_norm - 1.0).abs().mean()),
        "rotation6d_abs_cosine_mean": float(cosine.mean()),
        "rotation6d_abs_cosine_max": float(cosine.max()),
        "rotation_matrix_orthogonality_max": float(orthogonality.max()),
    }


def summarize_chain_rebase(
    c0_records: Sequence[Mapping[str, object]],
    arm_records: Sequence[Mapping[str, object]],
    stratum_weights: Mapping[str, float],
    *,
    arm: str,
    min_timestep: int = 0,
    seed: int = 42,
    replicates: int = 10000,
) -> Dict[str, object]:
    """Run the paired D4-B bootstrap and apply the arm-specific decision."""
    paired = []
    for c0, candidate in zip(c0_records, arm_records):
        if int(c0["data_idx"]) != int(candidate["data_idx"]):
            raise ValueError("D4-B c0 and candidate rows are not aligned")
        metrics = {}
        for name, value in c0["metrics"].items():
            metrics["c0_" + str(name)] = float(value)
        for name, value in candidate["metrics"].items():
            metrics["arm_" + str(name)] = float(value)
            if name.startswith("final_"):
                metrics["difference_" + str(name)] = float(value) - float(
                    c0["metrics"][name]
                )
        paired.append(
            {
                "episode_id": str(candidate["episode_id"]),
                "stratum": str(candidate["stratum"]),
                "metrics": metrics,
            }
        )

    names, point, boot, digest, episode_count = _episode_weighted_bootstrap(
        paired, stratum_weights, seed=seed, replicates=replicates
    )
    index = {name: position for position, name in enumerate(names)}
    denominator = (
        point[index["c0_final_first2_fk_acc_mps2"]]
        - point[index["c0_gt_first2_fk_acc_mps2"]]
    )
    numerator = (
        point[index["arm_final_first2_fk_acc_mps2"]]
        - point[index["arm_gt_first2_fk_acc_mps2"]]
    )
    ratio = float(numerator / denominator)
    boot_denominator = (
        boot[:, index["c0_final_first2_fk_acc_mps2"]]
        - boot[:, index["c0_gt_first2_fk_acc_mps2"]]
    )
    boot_numerator = (
        boot[:, index["arm_final_first2_fk_acc_mps2"]]
        - boot[:, index["arm_gt_first2_fk_acc_mps2"]]
    )
    valid = boot_denominator > 0.0
    ratio_samples = boot_numerator[valid] / boot_denominator[valid]

    third_denominator = (
        point[index["c0_final_cross_seam_third_difference_mps3"]]
        - point[index["c0_gt_cross_seam_third_difference_mps3"]]
    )
    third_numerator = (
        point[index["arm_final_cross_seam_third_difference_mps3"]]
        - point[index["arm_gt_cross_seam_third_difference_mps3"]]
    )
    third_ratio = float(third_numerator / third_denominator)
    boot_third_denominator = (
        boot[:, index["c0_final_cross_seam_third_difference_mps3"]]
        - boot[:, index["c0_gt_cross_seam_third_difference_mps3"]]
    )
    boot_third_numerator = (
        boot[:, index["arm_final_cross_seam_third_difference_mps3"]]
        - boot[:, index["arm_gt_cross_seam_third_difference_mps3"]]
    )
    third_valid = boot_third_denominator > 0.0
    third_ratio_samples = (
        boot_third_numerator[third_valid] / boot_third_denominator[third_valid]
    )

    guard_names = (
        "final_a2_fk_acc_mps2",
        "final_frame3_6_fk_error_m",
        "final_internal_frame3_8_fk_acc_mps2",
    )
    guards = {}
    for name in guard_names:
        key = "difference_" + name
        samples = boot[:, index[key]]
        ci = np.quantile(samples, (0.025, 0.975))
        guards[name] = {
            "difference": float(point[index[key]]),
            "ci": [float(ci[0]), float(ci[1])],
            "significantly_worse": bool(ci[0] > 0.0),
        }
    guards_pass = not any(value["significantly_worse"] for value in guards.values())

    if arm == "c3" and int(min_timestep) > 0:
        if third_ratio <= 0.4:
            decision = "PROCEED"
        elif third_ratio >= 0.6:
            decision = "DEPRIORITIZED"
        else:
            decision = "INCONCLUSIVE"
    elif arm == "c1":
        if ratio <= 0.5 and guards_pass:
            decision = "SUPPORTED"
        elif ratio >= 0.8:
            decision = "DEPRIORITIZED"
        elif ratio <= 0.5:
            decision = "GUARD_FAILED"
        else:
            decision = "INCONCLUSIVE"
    elif arm == "c2":
        if ratio <= 0.4:
            decision = "POSITION_DOMINATED"
        elif ratio >= 0.7:
            decision = "VELOCITY_DIRECTION_DOMINATED"
        else:
            decision = "INCONCLUSIVE"
    else:
        decision = "DESCRIPTIVE"

    return {
        "arm": str(arm),
        "rebase_min_timestep": int(min_timestep),
        "window_count": len(arm_records),
        "episode_count": episode_count,
        "metrics": _metric_summary(names, point, boot),
        "bootstrap": {
            "seed": int(seed),
            "replicates": int(replicates),
            "paired_holdout_resample_index_sha256": digest,
        },
        "derived": {
            "seam_excess_ratio": ratio,
            "seam_excess_ratio_ci_over_positive_denominator_replicates": [
                float(np.quantile(ratio_samples, 0.025)),
                float(np.quantile(ratio_samples, 0.975)),
            ],
            "c0_excess_mps2": float(denominator),
            "arm_excess_mps2": float(numerator),
            "third_difference_excess_ratio": third_ratio,
            "third_difference_excess_ratio_ci_over_positive_denominator_replicates": [
                float(np.quantile(third_ratio_samples, 0.025)),
                float(np.quantile(third_ratio_samples, 0.975)),
            ],
            "c0_third_difference_excess_mps3": float(third_denominator),
            "arm_third_difference_excess_mps3": float(third_numerator),
        },
        "guards": guards,
        "decision": decision,
    }


def d4_offline_decomp_metrics(
    target_joints: torch.Tensor,
    full_joints: torch.Tensor,
    root_joints: torch.Tensor,
    pose_joints: torch.Tensor,
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    fps: float,
) -> Dict[str, torch.Tensor]:
    """Decompose one predicted window into root, pose, and interaction terms."""
    if not fps > 0:
        raise ValueError("fps must be positive")
    expected = tuple(target_joints.shape)
    if any(tuple(value.shape) != expected for value in (full_joints, root_joints, pose_joints)):
        raise ValueError("all joint tensors must share shape")

    def first_two(value):
        clamped = torch.cat((target_joints[:, :2], value[:, 2:4]), dim=1)
        acceleration = (clamped[:, 2:4] - 2.0 * clamped[:, 1:3] + clamped[:, :2])
        return torch.linalg.vector_norm(acceleration * float(fps) ** 2, dim=-1).mean(dim=(1, 2))

    gt = first_two(target_joints)
    full = first_two(full_joints)
    root = first_two(root_joints)
    pose = first_two(pose_joints)

    predicted_root = predicted_positions[:, 2, :3]
    target_root = target_positions[:, :3, :3]
    error = predicted_root - target_root[:, 2]
    history_velocity = target_root[:, 1] - target_root[:, 0]
    horizontal_velocity = history_velocity[:, (0, 2)]
    horizontal_norm = torch.linalg.vector_norm(horizontal_velocity, dim=-1, keepdim=True)
    direction = horizontal_velocity / horizontal_norm.clamp_min(1e-12)
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    horizontal_error = error[:, (0, 2)]
    predicted_velocity = predicted_root - target_root[:, 1]
    cross = torch.linalg.vector_norm(
        torch.linalg.cross(predicted_velocity, target_root[:, 2] - target_root[:, 1], dim=-1),
        dim=-1,
    )
    dot = (predicted_velocity * (target_root[:, 2] - target_root[:, 1])).sum(dim=-1)

    return {
        "gt_first2_fk_acc_mps2": gt,
        "full_first2_fk_acc_mps2": full,
        "root_first2_fk_acc_mps2": root,
        "pose_first2_fk_acc_mps2": pose,
        "full_excess_mps2": full - gt,
        "root_excess_mps2": root - gt,
        "pose_excess_mps2": pose - gt,
        "interaction_excess_mps2": full - root - pose + gt,
        "frame2_root_error_parallel_m": (horizontal_error * direction).sum(dim=-1),
        "frame2_root_error_horizontal_orthogonal_m": (horizontal_error * perpendicular).sum(dim=-1),
        "frame2_root_error_vertical_m": error[:, 1],
        "frame2_root_error_m": torch.linalg.vector_norm(error, dim=-1),
        "history_horizontal_speed_m_per_frame": horizontal_norm[:, 0],
        "frame2_velocity_angle_deg": torch.rad2deg(torch.atan2(cross, dot)),
    }


def summarize_d4_offline_decomp(
    holdout_records: Sequence[Mapping[str, object]],
    train_records: Sequence[Mapping[str, object]],
    stratum_weights: Mapping[str, float],
    *,
    seed: int = 42,
    replicates: int = 10000,
) -> Dict[str, object]:
    """Summarize D4-A and apply the conditional rotation-arm gate."""
    metric_names, point, boot, digest, episode_count = _episode_weighted_bootstrap(
        holdout_records, stratum_weights, seed=seed, replicates=replicates
    )
    train_names = tuple(sorted(str(name) for name in train_records[0]["metrics"]))
    train_point, train_boot = _ordinary_window_bootstrap(
        train_records, seed=seed, replicates=replicates, metric_names=train_names
    )
    index = {name: position for position, name in enumerate(metric_names)}

    arms = ("d2_conditional", "d3_trace", "d3_final")
    decomposition = {}
    for arm in arms:
        full = point[index[arm + "_full_excess_mps2"]]
        values = {}
        for component in ("root", "pose", "interaction"):
            value = point[index[arm + "_" + component + "_excess_mps2"]]
            samples = (
                boot[:, index[arm + "_" + component + "_excess_mps2"]]
                / boot[:, index[arm + "_full_excess_mps2"]]
            )
            values[component + "_share"] = {
                "value": float(value / full),
                "ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            }
        decomposition[arm] = values

    pose_share = decomposition["d3_final"]["pose_share"]["value"]
    return {
        "holdout": {
            "window_count": len(holdout_records),
            "episode_count": episode_count,
            "metrics": _metric_summary(metric_names, point, boot),
        },
        "train": {
            "window_count": len(train_records),
            "metrics": _metric_summary(train_names, train_point, train_boot),
        },
        "bootstrap": {
            "seed": int(seed),
            "replicates": int(replicates),
            "holdout_resample_index_sha256": digest,
        },
        "decomposition": decomposition,
        "decision": {
            "c3_rotation_rebase_authorized": bool(pose_share >= 0.40),
            "gate_arm": "d3_final_holdout",
            "pose_share_threshold": 0.40,
            "pose_share": float(pose_share),
        },
    }


def _episode_weighted_bootstrap(
    records: Sequence[Mapping[str, object]],
    stratum_weights: Mapping[str, float],
    *,
    seed: int,
    replicates: int,
):
    if not records:
        raise ValueError("diagnostic records must be non-empty")
    metric_names = tuple(sorted(str(name) for name in records[0]["metrics"]))
    by_episode = defaultdict(list)
    episode_stratum = {}
    for record in records:
        values = np.asarray(
            [float(record["metrics"][name]) for name in metric_names], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError("diagnostic record contains non-finite metrics")
        episode_id = str(record["episode_id"])
        stratum = str(record["stratum"])
        by_episode[episode_id].append(values)
        episode_stratum[episode_id] = stratum
    episode_vectors = {
        episode_id: np.stack(values).mean(axis=0)
        for episode_id, values in by_episode.items()
    }
    strata = defaultdict(list)
    for episode_id in sorted(episode_vectors):
        strata[episode_stratum[episode_id]].append(episode_id)
    if set(strata) != set(stratum_weights):
        raise ValueError("diagnostic strata do not match the registered weights")
    weights = {str(key): float(value) for key, value in stratum_weights.items()}

    def combine(selected):
        total = np.zeros(len(metric_names), dtype=np.float64)
        for stratum, episode_ids in selected.items():
            total += weights[stratum] * np.stack(
                [episode_vectors[episode_id] for episode_id in episode_ids]
            ).mean(axis=0)
        return total

    point = combine({stratum: tuple(ids) for stratum, ids in strata.items()})
    boot = np.empty((int(replicates), len(metric_names)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    digest = hashlib.sha256()
    for replicate in range(int(replicates)):
        selected = {}
        for stratum in sorted(strata):
            episode_ids = strata[stratum]
            indices = rng.integers(0, len(episode_ids), size=len(episode_ids), dtype=np.int64)
            digest.update(indices.tobytes())
            selected[stratum] = [episode_ids[index] for index in indices]
        boot[replicate] = combine(selected)
    return metric_names, point, boot, digest.hexdigest(), len(by_episode)


def _ordinary_window_bootstrap(records, *, seed, replicates, metric_names):
    matrix = np.stack(
        [
            np.asarray([float(record["metrics"][name]) for name in metric_names])
            for record in records
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("diagnostic train records contain non-finite metrics")
    point = matrix.mean(axis=0)
    boot = np.empty((int(replicates), len(metric_names)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for replicate in range(int(replicates)):
        indices = rng.integers(0, len(matrix), size=len(matrix), dtype=np.int64)
        boot[replicate] = matrix[indices].mean(axis=0)
    return point, boot


def _metric_summary(metric_names, point, boot):
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


def summarize_predictor_decomp(
    holdout_records: Sequence[Mapping[str, object]],
    train_records: Sequence[Mapping[str, object]],
    stratum_weights: Mapping[str, float],
    *,
    seed: int = 42,
    replicates: int = 10000,
) -> Dict[str, object]:
    """Apply the preregistered D2 K1-K4 decisions."""
    metric_names, point, boot, digest, episode_count = _episode_weighted_bootstrap(
        holdout_records, stratum_weights, seed=seed, replicates=replicates
    )
    train_point, train_boot = _ordinary_window_bootstrap(
        train_records,
        seed=seed,
        replicates=replicates,
        metric_names=metric_names,
    )
    index = {name: position for position, name in enumerate(metric_names)}

    cfg_delta = (
        point[index["cfg_w1_first2_fk_acc_mps2"]]
        - point[index["cfg_w0_first2_fk_acc_mps2"]]
    )
    cfg_delta_boot = (
        boot[:, index["cfg_w1_first2_fk_acc_mps2"]]
        - boot[:, index["cfg_w0_first2_fk_acc_mps2"]]
    )
    cfg_delta_ci = np.quantile(cfg_delta_boot, (0.025, 0.975))
    if cfg_delta_ci[1] < 0.0:
        k1 = "CFG_COMPRESSES_SEAM"
    elif cfg_delta_ci[0] > 0.0 and cfg_delta >= 0.12:
        k1 = "SUPPORTED"
    elif cfg_delta_ci[1] < 0.12:
        k1 = "DEPRIORITIZED"
    else:
        k1 = "INCONCLUSIVE"

    k2_ratio = (
        point[index["conditional_frame2_fk_error_m"]]
        / point[index["constant_velocity_frame2_fk_error_m"]]
    )
    if k2_ratio >= 1.0:
        k2 = "SUPPORTED"
    elif k2_ratio <= 0.7:
        k2 = "DEPRIORITIZED"
    else:
        k2 = "INCONCLUSIVE"

    velocity_gain = point[index["history_velocity_gain"]]
    if velocity_gain < 0.5:
        k3 = "SUPPORTED"
    elif velocity_gain > 0.8:
        k3 = "DEPRIORITIZED"
    else:
        k3 = "INCONCLUSIVE"

    deterministic_ratio = (
        point[index["conditional_first2_fk_acc_mps2"]]
        / point[index["gt_first2_fk_acc_mps2"]]
    )
    d1_ratio = 2.0739302919402602
    k4_erratum = abs(deterministic_ratio - d1_ratio) > 0.05
    return {
        "holdout": {
            "window_count": len(holdout_records),
            "episode_count": episode_count,
            "metrics": _metric_summary(metric_names, point, boot),
        },
        "train": {
            "window_count": len(train_records),
            "metrics": _metric_summary(metric_names, train_point, train_boot),
        },
        "bootstrap": {
            "seed": int(seed),
            "replicates": int(replicates),
            "holdout_resample_index_sha256": digest,
        },
        "derived": {
            "cfg_w1_minus_w0_first2_fk_acc_mps2": {
                "value": float(cfg_delta),
                "ci": [float(cfg_delta_ci[0]), float(cfg_delta_ci[1])],
            },
            "conditional_over_constant_velocity_frame2_fk_error": float(k2_ratio),
            "history_velocity_gain": float(velocity_gain),
            "deterministic_conditional_over_gt": float(deterministic_ratio),
            "d1_stochastic_clamped_over_gt": d1_ratio,
            "d1_ratio_absolute_difference": float(abs(deterministic_ratio - d1_ratio)),
        },
        "decisions": {
            "k1_cfg": k1,
            "k2_constant_velocity": k2,
            "k3_history_velocity_use": k3,
            "k4_d1_erratum_required": bool(k4_erratum),
            "d3_w0_authorized": k1 == "SUPPORTED",
            "r3_res_may_be_proposed": k2 == "SUPPORTED" and k3 == "SUPPORTED",
        },
    }


def summarize_single_window_chain(
    records: Sequence[Mapping[str, object]],
    stratum_weights: Mapping[str, float],
    *,
    seed: int = 42,
    replicates: int = 10000,
) -> Dict[str, object]:
    """Apply the preregistered D3 chain-retention decision."""
    metric_names, point, boot, digest, episode_count = _episode_weighted_bootstrap(
        records, stratum_weights, seed=seed, replicates=replicates
    )
    index = {name: position for position, name in enumerate(metric_names)}
    denominator = (
        point[index["trace_t498_first2_fk_acc_mps2"]]
        - point[index["gt_first2_fk_acc_mps2"]]
    )
    numerator = (
        point[index["final_first2_fk_acc_mps2"]]
        - point[index["gt_first2_fk_acc_mps2"]]
    )
    if denominator <= 0.0:
        rho = None
        rho_ci = None
        decision = "INCONCLUSIVE"
    else:
        rho = float(numerator / denominator)
        boot_denominator = (
            boot[:, index["trace_t498_first2_fk_acc_mps2"]]
            - boot[:, index["gt_first2_fk_acc_mps2"]]
        )
        boot_numerator = (
            boot[:, index["final_first2_fk_acc_mps2"]]
            - boot[:, index["gt_first2_fk_acc_mps2"]]
        )
        valid = boot_denominator > 0.0
        rho_samples = boot_numerator[valid] / boot_denominator[valid]
        rho_ci = (
            None
            if not len(rho_samples)
            else [
                float(np.quantile(rho_samples, 0.025)),
                float(np.quantile(rho_samples, 0.975)),
            ]
        )
        if rho >= 0.8:
            decision = "SUPPORTED"
        elif rho <= 0.5:
            decision = "CHAIN_REPAIRS"
        else:
            decision = "INCONCLUSIVE"
    return {
        "window_count": len(records),
        "episode_count": episode_count,
        "metrics": _metric_summary(metric_names, point, boot),
        "bootstrap": {
            "seed": int(seed),
            "replicates": int(replicates),
            "holdout_resample_index_sha256": digest,
        },
        "derived": {
            "rho": rho,
            "rho_ci_over_positive_denominator_replicates": rho_ci,
            "point_denominator_mps2": float(denominator),
            "point_numerator_mps2": float(numerator),
        },
        "decision": {
            "chain_retains_predictor_seam": decision,
            "pause_regressor_training_line": decision == "CHAIN_REPAIRS",
        },
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
