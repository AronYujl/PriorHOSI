"""Scene-free 500-step diffusion utilities for HOIPrior."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from .representation import REPRESENTATION
from .sparse_relation import (
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    sparse_rest_object_point_cache,
    verify_sparse_rest_object_assets,
)
from .window_codec import project_to_so3


def normalize_progress(progress: torch.Tensor) -> torch.Tensor:
    """Map raw ``pi/end_pi/seq_length`` to the Phase 1A model condition."""
    if progress.shape[-1] != 3:
        raise ValueError(f"expected progress [...,3], got {tuple(progress.shape)}")
    denominator = progress[..., 2:3].clamp_min(1.0)
    return torch.cat((progress[..., :2] / denominator, torch.log1p(denominator) / 10.0), dim=-1)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def project_object_rotation_x0(clean: torch.Tensor) -> torch.Tensor:
    """Close predicted object rotations on SO(3) without changing the 232-D API."""
    if clean.ndim != 3 or clean.shape[1:] != (
        REPRESENTATION.window_frames, REPRESENTATION.dimension,
    ):
        raise ValueError(f"expected clean [B,16,232], got {tuple(clean.shape)}")
    result = clean.clone()
    predicted = result[:, REPRESENTATION.history_frames:, 219:228]
    projected = project_to_so3(predicted.reshape(*predicted.shape[:-1], 3, 3))
    result[:, REPRESENTATION.history_frames:, 219:228] = projected.reshape(*predicted.shape)
    return result


def prepare_clean_x0(
    clean: torch.Tensor,
    fixed_history: torch.Tensor,
    *,
    object_so3_x0: bool = False,
) -> torch.Tensor:
    """Restore immutable history, then optionally close predicted object x0 on SO(3)."""
    if fixed_history.shape != (
        clean.shape[0], REPRESENTATION.history_frames, REPRESENTATION.dimension,
    ):
        raise ValueError(f"invalid fixed history shape: {tuple(fixed_history.shape)}")
    result = clean.clone()
    result[:, :REPRESENTATION.history_frames] = fixed_history
    if object_so3_x0:
        result = project_object_rotation_x0(result)
    return result


class GaussianDiffusion(nn.Module):
    """Linear-beta x0-prediction diffusion with fixed two-frame history."""

    def __init__(self, timesteps: int = 500) -> None:
        super().__init__()
        if timesteps != REPRESENTATION.diffusion_steps:
            raise ValueError(f"HOIPrior diffusion must use {REPRESENTATION.diffusion_steps} steps")
        betas = torch.linspace(0.0001, 0.02, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_previous = torch.nn.functional.pad(alpha_bar[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", posterior_variance.clamp_min(1e-20).log())
        self.register_buffer(
            "posterior_mean_coef1", betas * alpha_bar_previous.sqrt() / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alpha_bar_previous) * alphas.sqrt() / (1.0 - alpha_bar),
        )
        self.timesteps = timesteps

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = (
            _extract(self.sqrt_alpha_bar, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_bar, timesteps, clean.shape) * noise
        )
        noisy[:, :REPRESENTATION.history_frames] = clean[:, :REPRESENTATION.history_frames]
        return noisy

    def posterior_mean(
        self,
        current: torch.Tensor,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return the production ``q(x_{t-1} | x_t, x_0)`` posterior mean.

        Keeping this formula in one helper makes the paired D2-H diagnostic and
        the production sampler consume the exact same registered coefficients.
        The helper deliberately performs no projection, clamp, or conditioning.
        """
        if current.shape != clean.shape:
            raise ValueError(
                f"posterior current/clean shapes differ: {tuple(current.shape)}/{tuple(clean.shape)}"
            )
        if current.ndim != 3 or current.shape[1:] != (
            REPRESENTATION.window_frames, REPRESENTATION.dimension,
        ):
            raise ValueError(f"expected posterior state [B,16,232], got {tuple(current.shape)}")
        if timesteps.shape != (current.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError(f"expected long posterior timesteps [B], got {timesteps.shape}/{timesteps.dtype}")
        if bool((timesteps < 0).any()) or bool((timesteps >= self.timesteps).any()):
            raise ValueError("posterior timestep is outside the registered diffusion schedule")
        return (
            _extract(self.posterior_mean_coef1, timesteps, current.shape) * clean
            + _extract(self.posterior_mean_coef2, timesteps, current.shape) * current
        )

    def posterior_sample(
        self,
        current: torch.Tensor,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        fixed_history: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one production reverse posterior step with explicit noise.

        ``noise`` is explicit so paired diagnostics can prove identity instead
        of relying on generator call order.  At timestep zero the registered
        posterior variance is zero, so a zero noise tensor preserves the
        production sampler's historical no-draw behavior.
        """
        if noise.shape != current.shape:
            raise ValueError(f"posterior noise shape mismatch: {tuple(noise.shape)}/{tuple(current.shape)}")
        expected_history = (
            current.shape[0], REPRESENTATION.history_frames, REPRESENTATION.dimension,
        )
        if fixed_history.shape != expected_history:
            raise ValueError(
                f"expected fixed history {expected_history}, got {tuple(fixed_history.shape)}"
            )
        mean = self.posterior_mean(current, clean, timesteps)
        result = mean + (
            0.5 * _extract(self.posterior_log_variance, timesteps, current.shape)
        ).exp() * noise
        result[:, :REPRESENTATION.history_frames] = fixed_history
        return result

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        fixed_history: torch.Tensor,
        text_embedding: torch.Tensor,
        object_bps: torch.Tensor,
        goals: torch.Tensor,
        progress: torch.Tensor,
        *,
        local_object_bps: Optional[torch.Tensor] = None,
        rest_object_points: Optional[torch.Tensor] = None,
        world_to_local_rotation: Optional[torch.Tensor] = None,
        object_rotation_reference: Optional[torch.Tensor] = None,
        position_minimum: Optional[torch.Tensor] = None,
        position_maximum: Optional[torch.Tensor] = None,
        object_minimum: Optional[torch.Tensor] = None,
        object_maximum: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        paired_repeats: int = 1,
        object_so3_x0: bool = False,
    ) -> torch.Tensor:
        batch = fixed_history.shape[0]
        shape = (batch, REPRESENTATION.window_frames, REPRESENTATION.dimension)
        if paired_repeats < 1 or batch % paired_repeats:
            raise ValueError("paired_repeats must evenly divide the batch")
        base_batch = batch // paired_repeats
        current = torch.randn(
            (base_batch, *shape[1:]), device=fixed_history.device, generator=generator,
        ).repeat(paired_repeats, 1, 1)
        current[:, :REPRESENTATION.history_frames] = fixed_history
        relation_values = (
            rest_object_points,
            world_to_local_rotation,
            object_rotation_reference,
            position_minimum,
            position_maximum,
            object_minimum,
            object_maximum,
        )
        if any(value is not None for value in relation_values) and any(
            value is None for value in relation_values
        ):
            raise ValueError("D2-AE sampling requires a complete sparse relation metadata set")
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((batch,), step, dtype=torch.long, device=current.device)
            if all(value is not None for value in relation_values):
                if local_object_bps is not None:
                    raise ValueError("D2-AE sparse relation metadata cannot be combined with D2-AD local BPS")
                clean = model(
                    current,
                    timesteps,
                    text_embedding,
                    object_bps,
                    goals,
                    progress,
                    rest_object_points=rest_object_points,
                    world_to_local_rotation=world_to_local_rotation,
                    object_rotation_reference=object_rotation_reference,
                    position_minimum=position_minimum,
                    position_maximum=position_maximum,
                    object_minimum=object_minimum,
                    object_maximum=object_maximum,
                )
            elif local_object_bps is not None:
                clean = model(
                    current,
                    timesteps,
                    text_embedding,
                    object_bps,
                    goals,
                    progress,
                    local_object_bps=local_object_bps,
                )
            else:
                clean = model(
                    current, timesteps, text_embedding, object_bps, goals, progress,
                )
            clean = prepare_clean_x0(
                clean, fixed_history, object_so3_x0=object_so3_x0,
            )
            if step:
                noise = torch.randn(
                    (base_batch, *current.shape[1:]), device=current.device, generator=generator,
                ).repeat(paired_repeats, 1, 1)
            else:
                noise = torch.zeros_like(current)
            current = self.posterior_sample(
                current, clean, timesteps, noise, fixed_history,
            )
        return current


class HOIPriorSampler:
    """Legacy-evaluator-compatible wrapper around the scene-free sampler.

    The compatibility method accepts the old evaluator's metadata, but never
    forwards scene occupancy or a scene flag to HOIPrior.
    """

    def __init__(
        self,
        device: str,
        auto_regre_num: int = 2,
        timesteps: int = 500,
        object_so3_x0: bool = False,
        **_: object,
    ) -> None:
        if auto_regre_num != REPRESENTATION.history_frames:
            raise ValueError("HOIPrior sampler requires exactly two history frames")
        self.device = torch.device(device)
        self.diffusion = GaussianDiffusion(timesteps).to(self.device)
        self.object_so3_x0 = bool(object_so3_x0)
        self.audit: Dict[str, int] = {
            "generated_values": 0,
            "nonfinite_values": 0,
            "position_values": 0,
            "position_outside_count": 0,
            "object_values": 0,
            "object_outside_count": 0,
        }
        self.sample_calls = 0
        self.local_bps_builder = None
        self.local_bps_build_seconds = 0.0
        self.local_bps_build_calls = 0
        self.local_bps_digest = hashlib.sha256()
        self.sparse_relation_enabled = False
        self.sparse_rest_point_cache = None
        self.sparse_relation_asset_contract = None
        self.sparse_relation_metadata_calls = 0
        self.sparse_relation_metadata_seconds = 0.0

    def reset_sampling_audit(self) -> None:
        """Exclude evaluator warmup calls from deterministic samples and audits."""
        for key in self.audit:
            self.audit[key] = 0
        self.sample_calls = 0
        self.local_bps_build_seconds = 0.0
        self.local_bps_build_calls = 0
        self.local_bps_digest = hashlib.sha256()
        self.sparse_relation_metadata_calls = 0
        self.sparse_relation_metadata_seconds = 0.0

    def set_dataset_and_model(self, dataset, model: nn.Module) -> None:
        if getattr(dataset, "load_scene", None):
            raise ValueError("HOIPrior evaluation dataset must have load_scene=false")
        self.dataset = dataset
        self.student_model = model
        if getattr(model, "architecture_variant", None) == (
            "d2ad_local_frame_interaction_adapter"
        ):
            from .d2ad import DEFAULT_QUERY_WORKERS, LocalObjectBPSBuilder
            self.local_bps_builder = LocalObjectBPSBuilder(
                Path(__file__).resolve().parents[2],
                query_workers=DEFAULT_QUERY_WORKERS,
            )
        else:
            self.local_bps_builder = None
        self.sparse_relation_enabled = (
            getattr(model, "architecture_variant", None)
            == "d2ae_sparse_relation_field"
        )
        self.sparse_rest_point_cache = None
        self.sparse_relation_asset_contract = None
        if self.sparse_relation_enabled:
            required = ("min_torch", "max_torch", "obj_min_torch", "obj_max_torch")
            missing = [name for name in required if not hasattr(dataset, name)]
            if missing:
                raise ValueError(f"D2-AE evaluator dataset is missing normalization tensors: {missing}")

    @torch.no_grad()
    def p_sample_loop(
        self, fixed_points, mat, scene_flag, text_emb, pelvis_goal, scene_goal, object_goal,
        need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object,
        obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, seq_name_dict,
        obj_rot_mat_prefix=None, object_only=False,
    ):
        del scene_flag, scene_goal, need_scene, need_pelvis_dir, need_pi
        del is_loco, object_points
        del obj_rot_mat_prefix, object_only
        batch = fixed_points.shape[0]
        if not bool(torch.as_tensor(is_object).all()):
            raise ValueError("HOIPrior evaluation received a non-object window")
        text = text_emb.reshape(batch, -1)
        bps = obj_bps_data.reshape(batch, -1, 1024, 3)[:, 0]
        raw_goal = object_goal.reshape(batch, 3)
        normalized_goal = self.dataset.normalize_torch(raw_goal, is_object=True)
        goals = torch.zeros(batch, 9, dtype=torch.float32, device=self.device)
        goals[:, :3] = pelvis_goal.reshape(batch, 3).to(device=self.device, dtype=torch.float32)
        goals[:, 1] = 0.0
        goals[:, 6:9] = normalized_goal
        raw_progress = torch.stack((pi.reshape(-1), end_pi.reshape(-1), seq_length.reshape(-1)), dim=-1).float()
        progress = normalize_progress(raw_progress)
        local_bps = None
        if self.local_bps_builder is not None:
            names = [str(seq_name_dict[index]) for index in range(batch)]
            started = time.perf_counter()
            local_bps = self.local_bps_builder.build_from_evaluator_inputs(
                mat,
                obj_rot_mat_ref,
                names,
            ).to(device=self.device, dtype=torch.float32)
            self.local_bps_build_seconds += time.perf_counter() - started
            self.local_bps_build_calls += 1
            self.local_bps_digest.update(
                local_bps.detach().cpu().contiguous().numpy().tobytes()
            )
        relation_arguments = {}
        if self.sparse_relation_enabled:
            started = time.perf_counter()
            if self.sparse_rest_point_cache is None:
                self.sparse_relation_asset_contract = verify_sparse_rest_object_assets(
                    obj_rest_verts,
                )
                self.sparse_rest_point_cache = sparse_rest_object_point_cache(
                    obj_rest_verts,
                )
            names = [str(seq_name_dict[index]) for index in range(batch)]
            object_names = [name.split("_")[1] for name in names]
            rest_points = torch.stack([
                self.sparse_rest_point_cache[name]
                for name in object_names
            ]).to(device=self.device, dtype=torch.float32)
            transform = mat.reshape(batch, 4, 4).to(
                device=self.device, dtype=torch.float32,
            )
            reference = obj_rot_mat_ref.reshape(batch, -1, 3, 3)[:, 0].to(
                device=self.device, dtype=torch.float32,
            )
            relation_arguments = {
                "rest_object_points": rest_points,
                "world_to_local_rotation": transform[:, :3, :3].transpose(-1, -2),
                "object_rotation_reference": reference,
                "position_minimum": self.dataset.min_torch.to(
                    device=self.device, dtype=torch.float32,
                ),
                "position_maximum": self.dataset.max_torch.to(
                    device=self.device, dtype=torch.float32,
                ),
                "object_minimum": self.dataset.obj_min_torch.to(
                    device=self.device, dtype=torch.float32,
                ),
                "object_maximum": self.dataset.obj_max_torch.to(
                    device=self.device, dtype=torch.float32,
                ),
            }
            self.sparse_relation_metadata_seconds += time.perf_counter() - started
            self.sparse_relation_metadata_calls += 1
        generator = torch.Generator(device=self.device)
        generator.manual_seed((int(torch.initial_seed()) + self.sample_calls * 1000003) % (2 ** 63 - 1))
        self.sample_calls += 1
        sample = self.diffusion.sample(
            self.student_model,
            fixed_points,
            text,
            bps,
            goals,
            progress,
            local_object_bps=local_bps,
            **relation_arguments,
            generator=generator,
            object_so3_x0=self.object_so3_x0,
        )
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(batch, REPRESENTATION.window_frames, 3, 3)
        ).reshape(batch, REPRESENTATION.window_frames, 9)
        self._update_audit(sample)
        return [sample], []

    def _update_audit(self, sample: torch.Tensor) -> None:
        finite = torch.isfinite(sample)
        positions = sample[..., :84]
        objects = sample[..., 216:219]
        self.audit["generated_values"] += sample.numel()
        self.audit["nonfinite_values"] += int((~finite).sum().item())
        self.audit["position_values"] += positions.numel()
        self.audit["position_outside_count"] += int((positions.abs() > 1.0).sum().item())
        self.audit["object_values"] += objects.numel()
        self.audit["object_outside_count"] += int((objects.abs() > 1.0).sum().item())

    def audit_dict(self) -> Dict[str, object]:
        value: Dict[str, object] = dict(self.audit)
        value["position_outside_rate"] = (
            self.audit["position_outside_count"] / self.audit["position_values"]
            if self.audit["position_values"] else None
        )
        value["object_outside_rate"] = (
            self.audit["object_outside_count"] / self.audit["object_values"]
            if self.audit["object_values"] else None
        )
        value["local_bps_build_calls"] = self.local_bps_build_calls
        value["local_bps_build_seconds"] = self.local_bps_build_seconds
        value["local_bps_sha256"] = (
            self.local_bps_digest.hexdigest()
            if self.local_bps_build_calls else None
        )
        value["sparse_relation_metadata_calls"] = self.sparse_relation_metadata_calls
        value["sparse_relation_metadata_seconds"] = self.sparse_relation_metadata_seconds
        value["sparse_relation_asset_contract"] = self.sparse_relation_asset_contract
        if self.sparse_relation_enabled:
            value["sparse_relation_expected_hashes"] = {
                "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
                "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
                "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
            }
        return value
