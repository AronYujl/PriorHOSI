"""Scene-free 500-step diffusion utilities for HOIPrior.

The expert-agnostic DDPM pieces (schedule buffers, ``q_sample`` history
pinning, ``posterior_mean``/``posterior_sample``) live in
``priors.core.ddpm``.  This module keeps only the HOI-coupled reverse loop and
the legacy-evaluator sampler wrapper, and re-exports ``normalize_progress`` and
``_extract`` so ``from priors.hoi.diffusion import ...`` keeps resolving them.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
from torch import nn

from ..core.ddpm import (
    GaussianDiffusion as _CoreGaussianDiffusion,
    _extract,
    normalize_progress,
)
from ..core.representation import REPRESENTATION
from ..core.window_codec import WindowFrame, WindowStateCodec, project_to_so3
from .sparse_relation import (
    ARCHITECTURE_VARIANT as D2AE_ARCHITECTURE_VARIANT,
    D2AF_ARCHITECTURE_VARIANT,
    D2AG_ARCHITECTURE_VARIANT,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    build_d2ag_relation_source,
    sparse_rest_object_point_cache,
    verify_sparse_rest_object_assets,
)

__all__ = [
    "GaussianDiffusion",
    "HOIPriorSampler",
    "_extract",
    "normalize_progress",
    "prepare_clean_x0",
    "project_object_rotation_x0",
]


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


class GaussianDiffusion(_CoreGaussianDiffusion):
    """HOI reverse loop on the frozen core DDPM.

    The class name and constructor signature are deliberately unchanged: every
    call site still writes ``GaussianDiffusion(int(cfg.diffusion_steps))`` and
    still receives the same buffers, so only the import path moved.
    """

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
        guidance: Optional[object] = None,
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
        selfcond_relation_source = (
            getattr(model, "architecture_variant", None) == D2AG_ARCHITECTURE_VARIANT
        )
        if selfcond_relation_source and not all(
            value is not None for value in relation_values
        ):
            raise ValueError("D2-AG sampling requires the sparse relation metadata set")
        # ``prev_x0`` is a local, so every ``sample`` call - that is, every
        # window - starts a fresh 500-step chain whose first reverse step has no
        # previous estimate and therefore uses the current-state source.
        prev_x0: Optional[torch.Tensor] = None
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((batch,), step, dtype=torch.long, device=current.device)
            if all(value is not None for value in relation_values):
                if local_object_bps is not None:
                    raise ValueError("D2-AE sparse relation metadata cannot be combined with D2-AD local BPS")
                selfcond_arguments = (
                    {
                        "relation_source": build_d2ag_relation_source(
                            current, prev_x0,
                        ),
                    }
                    if selfcond_relation_source and prev_x0 is not None
                    else {}
                )
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
                    **selfcond_arguments,
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
            if selfcond_relation_source:
                # Registered contract: ``prev_x0`` is the raw model ``x0_hat``,
                # captured before ``prepare_clean_x0`` restores history or closes
                # object rotations on SO(3).
                prev_x0 = clean.detach()
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
            # Preregistered P2 inference-time contact guidance.  ``guidance`` is
            # ``None`` for every registered result in this repository, so the
            # default path is bit-identical: no tensor is constructed, no RNG is
            # drawn and no dtype changes.  ``step`` is truthy on every reverse
            # step except the final one, so guidance is never applied at step 0.
            if guidance is not None and step:
                current = guidance.apply(current, clean, fixed_history, step)
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
        guidance: Optional[Mapping[str, object]] = None,
        **_: object,
    ) -> None:
        if auto_regre_num != REPRESENTATION.history_frames:
            raise ValueError("HOIPrior sampler requires exactly two history frames")
        self.device = torch.device(device)
        self.diffusion = GaussianDiffusion(timesteps).to(self.device)
        self.object_so3_x0 = bool(object_so3_x0)
        # Preregistered P2 guidance is inert unless explicitly enabled.  The
        # module is imported lazily so the default evaluation path keeps its
        # existing import graph.
        self.guidance_settings = None
        if guidance is not None:
            from .inference_guidance import GuidanceSettings
            settings = GuidanceSettings.from_config(guidance)
            if settings.enabled:
                self.guidance_settings = settings
        self.guidance_audit = None
        if self.guidance_settings is not None:
            from .inference_guidance import GuidanceAudit
            self.guidance_audit = GuidanceAudit()
        self.guidance_codec = None
        self.guidance_parents_24 = None
        self.guidance_rest_offsets = None
        self.guidance_rest_vertex_cache: Dict[str, torch.Tensor] = {}
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
        if self.guidance_settings is not None:
            from .inference_guidance import GuidanceAudit
            self.guidance_audit = GuidanceAudit()

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
            in {
                D2AE_ARCHITECTURE_VARIANT,
                D2AF_ARCHITECTURE_VARIANT,
                D2AG_ARCHITECTURE_VARIANT,
            }
        )
        self.sparse_rest_point_cache = None
        self.sparse_relation_asset_contract = None
        if self.sparse_relation_enabled:
            required = ("min_torch", "max_torch", "obj_min_torch", "obj_max_torch")
            missing = [name for name in required if not hasattr(dataset, name)]
            if missing:
                raise ValueError(
                    "sparse-relation evaluator dataset is missing normalization "
                    f"tensors: {missing}"
                )
        if self.guidance_settings is not None:
            self._prepare_guidance_context(dataset)

    def _prepare_guidance_context(self, dataset) -> None:
        """Bind the preregistered P2 guidance to the evaluator dataset.

        The evaluation runner (``code/test_infbagel_hoi.py``) is a byte-locked
        evaluator source, so the guidance recovers the per-sequence SMPL rest
        offsets from the dataset itself: ``scene_name[i]`` and
        ``rest_human_offsets[i]`` share one sequence index, exactly the pair the
        runner reads through ``__getitem__``.  The lookup fails closed if the
        names are not unique.
        """
        from .inference_guidance import GuidanceAudit
        required = ("min_torch", "max_torch", "obj_min_torch", "obj_max_torch")
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise ValueError(
                f"HOIPrior guidance dataset is missing normalization tensors: {missing}"
            )
        names = getattr(dataset, "scene_name", None)
        offsets = getattr(dataset, "rest_human_offsets", None)
        parents = getattr(dataset, "parents_24", None)
        if names is None or offsets is None or parents is None:
            raise ValueError(
                "HOIPrior guidance requires dataset scene_name, rest_human_offsets "
                "and parents_24"
            )
        offsets = torch.as_tensor(offsets, dtype=torch.float32)
        if offsets.ndim != 3 or offsets.shape[1:] != (24, 3) or offsets.shape[0] != len(names):
            raise ValueError(
                f"HOIPrior guidance rest offsets must be [sequences,24,3], "
                f"got {tuple(offsets.shape)} for {len(names)} sequences"
            )
        table: Dict[str, torch.Tensor] = {}
        for index, name in enumerate(names):
            key = str(name)
            if key in table and not bool(torch.equal(table[key], offsets[index])):
                raise ValueError(
                    f"HOIPrior guidance found conflicting rest offsets for {key}"
                )
            table[key] = offsets[index]
        self.guidance_rest_offsets = table
        self.guidance_parents_24 = torch.as_tensor(
            parents, dtype=torch.long, device=self.device,
        )
        self.guidance_codec = WindowStateCodec(
            dataset.min_torch,
            dataset.max_torch,
            dataset.obj_min_torch,
            dataset.obj_max_torch,
        )
        self.guidance_rest_vertex_cache = {}
        self.guidance_audit = GuidanceAudit()

    def _build_guidance(
        self, mat, obj_rot_mat_ref, obj_rest_verts, seq_name_dict, batch,
        ground_truth_contact=None, object_goal=None,
    ):
        """Build one window's guidance context from the evaluator's own inputs.

        ``ground_truth_contact`` is the NON-DEPLOYABLE P6 cell-U probe: ground
        truth contact does not exist at inference time, so it may only bound how
        much of the engagement gap guidance could recover given a perfect
        engagement decision.  It must be window-local and aligned frame-for-frame
        with the predicted contact the codec decodes, and it is validated as such
        by :meth:`_validate_ground_truth_contact_window` before use.
        """
        from .contact_guidance import deterministic_vertex_subset
        from .inference_guidance import HOIContactGuidance
        if self.guidance_codec is None:
            raise ValueError("HOIPrior guidance was not bound to a dataset")
        if not isinstance(obj_rest_verts, Mapping):
            raise ValueError("HOIPrior guidance requires the rest-vertex mapping")
        transform = mat.reshape(batch, 4, 4).to(device=self.device, dtype=torch.float32)
        frame = WindowFrame(
            transform[:, :3, 3],
            transform[:, :3, :3].transpose(-1, -2),
            obj_rot_mat_ref.reshape(batch, -1, 3, 3)[:, 0].to(
                device=self.device, dtype=torch.float32,
            ),
        )
        names = [str(seq_name_dict[index]) for index in range(batch)]
        rest_offsets = torch.stack([
            self.guidance_rest_offsets[name] for name in names
        ]).to(device=self.device, dtype=torch.float32)
        surfaces = []
        for name in names:
            object_name = name.split("_")[1]
            cached = self.guidance_rest_vertex_cache.get(object_name)
            if cached is None:
                cached = deterministic_vertex_subset(
                    obj_rest_verts[object_name].to(
                        device=self.device, dtype=torch.float32,
                    )
                )
                self.guidance_rest_vertex_cache[object_name] = cached
            surfaces.append(cached)
        if ground_truth_contact is not None:
            ground_truth_contact = self._validate_ground_truth_contact_window(
                ground_truth_contact, batch,
            )
        goal_world = None
        if object_goal is not None:
            # The codec decodes ``object_translation`` in GLOBAL coordinates
            # (window_codec.py:251) while ``sample_step`` has already converted
            # ``object_goal`` to the WINDOW-LOCAL frame (test_infbagel_hoi.py:377).
            # Differencing them as-is would pull the object toward a meaningless
            # point without raising, so the goal is lifted back to global with
            # the very same frame the decode uses.
            goal_world = WindowStateCodec.global_position(
                object_goal.reshape(batch, 1, 3).to(
                    device=self.device, dtype=torch.float32,
                ),
                frame,
            ).reshape(batch, 3)
        return HOIContactGuidance(
            self.guidance_settings,
            posterior_variance=self.diffusion.posterior_variance,
            codec=self.guidance_codec,
            frame=frame,
            rest_human_offsets=rest_offsets,
            parents_24=self.guidance_parents_24,
            rest_vertices=torch.stack(surfaces),
            audit=self.guidance_audit,
            ground_truth_contact=ground_truth_contact,
            object_goal=goal_world,
        )

    def _validate_ground_truth_contact_window(self, contact, batch):
        """Reject a misaligned or malformed cell-U ground-truth window.

        A misaligned slice does not crash: it produces a plausible-looking
        ceiling, which would then be used to decide whether to abandon
        inference-time guidance entirely.  The shape check below is what makes
        an off-by-``auto_regre_num`` slice impossible to pass silently, since a
        shifted window that ran off the end of a sequence would arrive short.

        The engagement fraction is recorded rather than compared against the
        evaluator's ``gt_contact_percent``: that statistic is a GEOMETRIC
        judgement (GT hand joints within ``contact_threshold`` of the GT object
        surface, ``eval_metrics.py:246``) computed on interpolated evaluation
        frames, whereas this tensor is the dataset's own annotation channel
        thresholded at 0.95.  Measured over all 482 test annotation files those
        two quantities are 0.66188 and 0.62794 respectively -- they are not the
        same statistic, so equating them would make the gate fire on correct
        alignment.
        """
        window_frames = REPRESENTATION.window_frames
        if contact.ndim != 3 or contact.shape[:2] != (batch, window_frames):
            raise ValueError(
                "cell-U ground-truth contact must be window-local "
                f"[{batch},{window_frames},4], got {tuple(contact.shape)}; a "
                "slice of the wrong length means the per-window alignment is "
                "off and the probe is void"
            )
        if contact.shape[-1] != 4:
            raise ValueError(
                "cell-U ground-truth contact must carry the 4 contact channels, "
                f"got {contact.shape[-1]}"
            )
        contact = contact.to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(contact).all()):
            raise ValueError("cell-U ground-truth contact contains non-finite values")
        engaged = (contact[..., -4:-2] > 0.95).any(dim=-1)
        if self.guidance_audit is not None:
            self.guidance_audit.record_ground_truth_window(
                int(engaged.sum()), int(engaged.numel()),
            )
        return contact

    @torch.no_grad()
    def p_sample_loop(
        self, fixed_points, mat, scene_flag, text_emb, pelvis_goal, scene_goal, object_goal,
        need_scene, need_pelvis_dir, pi, end_pi, seq_length, need_pi, is_loco, is_object,
        obj_bps_data, object_points, obj_rot_mat_ref, obj_rest_verts, seq_name_dict,
        obj_rot_mat_prefix=None, object_only=False, ground_truth_contact=None,
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
        guidance_arguments = {}
        if self.guidance_settings is not None:
            guidance_arguments = {
                "guidance": self._build_guidance(
                    mat, obj_rot_mat_ref, obj_rest_verts, seq_name_dict, batch,
                    ground_truth_contact=ground_truth_contact,
                    object_goal=raw_goal,
                ),
            }
            if self.guidance_audit is not None:
                self.guidance_audit.sample_calls += 1
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
            **guidance_arguments,
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
        if self.guidance_settings is not None:
            value["inference_guidance"] = self.guidance_settings.as_dict()
            if self.guidance_audit is not None:
                value["inference_guidance"].update(self.guidance_audit.as_dict())
        return value
