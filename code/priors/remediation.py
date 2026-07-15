"""Deterministic internal-only diagnostics for the Phase 1B remediation gate."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.ops import knn_points

from .representation import REPRESENTATION
from .window_codec import yup_to_zup_tensor


D0_TIMESTEPS: Tuple[int, ...] = (0, 1, 10, 50, 100, 250, 499)


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_internal_triples(dataset, count: int = 128) -> List[Tuple[int, int, int]]:
    """Select the preregistered sequence-disjoint D0 triples.

    Dataset positions, rather than global language-window indices, are returned.
    Each triple begins at ``pi=0`` and advances by the 42 source frames emitted
    by a 16-frame/2-history rollout window.  Sequence selection is exactly the
    hash ordering preregistered in ``docs/EXPERIMENT_PLAN.md``.
    """
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D0 sequence selection is internal-validation only")
    by_sequence: Dict[int, Dict[int, int]] = defaultdict(dict)
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        pi = int(dataset.language["pi"][global_index])
        by_sequence[sequence][pi] = position
    eligible = []
    for sequence, positions in by_sequence.items():
        if all(pi in positions for pi in (0, 42, 84)):
            name = str(dataset.scene_names[sequence])
            eligible.append((stable_digest("42:hoi-remediation:" + name), name, sequence, positions))
    eligible.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} internal sequences form a three-window rollout")
    return [tuple(value[3][pi] for pi in (0, 42, 84)) for value in eligible[:count]]


def select_teacher_windows(dataset, count: int = 512) -> List[int]:
    """Select a stable set of internal windows for every D0 timestep."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D0 teacher-window selection is internal-validation only")
    ranked = []
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        name = str(dataset.scene_names[sequence])
        pi = int(dataset.language["pi"][global_index])
        key = stable_digest(f"42:hoi-remediation-window:{name}:{pi}")
        ranked.append((key, name, pi, position))
    ranked.sort()
    if len(ranked) < count:
        raise ValueError(f"only {len(ranked)} internal windows are available")
    return [value[-1] for value in ranked[:count]]


def deterministic_derangement(count: int, *, device=None) -> torch.Tensor:
    if count < 2:
        raise ValueError("permutation sensitivity needs at least two samples")
    return torch.roll(torch.arange(count, device=device), shifts=1)


def field_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return non-history MSE for each locked 232-D representation field."""
    if prediction.shape != target.shape or prediction.shape[-1] != REPRESENTATION.dimension:
        raise ValueError(f"expected matching [...,232] tensors, got {prediction.shape}/{target.shape}")
    return {
        field.name: (
            prediction[:, REPRESENTATION.history_frames:, field.slice]
            - target[:, REPRESENTATION.history_frames:, field.slice]
        ).square().mean()
        for field in REPRESENTATION.fields
    }


def selection_sha256(values: Iterable[object]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bps_replay_equivalence_gate(
    recomputed: torch.Tensor,
    stored: torch.Tensor,
    basis: torch.Tensor,
    transformed_vertices: torch.Tensor,
    selected_vertex_indices: torch.Tensor,
    *,
    strict_component_max_abs: float = 1e-4,
    stored_mesh_residual_m_max: float = 1e-6,
    recomputed_mesh_residual_m_max: float = 1e-6,
    nearest_squared_distance_gap_m2_max: float = 1e-7,
    nearest_linear_distance_gap_m_max: float = float("inf"),
) -> Dict[str, object]:
    """Apply the D2-C strict-or-provable-tie BPS replay gate.

    This is an audit-only comparison against stored GT BPS.  It does not
    participate in sampler conditioning; generated BPS continues to come only
    from ``basis``, the immutable PLY vertices, and the current object pose.
    """
    if recomputed.shape != stored.shape or recomputed.ndim != 2 or recomputed.shape[-1] != 3:
        raise ValueError(f"expected matching [P,3] BPS tensors, got {recomputed.shape}/{stored.shape}")
    if basis.shape != recomputed.shape:
        raise ValueError(f"basis shape mismatch: {basis.shape}/{recomputed.shape}")
    if transformed_vertices.ndim != 2 or transformed_vertices.shape[-1] != 3:
        raise ValueError(f"expected [V,3] transformed vertices, got {transformed_vertices.shape}")
    if selected_vertex_indices.shape != recomputed.shape[:-1]:
        raise ValueError("selected vertex indices must have one value per basis point")
    if selected_vertex_indices.dtype != torch.long:
        raise ValueError("selected vertex indices must be torch.long")
    if bool((selected_vertex_indices < 0).any()) or bool(
        (selected_vertex_indices >= transformed_vertices.shape[0]).any()
    ):
        raise ValueError("selected vertex index is outside the immutable PLY")

    point_count = recomputed.shape[0]
    component_error = (recomputed - stored).abs().max(dim=-1).values
    mesh_finite = bool(torch.isfinite(transformed_vertices).all())
    finite = (
        torch.isfinite(recomputed).all(dim=-1)
        & torch.isfinite(stored).all(dim=-1)
        & torch.isfinite(basis).all(dim=-1)
        & mesh_finite
    )
    strict = finite & (component_error <= strict_component_max_abs)
    candidates = finite & ~strict
    tie = torch.zeros(point_count, dtype=torch.bool, device=recomputed.device)
    stored_indices = torch.full(
        (point_count,), -1, dtype=torch.long, device=recomputed.device,
    )
    proof_dtype = torch.float64
    stored_residual = torch.full(
        (point_count,), float("inf"), dtype=proof_dtype, device=recomputed.device,
    )
    recomputed_residual = torch.full(
        (point_count,), float("inf"), dtype=proof_dtype, device=recomputed.device,
    )
    squared_distance_gap = torch.full(
        (point_count,), float("inf"), dtype=proof_dtype, device=recomputed.device,
    )
    linear_distance_gap = torch.full(
        (point_count,), float("inf"), dtype=proof_dtype, device=recomputed.device,
    )
    if bool(candidates.any()):
        rows = torch.nonzero(candidates).flatten()
        stored_closest = basis[rows] + yup_to_zup_tensor(stored[rows])
        stored_nearest = knn_points(
            stored_closest[None], transformed_vertices[None], K=1, return_nn=True,
        )
        candidate_stored_indices = stored_nearest.idx[0, :, 0]
        stored_indices[rows] = candidate_stored_indices
        stored_vertices = transformed_vertices[candidate_stored_indices]
        selected_vertices = transformed_vertices[selected_vertex_indices[rows]]
        recomputed_closest = basis[rows] + yup_to_zup_tensor(recomputed[rows])
        # The points themselves define the two nearest distances; PLY vertices
        # independently prove that each point is on the immutable mesh.  Use
        # float64 for the proof so subtracting two ~1 m^2 distances does not
        # quantize the 1e-7 m^2 gate to float32 ULPs.
        candidate_stored_residual = (
            stored_closest.to(proof_dtype) - stored_vertices.to(proof_dtype)
        ).norm(dim=-1)
        candidate_recomputed_residual = (
            recomputed_closest.to(proof_dtype) - selected_vertices.to(proof_dtype)
        ).norm(dim=-1)
        stored_distance = stored[rows].to(proof_dtype).square().sum(dim=-1)
        recomputed_distance = recomputed[rows].to(proof_dtype).square().sum(dim=-1)
        candidate_gap = (stored_distance - recomputed_distance).abs()
        candidate_linear_gap = (
            torch.sqrt(stored_distance) - torch.sqrt(recomputed_distance)
        ).abs()
        stored_residual[rows] = candidate_stored_residual
        recomputed_residual[rows] = candidate_recomputed_residual
        squared_distance_gap[rows] = candidate_gap
        linear_distance_gap[rows] = candidate_linear_gap
        tie[rows] = (
            torch.isfinite(candidate_stored_residual)
            & torch.isfinite(candidate_recomputed_residual)
            & torch.isfinite(candidate_gap)
            & torch.isfinite(candidate_linear_gap)
            & (candidate_stored_residual <= stored_mesh_residual_m_max)
            & (candidate_recomputed_residual <= recomputed_mesh_residual_m_max)
            & (candidate_gap <= nearest_squared_distance_gap_m2_max)
            & (candidate_linear_gap <= nearest_linear_distance_gap_m_max)
        )
    accepted = strict | tie
    failure = ~accepted
    return {
        "passed": not bool(failure.any()),
        "component_error": component_error,
        "finite": finite,
        "strict": strict,
        "tie": tie,
        "failure": failure,
        "stored_vertex_indices": stored_indices,
        "stored_mesh_residual_m": stored_residual,
        "recomputed_mesh_residual_m": recomputed_residual,
        "nearest_squared_distance_gap_m2": squared_distance_gap,
        "nearest_linear_distance_gap_m": linear_distance_gap,
    }
