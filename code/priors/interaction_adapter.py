"""The fixed Phase 1B D2-AC local-object interaction adapter.

This module deliberately contains the complete, immutable BPS partition
contract.  The partition is derived from ``code/bps.pt`` only; no dataset,
contact, evaluator, or train-statistics input is permitted here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


BPS_SHA256 = "fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042"
ASSIGNMENT_SHA256 = (
    "b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f33607959fddeecd"
)
CANONICAL_ALGORITHM = "lexicographic-seed-farthest-point-16-v1"
CENTER_INDICES = (
    328, 903, 503, 817, 474, 1023, 382, 864,
    640, 431, 445, 960, 547, 829, 545, 756,
)
CLUSTER_SIZES = (
    39, 40, 57, 61, 65, 68, 70, 134,
    77, 64, 59, 79, 43, 46, 84, 38,
)
ROLE_NAMES = ("left_hand", "right_hand", "object_motion")
LOCAL_TOKEN_COUNT = 16
LOCAL_FEATURE_DIMENSION = 10
ADAPTER_DIMENSION = 128
ADAPTER_PARAMETER_COUNT = 349_697
BASE_PARAMETER_COUNT_512 = 29_673_448
TOTAL_PARAMETER_COUNT_512 = 30_023_145


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_assignment_payload(centers: Tuple[int, ...], assignment: np.ndarray) -> Dict[str, object]:
    return {
        "algorithm": CANONICAL_ALGORITHM,
        "centers": [int(value) for value in centers],
        "assignments": [int(value) for value in assignment.tolist()],
    }


def _canonical_assignment_sha256(payload: Dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_bps_partition(
    bps_path: Optional[str | Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Load and verify the fixed BPS basis and deterministic 16-way partition.

    Returns ``(basis, assignment, basis_means, cluster_sizes, metadata)``.
    The returned basis and means are float32 tensors; assignment and sizes are
    int64 tensors.  The metadata contains the canonical payload and hashes used
    in manifests and CPU contracts.
    """

    path = (
        Path(bps_path).resolve()
        if bps_path is not None
        else Path(__file__).resolve().parents[1] / "bps.pt"
    )
    actual_file_hash = _sha256_file(path)
    if actual_file_hash != BPS_SHA256:
        raise ValueError(
            f"D2-AC immutable BPS hash mismatch: {actual_file_hash} != {BPS_SHA256}"
        )
    value = torch.load(str(path), map_location="cpu")
    if not isinstance(value, dict) or "obj" not in value:
        raise ValueError("D2-AC BPS file must contain an 'obj' tensor")
    basis_value = value["obj"]
    if not torch.is_tensor(basis_value) or tuple(basis_value.shape) != (1, 1024, 3):
        raise ValueError(
            f"D2-AC BPS basis must have shape [1,1024,3], got {getattr(basis_value, 'shape', None)}"
        )
    if not torch.isfinite(basis_value).all():
        raise ValueError("D2-AC BPS basis is non-finite")

    # The canonical calculation intentionally uses float64 NumPy values.  This
    # is the exact serialization/calculation bound in the preregistration.
    coordinates = np.asarray(basis_value.detach().cpu(), dtype=np.float64).reshape(1024, 3)
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    centers = [int(order[0])]
    minimum_distance = ((coordinates - coordinates[centers[0]]) ** 2).sum(axis=1)
    for _ in range(LOCAL_TOKEN_COUNT - 1):
        maximum = float(minimum_distance.max())
        # flatnonzero is ascending, so this is the smallest original index on
        # an exact distance tie.
        next_index = int(np.flatnonzero(minimum_distance == maximum)[0])
        centers.append(next_index)
        next_distance = ((coordinates - coordinates[next_index]) ** 2).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, next_distance)
    if tuple(centers) != CENTER_INDICES:
        raise ValueError(f"D2-AC center mismatch: {centers} != {list(CENTER_INDICES)}")

    center_coordinates = coordinates[np.asarray(centers)]
    distances = ((coordinates[:, None, :] - center_coordinates[None, :, :]) ** 2).sum(axis=2)
    assignment = distances.argmin(axis=1).astype(np.int16)
    sizes = np.bincount(assignment, minlength=LOCAL_TOKEN_COUNT).astype(np.int64)
    if tuple(int(value) for value in sizes.tolist()) != CLUSTER_SIZES:
        raise ValueError(
            f"D2-AC cluster-size mismatch: {sizes.tolist()} != {list(CLUSTER_SIZES)}"
        )
    payload = _canonical_assignment_payload(tuple(centers), assignment)
    assignment_hash = _canonical_assignment_sha256(payload)
    if assignment_hash != ASSIGNMENT_SHA256:
        raise ValueError(
            "D2-AC assignment hash mismatch: "
            f"{assignment_hash} != {ASSIGNMENT_SHA256}"
        )

    basis = basis_value[0].detach().to(dtype=torch.float32).contiguous()
    assignment_tensor = torch.from_numpy(assignment.astype(np.int64, copy=False))
    sizes_tensor = torch.from_numpy(sizes)
    basis_means = torch.stack(
        [basis[assignment_tensor == index].mean(dim=0) for index in range(LOCAL_TOKEN_COUNT)],
        dim=0,
    )
    metadata: Dict[str, object] = {
        "bps_path": str(path),
        "bps_sha256": actual_file_hash,
        "algorithm": CANONICAL_ALGORITHM,
        "center_indices": list(CENTER_INDICES),
        "cluster_sizes": list(CLUSTER_SIZES),
        "assignment_sha256": assignment_hash,
        "canonical_assignment_payload": payload,
    }
    return basis, assignment_tensor, basis_means, sizes_tensor, metadata


def cluster_bps_features(
    object_bps: torch.Tensor,
    basis: torch.Tensor,
    assignment: torch.Tensor,
    basis_means: torch.Tensor,
    cluster_sizes: torch.Tensor,
    *,
    permute_delta_statistics: bool = False,
) -> torch.Tensor:
    """Build the fixed ``[B,16,10]`` local BPS statistics.

    The optional permutation is the preregistered causal locality probe:
    output cluster ``k`` receives only the delta statistics from ``k+8``.
    Basis means and object identity embeddings remain in their original order.
    """

    if object_bps.ndim != 3 or tuple(object_bps.shape[1:]) != (1024, 3):
        raise ValueError(f"expected object_bps [B,1024,3], got {tuple(object_bps.shape)}")
    if not torch.is_floating_point(object_bps):
        raise TypeError("object_bps must be floating point")
    batch = object_bps.shape[0]
    device = object_bps.device
    output_dtype = object_bps.dtype
    accumulation_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    deltas = object_bps.to(dtype=accumulation_dtype)
    assignment = assignment.to(device=device, dtype=torch.long)
    cluster_sizes = cluster_sizes.to(device=device, dtype=accumulation_dtype)
    index = assignment.view(1, -1, 1).expand(batch, -1, 3)
    sums = torch.zeros(
        batch, LOCAL_TOKEN_COUNT, 3, device=device, dtype=accumulation_dtype
    )
    sums.scatter_add_(1, index, deltas)
    mean_delta = sums / cluster_sizes.view(1, -1, 1)
    squared_sums = torch.zeros_like(sums)
    squared_sums.scatter_add_(1, index, deltas * deltas)
    rms_delta = torch.sqrt(torch.clamp(
        squared_sums / cluster_sizes.view(1, -1, 1), min=0.0
    ))
    norm_sums = torch.zeros(
        batch, LOCAL_TOKEN_COUNT, device=device, dtype=accumulation_dtype
    )
    norm_sums.scatter_add_(1, assignment.view(1, -1).expand(batch, -1),
                          torch.linalg.vector_norm(deltas, dim=-1))
    mean_norm = norm_sums / cluster_sizes.view(1, -1)

    if permute_delta_statistics:
        # ``new[k] = old[(k+8) mod 16]``; +/-8 are identical for 16 tokens.
        mean_delta = torch.roll(mean_delta, shifts=-8, dims=1)
        rms_delta = torch.roll(rms_delta, shifts=-8, dims=1)
        mean_norm = torch.roll(mean_norm, shifts=-8, dims=1)
    basis_means = basis_means.to(device=device, dtype=accumulation_dtype)
    features = torch.cat((
        basis_means.view(1, LOCAL_TOKEN_COUNT, 3).expand(batch, -1, -1),
        mean_delta,
        rms_delta,
        mean_norm.unsqueeze(-1),
    ), dim=-1)
    return features.to(dtype=output_dtype)


class LocalObjectInteractionAdapter(nn.Module):
    """The single fixed D2-AC adapter inserted between trunk layers 4 and 5."""

    def __init__(
        self,
        dim_model: int = 512,
        *,
        bps_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        if dim_model != 512:
            raise ValueError("D2-AC is locked to a 512-wide HOIPrior trunk")
        basis, assignment, basis_means, cluster_sizes, metadata = load_bps_partition(bps_path)
        self.register_buffer("bps_basis", basis, persistent=False)
        self.register_buffer("cluster_assignment", assignment, persistent=False)
        self.register_buffer("cluster_basis_means", basis_means, persistent=False)
        self.register_buffer("cluster_sizes", cluster_sizes, persistent=False)
        self.partition_metadata = metadata

        self.object_encoder = nn.Sequential(
            nn.Linear(LOCAL_FEATURE_DIMENSION, ADAPTER_DIMENSION),
            nn.SiLU(),
            nn.Linear(ADAPTER_DIMENSION, ADAPTER_DIMENSION),
        )
        self.object_identity = nn.Parameter(torch.empty(LOCAL_TOKEN_COUNT, ADAPTER_DIMENSION))
        self.object_norm = nn.LayerNorm(ADAPTER_DIMENSION)
        self.query_projection = nn.Linear(dim_model, ADAPTER_DIMENSION)
        self.part_embedding = nn.Parameter(torch.empty(len(ROLE_NAMES), ADAPTER_DIMENSION))
        self.query_norm = nn.LayerNorm(ADAPTER_DIMENSION)
        self.cross_attention = nn.MultiheadAttention(
            ADAPTER_DIMENSION, 4, dropout=0.0, batch_first=True,
        )
        self.writeback = nn.Linear(len(ROLE_NAMES) * ADAPTER_DIMENSION, dim_model)
        self.alpha = nn.Parameter(torch.zeros(()))

        nn.init.normal_(self.object_identity, std=0.02)
        nn.init.normal_(self.part_embedding, std=0.02)
        self._gate_override: Optional[float] = None
        self._permute_delta_statistics = False
        self._capture_attention = False
        self._last_attention = None

    @property
    def gate(self) -> torch.Tensor:
        if self._gate_override is None:
            return torch.tanh(self.alpha)
        return self.alpha.new_tensor(float(self._gate_override))

    def set_diagnostic_variant(self, variant: str) -> None:
        if variant not in {"full", "gate_ablated", "local_correspondence_permuted"}:
            raise ValueError(f"unknown D2-AC diagnostic variant: {variant}")
        self._gate_override = 0.0 if variant == "gate_ablated" else None
        self._permute_delta_statistics = variant == "local_correspondence_permuted"

    def set_gate_override(self, coefficient: Optional[float]) -> None:
        self._gate_override = None if coefficient is None else float(coefficient)

    def set_capture_attention(self, enabled: bool) -> None:
        self._capture_attention = bool(enabled)
        if not enabled:
            self._last_attention = None

    def clear_diagnostic_state(self) -> None:
        self.set_diagnostic_variant("full")
        self.set_capture_attention(False)

    def local_features(self, object_bps: torch.Tensor) -> torch.Tensor:
        return cluster_bps_features(
            object_bps,
            self.bps_basis,
            self.cluster_assignment,
            self.cluster_basis_means,
            self.cluster_sizes,
            permute_delta_statistics=self._permute_delta_statistics,
        )

    def forward(
        self, motion_tokens: torch.Tensor, object_bps: torch.Tensor,
    ) -> torch.Tensor:
        if motion_tokens.ndim != 3 or motion_tokens.shape[-1] != 512:
            raise ValueError(
                f"expected contextualized motion tokens [B,T,512], got {tuple(motion_tokens.shape)}"
            )
        features = self.local_features(object_bps)
        objects = self.object_norm(
            self.object_encoder(features)
            + self.object_identity.to(dtype=features.dtype, device=features.device).view(
                1, LOCAL_TOKEN_COUNT, ADAPTER_DIMENSION
            )
        )
        queries = self.query_projection(motion_tokens).unsqueeze(2)
        queries = queries + self.part_embedding.to(
            dtype=queries.dtype, device=queries.device
        ).view(1, 1, len(ROLE_NAMES), ADAPTER_DIMENSION)
        queries = self.query_norm(queries)
        batch, frames, roles, width = queries.shape
        query_flat = queries.reshape(batch, frames * roles, width)
        attended, weights = self.cross_attention(
            query_flat, objects, objects,
            need_weights=self._capture_attention,
            average_attn_weights=False,
        )
        if self._capture_attention:
            # [B,heads,T*roles,K] -> [B,T,roles,heads,K]
            self._last_attention = weights.reshape(
                batch, self.cross_attention.num_heads, frames, roles, LOCAL_TOKEN_COUNT
            ).permute(0, 2, 3, 1, 4).detach()
        else:
            self._last_attention = None
        writeback = self.writeback(
            attended.reshape(batch, frames, roles * ADAPTER_DIMENSION)
        )
        return motion_tokens + self.gate.to(dtype=writeback.dtype) * writeback

    def attention_snapshot(self):
        return self._last_attention

    def contract_metadata(self) -> Dict[str, object]:
        return {
            **self.partition_metadata,
            "roles": list(ROLE_NAMES),
            "local_token_count": LOCAL_TOKEN_COUNT,
            "local_feature_dimension": LOCAL_FEATURE_DIMENSION,
            "adapter_dimension": ADAPTER_DIMENSION,
            "attention_heads": self.cross_attention.num_heads,
            "attention_dropout": 0.0,
            "object_encoder": [10, 128, 128],
            "query_projection": [512, 128],
            "writeback_projection": [384, 512],
            "rezero_gate": "tanh(alpha)",
            "alpha_initial": 0.0,
            "adapter_parameters": ADAPTER_PARAMETER_COUNT,
        }
