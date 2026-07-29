"""D2-AE GPU-native sparse current-state role-relative object field.

The learned relation path is intentionally torch-only.  It consumes the current
diffusion state and immutable sparse object metadata; it never reads a clean
target, a scene tensor, contact labels, or a stored per-window relation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .diffusion_schedule import (
    DIFFUSION_STEPS,
    SQRT_ALPHA_BAR_SHA256,
    canonical_diffusion_schedule,
    diffusion_schedule_contract_metadata,
    tensor_sha256,
)
from .representation import REPRESENTATION
from .window_codec import project_to_so3


ARCHITECTURE_VARIANT = "d2ae_sparse_relation_field"
D2AF_ARCHITECTURE_VARIANT = "d2af_sqrt_alpha_bar_reliability"
TEMPORAL_ANCHORS: Tuple[int, ...] = (0, 5, 10, 15)
ROLE_JOINTS: Tuple[int, ...] = (24, 26, 0)
ROLE_NAMES: Tuple[str, ...] = ("left_hand", "right_hand", "pelvis")
TEMPORAL_SOURCE_PERMUTATION: Tuple[int, ...] = (2, 3, 0, 1)
ROLE_SWAP_PERMUTATION: Tuple[int, ...] = (1, 0, 2)
ROUTING_SLOTS: Tuple[int, ...] = (
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3,
)
DIAGNOSTIC_VARIANTS = frozenset({
    "full",
    "relation_gate_ablated",
    "temporal_correspondence_permuted",
    "left_right_role_swapped",
})
D2AF_DIAGNOSTIC_VARIANTS = frozenset((*DIAGNOSTIC_VARIANTS, "unit_rho"))

POINT_ENCODER_PARAMETER_COUNT = 17_152
PROJECTION_PARAMETER_COUNT = 393_728
TEMPORAL_EMBEDDING_PARAMETER_COUNT = 2_048
LAYER_NORM_PARAMETER_COUNT = 1_024
ALPHA_PARAMETER_COUNT = 1
SPARSE_RELATION_PARAMETER_COUNT = 413_953
BASE_PARAMETER_COUNT = 29_673_448
TOTAL_PARAMETER_COUNT = 30_087_401
PARAMETER_INCREASE_FRACTION = SPARSE_RELATION_PARAMETER_COUNT / BASE_PARAMETER_COUNT

OBJECT_NAMES: Tuple[str, ...] = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "monitor",
    "plasticbox",
    "smallbox",
    "smalltable",
    "suitcase",
    "trashcan",
    "tripod",
    "whitechair",
    "woodchair",
)
REST_MESH_SHA256: Mapping[str, str] = {
    "clothesstand.ply": "f7967db79cfa76c2dd154426c0e110369b5f28f756dcd9df5e5c9ecba3fed960",
    "floorlamp.ply": "c012a69bfb37542ad099ff087b17059b2622a47d2e04e09d67403ea929470e4a",
    "largebox.ply": "6b9ef6cf4564c0a9d052796b2f24bcb0557249304a6c417f25bfcacec50106e0",
    "largetable.ply": "087c3575a25b0c423a9dcc7ceea2519ff604ce0b8172a5e66e89eefd2d3cb7d1",
    "monitor.ply": "21f3bbae53b678eff92f34d597f18a7cdbb04db3195425f6c64c6a2790129bc6",
    "plasticbox.ply": "e0a9c29124314f02d6b189b36716c5a01aeeb973b4001e8f28ee76f3382a1a76",
    "smallbox.ply": "2aa65b4bf36476ffec1bf4582f2aded77bf111956c70d127d44e8bc76b2072b8",
    "smalltable.ply": "eb601c69e2aa96b2cf284dccdf63e2c49f192200f3dc298e5cd6143b42a25628",
    "suitcase.ply": "6ebfb9e5bf732314fe8fbd5dbf1ad500b4b2218fab74888dca7cf3fc2ad49845",
    "trashcan.ply": "7381b83dc33265872753b8425ec7da4a24da0fc0512b57ebda3307c298659d38",
    "tripod.ply": "b2222da71906a832826bd15b697e1a21e69627664da97f62704667fe7e871540",
    "whitechair.ply": "f3cc5036d305cbe5e53a701cdfa63c9181113e6b2249860bf190ab0a8a0bedd0",
    "woodchair.ply": "45d3e42b3f4cb537efc94e37d5919ff1a70967c4ce4c8ad399a0882e5d356a89",
}
SPARSE_POINT_MAPPING_SHA256 = (
    "1af35119c1dd54e2ad44c99f3cb91b62c1b88f62ca80cddcc96f4b201ffe0f5b"
)
SPARSE_POINT_MANIFEST_SHA256 = (
    "e88d74a7ee434f3e6320c95d1ebb74efdc8fe4740b70ff596e502666a096f7a7"
)
SPARSE_POINT_TENSOR_SHA256 = (
    "793dad6a805d0a908087b273590bf171e7bce4c026297cf94d40f8c651fe4cab"
)
SPARSE_POINT_SELECTION = (
    "numpy.linspace(0,n-1,min(100,n)).round().astype(int64)"
)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    cpu = value.detach().cpu().contiguous()
    return bytes(cpu.view(torch.uint8).reshape(-1).tolist())


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sparse_point_mapping_payload() -> Dict[str, object]:
    return {
        "algorithm": "sequence-name-second-underscore-field-v1",
        "names": list(OBJECT_NAMES),
    }


def sparse_point_indices(vertex_count: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
    count = int(vertex_count)
    if count < 100:
        raise ValueError(f"D2-AE rest object has {count} vertices, expected at least 100")
    return torch.linspace(
        0,
        count - 1,
        100,
        dtype=torch.float64,
        device=device,
    ).round().to(torch.long)


def select_sparse_rest_object_points(rest_vertices: torch.Tensor) -> torch.Tensor:
    """Select the immutable D2-X 100-point subset without a geometry query."""
    if rest_vertices.ndim != 2 or rest_vertices.shape[-1] != 3:
        raise ValueError(
            f"expected immutable rest vertices [N,3], got {tuple(rest_vertices.shape)}"
        )
    indices = sparse_point_indices(rest_vertices.shape[0], device=rest_vertices.device)
    return rest_vertices.index_select(0, indices).to(dtype=torch.float32).contiguous()


def sparse_rest_object_point_cache(
    rest_vertices_by_name: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    names = tuple(sorted(str(name) for name in rest_vertices_by_name))
    if names != OBJECT_NAMES:
        raise ValueError(f"D2-AE immutable rest-object mapping mismatch: {names}")
    return {
        name: select_sparse_rest_object_points(rest_vertices_by_name[name])
        for name in OBJECT_NAMES
    }


def sparse_point_manifest_payload(
    rest_vertices_by_name: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    cache = sparse_rest_object_point_cache(rest_vertices_by_name)
    records = []
    for name in OBJECT_NAMES:
        vertices = rest_vertices_by_name[name]
        indices = sparse_point_indices(vertices.shape[0])
        points = cache[name].cpu()
        records.append({
            "name": name,
            "source_mesh": f"{name}.ply",
            "source_mesh_sha256": REST_MESH_SHA256[f"{name}.ply"],
            "source_vertex_count": int(vertices.shape[0]),
            "point_count": int(points.shape[0]),
            "selection": SPARSE_POINT_SELECTION,
            "indices_sha256": _tensor_sha256(indices.to(torch.int64)),
            "points_float32_yup_sha256": _tensor_sha256(points.to(torch.float32)),
        })
    return {
        "algorithm": "d2x-rest-object-points-100-yup-linspace-vertex-v1",
        "objects": records,
    }


def verify_sparse_rest_object_assets(
    rest_vertices_by_name: Mapping[str, torch.Tensor],
    *,
    mesh_root: Optional[Path] = None,
) -> Dict[str, object]:
    if _canonical_json_sha256(sparse_point_mapping_payload()) != SPARSE_POINT_MAPPING_SHA256:
        raise ValueError("D2-AE sparse object-name mapping hash mismatch")
    if mesh_root is not None:
        actual = {
            path.name: _sha256_file(path)
            for path in sorted(Path(mesh_root).glob("*.ply"))
        }
        if actual != dict(REST_MESH_SHA256):
            raise ValueError("D2-AE immutable rest-mesh hashes do not match")
    manifest = sparse_point_manifest_payload(rest_vertices_by_name)
    manifest_sha256 = _canonical_json_sha256(manifest)
    if manifest_sha256 != SPARSE_POINT_MANIFEST_SHA256:
        raise ValueError(
            "D2-AE sparse point manifest hash mismatch: "
            f"{manifest_sha256} != {SPARSE_POINT_MANIFEST_SHA256}"
        )
    cache = sparse_rest_object_point_cache(rest_vertices_by_name)
    stacked = torch.stack([cache[name].cpu() for name in OBJECT_NAMES]).contiguous()
    tensor_sha256 = _tensor_sha256(stacked)
    if tensor_sha256 != SPARSE_POINT_TENSOR_SHA256:
        raise ValueError(
            "D2-AE sparse point tensor hash mismatch: "
            f"{tensor_sha256} != {SPARSE_POINT_TENSOR_SHA256}"
        )
    return {
        "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
        "manifest_sha256": manifest_sha256,
        "stacked_tensor_sha256": tensor_sha256,
        "shape": list(stacked.shape),
    }


def sparse_relation_contract_metadata() -> Dict[str, object]:
    """Return the exact architecture and provenance contract stored in checkpoints."""
    return {
        "architecture_variant": ARCHITECTURE_VARIANT,
        "temporal_anchors": list(TEMPORAL_ANCHORS),
        "roles": list(ROLE_NAMES),
        "role_joints": list(ROLE_JOINTS),
        "point_encoder": [4, 128, 128],
        "activation": "SiLU",
        "pooling": ["mean", "max"],
        "role_concat_order": list(ROLE_NAMES),
        "projection": [768, 512],
        "temporal_embeddings": 4,
        "routing_slots": list(ROUTING_SLOTS),
        "placement": "after_motion_input_before_condition_concat_position_and_full_trunk",
        "gate": "tanh(alpha)",
        "alpha_initial": 0.0,
        "sparse_relation_parameters": SPARSE_RELATION_PARAMETER_COUNT,
        "base_parameters": BASE_PARAMETER_COUNT,
        "total_parameters": TOTAL_PARAMETER_COUNT,
        "parameter_increase_fraction": PARAMETER_INCREASE_FRACTION,
        "rest_object_points": [100, 3],
        "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
        "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
        "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
        "current_state_only": True,
        "clean_target_used": False,
        "future_gt_used": False,
        "scene_used": False,
        "contact_used": False,
        "stored_relation_used": False,
    }


def validate_sparse_relation_contract(value: object) -> Dict[str, object]:
    """Reject checkpoints that do not carry the complete locked D2-AE contract."""
    if not isinstance(value, Mapping):
        raise ValueError("D2-AE checkpoint is missing sparse-relation provenance")
    expected = sparse_relation_contract_metadata()
    mismatches = sorted(
        key for key, expected_value in expected.items()
        if value.get(key) != expected_value
    )
    unexpected = sorted(set(value) - set(expected))
    if mismatches or unexpected:
        detail = []
        if mismatches:
            detail.append("mismatched=" + ",".join(mismatches))
        if unexpected:
            detail.append("unexpected=" + ",".join(unexpected))
        raise ValueError(
            "D2-AE checkpoint sparse-relation provenance mismatch: "
            + "; ".join(detail)
        )
    return dict(value)


def diffusion_reliability_contract_metadata() -> Dict[str, object]:
    """Return the independent D2-AF architecture/provenance contract."""
    value = sparse_relation_contract_metadata()
    value.update({
        "architecture_variant": D2AF_ARCHITECTURE_VARIANT,
        "writeback": (
            "motion + sqrt_alpha_bar[current_timestep] "
            "* tanh(alpha) * routed_relation"
        ),
        "current_timestep_per_sample": True,
        "train_sample_symmetric": True,
        "unit_rho_training": False,
        "unit_rho_diagnostic_only": True,
        "global_scaling_including_history_anchor": True,
        "per_anchor_scaling": False,
        "learned_schedule": False,
        "loss_or_snr_weighting": False,
        "schedule_buffer_persistent": False,
        "schedule": diffusion_schedule_contract_metadata(),
    })
    return value


def validate_diffusion_reliability_contract(value: object) -> Dict[str, object]:
    """Reject checkpoints that do not carry the complete locked D2-AF contract."""
    if not isinstance(value, Mapping):
        raise ValueError("D2-AF checkpoint is missing diffusion-reliability provenance")
    expected = diffusion_reliability_contract_metadata()
    mismatches = sorted(
        key for key, expected_value in expected.items()
        if value.get(key) != expected_value
    )
    unexpected = sorted(set(value) - set(expected))
    if mismatches or unexpected:
        detail = []
        if mismatches:
            detail.append("mismatched=" + ",".join(mismatches))
        if unexpected:
            detail.append("unexpected=" + ",".join(unexpected))
        raise ValueError(
            "D2-AF checkpoint diffusion-reliability provenance mismatch: "
            + "; ".join(detail)
        )
    return dict(value)


def _metadata_tensor(
    value: torch.Tensor,
    current: torch.Tensor,
    shape: Tuple[int, ...],
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"expected D2-AE {name} {shape}, got {getattr(value, 'shape', None)}")
    return value.to(device=current.device, dtype=current.dtype)


def _normalization_tensor(value: torch.Tensor, current: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() != 3:
        raise ValueError(f"expected D2-AE {name} with three values")
    return value.reshape(3).to(device=current.device, dtype=current.dtype)


def build_sparse_relation_geometry(
    current: torch.Tensor,
    rest_object_points: torch.Tensor,
    world_to_local_rotation: torch.Tensor,
    object_rotation_reference: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build D2-AE geometry from the current state only.

    Returns point features ``[B,4,3,100,4]`` and the intermediate surfaces and
    role anchors needed by parity/causal-contract tests.
    """
    if current.ndim != 3 or current.shape[1:] != (
        REPRESENTATION.window_frames,
        REPRESENTATION.dimension,
    ):
        raise ValueError(f"expected current D2-AE state [B,16,232], got {tuple(current.shape)}")
    batch = current.shape[0]
    points = _metadata_tensor(
        rest_object_points,
        current,
        (batch, 100, 3),
        "rest_object_points",
    )
    world_to_local = _metadata_tensor(
        world_to_local_rotation,
        current,
        (batch, 3, 3),
        "world_to_local_rotation",
    )
    reference = _metadata_tensor(
        object_rotation_reference,
        current,
        (batch, 3, 3),
        "object_rotation_reference",
    )
    position_minimum = _normalization_tensor(position_minimum, current, "position_minimum")
    position_maximum = _normalization_tensor(position_maximum, current, "position_maximum")
    object_minimum = _normalization_tensor(object_minimum, current, "object_minimum")
    object_maximum = _normalization_tensor(object_maximum, current, "object_maximum")
    anchors = torch.as_tensor(TEMPORAL_ANCHORS, device=current.device, dtype=torch.long)
    state = current.index_select(1, anchors)

    position_scale = (position_maximum - position_minimum).reshape(1, 1, 1, 3)
    position_base = position_minimum.reshape(1, 1, 1, 3)
    joints = (
        (state[..., :84].reshape(batch, 4, 28, 3) + 1.0)
        * position_scale
        / 2.0
        + position_base
    )
    object_scale = (object_maximum - object_minimum).reshape(1, 1, 3)
    object_base = object_minimum.reshape(1, 1, 3)
    object_translation = (
        (state[..., 216:219] + 1.0) * object_scale / 2.0 + object_base
    )
    relative = project_to_so3(state[..., 219:228].reshape(batch, 4, 3, 3))
    global_rotation = project_to_so3(relative @ reference[:, None])
    local_rotation = world_to_local[:, None] @ global_rotation
    surface = torch.einsum("bpc,bfdc->bfpd", points, local_rotation)
    surface = surface + object_translation[:, :, None]

    roles = torch.as_tensor(ROLE_JOINTS, device=current.device, dtype=torch.long)
    role_anchors = joints.index_select(2, roles)
    delta = surface[:, :, None] - role_anchors[:, :, :, None]
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    features = torch.cat((delta, distance), dim=-1)
    return {
        "features": features,
        "surface": surface,
        "role_anchors": role_anchors,
        "joints": joints,
        "object_translation": object_translation,
        "relative_rotation": relative,
        "local_rotation": local_rotation,
    }


class SparseCurrentStateRelationField(nn.Module):
    """Encode and route the fixed D2-AE/D2-AF sparse relation residual."""

    def __init__(
        self,
        dim_model: int,
        *,
        diffusion_reliability: bool = False,
    ) -> None:
        super().__init__()
        if int(dim_model) != 512:
            raise ValueError("sparse relation field requires dim_model=512")
        self.diffusion_reliability = bool(diffusion_reliability)
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.projection = nn.Linear(768, dim_model)
        self.temporal_embeddings = nn.Parameter(torch.empty(4, dim_model))
        nn.init.normal_(self.temporal_embeddings, std=0.02)
        self.relation_norm = nn.LayerNorm(dim_model)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "routing_slots",
            torch.as_tensor(ROUTING_SLOTS, dtype=torch.long),
            persistent=False,
        )
        if self.diffusion_reliability:
            schedule = canonical_diffusion_schedule()["sqrt_alpha_bar"]
            if tensor_sha256(schedule) != SQRT_ALPHA_BAR_SHA256:
                raise ValueError("D2-AF sqrt(alpha_bar) schedule hash mismatch")
            self.register_buffer(
                "sqrt_alpha_bar",
                schedule,
                persistent=False,
            )
        else:
            self.sqrt_alpha_bar = None
        self._diagnostic_variant = "full"
        self._gate_override: Optional[float] = None
        self._rho_override: Optional[float] = None
        self._capture = False
        self._snapshot: Optional[Dict[str, torch.Tensor]] = None

    def set_diagnostic_variant(self, variant: str) -> None:
        allowed = (
            D2AF_DIAGNOSTIC_VARIANTS
            if self.diffusion_reliability
            else DIAGNOSTIC_VARIANTS
        )
        if variant not in allowed:
            raise ValueError(f"unknown sparse-relation diagnostic variant: {variant}")
        self._diagnostic_variant = variant

    def set_gate_override(self, value: Optional[float]) -> None:
        if value is not None and not (-1.0 <= float(value) <= 1.0):
            raise ValueError("sparse-relation gate override must be in [-1,1]")
        self._gate_override = None if value is None else float(value)

    def set_rho_override(self, value: Optional[float]) -> None:
        if not self.diffusion_reliability:
            if value is not None:
                raise ValueError("rho override is restricted to D2-AF")
            self._rho_override = None
            return
        if value is not None and float(value) != 1.0:
            raise ValueError("D2-AF rho override is fixed to unit-rho only")
        self._rho_override = None if value is None else 1.0

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if not self._capture:
            self._snapshot = None

    def snapshot(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._snapshot is None:
            return None
        return {key: value.clone() for key, value in self._snapshot.items()}

    def _relation_vectors(self, pooled: torch.Tensor) -> torch.Tensor:
        flattened = pooled.reshape(pooled.shape[0], 4, 768)
        return self.relation_norm(
            self.projection(flattened) + self.temporal_embeddings[None].to(flattened)
        )

    def forward(
        self,
        motion: torch.Tensor,
        current: torch.Tensor,
        rest_object_points: torch.Tensor,
        world_to_local_rotation: torch.Tensor,
        object_rotation_reference: torch.Tensor,
        position_minimum: torch.Tensor,
        position_maximum: torch.Tensor,
        object_minimum: torch.Tensor,
        object_maximum: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if motion.shape != (current.shape[0], REPRESENTATION.window_frames, 512):
            raise ValueError(
                "expected sparse-relation motion embedding [B,16,512], "
                f"got {tuple(motion.shape)}"
            )
        rho: Optional[torch.Tensor] = None
        if self.diffusion_reliability:
            if not torch.is_tensor(timesteps):
                raise ValueError("D2-AF requires current timesteps [B]")
            if timesteps.shape != (current.shape[0],):
                raise ValueError(
                    f"D2-AF expected timesteps [{current.shape[0]}], got {timesteps.shape}"
                )
            if timesteps.dtype != torch.long:
                raise ValueError("D2-AF timesteps must be torch.long")
            if timesteps.device != current.device:
                raise ValueError("D2-AF timesteps must share the current-state device")
            if bool(
                ((timesteps < 0) | (timesteps >= DIFFUSION_STEPS)).any()
            ):
                raise ValueError("D2-AF timestep is outside the canonical schedule")
            if self._rho_override is not None or self._diagnostic_variant == "unit_rho":
                rho = motion.new_ones((current.shape[0], 1, 1))
            else:
                assert self.sqrt_alpha_bar is not None
                rho = self.sqrt_alpha_bar.gather(0, timesteps).to(
                    device=motion.device,
                    dtype=motion.dtype,
                ).reshape(current.shape[0], 1, 1)
        elif timesteps is not None:
            if timesteps.shape != (current.shape[0],):
                raise ValueError("D2-AE optional timesteps must match the batch")
        geometry = build_sparse_relation_geometry(
            current,
            rest_object_points,
            world_to_local_rotation,
            object_rotation_reference,
            position_minimum,
            position_maximum,
            object_minimum,
            object_maximum,
        )
        encoded = self.point_encoder(geometry["features"])
        pooled = torch.cat((encoded.mean(dim=-2), encoded.amax(dim=-2)), dim=-1)
        full_relation = self._relation_vectors(pooled)
        temporal_relation = None
        role_relation = None
        if (
            self._diagnostic_variant == "temporal_correspondence_permuted"
            or self._capture
        ):
            temporal_relation = self._relation_vectors(
                pooled[:, TEMPORAL_SOURCE_PERMUTATION]
            )
        if (
            self._diagnostic_variant == "left_right_role_swapped"
            or self._capture
        ):
            role_relation = self._relation_vectors(
                pooled[:, :, ROLE_SWAP_PERMUTATION]
            )
        if self._diagnostic_variant == "temporal_correspondence_permuted":
            assert temporal_relation is not None
            relation = temporal_relation
        elif self._diagnostic_variant == "left_right_role_swapped":
            assert role_relation is not None
            relation = role_relation
        else:
            relation = full_relation
        routed = relation.index_select(1, self.routing_slots)
        if (
            self._diagnostic_variant == "relation_gate_ablated"
            or self._gate_override == 0.0
        ):
            gate = motion.new_zeros(())
        elif self._gate_override is None:
            gate = torch.tanh(self.alpha).to(motion)
        else:
            gate = motion.new_tensor(self._gate_override)
        raw_writeback = gate * routed
        attenuated_writeback = (
            raw_writeback if rho is None else rho * raw_writeback
        )
        if self._capture:
            assert temporal_relation is not None and role_relation is not None
            with torch.no_grad():
                self._snapshot = {
                    "pooled_block_norm": pooled.float().norm(dim=-1).mean(dim=0),
                    "pooled_block_variance": pooled.float().var(
                        dim=(0, 3), unbiased=False,
                    ),
                    "relation_norm": full_relation.float().norm(dim=-1).mean(dim=0),
                    "temporal_permutation_sensitivity": (
                        full_relation.float() - temporal_relation.float()
                    ).norm(dim=-1).mean(dim=0),
                    "role_swap_sensitivity": (
                        full_relation.float() - role_relation.float()
                    ).norm(dim=-1).mean(dim=0),
                    "gate": gate.detach().float().reshape(1),
                }
                if rho is not None:
                    self._snapshot.update({
                        "rho": rho.detach().float().reshape(-1),
                        "raw_writeback_norm": raw_writeback.detach().float().norm(
                            dim=-1,
                        ).mean(dim=0),
                        "attenuated_writeback_norm": (
                            attenuated_writeback.detach().float().norm(dim=-1).mean(dim=0)
                        ),
                    })
        return motion + attenuated_writeback

    def contract_metadata(self) -> Dict[str, object]:
        if self.diffusion_reliability:
            return diffusion_reliability_contract_metadata()
        return sparse_relation_contract_metadata()
