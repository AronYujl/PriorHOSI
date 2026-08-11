"""Shared state codec for composable fixed-window motion experts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch3d import transforms
from pytorch3d.ops import knn_points

from .representation import REPRESENTATION


BPS_SHA256 = "fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_to_so3(matrix: torch.Tensor) -> torch.Tensor:
    """Project arbitrary 3x3 outputs to a proper rotation with stable gradients."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected [...,3,3], got {tuple(matrix.shape)}")
    first = matrix[..., :, 0]
    first_fallback = torch.zeros_like(first)
    first_fallback[..., 0] = 1.0
    first = torch.where(
        (torch.linalg.vector_norm(first, dim=-1, keepdim=True) < 1e-6), first_fallback, first,
    )
    first = F.normalize(first, dim=-1, eps=1e-6)
    second = matrix[..., :, 1]
    second = second - (second * first).sum(dim=-1, keepdim=True) * first
    fallback_index = first.abs().argmin(dim=-1)
    fallback = F.one_hot(fallback_index, num_classes=3).to(matrix)
    fallback = fallback - (fallback * first).sum(dim=-1, keepdim=True) * first
    second = torch.where(
        (torch.linalg.vector_norm(second, dim=-1, keepdim=True) < 1e-6), fallback, second,
    )
    second = F.normalize(second, dim=-1, eps=1e-6)
    third = torch.linalg.cross(first, second, dim=-1)
    second = torch.linalg.cross(third, first, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def rotation_geodesic(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = project_to_so3(first) @ project_to_so3(second).transpose(-1, -2)
    skew = torch.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
    ), dim=-1)
    sine = torch.linalg.vector_norm(skew, dim=-1) / 2.0
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    return torch.atan2(sine, cosine)


def zup_to_yup_tensor(value: torch.Tensor) -> torch.Tensor:
    converted = value[..., (0, 2, 1)].clone()
    converted[..., 2] *= -1.0
    return converted


def yup_to_zup_tensor(value: torch.Tensor) -> torch.Tensor:
    """Invert :func:`zup_to_yup_tensor` for author BPS delta replay."""
    return torch.stack((value[..., 0], -value[..., 2], value[..., 1]), dim=-1)


@dataclass(frozen=True)
class WindowFrame:
    """A window-local XZ origin/yaw frame plus its object rotation reference."""

    origin: torch.Tensor
    world_to_local: torch.Tensor
    object_reference: torch.Tensor

    def to(self, *, device=None, dtype=None) -> "WindowFrame":
        return WindowFrame(
            self.origin.to(device=device, dtype=dtype),
            self.world_to_local.to(device=device, dtype=dtype),
            self.object_reference.to(device=device, dtype=dtype),
        )


class WindowStateCodec:
    """Single encode/decode/rebase contract used by data, training and rollout."""

    def __init__(
        self,
        position_minimum: torch.Tensor,
        position_maximum: torch.Tensor,
        object_minimum: Optional[torch.Tensor] = None,
        object_maximum: Optional[torch.Tensor] = None,
        *,
        bps_path: Optional[Path] = None,
        verify_bps: bool = True,
    ) -> None:
        self.position_minimum = torch.as_tensor(position_minimum, dtype=torch.float32)
        self.position_maximum = torch.as_tensor(position_maximum, dtype=torch.float32)
        self.object_minimum = (
            torch.as_tensor(object_minimum, dtype=torch.float32) if object_minimum is not None else None
        )
        self.object_maximum = (
            torch.as_tensor(object_maximum, dtype=torch.float32) if object_maximum is not None else None
        )
        self.bps_path = Path(bps_path).resolve() if bps_path is not None else None
        self._basis: Optional[torch.Tensor] = None
        if self.bps_path is not None:
            if verify_bps and sha256_file(self.bps_path) != BPS_SHA256:
                raise ValueError(f"BPS basis hash mismatch: {self.bps_path}")
            payload = torch.load(self.bps_path, map_location="cpu")
            if not isinstance(payload, dict) or "obj" not in payload:
                raise ValueError(f"invalid BPS basis payload: {self.bps_path}")
            basis = torch.as_tensor(payload["obj"], dtype=torch.float32).reshape(-1, 3)
            if basis.shape != (1024, 3):
                raise ValueError(f"expected BPS basis [1024,3], got {tuple(basis.shape)}")
            self._basis = basis.contiguous()

    @staticmethod
    def frame_from_global(
        global_joints: torch.Tensor,
        global_human_rotation: torch.Tensor,
        global_object_rotation: Optional[torch.Tensor] = None,
    ) -> WindowFrame:
        if global_joints.shape[-2:] != (28, 3):
            raise ValueError(f"expected [...,T,28,3], got {tuple(global_joints.shape)}")
        if global_human_rotation.shape[-3:] != (22, 3, 3):
            raise ValueError(f"expected [...,T,22,3,3], got {tuple(global_human_rotation.shape)}")
        pelvis = global_joints[..., 0, 0, :]
        origin = pelvis.clone()
        origin[..., 1] = 0.0
        root = project_to_so3(global_human_rotation[..., 0, 0, :, :])
        yaw = transforms.matrix_to_euler_angles(root, "ZXY")[..., 2]
        zeros = torch.zeros_like(yaw)
        shift_euler = torch.stack((zeros, zeros, -yaw), dim=-1)
        world_to_local = transforms.euler_angles_to_matrix(shift_euler, "ZXY")
        if global_object_rotation is None:
            reference = torch.eye(3, device=global_joints.device, dtype=global_joints.dtype)
            reference = reference.expand(*origin.shape[:-1], 3, 3).clone()
        else:
            reference = project_to_so3(global_object_rotation[..., 0, :, :])
        return WindowFrame(origin, world_to_local, reference)

    @staticmethod
    def local_position(global_position: torch.Tensor, frame: WindowFrame) -> torch.Tensor:
        extra = global_position.ndim - frame.origin.ndim
        if extra < 1:
            raise ValueError("position must include at least one state/time dimension")
        origin = frame.origin
        rotation = frame.world_to_local.transpose(-1, -2)
        for _ in range(extra):
            origin = origin.unsqueeze(-2)
        for _ in range(extra - 1):
            rotation = rotation.unsqueeze(-3)
        return (global_position - origin) @ rotation

    @staticmethod
    def global_position(local_position: torch.Tensor, frame: WindowFrame) -> torch.Tensor:
        extra = local_position.ndim - frame.origin.ndim
        if extra < 1:
            raise ValueError("position must include at least one state/time dimension")
        origin = frame.origin
        rotation = frame.world_to_local
        for _ in range(extra):
            origin = origin.unsqueeze(-2)
        for _ in range(extra - 1):
            rotation = rotation.unsqueeze(-3)
        return local_position @ rotation + origin

    @staticmethod
    def _normalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        return -1.0 + 2.0 * (value - minimum) / (maximum - minimum)

    @staticmethod
    def _denormalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        return (value + 1.0) * (maximum - minimum) / 2.0 + minimum

    def _norms(self, value: torch.Tensor, *, object_position: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        minimum = self.object_minimum if object_position else self.position_minimum
        maximum = self.object_maximum if object_position else self.position_maximum
        if minimum is None or maximum is None:
            raise ValueError("object normalization is unavailable for this codec")
        return minimum.to(value), maximum.to(value)

    def encode(
        self,
        global_joints: torch.Tensor,
        global_human_rotation: torch.Tensor,
        *,
        frame: Optional[WindowFrame] = None,
        global_object_translation: Optional[torch.Tensor] = None,
        global_object_rotation: Optional[torch.Tensor] = None,
        contact: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, WindowFrame]:
        if frame is None:
            frame = self.frame_from_global(global_joints, global_human_rotation, global_object_rotation)
        frame = frame.to(device=global_joints.device, dtype=global_joints.dtype)
        encoded = torch.zeros(
            *global_joints.shape[:-2], REPRESENTATION.dimension,
            device=global_joints.device, dtype=global_joints.dtype,
        )
        local_joints = self.local_position(global_joints, frame)
        minimum, maximum = self._norms(local_joints)
        encoded[..., :84] = self._normalize(local_joints, minimum, maximum).reshape(*encoded.shape[:-1], 84)
        local_rotation = frame.world_to_local[..., None, None, :, :] @ project_to_so3(global_human_rotation)
        encoded[..., 84:216] = transforms.matrix_to_rotation_6d(local_rotation).reshape(
            *encoded.shape[:-1], 132,
        )
        if global_object_translation is not None or global_object_rotation is not None:
            if global_object_translation is None or global_object_rotation is None:
                raise ValueError("object translation and rotation must be provided together")
            local_object = self.local_position(global_object_translation, frame)
            object_minimum, object_maximum = self._norms(local_object, object_position=True)
            encoded[..., 216:219] = self._normalize(local_object, object_minimum, object_maximum)
            relative = project_to_so3(global_object_rotation) @ frame.object_reference.transpose(-1, -2)[..., None, :, :]
            encoded[..., 219:228] = relative.reshape(*encoded.shape[:-1], 9)
            if contact is not None:
                encoded[..., 228:232] = contact
        return encoded, frame

    def decode(self, encoded: torch.Tensor, frame: WindowFrame) -> Dict[str, torch.Tensor]:
        if encoded.shape[-1] != REPRESENTATION.dimension:
            raise ValueError(f"expected [...,232], got {tuple(encoded.shape)}")
        frame = frame.to(device=encoded.device, dtype=encoded.dtype)
        minimum, maximum = self._norms(encoded)
        local_joints = self._denormalize(
            encoded[..., :84].reshape(*encoded.shape[:-1], 28, 3), minimum, maximum,
        )
        joints = self.global_position(local_joints, frame)
        local_human_rotation = transforms.rotation_6d_to_matrix(
            encoded[..., 84:216].reshape(*encoded.shape[:-1], 22, 6)
        )
        human_rotation = frame.world_to_local.transpose(-1, -2)[..., None, None, :, :] @ local_human_rotation
        result: Dict[str, torch.Tensor] = {
            "joints": joints,
            "human_rotation": project_to_so3(human_rotation),
            "contact": encoded[..., 228:232],
        }
        if self.object_minimum is not None:
            object_minimum, object_maximum = self._norms(encoded, object_position=True)
            local_object = self._denormalize(encoded[..., 216:219], object_minimum, object_maximum)
            relative = project_to_so3(encoded[..., 219:228].reshape(*encoded.shape[:-1], 3, 3))
            result["object_translation"] = self.global_position(local_object, frame)
            result["object_rotation"] = project_to_so3(
                relative @ frame.object_reference[..., None, :, :]
            )
        return result

    def pelvis_goal(self, global_endpoint: torch.Tensor, frame: WindowFrame) -> torch.Tensor:
        value = self.local_position(global_endpoint[..., None, :], frame)[..., 0, :]
        value = value.clone()
        value[..., 1] = 0.0
        return value

    def object_goal(self, global_goal: torch.Tensor, frame: WindowFrame) -> torch.Tensor:
        local = self.local_position(global_goal[..., None, :], frame)[..., 0, :]
        minimum, maximum = self._norms(local, object_position=True)
        return self._normalize(local, minimum, maximum)

    def rebase(self, encoded: torch.Tensor, source: WindowFrame) -> Tuple[torch.Tensor, WindowFrame]:
        decoded = self.decode(encoded, source)
        frame = self.frame_from_global(
            decoded["joints"], decoded["human_rotation"], decoded.get("object_rotation"),
        )
        return self.encode(
            decoded["joints"], decoded["human_rotation"], frame=frame,
            global_object_translation=decoded.get("object_translation"),
            global_object_rotation=decoded.get("object_rotation"), contact=decoded["contact"],
        )

    @property
    def bps_basis(self) -> torch.Tensor:
        if self._basis is None:
            raise ValueError("this codec was created without a BPS basis")
        return self._basis

    def recompute_bps(
        self, rest_object_vertices_yup: torch.Tensor, global_object_rotation: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce stored OMOMO object-centered BPS from current generated pose."""
        if rest_object_vertices_yup.shape[-1] != 3 or global_object_rotation.shape[-2:] != (3, 3):
            raise ValueError("invalid rest-object vertices or rotation")
        rotation = project_to_so3(global_object_rotation)
        if rotation.ndim == 2:
            rotation = rotation[None]
        vertices = rest_object_vertices_yup.to(rotation)
        if vertices.ndim == 2:
            vertices = vertices[None].expand(rotation.shape[0], -1, -1)
        transformed = vertices @ rotation.transpose(-1, -2)
        basis = self.bps_basis.to(rotation)[None].expand(rotation.shape[0], -1, -1)
        nearest = knn_points(basis, transformed, K=1, return_nn=True).knn[..., 0, :]
        delta_raw = nearest - basis
        encoded = zup_to_yup_tensor(delta_raw)
        return encoded[0] if global_object_rotation.ndim == 2 else encoded
