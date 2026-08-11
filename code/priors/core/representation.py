"""Single source of truth for the external 232-dimensional motion representation."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class Field:
    name: str
    start: int
    stop: int
    semantics: str

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


class RepresentationSchema:
    dimension = 232
    window_frames = 16
    history_frames = 2
    diffusion_steps = 500
    coordinate_system = "Y-up, window-local XZ origin, initial-root-yaw aligned"
    fields: Tuple[Field, ...] = (
        Field("joint_positions", 0, 84, "28 joints x XYZ, OMOMO min/max normalized"),
        Field("joint_rotations_6d", 84, 216, "22 global joint rotations x continuous 6D"),
        Field("object_translation", 216, 219, "dynamic-object XYZ, OMOMO object min/max normalized"),
        Field("object_rotation", 219, 228, "3x3 object rotation relative to the BPS reference frame"),
        Field("contact", 228, 232, "four human-object contact labels"),
    )

    def __init__(self) -> None:
        cursor = 0
        for field in self.fields:
            if field.start != cursor:
                raise ValueError(f"non-contiguous representation before {field.name}")
            cursor = field.stop
        if cursor != self.dimension:
            raise ValueError(f"representation ends at {cursor}, expected {self.dimension}")

    def field(self, name: str) -> Field:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    def loss_mask(self, expert: str, *, device=None) -> torch.Tensor:
        mask = torch.ones(self.dimension, dtype=torch.bool, device=device)
        if expert == "hsi":
            mask[self.field("object_translation").start:] = False
        elif expert != "hoi":
            raise ValueError(f"unknown expert: {expert}")
        return mask

    def empty_channels(self, expert: str) -> Tuple[str, ...]:
        return ("object_translation", "object_rotation", "contact") if expert == "hsi" else ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "dimension": self.dimension,
            "window_frames": self.window_frames,
            "history_frames": self.history_frames,
            "diffusion_steps": self.diffusion_steps,
            "coordinate_system": self.coordinate_system,
            "fields": [field.__dict__ for field in self.fields],
            "hsi_empty_fields": list(self.empty_channels("hsi")),
            "hsi_active_indices": [0, self.field("object_translation").start],
        }


REPRESENTATION = RepresentationSchema()


def transform_object_points_for_next_window(
    rest_object_points: torch.Tensor,
    object_rotation: torch.Tensor,
    object_translation: torch.Tensor,
) -> torch.Tensor:
    """Apply the autoregressive object transform in the BPS tensor dtype.

    OMOMO object transforms can arrive from NumPy as float64 while the locked
    BPS representation is float32.  Keep the representation tensor authoritative
    so both ``bmm`` operands and the translated result use one explicit dtype.
    """
    if rest_object_points.ndim != 3 or rest_object_points.shape[-1] != 3:
        raise ValueError(f"expected rest object points [B,N,3], got {rest_object_points.shape}")
    batch_size = rest_object_points.shape[0]
    if object_rotation.shape != (batch_size, 3, 3):
        raise ValueError(f"expected object rotation [B,3,3], got {object_rotation.shape}")
    if object_translation.shape not in ((batch_size, 3), (batch_size, 1, 3)):
        raise ValueError(
            f"expected object translation [B,3] or [B,1,3], got {object_translation.shape}"
        )
    device = rest_object_points.device
    dtype = rest_object_points.dtype
    rotation = object_rotation.to(device=device, dtype=dtype)
    translation = object_translation.reshape(batch_size, 1, 3).to(device=device, dtype=dtype)
    transformed = rotation.bmm(rest_object_points.transpose(1, 2))
    return (transformed + translation.transpose(1, 2)).transpose(1, 2)


def masked_reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, expert: str, history_frames: int = 2,
) -> torch.Tensor:
    """MSE over predicted frames and domain-supervised channels only."""
    if prediction.shape != target.shape or prediction.shape[-1] != REPRESENTATION.dimension:
        raise ValueError(f"expected matching [...,232] tensors, got {prediction.shape}/{target.shape}")
    active = REPRESENTATION.loss_mask(expert, device=prediction.device)
    error = (prediction[:, history_frames:, active] - target[:, history_frames:, active]).square()
    if not error.numel():
        raise ValueError("loss mask selected no values")
    return error.mean()
