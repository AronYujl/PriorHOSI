"""Kinematically coherent composition of the two experts' clean predictions.

The raw body-group gate selects *global* rotations joint by joint.  Converting that
hybrid to locals afterwards makes every boundary joint absorb the experts' whole
parent-frame disagreement.  P2-ROOT put exactly that artefact at both hips.

This operator instead converts each expert to its own local rotations first, takes
the complete hip-to-foot branches from HSI and every other rotation from HOI, and
then runs FK once below HOI's root.  Object/contact remain bitwise HOI.  There are
no learned values and no checkpoint-specific thresholds in this module.
"""

from typing import Dict

import torch
from pytorch3d import transforms

from priors.core.representation import REPRESENTATION

from .body_groups import ROTATION_GROUPS
from .composition import ExpertOutputs, OBJECT_CHANNEL_START


# The 28-position representation extends the 22-rotation skeleton with six
# markers.  The 24-joint FK bundle already contains SMPL-X left_middle1 and
# right_middle1 at its last two slots; in the 28-joint representation those are
# slots 25 and 27.  The other four markers are transported rigidly in the local
# frame of their HOI parent rather than independently blended.
_FK_EXTRA_TO_POSITION = {22: 25, 23: 27}
_ATTACHED_MARKER_PARENTS = {22: 15, 23: 15, 24: 20, 26: 21}


def _expand_rest_offsets(value: torch.Tensor, batch: int, frames: int,
                         reference: torch.Tensor) -> torch.Tensor:
    """Return writable ``[B,T,24,3]`` offsets without guessing silently."""
    offsets = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if offsets.shape == (24, 3):
        offsets = offsets.reshape(1, 1, 24, 3)
    elif offsets.ndim == 3 and offsets.shape[-2:] == (24, 3):
        if offsets.shape[0] not in (1, batch):
            raise ValueError(
                f'rest offsets batch {offsets.shape[0]} does not match {batch}'
            )
        offsets = offsets[:, None]
    elif offsets.ndim != 4 or offsets.shape[-2:] != (24, 3):
        raise ValueError(
            'rest_human_offsets must be [24,3], [B,24,3] or [B,T,24,3], '
            f'got {tuple(offsets.shape)}'
        )
    if offsets.shape[0] not in (1, batch) or offsets.shape[1] not in (1, frames):
        raise ValueError(
            f'rest offsets {tuple(offsets.shape)} do not broadcast to '
            f'[{batch},{frames},24,3]'
        )
    return offsets.expand(batch, frames, 24, 3).clone()


def _apply_rotation(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.matmul(rotation, vector.unsqueeze(-1)).squeeze(-1)


class KinematicBodyComposer:
    """Attach HSI-local legs to an HOI carrier and rebuild the body by FK.

    The dataset supplies the same IK/FK implementation and rest offsets used by
    the evaluator.  Its normalization is also used directly: this module never
    duplicates min/max values and therefore cannot become checkpoint-specific.
    """

    lower_body_joints = tuple(ROTATION_GROUPS['lower_body'])

    def __init__(self):
        self.compose_calls = 0

    def describe(self) -> Dict[str, object]:
        return {
            'kind': 'kinematic_local_rotation_fk',
            'root_owner': 'hoi',
            'lower_body_owner': 'hsi',
            'other_rotation_owner': 'hoi',
            'extra_marker_owner': 'hoi_parent_attached',
            'object_contact_owner': 'hoi',
            'scene_query_pelvis': 'shared_current_and_previous_composed_x0',
            'learned_parameters': 0,
            'compose_calls': self.compose_calls,
        }

    def __call__(self, outputs: ExpertOutputs, *, dataset,
                 rest_human_offsets: torch.Tensor,
                 fixed_points: torch.Tensor) -> torch.Tensor:
        if not isinstance(outputs, ExpertOutputs):
            raise TypeError(f'outputs must be ExpertOutputs, got {type(outputs)!r}')
        if outputs.hoi is None or outputs.hsi is None:
            raise ValueError(
                'kinematic composition needs both HOI and HSI predictions; '
                f'present: {outputs.present()}'
            )
        hoi, hsi = outputs.hoi, outputs.hsi
        if hoi.shape != hsi.shape or hoi.ndim != 3:
            raise ValueError(
                f'expected matching [B,T,{REPRESENTATION.dimension}] predictions, '
                f'got {tuple(hoi.shape)} and {tuple(hsi.shape)}'
            )
        if hoi.shape[-1] != REPRESENTATION.dimension:
            raise ValueError(
                f'expected {REPRESENTATION.dimension} channels, got {hoi.shape[-1]}'
            )
        batch, frames, _ = hoi.shape
        if fixed_points.shape != (
            batch, REPRESENTATION.history_frames, REPRESENTATION.dimension,
        ):
            raise ValueError(
                'fixed_points must be '
                f'[{batch},{REPRESENTATION.history_frames},{REPRESENTATION.dimension}], '
                f'got {tuple(fixed_points.shape)}'
            )
        for name in ('normalize_torch', 'denormalize_torch',
                     'quat_ik_torch', 'quat_fk_torch'):
            if not hasattr(dataset, name):
                raise TypeError(f'dataset lacks required kinematic method {name}')

        hoi_positions = dataset.denormalize_torch(
            hoi[..., :84].reshape(batch, frames, 28, 3)
        )
        hoi_global = transforms.rotation_6d_to_matrix(
            hoi[..., 84:216].reshape(batch, frames, 22, 6)
        )
        hsi_global = transforms.rotation_6d_to_matrix(
            hsi[..., 84:216].reshape(batch, frames, 22, 6)
        )

        hoi_local = dataset.quat_ik_torch(hoi_global.reshape(-1, 22, 3, 3))
        hsi_local = dataset.quat_ik_torch(hsi_global.reshape(-1, 22, 3, 3))
        local = hoi_local.reshape(batch, frames, 22, 3, 3).clone()
        hsi_local = hsi_local.reshape(batch, frames, 22, 3, 3)
        local[..., self.lower_body_joints, :, :] = hsi_local[
            ..., self.lower_body_joints, :, :
        ]

        offsets = _expand_rest_offsets(
            rest_human_offsets, batch, frames, hoi_positions,
        )
        # quat_fk_torch defines lpos[:,0] in global/window coordinates and every
        # other entry as a parent-relative rest offset.
        offsets[..., 0, :] = hoi_positions[..., 0, :]
        composed_global_quat_flat, fk_positions_flat = dataset.quat_fk_torch(
            local.reshape(-1, 22, 3, 3), offsets.reshape(-1, 24, 3),
        )
        expected_rotations = (batch * frames, 22, 4)
        expected_positions = (batch * frames, 24, 3)
        if tuple(composed_global_quat_flat.shape) != expected_rotations:
            raise ValueError(
                'quat_fk_torch must return global quaternions shaped '
                f'{expected_rotations}, got {tuple(composed_global_quat_flat.shape)}'
            )
        if tuple(fk_positions_flat.shape) != expected_positions:
            raise ValueError(
                'quat_fk_torch must return positions shaped '
                f'{expected_positions}, got {tuple(fk_positions_flat.shape)}'
            )
        composed_global = transforms.quaternion_to_matrix(
            composed_global_quat_flat
        ).reshape(batch, frames, 22, 3, 3)
        fk_positions = fk_positions_flat.reshape(batch, frames, 24, 3)

        positions = torch.empty_like(hoi_positions)
        positions[..., :22, :] = fk_positions[..., :22, :]
        for fk_index, position_index in _FK_EXTRA_TO_POSITION.items():
            positions[..., position_index, :] = fk_positions[..., fk_index, :]

        # Transport the remaining HOI markers in their HOI-parent local frame.
        # This preserves the marker-to-parent vector exactly while allowing the
        # selected HSI leg branch to alter only the articulated skeleton it owns.
        for marker, parent in _ATTACHED_MARKER_PARENTS.items():
            delta = hoi_positions[..., marker, :] - hoi_positions[..., parent, :]
            local_delta = _apply_rotation(
                hoi_global[..., parent, :, :].transpose(-1, -2), delta,
            )
            positions[..., marker, :] = positions[..., parent, :] + _apply_rotation(
                composed_global[..., parent, :, :], local_delta,
            )

        result = hoi.clone()
        result[..., :84] = dataset.normalize_torch(positions).reshape(
            batch, frames, 84,
        )
        result[..., 84:216] = transforms.matrix_to_rotation_6d(
            composed_global
        ).reshape(batch, frames, 132)
        # The query pelvis and all object/contact channels are exact ownership
        # contracts, not merely round-trip-close values.
        result[..., :3] = hoi[..., :3]
        result[..., OBJECT_CHANNEL_START:] = hoi[..., OBJECT_CHANNEL_START:]
        result[:, :REPRESENTATION.history_frames] = fixed_points
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError('kinematic composition produced non-finite state')
        self.compose_calls += 1
        return result
