"""Per-joint channel groups, so a gate can split the body between the experts.

``ChannelBlockGate`` can only cut the representation at 84 -- positions against
rotations.  That is the wrong axis for the question the two experts actually pose.
Scene collision is decided by where the LEGS and the ROOT go; object manipulation
is decided by where the ARMS and HANDS go; and both of those live in the position
block and in the rotation block at once.  So the split has to be per joint inside
both blocks, which is an implementation gap rather than a configuration one.

THE LAYOUT, taken from the repository's own indices rather than assumed.

Rotations, channels 84:216, are 22 joints x 6D in SMPL body order.  Confirmed by
``eval_metrics.py:107-119`` (ankles 7/8, feet 10/11) and
``priors/hoi/losses.py:335-336`` (wrists 20/21, feet 7/8/10/11):

    0 pelvis | 1 L_Hip 2 R_Hip | 3 Spine1 | 4 L_Knee 5 R_Knee | 6 Spine2
    7 L_Ankle 8 R_Ankle | 9 Spine3 | 10 L_Foot 11 R_Foot | 12 Neck
    13 L_Collar 14 R_Collar | 15 Head | 16 L_Shoulder 17 R_Shoulder
    18 L_Elbow 19 R_Elbow | 20 L_Wrist 21 R_Wrist

There are NO hand rotations: the representation stops at 22 joints, so "HOI drives
the hands" is a claim about the four hand POSITION channels and about the arm chain
that carries them, not about finger articulation, which this representation cannot
express at all.

Positions, channels 0:84, are 28 joints x XYZ.  ``utils.py:300`` gives
``SMPLX_JOINTS_28 = [0..21, 23, 24, 25, 28, 40, 43]``, so slots 22-27 are SMPL-X
joints 23 (left eye), 24 (right eye), 25 (left_index1), 28 (left_middle1),
40 (right_index1) and 43 (right_middle1).  ``test_infbagel_hosi.py:379-380`` reads
hands at slots 24/26 and ``priors/hoi/losses.py:335`` at slots 25/27; both are
hands, so slots 24-27 are the hand group and 22-23 go with the head.

WHAT REACHES THE METRICS, which is why the root is its own group.  The evaluator
does not score the 84 position channels.  It takes ``points_all[:, 0]`` as the
root translation (``test_infbagel_hosi.py:885``), converts the 22 global rotations
to locals through ``quat_ik_torch``, and runs SMPL-X; every geometric metric is
computed on the vertices and joints that come back.  So of the position block only
channels 0:3 -- the pelvis -- reach the body that is measured.  The other 81 act
on the rollout instead: they are the autoregressive history the next window is
conditioned on, and they are what ``_compute_occ_sample`` reads to place the
temporal occupancy queries.  Both matter, but they matter through the chain rather
than through the score, and a gate design that conflates the two would be reasoning
about the wrong tensor.

WHERE THE SEAM SITS.  ``quat_ik_torch`` converts global rotations to local ones by
differencing against the parent, so a local rotation is well defined however the
globals were mixed and bone lengths cannot be violated by construction -- the rest
template supplies them.  What a split DOES create is a one-joint seam: if HSI owns
Spine3's global frame and HOI owns L_Collar's, the local collar rotation absorbs
the whole disagreement between the two experts' body headings as a shoulder twist.
That seam is unavoidable in any split; the group boundaries decide only where it
lands, which is why ``torso`` is a group of its own rather than being folded
silently into one side.
"""

from typing import Dict, Mapping, Optional, Sequence

import torch

from priors.core.representation import REPRESENTATION

#: 22 rotation joints, by group.  Every index appears exactly once.
ROTATION_GROUPS: Dict[str, Sequence[int]] = {
    'root': (0,),
    'lower_body': (1, 2, 4, 5, 7, 8, 10, 11),
    'torso': (3, 6, 9, 12, 15),
    'arms': (13, 14, 16, 17, 18, 19, 20, 21),
    'hands': (),
}

#: 28 position joints, by group.  Same group names, so one weight drives both
#: blocks and a caller cannot accidentally gate a joint's position one way and its
#: rotation the other.
POSITION_GROUPS: Dict[str, Sequence[int]] = {
    'root': (0,),
    'lower_body': (1, 2, 4, 5, 7, 8, 10, 11),
    'torso': (3, 6, 9, 12, 15, 22, 23),
    'arms': (13, 14, 16, 17, 18, 19, 20, 21),
    'hands': (24, 25, 26, 27),
}

GROUP_NAMES = tuple(ROTATION_GROUPS)

#: The default assignment: HSI drives the root and the legs, HOI drives the arms
#: and the hands.  ``torso`` follows HOI, so the seam lands at the collars rather
#: than inside the spine -- the arm chain keeps one consistent parent frame, which
#: is the side of the body the object constraint acts on.  It is a knob, not a
#: finding: nothing has measured which placement generates better motion.
DEFAULT_BODY_GROUP_GATE: Dict[str, float] = {
    'root': 1.0,
    'lower_body': 1.0,
    'torso': 0.0,
    'arms': 0.0,
    'hands': 0.0,
}

_POSITION_JOINTS = 28
_ROTATION_JOINTS = 22
_ROTATION_WIDTH = 6
_ROTATION_START = REPRESENTATION.field('joint_rotations_6d').start


def _validate_partition():
    """A partition, checked at import: no joint missing and none in two groups."""
    for label, groups, total in (
        ('rotation', ROTATION_GROUPS, _ROTATION_JOINTS),
        ('position', POSITION_GROUPS, _POSITION_JOINTS),
    ):
        seen = [index for group in groups.values() for index in group]
        if len(seen) != len(set(seen)):
            raise ValueError(f'{label} groups overlap')
        if sorted(seen) != list(range(total)):
            missing = sorted(set(range(total)) - set(seen))
            raise ValueError(
                f'{label} groups are not a partition of {total} joints; '
                f'missing {missing}'
            )
    if tuple(POSITION_GROUPS) != GROUP_NAMES:
        raise ValueError('position and rotation groups must use the same names')


_validate_partition()


def body_group_channel_gate(
    weights: Optional[Mapping[str, float]] = None,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """A [232] gate carrying one weight per body group.

    Object and contact channels 216:232 are left at zero.  That is belt and
    braces, not the mechanism: ``compose_x0`` multiplies by ``human_gate_mask()``
    unconditionally and refuses any mask that opens those channels, so contact
    comes from HOI whatever this function returns.
    """
    resolved = dict(DEFAULT_BODY_GROUP_GATE)
    if weights is not None:
        unknown = sorted(set(weights) - set(GROUP_NAMES))
        if unknown:
            raise ValueError(
                f'unknown body group(s) {unknown}; known groups are '
                f'{list(GROUP_NAMES)}'
            )
        resolved.update({name: float(value) for name, value in weights.items()})
    for name, value in resolved.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'gate value for {name!r} must lie in [0, 1], got {value}')

    gate = torch.zeros(REPRESENTATION.dimension, device=device, dtype=dtype)
    for name, value in resolved.items():
        for joint in POSITION_GROUPS[name]:
            gate[joint * 3:(joint + 1) * 3] = value
        for joint in ROTATION_GROUPS[name]:
            start = _ROTATION_START + joint * _ROTATION_WIDTH
            gate[start:start + _ROTATION_WIDTH] = value
    return gate


def describe_body_groups() -> Dict[str, Dict[str, object]]:
    """Group membership and channel counts, for a run payload."""
    described = {}
    for name in GROUP_NAMES:
        described[name] = {
            'position_joints': list(POSITION_GROUPS[name]),
            'rotation_joints': list(ROTATION_GROUPS[name]),
            'channels': 3 * len(POSITION_GROUPS[name])
            + _ROTATION_WIDTH * len(ROTATION_GROUPS[name]),
        }
    return described
