import numpy as np
import os

_SMPLH_KINTREE = os.path.join(os.path.dirname(__file__), '..', '..', 'smpl_models', 'smplh', 'kintree_table.npy')

def zup_to_yup(coord):
    # change the coordinate from z-up to y-up
    if len(coord.shape) > 1:
        coord = coord[..., [0, 2, 1]]
        coord[..., 2] *= -1
    else:
        coord = coord[[0, 2, 1]]
        coord[2] *= -1

    return coord

def yup_to_zup(coord):
    # change the coordinate from y-up to z-up
    if len(coord.shape) > 1:
        coord = coord[..., [0, 2, 1]]
        coord[..., 1] *= -1
    else:
        coord = coord[[0, 2, 1]]
        coord[1] *= -1

    return coord

import time
def get_occupancy_from_npy(data):
    # data = np.load(npy_path, allow_pickle=True).item()
    # Unpack bit data
    data = np.array(data)
    start_time = time.time()
    bs = data.shape[0]
    shape = [300, 100, 400]
    unpacked = np.unpackbits(data, axis=1)
    # Take only the required length (shape[0]*shape[1]*shape[2]) and reshape to 3D
    total_size = shape[0] * shape[1] * shape[2]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    return 1 - unpacked[:, :total_size].reshape(bs, shape[0], shape[1], shape[2])

def get_smpl_parents(use_joints24=True):
    ori_kintree_table = np.load(_SMPLH_KINTREE)  # 2 X 52

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents

# ---------------------------------------------------------------------------
# Asset world frame
# ---------------------------------------------------------------------------
#
# The released arrays do not all live in the same world frame, and the mismatch
# is silent because every one of them is float32 [.., 3].  Measured on
# 2026-08-18 (see tests/hsi/test_representation_frame.py, which re-derives every
# number below from the arrays themselves rather than trusting this comment):
#
#   human_joints_aligned.npy       y-up world, every corpus.
#   transl_aligned.npy             y-up world in LINGO.  In OMOMO it is the
#                                  y-up pelvis minus ``zup_to_yup(J0)``, i.e.
#                                  ``zup_to_yup`` applied to the genuine z-up
#                                  SMPL-X translation.
#   rest_human_offsets_aligned.npy ``zup_to_yup`` applied to the y-up SMPL
#                                  template, every corpus.  Invert with
#                                  ``rest_offsets_to_yup``.
#   human_pose.npy                 parent-relative local rotations, therefore
#                                  independent of the world frame; never
#                                  transform it.
#   human_orient.npy               y-up world in LINGO (``data/dataset``),
#                                  z-up world in OMOMO (``data/train``,
#                                  ``data/test``).
#
# A y-up pipeline therefore needs exactly two operations: left-multiply the root
# orientation by ``world_up_correction`` (identity for LINGO, ``zup_to_yup`` for
# OMOMO) and invert the baked-in ``zup_to_yup`` on the rest template.  Note that
# conjugating the axis-angles -- ``zup_to_yup(aa)``, which is exactly
# ``M R M^T`` because ``zup_to_yup`` is the proper rotation ``Rx(-90 deg)`` --
# is *not* a world-frame change: it also rotates the template, so it only
# composes correctly with a template that carries the same rotation.  That is
# why the released code's ``zup_to_yup(human_orient)`` + stored rest offsets
# reproduces OMOMO joints exactly and misses LINGO joints by 0.565 m.

_ZUP_TO_YUP_MATRIX = np.array([[1., 0., 0.],
                               [0., 0., 1.],
                               [0., -1., 0.]])

ASSET_WORLD_UP_CHOICES = ('auto', 'y', 'z')


def world_up_correction(asset_world_up):
    """Rotation carrying the asset's stored world frame to the y-up world."""
    if asset_world_up == 'y':
        return np.eye(3)
    if asset_world_up == 'z':
        return _ZUP_TO_YUP_MATRIX.copy()
    raise ValueError(f"asset_world_up must be 'y' or 'z', got {asset_world_up!r}")


def rest_offsets_to_yup(rest_offsets):
    """Undo the ``zup_to_yup`` baked into ``rest_human_offsets_aligned.npy``."""
    return yup_to_zup(np.array(rest_offsets, dtype=np.float64))


def _axis_angle_to_matrix(axis_angle):
    """Rodrigues for [..., 3] -> [..., 3, 3]; numpy only, float64."""
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-12, 1.0, theta)
    axis = axis_angle / safe
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1)
    skew = skew.reshape(axis.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3), skew.shape)
    sin = np.sin(theta)[..., None]
    cos = np.cos(theta)[..., None]
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def asset_frame_joint_error(root_axis_angle, body_pose, pelvis, rest_offsets_yup,
                           parents_22, correction):
    """Max per-joint FK error against a y-up joint array, in metres.

    ``root_axis_angle`` [F, 3], ``body_pose`` [F, 63], ``pelvis`` [F, 22, 3]
    (the reference joints; only ``[:, 0]`` seeds the chain), ``rest_offsets_yup``
    [F, >=22, 3] already in the y-up template frame, ``correction`` [3, 3] the
    world-frame rotation applied to the root only.
    """
    correction = np.asarray(correction, dtype=np.float64)
    root = correction[None] @ _axis_angle_to_matrix(root_axis_angle)
    body = _axis_angle_to_matrix(np.asarray(body_pose, dtype=np.float64).reshape(-1, 21, 3))
    frames = root.shape[0]
    globals_ = np.empty((frames, 22, 3, 3), dtype=np.float64)
    positions = np.empty((frames, 22, 3), dtype=np.float64)
    globals_[:, 0] = root
    positions[:, 0] = np.asarray(pelvis, dtype=np.float64)[:, 0]
    for joint in range(1, 22):
        parent = int(parents_22[joint])
        globals_[:, joint] = globals_[:, parent] @ body[:, joint - 1]
        positions[:, joint] = positions[:, parent] + np.einsum(
            'fij,fj->fi', globals_[:, parent], rest_offsets_yup[:, joint])
    reference = np.asarray(pelvis, dtype=np.float64)[:, :22]
    return float(np.abs(positions - reference).max())


def resolve_asset_world_up(orient, pose, joints, rest_offsets, sequence_starts,
                          parents_22, requested='auto', probe_sequences=32,
                          accept_metres=1e-3, reject_metres=1e-2):
    """Decide (and always verify) which world frame ``human_orient`` is in.

    The test is functional, not a heuristic: forward kinematics from the
    candidate world correction and the y-up rest template must reproduce
    ``human_joints_aligned.npy``.  The two hypotheses are separated by roughly
    six orders of magnitude (1e-7 m against 0.55 m on both corpora), so the
    decision has no grey zone, and an explicitly requested frame is verified
    with the same assertion rather than trusted.

    Returns ``(asset_world_up, {'y': error, 'z': error})`` with errors in metres.
    """
    if requested not in ASSET_WORLD_UP_CHOICES:
        raise ValueError(f"asset_world_up must be one of {ASSET_WORLD_UP_CHOICES}, got {requested!r}")
    sequence_starts = np.asarray(sequence_starts, dtype=np.int64)
    count = min(int(probe_sequences), len(sequence_starts))
    picks = np.unique(np.linspace(0, len(sequence_starts) - 1, count).round().astype(np.int64))
    frames = sequence_starts[picks]
    root = np.asarray(orient[frames], dtype=np.float64)
    body = np.asarray(pose[frames], dtype=np.float64)
    reference = np.asarray(joints[frames], dtype=np.float64)
    offsets = rest_offsets_to_yup(np.asarray(rest_offsets[picks], dtype=np.float64))
    errors = {
        up: asset_frame_joint_error(root, body, reference, offsets, parents_22,
                                    world_up_correction(up))
        for up in ('y', 'z')
    }
    if requested == 'auto':
        resolved = min(errors, key=errors.get)
    else:
        resolved = requested
    other = 'z' if resolved == 'y' else 'y'
    if not errors[resolved] < accept_metres:
        raise ValueError(
            "asset world frame '%s' does not reproduce human_joints_aligned.npy: "
            "max joint error %.6f m over %d probe frames (y=%.6f, z=%.6f). "
            "Either the arrays changed or rest_human_offsets_aligned.npy is no "
            "longer zup_to_yup(y-up template)."
            % (resolved, errors[resolved], len(frames), errors['y'], errors['z']))
    if not errors[other] > reject_metres:
        raise ValueError(
            "asset world frame is ambiguous: y=%.6f m, z=%.6f m over %d probe "
            "frames. The frame test requires the rejected hypothesis to fail by "
            "more than %.3f m." % (errors['y'], errors['z'], len(frames), reject_metres))
    return resolved, errors


def apply_world_correction_to_axis_angle(axis_angle, correction, chunk=1 << 18):
    """Left-multiply a world rotation onto axis-angle root orientations.

    This is a change of world frame, so it left-multiplies only.  Conjugating
    instead (``M R M^T``) would additionally rotate the body template and is the
    defect this function exists to avoid.
    """
    from scipy.spatial.transform import Rotation

    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    out = np.empty_like(axis_angle)
    for begin in range(0, len(axis_angle), int(chunk)):
        end = min(begin + int(chunk), len(axis_angle))
        matrices = Rotation.from_rotvec(np.array(axis_angle[begin:end])).as_matrix()
        out[begin:end] = Rotation.from_matrix(correction[None] @ matrices).as_rotvec()
    return out


# ---------------------------------------------------------------------------
# Step-0 window frame rule (EVALUATION ONLY)
# ---------------------------------------------------------------------------

STEP0_FRAME_RULE_CHOICES = ('repaired', 'historical_conjugated')
HISTORICAL_STEP0_FRAME_RULE = 'historical_conjugated'


def historical_conjugated_root_matrix(root_axis_angle, asset_world_up):
    """The root orientation the released code handed the window heading rule.

    ``datasets/infbagel.py`` canonicalizes each window's heading from the
    outermost EXTRINSIC y Euler angle of the first frame's root orientation, and
    that arithmetic is the released code's, unchanged.  What the 2026-08-18
    correction changed is its *input*: the released code conjugated the root,
    ``M R_stored M^T`` with ``M = zup_to_yup``, so the angle it read was a
    rotation about a horizontal axis; the repaired code left-multiplies the world
    correction only, ``C R_stored`` with ``C = world_up_correction(...)``.

    Given the REPAIRED root this reconstructs the released one exactly.  With
    ``R_rep = C R_stored`` and ``C`` orthogonal, ``R_stored = C^T R_rep``, hence

        R_hist = M R_stored M^T = M C^T R_rep M^T

    which is ``R_rep M^T`` on OMOMO (``C == M``) and ``M R_rep M^T`` on LINGO
    (``C == I``).  Deriving it from the repaired array rather than re-reading
    ``human_orient.npy`` is what makes the two step-0 frame rules two functions
    of one identical repaired dataset, which is the whole point of the P12
    CONTROL cell: it moves the evaluator's step-0 frame and nothing else.

    This function exists to REPRODUCE A KNOWN DEFECT for one controlled
    evaluation cell.  It must never be selected for training or for a primary
    result; see the 2026-08-19 section of
    ``docs/plan/PHASE_1B_HOI/07_REPRESENTATION_FRAME.md``.

    ``root_axis_angle`` is ``[..., 3]``; the return is ``[..., 3, 3]``.
    """
    from scipy.spatial.transform import Rotation

    conjugation = _ZUP_TO_YUP_MATRIX
    correction = world_up_correction(asset_world_up)
    repaired = Rotation.from_rotvec(
        np.asarray(root_axis_angle, dtype=np.float64)).as_matrix()
    return conjugation @ correction.T @ repaired @ conjugation.T
