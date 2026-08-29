"""The window representation's coordinate frame, against independent references.

Every assertion here is anchored on something the code under test does not
produce:

* ``human_joints_aligned.npy`` -- the released joint array.  Forward kinematics
  from the composed rotation channel and the rest template must reproduce it.
* ``human_orient.npy`` -- the released root orientation.  The channel's
  uprightness must equal the raw asset's, frame for frame.
* the hip line ``joints[:, 2] - joints[:, 1]`` -- a readout of the root
  orientation that touches no Euler convention at all, because the hips are
  direct children of the pelvis.  It measures whether the window shift really
  canonicalizes heading.
* ``constants.rest_pelvis`` and ``test_infbagel_hosi.get_mat`` -- the sampler's
  own joint-space heading estimator, which the dataset never consults.

The two composition paths (``datasets/infbagel.py`` and
``priors/core/window_codec.py``) are then required to agree with each other, which
is the property that had no test before 2026-08-18 and is why a 8.06 deg mean /
163.4 deg max convention split survived in the frozen contract.
"""

import contextlib
import io
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import pytorch3d.transforms as transforms

from constants import rest_pelvis
from datasets.infbagel import InfBaGelDataset
from datasets.utils import (get_smpl_parents, rest_offsets_to_yup, resolve_asset_world_up,
                            world_up_correction, yup_to_zup, zup_to_yup)
from priors.core.window_codec import WindowStateCodec, rotation_geodesic
from priors.hsi.data import _global_rotations
from utils import rigid_transform_3D

LINGO = REPO / "data/dataset"
OMOMO = REPO / "data/train"
WINDOW_SOURCE_FRAMES = 48
DATA_STEP = 3
WINDOWS = 64
PROBE_SEQUENCES = 24

# Tolerances.  All of them are set well inside the measured margin so a real
# regression trips them and float32 noise does not: the accepted/rejected FK
# hypotheses are ~1e-7 m against ~1.2 m apart, and the two composition paths agree
# to 2.3e-05 deg against the 163 deg they differed by before.
FK_TOLERANCE_M = 1e-4
UPRIGHTNESS_TOLERANCE_DEG = 1e-3
SHIFT_AXIS_TOLERANCE_DEG = 1e-6
GEODESIC_TOLERANCE_DEG = 1e-2
CHANNEL_TOLERANCE = 1e-5


def _degrees_between(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((first * second).sum(-1), -1.0, 1.0)))


def _wrap_degrees(radians):
    return np.degrees((radians + np.pi) % (2.0 * np.pi) - np.pi)


class _Corpus:
    """Memory-mapped release arrays plus the deterministic HSI window sample."""

    def __init__(self, root):
        self.root = Path(root)
        self.orient = np.load(self.root / "human_orient.npy", mmap_mode="r")
        self.pose = np.load(self.root / "human_pose.npy", mmap_mode="r")
        self.joints = np.load(self.root / "human_joints_aligned.npy", mmap_mode="r")
        self.rest = np.load(self.root / "rest_human_offsets_aligned.npy", mmap_mode="r")
        self.transl = np.load(self.root / "transl_aligned.npy", mmap_mode="r")
        self.norm = np.load(self.root / "norm.npy")
        self.sequence_starts = np.load(self.root / "start_idx.npy").astype(np.int64)
        self.sequence_ends = np.load(self.root / "end_idx.npy").astype(np.int64)
        self.parents = get_smpl_parents(use_joints24=False).copy()

    def windows(self, count=WINDOWS):
        path = self.root / "language_motion_dict/language_motion_dict__inter_and_loco__16.pkl"
        with path.open("rb") as handle:
            language = pickle.load(handle)
        starts = np.asarray(language["start_idx"], dtype=np.int64)
        ends = np.asarray(language["end_idx"], dtype=np.int64)
        sequences = np.asarray(language["ori_sequence_idx"], dtype=np.int64)
        keep = (np.asarray(language["left_hand_inter_frame"]) == -1)
        keep &= (np.asarray(language["right_hand_inter_frame"]) == -1)
        keep &= (ends - starts) == WINDOW_SOURCE_FRAMES
        keep &= (self.sequence_ends[sequences] - self.sequence_starts[sequences]) > WINDOW_SOURCE_FRAMES
        eligible = np.nonzero(keep)[0]
        picks = np.unique(np.linspace(0, len(eligible) - 1, count).round().astype(np.int64))
        return eligible[picks], starts, ends, sequences


def _available(root):
    return (Path(root) / "human_orient.npy").is_file()


class AssetWorldFrameTests(unittest.TestCase):
    """Which world frame each release corpus stores, decided by forward kinematics."""

    @unittest.skipUnless(_available(LINGO), "data/dataset is required")
    def test_lingo_root_orientation_is_already_y_up(self):
        corpus = _Corpus(LINGO)
        resolved, errors = resolve_asset_world_up(
            corpus.orient, corpus.pose, corpus.joints, corpus.rest,
            corpus.sequence_starts, corpus.parents, probe_sequences=PROBE_SEQUENCES,
        )
        self.assertEqual(resolved, "y")
        self.assertLess(errors["y"], FK_TOLERANCE_M)
        # The rejected hypothesis is the released code's own choice, so it must be
        # shown to fail rather than merely be unused.
        self.assertGreater(errors["z"], 0.1)

    @unittest.skipUnless(_available(OMOMO), "data/train is required")
    def test_omomo_root_orientation_is_z_up(self):
        corpus = _Corpus(OMOMO)
        resolved, errors = resolve_asset_world_up(
            corpus.orient, corpus.pose, corpus.joints, corpus.rest,
            corpus.sequence_starts, corpus.parents, probe_sequences=PROBE_SEQUENCES,
        )
        self.assertEqual(resolved, "z")
        self.assertLess(errors["z"], FK_TOLERANCE_M)
        self.assertGreater(errors["y"], 0.1)

    @unittest.skipUnless(_available(LINGO), "data/dataset is required")
    def test_rest_template_inversion_matches_the_kinematic_constant(self):
        """``rest_offsets_to_yup`` must land on ``constants.rest_pelvis``.

        ``rest_pelvis`` is an independent copy of the y-up SMPL hip triple: it
        lives in constants.py, is written as a literal, and is consumed only by
        ``get_mat``.  If the inversion direction were flipped, the two would
        disagree by 0.11 m.
        """
        corpus = _Corpus(LINGO)
        offsets = rest_offsets_to_yup(np.asarray(corpus.rest[0]))
        expected = np.asarray(rest_pelvis, dtype=np.float64)
        for joint in (1, 2):
            self.assertLess(float(np.abs(offsets[joint] - expected[joint]).max()), 5e-3)
        stored = np.asarray(corpus.rest[0], dtype=np.float64)
        self.assertGreater(float(np.abs(stored[1] - expected[1]).max()), 0.05)

    @unittest.skipUnless(_available(LINGO), "data/dataset is required")
    def test_local_pose_carries_no_world_frame(self):
        """The world correction applies to the root only.

        Conjugating the locals as well (the released ``zup_to_yup(human_pose)``)
        must be shown to break FK against ``human_joints_aligned.npy``, otherwise
        "do not transform human_pose" is an untested preference.
        """
        corpus = _Corpus(LINGO)
        frames = corpus.sequence_starts[:PROBE_SEQUENCES]
        root = np.asarray(corpus.orient[frames], dtype=np.float64)
        pose = np.asarray(corpus.pose[frames], dtype=np.float64)
        offsets = rest_offsets_to_yup(np.asarray(corpus.rest[:PROBE_SEQUENCES]))
        reference = np.asarray(corpus.joints[frames], dtype=np.float64)
        from datasets.utils import asset_frame_joint_error

        kept = asset_frame_joint_error(root, pose, reference, offsets, corpus.parents, np.eye(3))
        conjugated = asset_frame_joint_error(
            root, zup_to_yup(pose.reshape(-1, 3)).reshape(pose.shape), reference,
            offsets, corpus.parents, np.eye(3),
        )
        self.assertLess(kept, FK_TOLERANCE_M)
        self.assertGreater(conjugated, 0.1)


@unittest.skipUnless(_available(LINGO), "data/dataset is required")
class ChannelFrameTests(unittest.TestCase):
    """Gate A/B/C/E on real LINGO windows through the real dataset."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = _Corpus(LINGO)
        cls.indices, cls.starts, cls.ends, cls.sequences = cls.corpus.windows()
        cls.dataset = InfBaGelDataset(
            str(LINGO), torch.device("cpu"), None, 1, DATA_STEP, None, train=True,
            load_scene=False, load_language=True, load_object_goal=False,
            load_object_payload=False, vis=False, max_window_size=16,
        )
        # __getitem__ reads scene_name only to label the sample; load_scene=False
        # skips the Scene* tables, so supply the list the label comes from.
        with (LINGO / "scene_name.pkl").open("rb") as handle:
            cls.dataset.scene_name = pickle.load(handle)
        cls.items = [cls.dataset[int(index)] for index in cls.indices]

    def _channel_root(self, item):
        matrices = transforms.rotation_6d_to_matrix(item["global_rot_6d"].reshape(-1, 22, 6))
        return matrices[0, 0].numpy().astype(np.float64)

    def _shift(self, item):
        return np.linalg.inv(np.asarray(item["mat"][:3, :3], dtype=np.float64))

    def test_channel_uprightness_equals_the_raw_asset(self):
        """Gate A. The shift is a world-y rotation, so R[1, 1] is invariant under
        it and the channel's value must equal the raw ``human_orient`` value
        exactly -- not merely be small."""
        worst = 0.0
        for item, index in zip(self.items, self.indices):
            start = int(self.starts[index])
            raw = Rotation.from_rotvec(
                np.array(self.corpus.orient[start], dtype=np.float64)
            ).as_matrix()
            channel = self._channel_root(item)
            worst = max(worst, abs(float(channel[1, 1] - raw[1, 1])))
        self.assertLess(np.degrees(worst), UPRIGHTNESS_TOLERANCE_DEG)

    def test_channel_up_axis_agrees_with_the_joint_channel(self):
        """Gate A. The rotation channel's body-up direction against the joint
        channel's pelvis->neck direction, both read in the same post-shift frame.
        Before the correction this was 94.8 deg mean; the residual is the real
        anatomical angle between the two, which the raw asset also shows."""
        angles = []
        for item in self.items:
            joints = self.dataset.denormalize(item["joints"].reshape(-1, 28, 3)).astype(np.float64)
            up = self._channel_root(item) @ np.array([0.0, 1.0, 0.0])
            angles.append(_degrees_between(up, joints[0, 12] - joints[0, 0]))
        angles = np.asarray(angles)
        self.assertLess(float(np.median(angles)), 15.0)
        self.assertLess(float(angles.mean()), 20.0)

    def test_window_shift_is_a_rotation_about_world_up(self):
        """Gate B. Axis check plus the algebraic invariant: a world-y rotation
        cannot change how upright the body is."""
        for item, index in zip(self.items, self.indices):
            shift = self._shift(item)
            rotvec = Rotation.from_matrix(shift).as_rotvec()
            norm = float(np.linalg.norm(rotvec))
            if norm > 1e-9:
                axis = min(_degrees_between(rotvec, [0.0, 1.0, 0.0]),
                           _degrees_between(rotvec, [0.0, -1.0, 0.0]))
                self.assertLess(float(axis), SHIFT_AXIS_TOLERANCE_DEG)
            start, end = int(self.starts[index]), int(self.ends[index])
            raw = np.asarray(self.corpus.joints[start:end:DATA_STEP], dtype=np.float64)
            shifted = self.dataset.denormalize(
                item["joints"].reshape(-1, 28, 3)).astype(np.float64)
            before = _degrees_between(raw[0, 12] - raw[0, 0], [0.0, 1.0, 0.0])
            after = _degrees_between(shifted[0, 12] - shifted[0, 0], [0.0, 1.0, 0.0])
            self.assertLess(abs(float(before - after)), 1e-3)

    def test_window_shift_canonicalizes_the_hip_line_azimuth(self):
        """Gate B, against a reference that uses no Euler convention.

        The hips are direct children of the pelvis, so ``joints[2] - joints[1]``
        is ``R_root`` applied to a fixed template vector: its xz azimuth is a pure
        readout of the root's heading.  Before the correction the post-shift
        azimuths stayed uniform on the circle (p50 85 deg from their own circular
        mean); after it they collapse onto one direction.
        """
        before, after = [], []
        for item, index in zip(self.items, self.indices):
            start, end = int(self.starts[index]), int(self.ends[index])
            raw = np.asarray(self.corpus.joints[start:end:DATA_STEP], dtype=np.float64)
            shifted = self.dataset.denormalize(
                item["joints"].reshape(-1, 28, 3)).astype(np.float64)
            hip_raw = raw[0, 2] - raw[0, 1]
            hip_shifted = shifted[0, 2] - shifted[0, 1]
            before.append(np.arctan2(hip_raw[0], hip_raw[2]))
            after.append(np.arctan2(hip_shifted[0], hip_shifted[2]))
        before = np.asarray(before)
        after = np.asarray(after)
        # the input really is spread over the circle
        self.assertGreater(float(np.median(np.abs(np.degrees(before)))), 45.0)
        mean_direction = np.arctan2(np.sin(after).mean(), np.cos(after).mean())
        residual = np.abs(_wrap_degrees(after - mean_direction))
        self.assertLess(float(np.median(residual)), 5.0)
        self.assertLess(float(np.percentile(residual, 90)), 15.0)

    def test_dataset_heading_matches_the_samplers_joint_space_estimate(self):
        """``get_mat``'s heading, recomputed here, must equal the dataset's.

        ``get_mat`` fits the joint channel's pelvis/hip triple to the y-up
        ``constants.rest_pelvis`` and takes the extrinsic-'zxy' y angle; the
        dataset takes the same angle from the rotation channel's root.  Two
        independent readouts of one quantity: they disagreed by 85 deg p50 before
        the correction, which meant training windows and rollout windows were
        canonicalized in different frames.
        """
        worst = 0.0
        for item, index in zip(self.items, self.indices):
            start = int(self.starts[index])
            joints = np.asarray(self.corpus.joints[start], dtype=np.float64)
            with contextlib.redirect_stdout(io.StringIO()):
                _scale, fitted, _t = rigid_transform_3D(np.matrix(joints[:3]), rest_pelvis, False)
            joint_yaw = Rotation.from_matrix(np.asarray(fitted)).as_euler("zxy")[2]
            channel_yaw = Rotation.from_matrix(
                self._shift(item)).as_euler("zxy")[2]
            worst = max(worst, abs(float(_wrap_degrees(joint_yaw + channel_yaw))))
        self.assertLess(worst, 1.0)

    def test_legacy_and_core_paths_compose_the_same_channel(self):
        """Gate E. ``datasets/infbagel.py`` and ``priors/core/window_codec.py``
        must produce the same 232-dim window state for the same window.

        This is the assertion that did not exist: scipy's lowercase ``as_euler``
        is extrinsic and pytorch3d's convention string is intrinsic, so
        ``'zxy'[2]`` and ``"ZXY"[2]`` are different angles, and the two paths
        silently disagreed by 8.06 deg mean / 163.4 deg max.
        """
        codec = WindowStateCodec(
            torch.from_numpy(self.corpus.norm[0].astype(np.float32)),
            torch.from_numpy(self.corpus.norm[1].astype(np.float32)),
        )
        correction = world_up_correction("y")
        worst_rotation = worst_joint = 0.0
        for item, index in zip(self.items, self.indices):
            start, end = int(self.starts[index]), int(self.ends[index])
            rotations = _global_rotations(
                np.asarray(self.corpus.orient[start:end]),
                np.asarray(self.corpus.pose[start:end]),
                self.corpus.parents, correction,
            )[::DATA_STEP]
            joints = torch.from_numpy(
                np.array(self.corpus.joints[start:end:DATA_STEP], dtype=np.float32))
            encoded, _frame = codec.encode(joints, rotations)
            legacy = transforms.rotation_6d_to_matrix(item["global_rot_6d"].reshape(-1, 22, 6))
            core = transforms.rotation_6d_to_matrix(
                encoded[..., 84:216].reshape(-1, 22, 6))
            worst_rotation = max(
                worst_rotation, float(np.degrees(rotation_geodesic(legacy, core).max())))
            worst_joint = max(worst_joint, float(
                (encoded[..., :84] - item["joints"].reshape(-1, 84)).abs().max()))
        self.assertLess(worst_rotation, GEODESIC_TOLERANCE_DEG)
        self.assertLess(worst_joint, CHANNEL_TOLERANCE)

    def test_channel_forward_kinematics_reproduces_the_released_joint_array(self):
        """Gate C's kinematic half, with no SMPL-X dependency.

        Decode the channel back to the world frame the way the evaluators do
        (``mat[:3, :3] @ channel``), run the dataset's own IK/FK against the rest
        template it now serves, and compare against ``human_joints_aligned.npy``.
        """
        worst = 0.0
        for item, index in zip(self.items, self.indices):
            start, end = int(self.starts[index]), int(self.ends[index])
            mat = torch.from_numpy(np.asarray(item["mat"], dtype=np.float32))
            globals_ = mat[None, None, :3, :3] @ transforms.rotation_6d_to_matrix(
                item["global_rot_6d"].reshape(-1, 22, 6))
            locals_ = self.dataset.quat_ik_torch(globals_)
            offsets = torch.from_numpy(
                np.array(item["rest_human_offsets"], dtype=np.float32))
            positions = offsets[None].repeat(locals_.shape[0], 1, 1).clone()
            reference = np.asarray(self.corpus.joints[start:end:DATA_STEP], dtype=np.float32)
            positions[:, 0, :] = torch.from_numpy(reference[:, 0])
            _rotations, fk = self.dataset.quat_fk_torch(locals_, positions)
            worst = max(worst, float(
                np.abs(fk[:, :22].numpy() - reference[:, :22]).max()))
        self.assertLess(worst, FK_TOLERANCE_M)


if __name__ == "__main__":
    unittest.main()
