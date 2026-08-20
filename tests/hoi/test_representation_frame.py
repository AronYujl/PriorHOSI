"""The HOI window representation's coordinate frame, against independent references.

The 2026-08-18 correction put the rotation channel, the rest template, the
SMPL-X translation and the window heading in one y-up world.  Six source files
moved together, and the reason they had to move together is that several of the
pairs are *silently* compensating: undo one half and nothing raises, nothing
diverges, and one preregistered metric collapses toward zero.  This file exists
to make each half individually detectable.

Every assertion is anchored on something the code under test does not produce:

* ``human_joints_aligned.npy`` -- the released joint array.  Forward kinematics
  from the composed rotation channel and the rest template must reproduce it.
* ``human_orient.npy`` -- the released root orientation, read raw.  The
  channel's up-axis readout must equal the raw asset's own readout.
* ``constants.rest_pelvis`` -- an independent literal copy of the y-up SMPL hip
  triple, consumed nowhere on the HOI path, so it cannot have drifted with the
  arrays.
* the real SMPL-X body, run through the evaluator's own source text.

Two paths are then required to agree with each other: ``priors/hoi/data.py``
(HOIPrior training) and ``datasets/infbagel.py`` (HOI evaluation).  That
agreement is the property that had no test before 2026-08-18, which is why an
8.06 deg mean / 163.4 deg max convention split survived inside the frozen
``priors/core/`` contract.

Where a test executes source *text* extracted from a module rather than calling
a function, that is deliberate: the deleted SMPL-X sandwich and the transl
layout branch are not reachable through any public API, and pinning the text is
the only way a reintroduction fails a test instead of silently changing a
number.

``INFBAGEL_WORKER_EXPERT=hoi`` skips only the one test that loads real LINGO
files (``data/dataset``); everything else runs on the HOI worker, as
``AGENTS.md`` requires of representation tests.
"""

import contextlib
import inspect
import os
import pickle
import sys
import textwrap
import types
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
# code/config/config_eval_hoi_prior.yaml interpolates ${oc.env:ROOT_DIR}.
os.environ.setdefault("ROOT_DIR", str(REPO))

import pytorch3d.transforms as transforms  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from constants import rest_pelvis  # noqa: E402
from datasets.utils import (STEP0_FRAME_RULE_CHOICES,  # noqa: E402
                           get_smpl_parents, historical_conjugated_root_matrix,
                           rest_offsets_to_yup, resolve_asset_world_up,
                           world_up_correction, yup_to_zup, zup_to_yup)
from priors.core.window_codec import rotation_geodesic  # noqa: E402
from utils import create_smplx_model, run_smplx_model  # noqa: E402

OMOMO_TRAIN = REPO / "data/train"
OMOMO_TEST = REPO / "data/test"
LINGO = REPO / "data/dataset"
PROBE_SEQUENCES = 32
WINDOWS = 24

# ---------------------------------------------------------------------------
# Tolerances.  Every one is quoted with the margin it was set inside, measured
# 2026-08-18 on real OMOMO through the real evaluator.
# ---------------------------------------------------------------------------

# Absolute SMPL-X pelvis placement.  Correct code lands at 0.085 mm max /
# 0.1347 mm in the graft-apply gate; the four wrong ways to pair the source fix
# with the sandwich land at 0.3803 m (transl zeroed), 0.5378 m (transl
# restoration deleted), 0.7605 m (restoration direction reversed) and 1.4059 m
# (per-frame form broadcast onto the per-sequence array).
#
# The threshold is deliberately 1 mm and NOT 0.01 mm.  transl_aligned.npy ships
# float32 with about 85 um of quantisation, so rotating it by 90 deg
# necessarily costs on the order of 120 um; a 0.01 mm gate would fail on
# correct code.  1 mm keeps a 12x margin over correct code and is still 380x
# inside the cheapest failure.
PELVIS_ABSOLUTE_TOLERANCE_M = 1e-3
# The counterfactual sandwich displaces vertices by 0.54-0.92 m mean over the
# sampled windows (1.87 m on generated motion in the graft measurement).
SANDWICH_DISPLACEMENT_M = 0.2

# Step-0 window shift angle, degrees, against a by-hand reference built from the
# raw ``human_orient.npy``.  Measured 2026-08-19 over all 1314 evaluator
# windows: 2.36e-06 deg max for both rules, which is the float32 quantisation of
# the returned ``mat`` and nothing else.  1e-4 keeps a 42x margin over correct
# code while sitting 4e+03x inside the rejected codec-convention-flip route
# (0.41 deg mean at the 438 step-0 windows) and 7.8e+05x inside the CONTROL
# perturbation itself (77.83 deg mean).
STEP0_SHIFT_YAW_TOLERANCE_DEG = 1e-4


@contextlib.contextmanager
def _in_code_dir():
    """``constants.SMPL_DIR`` is ``'../smpl_models'`` and
    ``datasets/infbagel.py`` loads ``'./bps.pt'``, so both the dataset and the
    SMPL-X model resolve only from ``<repo>/code``.  Restored on exit so the
    rest of the suite is unaffected."""
    previous = os.getcwd()
    os.chdir(REPO / "code")
    try:
        yield
    finally:
        os.chdir(previous)


def _extract_block(path, first_line_startswith, last_line_startswith, *, inclusive):
    """Lift a contiguous source block out of a module and compile it.

    Both anchors are asserted unique so a refactor that duplicates them fails
    loudly here instead of silently pinning the wrong lines.
    """
    lines = Path(path).read_text().split("\n")
    begins = [i for i, line in enumerate(lines) if line.strip().startswith(first_line_startswith)]
    ends = [i for i, line in enumerate(lines) if last_line_startswith in line]
    if len(begins) != 1 or len(ends) != 1:
        raise AssertionError(
            f"{path}: expected one anchor each, got {len(begins)} start / {len(ends)} end"
        )
    stop = ends[0] + 1 if inclusive else ends[0]
    block = textwrap.dedent("\n".join(lines[begins[0]:stop]).rstrip())
    return block, compile(block, f"<{Path(path).name}:{begins[0] + 1}>", "exec")


@lru_cache(maxsize=1)
def _evaluator_smplx_block():
    """The HOI evaluator's own SMPL-X reconstruction, as source text.

    ``code/test_infbagel_hoi.py`` between ``root_trans =`` and the
    ``points_all_48`` accumulation is exactly the region the compensatory
    yup_to_zup/zup_to_yup sandwich occupied.  Executing the text means a
    reintroduced sandwich is *run* by this test rather than merely described by
    it.
    """
    return _extract_block(
        REPO / "code/test_infbagel_hoi.py",
        "root_trans =", "points_all_48 = torch.cat", inclusive=False,
    )


@lru_cache(maxsize=1)
def _legacy_dataset():
    """One ``InfBaGelDataset`` on ``data/test``, built exactly as
    ``test_infbagel_hoi.py:579`` builds it (the real hydra eval config).

    Cached because construction costs about 4.5 s and four test classes need
    it; ``np.random`` is seeded because ``__getitem__`` draws ``pi`` at random
    for windows whose ``need_pi`` is false (it reaches no channel this file
    asserts on, but a seeded run is cheaper to reason about than an argument).
    """
    from hydra import compose, initialize_config_dir
    from datasets.infbagel import InfBaGelDataset

    np.random.seed(42)
    torch.manual_seed(42)
    with initialize_config_dir(config_dir=str(REPO / "code/config"), version_base=None):
        cfg = compose(config_name="config_eval_hoi_prior",
                      overrides=["device=cpu", "exp_name=tests_representation_frame"])
    with _in_code_dir():
        return InfBaGelDataset(**cfg.dataset)


def _evaluator_windows(count=WINDOWS):
    """Deterministic spread of ``data/test`` language windows."""
    dataset = _legacy_dataset()
    total = len(dataset.start_ind)
    return [int(i) for i in np.unique(np.linspace(0, total - 1, count).round().astype(np.int64))]


@lru_cache(maxsize=1)
def _control_dataset():
    """``data/test`` under the P12 CONTROL step-0 window frame rule.

    Composed through the real hydra eval config with the exact override the
    CONTROL evaluation cell will pass, so this exercises the configuration path
    (``InfBaGelDataset(**cfg.dataset)`` at ``test_infbagel_hoi.py:579``) and not
    merely the constructor keyword.  ``step0_frame_rule`` is absent from
    ``config/dataset/omomo_test.yaml`` for the same reason ``asset_world_up``
    is: leaving it out keeps the PRIMARY cell's resolved config byte-identical
    to every earlier evaluation, so the switch cannot show up in a run that did
    not ask for it.
    """
    from hydra import compose, initialize_config_dir
    from datasets.infbagel import InfBaGelDataset

    np.random.seed(42)
    torch.manual_seed(42)
    with initialize_config_dir(config_dir=str(REPO / "code/config"), version_base=None):
        cfg = compose(config_name="config_eval_hoi_prior",
                      overrides=["device=cpu", "exp_name=tests_representation_frame",
                                 "+dataset.step0_frame_rule=historical_conjugated"])
    with _in_code_dir():
        return InfBaGelDataset(**cfg.dataset)


@lru_cache(maxsize=1)
def _step0_window_indices():
    """The 438 dataset indices the evaluator seeds a rollout from.

    ``test_infbagel_hoi.py:502-503`` builds ``[0] + list(seq_id.pkl.values())``
    and iterates ``seg_id_dict[seg_id]`` over ``range(len - 1)``, so the step-0
    windows are the first 438 entries.  The other two windows of each sequence
    take their frame from ``get_mat`` on generated motion, so this rule reaches
    exactly these 438 -- which is why every distribution below is quoted on them
    and not on all 1314.
    """
    with open(OMOMO_TEST / "seq_id.pkl", "rb") as handle:
        ids = [0] + list(pickle.load(handle).values())
    return tuple(int(index) for index in ids[:len(ids) - 1])


@lru_cache(maxsize=1)
def _raw_root_orientations():
    """``human_orient.npy`` as shipped, the reference nothing under test writes."""
    return np.load(OMOMO_TEST / "human_orient.npy", mmap_mode="r")


def _shift_yaw_from_mat(mat):
    """The window shift's y angle in degrees, read back out of ``mat``.

    ``__getitem__`` sets ``mat[:3, :3] = inv(S.T).T``, which for the orthogonal
    ``S = shift_rot_matrix`` is ``S.T``; ``S`` is ``Ry(a)``, so
    ``a = atan2(S[0, 2], S[0, 0])``.  Reading the SIGNED angle rather than a
    geodesic is load bearing: the shift reaches 179.94 deg on this corpus, where
    ``arccos`` of a float32 trace is ill-conditioned and inflates the same
    2.4e-06 deg disagreement to 1.6e-02 deg.
    """
    shift = np.asarray(mat, dtype=np.float64).reshape(4, 4)[:3, :3].T
    return float(np.degrees(np.arctan2(shift[0, 2], shift[0, 0])))


def _wrapped_degrees(value):
    """Signed angle difference folded into (-180, 180]."""
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _reference_shift_yaw(start, rule):
    """The shift angle for one window start, built by hand from the raw asset.

    Independent of the code under test at every step: it re-reads
    ``human_orient.npy``, applies the released ``zup_to_yup`` conjugation (or the
    repaired left multiplication) itself, and takes scipy's EXTRINSIC ``'zxy'``
    index 2 -- the outermost, i.e. world-y, factor.
    """
    raw = np.asarray(_raw_root_orientations()[int(start)], dtype=np.float64)
    if rule == "historical_conjugated":
        # exactly the released operation, zup_to_yup on the axis-angle
        root = Rotation.from_rotvec(zup_to_yup(raw.copy()[None])[0]).as_matrix()
    else:
        root = world_up_correction("z") @ Rotation.from_rotvec(raw).as_matrix()
    return -float(np.degrees(Rotation.from_matrix(root).as_euler("zxy")[2]))


class AbsolutePlacementTests(unittest.TestCase):
    """The paired source fix and sandwich deletion, in absolute world position.

    This is the only place in the change where undoing one half is silent.  The
    evaluator used to run SMPL-X in a z-up world -- ``yup_to_zup`` on the
    translation and the pose, ``zup_to_yup`` back on the vertices and joints --
    because OMOMO's rotation channel held rotations of a ``zup_to_yup``-rotated
    template and ``transl`` held ``-zup_to_yup(J0)`` instead of ``-J0``.  Now
    that ``datasets/infbagel.py`` fixes both at the source, keeping the
    sandwich moves the body about 0.54 m (vertices 0.54-0.92 m).  Nothing
    raises: the SDF query simply leaves the object's box and ``hand_pen`` /
    ``human_pen`` collapse toward zero, i.e. the model looks better.

    The assertion has to be on ABSOLUTE position.  Root-relative agreement is
    0.684 mm for the correct pairing *and* for the transl failures, so it has
    no discriminating power at all -- it is the reason an earlier root-relative
    gate scored the broken and the fixed trees as identical.
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = _legacy_dataset()
        cls.reference = np.load(OMOMO_TEST / "human_joints_aligned.npy", mmap_mode="r")
        cls.windows = _evaluator_windows(4)
        cls.cache = {}

    def _ground_truth_inputs(self, index):
        """The evaluator's GT branch (``test_infbagel_hoi.py:805-812``)."""
        item = self.dataset[index]
        frames = int(np.asarray(item["global_rot_6d_gt"]).shape[0])
        joints = torch.from_numpy(np.array(item["joints_gt"], dtype=np.float32))
        mat = torch.from_numpy(np.array(item["mat"], dtype=np.float32)).reshape(4, 4)
        rest = torch.from_numpy(np.array(item["rest_human_offsets"], dtype=np.float32))
        seeded = rest.unsqueeze(0).repeat(frames, 1, 1).reshape(-1, 24, 3).clone()
        seeded[:, 0, :] = joints.reshape(-1, 28, 3)[:, 0, :]
        globals_ = mat[None, :3, :3] @ transforms.rotation_6d_to_matrix(
            torch.from_numpy(np.array(item["global_rot_6d_gt"], dtype=np.float32)))
        locals_ = self.dataset.quat_ik_torch(globals_.reshape(-1, 22, 3, 3))
        return item, frames, joints, locals_, seeded

    def test_source_fix_and_sandwich_deletion_must_land_together(self):
        block_text, block = _evaluator_smplx_block()
        # A reintroduced sandwich would be executed below, but say so plainly
        # too: the numeric failure is the gate, this is the error message.
        self.assertNotIn("yup_to_zup(joints.reshape", block_text)
        self.assertNotIn("zup_to_yup(verts)", block_text)

        worst_correct = 0.0
        least_sandwich = float("inf")
        least_displacement = float("inf")
        with _in_code_dir():
            for index in self.windows:
                item, frames, joints, locals_, _seeded = self._ground_truth_inputs(index)
                transl = torch.from_numpy(np.array(item["transl"]))
                betas = torch.from_numpy(np.array(item["betas"]))
                namespace = dict(
                    joints=joints, transl=transl, betas=betas, gender=item["gender"],
                    local_jrot_mat_48=locals_, device="cpu", smplx_model_cache=self.cache,
                    create_smplx_model=create_smplx_model, run_smplx_model=run_smplx_model,
                    transforms=transforms, yup_to_zup=yup_to_zup, zup_to_yup=zup_to_yup,
                )
                exec(block, namespace)  # the evaluator's own text
                verts = namespace["verts"].numpy().astype(np.float64)
                pelvis = namespace["joints"].numpy().astype(np.float64)[:, 0]

                # The counterfactual: the deleted sandwich, re-applied to the
                # fixed source.  Computed here so the margin is asserted rather
                # than remembered from a scratch measurement.
                sandwiched_trans = yup_to_zup(joints.reshape(-1, 28, 3)[:, 0, :] + transl)
                sandwiched_pose = yup_to_zup(
                    transforms.matrix_to_axis_angle(locals_).reshape(-1, 22, 3))
                bad_verts, bad_joints = run_smplx_model(
                    sandwiched_pose, sandwiched_trans,
                    betas[None].repeat(sandwiched_trans.shape[0], 1), item["gender"],
                    joints_ind=None, smpl_model=self.cache[item["gender"]])
                bad_verts = zup_to_yup(bad_verts).numpy().astype(np.float64)
                bad_pelvis = zup_to_yup(bad_joints).numpy().astype(np.float64)[:, 0]

                start = int(self.dataset.start_ind[index])
                reference = np.asarray(self.reference[start:start + frames], dtype=np.float64)
                worst_correct = max(worst_correct, float(
                    np.abs(pelvis - reference[:, 0]).max()))
                least_sandwich = min(least_sandwich, float(
                    np.linalg.norm(bad_pelvis - reference[:, 0], axis=-1).mean()))
                least_displacement = min(least_displacement, float(
                    np.linalg.norm(bad_verts - verts, axis=-1).mean()))

        # Half one: the source fix.  Reverting any of the world correction, the
        # template inversion or the transl restoration moves this to 0.38-1.41 m.
        self.assertLess(worst_correct, PELVIS_ABSOLUTE_TOLERANCE_M)
        # Half two: the sandwich deletion.  The sandwich really is wrong now.
        self.assertGreater(least_sandwich, 0.1)
        self.assertGreater(least_displacement, SANDWICH_DISPLACEMENT_M)
class AssetWorldFrameTests(unittest.TestCase):
    """Which world frame each release corpus stores, decided by forward kinematics.

    ``resolve_asset_world_up`` is a functional probe, not a heuristic: FK from
    the candidate world correction and the y-up rest template must reproduce
    ``human_joints_aligned.npy``.  These tests assert the verdict *and* that the
    two hypotheses are separated by orders of magnitude, because a probe whose
    two candidates scored comparably would be a coin flip dressed as a
    measurement.
    """

    @staticmethod
    def _probe(root):
        orient = np.load(root / "human_orient.npy", mmap_mode="r")
        pose = np.load(root / "human_pose.npy", mmap_mode="r")
        joints = np.load(root / "human_joints_aligned.npy", mmap_mode="r")
        rest = np.load(root / "rest_human_offsets_aligned.npy", mmap_mode="r")
        starts = np.load(root / "start_idx.npy")
        parents = get_smpl_parents(use_joints24=False).copy()
        return resolve_asset_world_up(orient, pose, joints, rest, starts, parents,
                                      probe_sequences=PROBE_SEQUENCES)

    def _assert_no_grey_zone(self, root, expected):
        resolved, errors = self._probe(root)
        other = "z" if expected == "y" else "y"
        self.assertEqual(resolved, expected)
        self.assertLess(errors[expected], 1e-4)
        self.assertGreater(errors[other], 0.5)
        # Measured: data/train z=7.805e-07 vs y=1.286067; data/test z=5.421e-07
        # vs y=1.114070; data/dataset y=0.0 vs z=1.287.  Six orders of magnitude
        # is the property that makes the probe a decision rather than a guess.
        self.assertGreater(errors[other] / max(errors[expected], 1e-12), 1e4)
        return errors

    def test_omomo_train_world_frame_is_resolved_to_z_up(self):
        self._assert_no_grey_zone(OMOMO_TRAIN, "z")

    def test_omomo_test_world_frame_is_resolved_to_z_up(self):
        """``data/test`` is the corpus the HOI evaluator reads, so a wrong
        verdict here is a wrong number in every reported HOI metric."""
        self._assert_no_grey_zone(OMOMO_TEST, "z")

    @unittest.skipUnless((LINGO / "human_orient.npy").is_file(),
                         "data/dataset (LINGO) is absent; the HOI worker snapshot omits it")
    def test_lingo_world_frame_is_resolved_to_y_up(self):
        """The corpus that proves the correction is a *decision*, not a constant.

        LINGO stores an already-y-up ``human_orient.npy`` and OMOMO stores a
        z-up one.  A fix that hard-coded ``zup_to_yup`` for every corpus -- the
        released behaviour -- would score 1.287 m here.  This test also carries
        LINGO's rest-template inversion, since the probe consumes it.
        """
        self._assert_no_grey_zone(LINGO, "y")

    def test_world_correction_left_multiplies_and_is_not_a_conjugation(self):
        """A change of world frame left-multiplies the root only.

        ``zup_to_yup(axis_angle)`` is ``M R M^T``, which also rotates the body
        template, and that is exactly the released defect: it cancels against
        OMOMO's conjugated ``rest_human_offsets_aligned.npy`` and does not
        cancel against LINGO's y-up ``human_orient.npy``.  Asserting the two
        are *different* is the point -- otherwise "left-multiply" is an untested
        preference.
        """
        from datasets.utils import apply_world_correction_to_axis_angle

        correction = world_up_correction("z")
        axis_angle = np.array([[0.3, -1.1, 0.7], [2.0, 0.4, -0.2], [-0.9, 0.15, 1.4]])
        matrices = Rotation.from_rotvec(axis_angle).as_matrix()
        expected = Rotation.from_matrix(correction[None] @ matrices).as_rotvec()
        conjugated = Rotation.from_matrix(
            correction[None] @ matrices @ correction.T[None]).as_rotvec()

        got = apply_world_correction_to_axis_angle(axis_angle, correction)
        self.assertLess(float(np.abs(got - expected).max()), 1e-12)
        # measured 1.5486 rad apart, and the released op is provably the
        # conjugation to 2.2e-16
        self.assertGreater(float(np.abs(expected - conjugated).max()), 1.0)
        self.assertLess(float(np.abs(zup_to_yup(axis_angle.copy()) - conjugated).max()), 1e-14)
        self.assertTrue(np.array_equal(world_up_correction("y"), np.eye(3)))
        with self.assertRaises(ValueError):
            world_up_correction("x")

    def test_rest_template_inversion_lands_on_the_kinematic_constant(self):
        """``rest_offsets_to_yup`` must undo the baked-in ``zup_to_yup``.

        ``constants.rest_pelvis`` is an independent literal copy of the y-up
        SMPL hip triple; nothing on the HOI path consumes it, so it cannot have
        drifted with the arrays.  Measured hip error against it: inverted
        0.003-0.019 m (the spread is subject betas), stored as shipped
        0.114-0.125 m, inverted the wrong way 0.202-0.213 m.  Both OMOMO
        corpora are asserted; LINGO's copy rides along in the y-up probe test.
        """
        expected = np.asarray(rest_pelvis, dtype=np.float64)[1:3]
        for root in (OMOMO_TRAIN, OMOMO_TEST):
            with self.subTest(corpus=root.name):
                stored = np.asarray(
                    np.load(root / "rest_human_offsets_aligned.npy", mmap_mode="r")[0],
                    dtype=np.float64)
                inverted = rest_offsets_to_yup(stored)[1:3]
                good = float(np.abs(inverted - expected).max())
                as_shipped = float(np.abs(stored[1:3] - expected).max())
                wrong_way = float(np.abs(zup_to_yup(stored.copy())[1:3] - expected).max())
                self.assertLess(good, 0.03)
                self.assertGreater(as_shipped / good, 3.0)
                self.assertGreater(wrong_way / good, 3.0)
                # exact involution, so "inverse" is not merely approximate
                probe = np.array([[0.1, -0.2, 0.35], [1.0, 2.0, -3.0]])
                self.assertTrue(np.array_equal(
                    rest_offsets_to_yup(zup_to_yup(probe.copy())), probe))
@lru_cache(maxsize=1)
def _transl_restoration_block():
    """``datasets/infbagel.py``'s layout-aware transl restoration, as text.

    The branch is not reachable through any public API -- it runs once inside
    ``__init__`` -- and constructing the dataset on ``data/train`` to reach the
    per-frame arm costs 18.7 s.  Executing the extracted text against real
    per-corpus arrays reaches every arm, including the ``raise``, for about
    0.2 s.
    """
    return _extract_block(
        REPO / "code/datasets/infbagel.py",
        "transl = np.asarray(self.transl, dtype=np.float64)",
        "self.transl = transl.astype(transl_dtype)", inclusive=True,
    )


def _restore_transl(joints, transl, sequence_starts):
    _text, block = _transl_restoration_block()
    stub = types.SimpleNamespace(joints=joints, transl=transl,
                                 ori_sequence_start_idx=sequence_starts)
    exec(block, dict(self=stub, np=np, yup_to_zup=yup_to_zup))
    return stub.transl


class TranslRestorationTests(unittest.TestCase):
    """``transl_aligned.npy`` ships in two layouts and both must be restored.

    ``data/train`` holds it per frame (804460, 3), ``data/test`` per sequence
    (482, 3), ``data/dataset`` per frame (2915752, 3) but in the y-up branch
    that skips restoration entirely.  ``__getitem__`` consumes both
    conventions -- ``:602`` subtracts the pelvis, ``:604`` does not -- so a
    single formula cannot serve both, and broadcasting the per-frame form onto
    the per-sequence array is not a harmless fallback: it displaces the pelvis
    by 1.4059 m.  The HSI branch never loads ``data/test``, so this arm of the
    fix had no prior exposure at all.
    """

    def test_both_transl_layouts_restore_the_same_offset(self):
        """One operation, two layouts: the OFFSET is what gets rotated.

        ``yup_to_zup`` is linear, so per frame the restoration is
        ``pelvis - yup_to_zup(pelvis - transl)`` and per sequence, where the
        stored value already *is* the offset, it is ``yup_to_zup(transl)``.
        Both must therefore satisfy one invariant: the restored offset equals
        ``yup_to_zup`` of the stored offset.  Reversing either direction, or
        deleting either arm, breaks it here and costs 0.54-0.76 m of absolute
        pelvis error in ``AbsolutePlacementTests``.
        """
        # per-frame arm, real data/train values
        joints = np.asarray(np.load(OMOMO_TRAIN / "human_joints_aligned.npy",
                                    mmap_mode="r")[:4000])
        stored = np.asarray(np.load(OMOMO_TRAIN / "transl_aligned.npy", mmap_mode="r")[:4000])
        restored = _restore_transl(joints, stored, np.zeros((7, 3)))
        pelvis = np.asarray(joints[:, 0], dtype=np.float64)
        offset_before = np.asarray(stored, dtype=np.float64) - pelvis
        offset_after = np.asarray(restored, dtype=np.float64) - pelvis
        self.assertEqual(restored.dtype, stored.dtype)  # must not widen root_trans
        self.assertLess(float(np.abs(offset_after - yup_to_zup(offset_before.copy())).max()),
                        1e-6)
        # the per-frame array really does encode a per-sequence constant, which
        # is what licenses treating the two layouts as one quantity
        self.assertLess(float(np.abs(offset_before[:300] - offset_before[0]).max()), 1e-7)

        # per-sequence arm, real data/test values
        test_joints = np.asarray(np.load(OMOMO_TEST / "human_joints_aligned.npy",
                                         mmap_mode="r")[:5000])
        test_stored = np.load(OMOMO_TEST / "transl_aligned.npy")
        test_restored = _restore_transl(test_joints, test_stored, np.zeros((482, 3)))
        self.assertEqual(test_restored.dtype, test_stored.dtype)
        self.assertLess(float(np.abs(
            np.asarray(test_restored, dtype=np.float64)
            - yup_to_zup(np.asarray(test_stored, dtype=np.float64))).max()), 1e-12)

    def test_unrecognised_transl_layout_raises_instead_of_broadcasting(self):
        """A third layout must be a hard failure.

        Numpy would happily broadcast a length-1 or otherwise mismatched array
        and the run would continue with a 1.4 m displaced body.  The released
        code had no branch here at all, so this is the one place in the change
        where the safe behaviour is an exception.
        """
        test_joints = np.asarray(np.load(OMOMO_TEST / "human_joints_aligned.npy",
                                         mmap_mode="r")[:5000])
        stored = np.load(OMOMO_TEST / "transl_aligned.npy")
        for bad in (stored[:481], stored[:1], np.zeros((1, 3), dtype=stored.dtype)):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    _restore_transl(test_joints, bad, np.zeros((482, 3)))
class WindowChannelTests(unittest.TestCase):
    """Gate E, FK closure and the gate-A caveat, on real ``data/test`` windows.

    ``priors/hoi/data.py`` -> ``WindowStateCodec`` is what HOIPrior trains on;
    ``datasets/infbagel.py`` is what the HOI evaluator rolls out from.  Nothing
    imports one from the other, so before 2026-08-18 nothing compared them.
    """

    @classmethod
    def setUpClass(cls):
        from priors.hoi.data import PriorWindowDataset

        cls.legacy = _legacy_dataset()
        cls.windows = _evaluator_windows()
        cls.prior = PriorWindowDataset(str(REPO), "hoi", partition="test")
        # Select exactly the windows compared below.  PriorWindowDataset exposes
        # no index filter, and ``indices`` is the documented seam it uses itself.
        cls.prior.indices = np.asarray(cls.windows, dtype=np.int64)
        cls.reference = np.load(OMOMO_TEST / "human_joints_aligned.npy", mmap_mode="r")
        cls.parents = get_smpl_parents(use_joints24=False).copy()
        np.testing.assert_array_equal(np.asarray(cls.legacy.start_ind), cls.prior.starts)

    def test_training_and_evaluation_paths_compose_the_same_window_state(self):
        """Gate E: the two paths must build the same 232-dim window.

        scipy's lowercase ``as_euler('zxy')`` is EXTRINSIC, so index 2 is the
        outermost rotation -- the one about world up, i.e. the heading.
        pytorch3d's ``matrix_to_euler_angles(R, "ZXY")`` is INTRINSIC and its
        index 2 is the innermost y rotation, a body-frame rotation that is not a
        heading.  ``WindowStateCodec`` now takes ``"YXZ"[..., 0]``, provably the
        same angle as scipy's extrinsic ``'zxy'[2]``.

        Before the fix the two paths' rotation channels differed by 50.12 deg
        geodesic at the 438 evaluator window starts (56.86 deg over n=1653 whole
        windows every 37 frames) and the joint channel by 0.943 in normalized
        units.  After it: 2e-06 deg mean, 1.6e-05 deg max, and 1.79e-07 on the
        joint channel.  The thresholds below sit between those two worlds by
        three orders of magnitude in each direction.
        """
        worst_rotation = worst_joint = worst_origin = 0.0
        for slot, index in enumerate(self.windows):
            legacy = self.legacy[index]
            training = self.prior[slot]
            state = training["x"].numpy()
            worst_joint = max(worst_joint, float(np.abs(
                state[:, 0:84] - np.asarray(legacy["joints"]).reshape(-1, 84)).max()))
            legacy_rotation = transforms.rotation_6d_to_matrix(torch.from_numpy(
                np.array(legacy["global_rot_6d"], dtype=np.float32)).reshape(-1, 22, 6))
            core_rotation = transforms.rotation_6d_to_matrix(
                torch.from_numpy(state[:, 84:216]).reshape(-1, 22, 6))
            worst_rotation = max(worst_rotation, float(np.degrees(
                rotation_geodesic(legacy_rotation.double(), core_rotation.double()).max())))
            worst_origin = max(worst_origin, float(np.abs(
                np.asarray(legacy["mat"]).reshape(4, 4)[:3, 3]
                - training["window_origin"].numpy()).max()))
        self.assertLess(worst_rotation, 1e-2)
        self.assertLess(worst_joint, 1e-4)
        self.assertLess(worst_origin, 1e-6)

    def _forward_kinematics(self, globals_, offsets, reference):
        locals_ = self.legacy.quat_ik_torch(globals_.reshape(-1, 22, 3, 3))
        positions = offsets[None].repeat(locals_.shape[0], 1, 1).clone()
        positions[:, 0, :] = torch.from_numpy(reference[:, 0])
        _rotations, joints = self.legacy.quat_fk_torch(locals_, positions)
        return float(np.abs(joints[:, :22].numpy() - reference[:, :22]).max())

    def test_channel_forward_kinematics_reproduces_the_released_joint_array(self):
        """The criterion: channel plus template must rebuild the released joints.

        Both paths are checked, and so is the pairing.  The rotation correction
        and the template un-conjugation are a second silently compensating pair:
        the fixed channel against the FACTORY (still ``zup_to_yup``-baked)
        template misses by 1.05-1.11 m, six orders of magnitude outside the
        7e-07 m the correct pairing achieves.  Neither half can be shipped
        alone.
        """
        worst_evaluation = worst_training = 0.0
        least_mismatched = float("inf")
        for slot, index in enumerate(self.windows[:8]):
            legacy = self.legacy[index]
            frames = int(np.asarray(legacy["global_rot_6d_gt"]).shape[0])
            start = int(self.legacy.start_ind[index])
            reference = np.array(self.reference[start:start + frames], dtype=np.float32)
            mat = torch.from_numpy(np.array(legacy["mat"], dtype=np.float32))
            globals_ = mat[None, :3, :3] @ transforms.rotation_6d_to_matrix(
                torch.from_numpy(np.array(legacy["global_rot_6d_gt"], dtype=np.float32)))
            template = torch.from_numpy(np.array(legacy["rest_human_offsets"], dtype=np.float32))
            factory = torch.from_numpy(zup_to_yup(
                np.array(legacy["rest_human_offsets"], dtype=np.float32)))
            worst_evaluation = max(worst_evaluation,
                                   self._forward_kinematics(globals_, template, reference))
            least_mismatched = min(least_mismatched,
                                   self._forward_kinematics(globals_, factory, reference))

            training = self.prior[slot]
            shift = torch.from_numpy(
                training["world_to_local_rotation"].numpy().T.astype(np.float32))
            training_globals = shift[None, None] @ transforms.rotation_6d_to_matrix(
                torch.from_numpy(training["x"].numpy()[:, 84:216]).reshape(-1, 22, 6))
            coarse = np.array(self.reference[start:start + 48:3], dtype=np.float32)
            worst_training = max(worst_training, self._forward_kinematics(
                training_globals,
                torch.from_numpy(np.array(training["rest_human_offsets"], dtype=np.float32)),
                coarse))
        self.assertLess(worst_evaluation, 1e-4)
        self.assertLess(worst_training, 1e-4)
        self.assertGreater(least_mismatched, 0.5)

    def test_channel_up_axis_only_mirrors_the_raw_asset_and_cannot_gate(self):
        """Why "the channel reads ~90 deg from the joints" is NOT a defect test.

        The released OMOMO channel was ``G_yup @ M^T``, a uniform right
        multiplication of the whole chain.  Right multiplication cancels
        exactly against the equally rotated template, so FK still closes to
        2.6e-07 m -- old and new score the same.  What the right multiplication
        does change is any readout that dots the root against a world axis:
        ``M^T @ yhat == zhat``, so the old channel's "up" is really the body's
        fore-aft axis and the angle jumps from 22.54 deg to 68.61 deg on OMOMO.

        So this test asserts an IDENTITY against the raw asset, never an
        absolute angle.  The absolute value is corpus dependent (8.98 deg on
        LINGO, 22.54 deg on OMOMO) and the two distributions overlap, so no
        separating threshold exists -- asserting one would either fail on
        correct code or invite someone to "fix" the correction in the wrong
        direction.  FK closure above is the criterion.
        """
        orient = np.load(OMOMO_TEST / "human_orient.npy", mmap_mode="r")
        correction = world_up_correction("z")
        transpose = torch.from_numpy(correction.T).float()
        up = np.array([0.0, 1.0, 0.0])

        def angle(first, second):
            first = first / np.linalg.norm(first)
            second = second / np.linalg.norm(second)
            return float(np.degrees(np.arccos(np.clip(first @ second, -1.0, 1.0))))

        fixed, old = [], []
        worst_fixed = worst_old = 0.0
        for index in self.windows:
            legacy = self.legacy[index]
            start = int(self.legacy.start_ind[index])
            mat = np.asarray(legacy["mat"], dtype=np.float64).reshape(4, 4)[:3, :3]
            channel = transforms.rotation_6d_to_matrix(torch.from_numpy(
                np.array(legacy["global_rot_6d"], dtype=np.float32)).reshape(-1, 22, 6))
            joints = np.asarray(self.reference[start], dtype=np.float64)
            spine = joints[12] - joints[0]

            raw = np.asarray(orient[start], dtype=np.float64)
            reference_fixed = Rotation.from_rotvec(raw).as_matrix()
            # exactly the released operation, zup_to_yup on the axis-angle
            reference_old = Rotation.from_rotvec(zup_to_yup(raw.copy()[None])[0]).as_matrix()

            channel_fixed = angle((mat @ channel[0, 0].numpy().astype(np.float64)) @ up, spine)
            channel_old = angle(
                (mat @ (channel[0, 0] @ transpose).numpy().astype(np.float64)) @ up, spine)
            fixed.append(channel_fixed)
            old.append(channel_old)
            worst_fixed = max(worst_fixed,
                              abs(channel_fixed - angle(correction @ reference_fixed @ up, spine)))
            worst_old = max(worst_old, abs(channel_old - angle(reference_old @ up, spine)))

        # seven significant digits: 22.5412357 against 22.5412352 in the audit
        self.assertLess(worst_fixed, 1e-4)
        self.assertLess(worst_old, 1e-4)
        fixed, old = np.asarray(fixed), np.asarray(old)
        # the conjugation is what moved the readout, by tens of degrees
        self.assertGreater(float(old.mean() - fixed.mean()), 20.0)
        # ...and yet the ranges overlap, which is the whole point: there is no
        # threshold on this quantity that separates correct from broken.
        self.assertLess(float(old.min()), float(fixed.max()))

    def test_bone_lengths_are_rigid_on_the_coarse_window_only(self):
        """Bone-length rigidity is a valid channel check at 16 frames, not 48.

        On the coarse window the joint channel is rigid to 4e-07 m mean /
        1e-06 m max, 0 of 168 bones past 1e-2.  Push the same intact ground
        truth through ``interp_s=3`` and independent linear interpolation of
        joint positions stretches bones to 0.0019 m mean / 0.036 m max with 120
        of 168 past 1e-2.  Anyone who runs a rigidity check on exported motion
        and concludes the channel is broken will be reading the interpolator.
        """
        from utils import interpolate_joints

        coarse, interpolated = [], []
        for index in self.windows:
            legacy = self.legacy[index]
            normalized = np.asarray(legacy["joints"]).reshape(16, 84).astype(np.float32)
            fine = interpolate_joints(torch.from_numpy(normalized), 3).numpy()
            for source, sink in ((normalized, coarse), (fine, interpolated)):
                joints = self.legacy.denormalize(
                    source.reshape(-1, 28, 3)).astype(np.float64)
                lengths = np.linalg.norm(
                    joints[:, 1:22] - joints[:, self.parents[1:22]], axis=-1)
                sink.append(np.abs(lengths - lengths[0:1]).max(0))
        coarse = np.concatenate(coarse)
        interpolated = np.concatenate(interpolated)
        self.assertLess(float(coarse.max()), 1e-4)
        self.assertEqual(int((coarse > 1e-2).sum()), 0)
        self.assertGreater(float(interpolated.max()), 5e-3)
        self.assertGreater(float(interpolated.max()) / float(coarse.max()), 100.0)
class InterpolationGridTests(unittest.TestCase):
    """``interpolate_joints`` and ``interp_jrot`` must sample one grid.

    Both run on every evaluated window and their outputs are combined: the joint
    channel supplies the FK root, the rotation channel supplies the pose.  The
    released ``interpolate_joints`` used ``np.linspace(0, in_len - 1, out_len)``,
    whose step is ``(in_len-1)/(out_len-1)`` rather than ``1/scale``, so the two
    drifted apart by 0.32 frames per step at ``in_len=16, scale=3``, reaching
    0.638 frames by the end of a window -- 39.25 mm of root drift (mean of
    per-sequence maxima), 74.72 mm max, +1.23 cm of ``trans_dist`` on a reported
    7.70 cm and +0.040 of ``foot_sliding``.
    """

    def test_joint_and_rotation_interpolation_share_one_grid(self):
        """Both grids are read out of the functions, not restated.

        ``interp_jrot``'s grid is recovered by recording the blend parameter it
        hands ``quaternion_slerp`` on every call; ``interpolate_joints``' grid is
        recovered by feeding it an integer ramp, for which linear interpolation
        returns the sample positions themselves exactly.  Neither reference is
        this test's own arithmetic, so changing either function's grid fails
        here.  The comparison is bitwise, which is sound because both sides are
        ``base + j/scale`` evaluated in float64 and cast once.
        """
        import utils

        scale, frames = 3, 16
        recorded = []
        original = utils.quaternion_slerp

        def recorder(first, second, step, eps=1e-6):
            recorded.append(float(step))
            return original(first, second, step, eps)

        utils.quaternion_slerp = recorder
        try:
            identity = torch.zeros(frames, 22, 4)
            identity[..., 0] = 1.0
            utils.interp_jrot(identity, scale)
        finally:
            utils.quaternion_slerp = original

        self.assertEqual(len(recorded), (frames - 1) * scale)
        rotation_grid = np.array(
            [index // scale + offset for index, offset in enumerate(recorded)]
            # ``interp_jrot`` writes its final ``scale`` slots from the last
            # input frame rather than through slerp, so they are not recorded.
            + [float(frames - 1)] * scale)

        ramp = torch.arange(frames, dtype=torch.float64)[:, None].repeat(1, 3)
        joint_grid = utils.interpolate_joints(ramp, scale).numpy()[:, 0]
        self.assertTrue(np.array_equal(joint_grid, rotation_grid.astype(np.float32)))

        released = np.linspace(0, frames - 1, frames * scale)
        self.assertGreater(float(np.abs(released - rotation_grid).max()), 0.5)

    def test_mpjpe_cannot_see_the_joint_interpolation_grid(self):
        """The interpolation fix must not move MPJPE, and provably does not.

        ``compute_gt_difference`` subtracts joint 0 from both sides, and joint 0
        of the FK output is exactly the interpolated root that was written into
        ``positions[:, 0]``.  Every other joint is that root plus a chain of
        ``global_rotation @ offset`` terms taken from ``interp_jrot``.  So the
        grid cancels: on real ``data/test`` motion the root moves 33.1 mm while
        the root-relative skeleton changes by 1.2e-07 m, which is float32
        rounding of a 2 m coordinate and not a signal.  ``trans_dist`` and
        ``foot_sliding``, which read the root itself, do move -- that is the
        intended effect.
        """
        from scipy.interpolate import interp1d
        from utils import interpolate_joints

        dataset = _legacy_dataset()
        index = _evaluator_windows(8)[3]
        item = dataset[index]
        mat = torch.from_numpy(np.array(item["mat"], dtype=np.float32))
        globals_ = mat[None, :3, :3] @ transforms.rotation_6d_to_matrix(
            torch.from_numpy(np.array(item["global_rot_6d"], dtype=np.float32)).reshape(-1, 22, 6))
        locals_ = dataset.quat_ik_torch(globals_.reshape(-1, 22, 3, 3)).repeat(3, 1, 1, 1)[:48]
        offsets = torch.from_numpy(np.array(item["rest_human_offsets"], dtype=np.float32))
        joints = torch.from_numpy(dataset.denormalize(
            np.asarray(item["joints"]).reshape(-1, 28, 3)).astype(np.float32)).reshape(16, 84)

        fixed_root = interpolate_joints(joints, 3).reshape(48, 28, 3)[:, 0, :]
        released_root = torch.from_numpy(interp1d(np.arange(16), joints.numpy(), axis=0)(
            np.linspace(0, 15, 48))).float().reshape(48, 28, 3)[:, 0, :]

        outputs = []
        for root in (fixed_root, released_root):
            positions = offsets[None].repeat(48, 1, 1).clone()
            positions[:, 0, :] = root
            _rotations, joint_positions = dataset.quat_fk_torch(locals_, positions)
            outputs.append(joint_positions)
        root_shift = float((outputs[0][:, 0] - outputs[1][:, 0]).abs().max())
        relative = ((outputs[0] - outputs[0][:, 0:1])
                    - (outputs[1] - outputs[1][:, 0:1])).abs().max()
        self.assertGreater(root_shift, 5e-3)
        self.assertLess(float(relative), 1e-5)
        self.assertGreater(root_shift / max(float(relative), 1e-12), 1e3)


# ---------------------------------------------------------------------------
# P12 CONTROL: the step-0 window frame rule
# ---------------------------------------------------------------------------

WINDOW_ARRAY_KEYS = (
    "mat", "global_rot_6d", "global_rot_6d_gt", "joints", "joints_gt",
    "pelvis_goal", "scene_goal", "object_goal", "object_trans",
    "object_rot_mat", "obj_rot_mat_ref", "rest_human_offsets", "transl",
)


@contextlib.contextmanager
def _with_step0_rule(dataset, rule):
    """Flip one dataset's step-0 rule for the duration of a block.

    The rule is read in ``__getitem__``, never in ``__init__``, so flipping the
    attribute on an already built instance is the whole switch -- and proving
    that is half of what the first test below asserts.  Restored on exit because
    the instance is ``lru_cache``d and shared with every other test class here.
    """
    previous = dataset.step0_frame_rule
    dataset.step0_frame_rule = rule
    try:
        yield dataset
    finally:
        dataset.step0_frame_rule = previous


def _window_arrays(item):
    return {key: np.ascontiguousarray(np.asarray(item[key])) for key in WINDOW_ARRAY_KEYS}


class Step0FrameRuleTests(unittest.TestCase):
    """One checkpoint, two step-0 window frame rules: the P12 CONTROL cell.

    P12 changed the representation *and* the evaluator's step-0 orientation rule
    together, so no PRIMARY-minus-anything difference is attributable.  The
    CONTROL cell moves only the evaluator half: same checkpoint, same repaired
    data, and the window normalization frame built from the released rule
    instead.  ``historical_conjugated_root_matrix`` reconstructs the released
    input exactly -- ``M C^T R_rep M^T`` -- so the two cells are two functions of
    one identical dataset rather than two datasets.

    What that buys and what it does not:

    * ``|A - P|``, the mismatch CONTROL opens against the trained convention, is
      77.83 deg mean / 51.89 deg p50 at the 438 step-0 windows.  It is a real
      perturbation, not a no-op.
    * It is NOT the 50.12 deg the released code actually suffered.  That number
      is ``|A - B|``: the released EVALUATION rule against the released TRAINING
      rule (pytorch3d intrinsic ``"ZXY"[..., 2]``, the innermost y angle), two
      different conventions on the same conjugated input.  CONTROL instead pairs
      one convention with two different inputs.  Reproducing 50.12 deg would
      need the pre-fix codec convention as well, and that lives in the frozen
      ``priors/core/``.
    * ``|C - P|``, flipping the codec convention alone, is 0.41 deg mean / 0.13
      p50 -- the measured no-op, and the reason that route was rejected.

    Every assertion is anchored on ``human_orient.npy`` read raw, on the released
    joint array, or on source text; never on what the branch under test emits.
    """

    @classmethod
    def setUpClass(cls):
        cls.repaired = _legacy_dataset()
        cls.control = _control_dataset()
        cls.windows = _evaluator_windows()
        cls.reference = np.load(OMOMO_TEST / "human_joints_aligned.npy", mmap_mode="r")

    def test_the_default_rule_is_repaired_and_selects_the_released_arithmetic(self):
        """Switch off means byte-for-byte unchanged, and off is the default.

        Three independent statements, because a silently flipped default is the
        one failure of this change that would corrupt a PRIMARY result rather
        than merely fail:

        1. the constructor default is pinned to ``'repaired'``, and the dataset
           the real eval config builds reports it;
        2. the released three statements still appear once each in
           ``datasets/infbagel.py``, so neither was rewritten or duplicated into
           the CONTROL branch;
        3. the CONTROL instance, with its rule flipped back to ``'repaired'``,
           returns windows bitwise identical to the default instance's -- which
           also proves ``__init__`` consumed the rule for nothing but validation,
           since the two instances were built independently.

        Measured outside the suite over all 1314 evaluator windows and all 14
        returned arrays: the default path is bitwise identical to the pre-switch
        tree, ``maxabsdiff`` exactly 0.

        Point 3 is deliberately a RELATIVE check -- two instances agreeing -- and
        therefore cannot see a branch that applies the CONTROL rule
        unconditionally, since both instances would then move together.  The
        absolute anchor for the default path is the repaired-rule half of
        ``test_the_control_rule_is_the_released_conjugated_readout``, which
        compares it against the raw asset; deleting the ``if`` fails there.
        """
        from datasets.infbagel import InfBaGelDataset

        # ``Dataset`` is ``Generic``, so ``signature(cls)`` reports ``(*args,
        # **kwds)``; the class's own ``__init__`` is where the default lives.
        signature = inspect.signature(InfBaGelDataset.__init__)
        self.assertEqual(signature.parameters["step0_frame_rule"].default, "repaired")
        self.assertEqual(STEP0_FRAME_RULE_CHOICES, ("repaired", "historical_conjugated"))
        self.assertEqual(self.repaired.step0_frame_rule, "repaired")
        self.assertEqual(self.control.step0_frame_rule, "historical_conjugated")

        text = (REPO / "code/datasets/infbagel.py").read_text()
        for statement in (
            "init_global_orient_euler = R.from_rotvec(init_global_orient).as_euler('zxy')",
            "shift_euler = np.array([0, 0, -init_global_orient_euler[2]])",
            "shift_rot_matrix = R.from_euler('zxy', shift_euler).as_matrix()",
        ):
            self.assertEqual(text.count(statement), 1, statement)

        with _with_step0_rule(self.control, "repaired") as reverted:
            for index in self.windows:
                with self.subTest(window=index):
                    expected = _window_arrays(self.repaired[index])
                    actual = _window_arrays(reverted[index])
                    for key, value in expected.items():
                        self.assertEqual(value.dtype, actual[key].dtype, key)
                        self.assertEqual(value.shape, actual[key].shape, key)
                        self.assertEqual(value.tobytes(), actual[key].tobytes(), key)

    def test_the_control_rule_is_the_released_conjugated_readout(self):
        """CONTROL's shift equals a by-hand reference off the raw asset.

        The reference re-reads ``human_orient.npy``, applies the released
        ``zup_to_yup`` to the axis-angle itself, and takes scipy's extrinsic
        ``'zxy'`` index 2.  It shares no line of code with the branch under test,
        so this cannot pass by asserting what the branch emits.  Measured over
        all 1314 windows: 2.36e-06 deg max for CONTROL and 2.36e-06 deg for the
        repaired rule, both of which are the float32 quantisation of ``mat``.

        The same comparison against the *other* rule's reference is 179.93 deg
        apart at worst, so the two references are not interchangeable and the
        agreement above is not vacuous.
        """
        conjugation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        worst = {"repaired": 0.0, "historical_conjugated": 0.0}
        worst_crossed = 0.0
        for index in self.windows:
            start = int(self.repaired.start_ind[index])
            for rule, dataset in (("repaired", self.repaired),
                                  ("historical_conjugated", self.control)):
                measured = _shift_yaw_from_mat(dataset[index]["mat"])
                worst[rule] = max(worst[rule], abs(float(_wrapped_degrees(
                    measured - _reference_shift_yaw(start, rule)))))
                other = "repaired" if rule == "historical_conjugated" else "historical_conjugated"
                worst_crossed = max(worst_crossed, abs(float(_wrapped_degrees(
                    measured - _reference_shift_yaw(start, other)))))

            # the reconstruction itself, against the raw asset conjugated by hand
            raw = np.asarray(_raw_root_orientations()[start], dtype=np.float64)
            released = Rotation.from_rotvec(zup_to_yup(raw.copy()[None])[0]).as_matrix()
            rebuilt = historical_conjugated_root_matrix(
                np.asarray(self.repaired.global_orient[start], dtype=np.float64), "z")
            self.assertLess(float(np.abs(rebuilt - released).max()), 1e-12)
            self.assertLess(float(np.abs(
                conjugation @ Rotation.from_rotvec(raw).as_matrix() @ conjugation.T
                - released).max()), 1e-12)

        self.assertLess(worst["repaired"], STEP0_SHIFT_YAW_TOLERANCE_DEG)
        self.assertLess(worst["historical_conjugated"], STEP0_SHIFT_YAW_TOLERANCE_DEG)
        self.assertGreater(worst_crossed, 90.0)

    def test_both_rules_stay_exact_inverses_so_the_ground_truth_row_cannot_move(self):
        """``mat`` must undo whichever shift built the window, for both rules.

        This is the property that makes CONTROL a clean single-factor cell.  The
        evaluator's ground-truth reference row is
        ``transform_points(denormalize(joints), mat)`` and
        ``mat[:3, :3] @ rotation_6d_to_matrix(global_rot_6d_gt)``
        (``test_infbagel_hoi.py:784`` and ``:809``), both of which are
        frame-invariant only if the round trip is exact.  Measured: the two rules
        agree on absolute global joint position to 7.6e-07 m and each reproduces
        ``human_joints_aligned.npy`` to 4.1e-07 m, so the GT row is the same row
        in both cells and only the model's conditioning frame moves.

        The frame-independent channels are asserted bitwise, not approximately:
        ``object_rot_mat`` is reference-relative and ``joints_gt`` is the raw
        global array, so a rule that touched either would be re-framing the
        target as well as the input.
        """
        invariant = ("joints_gt", "object_rot_mat", "obj_rot_mat_ref",
                     "rest_human_offsets", "transl")
        worst_position = worst_reference = worst_rotation = 0.0
        worst_inverse = 0.0
        for index in self.windows:
            repaired_item, control_item = self.repaired[index], self.control[index]
            start = int(self.repaired.start_ind[index])
            reference = np.asarray(self.reference[start:start + 48:3], dtype=np.float64)
            globals_ = []
            for item in (repaired_item, control_item):
                mat = np.asarray(item["mat"], dtype=np.float64).reshape(4, 4)
                shift = mat[:3, :3].T
                worst_inverse = max(worst_inverse, float(np.abs(
                    shift @ mat[:3, :3] - np.eye(3)).max()))
                joints = np.asarray(self.repaired.denormalize(
                    np.asarray(item["joints"], dtype=np.float32).reshape(-1, 28, 3)),
                    dtype=np.float64)
                globals_.append(joints @ mat[:3, :3].T + mat[:3, 3])
                worst_reference = max(worst_reference, float(np.abs(
                    globals_[-1][:, :22] - reference[:, :22]).max()))
                # the GT FK expression, mat[:3,:3] @ the 48-frame rotation channel
                gt = transforms.rotation_6d_to_matrix(torch.from_numpy(
                    np.array(item["global_rot_6d_gt"], dtype=np.float32)))
                globals_.append(np.asarray(
                    mat[None, None, :3, :3] @ gt.double().numpy(), dtype=np.float64))
            worst_position = max(worst_position, float(np.abs(globals_[0] - globals_[2]).max()))
            worst_rotation = max(worst_rotation, float(np.abs(globals_[1] - globals_[3]).max()))
            for key in invariant:
                left = np.ascontiguousarray(np.asarray(repaired_item[key]))
                right = np.ascontiguousarray(np.asarray(control_item[key]))
                self.assertEqual(left.tobytes(), right.tobytes(), key)
        self.assertLess(worst_inverse, 1e-6)
        self.assertLess(worst_position, 1e-5)
        self.assertLess(worst_rotation, 1e-5)
        self.assertLess(worst_reference, 1e-4)

    def test_the_two_rules_differ_by_tens_of_degrees_at_the_438_step0_windows(self):
        """Is CONTROL worth an evaluation pass?  The distribution says yes.

        Four heading rules, all built here from the raw asset so the comparison
        does not depend on the branch under test, at the 438 window starts the
        evaluator actually seeds a rollout from:

        ==  ==================================================  ==============
        P   scipy extrinsic ``'zxy'[2]`` of ``C R_stored``       trained/repaired
        A   scipy extrinsic ``'zxy'[2]`` of ``M R_stored M^T``   released eval
        B   pytorch3d ``"ZXY"[..., 2]`` of ``M R_stored M^T``    released train
        C   pytorch3d ``"ZXY"[..., 2]`` of ``C R_stored``        codec flip only
        ==  ==================================================  ==============

        Measured 2026-08-19, mean / p50 / p95 / max in degrees:

        * ``|A - P|`` = 77.83 / 51.89 / 175.60 / 179.81 -- what CONTROL costs
          the model.  93.2 percent of windows move more than 5 deg.
        * ``|A - B|`` = 50.12 / 38.98 / 145.14 / 179.89 -- the mismatch the
          released code really had, and the reason this test asserts it: it
          reproduces the independently recorded 50.12 deg, which identifies A as
          the released EVALUATION rule and not something adjacent to it.
        * ``|C - P|`` = 0.41 / 0.13 / 1.79 / 4.62 -- the rejected route.  A
          CONTROL built this way would be a no-op and would credit the model with
          the whole difference.
        """
        starts = np.asarray(_step0_window_indices(), dtype=np.int64)
        starts = np.asarray([int(self.repaired.start_ind[i]) for i in starts], dtype=np.int64)
        raw = np.asarray(_raw_root_orientations()[starts], dtype=np.float64)
        stored = Rotation.from_rotvec(raw).as_matrix()
        conjugated = Rotation.from_rotvec(zup_to_yup(raw.copy())).as_matrix()
        repaired = world_up_correction("z")[None] @ stored

        def extrinsic(matrices):
            return np.degrees(Rotation.from_matrix(matrices).as_euler("zxy")[:, 2])

        def intrinsic(matrices):
            return np.degrees(transforms.matrix_to_euler_angles(
                torch.from_numpy(matrices), "ZXY")[..., 2].numpy())

        angles = {"P": extrinsic(repaired), "A": extrinsic(conjugated),
                  "B": intrinsic(conjugated), "C": intrinsic(repaired)}
        self.assertEqual(len(starts), 438)

        control = np.abs(_wrapped_degrees(angles["A"] - angles["P"]))
        historical = np.abs(_wrapped_degrees(angles["A"] - angles["B"]))
        codec_only = np.abs(_wrapped_degrees(angles["C"] - angles["P"]))

        # CONTROL is a real perturbation
        self.assertGreater(float(control.mean()), 45.0)
        self.assertGreater(float(np.percentile(control, 50)), 20.0)
        self.assertGreater(float((control > 5.0).mean()), 0.8)
        # ...and it is the released evaluation rule, identified by reproducing
        # the recorded 50.12 deg released train/eval mismatch
        self.assertGreater(float(historical.mean()), 45.0)
        self.assertLess(float(historical.mean()), 55.0)
        # ...and it is not the rejected codec-convention flip, which is a no-op
        self.assertLess(float(codec_only.mean()), 2.0)
        self.assertGreater(float(control.mean()) / max(float(codec_only.mean()), 1e-9), 20.0)

    def test_an_unrecognised_step0_frame_rule_raises_before_any_io(self):
        """A typo must be a hard failure, and must not cost a dataset load.

        The validation sits above the first ``np.load``, so a bad value raises
        without touching ``folder`` -- which is what lets this test run in
        milliseconds against a path that does not exist.
        """
        from datasets.infbagel import InfBaGelDataset

        for bad in ("historical", "Repaired", "", None, 0, "legacy"):
            with self.subTest(rule=bad):
                with self.assertRaises(ValueError):
                    InfBaGelDataset("/nonexistent/p12-control", "cpu", None, 1, 3,
                                    [32, 32, 32], step0_frame_rule=bad)

    def test_the_step0_rule_cannot_reach_the_hoi_training_path(self):
        """HOIPrior trains through ``priors/hoi/data.py``, never through here.

        Asserted rather than described, because the whole premise of an
        evaluation-only switch is that no training run can pick it up.
        ``train_hoi_prior.py`` imports ``PriorWindowDataset`` and never
        ``datasets.infbagel``, and neither the HOI training dataset nor the
        frozen codec mentions the rule.
        """
        trainer = (REPO / "code/train_hoi_prior.py").read_text()
        self.assertNotIn("datasets.infbagel", trainer)
        self.assertIn("from priors.hoi.data import PriorWindowDataset", trainer)
        for path in ("code/priors/hoi/data.py", "code/priors/core/window_codec.py"):
            self.assertNotIn("step0_frame_rule", (REPO / path).read_text(), path)


if __name__ == "__main__":
    unittest.main()
