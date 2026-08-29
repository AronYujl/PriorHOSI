"""The LINGO-only normalization box, and the FK term's history mask.

Two defects in the released code, both fixed on 2026-08-18, both previously
untested:

* ``InfBaGelMixDataset`` normalized the joint channel with one box and
  denormalized it with another.  ``InfBaGelDataset.__getitem__`` normalizes with
  ``<lingo_folder>/norm.npy``; the mix's ``unified_min``/``unified_max`` were
  loaded from ``<lingo_folder>/norm_inter_and_loco__16frames.npy``, a (2, 3) box
  whose per-axis range is ``[0.39924, 1.04313, 0.39456]`` of ``norm.npy``'s.  So
  ``mix.denormalize_torch(sub.normalize(v))`` was ``S v + c`` with
  ``S = [0.39924, 1.04313, 0.39456]`` and
  ``c = [-0.03386, -0.07942, -0.12771] m`` instead of ``v`` -- up to 1.03 m per
  joint in world metres.  Every consumer of the unified box was affected: the 13
  human-position normalizer sites in ``models/infbagel.py`` (including
  ``loss_fk``'s FK and ``_compute_occ``'s scene query) and the 7 in the two
  evaluators.  The existing coverage in ``test_representation_frame.py`` only
  ever exercised the sub-dataset's self-consistent pair, which cannot see a
  disagreement between two objects.

* ``Sampler.p_losses`` built ``mask_fk`` and never applied it, while
  ``Sampler.consistency_loss`` did.  Since the masked frames are exactly the
  ``auto_regre_num`` history frames -- which ``set_fixed_points`` overwrites at
  every sampling step, and which ``mask_inv`` already excludes from all five base
  losses -- the diffusion stage's ``loss_fk`` was ``(T - auto_regre_num) / T``
  times the consistency stage's on identical geometry, so B's and C's
  ``loss_w_fk`` were not comparable.

The metre-domain assertions are anchored on ``InfBaGelDataset``'s own inverse,
which ``test_representation_frame.py`` validates against the release arrays.
Each one also measures what the wrong box would have produced, so the test proves
it has the power to fail.
"""

import ast
import os
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from datasets.infbagel_mix import InfBaGelMixDataset

LINGO = REPO / "data/dataset"
OMOMO = REPO / "data/train"
SPLIT = REPO / "experiments/splits/lingo_scene_family_disjoint_v3_seed42.json"
MODEL_SOURCE = REPO / "code/models/infbagel.py"

WINDOW_FRAMES = 16
DATA_STEP = 3
NB_JOINTS = 28
JOINTS_IND = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
              19, 20, 21, 23, 24, 25, 28, 40, 43]
ROUND_TRIP_WINDOWS = 256

# Measured on 2026-08-18 over 200 real v3-train windows: the repaired path is
# exact to 4.8e-07 m and the defect was 1.16 m local / 1.03 m world, so a
# 1e-05 m gate sits five orders of magnitude inside the margin on both sides.
METRE_TOLERANCE = 1e-5
CHANNEL_TOLERANCE = 1e-5
# The smallest world displacement the released box produced on any of the 200
# probe windows was 0.20 m; 0.5 m is a floor the defect clears everywhere.
DEFECT_FLOOR_M = 0.5

WORKER_EXPERT = os.environ.get("INFBAGEL_WORKER_EXPERT")


def _available():
    return (LINGO / "human_joints_aligned.npy").is_file() and (OMOMO / "norm.npy").is_file()


@unittest.skipIf(WORKER_EXPERT == "hoi", "HOI worker intentionally has no real LINGO assets")
@unittest.skipUnless(_available(), "data/dataset and data/train are required")
class UnifiedNormalizationBoxTests(unittest.TestCase):
    """The real ``InfBaGelMixDataset.__init__`` under ``lingo_only``.

    ``tests/hsi/test_nearest_free_voxel.py`` builds the type through ``__new__``
    because it only needs three attributes.  Here the object under test *is* what
    ``__init__`` computes, so it has to run for real: the ordering matters, since
    ``__init__`` copies OMOMO's object rows onto the LINGO sub-dataset before
    ``_compute_unified_normalization_params`` reads them.
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = InfBaGelMixDataset(
            omomo_folder=str(OMOMO), lingo_folder=str(LINGO),
            device=torch.device("cpu"), mesh_grid=[-0.6, 0.6, 0.1, 1.2, -0.6, 0.6],
            batch_size=1, step=DATA_STEP, nb_voxels=[32, 32, 32], train=True,
            load_scene=True, load_language=True, load_pelvis_goal=False,
            load_scene_goal=False, load_object_goal=False,
            use_random_frame_bps=False, use_object_keypoints=False,
            max_window_size=WINDOW_FRAMES, use_pi=True, vis=False,
            lingo_only=True, random_seed=42,
            split_manifest=str(SPLIT), split_partition="train",
            joints_ind=JOINTS_IND, nb_joints=NB_JOINTS,
        )
        cls.sub = cls.dataset.lingo_dataset
        with (LINGO / "scene_name.pkl").open("rb") as handle:
            cls.sub.scene_name = pickle.load(handle)
        # Deterministic, spread over the whole LINGO-only index set.
        picks = np.unique(np.linspace(0, len(cls.sub) - 1, ROUND_TRIP_WINDOWS)
                          .round().astype(np.int64))
        items = [cls.sub[int(index)] for index in picks]
        cls.channel = torch.from_numpy(
            np.stack([item["joints"].reshape(WINDOW_FRAMES, NB_JOINTS, 3) for item in items]))
        cls.n_windows = len(picks)

    @classmethod
    def _released_box(cls):
        """The (2, 3) box the released ``lingo_only`` branch loaded."""
        released = np.load(LINGO / "norm_inter_and_loco__16frames.npy")
        cls_min = released[0].astype(np.float32)
        cls_max = released[1].astype(np.float32)
        return torch.tensor(cls_min), torch.tensor(cls_max)

    def _denormalize_with(self, minimum, maximum):
        return (self.channel + 1.0) * (maximum - minimum) / 2.0 + minimum

    def test_unified_position_box_is_the_sub_datasets_own_box(self):
        """The whole defect in one assertion, on the arrays rather than on a
        derived quantity, so swapping the file back fails here first."""
        np.testing.assert_array_equal(self.dataset.unified_min, self.sub.min)
        np.testing.assert_array_equal(self.dataset.unified_max, self.sub.max)
        self.assertTrue(torch.equal(self.dataset.unified_min_torch, self.sub.min_torch))
        self.assertTrue(torch.equal(self.dataset.unified_max_torch, self.sub.max_torch))
        # ...and those rows are norm.npy's, which core/contracts.py names.
        norm = np.load(LINGO / "norm.npy")
        np.testing.assert_array_equal(self.dataset.unified_min, norm[0].astype(np.float32))
        np.testing.assert_array_equal(self.dataset.unified_max, norm[1].astype(np.float32))

    def test_unified_position_box_is_not_the_released_two_row_file(self):
        """LINGO's own ``norm_inter_and_loco__16frames.npy`` fits the corpus far
        better and is deliberately not used, because the position box is shared
        with HOIPrior and the mixer.  Retuning it is a later, deliberate change;
        if it ever happens it must not happen by reinstating this file here."""
        released_min, released_max = self._released_box()
        self.assertFalse(np.allclose(self.dataset.unified_min, released_min.numpy()))
        self.assertFalse(np.allclose(self.dataset.unified_max, released_max.numpy()))
        ratio = ((released_max - released_min)
                 / (self.sub.max_torch - self.sub.min_torch)).numpy()
        np.testing.assert_allclose(ratio, [0.39924, 1.04313, 0.39456], atol=1e-5)

    def test_object_rows_keep_their_omomo_provenance(self):
        """``unified_obj_*`` are unchanged by the repair: LINGO carries no
        objects, so ``__init__`` copies OMOMO's rows onto the sub-dataset and both
        sides of the pair already agreed."""
        norm = np.load(LINGO / "norm.npy")
        np.testing.assert_allclose(self.dataset.unified_obj_min, norm[2], atol=0)
        np.testing.assert_allclose(self.dataset.unified_obj_max, norm[3], atol=0)
        np.testing.assert_array_equal(self.dataset.unified_obj_min, self.sub.obj_min)
        np.testing.assert_array_equal(self.dataset.unified_obj_max, self.sub.obj_max)

    def test_mix_decodes_real_windows_to_the_same_metres_as_the_sub_dataset(self):
        """The cross-object direction training actually uses: the sub-dataset
        normalizes in ``__getitem__``, the mix denormalizes in ``p_losses``,
        ``_compute_occ`` and the evaluator."""
        mix = self.dataset.denormalize_torch(self.channel)
        sub = self.sub.denormalize_torch(self.channel)
        self.assertLess(float((mix - sub).abs().max()), METRE_TOLERANCE)

        # The test has power: the released box moved these same windows metres.
        wrong = self._denormalize_with(*self._released_box())
        self.assertGreater(float((wrong - sub).abs().max()), DEFECT_FLOOR_M)

    def test_mix_and_sub_dataset_round_trip_the_channel_both_ways(self):
        """``normalize_torch`` and ``denormalize_torch`` must invert each other
        across the two objects, not only within one of them."""
        for decoder, label in ((self.dataset, "mix"), (self.sub, "sub")):
            metres = decoder.denormalize_torch(self.channel)
            for encoder in (self.dataset, self.sub):
                with self.subTest(decode=label, encode=type(encoder).__name__):
                    back = encoder.normalize_torch(metres)
                    self.assertLess(float((back - self.channel).abs().max()),
                                    CHANNEL_TOLERANCE)

        # The test has power: encoding the correct metres with the released box
        # displaces the channel far outside float32 noise.
        released_min, released_max = self._released_box()
        metres = self.sub.denormalize_torch(self.channel)
        wrong = -1.0 + 2.0 * (metres - released_min) / (released_max - released_min)
        self.assertGreater(float((wrong - self.channel).abs().max()), DEFECT_FLOOR_M)


def _fk_statements(function, source):
    """The FK-loss statements of one ``Sampler`` method, in source order.

    Each is returned twice: as whitespace-normalized source, which reads in a
    failure message, and as an ``ast.dump`` fingerprint, which ignores comments,
    line breaks and indentation entirely.  The two methods differ only in the
    name of the predicted tensor, so that one name is unified.
    """
    keep = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        text = ast.get_source_segment(source, node)
        if text is None:
            continue
        if "mask_fk" in text or "fk_hand_loss" in text or "fk_foot_loss" in text:
            keep.append((
                node.lineno,
                " ".join(text.split()).replace("predicted_noise", "model_pred"),
                ast.dump(node).replace("predicted_noise", "model_pred"),
            ))
    keep.sort()
    return [(text, dump) for _, text, dump in keep]


def _sampler_methods():
    source = MODEL_SOURCE.read_text()
    tree = ast.parse(source)
    sampler = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "Sampler")
    methods = {node.name: node for node in sampler.body
               if isinstance(node, ast.FunctionDef)}
    return source, methods["p_losses"], methods["consistency_loss"]


class MaskFkParityTests(unittest.TestCase):
    def test_both_stages_score_loss_fk_with_the_same_statements(self):
        """B distills into C, so the two stages' ``loss_fk`` must be the same
        function of the same geometry or ``loss_w_fk`` cannot be carried across.
        Before the repair ``p_losses`` omitted the ``[mask_fk]`` subscript that
        ``consistency_loss`` applies, and nothing compared them."""
        source, diffusion, consistency = _sampler_methods()
        left = _fk_statements(diffusion, source)
        right = _fk_statements(consistency, source)
        self.assertEqual([text for text, _ in left], [text for text, _ in right])
        self.assertEqual([dump for _, dump in left], [dump for _, dump in right])

    def test_each_stage_applies_the_history_mask(self):
        """Stated directly, so the failure message names the defect rather than
        only reporting that two blocks differ."""
        source, *methods = _sampler_methods()
        for function in methods:
            statements = [text for text, _ in _fk_statements(function, source)
                          if "F.mse_loss" in text]
            self.assertEqual(len(statements), 2, function.name)
            for text in statements:
                with self.subTest(method=function.name, statement=text):
                    self.assertEqual(text.count("[mask_fk]"), 2)

    def test_masking_the_history_frames_is_the_measured_0875_factor(self):
        """The size of the incommensurability, on synthetic geometry whose
        history frames are exact -- which is what ``set_fixed_points`` enforces
        during sampling.  auto_regre_num = 2 of 16 gives 14/16 = 0.875."""
        auto_regre_num, frames = 2, WINDOW_FRAMES
        torch.manual_seed(42)
        target = torch.randn(8, frames, 4, 3, dtype=torch.float64)
        prediction = target + 0.05 * torch.randn_like(target)
        prediction[:, :auto_regre_num] = target[:, :auto_regre_num]
        mask = torch.ones(8, frames, 4, 3, dtype=torch.bool)
        mask[:, :auto_regre_num] = False

        unmasked = torch.nn.functional.mse_loss(prediction, target)
        masked = torch.nn.functional.mse_loss(prediction[mask], target[mask])
        self.assertAlmostEqual(float(unmasked / masked),
                               (frames - auto_regre_num) / frames, places=12)


if __name__ == "__main__":
    unittest.main()
