"""The training-time ``need_scene`` conditioning gate, and its override.

``need_scene`` is a precomputed per-window boolean in the released LINGO asset
``data/dataset/language_motion_dict/language_motion_dict__inter_and_loco__16.pkl``.
``InfBaGelDataset.__getitem__`` emits it verbatim and
``models/infbagel.py:1433-1438`` zeroes all five scene tokens -- ``scene_emb``
(the goal-centred crop) and ``scene_emb_0..3`` (the current frame and the three
temporal voxels) -- for every row where it is False.  Measured on the v3 training
split: False on 522,818 of 1,343,667 windows, 38.9098%.  So the scene encoder
receives gradient from 61.09% of rows.

Inference never reads it.  ``test_infbagel_lingo_hsi.py:1362`` pins it True for
every one of the 375 sealed episodes, and ``need_scene=False`` appears there only
as the paired RDS null-scene rollout (:1848).  That is a train/test conditioning
mismatch rather than a shared blindness, and ``force_need_scene`` removes it.

Nothing here existed before: ``grep -rn need_scene tests/`` returned nothing, so
the gate was entirely uncovered on both sides of the switch.  The four things
that have to hold, and are asserted below:

1. **Default off.**  Absent the kwarg the flag is False and ``__getitem__``
   returns the pickle's value unchanged, so every sealed config -- training and
   evaluation -- keeps its exact meaning.
2. **On forces the gate open** for training windows, and the returned object
   still collates to ``torch.bool``.
3. **The flag reaches both inner datasets** through ``InfBaGelMixDataset``'s
   ``**kwargs``, which is the only path a Hydra ``dataset:`` key can take.
4. **Inference is unaffected**, for two independent reasons: no sampling config
   sets the flag, and the evaluator pins ``need_scene`` itself instead of reading
   the dataset's value.

Plus one negative: the model-side zeroing must still be there.  Deleting it would
have been the other way to implement this arm, and it would also have deleted the
evaluator's RDS null-scene pass, which *is* ``need_scene=False``.
"""

import ast
import gc
import os
import random
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from datasets import infbagel_mix as mix_module
from datasets.infbagel_mix import InfBaGelMixDataset

LINGO = REPO / "data/dataset"
OMOMO = REPO / "data/train"
SPLIT = REPO / "experiments/splits/lingo_scene_family_disjoint_v3_seed42.json"
DATASET_SOURCE = REPO / "code/datasets/infbagel.py"
MIX_SOURCE = REPO / "code/datasets/infbagel_mix.py"
MODEL_SOURCE = REPO / "code/models/infbagel.py"
EVALUATOR_SOURCE = REPO / "code/test_infbagel_lingo_hsi.py"
TRAINER_SOURCE = REPO / "code/train_infbagel.py"
HOSI_SOURCE = REPO / "code/test_infbagel_hosi.py"
CONFIG_DIR = REPO / "code/config"
ARM_CONFIG = CONFIG_DIR / "config_train_hsi_b_p16ns.yaml"

WORKER_EXPERT = os.environ.get("INFBAGEL_WORKER_EXPERT")

# The B-v2 recipe's dataset group, code/config/dataset/lingo_v3_train.yaml,
# resolved against config_train_hsi_b_lingo_full.yaml.  Only ``device`` differs:
# these tests never touch CUDA.
JOINTS_IND = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
              19, 20, 21, 23, 24, 25, 28, 40, 43]
DATASET_KWARGS = dict(
    omomo_folder=str(OMOMO), lingo_folder=str(LINGO),
    device=torch.device("cpu"), mesh_grid=[-0.6, 0.6, 0.1, 1.2, -0.6, 0.6],
    batch_size=512, step=3, nb_voxels=[32, 32, 32], train=True,
    load_scene=True, load_language=True, load_pelvis_goal=True,
    load_scene_goal=True, load_object_goal=True,
    use_random_frame_bps=True, use_object_keypoints=True,
    max_window_size=16, use_pi=True, vis=False, start_type="stand",
    human_only_ratio=0.4, lingo_scene_num=45, lingo_data_ratio=0.5,
    empty_omomo_scene=False, lingo_only=True, random_seed=42,
    split_manifest=str(SPLIT), split_partition="train",
    joints_ind=JOINTS_IND, nb_joints=28,
)

# Measured 2026-08-26 on this exact index set, and independently in
# .claude/scratch/p16-needscene/build.py from the released arrays alone.
EXPECTED_TRAIN_WINDOWS = 1343667
EXPECTED_NEED_SCENE_FALSE = 522818

# One construction is ~9 s and ~11.5 GiB, so the two real-data classes below each
# build one and release it; they are never both live.  600 __getitem__ calls is
# ~1 s and carries ~230 pickle-False windows, two orders of magnitude more power
# than the assertions need.
SAMPLE_WINDOWS = 600
PAIRED_WINDOWS = 60


def _data_available():
    return (LINGO / "human_joints_aligned.npy").is_file() and (OMOMO / "norm.npy").is_file()


def _sample_indices(dataset, count):
    """A deterministic sample of the dataloader-visible LINGO window ids.

    ``self.indices`` is what ``DistributedSampler`` draws from, so this is the
    population the training run actually sees -- not the raw 2,275,973-window
    pickle, and not the pre-filter scene selection.
    """
    ids = np.asarray([i for (source, i) in dataset.indices if source == 1], dtype=np.int64)
    picks = np.random.RandomState(20260826).choice(len(ids), size=count, replace=False)
    picks.sort()
    return ids, ids[picks], np.asarray(dataset.lingo_dataset.need_scene, dtype=bool)[ids[picks]]


def _item(dataset, window_id):
    """One window under a pinned RNG state.

    ``__getitem__`` draws from ``np.random`` at infbagel.py:471/474 to jitter the
    progress indicator, so two calls at the same index differ unless both draws
    start from the same state.  Seeding on the index makes the flag the only
    thing that can move.
    """
    np.random.seed(int(window_id) % (2 ** 31)); random.seed(int(window_id))
    return dataset.lingo_dataset[int(window_id)]


@unittest.skipIf(WORKER_EXPERT == "hoi", "HOI worker intentionally has no real LINGO assets")
@unittest.skipUnless(_data_available(), "data/dataset and data/train are required")
class NeedSceneGateDefaultTests(unittest.TestCase):
    """No kwarg.  This is the state every sealed config is in."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = InfBaGelMixDataset(**DATASET_KWARGS)
        cls.ids, cls.picks, cls.truth = _sample_indices(cls.dataset, SAMPLE_WINDOWS)

    @classmethod
    def tearDownClass(cls):
        del cls.dataset
        gc.collect()

    def test_the_flag_defaults_to_false_on_both_inner_datasets(self):
        self.assertIs(self.dataset.lingo_dataset.force_need_scene, False)
        self.assertIs(self.dataset.omomo_dataset.force_need_scene, False)

    def test_the_dataloader_visible_population_is_the_recorded_baseline(self):
        """Exact over all 1.34 M indices, not a sample.  If the split manifest,
        the is_pick filter or the sequence-length filter ever moves, the 38.91%
        that motivates this arm moves with it and this fails first."""
        need_scene = np.asarray(self.dataset.lingo_dataset.need_scene, dtype=bool)[self.ids]
        self.assertEqual(len(self.ids), EXPECTED_TRAIN_WINDOWS)
        self.assertEqual(int((~need_scene).sum()), EXPECTED_NEED_SCENE_FALSE)
        self.assertAlmostEqual(float((~need_scene).mean()), 0.389098, places=6)

    def test_getitem_returns_the_pickle_value_unchanged(self):
        returned = np.asarray([bool(_item(self.dataset, w)["need_scene"]) for w in self.picks])
        np.testing.assert_array_equal(returned, self.truth)
        # The test has power: the sample is not all-True, so an unconditional
        # override would fail here.
        self.assertGreater(int((~self.truth).sum()), 0.2 * SAMPLE_WINDOWS)


@unittest.skipIf(WORKER_EXPERT == "hoi", "HOI worker intentionally has no real LINGO assets")
@unittest.skipUnless(_data_available(), "data/dataset and data/train are required")
class NeedSceneGateForcedTests(unittest.TestCase):
    """``force_need_scene: true`` passed as a kwarg, exactly as
    ``hydra.utils.instantiate`` passes the resolved ``dataset:`` block."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = InfBaGelMixDataset(force_need_scene=True, **DATASET_KWARGS)
        cls.ids, cls.picks, cls.truth = _sample_indices(cls.dataset, SAMPLE_WINDOWS)

    @classmethod
    def tearDownClass(cls):
        del cls.dataset
        gc.collect()

    def test_the_kwarg_reaches_both_inner_datasets(self):
        """``InfBaGelMixDataset`` names none of its own parameters, so the only
        route is ``**kwargs`` -> both ``InfBaGelDataset(...)`` calls."""
        self.assertIs(self.dataset.lingo_dataset.force_need_scene, True)
        self.assertIs(self.dataset.omomo_dataset.force_need_scene, True)

    def test_every_training_window_is_scene_conditioned(self):
        returned = np.asarray([bool(_item(self.dataset, w)["need_scene"]) for w in self.picks])
        self.assertTrue(bool(returned.all()))
        # ...and the ones that moved are exactly the ones the pickle called
        # False, which is where the power is.
        self.assertGreater(int((~self.truth).sum()), 0.2 * SAMPLE_WINDOWS)

    def test_the_returned_object_still_collates_to_a_bool_tensor(self):
        """The trainer indexes this with a boolean mask at
        ``models/infbagel.py:1433``, so the collated dtype is load-bearing."""
        from torch.utils.data._utils.collate import default_collate

        self.addCleanup(setattr, self.dataset.lingo_dataset, "force_need_scene", True)
        false_side = [w for w, ns in zip(self.picks, self.truth) if not ns][:4]
        true_side = [w for w, ns in zip(self.picks, self.truth) if ns][:4]
        self.assertTrue(false_side and true_side)

        # np.bool_(True) rather than a Python bool, so the per-item type the
        # collate sees is the pickle's own.  Both spellings collate to
        # torch.bool, so this is conservatism rather than semantics -- but it
        # keeps the DeprecationWarning that torch already emits for np.bool_
        # scalars identical on both sides of the switch instead of one-sided.
        self.dataset.lingo_dataset.force_need_scene = False
        unforced = _item(self.dataset, false_side[0])["need_scene"]
        self.dataset.lingo_dataset.force_need_scene = True
        forced = _item(self.dataset, false_side[0])["need_scene"]
        self.assertIs(type(forced), type(unforced))
        self.assertIs(type(forced), np.bool_)
        self.assertTrue(bool(forced) and not bool(unforced))

        batch = default_collate([
            {"need_scene": _item(self.dataset, w)["need_scene"]}
            for w in list(false_side) + list(true_side)
        ])["need_scene"]
        self.assertEqual(batch.dtype, torch.bool)
        self.assertEqual(tuple(batch.shape), (len(false_side) + len(true_side),))
        self.assertTrue(bool(batch.all()))

    def test_no_other_field_changes(self):
        """The flag must move one key and nothing else.  Both sides come from the
        same object under the same RNG seed, so any difference is the flag's."""
        self.addCleanup(setattr, self.dataset.lingo_dataset, "force_need_scene", True)
        paired = [w for w, ns in zip(self.picks, self.truth) if not ns][:PAIRED_WINDOWS]
        self.assertGreaterEqual(len(paired), PAIRED_WINDOWS,
                                "need more pickle-False windows to have power")
        moved = {}
        for window in paired:
            self.dataset.lingo_dataset.force_need_scene = False
            off = _item(self.dataset, window)
            self.dataset.lingo_dataset.force_need_scene = True
            on = _item(self.dataset, window)
            self.assertEqual(set(off), set(on))
            for key in off:
                left, right = off[key], on[key]
                if isinstance(left, np.ndarray):
                    same = left.shape == right.shape and bool(
                        np.array_equal(left, right, equal_nan=True))
                elif torch.is_tensor(left):
                    same = bool(torch.equal(left, right))
                else:
                    same = bool(left == right)
                if not same:
                    moved[key] = moved.get(key, 0) + 1
        self.assertEqual(moved, {"need_scene": len(paired)},
                         "the flag moved a field other than need_scene")


class NeedSceneGatePlumbingTests(unittest.TestCase):
    """Data-free.  Runs on the HOI worker too, where the LINGO assets are absent
    by design, because the forwarding is what breaks silently under refactoring."""

    class _Stop(Exception):
        pass

    def _recorded(self, **extra):
        """The kwargs of both ``InfBaGelDataset(...)`` calls in
        ``InfBaGelMixDataset.__init__``, captured without touching the disk."""
        calls = []
        outer = self

        class Recorder:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 2:
                    raise outer._Stop()

        original = mix_module.InfBaGelDataset
        mix_module.InfBaGelDataset = Recorder
        try:
            with self.assertRaises(self._Stop):
                mix_module.InfBaGelMixDataset(
                    omomo_folder="/nonexistent", lingo_folder="/nonexistent",
                    device="cpu", mesh_grid=[0, 1, 0, 1, 0, 1], batch_size=1,
                    step=3, nb_voxels=[2, 2, 2], lingo_only=True, **extra)
        finally:
            mix_module.InfBaGelDataset = original
        self.assertEqual(len(calls), 2, "expected one OMOMO and one LINGO construction")
        return calls

    def test_the_mix_forwards_the_flag_to_both_inner_constructions(self):
        omomo_kwargs, lingo_kwargs = self._recorded(force_need_scene=True)
        self.assertIs(omomo_kwargs.get("force_need_scene"), True)
        self.assertIs(lingo_kwargs.get("force_need_scene"), True)

    def test_the_mix_does_not_invent_the_key_when_it_is_absent(self):
        """So an unset config cannot reach ``InfBaGelDataset`` as anything but
        the default."""
        for kwargs in self._recorded():
            self.assertNotIn("force_need_scene", kwargs)

    def test_the_read_defaults_to_false(self):
        """The literal default, from the AST rather than from a string search, so
        a comment mentioning ``True`` cannot satisfy it."""
        tree = ast.parse(DATASET_SOURCE.read_text())
        defaults = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "get" or not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "kwargs" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "force_need_scene":
                self.assertEqual(len(node.args), 2, "the read must carry an explicit default")
                self.assertIsInstance(node.args[1], ast.Constant)
                defaults.append(node.args[1].value)
        self.assertEqual(defaults, [False],
                         "exactly one kwargs.get('force_need_scene', False) is expected")


class NeedSceneGateInferenceInertnessTests(unittest.TestCase):
    """Data-free.  ``datasets/infbagel.py`` is on the evaluator's import path --
    ``_scene_only_dataset`` instantiates ``cfg.dataset`` and ``_lingo_item`` calls
    ``dataset.lingo_dataset[data_idx]`` -- so "the evaluator is untouched" is not
    a claim about files.  It is these two guards."""

    def test_the_model_side_gate_is_still_there(self):
        """Deleting the zeroing was the other way to implement the arm, and it
        would have deleted the evaluator's RDS null-scene pass with it: that pass
        *is* ``need_scene=False`` (test_infbagel_lingo_hsi.py:1848,
        payload ``rds.null_scene_mode``).  The arm must not take that route."""
        source = MODEL_SOURCE.read_text()
        self.assertIn("not_need_scene = torch.logical_not(need_scene)", source)
        for name in ("scene_emb", "scene_emb_0", "scene_emb_1", "scene_emb_2", "scene_emb_3"):
            self.assertRegex(source, rf"(?m)^\s*{re.escape(name)}\[not_need_scene\] = 0\.")
        self.assertNotIn("force_need_scene", source,
                         "the override belongs on the data side, not in the model")

    def test_the_trainer_passes_the_batch_value_straight_to_the_gate(self):
        """The last link in the chain the dataloader flip has to traverse:
        dataset -> collate -> ``b['need_scene']`` (train_infbagel.py:559) ->
        ``p_losses`` / ``consistency_loss`` -> the forward's gate.  Asserted from
        the AST, because a single rebinding anywhere in between would make the
        flag silently inert and no runtime test on the dataset could see it."""
        source = TRAINER_SOURCE.read_text()
        lines = source.splitlines()
        tree = ast.parse(source)
        stores = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Name) and node.id == "need_scene"
                  and isinstance(node.ctx, ast.Store)]
        self.assertEqual(len(stores), 1, "need_scene must be bound exactly once")
        binding = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.Assign) and stores[0] in ast.walk(node))
        text = "\n".join(lines[binding.lineno - 1:binding.end_lineno])
        self.assertIn("b['need_scene']", text.replace('"', "'"))
        # ...and it is then handed to both loss entry points unchanged.
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr in ("p_losses", "consistency_loss")]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertTrue(
                any(isinstance(arg, ast.Name) and arg.id == "need_scene"
                    for arg in call.args),
                "the batch's need_scene must reach the loss unchanged")

    def test_the_evaluator_pins_need_scene_itself(self):
        source = EVALUATOR_SOURCE.read_text()
        self.assertIn('"need_scene": torch.full((batch,), need_scene', source)
        for spelling in ('info["need_scene"]', "info['need_scene']"):
            self.assertNotIn(spelling, source,
                             "the evaluator must not read the dataset's need_scene")
        self.assertNotIn("force_need_scene", source)

    def test_the_hosi_sampler_pins_need_scene_itself(self):
        """``test_infbagel_hosi.py:17`` imports ``InfBaGelDataset`` and :478
        splats ``**cfg.dataset`` into it, so the patched file is on that import
        path too.  Inert there for the same two reasons: its config group is
        ``dataset: omomo_test``, which sets nothing (see the next test), and it
        overwrites the conditioning flag itself."""
        source = HOSI_SOURCE.read_text()
        self.assertIn("cond['need_scene'] = torch.ones((cfg.batch_size, ), dtype=torch.bool)",
                      source)
        self.assertNotIn("force_need_scene", source)

    def test_only_the_arm_config_sets_the_flag(self):
        """Every other yaml -- in particular every sampling config and
        dataset/lingo_v3_test.yaml -- leaves it unset, so it takes the default."""
        setters = sorted(
            path.relative_to(REPO).as_posix()
            for path in CONFIG_DIR.rglob("*.yaml")
            if "force_need_scene" in path.read_text()
        )
        self.assertEqual(setters, [ARM_CONFIG.relative_to(REPO).as_posix()])

    def test_the_arm_config_is_an_override_fragment_of_the_b_recipe(self):
        """``docs/EXPERIMENT_CONVENTIONS.md`` section 1: the delta only.  A copy of
        the base config would let the arm and its baseline drift apart silently on
        seed, layout, budget or objective."""
        from omegaconf import OmegaConf

        raw = OmegaConf.load(ARM_CONFIG)
        self.assertEqual(list(raw.defaults), ["config_train_hsi_b_lingo_full", "_self_"])
        self.assertEqual(raw.dataset.force_need_scene, True)
        self.assertEqual(raw.exp_name, "hsi_b_p16ns")
        # The delta is exactly these three keys plus the defaults list.
        self.assertEqual(
            sorted(raw.keys()), ["dataset", "defaults", "exp_name", "occ_permute_fix"]
        )
        self.assertIs(raw.occ_permute_fix, False)
        self.assertEqual(list(raw.dataset.keys()), ["force_need_scene"])


if __name__ == "__main__":
    unittest.main()
