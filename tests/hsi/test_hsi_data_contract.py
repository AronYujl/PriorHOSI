"""HSI dataset-contract tests.

These assertions are the Phase 1A HSI half of the pre-split
``tests/test_independent_priors.py``.  They also exist in
``tests/hoi/test_hoi_prior.py`` against ``priors.hoi.data``; this copy runs them
against ``priors.hsi.data`` so that ``phase/01c-hsi`` keeps a self-consistent
HSI side after it deletes ``code/priors/hoi/`` and ``tests/hoi/``.

The duplication is deliberate and branch-local: the two dataset implementations
are expected to diverge, while the contract they are checked against
(``priors.core.contracts.HSI_CONTRACT`` and the locked seed-42 scene-disjoint
LINGO split) is frozen in ``core`` and shared.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.core.contracts import HSI_CONTRACT
from priors.hsi.data import PriorWindowDataset, hsi_filter, partition_for_scenes


WORKER_EXPERT = os.environ.get("INFBAGEL_WORKER_EXPERT")
if WORKER_EXPERT not in {None, "hoi", "hsi"}:
    raise ValueError(f"invalid INFBAGEL_WORKER_EXPERT: {WORKER_EXPERT}")


class HSIDatasetContractTests(unittest.TestCase):
    def test_hsi_dynamic_object_filter_is_exact(self):
        actual = hsi_filter(np.array([-1, 3, -1, 2]), np.array([-1, -1, 4, 5]))
        np.testing.assert_array_equal(actual, np.array([True, False, False, False]))
        self.assertIn("seq_length <= 48", HSI_CONTRACT.filter_rule)

    def test_locked_split_has_no_family_or_scene_leakage(self):
        split = json.loads((REPO / HSI_CONTRACT.split).read_text())
        self.assertEqual(split["seed"], 42)
        self.assertFalse(set(split["train"]["scene_families"]) & set(split["validation"]["scene_families"]))
        self.assertFalse(set(split["train"]["scenes"]) & set(split["validation"]["scenes"]))
        scenes = np.asarray(split["train"]["scenes"][:2] + split["validation"]["scenes"][:2])
        sides = partition_for_scenes(split, scenes)
        np.testing.assert_array_equal(sides, np.asarray(["train", "train", "validation", "validation"], dtype=object))

    @unittest.skipIf(WORKER_EXPERT == "hoi", "HOI worker intentionally has no real LINGO assets")
    def test_real_hsi_item_exposes_only_authorized_conditions(self):
        hsi = PriorWindowDataset(str(REPO), "hsi", limit=1)[0]
        self.assertIn("scene_condition", hsi)
        self.assertNotIn("object_bps", hsi)
        self.assertEqual(float(hsi["x"][:, 216:].abs().max()), 0.0)


V2_SPLIT = Path("experiments/splits/lingo_scene_family_disjoint_v2_seed42.json")


class LingoMirrorDefectTests(unittest.TestCase):
    """Regression tests for the released-LINGO scene-label defect.

    The release labels every mirrored sequence ``005_mirror`` regardless of which
    of the 110 rooms it was captured in, so a label-disjoint split is content-
    duplicated across its own boundary.  ``scene-family-disjoint-v1`` is
    label-disjoint and therefore leaked every one of its 1,895 validation
    sequences; v2 repairs the labels before partitioning.  These assertions are
    what v1 lacked, so they must fail loudly if the repair ever regresses.
    """

    @classmethod
    def setUpClass(cls):
        cls.split = json.loads((REPO / V2_SPLIT).read_text())

    def test_every_mirror_pair_was_verified_not_sampled(self):
        verification = self.split["mirror_verification"]
        self.assertIs(verification["pairs_sampled"], False)
        self.assertGreater(verification["pairs_checked"], 0)
        for failure in (
            "length_equal_failures",
            "x_exactly_negated_failures",
            "yz_exactly_equal_failures",
        ):
            self.assertEqual(verification[failure], 0, failure)

    def test_relabel_is_unambiguous_and_actually_ran(self):
        relabel = self.split["mirror_relabel"]
        # A source label already ending in _mirror would make <src>_mirror ambiguous.
        self.assertEqual(relabel["source_labels_ending_in_mirror"], 0)
        self.assertGreater(relabel["relabelled_sequences"], 0)
        self.assertEqual(relabel["released_second_half_labels"], ["005_mirror"])

    def test_scene_labels_are_injective_with_respect_to_rooms(self):
        # The defect collapsed 110 rooms onto one label.  After the repair, every
        # label must map to a distinct occupancy grid.
        grids = self.split["scene_grid_verification"]
        self.assertEqual(grids["distinct_grid_sha256"], grids["labels_checked"])
        self.assertEqual(grids["grid_shape_axis0_reversal_failures"], 0)

    def test_three_way_partitions_are_disjoint_and_cover_every_family(self):
        sides = {name: self.split[name] for name in ("train", "validation", "test")}
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(set(sides[first]["scene_families"]) & set(sides[second]["scene_families"]))
            self.assertFalse(set(sides[first]["scenes"]) & set(sides[second]["scenes"]))
        covered = set().union(*(set(side["scene_families"]) for side in sides.values()))
        self.assertEqual(covered, set(self.split["scene_to_family"].values()))

    def test_held_out_sides_carry_no_mirrored_content(self):
        counts = self.split["counts"]
        assigned = counts["train_sequences"] + counts["validation_sequences"] + counts["test_sequences"]
        # Mirrors of held-out families are discarded outright; moving them to train
        # would recreate exactly the v1 leak.
        self.assertEqual(assigned + counts["discarded_mirror_sequences"], counts["sequences"])
        self.assertGreater(counts["discarded_mirror_sequences"], 0)

    def test_test_side_is_zero_shot_for_the_released_baseline(self):
        baseline = set(self.split["baseline_reference"]["selected_scenes"])
        self.assertTrue(baseline)
        self.assertFalse(baseline & set(self.split["test"]["scenes"]))

    @unittest.skipIf(WORKER_EXPERT == "hoi", "HOI worker intentionally has no real LINGO assets")
    def test_released_labels_really_do_collapse_the_mirrored_half(self):
        """Guard the premise itself, against the real dataset."""
        import pickle

        root = REPO / "data" / "dataset"
        with (root / "scene_name.pkl").open("rb") as handle:
            scene_names = np.asarray(pickle.load(handle))
        starts = np.load(root / "start_idx.npy")
        half = len(starts) // 2
        labels = scene_names[starts]
        self.assertEqual(set(labels[half:]), {"005_mirror"})
        self.assertGreater(len(set(labels[:half])), 100)
        transl = np.load(root / "transl_aligned.npy", mmap_mode="r")
        ends = np.load(root / "end_idx.npy")
        for index in (0, 7, half - 1):
            source = np.asarray(transl[starts[index] : ends[index] + 1])
            mirror = np.asarray(transl[starts[index + half] : ends[index + half] + 1])
            np.testing.assert_array_equal(source[:, 0], -mirror[:, 0])
            np.testing.assert_array_equal(source[:, 1:], mirror[:, 1:])



if __name__ == "__main__":
    unittest.main()
