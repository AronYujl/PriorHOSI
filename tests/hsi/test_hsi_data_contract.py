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


if __name__ == "__main__":
    unittest.main()
