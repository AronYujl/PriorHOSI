import unittest
from types import SimpleNamespace

import numpy as np
import torch

from priors.remediation import (
    deterministic_derangement,
    field_squared_error,
    select_internal_triples,
    select_teacher_windows,
)


class RemediationDiagnosticTest(unittest.TestCase):
    def _dataset(self):
        names = np.asarray(("seq_a", "seq_b"), dtype=object)
        sequence_ids = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int64)
        return SimpleNamespace(
            partition="internal_validation",
            indices=np.arange(6),
            sequence_ids=sequence_ids,
            scene_names=names,
            language={"pi": np.asarray((0, 42, 84, 0, 42, 84), dtype=np.int64)},
        )

    def test_selection_is_internal_only_and_deterministic(self):
        dataset = self._dataset()
        self.assertEqual(select_internal_triples(dataset, 2), select_internal_triples(dataset, 2))
        self.assertEqual(len(select_teacher_windows(dataset, 4)), 4)
        dataset.partition = "test"
        with self.assertRaisesRegex(ValueError, "internal-validation only"):
            select_internal_triples(dataset, 1)

    def test_derangement(self):
        permutation = deterministic_derangement(8)
        self.assertTrue(torch.all(permutation != torch.arange(8)))
        self.assertEqual(sorted(permutation.tolist()), list(range(8)))

    def test_fieldwise_error_excludes_history(self):
        target = torch.zeros(2, 16, 232)
        prediction = target.clone()
        prediction[:, :2] = 100.0
        errors = field_squared_error(prediction, target)
        self.assertTrue(all(float(value) == 0.0 for value in errors.values()))


if __name__ == "__main__":
    unittest.main()
