"""Evaluation-time provenance for ``aggregate_metrics.json``.

``aggregate_metrics.json`` recorded no execution environment and no
evaluation-time commit at all.  Its only commit was ``checkpoint.git_commit``,
the *training* commit, so the branch's most-referenced baseline --
``p1-hoi-p8-eval-w3-guided-s42-20260809``, which ran 12:24:25-12:27:43 on
2026-08-09 while the commit holding its code, ``5e89644``, landed at 12:43:32 --
had no recorded code identity at all.  Answering "which code produced this
number?" then took a two-run, two-host re-execution instead of one file read.

Three properties are defended here, all of them silent if broken.

1. **Numerical inertness.**  The evaluator's outputs are compared bitwise against
   sealed baselines.  Provenance collection must consume no RNG and must happen
   strictly after the last metric is computed.
2. **The pinned per-sequence hash.**  ``per_sequence_metrics.json`` carries no
   absolute paths, so its hash is a valid cross-host readout, and
   ``bbcd9e1b550d42bf4ac19f9a55db4b9eebb896a8ddb2d562b5226a11b297f6b2`` is pinned
   by ``docs/phase_summaries/PHASE_1B_P11_ROOT_DETACH.md``,
   ``experiments/results/p1_hoi_p11_root_detach_s42_20260815.json``,
   ``experiments/results/p1_hoi_p10_geometry_repair_2x2_s42_20260810.json`` and
   ``experiments/registry.jsonl``.  Provenance is host-dependent by design and
   must therefore never enter that file.
3. **Never raising.**  A CPU-only host, an absent ``nvidia-smi`` and a non-Git
   directory must each yield null fields.  Losing a finished evaluation to a
   failed metadata read would be strictly worse than a null.

The module also pins the line anchor at 502: twenty tracked sites cite absolute
line numbers in the evaluator, and seven of them are append-only records under
``experiments/`` that cannot be edited to follow a shifted line.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import test_infbagel_hoi as evaluator


EVALUATOR_SOURCE = (REPO / "code" / "test_infbagel_hoi.py").read_text(encoding="utf-8")

PROVENANCE_MARKER = "# --- Execution provenance"


def _test_body():
    """Source of ``test()`` alone, excluding the appended provenance helpers.

    The helpers live below ``test()`` so that no cited line number shifts, so a
    naive slice to ``if __name__`` would also swallow them -- and
    ``_null_execution_provenance(`` contains ``_execution_provenance(`` as a
    substring, which would silently break every count below.
    """
    body = EVALUATOR_SOURCE.split("def test(cfg: DictConfig) -> None:", 1)[1]
    return body.split(PROVENANCE_MARKER, 1)[0]

EXPECTED_KEYS = [
    "cuda_version",
    "cudnn_version",
    "device",
    "git",
    "gpu_name",
    "hostname",
    "nvidia_driver_version",
    "python_version",
    "torch_version",
]

EXPECTED_GIT_KEYS = ["branch", "commit", "dirty", "dirty_entry_count", "resolved_at"]


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _make_repo(directory):
    """A one-commit Git repository, returning its HEAD."""
    _git(directory, "init", "--quiet")
    _git(directory, "config", "user.email", "test@example.invalid")
    _git(directory, "config", "user.name", "test")
    (directory / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(directory, "add", "tracked.txt")
    _git(directory, "commit", "--quiet", "-m", "initial")
    return _git(directory, "rev-parse", "HEAD")


class ProvenanceShapeTests(unittest.TestCase):
    def test_the_block_is_populated_on_this_host(self):
        record = evaluator._execution_provenance("cpu")
        self.assertEqual(sorted(record), EXPECTED_KEYS)
        self.assertEqual(sorted(record["git"]), EXPECTED_GIT_KEYS)
        for key in ("hostname", "torch_version", "python_version"):
            self.assertTrue(record[key], f"{key} should be populated on this host")
        self.assertRegex(record["git"]["commit"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(record["git"]["dirty"], bool)
        self.assertEqual(record["git"]["resolved_at"], "metrics_write")

    def test_the_device_is_recorded_as_a_string(self):
        self.assertEqual(evaluator._execution_provenance("cuda:0")["device"], "cuda:0")

    def test_the_null_skeleton_matches_the_populated_key_set(self):
        """A failed collection must not change the schema, only the values."""
        null = evaluator._null_execution_provenance()
        self.assertEqual(sorted(null), EXPECTED_KEYS)
        self.assertEqual(sorted(null["git"]), EXPECTED_GIT_KEYS)
        self.assertEqual(
            [value for key, value in null.items() if key != "git"],
            [None] * (len(EXPECTED_KEYS) - 1),
        )

    def test_the_null_skeleton_does_not_share_its_git_dict(self):
        """Two calls must not alias one mutable sub-dict."""
        first = evaluator._null_execution_provenance()
        second = evaluator._null_execution_provenance()
        first["git"]["commit"] = "sentinel"
        self.assertIsNone(second["git"]["commit"])


class NeverRaisesTests(unittest.TestCase):
    def test_a_missing_nvidia_smi_yields_a_null_driver(self):
        with mock.patch("subprocess.check_output", side_effect=FileNotFoundError):
            self.assertIsNone(evaluator._nvidia_driver_version())

    def test_a_failing_nvidia_smi_yields_a_null_driver(self):
        error = subprocess.CalledProcessError(9, ["nvidia-smi"])
        with mock.patch("subprocess.check_output", side_effect=error):
            self.assertIsNone(evaluator._nvidia_driver_version())

    def test_a_hanging_binary_is_bounded_by_a_timeout(self):
        """A hang is not an exception, and both output files open with mode 'x'.

        Wedged on a sick driver, an unbounded ``nvidia-smi`` would block after
        ``per_sequence_metrics.json`` was written but before the aggregate, and
        the half-finished directory would then also block a clean retry.
        """
        error = subprocess.TimeoutExpired(["nvidia-smi"], 30)
        with mock.patch("subprocess.check_output", side_effect=error):
            self.assertIsNone(evaluator._nvidia_driver_version())
            self.assertIsNone(evaluator._git_provenance(REPO)["commit"])
            record = evaluator._execution_provenance("cpu")
        self.assertEqual(sorted(record), EXPECTED_KEYS)

    def test_every_subprocess_call_passes_a_timeout(self):
        self.assertIn("timeout=30,", EVALUATOR_SOURCE)
        self.assertEqual(EVALUATOR_SOURCE.count("subprocess.check_output("), 1)

    def test_no_external_binary_at_all_still_returns_the_full_schema(self):
        with mock.patch("subprocess.check_output", side_effect=FileNotFoundError):
            record = evaluator._execution_provenance("cpu")
        self.assertEqual(sorted(record), EXPECTED_KEYS)
        self.assertIsNone(record["nvidia_driver_version"])
        self.assertIsNone(record["git"]["commit"])
        self.assertIsNone(record["git"]["dirty"])
        self.assertTrue(record["hostname"], "hostname does not need a subprocess")
        self.assertTrue(record["torch_version"])

    def test_cuda_unavailable_yields_a_null_gpu_name(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIsNone(evaluator._gpu_name("cuda:0"))
            record = evaluator._execution_provenance("cuda:0")
        self.assertIsNone(record["gpu_name"])
        self.assertEqual(record["device"], "cuda:0")

    def test_a_cpu_device_never_touches_cuda(self):
        self.assertIsNone(evaluator._gpu_name("cpu"))

    def test_a_non_git_directory_yields_a_null_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            record = evaluator._git_provenance(Path(directory))
        self.assertIsNone(record["commit"])
        self.assertIsNone(record["dirty"])
        self.assertIsNone(record["dirty_entry_count"])
        self.assertEqual(record["resolved_at"], "metrics_write")

    def test_every_field_failing_still_returns_the_full_schema(self):
        with mock.patch("socket.gethostname", side_effect=OSError), \
             mock.patch("platform.python_version", side_effect=RuntimeError), \
             mock.patch("subprocess.check_output", side_effect=FileNotFoundError):
            record = evaluator._execution_provenance("cpu")
        self.assertEqual(sorted(record), EXPECTED_KEYS)
        self.assertIsNone(record["hostname"])
        self.assertIsNone(record["python_version"])


class DirtyFlagTests(unittest.TestCase):
    """``dirty`` must mean what it means everywhere else in the repository."""

    def test_a_clean_repository_reports_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            head = _make_repo(path)
            record = evaluator._git_provenance(path)
        self.assertEqual(record["commit"], head)
        self.assertFalse(record["dirty"])
        self.assertEqual(record["dirty_entry_count"], 0)

    def test_an_untracked_file_counts_as_dirty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _make_repo(path)
            (path / "scratch.txt").write_text("untracked\n", encoding="utf-8")
            record = evaluator._git_provenance(path)
        self.assertTrue(record["dirty"])
        self.assertEqual(record["dirty_entry_count"], 1)

    def test_a_modified_tracked_file_counts_as_dirty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _make_repo(path)
            (path / "tracked.txt").write_text("modified\n", encoding="utf-8")
            record = evaluator._git_provenance(path)
        self.assertTrue(record["dirty"])
        self.assertEqual(record["dirty_entry_count"], 1)

    def test_the_flag_agrees_with_the_repositorys_own_porcelain(self):
        status = _git(REPO, "status", "--porcelain=v1", "--untracked-files=all")
        entries = status.splitlines()
        record = evaluator._git_provenance(REPO)
        self.assertEqual(record["dirty"], bool(entries))
        self.assertEqual(record["dirty_entry_count"], len(entries))

    def test_the_definition_matches_the_other_two_gates(self):
        """``experiment.py`` and the trainer use exactly these flags."""
        self.assertIn(
            "['git', 'status', '--porcelain=v1', '--untracked-files=all']",
            EVALUATOR_SOURCE,
        )


class NumericalInertnessTests(unittest.TestCase):
    def test_collection_consumes_no_rng(self):
        """The evaluator is compared bitwise against sealed baselines."""
        torch.manual_seed(42)
        expected = torch.rand(4).tolist()
        torch.manual_seed(42)
        evaluator._execution_provenance("cpu")
        self.assertEqual(torch.rand(4).tolist(), expected)

    def test_collection_happens_after_the_metrics_are_written(self):
        body = _test_body()
        self.assertEqual(body.count("_execution_provenance(device)"), 1)
        self.assertLess(body.index("compute_metrics("), body.index("_execution_provenance("))
        self.assertLess(
            body.index("with open(per_sequence_path, 'x')"),
            body.index("_execution_provenance("),
        )

    def test_the_sampling_loop_gained_no_synchronization(self):
        """``synchronize_cuda`` is a timing boundary, not a provenance hook.

        The five pre-existing calls are the run-start and end-to-end boundaries
        plus the per-step generation timing pair.
        """
        self.assertEqual(_test_body().count("synchronize_cuda(device)"), 5)


class PinnedPerSequenceHashTests(unittest.TestCase):
    """Provenance is host-dependent; the per-sequence file's hash is pinned."""

    def _per_sequence_payload(self):
        block = EVALUATOR_SOURCE.split("with open(per_sequence_path, 'x') as handle:", 1)[1]
        return block.split("evaluation_result = {", 1)[0]

    def test_the_per_sequence_payload_keys_are_exactly_the_original_four(self):
        payload = self._per_sequence_payload()
        keys = [line.strip() for line in payload.splitlines() if line.strip().startswith("'")]
        self.assertEqual(
            keys,
            [
                "'schema_version': 1,",
                "'seed': int(cfg.seed),",
                "'sequence_count': len(per_sequence_metrics),",
                "'metrics': per_sequence_metrics,",
            ],
        )

    def test_the_per_sequence_file_stays_at_schema_version_one(self):
        self.assertIn("'schema_version': 1,", self._per_sequence_payload())

    def test_provenance_never_reaches_the_per_sequence_file(self):
        self.assertNotIn("provenance", self._per_sequence_payload())
        self.assertNotIn("hostname", self._per_sequence_payload())


class AggregateSchemaTests(unittest.TestCase):
    def test_the_aggregate_declares_provenance_once_as_a_nested_key(self):
        block = EVALUATOR_SOURCE.split("evaluation_result = {", 1)[1]
        block = block.split("metrics_path = os.path.join", 1)[0]
        self.assertEqual(block.count("'execution_provenance': _execution_provenance(device),"), 1)
        for scattered in ("'hostname':", "'gpu_name':", "'nvidia_driver_version':"):
            self.assertNotIn(scattered, block)

    def test_the_aggregate_schema_version_is_bumped_to_two(self):
        """v1 means "no provenance recorded"; v2 means it is there."""
        block = EVALUATOR_SOURCE.split("evaluation_result = {", 1)[1]
        block = block.split("metrics_path = os.path.join", 1)[0]
        self.assertIn("'schema_version': 2,", block)


class SealedLineAnchorTests(unittest.TestCase):
    """Twenty tracked sites cite absolute line numbers in this module.

    Seven of them are append-only records under ``experiments/`` -- including
    ``p1_hoi_p11_root_detach_s42_20260815.json``, which says "FileNotFoundError
    at code/test_infbagel_hoi.py:502 for ../data/test/seq_id.pkl" -- and two of
    those files are hash-verified by ``tools/diagnose_hoi_d2p.py`` and
    ``tools/diagnose_hoi_d2f.py``.  None may be edited to follow a shifted line,
    so nothing may be inserted above the highest cited line, 625.
    """

    def test_line_502_still_loads_seq_id_pkl(self):
        line = EVALUATOR_SOURCE.splitlines()[501]
        self.assertIn("seq_id.pkl", line)

    def test_the_provenance_block_lives_below_the_highest_cited_line(self):
        lines = EVALUATOR_SOURCE.splitlines()
        first = next(
            index for index, line in enumerate(lines, start=1)
            if line.startswith("def _execution_provenance(")
        )
        self.assertGreater(first, 625)


if __name__ == "__main__":
    unittest.main()
