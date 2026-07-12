import json
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # Minimal governance checks run before ML dependencies are installed.
    np = None

from tools import chois_evaluator, experiment, make_lingo_split, run_chois_evaluator


class SplitTests(unittest.TestCase):
    def test_scene_variants_stay_together_and_split_is_deterministic(self):
        records = [
            {"sequence_id": "0", "scene_name": "room_a"},
            {"sequence_id": "1", "scene_name": "room_a_mirror"},
            {"sequence_id": "2", "scene_name": "room_b_new-loco"},
            {"sequence_id": "3", "scene_name": "room_b"},
            {"sequence_id": "4", "scene_name": "room_c_action_variant_2"},
            {"sequence_id": "5", "scene_name": "room_d"},
            {"sequence_id": "6", "scene_name": "room_e_aug_1"},
            {"sequence_id": "7", "scene_name": "088-take_shower"},
            {"sequence_id": "8", "scene_name": "088-wash"},
        ]
        first = make_lingo_split.build_split(records, 42, 0.2, {})
        second = make_lingo_split.build_split(records, 42, 0.2, {})
        self.assertEqual(first, second)
        self.assertEqual(first["scene_to_family"]["room_a_mirror"], "room_a")
        self.assertEqual(first["scene_to_family"]["room_b_new-loco"], "room_b")
        self.assertEqual(first["scene_to_family"]["088-take_shower"], "088")
        self.assertEqual(first["scene_to_family"]["088-wash"], "088")
        train = set(first["train"]["scene_families"])
        validation = set(first["validation"]["scene_families"])
        self.assertFalse(train & validation)

    def test_duplicate_sequence_is_rejected(self):
        records = [
            {"sequence_id": "0", "scene_name": "a"},
            {"sequence_id": "0", "scene_name": "b"},
        ]
        with self.assertRaises(make_lingo_split.SplitError):
            make_lingo_split.build_split(records, 42, 0.2, {})


class ManifestTests(unittest.TestCase):
    def test_run_id_binds_phase_and_seed(self):
        experiment.validate_run_id("p1-hoi-smoke-s42-20260711", "p1", 42)
        with self.assertRaises(experiment.ManifestError):
            experiment.validate_run_id("p1-hoi-smoke-s7-20260711", "p1", 42)

    def test_directory_hash_depends_on_names_and_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_text("same", encoding="utf-8")
            before = experiment.sha256_path(root)["sha256"]
            (root / "a").rename(root / "b")
            after = experiment.sha256_path(root)["sha256"]
            self.assertNotEqual(before, after)

    def test_registry_duplicate_is_rejected(self):
        record = {
            "schema_version": 1,
            "experiment_id": "p0-test-s42-20260711",
            "phase": "p0",
            "status": "failed",
            "hypothesis": "h",
            "config": {},
            "results": {},
            "conclusion": "c",
            "next_action": "n",
            "created_at": "2026-07-11T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.jsonl"
            line = json.dumps(record) + "\n"
            path.write_text(line + line, encoding="utf-8")
            with self.assertRaises(experiment.ManifestError):
                experiment.validate_registry(path)

    def test_split_validation_rejects_family_leakage(self):
        value = {
            "algorithm": "scene-family-disjoint-v1",
            "seed": 42,
            "validation_ratio": 0.2,
            "scene_to_family": {"a": "room"},
            "train": {"scene_families": ["room"], "scenes": ["a"]},
            "validation": {"scene_families": ["room"], "scenes": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(experiment.ManifestError):
                experiment.validate_split(path)


@unittest.skipIf(np is None, "numpy is not installed")
class ChoisEvaluatorTests(unittest.TestCase):
    def _write_npz(self, folder, name, frames=8):
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{name}.npz",
            seq_name=np.asarray(name),
            global_jpos=np.zeros((frames, 24, 3), dtype=np.float32),
        )

    def test_input_pair_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "pred"
            truth = root / "gt"
            self._write_npz(predictions, "sub1_chair_000")
            self._write_npz(truth, "sub1_chair_000")
            result = chois_evaluator.validate_pair(predictions, truth)
            self.assertEqual(result["sequence_count"], 1)

    def test_input_pair_rejects_mismatched_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "pred"
            truth = root / "gt"
            self._write_npz(predictions, "a")
            self._write_npz(truth, "b")
            with self.assertRaises(chois_evaluator.EvaluatorError):
                chois_evaluator.validate_pair(predictions, truth)

    def test_pinned_checkout_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "t2m_eval").mkdir(parents=True)
            evaluator = root / "t2m_eval" / "final_evaluations.py"
            evaluator.write_text("# evaluator\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "init"],
                cwd=root,
                check=True,
            )
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            config = {
                "upstream_commit": commit,
                "files": {"t2m_eval/final_evaluations.py": chois_evaluator.sha256_file(evaluator)},
            }
            result = chois_evaluator.verify_upstream(root, config)
            self.assertEqual(result["commit"], commit)

    def test_text_to_motion_checkout_rejects_unpinned_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "options").mkdir()
            module = root / "options" / "train_options.py"
            module.write_text("# parser\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "init"],
                cwd=root,
                check=True,
            )
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            config = {
                "upstream_commit": commit,
                "files": {"options/train_options.py": chois_evaluator.sha256_file(module)},
            }
            result = run_chois_evaluator.verify_text_to_motion(root, config)
            self.assertEqual(result["commit"], commit)
            module.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(run_chois_evaluator.AdapterError):
                run_chois_evaluator.verify_text_to_motion(root, config)


if __name__ == "__main__":
    unittest.main()
