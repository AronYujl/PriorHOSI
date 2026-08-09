#!/usr/bin/env python3
"""Tests for the tracked paired sequence-level bootstrap tool.

The P10 preregistration makes a decision rest on this tool, so the protocol
properties are asserted directly: name-based pairing, the single shared
resample index matrix, the b - a sign convention, determinism under seed 42,
and NaN semantics for sequences the evaluator does not score.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.paired_bootstrap import (  # noqa: E402
    PairedBootstrapError,
    discover_metrics,
    format_table,
    load_per_sequence,
    main,
    pair_sequence_names,
    paired_bootstrap,
    resolve_per_sequence_path,
    run,
)

TOOL = REPO_ROOT / "tools" / "paired_bootstrap.py"
PYTHON = sys.executable


def make_payload(metrics):
    return {
        "schema_version": 1,
        "seed": 42,
        "sequence_count": len(metrics),
        "metrics": metrics,
    }


def write_payload(directory, name, metrics):
    path = Path(directory) / name
    path.write_text(json.dumps(make_payload(metrics)), encoding="utf-8")
    return path


def linear_metrics(count, offset=0.0, scale=1.0, start=0.0):
    """Deterministic per-sequence records with a controllable constant offset."""
    return {
        "seq_{0:03d}".format(index): {
            "object_name": "box",
            "value": start + scale * index + offset,
        }
        for index in range(count)
    }


class ResolutionTest(unittest.TestCase):
    def test_explicit_file_path_is_used_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_payload(tmp, "per_sequence_metrics.json", linear_metrics(3))
            self.assertEqual(resolve_per_sequence_path(str(path)), path.resolve())

    def test_run_directory_and_run_id_both_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            evaluation = root / "my-run-s42-20260809" / "evaluation"
            evaluation.mkdir(parents=True)
            path = write_payload(evaluation, "per_sequence_metrics.json", linear_metrics(3))
            self.assertEqual(
                resolve_per_sequence_path(str(evaluation.parent)), path.resolve()
            )
            self.assertEqual(
                resolve_per_sequence_path("my-run-s42-20260809", results_root=root),
                path.resolve(),
            )

    def test_unresolvable_target_lists_what_was_tried(self):
        with self.assertRaises(PairedBootstrapError) as caught:
            resolve_per_sequence_path("no-such-run", results_root=Path("/nonexistent"))
        message = str(caught.exception)
        self.assertIn("cannot resolve", message)
        self.assertIn("tried:", message)
        self.assertIn("no-such-run", message)


class LoadValidationTest(unittest.TestCase):
    def test_missing_metrics_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            with self.assertRaises(PairedBootstrapError) as caught:
                load_per_sequence(path)
            self.assertIn("no 'metrics' key", str(caught.exception))

    def test_declared_sequence_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            payload = make_payload(linear_metrics(3))
            payload["sequence_count"] = 99
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PairedBootstrapError) as caught:
                load_per_sequence(path)
            self.assertIn("internally inconsistent", str(caught.exception))

    def test_malformed_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(PairedBootstrapError):
                load_per_sequence(path)

    def test_provenance_records_hash_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_payload(tmp, "m.json", linear_metrics(4))
            _, provenance = load_per_sequence(path)
            self.assertEqual(provenance["sequence_count"], 4)
            self.assertEqual(len(provenance["sha256"]), 64)
            self.assertEqual(provenance["evaluation_seed"], 42)


class PairingTest(unittest.TestCase):
    def test_pairing_is_by_name_not_position(self):
        a = {"z": {"value": 1.0}, "a": {"value": 3.0}}
        b = {"a": {"value": 4.0}, "z": {"value": 5.0}}
        names = pair_sequence_names(a, b)
        self.assertEqual(names, ["a", "z"])
        results, _, _ = paired_bootstrap(a, b, names, replicates=50)
        # Paired by name: a: 4-3=1, z: 5-1=4 -> mean 2.5.
        # Paired by position it would have been (4-1 + 5-3)/2 = 2.5 only by
        # accident, so use asymmetric values that separate the two.
        self.assertAlmostEqual(results["value"]["mean_delta"], 2.5)

    def test_positional_pairing_would_give_a_different_answer(self):
        a = {"z": {"value": 0.0}, "a": {"value": 10.0}}
        b = {"z": {"value": 1.0}, "a": {"value": 30.0}}
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=50)
        # by name: (30-10 + 1-0)/2 = 10.5
        self.assertAlmostEqual(results["value"]["mean_delta"], 10.5)

    def test_name_mismatch_raises_with_symmetric_difference(self):
        a = {"shared": {"value": 1.0}, "only_a": {"value": 2.0}}
        b = {"shared": {"value": 1.0}, "only_b": {"value": 3.0}}
        with self.assertRaises(PairedBootstrapError) as caught:
            pair_sequence_names(a, b, "runA", "runB")
        message = str(caught.exception)
        self.assertIn("sequence sets differ", message)
        self.assertIn("refusing to silently intersect", message)
        self.assertIn("only_a", message)
        self.assertIn("only_b", message)
        self.assertIn("runA", message)
        self.assertIn("runB", message)

    def test_mismatch_preview_is_bounded_but_reports_full_counts(self):
        a = {"s{0}".format(i): {"value": float(i)} for i in range(40)}
        b = {"t{0}".format(i): {"value": float(i)} for i in range(40)}
        with self.assertRaises(PairedBootstrapError) as caught:
            pair_sequence_names(a, b)
        message = str(caught.exception)
        self.assertIn("only in a (40)", message)
        self.assertIn("+30 more", message)

    def test_end_to_end_mismatch_surfaces_through_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_payload(tmp, "a.json", {"x": {"value": 1.0}})
            second = write_payload(tmp, "b.json", {"y": {"value": 1.0}})
            with self.assertRaises(PairedBootstrapError) as caught:
                run(str(first), str(second), expected_sequences=0)
            self.assertIn("sequence sets differ", str(caught.exception))


class KnownOffsetTest(unittest.TestCase):
    def test_constant_offset_ci_excludes_zero_and_brackets_truth(self):
        offset = 2.5
        a = linear_metrics(60)
        b = linear_metrics(60, offset=offset)
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=2000)
        entry = results["value"]
        self.assertAlmostEqual(entry["mean_delta"], offset)
        # A constant paired offset has zero resampling variance: every
        # replicate mean is exactly the offset, so the interval is degenerate
        # at the truth and still strictly excludes zero.
        self.assertAlmostEqual(entry["ci_low"], offset)
        self.assertAlmostEqual(entry["ci_high"], offset)
        self.assertLessEqual(entry["ci_low"], offset)
        self.assertGreaterEqual(entry["ci_high"], offset)
        self.assertTrue(entry["significant"])
        self.assertEqual(entry["direction"], "b_greater")

    def test_noisy_offset_ci_excludes_zero_and_brackets_truth(self):
        # The noise is centred exactly, so the realised sample mean delta IS
        # the true offset. Asserting coverage of an uncentred draw would be a
        # one-sample coverage test that legitimately fails ~5% of the time.
        rng = np.random.default_rng(7)
        offset = 1.0
        noise = rng.normal(0.0, 0.5, size=200)
        noise = noise - noise.mean()
        names = ["seq_{0:03d}".format(index) for index in range(200)]
        a = {name: {"value": float(rng.normal(10.0, 3.0))} for name in names}
        b = {name: {"value": a[name]["value"] + offset + float(noise[index])}
             for index, name in enumerate(names)}
        paired = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, paired, replicates=4000)
        entry = results["value"]
        self.assertAlmostEqual(entry["mean_delta"], offset, places=9)
        self.assertTrue(entry["significant"])
        self.assertEqual(entry["direction"], "b_greater")
        # Non-degenerate interval that strictly brackets the truth.
        self.assertLess(entry["ci_low"], entry["ci_high"])
        self.assertLess(entry["ci_low"], offset)
        self.assertGreater(entry["ci_high"], offset)
        self.assertGreater(entry["ci_low"], 0.0)

    def test_negative_offset_is_reported_as_b_lower(self):
        a = linear_metrics(40)
        b = linear_metrics(40, offset=-3.0)
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=500)
        entry = results["value"]
        self.assertAlmostEqual(entry["mean_delta"], -3.0)
        self.assertTrue(entry["significant"])
        self.assertEqual(entry["direction"], "b_lower")

    def test_delta_convention_is_b_minus_a(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_payload(tmp, "a.json", linear_metrics(20, offset=0.0))
            second = write_payload(tmp, "b.json", linear_metrics(20, offset=4.0))
            report = run(str(first), str(second), expected_sequences=0,
                         label_a="A", label_b="B")
            self.assertEqual(report["delta_convention"], "b_minus_a")
            self.assertIn("b - a", report["delta_description"])
            self.assertAlmostEqual(report["metrics"]["value"]["mean_delta"], 4.0)
            self.assertAlmostEqual(report["metrics"]["value"]["mean_a"],
                                   report["metrics"]["value"]["mean_b"] - 4.0)


class NullCaseTest(unittest.TestCase):
    def test_identical_inputs_give_exactly_zero_and_a_degenerate_ci(self):
        metrics = linear_metrics(50)
        names = pair_sequence_names(metrics, metrics)
        results, _, _ = paired_bootstrap(metrics, metrics, names, replicates=1000)
        entry = results["value"]
        self.assertEqual(entry["mean_delta"], 0.0)
        self.assertEqual(entry["ci_low"], 0.0)
        self.assertEqual(entry["ci_high"], 0.0)
        self.assertFalse(entry["significant"])
        self.assertEqual(entry["direction"], "inconclusive")

    def test_degenerate_ci_at_zero_is_not_significant(self):
        # A CI whose endpoints are both exactly zero touches zero, so the
        # "strictly excludes zero" rule must return False.
        metrics = {"a": {"value": 1.0}, "b": {"value": 2.0}}
        names = pair_sequence_names(metrics, metrics)
        results, _, _ = paired_bootstrap(metrics, metrics, names, replicates=100)
        self.assertFalse(results["value"]["significant"])

    def test_same_file_on_both_sides_warns_and_yields_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_payload(tmp, "m.json", linear_metrics(10))
            warnings = []
            report = run(str(path), str(path), expected_sequences=0,
                         warn=warnings.append)
            self.assertTrue(any("same file" in text for text in warnings))
            self.assertEqual(report["metrics"]["value"]["mean_delta"], 0.0)


class SharedIndexTest(unittest.TestCase):
    def test_duplicated_metrics_receive_identical_intervals(self):
        rng = np.random.default_rng(11)
        a, b = {}, {}
        for index in range(120):
            name = "seq_{0:03d}".format(index)
            left = float(rng.normal(5.0, 2.0))
            right = left + float(rng.normal(0.7, 1.5))
            a[name] = {"metric_one": left, "metric_one_copy": left,
                       "unrelated": float(rng.normal(0.0, 1.0))}
            b[name] = {"metric_one": right, "metric_one_copy": right,
                       "unrelated": float(rng.normal(0.0, 1.0))}
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=3000)
        first = results["metric_one"]
        copy = results["metric_one_copy"]
        for key in ("mean_a", "mean_b", "mean_delta", "ci_low", "ci_high",
                    "significant", "direction"):
            self.assertEqual(first[key], copy[key], msg="differs on " + key)
        # Not a vacuous check: the third metric is genuinely different.
        self.assertNotEqual(first["ci_low"], results["unrelated"]["ci_low"])

    def test_scaled_copy_scales_the_interval_exactly(self):
        # A shared index matrix makes the bootstrap a linear functional of the
        # delta vector, so an exactly scaled metric yields an exactly scaled
        # interval. Freshly drawn per-metric indices would not do this.
        rng = np.random.default_rng(13)
        a, b = {}, {}
        for index in range(100):
            name = "seq_{0:03d}".format(index)
            left = float(rng.normal(0.0, 1.0))
            right = left + float(rng.normal(0.3, 1.0))
            a[name] = {"base": left, "scaled": left * 10.0}
            b[name] = {"base": right, "scaled": right * 10.0}
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=2000)
        for key in ("mean_delta", "ci_low", "ci_high"):
            self.assertAlmostEqual(results["scaled"][key],
                                   results["base"][key] * 10.0, places=9)

    def test_index_digest_is_reported_and_seed_dependent(self):
        a = linear_metrics(30)
        b = linear_metrics(30, offset=1.0)
        names = pair_sequence_names(a, b)
        _, _, digest_seed_42 = paired_bootstrap(a, b, names, seed=42, replicates=100)
        _, _, digest_again = paired_bootstrap(a, b, names, seed=42, replicates=100)
        _, _, digest_seed_43 = paired_bootstrap(a, b, names, seed=43, replicates=100)
        self.assertEqual(digest_seed_42, digest_again)
        self.assertNotEqual(digest_seed_42, digest_seed_43)
        self.assertEqual(len(digest_seed_42), 64)


class NanHandlingTest(unittest.TestCase):
    def test_null_on_either_side_drops_only_that_pair(self):
        a = {
            "s0": {"value": 1.0, "always": 1.0},
            "s1": {"value": None, "always": 1.0},
            "s2": {"value": 3.0, "always": 1.0},
            "s3": {"value": 4.0, "always": 1.0},
        }
        b = {
            "s0": {"value": 2.0, "always": 2.0},
            "s1": {"value": 9.0, "always": 2.0},
            "s2": {"value": None, "always": 2.0},
            "s3": {"value": 5.0, "always": 2.0},
        }
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=500)
        entry = results["value"]
        self.assertEqual(entry["n_pairs"], 4)
        self.assertEqual(entry["n_pairs_used"], 2)
        self.assertEqual(entry["n_pairs_dropped_nonfinite"], 2)
        self.assertAlmostEqual(entry["mean_delta"], 1.0)
        self.assertEqual(results["always"]["n_pairs_used"], 4)

    def test_nan_means_use_only_paired_observations(self):
        # mean_a and mean_b are nanmeans of their own columns; a sequence with
        # a null on one side still contributes to the other side's column mean.
        a = {"s0": {"value": 1.0}, "s1": {"value": None}}
        b = {"s0": {"value": 4.0}, "s1": {"value": 10.0}}
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=100)
        entry = results["value"]
        self.assertAlmostEqual(entry["mean_a"], 1.0)
        self.assertAlmostEqual(entry["mean_b"], 7.0)
        self.assertAlmostEqual(entry["mean_delta"], 3.0)

    def test_all_null_metric_is_reported_as_undefined_not_crashed(self):
        a = {"s0": {"value": None, "ok": 1.0}, "s1": {"value": None, "ok": 1.0}}
        b = {"s0": {"value": None, "ok": 2.0}, "s1": {"value": None, "ok": 2.0}}
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=200)
        entry = results["value"]
        self.assertEqual(entry["n_pairs_used"], 0)
        self.assertIsNone(entry["mean_delta"])
        self.assertIsNone(entry["ci_low"])
        self.assertIsNone(entry["significant"])
        self.assertEqual(entry["direction"], "undefined")
        self.assertIn("note", entry)
        self.assertTrue(results["ok"]["significant"])

    def test_disjoint_nulls_leave_no_usable_pair(self):
        a = {"s0": {"value": 1.0}, "s1": {"value": None}}
        b = {"s0": {"value": None}, "s1": {"value": 2.0}}
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=100)
        self.assertEqual(results["value"]["n_pairs_used"], 0)
        self.assertEqual(results["value"]["direction"], "undefined")

    def test_infinite_values_are_treated_as_missing(self):
        a = {"s0": {"value": 1.0}, "s1": {"value": 2.0}, "s2": {"value": 3.0}}
        b = {"s0": {"value": float("inf")}, "s1": {"value": 4.0},
             "s2": {"value": 5.0}}
        names = pair_sequence_names(a, b)
        results, _, _ = paired_bootstrap(a, b, names, replicates=500)
        entry = results["value"]
        self.assertEqual(entry["n_pairs_used"], 2)
        self.assertAlmostEqual(entry["mean_delta"], 2.0)
        self.assertTrue(np.isfinite(entry["ci_low"]))

    def test_json_nan_is_treated_as_missing_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.json"
            second = Path(tmp) / "b.json"
            # json.dump emits bare NaN by default, which json.load accepts.
            first.write_text(json.dumps(make_payload(
                {"s0": {"value": 1.0}, "s1": {"value": float("nan")}}
            )), encoding="utf-8")
            second.write_text(json.dumps(make_payload(
                {"s0": {"value": 3.0}, "s1": {"value": 9.0}}
            )), encoding="utf-8")
            report = run(str(first), str(second), expected_sequences=0)
            entry = report["metrics"]["value"]
            self.assertEqual(entry["n_pairs_used"], 1)
            self.assertAlmostEqual(entry["mean_delta"], 2.0)


class MetricCoverageTest(unittest.TestCase):
    def test_string_metrics_are_excluded_and_reported(self):
        a = {"s0": {"object_name": "box", "value": 1.0}}
        b = {"s0": {"object_name": "box", "value": 2.0}}
        coverage = discover_metrics(a, b, ["s0"])
        self.assertEqual(coverage["analyzed"], ["value"])
        self.assertEqual(coverage["excluded_non_numeric"], ["object_name"])

    def test_booleans_are_not_treated_as_numbers(self):
        a = {"s0": {"flag": True, "value": 1.0}}
        b = {"s0": {"flag": False, "value": 2.0}}
        coverage = discover_metrics(a, b, ["s0"])
        self.assertEqual(coverage["analyzed"], ["value"])
        self.assertIn("flag", coverage["excluded_non_numeric"])

    def test_one_sided_metrics_are_reported_not_silently_dropped(self):
        a = {"s0": {"shared": 1.0, "only_a": 1.0}}
        b = {"s0": {"shared": 2.0, "only_b": 1.0}}
        coverage = discover_metrics(a, b, ["s0"])
        self.assertEqual(coverage["analyzed"], ["shared"])
        self.assertEqual(coverage["only_in_a"], ["only_a"])
        self.assertEqual(coverage["only_in_b"], ["only_b"])

    def test_run_warns_about_one_sided_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_payload(tmp, "a.json", {"s0": {"shared": 1.0, "extra": 1.0}})
            second = write_payload(tmp, "b.json", {"s0": {"shared": 2.0}})
            warnings = []
            run(str(first), str(second), expected_sequences=0, warn=warnings.append)
            self.assertTrue(any("extra" in text for text in warnings))

    def test_every_shared_metric_is_analysed(self):
        a = {"s0": {"m1": 1.0, "m2": 2.0, "m3": 3.0}}
        b = {"s0": {"m1": 2.0, "m2": 4.0, "m3": 6.0}}
        report = paired_bootstrap(a, b, ["s0"], replicates=50)[0]
        self.assertEqual(sorted(report), ["m1", "m2", "m3"])


class ExpectedCountTest(unittest.TestCase):
    def test_wrong_count_warns_but_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_payload(tmp, "a.json", linear_metrics(7))
            second = write_payload(tmp, "b.json", linear_metrics(7, offset=1.0))
            warnings = []
            report = run(str(first), str(second), expected_sequences=438,
                         warn=warnings.append)
            self.assertTrue(any("expected 438" in text for text in warnings))
            self.assertFalse(report["sequences"]["matches_expected"])
            self.assertEqual(report["sequences"]["paired_count"], 7)

    def test_matching_count_does_not_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_payload(tmp, "a.json", linear_metrics(7))
            second = write_payload(tmp, "b.json", linear_metrics(7, offset=1.0))
            warnings = []
            report = run(str(first), str(second), expected_sequences=7,
                         warn=warnings.append)
            self.assertEqual(warnings, [])
            self.assertTrue(report["sequences"]["matches_expected"])

    def test_zero_disables_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_payload(tmp, "a.json", linear_metrics(7))
            second = write_payload(tmp, "b.json", linear_metrics(7, offset=1.0))
            report = run(str(first), str(second), expected_sequences=0)
            self.assertIsNone(report["sequences"]["matches_expected"])


class DeterminismTest(unittest.TestCase):
    def test_two_cli_runs_with_seed_42_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(3)
            a, b = {}, {}
            for index in range(150):
                name = "seq_{0:03d}".format(index)
                left = float(rng.normal(2.0, 1.0))
                a[name] = {"object_name": "box", "m": left,
                           "n": float(rng.normal(0.0, 1.0))}
                b[name] = {"object_name": "box", "m": left + float(rng.normal(0.4, 1.0)),
                           "n": float(rng.normal(0.0, 1.0))}
            first = write_payload(tmp, "a.json", a)
            second = write_payload(tmp, "b.json", b)
            command = [PYTHON, str(TOOL), "--a", str(first), "--b", str(second),
                       "--seed", "42", "--replicates", "2000",
                       "--expected-sequences", "0"]
            one = subprocess.run(command, capture_output=True, check=True)
            two = subprocess.run(command, capture_output=True, check=True)
            self.assertEqual(one.stdout, two.stdout)
            self.assertEqual(one.stderr, two.stderr)
            self.assertGreater(len(one.stdout), 0)
            json.loads(one.stdout.decode("utf-8"))

    def test_api_repeats_bit_for_bit_and_seed_changes_the_interval(self):
        rng = np.random.default_rng(5)
        a, b = {}, {}
        for index in range(120):
            name = "seq_{0:03d}".format(index)
            left = float(rng.normal(1.0, 2.0))
            a[name] = {"value": left}
            b[name] = {"value": left + float(rng.normal(0.5, 2.0))}
        names = pair_sequence_names(a, b)
        first = paired_bootstrap(a, b, names, seed=42, replicates=2000)[0]
        second = paired_bootstrap(a, b, names, seed=42, replicates=2000)[0]
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))
        other = paired_bootstrap(a, b, names, seed=1234, replicates=2000)[0]
        self.assertEqual(first["value"]["mean_delta"], other["value"]["mean_delta"])
        self.assertNotEqual(first["value"]["ci_low"], other["value"]["ci_low"])

    def test_insertion_order_does_not_change_the_result(self):
        rng = np.random.default_rng(17)
        base = {}
        for index in range(80):
            name = "seq_{0:03d}".format(index)
            base[name] = float(rng.normal(0.0, 1.0))
        a = {name: {"value": value} for name, value in base.items()}
        b = {name: {"value": value + 0.5} for name, value in base.items()}
        shuffled_a = {name: a[name] for name in reversed(list(a))}
        shuffled_b = {name: b[name] for name in sorted(b, key=lambda n: n[::-1])}
        forward = paired_bootstrap(a, b, pair_sequence_names(a, b), replicates=500)[0]
        reordered = paired_bootstrap(
            shuffled_a, shuffled_b,
            pair_sequence_names(shuffled_a, shuffled_b), replicates=500)[0]
        self.assertEqual(json.dumps(forward, sort_keys=True),
                         json.dumps(reordered, sort_keys=True))


class CliTest(unittest.TestCase):
    def _pair(self, tmp):
        a = linear_metrics(30)
        b = linear_metrics(30, offset=2.0)
        return write_payload(tmp, "a.json", a), write_payload(tmp, "b.json", b)

    def test_output_file_is_written_and_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = self._pair(tmp)
            out = Path(tmp) / "nested" / "report.json"
            code = main(["--a", str(first), "--b", str(second), "--output", str(out),
                         "--expected-sequences", "0", "--replicates", "200"])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["metrics"]["value"]["mean_delta"], 2.0)
            code = main(["--a", str(first), "--b", str(second), "--output", str(out),
                         "--expected-sequences", "0", "--replicates", "200"])
            self.assertEqual(code, 2)
            code = main(["--a", str(first), "--b", str(second), "--output", str(out),
                         "--expected-sequences", "0", "--replicates", "200",
                         "--overwrite"])
            self.assertEqual(code, 0)

    def test_unresolvable_input_exits_two_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, _ = self._pair(tmp)
            code = main(["--a", str(first), "--b", "no-such-run",
                         "--results-root", tmp, "--expected-sequences", "0"])
            self.assertEqual(code, 2)

    def test_table_states_the_convention_and_marks_significance(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = self._pair(tmp)
            report = run(str(first), str(second), expected_sequences=0,
                         label_a="H0", label_b="W3", replicates=500)
            table = format_table(report)
            self.assertIn("a = H0", table)
            self.assertIn("b = W3", table)
            self.assertIn("b - a", table)
            self.assertIn("delta(b-a)", table)
            self.assertIn("YES", table)
            self.assertIn("shared resample index matrix", table)

    def test_report_records_protocol_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = self._pair(tmp)
            report = run(str(first), str(second), expected_sequences=0,
                         replicates=777, seed=99)
            self.assertEqual(report["bootstrap"]["replicates"], 777)
            self.assertEqual(report["bootstrap"]["seed"], 99)
            self.assertEqual(report["bootstrap"]["percentiles"], [2.5, 97.5])
            self.assertTrue(report["bootstrap"]["shared_resample_index_matrix"])
            self.assertEqual(len(report["bootstrap"]["resample_index_sha256"]), 64)
            self.assertEqual(len(report["inputs"]["a"]["sha256"]), 64)
            self.assertNotEqual(report["inputs"]["a"]["sha256"],
                                report["inputs"]["b"]["sha256"])

    def test_rejects_nonpositive_replicates(self):
        a = linear_metrics(5)
        b = linear_metrics(5, offset=1.0)
        with self.assertRaises(PairedBootstrapError):
            paired_bootstrap(a, b, pair_sequence_names(a, b), replicates=0)


class RealRunSmokeTest(unittest.TestCase):
    """Exercise the tool on the two sealed P8 evaluations when present."""

    A = REPO_ROOT / "results/experiments/p1-hoi-p8-eval-h0-guided-s42-20260806"
    B = REPO_ROOT / "results/experiments/p1-hoi-p8-eval-w3-guided-s42-20260809"

    def test_real_p8_pair_produces_438_paired_sequences(self):
        for path in (self.A, self.B):
            if not (path / "evaluation" / "per_sequence_metrics.json").is_file():
                self.skipTest("sealed evaluation outputs are not present")
        report = run(str(self.A), str(self.B), label_a="H0", label_b="W3",
                     replicates=200)
        self.assertEqual(report["sequences"]["paired_count"], 438)
        self.assertTrue(report["sequences"]["matches_expected"])
        self.assertIn("contact_f1", report["metrics"])
        self.assertIn("object_name", report["metric_coverage"]["excluded_non_numeric"])
        # contact_percent is aggregate-only; it must not silently appear.
        self.assertNotIn("contact_percent", report["metrics"])
        # The penetration metrics are scored on a subset of the cohort.
        self.assertEqual(report["metrics"]["hand_pen_loss_omomo"]["n_pairs_used"], 181)
        self.assertEqual(report["metrics"]["contact_f1"]["n_pairs_used"], 438)


if __name__ == "__main__":
    unittest.main()
