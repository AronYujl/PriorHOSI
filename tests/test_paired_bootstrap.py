#!/usr/bin/env python3
"""Tests for the tracked paired sequence-level bootstrap tool.

The P10 preregistration makes a decision rest on this tool, so the protocol
properties are asserted directly: name-based pairing, the single shared
resample index matrix, the b - a sign convention, determinism under seed 42,
and NaN semantics for sequences the evaluator does not score.

P10 is a 2x2 factorial, so the factorial mode is held to the same standard plus
three properties a pairwise tool cannot have: main effects and the interaction
must recover a planted truth in closed form, the two algebraic forms of the
interaction must agree, and every contrast must ride the same shared resample
index matrix so the intervals are mutually comparable.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.paired_bootstrap import (  # noqa: E402
    CELL_KEYS,
    PairedBootstrapError,
    discover_metrics,
    discover_metrics_across,
    factorial_bootstrap,
    format_factorial_table,
    format_table,
    load_per_sequence,
    main,
    pair_sequence_names,
    pair_sequence_names_across,
    paired_bootstrap,
    resolve_per_sequence_path,
    run,
    run_factorial,
)
import tools.paired_bootstrap as paired_bootstrap_module  # noqa: E402

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


FACTOR1 = "hinge"
FACTOR2 = "detach"
MAIN1_KEY = "main_effect_" + FACTOR1
MAIN2_KEY = "main_effect_" + FACTOR2
INTERACTION_KEY = "interaction_{0}_x_{1}".format(FACTOR1, FACTOR2)
ALL_CONTRAST_KEYS = (
    "a00_minus_a00", "a10_minus_a00", "a01_minus_a00", "a11_minus_a00",
    "a11_minus_a01", "a11_minus_a10", MAIN1_KEY, MAIN2_KEY, INTERACTION_KEY,
)


def centred_noise(count, scale, seed):
    """Noise whose sample mean is exactly zero.

    The realised sample effect is then exactly the planted effect, so asserting
    that an interval brackets the truth is a statement about the tool and not a
    one-sample coverage test that legitimately fails ~5% of the time. This is
    the same device the pairwise noisy-offset test uses.
    """
    noise = np.random.default_rng(seed).normal(0.0, scale, size=count)
    return noise - noise.mean()


def planted_cells(count=200, hinge=2.0, detach=-1.5, interaction=0.75,
                  noise=0.4, seed=101, exact=None):
    """Four cells whose main effects and interaction are known in closed form.

    ``a00 = v``, ``a10 = v + H``, ``a01 = v + D``, ``a11 = v + H + D + I``,
    each of the three treated cells carrying its own centred noise. Because the
    noise means are exactly zero the realised sample values are exactly::

        a10 - a00         = H              a11 - a01 = H + I
        a01 - a00         = D              a11 - a10 = D + I
        a11 - a00         = H + D + I
        main effect of F1 = H + I/2
        main effect of F2 = D + I/2
        interaction       = I

    ``exact`` (default: on whenever ``noise`` is zero) builds the base column
    from small integers, so with dyadic offsets every cell value is exactly
    representable and the contrasts come out bit-exact. That matters for the
    degenerate-interval assertions: with a Gaussian base, ``(v+H+D) - (v+D)``
    is only ``H`` to within a rounding error, and a "zero" interaction lands
    at ~1e-18 rather than at 0.0. See
    ``test_a_float_noise_floor_is_reported_honestly``.
    """
    exact = (noise == 0) if exact is None else exact
    if exact:
        base = np.arange(count, dtype=np.float64)
    else:
        base = np.random.default_rng(seed).normal(0.0, 3.0, size=count)
    zero = np.zeros(count)
    columns = {
        "a00": base,
        "a10": base + hinge + (centred_noise(count, noise, seed + 1) if noise else zero),
        "a01": base + detach + (centred_noise(count, noise, seed + 2) if noise else zero),
        "a11": base + hinge + detach + interaction
               + (centred_noise(count, noise, seed + 3) if noise else zero),
    }
    names = ["seq_{0:03d}".format(index) for index in range(count)]
    cells = {
        key: {name: {"object_name": "box", "value": float(column[index])}
              for index, name in enumerate(names)}
        for key, column in columns.items()
    }
    truth = {
        "a00_minus_a00": 0.0,
        "a10_minus_a00": hinge,
        "a01_minus_a00": detach,
        "a11_minus_a00": hinge + detach + interaction,
        "a11_minus_a01": hinge + interaction,
        "a11_minus_a10": detach + interaction,
        MAIN1_KEY: hinge + interaction / 2.0,
        MAIN2_KEY: detach + interaction / 2.0,
        INTERACTION_KEY: interaction,
    }
    return cells, truth


def bootstrap_cells(cells, **kwargs):
    kwargs.setdefault("replicates", 3000)
    kwargs.setdefault("factor1_name", FACTOR1)
    kwargs.setdefault("factor2_name", FACTOR2)
    names = pair_sequence_names_across(cells)
    return factorial_bootstrap(cells, names, **kwargs)


def write_cells(directory, cells):
    return {key: str(write_payload(directory, key + ".json", metrics))
            for key, metrics in cells.items()}


class FactorialPairingTest(unittest.TestCase):
    def test_pairing_is_by_name_not_position_across_four_cells(self):
        # Every cell lists the same two sequences in a different order; only
        # name-based pairing recovers the planted +1 / +2 / +3 offsets.
        cells = {
            "a00": {"z": {"value": 1.0}, "a": {"value": 10.0}},
            "a10": {"a": {"value": 11.0}, "z": {"value": 2.0}},
            "a01": {"z": {"value": 3.0}, "a": {"value": 12.0}},
            "a11": {"a": {"value": 14.0}, "z": {"value": 5.0}},
        }
        self.assertEqual(pair_sequence_names_across(cells), ["a", "z"])
        contrasts = bootstrap_cells(cells, replicates=200)[0]
        self.assertAlmostEqual(
            contrasts["a10_minus_a00"]["metrics"]["value"]["difference"], 1.0)
        self.assertAlmostEqual(
            contrasts["a01_minus_a00"]["metrics"]["value"]["difference"], 2.0)
        self.assertAlmostEqual(
            contrasts[INTERACTION_KEY]["metrics"]["value"]["difference"], 1.0)

    def test_mismatch_in_any_of_the_four_cells_raises(self):
        for offender in CELL_KEYS:
            cells, _ = planted_cells(count=6, noise=0.0)
            broken = dict(cells)
            metrics = dict(cells[offender])
            del metrics["seq_002"]
            metrics["only_in_" + offender] = {"value": 1.0}
            broken[offender] = metrics
            with self.assertRaises(PairedBootstrapError) as caught:
                pair_sequence_names_across(broken)
            message = str(caught.exception)
            self.assertIn("sequence sets differ across 4 runs", message)
            self.assertIn("refusing to silently intersect", message)
            # Both halves of the symmetric difference are reported.
            self.assertIn("seq_002", message)
            self.assertIn("only_in_" + offender, message)
            self.assertIn(offender, message)

    def test_mismatch_names_the_offending_cell_label(self):
        cells, _ = planted_cells(count=4, noise=0.0)
        broken = dict(cells)
        broken["a11"] = {name: record for name, record in cells["a11"].items()
                         if name != "seq_001"}
        labels = {key: key.upper() + "-run" for key in CELL_KEYS}
        with self.assertRaises(PairedBootstrapError) as caught:
            pair_sequence_names_across(broken, labels)
        message = str(caught.exception)
        self.assertIn("A11-run", message)
        self.assertIn("missing 1 (seq_001)", message)

    def test_mismatch_surfaces_end_to_end_through_run_factorial(self):
        with tempfile.TemporaryDirectory() as tmp:
            cells, _ = planted_cells(count=5, noise=0.0)
            cells["a01"] = {name: record for name, record in cells["a01"].items()
                            if name != "seq_003"}
            targets = write_cells(tmp, cells)
            with self.assertRaises(PairedBootstrapError) as caught:
                run_factorial(targets, expected_sequences=0, replicates=50)
            self.assertIn("sequence sets differ", str(caught.exception))

    def test_a_missing_cell_is_rejected(self):
        cells, _ = planted_cells(count=5, noise=0.0)
        del cells["a01"]
        with self.assertRaises(PairedBootstrapError) as caught:
            factorial_bootstrap(cells, ["seq_000"], replicates=10)
        self.assertIn("missing cells a01", str(caught.exception))

    def test_mismatch_preview_is_bounded(self):
        cells = {key: {"s{0}".format(index): {"value": float(index)}
                       for index in range(40)}
                 for key in CELL_KEYS}
        cells["a10"] = {"t{0}".format(index): {"value": float(index)}
                        for index in range(40)}
        with self.assertRaises(PairedBootstrapError) as caught:
            pair_sequence_names_across(cells)
        self.assertIn("+30 more", str(caught.exception))


class FactorialPlantedEffectTest(unittest.TestCase):
    """A planted 2x2 whose every contrast is known in closed form."""

    def test_every_interval_brackets_its_planted_truth(self):
        cells, truth = planted_cells()
        contrasts = bootstrap_cells(cells)[0]
        for key in ALL_CONTRAST_KEYS:
            entry = contrasts[key]["metrics"]["value"]
            expected = truth[key]
            self.assertAlmostEqual(entry["difference"], expected, places=9,
                                   msg="point estimate wrong for " + key)
            self.assertLessEqual(entry["ci"][0], expected,
                                 msg="ci_low above truth for " + key)
            self.assertGreaterEqual(entry["ci"][1], expected,
                                    msg="ci_high below truth for " + key)

    def test_the_planted_interaction_is_detected(self):
        cells, truth = planted_cells(interaction=0.75)
        contrasts = bootstrap_cells(cells)[0]
        entry = contrasts[INTERACTION_KEY]["metrics"]["value"]
        self.assertAlmostEqual(entry["difference"], 0.75, places=9)
        self.assertFalse(entry["crosses_zero"])
        self.assertTrue(entry["significant"])
        self.assertEqual(entry["direction"], "positive")
        self.assertLess(entry["ci"][0], entry["ci"][1])
        self.assertGreater(entry["ci"][0], 0.0)

    def test_a_negative_interaction_is_reported_as_negative(self):
        cells, _ = planted_cells(interaction=-0.9)
        entry = bootstrap_cells(cells)[0][INTERACTION_KEY]["metrics"]["value"]
        self.assertAlmostEqual(entry["difference"], -0.9, places=9)
        self.assertEqual(entry["direction"], "negative")
        self.assertLess(entry["ci"][1], 0.0)

    def test_main_effects_are_not_just_the_simple_effects(self):
        # With a real interaction the main effect is offset from the simple
        # effect by exactly half the interaction; a tool that reported the
        # simple effect under a main-effect label would pass every other test.
        cells, _ = planted_cells(hinge=2.0, detach=-1.5, interaction=0.75)
        contrasts = bootstrap_cells(cells)[0]
        main = contrasts[MAIN1_KEY]["metrics"]["value"]["difference"]
        simple = contrasts["a10_minus_a00"]["metrics"]["value"]["difference"]
        self.assertAlmostEqual(main - simple, 0.75 / 2.0, places=9)
        main2 = contrasts[MAIN2_KEY]["metrics"]["value"]["difference"]
        simple2 = contrasts["a01_minus_a00"]["metrics"]["value"]["difference"]
        self.assertAlmostEqual(main2 - simple2, 0.75 / 2.0, places=9)

    def test_the_self_control_row_is_exactly_zero(self):
        cells, _ = planted_cells(count=60)
        entry = bootstrap_cells(cells)[0]["a00_minus_a00"]["metrics"]["value"]
        self.assertEqual(entry["difference"], 0.0)
        self.assertEqual(entry["ci"], [0.0, 0.0])
        self.assertTrue(entry["crosses_zero"])
        self.assertFalse(entry["significant"])

    def test_cell_summaries_report_every_cell(self):
        cells, _ = planted_cells(count=50, noise=0.0)
        cell_summaries = bootstrap_cells(cells, replicates=200)[1]
        summary = cell_summaries["value"]
        self.assertEqual(summary["complete_case_n"], 50)
        for key in CELL_KEYS:
            self.assertEqual(summary[key]["n_observed"], 50)
        self.assertAlmostEqual(summary["a10"]["mean"] - summary["a00"]["mean"], 2.0)
        self.assertAlmostEqual(summary["a01"]["mean"] - summary["a00"]["mean"], -1.5)


class FactorialAdditiveTest(unittest.TestCase):
    """A purely additive 2x2 must show an interaction interval containing zero."""

    def test_pure_additive_interaction_straddles_zero(self):
        cells, truth = planted_cells(interaction=0.0)
        self.assertEqual(truth[INTERACTION_KEY], 0.0)
        contrasts = bootstrap_cells(cells)[0]
        entry = contrasts[INTERACTION_KEY]["metrics"]["value"]
        self.assertAlmostEqual(entry["difference"], 0.0, places=9)
        self.assertTrue(entry["crosses_zero"])
        self.assertFalse(entry["significant"])
        self.assertEqual(entry["direction"], "inconclusive")
        # Not vacuous: the interval is genuinely wide, it is not degenerate at
        # zero because the three treated cells carry independent noise.
        self.assertLess(entry["ci"][0], 0.0)
        self.assertGreater(entry["ci"][1], 0.0)

    def test_pure_additive_main_effects_are_still_detected(self):
        contrasts = bootstrap_cells(planted_cells(interaction=0.0)[0])[0]
        for key, expected in ((MAIN1_KEY, 2.0), (MAIN2_KEY, -1.5)):
            entry = contrasts[key]["metrics"]["value"]
            self.assertAlmostEqual(entry["difference"], expected, places=9)
            self.assertFalse(entry["crosses_zero"])

    def test_pure_additive_main_effect_equals_its_simple_effect(self):
        contrasts = bootstrap_cells(planted_cells(interaction=0.0)[0])[0]
        self.assertAlmostEqual(
            contrasts[MAIN1_KEY]["metrics"]["value"]["difference"],
            contrasts["a10_minus_a00"]["metrics"]["value"]["difference"],
            places=9)
        self.assertAlmostEqual(
            contrasts["a11_minus_a01"]["metrics"]["value"]["difference"],
            contrasts["a10_minus_a00"]["metrics"]["value"]["difference"],
            places=9)

    def test_noiseless_additive_interaction_is_degenerate_at_zero(self):
        contrasts = bootstrap_cells(
            planted_cells(interaction=0.0, noise=0.0)[0], replicates=500)[0]
        entry = contrasts[INTERACTION_KEY]["metrics"]["value"]
        self.assertEqual(entry["difference"], 0.0)
        self.assertEqual(entry["ci"], [0.0, 0.0])
        # A degenerate interval at zero touches zero, so the "strictly
        # excludes zero" rule must call it inconclusive.
        self.assertTrue(entry["crosses_zero"])

    def test_a_float_noise_floor_is_reported_honestly(self):
        # Same additive design but on a Gaussian base, where (v+H+D) - (v+D)
        # is only H to within a rounding error. The tool must report the
        # residue it actually computed rather than snap it to zero; the
        # residue has to sit at the double-precision noise floor.
        contrasts = bootstrap_cells(
            planted_cells(interaction=0.0, noise=0.0, exact=False)[0],
            replicates=500)[0]
        entry = contrasts[INTERACTION_KEY]["metrics"]["value"]
        self.assertNotEqual(entry["difference"], 0.0)
        self.assertLess(abs(entry["difference"]), 1e-12)
        self.assertLess(max(abs(bound) for bound in entry["ci"]), 1e-12)


class FactorialIdentityTest(unittest.TestCase):
    """(A11-A01)-(A10-A00) and (A11-A10)-(A01-A00) are the same quantity."""

    def test_both_forms_agree_within_tolerance(self):
        cells, _ = planted_cells()
        identity = bootstrap_cells(cells)[4]
        self.assertEqual(identity["form_1"], "(a11 - a01) - (a10 - a00)")
        self.assertEqual(identity["form_2"], "(a11 - a10) - (a01 - a00)")
        self.assertTrue(identity["all_agree"])
        self.assertEqual(identity["disagreeing_metrics"], [])
        record = identity["per_metric"]["value"]
        self.assertLessEqual(record["column_max_abs_difference"],
                             record["tolerance"])
        self.assertLessEqual(record["ci_max_abs_difference"], record["tolerance"])

    def test_the_identity_also_holds_on_the_reported_numbers(self):
        # Built from the public report only: the four component contrasts are
        # bootstrapped by separate calls on separately constructed columns, so
        # this is an end-to-end algebraic check rather than a restatement.
        cells, _ = planted_cells()
        contrasts = bootstrap_cells(cells)[0]

        def difference(key):
            return contrasts[key]["metrics"]["value"]["difference"]

        reported = difference(INTERACTION_KEY)
        form_1 = difference("a11_minus_a01") - difference("a10_minus_a00")
        form_2 = difference("a11_minus_a10") - difference("a01_minus_a00")
        self.assertAlmostEqual(reported, form_1, places=9)
        self.assertAlmostEqual(reported, form_2, places=9)
        self.assertAlmostEqual(form_1, form_2, places=9)
        # And the main effects are the average of their two components.
        self.assertAlmostEqual(
            difference(MAIN1_KEY),
            0.5 * (difference("a10_minus_a00") + difference("a11_minus_a01")),
            places=9)
        self.assertAlmostEqual(
            difference(MAIN2_KEY),
            0.5 * (difference("a01_minus_a00") + difference("a11_minus_a10")),
            places=9)

    def test_a_broken_second_form_is_caught_and_raises(self):
        # The guard is only worth having if it fires. Corrupt the second form
        # the way a swapped cell would and confirm the tool refuses to report.
        cells, _ = planted_cells(count=40)
        original = paired_bootstrap_module._contrast_columns

        def sabotaged(columns, factor1_name, factor2_name):
            built, form_1, _ = original(columns, factor1_name, factor2_name)
            return built, form_1, columns["a11"] - columns["a00"]

        with mock.patch.object(paired_bootstrap_module, "_contrast_columns",
                               sabotaged):
            with self.assertRaises(PairedBootstrapError) as caught:
                bootstrap_cells(cells, replicates=100)
        message = str(caught.exception)
        self.assertIn("two algebraic forms of the interaction disagree", message)
        self.assertIn("implementation error", message)
        self.assertIn("value", message)


class FactorialSharedIndexTest(unittest.TestCase):
    """One resample index matrix for four runs, every contrast, every metric."""

    def _cells_with_a_twin(self):
        rng = np.random.default_rng(23)
        names = ["seq_{0:03d}".format(index) for index in range(120)]
        cells = {key: {} for key in CELL_KEYS}
        for name in names:
            base = float(rng.normal(0.0, 2.0))
            for offset, key in ((0.0, "a00"), (0.6, "a10"),
                                (-0.4, "a01"), (0.9, "a11")):
                value = base + offset + float(rng.normal(0.0, 0.5))
                cells[key][name] = {
                    "value": value,
                    "value_twin": value,
                    "value_scaled": value * 10.0,
                    "unrelated": float(rng.normal(0.0, 1.0)),
                }
        return cells

    def test_a_metric_duplicated_in_all_four_cells_matches_its_twin(self):
        contrasts = bootstrap_cells(self._cells_with_a_twin())[0]
        for key in ALL_CONTRAST_KEYS:
            first = contrasts[key]["metrics"]["value"]
            twin = contrasts[key]["metrics"]["value_twin"]
            for field in ("difference", "ci", "crosses_zero", "significant",
                          "direction", "n"):
                self.assertEqual(first[field], twin[field],
                                 msg="{0} differs on {1}".format(key, field))
        # Not vacuous: a genuinely different metric gets a different interval.
        self.assertNotEqual(
            contrasts[MAIN1_KEY]["metrics"]["value"]["ci"],
            contrasts[MAIN1_KEY]["metrics"]["unrelated"]["ci"])

    def test_a_scaled_metric_scales_every_interval_exactly(self):
        # A shared index matrix makes each contrast a linear functional of the
        # cell columns, so an exactly scaled metric yields exactly scaled
        # intervals. Freshly drawn per-metric or per-contrast indices would not.
        contrasts = bootstrap_cells(self._cells_with_a_twin())[0]
        for key in ALL_CONTRAST_KEYS:
            base = contrasts[key]["metrics"]["value"]
            scaled = contrasts[key]["metrics"]["value_scaled"]
            self.assertAlmostEqual(scaled["difference"],
                                   base["difference"] * 10.0, places=9,
                                   msg="difference not scaled for " + key)
            for position in (0, 1):
                self.assertAlmostEqual(scaled["ci"][position],
                                       base["ci"][position] * 10.0, places=9,
                                       msg="ci not scaled for " + key)

    def test_the_factorial_reuses_the_pairwise_index_matrix(self):
        # Same sequences, same seed, same replicate count: the a10-vs-a00
        # simple effect must be bit-identical to the pairwise b - a report.
        cells, _ = planted_cells(count=90, seed=31)
        names = pair_sequence_names_across(cells)
        contrasts, _, _, factorial_digest, _ = factorial_bootstrap(
            cells, names, seed=42, replicates=1500,
            factor1_name=FACTOR1, factor2_name=FACTOR2)
        pairwise, _, pairwise_digest = paired_bootstrap(
            cells["a00"], cells["a10"], names, seed=42, replicates=1500)
        self.assertEqual(factorial_digest, pairwise_digest)
        simple = contrasts["a10_minus_a00"]["metrics"]["value"]
        self.assertEqual(simple["difference"], pairwise["value"]["mean_delta"])
        self.assertEqual(simple["ci"][0], pairwise["value"]["ci_low"])
        self.assertEqual(simple["ci"][1], pairwise["value"]["ci_high"])

    def test_index_digest_is_reported_and_seed_dependent(self):
        cells, _ = planted_cells(count=40, noise=0.0)
        names = pair_sequence_names_across(cells)
        first = factorial_bootstrap(cells, names, seed=42, replicates=100)[3]
        again = factorial_bootstrap(cells, names, seed=42, replicates=100)[3]
        other = factorial_bootstrap(cells, names, seed=43, replicates=100)[3]
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 64)


class FactorialCoverageAndNanTest(unittest.TestCase):
    def test_a_metric_absent_from_one_cell_is_excluded_and_named(self):
        cells, _ = planted_cells(count=6, noise=0.0)
        for name in cells["a11"]:
            del cells["a11"][name]["value"]
            cells["a11"][name]["other"] = 1.0
        coverage = discover_metrics_across(cells, sorted(cells["a00"]))
        self.assertNotIn("value", coverage["analyzed"])
        self.assertEqual(coverage["excluded_not_in_every_cell"]["value"]["absent_from"],
                         ["a11"])
        self.assertEqual(coverage["excluded_not_in_every_cell"]["other"]["present_in"],
                         ["a11"])
        self.assertIn("object_name", coverage["excluded_non_numeric"])

    def test_run_factorial_warns_about_a_metric_missing_from_one_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            cells, _ = planted_cells(count=6, noise=0.0)
            for name in cells["a01"]:
                cells["a01"][name]["extra"] = 1.0
            warnings = []
            run_factorial(write_cells(tmp, cells), expected_sequences=0,
                          replicates=50, warn=warnings.append)
            self.assertTrue(any("'extra'" in text and "a00" in text
                                for text in warnings))

    def test_a_null_in_one_cell_only_drops_the_contrasts_that_need_it(self):
        cells, _ = planted_cells(count=30, noise=0.0)
        cells["a11"]["seq_005"]["value"] = None
        contrasts = bootstrap_cells(cells, replicates=200)[0]
        metrics = {key: contrasts[key]["metrics"]["value"]
                   for key in ALL_CONTRAST_KEYS}
        self.assertEqual(metrics["a10_minus_a00"]["n"], 30)
        self.assertEqual(metrics["a01_minus_a00"]["n"], 30)
        for key in ("a11_minus_a00", "a11_minus_a01", "a11_minus_a10",
                    MAIN1_KEY, MAIN2_KEY, INTERACTION_KEY):
            self.assertEqual(metrics[key]["n"], 29, msg="wrong n for " + key)
            self.assertEqual(metrics[key]["n_dropped_nonfinite"], 1)
        # The point estimates are unchanged because the design is noiseless.
        self.assertAlmostEqual(metrics[INTERACTION_KEY]["difference"], 0.75, places=9)

    def test_an_all_null_metric_is_undefined_rather_than_a_crash(self):
        cells, _ = planted_cells(count=8, noise=0.0)
        for key in CELL_KEYS:
            for name in cells[key]:
                cells[key][name]["blank"] = None
        contrasts = bootstrap_cells(cells, replicates=100)[0]
        entry = contrasts[INTERACTION_KEY]["metrics"]["blank"]
        self.assertEqual(entry["n"], 0)
        self.assertIsNone(entry["difference"])
        self.assertIsNone(entry["ci"])
        self.assertEqual(entry["direction"], "undefined")
        self.assertIn("note", entry)
        self.assertIsNotNone(
            contrasts[INTERACTION_KEY]["metrics"]["value"]["difference"])

    def test_an_infinity_is_treated_as_missing(self):
        cells, _ = planted_cells(count=12, noise=0.0)
        cells["a01"]["seq_004"]["value"] = float("inf")
        contrasts = bootstrap_cells(cells, replicates=100)[0]
        self.assertEqual(contrasts["a10_minus_a00"]["metrics"]["value"]["n"], 12)
        self.assertEqual(contrasts[INTERACTION_KEY]["metrics"]["value"]["n"], 11)


class FactorialDeterminismTest(unittest.TestCase):
    def test_two_cli_runs_with_seed_42_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets = write_cells(tmp, planted_cells(count=80, seed=57)[0])
            command = [PYTHON, str(TOOL), "--factorial"] + [
                "{0}={1}".format(key, targets[key]) for key in CELL_KEYS
            ] + ["--factor1-name", FACTOR1, "--factor2-name", FACTOR2,
                 "--seed", "42", "--replicates", "1500",
                 "--expected-sequences", "0"]
            one = subprocess.run(command, capture_output=True, check=True)
            two = subprocess.run(command, capture_output=True, check=True)
            self.assertEqual(one.stdout, two.stdout)
            self.assertEqual(one.stderr, two.stderr)
            payload = json.loads(one.stdout.decode("utf-8"))
            self.assertEqual(payload["mode"], "factorial_2x2")
            self.assertEqual(payload["bootstrap"]["seed"], 42)

    def test_api_repeats_bit_for_bit_and_the_seed_moves_only_the_interval(self):
        cells, _ = planted_cells(count=100, seed=71)
        names = pair_sequence_names_across(cells)
        first = factorial_bootstrap(cells, names, seed=42, replicates=1500)[0]
        second = factorial_bootstrap(cells, names, seed=42, replicates=1500)[0]
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))
        other = factorial_bootstrap(cells, names, seed=1234, replicates=1500)[0]
        key = "interaction_{0}_x_{1}".format("factor1", "factor2")
        self.assertEqual(first[key]["metrics"]["value"]["difference"],
                         other[key]["metrics"]["value"]["difference"])
        self.assertNotEqual(first[key]["metrics"]["value"]["ci"],
                            other[key]["metrics"]["value"]["ci"])

    def test_cell_insertion_order_does_not_change_the_result(self):
        cells, _ = planted_cells(count=60, seed=83)
        reordered = {key: {name: cells[key][name]
                           for name in reversed(list(cells[key]))}
                     for key in reversed(CELL_KEYS)}
        forward = bootstrap_cells(cells, replicates=400)[0]
        backward = bootstrap_cells(reordered, replicates=400)[0]
        self.assertEqual(json.dumps(forward, sort_keys=True),
                         json.dumps(backward, sort_keys=True))


class FactorialReportAndCliTest(unittest.TestCase):
    def _targets(self, tmp, **kwargs):
        return write_cells(tmp, planted_cells(count=40, **kwargs)[0])

    def _cli(self, targets, *extra):
        return ["--factorial"] + [
            "{0}={1}".format(key, targets[key]) for key in CELL_KEYS
        ] + ["--expected-sequences", "0", "--replicates", "200"] + list(extra)

    def test_report_records_the_design_protocol_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_factorial(self._targets(tmp), expected_sequences=0,
                                   replicates=300, seed=99,
                                   factor1_name=FACTOR1, factor2_name=FACTOR2)
            self.assertEqual(report["mode"], "factorial_2x2")
            self.assertEqual(report["design"]["factor1"]["name"], FACTOR1)
            self.assertEqual(report["design"]["factor2"]["name"], FACTOR2)
            self.assertEqual(report["design"]["factor1"]["level1_cells"],
                             ["a10", "a11"])
            self.assertEqual(report["design"]["factor2"]["level1_cells"],
                             ["a01", "a11"])
            self.assertEqual(report["design"]["reference_cell"], "a00")
            self.assertEqual(report["bootstrap"]["seed"], 99)
            self.assertTrue(report["bootstrap"]["shared_resample_index_matrix"])
            self.assertTrue(report["bootstrap"]["per_replicate_contrasts"])
            self.assertEqual(len(report["bootstrap"]["resample_index_sha256"]), 64)
            for key in CELL_KEYS:
                self.assertEqual(len(report["inputs"][key]["sha256"]), 64)
                self.assertEqual(report["inputs"][key]["cell"], key)
            self.assertEqual(sorted(report["contrasts"]), sorted(ALL_CONTRAST_KEYS))
            groups = report["design"]["contrast_groups"]
            self.assertEqual(len(groups["simple_effects_vs_a00"]), 4)
            self.assertEqual(len(groups["main_effects"]), 2)
            self.assertEqual(len(groups["component_effects"]), 4)

    def test_every_contrast_states_its_own_sign_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_factorial(self._targets(tmp), expected_sequences=0,
                                   replicates=100, factor1_name=FACTOR1,
                                   factor2_name=FACTOR2)
            for key, contrast in report["contrasts"].items():
                self.assertTrue(contrast["form"], msg=key)
                self.assertTrue(contrast["positive_means"], msg=key)
                self.assertTrue(contrast["null_hypothesis"], msg=key)
                self.assertTrue(contrast["cells_required_finite"], msg=key)
            self.assertIn("lower is better",
                          report["sign_convention"]["polarity_note"])
            self.assertIn("strictly excludes zero",
                          report["sign_convention"]["significance_rule"])

    def test_table_states_the_conventions_and_names_every_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_factorial(self._targets(tmp), expected_sequences=0,
                                   replicates=200, factor1_name=FACTOR1,
                                   factor2_name=FACTOR2,
                                   labels={"a00": "W3", "a10": "HINGE"})
            table = format_factorial_table(report)
            self.assertIn("2x2 factorial", table)
            self.assertIn("a<hinge level><detach level>", table)
            self.assertIn("W3", table)
            self.assertIn("HINGE", table)
            self.assertIn("SIGN CONVENTION", table)
            self.assertIn("POLARITY", table)
            self.assertIn("shared resample index matrix", table)
            self.assertIn("paired BY NAME across all four cells", table)
            self.assertIn("interaction identity", table)
            self.assertIn("AGREE", table)
            for group in ("main_effects", "interaction",
                          "simple_effects_vs_a00", "component_effects"):
                self.assertIn("=== {0} ===".format(group), table)
            self.assertIn(MAIN1_KEY + " = 0.5 * ((a10 - a00) + (a11 - a01))", table)
            self.assertIn(INTERACTION_KEY + " = (a11 - a01) - (a10 - a00)", table)

    def test_cli_writes_the_report_and_will_not_silently_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets = self._targets(tmp)
            out = Path(tmp) / "nested" / "factorial.json"
            self.assertEqual(main(self._cli(targets, "--output", str(out))), 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                payload["contrasts"]["interaction_factor1_x_factor2"]
                ["metrics"]["value"]["difference"], 0.75, places=9)
            self.assertEqual(main(self._cli(targets, "--output", str(out))), 2)
            self.assertEqual(
                main(self._cli(targets, "--output", str(out), "--overwrite")), 0)

    def test_cli_rejects_an_incomplete_or_malformed_cell_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets = self._targets(tmp)
            first = targets["a00"]
            self.assertEqual(main([
                "--factorial", "a00=" + first, "a10=" + first,
                "a01=" + first, "a99=" + first, "--expected-sequences", "0",
            ]), 2)
            self.assertEqual(main([
                "--factorial", "a00=" + first, "a10=" + first,
                "a01=" + first, "a01=" + first, "--expected-sequences", "0",
            ]), 2)
            self.assertEqual(main([
                "--factorial", "a00=" + first, "a10=" + first,
                "a01=" + first, first, "--expected-sequences", "0",
            ]), 2)

    def test_cli_refuses_to_mix_the_two_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets = self._targets(tmp)
            self.assertEqual(main(
                self._cli(targets, "--a", targets["a00"])), 2)
            self.assertEqual(main(
                self._cli(targets, "--label-a", "W3")), 2)
            self.assertEqual(main([
                "--a", targets["a00"], "--b", targets["a10"],
                "--cell-label", "a00=W3", "--expected-sequences", "0",
            ]), 2)
            self.assertEqual(main(["--expected-sequences", "0"]), 2)

    def test_pairwise_mode_still_works_through_the_same_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets = self._targets(tmp)
            out = Path(tmp) / "pairwise.json"
            self.assertEqual(main([
                "--a", targets["a00"], "--b", targets["a10"],
                "--expected-sequences", "0", "--replicates", "200",
                "--output", str(out),
            ]), 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "pairwise")
            self.assertAlmostEqual(payload["metrics"]["value"]["mean_delta"],
                                   2.0, places=9)

    def test_duplicate_cell_files_are_warned_about(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets = self._targets(tmp)
            targets["a01"] = targets["a00"]
            targets["a11"] = targets["a10"]
            warnings = []
            run_factorial(targets, expected_sequences=0, replicates=100,
                          warn=warnings.append)
            self.assertTrue(any("not a real 2x2" in text for text in warnings))
            self.assertTrue(any("a00/a01" in text for text in warnings))

    def test_wrong_sequence_count_warns_but_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            warnings = []
            report = run_factorial(self._targets(tmp), expected_sequences=438,
                                   replicates=100, warn=warnings.append)
            self.assertTrue(any("expected 438" in text for text in warnings))
            self.assertFalse(report["sequences"]["matches_expected"])

    def test_identical_factor_names_are_rejected(self):
        cells, _ = planted_cells(count=10, noise=0.0)
        with self.assertRaises(PairedBootstrapError) as caught:
            bootstrap_cells(cells, replicates=50, factor1_name="x",
                            factor2_name="x")
        self.assertIn("factor names must differ", str(caught.exception))


class FactorialDegenerateDesignTest(unittest.TestCase):
    """The a01=a00, a11=a10 collapse, which has closed-form answers.

    This is the mechanical plumbing check used on the real P8 evaluations:
    with only two distinct evaluations duplicated across the four cells the
    design is scientifically meaningless, but three identities must hold to
    float precision or the implementation is wrong.
    """

    def _report(self, tmp, count=60):
        cells, _ = planted_cells(count=count, seed=97)
        cells["a01"] = cells["a00"]
        cells["a11"] = cells["a10"]
        targets = write_cells(tmp, {"a00": cells["a00"], "a10": cells["a10"]})
        targets["a01"] = targets["a00"]
        targets["a11"] = targets["a10"]
        return run_factorial(targets, expected_sequences=0, replicates=1000,
                             factor1_name=FACTOR1, factor2_name=FACTOR2)

    def test_the_three_collapse_identities_hold_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            contrasts = self._report(tmp)["contrasts"]
            simple = contrasts["a10_minus_a00"]["metrics"]["value"]
            hinge = contrasts[MAIN1_KEY]["metrics"]["value"]
            detach = contrasts[MAIN2_KEY]["metrics"]["value"]
            interaction = contrasts[INTERACTION_KEY]["metrics"]["value"]
            # 1. the factor-1 main effect collapses onto the a10 - a00 delta
            self.assertEqual(hinge["difference"], simple["difference"])
            self.assertEqual(hinge["ci"], simple["ci"])
            # 2. the factor-2 main effect is exactly zero
            self.assertEqual(detach["difference"], 0.0)
            self.assertEqual(detach["ci"], [0.0, 0.0])
            # 3. the interaction is exactly zero
            self.assertEqual(interaction["difference"], 0.0)
            self.assertEqual(interaction["ci"], [0.0, 0.0])
            # and the collapse is not vacuous: factor 1 really did move
            self.assertNotEqual(simple["difference"], 0.0)
            self.assertFalse(simple["crosses_zero"])


class RealRunSmokeTest(unittest.TestCase):
    """Exercise the tool on the two sealed P8 evaluations when present."""

    A = REPO_ROOT / "results/experiments/p1-hoi-p8-eval-h0-guided-s42-20260806"
    B = REPO_ROOT / "results/experiments/p1-hoi-p8-eval-w3-guided-s42-20260809"

    def _require_sealed(self):
        for path in (self.A, self.B):
            if not (path / "evaluation" / "per_sequence_metrics.json").is_file():
                self.skipTest("sealed evaluation outputs are not present")

    def test_real_p8_pair_produces_438_paired_sequences(self):
        self._require_sealed()
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

    def test_real_collapsed_2x2_satisfies_the_three_identities(self):
        """Mechanical plumbing check on the only four evaluations that exist.

        With a01 = a00 and a11 = a10 this is NOT a 2x2 and nothing here is a
        scientific statement, but the arithmetic must still be exact: the
        factor-1 main effect collapses onto the a10 - a00 delta, the factor-2
        main effect is zero and the interaction is zero. The real files carry
        both a full-cohort metric (438) and a subset metric (181), so the
        identities are checked under real missingness rather than only on
        synthetic complete cases.
        """
        self._require_sealed()
        report = run_factorial(
            {"a00": str(self.A), "a10": str(self.B),
             "a01": str(self.A), "a11": str(self.B)},
            replicates=200, factor1_name=FACTOR1, factor2_name=FACTOR2,
        )
        self.assertEqual(report["sequences"]["paired_count"], 438)
        self.assertTrue(report["interaction_identity_check"]["all_agree"])
        contrasts = report["contrasts"]
        moved = 0
        for metric in report["metric_coverage"]["analyzed"]:
            simple = contrasts["a10_minus_a00"]["metrics"][metric]
            hinge = contrasts[MAIN1_KEY]["metrics"][metric]
            detach = contrasts[MAIN2_KEY]["metrics"][metric]
            interaction = contrasts[INTERACTION_KEY]["metrics"][metric]
            self.assertEqual(hinge["difference"], simple["difference"],
                             msg="factor-1 main effect diverged on " + metric)
            self.assertEqual(hinge["ci"], simple["ci"], msg=metric)
            self.assertEqual(detach["difference"], 0.0, msg=metric)
            self.assertEqual(detach["ci"], [0.0, 0.0], msg=metric)
            self.assertEqual(interaction["difference"], 0.0, msg=metric)
            self.assertEqual(interaction["ci"], [0.0, 0.0], msg=metric)
            if simple["difference"] != 0.0:
                moved += 1
        # Not vacuous: the two real evaluations genuinely differ.
        self.assertGreater(moved, 0)
        self.assertEqual(
            contrasts[INTERACTION_KEY]["metrics"]["hand_pen_loss_omomo"]["n"], 181)
        self.assertEqual(
            contrasts[INTERACTION_KEY]["metrics"]["contact_f1"]["n"], 438)


if __name__ == "__main__":
    unittest.main()
