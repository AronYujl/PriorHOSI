#!/usr/bin/env python3
"""Paired sequence-level bootstrap confidence intervals between two evaluations.

This promotes the ad-hoc analysis used to seal Phase 1B P6/P7 into tracked
tooling, because preregistered decisions from P10 onward rest on it. The
protocol is fixed by the 2026-08-09 P10 preregistration in
``docs/EXPERIMENT_PLAN.md``:

* the resampling unit is the evaluation sequence, and sequences are paired
  **by name** across the two runs (never by position);
* one shared resample index matrix is drawn once and reused for every metric,
  which is what makes the per-metric intervals mutually comparable;
* 10,000 replicates, seed 42, 2.5/97.5 percentiles;
* the reported quantity is ``delta = b - a`` and the sign convention is always
  restated in the output.

Only ``numpy`` is required beyond the standard library.

Scope note: the per-sequence file is the only admissible input, so only metrics
the evaluator emits per sequence can receive an interval. Aggregate-only
metrics (for example ``contact_percent``, which the evaluator computes as a
mean of per-sequence contact fractions but does not persist per sequence) have
no sequence-level sample and therefore cannot be bootstrapped here.

There is deliberately no flag to restrict the analysed metric set: every metric
present in both files is reported, so a decision cannot be based on a
post-hoc favourable subset.
"""

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_REPLICATES = 10000
DEFAULT_EXPECTED_SEQUENCES = 438
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "experiments"
PER_SEQUENCE_BASENAME = "per_sequence_metrics.json"
_MISMATCH_PREVIEW = 10


class PairedBootstrapError(RuntimeError):
    """A loud, actionable failure in input resolution, pairing or protocol."""


def resolve_per_sequence_path(target, results_root=DEFAULT_RESULTS_ROOT):
    """Resolve a run id, run directory or explicit file path to a metrics file.

    Accepted forms, in the order they are tried:

    1. an existing file (used verbatim);
    2. an existing directory (``<dir>/evaluation/per_sequence_metrics.json``
       then ``<dir>/per_sequence_metrics.json``);
    3. a run id under ``results_root``.
    """
    candidate = Path(target)
    attempted = []
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_dir():
        for relative in ("evaluation/" + PER_SEQUENCE_BASENAME, PER_SEQUENCE_BASENAME):
            probe = candidate / relative
            attempted.append(probe)
            if probe.is_file():
                return probe.resolve()
    else:
        attempted.append(candidate)
    run_probe = Path(results_root) / str(target) / "evaluation" / PER_SEQUENCE_BASENAME
    attempted.append(run_probe)
    if run_probe.is_file():
        return run_probe.resolve()
    listing = "\n".join("  tried: " + str(path) for path in attempted)
    raise PairedBootstrapError(
        "cannot resolve per-sequence metrics for {0!r}; pass a run id, a run "
        "directory, or a path to a {1} file\n{2}".format(
            target, PER_SEQUENCE_BASENAME, listing
        )
    )


def load_per_sequence(path):
    """Load and validate one ``per_sequence_metrics.json`` payload."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairedBootstrapError("{0} is not readable JSON: {1}".format(path, error))
    if not isinstance(payload, dict):
        raise PairedBootstrapError("{0} must contain a JSON object".format(path))
    if "metrics" not in payload:
        raise PairedBootstrapError(
            "{0} has no 'metrics' key; keys present: {1}".format(
                path, sorted(payload)
            )
        )
    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        raise PairedBootstrapError(
            "{0}['metrics'] must be a non-empty object keyed by sequence name".format(path)
        )
    for name, record in metrics.items():
        if not isinstance(record, dict):
            raise PairedBootstrapError(
                "{0}['metrics'][{1!r}] must be an object, found {2}".format(
                    path, name, type(record).__name__
                )
            )
    declared = payload.get("sequence_count")
    if isinstance(declared, int) and declared != len(metrics):
        raise PairedBootstrapError(
            "{0} is internally inconsistent: sequence_count={1} but "
            "metrics holds {2} sequences".format(path, declared, len(metrics))
        )
    provenance = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "schema_version": payload.get("schema_version"),
        "evaluation_seed": payload.get("seed"),
        "sequence_count": len(metrics),
    }
    return metrics, provenance


def pair_sequence_names(metrics_a, metrics_b, label_a="a", label_b="b"):
    """Return the sorted shared names, failing loudly on any set difference."""
    names_a, names_b = set(metrics_a), set(metrics_b)
    if names_a != names_b:
        only_a = sorted(names_a - names_b)
        only_b = sorted(names_b - names_a)
        raise PairedBootstrapError(
            "sequence sets differ; refusing to silently intersect.\n"
            "  {0} has {1} sequences, {2} has {3}, {4} shared\n"
            "  only in {0} ({5}): {6}\n"
            "  only in {2} ({7}): {8}".format(
                label_a,
                len(names_a),
                label_b,
                len(names_b),
                len(names_a & names_b),
                len(only_a),
                _preview(only_a),
                len(only_b),
                _preview(only_b),
            )
        )
    if not names_a:
        raise PairedBootstrapError("no sequences to pair")
    return sorted(names_a)


def _preview(names):
    if not names:
        return "none"
    head = ", ".join(names[:_MISMATCH_PREVIEW])
    if len(names) > _MISMATCH_PREVIEW:
        return head + ", ... (+{0} more)".format(len(names) - _MISMATCH_PREVIEW)
    return head


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def discover_metrics(metrics_a, metrics_b, names):
    """Classify every key seen in either file.

    A key is analysable when it appears in both files and every non-null value
    on both sides is a real number. ``None`` is admitted as a missing
    observation (the evaluator writes it for object classes excluded from the
    penetration metrics) and becomes NaN downstream.
    """
    keys_a, keys_b = set(), set()
    non_numeric = set()
    for source, keys in ((metrics_a, keys_a), (metrics_b, keys_b)):
        for name in names:
            for key, value in source[name].items():
                keys.add(key)
                if value is not None and not _is_number(value):
                    non_numeric.add(key)
    shared = keys_a & keys_b
    return {
        "analyzed": sorted(shared - non_numeric),
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "excluded_non_numeric": sorted(non_numeric & shared),
    }


def _column(source, names, metric):
    out = np.full(len(names), np.nan, dtype=np.float64)
    for position, name in enumerate(names):
        value = source[name].get(metric)
        if _is_number(value):
            out[position] = float(value)
    return out


def paired_bootstrap(metrics_a, metrics_b, names, seed=DEFAULT_SEED,
                     replicates=DEFAULT_REPLICATES):
    """Bootstrap ``b - a`` for every analysable metric on one shared index matrix.

    The index matrix is drawn once, before any metric is touched, and reused
    verbatim for all of them. Non-finite paired differences (either side
    missing, or an infinity) are excluded from every replicate mean via
    ``np.nanmean``, so a sequence only contributes where the pair exists.
    """
    if replicates < 1:
        raise PairedBootstrapError("replicates must be >= 1, got {0}".format(replicates))
    if not names:
        raise PairedBootstrapError("no sequences to bootstrap")
    coverage = discover_metrics(metrics_a, metrics_b, names)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(names), size=(replicates, len(names)))
    results = {}
    for metric in coverage["analyzed"]:
        column_a = _column(metrics_a, names, metric)
        column_b = _column(metrics_b, names, metric)
        delta = column_b - column_a
        delta[~np.isfinite(delta)] = np.nan
        used = int(np.count_nonzero(~np.isnan(delta)))
        entry = {
            "n_pairs": len(names),
            "n_pairs_used": used,
            "n_pairs_dropped_nonfinite": len(names) - used,
        }
        if used == 0:
            entry.update({
                "mean_a": None, "mean_b": None, "mean_delta": None,
                "ci_low": None, "ci_high": None, "significant": None,
                "direction": "undefined", "nan_replicates": replicates,
                "note": "no sequence has a finite value on both sides",
            })
            results[metric] = entry
            continue
        with warnings.catch_warnings():
            # An all-missing replicate is a legitimate outcome for a metric the
            # evaluator scores on only part of the cohort; it is counted below
            # rather than announced on stderr next to the table.
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore"):
                replicate_means = np.nanmean(delta[index], axis=1)
                nan_replicates = int(np.count_nonzero(np.isnan(replicate_means)))
                low = float(np.nanpercentile(replicate_means, LOWER_PERCENTILE))
                high = float(np.nanpercentile(replicate_means, UPPER_PERCENTILE))
                mean_delta = float(np.nanmean(delta))
                mean_a = float(np.nanmean(column_a))
                mean_b = float(np.nanmean(column_b))
        significant = bool(low > 0.0 or high < 0.0)
        entry.update({
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_delta": mean_delta,
            "ci_low": low,
            "ci_high": high,
            "significant": significant,
            "direction": ("b_greater" if significant and low > 0.0
                          else "b_lower" if significant
                          else "inconclusive"),
            "nan_replicates": nan_replicates,
        })
        results[metric] = entry
    index_digest = hashlib.sha256(
        np.ascontiguousarray(index, dtype=np.int64).tobytes()
    ).hexdigest()
    return results, coverage, index_digest


def run(a, b, results_root=DEFAULT_RESULTS_ROOT, seed=DEFAULT_SEED,
        replicates=DEFAULT_REPLICATES,
        expected_sequences=DEFAULT_EXPECTED_SEQUENCES,
        label_a=None, label_b=None, warn=None):
    """Resolve, pair, bootstrap and package the full report."""
    warn = warn if warn is not None else (lambda message: None)
    label_a = label_a or str(a)
    label_b = label_b or str(b)
    path_a = resolve_per_sequence_path(a, results_root)
    path_b = resolve_per_sequence_path(b, results_root)
    if path_a == path_b:
        warn("a and b resolve to the same file {0}; "
             "every delta will be exactly zero".format(path_a))
    metrics_a, provenance_a = load_per_sequence(path_a)
    metrics_b, provenance_b = load_per_sequence(path_b)
    names = pair_sequence_names(metrics_a, metrics_b, label_a, label_b)
    matches_expected = None
    if expected_sequences is not None and expected_sequences > 0:
        matches_expected = len(names) == expected_sequences
        if not matches_expected:
            warn("paired {0} sequences but expected {1}; continuing".format(
                len(names), expected_sequences))
    results, coverage, index_digest = paired_bootstrap(
        metrics_a, metrics_b, names, seed=seed, replicates=replicates
    )
    for key in ("only_in_a", "only_in_b"):
        if coverage[key]:
            warn("metrics present only in {0}, not analysed: {1}".format(
                label_a if key == "only_in_a" else label_b,
                ", ".join(coverage[key])))
    provenance_a["label"] = label_a
    provenance_b["label"] = label_b
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/paired_bootstrap.py",
        "delta_convention": "b_minus_a",
        "delta_description": (
            "delta = b - a, per sequence, where a={0!r} and b={1!r}; a positive "
            "delta means b reports a larger value than a".format(label_a, label_b)
        ),
        "bootstrap": {
            "paired": True,
            "resample_unit": "sequence",
            "replicates": replicates,
            "seed": seed,
            "percentiles": [LOWER_PERCENTILE, UPPER_PERCENTILE],
            "percentile_method": "linear",
            "shared_resample_index_matrix": True,
            "resample_index_sha256": index_digest,
            "significance_rule": "confidence interval strictly excludes zero",
            "missing_value_rule": (
                "a non-finite paired difference is excluded from each replicate "
                "mean via np.nanmean; the pair never contributes"
            ),
        },
        "inputs": {"a": provenance_a, "b": provenance_b},
        "sequences": {
            "paired_count": len(names),
            "expected_count": expected_sequences,
            "matches_expected": matches_expected,
            "order": "sorted-by-name",
        },
        "metric_coverage": coverage,
        "metrics": results,
    }


def format_table(report):
    """Render the report as a fixed-width table for human review."""
    rows = [(
        "metric", "n", "mean_a", "mean_b", "delta(b-a)",
        "ci_low", "ci_high", "sig",
    )]
    for metric in sorted(report["metrics"]):
        entry = report["metrics"][metric]
        if entry["mean_delta"] is None:
            rows.append((metric, str(entry["n_pairs_used"]),
                         "-", "-", "-", "-", "-", "n/a"))
            continue
        rows.append((
            metric,
            str(entry["n_pairs_used"]),
            "{0:.6g}".format(entry["mean_a"]),
            "{0:.6g}".format(entry["mean_b"]),
            "{0:+.6g}".format(entry["mean_delta"]),
            "{0:+.6g}".format(entry["ci_low"]),
            "{0:+.6g}".format(entry["ci_high"]),
            "YES" if entry["significant"] else "no",
        ))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = [
        "a = {0}".format(report["inputs"]["a"]["label"]),
        "b = {0}".format(report["inputs"]["b"]["label"]),
        report["delta_description"],
        "{0} sequences paired by name, {1} replicates, seed {2}, "
        "{3}/{4} percentiles, one shared resample index matrix".format(
            report["sequences"]["paired_count"],
            report["bootstrap"]["replicates"],
            report["bootstrap"]["seed"],
            *report["bootstrap"]["percentiles"],
        ),
        "",
    ]
    for position, row in enumerate(rows):
        lines.append("  ".join(
            cell.ljust(width) if column == 0 else cell.rjust(width)
            for column, (cell, width) in enumerate(zip(row, widths))
        ).rstrip())
        if position == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Paired sequence-level bootstrap CIs between two evaluation runs.",
        epilog="Every metric present in both files is reported; there is no "
               "flag to restrict the metric set.",
    )
    parser.add_argument("--a", required=True,
                        help="baseline run id, run directory or per_sequence_metrics.json path")
    parser.add_argument("--b", required=True,
                        help="comparison run id, run directory or per_sequence_metrics.json path")
    parser.add_argument("--label-a", default=None, help="name for a in the output")
    parser.add_argument("--label-b", default=None, help="name for b in the output")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT),
                        help="root holding <run-id>/evaluation directories")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="bootstrap seed (default: %(default)s)")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                        help="bootstrap replicates (default: %(default)s)")
    parser.add_argument("--expected-sequences", type=int, default=DEFAULT_EXPECTED_SEQUENCES,
                        help="warn if the paired count differs; 0 disables "
                             "(default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="write the JSON report here instead of stdout")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow --output to replace an existing file")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    def warn(message):
        print("warning: " + message, file=sys.stderr)

    try:
        report = run(
            args.a, args.b,
            results_root=Path(args.results_root),
            seed=args.seed,
            replicates=args.replicates,
            expected_sequences=args.expected_sequences,
            label_a=args.label_a,
            label_b=args.label_b,
            warn=warn,
        )
    except PairedBootstrapError as error:
        print("error: {0}".format(error), file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(format_table(report), file=sys.stderr)
    if args.output is None:
        sys.stdout.write(payload)
        return 0
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.overwrite:
        print("error: {0} exists; pass --overwrite to replace it".format(path),
              file=sys.stderr)
        return 2
    path.write_text(payload, encoding="utf-8")
    print("wrote {0}".format(path), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
