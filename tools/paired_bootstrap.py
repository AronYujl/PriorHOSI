#!/usr/bin/env python3
"""Paired sequence-level bootstrap confidence intervals between evaluations.

This promotes the ad-hoc analysis used to seal Phase 1B P6/P7 into tracked
tooling, because preregistered decisions from P10 onward rest on it. The
protocol is fixed by the 2026-08-09 P10 preregistration in
``docs/plan/PHASE_1B_HOI/06_GEOMETRY_TERM.md``:

* the resampling unit is the evaluation sequence, and sequences are paired
  **by name** across the runs (never by position);
* one shared resample index matrix is drawn once and reused for every metric,
  which is what makes the per-metric intervals mutually comparable;
* 10,000 replicates, seed 42, 2.5/97.5 percentiles;
* the reported quantity is a difference in the metric's own units and the sign
  convention is always restated in the output.

Two modes:

``--a/--b`` **pairwise**
    one A-vs-B contrast, ``delta = b - a``.

``--factorial a00=... a10=... a01=... a11=...`` **2x2 factorial**
    the P10 design. Cell ``a<f1><f2>`` carries factor 1 at level ``<f1>`` and
    factor 2 at level ``<f2>``, so for P10 ``a10`` is hinge=0.02/detach=false.
    Reports both main effects, the interaction, every cell against ``a00``, and
    the four conditional differences the main effects and interaction are built
    from. Every contrast is formed **per sequence first** and then bootstrapped
    on the **same** shared resample index matrix as every other contrast and
    metric, so the intervals are mutually comparable and the interaction
    interval carries the correct covariance between the four cells. Combining
    separately bootstrapped pairwise reports would not: it discards that
    covariance. The reporting vocabulary (``difference`` / ``ci`` /
    ``crosses_zero``, a spelled-out ``form`` and ``null_hypothesis``) matches
    the P3 interaction analysis in
    ``experiments/results/p1_hoi_p3_relation_field_guidance_s42_20260802.json``
    so a P10 table reads next to it.

Only ``numpy`` is required beyond the standard library.

Scope note: the per-sequence file is the only admissible input, so only metrics
the evaluator emits per sequence can receive an interval. Aggregate-only
metrics (for example ``contact_percent``, which the evaluator computes as a
mean of per-sequence contact fractions but does not persist per sequence) have
no sequence-level sample and therefore cannot be bootstrapped here.

There is deliberately no flag to restrict the analysed metric set: every metric
present in every input file is reported, so a decision cannot be based on a
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

# Cell naming for the 2x2: a<factor1 level><factor2 level>.
CELL_KEYS = ("a00", "a10", "a01", "a11")
DEFAULT_FACTOR1_NAME = "factor1"
DEFAULT_FACTOR2_NAME = "factor2"
# The interaction identity is checked at |form1 - form2| <= atol + rtol * scale,
# where scale is the largest absolute cell value seen for that metric. A real
# sign or cell mix-up moves the two forms by O(effect size), which is orders of
# magnitude above this; only floating-point non-associativity fits under it.
IDENTITY_ATOL = 1e-12
IDENTITY_RTOL = 1e-12


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


def pair_sequence_names_across(sources, labels=None):
    """Return the sorted names shared by *every* source, failing on any difference.

    ``sources`` maps a cell key to that cell's per-sequence ``metrics`` dict.
    The failure message reports, per cell, both halves of the symmetric
    difference against the union: which names the cell is missing and which
    names only it has.
    """
    if len(sources) < 2:
        raise PairedBootstrapError(
            "need at least two sources to pair, got {0}".format(len(sources))
        )
    labels = labels or {}
    name_sets = {key: set(value) for key, value in sources.items()}
    union = set()
    for value in name_sets.values():
        union |= value
    intersection = set(union)
    for value in name_sets.values():
        intersection &= value
    if not union:
        raise PairedBootstrapError("no sequences to pair")
    if len(intersection) != len(union):
        lines = [
            "sequence sets differ across {0} runs; refusing to silently "
            "intersect.".format(len(sources)),
            "  union holds {0} names, {1} are present in every run".format(
                len(union), len(intersection)
            ),
        ]
        for key in sources:
            others = set()
            for other_key, other_names in name_sets.items():
                if other_key != key:
                    others |= other_names
            absent_here = sorted(union - name_sets[key])
            only_here = sorted(name_sets[key] - others)
            lines.append(
                "  {0} [{1}]: {2} sequences, missing {3} ({4}), unique to it {5} ({6})".format(
                    key,
                    labels.get(key, key),
                    len(name_sets[key]),
                    len(absent_here),
                    _preview(absent_here),
                    len(only_here),
                    _preview(only_here),
                )
            )
        raise PairedBootstrapError("\n".join(lines))
    return sorted(intersection)


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


def discover_metrics_across(sources, names):
    """Classify every key seen in any source, for an N-way comparison.

    A key is analysable when it appears in **every** source and every non-null
    value everywhere is a real number. Anything else is reported by name so a
    dropped metric is visible rather than silent.
    """
    seen = {}
    non_numeric = set()
    for key, source in sources.items():
        keys = set()
        for name in names:
            for metric, value in source[name].items():
                keys.add(metric)
                if value is not None and not _is_number(value):
                    non_numeric.add(metric)
        seen[key] = keys
    everywhere = None
    anywhere = set()
    for keys in seen.values():
        anywhere |= keys
        everywhere = set(keys) if everywhere is None else (everywhere & keys)
    everywhere = everywhere or set()
    partial = {}
    for metric in sorted(anywhere - everywhere):
        partial[metric] = {
            "present_in": sorted(key for key in sources if metric in seen[key]),
            "absent_from": sorted(key for key in sources if metric not in seen[key]),
        }
    return {
        "analyzed": sorted(everywhere - non_numeric),
        "excluded_non_numeric": sorted(non_numeric & everywhere),
        "excluded_not_in_every_cell": partial,
    }


def _column(source, names, metric):
    out = np.full(len(names), np.nan, dtype=np.float64)
    for position, name in enumerate(names):
        value = source[name].get(metric)
        if _is_number(value):
            out[position] = float(value)
    return out


def _summarize_column(delta, index):
    """Bootstrap the mean of one per-sequence contrast column.

    ``delta`` holds one number per paired sequence; ``index`` is the shared
    ``(replicates, n)`` resample index matrix, drawn once by the caller before
    any metric or contrast is touched and reused verbatim for all of them.

    A non-finite entry (a cell the evaluator did not score, or an infinity)
    becomes NaN and is excluded from every replicate mean via ``np.nanmean``:
    the sequence never contributes to this contrast. Because a contrast column
    is built by arithmetic over the cells it needs, NaN propagation already
    restricts it to the complete-case subset of exactly those cells.
    """
    replicates = index.shape[0]
    delta = np.asarray(delta, dtype=np.float64).copy()
    delta[~np.isfinite(delta)] = np.nan
    used = int(np.count_nonzero(~np.isnan(delta)))
    entry = {
        "n": len(delta),
        "n_used": used,
        "n_dropped_nonfinite": len(delta) - used,
    }
    if used == 0:
        entry.update({
            "difference": None,
            "ci_low": None,
            "ci_high": None,
            "crosses_zero": None,
            "nan_replicates": replicates,
            "defined": False,
        })
        return entry
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
            difference = float(np.nanmean(delta))
    entry.update({
        "difference": difference,
        "ci_low": low,
        "ci_high": high,
        "crosses_zero": not bool(low > 0.0 or high < 0.0),
        "nan_replicates": nan_replicates,
        "defined": True,
    })
    return entry


def _draw_index(names, seed, replicates):
    """Draw the one shared resample index matrix and its digest."""
    if replicates < 1:
        raise PairedBootstrapError("replicates must be >= 1, got {0}".format(replicates))
    if not names:
        raise PairedBootstrapError("no sequences to bootstrap")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(names), size=(replicates, len(names)))
    digest = hashlib.sha256(
        np.ascontiguousarray(index, dtype=np.int64).tobytes()
    ).hexdigest()
    return index, digest


def paired_bootstrap(metrics_a, metrics_b, names, seed=DEFAULT_SEED,
                     replicates=DEFAULT_REPLICATES):
    """Bootstrap ``b - a`` for every analysable metric on one shared index matrix.

    The index matrix is drawn once, before any metric is touched, and reused
    verbatim for all of them. Non-finite paired differences (either side
    missing, or an infinity) are excluded from every replicate mean via
    ``np.nanmean``, so a sequence only contributes where the pair exists.
    """
    coverage = discover_metrics(metrics_a, metrics_b, names)
    index, index_digest = _draw_index(names, seed, replicates)
    results = {}
    for metric in coverage["analyzed"]:
        column_a = _column(metrics_a, names, metric)
        column_b = _column(metrics_b, names, metric)
        summary = _summarize_column(column_b - column_a, index)
        entry = {
            "n_pairs": summary["n"],
            "n_pairs_used": summary["n_used"],
            "n_pairs_dropped_nonfinite": summary["n_dropped_nonfinite"],
        }
        if not summary["defined"]:
            entry.update({
                "mean_a": None, "mean_b": None, "mean_delta": None,
                "ci_low": None, "ci_high": None, "significant": None,
                "direction": "undefined", "nan_replicates": replicates,
                "note": "no sequence has a finite value on both sides",
            })
            results[metric] = entry
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore"):
                mean_a = float(np.nanmean(column_a))
                mean_b = float(np.nanmean(column_b))
        significant = not summary["crosses_zero"]
        entry.update({
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_delta": summary["difference"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "significant": significant,
            "direction": ("b_greater" if significant and summary["ci_low"] > 0.0
                          else "b_lower" if significant
                          else "inconclusive"),
            "nan_replicates": summary["nan_replicates"],
        })
        results[metric] = entry
    return results, coverage, index_digest


def _contrast_specs(factor1_name, factor2_name):
    """Every contrast the 2x2 reports, with its arithmetic and its sign rule.

    Each entry is ``(key, form, cells_required, positive_means, null_hypothesis)``.
    A contrast appears exactly once here even when it belongs to two reporting
    groups (``a10_minus_a00`` is both the a10-vs-a00 simple effect and the
    ``factor1``-at-``factor2``-level-0 component of the main effect), so the
    groups below are indices into this list rather than duplicated numbers.
    """
    f1, f2 = factor1_name, factor2_name
    return [
        (
            "a00_minus_a00",
            "a00 - a00",
            ("a00",),
            "nothing: this is the self-control row and must be exactly 0.0 "
            "with a degenerate interval [0.0, 0.0]",
            "trivially true; a non-zero value here means the machinery is broken",
        ),
        (
            "a10_minus_a00",
            "a10 - a00",
            ("a00", "a10"),
            "cell a10 ({0}=1, {1}=0) reports a LARGER value than the a00 "
            "reference cell".format(f1, f2),
            "{0} has no effect while {1} is held at level 0".format(f1, f2),
        ),
        (
            "a01_minus_a00",
            "a01 - a00",
            ("a00", "a01"),
            "cell a01 ({0}=0, {1}=1) reports a LARGER value than the a00 "
            "reference cell".format(f1, f2),
            "{0} has no effect while {1} is held at level 0".format(f2, f1),
        ),
        (
            "a11_minus_a00",
            "a11 - a00",
            ("a00", "a11"),
            "cell a11 (both factors at level 1) reports a LARGER value than "
            "the a00 reference cell",
            "the two factors together have no combined effect",
        ),
        (
            "a11_minus_a01",
            "a11 - a01",
            ("a01", "a11"),
            "cell a11 reports a LARGER value than cell a01, i.e. turning {0} "
            "on raises the metric while {1} is held at level 1".format(f1, f2),
            "{0} has no effect while {1} is held at level 1".format(f1, f2),
        ),
        (
            "a11_minus_a10",
            "a11 - a10",
            ("a10", "a11"),
            "cell a11 reports a LARGER value than cell a10, i.e. turning {0} "
            "on raises the metric while {1} is held at level 1".format(f2, f1),
            "{0} has no effect while {1} is held at level 1".format(f2, f1),
        ),
        (
            "main_effect_" + f1,
            "0.5 * ((a10 - a00) + (a11 - a01))",
            CELL_KEYS,
            "averaged over both levels of {1}, turning {0} on RAISES the "
            "metric".format(f1, f2),
            "{0} has no average effect across the levels of {1}".format(f1, f2),
        ),
        (
            "main_effect_" + f2,
            "0.5 * ((a01 - a00) + (a11 - a10))",
            CELL_KEYS,
            "averaged over both levels of {1}, turning {0} on RAISES the "
            "metric".format(f2, f1),
            "{0} has no average effect across the levels of {1}".format(f2, f1),
        ),
        (
            "interaction_{0}_x_{1}".format(f1, f2),
            "(a11 - a01) - (a10 - a00)",
            CELL_KEYS,
            "the effect of {0} is MORE POSITIVE when {1} is on than when it is "
            "off; equivalently the effect of {1} is more positive when {0} is "
            "on".format(f1, f2),
            "the two factors are additive: the effect of {0} is the same at "
            "both levels of {1}".format(f1, f2),
        ),
    ]


def _contrast_groups(factor1_name, factor2_name):
    f1, f2 = factor1_name, factor2_name
    return {
        "simple_effects_vs_a00": [
            "a00_minus_a00", "a10_minus_a00", "a01_minus_a00", "a11_minus_a00",
        ],
        "main_effects": ["main_effect_" + f1, "main_effect_" + f2],
        "interaction": ["interaction_{0}_x_{1}".format(f1, f2)],
        "component_effects": [
            "a10_minus_a00", "a11_minus_a01", "a01_minus_a00", "a11_minus_a10",
        ],
    }


def _contrast_columns(columns, factor1_name, factor2_name):
    """Build every contrast as a per-sequence column, before any resampling.

    The two interaction forms are deliberately built from *different*
    expression trees rather than one shared weight vector, so comparing them
    is a real check on the implementation (a swapped cell or a flipped sign
    would separate them by O(effect size)) and not a tautology.
    """
    a00, a10, a01, a11 = (columns[key] for key in CELL_KEYS)
    simple_10 = a10 - a00
    simple_01 = a01 - a00
    conditional_11_01 = a11 - a01
    conditional_11_10 = a11 - a10
    built = {
        "a00_minus_a00": a00 - a00,
        "a10_minus_a00": simple_10,
        "a01_minus_a00": simple_01,
        "a11_minus_a00": a11 - a00,
        "a11_minus_a01": conditional_11_01,
        "a11_minus_a10": conditional_11_10,
        "main_effect_" + factor1_name: 0.5 * (simple_10 + conditional_11_01),
        "main_effect_" + factor2_name: 0.5 * (simple_01 + conditional_11_10),
    }
    interaction_form_1 = conditional_11_01 - simple_10
    interaction_form_2 = conditional_11_10 - simple_01
    built["interaction_{0}_x_{1}".format(factor1_name, factor2_name)] = interaction_form_1
    return built, interaction_form_1, interaction_form_2


def _p3_entry(summary, replicates):
    """Render one bootstrapped column in the P3 reporting vocabulary."""
    entry = {
        "n": summary["n_used"],
        "n_sequences": summary["n"],
        "n_dropped_nonfinite": summary["n_dropped_nonfinite"],
    }
    if not summary["defined"]:
        entry.update({
            "difference": None,
            "ci": None,
            "crosses_zero": None,
            "significant": None,
            "direction": "undefined",
            "nan_replicates": replicates,
            "note": "no sequence has a finite value in every cell this contrast needs",
        })
        return entry
    crosses = summary["crosses_zero"]
    entry.update({
        "difference": summary["difference"],
        "ci": [summary["ci_low"], summary["ci_high"]],
        "crosses_zero": crosses,
        "significant": not crosses,
        "direction": ("inconclusive" if crosses
                      else "positive" if summary["ci_low"] > 0.0 else "negative"),
        "nan_replicates": summary["nan_replicates"],
    })
    return entry


def _finite_max_abs(*columns):
    scale = 0.0
    for column in columns:
        finite = column[np.isfinite(column)]
        if finite.size:
            scale = max(scale, float(np.max(np.abs(finite))))
    return scale


def _identity_gap(first, second):
    """Largest absolute discrepancy between two columns, ignoring shared NaNs."""
    both = np.isfinite(first) & np.isfinite(second)
    if not np.any(both):
        return 0.0
    return float(np.max(np.abs(first[both] - second[both])))


def factorial_bootstrap(cells, names, seed=DEFAULT_SEED,
                        replicates=DEFAULT_REPLICATES,
                        factor1_name=DEFAULT_FACTOR1_NAME,
                        factor2_name=DEFAULT_FACTOR2_NAME):
    """Bootstrap the full 2x2 on one shared resample index matrix.

    ``cells`` maps ``a00``/``a10``/``a01``/``a11`` to that cell's per-sequence
    metrics dict. Every contrast, for every metric, is resampled with the
    **same** index matrix drawn once here, so the intervals are mutually
    comparable and the interaction interval carries the correct covariance
    between the four cells. Deriving these numbers from four separately
    bootstrapped pairwise reports would not: independent resamples destroy that
    covariance and the interaction interval would be wrong.

    Returns ``(contrasts, cell_summaries, coverage, index_digest, identity)``.
    """
    if factor1_name == factor2_name:
        raise PairedBootstrapError(
            "the two factor names must differ, both are {0!r}".format(factor1_name)
        )
    missing = [key for key in CELL_KEYS if key not in cells]
    if missing:
        raise PairedBootstrapError(
            "missing cells {0}; a 2x2 needs all of {1}".format(
                ", ".join(missing), ", ".join(CELL_KEYS))
        )
    coverage = discover_metrics_across(
        {key: cells[key] for key in CELL_KEYS}, names
    )
    index, index_digest = _draw_index(names, seed, replicates)
    specs = _contrast_specs(factor1_name, factor2_name)
    contrasts = {}
    for key, form, required, positive_means, null_hypothesis in specs:
        contrasts[key] = {
            "form": form,
            "cells_required_finite": list(required),
            "positive_means": positive_means,
            "null_hypothesis": null_hypothesis,
            "metrics": {},
        }
    cell_summaries = {}
    identity_per_metric = {}
    for metric in coverage["analyzed"]:
        columns = {key: _column(cells[key], names, metric) for key in CELL_KEYS}
        built, form_1, form_2 = _contrast_columns(columns, factor1_name, factor2_name)
        complete = np.ones(len(names), dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for key in CELL_KEYS:
                complete &= np.isfinite(columns[key])
            summary = {"complete_case_n": int(np.count_nonzero(complete))}
            for key in CELL_KEYS:
                finite = columns[key][np.isfinite(columns[key])]
                summary[key] = {
                    "mean": float(np.mean(finite)) if finite.size else None,
                    "n_observed": int(finite.size),
                }
        cell_summaries[metric] = summary
        for key, column in built.items():
            contrasts[key]["metrics"][metric] = _p3_entry(
                _summarize_column(column, index), replicates
            )
        gap = _identity_gap(form_1, form_2)
        tolerance = IDENTITY_ATOL + IDENTITY_RTOL * _finite_max_abs(
            *(columns[key] for key in CELL_KEYS)
        )
        second = _p3_entry(_summarize_column(form_2, index), replicates)
        first = contrasts["interaction_{0}_x_{1}".format(
            factor1_name, factor2_name)]["metrics"][metric]
        ci_gap = 0.0
        if first["difference"] is not None and second["difference"] is not None:
            ci_gap = max(
                abs(first["difference"] - second["difference"]),
                abs(first["ci"][0] - second["ci"][0]),
                abs(first["ci"][1] - second["ci"][1]),
            )
        identity_per_metric[metric] = {
            "column_max_abs_difference": gap,
            "ci_max_abs_difference": ci_gap,
            "tolerance": tolerance,
            "agrees": bool(gap <= tolerance and ci_gap <= tolerance),
        }
    disagreeing = sorted(
        metric for metric, record in identity_per_metric.items()
        if not record["agrees"]
    )
    identity = {
        "form_1": "(a11 - a01) - (a10 - a00)",
        "form_2": "(a11 - a10) - (a01 - a00)",
        "claim": "the two algebraic forms of the interaction are the same "
                 "linear functional of the four cells and must agree to "
                 "floating-point tolerance; a disagreement means a cell or a "
                 "sign is wrong in this implementation",
        "reported_form": "form_1",
        "tolerance_rule": "atol {0:g} + rtol {1:g} * max |cell value| for that "
                          "metric".format(IDENTITY_ATOL, IDENTITY_RTOL),
        "per_metric": identity_per_metric,
        "max_column_difference": max(
            [record["column_max_abs_difference"]
             for record in identity_per_metric.values()] or [0.0]),
        "max_ci_difference": max(
            [record["ci_max_abs_difference"]
             for record in identity_per_metric.values()] or [0.0]),
        "all_agree": not disagreeing,
        "disagreeing_metrics": disagreeing,
    }
    if disagreeing:
        raise PairedBootstrapError(
            "the two algebraic forms of the interaction disagree on {0} "
            "metric(s): {1}. (a11-a01)-(a10-a00) and (a11-a10)-(a01-a00) are "
            "the same quantity, so this is an implementation error, not a "
            "property of the data. Worst column gap {2:g}, worst CI gap "
            "{3:g}.".format(
                len(disagreeing), ", ".join(disagreeing),
                identity["max_column_difference"], identity["max_ci_difference"])
        )
    return contrasts, cell_summaries, coverage, index_digest, identity


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
        "mode": "pairwise",
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


def run_factorial(targets, results_root=DEFAULT_RESULTS_ROOT, seed=DEFAULT_SEED,
                  replicates=DEFAULT_REPLICATES,
                  expected_sequences=DEFAULT_EXPECTED_SEQUENCES,
                  labels=None, factor1_name=DEFAULT_FACTOR1_NAME,
                  factor2_name=DEFAULT_FACTOR2_NAME, warn=None):
    """Resolve, pair across four cells, bootstrap the 2x2 and package the report.

    ``targets`` maps ``a00``/``a10``/``a01``/``a11`` to a run id, run directory
    or explicit ``per_sequence_metrics.json`` path.
    """
    warn = warn if warn is not None else (lambda message: None)
    missing = [key for key in CELL_KEYS if key not in targets]
    if missing:
        raise PairedBootstrapError(
            "missing cells {0}; a 2x2 needs all of {1}".format(
                ", ".join(missing), ", ".join(CELL_KEYS))
        )
    unknown = sorted(set(targets) - set(CELL_KEYS))
    if unknown:
        raise PairedBootstrapError(
            "unknown cell key(s) {0}; expected exactly {1}".format(
                ", ".join(unknown), ", ".join(CELL_KEYS))
        )
    labels = dict(labels or {})
    for key in CELL_KEYS:
        labels.setdefault(key, str(targets[key]))
    paths = {key: resolve_per_sequence_path(targets[key], results_root)
             for key in CELL_KEYS}
    duplicates = {}
    for key in CELL_KEYS:
        duplicates.setdefault(paths[key], []).append(key)
    for path, keys in duplicates.items():
        if len(keys) > 1:
            warn("cells {0} resolve to the same file {1}; every contrast "
                 "between them is exactly zero by construction and the design "
                 "is not a real 2x2".format("/".join(keys), path))
    cells, provenance = {}, {}
    for key in CELL_KEYS:
        cells[key], provenance[key] = load_per_sequence(paths[key])
        provenance[key]["label"] = labels[key]
        provenance[key]["cell"] = key
    names = pair_sequence_names_across(cells, labels)
    matches_expected = None
    if expected_sequences is not None and expected_sequences > 0:
        matches_expected = len(names) == expected_sequences
        if not matches_expected:
            warn("paired {0} sequences but expected {1}; continuing".format(
                len(names), expected_sequences))
    contrasts, cell_summaries, coverage, index_digest, identity = factorial_bootstrap(
        cells, names, seed=seed, replicates=replicates,
        factor1_name=factor1_name, factor2_name=factor2_name,
    )
    for metric, record in sorted(coverage["excluded_not_in_every_cell"].items()):
        warn("metric {0!r} is not present in every cell (missing from {1}); "
             "not analysed".format(metric, ", ".join(record["absent_from"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/paired_bootstrap.py",
        "mode": "factorial_2x2",
        "design": {
            "cell_naming": "a<factor1 level><factor2 level>",
            "factor1": {
                "name": factor1_name,
                "level0_cells": ["a00", "a01"],
                "level1_cells": ["a10", "a11"],
            },
            "factor2": {
                "name": factor2_name,
                "level0_cells": ["a00", "a10"],
                "level1_cells": ["a01", "a11"],
            },
            "reference_cell": "a00",
            "contrast_groups": _contrast_groups(factor1_name, factor2_name),
            "group_note": "a contrast is listed once under 'contrasts' and may "
                          "appear in two groups; a10_minus_a00 and "
                          "a01_minus_a00 are both simple effects against a00 "
                          "and components of a main effect",
        },
        "sign_convention": {
            "units": "every reported quantity is a difference in the metric's "
                     "own units, never a ratio or a percentage",
            "direction": "positive means the first-named side of the stated "
                         "form is LARGER; each contrast repeats this in its "
                         "own 'positive_means' field",
            "polarity_note": "this tool does not encode metric polarity, so "
                             "'larger' is not 'better': for penetration and "
                             "error metrics (hand_pen_loss_omomo, "
                             "human_pen_loss_infbagel, mpjpe, end_obj_trans_err) "
                             "lower is better, for contact metrics "
                             "(contact_f1, contact_recall) higher is better",
            "significance_rule": "'crosses_zero' is false exactly when the "
                                 "interval strictly excludes zero; a degenerate "
                                 "interval at [0.0, 0.0] touches zero and is "
                                 "therefore reported as crossing it",
        },
        "bootstrap": {
            "paired": True,
            "resample_unit": "sequence",
            "replicates": replicates,
            "seed": seed,
            "percentiles": [LOWER_PERCENTILE, UPPER_PERCENTILE],
            "percentile_method": "linear",
            "shared_resample_index_matrix": True,
            "shared_across": "all four cells, all contrasts and all metrics",
            "resample_index_sha256": index_digest,
            "per_replicate_contrasts": True,
            "covariance_note": "each contrast is formed per sequence first and "
                               "then resampled, so main effects and the "
                               "interaction carry the correct covariance "
                               "between the four cells; combining separately "
                               "bootstrapped pairwise reports would not",
            "significance_rule": "confidence interval strictly excludes zero",
            "missing_value_rule": (
                "a contrast column is built by arithmetic over the cells it "
                "needs, so NaN propagation restricts it to the complete-case "
                "subset of exactly those cells; excluded via np.nanmean"
            ),
            "cell_mean_note": "cell means under 'cell_summaries' are taken over "
                              "each cell's own observed sequences, so a "
                              "difference of two cell means need not equal the "
                              "corresponding contrast when some cell has "
                              "missing observations; the contrast is the "
                              "complete-case quantity and is authoritative",
        },
        "inputs": {key: provenance[key] for key in CELL_KEYS},
        "sequences": {
            "paired_count": len(names),
            "expected_count": expected_sequences,
            "matches_expected": matches_expected,
            "order": "sorted-by-name",
            "paired_across": list(CELL_KEYS),
        },
        "metric_coverage": coverage,
        "cell_summaries": cell_summaries,
        "interaction_identity_check": identity,
        "contrasts": contrasts,
    }


def _factorial_metric_rows(contrast):
    rows = [("metric", "n", "difference", "ci_low", "ci_high", "crosses_zero")]
    for metric in sorted(contrast["metrics"]):
        entry = contrast["metrics"][metric]
        if entry["difference"] is None:
            rows.append((metric, str(entry["n"]), "-", "-", "-", "n/a"))
            continue
        rows.append((
            metric,
            str(entry["n"]),
            "{0:+.6g}".format(entry["difference"]),
            "{0:+.6g}".format(entry["ci"][0]),
            "{0:+.6g}".format(entry["ci"][1]),
            "yes" if entry["crosses_zero"] else "NO",
        ))
    return rows


def _render_rows(rows, indent="  "):
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for position, row in enumerate(rows):
        lines.append(indent + "  ".join(
            cell.ljust(width) if column == 0 else cell.rjust(width)
            for column, (cell, width) in enumerate(zip(row, widths))
        ).rstrip())
        if position == 0:
            lines.append(indent + "  ".join("-" * width for width in widths))
    return lines


def format_factorial_table(report):
    """Render the 2x2 report as fixed-width tables for human review."""
    design = report["design"]
    factor1 = design["factor1"]["name"]
    factor2 = design["factor2"]["name"]
    identity = report["interaction_identity_check"]
    lines = [
        "2x2 factorial paired sequence-level bootstrap",
        "",
        "cells, named a<{0} level><{1} level>:".format(factor1, factor2),
    ]
    cell_rows = [("cell", factor1, factor2, "label", "sequences")]
    for key in CELL_KEYS:
        cell_rows.append((
            key, key[1], key[2],
            str(report["inputs"][key]["label"]),
            str(report["inputs"][key]["sequence_count"]),
        ))
    lines.extend(_render_rows(cell_rows))
    lines.extend([
        "",
        "{0} sequences paired BY NAME across all four cells, {1} replicates, "
        "seed {2}, {3}/{4} percentiles.".format(
            report["sequences"]["paired_count"],
            report["bootstrap"]["replicates"],
            report["bootstrap"]["seed"],
            *report["bootstrap"]["percentiles"],
        ),
        "ONE shared resample index matrix (sha256 {0}) is reused for every "
        "cell,".format(report["bootstrap"]["resample_index_sha256"][:16]),
        "contrast and metric, so the intervals are mutually comparable and the",
        "interaction interval carries the correct covariance between cells.",
        "",
        "SIGN CONVENTION: " + report["sign_convention"]["direction"],
        "POLARITY: " + report["sign_convention"]["polarity_note"],
        "SIGNIFICANCE: crosses_zero=NO means the interval strictly excludes zero.",
        "",
        "interaction identity {0} vs {1}: max column gap {2:.3g}, max CI gap "
        "{3:.3g} -> {4}".format(
            identity["form_1"], identity["form_2"],
            identity["max_column_difference"], identity["max_ci_difference"],
            "AGREE" if identity["all_agree"] else "DISAGREE",
        ),
        "",
        "cell means (each over that cell's own observed sequences):",
    ])
    mean_rows = [("metric",) + CELL_KEYS + ("complete_case_n",)]
    for metric in sorted(report["cell_summaries"]):
        summary = report["cell_summaries"][metric]
        mean_rows.append((metric,) + tuple(
            "-" if summary[key]["mean"] is None
            else "{0:.6g}".format(summary[key]["mean"])
            for key in CELL_KEYS
        ) + (str(summary["complete_case_n"]),))
    lines.extend(_render_rows(mean_rows))
    seen = []
    for group in ("main_effects", "interaction", "simple_effects_vs_a00",
                  "component_effects"):
        lines.extend(["", "=== {0} ===".format(group)])
        for key in design["contrast_groups"][group]:
            contrast = report["contrasts"][key]
            note = " (already shown above)" if key in seen else ""
            seen.append(key)
            lines.extend([
                "",
                "{0} = {1}{2}".format(key, contrast["form"], note),
                "  positive means: " + contrast["positive_means"],
                "  null hypothesis: " + contrast["null_hypothesis"],
            ])
            lines.extend(_render_rows(_factorial_metric_rows(contrast), indent="    "))
    return "\n".join(lines)


def _parse_cell_assignments(values, what):
    parsed = {}
    for item in values or []:
        if "=" not in item:
            raise PairedBootstrapError(
                "{0} entry {1!r} must be KEY=VALUE with KEY one of {2}".format(
                    what, item, ", ".join(CELL_KEYS))
            )
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key not in CELL_KEYS:
            raise PairedBootstrapError(
                "{0} key {1!r} is not one of {2}".format(
                    what, key, ", ".join(CELL_KEYS))
            )
        if key in parsed:
            raise PairedBootstrapError(
                "{0} key {1!r} given twice".format(what, key)
            )
        if not value:
            raise PairedBootstrapError(
                "{0} key {1!r} has an empty value".format(what, key)
            )
        parsed[key] = value
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Paired sequence-level bootstrap CIs between evaluation runs: "
                    "either one A-vs-B pair (--a/--b) or a 2x2 factorial "
                    "(--factorial).",
        epilog="Every metric present in every input file is reported; there is "
               "no flag to restrict the metric set. Factorial example: "
               "--factorial a00=RUN_H0D0 a10=RUN_H1D0 a01=RUN_H0D1 "
               "a11=RUN_H1D1 --factor1-name hinge --factor2-name detach",
    )
    parser.add_argument("--a", default=None,
                        help="pairwise mode: baseline run id, run directory or "
                             "per_sequence_metrics.json path")
    parser.add_argument("--b", default=None,
                        help="pairwise mode: comparison run id, run directory or "
                             "per_sequence_metrics.json path")
    parser.add_argument("--label-a", default=None, help="name for a in the output")
    parser.add_argument("--label-b", default=None, help="name for b in the output")
    parser.add_argument("--factorial", nargs=4, metavar="CELL=TARGET", default=None,
                        help="factorial mode: the four cells as a00=..., a10=..., "
                             "a01=..., a11=... in any order, where a<i><j> has "
                             "factor 1 at level i and factor 2 at level j")
    parser.add_argument("--cell-label", action="append", metavar="CELL=NAME",
                        default=None,
                        help="factorial mode: readable name for one cell "
                             "(repeatable); defaults to the target string")
    parser.add_argument("--factor1-name", default=DEFAULT_FACTOR1_NAME,
                        help="factorial mode: name of the first factor "
                             "(default: %(default)s)")
    parser.add_argument("--factor2-name", default=DEFAULT_FACTOR2_NAME,
                        help="factorial mode: name of the second factor "
                             "(default: %(default)s)")
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


def _dispatch(args, warn):
    """Validate the mode selection and produce the report plus its renderer."""
    pairwise = args.a is not None or args.b is not None
    if pairwise and args.factorial is not None:
        raise PairedBootstrapError(
            "--a/--b and --factorial are different modes; pass one or the other"
        )
    if not pairwise and args.factorial is None:
        raise PairedBootstrapError(
            "nothing to compare: pass --a and --b for a pairwise contrast, or "
            "--factorial a00=... a10=... a01=... a11=... for the 2x2"
        )
    if pairwise:
        if args.a is None or args.b is None:
            raise PairedBootstrapError("pairwise mode needs both --a and --b")
        if args.cell_label is not None:
            raise PairedBootstrapError("--cell-label only applies to --factorial")
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
        return report, format_table
    for name in ("label_a", "label_b"):
        if getattr(args, name) is not None:
            raise PairedBootstrapError(
                "--{0} only applies to the pairwise mode; use --cell-label "
                "CELL=NAME".format(name.replace("_", "-"))
            )
    targets = _parse_cell_assignments(args.factorial, "--factorial")
    missing = [key for key in CELL_KEYS if key not in targets]
    if missing:
        raise PairedBootstrapError(
            "--factorial is missing cell(s) {0}; all of {1} are required".format(
                ", ".join(missing), ", ".join(CELL_KEYS))
        )
    report = run_factorial(
        targets,
        results_root=Path(args.results_root),
        seed=args.seed,
        replicates=args.replicates,
        expected_sequences=args.expected_sequences,
        labels=_parse_cell_assignments(args.cell_label, "--cell-label"),
        factor1_name=args.factor1_name,
        factor2_name=args.factor2_name,
        warn=warn,
    )
    return report, format_factorial_table


def main(argv=None):
    args = build_parser().parse_args(argv)

    def warn(message):
        print("warning: " + message, file=sys.stderr)

    try:
        report, renderer = _dispatch(args, warn)
    except PairedBootstrapError as error:
        print("error: {0}".format(error), file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(renderer(report), file=sys.stderr)
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
