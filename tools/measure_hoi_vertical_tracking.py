#!/usr/bin/env python3
"""Measure cross-sequence HOI pelvis-height tracking in Z-up exports.

This CPU-only diagnostic is deliberately separate from the official evaluator.
It reads already-produced CHOIS prediction and ground-truth NPZ directories,
fits model pelvis height against ground-truth pelvis height over fixed spans,
and optionally compares a teacher-forced arm with one paired bootstrap stream.
It does not load a checkpoint, instantiate a model, or run inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import chois_evaluator


class TrackingError(RuntimeError):
    """Raised when the tracking probe cannot establish its input contract."""


EXECUTION_CONTRACT = {
    "device": "cpu",
    "requires_gpu": False,
    "requires_checkpoint": False,
    "requires_model_inference": False,
    "requires_training": False,
    "writes_inside_run_directory": False,
    "read_only_inputs": True,
}

SPAN_DEFINITIONS = {
    "frame0": (0, 1),
    "w1": (0, 42),
    "w2": (42, 84),
    "w3": (84, 126),
    "f0": (0, 1),
    "f42": (42, 43),
    "f84": (84, 85),
}
SPAN_ALIASES = {"f0": "frame0"}
# Own-history spans: the level a window's output is regressed onto is the level
# in the 2 history frames it actually consumed, taken from the MODEL's own
# output rather than from ground truth.  The export keeps keyframes 2..15 of
# each 16-keyframe window, so keyframe k lands at within-window dense offset
# 3*(k-2); the 2 history keyframes 14 and 15 are therefore at offsets 36 and 39
# of the PREVIOUS window's 42-frame span, i.e. global dense 42*s+36 and 42*s+39
# for the window starting at 42*(s+1).  Secondary statistic, NO verdict
# authority: it measures how much the model shrinks whatever level it is given,
# which is the only channel that reaches the "intrinsic per-window contraction"
# hypothesis.  This definition is new here; it does NOT reproduce, and does not
# claim to reproduce, two earlier own-history slopes whose script was lost.
OWN_HISTORY_SPANS = {
    "w2": {"history_frames": (36, 39), "output": (42, 84)},
    "w3": {"history_frames": (78, 81), "output": (84, 126)},
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def _sha256_sequence_names(names: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{name}\n" for name in sorted(names)).encode("utf-8")).hexdigest()


def _load_pair(predictions: Path, ground_truth: Path) -> Dict[str, Any]:
    """Validate identity with the shared evaluator contract and load arrays."""
    pair = chois_evaluator.validate_pair(predictions.resolve(), ground_truth.resolve())
    names = sorted(pair["sequences"])
    prediction_arrays = _load_arrays(predictions.resolve(), pair["sequences"])
    truth_metadata, _ = chois_evaluator.read_npz_directory(ground_truth.resolve())
    truth_arrays = _load_arrays(ground_truth.resolve(), truth_metadata)
    return {
        "predictions": prediction_arrays,
        "ground_truth": truth_arrays,
        "sequence_names": names,
        "sequence_count": len(names),
        "sequence_names_sha256": _sha256_sequence_names(names),
        "input_contract": pair,
    }


def _load_arrays(directory: Path, metadata: Mapping[str, Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for name in sorted(metadata):
        file = directory / str(metadata[name]["file"])
        with np.load(file, allow_pickle=False) as value:
            actual_name = str(value["seq_name"].item())
            if actual_name != name:
                raise TrackingError(
                    f"{file.name} changed while loading: expected seq_name={name!r}, got {actual_name!r}"
                )
            joints = value["global_jpos"]
            if joints.shape != (126, 24, 3):
                raise TrackingError(
                    f"{file.name} global_jpos must have exact shape [126,24,3]; got {joints.shape}"
                )
            arrays[name] = np.asarray(joints, dtype=np.float64)
    return arrays


def _pelvis_heights(arm: Mapping[str, Any], span: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    start, stop = span
    names = arm["sequence_names"]
    prediction = np.asarray(
        [np.mean(arm["predictions"][name][start:stop, 0, 2]) for name in names], dtype=np.float64
    )
    truth = np.asarray(
        [np.mean(arm["ground_truth"][name][start:stop, 0, 2]) for name in names], dtype=np.float64
    )
    return prediction, truth


def _fit_statistics(prediction: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    if prediction.shape != truth.shape or prediction.ndim != 1:
        raise TrackingError("pelvis-height vectors must be one-dimensional and paired")
    if prediction.size < 2:
        raise TrackingError("pelvis-height tracking requires at least two sequences")
    centered_truth = truth - np.mean(truth)
    centered_prediction = prediction - np.mean(prediction)
    denominator = float(np.dot(centered_truth, centered_truth))
    if denominator == 0.0:
        raise TrackingError("ground-truth pelvis heights are constant; OLS slope is undefined")
    b = float(np.dot(centered_truth, centered_prediction) / denominator)
    a = float(np.mean(prediction) - b * np.mean(truth))
    correlation_denominator = float(
        np.sqrt(np.dot(centered_prediction, centered_prediction) * denominator)
    )
    r = float(np.dot(centered_prediction, centered_truth) / correlation_denominator) if correlation_denominator else 0.0
    return {
        "b": b,
        "a": a,
        "r": r,
        "pred_std": float(np.std(prediction)),
        "gt_std": float(np.std(truth)),
        "vertical_bias": float(np.mean(prediction - truth)),
    }


def _own_history_levels(
    arm: Mapping[str, Any], definition: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Model output level, and the model's OWN history level it was given."""
    names = arm["sequence_names"]
    history_frames = list(definition["history_frames"])
    start, stop = definition["output"]
    output_level = np.asarray(
        [np.mean(arm["predictions"][name][start:stop, 0, 2]) for name in names], dtype=np.float64
    )
    history_level = np.asarray(
        [np.mean(arm["predictions"][name][history_frames, 0, 2]) for name in names], dtype=np.float64
    )
    return output_level, history_level


def _own_history_statistics(arm: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name, definition in OWN_HISTORY_SPANS.items():
        output_level, history_level = _own_history_levels(arm, definition)
        # _fit_statistics regresses its first argument on its second, so the
        # history level is the regressor here, exactly as GT is elsewhere.
        statistics = _fit_statistics(output_level, history_level)
        result[name] = {
            "history_frames": list(definition["history_frames"]),
            "output_start": definition["output"][0],
            "output_stop": definition["output"][1],
            "b": statistics["b"],
            "a": statistics["a"],
            "r": statistics["r"],
            "output_std": statistics["pred_std"],
            "history_std": statistics["gt_std"],
            "level_shift": statistics["vertical_bias"],
            "verdict_authority": False,
        }
    return result


def _shared_resample_indices(count: int, replicates: int, seed: int) -> np.ndarray:
    if count < 2:
        raise TrackingError("paired bootstrap requires at least two sequences")
    random_state = np.random.RandomState(seed)
    indices = np.empty((replicates, count), dtype=np.int64)
    for replicate in range(replicates):
        indices[replicate] = random_state.randint(0, count, size=count)
    return indices


def _bootstrap_delta_b(
    baseline_prediction: np.ndarray,
    baseline_truth: np.ndarray,
    compare_prediction: np.ndarray,
    compare_truth: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    values = np.empty(indices.shape[0], dtype=np.float64)
    for replicate, selected in enumerate(indices):
        baseline = _fit_statistics(baseline_prediction[selected], baseline_truth[selected])["b"]
        comparison = _fit_statistics(compare_prediction[selected], compare_truth[selected])["b"]
        values[replicate] = comparison - baseline
    return values


def provenance() -> Dict[str, Any]:
    import torch

    try:
        commit = chois_evaluator.git_output(ROOT, "rev-parse", "HEAD")
        dirty = bool(chois_evaluator.git_output(ROOT, "status", "--porcelain"))
    except chois_evaluator.EvaluatorError:
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "tool_path": str(Path(__file__).resolve()),
        "tool_sha256": chois_evaluator.sha256_file(Path(__file__).resolve()),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--compare", nargs=2, metavar=("PREDICTIONS", "GROUND_TRUTH"),
        help="optional second arm (teacher-forced run) to compare with the primary arm",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise TrackingError(f"refusing to overwrite tracking-probe output: {output}")

    baseline = _load_pair(args.predictions, args.ground_truth)
    comparison = None
    if args.compare is not None:
        comparison = _load_pair(Path(args.compare[0]), Path(args.compare[1]))
        if set(baseline["sequence_names"]) != set(comparison["sequence_names"]):
            raise TrackingError(
                "comparison arms have different sequence identity sets; refusing paired bootstrap"
            )

    spans: Dict[str, Dict[str, Any]] = {}
    baseline_vectors: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, definition in SPAN_DEFINITIONS.items():
        prediction, truth = _pelvis_heights(baseline, definition)
        baseline_vectors[name] = (prediction, truth)
        spans[name] = {
            "start": definition[0],
            "stop": definition[1],
            **_fit_statistics(prediction, truth),
        }
        if name in SPAN_ALIASES:
            spans[name]["alias_of"] = SPAN_ALIASES[name]

    own_history = {"baseline": _own_history_statistics(baseline)}
    if comparison is not None:
        own_history["comparison"] = _own_history_statistics(comparison)

    bootstrap = None
    if comparison is not None:
        indices = _shared_resample_indices(
            baseline["sequence_count"], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
        )
        bootstrap = {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "method": "paired_shared_resample_indices",
            "percentiles": [2.5, 97.5],
            "percentile_method": "linear",
            "resample_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "resample_index_shape": list(indices.shape),
            "resample_index_dtype": str(indices.dtype),
        }
        for name, definition in SPAN_DEFINITIONS.items():
            compare_prediction, compare_truth = _pelvis_heights(comparison, definition)
            baseline_prediction, baseline_truth = baseline_vectors[name]
            baseline_b = spans[name]["b"]
            compare_b = _fit_statistics(compare_prediction, compare_truth)["b"]
            delta_series = _bootstrap_delta_b(
                baseline_prediction, baseline_truth,
                compare_prediction, compare_truth, indices,
            )
            interval = np.percentile(delta_series, [2.5, 97.5])
            spans[name]["comparison"] = {
                "baseline_b": baseline_b,
                "compare_b": compare_b,
                "delta_b": float(compare_b - baseline_b),
                "delta_b_ci95": [float(interval[0]), float(interval[1])],
            }

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution_contract": EXECUTION_CONTRACT,
        "provenance": provenance(),
        "coordinate_system": "Z-up",
        "vertical_axis": 2,
        "pelvis_joint_index": 0,
        "arms": {
            "baseline": {
                "predictions": str(args.predictions.resolve()),
                "ground_truth": str(args.ground_truth.resolve()),
                "sequence_count": baseline["sequence_count"],
                "sequence_names_sha256": baseline["sequence_names_sha256"],
            },
            **({
                "compare": {
                    "predictions": str(Path(args.compare[0]).resolve()),
                    "ground_truth": str(Path(args.compare[1]).resolve()),
                    "sequence_count": comparison["sequence_count"],
                    "sequence_names_sha256": comparison["sequence_names_sha256"],
                }
            } if comparison is not None else {}),
        },
        "span_definitions": {
            name: {"start": start, "stop": stop}
            for name, (start, stop) in SPAN_DEFINITIONS.items()
        },
        "span_aliases": dict(SPAN_ALIASES),
        "spans": spans,
        "own_history": own_history,
        "own_history_spans": {
            name: {k: list(v) if isinstance(v, tuple) else v for k, v in definition.items()}
            for name, definition in OWN_HISTORY_SPANS.items()
        },
        "bootstrap": bootstrap,
    }
    result["probe_sha256"] = chois_evaluator.sha256_file(Path(__file__).resolve())
    chois_evaluator.atomic_output(output, result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (TrackingError, chois_evaluator.EvaluatorError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "spans": sorted(result["spans"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
