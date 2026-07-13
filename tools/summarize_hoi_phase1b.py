#!/usr/bin/env python3
"""Summarize the preregistered three-seed Phase 1B HOI gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


SEEDS = (42, 123, 314)
T_95_DF2 = 4.302652729911275
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 42

NATIVE_MAPPING = {
    "object_goal_error_cm": "end_obj_trans_err",
    "pelvis_goal_error_cm": "xy_points_err",
    "feet_height": "feet_height",
    "foot_sliding": "foot_sliding",
    "contact_accuracy": "contact_acc",
    "contact_precision": "contact_precision",
    "contact_recall": "contact_recall",
    "contact_f1": "contact_f1",
    "contact_percent": "contact_percent",
    "gt_contact_percent": "gt_contact_percent",
    "mpjpe": "mpjpe",
    "translation_difference": "trans_dist",
    "object_translation_difference": "obj_trans_dist",
    "object_rotation_difference": "obj_rot_dist",
    "hand_object_penetration": "hand_pen_loss_omomo",
    "hand_penetration_ratio": "hand_pen_ratio",
    "human_object_penetration": "human_pen_loss_infbagel",
    "human_penetration_ratio": "human_pen_ratio",
}
NATIVE_GATE_KEYS = {
    "object_goal_error_cm", "pelvis_goal_error_cm", "foot_sliding",
    "contact_precision", "contact_recall", "contact_f1",
    "human_object_penetration", "human_penetration_ratio",
}
CHOIS_KEYS = (
    "FID", "MatchingScore", "R-Precision@1", "R-Precision@2", "R-Precision@3", "Diversity",
)
CHOIS_GATE_KEYS = {"FID", "R-Precision@1", "R-Precision@2", "R-Precision@3"}
HIGHER_IS_BETTER = {
    "contact_precision", "contact_recall", "contact_f1",
    "R-Precision@1", "R-Precision@2", "R-Precision@3",
}
PER_SEQUENCE_MAPPING = {
    "object_goal_error_cm": "end_obj_trans_err",
    "pelvis_goal_error_cm": "pelvis_goal_error_cm",
    "foot_sliding": "foot_sliding",
    "contact_precision": "contact_precision",
    "contact_recall": "contact_recall",
    "contact_f1": "contact_f1",
    "human_object_penetration": "human_pen_loss_infbagel",
    "human_penetration_ratio": "human_pen_ratio",
    "hand_object_penetration": "hand_pen_loss_omomo",
    "hand_penetration_ratio": "hand_pen_ratio",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def parse_seed_paths(values: Sequence[str], label: str) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"{label} must use SEED=PATH: {value}")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate {label} seed {seed}")
        path = Path(path_text).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[seed] = path
    if tuple(sorted(result)) != SEEDS:
        raise ValueError(f"{label} seeds must be {SEEDS}, got {tuple(sorted(result))}")
    return result


def seed_summary(values: Iterable[float]) -> dict:
    samples = [float(value) for value in values]
    if len(samples) != 3 or not all(math.isfinite(value) for value in samples):
        raise ValueError(f"three finite seed values required, got {samples}")
    mean = statistics.fmean(samples)
    standard_deviation = statistics.stdev(samples)
    half_width = T_95_DF2 * standard_deviation / math.sqrt(3)
    return {
        "values_by_seed": {str(seed): samples[index] for index, seed in enumerate(SEEDS)},
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "student_t_95_ci": [mean - half_width, mean + half_width],
    }


def paired_sequence_bootstrap(per_sequence: Mapping[int, dict]) -> dict:
    identifiers = [set(per_sequence[seed]["metrics"]) for seed in SEEDS]
    if any(ids != identifiers[0] for ids in identifiers[1:]) or len(identifiers[0]) != 438:
        raise ValueError("per-sequence inputs must contain the same 438 sequence IDs for all seeds")
    ordered = sorted(identifiers[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled_indices = rng.integers(0, len(ordered), size=(BOOTSTRAP_REPLICATES, len(ordered)))
    result = {}
    for output_key, input_key in PER_SEQUENCE_MAPPING.items():
        matrix = np.asarray([
            [per_sequence[seed]["metrics"][identifier].get(input_key) for identifier in ordered]
            for seed in SEEDS
        ], dtype=np.float64)
        finite_counts = np.isfinite(matrix).sum(axis=1)
        if np.any(finite_counts == 0):
            raise ValueError(f"no finite per-sequence values for {output_key}")
        paired_seed_mean = np.nanmean(matrix, axis=0)
        bootstrap = np.nanmean(paired_seed_mean[sampled_indices], axis=1)
        result[output_key] = {
            "paired_sequence_count": len(ordered),
            "finite_values_by_seed": {
                str(seed): int(finite_counts[index]) for index, seed in enumerate(SEEDS)
            },
            "mean": float(np.nanmean(paired_seed_mean)),
            "percentile_95_ci": [
                float(np.nanpercentile(bootstrap, 2.5)),
                float(np.nanpercentile(bootstrap, 97.5)),
            ],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--native", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--chois", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--per-sequence", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument(
        "--native-baseline", type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments/results/p0_hoi_table5_baseline_s42_20260712.json",
    )
    parser.add_argument(
        "--chois-baseline", type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments/results/p0_hoi_chois_matched_s42_20260712.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    paths = {
        "training": parse_seed_paths(args.training, "training"),
        "native": parse_seed_paths(args.native, "native"),
        "chois": parse_seed_paths(args.chois, "chois"),
        "per_sequence": parse_seed_paths(args.per_sequence, "per-sequence"),
    }
    documents = {
        kind: {seed: load(path) for seed, path in seed_paths.items()}
        for kind, seed_paths in paths.items()
    }
    for kind in documents:
        for seed in SEEDS:
            recorded = documents[kind][seed].get("seed")
            if recorded is None and kind == "chois":
                recorded = documents[kind][seed].get("runtime", {}).get("seed")
            if int(recorded) != seed:
                raise ValueError(f"{kind} seed mismatch for {seed}: {recorded}")

    native_baseline = load(args.native_baseline.resolve())["metrics"]
    chois_baseline = load(args.chois_baseline.resolve())["metrics"]
    summaries = {}
    gate = {}
    for output_key, input_key in NATIVE_MAPPING.items():
        summary = seed_summary(documents["native"][seed]["metrics"][input_key] for seed in SEEDS)
        summaries[output_key] = summary
        if output_key in NATIVE_GATE_KEYS:
            baseline = float(native_baseline[output_key])
            higher = output_key in HIGHER_IS_BETTER
            threshold = baseline * 0.95 if higher else baseline / 0.95
            passed = summary["mean"] >= threshold if higher else summary["mean"] <= threshold
            gate[output_key] = {
                "direction": "higher" if higher else "lower",
                "baseline": baseline,
                "threshold": threshold,
                "mean": summary["mean"],
                "passed": bool(passed),
            }
    for key in CHOIS_KEYS:
        summary = seed_summary(documents["chois"][seed]["metrics"][key] for seed in SEEDS)
        summaries[key] = summary
        if key in CHOIS_GATE_KEYS:
            baseline = float(chois_baseline[key])
            higher = key in HIGHER_IS_BETTER
            threshold = baseline * 0.95 if higher else baseline / 0.95
            passed = summary["mean"] >= threshold if higher else summary["mean"] <= threshold
            gate[key] = {
                "direction": "higher" if higher else "lower", "baseline": baseline,
                "threshold": threshold, "mean": summary["mean"], "passed": bool(passed),
            }

    evaluation_audits = {}
    for seed in SEEDS:
        native = documents["native"][seed]
        normalization = native.get("normalization_audit", {})
        contract = native.get("data_contract", {})
        checks = {
            "all_values_finite": normalization.get("nonfinite_values") == 0,
            "full_438_sequence_evaluation": native.get("sample_count") == 438,
            "scene_condition_absent": contract.get("scene_condition_loaded") is False,
            "no_short_sequence_windows": contract.get("short_sequence_windows") == 0,
            "complete_text_coverage": contract.get("text_coverage_rate") == 1.0,
            "chois_input_counts_438": (
                documents["chois"][seed].get("inputs", {}).get("predictions", {}).get("count") == 438
                and documents["chois"][seed].get("inputs", {}).get("ground_truth", {}).get("count") == 438
            ),
        }
        evaluation_audits[str(seed)] = {
            "checks": checks,
            "passed": all(checks.values()),
            "normalization": normalization,
            "data_contract": contract,
        }

    training = {
        "wall_seconds": seed_summary(documents["training"][seed]["wall_seconds"] for seed in SEEDS),
        "throughput_windows_per_second": seed_summary(
            documents["training"][seed]["throughput_windows_per_second"] for seed in SEEDS
        ),
        "peak_reserved_bytes": {
            str(seed): max(documents["training"][seed]["peak_memory_reserved_bytes_by_rank"])
            for seed in SEEDS
        },
        "terminal_validation_total": seed_summary(
            documents["training"][seed]["validation"][-1]["total"] for seed in SEEDS
        ),
    }
    result = {
        "schema_version": 1,
        "phase": "1B",
        "expert": "hoi",
        "seeds": list(SEEDS),
        "statistics_protocol": {
            "seed_summary": "mean, sample standard deviation, Student-t 95 percent CI with df=2",
            "paired_sequence_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "paired_sequence_bootstrap_seed": BOOTSTRAP_SEED,
            "gate": "higher mean >= 0.95*baseline; lower mean <= baseline/0.95",
        },
        "training": training,
        "metric_summaries": summaries,
        "paired_sequence_bootstrap": paired_sequence_bootstrap(documents["per_sequence"]),
        "gate_metrics": gate,
        "evaluation_audits": evaluation_audits,
        "all_required_audits_passed": all(item["passed"] for item in evaluation_audits.values()),
        "all_95_percent_gates_passed": all(item["passed"] for item in gate.values()),
        "inputs": {
            kind: {
                str(seed): {"path": str(path), "sha256": sha256(path)}
                for seed, path in seed_paths.items()
            }
            for kind, seed_paths in paths.items()
        },
        "baselines": {
            "native": {"path": str(args.native_baseline.resolve()), "sha256": sha256(args.native_baseline.resolve())},
            "chois": {"path": str(args.chois_baseline.resolve()), "sha256": sha256(args.chois_baseline.resolve())},
        },
    }
    result["phase_1b_evaluation_gate_passed"] = (
        result["all_95_percent_gates_passed"] and result["all_required_audits_passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "all_95_percent_gates_passed": result["all_95_percent_gates_passed"],
        "all_required_audits_passed": result["all_required_audits_passed"],
        "phase_1b_evaluation_gate_passed": result["phase_1b_evaluation_gate_passed"],
    }, indent=2))
    return 0 if result["phase_1b_evaluation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
