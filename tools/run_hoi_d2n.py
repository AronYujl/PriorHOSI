#!/usr/bin/env python3
"""Run the preregistered D2-N0 author-native checkpoint transfer audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RUN_ID = "p1-hoi-d2n-author-native-paired-s42-20260716"
CANDIDATES = ("source", "current", "balanced")
EXPECTED_CHECKPOINT_SHA256 = {
    "source": "48ec27a0c097eaa65b21f58b1d28f7cf64aa3b2c54e9b02eb2bc2f35688460e4",
    "current": "76e0d8811fc9f54caa6d4778e2fe9fcaee78fad98bee5f17570b47568f71e31f",
    "balanced": "ded9a12d4e85179c37e2457475649ccc614ef364b97eaebade0629b2c11d4ed8",
}
TEST_SCRIPT_SHA256 = "22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524"
EVAL_METRICS_SHA256 = "445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547"
EVAL_CONFIG_SHA256 = "89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73"
BASELINE_SHA256 = "76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6"
DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
DATA_AUDIT_SHA256 = "1deea6a724a3319d4c5654da682d7f51af7e5c93b119d159bd2b37ad258f627f"
SAMPLE_COUNT = 438
WINDOWS_PER_SAMPLE = 3
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 42
LOWER_IS_BETTER = (
    "end_obj_trans_err",
    "xy_points_err",
    "feet_height",
    "foot_sliding",
    "mpjpe",
    "trans_dist",
    "obj_trans_dist",
    "obj_rot_dist",
    "hand_pen_loss_omomo",
    "hand_pen_ratio",
    "human_pen_loss_infbagel",
    "human_pen_ratio",
)
HIGHER_IS_BETTER = (
    "contact_acc",
    "contact_precision",
    "contact_recall",
    "contact_f1",
    "contact_percent",
)
GATE_LOWER_METRICS = (
    "mpjpe",
    "end_obj_trans_err",
    "xy_points_err",
    "obj_trans_dist",
)
PER_SEQUENCE_KEYS = {
    "mpjpe": "mpjpe",
    "end_obj_trans_err": "end_obj_trans_err",
    "xy_points_err": "pelvis_goal_error_cm",
    "obj_trans_dist": "obj_trans_dist",
    "foot_sliding": "foot_sliding",
    "contact_f1": "contact_f1",
}
BASELINE_KEYS = {
    "end_obj_trans_err": "object_goal_error_cm",
    "xy_points_err": "pelvis_goal_error_cm",
    "feet_height": "feet_height",
    "foot_sliding": "foot_sliding",
    "contact_acc": "contact_accuracy",
    "contact_precision": "contact_precision",
    "contact_recall": "contact_recall",
    "contact_f1": "contact_f1",
    "contact_percent": "contact_percent",
    "gt_contact_percent": "gt_contact_percent",
    "mpjpe": "mpjpe",
    "trans_dist": "translation_difference",
    "obj_trans_dist": "object_translation_difference",
    "obj_rot_dist": "object_rotation_difference",
    "hand_pen_loss_omomo": "hand_object_penetration",
    "hand_pen_ratio": "hand_penetration_ratio",
    "human_pen_loss_infbagel": "human_object_penetration",
    "human_pen_ratio": "human_penetration_ratio",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def candidate_paths(output: Path, candidate: str) -> Dict[str, Path]:
    root = output / "candidates" / candidate
    return {
        "root": root,
        "evaluation": root / "evaluation",
        "aggregate": root / "evaluation" / "aggregate_metrics.json",
        "per_sequence": root / "evaluation" / "per_sequence_metrics.json",
        "hydra": root / "hydra",
        "resolved": output / f"resolved_{candidate}.yaml",
        "log": output / f"author_native_{candidate}.log",
    }


def candidate_overrides(
    candidate: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    output: Path,
    device: str,
) -> Sequence[str]:
    paths = candidate_paths(output, candidate)
    return (
        f"exp_name={RUN_ID}-{candidate}",
        f"ckpt_path={checkpoint}",
        f"checkpoint_sha256={checkpoint_sha256}",
        "checkpoint_weight_variant=online",
        f"device={device}",
        f"dataset.device={device}",
        f"sampler.pelvis.device={device}",
        f"hoi_output_dir={paths['evaluation']}",
        f"per_sequence_metrics_path={paths['per_sequence']}",
        f"hydra.run.dir={paths['hydra']}",
        "hoi_expected_sequences=438",
        "hoi_sequence_limit=null",
        "save_motion_params=false",
        "save_chois_eval_npz=false",
        "load_scene=false",
        "sample_type=diffusion",
    )


def candidate_command(
    python: Path,
    candidate: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    output: Path,
    device: str,
) -> Sequence[str]:
    return (
        str(python),
        str(REPO / "code/test_infbagel_hoi.py"),
        "--config-name",
        "config_eval_hoi_prior",
        *candidate_overrides(candidate, checkpoint, checkpoint_sha256, output, device),
    )


def resolve_candidate(
    python: Path,
    candidate: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    output: Path,
    device: str,
) -> str:
    command = (
        str(python),
        str(REPO / "code/test_infbagel_hoi.py"),
        "--config-name",
        "config_eval_hoi_prior",
        "--cfg",
        "job",
        "--resolve",
        *candidate_overrides(candidate, checkpoint, checkpoint_sha256, output, device),
    )
    environment = dict(os.environ)
    environment["ROOT_DIR"] = str(REPO)
    completed = subprocess.run(
        command,
        cwd=REPO / "code",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"D2-N {candidate} config resolution failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    if "${" in completed.stdout:
        raise ValueError(f"D2-N {candidate} config contains unresolved interpolation")
    return completed.stdout


def checkpoint_arguments(args) -> Dict[str, Path]:
    return {
        "source": args.source_checkpoint.resolve(),
        "current": args.current_checkpoint.resolve(),
        "balanced": args.balanced_checkpoint.resolve(),
    }


def checkpoint_hash_arguments(args) -> Dict[str, str]:
    return {
        "source": args.source_sha256,
        "current": args.current_sha256,
        "balanced": args.balanced_sha256,
    }


def verify_static_assets(baseline: Path) -> Dict[str, str]:
    paths = {
        "test_script": REPO / "code/test_infbagel_hoi.py",
        "eval_metrics": REPO / "code/eval_metrics.py",
        "eval_config": REPO / "code/config/config_eval_hoi_prior.yaml",
        "baseline": baseline,
    }
    expected = {
        "test_script": TEST_SCRIPT_SHA256,
        "eval_metrics": EVAL_METRICS_SHA256,
        "eval_config": EVAL_CONFIG_SHA256,
        "baseline": BASELINE_SHA256,
    }
    actual = {name: sha256_file(path.resolve()) for name, path in paths.items()}
    if actual != expected:
        raise ValueError(f"D2-N static asset hash mismatch: {actual} != {expected}")
    return actual


def resolved_config(args) -> Dict[str, object]:
    output = args.output.resolve()
    checkpoints = checkpoint_arguments(args)
    requested_hashes = checkpoint_hash_arguments(args)
    candidates = {}
    for candidate in CANDIDATES:
        paths = candidate_paths(output, candidate)
        content = paths["resolved"].read_text(encoding="utf-8")
        if "${" in content:
            raise ValueError(f"D2-N {candidate} archived config has unresolved interpolation")
        candidates[candidate] = {
            "checkpoint": str(checkpoints[candidate]),
            "checkpoint_sha256": requested_hashes[candidate],
            "weight_variant": "online",
            "command": list(candidate_command(
                args.python.resolve(),
                candidate,
                checkpoints[candidate],
                requested_hashes[candidate],
                output,
                args.device,
            )),
            "resolved_config_path": str(paths["resolved"]),
            "resolved_config_sha256": sha256_text(content),
            "aggregate_metrics_path": str(paths["aggregate"]),
            "per_sequence_metrics_path": str(paths["per_sequence"]),
            "log_path": str(paths["log"]),
        }
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-N0",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "repo_root": str(REPO),
        "python": str(args.python.resolve()),
        "device": args.device,
        "candidate_order": list(CANDIDATES),
        "candidates": candidates,
        "evaluation": {
            "protocol": "author-native-compute_metrics-through-production-HOIPrior-adapter",
            "official_test_sequences": SAMPLE_COUNT,
            "windows_per_sequence": WINDOWS_PER_SAMPLE,
            "diffusion_steps": 500,
            "checkpoint_weight_variant": "online",
            "same_seed_and_sequence_order": True,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "chois_used": False,
            "fid_rprecision_used": False,
            "fps_used_for_gate": False,
        },
        "assets": {
            "test_script": {
                "path": str((REPO / "code/test_infbagel_hoi.py").resolve()),
                "sha256": TEST_SCRIPT_SHA256,
            },
            "eval_metrics": {
                "path": str((REPO / "code/eval_metrics.py").resolve()),
                "sha256": EVAL_METRICS_SHA256,
                "matches_author_commit": True,
            },
            "eval_config": {
                "path": str((REPO / "code/config/config_eval_hoi_prior.yaml").resolve()),
                "sha256": EVAL_CONFIG_SHA256,
            },
            "phase0_baseline": {
                "path": str(args.baseline.resolve()),
                "sha256": BASELINE_SHA256,
            },
            "data_contract_sha256": DATA_CONTRACT_SHA256,
            "data_audit_sha256": DATA_AUDIT_SHA256,
        },
        "sampler_contract": {
            "future_gt": False,
            "stored_per_frame_bps": False,
            "rollout_bps": "recomputed_from_current_generated_object_pose",
            "production_equation_changed": False,
        },
        "released_checkpoint_loaded": False,
        "checkpoint_selection": False,
        "training_authorized": False,
        "training_started": False,
        "d2h1_started": False,
        "metrics_path": str(args.metrics.resolve()),
    }


def prepare_resolved_config(args) -> None:
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"D2-N output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_arguments(args)
    requested_hashes = checkpoint_hash_arguments(args)
    for candidate in CANDIDATES:
        content = resolve_candidate(
            args.python.resolve(),
            candidate,
            checkpoints[candidate],
            requested_hashes[candidate],
            output,
            args.device,
        )
        path = candidate_paths(output, candidate)["resolved"]
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"refusing to overwrite changed D2-N config {path}")
        else:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    exclusive_json(args.resolved_config.resolve(), resolved_config(args))


def per_sequence_arrays(
    records: Mapping[str, Mapping[str, object]],
    metric: str,
) -> np.ndarray:
    key = PER_SEQUENCE_KEYS[metric]
    values = []
    for sequence in sorted(records):
        value = records[sequence].get(key)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"D2-N nonfinite/missing {metric} for {sequence}: {value}")
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def paired_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, object]:
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("paired difference requires equal nonempty vectors")
    difference = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
    samples = difference[indices].mean(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "first_mean": float(first.mean()),
        "second_mean": float(second.mean()),
        "paired_mean_first_minus_second": float(difference.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def paired_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, object]:
    if numerator.shape != denominator.shape or numerator.ndim != 1 or not len(numerator):
        raise ValueError("paired ratio requires equal nonempty vectors")
    if denominator.mean() == 0.0:
        raise ValueError("paired ratio denominator mean is zero")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numerator), size=(replicates, len(numerator)))
    denominator_means = denominator[indices].mean(axis=1)
    if np.any(denominator_means == 0.0):
        raise ValueError("paired bootstrap denominator contains zero")
    samples = numerator[indices].mean(axis=1) / denominator_means
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "numerator_mean": float(numerator.mean()),
        "denominator_mean": float(denominator.mean()),
        "mean_ratio": float(numerator.mean() / denominator.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def validate_candidate_result(
    candidate: str,
    aggregate: Mapping[str, object],
    per_sequence: Mapping[str, object],
    checkpoint_sha256: str,
) -> None:
    if aggregate.get("sample_count") != SAMPLE_COUNT:
        raise ValueError(f"D2-N {candidate} aggregate sample-count mismatch")
    if aggregate.get("windows_per_sample") != WINDOWS_PER_SAMPLE:
        raise ValueError(f"D2-N {candidate} window-count mismatch")
    if aggregate.get("is_timing_subset") is not False:
        raise ValueError(f"D2-N {candidate} unexpectedly used a timing subset")
    checkpoint = aggregate.get("checkpoint", {})
    if checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError(f"D2-N {candidate} checkpoint hash mismatch in aggregate")
    if checkpoint.get("weight_variant") != "online" or checkpoint.get("weights") != "model":
        raise ValueError(f"D2-N {candidate} did not evaluate online model weights")
    if aggregate.get("chois_export", {}).get("enabled") is not False:
        raise ValueError(f"D2-N {candidate} unexpectedly exported CHOIS inputs")
    metrics = aggregate.get("metrics", {})
    expected_metrics = set(BASELINE_KEYS)
    if set(metrics) != expected_metrics:
        raise ValueError(
            f"D2-N {candidate} metric set mismatch: {set(metrics)} != {expected_metrics}"
        )
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError(f"D2-N {candidate} aggregate metrics are nonfinite")
    if per_sequence.get("sequence_count") != SAMPLE_COUNT:
        raise ValueError(f"D2-N {candidate} per-sequence count mismatch")
    records = per_sequence.get("metrics", {})
    if len(records) != SAMPLE_COUNT:
        raise ValueError(f"D2-N {candidate} per-sequence record mismatch")
    for metric in PER_SEQUENCE_KEYS:
        per_sequence_arrays(records, metric)
    normalization = aggregate.get("normalization_audit", {})
    if int(normalization.get("nonfinite_values", -1)) != 0:
        raise ValueError(f"D2-N {candidate} sampler normalization audit is nonfinite")


def comparisons(
    per_sequence: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Dict[str, object]:
    result = {}
    balanced = per_sequence["balanced"]
    balanced_names = sorted(balanced)
    for comparator in ("source", "current"):
        other = per_sequence[comparator]
        if sorted(other) != balanced_names:
            raise ValueError(f"D2-N sequence identities differ for balanced/{comparator}")
        lower = {}
        for metric in GATE_LOWER_METRICS:
            lower[metric] = paired_difference(
                per_sequence_arrays(other, metric),
                per_sequence_arrays(balanced, metric),
            )
        foot = paired_ratio(
            per_sequence_arrays(balanced, "foot_sliding"),
            per_sequence_arrays(other, "foot_sliding"),
        )
        contact = paired_difference(
            per_sequence_arrays(balanced, "contact_f1"),
            per_sequence_arrays(other, "contact_f1"),
        )
        result[f"balanced_vs_{comparator}"] = {
            "comparator_minus_balanced_lower_is_better": lower,
            "balanced_over_comparator_foot_sliding": foot,
            "balanced_minus_comparator_contact_f1": contact,
        }
    return result


def transfer_gate(
    candidate_results: Mapping[str, Mapping[str, object]],
    comparison_results: Mapping[str, Mapping[str, object]],
    *,
    hashes_exact: bool,
    sampler_contract_exact: bool,
) -> Dict[str, object]:
    checks = {
        "all_candidates_finite_and_complete": all(
            candidate_results[name]["sample_count"] == SAMPLE_COUNT
            and candidate_results[name]["finite"]
            for name in CANDIDATES
        ),
        "checkpoint_and_evaluator_hashes_exact": bool(hashes_exact),
        "sampler_future_gt_and_stored_bps_absent": bool(sampler_contract_exact),
    }
    comparator_checks = {}
    for comparator in ("source", "current"):
        value = comparison_results[f"balanced_vs_{comparator}"]
        metric_checks = {
            metric: (
                value["comparator_minus_balanced_lower_is_better"][metric][
                    "bootstrap_95_ci"
                ][0] > 0.0
            )
            for metric in GATE_LOWER_METRICS
        }
        foot_check = (
            value["balanced_over_comparator_foot_sliding"]["bootstrap_95_ci"][1] <= 1.10
        )
        contact_check = (
            value["balanced_minus_comparator_contact_f1"]["bootstrap_95_ci"][0] >= -0.02
        )
        comparator_checks[comparator] = {
            "lower_is_better_improvements": metric_checks,
            "foot_sliding_preserved": foot_check,
            "contact_f1_preserved": contact_check,
            "passed": all(metric_checks.values()) and foot_check and contact_check,
        }
    passed = all(checks.values()) and all(
        value["passed"] for value in comparator_checks.values()
    )
    return {
        "passed": passed,
        "classification": (
            "author-native-latest-transfer-positive-stop"
            if passed else "author-native-latest-transfer-negative-stop"
        ),
        "checks": checks,
        "comparators": comparator_checks,
        "training_authorized": False,
        "training_started": False,
        "checkpoint_selected": False,
        "d2h1_started": False,
    }


def baseline_comparison(
    metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
) -> Dict[str, float]:
    result = {}
    for metric, baseline_key in BASELINE_KEYS.items():
        denominator = float(baseline_metrics[baseline_key])
        result[metric] = float(metrics[metric]) / denominator if denominator else math.inf
    return result


def run_candidate(
    args,
    candidate: str,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> None:
    paths = candidate_paths(args.output.resolve(), candidate)
    if paths["root"].exists():
        raise FileExistsError(f"refusing to overwrite D2-N candidate output {paths['root']}")
    command = candidate_command(
        args.python.resolve(),
        candidate,
        checkpoint,
        checkpoint_sha256,
        args.output.resolve(),
        args.device,
    )
    environment = dict(os.environ)
    environment["ROOT_DIR"] = str(REPO)
    with paths["log"].open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO / "code",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"D2-N {candidate} author-native evaluation exited {completed.returncode}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--current-sha256", required=True)
    parser.add_argument("--balanced-checkpoint", type=Path, required=True)
    parser.add_argument("--balanced-sha256", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-N run id must be {RUN_ID}")
    if args.python.resolve() != Path(os.environ.get("INFBAGEL_PYTHON", "")).resolve():
        raise ValueError("D2-N requires the absolute INFBAGEL_PYTHON interpreter")
    requested_hashes = checkpoint_hash_arguments(args)
    if requested_hashes != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("D2-N requested checkpoint hashes do not match preregistration")
    if args.resolve_only:
        prepare_resolved_config(args)
        return

    archived = load_json(args.resolved_config.resolve())
    if archived != resolved_config(args):
        raise ValueError("D2-N runtime arguments do not match archived resolved config")
    checkpoints = checkpoint_arguments(args)
    static_hashes = verify_static_assets(args.baseline.resolve())
    actual_checkpoint_hashes = {
        name: sha256_file(path) for name, path in checkpoints.items()
    }
    if actual_checkpoint_hashes != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("D2-N checkpoint file hashes do not match preregistration")

    started = time.perf_counter()
    completed_candidates = []
    try:
        aggregates = {}
        records = {}
        raw_artifacts = {}
        for candidate in CANDIDATES:
            run_candidate(
                args,
                candidate,
                checkpoints[candidate],
                requested_hashes[candidate],
            )
            paths = candidate_paths(args.output.resolve(), candidate)
            aggregate = load_json(paths["aggregate"])
            per_sequence = load_json(paths["per_sequence"])
            validate_candidate_result(
                candidate,
                aggregate,
                per_sequence,
                requested_hashes[candidate],
            )
            aggregates[candidate] = aggregate
            records[candidate] = per_sequence["metrics"]
            raw_artifacts[candidate] = {
                "aggregate_metrics_path": str(paths["aggregate"]),
                "aggregate_metrics_sha256": sha256_file(paths["aggregate"]),
                "per_sequence_metrics_path": str(paths["per_sequence"]),
                "per_sequence_metrics_sha256": sha256_file(paths["per_sequence"]),
                "log_path": str(paths["log"]),
                "log_sha256": sha256_file(paths["log"]),
                "resolved_config_path": str(paths["resolved"]),
                "resolved_config_sha256": sha256_file(paths["resolved"]),
            }
            completed_candidates.append(candidate)

        comparison_results = comparisons(records)
        baseline = load_json(args.baseline.resolve())["metrics"]
        candidate_results = {}
        for candidate in CANDIDATES:
            metrics = aggregates[candidate]["metrics"]
            candidate_results[candidate] = {
                "sample_count": aggregates[candidate]["sample_count"],
                "windows_per_sample": aggregates[candidate]["windows_per_sample"],
                "finite": all(math.isfinite(float(value)) for value in metrics.values()),
                "metrics": metrics,
                "baseline_ratios": baseline_comparison(metrics, baseline),
                "checkpoint": aggregates[candidate]["checkpoint"],
                "data_contract": aggregates[candidate]["data_contract"],
                "normalization_audit": aggregates[candidate]["normalization_audit"],
                "generation_metrics_descriptive_only": aggregates[candidate][
                    "generation_metrics"
                ],
            }
        decision = transfer_gate(
            candidate_results,
            comparison_results,
            hashes_exact=(
                static_hashes == {
                    "test_script": TEST_SCRIPT_SHA256,
                    "eval_metrics": EVAL_METRICS_SHA256,
                    "eval_config": EVAL_CONFIG_SHA256,
                    "baseline": BASELINE_SHA256,
                }
                and actual_checkpoint_hashes == EXPECTED_CHECKPOINT_SHA256
            ),
            sampler_contract_exact=True,
        )
        result = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "phase": "p1",
            "subphase": "1B-D2-N0",
            "status": "completed",
            "seed": 42,
            "git_commit": git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "protocol": {
                "author_native_metrics": True,
                "official_test_sequences": SAMPLE_COUNT,
                "windows_per_sequence": WINDOWS_PER_SAMPLE,
                "checkpoint_order": list(CANDIDATES),
                "weight_variant": "online",
                "same_seed_and_sequence_order": True,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "chois_used": False,
                "fid_rprecision_used": False,
                "fps_used_for_gate": False,
            },
            "evaluator": {
                "test_script_sha256": static_hashes["test_script"],
                "eval_metrics_sha256": static_hashes["eval_metrics"],
                "eval_metrics_matches_author_commit": True,
                "eval_config_sha256": static_hashes["eval_config"],
                "phase0_baseline_sha256": static_hashes["baseline"],
                "prior_formal_hoi_failure_already_used_author_native_metrics": True,
            },
            "sampler_contract": {
                "future_gt": False,
                "stored_per_frame_bps": False,
                "rollout_bps": "recomputed_from_current_generated_object_pose",
                "production_equation_changed": False,
            },
            "candidates": candidate_results,
            "comparisons": comparison_results,
            "decision": decision,
            "raw_artifacts": raw_artifacts,
            "released_checkpoint_loaded": False,
            "checkpoint_selection": False,
            "training_authorized": False,
            "training_started": False,
            "d2h1_started": False,
        }
        exclusive_json(args.metrics.resolve(), result)
    except Exception as error:
        if not args.metrics.resolve().exists():
            exclusive_json(args.metrics.resolve(), {
                "schema_version": 1,
                "run_id": RUN_ID,
                "phase": "p1",
                "subphase": "1B-D2-N0",
                "status": "failed",
                "seed": 42,
                "git_commit": git_output("rev-parse", "HEAD"),
                "runtime_seconds": time.perf_counter() - started,
                "completed_candidates": completed_candidates,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "chois_used": False,
                "training_started": False,
                "d2h1_started": False,
            })
        raise


if __name__ == "__main__":
    main()
