#!/usr/bin/env python3
"""Evaluate the D2-X final online checkpoint against the completed D2-V control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.run_hoi_d2n import (  # noqa: E402
    BASELINE_KEYS,
    baseline_comparison,
    paired_difference,
    paired_ratio,
    per_sequence_arrays,
    validate_candidate_result,
)


RUN_ID = "p1-hoi-d2x-native-eval-r1-s42-20260723"
SUBPHASE = "1B-D2-X0-eval-r1"
CONTROL_CHECKPOINT_SHA256 = "e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4"
CONTROL_AGGREGATE_SHA256 = "21f6bb27fe8d38a5203c2e40dee02815470bc40b638ee3a616101faae5cf8f0e"
CONTROL_PER_SEQUENCE_SHA256 = "4d147ef0a76977146639bab4260c6f7c5c2f96d9b253fb52ad968269f649ce1a"
TEST_SCRIPT_SHA256 = "22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524"
EVAL_METRICS_SHA256 = "445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547"
EVAL_CONFIG_SHA256 = "89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73"
BASELINE_SHA256 = "76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6"
PROTECTION_RATIO_METRICS = (
    "mpjpe",
    "end_obj_trans_err",
    "xy_points_err",
    "obj_trans_dist",
    "hand_pen_loss_omomo",
    "human_pen_loss_infbagel",
)
PENETRATION_METRICS = ("hand_pen_loss_omomo", "human_pen_loss_infbagel")
PENETRATION_SEQUENCE_COUNT = 181
PENETRATION_SEQUENCE_IDS_SHA256 = (
    "2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec"
)
EFFECTIVE_RATIO_MAX = {
    "mpjpe": 1.30,
    "end_obj_trans_err": 2.00,
    "xy_points_err": 1.50,
    "obj_trans_dist": 1.50,
    "foot_sliding": 1.10,
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


def target_paths(output: Path) -> Dict[str, Path]:
    return {
        "evaluation": output / "evaluation",
        "aggregate": output / "evaluation" / "aggregate_metrics.json",
        "per_sequence": output / "evaluation" / "per_sequence_metrics.json",
        "hydra": output / "hydra",
        "resolved": output / "resolved_target.yaml",
        "log": output / "author_native_target.log",
    }


def target_overrides(args) -> Sequence[str]:
    paths = target_paths(args.output.resolve())
    return (
        f"exp_name={RUN_ID}-target",
        f"ckpt_path={args.target_checkpoint.resolve()}",
        f"checkpoint_sha256={args.target_sha256}",
        "checkpoint_weight_variant=online",
        f"device={args.device}",
        f"dataset.device={args.device}",
        f"sampler.pelvis.device={args.device}",
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


def target_command(args) -> Sequence[str]:
    return (
        str(args.python.resolve()),
        str(REPO / "code/test_infbagel_hoi.py"),
        "--config-name",
        "config_eval_hoi_prior",
        *target_overrides(args),
    )


def resolve_target(args) -> str:
    command = (*target_command(args)[:3], "config_eval_hoi_prior", "--cfg", "job", "--resolve", *target_overrides(args))
    environment = dict(os.environ)
    environment["ROOT_DIR"] = str(REPO)
    completed = subprocess.run(
        command, cwd=REPO / "code", env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    if "${" in completed.stdout:
        raise ValueError("D2-X evaluation config contains unresolved interpolation")
    return completed.stdout


def resolved_config(args) -> Dict[str, object]:
    paths = target_paths(args.output.resolve())
    content = paths["resolved"].read_text(encoding="utf-8")
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "execution_host": "infbagel-4gpu/node01",
        "python": str(args.python.resolve()),
        "device": args.device,
        "target": {
            "checkpoint": str(args.target_checkpoint.resolve()),
            "sha256": args.target_sha256,
            "weight_variant": "online",
            "training_metrics": str(args.training_metrics.resolve()),
            "training_metrics_sha256": args.training_metrics_sha256,
            "resolved_config": str(paths["resolved"]),
            "resolved_config_sha256": sha256_text(content),
            "aggregate": str(paths["aggregate"]),
            "per_sequence": str(paths["per_sequence"]),
        },
        "control": {
            "aggregate": str(args.control_aggregate.resolve()),
            "aggregate_sha256": CONTROL_AGGREGATE_SHA256,
            "per_sequence": str(args.control_per_sequence.resolve()),
            "per_sequence_sha256": CONTROL_PER_SEQUENCE_SHA256,
            "checkpoint_sha256": CONTROL_CHECKPOINT_SHA256,
            "regenerated": False,
        },
        "baseline": {"path": str(args.baseline.resolve()), "sha256": BASELINE_SHA256},
        "evaluation": {
            "official_test_sequences": 438,
            "windows_per_sequence": 3,
            "diffusion_steps": 500,
            "cfg": False,
            "dynamic_perception": False,
            "guidance": False,
            "chois_used": False,
            "fid_rprecision_used": False,
            "paired_unit": "sequence",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 42,
        },
        "metrics_path": str(args.metrics.resolve()),
        "training_started": False,
        "consistency_started": False,
    }


def prepare_resolved_config(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = target_paths(output)
    content = resolve_target(args)
    if paths["resolved"].exists():
        if paths["resolved"].read_text(encoding="utf-8") != content:
            raise FileExistsError("refusing to overwrite changed target resolved config")
    else:
        paths["resolved"].write_text(content, encoding="utf-8")
    exclusive_json(args.resolved_config.resolve(), resolved_config(args))


def verify_static_assets(args) -> Dict[str, str]:
    paths = {
        "test_script": REPO / "code/test_infbagel_hoi.py",
        "eval_metrics": REPO / "code/eval_metrics.py",
        "eval_config": REPO / "code/config/config_eval_hoi_prior.yaml",
        "baseline": args.baseline.resolve(),
    }
    expected = {
        "test_script": TEST_SCRIPT_SHA256,
        "eval_metrics": EVAL_METRICS_SHA256,
        "eval_config": EVAL_CONFIG_SHA256,
        "baseline": BASELINE_SHA256,
    }
    actual = {key: sha256_file(path) for key, path in paths.items()}
    if actual != expected:
        raise ValueError(f"D2-X evaluator static hash mismatch: {actual}")
    return actual


def finite_metric_mask(
    records: Mapping[str, Mapping[str, object]],
    metric: str,
) -> Tuple[str, ...]:
    return tuple(
        sequence
        for sequence in sorted(records)
        if records[sequence].get(metric) is not None
        and math.isfinite(float(records[sequence][metric]))
    )


def sequence_ids_sha256(sequences: Sequence[str]) -> str:
    return sha256_text("".join(f"{sequence}\n" for sequence in sequences))


def metric_arrays(
    records: Mapping[str, Mapping[str, object]],
    metric: str,
    *,
    sequences: Optional[Sequence[str]] = None,
) -> np.ndarray:
    if sequences is None:
        return per_sequence_arrays(records, metric)
    values = []
    for sequence in sequences:
        value = records[sequence].get(metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"D2-X nonfinite/missing {metric} for {sequence}: {value}")
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def compare_records(control: Mapping[str, object], target: Mapping[str, object]) -> Dict[str, object]:
    if sorted(control) != sorted(target):
        raise ValueError("D2-X/control sequence identities differ")
    control_masks = {
        metric: finite_metric_mask(control, metric) for metric in PENETRATION_METRICS
    }
    target_masks = {
        metric: finite_metric_mask(target, metric) for metric in PENETRATION_METRICS
    }
    locked_mask = control_masks[PENETRATION_METRICS[0]]
    mask_checks = {
        "control_penetration_fields_match": (
            control_masks[PENETRATION_METRICS[0]]
            == control_masks[PENETRATION_METRICS[1]]
        ),
        "control_finite_count": len(locked_mask) == PENETRATION_SEQUENCE_COUNT,
        "control_finite_ids_sha256": (
            sequence_ids_sha256(locked_mask) == PENETRATION_SEQUENCE_IDS_SHA256
        ),
        "target_hand_mask_matches_control": (
            target_masks[PENETRATION_METRICS[0]] == locked_mask
        ),
        "target_human_mask_matches_control": (
            target_masks[PENETRATION_METRICS[1]] == locked_mask
        ),
        "target_penetration_fields_match": (
            target_masks[PENETRATION_METRICS[0]]
            == target_masks[PENETRATION_METRICS[1]]
        ),
    }
    mask_passed = all(mask_checks.values())
    penetration_ratios = {}
    if mask_passed:
        penetration_ratios = {
            metric: paired_ratio(
                metric_arrays(target, metric, sequences=locked_mask),
                metric_arrays(control, metric, sequences=locked_mask),
            )
            for metric in PENETRATION_METRICS
        }
    protection_ratios = {
        metric: paired_ratio(
            metric_arrays(target, metric), metric_arrays(control, metric),
        )
        for metric in PROTECTION_RATIO_METRICS
        if metric not in PENETRATION_METRICS
    }
    protection_ratios.update(penetration_ratios)
    return {
        "control_minus_target_foot_sliding": paired_difference(
            metric_arrays(control, "foot_sliding"),
            metric_arrays(target, "foot_sliding"),
        ),
        "target_over_control_protection": protection_ratios,
        "penetration_mask_contract": {
            "passed": mask_passed,
            "checks": mask_checks,
            "official_sequences": len(control),
            "finite_sequences": len(locked_mask),
            "finite_sequence_ids_sha256": sequence_ids_sha256(locked_mask),
        },
        "target_minus_control_contact_f1": paired_difference(
            metric_arrays(target, "contact_f1"),
            metric_arrays(control, "contact_f1"),
        ),
    }


def classify(
    comparison: Mapping[str, object],
    target_metrics: Mapping[str, object],
    baseline_ratios: Mapping[str, float],
    *,
    contract_passed: bool,
) -> Dict[str, object]:
    mask_contract_passed = bool(
        comparison.get("penetration_mask_contract", {}).get("passed", False)
    )
    mechanism_checks = {
        "foot_sliding_improved": (
            comparison["control_minus_target_foot_sliding"]["bootstrap_95_ci"][0] > 0.0
        ),
        **{
            f"{metric}_preserved": (
                comparison["target_over_control_protection"][metric]["bootstrap_95_ci"][1]
                <= 1.10
            )
            for metric in PROTECTION_RATIO_METRICS
            if metric in comparison["target_over_control_protection"]
        },
        "contact_f1_preserved": (
            comparison["target_minus_control_contact_f1"]["bootstrap_95_ci"][0] >= -0.02
        ),
    }
    for metric in PENETRATION_METRICS:
        mechanism_checks.setdefault(f"{metric}_preserved", False)
    full_contract_passed = contract_passed and mask_contract_passed
    mechanism_passed = full_contract_passed and all(mechanism_checks.values())
    effective_checks = {
        metric: float(baseline_ratios[metric]) <= threshold
        for metric, threshold in EFFECTIVE_RATIO_MAX.items()
    }
    effective_checks["contact_f1_min"] = float(target_metrics["contact_f1"]) >= 0.60
    effective_passed = mechanism_passed and all(effective_checks.values())
    if not full_contract_passed:
        classification = "fk-foot-temporal-routing-contract-failure-stop"
    elif not mechanism_passed:
        classification = "fk-foot-temporal-routing-negative-stop"
    elif not effective_passed:
        classification = "fk-foot-temporal-routing-positive-but-not-effective-stop"
    else:
        classification = "fk-foot-temporal-routing-positive-candidate-stop"
    return {
        "classification": classification,
        "contract_passed": full_contract_passed,
        "penetration_mask_contract": comparison.get("penetration_mask_contract"),
        "mechanism_passed": mechanism_passed,
        "mechanism_checks": mechanism_checks,
        "effective_diffusion_passed": effective_passed,
        "effective_diffusion_checks": effective_checks,
        "checkpoint_selected": False,
        "consistency_authorized": False,
        "consistency_started": False,
    }


def validate_training_result(args) -> Dict[str, object]:
    if args.target_checkpoint.name != (
        "p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723_"
        "windows061440000.pth"
    ):
        raise ValueError("D2-X evaluation requires the registered final checkpoint basename")
    metrics = load_json(args.training_metrics.resolve())
    expected_optimization = {
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "learning_rate": 0.0001,
        "scheduler": "none",
        "warmup_windows": 0,
        "gradient_clipping": False,
        "gradient_clip_norm": None,
        "amp": False,
        "ema_decays": [],
        "primary_weight_variant": "online",
    }
    expected_losses = {
        "fk": 0.3569973401779424,
        "object_surface": 0.4772322188400037,
        "velocity": 0.1,
        "terminal_object_goal": 1.0,
    }
    expected_routing = {
        "fk_foot_temporal_routing": True,
        "foot_joint_indices": [7, 8, 10, 11],
        "routed_components": ["x", "z"],
        "velocity_weight": 0.1,
        "velocity_reduction": "mean_square",
    }
    checks = {
        "status": metrics.get("status") == "stable",
        "run_id": metrics.get("run_id") == (
            "p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723"
        ),
        "seed": metrics.get("seed") == 42,
        "initialization": metrics.get("initialization") == "random",
        "training_start": metrics.get("training_start") == "random",
        "released_checkpoint_used": metrics.get("released_checkpoint_used") is False,
        "processed_windows": metrics.get("processed_windows") == 61440000,
        "optimizer_updates": metrics.get("optimizer_updates") == 30000,
        "world_size": metrics.get("world_size") == 4,
        "effective_batch_size": metrics.get("effective_batch_size") == 2048,
        "optimization_contract": metrics.get("optimization_contract") == expected_optimization,
        "loss_weights": metrics.get("loss_weights") == expected_losses,
        "loss_routing": metrics.get("loss_routing") == expected_routing,
        "ema_decays": metrics.get("ema_decays") == [],
        "primary_weight_variant": metrics.get("primary_weight_variant") == "online",
        "weight_initialization": (
            metrics.get("weight_initialization", {}).get("mode") == "random"
            and metrics.get("weight_initialization", {}).get("restored_components") == []
            and metrics.get("weight_initialization", {}).get("source_checkpoint") is None
        ),
    }
    matching = [
        item for item in metrics.get("checkpoint_hashes", [])
        if item.get("processed_windows") == 61440000
        and item.get("sha256") == args.target_sha256
        and Path(item.get("path", "")).name == args.target_checkpoint.name
    ]
    checks["final_checkpoint_hash"] = len(matching) == 1
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError(f"D2-X training artifact contract mismatch: {failed}")
    return {"checks": checks, "metrics": metrics}


def run_target(args) -> None:
    paths = target_paths(args.output.resolve())
    if paths["evaluation"].exists() or paths["log"].exists():
        raise FileExistsError("refusing to overwrite D2-X target evaluation")
    environment = dict(os.environ)
    environment["ROOT_DIR"] = str(REPO)
    with paths["log"].open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            target_command(args), cwd=REPO / "code", env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"D2-X author-native evaluation exited {completed.returncode}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--training-metrics-sha256", required=True)
    parser.add_argument("--control-aggregate", type=Path, required=True)
    parser.add_argument("--control-per-sequence", type=Path, required=True)
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
        raise ValueError(f"D2-X evaluation run id must be {RUN_ID}")
    if not __import__("re").fullmatch(r"[0-9a-f]{64}", args.target_sha256):
        raise ValueError("D2-X target checkpoint SHA-256 must be lowercase hexadecimal")
    if not __import__("re").fullmatch(r"[0-9a-f]{64}", args.training_metrics_sha256):
        raise ValueError("D2-X training metrics SHA-256 must be lowercase hexadecimal")
    if args.python.resolve() != Path(os.environ.get("INFBAGEL_PYTHON", "")).resolve():
        raise ValueError("D2-X evaluation requires the absolute INFBAGEL_PYTHON")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi" or socket.gethostname() != "node01":
        raise RuntimeError("D2-X evaluation is restricted to the HOI worker")
    if args.resolve_only:
        prepare_resolved_config(args)
        return
    if git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("D2-X evaluation refuses a dirty worker checkout")
    if load_json(args.resolved_config.resolve()) != resolved_config(args):
        raise ValueError("D2-X runtime arguments do not match archived resolved config")

    started = time.perf_counter()
    try:
        static_hashes = verify_static_assets(args)
        actual_hashes = {
            "target": sha256_file(args.target_checkpoint.resolve()),
            "training_metrics": sha256_file(args.training_metrics.resolve()),
            "control_aggregate": sha256_file(args.control_aggregate.resolve()),
            "control_per_sequence": sha256_file(args.control_per_sequence.resolve()),
        }
        expected_hashes = {
            "target": args.target_sha256,
            "training_metrics": args.training_metrics_sha256,
            "control_aggregate": CONTROL_AGGREGATE_SHA256,
            "control_per_sequence": CONTROL_PER_SEQUENCE_SHA256,
        }
        if actual_hashes != expected_hashes:
            raise ValueError(f"D2-X runtime artifact hash mismatch: {actual_hashes}")

        training_contract = validate_training_result(args)
        control_aggregate = load_json(args.control_aggregate.resolve())
        control_per_sequence = load_json(args.control_per_sequence.resolve())
        validate_candidate_result(
            "control", control_aggregate, control_per_sequence, CONTROL_CHECKPOINT_SHA256,
        )
        run_target(args)
        paths = target_paths(args.output.resolve())
        target_aggregate = load_json(paths["aggregate"])
        target_per_sequence = load_json(paths["per_sequence"])
        validate_candidate_result("target", target_aggregate, target_per_sequence, args.target_sha256)

        comparison = compare_records(
            control_per_sequence["metrics"], target_per_sequence["metrics"],
        )
        baseline = load_json(args.baseline.resolve())["metrics"]
        target_metrics = target_aggregate["metrics"]
        baseline_ratios = baseline_comparison(target_metrics, baseline)
        decision = classify(
            comparison, target_metrics, baseline_ratios, contract_passed=True,
        )
        result = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "completed",
            "seed": 42,
            "git_commit": git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "protocol": {
                "official_test_sequences": 438,
                "windows_per_sequence": 3,
                "weight_variant": "online",
                "diffusion_steps": 500,
                "paired_unit": "sequence",
                "bootstrap_replicates": 10000,
                "bootstrap_seed": 42,
                "chois_used": False,
                "fid_rprecision_used": False,
            },
            "static_hashes": static_hashes,
            "runtime_artifact_hashes": actual_hashes,
            "training_contract": training_contract["checks"],
            "target": {
                "metrics": target_metrics,
                "baseline_ratios": baseline_ratios,
                "checkpoint": target_aggregate["checkpoint"],
                "normalization_audit": target_aggregate["normalization_audit"],
                "generation_metrics_descriptive_only": target_aggregate["generation_metrics"],
            },
            "control": {
                "metrics": control_aggregate["metrics"],
                "checkpoint": control_aggregate["checkpoint"],
                "reused_without_regeneration": True,
            },
            "comparison": comparison,
            "decision": decision,
            "raw_artifacts": {
                "target_aggregate": {"path": str(paths["aggregate"]), "sha256": sha256_file(paths["aggregate"])},
                "target_per_sequence": {"path": str(paths["per_sequence"]), "sha256": sha256_file(paths["per_sequence"])},
                "target_log": {"path": str(paths["log"]), "sha256": sha256_file(paths["log"])},
                "target_resolved": {"path": str(paths["resolved"]), "sha256": sha256_file(paths["resolved"])},
            },
            "released_checkpoint_loaded": False,
            "author_checkpoint_loaded": False,
            "training_started": False,
            "training_updates": 0,
            "consistency_started": False,
        }
        exclusive_json(args.metrics.resolve(), result)
    except Exception as error:
        if not args.metrics.resolve().exists():
            exclusive_json(args.metrics.resolve(), {
                "schema_version": 1,
                "run_id": RUN_ID,
                "phase": "p1",
                "subphase": SUBPHASE,
                "status": "failed",
                "seed": 42,
                "git_commit": git_output("rev-parse", "HEAD"),
                "runtime_seconds": time.perf_counter() - started,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "training_started": False,
                "consistency_started": False,
            })
        raise


if __name__ == "__main__":
    main()
