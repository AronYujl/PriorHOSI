#!/usr/bin/env python3
"""Evaluate D2-AB against the sealed D2-X native control."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Tuple


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import run_hoi_d2x_evaluation as shared  # noqa: E402
from tools.run_hoi_d2n import validate_candidate_result  # noqa: E402


RUN_ID = "p1-hoi-d2ab-native-eval-s42-20260725"
SUBPHASE = "1B-D2-AB0-eval"
TRAINING_RUN_ID = "p1-hoi-d2ab-predicted-support-no-slip-s42-20260725"
INTERNAL_RUN_ID = "p1-hoi-d2ab-predicted-support-no-slip-internal-s42-20260725"
CONTROL_CHECKPOINT_SHA256 = (
    "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51"
)
CONTROL_AGGREGATE_SHA256 = (
    "3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b"
)
CONTROL_PER_SEQUENCE_SHA256 = (
    "69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a"
)
INTERNAL_SELECTION_SHA256 = (
    "30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae"
)
SUPPORT_METADATA_SHA256 = (
    "807978580221910ad00260c2dff4f33ddacbb1bf72bad7443bf21ac48f31f079"
)
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e"
)
_shared_resolved_config = shared.resolved_config
_internal: Dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--training-metrics-sha256", required=True)
    parser.add_argument("--internal-diagnostic", type=Path, required=True)
    parser.add_argument("--internal-diagnostic-sha256", required=True)
    parser.add_argument("--support-metadata", type=Path, required=True)
    parser.add_argument("--support-metadata-sha256", required=True)
    parser.add_argument("--control-aggregate", type=Path, required=True)
    parser.add_argument("--control-per-sequence", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if name.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    return args


def resolved_config(args) -> Dict[str, object]:
    config = _shared_resolved_config(args)
    config["internal_diagnostic"] = {
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": args.internal_diagnostic_sha256,
        "run_id": INTERNAL_RUN_ID,
        "selection_sha256": INTERNAL_SELECTION_SHA256,
        "selection_use": False,
    }
    config["support_metadata"] = {
        "path": str(args.support_metadata.resolve()),
        "sha256": args.support_metadata_sha256,
        "registered_sha256": SUPPORT_METADATA_SHA256,
    }
    return config


def additional_runtime_artifact_hashes(
    args,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    actual = {
        "internal_diagnostic": shared.sha256_file(args.internal_diagnostic.resolve()),
        "support_metadata": shared.sha256_file(args.support_metadata.resolve()),
    }
    expected = {
        "internal_diagnostic": args.internal_diagnostic_sha256,
        "support_metadata": SUPPORT_METADATA_SHA256,
    }
    return actual, expected


def _validate_internal(args) -> Dict[str, object]:
    diagnostic = shared.load_json(args.internal_diagnostic.resolve())
    comparison = diagnostic.get("comparison", {})
    checks = {
        "schema_version": diagnostic.get("schema_version") == 1,
        "status": diagnostic.get("status") == "completed",
        "run_id": diagnostic.get("run_id") == INTERNAL_RUN_ID,
        "selection_sha256": (
            diagnostic.get("selection", {}).get("sha256") == INTERNAL_SELECTION_SHA256
        ),
        "selection_sequences": diagnostic.get("selection", {}).get("sequences") == 32,
        "selection_windows": diagnostic.get("selection", {}).get("windows") == 96,
        "control_checkpoint": (
            diagnostic.get("control_checkpoint", {}).get("sha256")
            == CONTROL_CHECKPOINT_SHA256
        ),
        "target_checkpoint": (
            diagnostic.get("target_checkpoint", {}).get("sha256")
            == args.target_sha256
        ),
        "support_metadata": (
            diagnostic.get("support_metadata", {}).get("sha256")
            == SUPPORT_METADATA_SHA256
        ),
        "pairing": diagnostic.get("pairing") == {
            "same_clean_windows": True,
            "same_timestep": True,
            "same_noise": True,
            "same_condition_dropout": True,
        },
        "comparison_contract": comparison.get("contract_passed") is True,
        "finite": comparison.get("finite") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AB internal diagnostic contract mismatch: {failed}")
    mechanism = comparison.get("mechanism_checks", {})
    support = comparison.get("support_sanity", {})
    mechanism_passed = bool(comparison.get("mechanism_passed", False))
    support_passed = bool(comparison.get("support_sanity_passed", False))
    return {
        "checks": checks,
        "mechanism_passed": mechanism_passed,
        "support_sanity_passed": support_passed,
        "optimization_gate_passed": mechanism_passed and support_passed,
        "mechanism_checks": mechanism,
        "support_sanity": support,
    }


def validate_training_result(args) -> Dict[str, object]:
    expected_name = f"{TRAINING_RUN_ID}_windows061440000.pth"
    if args.target_checkpoint.name != expected_name:
        raise ValueError("D2-AB evaluation requires the registered final checkpoint basename")
    metrics = shared.load_json(args.training_metrics.resolve())
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
    routing = metrics.get("loss_routing", {})
    expected_routing = {
        "fk_foot_temporal_routing": True,
        "foot_joint_indices": [7, 8, 10, 11],
        "routed_components": ["x", "z"],
        "velocity_weight": 0.1,
        "velocity_reduction": "mean_square",
        "d2ab_predicted_support_no_slip": True,
        "support_metadata_path": str(args.support_metadata.resolve()),
        "support_metadata_sha256": SUPPORT_METADATA_SHA256,
        "support_floor_source": "raw_immutable_train_sequence_toe_y_5th_linear_quantile",
        "support_pair_definition": "left_7_10_right_8_11_logmeanexp_soft_min",
        "support_scale_m": 0.03925712490454316,
        "sample_interval_s": 0.1,
        "physical_velocity": "horizontal_position_delta_over_0.1s",
        "velocity_scale_s_per_m": 0.029363068377844033,
        "first_future_previous": "immutable_gt_history",
        "later_previous": "predicted_fk_previous_frame",
        "gt_and_floor_stop_gradient": True,
        "zero_slip_target_when_supported": True,
        "weighted_slots": 8,
        "total_velocity_slots": 87,
    }
    initialization = metrics.get("weight_initialization", {})
    checks = {
        "status": metrics.get("status") == "stable",
        "run_id": metrics.get("run_id") == TRAINING_RUN_ID,
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
        "loss_routing": routing == expected_routing,
        "support_metadata": metrics.get("support_metadata") == {
            "path": str(args.support_metadata.resolve()),
            "sha256": SUPPORT_METADATA_SHA256,
        },
        "ema_decays": metrics.get("ema_decays") == [],
        "primary_weight_variant": metrics.get("primary_weight_variant") == "online",
        "weight_initialization": (
            initialization.get("mode") == "random"
            and initialization.get("restored_components") == []
            and initialization.get("source_checkpoint") is None
            and initialization.get("initial_model_state_sha256")
            == EXPECTED_INITIAL_MODEL_STATE_SHA256
        ),
    }
    matching = [
        item for item in metrics.get("checkpoint_hashes", [])
        if item.get("processed_windows") == 61440000
        and item.get("sha256") == args.target_sha256
        and Path(item.get("path", "")).name == expected_name
    ]
    checks["final_checkpoint_hash"] = len(matching) == 1
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError(f"D2-AB training artifact contract mismatch: {failed}")
    internal = _validate_internal(args)
    _internal.clear()
    _internal.update(internal)
    return {"checks": checks, "metrics": metrics, "internal_diagnostic": internal}


def classify(
    comparison: Mapping[str, object],
    target_metrics: Mapping[str, object],
    baseline_ratios: Mapping[str, float],
    *,
    contract_passed: bool,
) -> Dict[str, object]:
    mask_passed = bool(
        comparison.get("penetration_mask_contract", {}).get("passed", False)
    )
    protection_checks = {
        f"{metric}_preserved": (
            comparison["target_over_control_protection"][metric]["bootstrap_95_ci"][1]
            <= 1.10
        )
        for metric in shared.PROTECTION_RATIO_METRICS
        if metric in comparison["target_over_control_protection"]
    }
    for metric in shared.PENETRATION_METRICS:
        protection_checks.setdefault(f"{metric}_preserved", False)
    protection_checks["contact_f1_preserved"] = (
        comparison["target_minus_control_contact_f1"]["bootstrap_95_ci"][0] >= -0.02
    )
    foot_improved = (
        comparison["control_minus_target_foot_sliding"]["bootstrap_95_ci"][0] > 0.0
    )
    protection_passed = all(protection_checks.values())
    internal_passed = bool(_internal.get("optimization_gate_passed", False))
    full_contract = contract_passed and mask_passed
    mechanism_passed = full_contract and internal_passed and foot_improved and protection_passed
    effective_checks = {
        metric: float(baseline_ratios[metric]) <= threshold
        for metric, threshold in shared.EFFECTIVE_RATIO_MAX.items()
    }
    effective_checks["contact_f1_min"] = float(target_metrics["contact_f1"]) >= 0.60
    absolute_passed = all(effective_checks.values())
    effective_passed = mechanism_passed and absolute_passed
    if not full_contract:
        classification = "predicted-support-no-slip-contract-failure-stop"
    elif not internal_passed:
        classification = "predicted-support-no-slip-optimization-negative-stop"
    elif not foot_improved:
        classification = "predicted-support-no-slip-transfer-negative-stop"
    elif not protection_passed:
        classification = "predicted-support-no-slip-conflict-negative-stop"
    elif not absolute_passed:
        classification = "predicted-support-no-slip-positive-but-not-effective-stop"
    else:
        classification = "predicted-support-no-slip-positive-candidate-stop"
    return {
        "classification": classification,
        "contract_passed": full_contract,
        "internal_diagnostic_passed": internal_passed,
        "internal_diagnostic_used_for_selection": False,
        "penetration_mask_contract": comparison.get("penetration_mask_contract"),
        "official_foot_sliding_improved": foot_improved,
        "protection_passed": protection_passed,
        "protection_checks": protection_checks,
        "mechanism_passed": mechanism_passed,
        "effective_diffusion_passed": effective_passed,
        "effective_diffusion_checks": effective_checks,
        "checkpoint_selected": False,
        "consistency_authorized": False,
        "consistency_started": False,
    }


def configure_shared_module() -> None:
    shared.RUN_ID = RUN_ID
    shared.SUBPHASE = SUBPHASE
    shared.CONTROL_CHECKPOINT_SHA256 = CONTROL_CHECKPOINT_SHA256
    shared.CONTROL_AGGREGATE_SHA256 = CONTROL_AGGREGATE_SHA256
    shared.CONTROL_PER_SEQUENCE_SHA256 = CONTROL_PER_SEQUENCE_SHA256
    shared.parse_args = parse_args
    shared.resolved_config = resolved_config
    shared.additional_runtime_artifact_hashes = additional_runtime_artifact_hashes
    shared.validate_training_result = validate_training_result
    shared.compare_records = shared.compare_records
    shared.classify = classify


if __name__ == "__main__":
    configure_shared_module()
    shared.main()
