#!/usr/bin/env python3
"""Evaluate the D2-Y final online checkpoint against the sealed D2-X control."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import run_hoi_d2x_evaluation as shared


RUN_ID = "p1-hoi-d2y-native-eval-s42-20260724"
SUBPHASE = "1B-D2-Y0-eval"
TRAINING_RUN_ID = "p1-hoi-d2y-routed-foot-amplification-s42-20260723"
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
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e"
)

_shared_resolved_config = shared.resolved_config
_internal_decision: Dict[str, object] = {}
sha256_file = shared.sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--training-metrics-sha256", required=True)
    parser.add_argument("--internal-diagnostic", type=Path, required=True)
    parser.add_argument("--internal-diagnostic-sha256", required=True)
    parser.add_argument("--control-aggregate", type=Path, required=True)
    parser.add_argument("--control-per-sequence", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.internal_diagnostic_sha256):
        raise ValueError("D2-Y internal diagnostic SHA-256 must be lowercase hexadecimal")
    return args


def resolved_config(args) -> Dict[str, object]:
    config = _shared_resolved_config(args)
    config["internal_diagnostic"] = {
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": args.internal_diagnostic_sha256,
        "selection_sha256": INTERNAL_SELECTION_SHA256,
        "gate_timesteps": [249, 499],
    }
    return config


def _validate_internal_diagnostic(args) -> Dict[str, object]:
    actual_hash = shared.sha256_file(args.internal_diagnostic.resolve())
    if actual_hash != args.internal_diagnostic_sha256:
        raise ValueError("D2-Y internal diagnostic hash mismatch")
    diagnostic = shared.load_json(args.internal_diagnostic.resolve())
    timestep_gates = diagnostic.get("decision", {}).get("timestep_checks", {})
    checks = {
        "schema_version": diagnostic.get("schema_version") == 1,
        "status": diagnostic.get("status") == "completed",
        "run_id": diagnostic.get("run_id") == (
            "p1-hoi-d2y-routed-foot-amplification-internal-s42-20260724"
        ),
        "selection_sha256": (
            diagnostic.get("selection", {}).get("sha256") == INTERNAL_SELECTION_SHA256
        ),
        "selection_sequences": diagnostic.get("selection", {}).get("sequences") == 32,
        "selection_windows": diagnostic.get("selection", {}).get("windows") == 96,
        "control_checkpoint": (
            diagnostic.get("checkpoints", {}).get("d2x", {}).get("final_sha256")
            == CONTROL_CHECKPOINT_SHA256
        ),
        "target_checkpoint": (
            diagnostic.get("checkpoints", {}).get("d2y", {}).get("final_sha256")
            == args.target_sha256
        ),
        "timestep_249_present": "249" in timestep_gates,
        "timestep_499_present": "499" in timestep_gates,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-Y internal diagnostic contract mismatch: {failed}")
    for timestep in ("249", "499"):
        item = timestep_gates[timestep]
        lower = float(item["bootstrap_95_ci"][0])
        if bool(item.get("passed")) != (lower > 0.0):
            raise ValueError(f"D2-Y internal diagnostic gate inconsistency at t={timestep}")
    passed = all(bool(timestep_gates[timestep]["passed"]) for timestep in ("249", "499"))
    declared = diagnostic.get("decision", {}).get("mechanism_passed")
    if declared is not passed:
        raise ValueError("D2-Y internal diagnostic overall gate inconsistency")
    return {
        "checks": checks,
        "mechanism_passed": passed,
        "timestep_checks": timestep_gates,
        "sha256": actual_hash,
    }


def validate_training_result(args) -> Dict[str, object]:
    if args.target_checkpoint.name != (
        "p1-hoi-d2y-routed-foot-amplification-s42-20260723_"
        "windows061440000.pth"
    ):
        raise ValueError("D2-Y evaluation requires the registered final checkpoint basename")
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
    expected_routing = {
        "fk_foot_temporal_routing": True,
        "foot_joint_indices": [7, 8, 10, 11],
        "routed_components": ["x", "z"],
        "velocity_weight": 0.1,
        "velocity_reduction": "mean_square",
        "routed_foot_residual_multiplier": 1024.0,
        "nonrouted_residual_multiplier": 1.0,
        "weighted_slots": 8,
        "total_velocity_slots": 87,
    }
    weight_initialization = metrics.get("weight_initialization", {})
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
        "loss_routing": metrics.get("loss_routing") == expected_routing,
        "ema_decays": metrics.get("ema_decays") == [],
        "primary_weight_variant": metrics.get("primary_weight_variant") == "online",
        "weight_initialization": (
            weight_initialization.get("mode") == "random"
            and weight_initialization.get("restored_components") == []
            and weight_initialization.get("source_checkpoint") is None
            and weight_initialization.get("initial_model_state_sha256")
            == EXPECTED_INITIAL_MODEL_STATE_SHA256
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
        raise ValueError(f"D2-Y training artifact contract mismatch: {failed}")
    internal = _validate_internal_diagnostic(args)
    checks["internal_diagnostic_contract"] = True
    _internal_decision.clear()
    _internal_decision.update(internal)
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
    internal_passed = bool(_internal_decision.get("mechanism_passed", False))
    full_contract = contract_passed and mask_passed
    mechanism_passed = (
        full_contract and internal_passed and foot_improved
        and all(protection_checks.values())
    )
    effective_checks = {
        metric: float(baseline_ratios[metric]) <= threshold
        for metric, threshold in shared.EFFECTIVE_RATIO_MAX.items()
    }
    effective_checks["contact_f1_min"] = float(target_metrics["contact_f1"]) >= 0.60
    absolute_passed = all(effective_checks.values())
    effective_passed = mechanism_passed and absolute_passed
    if not full_contract:
        classification = "routed-foot-amplification-contract-failure-stop"
    elif not internal_passed:
        classification = "routed-foot-amplification-optimization-negative-stop"
    elif not foot_improved:
        classification = "routed-foot-amplification-transfer-negative-stop"
    elif not all(protection_checks.values()):
        classification = "routed-foot-amplification-conflict-negative-stop"
    elif not absolute_passed:
        classification = "routed-foot-amplification-positive-but-not-effective-stop"
    else:
        classification = "routed-foot-amplification-positive-candidate-stop"
    return {
        "classification": classification,
        "contract_passed": full_contract,
        "penetration_mask_contract": comparison.get("penetration_mask_contract"),
        "internal_mechanism_passed": internal_passed,
        "internal_timestep_checks": _internal_decision.get("timestep_checks"),
        "official_foot_sliding_improved": foot_improved,
        "protection_passed": all(protection_checks.values()),
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
    shared.validate_training_result = validate_training_result
    shared.classify = classify


def main() -> None:
    configure_shared_module()
    shared.main()


if __name__ == "__main__":
    main()
