#!/usr/bin/env python3
"""Evaluate D2-Z against sealed D2-X, reporting sealed D2-Y as comparator."""

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


RUN_ID = "p1-hoi-d2z-native-eval-s42-20260724"
SUBPHASE = "1B-D2-Z0-eval"
TRAINING_RUN_ID = "p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724"
INTERNAL_RUN_ID = (
    "p1-hoi-d2z-immutable-gt-near-ground-gating-internal-s42-20260724"
)
GATE_AUDIT_RUN_ID = "p1-hoi-d2z-gate-audit-r1-s42-20260724"
CONTROL_CHECKPOINT_SHA256 = (
    "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51"
)
CONTROL_AGGREGATE_SHA256 = (
    "3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b"
)
CONTROL_PER_SEQUENCE_SHA256 = (
    "69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a"
)
D2Y_CHECKPOINT_SHA256 = (
    "8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7"
)
D2Y_AGGREGATE_SHA256 = (
    "776e6c35acdaa190ffcbab047b170ed4ab559c23f454714c31ad980db4dd8c70"
)
D2Y_PER_SEQUENCE_SHA256 = (
    "ea2cde99372392c5f16446708e3acf3789a68be9f1b7cc95134fd45390b12c02"
)
INTERNAL_SELECTION_SHA256 = (
    "30524c88481f6cb81e8063073d510ad01543be92d91eb4ef9b2b8a376cc4fbae"
)
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e"
)

_shared_resolved_config = shared.resolved_config
_shared_compare_records = shared.compare_records
_comparator: Dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--training-metrics-sha256", required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--gate-audit-sha256", required=True)
    parser.add_argument("--internal-diagnostic", type=Path, required=True)
    parser.add_argument("--internal-diagnostic-sha256", required=True)
    parser.add_argument("--control-aggregate", type=Path, required=True)
    parser.add_argument("--control-per-sequence", type=Path, required=True)
    parser.add_argument("--d2y-aggregate", type=Path, required=True)
    parser.add_argument("--d2y-per-sequence", type=Path, required=True)
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
    config["gate_audit"] = {
        "path": str(args.gate_audit.resolve()),
        "sha256": args.gate_audit_sha256,
        "run_id": GATE_AUDIT_RUN_ID,
    }
    config["internal_diagnostic"] = {
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": args.internal_diagnostic_sha256,
        "run_id": INTERNAL_RUN_ID,
        "selection_sha256": INTERNAL_SELECTION_SHA256,
        "selection_use": False,
    }
    config["mechanism_comparator"] = {
        "name": "D2-Y",
        "checkpoint_sha256": D2Y_CHECKPOINT_SHA256,
        "aggregate": str(args.d2y_aggregate.resolve()),
        "aggregate_sha256": D2Y_AGGREGATE_SHA256,
        "per_sequence": str(args.d2y_per_sequence.resolve()),
        "per_sequence_sha256": D2Y_PER_SEQUENCE_SHA256,
        "regenerated": False,
        "selection_use": False,
    }
    return config


def additional_runtime_artifact_hashes(
    args,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    actual = {
        "gate_audit": shared.sha256_file(args.gate_audit.resolve()),
        "internal_diagnostic": shared.sha256_file(args.internal_diagnostic.resolve()),
        "d2y_aggregate": shared.sha256_file(args.d2y_aggregate.resolve()),
        "d2y_per_sequence": shared.sha256_file(args.d2y_per_sequence.resolve()),
    }
    expected = {
        "gate_audit": args.gate_audit_sha256,
        "internal_diagnostic": args.internal_diagnostic_sha256,
        "d2y_aggregate": D2Y_AGGREGATE_SHA256,
        "d2y_per_sequence": D2Y_PER_SEQUENCE_SHA256,
    }
    return actual, expected


def _validate_gate_audit(args) -> Dict[str, object]:
    audit = shared.load_json(args.gate_audit.resolve())
    checks = {
        "schema": audit.get("schema")
        == "d2z-immutable-gt-near-ground-gate-audit-v1",
        "status": audit.get("status") == "completed",
        "run_id": audit.get("run_id") == GATE_AUDIT_RUN_ID,
        "seed": audit.get("seed") == 42,
        "cpu_only": audit.get("cpu_only") is True,
        "selection": (
            audit.get("sealed_selection", {}).get("selection_sha256")
            == INTERNAL_SELECTION_SHA256
        ),
        "selection_counts": (
            audit.get("sealed_selection", {}).get("active_counts")
            == {"7": 1096, "8": 1081, "10": 1211, "11": 1232}
        ),
        "nonzero_active": (
            int(audit.get("partitions", {}).get("train", {}).get("total_active", 0)) > 0
        ),
        "finite_floor": all(
            partition.get("nonfinite_floor_count") == 0
            for partition in audit.get("partitions", {}).values()
        ),
        "finite_gate": all(
            partition.get("nonfinite_gate_count") == 0
            for partition in audit.get("partitions", {}).values()
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-Z gate audit contract mismatch: {failed}")
    return {"checks": checks, "payload_sha256": audit.get("payload_sha256")}


def _validate_internal(args) -> Dict[str, object]:
    diagnostic = shared.load_json(args.internal_diagnostic.resolve())
    checkpoints = diagnostic.get("checkpoints", {})
    results = diagnostic.get("results", {})
    record_checks = []
    finite_checks = []
    for expert in ("d2x", "d2y", "d2z"):
        for stratum in ("early", "mid", "final"):
            for timestep in ("0", "249", "499"):
                item = results.get(expert, {}).get(stratum, {}).get(
                    "timesteps", {},
                ).get(timestep, {})
                record_checks.append(
                    len(item.get("active_routed_residual_mse_by_sequence", [])) == 32
                    and len(item.get("inactive_routed_residual_mse_by_sequence", [])) == 32
                )
                values = (
                    item.get("active_routed_residual_rms"),
                    item.get("inactive_routed_residual_rms"),
                    item.get("gate_occupancy"),
                    item.get("uniform_vs_gated", {}).get("gated_gradient_norm"),
                    item.get("uniform_vs_gated", {}).get("uniform_gradient_norm"),
                )
                finite_checks.append(all(
                    value is not None and math.isfinite(float(value))
                    for value in values
                ))
    checks = {
        "schema_version": diagnostic.get("schema_version") == 1,
        "status": diagnostic.get("status") == "completed",
        "run_id": diagnostic.get("run_id") == INTERNAL_RUN_ID,
        "selection_sha256": (
            diagnostic.get("selection", {}).get("sha256") == INTERNAL_SELECTION_SHA256
        ),
        "selection_sequences": diagnostic.get("selection", {}).get("sequences") == 32,
        "selection_windows": diagnostic.get("selection", {}).get("windows") == 96,
        "gate_audit": (
            diagnostic.get("gate_audit", {}).get("sha256")
            == args.gate_audit_sha256
        ),
        "d2x_final": (
            checkpoints.get("d2x", {}).get("final_sha256")
            == CONTROL_CHECKPOINT_SHA256
        ),
        "d2y_final": (
            checkpoints.get("d2y", {}).get("final_sha256") == D2Y_CHECKPOINT_SHA256
        ),
        "d2z_final": (
            checkpoints.get("d2z", {}).get("final_sha256") == args.target_sha256
        ),
        "all_records": all(record_checks),
        "all_finite": all(finite_checks),
        "diagnostic_only": (
            diagnostic.get("diagnostic_summary", {}).get("selection_use") is False
            and diagnostic.get("checkpoint_selected") is False
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-Z internal diagnostic contract mismatch: {failed}")
    return {"checks": checks}


def validate_training_result(args) -> Dict[str, object]:
    expected_name = f"{TRAINING_RUN_ID}_windows061440000.pth"
    if args.target_checkpoint.name != expected_name:
        raise ValueError("D2-Z evaluation requires the registered final checkpoint basename")
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
        "amplification_support": "immutable_gt_previous_sampled_frame_near_ground",
        "gate_dtype": "bool",
        "gate_shape_per_window": [14, 4],
        "gate_stop_gradient": True,
        "floor_source": "complete_immutable_aligned_gt_sequence_30hz",
        "floor_algorithm": "code/eval_metrics.py::determine_floor_height_and_contacts",
        "foot_height_thresholds_m": {
            "7": 0.08, "8": 0.08, "10": 0.04, "11": 0.04,
        },
        "active_multiplier": 1024.0,
        "inactive_multiplier": 1.0,
        "gate_audit_path": str(args.gate_audit.resolve()),
        "gate_audit_sha256": args.gate_audit_sha256,
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
        "loss_routing": metrics.get("loss_routing") == expected_routing,
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
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-Z training artifact contract mismatch: {failed}")
    gate = _validate_gate_audit(args)
    internal = _validate_internal(args)
    checks["gate_audit_contract"] = True
    checks["internal_diagnostic_contract"] = True

    d2y_aggregate = shared.load_json(args.d2y_aggregate.resolve())
    d2y_per_sequence = shared.load_json(args.d2y_per_sequence.resolve())
    validate_candidate_result(
        "d2y-comparator",
        d2y_aggregate,
        d2y_per_sequence,
        D2Y_CHECKPOINT_SHA256,
    )
    _comparator.clear()
    _comparator.update({
        "aggregate": d2y_aggregate,
        "per_sequence": d2y_per_sequence,
        "aggregate_path": str(args.d2y_aggregate.resolve()),
        "per_sequence_path": str(args.d2y_per_sequence.resolve()),
        "gate_audit": gate,
        "internal": internal,
    })
    return {
        "checks": checks,
        "metrics": metrics,
        "gate_audit": gate,
        "internal_diagnostic": internal,
    }


def compare_records(
    control: Mapping[str, object],
    target: Mapping[str, object],
) -> Dict[str, object]:
    primary = _shared_compare_records(control, target)
    d2y_records = _comparator.get("per_sequence", {}).get("metrics")
    d2y_aggregate = _comparator.get("aggregate", {})
    if not isinstance(d2y_records, dict):
        raise ValueError("D2-Z evaluator has no validated D2-Y comparator records")
    full_comparison = _shared_compare_records(d2y_records, target)
    return {
        **primary,
        "mechanism_comparator_d2y": {
            "metrics": d2y_aggregate.get("metrics"),
            "full_paired_comparison": full_comparison,
            "d2y_minus_target_foot_sliding": shared.paired_difference(
                shared.metric_arrays(d2y_records, "foot_sliding"),
                shared.metric_arrays(target, "foot_sliding"),
            ),
            "target_over_d2y_protection": {
                metric: shared.paired_ratio(
                    shared.metric_arrays(target, metric),
                    shared.metric_arrays(d2y_records, metric),
                )
                for metric in (
                    "mpjpe", "end_obj_trans_err", "xy_points_err",
                    "obj_trans_dist", "contact_f1",
                )
            },
            "aggregate": {
                "path": _comparator["aggregate_path"],
                "sha256": D2Y_AGGREGATE_SHA256,
            },
            "per_sequence": {
                "path": _comparator["per_sequence_path"],
                "sha256": D2Y_PER_SEQUENCE_SHA256,
            },
            "reused_without_regeneration": True,
            "selection_use": False,
        },
    }


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
    full_contract = contract_passed and mask_passed
    mechanism_passed = full_contract and foot_improved and protection_passed
    effective_checks = {
        metric: float(baseline_ratios[metric]) <= threshold
        for metric, threshold in shared.EFFECTIVE_RATIO_MAX.items()
    }
    effective_checks["contact_f1_min"] = float(target_metrics["contact_f1"]) >= 0.60
    absolute_passed = all(effective_checks.values())
    effective_passed = mechanism_passed and absolute_passed
    if not full_contract:
        classification = "immutable-gt-near-ground-contract-failure-stop"
    elif foot_improved and protection_passed and not absolute_passed:
        classification = "immutable-gt-near-ground-positive-but-not-effective-stop"
    elif foot_improved and protection_passed:
        classification = "immutable-gt-near-ground-positive-candidate-stop"
    elif not foot_improved and protection_passed:
        classification = "immutable-gt-near-ground-transfer-negative-stop"
    elif foot_improved and not protection_passed:
        classification = "immutable-gt-near-ground-conflict-negative-stop"
    else:
        classification = "immutable-gt-near-ground-joint-negative-stop"
    return {
        "classification": classification,
        "contract_passed": full_contract,
        "penetration_mask_contract": comparison.get("penetration_mask_contract"),
        "official_foot_sliding_improved": foot_improved,
        "protection_passed": protection_passed,
        "protection_checks": protection_checks,
        "mechanism_passed": mechanism_passed,
        "effective_diffusion_passed": effective_passed,
        "effective_diffusion_checks": effective_checks,
        "d2y_comparator_used_for_selection": False,
        "internal_diagnostic_used_for_selection": False,
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
    shared.compare_records = compare_records
    shared.classify = classify


def main() -> None:
    configure_shared_module()
    shared.main()


if __name__ == "__main__":
    main()
