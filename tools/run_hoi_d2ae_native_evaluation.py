#!/usr/bin/env python3
"""Run fixed D2-AE0 native HOI evaluation against sealed D2-X."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from priors.d2ae_diagnostic import (  # noqa: E402
    CONTACT_F1_POINT_MINIMUM,
    PROTECTION_METRICS,
    native_gate,
)
from priors.models import HOI_ARCHITECTURE_D2AE  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
)
from tools import run_hoi_d2ac_native_evaluation as d2ac  # noqa: E402
from tools import run_hoi_d2x_evaluation as shared  # noqa: E402


_shared_resolved_config = shared.resolved_config


SUBPHASE = "1B-D2-AE0-native"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-native-eval"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
TRAINING_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-sparse-relation-field"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
INTERNAL_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-sparse-relation-field-internal"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
CONTROL_CHECKPOINT_SHA256 = d2ac.CONTROL_CHECKPOINT_SHA256
CONTROL_AGGREGATE_SHA256 = d2ac.CONTROL_AGGREGATE_SHA256
CONTROL_PER_SEQUENCE_SHA256 = d2ac.CONTROL_PER_SEQUENCE_SHA256
RELEASED_BASELINE_SHA256 = d2ac.RELEASED_BASELINE_SHA256
INTERNAL_SELECTION_SHA256 = d2ac.INTERNAL_SELECTION_SHA256

D2AC_CHECKPOINT_SHA256 = (
    "fede1c2b2f331407ceba7db16e3a4b30ccc6ffb6c8fc252861662bdcc96c7b96"
)
D2AC_AGGREGATE_SHA256 = (
    "3c996f0b3fcfb600b2dc7d88969b3f960799d57a98fad4562f9bbdd5ae88438c"
)
D2AC_PER_SEQUENCE_SHA256 = (
    "7cc9ab7688fd219b5c7fe9e22a373dbb93190cbfac1d6ba97593568946dfcb46"
)
D2AD_CHECKPOINT_SHA256 = (
    "f527d970243a42a1534b8db4437cd09dbc25334c832c3a13eb011f81db101c06"
)
D2AD_AGGREGATE_SHA256 = (
    "dab67e52af5a0a4cc443745fd9df713d445b06b069fc9469d7d3456b27f4fc27"
)
D2AD_PER_SEQUENCE_SHA256 = (
    "c6e15ab79c6d58bdc4419cd0ebe1ad260605d2ad310b611985ea117a26ef0460"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--training-metrics-sha256", required=True)
    parser.add_argument("--internal-diagnostic", type=Path, required=True)
    parser.add_argument("--internal-diagnostic-sha256", required=True)
    parser.add_argument("--control-aggregate", type=Path, required=True)
    parser.add_argument("--control-per-sequence", type=Path, required=True)
    parser.add_argument("--d2ac-aggregate", type=Path, required=True)
    parser.add_argument("--d2ac-per-sequence", type=Path, required=True)
    parser.add_argument("--d2ad-aggregate", type=Path, required=True)
    parser.add_argument("--d2ad-per-sequence", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def _validate_internal(args: argparse.Namespace) -> Dict[str, object]:
    value = d2ac.load_json(args.internal_diagnostic.resolve())
    decision = value.get("decision", {})
    checks = {
        "schema_version": value.get("schema_version") == 1,
        "status": value.get("status") == "completed",
        "run_id": bool(INTERNAL_RUN_ID_RE.fullmatch(str(value.get("run_id")))),
        "selection_sha256": value.get("selection", {}).get("sha256")
        == INTERNAL_SELECTION_SHA256,
        "selection_sequences": value.get("selection", {}).get("sequences") == 64,
        "selection_windows": value.get("selection", {}).get("windows") == 192,
        "target_checkpoint_sha256": value.get("target_checkpoint", {}).get(
            "sha256"
        ) == args.target_sha256,
        "target_checkpoint_run_id": value.get("target_checkpoint", {}).get(
            "run_id"
        ) == args.training_run_id,
        "contract_passed": decision.get("contract_passed") is True,
        "mechanism_fields": all(
            key in decision
            for key in (
                "relation_path_used",
                "temporal_routing_passed",
                "role_binding_passed",
                "mechanism_passed",
            )
        ),
        "paired_noise": value.get("contract", {}).get("paired_noise_identity")
        is True,
        "paired_exogenous_condition": value.get("contract", {}).get(
            "paired_exogenous_condition_identity"
        ) is True,
        "paired_initial_history": value.get("contract", {}).get(
            "paired_initial_history_identity"
        ) is True,
        "causal_window_overlap": value.get("contract", {}).get(
            "causal_window_overlap_exact"
        ) is True,
        "generator_draw_contract": value.get("contract", {}).get(
            "generator_draw_contract_exact"
        ) is True,
        "path_local_condition_provenance": value.get("contract", {}).get(
            "path_local_condition_provenance"
        ) is True,
        "current_state_relation": value.get("contract", {}).get(
            "current_state_relation_metadata_forwarded"
        ) is True,
        "relation_appendix": isinstance(
            value.get("sparse_relation_appendix"), dict,
        ),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AE internal diagnostic contract mismatch: {failed}")
    return {
        "checks": checks,
        "contract_passed": bool(decision["contract_passed"]),
        "relation_path_used": bool(decision["relation_path_used"]),
        "temporal_routing_passed": bool(decision["temporal_routing_passed"]),
        "role_binding_passed": bool(decision["role_binding_passed"]),
        "mechanism_passed": bool(decision["mechanism_passed"]),
        "classification": decision.get("classification"),
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": d2ac.sha256_file(args.internal_diagnostic.resolve()),
        "run_id": value["run_id"],
        "selection": value["selection"],
    }


def validate_training_result(args: argparse.Namespace) -> Dict[str, object]:
    metrics = d2ac.load_json(args.training_metrics.resolve())
    expected_name = f"{args.training_run_id}_windows061440000.pth"
    checkpoint_rows = [
        row for row in metrics.get("checkpoint_hashes", [])
        if row.get("processed_windows") == 61_440_000
        and row.get("sha256") == args.target_sha256
        and Path(row.get("path", "")).name == expected_name
    ]
    initialization = metrics.get("weight_initialization", {})
    routing = metrics.get("loss_routing", {})
    relation = metrics.get("sparse_relation_field", {})
    relation_contract = relation.get("contract", {})
    gradient_audit = metrics.get("sparse_relation_gradient_audit", {})
    initial_gradient = gradient_audit.get(
        "initial_zero_gate_alpha_gradient", {}
    ).get("alpha", {})
    activated_gradient = gradient_audit.get(
        "activated_relation_gradients", {}
    )
    activated_groups = activated_gradient.get("relation_groups", {})
    final_instance = relation.get("final_model_instance_contract", {})
    checks = {
        "status": metrics.get("status") == "stable",
        "run_id": metrics.get("run_id") == args.training_run_id,
        "seed": metrics.get("seed") == 42,
        "initialization": metrics.get("initialization") == "random",
        "training_start": metrics.get("training_start") == "random",
        "released_checkpoint_used": metrics.get("released_checkpoint_used") is False,
        "processed_windows": metrics.get("processed_windows") == 61_440_000,
        "processed_frames": metrics.get("processed_frames") == 983_040_000,
        "optimizer_updates": metrics.get("optimizer_updates") == 30_000,
        "world_size": metrics.get("world_size") == 4,
        "micro_batch_per_gpu": metrics.get("micro_batch_per_gpu") == 512,
        "effective_batch_size": metrics.get("effective_batch_size") == 2048,
        "optimization_contract": metrics.get("optimization_contract") == {
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
        },
        "loss_weights": metrics.get("loss_weights") == {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_object_goal": 1.0,
        },
        "routing": (
            routing.get("fk_foot_temporal_routing") is True
            and routing.get("d2ab_predicted_support_no_slip") is False
            and routing.get("d2ae_sparse_relation_field") is True
            and routing.get("global_bps_token_preserved") is True
        ),
        "model_config": metrics.get("model_config", {}).get(
            "architecture_variant"
        ) == HOI_ARCHITECTURE_D2AE,
        "sparse_relation_contract": (
            relation.get("architecture_variant") == HOI_ARCHITECTURE_D2AE
            and relation_contract.get("sparse_relation_parameters")
            == SPARSE_RELATION_PARAMETER_COUNT
            and relation_contract.get("mapping_sha256")
            == SPARSE_POINT_MAPPING_SHA256
            and relation_contract.get("manifest_sha256")
            == SPARSE_POINT_MANIFEST_SHA256
            and relation_contract.get("stacked_tensor_sha256")
            == SPARSE_POINT_TENSOR_SHA256
            and relation_contract.get("current_state_only") is True
            and relation_contract.get("clean_target_used") is False
            and relation_contract.get("future_gt_used") is False
            and relation_contract.get("scene_used") is False
            and relation_contract.get("contact_used") is False
            and relation_contract.get("stored_relation_used") is False
            and math.isfinite(float(relation.get("alpha", float("nan"))))
            and math.isfinite(float(relation.get("gate", float("nan"))))
        ),
        "sparse_relation_gradient_audit": (
            gradient_audit.get("schema_version") == 1
            and gradient_audit.get("run_id") == args.training_run_id
            and gradient_audit.get("seed") == 42
            and gradient_audit.get("optimizer_updates_observed") == [0, 1]
            and gradient_audit.get("probe_or_override_used") is False
            and initial_gradient.get("finite") is True
            and initial_gradient.get("nonzero") is True
            and activated_gradient.get("alpha", {}).get("finite") is True
            and activated_gradient.get("alpha", {}).get("nonzero") is True
            and bool(activated_groups)
            and all(
                value.get("finite") is True and value.get("nonzero") is True
                for value in activated_groups.values()
            )
        ),
        "sparse_relation_model_instance": (
            final_instance.get("base_parameters") == 29_673_448
            and final_instance.get("relation_parameters") == 413_953
            and final_instance.get("total_parameters") == 30_087_401
            and all(final_instance.get("checks", {}).values())
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(metrics.get("terminal_model_state_sha256", "")),
            ) is not None
        ),
        "random_weight_initialization": (
            initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("source_checkpoint_sha256") is None
            and initialization.get("source_model_state_sha256") is None
            and initialization.get("restored_components") == []
            and all(
                initialization.get(name) == 0
                for name in (
                    "old_optimizer_states_loaded",
                    "old_ema_models_loaded",
                    "old_scheduler_states_loaded",
                    "old_scaler_states_loaded",
                    "old_rng_states_loaded",
                )
            )
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(initialization.get("initial_model_state_sha256", "")),
            ) is not None
        ),
        "no_ema": metrics.get("ema_decays") == [],
        "primary_online": metrics.get("primary_weight_variant") == "online",
        "target_checkpoint_basename": args.target_checkpoint.name == expected_name,
        "final_checkpoint_hash": len(checkpoint_rows) == 1,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AE training artifact contract mismatch: {failed}")
    return {"checks": checks, "metrics": metrics}


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    value = _shared_resolved_config(args)
    value["subphase"] = SUBPHASE
    value["training_run_id"] = args.training_run_id
    value["internal_diagnostic"] = {
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": args.internal_diagnostic_sha256,
        "selection_sha256": INTERNAL_SELECTION_SHA256,
        "paired_unit": "sequence",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "selection_use": False,
    }
    value["sparse_relation"] = {
        "architecture_variant": HOI_ARCHITECTURE_D2AE,
        "current_diffusion_state_only": True,
        "rest_object_points": [100, 3],
        "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
        "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
        "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
        "scene": False,
        "future_gt": False,
        "stored_relation": False,
        "global_bps_token_preserved": True,
    }
    value["sealed_d2ac_descriptive_comparison"] = {
        "aggregate": str(args.d2ac_aggregate.resolve()),
        "aggregate_sha256": D2AC_AGGREGATE_SHA256,
        "per_sequence": str(args.d2ac_per_sequence.resolve()),
        "per_sequence_sha256": D2AC_PER_SEQUENCE_SHA256,
        "checkpoint_sha256": D2AC_CHECKPOINT_SHA256,
        "regenerated": False,
        "selection_use": False,
    }
    value["sealed_d2ad_descriptive_comparison"] = {
        "aggregate": str(args.d2ad_aggregate.resolve()),
        "aggregate_sha256": D2AD_AGGREGATE_SHA256,
        "per_sequence": str(args.d2ad_per_sequence.resolve()),
        "per_sequence_sha256": D2AD_PER_SEQUENCE_SHA256,
        "checkpoint_sha256": D2AD_CHECKPOINT_SHA256,
        "regenerated": False,
        "selection_use": False,
    }
    value["evaluation"]["protection_metrics"] = list(PROTECTION_METRICS)
    value["evaluation"]["native_gates"] = {
        "contact_f1_ci_lower_gt_zero": True,
        "contact_recall_ci_lower_gt_zero": True,
        "contact_f1_gap_closure_min": 0.25,
        "contact_f1_point_estimate_min": CONTACT_F1_POINT_MINIMUM,
        "protection_ratio_ci_upper_max": 1.10,
        "contact_precision_ci_lower_min": -0.02,
        "released_effectiveness_point_gate": "baseline 95 percent",
    }
    return value


def additional_runtime_artifact_hashes(args):
    return {
        "internal_diagnostic": shared.sha256_file(
            args.internal_diagnostic.resolve()
        ),
        "d2ac_aggregate": shared.sha256_file(args.d2ac_aggregate.resolve()),
        "d2ac_per_sequence": shared.sha256_file(
            args.d2ac_per_sequence.resolve()
        ),
        "d2ad_aggregate": shared.sha256_file(args.d2ad_aggregate.resolve()),
        "d2ad_per_sequence": shared.sha256_file(
            args.d2ad_per_sequence.resolve()
        ),
    }, {
        "internal_diagnostic": args.internal_diagnostic_sha256,
        "d2ac_aggregate": D2AC_AGGREGATE_SHA256,
        "d2ac_per_sequence": D2AC_PER_SEQUENCE_SHA256,
        "d2ad_aggregate": D2AD_AGGREGATE_SHA256,
        "d2ad_per_sequence": D2AD_PER_SEQUENCE_SHA256,
    }


_internal: Dict[str, object] = {}
_d2ac_aggregate: Dict[str, object] = {}
_d2ac_per_sequence: Dict[str, object] = {}
_d2ad_aggregate: Dict[str, object] = {}
_d2ad_per_sequence: Dict[str, object] = {}


def compare_records(
    control: Mapping[str, Mapping[str, object]],
    target: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    comparison = d2ac.compare_records(control, target)
    d2ac_descriptive = d2ac.compare_records(_d2ac_per_sequence, target)
    d2ac_descriptive["sealed_predecessor_metrics"] = _d2ac_aggregate
    d2ac_descriptive["selection_use"] = False
    d2ad_descriptive = d2ac.compare_records(_d2ad_per_sequence, target)
    d2ad_descriptive["sealed_predecessor_metrics"] = _d2ad_aggregate
    d2ad_descriptive["selection_use"] = False
    comparison["target_vs_sealed_d2ac_descriptive"] = d2ac_descriptive
    comparison["target_vs_sealed_d2ad_descriptive"] = d2ad_descriptive
    return comparison


def classify(
    comparison: Mapping[str, object],
    target_metrics: Mapping[str, object],
    baseline_ratios: Mapping[str, float],
    *,
    contract_passed: bool,
) -> Dict[str, object]:
    target_f1 = float(
        comparison["target_minus_control_contact_f1"]["first_mean"]
    )
    control_f1 = float(
        comparison["target_minus_control_contact_f1"]["second_mean"]
    )
    released_f1 = (
        float(target_metrics["contact_f1"])
        / float(baseline_ratios["contact_f1"])
        if float(baseline_ratios["contact_f1"]) else float("nan")
    )
    denominator = released_f1 - control_f1
    gap = (
        (target_f1 - control_f1) / denominator
        if math.isfinite(denominator) and denominator != 0.0
        else float("nan")
    )
    if isinstance(comparison, dict):
        comparison["contact_f1_released_gap_closure"] = gap
    comparison_value = dict(comparison)
    comparison_value["contact_f1_released_gap_closure"] = gap
    value = native_gate(
        contract_passed=contract_passed,
        internal=_internal,
        comparison=comparison_value,
        target_metrics=target_metrics,
        baseline_ratios=baseline_ratios,
    )
    value["internal_diagnostic"] = _internal
    value["comparison_gap_inputs"] = {
        "target_contact_f1_mean": target_f1,
        "control_contact_f1_mean": control_f1,
        "released_contact_f1": released_f1,
    }
    value["sealed_d2ac_descriptive_comparison_used_for_selection"] = False
    value["sealed_d2ad_descriptive_comparison_used_for_selection"] = False
    value["official_test_used"] = True
    return value


def configure_shared(
    args: argparse.Namespace,
    internal: Mapping[str, object],
    d2ac_aggregate: Mapping[str, object],
    d2ac_per_sequence: Mapping[str, object],
    d2ad_aggregate: Mapping[str, object],
    d2ad_per_sequence: Mapping[str, object],
) -> None:
    global _internal, _d2ac_aggregate, _d2ac_per_sequence
    global _d2ad_aggregate, _d2ad_per_sequence
    _internal = dict(internal)
    _d2ac_aggregate = dict(d2ac_aggregate)
    _d2ac_per_sequence = dict(d2ac_per_sequence)
    _d2ad_aggregate = dict(d2ad_aggregate)
    _d2ad_per_sequence = dict(d2ad_per_sequence)
    shared.RUN_ID = args.run_id
    shared.SUBPHASE = SUBPHASE
    shared.CONTROL_CHECKPOINT_SHA256 = CONTROL_CHECKPOINT_SHA256
    shared.CONTROL_AGGREGATE_SHA256 = CONTROL_AGGREGATE_SHA256
    shared.CONTROL_PER_SEQUENCE_SHA256 = CONTROL_PER_SEQUENCE_SHA256
    shared.BASELINE_SHA256 = RELEASED_BASELINE_SHA256
    shared.parse_args = lambda: args
    shared.resolved_config = resolved_config
    shared.additional_runtime_artifact_hashes = additional_runtime_artifact_hashes
    shared.validate_training_result = validate_training_result
    shared.compare_records = compare_records
    shared.classify = classify


def main() -> None:
    args = parse_args()
    run_id_match = RUN_ID_RE.fullmatch(args.run_id)
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if run_id_match is None or run_id_match.group("date") != actual_date:
        raise ValueError(
            "D2-AE native run id must use the locked stem and actual date"
        )
    if not TRAINING_RUN_ID_RE.fullmatch(args.training_run_id):
        raise ValueError("invalid D2-AE formal training run id")
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if (
        not configured_python
        or not Path(configured_python).is_absolute()
        or Path(sys.executable).resolve() != Path(configured_python).resolve()
        or args.python.resolve() != Path(configured_python).resolve()
    ):
        raise ValueError("D2-AE native evaluation requires the absolute INFBAGEL_PYTHON")
    for name in (
        "target_sha256",
        "training_metrics_sha256",
        "internal_diagnostic_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    if args.resolve_only:
        configure_shared(args, {}, {}, {}, {}, {})
        shared.prepare_resolved_config(args)
        return
    internal = _validate_internal(args)
    predecessor_hashes = {
        "d2ac_aggregate": d2ac.sha256_file(args.d2ac_aggregate.resolve()),
        "d2ac_per_sequence": d2ac.sha256_file(args.d2ac_per_sequence.resolve()),
        "d2ad_aggregate": d2ac.sha256_file(args.d2ad_aggregate.resolve()),
        "d2ad_per_sequence": d2ac.sha256_file(args.d2ad_per_sequence.resolve()),
    }
    if predecessor_hashes != {
        "d2ac_aggregate": D2AC_AGGREGATE_SHA256,
        "d2ac_per_sequence": D2AC_PER_SEQUENCE_SHA256,
        "d2ad_aggregate": D2AD_AGGREGATE_SHA256,
        "d2ad_per_sequence": D2AD_PER_SEQUENCE_SHA256,
    }:
        raise ValueError(
            f"sealed D2-AC/D2-AD artifact hash mismatch: {predecessor_hashes}"
        )
    d2ac_aggregate = d2ac.load_json(args.d2ac_aggregate.resolve())
    d2ac_per_sequence = d2ac.load_json(args.d2ac_per_sequence.resolve())
    d2ad_aggregate = d2ac.load_json(args.d2ad_aggregate.resolve())
    d2ad_per_sequence = d2ac.load_json(args.d2ad_per_sequence.resolve())
    shared.validate_candidate_result(
        "sealed-d2ac",
        d2ac_aggregate,
        d2ac_per_sequence,
        D2AC_CHECKPOINT_SHA256,
    )
    shared.validate_candidate_result(
        "sealed-d2ad",
        d2ad_aggregate,
        d2ad_per_sequence,
        D2AD_CHECKPOINT_SHA256,
    )
    configure_shared(
        args,
        internal,
        d2ac_aggregate.get("metrics", {}),
        d2ac_per_sequence.get("metrics", {}),
        d2ad_aggregate.get("metrics", {}),
        d2ad_per_sequence.get("metrics", {}),
    )
    shared.main()


if __name__ == "__main__":
    main()
