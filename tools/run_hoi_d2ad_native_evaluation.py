#!/usr/bin/env python3
"""Run fixed D2-AD0 native HOI evaluation against sealed D2-X."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from priors.d2ad import (  # noqa: E402
    BPS_YUP_TENSOR_SHA256,
    DEFAULT_QUERY_WORKERS,
    OBJECT_MAPPING_SHA256,
    REST_MESH_MANIFEST_SHA256,
)
from priors.interaction_diagnostic import native_gate  # noqa: E402
from tools import run_hoi_d2ac_native_evaluation as d2ac  # noqa: E402
from tools import run_hoi_d2x_evaluation as shared  # noqa: E402


_shared_resolved_config = shared.resolved_config


SUBPHASE = "1B-D2-AD0-native"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ad-native-eval(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
TRAINING_RUN_ID = "p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728"
INTERNAL_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ad-local-frame-interaction-adapter-internal"
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
CLASSIFICATION_MAP = {
    "interaction-adapter-contract-failure-stop":
        "local-frame-interaction-adapter-contract-failure-stop",
    "interaction-adapter-unused-optimization-negative-stop":
        "local-frame-interaction-adapter-unused-optimization-negative-stop",
    "interaction-adapter-locality-negative-stop":
        "local-frame-interaction-adapter-locality-negative-stop",
    "interaction-adapter-transfer-negative-stop":
        "local-frame-interaction-adapter-transfer-negative-stop",
    "interaction-adapter-conflict-negative-stop":
        "local-frame-interaction-adapter-conflict-negative-stop",
    "interaction-adapter-positive-but-not-effective-stop":
        "local-frame-interaction-adapter-positive-but-not-effective-stop",
    "interaction-adapter-positive-candidate-stop":
        "local-frame-interaction-adapter-positive-candidate-stop",
}


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
    parser.add_argument("--control-aggregate", type=Path, required=True)
    parser.add_argument("--control-per-sequence", type=Path, required=True)
    parser.add_argument("--d2ac-aggregate", type=Path, required=True)
    parser.add_argument("--d2ac-per-sequence", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def _validate_internal(args: argparse.Namespace) -> Dict[str, object]:
    value = d2ac.load_json(args.internal_diagnostic.resolve())
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
        "contract_passed": value.get("decision", {}).get("contract_passed") is True,
        "mechanism_fields": all(
            key in value.get("decision", {})
            for key in ("adapter_used", "locality_passed", "mechanism_passed")
        ),
        "paired_noise": value.get("contract", {}).get("paired_noise_identity")
        is True,
        "local_bps_recomputed": value.get("contract", {}).get(
            "local_bps_current_frame_recomputed"
        ) is True,
        "local_bps_appendix": isinstance(
            value.get("local_bps_appendix"), dict,
        ),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AD internal diagnostic contract mismatch: {failed}")
    return {
        "checks": checks,
        "contract_passed": bool(value["decision"]["contract_passed"]),
        "adapter_used": bool(value["decision"]["adapter_used"]),
        "locality_passed": bool(value["decision"]["locality_passed"]),
        "mechanism_passed": bool(value["decision"]["mechanism_passed"]),
        "classification": value["decision"].get("classification"),
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": d2ac.sha256_file(args.internal_diagnostic.resolve()),
        "run_id": value["run_id"],
        "selection": value["selection"],
    }


def validate_training_result(args: argparse.Namespace) -> Dict[str, object]:
    metrics = d2ac.load_json(args.training_metrics.resolve())
    expected_name = f"{TRAINING_RUN_ID}_windows061440000.pth"
    checkpoint_rows = [
        row for row in metrics.get("checkpoint_hashes", [])
        if row.get("processed_windows") == 61_440_000
        and row.get("sha256") == args.target_sha256
        and Path(row.get("path", "")).name == expected_name
    ]
    adapter = metrics.get("interaction_adapter", {})
    local_builder = metrics.get("local_bps_builder", {})
    initialization = metrics.get("weight_initialization", {})
    local_seconds = metrics.get("local_bps_build_seconds_by_rank")
    local_batches = metrics.get("local_bps_build_batches_by_rank")
    checks = {
        "status": metrics.get("status") == "stable",
        "run_id": metrics.get("run_id") == TRAINING_RUN_ID,
        "seed": metrics.get("seed") == 42,
        "initialization": metrics.get("initialization") == "random",
        "training_start": metrics.get("training_start") == "random",
        "released_checkpoint_used": metrics.get("released_checkpoint_used") is False,
        "processed_windows": metrics.get("processed_windows") == 61_440_000,
        "processed_frames": metrics.get("processed_frames") == 983_040_000,
        "optimizer_updates": metrics.get("optimizer_updates") == 30_000,
        "world_size": metrics.get("world_size") == 4,
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
            metrics.get("loss_routing", {}).get("fk_foot_temporal_routing")
            is True
            and metrics.get("loss_routing", {}).get(
                "d2ab_predicted_support_no_slip"
            ) is False
            and metrics.get("loss_routing", {}).get(
                "d2ad_local_frame_interaction_adapter"
            ) is True
            and metrics.get("loss_routing", {}).get(
                "global_bps_token_preserved"
            ) is True
        ),
        "model_config": metrics.get("model_config", {}).get(
            "architecture_variant"
        ) == "d2ad_local_frame_interaction_adapter",
        "adapter_contract": (
            adapter.get("architecture_variant")
            == "d2ad_local_frame_interaction_adapter"
            and adapter.get("contract", {}).get("adapter_parameters") == 349_697
            and adapter.get("contract", {}).get("basis_coordinate_system")
            == "human_window_local_y_up"
        ),
        "local_builder_contract": (
            local_builder.get("basis_coordinate_system")
            == "human_window_local_y_up"
            and local_builder.get("basis_yup_tensor_sha256")
            == BPS_YUP_TENSOR_SHA256
            and local_builder.get("rest_mesh_manifest_sha256")
            == REST_MESH_MANIFEST_SHA256
            and local_builder.get("object_mapping_sha256")
            == OBJECT_MAPPING_SHA256
            and local_builder.get("query_workers") == DEFAULT_QUERY_WORKERS
            and local_builder.get("full_rest_mesh") is True
            and local_builder.get("mesh_subsample") is False
            and local_builder.get("stored_per_window_local_bps") is False
        ),
        "local_bps_timing": (
            isinstance(local_seconds, list)
            and len(local_seconds) == 4
            and all(float(value) > 0.0 for value in local_seconds)
            and isinstance(local_batches, list)
            and len(local_batches) == 4
            and all(int(value) > 0 for value in local_batches)
        ),
        "random_weight_initialization": (
            initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("restored_components") == []
        ),
        "target_checkpoint_basename": args.target_checkpoint.name == expected_name,
        "final_checkpoint_hash": len(checkpoint_rows) == 1,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AD training artifact contract mismatch: {failed}")
    return {"checks": checks, "metrics": metrics}


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    value = _shared_resolved_config(args)
    value["subphase"] = SUBPHASE
    value["training_run_id"] = TRAINING_RUN_ID
    value["internal_diagnostic"] = {
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": args.internal_diagnostic_sha256,
        "selection_sha256": INTERNAL_SELECTION_SHA256,
        "paired_unit": "sequence",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "selection_use": False,
    }
    value["local_geometry"] = {
        "coordinate_system": "human_window_local_y_up",
        "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
        "rest_mesh_manifest_sha256": REST_MESH_MANIFEST_SHA256,
        "object_mapping_sha256": OBJECT_MAPPING_SHA256,
        "query_backend": "scipy.spatial.cKDTree.query",
        "query_parameters": {"k": 1, "eps": 0.0, "p": 2},
        "query_workers": DEFAULT_QUERY_WORKERS,
        "full_rest_mesh": True,
        "mesh_subsample": False,
        "stored_per_window_local_bps": False,
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
    value["evaluation"]["protection_metrics"] = list(d2ac.PROTECTION_METRICS)
    value["evaluation"]["native_gates"] = {
        "contact_f1_ci_lower_gt_zero": True,
        "contact_recall_ci_lower_gt_zero": True,
        "contact_f1_gap_closure_min": 0.25,
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
    }, {
        "internal_diagnostic": args.internal_diagnostic_sha256,
        "d2ac_aggregate": D2AC_AGGREGATE_SHA256,
        "d2ac_per_sequence": D2AC_PER_SEQUENCE_SHA256,
    }


_internal: Dict[str, object] = {}
_d2ac_aggregate: Dict[str, object] = {}
_d2ac_per_sequence: Dict[str, object] = {}


def compare_records(
    control: Mapping[str, Mapping[str, object]],
    target: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    comparison = d2ac.compare_records(control, target)
    descriptive = d2ac.compare_records(_d2ac_per_sequence, target)
    descriptive["sealed_d2ac_target_metrics"] = _d2ac_aggregate
    descriptive["selection_use"] = False
    comparison["target_vs_sealed_d2ac_descriptive"] = descriptive
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
    value["classification"] = CLASSIFICATION_MAP[value["classification"]]
    selectable = (
        value["classification"]
        == "local-frame-interaction-adapter-positive-candidate-stop"
    )
    value["checkpoint_selected"] = selectable
    value["selectable_autonomous_diffusion_candidate"] = selectable
    value.pop("d2ac1_authorized", None)
    value["d2ad1_authorized"] = False
    value["internal_diagnostic"] = _internal
    value["comparison_gap_inputs"] = {
        "target_contact_f1_mean": target_f1,
        "control_contact_f1_mean": control_f1,
        "released_contact_f1": released_f1,
    }
    value["sealed_d2ac_descriptive_comparison_used_for_selection"] = False
    value["official_test_used"] = True
    return value


def configure_shared(
    args: argparse.Namespace,
    internal: Mapping[str, object],
    d2ac_aggregate: Mapping[str, object],
    d2ac_per_sequence: Mapping[str, object],
) -> None:
    global _internal, _d2ac_aggregate, _d2ac_per_sequence
    _internal = dict(internal)
    _d2ac_aggregate = dict(d2ac_aggregate)
    _d2ac_per_sequence = dict(d2ac_per_sequence)
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
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("invalid D2-AD native lifecycle run id")
    for name in (
        "target_sha256",
        "training_metrics_sha256",
        "internal_diagnostic_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    if args.resolve_only:
        configure_shared(args, {}, {}, {})
        shared.prepare_resolved_config(args)
        return
    internal = _validate_internal(args)
    d2ac_hashes = {
        "aggregate": d2ac.sha256_file(args.d2ac_aggregate.resolve()),
        "per_sequence": d2ac.sha256_file(args.d2ac_per_sequence.resolve()),
    }
    if d2ac_hashes != {
        "aggregate": D2AC_AGGREGATE_SHA256,
        "per_sequence": D2AC_PER_SEQUENCE_SHA256,
    }:
        raise ValueError(f"sealed D2-AC artifact hash mismatch: {d2ac_hashes}")
    d2ac_aggregate = d2ac.load_json(args.d2ac_aggregate.resolve())
    d2ac_per_sequence = d2ac.load_json(args.d2ac_per_sequence.resolve())
    shared.validate_candidate_result(
        "sealed-d2ac",
        d2ac_aggregate,
        d2ac_per_sequence,
        D2AC_CHECKPOINT_SHA256,
    )
    configure_shared(
        args,
        internal,
        d2ac_aggregate.get("metrics", {}),
        d2ac_per_sequence.get("metrics", {}),
    )
    shared.main()


if __name__ == "__main__":
    main()
