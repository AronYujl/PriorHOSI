#!/usr/bin/env python3
"""Run the fixed D2-AC0 native HOI evaluation against sealed D2-X."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from priors.interaction_diagnostic import (  # noqa: E402
    PROTECTION_METRICS,
    paired_difference_fixed,
    paired_ratio_fixed,
    native_gate,
)
from tools import run_hoi_d2x_evaluation as shared  # noqa: E402


_shared_resolved_config = shared.resolved_config


SUBPHASE = "1B-D2-AC0-native"
RUN_ID_RE = re.compile(r"^p1-hoi-d2ac-native-eval-s42-[0-9]{8}$")
TRAINING_RUN_ID = "p1-hoi-d2ac-interaction-adapter-s42-20260726"
INTERNAL_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ac-interaction-adapter-internal-s42-[0-9]{8}$"
)
CONTROL_CHECKPOINT_SHA256 = (
    "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51"
)
CONTROL_AGGREGATE_SHA256 = (
    "3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b"
)
CONTROL_PER_SEQUENCE_SHA256 = (
    "69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a"
)
RELEASED_BASELINE_SHA256 = (
    "76fd86a3b28fa354ba552c004215acaf11e3396dc8eeb4752e0fc7a8186231e6"
)
INTERNAL_SELECTION_SHA256 = (
    "1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a"
)
PENETRATION_SEQUENCE_COUNT = 181
PENETRATION_SEQUENCE_IDS_SHA256 = (
    "2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec"
)
PER_SEQUENCE_KEYS = {
    "end_obj_trans_err": "end_obj_trans_err",
    "xy_points_err": "pelvis_goal_error_cm",
    "foot_sliding": "foot_sliding",
    "human_pen_loss_infbagel": "human_object_penetration",
    "hand_pen_loss_omomo": "hand_object_penetration",
    "mpjpe": "mpjpe",
    "trans_dist": "translation_difference",
    "obj_trans_dist": "object_translation_difference",
    "obj_rot_dist": "object_rotation_difference",
    "contact_precision": "contact_precision",
    "contact_recall": "contact_recall",
    "contact_f1": "contact_f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive_json(path: Path, value: object) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def _metric_value(record: Mapping[str, object], metric: str) -> float | None:
    key = PER_SEQUENCE_KEYS[metric]
    value = record.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_array(
    records: Mapping[str, Mapping[str, object]],
    metric: str,
    *,
    sequences: Sequence[str] | None = None,
) -> np.ndarray:
    names = sorted(records) if sequences is None else list(sequences)
    values = [_metric_value(records[name], metric) for name in names]
    if any(value is None for value in values):
        raise ValueError(f"native metric {metric} contains missing/nonfinite values")
    return np.asarray(values, dtype=np.float64)


def _finite_ids(
    records: Mapping[str, Mapping[str, object]],
    metric: str,
) -> Tuple[str, ...]:
    return tuple(
        name for name in sorted(records)
        if _metric_value(records[name], metric) is not None
    )


def _ids_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{name}\n" for name in names).encode("utf-8")
    ).hexdigest()


def _validate_internal(args: argparse.Namespace) -> Dict[str, object]:
    value = load_json(args.internal_diagnostic.resolve())
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
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AC internal diagnostic contract mismatch: {failed}")
    return {
        "checks": checks,
        "contract_passed": bool(value["decision"]["contract_passed"]),
        "adapter_used": bool(value["decision"]["adapter_used"]),
        "locality_passed": bool(value["decision"]["locality_passed"]),
        "mechanism_passed": bool(value["decision"]["mechanism_passed"]),
        "classification": value["decision"].get("classification"),
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": sha256_file(args.internal_diagnostic.resolve()),
        "run_id": value["run_id"],
        "selection": value["selection"],
    }


def validate_training_result(args: argparse.Namespace) -> Dict[str, object]:
    metrics = load_json(args.training_metrics.resolve())
    expected_name = f"{TRAINING_RUN_ID}_windows061440000.pth"
    checkpoint_rows = [
        row for row in metrics.get("checkpoint_hashes", [])
        if row.get("processed_windows") == 61_440_000
        and row.get("sha256") == args.target_sha256
        and Path(row.get("path", "")).name == expected_name
    ]
    adapter = metrics.get("interaction_adapter", {})
    initialization = metrics.get("weight_initialization", {})
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
        ),
        "model_config": metrics.get("model_config", {}).get(
            "architecture_variant"
        ) == "d2ac_interaction_adapter",
        "adapter_contract": (
            adapter.get("architecture_variant") == "d2ac_interaction_adapter"
            and adapter.get("contract", {}).get("adapter_parameters") == 349_697
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
        raise ValueError(f"D2-AC training artifact contract mismatch: {failed}")
    return {"checks": checks, "metrics": metrics}


def compare_records(
    control: Mapping[str, Mapping[str, object]],
    target: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    if sorted(control) != sorted(target):
        raise ValueError("D2-AC native control/target sequence identities differ")
    control_hand = _finite_ids(control, "hand_pen_loss_omomo")
    control_human = _finite_ids(control, "human_pen_loss_infbagel")
    target_hand = _finite_ids(target, "hand_pen_loss_omomo")
    target_human = _finite_ids(target, "human_pen_loss_infbagel")
    mask_checks = {
        "control_penetration_fields_match": control_hand == control_human,
        "control_finite_count": len(control_hand) == PENETRATION_SEQUENCE_COUNT,
        "control_finite_ids_sha256": _ids_sha256(control_hand)
        == PENETRATION_SEQUENCE_IDS_SHA256,
        "target_hand_mask_matches_control": target_hand == control_hand,
        "target_human_mask_matches_control": target_human == control_hand,
        "target_penetration_fields_match": target_hand == target_human,
    }
    mask_passed = all(mask_checks.values())
    protection = {}
    for metric in PROTECTION_METRICS:
        if metric in ("hand_pen_loss_omomo", "human_pen_loss_infbagel"):
            if not mask_passed:
                continue
            numerator = _metric_array(target, metric, sequences=control_hand)
            denominator = _metric_array(control, metric, sequences=control_hand)
        else:
            numerator = _metric_array(target, metric)
            denominator = _metric_array(control, metric)
        protection[metric] = paired_ratio_fixed(numerator, denominator)
    comparison = {
        "target_over_control_protection": protection,
        "penetration_mask_contract": {
            "passed": mask_passed,
            "checks": mask_checks,
            "official_sequences": len(control),
            "finite_sequences": len(control_hand),
            "finite_sequence_ids_sha256": _ids_sha256(control_hand),
        },
        "target_minus_control_contact_precision": paired_difference_fixed(
            _metric_array(target, "contact_precision"),
            _metric_array(control, "contact_precision"),
        ),
        "target_minus_control_contact_recall": paired_difference_fixed(
            _metric_array(target, "contact_recall"),
            _metric_array(control, "contact_recall"),
        ),
        "target_minus_control_contact_f1": paired_difference_fixed(
            _metric_array(target, "contact_f1"),
            _metric_array(control, "contact_f1"),
        ),
        "control_minus_target_foot_sliding": paired_difference_fixed(
            _metric_array(control, "foot_sliding"),
            _metric_array(target, "foot_sliding"),
        ),
    }
    return comparison


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    # The shared D2-X wrapper supplies the exact production command and sealed
    # evaluator hashes.  Add only the D2-AC-specific fixed artifacts.
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
    value["evaluation"]["protection_metrics"] = list(PROTECTION_METRICS)
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
    }, {
        "internal_diagnostic": args.internal_diagnostic_sha256,
    }


_internal: Dict[str, object] = {}


def classify(
    comparison: Mapping[str, object],
    target_metrics: Mapping[str, object],
    baseline_ratios: Mapping[str, float],
    *,
    contract_passed: bool,
) -> Dict[str, object]:
    # The released aggregate is recoverable from the target/baseline ratio
    # supplied by the sealed shared wrapper.  Use the per-sequence means for
    # the paired gap numerator/denominator.
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
    value["official_test_used"] = True
    return value


def configure_shared(args: argparse.Namespace, internal: Mapping[str, object]):
    global _internal
    _internal = dict(internal)
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
        raise ValueError("invalid D2-AC native lifecycle run id")
    for name in ("target_sha256", "training_metrics_sha256", "internal_diagnostic_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    if args.resolve_only:
        # No diagnostic/checkpoint loading is performed by config resolution.
        configure_shared(args, {})
        shared.prepare_resolved_config(args)
        return
    internal = _validate_internal(args)
    configure_shared(args, internal)
    shared.main()


if __name__ == "__main__":
    main()
