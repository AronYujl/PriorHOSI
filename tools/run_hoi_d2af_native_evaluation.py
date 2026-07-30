#!/usr/bin/env python3
"""Run fixed D2-AF0 native evaluation against sealed D2-AE and D2-X."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

from priors.d2af_diagnostic import (  # noqa: E402
    CONTACT_F1_POINT_MINIMUM,
    PROTECTION_METRICS,
    VARIANTS,
    internal_mechanism_gate,
    native_gate,
    paired_comparisons,
)
from priors.models import HOI_ARCHITECTURE_D2AF  # noqa: E402
from priors.remediation import selection_sha256  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    diffusion_reliability_contract_metadata,
)
from tools import run_hoi_d2ac_native_evaluation as d2ac  # noqa: E402
from tools import run_hoi_d2af_internal as internal_runner  # noqa: E402
from tools import run_hoi_d2x_evaluation as shared  # noqa: E402


_shared_resolved_config = shared.resolved_config


SUBPHASE = "1B-D2-AF0-native"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-native-eval"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
TRAINING_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-sqrt-alpha-bar-reliability"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
INTERNAL_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-sqrt-alpha-bar-reliability-internal"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
CONTROL_CHECKPOINT_SHA256 = d2ac.CONTROL_CHECKPOINT_SHA256
CONTROL_AGGREGATE_SHA256 = d2ac.CONTROL_AGGREGATE_SHA256
CONTROL_PER_SEQUENCE_SHA256 = d2ac.CONTROL_PER_SEQUENCE_SHA256
RELEASED_BASELINE_SHA256 = d2ac.RELEASED_BASELINE_SHA256
INTERNAL_SELECTION_SHA256 = d2ac.INTERNAL_SELECTION_SHA256
SEALED_D2AE_CHECKPOINT_SHA256 = (
    "b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840"
)
SEALED_D2AE_AGGREGATE_SHA256 = (
    "157acda463036bdf787618c217262c14c77a09a3f409cbeada03de06e9b902a1"
)
SEALED_D2AE_PER_SEQUENCE_SHA256 = (
    "8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c"
)
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c"
)
INTERNAL_SEQUENCE_NAMES_SHA256 = (
    "5eb8c14fe7e200481cf5cbaabb2e647b54de6e79082842ab13435c67aa179ac6"
)
EXPECTED_STREAM_COORDINATES = tuple(
    (chunk_index, window_index)
    for chunk_index in range(8)
    for window_index in range(3)
)
EXOGENOUS_SHAPES = {
    "global_object_goal": [8, 3],
    "global_pelvis_goal": [8, 3],
    "raw_progress": [8, 3],
    "rest_object_points": [8, 100, 3],
    "text": [8, 768],
}
MODEL_INPUT_SHAPES = {
    "fixed_history": [8, 2, 232],
    "global_bps": [8, 1024, 3],
    "local_goals": [8, 9],
    "normalized_progress": [8, 3],
    "object_maximum": [3],
    "object_minimum": [3],
    "object_rotation_reference": [8, 3, 3],
    "position_maximum": [3],
    "position_minimum": [3],
    "rest_object_points": [8, 100, 3],
    "world_to_local_rotation": [8, 3, 3],
}
RELATION_TRACE_SHAPES = {
    "pooled_block_norm": [500, 4, 3],
    "pooled_block_variance": [500, 4, 3],
    "relation_norm": [500, 4],
    "temporal_permutation_sensitivity": [500, 4],
    "role_swap_sensitivity": [500, 4],
    "gate": [500, 1],
    "rho": [500, 1],
    "raw_writeback_norm_by_anchor": [500, 4],
    "raw_writeback_variance_by_anchor": [500, 4],
    "attenuated_writeback_norm_by_anchor": [500, 4],
    "attenuated_writeback_variance_by_anchor": [500, 4],
}
RELATION_MEAN_SHAPES = {
    key: shape[1:] for key, shape in RELATION_TRACE_SHAPES.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--formal-manifest-sha256", required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--training-metrics-sha256", required=True)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--training-state-sha256", required=True)
    parser.add_argument("--resume-contract", type=Path, required=True)
    parser.add_argument("--resume-contract-sha256", required=True)
    parser.add_argument("--internal-diagnostic", type=Path, required=True)
    parser.add_argument("--internal-diagnostic-sha256", required=True)
    parser.add_argument("--control-aggregate", type=Path, required=True)
    parser.add_argument("--control-per-sequence", type=Path, required=True)
    parser.add_argument("--d2ae-aggregate", type=Path, required=True)
    parser.add_argument("--d2ae-per-sequence", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


INTERNAL_SUPPORT_ARTIFACTS = (
    "paired_noise",
    "paired_conditioning",
    "causal_window_overlap",
    "diffusion_reliability_appendix",
)
INTERNAL_ARTIFACT_FILENAMES = {
    **{variant: f"{variant}.json" for variant in VARIANTS},
    "paired_noise": "paired_noise.json",
    "paired_conditioning": "paired_conditioning.json",
    "causal_window_overlap": "causal_window_overlap.json",
    "diffusion_reliability_appendix": "diffusion_reliability_appendix.json",
}


def _closed_artifact_path(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
        raise ValueError("D2-AF internal artifact path must be relative")
    candidate = root / relative_path
    if candidate.is_symlink():
        raise ValueError("D2-AF internal artifact must not be a symlink")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("D2-AF internal artifact escapes its run directory") from error
    return path


def _lower_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite_nested(value: object) -> bool:
    if _finite_scalar(value):
        return True
    return (
        isinstance(value, list)
        and bool(value)
        and all(_finite_nested(item) for item in value)
    )


def _finite_scalar(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _shape_exact(value: object, shape: Sequence[int]) -> bool:
    if not shape:
        return _finite_scalar(value)
    return (
        isinstance(value, list)
        and len(value) == shape[0]
        and all(_shape_exact(item, shape[1:]) for item in value)
    )


def _stream_coordinates(rows: object) -> Optional[List[tuple[int, int]]]:
    if not isinstance(rows, list):
        return None
    coordinates = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        try:
            coordinates.append((
                int(row.get("chunk_index", -1)),
                int(row.get("window_index", -1)),
            ))
        except (TypeError, ValueError):
            return None
    return coordinates


def _hash_bundle_exact(
    value: object,
    expected_shapes: Mapping[str, Sequence[int]],
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"shapes", "sha256"}:
        return False
    shapes = value.get("shapes")
    hashes = value.get("sha256")
    return bool(
        isinstance(shapes, Mapping)
        and isinstance(hashes, Mapping)
        and dict(shapes) == {
            key: list(shape) for key, shape in expected_shapes.items()
        }
        and set(hashes) == set(expected_shapes)
        and all(_lower_sha256(item) for item in hashes.values())
    )


def _expected_provenance(window_index: int) -> Dict[str, str]:
    causal_source = (
        "immutable_selected_window_history"
        if window_index == 0
        else "previous_generated_tail_from_same_variant"
    )
    return {
        "fixed_history_source": causal_source,
        "frame_source": (
            "immutable_selected_window_frame"
            if window_index == 0
            else "previous_generated_tail_from_same_variant"
        ),
        "global_bps_reference": "same_path_local_frame.object_reference",
        "local_goal_reference": "same_path_local_frame",
        "relation_rotation_reference": "same_path_local_frame",
        "intervention_scope_per_model_call": (
            "rho_or_gate_or_temporal_geometry_blocks_or_"
            "left_right_pooled_blocks_only"
        ),
    }


def _noise_protocol(rows: object) -> bool:
    if _stream_coordinates(rows) != list(EXPECTED_STREAM_COORDINATES):
        return False
    assert isinstance(rows, list)
    expected_keys = {
        "chunk_index",
        "window_index",
        "label",
        "seed",
        "generator_initial_state_sha256",
        "generator_final_state_sha256",
        "draw_contract",
    }
    for row, (chunk_index, window_index) in zip(
        rows, EXPECTED_STREAM_COORDINATES,
    ):
        label = internal_runner.sampler_seed_label(chunk_index, window_index)
        if (
            set(row) != expected_keys
            or row.get("label") != label
            or row.get("seed") != internal_runner.base.stable_seed(label)
            or not _lower_sha256(row.get("generator_initial_state_sha256"))
            or not _lower_sha256(row.get("generator_final_state_sha256"))
            or row.get("draw_contract") != {
                "initial_latent_draws": 1,
                "posterior_noise_draws": 499,
                "total_generator_draws": 500,
                "draw_shape": [8, 16, 232],
                "timestep_zero_noise": "zeros_without_generator_draw",
            }
        ):
            return False
    return True


def _conditioning_protocol(rows: object) -> bool:
    if _stream_coordinates(rows) != list(EXPECTED_STREAM_COORDINATES):
        return False
    assert isinstance(rows, list)
    expected_keys = {
        "chunk_index",
        "window_index",
        "path_local_provenance",
        "exogenous",
        "path_local_model_inputs",
    }
    for row, (_, window_index) in zip(rows, EXPECTED_STREAM_COORDINATES):
        if (
            set(row) != expected_keys
            or row.get("path_local_provenance")
            != _expected_provenance(window_index)
            or not _hash_bundle_exact(row.get("exogenous"), EXOGENOUS_SHAPES)
            or not _hash_bundle_exact(
                row.get("path_local_model_inputs"), MODEL_INPUT_SHAPES,
            )
            or row["exogenous"]["sha256"]["rest_object_points"]
            != row["path_local_model_inputs"]["sha256"]["rest_object_points"]
        ):
            return False
    return True


def _variant_sequence_protocol(
    raw_variants: Mapping[str, Mapping[str, object]],
) -> tuple[bool, List[str]]:
    names_by_variant: Dict[str, List[str]] = {}
    identities_by_variant: Dict[str, List[object]] = {}
    for variant in VARIANTS:
        records = raw_variants[variant].get("per_sequence")
        if not isinstance(records, list) or len(records) != 64:
            return False, []
        names = []
        identities = []
        for record in records:
            if not isinstance(record, Mapping):
                return False, []
            name = record.get("sequence")
            positions = record.get("positions")
            pi = record.get("pi")
            if (
                not isinstance(name, str)
                or not isinstance(positions, list)
                or len(positions) != 3
                or not all(isinstance(item, int) for item in positions)
                or [item - positions[0] for item in positions] != [0, 42, 84]
                or pi != [14, 56, 98]
            ):
                return False, []
            names.append(name)
            identities.append({
                "sequence": name,
                "sequence_index": record.get("sequence_index"),
                "object_category": record.get("object_category"),
                "positions": positions,
                "pi": pi,
            })
        names_by_variant[variant] = names
        identities_by_variant[variant] = identities
    reference_names = names_by_variant["full_rho"]
    reference_identities = identities_by_variant["full_rho"]
    valid = bool(
        internal_runner.base.sequence_names_sha256(reference_names)
        == INTERNAL_SEQUENCE_NAMES_SHA256
        and all(
            names_by_variant[variant] == reference_names
            and identities_by_variant[variant] == reference_identities
            for variant in VARIANTS[1:]
        )
    )
    return valid, reference_names


def _causal_overlap_protocol(
    value: object,
    expected_names: Sequence[str],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    rows = value.get("rows")
    if (
        value.get("schema_version") != 1
        or value.get("phase_offsets") != [14, 56, 98]
        or value.get("prior_rollout_offsets") != [0, 42, 84]
        or value.get("source_window_frames") != 48
        or value.get("subsample_stride") != 3
        or value.get("model_window_frames") != 16
        or value.get("history_frames") != 2
        or value.get("sequences") != 64
        or value.get("all_exact") is not True
        or not isinstance(rows, list)
        or len(rows) != 64
    ):
        return False
    names = []
    global_indices = []
    expected_checks = {
        "single_sequence": True,
        "phase_offsets": True,
        "source_window_lengths": True,
        "sampled_window_lengths": True,
        "rollout_offsets": True,
        "history_overlap": True,
    }
    for cohort_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            return False
        starts = row.get("source_starts")
        ends = row.get("source_ends")
        overlap = row.get("sampled_tail_to_next_history")
        indices = row.get("global_indices")
        if (
            row.get("cohort_index") != cohort_index
            or not isinstance(row.get("sequence"), str)
            or not isinstance(indices, list)
            or len(indices) != 3
            or not all(isinstance(item, int) for item in indices)
            or row.get("phase_offsets") != [14, 56, 98]
            or not isinstance(starts, list)
            or not isinstance(ends, list)
            or len(starts) != 3
            or len(ends) != 3
            or not all(isinstance(item, int) for item in starts + ends)
            or [end - start for start, end in zip(starts, ends)]
            != [48, 48, 48]
            or [start - starts[0] for start in starts] != [0, 42, 84]
            or row.get("source_start_offsets") != [0, 42, 84]
            or not isinstance(overlap, list)
            or len(overlap) != 2
            or row.get("checks") != expected_checks
        ):
            return False
        expected_overlap = [
            {
                "previous_tail": [starts[step] + 42, starts[step] + 45],
                "next_history": [starts[step + 1], starts[step + 1] + 3],
                "exact": True,
            }
            for step in range(2)
        ]
        if overlap != expected_overlap:
            return False
        names.append(str(row["sequence"]))
        global_indices.extend(indices)
    return bool(
        names == list(expected_names)
        and internal_runner.base.sequence_names_sha256(names)
        == INTERNAL_SEQUENCE_NAMES_SHA256
        and selection_sha256(global_indices) == INTERNAL_SELECTION_SHA256
    )


def _relation_window_protocol(variant: str, windows: object) -> bool:
    if _stream_coordinates(windows) != list(EXPECTED_STREAM_COORDINATES):
        return False
    assert isinstance(windows, list)
    expected_mode = "unit" if variant == "unit_rho" else "canonical"
    expected_axis = {
        "temporal_anchors": [0, 5, 10, 15],
        "roles": ["left_hand", "right_hand", "pelvis"],
    }
    expected_rho = (
        [1.0] * 500
        if expected_mode == "unit"
        else list(reversed(
            internal_runner.canonical_diffusion_schedule()[
                "sqrt_alpha_bar"
            ].tolist()
        ))
    )
    expected_keys = {
        "chunk_index",
        "window_index",
        "forward_calls",
        "rho_mode",
        "rho_canonical_max_abs",
        "rho_unit_max_abs",
        "rho_batch_spread_max_abs",
        "rho_sentinels",
        "sqrt_alpha_bar_sha256",
        "axis",
        "values",
        "by_timestep",
        "by_timestep_sha256",
        "metadata",
    }
    for window in windows:
        if (
            set(window) != expected_keys
            or window.get("forward_calls") != 500
            or window.get("rho_mode") != expected_mode
            or window.get("sqrt_alpha_bar_sha256")
            != internal_runner.SQRT_ALPHA_BAR_SHA256
            or window.get("axis") != expected_axis
            or not isinstance(window.get("values"), Mapping)
            or set(window["values"]) != set(RELATION_MEAN_SHAPES)
            or not all(
                _shape_exact(window["values"][key], shape)
                for key, shape in RELATION_MEAN_SHAPES.items()
            )
            or not _finite_scalar(window.get("rho_batch_spread_max_abs"))
            or float(window["rho_batch_spread_max_abs"]) > 1.0e-7
        ):
            return False
        if expected_mode == "unit":
            if (
                not _finite_scalar(window.get("rho_unit_max_abs"))
                or float(window["rho_unit_max_abs"]) > 1.0e-7
            ):
                return False
        elif (
            not _finite_scalar(window.get("rho_canonical_max_abs"))
            or float(window["rho_canonical_max_abs"]) > 1.0e-7
        ):
            return False
        metadata = window.get("metadata")
        if metadata != {
            "rest_object_points_shape": [8, 100, 3],
            "world_to_local_rotation_shape": [8, 3, 3],
            "object_rotation_reference_shape": [8, 3, 3],
            "device": "cuda:0",
            "dtype": "torch.float32",
            "finite": True,
        }:
            return False
        by_timestep = window.get("by_timestep")
        if (
            not isinstance(by_timestep, Mapping)
            or by_timestep.get("timesteps") != list(reversed(range(500)))
            or by_timestep.get("axis") != expected_axis
            or not isinstance(by_timestep.get("values"), Mapping)
            or set(by_timestep["values"]) != set(RELATION_TRACE_SHAPES)
            or not all(
                _shape_exact(by_timestep["values"][key], shape)
                for key, shape in RELATION_TRACE_SHAPES.items()
            )
            or internal_runner.sha256_json(by_timestep)
            != window.get("by_timestep_sha256")
        ):
            return False
        observed_rho = [
            float(row[0]) for row in by_timestep["values"]["rho"]
        ]
        if any(
            abs(observed - expected) > 1.0e-7
            for observed, expected in zip(observed_rho, expected_rho)
        ):
            return False
        sentinels = window.get("rho_sentinels")
        if not isinstance(sentinels, Mapping):
            return False
        for timestep in internal_runner.SQRT_ALPHA_BAR_SENTINELS:
            observed = sentinels.get(str(timestep))
            expected = expected_rho[499 - timestep]
            if (
                not _finite_scalar(observed)
                or abs(float(observed) - expected) > 1.0e-7
            ):
                return False
    return True


def _formal_summary_matches(
    summary: object,
    current: Mapping[str, object],
) -> bool:
    if not isinstance(summary, Mapping):
        return False
    scalar_keys = (
        "training_run_id",
        "checkpoint_source_commit",
        "execution_target_commit",
        "execution_diff_sha256",
        "final_checkpoint_sha256",
        "final_model_state_sha256",
        "cadence_main_checkpoints",
        "cadence_rng_sidecars",
    )
    if any(summary.get(key) != current.get(key) for key in scalar_keys):
        return False
    if not summary.get("checks") or not all(summary.get("checks", {}).values()):
        return False
    summary_artifacts = summary.get("artifacts", {})
    current_artifacts = current.get("artifacts", {})
    return (
        isinstance(summary_artifacts, Mapping)
        and isinstance(current_artifacts, Mapping)
        and set(summary_artifacts) == set(current_artifacts)
        and all(
            summary_artifacts[name].get("sha256")
            == current_artifacts[name].get("sha256")
            for name in current_artifacts
        )
    )


def _validate_internal(
    args: argparse.Namespace,
    formal_lineage: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    internal_path = args.internal_diagnostic.resolve()
    internal_sha256 = d2ac.sha256_file(internal_path)
    if internal_sha256 != args.internal_diagnostic_sha256:
        raise ValueError("D2-AF internal diagnostic SHA-256 mismatch")
    if formal_lineage is None:
        formal_lineage = internal_runner.formal_artifact_contract(args)
    value = d2ac.load_json(internal_path)
    run_id = str(value.get("run_id"))
    decision = value.get("decision", {})
    contract = value.get("contract", {})
    closure = value.get("artifact_closure", {})
    raw_closure_artifacts = (
        closure.get("artifacts", {})
        if isinstance(closure, Mapping)
        else {}
    )
    closure_artifacts = (
        raw_closure_artifacts
        if isinstance(raw_closure_artifacts, Mapping)
        else {}
    )
    summary_variants = value.get("variants", {})
    if not isinstance(summary_variants, Mapping):
        summary_variants = {}
    expected_artifacts = set(VARIANTS) | set(INTERNAL_SUPPORT_ARTIFACTS)
    root = internal_path.parent
    payloads: Dict[str, Dict[str, object]] = {}
    resolved_artifacts: Dict[str, Dict[str, object]] = {}
    artifact_checks = {}
    for artifact_id in sorted(expected_artifacts):
        record = (
            closure_artifacts.get(artifact_id, {})
            if isinstance(closure_artifacts, Mapping)
            else {}
        )
        try:
            path = _closed_artifact_path(root, record.get("relative_path"))
            actual_sha256 = d2ac.sha256_file(path)
            payload = d2ac.load_json(path)
            artifact_checks[artifact_id] = bool(
                record.get("artifact_id") == artifact_id
                and record.get("run_id") == run_id
                and record.get("sha256") == actual_sha256
                and record.get("bytes") == path.stat().st_size
                and path.name == INTERNAL_ARTIFACT_FILENAMES[artifact_id]
                and payload.get("schema_version") == 1
                and payload.get("run_id") == run_id
            )
            payloads[artifact_id] = payload
            resolved_artifacts[artifact_id] = {
                **record,
                "path": str(path),
            }
        except (FileNotFoundError, OSError, TypeError, ValueError):
            artifact_checks[artifact_id] = False

    raw_variants = {
        variant: payloads.get(variant, {}) for variant in VARIANTS
    }
    variant_semantics = True
    for variant in VARIANTS:
        history_max_abs = raw_variants[variant].get("history_max_abs")
        variant_semantics = bool(
            variant_semantics
            and raw_variants[variant].get("variant") == variant
            and raw_variants[variant].get("training_run_id")
            == args.training_run_id
            and raw_variants[variant].get("target_checkpoint_sha256")
            == args.target_sha256
            and raw_variants[variant].get("rho_mode")
            == ("unit" if variant == "unit_rho" else "canonical")
            and raw_variants[variant].get("finite") is True
            and raw_variants[variant].get("all_fields_reported") is True
            and not isinstance(history_max_abs, bool)
            and isinstance(history_max_abs, (int, float))
            and math.isfinite(float(history_max_abs))
            and float(history_max_abs) <= internal_runner.HISTORY_MAX_ABS
            and raw_variants[variant].get("aggregate")
            == summary_variants.get(variant, {}).get("aggregate")
            and summary_variants.get(variant, {}).get("finite") is True
            and summary_variants.get(variant, {}).get(
                "all_fields_reported"
            ) is True
            and summary_variants.get(variant, {}).get("history_max_abs")
            == history_max_abs
            and summary_variants.get(variant, {}).get(
                "artifact", {}
            ).get("sha256")
            == closure_artifacts.get(variant, {}).get("sha256")
        )
    sequence_protocol, sequence_names = _variant_sequence_protocol(
        raw_variants,
    )
    records_by_variant = {
        variant: raw_variants[variant].get("per_sequence", [])
        for variant in VARIANTS
    }
    try:
        recomputed_comparisons = paired_comparisons(records_by_variant)
        recomputed_decision = internal_mechanism_gate(
            contract, recomputed_comparisons,
        )
    except (KeyError, TypeError, ValueError):
        recomputed_comparisons = {}
        recomputed_decision = {}

    paired_noise = payloads.get("paired_noise", {})
    paired_conditioning = payloads.get("paired_conditioning", {})
    causal_overlap = payloads.get("causal_window_overlap", {})
    appendix = payloads.get("diffusion_reliability_appendix", {})
    noise_protocol = all(
        _noise_protocol(raw_variants[variant].get("noise_streams"))
        for variant in VARIANTS
    )
    conditioning_protocol = all(
        _conditioning_protocol(
            raw_variants[variant].get("conditioning_streams")
        )
        for variant in VARIANTS
    )
    try:
        relation_protocol = all(
            _relation_window_protocol(
                variant, raw_variants[variant].get("relation_windows"),
            )
            for variant in VARIANTS
        )
    except (KeyError, TypeError, ValueError):
        relation_protocol = False
    raw_noise_identity = bool(
        noise_protocol
        and all(
            raw_variants[variant].get("noise_streams")
            == raw_variants["full_rho"].get("noise_streams")
            for variant in VARIANTS[1:]
        )
    )
    raw_exogenous_identity = bool(
        conditioning_protocol
        and all(
            [
                row["exogenous"]
                for row in raw_variants[variant]["conditioning_streams"]
            ]
            == [
                row["exogenous"]
                for row in raw_variants["full_rho"]["conditioning_streams"]
            ]
            for variant in VARIANTS[1:]
        )
    )
    raw_first_window_identity = bool(
        conditioning_protocol
        and all(
            [
                row["path_local_model_inputs"]
                for row in raw_variants[variant]["conditioning_streams"]
                if row["window_index"] == 0
            ]
            == [
                row["path_local_model_inputs"]
                for row in raw_variants["full_rho"]["conditioning_streams"]
                if row["window_index"] == 0
            ]
            for variant in VARIANTS[1:]
        )
    )
    paired_noise_exact = bool(
        raw_noise_identity
        and paired_noise.get("shared") is raw_noise_identity
        and set(paired_noise.get("variants", {})) == set(VARIANTS)
        and all(
            paired_noise["variants"][variant]
            == raw_variants[variant].get("noise_streams")
            for variant in VARIANTS
        )
    )
    paired_conditioning_exact = bool(
        raw_exogenous_identity
        and raw_first_window_identity
        and paired_conditioning.get("shared_exogenous")
        is raw_exogenous_identity
        and paired_conditioning.get("shared_first_window_model_inputs")
        is raw_first_window_identity
        and paired_conditioning.get("path_local_provenance_exact")
        is conditioning_protocol
        and paired_conditioning.get("first_window_model_input_keys")
        == sorted(internal_runner.FIRST_WINDOW_MODEL_INPUT_KEYS)
        and paired_conditioning.get("later_model_inputs")
        == "path-local after causal rollout divergence"
        and set(paired_conditioning.get("variants", {})) == set(VARIANTS)
        and all(
            paired_conditioning["variants"][variant]
            == raw_variants[variant].get("conditioning_streams")
            for variant in VARIANTS
        )
    )
    causal_overlap_exact = _causal_overlap_protocol(
        causal_overlap, sequence_names,
    )
    recomputed_relation: Dict[str, object] = {}
    if relation_protocol:
        try:
            recomputed_relation = {
                variant: internal_runner.aggregate_relation_windows(
                    raw_variants[variant]["relation_windows"]
                )
                for variant in VARIANTS
            }
        except (KeyError, TypeError, ValueError):
            recomputed_relation = {}
    appendix_exact = False
    if (
        recomputed_relation
        and isinstance(appendix, Mapping)
        and appendix.get("selection_use") is False
        and appendix.get("schedule")
        == internal_runner.diffusion_schedule_contract_metadata()
        and appendix.get("temporal_anchors") == [0, 5, 10, 15]
        and appendix.get("roles") == ["left_hand", "right_hand", "pelvis"]
        and set(appendix.get("variants", {})) == set(VARIANTS)
    ):
        try:
            appendix_exact = all(
                raw_variants[variant].get("aggregate", {}).get(
                    "diffusion_reliability"
                ) == recomputed_relation[variant]
                and appendix["variants"][variant].get("aggregate")
                == recomputed_relation[variant]
                and appendix["variants"][variant].get("per_window") == [
                    {
                        key: item
                        for key, item in window.items()
                        if key != "by_timestep"
                    }
                    for window in raw_variants[variant]["relation_windows"]
                ]
                for variant in VARIANTS
            )
        except (AttributeError, KeyError, TypeError):
            appendix_exact = False
    support_hash_bindings = all(
        value.get(artifact_id, {}).get("sha256")
        == closure_artifacts.get(artifact_id, {}).get("sha256")
        for artifact_id in INTERNAL_SUPPORT_ARTIFACTS
    )
    checks = {
        "schema_version": value.get("schema_version") == 1,
        "status": value.get("status") == "completed",
        "run_id": bool(INTERNAL_RUN_ID_RE.fullmatch(run_id)),
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
        "formal_lineage": _formal_summary_matches(
            value.get("formal_lineage"), formal_lineage,
        ),
        "artifact_closure_identity": (
            isinstance(closure, Mapping)
            and closure.get("schema_version") == 1
            and closure.get("run_id") == run_id
            and closure.get("training_run_id") == args.training_run_id
            and closure.get("target_checkpoint_sha256") == args.target_sha256
            and set(closure_artifacts) == expected_artifacts
        ),
        "artifact_hashes": bool(artifact_checks)
        and all(artifact_checks.values()),
        "variant_semantics": variant_semantics,
        "locked_sequence_cohort": sequence_protocol,
        "noise_raw_protocol": noise_protocol,
        "conditioning_raw_protocol": conditioning_protocol,
        "relation_raw_protocol": relation_protocol,
        "support_hash_bindings": support_hash_bindings,
        "paired_noise": paired_noise_exact,
        "paired_exogenous": raw_exogenous_identity,
        "paired_conditioning": paired_conditioning_exact,
        "first_window_model_input_identity": (
            raw_first_window_identity
            and contract.get(
                "paired_first_window_model_input_identity"
            ) is True
        ),
        "causal_window_overlap": (
            causal_overlap_exact
            and value.get("causal_window_overlap", {}).get("all_exact")
            is True
            and contract.get("causal_window_overlap_exact") is True
        ),
        "diffusion_reliability_appendix": appendix_exact,
        "generator_draw_contract": (
            noise_protocol
            and contract.get("generator_draw_contract_exact") is True
        ),
        "path_local_condition_provenance": (
            conditioning_protocol
            and contract.get("path_local_condition_provenance") is True
        ),
        "paired_noise_contract": (
            raw_noise_identity
            and contract.get("paired_noise_identity") is True
        ),
        "paired_exogenous_contract": (
            raw_exogenous_identity
            and contract.get("paired_exogenous_condition_identity") is True
        ),
        "comparisons_recomputed": (
            recomputed_comparisons == value.get("comparisons")
        ),
        "decision_recomputed": recomputed_decision == decision,
        "decision_evidence": (
            value.get("decision_evidence")
            == internal_runner.internal_decision_evidence(
                recomputed_decision
            )
        ),
        "seven_gate_checks": (
            isinstance(decision, Mapping)
            and isinstance(decision.get("checks"), Mapping)
            and len(decision["checks"]) == 7
        ),
        "contract_passed": decision.get("contract_passed") is True,
        "internal_status": decision.get("internal_status") in {
            "unused",
            "schedule-negative",
            "temporal-negative",
            "role-negative",
            "passed",
        },
        "native_authorized": (
            decision.get("native_evaluation_authorized") is True
            and value.get("native_evaluation_authorized") is True
        ),
        "current_state_relation": contract.get(
            "current_state_relation_metadata_forwarded"
        ) is True,
        "current_timestep": contract.get("current_timestep_forwarded") is True,
        "rho_variant_identity": contract.get("rho_variant_identity") is True,
        "schedule_hash": (
            relation_protocol
            and contract.get("canonical_schedule_hash") is True
        ),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AF internal diagnostic contract mismatch: {failed}")
    return {
        "checks": checks,
        "contract_passed": bool(decision["contract_passed"]),
        "internal_status": str(decision["internal_status"]),
        "relation_path_used": bool(decision["relation_path_used"]),
        "schedule_reliability_passed": bool(
            decision["schedule_reliability_passed"]
        ),
        "temporal_routing_passed": bool(decision["temporal_routing_passed"]),
        "role_binding_passed": bool(decision["role_binding_passed"]),
        "mechanism_passed": bool(decision["mechanism_passed"]),
        "classification": decision.get("classification"),
        "path": str(internal_path),
        "sha256": internal_sha256,
        "run_id": run_id,
        "selection": value["selection"],
        "artifact_closure": {
            **closure,
            "artifacts": resolved_artifacts,
        },
        "decision": decision,
    }


def validate_training_result(args: argparse.Namespace) -> Dict[str, object]:
    formal_lineage = (
        _formal_lineage
        if _formal_lineage
        else internal_runner.formal_artifact_contract(args)
    )
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
    initial_instance = relation.get("initial_model_instance_contract", {})
    final_instance = relation.get("final_model_instance_contract", {})
    lifecycle = metrics.get("d2af_lifecycle_contract", {})
    performance = lifecycle.get("performance_gate", {})
    eligibility = lifecycle.get("eligibility_gate", {})
    checks = {
        "formal_lineage": bool(formal_lineage.get("checks"))
        and all(formal_lineage["checks"].values()),
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
            and routing.get("d2ae_sparse_relation_field") is not True
            and routing.get("d2af_sqrt_alpha_bar_reliability") is True
            and routing.get("global_bps_token_preserved") is True
        ),
        "model_config": metrics.get("model_config", {}).get(
            "architecture_variant"
        ) == HOI_ARCHITECTURE_D2AF,
        "diffusion_reliability_contract": (
            relation.get("architecture_variant") == HOI_ARCHITECTURE_D2AF
            and relation_contract == diffusion_reliability_contract_metadata()
            and relation.get("diffusion_reliability") is True
            and relation.get("diagnostic_variant") == "full"
            and relation.get("gate_override") is None
            and relation.get("rho_override") is None
            and relation.get("capture_enabled") is False
            and math.isfinite(float(relation.get("alpha", float("nan"))))
            and math.isfinite(float(relation.get("gate", float("nan"))))
        ),
        "sparse_relation_assets": (
            relation_contract.get("sparse_relation_parameters")
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
            and relation_contract.get("loss_or_snr_weighting") is False
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
        "model_instances": (
            initial_instance.get("base_parameters") == 29_673_448
            and initial_instance.get("relation_parameters") == 413_953
            and initial_instance.get("total_parameters") == 30_087_401
            and all(initial_instance.get("checks", {}).values())
            and final_instance.get("base_parameters") == 29_673_448
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
            and initialization.get("initial_model_state_sha256")
            == EXPECTED_INITIAL_MODEL_STATE_SHA256
        ),
        "pretraining_gates": (
            lifecycle.get("eligibility_gate_required") is True
            and lifecycle.get("performance_gate_required") is True
            and isinstance(performance, Mapping)
            and bool(performance)
            and all(performance.get("checks", {}).values())
            and isinstance(eligibility, Mapping)
            and bool(eligibility)
            and all(eligibility.get("checks", {}).values())
            and lifecycle.get("schedule")
            == diffusion_reliability_contract_metadata()["schedule"]
        ),
        "predecessor_lifecycle_absent": (
            metrics.get("d2ae_lifecycle_contract") is None
        ),
        "no_ema": metrics.get("ema_decays") == [],
        "primary_online": metrics.get("primary_weight_variant") == "online",
        "target_checkpoint_basename": args.target_checkpoint.name == expected_name,
        "final_checkpoint_hash": len(checkpoint_rows) == 1,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AF training artifact contract mismatch: {failed}")
    return {"checks": checks, "metrics": metrics}


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    value = _shared_resolved_config(args)
    value["subphase"] = SUBPHASE
    value["training_run_id"] = args.training_run_id
    value["formal_lineage"] = {
        "manifest": {
            "path": str(args.formal_manifest.resolve()),
            "sha256": args.formal_manifest_sha256,
        },
        "metrics": {
            "path": str(args.training_metrics.resolve()),
            "sha256": args.training_metrics_sha256,
        },
        "training_state": {
            "path": str(args.training_state.resolve()),
            "sha256": args.training_state_sha256,
        },
        "resume_contract": {
            "path": str(args.resume_contract.resolve()),
            "sha256": args.resume_contract_sha256,
        },
        "checkpoint_source_commit": (
            internal_runner.FORMAL_CHECKPOINT_SOURCE_COMMIT
        ),
        "execution_target_commit": (
            internal_runner.FORMAL_EXECUTION_TARGET_COMMIT
        ),
        "execution_diff_sha256": internal_runner.FORMAL_EXECUTION_DIFF_SHA256,
        "final_checkpoint_sha256": (
            internal_runner.FORMAL_FINAL_CHECKPOINT_SHA256
        ),
        "final_model_state_sha256": (
            internal_runner.FORMAL_FINAL_MODEL_STATE_SHA256
        ),
    }
    value["internal_diagnostic"] = {
        "path": str(args.internal_diagnostic.resolve()),
        "sha256": args.internal_diagnostic_sha256,
        "selection_sha256": INTERNAL_SELECTION_SHA256,
        "paired_unit": "sequence",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "selection_use": False,
        "native_runs_regardless_of_internal_mechanism": True,
        "artifact_closure": _internal.get("artifact_closure", {}),
    }
    value["diffusion_reliability"] = {
        "architecture_variant": HOI_ARCHITECTURE_D2AF,
        "formula": (
            "motion + sqrt_alpha_bar[current_timestep] "
            "* tanh(alpha) * routed_relation"
        ),
        "contract": diffusion_reliability_contract_metadata(),
        "current_diffusion_state_only": True,
        "current_timestep_per_sample": True,
        "train_sample_symmetric": True,
        "rest_object_points": [100, 3],
        "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
        "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
        "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
        "scene": False,
        "future_gt": False,
        "stored_relation": False,
        "global_bps_token_preserved": True,
    }
    value["sealed_d2ae_repair_comparison"] = {
        "aggregate": str(args.d2ae_aggregate.resolve()),
        "aggregate_sha256": SEALED_D2AE_AGGREGATE_SHA256,
        "per_sequence": str(args.d2ae_per_sequence.resolve()),
        "per_sequence_sha256": SEALED_D2AE_PER_SEQUENCE_SHA256,
        "checkpoint_sha256": SEALED_D2AE_CHECKPOINT_SHA256,
        "checkpoint_loaded": False,
        "regenerated": False,
        "initializer_or_selector": False,
    }
    value["evaluation"]["protection_metrics"] = list(PROTECTION_METRICS)
    value["evaluation"]["native_gates"] = {
        "d2ae_repair": {
            "contact_f1_ci_lower_gt_zero": True,
            "contact_recall_ci_lower_gt_zero": True,
            "end_object_ratio_ci_upper_lt": 1.0,
            "foot_sliding_ratio_ci_upper_lt": 1.0,
        },
        "d2x_candidate": {
            "contact_f1_ci_lower_gt_zero": True,
            "contact_recall_ci_lower_gt_zero": True,
            "contact_f1_gap_closure_min": 0.25,
            "contact_f1_point_estimate_min": CONTACT_F1_POINT_MINIMUM,
        },
        "protection_ratio_ci_upper_max": 1.10,
        "contact_precision_ci_lower_min": -0.02,
        "released_effectiveness_point_gate": "baseline 95 percent",
        "all_conditions_conjunctive": True,
    }
    return value


def additional_runtime_artifact_hashes(args):
    actual = {
        "formal_manifest": shared.sha256_file(args.formal_manifest.resolve()),
        "training_state": shared.sha256_file(args.training_state.resolve()),
        "resume_contract": shared.sha256_file(args.resume_contract.resolve()),
        "internal_diagnostic": shared.sha256_file(
            args.internal_diagnostic.resolve()
        ),
        "sealed_d2ae_aggregate": shared.sha256_file(
            args.d2ae_aggregate.resolve()
        ),
        "sealed_d2ae_per_sequence": shared.sha256_file(
            args.d2ae_per_sequence.resolve()
        ),
    }
    expected = {
        "formal_manifest": args.formal_manifest_sha256,
        "training_state": args.training_state_sha256,
        "resume_contract": args.resume_contract_sha256,
        "internal_diagnostic": args.internal_diagnostic_sha256,
        "sealed_d2ae_aggregate": SEALED_D2AE_AGGREGATE_SHA256,
        "sealed_d2ae_per_sequence": SEALED_D2AE_PER_SEQUENCE_SHA256,
    }
    for artifact_id, record in _internal.get(
        "artifact_closure", {}
    ).get("artifacts", {}).items():
        key = f"internal_{artifact_id}"
        actual[key] = shared.sha256_file(Path(record["path"]))
        expected[key] = record["sha256"]
    return actual, expected


_internal: Dict[str, object] = {}
_formal_lineage: Dict[str, object] = {}
_d2ae_aggregate: Dict[str, object] = {}
_d2ae_per_sequence: Dict[str, object] = {}


def compare_records(
    control: Mapping[str, Mapping[str, object]],
    target: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    comparison = d2ac.compare_records(control, target)
    repair = d2ac.compare_records(_d2ae_per_sequence, target)
    repair["sealed_predecessor_metrics"] = _d2ae_aggregate
    repair["sealed_checkpoint_sha256"] = SEALED_D2AE_CHECKPOINT_SHA256
    repair["checkpoint_loaded"] = False
    repair["regenerated"] = False
    repair["initializer_or_selector"] = False
    comparison["target_vs_sealed_d2ae_repair"] = repair
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
    value["sealed_d2ae_checkpoint_loaded"] = False
    value["sealed_d2ae_regenerated"] = False
    value["sealed_d2ae_used_for_initialization_resume_or_selection"] = False
    value["official_test_used"] = True
    return value


def configure_shared(
    args: argparse.Namespace,
    internal: Mapping[str, object],
    formal_lineage: Mapping[str, object],
    d2ae_aggregate: Mapping[str, object],
    d2ae_per_sequence: Mapping[str, object],
) -> None:
    global _internal, _formal_lineage, _d2ae_aggregate, _d2ae_per_sequence
    _internal = dict(internal)
    _formal_lineage = dict(formal_lineage)
    _d2ae_aggregate = dict(d2ae_aggregate)
    _d2ae_per_sequence = dict(d2ae_per_sequence)
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
            "D2-AF native run id must use the locked stem and actual date"
        )
    if not TRAINING_RUN_ID_RE.fullmatch(args.training_run_id):
        raise ValueError("invalid D2-AF formal training run id")
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if (
        not configured_python
        or not Path(configured_python).is_absolute()
        or Path(sys.executable).resolve() != Path(configured_python).resolve()
        or args.python.resolve() != Path(configured_python).resolve()
    ):
        raise ValueError("D2-AF native evaluation requires the absolute INFBAGEL_PYTHON")
    for name in (
        "target_sha256",
        "formal_manifest_sha256",
        "training_metrics_sha256",
        "training_state_sha256",
        "resume_contract_sha256",
        "internal_diagnostic_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    formal_lineage = internal_runner.formal_artifact_contract(args)
    internal = _validate_internal(args, formal_lineage)
    if args.resolve_only:
        configure_shared(args, internal, formal_lineage, {}, {})
        shared.prepare_resolved_config(args)
        return

    predecessor_hashes = {
        "d2ae_aggregate": d2ac.sha256_file(args.d2ae_aggregate.resolve()),
        "d2ae_per_sequence": d2ac.sha256_file(
            args.d2ae_per_sequence.resolve()
        ),
    }
    if predecessor_hashes != {
        "d2ae_aggregate": SEALED_D2AE_AGGREGATE_SHA256,
        "d2ae_per_sequence": SEALED_D2AE_PER_SEQUENCE_SHA256,
    }:
        raise ValueError(
            f"sealed D2-AE artifact hash mismatch: {predecessor_hashes}"
        )
    d2ae_aggregate = d2ac.load_json(args.d2ae_aggregate.resolve())
    d2ae_per_sequence = d2ac.load_json(args.d2ae_per_sequence.resolve())
    shared.validate_candidate_result(
        "sealed-d2ae",
        d2ae_aggregate,
        d2ae_per_sequence,
        SEALED_D2AE_CHECKPOINT_SHA256,
    )
    configure_shared(
        args,
        internal,
        formal_lineage,
        d2ae_aggregate.get("metrics", {}),
        d2ae_per_sequence.get("metrics", {}),
    )
    shared.main()


if __name__ == "__main__":
    main()
