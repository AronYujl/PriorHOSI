#!/usr/bin/env python3
"""Run fixed D2-AG0 native evaluation against the sealed D2-X control."""

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

from priors.d2ag_diagnostic import (  # noqa: E402
    CONTACT_F1_POINT_MINIMUM,
    PROTECTION_METRICS,
    REFERENCE_VARIANT,
    SOURCE_IS_CURRENT_STEP_COUNTS,
    VARIANTS,
    internal_mechanism_gate,
    native_gate,
    paired_comparisons,
)
from priors.models import HOI_ARCHITECTURE_D2AG  # noqa: E402
from priors.remediation import selection_sha256  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    selfcond_relation_source_contract_metadata,
)
from tools import run_hoi_d2ac_native_evaluation as d2ac  # noqa: E402
from tools import run_hoi_d2ag_internal as internal_runner  # noqa: E402
from tools import run_hoi_d2x_evaluation as shared  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    D2AG_FORBIDDEN_WAIVED_STATUS,
    D2AG_MAXIMUM_ETA_HOURS,
    D2AG_MINIMUM_THROUGHPUT,
    D2AG_PERFORMANCE_FAILURE_CLASSIFICATION,
    D2AG_PERFORMANCE_WAIVER_CLASSIFICATION,
    D2AG_PERFORMANCE_WAIVER_STATUS,
    D2AG_WAIVED_BENCHMARK_RUN_ID,
    D2AG_WAIVED_BENCHMARK_SHA256,
    D2AG_WAIVED_ETA_HOURS,
    D2AG_WAIVED_FORMAL_RUN_ID,
    D2AG_WAIVED_THROUGHPUT,
)


_shared_resolved_config = shared.resolved_config


SUBPHASE = "1B-D2-AG0-native"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-native-eval(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
TRAINING_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-selfcond-relation-source"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
INTERNAL_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-selfcond-relation-source-internal"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
# Sealed D2-X control and released baseline, reused by reference and never
# recomputed.  Delegated so a single sealed source of truth stays authoritative.
CONTROL_CHECKPOINT_SHA256 = d2ac.CONTROL_CHECKPOINT_SHA256
CONTROL_AGGREGATE_SHA256 = d2ac.CONTROL_AGGREGATE_SHA256
CONTROL_PER_SEQUENCE_SHA256 = d2ac.CONTROL_PER_SEQUENCE_SHA256
RELEASED_BASELINE_SHA256 = d2ac.RELEASED_BASELINE_SHA256
INTERNAL_SELECTION_SHA256 = d2ac.INTERNAL_SELECTION_SHA256
# Zero new parameters and the same 64-sequence internal cohort, so both values
# are byte-identical to D2-AE/D2-AF.  A test pins them against the D2-AF module.
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
    "writeback_norm_by_anchor": [500, 4],
    "writeback_variance_by_anchor": [500, 4],
    "source_minus_current_l2_by_anchor": [500, 4],
    "source_history_max_abs": [500, 1],
}
RELATION_MEAN_SHAPES = {
    key: shape[1:] for key, shape in RELATION_TRACE_SHAPES.items()
}
INTERNAL_SUPPORT_ARTIFACTS = (
    "paired_noise",
    "paired_conditioning",
    "causal_window_overlap",
    "self_conditioning_appendix",
    "reported_only_quantities",
)
INTERNAL_ARTIFACT_FILENAMES = {
    **{variant: f"{variant}.json" for variant in VARIANTS},
    "paired_noise": "paired_noise.json",
    "paired_conditioning": "paired_conditioning.json",
    "causal_window_overlap": "causal_window_overlap.json",
    "self_conditioning_appendix": internal_runner.SELFCOND_APPENDIX_FILENAME,
    "reported_only_quantities": internal_runner.REPORTED_ONLY_FILENAME,
}


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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def _closed_artifact_path(root: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
        raise ValueError("D2-AG internal artifact path must be relative")
    candidate = root / relative_path
    if candidate.is_symlink():
        raise ValueError("D2-AG internal artifact must not be a symlink")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "D2-AG internal artifact escapes its run directory"
        ) from error
    return path


def _lower_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


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


def _stream_coordinates(rows: object) -> Optional[List[tuple]]:
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
        "self_conditioning_source": "same_path_prev_x0",
        "intervention_scope_per_model_call": internal_runner.INTERVENTION_SCOPE,
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
        # ``relation_source`` is path-local generated state that gates 1-3 change
        # by design, so it must never appear in either hashed identity bundle.
        if (
            "relation_source" in row["exogenous"]["sha256"]
            or "relation_source" in row["path_local_model_inputs"]["sha256"]
        ):
            return False
    return True


def _variant_sequence_protocol(
    raw_variants: Mapping[str, Mapping[str, object]],
) -> tuple:
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
    reference_names = names_by_variant[REFERENCE_VARIANT]
    reference_identities = identities_by_variant[REFERENCE_VARIANT]
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
            or [end - start for start, end in zip(starts, ends)] != [48, 48, 48]
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
    """Structural re-derivation of one variant's per-window capture records."""
    if _stream_coordinates(windows) != list(EXPECTED_STREAM_COORDINATES):
        return False
    assert isinstance(windows, list)
    expected_axis = {
        "temporal_anchors": [0, 5, 10, 15],
        "roles": ["left_hand", "right_hand", "pelvis"],
    }
    expected_count = SOURCE_IS_CURRENT_STEP_COUNTS[variant]
    expected_first = internal_runner.expected_first_step_source(variant)
    expected_keys = {
        "chunk_index",
        "window_index",
        "forward_calls",
        "sources",
        "source_is_x_t_count",
        "first_step_source",
        "source_history_max_abs",
        "estimate_pass_observed",
        "axis",
        "values",
        "by_timestep",
        "by_timestep_sha256",
        "metadata",
    }
    for window in windows:
        sources = window.get("sources")
        if (
            set(window) != expected_keys
            or window.get("forward_calls") != 500
            or window.get("estimate_pass_observed") is not False
            or window.get("first_step_source") != expected_first
            or window.get("source_is_x_t_count") != expected_count
            or not _finite_scalar(window.get("source_history_max_abs"))
            or float(window["source_history_max_abs"]) != 0.0
            or window.get("axis") != expected_axis
            or not isinstance(sources, list)
            or len(sources) != 500
            or not all(
                item in {
                    internal_runner.SOURCE_LABEL_CURRENT,
                    internal_runner.SOURCE_LABEL_PREVIOUS,
                }
                for item in sources
            )
            or sources[0] != expected_first
            or sum(
                1 for item in sources
                if item == internal_runner.SOURCE_LABEL_CURRENT
            ) != expected_count
            or not isinstance(window.get("values"), Mapping)
            or set(window["values"]) != set(RELATION_MEAN_SHAPES)
            or not all(
                _shape_exact(window["values"][key], shape)
                for key, shape in RELATION_MEAN_SHAPES.items()
            )
        ):
            return False
        if window.get("metadata") != {
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
            or by_timestep.get("sources") != sources
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
        # The per-step source label must agree with the recorded per-anchor
        # source-minus-current L2: an ``x_t`` step is exactly zero everywhere.
        for label, row in zip(
            sources, by_timestep["values"]["source_minus_current_l2_by_anchor"],
        ):
            zero = all(float(value) == 0.0 for value in row)
            if (label == internal_runner.SOURCE_LABEL_CURRENT) != zero:
                return False
        if any(
            float(value[0]) != 0.0
            for value in by_timestep["values"]["source_history_max_abs"]
        ):
            return False
    return True


def _checkpoint_summary_matches(
    summary: object,
    args: argparse.Namespace,
) -> bool:
    """Bind the internal diagnostic's own checkpoint record to this CLI.

    D2-AG is straight-through, so there is no sealed multi-artifact lineage to
    recompute here.  What must still hold is that the diagnostic evaluated the
    same file this evaluation is about, that it recorded the fixed final-online
    identity, and that every semantic checkpoint check it ran passed.
    """
    if not isinstance(summary, Mapping):
        return False
    expected_name = f"{args.training_run_id}_windows061440000.pth"
    checks = summary.get("checks")
    return bool(
        summary.get("sha256") == args.target_sha256
        and summary.get("run_id") == args.training_run_id
        and summary.get("basename") == expected_name
        and Path(str(summary.get("path", ""))).name == expected_name
        and summary.get("processed_windows") == 61_440_000
        and summary.get("weight_variant") == "online"
        and re.fullmatch(
            r"[0-9a-f]{64}", str(summary.get("initial_model_state_sha256", "")),
        )
        and summary.get("initial_model_state_sha256")
        == EXPECTED_INITIAL_MODEL_STATE_SHA256
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(checks.values())
    )


def _validate_internal(args: argparse.Namespace) -> Dict[str, object]:
    internal_path = args.internal_diagnostic.resolve()
    internal_sha256 = d2ac.sha256_file(internal_path)
    if internal_sha256 != args.internal_diagnostic_sha256:
        raise ValueError("D2-AG internal diagnostic SHA-256 mismatch")
    value = d2ac.load_json(internal_path)
    run_id = str(value.get("run_id"))
    decision = value.get("decision", {})
    contract = value.get("contract", {})
    closure = value.get("artifact_closure", {})
    raw_closure_artifacts = (
        closure.get("artifacts", {}) if isinstance(closure, Mapping) else {}
    )
    closure_artifacts = (
        raw_closure_artifacts
        if isinstance(raw_closure_artifacts, Mapping) else {}
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
            if isinstance(closure_artifacts, Mapping) else {}
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
            resolved_artifacts[artifact_id] = {**record, "path": str(path)}
        except (FileNotFoundError, OSError, TypeError, ValueError):
            artifact_checks[artifact_id] = False

    raw_variants = {variant: payloads.get(variant, {}) for variant in VARIANTS}
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
            and raw_variants[variant].get("registered_source_is_x_t_count")
            == SOURCE_IS_CURRENT_STEP_COUNTS[variant]
            and raw_variants[variant].get("registered_first_step_source")
            == internal_runner.expected_first_step_source(variant)
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
            and summary_variants.get(variant, {}).get("artifact", {}).get(
                "sha256"
            ) == closure_artifacts.get(variant, {}).get("sha256")
        )
    sequence_protocol, sequence_names = _variant_sequence_protocol(raw_variants)
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
    appendix = payloads.get("self_conditioning_appendix", {})
    reported_only = payloads.get("reported_only_quantities", {})
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
            == raw_variants[REFERENCE_VARIANT].get("noise_streams")
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
                for row in raw_variants[REFERENCE_VARIANT][
                    "conditioning_streams"
                ]
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
                for row in raw_variants[REFERENCE_VARIANT][
                    "conditioning_streams"
                ]
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
        and paired_conditioning.get("relation_source_in_identity_set") is False
        and paired_conditioning.get("first_window_model_input_keys")
        == sorted(MODEL_INPUT_SHAPES)
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
        and appendix.get("sqrt_alpha_bar_attenuation") is False
        and appendix.get("schedule")
        == internal_runner.diffusion_schedule_contract_metadata()
        and appendix.get("temporal_anchors") == [0, 5, 10, 15]
        and appendix.get("variable_anchors") == list(D2AG_VARIABLE_ANCHORS)
        and appendix.get("roles") == ["left_hand", "right_hand", "pelvis"]
        and appendix.get("registered_source_is_x_t_counts")
        == dict(SOURCE_IS_CURRENT_STEP_COUNTS)
        and set(appendix.get("variants", {})) == set(VARIANTS)
    ):
        try:
            appendix_exact = all(
                raw_variants[variant].get("aggregate", {}).get(
                    "self_conditioning"
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
    reported_only_exact = bool(
        isinstance(reported_only, Mapping)
        and reported_only.get("reported_only") is True
        and reported_only.get("enters_internal_mechanism_gate") is False
        and reported_only.get("enters_native_gate") is False
        and reported_only.get("selection_use") is False
        and set(reported_only.get("variants", {})) == set(VARIANTS)
        and all(
            raw_variants[variant].get("reported_only")
            == reported_only["variants"][variant]
            for variant in VARIANTS
        )
    )
    # Independent proof that the report-only quantities are inert: recomputing
    # the gate from the contract alone must reproduce the recorded decision, and
    # the decision payload must declare that it never consulted them.
    reported_only_inert = bool(
        recomputed_decision
        and recomputed_decision == decision
        and decision.get("reported_only_quantities_used") is False
    )
    checks = {
        "schema_version": value.get("schema_version") == 1,
        "status": value.get("status") == "completed",
        "run_id": bool(INTERNAL_RUN_ID_RE.fullmatch(run_id)),
        "selection_sha256": value.get("selection", {}).get("sha256")
        == INTERNAL_SELECTION_SHA256,
        "selection_sequences": value.get("selection", {}).get("sequences") == 64,
        "selection_windows": value.get("selection", {}).get("windows") == 192,
        "target_checkpoint_contract": _checkpoint_summary_matches(
            value.get("target_checkpoint"), args,
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
        "six_variants_no_gate_ablation": (
            len(VARIANTS) == 6 and "relation_gate_ablated" not in VARIANTS
        ),
        "locked_sequence_cohort": sequence_protocol,
        "noise_raw_protocol": noise_protocol,
        "conditioning_raw_protocol": conditioning_protocol,
        "relation_raw_protocol": relation_protocol,
        "paired_noise": paired_noise_exact,
        "paired_exogenous": raw_exogenous_identity,
        "paired_conditioning": paired_conditioning_exact,
        "relation_source_excluded_from_identity_set": (
            conditioning_protocol
            and contract.get("relation_source_excluded_from_identity_set")
            is True
        ),
        "first_window_model_input_identity": (
            raw_first_window_identity
            and contract.get("paired_first_window_model_input_identity") is True
        ),
        "causal_window_overlap": (
            causal_overlap_exact
            and value.get("causal_window_overlap", {}).get("all_exact") is True
            and contract.get("causal_window_overlap_exact") is True
        ),
        "self_conditioning_appendix": appendix_exact,
        "reported_only_artifact": reported_only_exact,
        "reported_only_never_entered_the_gate": reported_only_inert,
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
            == internal_runner.internal_decision_evidence(recomputed_decision)
        ),
        "nine_gate_checks": (
            isinstance(decision, Mapping)
            and isinstance(decision.get("checks"), Mapping)
            and len(decision["checks"]) == 9
        ),
        "contract_passed": decision.get("contract_passed") is True,
        "internal_status": decision.get("internal_status") in {
            "unused",
            "source-provenance-negative",
            "high-t-negative",
            "object-following-negative",
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
        "source_variant_identity": contract.get("source_variant_identity")
        is True,
        "prev_x0_is_raw_x0_hat": contract.get("sampler_prev_x0_is_raw_x0_hat")
        is True,
        "shared_source_builder": contract.get("sampler_shared_source_builder")
        is True,
        "estimate_forward_absent_during_sampling": contract.get(
            "estimate_forward_absent_during_sampling"
        ) is True,
        "history_pin_exact_every_step": contract.get(
            "history_pin_exact_every_step"
        ) is True,
        "field_schedule_buffer_absent": contract.get(
            "field_schedule_buffer_absent"
        ) is True,
        "schedule_hash": (
            relation_protocol
            and contract.get("canonical_schedule_hash") is True
        ),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AG internal diagnostic contract mismatch: {failed}")
    return {
        "checks": checks,
        "contract_passed": bool(decision["contract_passed"]),
        "internal_status": str(decision["internal_status"]),
        "source_provenance_passed": bool(decision["source_provenance_passed"]),
        "high_t_provenance_passed": bool(decision["high_t_provenance_passed"]),
        "object_following_passed": bool(decision["object_following_passed"]),
        "temporal_routing_passed": bool(decision["temporal_routing_passed"]),
        "role_binding_passed": bool(decision["role_binding_passed"]),
        "mechanism_passed": bool(decision["mechanism_passed"]),
        "classification": decision.get("classification"),
        "path": str(internal_path),
        "sha256": internal_sha256,
        "run_id": run_id,
        "selection": value["selection"],
        "target_checkpoint": value.get("target_checkpoint"),
        "artifact_closure": {**closure, "artifacts": resolved_artifacts},
        "decision": decision,
        "reported_only_quantities_used": False,
    }


def _waived_performance_gate_accepted(
    lifecycle: Mapping[str, object],
    performance: Mapping[str, object],
) -> bool:
    """Accept exactly the one user-authorized ``failed-waived`` D2-AG lineage.

    A waived lineage fails the benchmark's throughput and ETA checks by
    construction, so ``all(checks)`` cannot be required.  Everything else stays
    mandatory: the waiver must be present and bound to the exact benchmark and
    formal run, the failure must be confined to throughput/ETA (plus the
    derived status/classification/authorization labels), every non-speed
    contract must still pass, and the waived status must be recorded
    explicitly and never as ``performance-gate-passed``.
    """
    waiver = performance.get("performance_waiver")
    if not isinstance(waiver, Mapping) or not waiver:
        return False
    benchmark_checks = performance.get("benchmark_checks")
    if not isinstance(benchmark_checks, Mapping) or not benchmark_checks:
        return False
    waived_checks = {"classification", "eta", "formal_authorized", "status", "throughput"}
    non_speed_pass = all(
        passed
        for name, passed in benchmark_checks.items()
        if name not in waived_checks
    )
    speed_only_failure = {
        name for name, passed in benchmark_checks.items() if not passed
    } <= waived_checks
    labels = (
        performance.get("status"),
        performance.get("classification"),
        performance.get("benchmark_status"),
        performance.get("benchmark_classification"),
        lifecycle.get("performance_gate_state"),
        lifecycle.get("performance_gate_classification"),
        waiver.get("status"),
        waiver.get("classification"),
    )
    return (
        non_speed_pass
        and speed_only_failure
        and all(waiver.get("checks", {}).values())
        and bool(waiver.get("checks"))
        and performance.get("status") == D2AG_PERFORMANCE_WAIVER_STATUS
        and performance.get("classification")
        == D2AG_PERFORMANCE_WAIVER_CLASSIFICATION
        and performance.get("original_gate_passed") is False
        and performance.get("formal_authorization")
        == "explicit-single-run-waiver"
        and performance.get("benchmark_status") == "failed"
        and performance.get("benchmark_classification")
        == D2AG_PERFORMANCE_FAILURE_CLASSIFICATION
        and performance.get("benchmark_failed_checks") == sorted(waived_checks)
        and performance.get("run_id") == D2AG_WAIVED_BENCHMARK_RUN_ID
        and performance.get("sha256") == D2AG_WAIVED_BENCHMARK_SHA256
        and performance.get("formal_run_id") == D2AG_WAIVED_FORMAL_RUN_ID
        and float(performance.get("throughput_windows_per_second", -1.0))
        == D2AG_WAIVED_THROUGHPUT
        and float(performance.get("full_budget_eta_hours", -1.0))
        == D2AG_WAIVED_ETA_HOURS
        and waiver.get("benchmark_run_id") == D2AG_WAIVED_BENCHMARK_RUN_ID
        and waiver.get("benchmark_sha256") == D2AG_WAIVED_BENCHMARK_SHA256
        and waiver.get("formal_run_id") == D2AG_WAIVED_FORMAL_RUN_ID
        # The waived state must be recorded on the lifecycle contract too.
        and lifecycle.get("performance_gate_state")
        == D2AG_PERFORMANCE_WAIVER_STATUS
        and lifecycle.get("performance_gate_classification")
        == D2AG_PERFORMANCE_WAIVER_CLASSIFICATION
        and lifecycle.get("performance_waiver_sha256") == waiver.get("sha256")
        and lifecycle.get("d2af_waiver_inherited") is False
        and D2AG_FORBIDDEN_WAIVED_STATUS not in labels
    )


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
    activated_gradient = gradient_audit.get("activated_relation_gradients", {})
    activated_groups = activated_gradient.get("relation_groups", {})
    initial_instance = relation.get("initial_model_instance_contract", {})
    final_instance = relation.get("final_model_instance_contract", {})
    lifecycle = metrics.get("d2ag_lifecycle_contract", {})
    performance = lifecycle.get("performance_gate", {})
    checks = {
        "status": metrics.get("status") == "stable",
        "run_id": metrics.get("run_id") == args.training_run_id,
        "seed": metrics.get("seed") == 42,
        "initialization": metrics.get("initialization") == "random",
        "training_start": metrics.get("training_start") == "random",
        "released_checkpoint_used": metrics.get("released_checkpoint_used")
        is False,
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
            and routing.get("d2af_sqrt_alpha_bar_reliability") is not True
            and routing.get("d2ag_selfcond_relation_source") is True
            and routing.get("global_bps_token_preserved") is True
        ),
        "model_config": metrics.get("model_config", {}).get(
            "architecture_variant"
        ) == HOI_ARCHITECTURE_D2AG,
        "selfcond_relation_source_contract": (
            relation.get("architecture_variant") == HOI_ARCHITECTURE_D2AG
            and relation_contract
            == selfcond_relation_source_contract_metadata()
            and relation.get("selfcond_relation_source") is True
            and relation.get("diffusion_reliability") is not True
            and relation.get("diagnostic_variant") == "full"
            and relation.get("gate_override") is None
            and relation.get("rho_override") is None
            and relation.get("capture_enabled") is False
            and math.isfinite(float(relation.get("alpha", float("nan"))))
            and math.isfinite(float(relation.get("gate", float("nan"))))
        ),
        # The registered self-conditioning semantics are carried by the sealed
        # contract inside ``sparse_relation_field``, not by a separate top-level
        # metrics block; asserting them here keeps a single source of truth.
        "selfcond_runtime_record": (
            relation.get("selection_probability")
            == D2AG_SELF_CONDITION_PROBABILITY
            and relation.get("mask_generator_seed_formula")
            == "cfg.seed * 1000003 + processed_windows + rank"
            and relation_contract.get("selection_probability")
            == D2AG_SELF_CONDITION_PROBABILITY
            and relation_contract.get("variable_anchors")
            == list(D2AG_VARIABLE_ANCHORS)
            and relation_contract.get("variable_anchor_source")
            == "detached_model_x0_hat"
            and relation_contract.get("history_anchor_source")
            == "current_noisy_state"
            and relation_contract.get("relation_exposure_fraction") == 1.0
            and relation_contract.get("relation_zero_branch") is False
            and relation_contract.get("estimate_forward_eval_mode") is True
            and relation_contract.get("estimate_forward_selected_subset_only")
            is True
            and relation_contract.get("x0_hat_detached") is True
            and relation_contract.get("learned_selection") is False
            and relation_contract.get("sqrt_alpha_bar_attenuation") is False
            and relation_contract.get("schedule_buffer_registered") is False
            and relation_contract.get(
                "prepare_clean_x0_applied_to_relation_source"
            ) is False
            and relation_contract.get("relation_source_so3_projected") is False
            and relation_contract.get("per_anchor_scaling") is False
            and relation.get("builder", {}).get("source")
            == "variable_anchors_detached_model_x0_hat_history_current_x_t"
            and relation.get("builder", {}).get("stored_relation") is False
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
            lifecycle.get("performance_gate_required") is True
            and isinstance(performance, Mapping)
            and bool(performance)
            # Either an unqualified pass, or exactly the one authorized waiver.
            and (
                (
                    all(performance.get("checks", {}).values())
                    and performance.get("performance_waiver") in (None, False)
                )
                or _waived_performance_gate_accepted(lifecycle, performance)
            )
            # The gate must be D2-AG's own floor, never a predecessor's.
            and float(
                performance.get("minimum_throughput_windows_per_second", -1.0)
            ) == D2AG_MINIMUM_THROUGHPUT
            and float(
                performance.get("maximum_full_budget_eta_hours", -1.0)
            ) == D2AG_MAXIMUM_ETA_HOURS
            and float(
                lifecycle.get("minimum_throughput_windows_per_second", -1.0)
            ) == D2AG_MINIMUM_THROUGHPUT
            and float(
                lifecycle.get("maximum_full_budget_eta_hours", -1.0)
            ) == D2AG_MAXIMUM_ETA_HOURS
        ),
        "eligibility_gate_absent": (
            lifecycle.get("eligibility_gate") is None
            and lifecycle.get("eligibility_gate_required") in (None, False)
        ),
        "predecessor_lifecycle_absent": (
            metrics.get("d2ae_lifecycle_contract") is None
            and metrics.get("d2af_lifecycle_contract") is None
        ),
        "no_ema": metrics.get("ema_decays") == [],
        "primary_online": metrics.get("primary_weight_variant") == "online",
        "target_checkpoint_basename": args.target_checkpoint.name
        == expected_name,
        "final_checkpoint_hash": len(checkpoint_rows) == 1,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AG training artifact contract mismatch: {failed}")
    return {"checks": checks, "metrics": metrics}


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    value = _shared_resolved_config(args)
    value["subphase"] = SUBPHASE
    value["training_run_id"] = args.training_run_id
    # Straight-through formal lineage: the target checkpoint's own semantic
    # record, as verified by the internal diagnostic, is the binding.
    value["formal_training"] = {
        "run_id": args.training_run_id,
        "final_checkpoint": str(args.target_checkpoint.resolve()),
        "final_checkpoint_sha256": args.target_sha256,
        "final_checkpoint_basename": (
            f"{args.training_run_id}_windows061440000.pth"
        ),
        "processed_windows": 61_440_000,
        "weight_variant": "online",
        "resumed": False,
        "checkpoint_source_commit": (
            _internal.get("target_checkpoint", {}) or {}
        ).get("git_commit"),
        "initial_model_state_sha256": EXPECTED_INITIAL_MODEL_STATE_SHA256,
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
        "variants": list(VARIANTS),
        "gate_ablation_variant_used": False,
        "reported_only_quantities_used": False,
        "artifact_closure": _internal.get("artifact_closure", {}),
    }
    value["self_conditioning"] = {
        "architecture_variant": HOI_ARCHITECTURE_D2AG,
        "formula": "motion + tanh(alpha) * routed_relation(relation_source)",
        "contract": selfcond_relation_source_contract_metadata(),
        "variable_anchors": list(D2AG_VARIABLE_ANCHORS),
        "history_anchor_source": "current_noisy_state",
        "train_selection_probability": D2AG_SELF_CONDITION_PROBABILITY,
        "relation_exposure_fraction": 1.0,
        "relation_zero_branch": False,
        "sqrt_alpha_bar_attenuation": False,
        "field_schedule_buffer_registered": False,
        "first_reverse_step_source": "current_noisy_state",
        "later_reverse_step_source": "previous_step_raw_x0_hat",
        "prepare_clean_x0_applied_to_relation_source": False,
        "relation_source_so3_projected": False,
        "train_sample_symmetric": True,
        "rest_object_points": [100, 3],
        "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
        "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
        "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
        "clean_target": False,
        "scene": False,
        "future_gt": False,
        "stored_relation": False,
        "global_bps_token_preserved": True,
        "registered_source_is_x_t_counts": dict(SOURCE_IS_CURRENT_STEP_COUNTS),
    }
    # D2-AG registers no predecessor-specific repair comparison: the sealed D2-X
    # control and the released baseline are the only native references.
    value["predecessor_repair_comparison"] = {
        "registered": False,
        "reason": (
            "D2-AG is not registered as a repair of D2-AE or D2-AF; the native "
            "gate is the standard D2-X transfer, protection and release set"
        ),
    }
    value["evaluation"]["protection_metrics"] = list(PROTECTION_METRICS)
    value["evaluation"]["native_gates"] = {
        "d2x_transfer": {
            "contact_f1_ci_lower_gt_zero": True,
            "contact_recall_ci_lower_gt_zero": True,
            "contact_f1_gap_closure_min": 0.25,
            "contact_f1_point_estimate_min": CONTACT_F1_POINT_MINIMUM,
        },
        "protection_ratio_ci_upper_max": 1.10,
        "contact_precision_ci_lower_min": -0.02,
        "released_effectiveness_point_gate": "baseline 95 percent",
        "predecessor_repair_gate": False,
        "all_conditions_conjunctive": True,
        "internal_mechanism_reported_beside_native_headline": True,
        "reported_only_quantities_enter_gate": False,
    }
    return value


def additional_runtime_artifact_hashes(args):
    """Deliberately empty: nothing is hashed twice in one process.

    ``_validate_internal`` already hashes the internal diagnostic against
    ``--internal-diagnostic-sha256`` and every closed internal artifact against
    its recorded closure entry, earlier in this same process.  Repeating those
    reads here would add no evidence, so this override pins the shared D2-X hook
    to no second pass rather than inheriting a predecessor's.
    """
    del args
    return {}, {}


_internal: Dict[str, object] = {}


def compare_records(
    control: Mapping[str, Mapping[str, object]],
    target: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Standard D2-X comparison; D2-AG adds no predecessor repair block."""
    comparison = d2ac.compare_records(control, target)
    comparison["predecessor_repair_comparison_registered"] = False
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
    value["predecessor_repair_gate_registered"] = False
    value["official_test_used"] = True
    return value


def configure_shared(
    args: argparse.Namespace,
    internal: Mapping[str, object],
) -> None:
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
    run_id_match = RUN_ID_RE.fullmatch(args.run_id)
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if run_id_match is None or run_id_match.group("date") != actual_date:
        raise ValueError(
            "D2-AG native run id must use the locked stem and actual date"
        )
    if not TRAINING_RUN_ID_RE.fullmatch(args.training_run_id):
        raise ValueError("invalid D2-AG formal training run id")
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if (
        not configured_python
        or not Path(configured_python).is_absolute()
        or Path(sys.executable).resolve() != Path(configured_python).resolve()
        or args.python.resolve() != Path(configured_python).resolve()
    ):
        raise ValueError(
            "D2-AG native evaluation requires the absolute INFBAGEL_PYTHON"
        )
    for name in (
        "target_sha256",
        "training_metrics_sha256",
        "internal_diagnostic_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    internal = _validate_internal(args)
    configure_shared(args, internal)
    if args.resolve_only:
        shared.prepare_resolved_config(args)
        return
    shared.main()


if __name__ == "__main__":
    main()
