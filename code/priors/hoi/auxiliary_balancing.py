"""Fixed auxiliary-weight counterfactual routing for Phase 1B D2-L0."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .adamw_routing import DIRECTIONS, adamw_directions
from .gradient_clipping import (
    BLOCK_SIZE,
    FIELD_COMPONENTS,
    GATE_TIMESTEPS,
    LOSS_COMPONENTS,
    PRIMARY_WINDOWS,
    TIMESTEPS,
)
from .gradient_routing import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINTS,
    PARAMETER_GROUPS,
    _cosine,
    _norm,
    _relative_l2,
    _sum_gradients,
)
from .remediation import selection_sha256, stable_digest


CANDIDATES: Tuple[str, ...] = ("current", "balanced")
RAW_COMPONENTS: Tuple[str, ...] = (
    *FIELD_COMPONENTS,
    "fk",
    "object_surface",
    "velocity",
    "terminal_goal",
)
CURRENT_WEIGHTS = {
    "fk": 50.0,
    "object_surface": 50.0,
    "velocity": 0.1,
    "terminal_goal": 1.0,
}
BALANCED_WEIGHTS = {
    "fk": 0.3569973401779424,
    "object_surface": 0.4772322188400037,
    "velocity": 0.1,
    "terminal_goal": 1.0,
}
WEIGHTS = {"current": CURRENT_WEIGHTS, "balanced": BALANCED_WEIGHTS}
EXPECTED_PRIMARY_SHA256 = "b5faa79316c6bd7aa9df0687a2554d458a459bd331c94648a99380d5c3b43a75"
WEIGHT_SOURCE_RUN = "p1-hoi-d2i-gradient-dominance-s42-20260715"
WEIGHT_SOURCE_METRICS_SHA256 = "910998c54487cb127343e783773d3dbf13d24b359caf0442695f066bc271bf56"
DERIVATION_TARGET_NORM = 0.6279429736100133
DERIVATION_RAW_FK_NORM = 1.7589570087469566
DERIVATION_RAW_OBJECT_SURFACE_NORM = 1.3158017183675033


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometric_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("D2-L weight derivation requires finite one-dimensional values")
    if bool((array <= 0).any()):
        raise ValueError("D2-L weight derivation requires strictly positive norms")
    return float(math.exp(np.log(array).mean()))


def derive_locked_weights(metrics: Mapping[str, object]) -> Dict[str, object]:
    """Reproduce the preregistered D2-I-only weight derivation."""
    if metrics.get("run_id") != WEIGHT_SOURCE_RUN:
        raise ValueError("D2-L weight source run id mismatch")
    records = []
    for checkpoint in CHECKPOINTS:
        candidate = metrics["candidates"][checkpoint]
        cohort = candidate["cohorts"]["primary"]
        for timestep in GATE_TIMESTEPS:
            blocks = cohort["timesteps"][str(timestep)]["blocks"]
            if len(blocks) != 8:
                raise ValueError("D2-L weight source requires exactly eight blocks per cell")
            records.extend(block["groups"]["all_parameters"]["gradient_l2_norm"] for block in blocks)
    if len(records) != 32:
        raise ValueError("D2-L weight source must contain exactly 32 high-noise records")
    targets = [
        math.sqrt(record["human_reconstruction"] * record["object_reconstruction"])
        for record in records
    ]
    raw_fk = [record["weighted_fk"] / 50.0 for record in records]
    raw_surface = [record["weighted_object_surface"] / 50.0 for record in records]
    target = _geometric_mean(targets)
    raw_fk_norm = _geometric_mean(raw_fk)
    raw_surface_norm = _geometric_mean(raw_surface)
    result = {
        "source_run": WEIGHT_SOURCE_RUN,
        "records": len(records),
        "target_norm": target,
        "raw_fk_norm_geomean": raw_fk_norm,
        "raw_object_surface_norm_geomean": raw_surface_norm,
        "balanced_weights": {
            "fk": target / raw_fk_norm,
            "object_surface": target / raw_surface_norm,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        },
    }
    expected = {
        "target_norm": DERIVATION_TARGET_NORM,
        "raw_fk_norm_geomean": DERIVATION_RAW_FK_NORM,
        "raw_object_surface_norm_geomean": DERIVATION_RAW_OBJECT_SURFACE_NORM,
    }
    for key, value in expected.items():
        if not math.isclose(result[key], value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"D2-L locked derivation {key} mismatch")
    for key, value in BALANCED_WEIGHTS.items():
        if not math.isclose(
            result["balanced_weights"][key], value, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"D2-L locked {key} weight mismatch")
    return result


def verify_weight_source(path: Path) -> Dict[str, object]:
    path = path.resolve()
    actual = _sha256_file(path)
    if actual != WEIGHT_SOURCE_METRICS_SHA256:
        raise ValueError(f"D2-L weight-source metrics hash mismatch: {actual}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "sha256": actual,
        **derive_locked_weights(metrics),
    }


def select_fresh_primary(dataset) -> Dict[str, object]:
    """Select locked D0 ranks 898--1025 and prove earlier-audit disjointness."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-L selection is internal-validation only")
    ranked = []
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        name = str(dataset.scene_names[sequence])
        pi = int(dataset.language["pi"][global_index])
        terminal = int(dataset.ends[global_index]) == int(dataset.seq_ends[sequence]) - 1
        ranked.append((
            stable_digest(f"42:hoi-remediation-window:{name}:{pi}"),
            name, pi, position, int(global_index), bool(terminal),
        ))
    ranked.sort()
    if len(ranked) < 1026:
        raise ValueError("D2-L requires at least 1026 ranked internal windows")
    rows = ranked[898:1026]
    if any(row[-1] for row in rows):
        raise ValueError("D2-L locked rank interval unexpectedly contains a terminal window")
    positions = [row[-3] for row in rows]
    global_indices = [row[-2] for row in rows]
    prior = {row[-2] for row in ranked[:898]}
    if set(global_indices) & prior:
        raise AssertionError("D2-L selection overlaps D2-H/I/J/K")
    return {
        "positions": positions,
        "global_indices": global_indices,
        "sha256": selection_sha256(global_indices),
        "terminal_windows": 0,
        "selected_ranks": list(range(898, 1026)),
        "prior_global_indices": prior,
    }


def _scale_gradients(
    gradients: Sequence[torch.Tensor | None], weight: float,
) -> Tuple[torch.Tensor | None, ...]:
    return tuple(None if value is None else value * float(weight) for value in gradients)


def candidate_gradient_components(
    raw_gradients: Mapping[str, Sequence[torch.Tensor | None]],
    weights: Mapping[str, float],
) -> Dict[str, Tuple[torch.Tensor | None, ...]]:
    """Build all field/aggregate gradients for one fixed weight candidate."""
    if tuple(raw_gradients) != RAW_COMPONENTS:
        raise ValueError(f"expected ordered raw components {RAW_COMPONENTS}")
    gradients = {
        name: tuple(raw_gradients[name]) for name in FIELD_COMPONENTS
    }
    gradients["human_reconstruction"] = _sum_gradients(
        gradients, ("joint_position", "joint_rotation"),
    )
    gradients["object_reconstruction"] = _sum_gradients(
        gradients, ("object_translation", "object_rotation"),
    )
    gradients["reconstruction"] = _sum_gradients(
        gradients, ("human_reconstruction", "object_reconstruction", "contact"),
    )
    gradients["weighted_fk"] = _scale_gradients(raw_gradients["fk"], weights["fk"])
    gradients["weighted_object_surface"] = _scale_gradients(
        raw_gradients["object_surface"], weights["object_surface"],
    )
    gradients["weighted_velocity"] = _scale_gradients(
        raw_gradients["velocity"], weights["velocity"],
    )
    gradients["terminal_goal"] = _scale_gradients(
        raw_gradients["terminal_goal"], weights["terminal_goal"],
    )
    gradients["auxiliary_sum"] = _sum_gradients(
        gradients,
        ("weighted_fk", "weighted_object_surface", "weighted_velocity", "terminal_goal"),
    )
    gradients["total"] = _sum_gradients(gradients, ("reconstruction", "auxiliary_sum"))
    return gradients


def _candidate_geometry(
    gradients: Mapping[str, Sequence[torch.Tensor | None]],
    direct_total: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    states: Sequence[Mapping[str, object]],
    optimizer_group: Mapping[str, object],
    parameter_groups: Mapping[str, Sequence[int]],
) -> Dict[str, object]:
    replay = _relative_l2(gradients["total"], direct_total)
    update = adamw_directions(direct_total, parameters, states, optimizer_group)
    directions = update.pop("directions")
    groups = {}
    for group_name in PARAMETER_GROUPS:
        indices = parameter_groups[group_name]
        loss_norms = {name: _norm(gradients[name], indices) for name in LOSS_COMPONENTS}
        direction_norms = {name: _norm(directions[name], indices) for name in DIRECTIONS}
        direction_loss = {
            direction: {
                loss: _cosine(directions[direction], gradients[loss], indices)
                for loss in LOSS_COMPONENTS
            }
            for direction in DIRECTIONS
        }
        direction_cosine = {
            first: {
                second: _cosine(directions[first], directions[second], indices)
                for second in DIRECTIONS
            }
            for first in DIRECTIONS
        }
        groups[group_name] = {
            "loss_gradient_l2_norm": loss_norms,
            "direction_l2_norm": direction_norms,
            "direction_loss_cosine": direction_loss,
            "direction_cosine": direction_cosine,
        }
    finite = bool(
        math.isfinite(replay)
        and math.isfinite(update["adamw_decomposition_relative_l2"])
        and all(math.isfinite(value) for value in update["clipping"].values())
        and all(
            math.isfinite(value)
            for record in groups.values()
            for values in (record["loss_gradient_l2_norm"], record["direction_l2_norm"])
            for value in values.values()
        )
        and all(
            math.isfinite(value["value"])
            for record in groups.values()
            for matrix in (record["direction_loss_cosine"], record["direction_cosine"])
            for row in matrix.values()
            for value in row.values()
        )
    )
    return {
        "groups": groups,
        "total_gradient_formula_relative_l2": float(replay),
        **update,
        "finite": finite,
    }


def paired_routing_geometry(
    raw_gradients: Mapping[str, Sequence[torch.Tensor | None]],
    direct_totals: Mapping[str, Sequence[torch.Tensor | None]],
    parameters: Sequence[torch.nn.Parameter],
    states: Sequence[Mapping[str, object]],
    optimizer_group: Mapping[str, object],
    parameter_groups: Mapping[str, Sequence[int]],
) -> Dict[str, object]:
    """Report both fixed candidates and their paired routing differences."""
    if tuple(direct_totals) != CANDIDATES:
        raise ValueError(f"expected ordered candidates {CANDIDATES}")
    candidates = {}
    for candidate in CANDIDATES:
        gradients = candidate_gradient_components(raw_gradients, WEIGHTS[candidate])
        candidates[candidate] = _candidate_geometry(
            gradients, direct_totals[candidate], parameters, states,
            optimizer_group, parameter_groups,
        )
    paired = {}
    for group in PARAMETER_GROUPS:
        paired[group] = {
            direction: {
                loss: (
                    candidates["balanced"]["groups"][group]["direction_loss_cosine"][
                        direction
                    ][loss]["value"]
                    - candidates["current"]["groups"][group]["direction_loss_cosine"][
                        direction
                    ][loss]["value"]
                )
                for loss in LOSS_COMPONENTS
            }
            for direction in DIRECTIONS
        }
    finite = bool(
        all(candidates[name]["finite"] for name in CANDIDATES)
        and all(
            math.isfinite(value)
            for group in paired.values()
            for direction in group.values()
            for value in direction.values()
        )
    )
    return {"candidates": candidates, "paired_candidate_difference": paired, "finite": finite}


def _bootstrap(values: Sequence[float]) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("D2-L bootstrap requires finite one-dimensional values")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    sampled = array[indices].mean(axis=1)
    lower, upper = np.quantile(sampled, (0.025, 0.975))
    return {
        "estimate": float(array.mean()),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "blocks": array.tolist(),
    }


def mechanism_gate(checkpoints: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    results = {}
    for checkpoint in CHECKPOINTS:
        record = checkpoints.get(checkpoint)
        if record is None:
            results[checkpoint] = {"passed": False, "missing": True}
            continue
        timesteps = {}
        for timestep in GATE_TIMESTEPS:
            blocks = record["timesteps"][str(timestep)]["blocks"]
            current = [
                block["candidates"]["current"]["groups"]["all_parameters"]
                for block in blocks
            ]
            balanced = [
                block["candidates"]["balanced"]["groups"]["all_parameters"]
                for block in blocks
            ]

            def efficiency(candidate_records, direction, loss):
                return [
                    item["direction_loss_cosine"][direction][loss]["value"]
                    for item in candidate_records
                ]

            clipped_current_human = efficiency(
                current, "clipped_total", "human_reconstruction",
            )
            clipped_balanced_human = efficiency(
                balanced, "clipped_total", "human_reconstruction",
            )
            clipped_balanced_object = efficiency(
                balanced, "clipped_total", "object_reconstruction",
            )
            adamw_current_human = efficiency(current, "adamw_full", "human_reconstruction")
            adamw_balanced_human = efficiency(
                balanced, "adamw_full", "human_reconstruction",
            )
            adamw_balanced_object = efficiency(
                balanced, "adamw_full", "object_reconstruction",
            )
            metrics = {
                "balanced_minus_current_clipped_human": _bootstrap([
                    right - left
                    for left, right in zip(clipped_current_human, clipped_balanced_human)
                ]),
                "balanced_clipped_human": _bootstrap(clipped_balanced_human),
                "balanced_clipped_object": _bootstrap(clipped_balanced_object),
                "balanced_minus_current_adamw_human": _bootstrap([
                    right - left
                    for left, right in zip(adamw_current_human, adamw_balanced_human)
                ]),
                "balanced_adamw_human": _bootstrap(adamw_balanced_human),
                "balanced_adamw_object": _bootstrap(adamw_balanced_object),
            }
            checks = {
                "finite": bool(all(block["finite"] for block in blocks)),
                "current_gradient_formula": max(
                    block["candidates"]["current"]["total_gradient_formula_relative_l2"]
                    for block in blocks
                ) <= 1e-5,
                "balanced_gradient_formula": max(
                    block["candidates"]["balanced"]["total_gradient_formula_relative_l2"]
                    for block in blocks
                ) <= 1e-5,
                "current_clip_formula": max(
                    block["candidates"]["current"]["clipping"]["formula_replay_max_abs"]
                    for block in blocks
                ) <= 1e-5,
                "balanced_clip_formula": max(
                    block["candidates"]["balanced"]["clipping"]["formula_replay_max_abs"]
                    for block in blocks
                ) <= 1e-5,
                "current_adamw_formula": max(
                    block["candidates"]["current"]["adamw_decomposition_relative_l2"]
                    for block in blocks
                ) <= 1e-5,
                "balanced_adamw_formula": max(
                    block["candidates"]["balanced"]["adamw_decomposition_relative_l2"]
                    for block in blocks
                ) <= 1e-5,
                "clipped_human_delta_lower": metrics[
                    "balanced_minus_current_clipped_human"
                ]["bootstrap_95_ci"][0] >= 0.10,
                "balanced_clipped_human_lower": metrics[
                    "balanced_clipped_human"
                ]["bootstrap_95_ci"][0] >= 0.15,
                "balanced_clipped_object_lower": metrics[
                    "balanced_clipped_object"
                ]["bootstrap_95_ci"][0] >= 0.15,
                "adamw_human_delta_lower": metrics[
                    "balanced_minus_current_adamw_human"
                ]["bootstrap_95_ci"][0] >= 0.10,
                "balanced_adamw_human_lower": metrics[
                    "balanced_adamw_human"
                ]["bootstrap_95_ci"][0] >= 0.15,
                "balanced_adamw_object_lower": metrics[
                    "balanced_adamw_object"
                ]["bootstrap_95_ci"][0] >= 0.15,
            }
            timesteps[str(timestep)] = {
                "passed": all(checks.values()), "checks": checks, **metrics,
            }
        checks = {
            "all_finite": bool(record.get("finite", False)),
            "model_state_unchanged": (
                record.get("model_state_sha256_before")
                == record.get("model_state_sha256_after")
            ),
            "optimizer_state_unchanged": (
                record.get("optimizer_state_sha256_before")
                == record.get("optimizer_state_sha256_after")
            ),
            "mapped_optimizer_state_unchanged": (
                record.get("mapped_state_sha256_before")
                == record.get("mapped_state_sha256_after")
            ),
            "parameter_grad_buffers_clear": bool(
                record.get("parameter_grad_buffers_clear", False)
            ),
            "optimizer_contract_exact": bool(record.get("optimizer_contract_exact", False)),
            "weight_provenance_exact": bool(record.get("weight_provenance_exact", False)),
            "all_gate_timesteps": all(item["passed"] for item in timesteps.values()),
        }
        results[checkpoint] = {
            "passed": all(checks.values()), "checks": checks, "timesteps": timesteps,
        }
    passed = all(results.get(name, {}).get("passed", False) for name in CHECKPOINTS)
    return {
        "passed": passed,
        "classification": (
            "gradient-balanced-auxiliary-routing-positive-stop"
            if passed else "gradient-balanced-auxiliary-routing-negative-stop"
        ),
        "checkpoint_results": results,
        "training_authorized": False,
        "training_started": False,
    }
