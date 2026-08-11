"""Frozen gradient-clipping routing helpers for the Phase 1B D2-J0 audit."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

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


TIMESTEPS: Tuple[int, ...] = (0, 1, 10, 50, 100, 250, 499)
GATE_TIMESTEPS: Tuple[int, ...] = (250, 499)
FIELD_COMPONENTS: Tuple[str, ...] = (
    "joint_position",
    "joint_rotation",
    "object_translation",
    "object_rotation",
    "contact",
)
AUXILIARY_COMPONENTS: Tuple[str, ...] = (
    "weighted_fk",
    "weighted_object_surface",
    "weighted_velocity",
    "terminal_goal",
)
BASE_COMPONENTS: Tuple[str, ...] = FIELD_COMPONENTS + AUXILIARY_COMPONENTS
LOSS_COMPONENTS: Tuple[str, ...] = (
    *FIELD_COMPONENTS,
    "human_reconstruction",
    "object_reconstruction",
    "reconstruction",
    *AUXILIARY_COMPONENTS,
    "auxiliary_sum",
    "total",
)
PRIMARY_WINDOWS = 128
BLOCK_SIZE = 16
GRADIENT_CLIP_NORM = 1.0
EXPECTED_PRIMARY_SHA256 = "a75012dda01cfd59c413bb622f4d867ffb6c2c48cf5d9dcfba4fe800e172432a"


def select_fresh_primary(dataset) -> Dict[str, object]:
    """Select locked D0-ordering ranks 640--767 and prove prior disjointness."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-J selection is internal-validation only")
    ranked = []
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        name = str(dataset.scene_names[sequence])
        pi = int(dataset.language["pi"][global_index])
        ranked.append((
            stable_digest(f"42:hoi-remediation-window:{name}:{pi}"),
            name, pi, position, int(global_index),
        ))
    ranked.sort()
    if len(ranked) < 768:
        raise ValueError("D2-J requires at least 768 ranked internal windows")
    d2h_global = {row[-1] for row in ranked[:512]}
    d2i_global = {row[-1] for row in ranked[512:640]}
    rows = ranked[640:768]
    positions = [row[-2] for row in rows]
    global_indices = [row[-1] for row in rows]
    terminal_windows = sum(
        int(
            int(dataset.ends[index])
            == int(dataset.seq_ends[int(dataset.sequence_ids[index])]) - 1
        )
        for index in global_indices
    )
    if set(global_indices) & (d2h_global | d2i_global):
        raise AssertionError("D2-J selection overlaps D2-H0 or D2-I0")
    return {
        "positions": positions,
        "global_indices": global_indices,
        "sha256": selection_sha256(global_indices),
        "terminal_windows": terminal_windows,
        "d2h_global_indices": d2h_global,
        "d2i_global_indices": d2i_global,
    }


def clipping_replay(preclip_norm: float, max_norm: float = GRADIENT_CLIP_NORM) -> Dict[str, float]:
    """Replay the production PyTorch global-clipping scalar formula."""
    if not math.isfinite(preclip_norm) or preclip_norm < 0:
        raise ValueError("preclip norm must be finite and nonnegative")
    synthetic = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    synthetic.grad = torch.tensor([preclip_norm], dtype=torch.float64)
    returned = float(torch.nn.utils.clip_grad_norm_([synthetic], max_norm))
    synthetic_postclip = abs(float(synthetic.grad.item()))
    formula_coefficient = min(1.0, max_norm / (preclip_norm + 1e-6))
    formula_postclip = preclip_norm * formula_coefficient
    synthetic_coefficient = (
        synthetic_postclip / preclip_norm if preclip_norm else 1.0
    )
    replay_error = max(
        abs(returned - preclip_norm),
        abs(synthetic_coefficient - formula_coefficient),
        abs(synthetic_postclip - formula_postclip),
    )
    return {
        "max_norm": float(max_norm),
        "preclip_norm": float(preclip_norm),
        "clip_coefficient": float(formula_coefficient),
        "postclip_norm": float(formula_postclip),
        "synthetic_returned_preclip_norm": returned,
        "synthetic_clip_coefficient": float(synthetic_coefficient),
        "synthetic_postclip_norm": synthetic_postclip,
        "formula_replay_max_abs": float(replay_error),
    }


def clip_gradient_geometry(
    base_gradients: Mapping[str, Sequence[torch.Tensor | None]],
    direct_total: Sequence[torch.Tensor | None],
    groups: Mapping[str, Sequence[int]],
) -> Dict[str, object]:
    """Report field-complete geometry plus production clip statistics."""
    if tuple(base_gradients) != BASE_COMPONENTS:
        raise ValueError(f"expected {BASE_COMPONENTS}, got {tuple(base_gradients)}")
    gradients = dict(base_gradients)
    gradients["human_reconstruction"] = _sum_gradients(
        gradients, ("joint_position", "joint_rotation"),
    )
    gradients["object_reconstruction"] = _sum_gradients(
        gradients, ("object_translation", "object_rotation"),
    )
    gradients["reconstruction"] = _sum_gradients(
        gradients, ("human_reconstruction", "object_reconstruction", "contact"),
    )
    gradients["auxiliary_sum"] = _sum_gradients(gradients, AUXILIARY_COMPONENTS)
    replay_total = _sum_gradients(gradients, ("reconstruction", "auxiliary_sum"))
    gradients["total"] = tuple(direct_total)
    replay_error = _relative_l2(replay_total, direct_total)
    records = {}
    for group in PARAMETER_GROUPS:
        indices = groups[group]
        norms = {name: _norm(gradients[name], indices) for name in LOSS_COMPONENTS}
        cosine = {
            first: {
                second: _cosine(gradients[first], gradients[second], indices)
                for second in LOSS_COMPONENTS
            }
            for first in LOSS_COMPONENTS
        }
        records[group] = {
            "gradient_l2_norm": norms,
            "cosine_matrix": cosine,
            "human_directional_efficiency": cosine["total"]["human_reconstruction"],
            "object_directional_efficiency": cosine["total"]["object_reconstruction"],
        }
    clipping = clipping_replay(records["all_parameters"]["gradient_l2_norm"]["total"])
    finite = bool(
        math.isfinite(replay_error)
        and all(math.isfinite(value) for value in clipping.values())
        and all(
            math.isfinite(value)
            for record in records.values()
            for value in record["gradient_l2_norm"].values()
        )
        and all(
            math.isfinite(value["value"])
            for record in records.values()
            for row in record["cosine_matrix"].values()
            for value in row.values()
        )
    )
    return {
        "groups": records,
        "clipping": clipping,
        "total_gradient_formula_relative_l2": float(replay_error),
        "finite": finite,
    }


def _bootstrap(values: Sequence[float]) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("D2-J bootstrap requires finite one-dimensional values")
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


def mechanism_gate(candidates: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    checkpoint_results = {}
    for checkpoint in CHECKPOINTS:
        candidate = candidates.get(checkpoint)
        if candidate is None:
            checkpoint_results[checkpoint] = {"passed": False, "missing": True}
            continue
        timestep_results = {}
        for timestep in GATE_TIMESTEPS:
            blocks = candidate["timesteps"][str(timestep)]["blocks"]
            preclip = _bootstrap([block["clipping"]["preclip_norm"] for block in blocks])
            coefficient = _bootstrap([block["clipping"]["clip_coefficient"] for block in blocks])
            human = _bootstrap([
                block["groups"]["all_parameters"]["human_directional_efficiency"]["value"]
                for block in blocks
            ])
            objects = _bootstrap([
                block["groups"]["all_parameters"]["object_directional_efficiency"]["value"]
                for block in blocks
            ])
            checks = {
                "finite": bool(all(block["finite"] for block in blocks)),
                "formula_replay": bool(max(
                    block["total_gradient_formula_relative_l2"] for block in blocks
                ) <= 1e-5),
                "clipping_replay": bool(max(
                    block["clipping"]["formula_replay_max_abs"] for block in blocks
                ) <= 1e-6),
                "preclip_norm_bootstrap_lower": preclip["bootstrap_95_ci"][0] >= 50.0,
                "clip_coefficient_bootstrap_upper": coefficient["bootstrap_95_ci"][1] <= 0.02,
                "human_efficiency_bootstrap_upper": human["bootstrap_95_ci"][1] <= 0.15,
                "object_efficiency_bootstrap_lower": objects["bootstrap_95_ci"][0] >= 0.15,
            }
            timestep_results[str(timestep)] = {
                "passed": all(checks.values()),
                "checks": checks,
                "preclip_norm": preclip,
                "clip_coefficient": coefficient,
                "human_directional_efficiency": human,
                "object_directional_efficiency": objects,
            }
        state_unchanged = candidate.get("state_dict_sha256_before") == candidate.get("state_dict_sha256_after")
        checks = {
            "all_finite": bool(candidate.get("finite", False)),
            "state_dict_unchanged": state_unchanged,
            "parameter_grad_buffers_clear": bool(candidate.get("parameter_grad_buffers_clear", False)),
            "all_gate_timesteps": all(record["passed"] for record in timestep_results.values()),
        }
        checkpoint_results[checkpoint] = {
            "passed": all(checks.values()),
            "checks": checks,
            "timesteps": timestep_results,
        }
    passed = all(checkpoint_results.get(name, {}).get("passed", False) for name in CHECKPOINTS)
    return {
        "passed": passed,
        "classification": (
            "gradient-clip-routing-positive-stop" if passed
            else "gradient-clip-routing-negative-stop"
        ),
        "checkpoint_results": checkpoint_results,
        "training_authorized": False,
        "training_started": False,
    }
