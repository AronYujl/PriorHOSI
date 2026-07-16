#!/usr/bin/env python3
"""Create the tracked compact aggregate from a sealed D2-L0 artifact tree."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.adamw_routing import DIRECTIONS  # noqa: E402
from priors.auxiliary_balancing import CANDIDATES, RAW_COMPONENTS  # noqa: E402
from priors.gradient_clipping import LOSS_COMPONENTS, TIMESTEPS  # noqa: E402
from priors.gradient_routing import CHECKPOINTS, PARAMETER_GROUPS  # noqa: E402
from tools.diagnose_hoi_d2h import exclusive_json  # noqa: E402
from tools.experiment import sha256_file, sha256_path  # noqa: E402


RUN_ID = "p1-hoi-d2l-aux-balance-s42-20260716"


def validate_run_identity(metrics, manifest, resolved) -> None:
    identifiers = (
        metrics.get("run_id"), manifest.get("experiment_id"), resolved.get("run_id"),
    )
    if identifiers != (RUN_ID, RUN_ID, RUN_ID):
        raise ValueError(f"D2-L0 run-id mismatch across sealed artifacts: {identifiers}")


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("D2-L0 compact summary received nonfinite values")
    return float(array.mean())


def _geomean_nonnegative(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or bool((array < 0).any()):
        raise ValueError("D2-L0 compact geomean received invalid values")
    if bool((array == 0).any()):
        return 0.0
    return float(math.exp(np.log(array).mean()))


def _mean_cosine(records, matrix, first, second):
    defined = [
        record[matrix][first][second]["value"]
        for record in records if record[matrix][first][second]["defined"]
    ]
    return {"value": _mean(defined) if defined else 0.0, "defined_blocks": len(defined)}


def _compact_candidate(blocks, candidate):
    candidate_blocks = [block["candidates"][candidate] for block in blocks]
    groups = {}
    for group in PARAMETER_GROUPS:
        records = [item["groups"][group] for item in candidate_blocks]
        groups[group] = {
            "loss_gradient_l2_norm_geometric_mean": {
                loss: _geomean_nonnegative([
                    record["loss_gradient_l2_norm"][loss] for record in records
                ])
                for loss in LOSS_COMPONENTS
            },
            "direction_l2_norm_geometric_mean": {
                direction: _geomean_nonnegative([
                    record["direction_l2_norm"][direction] for record in records
                ])
                for direction in DIRECTIONS
            },
            "direction_loss_cosine_mean": {
                direction: {
                    loss: _mean_cosine(
                        records, "direction_loss_cosine", direction, loss,
                    )
                    for loss in LOSS_COMPONENTS
                }
                for direction in DIRECTIONS
            },
            "direction_cosine_mean": {
                first: {
                    second: _mean_cosine(records, "direction_cosine", first, second)
                    for second in DIRECTIONS
                }
                for first in DIRECTIONS
            },
        }
    return {
        "finite": bool(all(item["finite"] for item in candidate_blocks)),
        "maximum_total_gradient_formula_relative_l2": max(
            float(item["total_gradient_formula_relative_l2"])
            for item in candidate_blocks
        ),
        "maximum_clipping_formula_replay_abs": max(
            float(item["clipping"]["formula_replay_max_abs"])
            for item in candidate_blocks
        ),
        "maximum_adamw_decomposition_relative_l2": max(
            float(item["adamw_decomposition_relative_l2"])
            for item in candidate_blocks
        ),
        "clipping": {
            key: _mean([item["clipping"][key] for item in candidate_blocks])
            for key in ("preclip_norm", "clip_coefficient", "postclip_norm")
        },
        "stored_lr": _mean([item["stored_lr"] for item in candidate_blocks]),
        "loss_values_mean": {
            loss: _mean([
                block["candidate_loss_values"][candidate][loss] for block in blocks
            ])
            for loss in LOSS_COMPONENTS
        },
        "groups": groups,
    }


def compact_blocks(blocks: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    paired = {}
    for group in PARAMETER_GROUPS:
        paired[group] = {
            direction: {
                loss: _mean([
                    block["paired_candidate_difference"][group][direction][loss]
                    for block in blocks
                ])
                for loss in LOSS_COMPONENTS
            }
            for direction in DIRECTIONS
        }
    return {
        "blocks": len(blocks),
        "windows": sum(int(block["windows"]) for block in blocks),
        "q_noise_sha256": [block["q_noise_sha256"] for block in blocks],
        "finite": bool(all(block["finite"] for block in blocks)),
        "maximum_production_total_value_replay_abs": max(
            float(block["production_total_value_replay_abs"]) for block in blocks
        ),
        "raw_loss_values_mean": {
            name: _mean([block["raw_loss_values"][name] for block in blocks])
            for name in RAW_COMPONENTS
        },
        "candidates": {
            candidate: _compact_candidate(blocks, candidate) for candidate in CANDIDATES
        },
        "balanced_minus_current_direction_loss_cosine_mean": paired,
    }


def compact_checkpoint(record: Mapping[str, object]) -> Dict[str, object]:
    return {
        "checkpoint": record["checkpoint"],
        "finite": record["finite"],
        "optimizer_contract": record["optimizer_contract"],
        "optimizer_contract_exact": record["optimizer_contract_exact"],
        "weight_provenance": record["weight_provenance"],
        "weight_provenance_exact": record["weight_provenance_exact"],
        "maximum_production_total_value_replay_abs": record[
            "maximum_production_total_value_replay_abs"
        ],
        "maximum_formula_replay": record["maximum_formula_replay"],
        "model_state_sha256_before": record["model_state_sha256_before"],
        "model_state_sha256_after": record["model_state_sha256_after"],
        "optimizer_state_sha256_before": record["optimizer_state_sha256_before"],
        "optimizer_state_sha256_after": record["optimizer_state_sha256_after"],
        "mapped_state_sha256_before": record["mapped_state_sha256_before"],
        "mapped_state_sha256_after": record["mapped_state_sha256_after"],
        "parameter_grad_buffers_clear": record["parameter_grad_buffers_clear"],
        "optimizer_created": record["optimizer_created"],
        "training_updates": record["training_updates"],
        "timesteps": {
            str(timestep): compact_blocks(record["timesteps"][str(timestep)]["blocks"])
            for timestep in TIMESTEPS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--metrics", default="metrics.json")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--resolved-config", default="resolved_config.json")
    parser.add_argument("--preflight", default="preflight.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    paths = {
        "metrics": artifact_dir / args.metrics,
        "manifest": artifact_dir / args.manifest,
        "resolved_config": artifact_dir / args.resolved_config,
        "preflight": artifact_dir / args.preflight,
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"D2-L0 artifact is missing: {missing}")
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    resolved = json.loads(paths["resolved_config"].read_text(encoding="utf-8"))
    validate_run_identity(metrics, manifest, resolved)
    if metrics["git_commit"] != manifest["git"]["commit"]:
        raise ValueError("D2-L0 metrics/manifest Git commit mismatch")
    tree = sha256_path(artifact_dir)
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": "1B-D2-L0",
        "status": manifest["status"],
        "git_commit": metrics["git_commit"],
        "selection": metrics["selection"],
        "selection_disjoint_from_d2h0_d2i0_d2j0_d2k0": metrics[
            "selection_disjoint_from_d2h0_d2i0_d2j0_d2k0"
        ],
        "weight_provenance": metrics["weight_provenance"],
        "weights": metrics["weights"],
        "weight_sweep": metrics["weight_sweep"],
        "assets": metrics["assets"],
        "checkpoints": {
            name: compact_checkpoint(metrics["checkpoints"][name])
            for name in CHECKPOINTS
        },
        "paired_noise_identity": metrics["paired_noise_identity"],
        "decision": metrics["decision"],
        "runtime_seconds": metrics["runtime_seconds"],
        "gpu": metrics["gpu"],
        "training_updates": 0,
        "optimizer_created": False,
        "checkpoint_write": False,
        "production_loss_change": False,
        "model_change": False,
        "representation_change": False,
        "condition_change": False,
        "sampler_change": False,
        "released_checkpoint_used": False,
        "ema_used": False,
        "official_test_used": False,
        "chois_used": False,
        "artifact": {
            "authority_staging": str(artifact_dir),
            "tree_sha256": tree["sha256"],
            "tree_files": tree["files"],
            "tree_bytes": tree["bytes"],
            **{f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
        },
    }
    exclusive_json(Path(args.output).resolve(), output)


if __name__ == "__main__":
    main()
