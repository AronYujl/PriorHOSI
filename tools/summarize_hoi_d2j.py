#!/usr/bin/env python3
"""Create the tracked compact aggregate from a sealed D2-J0 artifact tree."""

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

from priors.gradient_clipping import LOSS_COMPONENTS, TIMESTEPS  # noqa: E402
from priors.gradient_routing import CHECKPOINTS, PARAMETER_GROUPS  # noqa: E402
from tools.diagnose_hoi_d2h import exclusive_json  # noqa: E402
from tools.experiment import sha256_file, sha256_path  # noqa: E402


RUN_ID = "p1-hoi-d2j-clip-routing-s42-20260716"


def validate_run_identity(metrics, manifest, resolved) -> None:
    identifiers = (metrics.get("run_id"), manifest.get("experiment_id"), resolved.get("run_id"))
    if identifiers != (RUN_ID, RUN_ID, RUN_ID):
        raise ValueError(f"D2-J0 run-id mismatch across sealed artifacts: {identifiers}")


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("D2-J0 compact summary received nonfinite values")
    return float(array.mean())


def _geomean_nonnegative(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or bool((array < 0).any()):
        raise ValueError("D2-J0 compact geomean received invalid values")
    if bool((array == 0).any()):
        return 0.0
    return float(math.exp(np.log(array).mean()))


def compact_blocks(blocks: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    groups = {}
    for group in PARAMETER_GROUPS:
        records = [block["groups"][group] for block in blocks]
        groups[group] = {
            "gradient_l2_norm_geometric_mean": {
                loss: _geomean_nonnegative([
                    record["gradient_l2_norm"][loss] for record in records
                ])
                for loss in LOSS_COMPONENTS
            },
            "cosine_matrix_mean": {
                first: {
                    second: {
                        "value": _mean([
                            record["cosine_matrix"][first][second]["value"]
                            for record in records
                            if record["cosine_matrix"][first][second]["defined"]
                        ]) if any(
                            record["cosine_matrix"][first][second]["defined"]
                            for record in records
                        ) else 0.0,
                        "defined_blocks": sum(
                            int(record["cosine_matrix"][first][second]["defined"])
                            for record in records
                        ),
                    }
                    for second in LOSS_COMPONENTS
                }
                for first in LOSS_COMPONENTS
            },
            "human_directional_efficiency_mean": _mean([
                record["human_directional_efficiency"]["value"] for record in records
            ]),
            "object_directional_efficiency_mean": _mean([
                record["object_directional_efficiency"]["value"] for record in records
            ]),
        }
    return {
        "blocks": len(blocks),
        "windows": sum(int(block["windows"]) for block in blocks),
        "q_noise_sha256": [block["q_noise_sha256"] for block in blocks],
        "finite": bool(all(block["finite"] for block in blocks)),
        "maximum_total_gradient_formula_relative_l2": max(
            float(block["total_gradient_formula_relative_l2"]) for block in blocks
        ),
        "maximum_clipping_formula_replay_abs": max(
            float(block["clipping"]["formula_replay_max_abs"]) for block in blocks
        ),
        "clipping": {
            key: _mean([block["clipping"][key] for block in blocks])
            for key in ("preclip_norm", "clip_coefficient", "postclip_norm")
        },
        "loss_values_mean": {
            loss: _mean([block["loss_values"][loss] for block in blocks])
            for loss in LOSS_COMPONENTS
        },
        "groups": groups,
    }


def compact_candidate(candidate: Mapping[str, object]) -> Dict[str, object]:
    return {
        "checkpoint": candidate["checkpoint"],
        "finite": candidate["finite"],
        "maximum_total_gradient_formula_relative_l2": candidate[
            "maximum_total_gradient_formula_relative_l2"
        ],
        "maximum_clipping_formula_replay_abs": candidate[
            "maximum_clipping_formula_replay_abs"
        ],
        "state_dict_sha256_before": candidate["state_dict_sha256_before"],
        "state_dict_sha256_after": candidate["state_dict_sha256_after"],
        "state_dict_unchanged": candidate["state_dict_unchanged"],
        "parameter_grad_buffers_clear": candidate["parameter_grad_buffers_clear"],
        "optimizer_created": candidate["optimizer_created"],
        "training_updates": candidate["training_updates"],
        "timesteps": {
            str(timestep): compact_blocks(candidate["timesteps"][str(timestep)]["blocks"])
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
        raise FileNotFoundError(f"D2-J0 artifact is missing: {missing}")
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    resolved = json.loads(paths["resolved_config"].read_text(encoding="utf-8"))
    validate_run_identity(metrics, manifest, resolved)
    if metrics["git_commit"] != manifest["git"]["commit"]:
        raise ValueError("D2-J0 metrics/manifest Git commit mismatch")
    tree = sha256_path(artifact_dir)
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": "1B-D2-J0",
        "status": manifest["status"],
        "git_commit": metrics["git_commit"],
        "selection": metrics["selection"],
        "selection_disjoint_from_d2h0": metrics["selection_disjoint_from_d2h0"],
        "selection_disjoint_from_d2i0": metrics["selection_disjoint_from_d2i0"],
        "assets": metrics["assets"],
        "candidates": {
            name: compact_candidate(metrics["candidates"][name]) for name in CHECKPOINTS
        },
        "paired_noise_identity": metrics["paired_noise_identity"],
        "decision": metrics["decision"],
        "runtime_seconds": metrics["runtime_seconds"],
        "gpu": metrics["gpu"],
        "training_updates": 0,
        "optimizer_created": False,
        "production_training_change": False,
        "loss_or_weight_change": False,
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
