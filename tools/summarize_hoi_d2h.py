#!/usr/bin/env python3
"""Create the tracked compact aggregate from a sealed D2-H0 artifact tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.exposure import CHECKPOINTS, CONDITION_VARIANTS, TARGET_TIMESTEPS, compact_metric  # noqa: E402
from tools.experiment import sha256_file, sha256_path  # noqa: E402


RUN_ID = "p1-hoi-d2h-exposure-paired-s42-20260715"


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def compact_displacement(value: Mapping[str, object]) -> Dict[str, object]:
    return {
        name: {
            key: metric for key, metric in record.items() if key != "per_sample_mse"
        }
        for name, record in value.items()
    }


def validate_run_identity(
    metrics: Mapping[str, object],
    manifest: Mapping[str, object],
    resolved: Mapping[str, object],
) -> None:
    """Validate the metric/config ids against the experiment manifest schema."""
    identifiers = (
        metrics.get("run_id"), manifest.get("experiment_id"), resolved.get("run_id"),
    )
    if identifiers != (RUN_ID, RUN_ID, RUN_ID):
        raise ValueError(f"D2-H0 run-id mismatch across sealed artifacts: {identifiers}")


def compact_candidate(candidate: Mapping[str, object]) -> Dict[str, object]:
    timesteps = {}
    for timestep in TARGET_TIMESTEPS:
        raw = candidate["timesteps"][str(timestep)]
        record = {
            "target_timestep": raw["target_timestep"],
            "parent_timestep": raw["parent_timestep"],
            "parent_q_noise_sha256": raw["parent_q_noise_sha256"],
            "posterior_noise_sha256": raw["posterior_noise_sha256"],
        }
        for variant in CONDITION_VARIANTS:
            raw_variant = raw[variant]
            record[variant] = {
                "field_comparison": {
                    name: compact_metric(metric)
                    for name, metric in raw_variant["field_comparison"].items()
                },
                "physical_comparison": {
                    name: compact_metric(metric)
                    for name, metric in raw_variant["physical_comparison"].items()
                },
                "state_displacement_model_parent_vs_oracle_parent": compact_displacement(
                    raw_variant["state_displacement_model_parent_vs_oracle_parent"]
                ),
            }
        timesteps[str(timestep)] = record
    return {
        "checkpoint": candidate["checkpoint"],
        "finite": candidate["finite"],
        "history_max_abs": candidate["history_max_abs"],
        "posterior_formula_replay_max_abs": candidate["posterior_formula_replay_max_abs"],
        "paired_parent_q_noise": candidate["paired_parent_q_noise"],
        "paired_posterior_noise": candidate["paired_posterior_noise"],
        "object_so3_projection": candidate["object_so3_projection"],
        "support_clamp": candidate["support_clamp"],
        "cfg": candidate["cfg"],
        "autograd_detached": candidate["autograd_detached"],
        "timesteps": timesteps,
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
        raise FileNotFoundError(f"D2-H0 artifact is missing: {missing}")
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    resolved = json.loads(paths["resolved_config"].read_text(encoding="utf-8"))
    validate_run_identity(metrics, manifest, resolved)
    if metrics["git_commit"] != manifest["git"]["commit"]:
        raise ValueError("D2-H0 metrics/manifest Git commit mismatch")
    tree = sha256_path(artifact_dir)
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": "1B-D2-H0",
        "status": manifest["status"],
        "git_commit": metrics["git_commit"],
        "selection": metrics["selection"],
        "assets": metrics["assets"],
        "contract_checks": metrics["contract_checks"],
        "candidates": {
            name: compact_candidate(metrics["candidates"][name]) for name in CHECKPOINTS
        },
        "decision": metrics["decision"],
        "implementation_parity": metrics["implementation_parity"],
        "runtime_seconds": metrics["runtime_seconds"],
        "gpu": metrics["gpu"],
        "checkpoint_selection": False,
        "training_updates": 0,
        "official_test_used": False,
        "chois_used": False,
        "sampler_stored_per_frame_bps": False,
        "sampler_future_gt": False,
        "d2h1_started": False,
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
