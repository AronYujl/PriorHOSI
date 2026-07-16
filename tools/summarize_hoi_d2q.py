#!/usr/bin/env python3
"""Create the tracked compact aggregate for a completed Phase 1B D2-Q0 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


RUN_ID = "p1-hoi-d2q-author-contact-guidance-s42-20260716"
STRIP_KEYS = {
    "per_frame",
    "per_sequence",
    "per_sequence_window",
    "per_step",
    "per_window",
    "noise_streams",
    "by_object_category",
    "bins",
    "lengths",
    "per_unit",
    "global_indices",
    "sequence_names",
    "guidance_steps",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_identity(
    metrics: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    if metrics.get("run_id") != RUN_ID or manifest.get("experiment_id") != RUN_ID:
        raise ValueError("D2-Q summary run-id mismatch")
    if metrics.get("git_commit") != manifest.get("git", {}).get("commit"):
        raise ValueError("D2-Q summary manifest/workload commit mismatch")
    if metrics.get("status") != "completed":
        raise ValueError("D2-Q summary requires completed metrics")


def compact(value):
    if isinstance(value, dict):
        return {
            key: compact(item)
            for key, item in value.items()
            if key not in STRIP_KEYS
        }
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = load_json(args.metrics.resolve())
    manifest = load_json(args.manifest.resolve())
    validate_identity(metrics, manifest)
    result = compact(metrics)
    result["artifact"] = {
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "metrics_sha256": sha256_file(args.metrics.resolve()),
        "resolved_config_sha256": sha256_file(args.resolved_config.resolve()),
        "preflight_sha256": sha256_file(args.preflight.resolve()),
        "registry_sha256": sha256_file(args.registry.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
