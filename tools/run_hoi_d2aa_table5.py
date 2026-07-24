#!/usr/bin/env python3
"""Complete Table-5-aligned reporting for fixed D2-V/X/Y/Z checkpoints.

This is a non-selection evaluator.  It regenerates every fixed final-online
candidate symmetrically, enables only the matched CHOIS NPZ export, runs the
pinned CHOIS feature evaluator, and captures a separate batch-1 timing result.
It never creates an optimizer, trains, resumes, or writes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import chois_evaluator  # noqa: E402


RUN_ID = "p1-hoi-d2aa-table5-completion-s42-20260724"
SUBPHASE = "1B-D2-AA0"
EXPECTED_GT_TREE_SHA256 = "d439a98ea32f5d67964bc98431fe25bdffc24b63e00b42601c5355445d01742c"
EXPECTED_CHOIS_COMMIT = "8ec585aa0200fd2a890ffb12897bcf69ae719463"
EXPECTED_TEXT_TO_MOTION_COMMIT = "72df96ec453edea2fbe9603b1d58a955eaf71636"
EXPECTED_FEATURE_CHECKPOINT_SHA256 = (
    "a125bc15ffd9772686737111c7501ecee0a2d8571d9aca348ec1195ddef78775"
)
NATIVE_ABSOLUTE_TOLERANCE = 1e-9
NATIVE_RELATIVE_TOLERANCE = 1e-9

CANDIDATES: Mapping[str, Mapping[str, str]] = {
    "d2v": {
        "training_run": "p1-hoi-d2v-balanced-long-budget-s42-20260722",
        "checkpoint": (
            "p1-hoi-d2v-balanced-long-budget-s42-20260722_"
            "windows061440000.pth"
        ),
        "checkpoint_sha256": (
            "e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4"
        ),
        "sealed_eval_run": "p1-hoi-d2v-native-eval-s42-20260722",
        "sealed_aggregate_sha256": (
            "21f6bb27fe8d38a5203c2e40dee02815470bc40b638ee3a616101faae5cf8f0e"
        ),
        "sealed_per_sequence_sha256": (
            "4d147ef0a76977146639bab4260c6f7c5c2f96d9b253fb52ad968269f649ce1a"
        ),
    },
    "d2x": {
        "training_run": "p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723",
        "checkpoint": (
            "p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723_"
            "windows061440000.pth"
        ),
        "checkpoint_sha256": (
            "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51"
        ),
        "sealed_eval_run": "p1-hoi-d2x-native-eval-r1-s42-20260723",
        "sealed_aggregate_sha256": (
            "3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b"
        ),
        "sealed_per_sequence_sha256": (
            "69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a"
        ),
    },
    "d2y": {
        "training_run": "p1-hoi-d2y-routed-foot-amplification-s42-20260723",
        "checkpoint": (
            "p1-hoi-d2y-routed-foot-amplification-s42-20260723_"
            "windows061440000.pth"
        ),
        "checkpoint_sha256": (
            "8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7"
        ),
        "sealed_eval_run": "p1-hoi-d2y-native-eval-s42-20260724",
        "sealed_aggregate_sha256": (
            "776e6c35acdaa190ffcbab047b170ed4ab559c23f454714c31ad980db4dd8c70"
        ),
        "sealed_per_sequence_sha256": (
            "ea2cde99372392c5f16446708e3acf3789a68be9f1b7cc95134fd45390b12c02"
        ),
    },
    "d2z": {
        "training_run": "p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724",
        "checkpoint": (
            "p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724_"
            "windows061440000.pth"
        ),
        "checkpoint_sha256": (
            "44c1ff8c8cf4abc2c7312923f64183e1a4a307166d187c9fcaff03abdcc162b6"
        ),
        "sealed_eval_run": "p1-hoi-d2z-native-eval-s42-20260724",
        "sealed_aggregate_sha256": (
            "fb58a5ab3bd5ad0336ce02ff9a15cd7d97af8446599b147c9e2c806208a56162"
        ),
        "sealed_per_sequence_sha256": (
            "9f0f0e65bd0eaa4fe3ec1f495f6e4a4489c88d842256dccc3a6b9b57a1e9113f"
        ),
    },
}

STATIC_FILES = (
    "code/test_infbagel_hoi.py",
    "code/eval_metrics.py",
    "code/config/config_eval_hoi_prior.yaml",
    "tools/chois_evaluator.py",
    "tools/run_chois_evaluator.py",
    "experiments/evaluators/chois_omomo.json",
    "experiments/evaluators/text_to_motion.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def exclusive_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True,
    ).strip()


def candidate_source_paths(name: str) -> Dict[str, Path]:
    spec = CANDIDATES[name]
    results = REPO / "results" / "experiments"
    return {
        "checkpoint": (
            results / spec["training_run"] / "checkpoints" / spec["checkpoint"]
        ),
        "sealed_aggregate": (
            results / spec["sealed_eval_run"] / "evaluation" / "aggregate_metrics.json"
        ),
        "sealed_per_sequence": (
            results / spec["sealed_eval_run"] / "evaluation" /
            "per_sequence_metrics.json"
        ),
    }


def candidate_output_paths(output: Path, name: str) -> Dict[str, Path]:
    root = output / "candidates" / name
    return {
        "root": root,
        "native_root": root / "native",
        "native_evaluation": root / "native" / "evaluation",
        "native_aggregate": root / "native" / "evaluation" / "aggregate_metrics.json",
        "native_per_sequence": (
            root / "native" / "evaluation" / "per_sequence_metrics.json"
        ),
        "native_hydra": root / "native" / "hydra",
        "native_resolved": root / "native" / "resolved.yaml",
        "native_log": root / "native" / "run.log",
        "native_exit_code": root / "native" / "exit_code.txt",
        "predictions": root / "native" / "chois" / "predictions",
        "ground_truth": root / "native" / "chois" / "ground_truth",
        "chois_root": root / "chois",
        "chois_resolved": root / "chois" / "resolved.json",
        "chois_metrics": root / "chois" / "metrics.json",
        "chois_log": root / "chois" / "run.log",
        "chois_exit_code": root / "chois" / "exit_code.txt",
        "timing_root": root / "timing_batch1",
        "timing_evaluation": root / "timing_batch1" / "evaluation",
        "timing_aggregate": (
            root / "timing_batch1" / "evaluation" / "aggregate_metrics.json"
        ),
        "timing_per_sequence": (
            root / "timing_batch1" / "evaluation" / "per_sequence_metrics.json"
        ),
        "timing_hydra": root / "timing_batch1" / "hydra",
        "timing_resolved": root / "timing_batch1" / "resolved.yaml",
        "timing_log": root / "timing_batch1" / "run.log",
        "timing_exit_code": root / "timing_batch1" / "exit_code.txt",
    }


def native_overrides(args: argparse.Namespace, name: str, timing: bool) -> Tuple[str, ...]:
    source = candidate_source_paths(name)
    paths = candidate_output_paths(args.output.resolve(), name)
    prefix = "timing" if timing else "native"
    overrides = [
        f"exp_name={RUN_ID}-{name}-{'timing-b1' if timing else 'export'}",
        f"ckpt_path={source['checkpoint'].resolve()}",
        f"checkpoint_sha256={CANDIDATES[name]['checkpoint_sha256']}",
        "checkpoint_weight_variant=online",
        f"device={args.device}",
        f"dataset.device={args.device}",
        f"sampler.pelvis.device={args.device}",
        f"hoi_output_dir={paths[f'{prefix}_evaluation']}",
        f"per_sequence_metrics_path={paths[f'{prefix}_per_sequence']}",
        f"hydra.run.dir={paths[f'{prefix}_hydra']}",
        "hoi_expected_sequences=438",
        f"hoi_sequence_limit={'1' if timing else 'null'}",
        "save_motion_params=false",
        f"save_chois_eval_npz={'false' if timing else 'true'}",
        "load_scene=false",
        "sample_type=diffusion",
    ]
    if not timing:
        overrides.extend([
            f"chois_eval_output_dir={paths['predictions']}",
            f"chois_eval_ground_truth_dir={paths['ground_truth']}",
        ])
    return tuple(overrides)


def native_command(args: argparse.Namespace, name: str, timing: bool) -> Tuple[str, ...]:
    return (
        str(args.python.resolve()),
        str(REPO / "code" / "test_infbagel_hoi.py"),
        "--config-name",
        "config_eval_hoi_prior",
        *native_overrides(args, name, timing),
    )


def resolve_native(args: argparse.Namespace, name: str, timing: bool) -> str:
    command = (
        str(args.python.resolve()),
        str(REPO / "code" / "test_infbagel_hoi.py"),
        "--config-name",
        "config_eval_hoi_prior",
        "--cfg",
        "job",
        "--resolve",
        *native_overrides(args, name, timing),
    )
    environment = dict(os.environ)
    environment["ROOT_DIR"] = str(REPO)
    completed = subprocess.run(
        command,
        cwd=REPO / "code",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    if "${" in completed.stdout:
        raise ValueError(f"{name} {'timing' if timing else 'native'} config unresolved")
    return completed.stdout


def chois_command(args: argparse.Namespace, name: str) -> Tuple[str, ...]:
    paths = candidate_output_paths(args.output.resolve(), name)
    return (
        str(args.python.resolve()),
        str(REPO / "tools" / "run_chois_evaluator.py"),
        "--chois-root", str(args.chois_root.resolve()),
        "--text-to-motion-root", str(args.text_to_motion_root.resolve()),
        "--predictions", str(paths["predictions"]),
        "--ground-truth", str(paths["ground_truth"]),
        "--data-root", str(args.data_root.resolve()),
        "--glove-root", str(args.glove_root.resolve()),
        "--checkpoints-dir", str(args.checkpoints_dir.resolve()),
        "--checkpoint", str(args.feature_checkpoint.resolve()),
        "--dataset-name", "omomo",
        "--device", "cuda",
        "--batch-size", "32",
        "--workers", "0",
        "--seed", "42",
        "--diversity-times", "300",
        "--require-matched-ids",
        "--bootstrap-replicates", "10000",
        "--fid-bootstrap-replicates", "200",
        "--bootstrap-seed", "42",
        "--output", str(paths["chois_metrics"]),
    )


def chois_resolved(args: argparse.Namespace, name: str) -> Dict[str, Any]:
    paths = candidate_output_paths(args.output.resolve(), name)
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "candidate": name,
        "command": list(chois_command(args, name)),
        "inputs": {
            "predictions": str(paths["predictions"]),
            "ground_truth": str(paths["ground_truth"]),
        },
        "evaluator": {
            "chois_root": str(args.chois_root.resolve()),
            "text_to_motion_root": str(args.text_to_motion_root.resolve()),
            "data_root": str(args.data_root.resolve()),
            "glove_root": str(args.glove_root.resolve()),
            "checkpoints_dir": str(args.checkpoints_dir.resolve()),
            "feature_checkpoint": str(args.feature_checkpoint.resolve()),
            "batch_size": 32,
            "drop_last": True,
            "seed": 42,
            "bootstrap_replicates": 10000,
            "fid_bootstrap_replicates": 200,
            "bootstrap_seed": 42,
        },
    }


def resolved_lifecycle(args: argparse.Namespace) -> Dict[str, Any]:
    static_hashes = {
        relative: sha256_file(REPO / relative) for relative in STATIC_FILES
    }
    candidates: Dict[str, Any] = {}
    for name in CANDIDATES:
        source = candidate_source_paths(name)
        output = candidate_output_paths(args.output.resolve(), name)
        candidates[name] = {
            "source": {
                key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for key, path in source.items()
            },
            "expected": dict(CANDIDATES[name]),
            "native_resolved": {
                "path": str(output["native_resolved"]),
                "sha256": sha256_file(output["native_resolved"]),
            },
            "timing_resolved": {
                "path": str(output["timing_resolved"]),
                "sha256": sha256_file(output["timing_resolved"]),
            },
            "chois_resolved": {
                "path": str(output["chois_resolved"]),
                "sha256": sha256_file(output["chois_resolved"]),
            },
        }
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "execution_host": "infbagel-4gpu/node01",
        "python": str(args.python.resolve()),
        "device": args.device,
        "output": str(args.output.resolve()),
        "preflight": {
            "path": str(args.preflight.resolve()),
            "sha256": sha256_file(args.preflight.resolve()),
        },
        "static_hashes": static_hashes,
        "candidates": candidates,
        "fixed": {
            "official_sequences": 438,
            "windows_per_sequence": 3,
            "diffusion_steps": 500,
            "weight_variant": "online",
            "sample_type": "diffusion",
            "save_chois_eval_npz": True,
            "expected_gt_tree_sha256": EXPECTED_GT_TREE_SHA256,
            "native_absolute_tolerance": NATIVE_ABSOLUTE_TOLERANCE,
            "native_relative_tolerance": NATIVE_RELATIVE_TOLERANCE,
            "checkpoint_selection": False,
            "training": False,
            "consistency": False,
        },
    }


def prepare_resolved(args: argparse.Namespace) -> None:
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    for name in CANDIDATES:
        paths = candidate_output_paths(args.output.resolve(), name)
        exclusive_text(paths["native_resolved"], resolve_native(args, name, False))
        exclusive_text(paths["timing_resolved"], resolve_native(args, name, True))
        exclusive_json(paths["chois_resolved"], chois_resolved(args, name))
    exclusive_json(args.resolved_config.resolve(), resolved_lifecycle(args))


def verify_preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    value = load_json(args.preflight.resolve())
    checks = {
        "run_id": value.get("run_id") == RUN_ID,
        "hostname": value.get("hostname") == "node01",
        "git_commit": value.get("git_commit") == git_output("rev-parse", "HEAD"),
        "git_clean": value.get("git_clean") is True,
        "device": value.get("device") == args.device,
        "cuda_available": value.get("cuda_available") is True,
        "gpu_count": int(value.get("gpu_count", 0)) == 4,
        "compute_processes": value.get("compute_processes") == [],
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AA preflight mismatch: {failed}")
    return {"checks": checks, "value": value}


def verify_candidate_sources(name: str) -> Dict[str, str]:
    source = candidate_source_paths(name)
    expected = {
        "checkpoint": CANDIDATES[name]["checkpoint_sha256"],
        "sealed_aggregate": CANDIDATES[name]["sealed_aggregate_sha256"],
        "sealed_per_sequence": CANDIDATES[name]["sealed_per_sequence_sha256"],
    }
    actual = {key: sha256_file(path) for key, path in source.items()}
    if actual != expected:
        raise ValueError(f"{name} source hash mismatch: {actual}")
    aggregate = load_json(source["sealed_aggregate"])
    per_sequence = load_json(source["sealed_per_sequence"])
    checks = {
        "sample_count": aggregate.get("sample_count") == 438,
        "windows_per_sample": aggregate.get("windows_per_sample") == 3,
        "checkpoint_sha256": (
            aggregate.get("checkpoint", {}).get("sha256")
            == CANDIDATES[name]["checkpoint_sha256"]
        ),
        "weight_variant": (
            aggregate.get("checkpoint", {}).get("weight_variant") == "online"
        ),
        "chois_export_disabled": (
            aggregate.get("chois_export", {}).get("enabled") is False
        ),
        "per_sequence_count": per_sequence.get("sequence_count") == 438,
        "per_sequence_metrics": len(per_sequence.get("metrics", {})) == 438,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{name} sealed native contract mismatch: {failed}")
    return actual


def _numeric_close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=NATIVE_RELATIVE_TOLERANCE,
        abs_tol=NATIVE_ABSOLUTE_TOLERANCE,
    )


def compare_nested(left: Any, right: Any, path: str = "") -> Dict[str, Any]:
    mismatches = []
    max_absolute_difference = 0.0
    compared_numbers = 0

    def visit(a: Any, b: Any, current: str) -> None:
        nonlocal max_absolute_difference, compared_numbers
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                mismatches.append({
                    "path": current,
                    "left_keys": sorted(a),
                    "right_keys": sorted(b),
                })
                return
            for key in sorted(a):
                visit(a[key], b[key], f"{current}/{key}")
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                mismatches.append({
                    "path": current, "left_length": len(a), "right_length": len(b),
                })
                return
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                visit(left_item, right_item, f"{current}/{index}")
            return
        if (
            isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
        ):
            compared_numbers += 1
            difference = abs(float(a) - float(b))
            max_absolute_difference = max(max_absolute_difference, difference)
            if not _numeric_close(float(a), float(b)):
                mismatches.append({
                    "path": current, "left": a, "right": b,
                    "absolute_difference": difference,
                })
            return
        if a != b:
            mismatches.append({"path": current, "left": a, "right": b})

    visit(left, right, path)
    return {
        "passed": not mismatches,
        "compared_numbers": compared_numbers,
        "max_absolute_difference": max_absolute_difference,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log: Path,
    exit_code: Path,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    exclusive_text(exit_code, f"{completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(
            f"command exited {completed.returncode}: {' '.join(command[:3])}"
        )


def validate_native_regeneration(name: str) -> Dict[str, Any]:
    source = candidate_source_paths(name)
    paths = candidate_output_paths(_RUNTIME_OUTPUT, name)
    regenerated_aggregate = load_json(paths["native_aggregate"])
    regenerated_per_sequence = load_json(paths["native_per_sequence"])
    sealed_aggregate = load_json(source["sealed_aggregate"])
    sealed_per_sequence = load_json(source["sealed_per_sequence"])
    contract = {
        "sample_count": regenerated_aggregate.get("sample_count") == 438,
        "windows_per_sample": regenerated_aggregate.get("windows_per_sample") == 3,
        "checkpoint_sha256": (
            regenerated_aggregate.get("checkpoint", {}).get("sha256")
            == CANDIDATES[name]["checkpoint_sha256"]
        ),
        "weight_variant": (
            regenerated_aggregate.get("checkpoint", {}).get("weight_variant")
            == "online"
        ),
        "chois_export_enabled": (
            regenerated_aggregate.get("chois_export", {}).get("enabled") is True
        ),
        "per_sequence_count": regenerated_per_sequence.get("sequence_count") == 438,
    }
    failed = sorted(key for key, passed in contract.items() if not passed)
    if failed:
        raise ValueError(f"{name} regenerated native contract mismatch: {failed}")
    aggregate_comparison = compare_nested(
        sealed_aggregate["metrics"], regenerated_aggregate["metrics"], "/metrics",
    )
    per_sequence_comparison = compare_nested(
        sealed_per_sequence["metrics"],
        regenerated_per_sequence["metrics"],
        "/per_sequence_metrics",
    )
    if not aggregate_comparison["passed"] or not per_sequence_comparison["passed"]:
        raise ValueError(
            f"{name} native regeneration differs: "
            f"aggregate={aggregate_comparison['mismatches'][:3]}, "
            f"per_sequence={per_sequence_comparison['mismatches'][:3]}"
        )
    export = chois_evaluator.validate_pair(
        paths["predictions"], paths["ground_truth"],
    )
    if export["sequence_count"] != 438:
        raise ValueError(f"{name} CHOIS export count is not 438")
    if export["ground_truth"]["sha256"] != EXPECTED_GT_TREE_SHA256:
        raise ValueError(
            f"{name} GT tree mismatch: {export['ground_truth']['sha256']}"
        )
    frames = {value["frames"] for value in export["sequences"].values()}
    dtypes = {value["dtype"] for value in export["sequences"].values()}
    if frames != {126} or dtypes != {"float32"}:
        raise ValueError(f"{name} CHOIS export frames/dtypes differ: {frames}/{dtypes}")
    return {
        "contract": contract,
        "aggregate_comparison": aggregate_comparison,
        "per_sequence_comparison": per_sequence_comparison,
        "export": export,
        "aggregate": {
            "path": str(paths["native_aggregate"]),
            "sha256": sha256_file(paths["native_aggregate"]),
        },
        "per_sequence": {
            "path": str(paths["native_per_sequence"]),
            "sha256": sha256_file(paths["native_per_sequence"]),
        },
        "log": {
            "path": str(paths["native_log"]),
            "sha256": sha256_file(paths["native_log"]),
        },
    }


def validate_chois(name: str) -> Dict[str, Any]:
    paths = candidate_output_paths(_RUNTIME_OUTPUT, name)
    value = load_json(paths["chois_metrics"])
    metrics = value.get("metrics", {})
    expected_metrics = {
        "FID", "MatchingScore", "R-Precision@1", "R-Precision@2",
        "R-Precision@3", "Diversity",
    }
    if set(metrics) != expected_metrics:
        raise ValueError(f"{name} CHOIS metric keys differ: {sorted(metrics)}")
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError(f"{name} CHOIS metrics contain nonfinite values")
    protocol = value.get("embedding_protocol", {})
    protocol_checks = {
        "batch_size": protocol.get("batch_size") == 32,
        "drop_last": protocol.get("drop_last") is True,
        "matched_ids_required": protocol.get("matched_ids_required") is True,
        "exports": (
            protocol.get("exported_prediction_count") == 438
            and protocol.get("exported_ground_truth_count") == 438
        ),
        "embedded": protocol.get("embedded_count") == 416,
        "dropped": protocol.get("dropped_prediction_count") == 22,
    }
    uncertainty = value.get("uncertainty", {})
    additive = uncertainty.get("additive_metrics", {})
    fid = uncertainty.get("FID", {})
    uncertainty_checks = {
        "additive_replicates": additive.get("replicates") == 10000,
        "additive_seed": additive.get("seed") == 42,
        "fid_replicates": fid.get("replicates") == 200,
        "fid_seed": fid.get("seed") == 42,
    }
    for metric in ("MatchingScore", "R-Precision@1", "R-Precision@2", "R-Precision@3"):
        interval = additive.get(metric, {}).get("bootstrap_95_ci", [])
        uncertainty_checks[f"{metric}_finite_ci"] = (
            len(interval) == 2
            and all(math.isfinite(float(item)) for item in interval)
            and float(interval[0]) <= float(interval[1])
        )
    fid_interval = fid.get("bootstrap_95_ci", [])
    uncertainty_checks["FID_finite_ci"] = (
        len(fid_interval) == 2
        and all(math.isfinite(float(item)) for item in fid_interval)
        and float(fid_interval[0]) <= float(fid_interval[1])
    )
    failed = sorted(
        key for key, passed in {**protocol_checks, **uncertainty_checks}.items()
        if not passed
    )
    if failed:
        raise ValueError(f"{name} CHOIS reporting contract mismatch: {failed}")
    return {
        "metrics": metrics,
        "embedding_protocol": protocol,
        "uncertainty": uncertainty,
        "protocol_checks": protocol_checks,
        "uncertainty_checks": uncertainty_checks,
        "artifact": {
            "path": str(paths["chois_metrics"]),
            "sha256": sha256_file(paths["chois_metrics"]),
        },
        "log": {
            "path": str(paths["chois_log"]),
            "sha256": sha256_file(paths["chois_log"]),
        },
    }


def validate_timing(name: str) -> Dict[str, Any]:
    paths = candidate_output_paths(_RUNTIME_OUTPUT, name)
    aggregate = load_json(paths["timing_aggregate"])
    per_sequence = load_json(paths["timing_per_sequence"])
    generation = aggregate.get("generation_metrics", {})
    checks = {
        "sample_count": aggregate.get("sample_count") == 1,
        "dataset_sequence_count": aggregate.get("dataset_sequence_count") == 438,
        "is_timing_subset": aggregate.get("is_timing_subset") is True,
        "windows_per_sample": aggregate.get("windows_per_sample") == 3,
        "checkpoint_sha256": (
            aggregate.get("checkpoint", {}).get("sha256")
            == CANDIDATES[name]["checkpoint_sha256"]
        ),
        "weight_variant": (
            aggregate.get("checkpoint", {}).get("weight_variant") == "online"
        ),
        "chois_disabled": aggregate.get("chois_export", {}).get("enabled") is False,
        "warmup": generation.get("warmup_batches_excluded") == 1,
        "generated_frames": generation.get("generated_frames") == 126,
        "timing_synchronized": generation.get("timing_cuda_synchronized") is True,
        "per_sequence_count": per_sequence.get("sequence_count") == 1,
    }
    for key in ("generation_seconds", "fps", "end_to_end_seconds"):
        value = generation.get(key)
        checks[f"{key}_finite_positive"] = (
            value is not None and math.isfinite(float(value)) and float(value) > 0
        )
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{name} timing contract mismatch: {failed}")
    sequence_ids = sorted(per_sequence["metrics"])
    return {
        "checks": checks,
        "sequence_id": sequence_ids[0],
        "generation_metrics": generation,
        "aggregate": {
            "path": str(paths["timing_aggregate"]),
            "sha256": sha256_file(paths["timing_aggregate"]),
        },
        "per_sequence": {
            "path": str(paths["timing_per_sequence"]),
            "sha256": sha256_file(paths["timing_per_sequence"]),
        },
        "log": {
            "path": str(paths["timing_log"]),
            "sha256": sha256_file(paths["timing_log"]),
        },
    }


def table5_row(
    native: Mapping[str, Any],
    chois: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = native["metrics"]
    generation = native["generation_metrics"]
    return {
        "Te_cm": metrics["end_obj_trans_err"],
        "Txy_cm": metrics["xy_points_err"],
        "FS": metrics["foot_sliding"],
        "Rprec_paper_scalar": None,
        "Rprec_paper_scalar_note": (
            "not mapped; pinned evaluator reports R-Precision@1/2/3"
        ),
        "R-Precision@1": chois["metrics"]["R-Precision@1"],
        "R-Precision@2": chois["metrics"]["R-Precision@2"],
        "R-Precision@3": chois["metrics"]["R-Precision@3"],
        "FID": chois["metrics"]["FID"],
        "Cprec": metrics["contact_precision"],
        "Crec": metrics["contact_recall"],
        "Cf1": metrics["contact_f1"],
        "C_percent": metrics["contact_percent"],
        "Pbody": metrics["human_pen_loss_infbagel"],
        "MPJPE_cm": metrics["mpjpe"],
        "Troot_cm": metrics["trans_dist"],
        "Tobj_cm": metrics["obj_trans_dist"],
        "Oobj": metrics["obj_rot_dist"],
        "FPS_batch438_descriptive": generation["fps"],
        "FPS_batch1_local": timing["generation_metrics"]["fps"],
        "MatchingScore": chois["metrics"]["MatchingScore"],
        "Diversity": chois["metrics"]["Diversity"],
    }


_RUNTIME_OUTPUT = Path(".")


def run(args: argparse.Namespace) -> None:
    global _RUNTIME_OUTPUT
    _RUNTIME_OUTPUT = args.output.resolve()
    if git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("D2-AA refuses a dirty worker checkout")
    if load_json(args.resolved_config.resolve()) != resolved_lifecycle(args):
        raise ValueError("D2-AA runtime does not match archived resolved lifecycle")
    preflight = verify_preflight(args)
    environment = dict(os.environ)
    environment["ROOT_DIR"] = str(REPO)
    started = time.perf_counter()
    result_candidates: Dict[str, Any] = {}
    timing_sequence_id: Optional[str] = None
    for name in CANDIDATES:
        source_hashes = verify_candidate_sources(name)
        paths = candidate_output_paths(args.output.resolve(), name)
        run_logged(
            native_command(args, name, False),
            cwd=REPO / "code",
            environment=environment,
            log=paths["native_log"],
            exit_code=paths["native_exit_code"],
        )
        native_audit = validate_native_regeneration(name)
        run_logged(
            chois_command(args, name),
            cwd=REPO,
            environment=environment,
            log=paths["chois_log"],
            exit_code=paths["chois_exit_code"],
        )
        chois_result = validate_chois(name)
        run_logged(
            native_command(args, name, True),
            cwd=REPO / "code",
            environment=environment,
            log=paths["timing_log"],
            exit_code=paths["timing_exit_code"],
        )
        timing = validate_timing(name)
        if timing_sequence_id is None:
            timing_sequence_id = timing["sequence_id"]
        elif timing["sequence_id"] != timing_sequence_id:
            raise ValueError(
                f"{name} timing sequence differs: {timing['sequence_id']} "
                f"!= {timing_sequence_id}"
            )
        native_aggregate = load_json(paths["native_aggregate"])
        result_candidates[name] = {
            "source_hashes": source_hashes,
            "native_regeneration": native_audit,
            "chois": chois_result,
            "timing_batch1": timing,
            "table5_aligned": table5_row(
                native_aggregate,
                chois_result,
                timing,
            ),
        }
    ground_truth_hashes = {
        value["native_regeneration"]["export"]["ground_truth"]["sha256"]
        for value in result_candidates.values()
    }
    if ground_truth_hashes != {EXPECTED_GT_TREE_SHA256}:
        raise ValueError(f"D2-AA candidate GT trees differ: {ground_truth_hashes}")
    exclusive_json(args.metrics.resolve(), {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "status": "completed",
        "classification": "table5-completion-pass-nonselection-stop",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "runtime_seconds": time.perf_counter() - started,
        "preflight": preflight["checks"],
        "protocol": {
            "candidate_scope": list(CANDIDATES),
            "official_sequences": 438,
            "windows_per_sequence": 3,
            "diffusion_steps": 500,
            "weight_variant": "online",
            "only_quality_delta": "save_chois_eval_npz=true",
            "timing_sequence_id": timing_sequence_id,
            "paper_scalar_rprec_mapped": False,
            "checkpoint_selection": False,
        },
        "candidates": result_candidates,
        "training_started": False,
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_written": False,
        "checkpoint_selected": False,
        "consistency_started": False,
        "hsi_or_mixer_started": False,
    })


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--chois-root", type=Path, required=True)
    parser.add_argument("--text-to-motion-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--glove-root", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--feature-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-AA run id must be {RUN_ID}")
    if args.python.resolve() != Path(os.environ.get("INFBAGEL_PYTHON", "")).resolve():
        raise ValueError("D2-AA requires the absolute INFBAGEL_PYTHON")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-AA requires INFBAGEL_WORKER_EXPERT=hoi")
    if socket.gethostname() != "node01":
        raise RuntimeError("D2-AA is restricted to infbagel-4gpu/node01")
    if args.device != "cuda:0":
        raise ValueError("D2-AA is fixed to worker cuda:0")
    if args.resolve_only:
        prepare_resolved(args)
        return 0
    started = time.perf_counter()
    try:
        run(args)
    except Exception as error:
        if not args.metrics.resolve().exists():
            exclusive_json(args.metrics.resolve(), {
                "schema_version": 1,
                "run_id": RUN_ID,
                "phase": "p1",
                "subphase": SUBPHASE,
                "status": "failed",
                "classification": "table5-completion-contract-failure-stop",
                "seed": 42,
                "git_commit": git_output("rev-parse", "HEAD"),
                "runtime_seconds": time.perf_counter() - started,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "training_started": False,
                "optimizer_created": False,
                "training_updates": 0,
                "checkpoint_written": False,
                "checkpoint_selected": False,
                "consistency_started": False,
                "hsi_or_mixer_started": False,
            })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
