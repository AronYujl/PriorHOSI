#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-M0 paired fresh-optimizer smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.optimizer_reset import (  # noqa: E402
    CANDIDATES,
    EFFECTIVE_BATCH_SIZE,
    OPTIMIZER_UPDATES,
    PROCESSED_FRAMES,
    PROCESSED_WINDOWS,
    RUN_ID,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_OPTIMIZER_LR,
    WEIGHTS,
    WEIGHT_SOURCE_METRICS_SHA256,
    WEIGHT_SOURCE_RUN,
    mechanism_gate,
)
from tools.evaluate_hoi_d2m import evaluate, sha256_file  # noqa: E402
from priors.window_codec import BPS_SHA256  # noqa: E402


EXPECTED_NORMALIZATION_SHA256 = "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def candidate_paths(output: Path, candidate: str) -> Dict[str, Path]:
    root = output / "candidates" / candidate
    run_id = f"{RUN_ID}-{candidate}"
    return {
        "root": root,
        "metrics": root / "metrics.json",
        "state": root / "training_state.json",
        "checkpoint_dir": root / "checkpoints",
        "hydra": root / "hydra",
        "log": output / f"train_{candidate}.log",
        "resolved": output / f"resolved_{candidate}.yaml",
        "run_id": run_id,
    }


def candidate_overrides(
    source_checkpoint: Path,
    output: Path,
    candidate: str,
) -> Sequence[str]:
    paths = candidate_paths(output, candidate)
    weights = WEIGHTS[candidate]
    return (
        f"run_id={paths['run_id']}",
        f"d2m_candidate={candidate}",
        f"weight_init_checkpoint={source_checkpoint}",
        f"weight_init_sha256={SOURCE_CHECKPOINT_SHA256}",
        "weight_init_variant=online",
        f"fk_weight={weights['fk']}",
        f"object_surface_weight={weights['object_surface']}",
        f"velocity_weight={weights['velocity']}",
        f"goal_weight={weights['terminal_goal']}",
        f"output_dir={paths['root']}",
        f"checkpoint_dir={paths['checkpoint_dir']}",
        f"metrics_path={paths['metrics']}",
        f"state_path={paths['state']}",
        f"hydra.run.dir={paths['hydra']}",
    )


def candidate_command(
    python: Path,
    source_checkpoint: Path,
    output: Path,
    candidate: str,
) -> Sequence[str]:
    return (
        str(python),
        str(REPO / "code/train_hoi_prior.py"),
        "--config-name",
        "config_train_hoi_prior_d2m",
        *candidate_overrides(source_checkpoint, output, candidate),
    )


def resolve_candidate(
    python: Path,
    source_checkpoint: Path,
    output: Path,
    candidate: str,
) -> str:
    command = (
        str(python),
        str(REPO / "code/train_hoi_prior.py"),
        "--config-name",
        "config_train_hoi_prior_d2m",
        "--cfg",
        "job",
        "--resolve",
        *candidate_overrides(source_checkpoint, output, candidate),
    )
    environment = dict(os.environ)
    environment["ROOT_DIR"] = str(REPO)
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"D2-M {candidate} config resolution failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    if "${" in completed.stdout:
        raise ValueError(f"D2-M {candidate} config contains unresolved interpolation")
    return completed.stdout


def resolved_config(args) -> Dict[str, object]:
    output = args.output.resolve()
    source = args.source_checkpoint.resolve()
    candidates = {}
    for candidate in CANDIDATES:
        paths = candidate_paths(output, candidate)
        if not paths["resolved"].is_file():
            raise FileNotFoundError(paths["resolved"])
        content = paths["resolved"].read_text(encoding="utf-8")
        if "${" in content:
            raise ValueError(f"D2-M {candidate} archived config contains unresolved interpolation")
        candidates[candidate] = {
            "command": list(candidate_command(args.python.resolve(), source, output, candidate)),
            "resolved_config_path": str(paths["resolved"]),
            "resolved_config_sha256": sha256_text(content),
            "output_dir": str(paths["root"]),
            "metrics_path": str(paths["metrics"]),
            "training_log": str(paths["log"]),
            "weights": WEIGHTS[candidate],
        }
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-M0",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "repo_root": str(REPO),
        "python": str(args.python.resolve()),
        "source_checkpoint": {
            "path": str(source),
            "sha256": args.source_sha256,
            "weight_variant": "online",
            "restored_components": ["model"],
            "forbidden_restored_components": [
                "optimizer", "ema_models", "ema_model", "scheduler", "scaler", "rng",
            ],
        },
        "weight_source": {
            "run_id": WEIGHT_SOURCE_RUN,
            "path": str(args.weight_source.resolve()),
            "sha256": args.weight_source_sha256,
        },
        "assets": {
            "normalization": {
                "path": str((REPO / "data/train/norm.npy").resolve()),
                "sha256": EXPECTED_NORMALIZATION_SHA256,
            },
            "bps": {
                "path": str((REPO / "code/bps.pt").resolve()),
                "sha256": BPS_SHA256,
            },
        },
        "training": {
            "candidates": candidates,
            "gpu_count": 4,
            "micro_batch_per_gpu": 768,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "optimizer_updates_per_candidate": OPTIMIZER_UPDATES,
            "processed_windows_per_candidate": PROCESSED_WINDOWS,
            "processed_frames_per_candidate": PROCESSED_FRAMES,
            "learning_rate": SOURCE_OPTIMIZER_LR,
            "warmup_windows": 0,
            "minimum_lr_ratio": 1.0,
            "paired_training_rng": True,
        },
        "evaluation": {
            "device": args.device,
            "teacher_batch_size": args.teacher_batch_size,
            "weight_variants": {"source": "online", "current": "online", "balanced": "online"},
            "official_test_used": False,
            "chois_used": False,
        },
        "output": str(output),
        "metrics_path": str(args.metrics.resolve()),
        "released_checkpoint_used": False,
        "d2h1_started": False,
        "full_training_started": False,
    }


def prepare_resolved_config(args) -> None:
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"D2-M output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for candidate in CANDIDATES:
        content = resolve_candidate(
            args.python.resolve(),
            args.source_checkpoint.resolve(),
            output,
            candidate,
        )
        paths = candidate_paths(output, candidate)
        if paths["resolved"].exists():
            if paths["resolved"].read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"refusing to overwrite changed D2-M candidate config {paths['resolved']}"
                )
        else:
            with paths["resolved"].open("x", encoding="utf-8") as handle:
                handle.write(content)
    exclusive_json(args.resolved_config.resolve(), resolved_config(args))


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_summary(
    source_checkpoint: Path,
    source_sha256: str,
    candidate_metrics: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    initial_hashes = {
        name: candidate_metrics[name]["weight_initialization"]["initial_model_state_sha256"]
        for name in CANDIDATES
    }
    source_model_hashes = {
        name: candidate_metrics[name]["weight_initialization"]["source_model_state_sha256"]
        for name in CANDIDATES
    }
    old_load_fields = (
        "old_optimizer_states_loaded",
        "old_ema_models_loaded",
        "old_scheduler_states_loaded",
        "old_scaler_states_loaded",
        "old_rng_states_loaded",
    )
    candidates = {}
    for name in CANDIDATES:
        metrics = candidate_metrics[name]
        candidates[name] = {
            "status": metrics.get("status"),
            "loss_finite": metrics.get("loss_finite"),
            "optimizer_updates": metrics.get("optimizer_updates"),
            "processed_windows": metrics.get("processed_windows"),
            "amp_overflow_skips_by_rank": metrics.get("amp_overflow_skips_by_rank"),
            "initial_optimizer_state_count": metrics.get("initial_optimizer_state_count"),
            "terminal_optimizer_state_count": metrics.get("terminal_optimizer_state_count"),
            "terminal_optimizer_step_min": metrics.get("terminal_optimizer_step_min"),
            "terminal_optimizer_step_max": metrics.get("terminal_optimizer_step_max"),
            "training_rng_sha256_by_rank": metrics.get("training_rng_sha256_by_rank"),
            "terminal_checkpoint": metrics.get("terminal_checkpoint"),
            "terminal_checkpoint_sha256": metrics.get("terminal_checkpoint_sha256"),
            "weight_initialization": metrics.get("weight_initialization"),
            "mean_training_losses": metrics.get("mean_training_losses"),
            "peak_memory_allocated_bytes_by_rank": metrics.get(
                "peak_memory_allocated_bytes_by_rank"
            ),
            "peak_memory_reserved_bytes_by_rank": metrics.get(
                "peak_memory_reserved_bytes_by_rank"
            ),
            "throughput_windows_per_second": metrics.get("throughput_windows_per_second"),
            "wall_seconds": metrics.get("wall_seconds"),
        }
    all_finite = all(
        candidate_metrics[name].get("status") == "stable"
        and bool(candidate_metrics[name].get("loss_finite"))
        and int(candidate_metrics[name].get("optimizer_updates", -1)) == OPTIMIZER_UPDATES
        and int(candidate_metrics[name].get("processed_windows", -1)) == PROCESSED_WINDOWS
        and all(
            int(value) == 0
            for value in candidate_metrics[name].get("amp_overflow_skips_by_rank", [])
        )
        for name in CANDIDATES
    )
    old_state_load_counts_zero = all(
        int(candidate_metrics[name]["weight_initialization"].get(field, -1)) == 0
        for name in CANDIDATES for field in old_load_fields
    )
    paired_rng = (
        candidate_metrics["current"].get("training_rng_sha256_by_rank")
        == candidate_metrics["balanced"].get("training_rng_sha256_by_rank")
        and bool(candidate_metrics["current"].get("training_rng_sha256_by_rank"))
    )
    source_model_exact = all(
        initial_hashes[name] == source_model_hashes[name] for name in CANDIDATES
    )
    return {
        "source_checkpoint_hash_exact": (
            source_sha256 == SOURCE_CHECKPOINT_SHA256
            and sha256_file(source_checkpoint) == SOURCE_CHECKPOINT_SHA256
        ),
        "source_model_hash_exact": source_model_exact,
        "source_model_state_sha256": source_model_hashes["current"],
        "initial_model_state_sha256": initial_hashes,
        "initial_model_hashes_equal": len(set(initial_hashes.values())) == 1,
        "old_state_load_counts_zero": old_state_load_counts_zero,
        "paired_training_rng_audit": paired_rng,
        "all_finite": all_finite,
        "candidates": candidates,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--weight-source", type=Path, required=True)
    parser.add_argument("--weight-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-M0 run id must be {RUN_ID}")
    if args.source_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("D2-M0 source checkpoint requested hash mismatch")
    if args.weight_source_sha256 != WEIGHT_SOURCE_METRICS_SHA256:
        raise ValueError("D2-M0 weight-source metrics requested hash mismatch")
    if args.teacher_batch_size < 2:
        raise ValueError("D2-M0 teacher batch size must be at least two")
    if args.python.resolve() != Path(os.environ.get("INFBAGEL_PYTHON", "")).resolve():
        raise ValueError("D2-M0 requires the absolute INFBAGEL_PYTHON interpreter")
    if args.resolve_only:
        prepare_resolved_config(args)
        return
    archived = _load_json(args.resolved_config.resolve())
    if archived != resolved_config(args):
        raise ValueError("D2-M0 runtime arguments do not match the archived resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-M0 requires INFBAGEL_WORKER_EXPERT=hoi")
    if git_output("status", "--porcelain"):
        raise RuntimeError("D2-M0 refuses a dirty worker checkout")
    if sha256_file(args.source_checkpoint.resolve()) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("D2-M0 source checkpoint file hash mismatch")
    asset_hashes = {
        "weight_source": sha256_file(args.weight_source.resolve()),
        "normalization": sha256_file((REPO / "data/train/norm.npy").resolve()),
        "bps": sha256_file((REPO / "code/bps.pt").resolve()),
    }
    expected_asset_hashes = {
        "weight_source": WEIGHT_SOURCE_METRICS_SHA256,
        "normalization": EXPECTED_NORMALIZATION_SHA256,
        "bps": BPS_SHA256,
    }
    if asset_hashes != expected_asset_hashes:
        raise ValueError(
            f"D2-M0 asset hash mismatch: actual={asset_hashes}, expected={expected_asset_hashes}"
        )
    if args.metrics.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.metrics.resolve()}")
    started = time.time()
    environment = dict(os.environ)
    environment["ROOT_DIR"] = str(REPO)
    candidate_metrics = {}
    for candidate in CANDIDATES:
        paths = candidate_paths(args.output.resolve(), candidate)
        if paths["root"].exists():
            raise FileExistsError(f"refusing to overwrite D2-M candidate directory {paths['root']}")
        with paths["log"].open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                archived["training"]["candidates"][candidate]["command"],
                cwd=REPO,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"D2-M {candidate} training failed with return code {completed.returncode}"
            )
        candidate_metrics[candidate] = _load_json(paths["metrics"])
    training = _training_summary(
        args.source_checkpoint.resolve(),
        args.source_sha256,
        candidate_metrics,
    )
    training["asset_hashes_exact"] = True
    training["asset_hashes"] = asset_hashes
    terminal_checkpoints = {
        name: Path(str(candidate_metrics[name]["terminal_checkpoint"])).resolve()
        for name in CANDIDATES
    }
    terminal_hashes = {
        name: str(candidate_metrics[name]["terminal_checkpoint_sha256"])
        for name in CANDIDATES
    }
    evaluation = evaluate(
        {
            "source": args.source_checkpoint.resolve(),
            "current": terminal_checkpoints["current"],
            "balanced": terminal_checkpoints["balanced"],
        },
        {
            "source": args.source_sha256,
            "current": terminal_hashes["current"],
            "balanced": terminal_hashes["balanced"],
        },
        device=torch.device(args.device),
        teacher_batch_size=args.teacher_batch_size,
    )
    decision = mechanism_gate(
        training,
        evaluation["teacher"],
        evaluation["native"],
    )
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-M0",
        "seed": 42,
        "git_commit": git_output("rev-parse", "HEAD"),
        "status": "completed",
        "source_checkpoint": {
            "path": str(args.source_checkpoint.resolve()),
            "sha256": args.source_sha256,
            "weight_variant": "online",
        },
        "training": training,
        "evaluation": evaluation,
        "decision": decision,
        "runtime_seconds": time.time() - started,
        "released_checkpoint_used": False,
        "source_ema_used": False,
        "terminal_ema_used": False,
        "checkpoint_selection": False,
        "official_test_used": False,
        "chois_used": False,
        "production_model_change": False,
        "representation_change": False,
        "condition_change": False,
        "sampler_change": False,
        "d2h1_started": False,
        "from_random_screen_started": False,
        "full_training_started": False,
    }
    exclusive_json(args.metrics.resolve(), output)


if __name__ == "__main__":
    main()
