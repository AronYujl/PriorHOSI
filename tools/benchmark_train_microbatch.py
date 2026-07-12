#!/usr/bin/env python3
"""Run the preregistered 8-GPU InfBaGel training micro-batch audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CANDIDATES = (32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, nargs="+", default=DEFAULT_CANDIDATES)
    return parser.parse_args()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    code_dir = repo / "code"
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    run_dir.mkdir(parents=True)

    jobs: List[Dict[str, Any]] = []
    preflight_records: List[Dict[str, Any]] = []
    for micro_batch in args.candidates:
        denominator = micro_batch * args.world_size
        if args.effective_batch_size % denominator:
            raise ValueError(
                f"effective batch {args.effective_batch_size} is not divisible by "
                f"micro-batch {micro_batch} x world size {args.world_size}"
            )
        accumulation = args.effective_batch_size // denominator
        candidate_dir = run_dir / f"mb{micro_batch}"
        metrics_path = candidate_dir / "benchmark_metrics.json"
        log_path = candidate_dir / "train.log"
        candidate_dir.mkdir()
        command = [
            args.python,
            "train_infbagel.py",
            f"exp_name=p0-microbatch-mb{micro_batch}",
            f"seed={args.seed}",
            f"batch_size={micro_batch}",
            f"num_gpus={args.world_size}",
            f"effective_batch_size={args.effective_batch_size}",
            f"gradient_accumulation_steps={accumulation}",
            "epochs=1",
            "max_optimizer_updates=1",
            "save_checkpoints=false",
            "use_tensorboard=false",
            f"exp_dir={candidate_dir}",
            f"benchmark_metrics_path={metrics_path}",
            f"hydra.run.dir={candidate_dir / 'hydra'}",
        ]
        resolved_path = candidate_dir / "resolved_config.yaml"
        preflight = subprocess.run(
            command + ["--cfg", "job", "--resolve"],
            cwd=code_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        resolved_path.write_text(preflight.stdout, encoding="utf-8")
        preflight_record = {
            "micro_batch_per_gpu": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": args.effective_batch_size,
            "returncode": preflight.returncode,
            "resolved_config_path": str(resolved_path),
            "resolved_config_sha256": hashlib.sha256(
                preflight.stdout.encode("utf-8")
            ).hexdigest(),
            "command": command,
        }
        if preflight.returncode != 0:
            preflight_record.update({
                "stable": False,
                "failure_reason": "Hydra resolved-config preflight failed",
            })
        preflight_records.append(preflight_record)
        jobs.append({
            "micro_batch": micro_batch,
            "accumulation": accumulation,
            "candidate_dir": candidate_dir,
            "metrics_path": metrics_path,
            "log_path": log_path,
            "resolved_path": resolved_path,
            "command": command,
        })

    resolved_archive = run_dir / "resolved_configs.json"
    atomic_json(resolved_archive, {
        "schema_version": 1,
        "all_preflights_passed": all(
            record["returncode"] == 0 for record in preflight_records
        ),
        "jobs": preflight_records,
    })
    if any(record["returncode"] != 0 for record in preflight_records):
        result = {
            "schema_version": 1,
            "protocol": "Hydra resolved-config preflight; no GPU workload launched",
            "seed": args.seed,
            "world_size": args.world_size,
            "effective_batch_size": args.effective_batch_size,
            "candidates": preflight_records,
            "resolved_configs_path": str(resolved_archive),
            "selected_micro_batch_per_gpu": None,
            "all_candidates_stable": False,
        }
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    records: List[Dict[str, Any]] = []
    for job in jobs:
        micro_batch = job["micro_batch"]
        accumulation = job["accumulation"]
        metrics_path = job["metrics_path"]
        log_path = job["log_path"]
        resolved_path = job["resolved_path"]
        command = job["command"]
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=code_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        record: Dict[str, Any] = {
            "micro_batch_per_gpu": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": args.effective_batch_size,
            "returncode": completed.returncode,
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_path": str(log_path),
            "resolved_config_path": str(resolved_path),
        }
        if completed.returncode == 0 and metrics_path.is_file():
            record.update(json.loads(metrics_path.read_text(encoding="utf-8")))
            record["stable"] = True
        else:
            record["stable"] = False
            record["failure_reason"] = "training subprocess failed or did not emit metrics"
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            record["log_tail"] = tail
        records.append(record)

    stable = [record for record in records if record["stable"]]
    result = {
        "schema_version": 1,
        "protocol": "one real optimizer update on OMOMO train data; full forward/backward; DDP",
        "seed": args.seed,
        "world_size": args.world_size,
        "effective_batch_size": args.effective_batch_size,
        "candidates": records,
        "resolved_configs_path": str(resolved_archive),
        "selected_micro_batch_per_gpu": max(
            (record["micro_batch_per_gpu"] for record in stable), default=None
        ),
        "all_candidates_stable": len(stable) == len(records),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
