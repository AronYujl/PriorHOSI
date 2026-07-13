#!/usr/bin/env python3
"""Run the preregistered four-GPU Phase 1B HOIPrior capacity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


CANDIDATES = ((128, 512), (256, 1024), (512, 2048), (768, 3072))
PROCESSED_WINDOWS = 24576


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_snapshot() -> List[Dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    records = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        records.append({
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]),
            "memory_free_mib": int(fields[5]), "utilization_percent": int(fields[6]),
            "temperature_c": int(fields[7]),
        })
    return records


def compute_processes() -> List[str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


class Monitor:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.samples: List[List[Dict[str, object]]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                self.samples.append(gpu_snapshot())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop.wait(0.25)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> Dict[str, object]:
        self.stop.set()
        self.thread.join()
        by_gpu: Dict[int, Dict[str, List[int]]] = {}
        for sample in self.samples:
            for gpu in sample:
                values = by_gpu.setdefault(int(gpu["index"]), {"memory": [], "utilization": [], "temperature": []})
                values["memory"].append(int(gpu["memory_used_mib"]))
                values["utilization"].append(int(gpu["utilization_percent"]))
                values["temperature"].append(int(gpu["temperature_c"]))
        return {
            "sample_count": len(self.samples),
            "by_gpu": {
                str(index): {
                    "max_memory_used_mib": max(values["memory"]),
                    "mean_utilization_percent": statistics.fmean(values["utilization"]),
                    "max_utilization_percent": max(values["utilization"]),
                    "max_temperature_c": max(values["temperature"]),
                }
                for index, values in sorted(by_gpu.items()) if values["memory"]
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    code = repo / "code"
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    occupied = [run_dir / f"mb{micro_batch}" for micro_batch, _ in CANDIDATES]
    occupied.extend((args.output.resolve(), run_dir / "resolved_configs.json"))
    existing = [path for path in occupied if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite capacity artifacts: {existing}")

    initial_gpus = gpu_snapshot()
    initial_processes = compute_processes()
    idle = len(initial_gpus) == 4 and not initial_processes and all(
        int(gpu["memory_used_mib"]) <= 128 and int(gpu["utilization_percent"]) == 0
        for gpu in initial_gpus
    )
    if not idle:
        result = {
            "schema_version": 1,
            "status": "aborted",
            "failure_stage": "four_gpu_idle_preflight",
            "gpu_workload_started": False,
            "initial_gpus": initial_gpus,
            "initial_compute_processes": initial_processes,
            "candidates": [],
        }
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    jobs = []
    preflights = []
    for micro_batch, effective_batch in CANDIDATES:
        candidate = run_dir / f"mb{micro_batch}"
        candidate.mkdir()
        metrics = candidate / "metrics.json"
        state = candidate / "training_state.json"
        checkpoints = candidate / "checkpoints"
        hydra_dir = candidate / "hydra"
        candidate_run_id = f"{args.run_id}-mb{micro_batch}"
        command = [
            args.python,
            "train_hoi_prior.py",
            f"run_id={candidate_run_id}",
            "mode=capacity",
            "seed=42",
            "num_gpus=4",
            f"batch_size={micro_batch}",
            f"effective_batch_size={effective_batch}",
            "gradient_accumulation_steps=1",
            "num_workers=4",
            "dataset_limit=0",
            f"max_processed_windows={PROCESSED_WINDOWS}",
            "validation_windows=0",
            "validation_interval_windows=0",
            "checkpoint_interval_windows=0",
            "learning_rate=0.0001",
            "warmup_windows=0",
            "profile_every_update=true",
            f"output_dir={candidate}",
            f"checkpoint_dir={checkpoints}",
            f"metrics_path={metrics}",
            f"state_path={state}",
            f"hydra.run.dir={hydra_dir}",
        ]
        resolved = candidate / "resolved_config.yaml"
        completed = subprocess.run(
            command + ["--cfg", "job", "--resolve"], cwd=code, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        resolved.write_text(completed.stdout, encoding="utf-8")
        preflights.append({
            "micro_batch_per_gpu": micro_batch,
            "effective_batch_size": effective_batch,
            "gradient_accumulation_steps": 1,
            "processed_windows": PROCESSED_WINDOWS,
            "optimizer_updates": PROCESSED_WINDOWS // effective_batch,
            "returncode": completed.returncode,
            "resolved_config_path": str(resolved),
            "resolved_config_sha256": sha256(resolved),
            "command": command,
        })
        jobs.append((micro_batch, effective_batch, candidate, metrics, resolved, command))
    resolved_archive = run_dir / "resolved_configs.json"
    atomic_json(resolved_archive, {
        "schema_version": 1,
        "all_preflights_passed": all(record["returncode"] == 0 for record in preflights),
        "jobs": preflights,
    })
    if any(record["returncode"] != 0 for record in preflights):
        result = {
            "schema_version": 1, "status": "failed", "failure_stage": "resolved_config_preflight",
            "gpu_workload_started": False, "preflights": preflights, "candidates": [],
        }
        atomic_json(args.output, result)
        return 2

    records = []
    for micro_batch, effective_batch, candidate, metrics, resolved, command in jobs:
        log = candidate / "train.log"
        monitor = Monitor()
        monitor.start()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, cwd=code, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False,
            )
        utilization = monitor.finish()
        record: Dict[str, object] = {
            "micro_batch_per_gpu": micro_batch,
            "effective_batch_size": effective_batch,
            "gradient_accumulation_steps": 1,
            "processed_windows": PROCESSED_WINDOWS,
            "processed_frames": PROCESSED_WINDOWS * 16,
            "expected_optimizer_updates": PROCESSED_WINDOWS // effective_batch,
            "returncode": completed.returncode,
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_path": str(log),
            "log_sha256": sha256(log),
            "resolved_config_path": str(resolved),
            "resolved_config_sha256": sha256(resolved),
            "gpu_monitor": utilization,
        }
        if completed.returncode == 0 and metrics.is_file():
            values = json.loads(metrics.read_text(encoding="utf-8"))
            record.update(values)
            headroom = min(values["memory_headroom_bytes_by_rank"])
            threshold = max(2 * 1024 ** 3, int(values["device_total_memory_bytes"] * 0.10))
            record["headroom_threshold_bytes"] = threshold
            record["headroom_passed"] = headroom >= threshold
            record["stable"] = bool(values["loss_finite"] and values["key_gradient_present"] and headroom >= threshold)
        else:
            record["stable"] = False
            text = log.read_text(encoding="utf-8", errors="replace")
            record["oom"] = "out of memory" in text.lower()
            record["failure_reason"] = "training subprocess failed or omitted metrics"
            record["log_tail"] = text.splitlines()[-80:]
        records.append(record)

    stable = [record for record in records if record.get("stable")]
    selected = max((int(record["micro_batch_per_gpu"]) for record in stable), default=None)
    selected_record = next((record for record in stable if record["micro_batch_per_gpu"] == selected), None)
    result = {
        "schema_version": 1,
        "status": "completed" if stable else "failed",
        "protocol": "real scene-free HOIPrior DDP forward/backward/optimizer multi-update soak",
        "seed": 42,
        "world_size": 4,
        "initial_gpus": initial_gpus,
        "initial_compute_processes": initial_processes,
        "external_contention": False,
        "preflights": preflights,
        "resolved_configs_path": str(resolved_archive),
        "resolved_configs_sha256": sha256(resolved_archive),
        "candidates": records,
        "selected_micro_batch_per_gpu": selected,
        "selected_effective_batch_size": selected_record["effective_batch_size"] if selected_record else None,
        "selected_gradient_accumulation_steps": 1 if selected_record else None,
        "selection_rule": "largest stable candidate with >=max(2GiB,10%) headroom and no contention",
        "all_failures_and_ooms_retained": True,
        "ended_gpus": gpu_snapshot(),
        "ended_compute_processes": compute_processes(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
