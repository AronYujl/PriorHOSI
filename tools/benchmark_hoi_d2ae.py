#!/usr/bin/env python3
"""Registered 4-GPU full-micro-batch performance gate for D2-AE0."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors import sparse_relation as sparse_relation_module  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion  # noqa: E402
from priors.models import HOI_ARCHITECTURE_D2AE, build_expert  # noqa: E402
from tools.smoke_hoi_d2ac import _atomic_json, _gpu_contention  # noqa: E402
from tools.smoke_hoi_d2ae import (  # noqa: E402
    _atomic_text,
    _resolved_config,
    _sha256_file,
    _verify_worker,
)
from train_hoi_prior import (  # noqa: E402
    D2AE_MAXIMUM_ETA_HOURS,
    D2AE_MINIMUM_THROUGHPUT,
    LOSS_KEYS,
    _build_optimizer,
    _d2ae_formal_source_contract,
    _d2ae_gradient_audit,
    _forward_losses,
    _gradient_l2_norm,
    _move_batch,
    _state_dict_sha256,
    _validate_author_update_execution_host,
    _validate_d2ae_contract,
    _validate_d2ae_formal_run_id,
    _validate_fk_foot_temporal_routing_mode,
)


RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ae-performance-benchmark"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
WORLD_SIZE = 4
MICRO_BATCH_PER_GPU = 512
EFFECTIVE_BATCH = 2048
WARMUP_UPDATES = 64
MEASURED_UPDATES = 256
TOTAL_UPDATES = 320
MEASURED_WINDOWS = 524_288
FORMAL_WINDOWS = 61_440_000
SEALED_D2X_THROUGHPUT = 3243.0357134915853
MINIMUM_THROUGHPUT = D2AE_MINIMUM_THROUGHPUT
MAXIMUM_ETA_HOURS = D2AE_MAXIMUM_ETA_HOURS
MINIMUM_HEADROOM_BYTES = 2 * 1024**3
FAILURE_CLASSIFICATION = "sparse-relation-field-performance-negative-stop"


def _mean(values):
    return float(statistics.fmean(values)) if values else None


def _summary(values) -> dict:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "sum": float(sum(values)),
        "mean": _mean(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _resolved_workload_config(
    cfg,
    *,
    repo: Path,
    run_id: str,
    expected_commit: str,
    formal_source_contract: dict,
    output_dir: Path,
    resolved_config_output: Path,
) -> str:
    command = [
        str(Path(sys.executable).resolve()), "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=4",
        "tools/benchmark_hoi_d2ae.py",
        "--repo-root", str(repo),
        "--output-dir", str(output_dir.resolve()),
        "--resolved-config-output", str(resolved_config_output.resolve()),
        "--expected-commit", expected_commit,
        "--run-id", run_id,
        "--formal-run-id", str(cfg.run_id),
    ]
    value = {
        "schema_version": 1,
        "lifecycle": "d2ae_four_gpu_full_micro_batch_performance_benchmark",
        "run_id": run_id,
        "formal_run_id": str(cfg.run_id),
        "expected_git_commit": expected_commit,
        "formal_source_contract": formal_source_contract,
        "repo_root": str(repo),
        "python": str(Path(sys.executable).resolve()),
        "output_dir": str(output_dir.resolve()),
        "resolved_config_output": str(resolved_config_output.resolve()),
        "launcher": command,
        "workload": {
            "host": "node01",
            "world_size": WORLD_SIZE,
            "gpu_model": "RTX 3090",
            "micro_batch_per_gpu": MICRO_BATCH_PER_GPU,
            "effective_batch_size": EFFECTIVE_BATCH,
            "gradient_accumulation_steps": 1,
            "warmup_updates": WARMUP_UPDATES,
            "measured_updates": MEASURED_UPDATES,
            "total_updates": TOTAL_UPDATES,
            "measured_windows": MEASURED_WINDOWS,
            "seed": 42,
            "random_initialization": True,
            "optimizer": "FP32 Adam",
            "learning_rate": 1.0e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "amp": False,
            "checkpoint_loads": 0,
            "checkpoint_writes": 0,
            "weights_reusable": False,
        },
        "performance_gate": {
            "minimum_throughput_windows_per_second": MINIMUM_THROUGHPUT,
            "maximum_full_budget_eta_hours": MAXIMUM_ETA_HOURS,
            "minimum_memory_headroom": "max(2 GiB, 10% device memory)",
            "failure_classification": FAILURE_CLASSIFICATION,
            "sweep_on_failure": False,
        },
        "timing": {
            "cuda_synchronized_boundaries": True,
            "categories": [
                "loader_wait", "h2d", "gpu_relation_geometry",
                "gpu_relation_module", "forward_and_loss", "backward",
                "ddp_allreduce_bucket_wall", "gradient_validation", "optimizer",
            ],
            "ddp_note": (
                "bucket wall durations may overlap; backward CUDA time is the "
                "authoritative inclusive DDP critical-path measurement"
            ),
        },
        "formal_training_config_reference": OmegaConf.to_container(
            cfg, resolve=True,
        ),
    }
    resolved = OmegaConf.to_yaml(OmegaConf.create(value), resolve=True)
    if "${" in resolved:
        raise RuntimeError("D2-AE benchmark resolved config is incomplete")
    return resolved


class UtilizationMonitor:
    def __init__(self) -> None:
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._previous_cpu = None

    @staticmethod
    def _cpu_totals():
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    def _sample(self) -> None:
        while not self._stop.is_set():
            total, idle = self._cpu_totals()
            cpu_percent = None
            if self._previous_cpu is not None:
                previous_total, previous_idle = self._previous_cpu
                delta_total = total - previous_total
                if delta_total > 0:
                    cpu_percent = 100.0 * (
                        1.0 - (idle - previous_idle) / delta_total
                    )
            self._previous_cpu = (total, idle)
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            gpus = []
            if query.returncode == 0:
                for line in query.stdout.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if len(fields) == 7:
                        gpus.append({
                            "index": int(fields[0]),
                            "gpu_utilization_percent": float(fields[1]),
                            "memory_utilization_percent": float(fields[2]),
                            "memory_used_mib": float(fields[3]),
                            "memory_total_mib": float(fields[4]),
                            "temperature_c": float(fields[5]),
                            "power_w": float(fields[6]),
                        })
            self.samples.append({
                "time": time.time(),
                "cpu_utilization_percent": cpu_percent,
                "load_average": list(os.getloadavg()),
                "gpus": gpus,
            })
            self._stop.wait(1.0)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def summary(self) -> dict:
        cpu = [
            sample["cpu_utilization_percent"] for sample in self.samples
            if sample["cpu_utilization_percent"] is not None
        ]
        per_gpu = {}
        for sample in self.samples:
            for gpu in sample["gpus"]:
                per_gpu.setdefault(str(gpu["index"]), []).append(gpu)
        return {
            "sample_count": len(self.samples),
            "cpu_utilization_percent": _summary(cpu),
            "load_average_max": [
                max((sample["load_average"][index] for sample in self.samples), default=0.0)
                for index in range(3)
            ],
            "gpus": {
                index: {
                    key: _summary([row[key] for row in rows])
                    for key in (
                        "gpu_utilization_percent", "memory_utilization_percent",
                        "memory_used_mib", "temperature_c", "power_w",
                    )
                }
                for index, rows in per_gpu.items()
            },
            "samples": self.samples,
        }


class Profiler:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.active = False
        self.cuda_pairs = {}
        self.loader_wait = []
        self.relation_shapes = {}
        self.relation_runtime = {}

    def pair(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if self.active:
            self.cuda_pairs.setdefault(name, []).append((start, end))
        return end

    def finish(self, end) -> None:
        end.record()

    def summaries(self) -> dict:
        return {
            name: _summary([
                start.elapsed_time(end) / 1000.0 for start, end in pairs
            ])
            for name, pairs in self.cuda_pairs.items()
        }


class DDPCommState:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.active = False
        self.current = []
        self.steps = []

    def begin(self, active: bool) -> None:
        self.active = active
        self.current = []

    def end(self) -> None:
        if self.active:
            self.steps.append(list(self.current))
        self.active = False


def _ddp_hook(state: DDPCommState, bucket):
    started = time.perf_counter()
    future = dist.all_reduce(bucket.buffer(), async_op=True).get_future()

    def complete(completed):
        value = completed.value()[0]
        value.div_(state.world_size)
        if state.active:
            state.current.append({
                "bucket_elements": int(value.numel()),
                "wall_seconds": time.perf_counter() - started,
            })
        return value

    return future.then(complete)


def _aggregate_rank_values(value: float, device: torch.device) -> dict:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    total = tensor.clone()
    minimum = tensor.clone()
    maximum = tensor.clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return {
        "mean": float((total / WORLD_SIZE).item()),
        "minimum": float(minimum.item()),
        "maximum": float(maximum.item()),
    }


def _external_compute_processes(snapshot: dict, allowed_pids) -> list:
    allowed = {int(value) for value in allowed_pids}
    external = []
    for record in snapshot.get("compute_processes", []):
        fields = [item.strip() for item in str(record).split(",")]
        if len(fields) < 2:
            external.append(str(record))
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            external.append(str(record))
            continue
        if pid not in allowed:
            external.append(str(record))
    return external


def _next_batch(state: dict):
    started = time.perf_counter()
    try:
        batch = next(state["iterator"])
    except StopIteration:
        state["epoch"] += 1
        state["sampler"].set_epoch(state["epoch"])
        state["iterator"] = iter(state["loader"])
        batch = next(state["iterator"])
    return batch, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolved-config-output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-run-id", required=True)
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    run_id_match = RUN_ID_RE.fullmatch(args.run_id)
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if run_id_match is None or run_id_match.group("date") != actual_date:
        raise ValueError(
            "D2-AE benchmark run id must use the locked stem and actual date"
        )
    formal_run_id_contract = _validate_d2ae_formal_run_id(
        args.formal_run_id
    )
    if formal_run_id_contract["date"] != run_id_match.group("date"):
        raise ValueError(
            "D2-AE benchmark and intended formal run ids must use the same date"
        )
    formal_run_id = args.formal_run_id

    cfg = _resolved_config(repo, formal_run_id)
    _validate_fk_foot_temporal_routing_mode(cfg)
    _validate_d2ae_contract(
        cfg, WORLD_SIZE, require_performance_gate=False,
    )
    identity = _verify_worker(repo, args.expected_commit)
    formal_source_contract = _d2ae_formal_source_contract(repo)
    _validate_author_update_execution_host(cfg)
    resolved_yaml = _resolved_workload_config(
        cfg,
        repo=repo,
        run_id=args.run_id,
        expected_commit=args.expected_commit,
        formal_source_contract=formal_source_contract,
        output_dir=output_dir,
        resolved_config_output=args.resolved_config_output,
    )
    if args.resolve_only:
        _atomic_text(args.resolved_config_output, resolved_yaml)
        print(json.dumps({
            "schema_version": 1,
            "status": "resolved-config-archived",
            "run_id": args.run_id,
            "resolved_config_path": str(args.resolved_config_output.resolve()),
            "gpu_workload_started": False,
        }, indent=2, sort_keys=True), flush=True)
        return 0
    resolved_path = args.resolved_config_output.resolve()
    if not resolved_path.is_file() or resolved_path.read_text(encoding="utf-8") != resolved_yaml:
        raise RuntimeError("D2-AE benchmark differs from its archived resolved config")
    resolved_sha256 = _sha256_file(resolved_path)

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE) or local_rank not in range(WORLD_SIZE):
        raise RuntimeError("D2-AE benchmark requires torchrun with exactly four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("D2-AE benchmark requires exactly four visible CUDA devices")
    for index in range(WORLD_SIZE):
        if "RTX 3090" not in torch.cuda.get_device_name(index):
            raise RuntimeError("D2-AE benchmark requires 4x RTX 3090")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    local_pid = torch.tensor(os.getpid(), device=device, dtype=torch.int64)
    rank_pid_tensors = [torch.zeros_like(local_pid) for _ in range(WORLD_SIZE)]
    dist.all_gather(rank_pid_tensors, local_pid)
    rank_pids = [int(value.item()) for value in rank_pid_tensors]

    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = PriorWindowDataset(
        str(repo), "hoi", partition="train", split_manifest=str(cfg.split_manifest),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=rank, shuffle=True, seed=42,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH_PER_GPU,
        sampler=sampler,
        drop_last=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=True,
    )
    model = build_expert(
        "hoi", init_checkpoint=None, dim_model=512, num_heads=16, num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AE,
    ).to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False,
    )
    initial_model_sha256 = _state_dict_sha256(model.module.state_dict())
    comm_state = DDPCommState(WORLD_SIZE)
    model.register_comm_hook(comm_state, _ddp_hook)
    optimizer = _build_optimizer(cfg, model.parameters())
    diffusion = GaussianDiffusion(int(cfg.diffusion_steps)).to(device)
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    norm = np.load(repo / "data/train/norm.npy")
    minimum = torch.as_tensor(norm[0], device=device, dtype=torch.float32)
    maximum = torch.as_tensor(norm[1], device=device, dtype=torch.float32)
    object_minimum = torch.as_tensor(norm[2], device=device, dtype=torch.float32)
    object_maximum = torch.as_tensor(norm[3], device=device, dtype=torch.float32)

    profiler = Profiler(device)
    field = model.module.network.sparse_relation_field
    original_builder = sparse_relation_module.build_sparse_relation_geometry

    def timed_builder(*builder_args, **builder_kwargs):
        end = profiler.pair("gpu_relation_geometry")
        result = original_builder(*builder_args, **builder_kwargs)
        profiler.finish(end)
        if not profiler.relation_shapes:
            encoded_shape = [
                int(builder_args[0].shape[0]), 4, 3, 100, 128,
            ]
            profiler.relation_shapes = {
                "current": list(builder_args[0].shape),
                "rest_object_points": list(builder_args[1].shape),
                "surface": list(result["surface"].shape),
                "features": list(result["features"].shape),
                "encoded_points": encoded_shape,
                "pooled_blocks": [encoded_shape[0], 4, 3, 256],
                "projection_input": [encoded_shape[0], 4, 768],
                "relation_vectors": [encoded_shape[0], 4, 512],
                "routed_relation": [encoded_shape[0], 16, 512],
            }
            profiler.relation_runtime = {
                "input_device": str(builder_args[0].device),
                "input_dtype": str(builder_args[0].dtype),
                "output_devices": {
                    key: str(value.device) for key, value in result.items()
                },
                "all_cuda": (
                    builder_args[0].device.type == "cuda"
                    and all(value.device.type == "cuda" for value in result.values())
                ),
            }
        return result

    sparse_relation_module.build_sparse_relation_geometry = timed_builder
    relation_hook = {}

    def field_pre_hook(_module, _inputs):
        relation_hook["end"] = profiler.pair("gpu_relation_module")

    def field_post_hook(_module, _inputs, _output):
        profiler.finish(relation_hook.pop("end"))

    handles = [
        field.register_forward_pre_hook(field_pre_hook),
        field.register_forward_hook(field_post_hook),
    ]
    sampler.set_epoch(0)
    loader_state = {
        "epoch": 0, "sampler": sampler, "loader": loader,
        "iterator": iter(loader),
    }
    optimizer.zero_grad(set_to_none=True)
    contention_before = _gpu_contention() if rank == 0 else None
    monitor = UtilizationMonitor() if rank == 0 else None
    loss_finite = True
    gradient_finite = True
    gradient_audits = {}
    measured_losses = {key: [] for key in LOSS_KEYS}
    measured_started = None

    try:
        for update in range(TOTAL_UPDATES):
            if update == WARMUP_UPDATES:
                torch.cuda.synchronize(device)
                dist.barrier()
                torch.cuda.reset_peak_memory_stats(device)
                profiler.active = True
                if monitor is not None:
                    monitor.start()
                measured_started = time.perf_counter()
            raw_batch, loader_wait = _next_batch(loader_state)
            if "local_object_bps" in raw_batch:
                raise RuntimeError("D2-AE benchmark received CPU dynamic local geometry")
            if profiler.active:
                profiler.loader_wait.append(loader_wait)
            end = profiler.pair("h2d")
            batch = _move_batch(raw_batch, device)
            profiler.finish(end)

            comm_state.begin(profiler.active)
            end = profiler.pair("forward_and_loss")
            losses = _forward_losses(
                model, diffusion, batch, parents, minimum, maximum,
                object_minimum, object_maximum, cfg,
            )
            profiler.finish(end)
            loss = losses["total"]
            if not bool(torch.isfinite(loss)):
                loss_finite = False
                raise FloatingPointError(f"non-finite benchmark loss at update {update}")
            end = profiler.pair("backward")
            loss.backward()
            profiler.finish(end)
            comm_state.end()

            end = profiler.pair("gradient_validation")
            key_gradient = model.module.network.motion_input.weight.grad
            local_nonfinite = torch.tensor(
                int(any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                )),
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(local_nonfinite, op=dist.ReduceOp.MAX)
            if int(local_nonfinite.item()) or key_gradient is None:
                gradient_finite = False
                raise FloatingPointError(f"non-finite benchmark gradient at update {update}")
            gradient_norm = _gradient_l2_norm(model.parameters())
            if not bool(torch.isfinite(gradient_norm)) or not bool(torch.any(key_gradient != 0)):
                gradient_finite = False
                raise FloatingPointError(f"invalid benchmark gradient at update {update}")
            profiler.finish(end)
            if update == 0:
                gradient_audits["initial_zero_gate_alpha_gradient"] = (
                    _d2ae_gradient_audit(model.module, require_relation_paths=False)
                )
            elif update == 1:
                gradient_audits["activated_relation_gradients"] = (
                    _d2ae_gradient_audit(model.module, require_relation_paths=True)
                )
            end = profiler.pair("optimizer")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            profiler.finish(end)
            if profiler.active:
                for key in LOSS_KEYS:
                    measured_losses[key].append(float(losses[key].detach()))

        torch.cuda.synchronize(device)
        dist.barrier()
        measured_wall_local = time.perf_counter() - measured_started
    finally:
        profiler.active = False
        if monitor is not None:
            monitor.stop()
        sparse_relation_module.build_sparse_relation_geometry = original_builder
        for handle in handles:
            handle.remove()

    wall = _aggregate_rank_values(measured_wall_local, device)
    throughput = MEASURED_WINDOWS / wall["maximum"]
    eta_hours = FORMAL_WINDOWS / throughput / 3600.0
    total_memory = torch.cuda.get_device_properties(device).total_memory
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    headroom = total_memory - peak_reserved
    required_headroom = max(MINIMUM_HEADROOM_BYTES, int(0.10 * total_memory))
    headroom_min = torch.tensor(headroom, device=device, dtype=torch.int64)
    dist.all_reduce(headroom_min, op=dist.ReduceOp.MIN)
    local_timing = profiler.summaries()
    loader_aggregate = _aggregate_rank_values(sum(profiler.loader_wait), device)
    cuda_aggregate = {
        name: _aggregate_rank_values(value["sum"], device)
        for name, value in local_timing.items()
    }
    ddp_bucket_seconds = [
        record["wall_seconds"] for step in comm_state.steps for record in step
    ]
    ddp_bucket_aggregate = _aggregate_rank_values(sum(ddp_bucket_seconds), device)
    rank_result = {
        "schema_version": 1,
        "rank": rank,
        "local_rank": local_rank,
        "identity": identity,
        "initial_model_state_sha256": initial_model_sha256,
        "terminal_model_state_sha256": _state_dict_sha256(
            model.module.state_dict()
        ),
        "measured_wall_seconds": measured_wall_local,
        "loader_wait": _summary(profiler.loader_wait),
        "cuda_timing": local_timing,
        "ddp_bucket_wall": _summary(ddp_bucket_seconds),
        "ddp_bucket_steps": comm_state.steps,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "device_total_bytes": total_memory,
        "headroom_bytes": headroom,
        "losses": {
            key: _summary(values) for key, values in measured_losses.items()
        },
        "loss_finite": loss_finite,
        "gradient_finite": gradient_finite,
        "gradient_audits": gradient_audits,
        "relation_intermediate_shapes": profiler.relation_shapes,
        "relation_runtime": profiler.relation_runtime,
        "relation_gpu_only": profiler.relation_runtime.get("all_cuda") is True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / f"rank{rank}_metrics.json", rank_result)
    dist.barrier()

    if rank == 0:
        contention_after = _gpu_contention()
        utilization = monitor.summary()
        external_contention_before = _external_compute_processes(
            contention_before, rank_pids,
        )
        external_contention_after = _external_compute_processes(
            contention_after, rank_pids,
        )
        contention_pass = not (
            external_contention_before or external_contention_after
        )
        rank_metrics = [
            json.loads(
                (output_dir / f"rank{index}_metrics.json").read_text(
                    encoding="utf-8",
                )
            )
            for index in range(WORLD_SIZE)
        ]
        expected_shapes = {
            "current": [MICRO_BATCH_PER_GPU, 16, 232],
            "rest_object_points": [MICRO_BATCH_PER_GPU, 100, 3],
            "surface": [MICRO_BATCH_PER_GPU, 4, 100, 3],
            "features": [MICRO_BATCH_PER_GPU, 4, 3, 100, 4],
            "encoded_points": [MICRO_BATCH_PER_GPU, 4, 3, 100, 128],
            "pooled_blocks": [MICRO_BATCH_PER_GPU, 4, 3, 256],
            "projection_input": [MICRO_BATCH_PER_GPU, 4, 768],
            "relation_vectors": [MICRO_BATCH_PER_GPU, 4, 512],
            "routed_relation": [MICRO_BATCH_PER_GPU, 16, 512],
        }
        all_rank_contract_pass = all(
            record.get("relation_gpu_only") is True
            and record.get("relation_intermediate_shapes") == expected_shapes
            and record.get("loss_finite") is True
            and record.get("gradient_finite") is True
            for record in rank_metrics
        )
        headroom_pass = int(headroom_min.item()) >= required_headroom
        performance_pass = (
            throughput >= MINIMUM_THROUGHPUT
            and eta_hours <= MAXIMUM_ETA_HOURS
            and headroom_pass
            and loss_finite
            and gradient_finite
            and all_rank_contract_pass
            and contention_pass
        )
        result = {
            "schema_version": 1,
            "status": "passed" if performance_pass else "failed",
            "classification": (
                "performance-gate-passed"
                if performance_pass else FAILURE_CLASSIFICATION
            ),
            "run_id": args.run_id,
            "formal_run_id": formal_run_id,
            "identity": identity,
            "formal_source_contract": formal_source_contract,
            "subphase": "1B-D2-AE0-performance-benchmark",
            "seed": 42,
            "world_size": WORLD_SIZE,
            "micro_batch_per_gpu": MICRO_BATCH_PER_GPU,
            "effective_batch_size": EFFECTIVE_BATCH,
            "warmup_updates": WARMUP_UPDATES,
            "measured_updates": MEASURED_UPDATES,
            "total_updates": TOTAL_UPDATES,
            "measured_windows": MEASURED_WINDOWS,
            "wall_seconds_across_ranks": wall,
            "throughput_windows_per_second": throughput,
            "minimum_throughput_windows_per_second": MINIMUM_THROUGHPUT,
            "throughput_fraction_of_sealed_d2x": throughput / SEALED_D2X_THROUGHPUT,
            "full_budget_eta_hours": eta_hours,
            "maximum_full_budget_eta_hours": MAXIMUM_ETA_HOURS,
            "memory_headroom_min_bytes": int(headroom_min.item()),
            "memory_headroom_required_bytes": required_headroom,
            "memory_headroom_pass": headroom_pass,
            "losses_finite": loss_finite,
            "gradients_finite": gradient_finite,
            "relation_gpu_only": all(
                record.get("relation_gpu_only") is True
                for record in rank_metrics
            ),
            "all_rank_contract_pass": all_rank_contract_pass,
            "timing_aggregate_rank_seconds": {
                "loader_wait": loader_aggregate,
                **cuda_aggregate,
                "ddp_allreduce_bucket_wall_nonadditive": ddp_bucket_aggregate,
            },
            "ddp_timing_note": (
                "bucket wall durations may overlap; backward CUDA time is the "
                "inclusive DDP critical-path measurement"
            ),
            "relation_intermediate_shapes": profiler.relation_shapes,
            "relation_runtime": profiler.relation_runtime,
            "cpu_gpu_utilization": utilization,
            "contention_before": contention_before,
            "contention_after": contention_after,
            "external_contention_before": external_contention_before,
            "external_contention_after": external_contention_after,
            "contention_pass": contention_pass,
            "resolved_config_path": str(resolved_path),
            "resolved_config_sha256": resolved_sha256,
            "optimizer": "FP32 Adam",
            "optimizer_updates": TOTAL_UPDATES,
            "checkpoint_loads": 0,
            "checkpoint_writes": 0,
            "benchmark_weights_reusable": False,
            "cpu_dynamic_geometry": False,
            "relation_build_device": "cuda",
            "cuda_timing_synchronized": True,
            "formal_training_authorized": performance_pass,
            "sweep_authorized_on_failure": False,
            "rank_metrics": [
                str(output_dir / f"rank{index}_metrics.json")
                for index in range(WORLD_SIZE)
            ],
            "rank_relation_contracts": [
                {
                    "rank": record["rank"],
                    "relation_gpu_only": record["relation_gpu_only"],
                    "relation_intermediate_shapes": record[
                        "relation_intermediate_shapes"
                    ],
                }
                for record in rank_metrics
            ],
        }
        _atomic_json(output_dir / "benchmark_summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{FAILURE_CLASSIFICATION}: {error}", file=sys.stderr, flush=True)
        raise
