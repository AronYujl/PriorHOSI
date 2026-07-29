#!/usr/bin/env python3
"""Registered 4-GPU full-micro-batch performance gate for D2-AF0."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
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
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SHA256,
    tensor_sha256,
)
from priors.models import HOI_ARCHITECTURE_D2AF, build_expert  # noqa: E402
from priors.sparse_relation import diffusion_reliability_contract_metadata  # noqa: E402
from tools.benchmark_hoi_d2ae import (  # noqa: E402
    DDPCommState,
    Profiler,
    UtilizationMonitor,
    _aggregate_rank_values,
    _ddp_hook,
    _external_compute_processes,
    _next_batch,
    _summary,
)
from tools.smoke_hoi_d2ac import _atomic_json, _gpu_contention  # noqa: E402
from tools.smoke_hoi_d2af import (  # noqa: E402
    EXPECTED_INITIAL_MODEL_SHA256,
    _atomic_text,
    _resolved_config,
    _sha256_file,
    _verify_worker,
)
from train_hoi_prior import (  # noqa: E402
    D2AF_MAXIMUM_ETA_HOURS,
    D2AF_MINIMUM_THROUGHPUT,
    LOSS_KEYS,
    _build_optimizer,
    _d2af_formal_source_contract,
    _d2ae_gradient_audit,
    _forward_losses,
    _gradient_l2_norm,
    _move_batch,
    _state_dict_sha256,
    _validate_author_update_execution_host,
    _validate_d2af_contract,
    _validate_d2af_formal_run_id,
    _validate_fk_foot_temporal_routing_mode,
)


RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-performance-benchmark"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WORLD_SIZE = 4
MICRO_BATCH_PER_GPU = 512
EFFECTIVE_BATCH = 2048
WARMUP_UPDATES = 64
MEASURED_UPDATES = 256
TOTAL_UPDATES = 320
MEASURED_WINDOWS = 524_288
FORMAL_WINDOWS = 61_440_000
SEALED_D2AE_THROUGHPUT = 3347.0419610997483
MINIMUM_THROUGHPUT = D2AF_MINIMUM_THROUGHPUT
MAXIMUM_ETA_HOURS = D2AF_MAXIMUM_ETA_HOURS
MINIMUM_HEADROOM_BYTES = 2 * 1024**3
IDLE_CONTENTION_SAMPLES = 3
IDLE_CONTENTION_SAMPLE_INTERVAL_SECONDS = 1.0
FAILURE_CLASSIFICATION = "diffusion-reliability-performance-negative-stop"


def _resolved_workload_config(
    cfg,
    *,
    repo: Path,
    run_id: str,
    expected_commit: str,
    formal_source_contract: dict,
    eligibility_contract: dict,
    output_dir: Path,
    resolved_config_output: Path,
) -> str:
    command = [
        str(Path(sys.executable).resolve()), "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=4",
        "tools/benchmark_hoi_d2af.py",
        "--repo-root", str(repo),
        "--output-dir", str(output_dir.resolve()),
        "--resolved-config-output", str(resolved_config_output.resolve()),
        "--expected-commit", expected_commit,
        "--run-id", run_id,
        "--formal-run-id", str(cfg.run_id),
        "--eligibility-path", str(
            Path(str(cfg.d2af_clean_signal_eligibility_path)).resolve()
        ),
        "--eligibility-sha256", str(cfg.d2af_clean_signal_eligibility_sha256),
    ]
    value = {
        "schema_version": 1,
        "lifecycle": "d2af_four_gpu_full_micro_batch_performance_benchmark",
        "run_id": run_id,
        "formal_run_id": str(cfg.run_id),
        "expected_git_commit": expected_commit,
        "formal_source_contract": formal_source_contract,
        "eligibility_contract": eligibility_contract,
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
            "relation_source": "current_noisy_state_only",
            "reliability": "sqrt_alpha_bar[current_timestep]",
        },
        "identity_contracts": {
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
            "expected_initial_model_state_sha256": EXPECTED_INITIAL_MODEL_SHA256,
            "sqrt_alpha_bar_sha256": SQRT_ALPHA_BAR_SHA256,
            "four_rank_schedule_hash_identity": True,
            "eligibility_sha256": eligibility_contract["sha256"],
        },
        "performance_gate": {
            "sealed_d2ae_throughput_windows_per_second": (
                SEALED_D2AE_THROUGHPUT
            ),
            "minimum_throughput_windows_per_second": MINIMUM_THROUGHPUT,
            "maximum_full_budget_eta_hours": MAXIMUM_ETA_HOURS,
            "minimum_memory_headroom": "max(2 GiB, 10% device memory)",
            "failure_classification": FAILURE_CLASSIFICATION,
            "sweep_on_failure": False,
        },
        "timing": {
            "cuda_synchronized_boundaries": True,
            "categories": [
                "loader_wait",
                "h2d",
                "gpu_relation_geometry",
                "gpu_point_encoder",
                "gpu_relation_projection",
                "gpu_relation_norm",
                "gpu_pool_route_rho_writeback_derived",
                "gpu_relation_module",
                "forward_and_loss",
                "backward",
                "ddp_allreduce_bucket_wall",
                "gradient_validation",
                "optimizer",
            ],
            "rho_timing_note": (
                "pool/route/rho/writeback is the synchronized per-call residual "
                "after subtracting geometry, point encoder, projection and "
                "relation LayerNorm from the complete field duration"
            ),
            "ddp_note": (
                "bucket wall durations may overlap; backward CUDA time is the "
                "authoritative inclusive DDP critical-path measurement"
            ),
        },
        "contention_contract": {
            "idle_samples_before": IDLE_CONTENTION_SAMPLES,
            "idle_samples_after": IDLE_CONTENTION_SAMPLES,
            "sample_interval_seconds": IDLE_CONTENTION_SAMPLE_INTERVAL_SECONDS,
            "external_compute_processes_forbidden": True,
            "pstate_recorded_but_not_independently_gating": True,
        },
        "formal_training_config_reference": OmegaConf.to_container(
            cfg, resolve=True,
        ),
    }
    resolved = OmegaConf.to_yaml(OmegaConf.create(value), resolve=True)
    if "${" in resolved:
        raise RuntimeError("D2-AF benchmark resolved config is incomplete")
    return resolved


def _contention_samples(count: int) -> list:
    records = []
    for index in range(count):
        records.append({
            "sample_index": index,
            "captured_at": datetime.now().astimezone().isoformat(),
            **_gpu_contention(),
        })
        if index + 1 < count:
            time.sleep(IDLE_CONTENTION_SAMPLE_INTERVAL_SECONDS)
    return records


def _external_from_samples(samples: list, allowed_pids: list) -> list:
    values = []
    for sample in samples:
        values.extend(_external_compute_processes(sample, allowed_pids))
    return sorted(set(values))


def _pstate_observations(samples: list) -> dict:
    per_gpu = {}
    for sample in samples:
        for gpu in sample.get("gpus", []):
            per_gpu.setdefault(str(gpu["index"]), []).append({
                "captured_at": sample["captured_at"],
                "memory_used_mib": gpu["memory_used_mib"],
                "utilization_percent": gpu["utilization_percent"],
                "pstate": gpu["pstate"],
            })
    return per_gpu


def _derived_pool_route_rho_summary(profiler: Profiler) -> dict:
    component_names = (
        "gpu_relation_geometry",
        "gpu_point_encoder",
        "gpu_relation_projection",
        "gpu_relation_norm",
    )
    complete = profiler.cuda_pairs.get("gpu_relation_module", [])
    components = [
        profiler.cuda_pairs.get(name, []) for name in component_names
    ]
    if not complete or any(len(value) != len(complete) for value in components):
        return _summary([])
    residuals = []
    for index, (start, end) in enumerate(complete):
        complete_seconds = start.elapsed_time(end) / 1000.0
        component_seconds = sum(
            pairs[index][0].elapsed_time(pairs[index][1]) / 1000.0
            for pairs in components
        )
        residuals.append(max(0.0, complete_seconds - component_seconds))
    return _summary(residuals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolved-config-output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-run-id", required=True)
    parser.add_argument("--eligibility-path", type=Path, required=True)
    parser.add_argument("--eligibility-sha256", required=True)
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    run_id_match = RUN_ID_RE.fullmatch(args.run_id)
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if run_id_match is None or run_id_match.group("date") != actual_date:
        raise ValueError(
            "D2-AF benchmark run id must use the locked stem and actual date"
        )
    formal_run_id_contract = _validate_d2af_formal_run_id(args.formal_run_id)
    if formal_run_id_contract["date"] != run_id_match.group("date"):
        raise ValueError(
            "D2-AF benchmark and intended formal run ids must use the same date"
        )
    if not args.eligibility_path.is_absolute():
        raise ValueError("D2-AF benchmark eligibility path must be absolute")
    if SHA256_RE.fullmatch(str(args.eligibility_sha256)) is None:
        raise ValueError("D2-AF benchmark eligibility SHA-256 is malformed")
    formal_run_id = args.formal_run_id

    cfg = _resolved_config(repo, formal_run_id)
    cfg.d2af_clean_signal_eligibility_path = str(
        args.eligibility_path.resolve()
    )
    cfg.d2af_clean_signal_eligibility_sha256 = str(args.eligibility_sha256)
    OmegaConf.resolve(cfg)
    _validate_fk_foot_temporal_routing_mode(cfg)
    lifecycle_contract = _validate_d2af_contract(
        cfg,
        WORLD_SIZE,
        require_eligibility_gate=True,
        require_performance_gate=False,
    )
    eligibility_contract = lifecycle_contract["eligibility_gate"]
    identity = _verify_worker(repo, args.expected_commit)
    formal_source_contract = _d2af_formal_source_contract(repo)
    if eligibility_contract["formal_source_contract"] != formal_source_contract:
        raise RuntimeError(
            "D2-AF benchmark and eligibility source contracts differ"
        )
    _validate_author_update_execution_host(cfg)
    resolved_yaml = _resolved_workload_config(
        cfg,
        repo=repo,
        run_id=args.run_id,
        expected_commit=args.expected_commit,
        formal_source_contract=formal_source_contract,
        eligibility_contract=eligibility_contract,
        output_dir=output_dir,
        resolved_config_output=args.resolved_config_output,
    )
    if args.resolve_only:
        _atomic_text(args.resolved_config_output, resolved_yaml)
        print(json.dumps({
            "schema_version": 1,
            "status": "resolved-config-archived",
            "run_id": args.run_id,
            "formal_run_id": formal_run_id,
            "eligibility_path": eligibility_contract["path"],
            "eligibility_sha256": eligibility_contract["sha256"],
            "formal_source_contract": formal_source_contract,
            "resolved_config_path": str(args.resolved_config_output.resolve()),
            "resolved_config_sha256": _sha256_file(
                args.resolved_config_output.resolve()
            ),
            "gpu_workload_started": False,
        }, indent=2, sort_keys=True), flush=True)
        return 0
    resolved_path = args.resolved_config_output.resolve()
    if (
        not resolved_path.is_file()
        or resolved_path.read_text(encoding="utf-8") != resolved_yaml
    ):
        raise RuntimeError("D2-AF benchmark differs from its archived resolved config")
    resolved_sha256 = _sha256_file(resolved_path)

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if (
        world_size != WORLD_SIZE
        or rank not in range(WORLD_SIZE)
        or local_rank not in range(WORLD_SIZE)
    ):
        raise RuntimeError("D2-AF benchmark requires torchrun with exactly four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("D2-AF benchmark requires exactly four visible CUDA devices")
    for index in range(WORLD_SIZE):
        if "RTX 3090" not in torch.cuda.get_device_name(index):
            raise RuntimeError("D2-AF benchmark requires 4x RTX 3090")
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
        str(repo),
        "hoi",
        partition="train",
        split_manifest=str(cfg.split_manifest),
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=WORLD_SIZE,
        rank=rank,
        shuffle=True,
        seed=42,
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
        "hoi",
        init_checkpoint=None,
        dim_model=512,
        num_heads=16,
        num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AF,
    ).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
    )
    initial_model_sha256 = _state_dict_sha256(model.module.state_dict())
    field = model.module.network.sparse_relation_field
    schedule_sha256 = tensor_sha256(field.sqrt_alpha_bar)
    diffusion = GaussianDiffusion(int(cfg.diffusion_steps)).to(device)
    diffusion_schedule_sha256 = tensor_sha256(diffusion.sqrt_alpha_bar)
    if (
        initial_model_sha256 != EXPECTED_INITIAL_MODEL_SHA256
        or schedule_sha256 != SQRT_ALPHA_BAR_SHA256
        or diffusion_schedule_sha256 != SQRT_ALPHA_BAR_SHA256
    ):
        raise RuntimeError("D2-AF benchmark initialization/schedule identity mismatch")

    comm_state = DDPCommState(WORLD_SIZE)
    model.register_comm_hook(comm_state, _ddp_hook)
    optimizer = _build_optimizer(cfg, model.parameters())
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    norm = np.load(repo / "data/train/norm.npy")
    minimum = torch.as_tensor(norm[0], device=device, dtype=torch.float32)
    maximum = torch.as_tensor(norm[1], device=device, dtype=torch.float32)
    object_minimum = torch.as_tensor(norm[2], device=device, dtype=torch.float32)
    object_maximum = torch.as_tensor(norm[3], device=device, dtype=torch.float32)

    profiler = Profiler(device)
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
                "rho_indices": [encoded_shape[0]],
                "rho_broadcast": [encoded_shape[0], 1, 1],
                "raw_writeback": [encoded_shape[0], 16, 512],
                "attenuated_writeback": [encoded_shape[0], 16, 512],
            }
            profiler.relation_runtime = {
                "input_device": str(builder_args[0].device),
                "input_dtype": str(builder_args[0].dtype),
                "output_devices": {
                    key: str(value.device) for key, value in result.items()
                },
                "schedule_buffer_device": str(field.sqrt_alpha_bar.device),
                "schedule_buffer_dtype": str(field.sqrt_alpha_bar.dtype),
                "sqrt_alpha_bar_sha256": schedule_sha256,
                "all_cuda": (
                    builder_args[0].device.type == "cuda"
                    and field.sqrt_alpha_bar.device.type == "cuda"
                    and all(value.device.type == "cuda" for value in result.values())
                ),
            }
        return result

    sparse_relation_module.build_sparse_relation_geometry = timed_builder
    hook_ends = {}

    def pre_hook(name):
        def hook(_module, _inputs):
            hook_ends[name] = profiler.pair(name)
        return hook

    def post_hook(name):
        def hook(_module, _inputs, _output):
            profiler.finish(hook_ends.pop(name))
        return hook

    handles = []
    for module, name in (
        (field, "gpu_relation_module"),
        (field.point_encoder, "gpu_point_encoder"),
        (field.projection, "gpu_relation_projection"),
        (field.relation_norm, "gpu_relation_norm"),
    ):
        handles.extend([
            module.register_forward_pre_hook(pre_hook(name)),
            module.register_forward_hook(post_hook(name)),
        ])

    sampler.set_epoch(0)
    loader_state = {
        "epoch": 0,
        "sampler": sampler,
        "loader": loader,
        "iterator": iter(loader),
    }
    optimizer.zero_grad(set_to_none=True)
    contention_before_samples = (
        _contention_samples(IDLE_CONTENTION_SAMPLES) if rank == 0 else None
    )
    monitor = UtilizationMonitor() if rank == 0 else None
    loss_finite = True
    gradient_finite = True
    gradient_audits = {}
    reliability_runtime_sample = None
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
                raise RuntimeError(
                    "D2-AF benchmark received CPU dynamic local geometry"
                )
            if profiler.active:
                profiler.loader_wait.append(loader_wait)
            end = profiler.pair("h2d")
            batch = _move_batch(raw_batch, device)
            profiler.finish(end)

            comm_state.begin(profiler.active)
            if update == 1:
                field.set_capture(True)
            end = profiler.pair("forward_and_loss")
            losses = _forward_losses(
                model,
                diffusion,
                batch,
                parents,
                minimum,
                maximum,
                object_minimum,
                object_maximum,
                cfg,
            )
            profiler.finish(end)
            if update == 1:
                snapshot = field.snapshot()
                field.set_capture(False)
                if snapshot is None or not all(
                    name in snapshot
                    for name in (
                        "rho",
                        "raw_writeback_norm",
                        "attenuated_writeback_norm",
                        "relation_norm",
                        "gate",
                    )
                ):
                    raise RuntimeError(
                        "D2-AF benchmark failed to capture rho/writeback values"
                    )
                reliability_runtime_sample = {
                    "capture_update": 1,
                    "capture_in_warmup": True,
                    "rho": snapshot["rho"].tolist(),
                    "rho_sha256": tensor_sha256(snapshot["rho"]),
                    "rho_minimum": float(snapshot["rho"].amin().item()),
                    "rho_maximum": float(snapshot["rho"].amax().item()),
                    "rho_mean": float(snapshot["rho"].mean().item()),
                    "raw_writeback_norm_by_frame": (
                        snapshot["raw_writeback_norm"].tolist()
                    ),
                    "attenuated_writeback_norm_by_frame": (
                        snapshot["attenuated_writeback_norm"].tolist()
                    ),
                    "relation_norm_by_anchor": (
                        snapshot["relation_norm"].tolist()
                    ),
                    "gate": float(snapshot["gate"].item()),
                    "finite": all(
                        bool(torch.isfinite(snapshot[name]).all())
                        for name in (
                            "rho",
                            "raw_writeback_norm",
                            "attenuated_writeback_norm",
                            "relation_norm",
                            "gate",
                        )
                    ),
                }
            loss = losses["total"]
            if not bool(torch.isfinite(loss)):
                loss_finite = False
                raise FloatingPointError(
                    f"non-finite D2-AF benchmark loss at update {update}"
                )
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
                raise FloatingPointError(
                    f"non-finite D2-AF benchmark gradient at update {update}"
                )
            gradient_norm = _gradient_l2_norm(model.parameters())
            if (
                not bool(torch.isfinite(gradient_norm))
                or not bool(torch.any(key_gradient != 0))
            ):
                gradient_finite = False
                raise FloatingPointError(
                    f"invalid D2-AF benchmark gradient at update {update}"
                )
            profiler.finish(end)
            if update == 0:
                gradient_audits["initial_zero_gate_alpha_gradient"] = (
                    _d2ae_gradient_audit(
                        model.module,
                        require_relation_paths=False,
                    )
                )
            elif update == 1:
                gradient_audits["activated_relation_gradients"] = (
                    _d2ae_gradient_audit(
                        model.module,
                        require_relation_paths=True,
                    )
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
        field.set_capture(False)
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
    required_headroom = max(
        MINIMUM_HEADROOM_BYTES,
        int(0.10 * total_memory),
    )
    headroom_min = torch.tensor(headroom, device=device, dtype=torch.int64)
    dist.all_reduce(headroom_min, op=dist.ReduceOp.MIN)
    local_timing = profiler.summaries()
    local_timing["gpu_pool_route_rho_writeback_derived"] = (
        _derived_pool_route_rho_summary(profiler)
    )
    loader_aggregate = _aggregate_rank_values(
        sum(profiler.loader_wait),
        device,
    )
    cuda_aggregate = {
        name: _aggregate_rank_values(value["sum"], device)
        for name, value in local_timing.items()
    }
    ddp_bucket_seconds = [
        record["wall_seconds"]
        for step in comm_state.steps
        for record in step
    ]
    ddp_bucket_aggregate = _aggregate_rank_values(
        sum(ddp_bucket_seconds),
        device,
    )
    terminal_schedule_sha256 = tensor_sha256(field.sqrt_alpha_bar)
    terminal_diffusion_schedule_sha256 = tensor_sha256(
        diffusion.sqrt_alpha_bar
    )
    if (
        terminal_schedule_sha256 != SQRT_ALPHA_BAR_SHA256
        or terminal_diffusion_schedule_sha256 != SQRT_ALPHA_BAR_SHA256
    ):
        raise RuntimeError("D2-AF benchmark schedule changed during optimization")
    rank_result = {
        "schema_version": 1,
        "rank": rank,
        "local_rank": local_rank,
        "identity": identity,
        "formal_source_contract": formal_source_contract,
        "eligibility_sha256": eligibility_contract["sha256"],
        "initial_model_state_sha256": initial_model_sha256,
        "terminal_model_state_sha256": _state_dict_sha256(
            model.module.state_dict()
        ),
        "sqrt_alpha_bar_sha256": terminal_schedule_sha256,
        "diffusion_sqrt_alpha_bar_sha256": (
            terminal_diffusion_schedule_sha256
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
        "reliability_runtime_sample": reliability_runtime_sample,
        "relation_gpu_only": profiler.relation_runtime.get("all_cuda") is True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / f"rank{rank}_metrics.json", rank_result)
    dist.barrier()

    if rank == 0:
        contention_after_samples = _contention_samples(
            IDLE_CONTENTION_SAMPLES
        )
        utilization = monitor.summary()
        external_contention_before = _external_from_samples(
            contention_before_samples,
            rank_pids,
        )
        external_contention_after = _external_from_samples(
            contention_after_samples,
            rank_pids,
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
            "rho_indices": [MICRO_BATCH_PER_GPU],
            "rho_broadcast": [MICRO_BATCH_PER_GPU, 1, 1],
            "raw_writeback": [MICRO_BATCH_PER_GPU, 16, 512],
            "attenuated_writeback": [MICRO_BATCH_PER_GPU, 16, 512],
        }
        schedule_hashes = [
            record.get("sqrt_alpha_bar_sha256")
            for record in rank_metrics
        ]
        diffusion_schedule_hashes = [
            record.get("diffusion_sqrt_alpha_bar_sha256")
            for record in rank_metrics
        ]
        initial_model_hashes = [
            record.get("initial_model_state_sha256")
            for record in rank_metrics
        ]
        four_rank_schedule_hash_pass = (
            schedule_hashes
            == [SQRT_ALPHA_BAR_SHA256] * WORLD_SIZE
            and diffusion_schedule_hashes
            == [SQRT_ALPHA_BAR_SHA256] * WORLD_SIZE
        )
        four_rank_initial_model_hash_pass = (
            initial_model_hashes
            == [EXPECTED_INITIAL_MODEL_SHA256] * WORLD_SIZE
        )
        all_rank_contract_pass = all(
            record.get("relation_gpu_only") is True
            and record.get("relation_intermediate_shapes") == expected_shapes
            and record.get("loss_finite") is True
            and record.get("gradient_finite") is True
            and record.get("reliability_runtime_sample", {}).get("finite")
            is True
            and len(record.get("reliability_runtime_sample", {}).get("rho", []))
            == MICRO_BATCH_PER_GPU
            and record.get("eligibility_sha256")
            == eligibility_contract["sha256"]
            and record.get("formal_source_contract")
            == formal_source_contract
            and record.get("cuda_timing", {}).get(
                "gpu_pool_route_rho_writeback_derived", {}
            ).get("count") == MEASURED_UPDATES
            for record in rank_metrics
        ) and four_rank_schedule_hash_pass and four_rank_initial_model_hash_pass
        headroom_pass = int(headroom_min.item()) >= required_headroom
        losses_finite = all(
            record.get("loss_finite") is True for record in rank_metrics
        )
        gradients_finite = all(
            record.get("gradient_finite") is True for record in rank_metrics
        )
        performance_pass = (
            throughput >= MINIMUM_THROUGHPUT
            and eta_hours <= MAXIMUM_ETA_HOURS
            and headroom_pass
            and losses_finite
            and gradients_finite
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
            "eligibility_path": eligibility_contract["path"],
            "eligibility_sha256": eligibility_contract["sha256"],
            "eligibility_run_id": eligibility_contract["run_id"],
            "subphase": "1B-D2-AF0-performance-benchmark",
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
            "sealed_d2ae_throughput_windows_per_second": (
                SEALED_D2AE_THROUGHPUT
            ),
            "minimum_throughput_windows_per_second": MINIMUM_THROUGHPUT,
            "throughput_fraction_of_sealed_d2ae": (
                throughput / SEALED_D2AE_THROUGHPUT
            ),
            "full_budget_eta_hours": eta_hours,
            "maximum_full_budget_eta_hours": MAXIMUM_ETA_HOURS,
            "memory_headroom_min_bytes": int(headroom_min.item()),
            "memory_headroom_required_bytes": required_headroom,
            "memory_headroom_pass": headroom_pass,
            "rank_memory": [
                {
                    "rank": record["rank"],
                    "peak_allocated_bytes": record["peak_allocated_bytes"],
                    "peak_reserved_bytes": record["peak_reserved_bytes"],
                    "device_total_bytes": record["device_total_bytes"],
                    "headroom_bytes": record["headroom_bytes"],
                }
                for record in rank_metrics
            ],
            "losses_finite": losses_finite,
            "gradients_finite": gradients_finite,
            "relation_gpu_only": all(
                record.get("relation_gpu_only") is True
                for record in rank_metrics
            ),
            "all_rank_contract_pass": all_rank_contract_pass,
            "sqrt_alpha_bar_sha256": SQRT_ALPHA_BAR_SHA256,
            "four_rank_schedule_hashes": schedule_hashes,
            "four_rank_diffusion_schedule_hashes": diffusion_schedule_hashes,
            "four_rank_schedule_hash_pass": four_rank_schedule_hash_pass,
            "four_rank_initial_model_state_hashes": initial_model_hashes,
            "four_rank_initial_model_hash_pass": (
                four_rank_initial_model_hash_pass
            ),
            "timing_aggregate_rank_seconds": {
                "loader_wait": loader_aggregate,
                **cuda_aggregate,
                "ddp_allreduce_bucket_wall_nonadditive": ddp_bucket_aggregate,
            },
            "rho_timing_note": (
                "gpu_pool_route_rho_writeback_derived is the synchronized "
                "complete-field residual after subtracting geometry, point "
                "encoder, projection and relation LayerNorm"
            ),
            "ddp_timing_note": (
                "bucket wall durations may overlap; backward CUDA time is the "
                "inclusive DDP critical-path measurement"
            ),
            "relation_intermediate_shapes": profiler.relation_shapes,
            "relation_runtime": profiler.relation_runtime,
            "rank_reliability_runtime_samples": [
                {
                    "rank": record["rank"],
                    **record["reliability_runtime_sample"],
                }
                for record in rank_metrics
            ],
            "cpu_gpu_utilization": utilization,
            "contention_before_samples": contention_before_samples,
            "contention_after_samples": contention_after_samples,
            "external_contention_before": external_contention_before,
            "external_contention_after": external_contention_after,
            "contention_pass": contention_pass,
            "pstate_observations": {
                "before": _pstate_observations(contention_before_samples),
                "after": _pstate_observations(contention_after_samples),
                "independently_gating": False,
            },
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
                    "sqrt_alpha_bar_sha256": record[
                        "sqrt_alpha_bar_sha256"
                    ],
                    "initial_model_state_sha256": record[
                        "initial_model_state_sha256"
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
