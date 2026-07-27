#!/usr/bin/env python3
"""Run the fixed D2-AD0 internal causal interaction diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

import utils as author_utils  # noqa: E402
from datasets.utils import get_smpl_parents  # noqa: E402
from priors.d2ad import (  # noqa: E402
    BPS_YUP_TENSOR_SHA256,
    DEFAULT_QUERY_WORKERS,
    OBJECT_MAPPING_SHA256,
    REST_MESH_MANIFEST_SHA256,
    D2ADPriorWindowDataset,
    LocalObjectBPSBuilder,
)
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.interaction_adapter import ASSIGNMENT_SHA256, BPS_SHA256  # noqa: E402
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_D2AD,
    load_trained_hoi_prior,
)
from priors.window_codec import project_to_so3  # noqa: E402
from tools import run_hoi_d2ac_internal as base  # noqa: E402


SUBPHASE = "1B-D2-AD0-internal"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ad-local-frame-interaction-adapter-internal"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
TRAINING_RUN_ID = "p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728"
EXPECTED_PYTHON = "/home/yujinlun/data/envs/infbagel/bin/python"
FAILURE_CLASSIFICATION = "local-frame-interaction-adapter-contract-failure-stop"
CLASSIFICATION_MAP = {
    "interaction-adapter-contract-failure-stop":
        "local-frame-interaction-adapter-contract-failure-stop",
    "interaction-adapter-unused-optimization-negative-stop":
        "local-frame-interaction-adapter-unused-optimization-negative-stop",
    "interaction-adapter-locality-negative-stop":
        "local-frame-interaction-adapter-locality-negative-stop",
    "interaction-adapter-internal-positive-continue":
        "local-frame-interaction-adapter-internal-positive-continue",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def sampler_seed_label(chunk_index: int, window_index: int) -> str:
    if chunk_index < 0 or window_index not in range(base.WINDOWS_PER_SEQUENCE):
        raise ValueError("invalid D2-AD sampler seed coordinates")
    return f"D2:d2ad-shared:chunk:{chunk_index}:window:{window_index}"


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:15], 16)


def checkpoint_contract(path: Path, expected_sha256: str) -> Dict[str, object]:
    actual = sha256_file(path)
    expected_name = f"{TRAINING_RUN_ID}_windows061440000.pth"
    if actual != expected_sha256:
        raise ValueError(f"D2-AD final checkpoint hash mismatch: {actual}")
    if path.name != expected_name:
        raise ValueError("D2-AD internal requires the fixed final checkpoint basename")
    checkpoint = torch.load(path, map_location="cpu")
    initialization = checkpoint.get("weight_initialization", {})
    adapter = checkpoint.get("interaction_adapter_contract", {})
    resume = checkpoint.get("resume_contract", {})
    checks = {
        "checkpoint_type": checkpoint.get("checkpoint_type") == "hoi_prior_phase1b",
        "expert": checkpoint.get("expert") == "hoi",
        "run_id": checkpoint.get("run_id") == TRAINING_RUN_ID,
        "seed": checkpoint.get("seed") == 42,
        "processed_windows": checkpoint.get("processed_windows") == 61_440_000,
        "processed_frames": checkpoint.get("processed_frames") == 983_040_000,
        "optimizer_updates": checkpoint.get("optimizer_updates") == 30_000,
        "world_size": checkpoint.get("world_size") == 4,
        "effective_batch_size": checkpoint.get("effective_batch_size") == 2048,
        "architecture_variant": (
            checkpoint.get("architecture_variant") == HOI_ARCHITECTURE_D2AD
            and checkpoint.get("model_config", {}).get("architecture_variant")
            == HOI_ARCHITECTURE_D2AD
        ),
        "adapter_provenance": (
            adapter.get("bps_sha256") == BPS_SHA256
            and adapter.get("assignment_sha256") == ASSIGNMENT_SHA256
            and adapter.get("adapter_parameters") == 349_697
            and adapter.get("alpha_initial") == 0.0
            and adapter.get("basis_coordinate_system")
            == "human_window_local_y_up"
            and adapter.get("basis_yup_tensor_sha256")
            == BPS_YUP_TENSOR_SHA256
            and adapter.get("rest_mesh_manifest_sha256")
            == REST_MESH_MANIFEST_SHA256
            and adapter.get("object_mapping_sha256")
            == OBJECT_MAPPING_SHA256
            and adapter.get("query_backend")
            == "scipy.spatial.cKDTree.query"
            and adapter.get("query_parameters")
            == {"k": 1, "eps": 0.0, "p": 2}
            and adapter.get("query_workers") == DEFAULT_QUERY_WORKERS
            and adapter.get("full_rest_mesh") is True
            and adapter.get("mesh_subsample") is False
            and adapter.get("stored_per_window_local_bps") is False
        ),
        "data_contract": (
            checkpoint.get("data_contract_sha256")
            == base.EXPECTED_DATA_CONTRACT_SHA256
        ),
        "split": checkpoint.get("split_sha256") == base.EXPECTED_SPLIT_SHA256,
        "random_initialization": (
            checkpoint.get("initialization") == "random"
            and initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("restored_components") == []
        ),
        "no_ema": checkpoint.get("ema_models") == {},
        "online_model": isinstance(checkpoint.get("model"), dict),
        "d2x_routing": resume.get("fk_foot_temporal_routing") is True,
        "d2ab_disabled": resume.get("d2ab_predicted_support_no_slip") is False,
        "d2ad_enabled": (
            resume.get("d2ad_local_frame_interaction_adapter") is True
            and resume.get("architecture_variant") == HOI_ARCHITECTURE_D2AD
        ),
        "query_workers_locked": (
            resume.get("local_bps_query_workers") == DEFAULT_QUERY_WORKERS
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AD final checkpoint contract mismatch: {failed}")
    return {
        "path": str(path),
        "sha256": actual,
        "git_commit": checkpoint.get("git_commit"),
        "checks": checks,
        "initial_model_state_sha256": initialization.get(
            "initial_model_state_sha256"
        ),
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    value = base.resolved_config(args)
    value["subphase"] = SUBPHASE
    value["mode"] = "local-frame-interaction-adapter-internal-causal-diagnostic"
    value["training_run_id"] = TRAINING_RUN_ID
    value["target_checkpoint"]["run_id"] = TRAINING_RUN_ID
    value["sampling"].update({
        "adapter_local_bps": (
            "recomputed_current-human-local full-mesh BPS for every generated "
            "autoregressive window"
        ),
        "adapter_local_bps_query_workers": DEFAULT_QUERY_WORKERS,
        "adapter_local_bps_future_gt": False,
        "adapter_local_bps_stored_condition": False,
        "global_bps_path_unchanged": True,
    })
    value["assets"].update({
        "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
        "rest_mesh_manifest_sha256": REST_MESH_MANIFEST_SHA256,
        "object_mapping_sha256": OBJECT_MAPPING_SHA256,
    })
    value["local_geometry"] = {
        "coordinate_system": "human_window_local_y_up",
        "query_backend": "scipy.spatial.cKDTree.query",
        "query_parameters": {"k": 1, "eps": 0.0, "p": 2},
        "query_workers": DEFAULT_QUERY_WORKERS,
        "full_rest_mesh": True,
        "mesh_subsample": False,
        "stored_per_window_local_bps": False,
        "global_bps_token_preserved": True,
    }
    return value


@torch.no_grad()
def rollout_chunk(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    dataset: D2ADPriorWindowDataset,
    triples: Sequence[Sequence[int]],
    device: torch.device,
    rest_vertices: Mapping[str, torch.Tensor],
    parents_24: torch.Tensor,
    local_bps_builder: LocalObjectBPSBuilder,
    *,
    chunk_index: int,
) -> Dict[str, object]:
    positions_by_step = [
        [triple[step] for triple in triples]
        for step in range(base.WINDOWS_PER_SEQUENCE)
    ]
    items_by_step = [
        [dataset[position] for position in positions]
        for positions in positions_by_step
    ]
    names = [
        base._sequence_name(dataset, position)
        for position in positions_by_step[0]
    ]
    object_indices = local_bps_builder.object_indices_from_names(names)
    first_items = items_by_step[0]
    frame = base.stack_frames(first_items, device)
    fixed = torch.stack([item["x"][:2] for item in first_items]).to(device)
    decoded_steps = []
    fk_steps = []
    noise_streams = []
    attention_by_window: List[List[Dict[str, object]]] = []
    local_bps_by_window = []
    history_max_abs = 0.0
    for window_index in range(base.WINDOWS_PER_SEQUENCE):
        items = items_by_step[window_index]
        gt_frame = base.stack_frames(items, device)
        pelvis_global, object_global = base.global_goals(
            dataset, items, gt_frame, device,
        )
        goals = torch.zeros(len(triples), 9, device=device)
        goals[:, :3] = dataset.codec.pelvis_goal(pelvis_global, frame)
        goals[:, 6:9] = dataset.codec.object_goal(object_global, frame)
        text = torch.stack([item["text_embedding"] for item in items]).to(device)
        global_bps = base.current_bps(
            dataset, frame.object_reference, names, rest_vertices,
        )
        local_started = time.perf_counter()
        local_bps = local_bps_builder.build(
            frame.world_to_local,
            frame.object_reference,
            object_indices,
        ).to(device=device, dtype=torch.float32)
        local_seconds = time.perf_counter() - local_started
        progress = normalize_progress(
            torch.stack([item["progress"] for item in items]).to(device)
        )
        rest_offsets = torch.stack([
            item["rest_human_offsets"] for item in items
        ]).to(device)
        label = sampler_seed_label(chunk_index, window_index)
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(label))
        initial_state = base.sha256_tensor_state(generator.get_state())
        capture = base.AttentionCapture(len(triples), device)
        adapter = model.network.interaction_adapter
        model.network.set_interaction_attention_capture(True)
        hook = adapter.register_forward_hook(capture.hook)
        try:
            sample = diffusion.sample(
                model,
                fixed,
                text,
                global_bps,
                goals,
                progress,
                local_object_bps=local_bps,
                generator=generator,
            )
        finally:
            hook.remove()
            model.network.set_interaction_attention_capture(False)
        final_state = base.sha256_tensor_state(generator.get_state())
        attention_by_window.append(capture.result())
        local_bps_by_window.append({
            "chunk_index": chunk_index,
            "window_index": window_index,
            "shape": list(local_bps.shape),
            "dtype": str(local_bps.dtype),
            "finite": bool(torch.isfinite(local_bps).all()),
            "sha256": tensor_sha256(local_bps),
            "build_seconds": local_seconds,
        })
        sample[..., 219:228] = project_to_so3(
            sample[..., 219:228].reshape(len(triples), 16, 3, 3)
        ).reshape(len(triples), 16, 9)
        history_max_abs = max(
            history_max_abs,
            float((sample[:, :2] - fixed).abs().max().detach().cpu()),
        )
        decoded = dataset.codec.decode(sample, frame)
        fk = base.decoded_fk_positions(decoded, rest_offsets, parents_24)
        decoded_steps.append({
            key: value.detach().cpu()
            for key, value in decoded.items()
        })
        fk_steps.append(fk.detach().cpu())
        noise_streams.append({
            "chunk_index": chunk_index,
            "window_index": window_index,
            "label": label,
            "seed": stable_seed(label),
            "generator_initial_state_sha256": initial_state,
            "generator_final_state_sha256": final_state,
        })
        if window_index < base.WINDOWS_PER_SEQUENCE - 1:
            fixed, frame = dataset.codec.encode(
                decoded["joints"][:, -2:],
                decoded["human_rotation"][:, -2:],
                global_object_translation=decoded["object_translation"][:, -2:],
                global_object_rotation=decoded["object_rotation"][:, -2:],
                contact=decoded["contact"][:, -2:],
            )

    generated = []
    for row in range(len(triples)):
        value = {
            key: torch.cat([
                decoded_steps[step][key][row, 2:]
                for step in range(base.WINDOWS_PER_SEQUENCE)
            ])
            for key in (
                "joints",
                "human_rotation",
                "object_translation",
                "object_rotation",
                "contact",
            )
        }
        value["fk_joints"] = torch.cat([
            fk_steps[step][row, 2:]
            for step in range(base.WINDOWS_PER_SEQUENCE)
        ])
        value["attention_entropy"] = {
            role: {
                statistic: float(np.mean([
                    attention_by_window[step][row]["roles"][role][statistic]
                    for step in range(base.WINDOWS_PER_SEQUENCE)
                ]))
                for statistic in ("entropy_nats", "entropy_normalized")
            }
            for role in base.ROLE_NAMES
        }
        value["attention_forward_calls"] = sum(
            int(attention_by_window[step][row]["forward_calls"])
            for step in range(base.WINDOWS_PER_SEQUENCE)
        )
        generated.append(value)
    return {
        "generated": generated,
        "decoded_steps": decoded_steps,
        "noise_streams": noise_streams,
        "local_bps": local_bps_by_window,
        "history_max_abs": history_max_abs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=base.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("invalid D2-AD internal lifecycle run id")
    if not re.fullmatch(r"[0-9a-f]{64}", args.target_sha256):
        raise ValueError("D2-AD target SHA-256 must be lowercase hexadecimal")
    if args.batch_size <= 0 or 64 % args.batch_size:
        raise ValueError("D2-AD internal batch size must evenly divide 64")
    config = resolved_config(args)
    if args.resolve_only:
        base.exclusive_json(args.resolved_config.resolve(), config)
        return
    if Path(sys.executable).resolve() != Path(
        os.environ.get("INFBAGEL_PYTHON", ""),
    ).resolve():
        raise ValueError("D2-AD internal requires the absolute INFBAGEL_PYTHON")
    if Path(sys.executable).resolve() != Path(EXPECTED_PYTHON).resolve():
        raise ValueError(f"D2-AD internal requires {EXPECTED_PYTHON}")
    if (
        os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi"
        or socket.gethostname() != "node01"
    ):
        raise RuntimeError("D2-AD internal is restricted to the HOI worker")
    if base.git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("D2-AD internal refuses a dirty worker checkout")
    if json.loads(args.resolved_config.read_text(encoding="utf-8")) != config:
        raise ValueError("D2-AD internal runtime differs from archived config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-AD internal requires worker CUDA")
    if args.output_dir.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir.resolve()}")
    if args.metrics.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.metrics.resolve()}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    author_utils.SMPL_DIR = str((REPO / "smpl_models").resolve())
    try:
        checkpoint = checkpoint_contract(
            args.target_checkpoint.resolve(), args.target_sha256,
        )
        asset_hashes = {
            "normalization": sha256_file((REPO / "data/train/norm.npy").resolve()),
            "bps": sha256_file((REPO / "code/bps.pt").resolve()),
            "split": sha256_file(
                REPO
                / "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
            "basis_yup_tensor": BPS_YUP_TENSOR_SHA256,
            "rest_mesh_manifest": REST_MESH_MANIFEST_SHA256,
            "object_mapping": OBJECT_MAPPING_SHA256,
        }
        if asset_hashes != {
            "normalization": base.EXPECTED_NORMALIZATION_SHA256,
            "bps": BPS_SHA256,
            "split": base.EXPECTED_SPLIT_SHA256,
            "basis_yup_tensor": BPS_YUP_TENSOR_SHA256,
            "rest_mesh_manifest": REST_MESH_MANIFEST_SHA256,
            "object_mapping": OBJECT_MAPPING_SHA256,
        }:
            raise ValueError(f"D2-AD internal asset hash mismatch: {asset_hashes}")

        base.seed_everything(42)
        dataset = D2ADPriorWindowDataset(
            str(REPO),
            "hoi",
            partition="internal_validation",
            split_manifest=(
                "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
        )
        selection = base.select_contact_holdout(dataset)
        if (
            selection["sha256"] != base.SELECTION_SHA256
            or selection["sequences"] != 64
            or selection["windows"] != 192
            or selection["phase_offsets"] != [14, 56, 98]
        ):
            raise ValueError("D2-AD internal selection contract mismatch")
        triples = selection["triples"]
        parents_24 = torch.from_numpy(
            get_smpl_parents(use_joints24=True).copy()
        ).long().to(device)
        parents_22 = torch.from_numpy(
            get_smpl_parents(use_joints24=False).copy()
        ).long().to(device)
        targets = base.prepare_targets(dataset, triples, parents_24.cpu())
        for target, triple in zip(targets, triples):
            target["sequence_index"] = int(
                dataset[triple[0]]["sequence_index"].item()
            )
        rest_vertices = base.load_rest_vertices(dataset, triples, device)
        penetration_assets = base.load_penetration_assets(REPO)
        betas = np.load(REPO / "data/train/betas.npy", mmap_mode="r")
        translations = np.load(
            REPO / "data/train/transl_aligned.npy", mmap_mode="r"
        )
        with (REPO / "data/train/gender.pkl").open("rb") as handle:
            genders = pickle.load(handle)
        smpl_cache: Dict[str, torch.nn.Module] = {}
        diffusion = GaussianDiffusion(500).to(device)
        model, metadata = load_trained_hoi_prior(
            str(args.target_checkpoint.resolve()),
            device,
            weight_variant="online",
            expected_architecture_variant=HOI_ARCHITECTURE_D2AD,
        )
        if metadata["data_contract_sha256"] != base.EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError("D2-AD internal checkpoint data-contract mismatch")
        model.eval()
        model_before = state_dict_sha256(model)
        local_bps_builder = LocalObjectBPSBuilder(
            REPO, query_workers=DEFAULT_QUERY_WORKERS,
        )

        variants: Dict[str, object] = {}
        records_by_variant: Dict[str, List[Dict[str, object]]] = {}
        noise_by_variant: Dict[str, object] = {}
        local_bps_by_variant: Dict[str, object] = {}
        attention_appendix: Dict[str, object] = {}
        for variant in base.VARIANTS:
            model.network.set_interaction_diagnostic_variant(variant)
            records: List[Dict[str, object]] = []
            decoded_chunks = []
            noise_streams = []
            local_bps_records = []
            history_max_abs = 0.0
            for chunk_index, offset in enumerate(
                range(0, len(triples), args.batch_size)
            ):
                selected = triples[offset:offset + args.batch_size]
                rollout = rollout_chunk(
                    model,
                    diffusion,
                    dataset,
                    selected,
                    device,
                    rest_vertices,
                    parents_24,
                    local_bps_builder,
                    chunk_index=chunk_index,
                )
                decoded_chunks.append(rollout["decoded_steps"])
                noise_streams.extend(rollout["noise_streams"])
                local_bps_records.extend(rollout["local_bps"])
                history_max_abs = max(
                    history_max_abs, float(rollout["history_max_abs"])
                )
                for target, generated in zip(
                    targets[offset:offset + args.batch_size],
                    rollout["generated"],
                ):
                    penetration = base.sequence_penetration(
                        generated,
                        sequence_index=int(target["sequence_index"]),
                        object_name=str(target["object_category"]),
                        device=device,
                        parents_22=parents_22,
                        betas=betas,
                        genders=genders,
                        translations=translations,
                        penetration_assets=penetration_assets,
                        smpl_cache=smpl_cache,
                    )
                    records.append(base.analyze_sequence(
                        target,
                        generated,
                        rest_vertices,
                        device,
                        penetration=penetration,
                    ))
            decoded_steps = base.concatenate_decoded_steps(
                decoded_chunks, device,
            )
            kinematics = base.physical_summary(
                dataset, triples, decoded_steps, device,
            )
            sequence_names = [str(record["sequence"]) for record in records]
            mapped_kinematics = base.kinematics_by_sequence(
                kinematics, sequence_names,
            )
            for record in records:
                record["kinematics"] = mapped_kinematics[str(record["sequence"])]
            semantic_geometry = base._summary_for_records(
                records, include_categories=True,
            )
            penetration_summary = base.aggregate_penetration(records)
            attention_summary = base.aggregate_attention(records)
            finite = bool(
                base.all_finite(semantic_geometry)
                and base.all_finite(kinematics)
                and base.all_finite(attention_summary)
                and all(bool(row["finite"]) for row in local_bps_records)
                and all(
                    all(
                        value is None or math.isfinite(float(value))
                        for key, value in record["penetration"].items()
                        if key not in {
                            "finite", "excluded_by_official_contract",
                        }
                    )
                    for record in records
                )
            )
            complete = base.variant_complete(records, kinematics)
            variant_value = {
                "variant": variant,
                "history_max_abs": history_max_abs,
                "aggregate": {
                    "semantic_and_geometry": semantic_geometry,
                    "kinematics": kinematics["aggregate"],
                    "penetration": penetration_summary,
                    "attention_entropy": attention_summary,
                    "local_bps_build_seconds": float(sum(
                        row["build_seconds"] for row in local_bps_records
                    )),
                },
                "kinematics_full": kinematics,
                "per_sequence": records,
                "noise_streams": noise_streams,
                "local_bps": local_bps_records,
                "finite": finite,
                "all_fields_reported": complete,
            }
            variant_path = output_dir / f"{variant}.json"
            base.exclusive_json(variant_path, variant_value)
            variants[variant] = {
                "artifact": {
                    "path": str(variant_path),
                    "sha256": sha256_file(variant_path),
                    "bytes": variant_path.stat().st_size,
                },
                "history_max_abs": history_max_abs,
                "aggregate": variant_value["aggregate"],
                "finite": finite,
                "all_fields_reported": complete,
            }
            records_by_variant[variant] = records
            noise_by_variant[variant] = noise_streams
            local_bps_by_variant[variant] = local_bps_records
            attention_appendix[variant] = {
                "aggregate": attention_summary,
                "per_sequence": [
                    {
                        "sequence": record["sequence"],
                        "roles": record["attention_entropy"],
                        "forward_calls": record["attention_forward_calls"],
                    }
                    for record in records
                ],
            }
            model.network.interaction_adapter.clear_diagnostic_state()
            torch.cuda.empty_cache()

        model_after = state_dict_sha256(model)
        comparisons = base.paired_comparisons(records_by_variant)
        finite_masks = [
            comparisons[f"full_vs_{variant}"][
                "other_minus_full_gt_contact_distance_cm"
            ]["finite_sequence_names"]
            for variant in base.VARIANTS[1:]
        ]
        gt_contact_mask_exact = bool(
            finite_masks[0] == finite_masks[1]
            and len(finite_masks[0]) == base.GT_CONTACT_FINITE_SEQUENCE_COUNT
            and base.sequence_names_sha256(finite_masks[0])
            == base.GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256
        )
        paired_noise_identity = all(
            noise_by_variant[variant] == noise_by_variant["full"]
            for variant in base.VARIANTS[1:]
        )
        paired_noise_path = output_dir / "paired_noise.json"
        base.exclusive_json(paired_noise_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "shared": paired_noise_identity,
            "variants": noise_by_variant,
        })
        local_bps_path = output_dir / "local_bps_appendix.json"
        base.exclusive_json(local_bps_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "coordinate_system": "human_window_local_y_up",
            "query_workers": DEFAULT_QUERY_WORKERS,
            "full_rest_mesh": True,
            "variants": local_bps_by_variant,
        })
        attention_path = output_dir / "attention_appendix.json"
        base.exclusive_json(attention_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "selection_use": False,
            "roles": list(base.ROLE_NAMES),
            "variants": attention_appendix,
        })

        sampler_source = inspect.getsource(rollout_chunk)
        contract = {
            "checkpoint_contract": all(checkpoint["checks"].values()),
            "checkpoint_architecture_variant": (
                metadata["architecture_variant"] == HOI_ARCHITECTURE_D2AD
            ),
            "asset_hashes_exact": True,
            "selection_exact": True,
            "gt_contact_finite_mask_exact": gt_contact_mask_exact,
            "paired_noise_identity": paired_noise_identity,
            "history_restoration": all(
                float(variants[variant]["history_max_abs"])
                <= base.HISTORY_MAX_ABS
                for variant in base.VARIANTS
            ),
            "all_variants_finite": all(
                bool(variants[variant]["finite"]) for variant in base.VARIANTS
            ),
            "all_fields_reported": all(
                bool(variants[variant]["all_fields_reported"])
                for variant in base.VARIANTS
            ),
            "model_state_unchanged": model_before == model_after,
            "parameter_grad_buffers_clear": all(
                parameter.grad is None for parameter in model.parameters()
            ),
            "attention_capture_descriptive_only": True,
            "penetration_zero_denominator_explicit": True,
            "local_bps_current_frame_recomputed": (
                "local_bps_builder.build(" in sampler_source
            ),
            "global_bps_recomputed_unchanged": (
                "base.current_bps(" in sampler_source
            ),
            "sampler_future_gt_absent": "future_gt" not in sampler_source,
            "sampler_stored_per_frame_bps_absent": (
                "stored_per_frame_bps" not in sampler_source
                and 'item["object_bps"]' not in sampler_source
            ),
            "mesh_subsample_absent": "subsample" not in sampler_source,
            "optimizer_absent": True,
            "checkpoint_write_absent": True,
            "official_test_absent": True,
        }
        decision = base.internal_mechanism_gate(contract, comparisons)
        decision["classification"] = CLASSIFICATION_MAP[
            decision["classification"]
        ]
        result = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "completed",
            "seed": 42,
            "git_commit": base.git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "selection": {
                key: value
                for key, value in selection.items()
                if key != "triples"
            },
            "target_checkpoint": checkpoint,
            "checkpoint_metadata": metadata,
            "learned_adapter": {
                "alpha": float(
                    model.network.interaction_adapter.alpha.detach().cpu()
                ),
                "gate": float(torch.tanh(
                    model.network.interaction_adapter.alpha.detach()
                ).cpu()),
                "contract": (
                    model.network.interaction_adapter.contract_metadata()
                ),
            },
            "assets": {
                **asset_hashes,
                "data_contract_sha256": base.EXPECTED_DATA_CONTRACT_SHA256,
                "penetration_hand_vertex_ids_sha256": penetration_assets[
                    "hand_ids_sha256"
                ],
            },
            "variants": variants,
            "comparisons": comparisons,
            "contract": contract,
            "decision": decision,
            "paired_noise": {
                "path": str(paired_noise_path),
                "sha256": sha256_file(paired_noise_path),
            },
            "local_bps_appendix": {
                "path": str(local_bps_path),
                "sha256": sha256_file(local_bps_path),
            },
            "attention_appendix": {
                "path": str(attention_path),
                "sha256": sha256_file(attention_path),
                "selection_use": False,
            },
            "optimizer_created": False,
            "training_updates": 0,
            "checkpoint_writes": 0,
            "checkpoint_selection": False,
            "consistency_started": False,
            "official_test_used": False,
            "gpu": {
                "device": str(device),
                "name": torch.cuda.get_device_name(device),
                "maximum_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "maximum_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            },
        }
        base.exclusive_json(args.metrics.resolve(), result)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "failed",
            "seed": 42,
            "git_commit": base.git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "classification": FAILURE_CLASSIFICATION,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "optimizer_created": False,
            "training_updates": 0,
            "checkpoint_writes": 0,
            "checkpoint_selection": False,
            "consistency_started": False,
            "official_test_used": False,
        }
        failure_path = output_dir / "failure.json"
        if not failure_path.exists():
            base.exclusive_json(failure_path, failure)
        if not args.metrics.resolve().exists():
            base.exclusive_json(args.metrics.resolve(), failure)
        raise


if __name__ == "__main__":
    main()
