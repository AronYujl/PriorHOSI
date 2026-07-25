#!/usr/bin/env python3
"""Run the fixed D2-AB predicted-support internal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.d2ab import (  # noqa: E402
    D2ABPriorWindowDataset,
    _d2ab_physical_terms,
    sha256_file,
)
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.optimizer_reset import (  # noqa: E402
    NATIVE_SELECTION_SHA256,
    paired_difference,
    paired_mean_ratio as paired_ratio,
    select_native_holdout,
    stable_seed,
)
from priors.representation import REPRESENTATION  # noqa: E402
from tools.diagnose_hoi_d2y import _predicted_fk  # noqa: E402


RUN_ID = "p1-hoi-d2ab-predicted-support-no-slip-internal-s42-20260725"
SUBPHASE = "1B-D2-AB0-internal"
EXPECTED_PYTHON = "/home/yujinlun/data/envs/infbagel/bin/python"
EXPECTED_DATA_CONTRACT_SHA256 = (
    "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
)
CONTROL_CHECKPOINT_SHA256 = (
    "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51"
)
CONTROL_RUN_ID = "p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723"
TARGET_RUN_ID = "p1-hoi-d2ab-predicted-support-no-slip-s42-20260725"
TARGET_METADATA_SHA256 = (
    "807978580221910ad00260c2dff4f33ddacbb1bf72bad7443bf21ac48f31f079"
)
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e"
)
TIMESTEPS = (249, 499)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 42


def _exclusive_json(path: Path, value: object) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _checkpoint_contract(
    path: Path,
    expected_sha256: str,
    *,
    run_id: str,
    expected_windows: int,
) -> Dict[str, object]:
    actual = sha256_file(path)
    expected_name = f"{run_id}_windows{expected_windows:09d}.pth"
    if actual != expected_sha256:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    if path.name != expected_name:
        raise ValueError(f"checkpoint basename mismatch: {path.name}")
    raw = torch.load(path, map_location="cpu")
    initialization = raw.get("weight_initialization", {})
    checks = {
        "checkpoint_type": raw.get("checkpoint_type") == "hoi_prior_phase1b",
        "run_id": raw.get("run_id") == run_id,
        "seed": raw.get("seed") == 42,
        "processed_windows": raw.get("processed_windows") == expected_windows,
        "optimizer_updates": raw.get("optimizer_updates") == expected_windows // 2048,
        "world_size": raw.get("world_size") == 4,
        "effective_batch_size": raw.get("effective_batch_size") == 2048,
        "data_contract": raw.get("data_contract_sha256") == EXPECTED_DATA_CONTRACT_SHA256,
        "random_initialization": (
            initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("restored_components") == []
            and initialization.get("initial_model_state_sha256")
            == EXPECTED_INITIAL_MODEL_STATE_SHA256
        ),
        "online_model": isinstance(raw.get("model"), dict),
    }
    if run_id == TARGET_RUN_ID:
        checks.update({
            "d2ab_mode": raw.get("resume_contract", {}).get(
                "d2ab_predicted_support_no_slip"
            ) is True,
            "metadata": raw.get("resume_contract", {}).get(
                "d2ab_support_metadata_sha256"
            ) == TARGET_METADATA_SHA256,
        })
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"checkpoint contract mismatch for {run_id}: {failed}")
    return {
        "path": str(path),
        "sha256": actual,
        "run_id": run_id,
        "processed_windows": expected_windows,
        "optimizer_updates": expected_windows // 2048,
        "git_commit": raw.get("git_commit"),
        "checks": checks,
    }


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = (
        "x",
        "text_embedding",
        "object_bps",
        "goals",
        "progress",
        "rest_human_offsets",
        "terminal_window",
        "rest_object_points",
        "world_to_local_rotation",
        "object_rotation_reference",
        "d2ab_floor_m",
    )
    return {key: batch[key].to(device) for key in keys}


def _per_sequence(values: torch.Tensor) -> List[float]:
    if values.shape[0] != 96:
        raise ValueError(f"D2-AB internal expected 96 windows, got {values.shape[0]}")
    per_window = values.reshape(96, -1).mean(dim=1)
    per_sequence = (
        per_window.reshape(32, 3).mean(dim=1).detach().double().cpu().numpy()
    )
    if not np.isfinite(per_sequence).all():
        raise ValueError("D2-AB internal per-sequence value is non-finite")
    return [float(value) for value in per_sequence]


def _evaluate_model_timestep(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    *,
    timestep: int,
    seed_label: str,
) -> Dict[str, object]:
    model.train()
    clean = batch["x"]
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(stable_seed(seed_label + ":noise"))
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    timesteps = torch.full(
        (clean.shape[0],), timestep, device=clean.device, dtype=torch.long,
    )
    dropout_seed = stable_seed(seed_label + ":dropout")
    torch.manual_seed(dropout_seed)
    torch.cuda.manual_seed_all(dropout_seed)
    noisy = diffusion.q_sample(clean, timesteps, noise)
    with torch.no_grad():
        prediction = model(
            noisy,
            timesteps,
            batch["text_embedding"],
            batch["object_bps"],
            batch["goals"],
            normalize_progress(batch["progress"]),
        )
        predicted_fk = _predicted_fk(
            prediction,
            batch["rest_human_offsets"],
            parents,
            minimum,
            maximum,
        )
        terms = _d2ab_physical_terms(
            prediction,
            clean,
            predicted_fk,
            minimum,
            maximum,
            batch["d2ab_floor_m"],
        )
        support = terms["support_by_joint"]
        predicted_velocity = terms["predicted_velocity"]
        target_velocity = terms["target_velocity"]
        residual = predicted_velocity - (
            1.0 - support.unsqueeze(-1)
        ) * target_velocity
        supported_velocity = (
            support.unsqueeze(-1) * predicted_velocity.square()
        ).sum(dim=-1)
        no_slip = residual.square().mean(dim=-1)
        support_mass = support.mean(dim=-1)
        previous_height = terms["predicted_foot_previous"][..., 1].mean(dim=-1)
    return {
        "timestep": timestep,
        "noise_sha256": hashlib.sha256(
            noise.detach().contiguous().cpu().numpy().tobytes()
        ).hexdigest(),
        "dropout_seed": dropout_seed,
        "supported_velocity_m2_s2_by_sequence": _per_sequence(supported_velocity),
        "no_slip_residual_m2_s2_by_sequence": _per_sequence(no_slip),
        "support_mass_by_sequence": _per_sequence(support_mass),
        "previous_foot_height_m_by_sequence": _per_sequence(previous_height),
        "supported_velocity_mean_m2_s2": float(supported_velocity.mean().item()),
        "no_slip_residual_mean_m2_s2": float(no_slip.mean().item()),
        "support_mass_mean": float(support_mass.mean().item()),
        "support_mass_min": float(support_mass.min().item()),
        "support_mass_max": float(support_mass.max().item()),
        "previous_foot_height_mean_m": float(previous_height.mean().item()),
    }


def _load_and_evaluate(
    path: Path,
    expected_sha256: str,
    run_id: str,
    device: torch.device,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> Dict[str, object]:
    contract = _checkpoint_contract(
        path,
        expected_sha256,
        run_id=run_id,
        expected_windows=61_440_000,
    )
    model, metadata = load_trained_hoi_prior(
        str(path), device, weight_variant="online",
    )
    if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
        raise ValueError("loaded model data-contract mismatch")
    results = {
        str(timestep): _evaluate_model_timestep(
            model,
            diffusion,
            batch,
            parents,
            minimum,
            maximum,
            timestep=timestep,
            seed_label=f"D2AB:paired:t{timestep}",
        )
        for timestep in TIMESTEPS
    }
    del model
    torch.cuda.empty_cache()
    return {"checkpoint_contract": contract, "metadata": metadata, "timesteps": results}


def _aggregate_comparison(
    control: Mapping[str, object],
    target: Mapping[str, object],
) -> Dict[str, object]:
    comparisons = {}
    support_sanity = {}
    finite = True
    for timestep in (str(value) for value in TIMESTEPS):
        control_item = control["timesteps"][timestep]
        target_item = target["timesteps"][timestep]
        control_velocity = np.asarray(
            control_item["supported_velocity_m2_s2_by_sequence"], dtype=np.float64,
        )
        target_velocity = np.asarray(
            target_item["supported_velocity_m2_s2_by_sequence"], dtype=np.float64,
        )
        control_support = np.asarray(
            control_item["support_mass_by_sequence"], dtype=np.float64,
        )
        target_support = np.asarray(
            target_item["support_mass_by_sequence"], dtype=np.float64,
        )
        comparisons[timestep] = {
            "control_minus_target_supported_velocity": paired_difference(
                control_velocity,
                target_velocity,
                seed=BOOTSTRAP_SEED,
                replicates=BOOTSTRAP_REPLICATES,
            ),
            "control_minus_target_no_slip_residual": paired_difference(
                np.asarray(
                    control_item["no_slip_residual_m2_s2_by_sequence"],
                    dtype=np.float64,
                ),
                np.asarray(
                    target_item["no_slip_residual_m2_s2_by_sequence"],
                    dtype=np.float64,
                ),
                seed=BOOTSTRAP_SEED,
                replicates=BOOTSTRAP_REPLICATES,
            ),
        }
        support_sanity[timestep] = paired_ratio(
            target_support,
            control_support,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
        )
        finite = finite and all(
            math.isfinite(float(value))
            for value in (
                control_velocity.mean(),
                target_velocity.mean(),
                control_support.mean(),
                target_support.mean(),
            )
        )
    mechanism = {
        timestep: (
            comparisons[timestep][
                "control_minus_target_supported_velocity"
            ]["bootstrap_95_ci"][0] > 0.0
        )
        for timestep in comparisons
    }
    support_passed = all(
        0.80 <= item["bootstrap_95_ci"][0]
        and item["bootstrap_95_ci"][1] <= 1.20
        for item in support_sanity.values()
    )
    return {
        "comparisons": comparisons,
        "support_sanity": support_sanity,
        "mechanism_checks": mechanism,
        "mechanism_passed": all(mechanism.values()),
        "support_sanity_passed": support_passed,
        "contract_passed": finite,
        "finite": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--d2x-checkpoint", type=Path, required=True)
    parser.add_argument("--d2x-sha256", required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--support-metadata", type=Path, required=True)
    parser.add_argument("--support-metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-AB internal run id must be {RUN_ID}")
    if args.python.resolve() != Path(os.environ.get("INFBAGEL_PYTHON", "")).resolve():
        raise ValueError("D2-AB internal diagnostic requires absolute INFBAGEL_PYTHON")
    if args.python.resolve() != Path(EXPECTED_PYTHON).resolve():
        raise ValueError(f"D2-AB internal diagnostic requires {EXPECTED_PYTHON}")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi" or socket.gethostname() != "node01":
        raise RuntimeError("D2-AB internal diagnostic is restricted to the HOI worker")
    if subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        text=True,
    ).strip():
        raise RuntimeError("D2-AB internal diagnostic refuses a dirty worker checkout")
    if not re.fullmatch(r"[0-9a-f]{64}", args.target_sha256):
        raise ValueError("target checkpoint SHA-256 must be lowercase hexadecimal")
    if args.target_sha256 == CONTROL_CHECKPOINT_SHA256:
        raise ValueError("D2-AB target may not reuse D2-X control checkpoint")
    if args.d2x_sha256 != CONTROL_CHECKPOINT_SHA256:
        raise ValueError("D2-X internal control hash is not the sealed final checkpoint")
    if sha256_file(args.support_metadata.resolve()) != args.support_metadata_sha256:
        raise ValueError("D2-AB support metadata hash mismatch")
    if args.support_metadata_sha256 != TARGET_METADATA_SHA256:
        raise ValueError("D2-AB support metadata is not the registered artifact")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-AB internal diagnostic requires worker CUDA")

    dataset = D2ABPriorWindowDataset(
        str(REPO),
        "hoi",
        partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        support_metadata_path=str(args.support_metadata.resolve()),
        support_metadata_sha256=args.support_metadata_sha256,
    )
    selection = select_native_holdout(dataset)
    if selection["sha256"] != NATIVE_SELECTION_SHA256 or selection["sequences"] != 32:
        raise ValueError("D2-AB internal selection mismatch")
    items = [
        dataset[position]
        for triple in selection["triples"]
        for position in triple
    ]
    batch = _move_batch(default_collate(items), device)
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    minimum = torch.as_tensor(dataset.minimum, device=device)
    maximum = torch.as_tensor(dataset.maximum, device=device)
    diffusion = GaussianDiffusion(REPRESENTATION.diffusion_steps).to(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    control = _load_and_evaluate(
        args.d2x_checkpoint.resolve(),
        args.d2x_sha256,
        run_id=CONTROL_RUN_ID,
        device=device,
        diffusion=diffusion,
        batch=batch,
        parents=parents,
        minimum=minimum,
        maximum=maximum,
    )
    target = _load_and_evaluate(
        args.target_checkpoint.resolve(),
        args.target_sha256,
        run_id=TARGET_RUN_ID,
        device=device,
        diffusion=diffusion,
        batch=batch,
        parents=parents,
        minimum=minimum,
        maximum=maximum,
    )
    comparison = _aggregate_comparison(control, target)
    if not comparison["contract_passed"]:
        raise ValueError("D2-AB internal diagnostic produced non-finite values")
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "status": "completed",
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "eligible_sequence_ranks": [128, 159],
            "sequences": 32,
            "windows": 96,
            "sha256": selection["sha256"],
            "global_indices": selection["global_indices"],
            "official_test_sequences": 0,
        },
        "pairing": {
            "same_clean_windows": True,
            "same_timestep": True,
            "same_noise": True,
            "same_condition_dropout": True,
        },
        "control_checkpoint": {
            "path": str(args.d2x_checkpoint.resolve()),
            "sha256": args.d2x_sha256,
            "run_id": CONTROL_RUN_ID,
            "contract": control["checkpoint_contract"],
        },
        "target_checkpoint": {
            "path": str(args.target_checkpoint.resolve()),
            "sha256": args.target_sha256,
            "run_id": TARGET_RUN_ID,
            "contract": target["checkpoint_contract"],
        },
        "support_metadata": {
            "path": str(args.support_metadata.resolve()),
            "sha256": args.support_metadata_sha256,
        },
        "timesteps": list(TIMESTEPS),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "sequence",
        },
        "control": control,
        "target": target,
        "comparison": comparison,
        "runtime_seconds": time.perf_counter() - started,
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_write": False,
        "checkpoint_selected": False,
        "official_test_used": False,
        "consistency_authorized": False,
        "consistency_started": False,
    }
    _exclusive_json(args.output.resolve(), output)


if __name__ == "__main__":
    main()
