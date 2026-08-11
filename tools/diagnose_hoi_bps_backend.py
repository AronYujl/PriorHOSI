#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-B author-BPS backend replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import trimesh
from pytorch3d import __version__ as pytorch3d_version
from pytorch3d.ops import knn_points


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import zup_to_yup  # noqa: E402
from priors.hoi.data import PriorWindowDataset  # noqa: E402
from priors.hoi.remediation import select_internal_triples, selection_sha256  # noqa: E402
from priors.core.window_codec import (  # noqa: E402
    BPS_SHA256,
    project_to_so3,
    zup_to_yup_tensor,
)


RUN_ID = "p1-hoi-d2b-bps-replay-s42-20260715"
AUTHOR_BASELINE = "b9a158f75ab0740c91c9cfc8863a65fa381b014c"
AUTHOR_BPS_BLOB = "03b0851af882192913c90d3485559c6c034455ed"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
BPS_TOLERANCE = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def yup_to_zup_tensor(value: torch.Tensor) -> torch.Tensor:
    """Invert the author dataset's [x,z,-y] vector conversion."""
    return torch.stack((value[..., 0], -value[..., 2], value[..., 1]), dim=-1)


def classify_backend(cpu: Mapping[str, object], cuda: Mapping[str, object]) -> str:
    del cpu  # CPU failure is already established; CUDA is the preregistered decision gate.
    return "cpu-knn-tie-backend-artifact" if bool(cuda["passed"]) else "backend-replay-unresolved"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def asset_provenance() -> Dict[str, object]:
    bps_path = REPO / "code/bps.pt"
    author_blob = _git("rev-parse", f"{AUTHOR_BASELINE}:code/bps.pt")
    current_blob = _git("hash-object", str(bps_path))
    if author_blob != AUTHOR_BPS_BLOB or current_blob != AUTHOR_BPS_BLOB:
        raise ValueError(f"author/current BPS blob mismatch: {author_blob}/{current_blob}")
    if sha256_file(bps_path) != BPS_SHA256:
        raise ValueError(f"BPS SHA-256 mismatch: {bps_path}")
    split_root = REPO / "data/train/rest_object_geo"
    repository_root = REPO / "data/object/rest_object_geo"
    pairs = []
    for source in sorted(path for path in split_root.iterdir() if path.suffix in {".ply", ".npy"}):
        other = repository_root / source.name
        if not other.is_file():
            raise ValueError(f"missing paired rest-geometry asset: {other}")
        source_hash = sha256_file(source)
        other_hash = sha256_file(other)
        pairs.append({
            "name": source.name,
            "split_sha256": source_hash,
            "repository_sha256": other_hash,
            "identical": source_hash == other_hash,
        })
    if len(pairs) != 26 or not all(bool(pair["identical"]) for pair in pairs):
        raise ValueError("rest-geometry provenance mismatch")
    return {
        "author_baseline_commit": AUTHOR_BASELINE,
        "author_bps_blob": author_blob,
        "current_bps_blob": current_blob,
        "bps_sha256": BPS_SHA256,
        "rest_geometry_object_count": 13,
        "rest_geometry_file_pairs": pairs,
    }


def load_rest_vertices(object_name: str, device: torch.device) -> torch.Tensor:
    mesh = trimesh.load_mesh(
        REPO / "data/train/rest_object_geo" / f"{object_name}.ply", process=False,
    )
    vertices = zup_to_yup(np.asarray(mesh.vertices, dtype=np.float32).copy())
    return torch.from_numpy(vertices).to(device)


@torch.no_grad()
def replay_device(
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
) -> Dict[str, object]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    failures: List[Dict[str, object]] = []
    maxima: List[float] = []
    rms_values: List[float] = []
    failed_basis_total = 0
    cache: Dict[str, torch.Tensor] = {}
    for position in positions:
        item = dataset[position]
        global_index = int(dataset.indices[position])
        sequence = int(item["sequence_index"])
        sequence_name = str(dataset.scene_names[sequence])
        object_name = sequence_name.split("_")[1]
        if object_name not in cache:
            cache[object_name] = load_rest_vertices(object_name, device)
        rest = cache[object_name]
        reference = item["object_rotation_reference"].to(device)
        stored = item["object_bps"].to(device)
        replay = dataset.codec.recompute_bps(rest, reference)
        error = (replay - stored).abs()
        maximum = float(error.max())
        rms = float(error.square().mean().sqrt())
        maxima.append(maximum)
        rms_values.append(rms)
        failed_basis = torch.nonzero(error.max(dim=-1).values > BPS_TOLERANCE).flatten()
        failed_basis_total += int(failed_basis.numel())
        if failed_basis.numel():
            rotation = project_to_so3(reference)
            transformed = rest @ rotation.transpose(-1, -2)
            basis = dataset.codec.bps_basis.to(device)
            nearest = knn_points(basis[None], transformed[None], K=1, return_nn=True)
            selected_indices = nearest.idx[0, failed_basis, 0]
            stored_closest = basis[failed_basis] + yup_to_zup_tensor(stored[failed_basis])
            stored_nearest = knn_points(
                stored_closest[None], transformed[None], K=1, return_nn=True,
            )
            stored_indices = stored_nearest.idx[0, :, 0]
            selected_points = transformed[selected_indices]
            stored_points = transformed[stored_indices]
            selected_squared = (selected_points - basis[failed_basis]).square().sum(dim=-1)
            stored_squared = (stored_points - basis[failed_basis]).square().sum(dim=-1)
            stored_residual = (stored_points - stored_closest).norm(dim=-1)
            failures.append({
                "sequence": sequence_name,
                "dataset_position": int(position),
                "global_window_index": global_index,
                "max_abs": maximum,
                "rms": rms,
                "failed_basis_indices": failed_basis.cpu().tolist(),
                "selected_vertex_indices": selected_indices.cpu().tolist(),
                "stored_vertex_indices": stored_indices.cpu().tolist(),
                "stored_vertex_residual_m": stored_residual.cpu().tolist(),
                "stored_minus_selected_squared_distance": (
                    stored_squared - selected_squared
                ).cpu().tolist(),
            })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "device": str(device),
        "windows": len(positions),
        "max_abs": max(maxima),
        "mean_window_rms": float(np.mean(rms_values)),
        "failed_windows": len(failures),
        "failed_basis_points": failed_basis_total,
        "tolerance": BPS_TOLERANCE,
        "passed": bool(max(maxima) <= BPS_TOLERANCE),
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B-D2-B",
        "mode": "author-bps-backend-replay-only",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "partition": "internal_validation",
        "windows": 32,
        "devices": ["cpu", args.cuda_device],
        "bps_max_abs_tolerance": BPS_TOLERANCE,
        "author_baseline_commit": AUTHOR_BASELINE,
        "author_bps_blob": AUTHOR_BPS_BLOB,
        "bps_sha256": BPS_SHA256,
        "official_test_used": False,
        "chois_used": False,
        "checkpoint_count_loaded": 0,
        "checkpoint_selection": False,
        "model_forward_calls": 0,
        "training_updates": 0,
        "output": str(Path(args.output).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-B run id must be {RUN_ID}")
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match the archived D2-B resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-B requires INFBAGEL_WORKER_EXPERT=hoi")
    cuda_device = torch.device(args.cuda_device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-B requires the HOI worker CUDA backend")
    provenance = asset_provenance()
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    triples = select_internal_triples(dataset, 128)
    positions = [triple[0] for triple in triples[:32]]
    cpu = replay_device(dataset, positions, torch.device("cpu"))
    cuda = replay_device(dataset, positions, cuda_device)
    classification = classify_backend(cpu, cuda)
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-B",
        "seed": 42,
        "git_commit": _git("rev-parse", "HEAD"),
        "selection": {
            "partition": "internal_validation",
            "windows": len(positions),
            "global_window_indices_sha256": selection_sha256(
                int(dataset.indices[position]) for position in positions
            ),
            "sequence_names_sha256": selection_sha256(
                str(dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])])
                for position in positions
            ),
        },
        "provenance": provenance,
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "pytorch3d": pytorch3d_version,
            "trimesh": trimesh.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(cuda_device),
        },
        "cpu": cpu,
        "cuda": cuda,
        "classification": classification,
        "conditional_continuation_allowed": classification == "cpu-knn-tie-backend-artifact",
        "future_gt_used_for_condition": False,
        "official_test_used": False,
        "chois_used": False,
        "checkpoint_count_loaded": 0,
        "checkpoint_selection": False,
        "model_forward_calls": 0,
        "training_updates": 0,
    }
    exclusive_json(Path(args.output).resolve(), output)


if __name__ == "__main__":
    main()
