#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-C immutable-mesh BPS tie audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
from priors.hoi.remediation import bps_replay_equivalence_gate, selection_sha256  # noqa: E402
from priors.core.window_codec import BPS_SHA256, project_to_so3, zup_to_yup_tensor  # noqa: E402


RUN_ID = "p1-hoi-d2c-bps-equivalence-s42-20260715"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
WINDOWS_PER_CLASS = 64
EXPECTED_WINDOWS = 832
EXPECTED_GLOBAL_SELECTION_SHA256 = "5f13844f9c3c1540d89d19b304e484cba6e84cc8adcb7276ebf4d17fb803db72"
EXPECTED_SEQUENCE_WINDOW_SHA256 = "e7827e83b88058e9d87dc4d56ccdbfe5929ea245645356fab278364b1aae1f38"
STRICT_COMPONENT_MAX_ABS = 1e-4
STORED_MESH_RESIDUAL_M_MAX = 1e-6
RECOMPUTED_MESH_RESIDUAL_M_MAX = 1e-6
NEAREST_SQUARED_DISTANCE_GAP_M2_MAX = 1e-7
PLY_SHA256 = {
    "clothesstand": "f7967db79cfa76c2dd154426c0e110369b5f28f756dcd9df5e5c9ecba3fed960",
    "floorlamp": "c012a69bfb37542ad099ff087b17059b2622a47d2e04e09d67403ea929470e4a",
    "largebox": "6b9ef6cf4564c0a9d052796b2f24bcb0557249304a6c417f25bfcacec50106e0",
    "largetable": "087c3575a25b0c423a9dcc7ceea2519ff604ce0b8172a5e66e89eefd2d3cb7d1",
    "monitor": "21f3bbae53b678eff92f34d597f18a7cdbb04db3195425f6c64c6a2790129bc6",
    "plasticbox": "e0a9c29124314f02d6b189b36716c5a01aeeb973b4001e8f28ee76f3382a1a76",
    "smallbox": "2aa65b4bf36476ffec1bf4582f2aded77bf111956c70d127d44e8bc76b2072b8",
    "smalltable": "eb601c69e2aa96b2cf284dccdf63e2c49f192200f3dc298e5cd6143b42a25628",
    "suitcase": "6ebfb9e5bf732314fe8fbd5dbf1ad500b4b2218fab74888dca7cf3fc2ad49845",
    "trashcan": "7381b83dc33265872753b8425ec7da4a24da0fc0512b57ebda3307c298659d38",
    "tripod": "b2222da71906a832826bd15b697e1a21e69627664da97f62704667fe7e871540",
    "whitechair": "f3cc5036d305cbe5e53a701cdfa63c9181113e6b2249860bf190ab0a8a0bedd0",
    "woodchair": "45d3e42b3f4cb537efc94e37d5919ff1a70967c4ce4c8ad399a0882e5d356a89",
}


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


def select_d2c_windows(dataset) -> Tuple[List[int], Dict[str, int]]:
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-C selection is internal-validation only")
    groups: Dict[str, List[Tuple[str, int, int, str]]] = defaultdict(list)
    for position, global_index in enumerate(np.asarray(dataset.indices).tolist()):
        sequence = int(dataset.sequence_ids[global_index])
        sequence_name = str(dataset.scene_names[sequence])
        object_name = sequence_name.split("_")[1]
        key = hashlib.sha256(
            f"42:hoi-d2c:{object_name}:{sequence_name}:{global_index}".encode("utf-8")
        ).hexdigest()
        groups[object_name].append((key, position, global_index, sequence_name))
    if set(groups) != set(PLY_SHA256):
        raise ValueError(f"unexpected D2-C object classes: {sorted(groups)}")
    selected: List[int] = []
    coverage = {}
    for object_name in sorted(groups):
        ranked = sorted(groups[object_name])
        if len(ranked) < WINDOWS_PER_CLASS:
            raise ValueError(f"only {len(ranked)} windows for {object_name}")
        chosen = ranked[:WINDOWS_PER_CLASS]
        selected.extend(value[1] for value in chosen)
        coverage[object_name] = len(chosen)
    global_hash = selection_sha256(int(dataset.indices[position]) for position in selected)
    sequence_window_payload = "\n".join(
        f"{dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])]}:"
        f"{int(dataset.indices[position])}"
        for position in selected
    )
    sequence_window_hash = hashlib.sha256(sequence_window_payload.encode("utf-8")).hexdigest()
    if len(selected) != EXPECTED_WINDOWS:
        raise ValueError(f"D2-C selected {len(selected)} windows, expected {EXPECTED_WINDOWS}")
    if global_hash != EXPECTED_GLOBAL_SELECTION_SHA256:
        raise ValueError(f"D2-C global selection hash mismatch: {global_hash}")
    if sequence_window_hash != EXPECTED_SEQUENCE_WINDOW_SHA256:
        raise ValueError(f"D2-C sequence/window hash mismatch: {sequence_window_hash}")
    return selected, coverage


def verify_assets() -> Dict[str, object]:
    bps_path = REPO / "code/bps.pt"
    if sha256_file(bps_path) != BPS_SHA256:
        raise ValueError("D2-C BPS basis hash mismatch")
    records = {}
    for object_name, expected in sorted(PLY_SHA256.items()):
        split_path = REPO / "data/train/rest_object_geo" / f"{object_name}.ply"
        repository_path = REPO / "data/object/rest_object_geo" / f"{object_name}.ply"
        split_hash = sha256_file(split_path)
        repository_hash = sha256_file(repository_path)
        if split_hash != expected or repository_hash != expected:
            raise ValueError(f"D2-C immutable PLY hash mismatch: {object_name}")
        records[object_name] = {
            "sha256": expected,
            "split_path": str(split_path),
            "repository_path": str(repository_path),
        }
    return {"bps_sha256": BPS_SHA256, "rest_object_ply": records}


def _gate_details(
    gate: Mapping[str, object],
    selected_indices: torch.Tensor,
    rows: torch.Tensor,
) -> List[Dict[str, object]]:
    return [{
        "basis_index": int(row),
        "component_max_abs": float(gate["component_error"][row]),
        "selected_vertex_index": int(selected_indices[row]),
        "stored_vertex_index": int(gate["stored_vertex_indices"][row]),
        "stored_mesh_residual_m": float(gate["stored_mesh_residual_m"][row]),
        "recomputed_mesh_residual_m": float(gate["recomputed_mesh_residual_m"][row]),
        "nearest_squared_distance_gap_m2": float(gate["nearest_squared_distance_gap_m2"][row]),
        "nearest_linear_distance_gap_m": float(gate["nearest_linear_distance_gap_m"][row]),
    } for row in rows.cpu().tolist()]


@torch.no_grad()
def replay_device(
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
    *,
    nearest_squared_distance_gap_m2_max: float = NEAREST_SQUARED_DISTANCE_GAP_M2_MAX,
    nearest_linear_distance_gap_m_max: float = float("inf"),
    expected_windows_per_class: Optional[int] = WINDOWS_PER_CLASS,
    expected_object_classes: Optional[int] = 13,
) -> Dict[str, object]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    basis = dataset.codec.bps_basis.to(device)
    rest_cache: Dict[str, torch.Tensor] = {}
    strict_count = 0
    tie_count = 0
    failure_count = 0
    finite_failures = 0
    max_component = 0.0
    ties: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    per_class: Dict[str, Dict[str, object]] = {}
    for position in positions:
        global_index = int(dataset.indices[position])
        sequence = int(dataset.sequence_ids[global_index])
        sequence_name = str(dataset.scene_names[sequence])
        object_name = sequence_name.split("_")[1]
        source_frame = int(dataset.starts[global_index])
        offset = source_frame - int(dataset.seq_starts[sequence])
        if object_name not in rest_cache:
            mesh = trimesh.load_mesh(
                REPO / "data/train/rest_object_geo" / f"{object_name}.ply", process=False,
            )
            vertices = zup_to_yup(np.asarray(mesh.vertices, dtype=np.float32).copy())
            rest_cache[object_name] = torch.from_numpy(vertices).to(device)
        rest = rest_cache[object_name]
        reference = torch.from_numpy(
            np.array(dataset.object_rot[source_frame], dtype=np.float32, copy=True)
        ).to(device)
        rotation = project_to_so3(reference)
        transformed = rest @ rotation.transpose(-1, -2)
        nearest = knn_points(basis[None], transformed[None], K=1, return_nn=True)
        selected_indices = nearest.idx[0, :, 0]
        recomputed = zup_to_yup_tensor(nearest.knn[0, :, 0] - basis)
        stored = torch.from_numpy(zup_to_yup(np.asarray(
            dataset._bps(sequence_name)[offset], dtype=np.float32,
        ).copy())).to(device)
        gate = bps_replay_equivalence_gate(
            recomputed, stored, basis, transformed, selected_indices,
            strict_component_max_abs=STRICT_COMPONENT_MAX_ABS,
            stored_mesh_residual_m_max=STORED_MESH_RESIDUAL_M_MAX,
            recomputed_mesh_residual_m_max=RECOMPUTED_MESH_RESIDUAL_M_MAX,
            nearest_squared_distance_gap_m2_max=nearest_squared_distance_gap_m2_max,
            nearest_linear_distance_gap_m_max=nearest_linear_distance_gap_m_max,
        )
        strict_rows = torch.nonzero(gate["strict"]).flatten()
        tie_rows = torch.nonzero(gate["tie"]).flatten()
        failure_rows = torch.nonzero(gate["failure"]).flatten()
        strict_count += int(strict_rows.numel())
        tie_count += int(tie_rows.numel())
        failure_count += int(failure_rows.numel())
        finite_failures += int((~gate["finite"]).sum())
        window_max = float(gate["component_error"].max())
        max_component = max(max_component, window_max)
        class_record = per_class.setdefault(object_name, {
            "windows": 0, "strict_basis_points": 0, "tie_basis_points": 0,
            "unexplained_basis_points": 0, "max_component_abs": 0.0,
        })
        class_record["windows"] += 1
        class_record["strict_basis_points"] += int(strict_rows.numel())
        class_record["tie_basis_points"] += int(tie_rows.numel())
        class_record["unexplained_basis_points"] += int(failure_rows.numel())
        class_record["max_component_abs"] = max(class_record["max_component_abs"], window_max)
        base = {
            "object_name": object_name,
            "sequence": sequence_name,
            "dataset_position": int(position),
            "global_window_index": global_index,
            "source_frame": source_frame,
            "stored_bps_offset": offset,
        }
        for detail in _gate_details(gate, selected_indices, tie_rows):
            ties.append(dict(base, **detail))
        for detail in _gate_details(gate, selected_indices, failure_rows):
            failures.append(dict(base, **detail))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total = len(positions) * 1024
    passed = (
        (expected_object_classes is None or len(per_class) == expected_object_classes)
        and (
            expected_windows_per_class is None
            or all(record["windows"] == expected_windows_per_class for record in per_class.values())
        )
        and failure_count == 0
        and finite_failures == 0
        and strict_count + tie_count == total
    )
    return {
        "device": str(device),
        "windows": len(positions),
        "object_classes": len(per_class),
        "basis_points": total,
        "strict_basis_points": strict_count,
        "tie_basis_points": tie_count,
        "unexplained_basis_points": failure_count,
        "nonfinite_basis_points": finite_failures,
        "max_component_abs_including_accepted_ties": max_component,
        "per_class": per_class,
        "ties": ties,
        "failures": failures,
        "passed": passed,
        "runtime_seconds": time.perf_counter() - started,
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B-D2-C",
        "mode": "immutable-mesh-bps-geometric-equivalence-audit",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "partition": "internal_validation",
        "selection": {
            "object_classes": 13,
            "windows_per_class": WINDOWS_PER_CLASS,
            "windows": EXPECTED_WINDOWS,
            "global_window_indices_sha256": EXPECTED_GLOBAL_SELECTION_SHA256,
            "sequence_window_sha256": EXPECTED_SEQUENCE_WINDOW_SHA256,
        },
        "gate": {
            "strict_component_max_abs": STRICT_COMPONENT_MAX_ABS,
            "stored_mesh_residual_m_max": STORED_MESH_RESIDUAL_M_MAX,
            "recomputed_mesh_residual_m_max": RECOMPUTED_MESH_RESIDUAL_M_MAX,
            "nearest_squared_distance_gap_m2_max": NEAREST_SQUARED_DISTANCE_GAP_M2_MAX,
        },
        "devices": ["cpu", args.cuda_device],
        "execution_order": "cpu-first; stop before CUDA on any unexplained CPU basis point",
        "data_contract_sha256": EXPECTED_DATA_CONTRACT_SHA256,
        "bps_sha256": BPS_SHA256,
        "stored_per_frame_bps_use": "diagnostic_gt_replay_only",
        "sampler_stored_per_frame_bps": False,
        "sampler_future_gt": False,
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
        raise ValueError(f"D2-C run id must be {RUN_ID}")
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match the archived D2-C resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-C requires INFBAGEL_WORKER_EXPERT=hoi")
    cuda_device = torch.device(args.cuda_device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-C requires the HOI worker CUDA backend")
    assets = verify_assets()
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    positions, coverage = select_d2c_windows(dataset)
    cpu = replay_device(dataset, positions, torch.device("cpu"))
    if cpu["passed"]:
        cuda = replay_device(dataset, positions, cuda_device)
    else:
        cuda = {
            "device": str(cuda_device),
            "status": "skipped_due_to_cpu_gate",
            "reason": "D2-C requires CPU and CUDA to pass; CPU has unexplained basis points",
            "windows": 0,
            "basis_points": 0,
            "passed": None,
            "gpu_kernel_calls": 0,
        }
    passed = bool(cpu["passed"] and cuda["passed"] is True)
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-C",
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "available_windows": len(dataset.indices),
            "windows": len(positions),
            "coverage": coverage,
            "global_window_indices_sha256": EXPECTED_GLOBAL_SELECTION_SHA256,
            "sequence_window_sha256": EXPECTED_SEQUENCE_WINDOW_SHA256,
        },
        "assets": assets,
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "pytorch3d": pytorch3d_version,
            "trimesh": trimesh.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": (
                torch.cuda.get_device_name(cuda_device) if cpu["passed"]
                else "not queried after CPU gate failure"
            ),
        },
        "cpu": cpu,
        "cuda": cuda,
        "passed": passed,
        "classification": "geometric-equivalence-validated" if passed else "geometric-equivalence-unexplained-failure",
        "conditional_d2p_authorized": passed,
        "stored_per_frame_bps_use": "diagnostic_gt_replay_only",
        "sampler_stored_per_frame_bps": False,
        "sampler_future_gt": False,
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
