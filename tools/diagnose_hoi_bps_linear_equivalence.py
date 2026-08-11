#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-E linear BPS equivalence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import trimesh
from pytorch3d import __version__ as pytorch3d_version


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.hoi.data import PriorWindowDataset  # noqa: E402
from priors.hoi.remediation import selection_sha256  # noqa: E402
from priors.core.window_codec import BPS_SHA256  # noqa: E402
from tools.diagnose_hoi_bps_equivalence import (  # noqa: E402
    PLY_SHA256,
    exclusive_json,
    replay_device,
    verify_assets,
)


RUN_ID = "p1-hoi-d2e-bps-linear-equivalence-r1-s42-20260715"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
DISCLOSED_WINDOWS_PER_CLASS = 128
FRESH_WINDOWS_PER_CLASS = 64
EXPECTED_DISCLOSED_WINDOWS = 1664
EXPECTED_FRESH_WINDOWS = 832
EXPECTED_COMBINED_WINDOWS = 2496
STRICT_COMPONENT_MAX_ABS = 1e-4
STORED_MESH_RESIDUAL_M_MAX = 1e-6
RECOMPUTED_MESH_RESIDUAL_M_MAX = 1e-6
NEAREST_LINEAR_DISTANCE_GAP_M_MAX = 2.5e-7
EXPECTED_HASHES = {
    "disclosed_calibration": {
        "global": "e58bc72326ec4ec193b7e8371c9a034f64d09761f188ee863f1ccd63ef21bf87",
        "sequence_window": "f629e28cb4b277ad53c3bae4df96726a5224457fbbb1c779b4214de8385392ad",
    },
    "fresh_holdout": {
        "global": "44fdc7154902c922310f54ad2eb97d26ca710902d5c9d76c354e50f650e28316",
        "sequence_window": "70301049201e3c945570f73103d6313f166f592b615908001b41fcc513016b91",
    },
    "combined": {
        "global": "bdf93ebf796baa345f163194aa13b1720e1410d4b1c62c19a29c3a88ee40dc69",
        "sequence_window": "258254cf9e277b7e7f40f8fed087a74c128ecfaf7c6279bd47df662066b42e3c",
    },
}


def _selection_hashes(dataset, positions: Sequence[int]) -> Dict[str, str]:
    global_hash = selection_sha256(int(dataset.indices[position]) for position in positions)
    payload = "\n".join(
        f"{dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])]}:"
        f"{int(dataset.indices[position])}"
        for position in positions
    )
    return {
        "global": global_hash,
        "sequence_window": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def select_d2e_windows(dataset) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, int]]]:
    """Return disclosed rank-0:128 calibration and untouched rank-128:192 holdout."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-E selection is internal-validation only")
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
        raise ValueError(f"unexpected D2-E object classes: {sorted(groups)}")
    first: List[int] = []
    second: List[int] = []
    fresh: List[int] = []
    coverage = {"disclosed_calibration": {}, "fresh_holdout": {}}
    for object_name in sorted(groups):
        ranked = sorted(groups[object_name])
        if len(ranked) < DISCLOSED_WINDOWS_PER_CLASS + FRESH_WINDOWS_PER_CLASS:
            raise ValueError(f"only {len(ranked)} windows for {object_name}")
        first.extend(value[1] for value in ranked[:FRESH_WINDOWS_PER_CLASS])
        second.extend(value[1] for value in ranked[FRESH_WINDOWS_PER_CLASS:DISCLOSED_WINDOWS_PER_CLASS])
        fresh.extend(value[1] for value in ranked[DISCLOSED_WINDOWS_PER_CLASS:192])
        coverage["disclosed_calibration"][object_name] = DISCLOSED_WINDOWS_PER_CLASS
        coverage["fresh_holdout"][object_name] = FRESH_WINDOWS_PER_CLASS
    disclosed = first + second
    subsets = {"disclosed_calibration": disclosed, "fresh_holdout": fresh}
    if len(disclosed) != EXPECTED_DISCLOSED_WINDOWS or len(fresh) != EXPECTED_FRESH_WINDOWS:
        raise ValueError("D2-E subset window count mismatch")
    if set(disclosed) & set(fresh):
        raise ValueError("D2-E disclosed calibration and fresh holdout overlap")
    combined = disclosed + fresh
    if len(combined) != EXPECTED_COMBINED_WINDOWS:
        raise ValueError("D2-E combined window count mismatch")
    observed = {
        "disclosed_calibration": _selection_hashes(dataset, disclosed),
        "fresh_holdout": _selection_hashes(dataset, fresh),
        "combined": _selection_hashes(dataset, combined),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"D2-E selection hash mismatch: {observed}")
    return subsets, coverage


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B-D2-E",
        "mode": "immutable-mesh-bps-linear-equivalence-audit",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "partition": "internal_validation",
        "selection": {
            "object_classes": 13,
            "disclosed_windows_per_class": DISCLOSED_WINDOWS_PER_CLASS,
            "fresh_windows_per_class": FRESH_WINDOWS_PER_CLASS,
            "disclosed_calibration_windows": EXPECTED_DISCLOSED_WINDOWS,
            "fresh_holdout_windows": EXPECTED_FRESH_WINDOWS,
            "combined_windows": EXPECTED_COMBINED_WINDOWS,
            "fresh_holdout_rank_range_zero_based": [128, 191],
            "hashes": EXPECTED_HASHES,
        },
        "gate": {
            "strict_component_max_abs": STRICT_COMPONENT_MAX_ABS,
            "stored_mesh_residual_m_max": STORED_MESH_RESIDUAL_M_MAX,
            "recomputed_mesh_residual_m_max": RECOMPUTED_MESH_RESIDUAL_M_MAX,
            "nearest_linear_distance_gap_m_max": NEAREST_LINEAR_DISTANCE_GAP_M_MAX,
            "nearest_squared_distance_gap_m2": "report_only",
        },
        "devices": ["cpu", args.cuda_device],
        "execution_order": "CPU disclosed+fresh first; stop before CUDA on any CPU failure",
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


def _run_subsets(dataset, subsets, device) -> Dict[str, object]:
    expected = {
        "disclosed_calibration": DISCLOSED_WINDOWS_PER_CLASS,
        "fresh_holdout": FRESH_WINDOWS_PER_CLASS,
    }
    results = {
        name: replay_device(
            dataset,
            positions,
            device,
            nearest_squared_distance_gap_m2_max=float("inf"),
            nearest_linear_distance_gap_m_max=NEAREST_LINEAR_DISTANCE_GAP_M_MAX,
            expected_windows_per_class=expected[name],
        )
        for name, positions in subsets.items()
    }
    results["passed"] = all(bool(result["passed"]) for result in results.values())
    return results


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
        raise ValueError(f"D2-E run id must be {RUN_ID}")
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match the archived D2-E resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-E requires INFBAGEL_WORKER_EXPERT=hoi")
    cuda_device = torch.device(args.cuda_device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-E requires the HOI worker CUDA backend")
    assets = verify_assets()
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    subsets, coverage = select_d2e_windows(dataset)
    cpu = _run_subsets(dataset, subsets, torch.device("cpu"))
    if cpu["passed"]:
        cuda = _run_subsets(dataset, subsets, cuda_device)
    else:
        cuda = {
            "status": "skipped_due_to_cpu_gate",
            "reason": "D2-E requires both CPU subsets to pass before CUDA",
            "disclosed_calibration_windows": 0,
            "fresh_holdout_windows": 0,
            "basis_points": 0,
            "gpu_kernel_calls": 0,
            "passed": None,
        }
    passed = bool(cpu["passed"] and cuda["passed"] is True)
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-E",
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "available_windows": len(dataset.indices),
            "coverage": coverage,
            "hashes": EXPECTED_HASHES,
            "overlap": 0,
        },
        "gate": config["gate"],
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
        "classification": (
            "linear-geometric-equivalence-validated"
            if passed else "linear-geometric-equivalence-failure"
        ),
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
