#!/usr/bin/env python3
"""Run the preregistered Phase 1B D2-D BPS tolerance calibration audit."""

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

from priors.data import PriorWindowDataset  # noqa: E402
from priors.remediation import selection_sha256  # noqa: E402
from priors.window_codec import BPS_SHA256  # noqa: E402
from tools.diagnose_hoi_bps_equivalence import (  # noqa: E402
    PLY_SHA256,
    exclusive_json,
    replay_device,
    verify_assets,
)


RUN_ID = "p1-hoi-d2d-bps-tolerance-s42-20260715"
EXPECTED_DATA_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
WINDOWS_PER_CLASS_PER_SUBSET = 64
EXPECTED_SUBSET_WINDOWS = 832
EXPECTED_COMBINED_WINDOWS = 1664
STRICT_COMPONENT_MAX_ABS = 1e-4
STORED_MESH_RESIDUAL_M_MAX = 1e-6
RECOMPUTED_MESH_RESIDUAL_M_MAX = 1e-6
NEAREST_SQUARED_DISTANCE_GAP_M2_MAX = 2.5e-7
NEAREST_LINEAR_DISTANCE_GAP_M_MAX = 2.5e-7
EXPECTED_HASHES = {
    "calibration": {
        "global": "5f13844f9c3c1540d89d19b304e484cba6e84cc8adcb7276ebf4d17fb803db72",
        "sequence_window": "e7827e83b88058e9d87dc4d56ccdbfe5929ea245645356fab278364b1aae1f38",
    },
    "holdout": {
        "global": "750378d6933a6e190ceebfe582b00fac16b403dad853bd9c30f2bfe0b8fdc00a",
        "sequence_window": "f2d8fbc4ee42150727a981fbca1b7ef45b9b0902cfb64f859d797bc8ce9a944e",
    },
    "combined": {
        "global": "e58bc72326ec4ec193b7e8371c9a034f64d09761f188ee863f1ccd63ef21bf87",
        "sequence_window": "f629e28cb4b277ad53c3bae4df96726a5224457fbbb1c779b4214de8385392ad",
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


def select_d2d_windows(dataset) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, int]]]:
    """Select the locked D2-C calibration set and its disjoint rank-64:128 holdout."""
    if getattr(dataset, "partition", None) != "internal_validation":
        raise ValueError("D2-D selection is internal-validation only")
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
        raise ValueError(f"unexpected D2-D object classes: {sorted(groups)}")
    subsets = {"calibration": [], "holdout": []}
    coverage = {"calibration": {}, "holdout": {}}
    for object_name in sorted(groups):
        ranked = sorted(groups[object_name])
        if len(ranked) < 2 * WINDOWS_PER_CLASS_PER_SUBSET:
            raise ValueError(f"only {len(ranked)} windows for {object_name}")
        chosen = {
            "calibration": ranked[:WINDOWS_PER_CLASS_PER_SUBSET],
            "holdout": ranked[
                WINDOWS_PER_CLASS_PER_SUBSET:2 * WINDOWS_PER_CLASS_PER_SUBSET
            ],
        }
        for subset_name, values in chosen.items():
            subsets[subset_name].extend(value[1] for value in values)
            coverage[subset_name][object_name] = len(values)
    if any(len(values) != EXPECTED_SUBSET_WINDOWS for values in subsets.values()):
        raise ValueError("D2-D subset window count mismatch")
    if set(subsets["calibration"]) & set(subsets["holdout"]):
        raise ValueError("D2-D calibration and holdout overlap")
    combined = subsets["calibration"] + subsets["holdout"]
    if len(combined) != EXPECTED_COMBINED_WINDOWS:
        raise ValueError("D2-D combined window count mismatch")
    observed = {
        "calibration": _selection_hashes(dataset, subsets["calibration"]),
        "holdout": _selection_hashes(dataset, subsets["holdout"]),
        "combined": _selection_hashes(dataset, combined),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"D2-D selection hash mismatch: {observed}")
    return subsets, coverage


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "p1",
        "subphase": "1B-D2-D",
        "mode": "immutable-mesh-bps-dual-tolerance-audit",
        "run_id": args.run_id,
        "seed": 42,
        "repo_root": str(REPO),
        "partition": "internal_validation",
        "selection": {
            "object_classes": 13,
            "windows_per_class_per_subset": WINDOWS_PER_CLASS_PER_SUBSET,
            "calibration_windows": EXPECTED_SUBSET_WINDOWS,
            "holdout_windows": EXPECTED_SUBSET_WINDOWS,
            "combined_windows": EXPECTED_COMBINED_WINDOWS,
            "holdout_rank_range_zero_based": [64, 127],
            "hashes": EXPECTED_HASHES,
        },
        "gate": {
            "strict_component_max_abs": STRICT_COMPONENT_MAX_ABS,
            "stored_mesh_residual_m_max": STORED_MESH_RESIDUAL_M_MAX,
            "recomputed_mesh_residual_m_max": RECOMPUTED_MESH_RESIDUAL_M_MAX,
            "nearest_squared_distance_gap_m2_max": NEAREST_SQUARED_DISTANCE_GAP_M2_MAX,
            "nearest_linear_distance_gap_m_max": NEAREST_LINEAR_DISTANCE_GAP_M_MAX,
        },
        "devices": ["cpu", args.cuda_device],
        "execution_order": "CPU calibration+holdout first; stop before CUDA on any CPU failure",
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
    results = {
        name: replay_device(
            dataset,
            positions,
            device,
            nearest_squared_distance_gap_m2_max=NEAREST_SQUARED_DISTANCE_GAP_M2_MAX,
            nearest_linear_distance_gap_m_max=NEAREST_LINEAR_DISTANCE_GAP_M_MAX,
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
        raise ValueError(f"D2-D run id must be {RUN_ID}")
    config = resolved_config(args)
    config_path = Path(args.resolved_config).resolve()
    if args.resolve_only:
        exclusive_json(config_path, config)
        return
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("runtime arguments do not match the archived D2-D resolved config")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi":
        raise RuntimeError("D2-D requires INFBAGEL_WORKER_EXPERT=hoi")
    cuda_device = torch.device(args.cuda_device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-D requires the HOI worker CUDA backend")
    assets = verify_assets()
    dataset = PriorWindowDataset(
        str(REPO), "hoi", partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    subsets, coverage = select_d2d_windows(dataset)
    cpu = _run_subsets(dataset, subsets, torch.device("cpu"))
    if cpu["passed"]:
        cuda = _run_subsets(dataset, subsets, cuda_device)
    else:
        cuda = {
            "status": "skipped_due_to_cpu_gate",
            "reason": "D2-D requires both CPU subsets to pass before CUDA",
            "calibration_windows": 0,
            "holdout_windows": 0,
            "basis_points": 0,
            "gpu_kernel_calls": 0,
            "passed": None,
        }
    passed = bool(cpu["passed"] and cuda["passed"] is True)
    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": "1B-D2-D",
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
            "dual-tolerance-geometric-equivalence-validated"
            if passed else "dual-tolerance-geometric-equivalence-failure"
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
