"""Compare exported direct positions with the deployed SMPL-X reconstruction."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from utils import create_smplx_model, interpolate_joints, run_smplx_model
from priors.hsi.visualization import _motion_paths


# The direct-position data channel uses index1/ring1, including SMPL-X 34/49.
POSITION_JOINTS_28 = list(range(22)) + [23, 24, 25, 34, 40, 49]


def position_fk_metrics(direct, reconstructed, seams, scale=3):
    """Distances in metres, on the existing deployment interpolation grid."""
    error = torch.linalg.vector_norm(direct - reconstructed, dim=-1)
    relative = torch.linalg.vector_norm(
        (direct[:, 1:22] - direct[:, :1])
        - (reconstructed[:, 1:22] - reconstructed[:, :1]), dim=-1,
    )
    boundary = torch.zeros(len(direct), device=direct.device, dtype=torch.bool)
    for seam in seams:
        boundary[int(seam) * scale:(int(seam) + 2) * scale] = True
    result = {
        "root_relative_body_m": relative.mean().item(),
        "body_mpjpe_m": error[:, :22].mean().item(),
        "root_error_m": error[:, 0].mean().item(),
        "matched28_mpjpe_m": error.mean().item(),
        "body_point_error_p95_m": torch.quantile(error[:, :22].flatten(), .95).item(),
        "interior_root_relative_body_m": relative[~boundary].mean().item(),
        "boundary_root_relative_body_m": relative[boundary].mean().item() if boundary.any() else None,
    }
    return result, error.mean(dim=0).tolist()


def evaluate_position_fk(cfg):
    device = torch.device("cuda:0")
    shard, count = int(cfg.representation_shard_index), int(cfg.representation_shard_count)
    batch = int(cfg.representation_batch_size)
    paths = {arm: _motion_paths(root) for arm, root in cfg.representation_inputs.items()}
    ids = sorted(paths["gt"])[shard::count]
    models, records, timings = {}, {}, []
    full_batches = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        for arm, arm_paths in paths.items():
            records[arm] = {}
            for sequence_id in ids:
                with np.load(arm_paths[sequence_id]) as data:
                    gender = str(data["gender"])
                    if gender not in models:
                        models[gender] = create_smplx_model(gender, device)
                    pose = torch.as_tensor(np.concatenate(
                        [data["global_orient"][:, None], data["body_pose"]], axis=1
                    ), device=device, dtype=torch.float32)
                    transl = torch.as_tensor(data["transl"], device=device, dtype=torch.float32)
                    betas = torch.as_tensor(data["betas"], device=device, dtype=torch.float32)
                    rebuilt = []
                    for start in range(0, len(pose), batch):
                        part = pose[start:start + batch]
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        vertices, joints = run_smplx_model(
                            part, transl[start:start + batch], betas, gender,
                            joints_ind=POSITION_JOINTS_28, smpl_model=models[gender],
                        )
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - before
                        if len(part) == batch:
                            full_batches += 1
                            if full_batches > 4:
                                timings.append(elapsed)
                        rebuilt.append(joints)
                        del vertices
                    scale = int(data["interp_scale"])
                    direct = interpolate_joints(torch.as_tensor(
                        data["global_jpos"], device=device, dtype=torch.float32
                    ), scale).reshape(-1, 28, 3)
                    values, joint_means = position_fk_metrics(
                        direct, torch.cat(rebuilt), data["seams"], scale,
                    )
                    records[arm][sequence_id] = dict(
                        values, per_joint_mean_m=joint_means, frames=len(direct),
                    )
            print("Completed %s: %d sequences" % (arm, len(ids)), flush=True)
    output = Path(cfg.representation_output) / ("shard%02d.json" % shard)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        metrics=records, joint_indices=POSITION_JOINTS_28,
        shard_index=shard, shard_count=count, batch_size=batch,
        timing=dict(warmup_full_batches=4, measured_full_batches=len(timings),
                    mean_batch_seconds=float(np.mean(timings)),
                    frames_per_second=batch / float(np.mean(timings)),
                    wall_seconds=time.perf_counter() - started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()),
    )
    with output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return output


def merge_position_fk(cfg):
    root = Path(cfg.representation_output)
    shards = [json.loads((root / ("shard%02d.json" % i)).read_text())
              for i in range(int(cfg.representation_shard_count))]
    for arm in cfg.representation_inputs:
        records = {}
        for shard in shards:
            records.update(shard["metrics"][arm])
        arm_root = root / arm
        arm_root.mkdir()
        with (arm_root / "per_sequence_metrics.json").open("x") as handle:
            json.dump(dict(metrics=records), handle, indent=2, allow_nan=False)
    output = root / "summary.json"
    with output.open("x") as handle:
        json.dump(dict(shards=[s["timing"] for s in shards],
                       counts={arm: sum(len(s["metrics"][arm]) for s in shards)
                               for arm in cfg.representation_inputs}), handle, indent=2)
    return output
