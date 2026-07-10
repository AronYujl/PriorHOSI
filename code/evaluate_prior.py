"""Deterministic held-out evaluation for independent HOI and HSI priors."""

import json
import os
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from prior_utils import (
    balanced_subset_indices,
    build_motion_state,
    extract_model_state,
    make_prefix_mask,
    move_training_batch,
    split_dataset_indices,
)
from utils import transform_points

os.environ.setdefault("ROOT_DIR", str(Path(__file__).resolve().parent.parent))


def _single_entry(config_group):
    if "_target_" in config_group:
        return config_group
    values = list(config_group.values())
    if len(values) != 1:
        raise ValueError(f"Expected one configured component, got {len(values)}")
    return values[0]


class MetricAccumulator:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(float)

    def add(self, name, value, count=1):
        if torch.is_tensor(value):
            value = value.detach().double().sum().item()
        self.sums[name] += float(value)
        self.counts[name] += float(count)

    def means(self):
        return {
            name: self.sums[name] / max(self.counts[name], 1.0)
            for name in sorted(self.sums)
        }


def _project_to_rotation(matrix):
    u, _, vh = torch.linalg.svd(matrix)
    rotation = u @ vh
    determinant = torch.det(rotation)
    correction = torch.ones(*matrix.shape[:-2], 3, device=matrix.device, dtype=matrix.dtype)
    correction[..., -1] = torch.where(determinant < 0, -1.0, 1.0)
    return u @ torch.diag_embed(correction) @ vh


def _rotation_geodesic_degrees(predicted, target):
    predicted = _project_to_rotation(predicted)
    target = _project_to_rotation(target)
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)))


def _predict_x0(model, trainer, dataset, b, state, timestep, noise):
    mask = make_prefix_mask(state, trainer.auto_regre_num)
    noise = noise.clone()
    noise[mask] = 0.0
    timesteps = torch.full(
        (state.shape[0],), int(timestep), device=state.device, dtype=torch.long
    )
    noisy = trainer.q_sample(state, timesteps, noise)
    noisy[mask] = state[mask]

    if dataset.load_scene:
        occ, occ_list, occ_pos = trainer._compute_occ(
            noisy, state, b["joints"], b["mat"], b["scene_flag"],
            b["object_points"], b["pelvis_goal"], b["scene_goal"],
            b["object_goal"], b["is_loco"], b["is_object"],
            b["need_pelvis_dir"], b["obj_rot_mat_ref"],
        )
    else:
        occ = occ_list = occ_pos = None

    predicted = model(
        noisy, occ, timesteps, b["text_clip_embedding"], b["pelvis_goal"],
        b["scene_goal"], b["is_loco"], b["need_scene"],
        b["need_pelvis_dir"], b["pi"], b["end_pi"], b["seg_len"],
        b["need_pi"], b["object_goal"], b["is_object"],
        b["obj_bps_data"], occ_list, occ_pos,
    )
    return predicted, (noisy, occ, occ_list, occ_pos, timesteps)


def _human_metrics(metrics, prefix, predicted, target, b, dataset):
    batch_size = predicted.shape[0]
    pred_joints = dataset.denormalize_torch(predicted[..., :84]).reshape(
        batch_size, predicted.shape[1], 28, 3
    )
    target_joints = dataset.denormalize_torch(target[..., :84]).reshape_as(pred_joints)
    mpjpe = torch.linalg.vector_norm(pred_joints - target_joints, dim=-1)
    metrics.add(f"{prefix}/human_mpjpe_m", mpjpe, mpjpe.numel())

    rotation_l1 = torch.abs(predicted[..., 84:216] - target[..., 84:216])
    metrics.add(f"{prefix}/human_rotation_l1", rotation_l1, rotation_l1.numel())

    active_goal = b["need_pelvis_dir"].bool()
    if active_goal.any():
        desired_goal = torch.where(
            b["is_loco"].unsqueeze(-1), b["pelvis_goal"], b["scene_goal"]
        )
        goal_error = torch.linalg.vector_norm(
            pred_joints[:, -1, 0] - desired_goal, dim=-1
        )
        metrics.add(
            f"{prefix}/pelvis_goal_error_m",
            goal_error[active_goal],
            active_goal.sum().item(),
        )

    if dataset.load_scene:
        global_joints = transform_points(
            pred_joints.reshape(batch_size, predicted.shape[1], -1), b["mat"]
        ).reshape_as(pred_joints)
        occupancy = dataset.get_occ_for_points(global_joints, None, b["scene_flag"])
        penetrating = occupancy == 1
        metrics.add(
            f"{prefix}/scene_joint_penetration_rate",
            penetrating.float(),
            penetrating.numel(),
        )
    return pred_joints, target_joints


def _hoi_metrics(metrics, prefix, predicted, target, b, dataset,
                 pred_joints, target_joints):
    pred_object = dataset.denormalize_torch(
        predicted[..., 216:219], is_object=True
    )
    target_object = dataset.denormalize_torch(
        target[..., 216:219], is_object=True
    )
    object_error = torch.linalg.vector_norm(pred_object - target_object, dim=-1)
    metrics.add(f"{prefix}/object_translation_error_m", object_error, object_error.numel())

    pred_rotation = predicted[..., 219:228].reshape(*predicted.shape[:2], 3, 3)
    target_rotation = target[..., 219:228].reshape_as(pred_rotation)
    rotation_error = _rotation_geodesic_degrees(pred_rotation, target_rotation)
    metrics.add(f"{prefix}/object_rotation_error_deg", rotation_error, rotation_error.numel())

    contact_target = target[..., 228:232] > 0.5
    contact_predicted = predicted[..., 228:232] > 0.5
    contact_mae = torch.abs(predicted[..., 228:232] - target[..., 228:232])
    metrics.add(f"{prefix}/contact_mae", contact_mae, contact_mae.numel())
    true_positive = (contact_predicted & contact_target).sum().item()
    false_positive = (contact_predicted & ~contact_target).sum().item()
    false_negative = (~contact_predicted & contact_target).sum().item()
    metrics.add(f"{prefix}/contact_tp", true_positive)
    metrics.add(f"{prefix}/contact_fp", false_positive)
    metrics.add(f"{prefix}/contact_fn", false_negative)

    active_object = b["is_object"].bool()
    if active_object.any():
        object_goal_error = torch.linalg.vector_norm(
            pred_object[:, -1] - b["object_goal"], dim=-1
        )
        metrics.add(
            f"{prefix}/object_goal_error_m",
            object_goal_error[active_object],
            active_object.sum().item(),
        )

    hand_indices = [20, 21, 25, 27]
    pred_relative = pred_joints[..., hand_indices, :] - pred_object.unsqueeze(-2)
    target_relative = target_joints[..., hand_indices, :] - target_object.unsqueeze(-2)
    relative_error = torch.linalg.vector_norm(pred_relative - target_relative, dim=-1)
    if contact_target.any():
        metrics.add(
            f"{prefix}/contact_relative_error_m",
            relative_error[contact_target],
            contact_target.sum().item(),
        )


def _finalize_contact_f1(results):
    prefixes = {
        key.rsplit("/", 1)[0]
        for key in results
        if key.endswith("/contact_tp")
    }
    for prefix in prefixes:
        tp = results.pop(f"{prefix}/contact_tp")
        fp = results.pop(f"{prefix}/contact_fp")
        fn = results.pop(f"{prefix}/contact_fn")
        results[f"{prefix}/contact_f1"] = 2 * tp / max(2 * tp + fp + fn, 1.0)


@hydra.main(version_base=None, config_path="config", config_name="config_eval_hoi_prior")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    dataset = hydra.utils.instantiate(cfg.dataset)
    if cfg.evaluation_split == "all":
        indices = list(range(len(dataset)))
    else:
        train_indices, validation_indices = split_dataset_indices(
            dataset, cfg.validation.fraction, cfg.validation.split_unit,
            cfg.validation.split_seed,
        )
        indices = validation_indices if cfg.evaluation_split == "validation" else train_indices

    num_available_windows = len(indices)
    if cfg.max_batches > 0:
        indices = balanced_subset_indices(
            dataset,
            indices,
            split_unit=cfg.validation.split_unit,
            max_items=cfg.max_batches * cfg.batch_size,
            seed=cfg.noise_seed,
        )

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = hydra.utils.instantiate(_single_entry(cfg.model)).to(device)
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    checkpoint_prior = checkpoint.get("prior_type") if isinstance(checkpoint, dict) else None
    if checkpoint_prior and checkpoint_prior != cfg.prior_type:
        raise ValueError(
            f"Checkpoint contains {checkpoint_prior!r} prior, expected {cfg.prior_type!r}"
        )
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    model.eval()

    trainer = hydra.utils.instantiate(_single_entry(cfg.sampler))
    trainer.set_dataset_and_model(dataset, model)
    metrics = MetricAccumulator()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if cfg.max_batches > 0 and batch_index >= cfg.max_batches:
                break
            b = move_training_batch(batch, device)
            target = build_motion_state(b, cfg.prior_type)
            generator = torch.Generator(device=device)
            generator.manual_seed(cfg.noise_seed + batch_index)
            noise = torch.randn(target.shape, generator=generator, device=device)

            for timestep in cfg.timesteps:
                predicted, prediction_inputs = _predict_x0(
                    model, trainer, dataset, b, target, timestep, noise
                )
                prefix = f"t{int(timestep):03d}"
                pred_joints, target_joints = _human_metrics(
                    metrics, prefix, predicted, target, b, dataset
                )
                if cfg.prior_type == "hoi":
                    _hoi_metrics(
                        metrics, prefix, predicted, target, b, dataset,
                        pred_joints, target_joints,
                    )
                else:
                    noisy, occ, occ_list, occ_pos, timesteps = prediction_inputs
                    no_scene = torch.zeros_like(b["need_scene"], dtype=torch.bool)
                    ablated = model(
                        noisy, occ, timesteps, b["text_clip_embedding"],
                        b["pelvis_goal"], b["scene_goal"], b["is_loco"],
                        no_scene, b["need_pelvis_dir"], b["pi"], b["end_pi"],
                        b["seg_len"], b["need_pi"], b["object_goal"],
                        b["is_object"], b["obj_bps_data"], occ_list, occ_pos,
                    )
                    scene_effect = dataset.denormalize_torch(
                        predicted[..., :84]
                    ).reshape(*predicted.shape[:2], 28, 3) - dataset.denormalize_torch(
                        ablated[..., :84]
                    ).reshape(*predicted.shape[:2], 28, 3)
                    scene_effect = torch.linalg.vector_norm(scene_effect, dim=-1)
                    metrics.add(
                        f"{prefix}/scene_condition_effect_m",
                        scene_effect,
                        scene_effect.numel(),
                    )

            if cfg.sampling.enabled and batch_index < cfg.sampling.max_batches:
                generated_steps, _ = trainer.p_sample_loop(
                    target[:, :cfg.auto_regre_num],
                    b["mat"], b["scene_flag"], b["text_clip_embedding"],
                    b["pelvis_goal"], b["scene_goal"], b["object_goal"],
                    b["need_scene"], b["need_pelvis_dir"], b["pi"],
                    b["end_pi"], b["seg_len"], b["need_pi"], b["is_loco"],
                    b["is_object"], b["obj_bps_data"], b["object_points"],
                    b["obj_rot_mat_ref"],
                    getattr(dataset, "obj_rest_verts", {}),
                    list(batch["seq_name"]),
                    obj_rot_mat_prefix=None,
                    object_only=False,
                )
                generated = generated_steps[-1]
                generated_joints, target_joints = _human_metrics(
                    metrics, "sample", generated, target, b, dataset
                )
                if cfg.prior_type == "hoi":
                    _hoi_metrics(
                        metrics, "sample", generated, target, b, dataset,
                        generated_joints, target_joints,
                    )

    results = metrics.means()
    _finalize_contact_f1(results)
    report = {
        "prior_type": cfg.prior_type,
        "checkpoint": str(Path(cfg.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "evaluation_split": cfg.evaluation_split,
        "num_available_windows": num_available_windows,
        "num_selected_windows": len(indices),
        "num_evaluated_batches": min(len(loader), cfg.max_batches) if cfg.max_batches > 0 else len(loader),
        "timesteps": [int(t) for t in cfg.timesteps],
        "sampling": OmegaConf.to_container(cfg.sampling, resolve=True),
        "metrics": results,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output_path = Path(cfg.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
