"""Shared state, split, reproducibility, and checkpoint utilities for priors."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.priors import PRIOR_SPECS


TRAIN_BATCH_KEYS = (
    "joints", "mat", "object_trans", "object_rot_mat", "scene_flag",
    "text_clip_embedding", "pelvis_goal", "scene_goal", "object_goal",
    "need_scene", "need_pelvis_dir", "pi", "need_pi", "is_loco",
    "is_object", "obj_bps_data", "obj_rot_mat_ref",
    "rest_pose_obj_nn_pts", "transformed_obj_verts", "object_points",
    "global_rot_6d", "contact_label", "rest_human_offsets", "seg_len",
    "end_pi",
)


class DeterministicSubset(Dataset):
    """Subset whose Python/NumPy data augmentation is stable per source index."""

    def __init__(self, dataset, indices, seed):
        self.dataset = dataset
        self.indices = list(indices)
        self.seed = int(seed)

    def __getitem__(self, item):
        source_index = int(self.indices[item])
        digest = hashlib.sha256(
            f"{self.seed}:{source_index}".encode("utf-8")
        ).digest()
        item_seed = int.from_bytes(digest[:4], "big")
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        random.seed(item_seed)
        np.random.seed(item_seed)
        try:
            return self.dataset[source_index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def __len__(self):
        return len(self.indices)


def format_duration(seconds):
    seconds = max(0, int(round(float(seconds))))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def get_prior_spec(prior_type):
    try:
        return PRIOR_SPECS[str(prior_type).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown prior type: {prior_type!r}") from exc


def build_motion_state(batch, prior_type):
    spec = get_prior_spec(prior_type)
    global_rot_6d = batch["global_rot_6d"].reshape(
        batch["global_rot_6d"].shape[0], batch["global_rot_6d"].shape[1], -1
    )
    human_state = torch.cat([batch["joints"], global_rot_6d], dim=-1)
    if spec.uses_object:
        state = torch.cat(
            [
                human_state,
                batch["object_trans"],
                batch["object_rot_mat"].reshape(
                    batch["object_rot_mat"].shape[0],
                    batch["object_rot_mat"].shape[1],
                    -1,
                ),
                batch["contact_label"],
            ],
            dim=-1,
        )
    else:
        state = human_state

    if state.shape[-1] != spec.state_dim:
        raise ValueError(
            f"Built {state.shape[-1]} channels for {prior_type}, expected {spec.state_dim}"
        )
    return state.float()


def move_training_batch(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in TRAIN_BATCH_KEYS
    }


def make_prefix_mask(state, fixed_frames):
    mask = torch.zeros_like(state, dtype=torch.bool)
    if fixed_frames > 0:
        mask[:, :fixed_frames] = True
    return mask


def seed_everything(seed, rank=0, deterministic=False):
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    return effective_seed


def _stable_unit_interval(value, seed):
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _window_group_ids(dataset, split_unit):
    sequence_ids = np.asarray(dataset.ori_sequence_idx, dtype=np.int64)
    if split_unit == "sequence":
        return sequence_ids
    if split_unit != "scene":
        raise ValueError("split_unit must be 'sequence' or 'scene'")

    sequence_starts = np.asarray(dataset.ori_sequence_start_idx, dtype=np.int64)
    scene_by_sequence = [dataset.scene_name[int(i)] for i in sequence_starts]
    scene_to_id = {
        scene_name: int.from_bytes(
            hashlib.sha256(str(scene_name).encode("utf-8")).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        for scene_name in set(scene_by_sequence)
    }
    encoded = np.asarray([scene_to_id[name] for name in scene_by_sequence], dtype=np.int64)
    return encoded[sequence_ids]


def split_dataset_indices(dataset, val_fraction, split_unit, seed):
    """Split windows by source sequence or scene, never by individual window."""
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must lie strictly between 0 and 1")

    groups = _window_group_ids(dataset, split_unit)
    unique_groups = np.unique(groups)
    val_groups = {
        group
        for group in unique_groups.tolist()
        if _stable_unit_interval(group, seed) < float(val_fraction)
    }
    if not val_groups or len(val_groups) == len(unique_groups):
        ranked = sorted(unique_groups.tolist(), key=lambda x: _stable_unit_interval(x, seed))
        val_count = min(max(1, round(len(ranked) * float(val_fraction))), len(ranked) - 1)
        val_groups = set(ranked[:val_count])

    is_val = np.isin(groups, np.fromiter(val_groups, dtype=groups.dtype))
    train_indices = np.flatnonzero(~is_val).tolist()
    val_indices = np.flatnonzero(is_val).tolist()
    return train_indices, val_indices


def balanced_subset_indices(dataset, indices, split_unit, max_items, seed):
    """Select a deterministic group-balanced subset from preselected windows."""
    if max_items <= 0 or len(indices) <= max_items:
        return list(indices)

    indices_array = np.asarray(indices, dtype=np.int64)
    groups = _window_group_ids(dataset, split_unit)[indices_array]
    rng = np.random.RandomState(int(seed))
    queues = []
    for group in sorted(np.unique(groups).tolist()):
        group_indices = indices_array[groups == group].copy()
        rng.shuffle(group_indices)
        queues.append(group_indices.tolist())

    selected = []
    offset = 0
    while len(selected) < max_items:
        made_progress = False
        for queue in queues:
            if offset < len(queue):
                selected.append(queue[offset])
                made_progress = True
                if len(selected) == max_items:
                    break
        if not made_progress:
            break
        offset += 1
    return selected


def array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def dataset_contract(dataset, prior_type, split_unit, val_fraction, split_seed):
    spec = get_prior_spec(prior_type)
    norm = np.stack([dataset.min, dataset.max])
    contract = {
        "prior": spec.to_dict(),
        "folder": str(Path(dataset.folder).resolve()),
        "window_size": int(dataset.max_window_size),
        "step": int(dataset.step),
        "num_windows": int(len(dataset)),
        "num_source_sequences": int(len(dataset.ori_sequence_start_idx)),
        "human_norm_sha256": array_sha256(norm),
        "split_unit": str(split_unit),
        "validation_fraction": float(val_fraction),
        "split_seed": int(split_seed),
    }
    if spec.uses_scene:
        contract["num_scenes"] = int(len({
            dataset.scene_name[int(start)]
            for start in dataset.ori_sequence_start_idx
        }))
    return contract


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def save_checkpoint(path, model, optimizer, scaler, epoch, global_step, cfg,
                    prior_type, data_contract, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_unwrapped = unwrap_model(model)
    payload = {
        "schema_version": 1,
        "prior_type": str(prior_type),
        "prior_spec": get_prior_spec(prior_type).to_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model_unwrapped.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": cfg,
        "dataset_contract": data_contract,
        "metrics": metrics or {},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_training_checkpoint(path, model, optimizer=None, scaler=None,
                             expected_prior=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    if expected_prior and isinstance(checkpoint, dict):
        actual_prior = checkpoint.get("prior_type")
        if actual_prior is not None and actual_prior != expected_prior:
            raise ValueError(
                f"Checkpoint prior is {actual_prior!r}, expected {expected_prior!r}"
            )
    missing, unexpected = unwrap_model(model).load_state_dict(
        extract_model_state(checkpoint), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    if isinstance(checkpoint, dict):
        if optimizer is not None and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint
    return {"epoch": -1, "global_step": 0}
