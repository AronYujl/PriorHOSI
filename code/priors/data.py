"""Lean real-data windows for Phase 1A expert smoke updates."""

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from pytorch3d import transforms
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from datasets.utils import get_smpl_parents, zup_to_yup
from .representation import REPRESENTATION


def hsi_filter(left, right) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("hand-interaction arrays have different shapes")
    return (left == -1) & (right == -1)


def partition_for_scenes(split: Dict[str, object], scenes: np.ndarray) -> np.ndarray:
    train = set(split["train"]["scenes"])
    validation = set(split["validation"]["scenes"])
    if train & validation:
        raise ValueError("scene leakage in split")
    values = np.full(len(scenes), "unassigned", dtype=object)
    values[np.isin(scenes, list(train))] = "train"
    values[np.isin(scenes, list(validation))] = "validation"
    if np.any(values == "unassigned"):
        missing = sorted(set(scenes[values == "unassigned"].tolist()))
        raise ValueError(f"scenes absent from locked split: {missing[:10]}")
    return values


def _rotation_6d(root_orient: np.ndarray, pose: np.ndarray, shift: np.ndarray, parents: np.ndarray) -> torch.Tensor:
    root = zup_to_yup(root_orient.copy()).reshape(-1, 1, 3)
    body = zup_to_yup(pose.copy().reshape(-1, 3)).reshape(-1, 21, 3)
    local = transforms.axis_angle_to_matrix(torch.from_numpy(np.concatenate((root, body), axis=1)).float())
    global_rotation = local.clone()
    for joint, parent in enumerate(parents):
        if parent >= 0:
            global_rotation[:, joint] = global_rotation[:, parent] @ global_rotation[:, joint]
    shifted = torch.from_numpy(shift).float()[None, None] @ global_rotation
    return transforms.matrix_to_rotation_6d(shifted)


class PriorWindowDataset(Dataset):
    """Loads only fields authorized by the selected expert contract."""
    def __init__(self, repo_root: str, expert: str, partition: str = "train", limit: int = 0) -> None:
        self.repo = Path(repo_root).resolve()
        self.expert = expert
        if expert not in {"hoi", "hsi"}:
            raise ValueError(expert)
        self.root = self.repo / ("data/train" if expert == "hoi" else "data/dataset")
        language_path = self.root / "language_motion_dict/language_motion_dict__inter_and_loco__16.pkl"
        with language_path.open("rb") as handle:
            self.language = pickle.load(handle)
        self.starts = np.asarray(self.language["start_idx"], dtype=np.int64)
        self.ends = np.asarray(self.language["end_idx"], dtype=np.int64)
        self.sequence_ids = np.asarray(self.language["ori_sequence_idx"], dtype=np.int64)
        indices = np.arange(len(self.starts), dtype=np.int64)
        self.scene_names = self._load_pickle("scene_name.pkl")
        if expert == "hsi":
            keep = hsi_filter(self.language["left_hand_inter_frame"], self.language["right_hand_inter_frame"])
            split = json.loads((self.repo / "experiments/splits/lingo_scene_disjoint_seed42.json").read_text())
            window_scenes = np.asarray([self.scene_names[int(self.starts[i])] for i in indices], dtype=object)
            sides = partition_for_scenes(split, window_scenes)
            sequence_starts = np.load(self.root / "start_idx.npy", mmap_mode="r")
            sequence_ends = np.load(self.root / "end_idx.npy", mmap_mode="r")
            valid_length = (sequence_ends[self.sequence_ids] - sequence_starts[self.sequence_ids]) > 48
            indices = indices[keep & valid_length & (sides == partition)]
        elif partition != "train":
            raise ValueError("HOI smoke dataset uses the official train root")
        self.indices = indices[:limit] if limit else indices
        self.joints = np.load(self.root / "human_joints_aligned.npy", mmap_mode="r")
        self.orient = np.load(self.root / "human_orient.npy", mmap_mode="r")
        self.pose = np.load(self.root / "human_pose.npy", mmap_mode="r")
        self.seq_starts = np.load(self.root / "start_idx.npy", mmap_mode="r")
        self.seq_ends = np.load(self.root / "end_idx.npy", mmap_mode="r")
        self.norm = np.load(self.root / "norm.npy")
        self.minimum, self.maximum = self.norm[0].astype(np.float32), self.norm[1].astype(np.float32)
        self.text_features = np.load(self.root / "clip_features.npy", mmap_mode="r")
        self.text_to_feature = self._load_pickle("text2features_idx.pkl")
        self.parents = get_smpl_parents(use_joints24=False).copy()
        if expert == "hoi":
            self.object_trans = np.load(self.root / "object_trans.npy", mmap_mode="r")
            self.object_rot = np.load(self.root / "object_rot_mat.npy", mmap_mode="r")
            self.object_minimum, self.object_maximum = self.norm[2].astype(np.float32), self.norm[3].astype(np.float32)

    def _load_pickle(self, relative: str):
        with (self.root / relative).open("rb") as handle:
            return pickle.load(handle)

    def __len__(self) -> int:
        return len(self.indices)

    @lru_cache(maxsize=16)
    def _bps(self, sequence_name: str) -> np.ndarray:
        value = np.load(self.root / "cano_object_bps_npy_files_joints24_120" / f"{sequence_name}.npy", mmap_mode="r")
        return value

    @lru_cache(maxsize=32)
    def _contact(self, sequence_name: str) -> np.ndarray:
        return np.load(self.root / "contact_label_npy_files" / f"{sequence_name}.npy", mmap_mode="r")

    @lru_cache(maxsize=16)
    def _scene(self, scene_name: str) -> np.ndarray:
        occupancy = np.load(self.root / "Scene" / f"{scene_name}.npy", mmap_mode="r")
        x = np.linspace(0, occupancy.shape[0] - 1, 8).round().astype(int)
        y = np.linspace(0, occupancy.shape[1] - 1, 8).round().astype(int)
        z = np.linspace(0, occupancy.shape[2] - 1, 8).round().astype(int)
        return np.asarray(occupancy[np.ix_(x, y, z)], dtype=np.float32)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        index = int(self.indices[item])
        start, end = int(self.starts[index]), int(self.ends[index])
        if end - start != 48:
            raise ValueError(f"window {index} has {end-start} source frames, expected 48")
        frames = np.arange(start, end, 3)
        joints = np.asarray(self.joints[frames], dtype=np.float32)
        initial = np.array((joints[0, 0, 0], 0.0, joints[0, 0, 2]), dtype=np.float32)
        oriented = zup_to_yup(np.asarray(self.orient[start], dtype=np.float64).copy())
        yaw = Rotation.from_rotvec(oriented).as_euler("zxy")[2]
        shift = Rotation.from_euler("zxy", (0.0, 0.0, -yaw)).as_matrix().astype(np.float32)
        joints = (joints - initial) @ shift.T
        joints = (-1.0 + 2.0 * (joints - self.minimum) / (self.maximum - self.minimum)).reshape(16, 84)
        rotations = _rotation_6d(
            np.asarray(self.orient[start:end]), np.asarray(self.pose[start:end]), shift, self.parents,
        )[::3].numpy()
        representation = np.zeros((16, REPRESENTATION.dimension), dtype=np.float32)
        representation[:, :84] = joints
        representation[:, 84:216] = rotations.reshape(16, 132)
        sequence = int(self.sequence_ids[index])
        sequence_name = str(self.scene_names[sequence] if self.expert == "hoi" else self.scene_names[start])
        object_bps = np.zeros((1024, 3), dtype=np.float32)
        scene = np.zeros((8, 8, 8), dtype=np.float32)
        goals = np.zeros(9, dtype=np.float32)
        if self.expert == "hoi":
            trans = (np.asarray(self.object_trans[frames], dtype=np.float32) - initial) @ shift.T
            representation[:, 216:219] = -1.0 + 2.0 * (trans - self.object_minimum) / (self.object_maximum - self.object_minimum)
            reference = np.asarray(self.object_rot[start], dtype=np.float32)
            representation[:, 219:228] = (np.asarray(self.object_rot[frames]) @ reference.T).reshape(16, 9)
            offset = start - int(self.seq_starts[sequence])
            object_bps = zup_to_yup(np.asarray(self._bps(sequence_name)[offset], dtype=np.float32).copy())
            representation[:, 228:232] = np.asarray(self._contact(sequence_name)[offset:offset + 48:3], dtype=np.float32)
            goals[6:9] = representation[-1, 216:219]
        else:
            scene = self._scene(sequence_name)
            goals[:3] = joints[-1, :3]
        text = self.language["text"][index][0]
        embedding = np.asarray(self.text_features[self.text_to_feature[text]], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError(f"invalid text embedding for {text!r}")
        embedding = embedding / norm
        seq_length = int(self.seq_ends[sequence] - self.seq_starts[sequence])
        pi = int(self.language["pi"][index])
        end_pi = min(pi + 48, seq_length)
        progress = np.asarray((pi, end_pi, seq_length), dtype=np.float32)
        result = {
            "x": torch.from_numpy(representation),
            "text_embedding": torch.from_numpy(embedding),
            "goals": torch.from_numpy(goals),
            "progress": torch.from_numpy(progress),
        }
        if self.expert == "hoi":
            result["object_bps"] = torch.from_numpy(object_bps)
        else:
            result["scene_condition"] = torch.from_numpy(scene)
        return result
