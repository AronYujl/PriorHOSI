"""HSI window dataset.

Forked byte-identically from ``code/priors/hoi/data.py`` at commit c77b9d8 when
``code/priors/`` was split into ``core``/``hoi``/``hsi``.  The pre-split module
held the only implementation of ``hsi_filter``, ``partition_for_scenes`` and the
``expert == "hsi"`` branch, so filing it under ``hoi/`` alone would have deleted
the HSI dataset the moment ``phase/01c-hsi`` removed ``hoi/``.

This copy is the HSI branch's starting point and is expected to diverge: the HOI
paths (``_bps``, ``_contact``, ``_rest_object_points`` and their ``__getitem__``
branches) exist here only so the Phase 1A HSI contract keeps passing from day
one.  Strip them deliberately, with the tests in ``tests/hsi/`` as the gate --
the dataset is NOT part of the frozen ``core`` contract and needs no
cross-branch approval to change.
"""

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh
from pytorch3d import transforms
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from datasets.utils import get_smpl_parents, zup_to_yup
from ..core.representation import REPRESENTATION
from ..core.window_codec import WindowStateCodec


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


def _global_rotations(root_orient: np.ndarray, pose: np.ndarray, parents: np.ndarray) -> torch.Tensor:
    root = zup_to_yup(root_orient.copy()).reshape(-1, 1, 3)
    body = zup_to_yup(pose.copy().reshape(-1, 3)).reshape(-1, 21, 3)
    local = transforms.axis_angle_to_matrix(torch.from_numpy(np.concatenate((root, body), axis=1)).float())
    global_rotation = local.clone()
    for joint, parent in enumerate(parents):
        if parent >= 0:
            global_rotation[:, joint] = global_rotation[:, parent] @ global_rotation[:, joint]
    return global_rotation


def _rotation_6d(root_orient: np.ndarray, pose: np.ndarray, shift: np.ndarray, parents: np.ndarray) -> torch.Tensor:
    global_rotation = _global_rotations(root_orient, pose, parents)
    shifted = torch.from_numpy(shift).float()[None, None] @ global_rotation
    return transforms.matrix_to_rotation_6d(shifted)


class PriorWindowDataset(Dataset):
    """Loads only fields authorized by the selected expert contract."""
    def __init__(
        self,
        repo_root: str,
        expert: str,
        partition: str = "train",
        limit: int = 0,
        split_manifest: Optional[str] = None,
    ) -> None:
        self.repo = Path(repo_root).resolve()
        self.expert = expert
        self.partition = partition
        if expert not in {"hoi", "hsi"}:
            raise ValueError(expert)
        if expert == "hoi":
            if partition in {"validation", "test"}:
                self.root = self.repo / "data/test"
            elif partition in {"train", "internal_validation"}:
                self.root = self.repo / "data/train"
            else:
                raise ValueError(f"unknown HOI partition: {partition}")
        else:
            self.root = self.repo / "data/dataset"
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
        elif split_manifest:
            split_path = Path(split_manifest)
            if not split_path.is_absolute():
                split_path = self.repo / split_path
            split = json.loads(split_path.read_text(encoding="utf-8"))
            if split.get("algorithm") != "omomo-sequence-sha256-seed42-v1":
                raise ValueError(f"unexpected HOI split algorithm: {split_path}")
            selected = set(int(value) for value in split[partition]["sequence_indices"])
            indices = indices[np.isin(self.sequence_ids, list(selected))]
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
            self.rest_human_offsets = np.load(self.root / "rest_human_offsets_aligned.npy", mmap_mode="r")
            self.codec = WindowStateCodec(
                torch.from_numpy(self.minimum), torch.from_numpy(self.maximum),
                torch.from_numpy(self.object_minimum), torch.from_numpy(self.object_maximum),
                bps_path=self.repo / "code/bps.pt",
            )
        else:
            self.codec = WindowStateCodec(torch.from_numpy(self.minimum), torch.from_numpy(self.maximum))

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
    def _rest_object_points(self, object_name: str) -> np.ndarray:
        mesh = trimesh.load_mesh(self.repo / "data/object/rest_object_geo" / f"{object_name}.ply", process=False)
        vertices = zup_to_yup(np.asarray(mesh.vertices, dtype=np.float32).copy())
        indices = np.linspace(0, len(vertices) - 1, min(100, len(vertices))).round().astype(np.int64)
        return np.asarray(vertices[indices], dtype=np.float32)

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
        global_rotations = _global_rotations(
            np.asarray(self.orient[start:end]), np.asarray(self.pose[start:end]), self.parents,
        )[::3]
        sequence = int(self.sequence_ids[index])
        sequence_name = str(self.scene_names[sequence] if self.expert == "hoi" else self.scene_names[start])
        object_bps = np.zeros((1024, 3), dtype=np.float32)
        scene = np.zeros((8, 8, 8), dtype=np.float32)
        goals = np.zeros(9, dtype=np.float32)
        if self.expert == "hoi":
            object_translation = torch.from_numpy(np.asarray(self.object_trans[frames], dtype=np.float32))
            object_rotation = torch.from_numpy(np.asarray(self.object_rot[frames], dtype=np.float32))
            offset = start - int(self.seq_starts[sequence])
            object_bps = zup_to_yup(np.asarray(self._bps(sequence_name)[offset], dtype=np.float32).copy())
            contact = torch.from_numpy(
                np.asarray(self._contact(sequence_name)[offset:offset + 48:3], dtype=np.float32)
            )
            representation_tensor, frame = self.codec.encode(
                torch.from_numpy(joints), global_rotations,
                global_object_translation=object_translation,
                global_object_rotation=object_rotation,
                contact=contact,
            )
            representation = representation_tensor.numpy()
            if not bool(self.language["need_object"][index]):
                raise ValueError(f"HOI window {index} does not contain a dynamic object")
            pelvis_endpoint = torch.from_numpy(
                np.array(self.joints[frames[-1], 0], dtype=np.float32, copy=True)
            )
            goals[:3] = self.codec.pelvis_goal(pelvis_endpoint, frame).numpy()
            goal_frame = int(self.language["end_range"][index]) - 4
            goal_frame = min(max(goal_frame, int(self.seq_starts[sequence])), int(self.seq_ends[sequence]) - 1)
            goal_trans = torch.from_numpy(
                np.array(self.object_trans[goal_frame], dtype=np.float32, copy=True)
            )
            goals[6:9] = self.codec.object_goal(goal_trans, frame).numpy()
        else:
            representation_tensor, frame = self.codec.encode(torch.from_numpy(joints), global_rotations)
            representation = representation_tensor.numpy()
            scene = self._scene(sequence_name)
            goals[:3] = representation[-1, :3]
        text = self.language["text"][index][0]
        embedding = np.asarray(self.text_features[self.text_to_feature[text]], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError(f"invalid text embedding for {text!r}")
        embedding = embedding / norm
        seq_length = int(self.seq_ends[sequence] - self.seq_starts[sequence])
        pi = int(self.language["pi"][index])
        end_pi = min(pi + 48, seq_length)
        # OMOMO's final language window ends one stored sentinel frame before
        # ``seq_ends``.  Expose that window as the inclusive terminal progress
        # endpoint so the preregistered mask is exactly ``end_pi == seq_length``.
        if end == int(self.seq_ends[sequence]) - 1:
            end_pi = seq_length
        progress = np.asarray((pi, end_pi, seq_length), dtype=np.float32)
        result = {
            "x": torch.from_numpy(representation),
            "text_embedding": torch.from_numpy(embedding),
            "goals": torch.from_numpy(goals),
            "progress": torch.from_numpy(progress),
        }
        if self.expert == "hoi":
            result["object_bps"] = torch.from_numpy(object_bps)
            result["rest_human_offsets"] = torch.from_numpy(
                np.array(self.rest_human_offsets[sequence], dtype=np.float32, copy=True)
            )
            result["sequence_index"] = torch.tensor(sequence, dtype=torch.long)
            result["window_index"] = torch.tensor(index, dtype=torch.long)
            result["terminal_window"] = torch.tensor(end_pi == seq_length, dtype=torch.bool)
            result["window_origin"] = frame.origin.detach().clone()
            result["world_to_local_rotation"] = frame.world_to_local.detach().clone()
            result["object_rotation_reference"] = frame.object_reference.detach().clone()
            object_name = sequence_name.split("_")[1]
            result["rest_object_points"] = torch.from_numpy(
                np.array(self._rest_object_points(object_name), dtype=np.float32, copy=True)
            )
        else:
            result["scene_condition"] = torch.from_numpy(scene)
        return result
