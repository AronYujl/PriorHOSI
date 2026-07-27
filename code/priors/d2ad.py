"""Fixed D2-AD human-local full-mesh BPS construction.

The builder is deliberately non-learned and condition-local.  It caches only
the immutable per-object spatial index; no per-window BPS condition is stored.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from torch.utils.data._utils.collate import default_collate

from datasets.utils import zup_to_yup

from .data import PriorWindowDataset
from .interaction_adapter import BPS_SHA256, load_bps_partition
from .window_codec import zup_to_yup_tensor


BPS_YUP_TENSOR_SHA256 = (
    "02b4f8f3510e723174010a823630f663ddda9875ad82a2f8de807d2bdccebd7d"
)
REST_MESH_MANIFEST_SHA256 = (
    "ce8328ef2bf873a79d74fb5fd20cc488551a20d56fe5c5ecabf609824b0654d1"
)
OBJECT_MAPPING_SHA256 = (
    "424fc96102c576a1d11b0824cc0ee616d52cd9e39524819f49b207d1598fe41b"
)
OBJECT_NAMES: Tuple[str, ...] = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "monitor",
    "plasticbox",
    "smallbox",
    "smalltable",
    "suitcase",
    "trashcan",
    "tripod",
    "whitechair",
    "woodchair",
)
REST_MESH_SHA256: Mapping[str, str] = {
    "clothesstand.ply": "f7967db79cfa76c2dd154426c0e110369b5f28f756dcd9df5e5c9ecba3fed960",
    "floorlamp.ply": "c012a69bfb37542ad099ff087b17059b2622a47d2e04e09d67403ea929470e4a",
    "largebox.ply": "6b9ef6cf4564c0a9d052796b2f24bcb0557249304a6c417f25bfcacec50106e0",
    "largetable.ply": "087c3575a25b0c423a9dcc7ceea2519ff604ce0b8172a5e66e89eefd2d3cb7d1",
    "monitor.ply": "21f3bbae53b678eff92f34d597f18a7cdbb04db3195425f6c64c6a2790129bc6",
    "plasticbox.ply": "e0a9c29124314f02d6b189b36716c5a01aeeb973b4001e8f28ee76f3382a1a76",
    "smallbox.ply": "2aa65b4bf36476ffec1bf4582f2aded77bf111956c70d127d44e8bc76b2072b8",
    "smalltable.ply": "eb601c69e2aa96b2cf284dccdf63e2c49f192200f3dc298e5cd6143b42a25628",
    "suitcase.ply": "6ebfb9e5bf732314fe8fbd5dbf1ad500b4b2218fab74888dca7cf3fc2ad49845",
    "trashcan.ply": "7381b83dc33265872753b8425ec7da4a24da0fc0512b57ebda3307c298659d38",
    "tripod.ply": "b2222da71906a832826bd15b697e1a21e69627664da97f62704667fe7e871540",
    "whitechair.ply": "f3cc5036d305cbe5e53a701cdfa63c9181113e6b2249860bf190ab0a8a0bedd0",
    "woodchair.ply": "45d3e42b3f4cb537efc94e37d5919ff1a70967c4ce4c8ad399a0882e5d356a89",
}
DEFAULT_QUERY_WORKERS = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def object_mapping_payload() -> Dict[str, object]:
    return {
        "algorithm": "sorted-ply-stem-v1",
        "names": list(OBJECT_NAMES),
    }


def rest_mesh_manifest_payload() -> Dict[str, object]:
    return {
        "algorithm": "sorted-name-sha256-v1",
        "files": [
            {"name": name, "sha256": REST_MESH_SHA256[name]}
            for name in sorted(REST_MESH_SHA256)
        ],
    }


def local_bps_basis(
    bps_path: Optional[str | Path] = None,
) -> torch.Tensor:
    basis, _, _, _, _ = load_bps_partition(bps_path)
    converted = zup_to_yup_tensor(basis).contiguous()
    actual = _tensor_sha256(converted)
    if actual != BPS_YUP_TENSOR_SHA256:
        raise ValueError(
            f"D2-AD Y-up BPS tensor hash mismatch: {actual} != {BPS_YUP_TENSOR_SHA256}"
        )
    return converted


class LocalObjectBPSBuilder:
    """Build exact current-pose BPS deltas against immutable full meshes."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        query_workers: int = DEFAULT_QUERY_WORKERS,
        verify_assets: bool = True,
    ) -> None:
        self.repo = Path(repo_root).resolve()
        self.query_workers = int(query_workers)
        if self.query_workers == 0 or self.query_workers < -1:
            raise ValueError("D2-AD query_workers must be -1 or a positive integer")
        self.bps_path = self.repo / "code/bps.pt"
        self.mesh_root = self.repo / "data/object/rest_object_geo"
        if verify_assets:
            if _sha256_file(self.bps_path) != BPS_SHA256:
                raise ValueError("D2-AD immutable BPS file hash mismatch")
            names = tuple(sorted(path.stem for path in self.mesh_root.glob("*.ply")))
            if names != OBJECT_NAMES:
                raise ValueError(f"D2-AD rest-object mapping mismatch: {names}")
            if _canonical_json_sha256(object_mapping_payload()) != OBJECT_MAPPING_SHA256:
                raise ValueError("D2-AD object mapping hash mismatch")
            actual_mesh_hashes = {
                path.name: _sha256_file(path)
                for path in sorted(self.mesh_root.glob("*.ply"))
            }
            if actual_mesh_hashes != dict(REST_MESH_SHA256):
                raise ValueError("D2-AD immutable rest-mesh hash mismatch")
            if (
                _canonical_json_sha256(rest_mesh_manifest_payload())
                != REST_MESH_MANIFEST_SHA256
            ):
                raise ValueError("D2-AD rest-mesh manifest hash mismatch")
        self.basis = local_bps_basis(self.bps_path).numpy()
        self._geometry: Dict[int, Tuple[np.ndarray, cKDTree]] = {}

    def _load_geometry(self, object_index: int) -> Tuple[np.ndarray, cKDTree]:
        object_index = int(object_index)
        if object_index < 0 or object_index >= len(OBJECT_NAMES):
            raise ValueError(f"D2-AD object index is outside the fixed mapping: {object_index}")
        if object_index not in self._geometry:
            object_name = OBJECT_NAMES[object_index]
            path = self.mesh_root / f"{object_name}.ply"
            mesh = trimesh.load_mesh(path, process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            vertices = np.asarray(zup_to_yup(vertices.copy()), dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError(f"invalid D2-AD rest mesh: {path}")
            self._geometry[object_index] = (vertices, cKDTree(vertices))
        return self._geometry[object_index]

    @staticmethod
    def object_indices_from_names(names: Sequence[str]) -> torch.Tensor:
        mapping = {name: index for index, name in enumerate(OBJECT_NAMES)}
        indices = []
        for value in names:
            parts = str(value).split("_")
            object_name = parts[1] if len(parts) > 1 else parts[0]
            if object_name not in mapping:
                raise ValueError(f"unknown D2-AD object name: {value}")
            indices.append(mapping[object_name])
        return torch.tensor(indices, dtype=torch.long)

    def build(
        self,
        world_to_local_rotation: torch.Tensor,
        object_rotation_reference: torch.Tensor,
        object_indices: torch.Tensor | Sequence[int],
        *,
        return_indices: bool = False,
    ):
        if not torch.is_tensor(world_to_local_rotation) or not torch.is_tensor(
            object_rotation_reference
        ):
            raise TypeError("D2-AD rotations must be torch tensors")
        if world_to_local_rotation.ndim != 3 or world_to_local_rotation.shape[1:] != (3, 3):
            raise ValueError("expected D2-AD world_to_local_rotation [B,3,3]")
        if object_rotation_reference.shape != world_to_local_rotation.shape:
            raise ValueError("D2-AD object rotation/reference shape mismatch")
        if (
            not bool(torch.isfinite(world_to_local_rotation).all())
            or not bool(torch.isfinite(object_rotation_reference).all())
        ):
            raise ValueError("D2-AD rotations must be finite")
        batch = world_to_local_rotation.shape[0]
        indices_tensor = torch.as_tensor(object_indices, dtype=torch.long).reshape(-1)
        if indices_tensor.shape != (batch,):
            raise ValueError(f"expected D2-AD object indices [{batch}]")
        target_device = world_to_local_rotation.device
        world_to_local = world_to_local_rotation.detach().cpu().to(torch.float32).numpy()
        object_rotation = object_rotation_reference.detach().cpu().to(torch.float32).numpy()
        object_index_array = indices_tensor.cpu().numpy()
        local_rotation = world_to_local @ object_rotation
        output = np.empty((batch, 1024, 3), dtype=np.float32)
        nearest_indices = np.empty((batch, 1024), dtype=np.int64)
        for object_index in sorted(set(int(value) for value in object_index_array.tolist())):
            rows = np.flatnonzero(object_index_array == object_index)
            vertices, tree = self._load_geometry(object_index)
            queries = np.einsum(
                "qc,bcf->bqf", self.basis, local_rotation[rows],
            ).reshape(-1, 3)
            _, selected = tree.query(
                queries,
                k=1,
                eps=0.0,
                p=2,
                workers=self.query_workers,
            )
            selected = np.asarray(selected, dtype=np.int64).reshape(len(rows), 1024)
            nearest = vertices[selected]
            nearest_local = np.einsum(
                "bqc,bfc->bqf", nearest, local_rotation[rows],
            )
            output[rows] = nearest_local - self.basis[None]
            nearest_indices[rows] = selected
        if not np.isfinite(output).all():
            raise FloatingPointError("D2-AD local BPS contains non-finite values")
        result = torch.from_numpy(output).to(device=target_device)
        if not return_indices:
            return result
        return result, torch.from_numpy(nearest_indices).to(device=target_device)

    def build_from_evaluator_inputs(
        self,
        local_to_world_transform: torch.Tensor,
        object_rotation_reference: torch.Tensor,
        sequence_names: Sequence[str],
        *,
        return_indices: bool = False,
    ):
        """Use the official evaluator's current-window frame without reinterpreting it.

        ``mat`` in the native evaluator maps current local coordinates to the
        evaluator's Y-up global frame.  Its rotation transpose is therefore
        exactly the ``WindowFrame.world_to_local`` used by training.
        """

        if (
            not torch.is_tensor(local_to_world_transform)
            or local_to_world_transform.ndim != 3
            or local_to_world_transform.shape[1:] != (4, 4)
        ):
            raise ValueError("expected evaluator local-to-world transforms [B,4,4]")
        batch = local_to_world_transform.shape[0]
        reference = object_rotation_reference.reshape(batch, 3, 3)
        if len(sequence_names) != batch:
            raise ValueError(f"expected {batch} D2-AD evaluator sequence names")
        return self.build(
            local_to_world_transform[:, :3, :3].transpose(-1, -2),
            reference,
            self.object_indices_from_names(sequence_names),
            return_indices=return_indices,
        )

    def contract_metadata(self) -> Dict[str, object]:
        return {
            "bps_sha256": BPS_SHA256,
            "basis_coordinate_system": "human_window_local_y_up",
            "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
            "rest_mesh_manifest_sha256": REST_MESH_MANIFEST_SHA256,
            "object_mapping": list(OBJECT_NAMES),
            "object_mapping_sha256": OBJECT_MAPPING_SHA256,
            "query_backend": "scipy.spatial.cKDTree.query",
            "query_parameters": {"k": 1, "eps": 0.0, "p": 2},
            "query_workers": self.query_workers,
            "full_rest_mesh": True,
            "mesh_subsample": False,
            "window_condition_cache": False,
        }


class D2ADPriorWindowDataset(PriorWindowDataset):
    """Adds only the immutable object-geometry selector needed by the collator."""

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        sequence = int(result["sequence_index"])
        sequence_name = str(self.scene_names[sequence])
        result["object_geometry_index"] = LocalObjectBPSBuilder.object_indices_from_names(
            [sequence_name]
        )[0]
        return result


class D2ADBatchCollator:
    """Default-collate a batch, then build its current-pose local BPS once."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        query_workers: int = DEFAULT_QUERY_WORKERS,
    ) -> None:
        self.repo_root = str(Path(repo_root).resolve())
        self.query_workers = int(query_workers)
        self._builder: Optional[LocalObjectBPSBuilder] = None

    @property
    def builder(self) -> LocalObjectBPSBuilder:
        if self._builder is None:
            self._builder = LocalObjectBPSBuilder(
                self.repo_root,
                query_workers=self.query_workers,
            )
        return self._builder

    def __call__(self, items: Iterable[Mapping[str, torch.Tensor]]) -> Dict[str, object]:
        batch = default_collate(list(items))
        started = time.perf_counter()
        batch["local_object_bps"] = self.builder.build(
            batch["world_to_local_rotation"],
            batch["object_rotation_reference"],
            batch["object_geometry_index"],
        )
        batch["local_bps_build_seconds"] = torch.tensor(
            time.perf_counter() - started,
            dtype=torch.float64,
        )
        return batch
