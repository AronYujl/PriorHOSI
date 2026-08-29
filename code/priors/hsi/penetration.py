"""Batched, mirror-aware signed-distance lookup for LINGO scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch

from .scene_field import (
    BUILD_VERSION,
    SDF_EXACT_BAND,
    SDF_PAD,
    SDF_VOXEL_SIZE,
    SceneGeometry,
    _mesh_sha256,
    default_cache_dir,
)


DEFAULT_LINGO_MESH_ROOT = Path("/data/yujinlun/datasets/LINGO/Scene_mesh")

__all__ = ["DEFAULT_LINGO_MESH_ROOT", "SceneSDFBank", "resolve_sdf_dtype"]


def resolve_sdf_dtype(dtype: object) -> torch.dtype:
    """Resolve the config spelling and restrict storage to the audited choices."""
    if isinstance(dtype, str):
        if dtype.startswith("torch."):
            dtype = dtype[len("torch."):]
        dtype = {"float16": torch.float16, "half": torch.float16,
                 "float32": torch.float32}.get(dtype, dtype)
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("pen_sdf_dtype must be float16 or float32")
    return dtype


class SceneSDFBank:
    """LINGO source SDFs in one flat buffer with a batched trilinear gather.

    Mirror flags reuse their source field by negating query x.  The fields are
    stored at ``dtype`` but all query coordinates, weights, and interpolated
    values are evaluated in float32.
    """

    def __init__(
        self,
        flat_field: torch.Tensor,
        offsets: torch.Tensor,
        shapes: torch.Tensor,
        origins: torch.Tensor,
        flag_keys: torch.Tensor,
        flag_source_index: torch.Tensor,
        flag_mirror: torch.Tensor,
        scene_names,
        *,
        voxel_size: float = SDF_VOXEL_SIZE,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        dtype = resolve_sdf_dtype(dtype)
        target_device = torch.device(device) if device is not None else flat_field.device
        self.flat_field = flat_field.to(device=target_device, dtype=dtype).contiguous()
        self.offsets = offsets.to(device=target_device, dtype=torch.long).contiguous()
        self.shapes = shapes.to(device=target_device, dtype=torch.long).contiguous()
        self.origins = origins.to(device=target_device, dtype=torch.float32).contiguous()
        self.flag_keys = flag_keys.to(device=target_device, dtype=torch.long).contiguous()
        self.flag_source_index = flag_source_index.to(
            device=target_device, dtype=torch.long
        ).contiguous()
        self.flag_mirror = flag_mirror.to(device=target_device, dtype=torch.bool).contiguous()
        self.scene_names = tuple(scene_names)
        self.voxel_size = float(voxel_size)
        self.dtype = dtype
        self.device = target_device

    @classmethod
    def from_scene_flags(
        cls,
        flag_to_name: Mapping[int, str],
        *,
        dataset_root,
        mesh_root,
        cache_dir=None,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        require_cache: bool = True,
    ) -> "SceneSDFBank":
        """Load the unique source fields named by a unified scene-flag table."""
        if not flag_to_name:
            raise ValueError("flag_to_name must contain at least one LINGO scene")

        normalized = {int(flag): str(name) for flag, name in flag_to_name.items()}
        source_names = sorted({
            name[:-7] if name.endswith("_mirror") else name
            for name in normalized.values()
        })
        dtype = resolve_sdf_dtype(dtype)
        if require_cache:
            for name in source_names:
                cache_path = cls._cache_path(name, mesh_root, cache_dir)
                if not cache_path.is_file():
                    raise FileNotFoundError(
                        f"missing prebuilt SDF cache for scene {name!r}: {cache_path}; "
                        "E3 refuses to build a field during training"
                    )
        fields = []
        shapes = []
        origins = []
        voxel_size = None
        for name in source_names:
            geometry = SceneGeometry.from_scene(
                name,
                dataset_root=Path(dataset_root),
                mesh_root=Path(mesh_root),
                cache_dir=cache_dir,
            )
            fields.append(geometry.field.reshape(-1).to(dtype=dtype))
            shapes.append(geometry.grid_shape)
            origins.append(geometry.bounds[0])
            if voxel_size is None:
                voxel_size = geometry.voxel_size
        return cls._from_packed(
            normalized,
            source_names,
            fields,
            shapes,
            origins,
            voxel_size,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _cache_path(scene_name, mesh_root, cache_dir):
        mesh_path = Path(mesh_root) / str(scene_name) / "mesh_low.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"no mesh for scene {scene_name!r} at {mesh_path}; "
                "the SDF bank cannot be constructed"
            )
        digest = _mesh_sha256(mesh_path)
        cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        return cache_root / (
            f"{scene_name}__{digest[:16]}__h{round(SDF_VOXEL_SIZE * 1000)}mm"
            f"__p{round(SDF_PAD * 1000)}mm__b{SDF_EXACT_BAND}__v{BUILD_VERSION}.npz"
        )

    @classmethod
    def from_geometries(
        cls,
        flag_to_geometry: Mapping[int, SceneGeometry],
        *,
        flag_to_name: Optional[Mapping[int, str]] = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> "SceneSDFBank":
        """Pack in-memory geometries; useful for tests and synthetic callers."""
        if not flag_to_geometry:
            raise ValueError("flag_to_geometry must contain at least one scene")
        normalized = {int(flag): geometry for flag, geometry in flag_to_geometry.items()}
        names = {
            flag: str(flag_to_name[flag]) if flag_to_name is not None else geometry.scene_name
            for flag, geometry in normalized.items()
        }
        geometries = {}
        for flag, geometry in normalized.items():
            name = names[flag]
            source_name = name[:-7] if name.endswith("_mirror") else name
            geometries.setdefault(source_name, geometry)
        return cls._from_geometries(names, geometries, dtype=dtype, device=device)

    @classmethod
    def _from_geometries(cls, flag_to_name, geometries, *, dtype, device):
        dtype = resolve_sdf_dtype(dtype)
        source_names = sorted(geometries)
        fields = [geometries[name].field.reshape(-1).to(dtype=dtype) for name in source_names]
        shapes = [geometries[name].grid_shape for name in source_names]
        origins = [geometries[name].bounds[0] for name in source_names]
        return cls._from_packed(
            flag_to_name,
            source_names,
            fields,
            shapes,
            origins,
            geometries[source_names[0]].voxel_size,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def _from_packed(
        cls,
        flag_to_name,
        source_names,
        fields,
        shapes,
        origins,
        voxel_size,
        *,
        dtype,
        device,
    ):
        flat_field = torch.cat(fields, dim=0)

        offsets = []
        cursor = 0
        for field in fields:
            offsets.append(cursor)
            cursor += field.numel()

        source_index = {name: index for index, name in enumerate(source_names)}
        flag_keys = sorted(int(flag) for flag in flag_to_name)
        flag_source_index = []
        flag_mirror = []
        for flag in flag_keys:
            name = str(flag_to_name[flag])
            source_name = name[:-7] if name.endswith("_mirror") else name
            flag_source_index.append(source_index[source_name])
            flag_mirror.append(name.endswith("_mirror"))

        return cls(
            flat_field,
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(shapes, dtype=torch.long),
            torch.stack(origins).to(dtype=torch.float32),
            torch.tensor(flag_keys, dtype=torch.long),
            torch.tensor(flag_source_index, dtype=torch.long),
            torch.tensor(flag_mirror, dtype=torch.bool),
            source_names,
            voxel_size=voxel_size,
            dtype=dtype,
            device=device,
        )

    def signed_distance(self, points_world: torch.Tensor, scene_flag: torch.Tensor):
        """Return signed distance and out-of-bounds flags for ``[B,...,3]`` points."""
        if points_world.shape[-1] != 3:
            raise ValueError(
                f"points_world must have trailing dim 3, got {tuple(points_world.shape)}"
            )
        if points_world.ndim < 2:
            raise ValueError("points_world must have a batch dimension")

        leading = points_world.shape[:-1]
        batch_size = leading[0]
        flags = scene_flag.to(device=points_world.device, dtype=torch.long).reshape(-1)
        if flags.numel() != batch_size:
            raise ValueError(
                f"scene_flag has {flags.numel()} entries for a batch of {batch_size}"
            )

        flag_keys = self.flag_keys.to(device=points_world.device)
        flag_source_index = self.flag_source_index.to(device=points_world.device)
        flag_mirror = self.flag_mirror.to(device=points_world.device)
        flag_index = torch.searchsorted(flag_keys, flags)
        flag_index = flag_index.clamp(max=flag_keys.numel() - 1)
        exact_match = flag_keys[flag_index] == flags
        if not bool(exact_match.all()):
            offending = torch.unique(flags[~exact_match]).detach().cpu().tolist()
            raise ValueError(
                f"scene_flag(s) not present in SceneSDFBank: {offending}"
            )
        source_per_batch = flag_source_index[flag_index]
        mirror_per_batch = flag_mirror[flag_index]

        expand_shape = (batch_size,) + (1,) * (len(leading) - 1)
        source_index = source_per_batch.reshape(expand_shape).expand(leading).reshape(-1)
        mirror = mirror_per_batch.reshape(expand_shape).expand(leading).reshape(-1)

        points = points_world.to(device=points_world.device, dtype=torch.float32).reshape(-1, 3)
        mirror_sign = torch.where(
            mirror, torch.full_like(points[:, 0], -1.0), torch.ones_like(points[:, 0])
        )
        query = points * torch.stack(
            (mirror_sign, torch.ones_like(mirror_sign), torch.ones_like(mirror_sign)), dim=-1
        )

        origins = self.origins.to(device=points.device)[source_index]
        shapes = self.shapes.to(device=points.device)[source_index]
        offsets = self.offsets.to(device=points.device)[source_index]
        last = shapes.to(dtype=torch.float32) - 1.0

        # Keep this arithmetic in lockstep with SceneGeometry.signed_distance.
        g = (query - origins) / self.voxel_size - 0.5
        g = torch.clamp(g, torch.zeros_like(last), last)
        base = torch.nan_to_num(g, nan=0.0).floor()
        i0 = base.to(torch.long)
        i1 = torch.minimum(i0 + 1, last.to(torch.long))
        w = g - base
        wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
        sx, sy, sz = shapes[:, 0], shapes[:, 1], shapes[:, 2]

        def linear_index(x, y, z):
            return offsets + (x * sy + y) * sz + z

        indices = torch.stack(
            (
                linear_index(x0, y0, z0),
                linear_index(x0, y0, z1),
                linear_index(x0, y1, z0),
                linear_index(x0, y1, z1),
                linear_index(x1, y0, z0),
                linear_index(x1, y0, z1),
                linear_index(x1, y1, z0),
                linear_index(x1, y1, z1),
            ),
            dim=-1,
        )
        corners = self.flat_field.to(device=points.device)[indices].to(dtype=torch.float32)
        c000, c001, c010, c011, c100, c101, c110, c111 = corners.unbind(dim=-1)

        c00 = c000 + (c100 - c000) * wx
        c01 = c001 + (c101 - c001) * wx
        c10 = c010 + (c110 - c010) * wx
        c11 = c011 + (c111 - c011) * wx
        c0 = c00 + (c10 - c00) * wy
        c1 = c01 + (c11 - c01) * wy
        interpolated = c0 + (c1 - c0) * wz

        high = origins + shapes.to(dtype=torch.float32) * self.voxel_size
        overshoot = torch.clamp(origins - query, min=0.0) + torch.clamp(
            query - high, min=0.0
        )
        outside = torch.linalg.vector_norm(overshoot, dim=-1)
        result = torch.where(
            outside > 0, outside + torch.clamp(interpolated, min=0.0), interpolated
        )
        out_of_bounds = ~((query >= origins) & (query <= high)).all(dim=-1)
        return result.reshape(leading), out_of_bounds.reshape(leading)

    def out_of_bounds(self, points_world: torch.Tensor, scene_flag: torch.Tensor):
        """Return the out-of-bounds mask for the same batched query API."""
        return self.signed_distance(points_world, scene_flag)[1]
