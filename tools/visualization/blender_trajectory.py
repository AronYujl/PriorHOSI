"""Render OMOMO Figure 6-style multi-pose stills from an accepted mesh cache."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from .blender import (
    DEFAULT_BLENDER_SCRIPT,
    BlenderRenderError,
    _blender_command,
    _run_blender,
    _validate_render_settings,
    _write_json,
)
from .headless import _sha256


DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 800
DEFAULT_SAMPLES = 64
DEFAULT_MATERIAL_STYLE = "omomo"
MATERIAL_STYLE_CHOICES = (DEFAULT_MATERIAL_STYLE, "lingo")
LINGO_FOREGROUND_MATERIALS = {
    "style": "lingo_principled_v1",
    "human": {
        "base_color": [0.20, 0.42, 0.56, 1.0],
        "roughness": 0.46,
        "specular": 0.35,
    },
    "object": {
        "base_color": [0.42, 0.56, 0.43, 1.0],
        "roughness": 0.66,
        "specular": 0.26,
    },
    "smooth_shading": True,
}


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise BlenderRenderError("%s does not exist: %s" % (label, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlenderRenderError("cannot read %s: %s" % (label, path)) from exc
    if not isinstance(value, dict):
        raise BlenderRenderError("%s must contain a JSON object" % label)
    return value


def _validate_keyframes(frames: Sequence[int], frame_count: int) -> list[int]:
    selected = []
    for frame in frames:
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise BlenderRenderError("multi-pose keyframes must be integers")
        selected.append(int(frame))
    if len(selected) < 2 or len(selected) > 12:
        raise BlenderRenderError("multi-pose composition requires 2 to 12 keyframes")
    if any(left >= right for left, right in zip(selected, selected[1:])):
        raise BlenderRenderError("multi-pose keyframes must be unique and increasing")
    if selected[0] < 0 or selected[-1] >= frame_count:
        raise BlenderRenderError("multi-pose keyframe is outside the source timeline")
    return selected


def _resolve_materials(
    source_materials: Mapping[str, Any], material_style: str
) -> Dict[str, Any]:
    if material_style == DEFAULT_MATERIAL_STYLE:
        required = ("human_source", "object_source")
        missing = [key for key in required if key not in source_materials]
        if missing:
            raise BlenderRenderError(
                "source OMOMO materials are missing: %s" % ", ".join(missing)
            )
        # Preserve the accepted V3a material configuration rather than
        # approximating the source .blend materials with new shader nodes.
        return copy.deepcopy(dict(source_materials))
    if material_style == "lingo":
        return copy.deepcopy(LINGO_FOREGROUND_MATERIALS)
    raise BlenderRenderError(
        "unsupported multi-pose material style: %s" % material_style
    )


def _validate_sources(
    cache: Path,
    cache_manifest: Path,
    source_render_manifest: Path,
    blend_scene: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], int]:
    if not cache.is_file():
        raise BlenderRenderError("mesh cache does not exist: %s" % cache)
    if not blend_scene.is_file():
        raise BlenderRenderError("OMOMO blend scene does not exist: %s" % blend_scene)
    cache_record = _load_json(cache_manifest, "mesh-cache manifest")
    render_record = _load_json(source_render_manifest, "source render manifest")
    cache_sha256 = _sha256(cache)
    cache_manifest_sha256 = _sha256(cache_manifest)
    if cache_record.get("cache_sha256") != cache_sha256:
        raise BlenderRenderError("mesh cache hash does not match its manifest")
    if render_record.get("mesh_cache_sha256") != cache_sha256:
        raise BlenderRenderError("mesh cache hash does not match source render")
    if render_record.get("mesh_cache_manifest_sha256") != cache_manifest_sha256:
        raise BlenderRenderError("mesh-cache manifest hash does not match source render")
    if render_record.get("blend_scene_sha256") != _sha256(blend_scene):
        raise BlenderRenderError("blend scene does not match source render")
    if cache_record.get("sequence_id") != render_record.get("sequence_id"):
        raise BlenderRenderError("source manifests disagree on sequence identity")
    if cache_record.get("ground_correction") != render_record.get(
        "ground_correction"
    ):
        raise BlenderRenderError("source manifests disagree on ground correction")
    try:
        with np.load(cache, allow_pickle=False) as loaded:
            human = np.asarray(loaded["human_vertices"])
            manipulated_object = np.asarray(loaded["object_vertices"])
            human_faces = np.asarray(loaded["human_faces"])
            object_faces = np.asarray(loaded["object_faces"])
            frame_index = np.asarray(loaded["frame_index"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise BlenderRenderError("cannot validate Blender mesh cache arrays") from exc
    if (
        human.ndim != 3
        or manipulated_object.ndim != 3
        or human.shape[0] != manipulated_object.shape[0]
        or human.shape[-1] != 3
        or manipulated_object.shape[-1] != 3
        or human_faces.ndim != 2
        or object_faces.ndim != 2
        or human_faces.shape[1] != 3
        or object_faces.shape[1] != 3
    ):
        raise BlenderRenderError("Blender mesh cache has incompatible topology")
    frame_count = int(human.shape[0])
    if frame_count != cache_record.get("frame_count"):
        raise BlenderRenderError("mesh cache frame count disagrees with its manifest")
    if not np.array_equal(frame_index, np.arange(frame_count, dtype=frame_index.dtype)):
        raise BlenderRenderError("mesh cache frame index is not the complete timeline")
    return cache_record, render_record, frame_count


def _build_config(
    source_config: Mapping[str, Any],
    *,
    cache: Path,
    frame_count: int,
    frames: Sequence[int],
    output_image: str,
    width: int,
    height: int,
    samples: int,
    material_style: str = DEFAULT_MATERIAL_STYLE,
) -> Dict[str, Any]:
    required = (
        "camera",
        "color_management",
        "device",
        "engine",
        "floor",
        "materials",
    )
    missing = [key for key in required if key not in source_config]
    if missing:
        raise BlenderRenderError(
            "source Blender config is missing: %s" % ", ".join(missing)
        )
    selected = _validate_keyframes(frames, frame_count)
    materials = _resolve_materials(source_config["materials"], material_style)
    return {
        "cache": str(cache),
        "scene_report": "blender-scene-report.json",
        "render_mode": "multi_pose",
        "output_image": output_image,
        "frame_count": frame_count,
        "selected_frame_indices": selected,
        "selection_rule": "explicit_increasing_source_frames",
        "width": width,
        "height": height,
        "samples": samples,
        "engine": source_config["engine"],
        "device": source_config["device"],
        "camera": source_config["camera"],
        "floor": source_config["floor"],
        "materials": materials,
        "color_management": source_config["color_management"],
        "composition": {
            "mode": "opaque_multi_pose_shared_scene",
            "selected_frame_indices": selected,
            "selection_rule": "explicit_increasing_source_frames",
            "pose_layout": "unaltered_source_world_coordinates",
            "camera_fit_source": "complete_mesh_cache_all_frames",
            "human_object_pair_count": len(selected),
            "material_style": material_style,
            "material_change_scope": "foreground_human_and_object_only",
        },
    }


def render_multi_pose_figure(
    cache_path: Path | str,
    *,
    cache_manifest_path: Path | str,
    source_render_manifest_path: Path | str,
    output_dir: Path | str,
    blend_scene: Path | str,
    blender_binary: Path | str,
    frames: Sequence[int],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    samples: int = DEFAULT_SAMPLES,
    material_style: str = DEFAULT_MATERIAL_STYLE,
    renderer_commit: str = "local-unrecorded",
    blender_script: Path | str = DEFAULT_BLENDER_SCRIPT,
) -> Dict[str, Any]:
    cache = Path(cache_path).resolve()
    cache_manifest = Path(cache_manifest_path).resolve()
    source_render_manifest = Path(source_render_manifest_path).resolve()
    destination = Path(output_dir).resolve()
    scene_asset = Path(blend_scene).resolve()
    blender = Path(blender_binary).resolve()
    script = Path(blender_script).resolve()
    _validate_render_settings(
        width=width,
        height=height,
        fps=30.0,
        samples=samples,
        figure_columns=1,
    )
    if destination.exists():
        raise BlenderRenderError(
            "refusing to overwrite artifact directory %s" % destination
        )
    if not blender.is_file():
        raise BlenderRenderError("Blender binary does not exist: %s" % blender)
    if not script.is_file():
        raise BlenderRenderError("Blender scene script does not exist: %s" % script)
    cache_record, source_record, frame_count = _validate_sources(
        cache, cache_manifest, source_render_manifest, scene_asset
    )
    selected = _validate_keyframes(frames, frame_count)
    source_config = source_record.get("config")
    if not isinstance(source_config, dict):
        raise BlenderRenderError("source render manifest has no Blender config")

    destination.parent.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    staging = destination.with_name(".%s.%s.staging" % (destination.name, identity))
    staging.mkdir()
    if material_style == DEFAULT_MATERIAL_STYLE:
        image_name = "trajectory-k%d-omomo-grounded-style.png" % len(selected)
    else:
        image_name = (
            "trajectory-k%d-omomo-grounded-layout-lingo-materials.png"
            % len(selected)
        )
    image_path = staging / image_name
    config_path = staging / "blender-trajectory-config.json"
    scene_report = staging / "blender-scene-report.json"
    log_path = staging / "blender.log"
    figure_manifest = staging / "figure.manifest.json"
    config = _build_config(
        source_config,
        cache=cache,
        frame_count=frame_count,
        frames=selected,
        output_image=image_name,
        width=width,
        height=height,
        samples=samples,
        material_style=material_style,
    )
    _write_json(config_path, config)
    _run_blender(_blender_command(blender, scene_asset, script, config_path), log_path)
    if not image_path.is_file():
        raise BlenderRenderError("Blender did not produce the multi-pose image")
    with Image.open(image_path) as loaded:
        if loaded.size != (width, height):
            raise BlenderRenderError("multi-pose image dimensions do not match request")
    report = _load_json(scene_report, "Blender scene report")
    version = subprocess.run(
        [str(blender), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).stdout.splitlines()[0]
    ground_correction = cache_record.get("ground_correction", {})
    record = {
        "schema": "infbagel-blender-multi-pose-figure-v1",
        "sequence_id": cache_record["sequence_id"],
        "renderer_commit": renderer_commit,
        "renderer_backend": "blender-cycles-omomo-multi-pose-scene",
        "material_style": material_style,
        "composition": config["composition"],
        "ground_correction": ground_correction,
        "visualization_only": True,
        "evaluation_forbidden": True,
        "source_motion_sha256": cache_record.get("source_motion_sha256"),
        "mesh_cache_path": str(cache),
        "mesh_cache_sha256": _sha256(cache),
        "mesh_cache_manifest_path": str(cache_manifest),
        "mesh_cache_manifest_sha256": _sha256(cache_manifest),
        "source_render_manifest_path": str(source_render_manifest),
        "source_render_manifest_sha256": _sha256(source_render_manifest),
        "blend_scene_path": str(scene_asset),
        "blend_scene_sha256": _sha256(scene_asset),
        "blender_binary_path": str(blender),
        "blender_binary_sha256": _sha256(blender),
        "blender_version": version,
        "blender_script_sha256": _sha256(script),
        "config": config,
        "scene_report": report,
        "image": {
            "path": image_name,
            "dimensions": [width, height],
            "sha256": _sha256(image_path),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(figure_manifest, record)
    if destination.exists():
        raise BlenderRenderError("artifact directory appeared during rendering")
    os.rename(staging, destination)
    return {
        "sequence_id": cache_record["sequence_id"],
        "output_dir": str(destination),
        "image_path": str(destination / image_name),
        "figure_manifest_path": str(destination / figure_manifest.name),
        "selected_frame_indices": selected,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an OMOMO Figure 6-style shared-scene motion still"
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--source-render-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blend-scene", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--material-style",
        choices=MATERIAL_STYLE_CHOICES,
        default=DEFAULT_MATERIAL_STYLE,
        help="foreground palette; omomo preserves the accepted source materials",
    )
    parser.add_argument("--renderer-commit", default="local-unrecorded")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = render_multi_pose_figure(
            args.cache,
            cache_manifest_path=args.cache_manifest,
            source_render_manifest_path=args.source_render_manifest,
            output_dir=args.output_dir,
            blend_scene=args.blend_scene,
            blender_binary=args.blender,
            frames=args.frames,
            width=args.width,
            height=args.height,
            samples=args.samples,
            material_style=args.material_style,
            renderer_commit=args.renderer_commit,
        )
    except (BlenderRenderError, OSError, ValueError) as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
