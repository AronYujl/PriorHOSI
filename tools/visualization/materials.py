"""Shared foreground material presets for HOI Blender renders."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping


DEFAULT_MATERIAL_STYLE = "omomo"
LINGO_MATERIAL_STYLE = "lingo"
TIME_GRADIENT_MATERIAL_STYLE = "white-yellow-time-lingo-object"
ORANGE_TIME_GRADIENT_MATERIAL_STYLE = "white-orange-time-lingo-object"
MATERIAL_STYLE_CHOICES = (
    DEFAULT_MATERIAL_STYLE,
    LINGO_MATERIAL_STYLE,
    TIME_GRADIENT_MATERIAL_STYLE,
    ORANGE_TIME_GRADIENT_MATERIAL_STYLE,
)
TEMPORAL_GRADIENT_MATERIAL_STYLES = (
    TIME_GRADIENT_MATERIAL_STYLE,
    ORANGE_TIME_GRADIENT_MATERIAL_STYLE,
)

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

TIME_GRADIENT_LINGO_OBJECT_MATERIALS = {
    "style": "timeline_white_to_yellow_lingo_object_v1",
    "human": {
        "color_mode": "timeline_linear",
        "start_color": [0.92, 0.92, 0.90, 1.0],
        "end_color": [1.0, 0.62, 0.03, 1.0],
        "timeline_normalization": "source_frame_index/(frame_count-1)",
        "roughness": 0.58,
        "specular": 0.24,
    },
    "object": copy.deepcopy(LINGO_FOREGROUND_MATERIALS["object"]),
    "smooth_shading": True,
}

ORANGE_TIME_GRADIENT_LINGO_OBJECT_MATERIALS = {
    "style": "timeline_white_to_orange_lingo_object_v1",
    "human": {
        "color_mode": "timeline_linear",
        "start_color": [0.92, 0.92, 0.90, 1.0],
        "end_color": [0.82, 0.32, 0.055, 1.0],
        "timeline_normalization": "source_frame_index/(frame_count-1)",
        "roughness": 0.62,
        "specular": 0.20,
    },
    "object": copy.deepcopy(LINGO_FOREGROUND_MATERIALS["object"]),
    "smooth_shading": True,
}


def resolve_foreground_materials(
    source_materials: Mapping[str, Any], material_style: str
) -> Dict[str, Any]:
    if material_style == DEFAULT_MATERIAL_STYLE:
        required = ("human_source", "object_source")
        missing = [key for key in required if key not in source_materials]
        if missing:
            raise ValueError(
                "source OMOMO materials are missing: %s" % ", ".join(missing)
            )
        return copy.deepcopy(dict(source_materials))
    if material_style == LINGO_MATERIAL_STYLE:
        return copy.deepcopy(LINGO_FOREGROUND_MATERIALS)
    if material_style == TIME_GRADIENT_MATERIAL_STYLE:
        return copy.deepcopy(TIME_GRADIENT_LINGO_OBJECT_MATERIALS)
    if material_style == ORANGE_TIME_GRADIENT_MATERIAL_STYLE:
        return copy.deepcopy(ORANGE_TIME_GRADIENT_LINGO_OBJECT_MATERIALS)
    raise ValueError("unsupported foreground material style: %s" % material_style)
