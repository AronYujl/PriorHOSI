"""Blender-bundled entry point for the OMOMO-style render.

This file intentionally depends only on Blender's bundled ``bpy``, ``numpy``
and the Python standard library.  It is launched by the host-side Blender
orchestrators and must not import InfBaGel training code.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _arguments():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def _mesh_object(name, vertices, faces, material):
    mesh = bpy.data.meshes.new(name + ".Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update(calc_edges=False)
    mesh.vertices.foreach_set("co", vertices.reshape(-1))
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _copy_omomo_material(source_name, name):
    source = bpy.data.materials.get(source_name)
    if source is None:
        raise RuntimeError("OMOMO material is missing: %s" % source_name)
    material = source.copy()
    material.name = name
    return material


def _wood_material(settings):
    material = bpy.data.materials.new("PriorHOSI.ProceduralWoodFloor")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (760, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (500, 0)
    shader.inputs["Roughness"].default_value = float(settings["roughness"])
    shader.inputs["Specular"].default_value = 0.38

    coordinates = nodes.new("ShaderNodeTexCoord")
    coordinates.location = (-900, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 0)
    mapping.vector_type = "POINT"
    mapping.inputs["Scale"].default_value = (1.0, 5.5, 1.0)

    brick = nodes.new("ShaderNodeTexBrick")
    brick.location = (-430, 100)
    brick.offset = 0.5
    brick.offset_frequency = 2
    brick.squash = 1.0
    brick.squash_frequency = 2
    brick.inputs["Color1"].default_value = (0.72, 0.32, 0.055, 1.0)
    brick.inputs["Color2"].default_value = (0.43, 0.15, 0.025, 1.0)
    brick.inputs["Mortar"].default_value = (0.24, 0.095, 0.025, 1.0)
    brick.inputs["Scale"].default_value = float(settings["brick_scale"])
    brick.inputs["Mortar Size"].default_value = float(settings["mortar_size"])
    brick.inputs["Mortar Smooth"].default_value = 0.002
    brick.inputs["Brick Width"].default_value = 2.8
    brick.inputs["Row Height"].default_value = 0.32

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-430, -170)
    noise.inputs["Scale"].default_value = 38.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.65
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-180, -170)
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (0.58, 0.58, 0.58, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (1.0, 0.92, 0.76, 1.0)

    multiply = nodes.new("ShaderNodeMixRGB")
    multiply.location = (100, 100)
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 0.22
    multiply.inputs[2].default_value = (1.0, 1.0, 1.0, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.location = (260, -130)
    bump.inputs["Strength"].default_value = float(settings["bump_strength"])
    bump.inputs["Distance"].default_value = 0.018
    bump.invert = True

    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], brick.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(brick.outputs["Color"], multiply.inputs[1])
    links.new(ramp.outputs["Color"], multiply.inputs[2])
    links.new(multiply.outputs["Color"], shader.inputs["Base Color"])
    links.new(brick.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _camera_direction(elev_degrees, azim_degrees):
    elev = math.radians(float(elev_degrees))
    azim = math.radians(float(azim_degrees))
    # InfBaGel y-up direction rotated by [x, -z, y] into Blender z-up.
    direction = np.asarray(
        [
            math.cos(elev) * math.cos(azim),
            -math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ],
        dtype=np.float64,
    )
    return direction / np.linalg.norm(direction)


def _fit_camera(camera, arrays, settings, width, height):
    lower = np.minimum(
        arrays[0].min(axis=(0, 1)), arrays[1].min(axis=(0, 1))
    ).astype(np.float64)
    upper = np.maximum(
        arrays[0].max(axis=(0, 1)), arrays[1].max(axis=(0, 1))
    ).astype(np.float64)
    target = (lower + upper) / 2.0
    direction = _camera_direction(
        settings["elev_degrees"], settings["azim_degrees"]
    )
    forward = -direction
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)

    projected_lower = np.full(2, np.inf, dtype=np.float64)
    projected_upper = np.full(2, -np.inf, dtype=np.float64)
    for array in arrays:
        flat = array.reshape(-1, 3)
        for offset in range(0, flat.shape[0], 250000):
            relative = flat[offset : offset + 250000] - target
            projected = np.stack(
                [relative @ right, relative @ camera_up], axis=1
            )
            projected_lower = np.minimum(projected_lower, projected.min(axis=0))
            projected_upper = np.maximum(projected_upper, projected.max(axis=0))
    projected_center = (projected_lower + projected_upper) / 2.0
    target = target + right * projected_center[0] + camera_up * projected_center[1]
    span = np.maximum(projected_upper - projected_lower, 0.35)
    aspect = float(width) / float(height)
    # Blender's orthographic scale is the horizontal camera frame when sensor
    # fit is AUTO/HORIZONTAL; its vertical span is scale/aspect.
    ortho_scale = max(float(span[0]), float(span[1]) * aspect) * float(
        settings["padding"]
    )
    distance = max(float((upper - lower).max()) * 4.0, 8.0)
    eye = target + direction * distance
    camera.location = eye.tolist()
    camera.rotation_euler = (Vector(target.tolist()) - camera.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()
    camera.data.type = "ORTHO"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.shift_x = 0.0
    camera.data.shift_y = 0.0
    camera.data.ortho_scale = ortho_scale
    return {
        "projection": "orthographic",
        "location": list(camera.location),
        "rotation_euler": list(camera.rotation_euler),
        "target": target.tolist(),
        "ortho_scale": float(ortho_scale),
        "elev_degrees": float(settings["elev_degrees"]),
        "azim_degrees": float(settings["azim_degrees"]),
        "padding": float(settings["padding"]),
        "motion_bounds": [lower.tolist(), upper.tolist()],
    }


def _add_camera_fill(camera_report):
    data = bpy.data.lights.new("PriorHOSI.CameraFill", type="AREA")
    data.energy = 400.0
    data.shape = "DISK"
    data.size = 5.0
    obj = bpy.data.objects.new("PriorHOSI.CameraFill", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = camera_report["location"]
    target = Vector(camera_report["target"])
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def _create_floor(arrays, settings):
    lower = np.minimum(
        arrays[0].min(axis=(0, 1)), arrays[1].min(axis=(0, 1))
    )
    upper = np.maximum(
        arrays[0].max(axis=(0, 1)), arrays[1].max(axis=(0, 1))
    )
    margin = float(settings["margin"])
    x0, y0 = float(lower[0] - margin), float(lower[1] - margin)
    x1, y1 = float(upper[0] + margin), float(upper[1] + margin)
    z = float(settings["height"])
    vertices = np.asarray(
        [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return _mesh_object(
        "PriorHOSI.WoodFloor", vertices, faces, _wood_material(settings)
    )


def _scene_report(scene, camera_report, config, floor):
    lights = []
    for obj in scene.objects:
        if obj.type == "LIGHT" and not obj.hide_render:
            lights.append(
                {
                    "name": obj.name,
                    "type": obj.data.type,
                    "energy": float(obj.data.energy),
                    "color": list(obj.data.color),
                    "location": list(obj.location),
                    "rotation_euler": list(obj.rotation_euler),
                    "use_shadow": bool(obj.data.use_shadow),
                    "angle": float(getattr(obj.data, "angle", 0.0)),
                }
            )
    return {
        "blender_version": bpy.app.version_string,
        "base_blend_path": bpy.data.filepath,
        "engine": scene.render.engine,
        "cycles_device": scene.cycles.device,
        "cycles_samples": int(scene.cycles.samples),
        "cycles_use_denoising": bool(scene.cycles.use_denoising),
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "resolution_percentage": scene.render.resolution_percentage,
        "film_transparent": bool(scene.render.film_transparent),
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": float(scene.view_settings.exposure),
        "gamma": float(scene.view_settings.gamma),
        "camera": camera_report,
        "lights": lights,
        "floor": {
            **config["floor"],
            "name": floor.name,
            "material": floor.active_material.name,
        },
        "materials": config["materials"],
        "composition": config.get(
            "composition", {"mode": "animation", "selection_rule": "all_frames"}
        ),
    }


def main():
    args = _arguments()
    config_path = args.config.resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_path = _resolve(base, config["cache"])
    report_path = _resolve(base, config["scene_report"])
    render_mode = config.get("render_mode", "animation")
    if render_mode not in ("animation", "multi_pose"):
        raise RuntimeError("unsupported render_mode: %s" % render_mode)
    frames_dir = None
    if render_mode == "animation":
        frames_dir = _resolve(base, config["frames_dir"])
        frames_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cache_path, allow_pickle=False) as cache:
        human_vertices = np.asarray(cache["human_vertices"], dtype=np.float32)
        human_faces = np.asarray(cache["human_faces"], dtype=np.int32)
        object_vertices = np.asarray(cache["object_vertices"], dtype=np.float32)
        object_faces = np.asarray(cache["object_faces"], dtype=np.int32)
    frame_count = int(config["frame_count"])
    if human_vertices.shape[0] != frame_count or object_vertices.shape[0] != frame_count:
        raise RuntimeError("mesh cache frame count does not match Blender config")
    if render_mode == "animation":
        render_frames = [
            int(frame)
            for frame in config.get("render_frame_indices", range(frame_count))
        ]
        if not render_frames or any(
            frame < 0 or frame >= frame_count for frame in render_frames
        ):
            raise RuntimeError("render_frame_indices contains an invalid source frame")
        selected_frames = []
    else:
        selected_frames = [
            int(frame) for frame in config.get("selected_frame_indices", [])
        ]
        if (
            not selected_frames
            or len(set(selected_frames)) != len(selected_frames)
            or any(frame < 0 or frame >= frame_count for frame in selected_frames)
        ):
            raise RuntimeError("selected_frame_indices contains an invalid source frame")
        render_frames = []

    scene = bpy.context.scene
    for obj in list(scene.objects):
        if obj.type == "MESH":
            obj.hide_render = True
    human_material = _copy_omomo_material(
        config["materials"]["human_source"], "PriorHOSI.Human.OMOMOBlue"
    )
    object_material = _copy_omomo_material(
        config["materials"]["object_source"], "PriorHOSI.Object.OMOMOPurple"
    )
    if render_mode == "animation":
        human = _mesh_object(
            "PriorHOSI.Human", human_vertices[0], human_faces, human_material
        )
        manipulated_object = _mesh_object(
            "PriorHOSI.Object", object_vertices[0], object_faces, object_material
        )
    else:
        for order, frame in enumerate(selected_frames):
            _mesh_object(
                "PriorHOSI.Human.%02d.Frame%05d" % (order, frame),
                human_vertices[frame],
                human_faces,
                human_material,
            )
            _mesh_object(
                "PriorHOSI.Object.%02d.Frame%05d" % (order, frame),
                object_vertices[frame],
                object_faces,
                object_material,
            )
    floor = _create_floor([human_vertices, object_vertices], config["floor"])

    camera = scene.camera
    if camera is None:
        camera_data = bpy.data.cameras.new("PriorHOSI.Camera")
        camera = bpy.data.objects.new("PriorHOSI.Camera", camera_data)
        scene.collection.objects.link(camera)
        scene.camera = camera
    camera_report = _fit_camera(
        camera,
        [human_vertices, object_vertices],
        config["camera"],
        int(config["width"]),
        int(config["height"]),
    )
    _add_camera_fill(camera_report)

    scene.render.engine = config["engine"]
    scene.cycles.device = config["device"]
    scene.cycles.samples = int(config["samples"])
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.render.resolution_x = int(config["width"])
    scene.render.resolution_y = int(config["height"])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = config["color_management"]["view_transform"]
    scene.view_settings.look = config["color_management"]["look"]
    scene.frame_start = 0
    scene.frame_end = frame_count - 1 if render_mode == "animation" else 0

    report = _scene_report(scene, camera_report, config, floor)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if render_mode == "multi_pose":
        output_image = _resolve(base, config["output_image"])
        output_image.parent.mkdir(parents=True, exist_ok=True)
        scene.frame_set(0)
        scene.render.filepath = str(output_image)
        bpy.ops.render.render(write_still=True)
        print(
            "V3A_RENDER poses=%d frames=%s"
            % (len(selected_frames), ",".join(str(frame) for frame in selected_frames)),
            flush=True,
        )
    else:
        if frames_dir is None:
            raise RuntimeError("animation frames directory is unavailable")
        for progress, frame in enumerate(render_frames):
            scene.frame_set(frame)
            human.data.vertices.foreach_set("co", human_vertices[frame].reshape(-1))
            manipulated_object.data.vertices.foreach_set(
                "co", object_vertices[frame].reshape(-1)
            )
            human.data.update()
            manipulated_object.data.update()
            scene.render.filepath = str(frames_dir / ("%05d.png" % frame))
            bpy.ops.render.render(write_still=True)
            print(
                "V2B3_RENDER %d/%d source=%d"
                % (progress + 1, len(render_frames), frame),
                flush=True,
            )


if __name__ == "__main__":
    main()
