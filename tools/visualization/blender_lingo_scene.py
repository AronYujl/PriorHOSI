"""Blender-bundled consumer for one HSI motion in a full LINGO room.

Only Blender's bundled Python, NumPy, and the standard library are used.  The
host process performs SMPL-X reconstruction and supplies an immutable cache.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector


def _arguments():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def _material(name, settings):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        raise RuntimeError("Principled BSDF node is unavailable")
    shader.inputs["Base Color"].default_value = settings["base_color"]
    shader.inputs["Roughness"].default_value = float(settings["roughness"])
    shader.inputs["Specular"].default_value = float(settings["specular"])
    return material


def _mesh_object(name, vertices, faces, material, smooth=True):
    mesh = bpy.data.meshes.new(name + ".Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update(calc_edges=False)
    mesh.vertices.foreach_set("co", vertices.reshape(-1))
    for polygon in mesh.polygons:
        polygon.use_smooth = bool(smooth)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _world_bounds(objects):
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    for obj in objects:
        for corner in obj.bound_box:
            value = np.asarray(obj.matrix_world @ Vector(corner), dtype=np.float64)
            lower = np.minimum(lower, value)
            upper = np.maximum(upper, value)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError("cannot determine imported LINGO scene bounds")
    return lower, upper


def _apply_cutaway(obj, settings):
    if not settings.get("enabled", False):
        return {"enabled": False, "removed_face_count": 0}
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    coordinates = np.asarray([vertex.co[:] for vertex in bm.verts], dtype=np.float64)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    depth = float(settings["boundary_depth_m"])
    threshold = float(settings["normal_axis_threshold"])
    sides = set(settings["sides_blender_z_up"])
    remove = []
    for face in bm.faces:
        center = face.calc_center_median()
        normal = face.normal
        if (
            ("z_max" in sides and center.z >= upper[2] - depth and abs(normal.z) >= threshold)
            or ("x_max" in sides and center.x >= upper[0] - depth and abs(normal.x) >= threshold)
            or ("x_min" in sides and center.x <= lower[0] + depth and abs(normal.x) >= threshold)
            or ("y_max" in sides and center.y >= upper[1] - depth and abs(normal.y) >= threshold)
            or ("y_min" in sides and center.y <= lower[1] + depth and abs(normal.y) >= threshold)
        ):
            remove.append(face)
    before = len(bm.faces)
    bmesh.ops.delete(bm, geom=remove, context="FACES")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return {
        **settings,
        "removed_face_count": len(remove),
        "pre_cutaway_face_count": before,
        "post_cutaway_face_count": len(mesh.polygons),
        "pre_cutaway_bounds_blender_z_up": [lower.tolist(), upper.tolist()],
    }


def _assign_surface_materials(obj, material_count, settings):
    if material_count < 3:
        return {"policy": "single_material"}
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    coordinates = np.asarray([vertex.co[:] for vertex in bm.verts], dtype=np.float64)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    floor_depth = float(settings["floor_depth_m"])
    wall_depth = float(settings["wall_depth_m"])
    normal_threshold = float(settings["normal_axis_threshold"])
    counts = [0 for _ in range(material_count)]
    for face in bm.faces:
        center = face.calc_center_median()
        normal = face.normal
        if center.z <= lower[2] + floor_depth and abs(normal.z) >= normal_threshold:
            material_index = 0
        elif (
            (
                center.x <= lower[0] + wall_depth
                or center.x >= upper[0] - wall_depth
            )
            and abs(normal.x) >= normal_threshold
        ) or (
            (
                center.y <= lower[1] + wall_depth
                or center.y >= upper[1] - wall_depth
            )
            and abs(normal.y) >= normal_threshold
        ):
            material_index = 1
        else:
            material_index = 2
        face.material_index = material_index
        counts[material_index] += 1
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return {
        **settings,
        "policy": "floor_wall_geometry_plus_uniform_furniture",
        "material_face_counts": {
            str(index): int(count) for index, count in enumerate(counts)
        },
    }


def _import_scene(path, settings, materials):
    before = set(bpy.context.scene.objects)
    # Blender's OBJ importer applies the source (-Z forward, +Y up) to Blender
    # conversion.  This is the same +90 degree X rotation used for the human
    # cache: [x,y,z] -> [x,-z,y].
    bpy.ops.import_scene.obj(
        filepath=str(path),
        axis_forward="-Z",
        axis_up="Y",
        use_image_search=False,
        use_split_objects=False,
        use_split_groups=False,
    )
    imported = [
        obj
        for obj in bpy.context.scene.objects
        if obj not in before and obj.type == "MESH"
    ]
    if not imported:
        raise RuntimeError("Blender imported no LINGO scene mesh")
    original_vertices = sum(len(obj.data.vertices) for obj in imported)
    original_faces = sum(len(obj.data.polygons) for obj in imported)
    import_matrices = [
        [list(row) for row in obj.matrix_world]
        for obj in imported
    ]
    ratio = float(settings["decimate_ratio"])
    cutaway_reports = []
    surface_reports = []
    for obj in imported:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        obj.data.materials.clear()
        for material in materials:
            obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = False
        if ratio < 0.999999:
            modifier = obj.modifiers.new("PriorHOSI.LINGO.Decimate", "DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.ratio = ratio
            modifier.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        cutaway_reports.append(_apply_cutaway(obj, settings["cutaway"]))
        surface_reports.append(
            _assign_surface_materials(
                obj,
                len(materials),
                settings["surface_palette"],
            )
        )
        obj.select_set(False)
    lower, upper = _world_bounds(imported)
    return imported, {
        "object_count": len(imported),
        "source_vertex_count": original_vertices,
        "source_face_count": original_faces,
        "render_vertex_count": sum(len(obj.data.vertices) for obj in imported),
        "render_face_count": sum(len(obj.data.polygons) for obj in imported),
        "decimate_ratio": ratio,
        "bounds_blender_z_up": [lower.tolist(), upper.tolist()],
        "obj_import_axis_forward": "-Z",
        "obj_import_axis_up": "Y",
        "obj_import_matrices": import_matrices,
        "obj_import_transform_applied_to_mesh": True,
        "cutaway": cutaway_reports,
        "surface_palette": surface_reports,
        "coordinate_transform": [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        "coordinate_transform_application_count": 1,
    }


def _camera_direction(elev_degrees, azim_degrees):
    elev = math.radians(float(elev_degrees))
    azim = math.radians(float(azim_degrees))
    direction = np.asarray(
        [
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ],
        dtype=np.float64,
    )
    return direction / np.linalg.norm(direction)


def _bounds_corners(lower, upper):
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )


def _fit_camera(camera, lower, upper, settings, width, height):
    target = (lower + upper) / 2.0
    direction = _camera_direction(
        settings["elev_degrees"], settings["azim_degrees"]
    )
    forward = -direction
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    relative = _bounds_corners(lower, upper) - target
    projected = np.stack([relative @ right, relative @ camera_up], axis=1)
    projected_lower = projected.min(axis=0)
    projected_upper = projected.max(axis=0)
    projected_center = (projected_lower + projected_upper) / 2.0
    target = target + right * projected_center[0] + camera_up * projected_center[1]
    span = np.maximum(projected_upper - projected_lower, 0.35)
    aspect = float(width) / float(height)
    ortho_scale = max(float(span[0]), float(span[1]) * aspect) * float(
        settings["padding"]
    )
    distance = max(float((upper - lower).max()) * 3.0, 10.0)
    eye = target + direction * distance
    camera.location = eye.tolist()
    camera.rotation_euler = (Vector(target.tolist()) - camera.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()
    camera.data.type = "ORTHO"
    camera.data.sensor_fit = "HORIZONTAL"
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
        "fit_bounds": [lower.tolist(), upper.tolist()],
    }


def _point_light(name, light_type, energy, size, location, target):
    data = bpy.data.lights.new(name, type=light_type)
    data.energy = float(energy)
    if light_type == "AREA":
        data.shape = "DISK"
        data.size = float(size)
    data.use_shadow = True
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = list(location)
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()
    return obj


def _configure_lighting(scene, bounds, settings):
    lower, upper = bounds
    target = ((lower + upper) / 2.0).tolist()
    span = upper - lower
    key = _point_light(
        "PriorHOSI.LINGO.Key",
        "AREA",
        settings["key_energy"],
        settings["key_size"],
        [lower[0] - span[0] * 0.15, lower[1] - span[1] * 0.10, upper[2] + 3.0],
        target,
    )
    fill = _point_light(
        "PriorHOSI.LINGO.Fill",
        "AREA",
        settings["fill_energy"],
        settings["fill_size"],
        [upper[0] + span[0] * 0.10, upper[1] + span[1] * 0.10, upper[2] + 1.5],
        target,
    )
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("PriorHOSI.LINGO.World")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = list(settings["world_color"]) + [1.0]
    background.inputs["Strength"].default_value = float(settings["world_strength"])
    return [key, fill]


def _configure_render(scene, config, width, height):
    scene.render.engine = config["engine"]
    if config["engine"] == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = int(config["samples"])
        scene.cycles.use_denoising = True
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.08
        scene.cycles.max_bounces = 4
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 2
        scene.cycles.transparent_max_bounces = 2
        scene.render.use_persistent_data = True
    elif config["engine"] == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = int(config["samples"])
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = float(
            config["lighting"]["ambient_occlusion_distance"]
        )
        scene.eevee.gtao_factor = float(
            config["lighting"]["ambient_occlusion_factor"]
        )
        scene.eevee.use_soft_shadows = True
    else:
        raise RuntimeError("unsupported Blender render engine")
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = config["color_management"]["view_transform"]
    scene.view_settings.look = config["color_management"]["look"]
    scene.view_settings.exposure = float(config["color_management"]["exposure"])


def _light_report(lights):
    return [
        {
            "name": obj.name,
            "type": obj.data.type,
            "energy": float(obj.data.energy),
            "size": float(getattr(obj.data, "size", 0.0)),
            "location": list(obj.location),
            "rotation_euler": list(obj.rotation_euler),
            "use_shadow": bool(obj.data.use_shadow),
        }
        for obj in lights
    ]


def main():
    args = _arguments()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_path = Path(config["cache"])
    scene_path = Path(config["scene_mesh"])
    frames_dir = Path(config["frames_dir"])
    output_figure = Path(config["output_figure"])
    scene_report_path = Path(config["scene_report"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    with np.load(cache_path, allow_pickle=False) as cache:
        human_vertices = np.asarray(cache["human_vertices"], dtype=np.float32)
        human_faces = np.asarray(cache["human_faces"], dtype=np.int32)
    frame_count = int(config["frame_count"])
    if human_vertices.shape[0] != frame_count:
        raise RuntimeError("human cache frame count does not match config")
    render_frames = [int(frame) for frame in config["render_frame_indices"]]
    figure_frames = [int(frame) for frame in config["selected_figure_frames"]]
    if any(frame < 0 or frame >= frame_count for frame in render_frames + figure_frames):
        raise RuntimeError("source frame index is outside cache timeline")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene_materials = [
        _material("PriorHOSI.LINGO.Scene.%02d" % index, settings)
        for index, settings in enumerate(config["scene"]["materials"])
    ]
    human_material = _material("PriorHOSI.LINGO.Human", config["human_material"])
    scene_objects, scene_geometry = _import_scene(
        scene_path, config["scene"], scene_materials
    )
    scene_lower = np.asarray(scene_geometry["bounds_blender_z_up"][0])
    scene_upper = np.asarray(scene_geometry["bounds_blender_z_up"][1])
    human_lower = human_vertices.min(axis=(0, 1)).astype(np.float64)
    human_upper = human_vertices.max(axis=(0, 1)).astype(np.float64)
    fit_lower = np.minimum(scene_lower, human_lower)
    fit_upper = np.maximum(scene_upper, human_upper)

    camera_data = bpy.data.cameras.new("PriorHOSI.LINGO.Camera")
    camera = bpy.data.objects.new("PriorHOSI.LINGO.Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    video_size = config["video"]
    camera_report = _fit_camera(
        camera,
        fit_lower,
        fit_upper,
        config["camera"],
        int(video_size["width"]),
        int(video_size["height"]),
    )
    lights = _configure_lighting(scene, (fit_lower, fit_upper), config["lighting"])
    _configure_render(
        scene, config, int(video_size["width"]), int(video_size["height"])
    )

    human = _mesh_object(
        "PriorHOSI.LINGO.Human.Animation",
        human_vertices[0],
        human_faces,
        human_material,
        smooth=True,
    )
    scene.frame_start = 0
    scene.frame_end = frame_count - 1
    for progress, frame in enumerate(render_frames):
        human.data.vertices.foreach_set("co", human_vertices[frame].reshape(-1))
        human.data.update()
        scene.frame_set(frame)
        scene.render.filepath = str(frames_dir / ("%05d.png" % frame))
        bpy.ops.render.render(write_still=True)
        print(
            "LINGO_RENDER %d/%d source=%d"
            % (progress + 1, len(render_frames), frame),
            flush=True,
        )

    human.hide_render = True
    figure_objects = []
    for order, frame in enumerate(figure_frames):
        figure_objects.append(
            _mesh_object(
                "PriorHOSI.LINGO.Human.%02d.Frame%05d" % (order, frame),
                human_vertices[frame],
                human_faces,
                human_material,
                smooth=True,
            )
        )
    figure_size = config["figure"]
    _configure_render(
        scene, config, int(figure_size["width"]), int(figure_size["height"])
    )
    # Refit for the figure aspect ratio while retaining identical world bounds,
    # view direction, and source positions.
    figure_camera_report = _fit_camera(
        camera,
        fit_lower,
        fit_upper,
        config["camera"],
        int(figure_size["width"]),
        int(figure_size["height"]),
    )
    scene.frame_set(0)
    scene.render.filepath = str(output_figure)
    bpy.ops.render.render(write_still=True)
    print(
        "LINGO_FIGURE poses=%d frames=%s"
        % (len(figure_frames), ",".join(str(frame) for frame in figure_frames)),
        flush=True,
    )

    report = {
        "blender_version": bpy.app.version_string,
        "engine": scene.render.engine,
        "samples": int(config["samples"]),
        "cycles": {
            "device": scene.cycles.device if config["engine"] == "CYCLES" else None,
            "denoising": bool(scene.cycles.use_denoising)
            if config["engine"] == "CYCLES"
            else None,
            "adaptive_sampling": bool(scene.cycles.use_adaptive_sampling)
            if config["engine"] == "CYCLES"
            else None,
            "adaptive_threshold": float(scene.cycles.adaptive_threshold)
            if config["engine"] == "CYCLES"
            else None,
        },
        "ambient_occlusion": {
            "enabled": bool(scene.eevee.use_gtao)
            if config["engine"] == "BLENDER_EEVEE"
            else False,
            "distance": float(config["lighting"]["ambient_occlusion_distance"]),
            "factor": float(config["lighting"]["ambient_occlusion_factor"]),
            "note": "GTAO is Eevee-only; Cycles uses path-traced contact shadows",
        },
        "scene_geometry": scene_geometry,
        "human_bounds_blender_z_up": [human_lower.tolist(), human_upper.tolist()],
        "camera_video": camera_report,
        "camera_figure": figure_camera_report,
        "lights": _light_report(lights),
        "world": {
            "color": config["lighting"]["world_color"],
            "strength": config["lighting"]["world_strength"],
        },
        "materials": {
            "scene": config["scene"]["materials"],
            "human": config["human_material"],
        },
        "color_management": config["color_management"],
        "composition": config["composition"],
        "rendered_frame_indices": render_frames,
        "selected_figure_frames": figure_frames,
        "scene_object_count": len(scene_objects),
        "figure_human_count": len(figure_objects),
    }
    scene_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
