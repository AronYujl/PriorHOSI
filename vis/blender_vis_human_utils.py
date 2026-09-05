import numpy as np
import json
import os
import math
import argparse

import bpy


def render_paired(config_path):
    """Render two motions in one unchanged room with one fixed camera."""
    from pathlib import Path
    from mathutils import Vector
    import bmesh

    config = json.loads(Path(config_path).read_text())
    output = Path(config['output_dir'])
    vertices = np.load(config['vertices'], mmap_mode='r')
    faces = np.load(config['faces'])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = config['engine']
    if config['engine'] == 'CYCLES':
        scene.cycles.device = 'GPU'
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'CUDA'
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == 'CUDA'
        scene.cycles.samples = int(config['samples'])
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 4
        scene.cycles.transparent_max_bounces = 8
        scene.render.use_persistent_data = True
    else:
        scene.eevee.taa_render_samples = int(config['samples'])
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 1.5
        scene.eevee.gtao_factor = 1.1
        scene.eevee.use_soft_shadows = True
    scene.render.resolution_x = int(config['width'])
    scene.render.resolution_y = int(config['height'])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'Medium High Contrast'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new('White studio')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .7
    scene.world = world

    def material(name, color, alpha=1.):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        shader = mat.node_tree.nodes['Principled BSDF']
        shader.inputs['Base Color'].default_value = tuple(color) + (1.,)
        shader.inputs['Roughness'].default_value = .72
        shader.inputs['Specular'].default_value = .25
        if alpha < 1:
            tree = mat.node_tree
            transparent = tree.nodes.new('ShaderNodeBsdfTransparent')
            mix = tree.nodes.new('ShaderNodeMixShader')
            mix.inputs[0].default_value = alpha
            tree.links.new(transparent.outputs[0], mix.inputs[1])
            tree.links.new(shader.outputs[0], mix.inputs[2])
            tree.links.new(mix.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
            mat.blend_method = 'BLEND'
            mat.use_screen_refraction = False
        return mat

    floor_mat = material('Floor · warm grey', (.68, .65, .60))
    wall_mat = material('Walls · pale grey', (.77, .79, .79))
    furniture_mat = material('Furniture · muted blue grey', (.45, .57, .58))
    human_mat = material('Human · warm gold', (.75, .46, .23))
    bpy.ops.import_scene.obj(filepath=config['scene_mesh'], axis_forward='-Z', axis_up='Y',
                             use_image_search=False, use_split_objects=False, use_split_groups=False)
    rooms = [obj for obj in scene.objects if obj.type == 'MESH']
    for obj in rooms:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        obj.select_set(False)
    room_points = np.concatenate([np.asarray([obj.matrix_world @ v.co for v in obj.data.vertices]) for obj in rooms])
    room_lower, room_upper = room_points.min(0), room_points.max(0)
    azimuth = math.radians(config['camera_azimuth_degrees'])
    elevation = math.radians(config['camera_elevation_degrees'])
    direction = np.array([math.cos(elevation)*math.cos(azimuth),
                          math.cos(elevation)*math.sin(azimuth), math.sin(elevation)])
    removed_count = 0
    for obj in rooms:
        obj.data.materials.clear()
        for mat in (floor_mat, wall_mat, furniture_mat):
            obj.data.materials.append(mat)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        remove = []
        for face in bm.faces:
            center = np.asarray(obj.matrix_world @ face.calc_center_median())
            normal = face.normal
            camera_wall = any(
                abs(normal[axis]) > .95 and (
                    (direction[axis] > 0 and center[axis] > room_upper[axis]-.06) or
                    (direction[axis] < 0 and center[axis] < room_lower[axis]+.06)
                ) for axis in (0, 1)
            )
            ceiling = center[2] > room_upper[2]-.06 and abs(normal.z) > .9
            if ceiling or camera_wall:
                remove.append(face)
            elif center[2] < room_lower[2]+.12 and abs(normal.z) > .8:
                face.material_index = 0
            elif center[2] > room_lower[2]+1.75 and abs(normal.z) < .2:
                face.material_index = 1
            else:
                face.material_index = 2
        removed_count += len(remove)
        bmesh.ops.delete(bm, geom=remove, context='FACES')
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()

    lower = np.minimum(room_lower, np.asarray(config['motion_bounds'][0]))
    upper = np.maximum(room_upper, np.asarray(config['motion_bounds'][1]))
    target = (lower + upper) / 2
    camera_data = bpy.data.cameras.new('Shared camera')
    camera = bpy.data.objects.new('Shared camera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = target + direction * max(10., np.max(upper-lower)*3)
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera_data.type = 'ORTHO'
    camera_data.sensor_fit = 'HORIZONTAL'
    corners = np.asarray([[x,y,z] for x in (lower[0],upper[0])
                          for y in (lower[1],upper[1]) for z in (lower[2],upper[2])])
    right = np.cross(-direction, np.array([0.,0.,1.]))
    right /= np.linalg.norm(right)
    up = np.cross(right, -direction)
    extent = np.ptp(np.stack([(corners-target) @ right, (corners-target) @ up], axis=1), axis=0)
    aspect = config['width']/config['height']
    camera_data.ortho_scale = float(max(extent[0], extent[1]*aspect)*1.07)

    for name, energy, offset, size in [('Key', 1200., (-3.,-4.,8.), 6.), ('Fill', 650., (4.,2.,6.), 5.)]:
        light_data = bpy.data.lights.new(name, 'AREA')
        light_data.energy = energy
        light_data.shape = 'DISK'
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = target + np.asarray(offset)
        light.rotation_euler = (Vector(target)-light.location).to_track_quat('-Z', 'Y').to_euler()

    def human_object(name, points, mat):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(points.tolist(), [], faces.tolist())
        mesh.update()
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        mesh.materials.append(mat)
        return obj

    body = human_object('Animated human', vertices[0, 0], human_mat)
    if config['video']:
        for frame in config.get('render_frames', range(config['frame_count'])):
            for arm, name in enumerate(('ground_truth', 'generated')):
                folder = output / name
                folder.mkdir(exist_ok=True)
                body.data.vertices.foreach_set('co', vertices[arm, frame].reshape(-1))
                body.data.update()
                scene.render.filepath = str(folder / ('%05d.png' % frame))
                bpy.ops.render.render(write_still=True)
            print('PAIRED_RENDER %s %d/%d' % (config['case_id'], frame+1, config['frame_count']), flush=True)
    body.hide_render = True
    scene.render.resolution_x = int(config['figure_width'])
    scene.render.resolution_y = int(config['figure_height'])
    for arm, name in enumerate(('ground_truth', 'generated')):
        figure_objects = []
        for order, frame in enumerate(config['keyframes']):
            frac = order / (len(config['keyframes'])-1)
            color = np.asarray([.80,.73,.60])*(1-frac) + np.asarray([.78,.44,.18])*frac
            alpha = 1. if order in (0,len(config['keyframes'])-1) else .28
            mat = material('%s time %d' % (name, frame), color, alpha)
            figure_objects.append(human_object('%s frame %d' % (name, frame), vertices[arm, frame], mat))
        scene.render.filepath = str(output / ('teaser_%s.png' % name))
        bpy.ops.render.render(write_still=True)
        for obj in figure_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    report = {'engine': scene.render.engine, 'render_device': 'GPU',
              'shared_camera': {'location': list(camera.location), 'rotation': list(camera.rotation_euler),
                                'ortho_scale': camera_data.ortho_scale},
              'exterior_cutaway_faces': removed_count, 'motion_transform': 'y-up to Blender only',
              'frame_count': config['frame_count'], 'keyframes': config['keyframes'],
              'scene_mesh': config['scene_mesh'], 'body_geometry_edited': False,
              'furniture_geometry_edited': False}
    (output / 'render_report.json').write_text(json.dumps(report, indent=2)+'\n')

if __name__ == "__main__":
    import sys
    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--")+1:]

    print("argsv:{0}".format(argv))
    if '--paired-config' in argv:
        render_paired(argv[argv.index('--paired-config')+1])
        sys.exit(0)
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment.')
    parser.add_argument('--folder', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='')
    parser.add_argument('--out-folder', type=str, metavar='PATH',
                        help='path to output folder which include rendered img files',
                        default='')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .blend path for 3D scene',
                        default='')
    parser.add_argument('--material-color', type=str, 
                        help='material, decides color',
                        default='blue')
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    # Load the world
    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    scene_name = args.scene.split("/")[-1].replace("_scene.blend", "")
    print("scene name:{0}".format(scene_name))
   
    obj_folder = args.folder
    output_dir = args.out_folder
    print("obj_folder:{0}".format(obj_folder))
    print("output dir:{0}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare ply paths 
    ori_obj_files = os.listdir(obj_folder)
    ori_obj_files.sort()
    obj_files = []
    for tmp_name in ori_obj_files:
        if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            obj_files.append(tmp_name)

    for frame_idx in range(len(obj_files)):
        file_name = obj_files[frame_idx]

        # Iterate folder to process all model
        path_to_file = os.path.join(obj_folder, file_name)

        # Load human mesh and set material 
        if ".obj" in path_to_file:
            human_new_obj = bpy.ops.import_scene.obj(filepath=path_to_file, split_mode ="OFF")
        elif ".ply" in path_to_file:
            human_new_obj = bpy.ops.import_mesh.ply(filepath=path_to_file)
        # obj_object = bpy.context.selected_objects[0]
        # if file_name == "00000.obj":
        #     human_obj_object = bpy.data.objects[str(file_name.replace(".ply", "").replace(".obj", ""))+".004"]
        # else:
        human_obj_object = bpy.data.objects[str(file_name.replace(".ply", "").replace(".obj", ""))]
        # obj_object.scale = (0.3, 0.3, 0.3)
        human_mesh = human_obj_object.data
        for f in human_mesh.polygons:
            f.use_smooth = True
        
        human_obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(0)) # The default seems 90, 0, 0 while importing .obj into blender 
        # obj_object.location.y = 0

        human_mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
        human_obj_object.data.materials.append(human_mat)
        human_mat.use_nodes = True
        principled_bsdf = human_mat.node_tree.nodes['Principled BSDF']
        if principled_bsdf is not None:
            # principled_bsdf.inputs[0].default_value = (220/255.0, 220/255.0, 220/255.0, 1) # Gray, close to white after rendering 
            principled_bsdf.inputs[0].default_value = (10/255.0, 30/255.0, 225/255.0, 1) # Light Blue, used for floor scene 

        # principled_bsdf.inputs[0].default_value = (153/255.0, 51/255.0, 225/255.0, 1) # Light Blue, used for floor scene 

        human_obj_object.active_material = human_mat
        # if args.material_color == "orange":
        #     human_obj_object.active_material = bpy.data.materials.get("orange")
        # elif args.material_color == "blue":
        #     human_obj_object.active_material = bpy.data.materials.get("blue")
        # elif args.material_color == "purple":
        #     human_obj_object.active_material = bpy.data.materials.get("purple")
        # elif args.material_color == "green":
        #     human_obj_object.active_material = bpy.data.materials.get("green")

        # bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, file_name.replace(".ply", ".png"))
        bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, ("%05d"%frame_idx)+".jpg")
        bpy.ops.render.render(write_still=True)

        # Delet materials
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        bpy.data.objects.remove(human_obj_object, do_unlink=True)     

    bpy.ops.wm.quit_blender()
