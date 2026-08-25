# Headless rendering and paper figures

The Linux authority host is not required to run a Blender GUI. Rendering is a
consumer of the motion artifact and is separate from model inference.

## Two rendering tiers

### Tier A: deterministic trajectory PNG

This tier is the first implementation target and needs no Blender, display
server, or scene mesh. It reads `global_jpos` and object translations and
produces:

- a top-down or fixed orthographic projection;
- pelvis and object trajectories;
- start, goal, and end markers;
- selected human skeleton poses at common keyframes;
- optional window/stage boundaries and labels.

It is suitable for debugging, long-horizon state traces, and a first paper
figure when a full room mesh is unavailable. The renderer should use a fixed
camera/config and deterministic alpha/compositing order.

### Tier B: mesh PNG/MP4

This tier reconstructs SMPL-X vertices from the exported pose/translation/beta
arrays and applies object transforms to rest meshes. It can use a headless
Linux backend (`pyrender` with EGL/OSMesa, or a controlled software path) or a
Windows Blender consumer. It requires scene mesh assets to reproduce the
furnished rooms in the InfBaGel figures.

The current CPU smoke implementation is
`tools.visualization.headless`. It uses Matplotlib's Agg backend and SMPL-X
forward kinematics, so it does not require Blender, EGL, or a display server.
It overlays fixed keyframes from one export and records the camera, frame
indices, asset hashes, and output hash in a render manifest. The default HOI
smoke uses a convex-hull proxy for the object; pass `--object-geometry full`
with an appropriate face budget when the full rest mesh is needed.

Use `--style paper --object-geometry full --max-faces 0` for an axis-free,
orthographic, content-cropped trajectory still. The full-face setting avoids
the disconnected triangles caused by naive face-stride subsampling. Paper
style is still a mesh compositor, not a furnished-scene reproduction.

If paired `[F,45]` hand-pose arrays exist, the renderer consumes them. When an
export such as P12 has no finger articulation, `--hand-pose-fallback mean`
uses the static SMPL-X natural mean hand and records `smplx_mean` in the render
manifest. It must not be described as generated finger motion.

`data/hosi_test/Scene_sdf` and occupancy are evaluation geometry. They are not
automatically equivalent to the room/furniture meshes used in a paper render.
If no scene mesh is available, the output must say so rather than presenting an
occupancy visualization as a reproduction of the paper scene.

## Multiple poses in one image

The trajectory-style images in the paper are made by selecting several
timesteps from one generated sequence and placing those poses in a common
camera/scene before compositing. They are not obtained by changing the model
or by concatenating unrelated frames.

The reproducible procedure is:

1. Select keyframes from the exported frame/stage timeline using a recorded
   rule (for example, start, evenly spaced interior frames, and final frame).
2. Reconstruct a human/object mesh for each selected frame.
3. Keep one camera, world transform, and scene asset fixed for all selected
   frames.
4. Draw later/earlier poses with a fixed alpha and palette, or render each
   pose to a transparent layer and alpha-composite them.
5. Record the exact frame indices, camera config, palette, alpha, and renderer
   version in the render manifest.
6. Add labels/circles/arrows in a separate composition step; these annotations
   are not motion data.

For videos, render one pose per frame instead of compositing all keyframes.
The same export and camera manifest must be retained so a video frame and a
trajectory figure can be traced to the same sequence.

## Windows/Blender hand-off

Windows may run Blender interactively or headless using the same NPZ and
scene/asset manifest. It must not rerun model inference merely to render. A
Blender-side importer should consume the canonical pose fields, set SMPL-X
animation, apply object transforms, and save PNG/MP4 to a new artifact
directory.

## Render manifest

Each render output records:

```text
source_motion_sha256
source_manifest_sha256
renderer_commit
renderer_backend
scene_asset_id_and_sha256
smpl_asset_sha256
camera_projection_and_parameters
selected_frame_indices
stage/window selection rule
image dimensions / fps
created_at
```

The output directory is write-once. A camera change, keyframe change, palette
change, or backend change creates a new render identity; it never overwrites a
previous figure or video.

## Failure boundaries

Missing scene meshes, missing SMPL-X assets, unsupported coordinate frames, and
bad frame/rate relationships are explicit render failures. They must not be
silently replaced with a different scene, a different body model, or an
unrecorded coordinate conversion.
