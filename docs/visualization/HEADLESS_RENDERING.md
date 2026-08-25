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

The implemented video entry point is `tools.visualization.video`. It uses
PyTorch3D's CPU rasterizer and streams raw RGB frames directly to system
FFmpeg/libx264, so it needs neither Blender, EGL, an X display, nor a temporary
PNG sequence. The camera is fitted once against every human and object vertex
in the complete motion, then held fixed. The default H.264 output is 640x480,
30 FPS, CRF 18, and `yuv420p` for broad Windows/browser compatibility.

```bash
"$INFBAGEL_PYTHON" -m tools.visualization.video \
  /absolute/path/to/motion.npz \
  --manifest /absolute/path/to/motion.manifest.json \
  --output /new/write-once/path/motion.mp4 \
  --render-manifest /new/write-once/path/motion.video.render.json \
  --smpl-models /absolute/path/to/smpl_models \
  --object-mesh /absolute/path/to/rest_object.ply \
  --object-rest-frame z_up --object-geometry full \
  --fps 30 --width 640 --height 480 \
  --hand-pose-fallback mean \
  --renderer-commit "$(git rev-parse HEAD)"
```

The renderer verifies the encoded dimensions, FPS, and frame count with
`ffprobe` before atomically promoting the MP4 and its manifest. Existing output
or manifest paths are rejected. For P12, `mean` remains a static natural SMPL-X
hand fallback because the checkpoint does not produce articulated finger
motion; the manifest records this as `smplx_mean`.

### OMOMO-style Blender quality tier

`tools.visualization.blender` is the high-quality offline path. Inspection of
the official local OMOMO repository showed that its released visualization
uses Blender 3.2/Cycles, smooth mesh normals, Principled BSDF materials, the
`floor_colorful_mat.blend` camera/light/world scene, Filmic color management,
and per-frame image rendering before video encoding. Those are structural
differences from the fast PyTorch3D preview.

The Blender tier reconstructs an immutable mesh cache with the verified
InfBaGel Python environment, rotates it from canonical y-up into Blender's
right-handed z-up coordinates, then launches Blender headlessly. Blender reads
only the cache. It reuses the OMOMO blue/purple materials and Sun/world setup,
adds a recorded camera-side fill light and procedural staggered wood floor,
uses Cycles denoising and smooth shading, and fits one orthographic camera to
all 126 frames. It retains the per-frame PNGs, encodes a verified H.264 MP4,
and makes a labelled 3x2 process figure from frames
`[0,25,50,75,100,125]`.

The locally verified portable binary is Blender 3.2.0 at
`/data/yujinlun/tools/blender-3.2.0-linux-x64/blender`. It was downloaded from
the official Blender release archive and verified against the official archive
SHA256
`07c9380518ee1ee1ee3d5353e47bf105569cb2860f8bf45a35743b4f8cd6b742`.
It is a machine-local rendering dependency, not a repository file.

```bash
"$INFBAGEL_PYTHON" -m tools.visualization.blender \
  /absolute/path/to/motion.npz \
  --manifest /absolute/path/to/motion.manifest.json \
  --output-dir /new/write-once/artifact-directory \
  --smpl-models /absolute/path/to/smpl_models \
  --object-mesh /absolute/path/to/rest_object.ply \
  --object-rest-frame z_up \
  --blend-scene /data/yujinlun/omomo_release/manip/vis/floor_colorful_mat.blend \
  --blender /data/yujinlun/tools/blender-3.2.0-linux-x64/blender \
  --width 1024 --height 768 --samples 64 --fps 30 \
  --hand-pose-fallback mean \
  --renderer-commit "$(git rev-parse HEAD)"
```

The entire artifact directory is staged under a unique hidden sibling and
promoted only after Blender completes, FFmpeg/ffprobe validate all 126 frames,
and the process figure and provenance manifest are written. It never overwrites
the faster V2b.2 output.

### Visualization-only ground correction

The optional `visual_contact_aware_v1` mode addresses floor inconsistencies
that become visible only after adding a physical-looking floor. It does not
rewrite SMPL-X parameters, object transforms, the canonical motion NPZ, or any
evaluation artifact. Instead, it derives a separate Blender mesh cache and
marks both its cache and render manifests with `visualization_only=true` and
`evaluation_forbidden=true`.

The correction grounds the lowest human support vertices, raises an object
only when its mesh penetrates the floor, and detects human-object contact from
the original meshes. During contact, a recorded height-based smoothstep blend
lets the human upper body follow the object's vertical correction while the
feet retain their support correction. This avoids destroying a visible grasp
by independently translating the complete human and object meshes. The cache
records all correction streams, thresholds, pre/post floor statistics, contact
ranges, maximum correction, within-body vertical differential, and contact
distance change.

This mode deliberately requires the accepted uncorrected cache as immutable
provenance. The renderer reconstructs the meshes again and refuses to proceed
unless their arrays and topology exactly match that cache.

```bash
"$INFBAGEL_PYTHON" -m tools.visualization.blender \
  /absolute/path/to/motion.npz \
  --manifest /absolute/path/to/motion.manifest.json \
  --output-dir /new/write-once/grounded-artifact-directory \
  --smpl-models /absolute/path/to/smpl_models \
  --object-mesh /absolute/path/to/rest_object.ply \
  --object-rest-frame z_up \
  --blend-scene /data/yujinlun/omomo_release/manip/vis/floor_colorful_mat.blend \
  --blender /data/yujinlun/tools/blender-3.2.0-linux-x64/blender \
  --width 1024 --height 768 --samples 64 --fps 30 \
  --hand-pose-fallback mean \
  --ground-correction visual_contact_aware_v1 \
  --uncorrected-cache /absolute/path/to/accepted/mesh-cache.npz \
  --renderer-commit "$(git rev-parse HEAD)"
```

Ground-corrected outputs are presentation aids only. They must never be used
for metrics, qualitative claims about physical plausibility, or comparisons
against uncorrected methods without applying and disclosing the same policy.

### OMOMO Figure 6-style multi-pose still

`tools.visualization.blender_trajectory` places several complete human/object
mesh pairs from one accepted cache into a single Blender scene. Unlike the
diagnostic 3x2 process grid, the result has one floor, one lighting setup, one
camera, and coherent depth occlusion and shadows across all selected poses.
The poses are opaque and retain their source world coordinates; the renderer
does not move them apart for layout or use transparency to fake a trajectory.

This paper-composition step consumes the existing mesh cache and performs no
SMPL-X reconstruction or inference. It checks that the cache SHA256 matches
both its own manifest and the source Blender render manifest, and checks that
the blend scene and ground-correction records agree before rendering.

```bash
"$INFBAGEL_PYTHON" -m tools.visualization.blender_trajectory \
  /absolute/path/to/accepted/mesh-cache.npz \
  --cache-manifest /absolute/path/to/mesh-cache.manifest.json \
  --source-render-manifest /absolute/path/to/render.manifest.json \
  --output-dir /new/write-once/multi-pose-figure-directory \
  --blend-scene /data/yujinlun/omomo_release/manip/vis/floor_colorful_mat.blend \
  --blender /data/yujinlun/tools/blender-3.2.0-linux-x64/blender \
  --frames 0 42 83 125 \
  --width 1600 --height 800 --samples 64 \
  --renderer-commit "$(git rev-parse HEAD)"
```

The output manifest records the ordered source frames, explicit selection
rule, unaltered-world-layout policy, complete-cache camera fitting, source
hashes, render settings, scene report, and image hash. It always marks the
figure `visualization_only=true` and `evaluation_forbidden=true`.

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
