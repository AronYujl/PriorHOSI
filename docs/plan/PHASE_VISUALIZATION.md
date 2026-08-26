# Visualization Worktree Phase

Status: HOI and HSI visualization paths passed through shared-scene paper figures and verified videos; worker2 workflow implemented and dry-run validated; mixer work remains pending
Date: 2026-08-25 (Asia/Shanghai)  
Branch: `visualization/renderer`  
Worktree: `/data/yujinlun/InfBaGel-visualization`

## Purpose

This phase defines an isolated consumer-side visualization pipeline for
HOIPrior, HSIPrior, and the future PriorHOSI mixer. It must not change the
training, evaluation, configuration, registry, or frozen-core files owned by
`phase/01b-hoi` or `phase/01c-hsi`.

The pipeline has two boundaries:

```text
expert/mixer checkpoint + matching inference code
        -> motion export (the only model-dependent step)
        -> versioned motion artifact
        -> renderer / keyframe compositor (model-independent)
```

The exported artifact is the hand-off. Once it exists, changing a checkpoint,
sampler, denoising schedule, or mixer implementation must not be required to
re-render a figure or video.

## Isolation contract

The visualization worktree is based on `research/state-compositional-priors`
and lives on `visualization/renderer`. It is intentionally not a worktree of
either expert branch and does not merge or cherry-pick their in-progress
commits.

The following paths are out of scope for this branch:

- `code/priors/core/`;
- `code/priors/hoi/` and `code/priors/hsi/` model/training changes;
- expert trainer/evaluator behavior and expert Hydra configs;
- expert `docs/plan/` or phase summaries;
- expert `experiments/registry.jsonl` history.

Expert results are external immutable inputs. When an adapter eventually needs
expert code, it must consume a pinned integration snapshot or an exported
artifact; it must not edit an expert checkout while that branch is training.

## Scope of this design phase

1. Define a provisional, versioned NPZ motion-export contract.
2. Specify adapters for the existing HOI export, the planned HSI export, and
   a future mixer export whose stage metadata is not yet fixed.
3. Define a headless Linux renderer that needs no Blender GUI.
4. Define an optional Windows/Blender consumer that reads the same artifact.
5. Define keyframe selection and multi-pose compositing for paper figures.
6. Establish provenance and non-overwrite rules for exported artifacts.

No checkpoint is loaded and no GPU workload is started in this design phase.
No generated motion, checkpoint, scene mesh, video, or per-sample output is
tracked in Git.

## Planned subphases and gates

### V0 — Contract and boundary (passed 2026-08-25)

Deliver the schema document, artifact naming rules, coordinate-frame rules,
and worktree boundary. The gate passed with ten tests covering valid HSI and
HOI payloads, object-stream requirements, frame/rate consistency, finite
values, schema rejection, and provenance hash mismatch. The validator imports
only NumPy and the Python standard library; it does not import either expert
package.

### V1a — HOI read-only adapter (passed 2026-08-25)

The adapter reads the existing HOI `motion_params/*.pkl` without importing the
HOI package, selects one sample from the legacy candidate dimension, and
writes a new canonical NPZ plus an explicit legacy provenance manifest. The
gate passed with synthetic multi-sample fixtures and a real 42-frame HOI
pickle; the output validated with the V0 schema and preserved pose/object
frame counts. The real pickle predates the P12 frame repair, so the adapter
must reverse its human-only `yup_to_zup` storage convention before rendering.
The legacy source and generated artifacts remain outside Git.

### V1b — HSI read-only adapter (passed 2026-08-26)

The Phase 1C evaluator now exports native schema-3, non-pickle NPZ files. Add a
read-only adapter into the common visualization schema without importing or
editing the HSI worktree. The adapter must preserve the complete
`global_orient`, `body_pose`, `transl`, `betas`, `gender`, coarse
`global_jpos`, window/seam structure, caption, scene identity, and source
hash. Native `fps=10` describes the coarse rollout; the canonical artifact
must record `fps=30` for its `174` fine SMPL-X frames when
`interp_scale=3`, while retaining the 10 Hz value separately. Schema-3
`smplx_output_transform="identity"` means FK output stays in LINGO y-up world
metres and receives exactly one y-up-to-Blender-z-up conversion downstream.

Gate: synthetic tests reject malformed native exports and prove array/rate
preservation; one real schema-3 sequence validates as common schema 1; all
adapter outputs and manifests are write-once and bind the untouched source
NPZ, shard evaluation report, SMPL-X assets, and scene mesh by SHA256.

Implementation commit `0ff029a` adds the native schema-3 validator/adapter and
nine HSI/LINGO tests. The real `071-write:007162` source SHA256 is
`0ada35f65942fa1783084ea75701dcf6b9de6e4112a6f57f29960c902b56fb65`;
the normalized common-schema NPZ SHA256 is
`efb64e67ff4b57a717c97e91c8a2a490e395fcf4180dbca618dce43af72242f1`.
All SMPL-X pose, translation, shape, coarse-joint and stitch arrays are
unchanged. The adapter records 58 coarse frames at 10 Hz, 174 fine frames at
30 Hz, `interp_scale=3`, and a 5.8-second duration.

### V1c — Worker2 checkpoint-to-artifact workflow (approved 2026-08-25)

Provide one authority-side entry point whose only required scientific input is
an absolute checkpoint path.  The tool must inspect trusted checkpoint
metadata, select a version-controlled inference profile, and fail closed when
no profile matches.  A profile pins the inference source commit separately
from the checkpoint's training commit, because the P12 checkpoint records
training commit `25931627a7a5668598e3120f57762546306c13a7` while its validated
post-repair inference path is commit
`8742d1a3b88800161324a8e45c597ffafdcbb607`.

The authority controls worker2 only through the loopback reverse endpoint
`127.0.0.1:22216`.  Checkpoint/Git inputs and returned artifacts remain
worker2-initiated over direct campus routing to `10.184.17.253`, with
`ssh -F /dev/null`; Windows and its proxy are not data-plane participants.
Worker2 state lives under `/data2/yujinlun/infbagel-inference`, and each pinned
inference commit gets an immutable detached checkout rather than changing an
expert worktree.

Gate:

1. `start CHECKPOINT` derives the checkpoint hash, compatible profile, source
   commit, run id, worker paths, exact Hydra overrides, and return destination;
2. preflight verifies the clean pinned checkout, checkpoint hash, assets, idle
   selected GPU, and a fully resolved config before starting inference;
3. inference runs in a worker-owned persistent systemd unit and automatically
   creates per-file motion hashes and pushes raw motion, evaluation records,
   logs, and provenance into a non-overwriting authority staging directory;
4. a successful return atomically promotes staging to the final artifact root,
   while a failed/interrupted run remains inspectable and is never reused;
5. dry-run/unit tests prove command construction, profile matching, shell
   quoting, transfer direction, non-overwrite behavior, and expert-worktree
   isolation without SSH, checkpoint transfer, or GPU execution.

Implementation status (2026-08-25): the profile registry, authority CLI,
`start`/`status`/`retry-return` lifecycle, persistent unit command, atomic
return contract, documentation, and nine workflow-specific tests are present.
The complete repository suite passes 64 tests and registry validation passes
27 records.  A real dry-run against checkpoint SHA256
`722d83ee7755b051e2095ccd01d4094bacce99589e679f89379f54661fb43704`
selected `hoi-p12-armb`, pinned inference commit `8742d1a3...`, and produced
remote preflight plan SHA256
`bb5ef01721059b5c596a46b2c41e7facca6b5f80985d3172ce0afe8c6e602c73`.
Worker2's base Git repository, environment, data/SMPL-X assets, source commit,
direct campus route, and idle GPU 0 also passed a read-only check.

No duplicate GPU rollout was started for tool validation.  The first new
user-requested checkpoint job is the remaining live acceptance of the
automatic systemd generation-and-return path; V1c must not be labelled fully
passed until that job reaches atomic promotion.

### V2a — Headless HOI mesh smoke (passed 2026-08-25)

The CPU/Agg renderer produced a deterministic PNG from the real converted HOI
artifact on Linux without Blender or a display server. It selected frames
`[0,14,27,41]`, reconstructed SMPL-X on CPU, applied the object rest mesh, and
wrote a render manifest with source/asset/output hashes. The full furnished
scene and paper-quality camera remain a later gate. The smoke artifact root is
`/data/yujinlun/InfBaGel-visualization-artifacts/hoi-smoke-20260825` (outside
Git); the NPZ SHA256 is
`2f76c86c21d0e27eee908594249007d04e22888b9a00b478e359c28233a3dff9`, and the
PNG SHA256 is
`d20f74ba0f45c00057b79c9a2aa9b66aa21f741c16bf924d9ce62708c37f8298`.

### V2b — Headless 2D/3D rendering gate (pending)

Implement a CPU/headless renderer for trajectory figures and optional mesh
frames. Gate: it produces deterministic PNG output from a fixture on Linux
without Blender or a display server.

#### V2b.1 — P12 long-window still (passed 2026-08-25)

The V2a review found that P12 object arrays shaped `[3,42,...]` are three
consecutive autoregressive windows, not three candidate samples. Correct the
adapter with an explicit layout flag and retain the complete 126-frame human
and object timeline with seams `[42,84]`. Produce one write-once paper-style
still for `sub16_clothesstand_000` using the full SMPL-X and clothesstand
meshes, six deterministic keyframes, no axes, and an explicit natural
mean-hand fallback because P12 does not predict articulated fingers.

Gate: the real P12 pickle validates as one 126-frame NPZ, the renderer manifest
records window semantics, exact assets, full-mesh settings, camera/style,
selected frames, hand fallback, and hashes, and the visualization test suite
passes. Batch conversion of the other 437 provisional-model sequences is
deferred. MP4 generation remains outside this substep because the authority
environment currently has no maintained EGL renderer; the deterministic PNG
path must be accepted before adding a video backend.

Implementation commits `ee32cff` and `9b4d12d` make legacy layout selection
mandatory, flatten the P12 windows into one timeline, validate optional paired
hand-pose arrays, use a recorded SMPL-X mean-hand fallback when absent, and add
the axis-free orthographic `paper` style with full meshes and deterministic
content cropping. The complete repository suite passes 73 tests; research
metadata validation passes 28 registry records, one split, two evaluators, and
one training protocol.

The accepted write-once artifact is
`/data/yujinlun/InfBaGel-visualization-artifacts/hoi/p12-armb/visualization-v2b1-r2-20260825/sub16_clothesstand_000`.
It contains 126 frames with window lengths `[42,42,42]`, seams `[42,84]`, and
selected frames `[0,25,50,75,100,125]`; their window IDs are
`[0,0,1,1,2,2]`. The canonical NPZ SHA256 is
`6b5f3f2a59320feee4fb89f2d31bfd11e8b35a21326438bfaee368fd3b8fc99c`,
the PNG SHA256 is
`9ab3788e85120ec388eb932c671bee627f393b028e89dbdabe987b99ecd871d7`,
and the render-manifest SHA256 is
`40c2993a4c165cbc4244eab0ea9b223f790b13936b1b034b7168f0b663834ad6`.
The earlier `visualization-v2b1-20260825` directory is retained but superseded:
its PNG is identical, while its render manifest predates the explicit source
window fields. No Blender installation was required.

#### V2b.2 — P12 fixed-camera video (passed 2026-08-25)

Add one Linux-headless MP4 for the accepted 126-frame
`sub16_clothesstand_000` export before considering batch rendering. The video
must reuse the canonical NPZ and its complete SMPL-X/clothesstand meshes,
render exactly one synchronized human/object pose per source frame at 30 FPS,
use a fixed orthographic camera derived from the whole sequence, and retain the
V2b.1 natural mean-hand fallback. It must not rerun inference or interpolate
the three autoregressive windows.

Gate: a CPU PyTorch3D renderer streams RGB frames to the system FFmpeg
`libx264` encoder without Blender, EGL, or a display server; the write-once MP4
has 126 frames and a 30 FPS time base; and its render manifest records source,
assets, source-window semantics, camera bounds, encoder settings, dimensions,
frame count, duration, hand-pose source, renderer commit, and output SHA256.
Unit tests cover fail-closed settings, camera framing, FFmpeg command
construction, and overwrite refusal. Batch rendering of the remaining 437
sequences remains deferred.

Implementation commit `f0b8b8f` adds the CPU PyTorch3D/FFmpeg renderer and
passes the complete 82-test repository suite plus research-metadata
validation. The write-once artifact is
`/data/yujinlun/InfBaGel-visualization-artifacts/hoi/p12-armb/visualization-v2b2-video-20260825/sub16_clothesstand_000/sub16_clothesstand_000-motion126-fixed-camera.mp4`.
`ffprobe` verifies H.264/yuv420p, 640x480, 30 FPS, 126 frames, and 4.2 seconds.
The three source windows contribute 42 frames each and retain seams `[42,84]`.
The full clothesstand and SMPL-X meshes, fixed camera, and `smplx_mean` hand
source were visually checked at frames 0, 63, and 125. The MP4 SHA256 is
`9d86359594de8258e688b3e9d20c93c0ea78517f48a26fbd70634ea3fe39f672`;
its render-manifest SHA256 is
`f8f4cd0fee052ec7ac71c9aa4605fe7f052161ff7f115f1bb3b2a1469b156666`.

#### V2b.3 — OMOMO-quality Blender render (passed 2026-08-25)

Add a high-quality Linux-headless tier after review found that the V2b.2
PyTorch3D preview has flat plastic-like shading, no floor or cast shadows, and
visible raster edges. Reproduce the relevant rendering ingredients from the
local official OMOMO release without coupling rendering back into inference:
Blender 3.2 headless rendering, smooth mesh normals, Principled BSDF materials,
the hash-recorded OMOMO `floor_colorful_mat.blend` scene, its lighting and
floor, shadows, color management, and supersampled/anti-aliased output.

The canonical 126-frame NPZ remains the only motion input. A preparation step
may reconstruct one immutable mesh cache outside Git; Blender must consume
that cache without importing the training code or rerunning inference. Produce
both a 30 FPS video and one fixed-camera six-keyframe process figure from the
same cache, scene, material palette, camera policy, and renderer commit. Keep
the V2b.2 PyTorch3D path as the fast preview backend rather than overwriting it.

Gate: the new write-once artifact records hashes for the motion, source
manifest, mesh cache, SMPL-X assets, object mesh, OMOMO blend scene, Blender
binary/version, camera, lights/world/color management, Cycles/Eevee settings,
materials, floor, selected figure frames, video probe, and every output. The
video must contain 126 verified frames at 30 FPS; the process figure must show
six recorded frames with a common camera and floor. Review must confirm smooth
silhouettes, non-flat material response, a visible floor, and coherent cast or
contact shadows before this gate passes. The remaining 437 sequences stay
deferred.

Implementation commit `fe56a5d` adds the immutable mesh-cache builder,
standalone Blender-bundled scene consumer, Cycles renderer, FFmpeg verifier,
process-figure compositor, tests, and documentation. A complete 126-frame
low-sample orchestration smoke passed before the formal render; the repository
suite passes 91 tests and research-metadata validation.

The formal write-once artifact is
`/data/yujinlun/InfBaGel-visualization-artifacts/hoi/p12-armb/visualization-v2b3-blender-20260825/sub16_clothesstand_000`.
It uses Blender 3.2.0, the official OMOMO blend-scene SHA256
`bc7e2d7de5fe2129c9808c21a00629fd294d97ee52d8d1ca543bb3d748312f1f`,
Cycles CPU with 64 samples and denoising, smooth OMOMO blue/purple Principled
materials, the official Sun/world plus a recorded 400 W camera fill, Filmic
color management, a procedural staggered wood floor, and one sequence-fitted
orthographic camera. The natural `smplx_mean` hand fallback remains explicit.

`ffprobe` verifies the H.264/yuv420p video as 126 frames, 1024x768, 30 FPS,
and 4.2 seconds. Its SHA256 is
`d9374c52f3637bc47848f354b5a0586e0e912f4f870527a6240edc235c91ebe3`.
All 126 lossless source PNGs are retained with tree SHA256
`b97022bcea6e6f8f80a5b0d0ca56de064fc2d7c4465b14b943e337d040143f12`.
The 3104x1648 process figure uses frames `[0,25,50,75,100,125]` and has
SHA256 `65ec057978f8b56c722a9031ed015630230c566e6013c0c8b0114311ccec8a8d`.
The mesh-cache SHA256 is
`f5b08a9a6bba1655afd29e1a72840dc46cd413687f23c8f198f1974d11493959`;
the render-manifest SHA256 is
`d68ac3b962691df4586a49884a429d112e322acc786d410facdb02f9b82b1991`.
Visual review of the six formal frames confirms full floor coverage, smooth
silhouettes without the V2b.2 stair-step edges, non-flat material highlights,
coherent human/object shadows, and no first/last-frame camera clipping.

#### V2b.4 — Visualization-only ground correction (passed 2026-08-25)

The V2b.3 floor exposes source-motion support inconsistencies: across the
accepted 126-frame sequence the lowest human vertex remains 1.45--4.52 cm
above the 1.5 cm render floor, while the clothesstand penetrates it by more
than 1 cm in 60 frames and by up to 5.20 cm. A single shared rigid vertical
offset cannot satisfy both constraints; after enforcing object non-penetration
it would leave more than 2 cm of human float in 65 frames.

Add an explicitly visualization-only derived mesh-cache mode without changing
or overwriting the canonical motion NPZ. Smoothly ground the lowest human
support vertices, raise the object only when its mesh would penetrate the
floor, and preserve human/object proximity during detected interaction by
blending the object correction into the upper body while retaining the foot
correction at the floor. Store the per-frame foot, upper-body, object, and
contact-strength corrections plus the vertical weighting formula in the cache.

Gate: the derived cache and render manifest state `visualization_only` and
`evaluation_forbidden`, retain hashes of the untouched canonical motion and
uncorrected cache, and report pre/post floor gaps, penetration, contact ranges,
maximum rigid correction, maximum within-body vertical differential, and
contact-distance change. A three-frame comparison must be reviewed before a
new write-once 126-frame video and process figure are rendered. The corrected
cache must leave no object penetration and keep human support within 5 mm of
the floor without modifying source SMPL-X/object parameters or any V2b.3
artifact.

Implementation commit `b9a7e35` adds the contact-aware derived-cache mode,
fail-closed validation against the accepted uncorrected cache, correction
streams and audit statistics, CLI options, tests, and operating documentation.
Visual comparison of frames `[0,63,100,125]` covered the worst initial object
penetration, a lifted-object contact frame, simultaneous contact and floor
conflict, and the final grounded pose before the formal render was started.

The formal write-once artifact is
`/data/yujinlun/InfBaGel-visualization-artifacts/hoi/p12-armb/visualization-v2b4-grounded-blender-20260825/sub16_clothesstand_000`.
Its manifests mark the output `visualization_only=true` and
`evaluation_forbidden=true`, reference the untouched V2b.3 cache SHA256
`f5b08a9a6bba1655afd29e1a72840dc46cd413687f23c8f198f1974d11493959`,
and retain the canonical motion SHA256. Across 126 frames, human support is
within -0.50--0.50 cm of the floor and object penetration is zero. The
75-frame original-contact interval `[37,111]` changes nearest mesh distance by
less than 0.000005 cm. Maximum rigid vertical correction is 5.20 cm and the
maximum foot-to-upper-body correction differential is 7.04 cm.

`ffprobe` verifies the H.264/yuv420p output as 126 frames, 1024x768, 30 FPS,
and 4.2 seconds. The video SHA256 is
`b69ada9a6f98ae5525e99bb663dc2d6b1c635f0ad9cf489ed3f1ee90a1ad6ef7`;
the six-frame process-figure SHA256 is
`e20226dd37365791fa2bbe75aa02951652639a38c7a822112489c36e8b73c542`;
the derived mesh-cache SHA256 is
`df66e1247934c30c16448f7f94b6da5db47c65f19c023e497e78de31f87254de`;
the 126-frame PNG tree SHA256 is
`e1fe6df1a4a127ecc3d274d7a818b9fca0dd1ec3fe10e69ca2acf5a7d5d92503`;
and the render-manifest SHA256 is
`d8ae2aa62a14d4c7119b62a00d42f775bd0b3a3c12cf0cecc5a0e12b2eb65dec`.
The repository suite passes 92 tests and research-metadata validation passes
28 registry records, one split, two evaluators, and one training protocol.

### V3 — Paper composition

Implement fixed-keyframe selection, alpha compositing, labels, and montage
layout. Gate: the same input artifact and render config reproduce the same
figure hash; method comparison uses the same keyframes and camera.

#### V3a — OMOMO Figure 6-style multi-pose scene (passed 2026-08-25)

Replace the diagnostic 3x2 grid as the primary paper still with one wide
Blender image that instantiates several selected human/object mesh pairs in a
shared scene. Match the structural composition of OMOMO Figure 6: opaque
smooth-shaded meshes, one camera, one floor and lighting setup, coherent
shadows, and the original world-space displacement between poses. Do not fake
motion by translating poses for layout and do not merge this presentation
artifact back into the canonical motion or evaluation path.

Implement the composition as a consumer of an accepted immutable Blender mesh
cache. It must fail closed unless the cache hash matches both its cache
manifest and source render manifest. Record the exact ordered keyframes,
selection rule, shared camera, scene/material/light configuration, source
hashes, renderer commit, image dimensions, and output hash in a write-once
manifest. First render a five-keyframe grounded P12 prototype; reduce to four
or adjust the fixed indices only if visual review finds occlusion that obscures
the action. Keep the V2b.4 video and diagnostic grid unchanged.

Implementation commit `3c83142` adds the cache-only paper-figure orchestrator,
the shared-scene mode in the Blender consumer, fail-closed dual-manifest and
scene-hash validation, tests, and documentation. The five-frame prototype
`[0,31,63,94,125]` was rejected because the middle poses obscured each other.
The reviewed four-frame selection `[0,42,83,125]` clearly preserves approach,
interaction, transport, and final states without changing their world-space
positions. A separate animation-mode Blender smoke confirms that the existing
video path remains functional.

The formal write-once artifact is
`/data/yujinlun/InfBaGel-visualization-artifacts/hoi/p12-armb/visualization-v3a-multipose-20260825/sub16_clothesstand_000`.
It is a 1600x800 RGB PNG rendered with Cycles at 64 samples. All four opaque
human/object pairs share one complete-cache-fitted orthographic camera, wood
floor, OMOMO material palette, lights, shadows, and source world frame. The
image SHA256 is
`2df5844d6231f35b9a11d2d7d5611ad9c6a319572d258a80fb1b41ae846ed8ed`;
the figure-manifest SHA256 is
`9b0a4d60243df226b9ca46a0b4bacb36d5b28bbe022119e248aae049eeaef963`;
and the scene-report SHA256 is
`a301316a2f62de33204869d5f6cbefe770de1a7fa9e637471a23bb5e47612b6b`.
The figure manifest binds the V2b.4 mesh-cache SHA256
`df66e1247934c30c16448f7f94b6da5db47c65f19c023e497e78de31f87254de`
and source render-manifest SHA256
`d8ae2aa62a14d4c7119b62a00d42f775bd0b3a3c12cf0cecc5a0e12b2eb65dec`,
and marks the output visualization-only and evaluation-forbidden. The complete
repository suite passes 101 tests and research-metadata validation.

### V3b — LINGO-style HSI scene visualization (passed 2026-08-26)

Render one HSIPrior sequence in its matching full LINGO scene, independently
of training and inference. The first deterministic smoke sample is
`071-write:007162` (`write on blackboard with right hand`): it has 174 fine
frames, a visible 0.68 m root-trajectory span suitable for a shared-scene
multi-pose still, and only 0.000818 reported non-floor penetration mean. The
initial lower-error candidate `007212` was rejected at geometry preflight
because its complete trajectory spans only about 0.09 m and its opaque poses
would almost completely overlap. This selection avoids both a known-broken
input and an unreadable composition; it is not a best-of-N scientific result
and must not be used as quantitative evidence.

The scene source is
`Scene_mesh/071-write/mesh_low.obj`, not an occupancy grid. Because this OBJ
contains geometry only, the Blender consumer supplies a recorded neutral
indoor palette, soft area lighting, ambient world light, shadows, smooth human
normals, and a fixed elevated room camera. It must produce both a 30 FPS video
with one body per source frame and one wide shared-scene still containing
several opaque source poses at their unmodified world locations, matching the
structural presentation of the LINGO teaser. The source NPZ and its SMPL-X
parameters remain untouched; no presentation-only floor or collision repair
is allowed in this first HSI smoke because a global vertical correction could
break blackboard/hand alignment.

Gate: a low-sample alignment render confirms that the body and room share one
coordinate frame and the camera exposes the interaction. The promoted
write-once artifact then records native/canonical motion hashes, shard report,
scene mesh, SMPL-X assets, the single coordinate transform, fine FPS
derivation, selected still frames, Blender/FFmpeg settings and hashes, and
contains a verified 174-frame H.264 MP4 plus one multi-pose PNG. Large inputs
and outputs remain outside Git under the ignored `artifacts` link.

Implementation commit `0ff029a` adds the isolated adapter, immutable human
mesh cache, Blender-bundled scene consumer, camera-side dollhouse cutaway,
recorded three-class presentation palette, Cycles CPU render, FFmpeg/ffprobe
validation, shared-scene figure, tests and operating documentation. The first
smoke failed before rendering because a factory-empty Blender scene has no
World; the second proved that Eevee cannot open a display on this headless
host. Both failures remained unpromoted. Cycles smoke then exposed the closed
ceiling/walls and drove the explicit cutaway. The accepted style uses the
geometry-only OBJ without pretending to recover missing textures.

The formal write-once artifact is
`/data/yujinlun/InfBaGel-visualization-artifacts/hsi/b-v2-unguided/visualization-v3b-lingo-blender-20260826/071-write_007162`.
It uses Blender 3.2.0 Cycles CPU, 24 adaptive samples with denoising, a 10%
scene decimation followed by the recorded cutaway (1,947,290 source faces;
97,058 rendered faces), 1280x720 frames, and renderer commit `0ff029a`.
`ffprobe` verifies H.264/yuv420p at 30 FPS, 174 frames and 5.8 seconds. The MP4
SHA256 is
`5ee5d8bca511b231b80bdf088adff21dba2797026ac770b2556b73b588e941ae`;
the 1800x1000 shared-scene figure at frames `[0,25,45,75]` has SHA256
`411505e683ff9496bf2a4b7d42ffc7ad035d48652d1341bc7ce5f03b3e7cda2d`;
the human mesh-cache SHA256 is
`6b2c2c17d3e99ff0634f462b8527db84ad113e40e810cdc67396f1dc45facb25`;
and the render-manifest SHA256 is
`64497c5e23c78f76430e1fcbeab7d5cc77acf108bf623aa8f652adc4f6f4e565`.

Visual inspection of frames 0, 75, 100, 125, 150 and 173 confirms that the
body remains in the matching furnished room and reaches the board. No
vertical/collision correction is applied. The manifest records that the
lowest body vertex lies 4.58--5.72 cm below the scene floor; this is retained
rather than shifting the body and silently changing hand-to-board height.
The relaxed mean-hand fallback is also recorded as visualization-only because
the HSI export contains no articulated finger parameters. The repository
`tests/` suite passes 110 tests and research metadata validates 29 records
before the completion record below is appended.

### V4 — Mixer/long-horizon extension

Add optional `stage_id`, `state`, `window_id`, and guard-event metadata once
PriorHOSI output semantics are fixed. Gate: an episode can be rendered as a
single trajectory while retaining stage boundaries and seam locations.

## Current inputs and assumptions

- The HOI branch currently has a legacy SMPL-X/object motion-parameter export.
- The HSI branch is expected to export the same human-parameter family, but its
  final export is not yet treated as a stable contract here.
- PriorHOSI/mixer output format is intentionally unspecified at V0; the schema
  therefore reserves optional stage and routing metadata rather than freezing
  a mixer API prematurely.
- SMPL-X model files, object rest meshes, and scene assets are code-independent
  assets, but each render manifest must record their hashes.
- `data/hosi_test` SDF/occupancy files are evaluation geometry, not by
  themselves a replacement for the furniture/room meshes seen in paper figures.

## Non-goals

- changing expert training or model selection;
- deciding the mixer architecture or state-machine protocol;
- claiming that an export artifact is a scientific evaluation result;
- reproducing the authors' unpublished figure-layout script;
- requiring Blender for training, inference, motion export, or the fast
  PyTorch3D preview; the optional paper-quality tier uses a pinned portable
  Blender binary on the Linux authority host.

## Entry and exit points

Entry: this committed design on `visualization/renderer`.  
Exit for the current HOI subphase: the schema, HOI adapter, headless renderer,
isolation documents, synthetic tests, and real HOI smoke artifact are
committed/recorded on this worktree; expert branches remain untouched; no run
ID or GPU workload was created. HSI adapter and full scene renderer remain
pending until their inputs/assets are stable.
