# Phase 5: Multi-task inference

## 2026-09-09 — Phase 5.1: standing handoff feasibility

The user approved confirming HSI-generated standing transitions from the settled
HOI chain and asked to inspect existing motion-inbetweening models. This session
implements only the bounded feasibility diagnostic on `phase/05a-stand-wait`.
The accepted starting point is Phase 2.34's P15+guide, relation-preserving edit,
R2 body guidance and terminal repair. The earlier learned-mixer and implicit-policy
phase proposals are superseded for this inference entry point by the user's
decision to keep the current trained system. Expert training is outside this run.

Hypothesis: a scene-conditioned HSI rollout, initialized from the actual generated
HOI terminal history and supplied a stationary pelvis goal and fixed text, can
produce a short upright, slow and released handoff. This is an exploratory
feasibility check on historically used HOSI-test tasks, not a held-out benchmark.

### Inputs and sampling

- Audit all 469 saved `correct_terminal_draw0` motions from the Phase 2.34 run.
  Draw 0 is fixed before inspection; retain the other draw's source identity by
  reference. Load existing source/geometry/terminal manifests by reference.
- Remove the two manufactured final interpolation frames before measuring source
  velocities. Measure the last 0.3 seconds. Standing proxy: neck-pelvis tilt at
  most 25 degrees, pelvis at least 0.65 m above floor, and at least one of the four
  foot markers within 0.08 m of the scene floor (world Y=0) throughout that tail.
- Released-grounded candidate: both native hand markers at least 0.08 m from the
  object surface throughout the tail; object lowest vertex within 0.05 m of the
  floor; object tail translation speed at most 0.10 m/s and angular speed at most
  0.5 rad/s. These are declared operational proxies, not measured contact forces.
- Choose at most 12 standing sources deterministically: released-grounded
  candidates first, then other standing sources as explicitly labelled stress
  cases. Within each group cycle the seven object names alphabetically, choosing
  ascending task ordinals and preferring unused scenes. Keep every audit result
  and exclusion reason; source selection uses no new HSI output.
- Each selected source receives exactly one native 16-frame HSI window with its
  two actual stride-3 historical frames fixed. It adds 14 samples (1.4 seconds at
  10 Hz). Hold all object geometry at the actual final pose for scene queries;
  feed HSI the existing known-empty object-channel view. Keep native R2 epoch222,
  500 diffusion steps, CFG=1, and posterior-coefficient scene guidance fixed.
- Primary text: `stand up and wait`. Paired wording diagnostic:
  `Stand upright and remain in place.` Both use seed42 and identical latent and
  posterior random draws, input history, stationary pelvis goal and step budget.
  No output-based prompt choice, resampling, endpoint-pose clamp or new training.
- The native `need_pelvis_dir` flag enables the entire pelvis-goal token, so it
  stays true to supply the stationary position; this API has no separate heading
  goal. Human goal remains the final source pelvis XZ. Scene target is empty and
  semantic timing uses pi=0/end_pi=48/seq_length=48. Decode
  through the existing native SMPL-X path with source betas and translation
  convention. Check the two conditioned poses after encoding and decoding.

### Measures and decision

Record each source and generated tail: upright tilt, root height/speed/angular
speed, foot height/speed, hand-object distance, object support proxies; all-frame
root drift, human-scene and human-object distances, seam position/velocity, and
history reconstruction error. The generated tail is its last 0.3 seconds with
terminal interpolation padding removed. Report short reference-pose holding as
a geometric diagnostic only, without treating frozen frames as generated motion.

Standing success requires tilt<=25deg, height>=0.65m, foot distance<=0.08m,
root speed<=0.15m/s, root angular speed<=0.5rad/s, root XZ drift<=0.10m,
and both hands >=0.08m from the object throughout the tail. Separately report
all-frame scene penetration mean<=0.005m/max<=0.05m, out-of-bounds, source eligibility
and object persistence. Full handoff success additionally requires an eligible
released-grounded source, geometry criteria and history error<=1e-4m.
These thresholds are explicit first-pass definitions and will not be tuned here.

At least 80% standing success on the selected sources supports testing this fixed
transition in a short task chain; report numerator/denominator and eligible/stress
strata, so a small eligible sample cannot establish general release support.
Lower success stops this text-driven recipe. No subsequent model or prompt tuning
follows automatically. External inbetweening review checks official pretrained
availability, endpoint requirements, body representation and scene/object scope.

### Execution and deliverables

Use the existing Hydra `code/test_infbagel_hosi.py` entry and a named diagnostic
under `code/mixer/`; add one config fragment and component tests. No new tool
script, expert/core changes, extra smoke workload or new hashing machinery.
Use the verified infbagel interpreter, one shared RTX3090 with documented
contention and <8GiB intended allocation; existing training remains running.
Archive resolved config and machine preflight with the standard experiment
manifest before sampling. All failures and partial outputs are retained.
Run the full authority suite and registry validation before the reportable run.
The first formal sample supplies runtime/memory evidence. The source scan and
sample calculations use CUDA. Save per-task motion and metrics, aggregate results,
paired task bootstrap, a concise phase summary and the external-model review.
Deliver three commits: preregistration, implementation, completion. This subphase
ends after confirming feasibility; benchmark construction and LLM execution are
the next user-directed work.

### Execution correction before denoising

The initial detached launch produced no Python output and its manifest was sealed
failed. Run r1 completed the unchanged 469-source audit (384 standing,22 eligible)
and selected12 eligible sources, then failed before the first denoising step:
HOSI `Scene_vis` keys use the raw scene ID, while the new adapter prepended `occ_`,
a convention belonging to another scene-loading path. Use the evaluator's raw
task scene key directly. Keep both failed runs, all source audit results and the
fixed thresholds/selection/prompts. The next run uses a new ID and the corrected
source commit; it is the same preregistered diagnostic.

### Measured scheduling amendment

Run r2 produced seven complete cells: both texts for tasks7,29,151 and the primary
text for task20. Each window took60–64seconds with about0.54GiB peak allocated
memory on the shared GPU. The serial process was then stopped and sealed aborted
for scheduling, preserving those seven outputs and the interrupted window.
The remaining17 fixed cells run concurrently on eight shared RTX3090 devices,
using one-task manifests and the same existing sampler/config. Task20 runs only
the missing secondary text; tasks43,277,230,57,384,64,419,71 run both texts.
Every cell still uses seed42, the same two source history frames and500steps.
The task20 lane's local prompt index0 denotes its sole secondary text; aggregation
uses the exact prompt string to map to the original two-arm protocol.
The final table contains each of the24 specified task/text cells exactly once.
The full469source audit and eligibility selection are reused from r2; per-lane
audits confirm only their assigned source. The source code is unchanged by this
resource amendment, and the latest1119-pass/6-skip authority suite still applies.
All timings describe contention; no exclusive-device speed claim is made.

### Completion

All24fixed cells are complete: literal text2/12handoffs, upright text1/12;
both12/12meet the registered scene-geometry tolerances. Both texts fail foot
support in8/12cases. The fixed80%gate fails and this text-only standing recipe
stops here. Full native history reconstruction remains within0.000358mm.
The complete record, operational failures, uncertainty and external-model review
are in [Phase5.1 summary](../phase_summaries/PHASE_5A_STANDING_TRANSITION.md).
Current experts remain the accepted input system. The next proposal should
specify both endpoint contexts for an existing inbetweening model.

## 2026-09-09 — Phase 5.2: pretrained inbetween deployment

The user supplied CondMDI code and Kimodo-SMPLX weights under
`/data/yujinlun/motion-inbetween` and authorized conda setup and transition
generation. Work on `phase/05b-inbetween`, using separate `condmdi` and `kimodo`
environments; keep the verified infbagel environment for native geometry.

Hypothesis: explicit endpoint contexts permit a usable HOI-to-standing bridge
where the previous fixed-text continuation was unreliable. This is a deployment
and adaptation diagnostic on the same twelve historical sources, not a new
benchmark or an expert-training comparison.

Use all twelve Phase5.1 selected sources, seed42, one sample per source/model.
The default endpoints are the actual last0.3s of HOI and a repeated natural
standing target at the same XZ and heading. Select one real standing reference
pose from these saved source motions before generation, preferring upright poses
with lowered wrists; archive the selected source/frame and selection measures.
Transfer its body rotations to each source's betas and gender, align heading,
and place its lowest body vertex on worldY=0. The target is a prescribed pose,
not an independently completed downstream task. Preserve the released object's
actual terminal transform throughout the bridge.

The complete conditioned clip spans2.0s: prefix0–0.3s, generated interval
0.3–1.7s, suffix1.7–2.0s. Kimodo uses61frames at30Hz; CondMDI uses41frames at20Hz
and is resampled to the same61native timestamps. Both receive full-body prefix
and suffix conditions. Use pose-only generation (the official empty-text
condition), so no LLM text encoder or prompt choice is needed. Kimodo uses its
official50-step DDIM, constraint CFG2, full-body plus hand/foot orientation
constraints and official constraint/foot postprocessing. CondMDI uses its
released random-joints conditional UNet,1000 diffusion steps, keyframe CFG1,
empty text and full-channel endpoint imputation. Keep raw model outputs as well
as native-body adaptations, recording any fitting or endpoint correction.

Measure representation roundtrip error, raw/final endpoint position and rotation
error, boundary joint/root velocity change, foot/body surface height, foot sliding,
root drift, human-scene and human-object penetration, out-of-bounds, synchronized
GPU generation/adaptation latency and memory. Reuse the prior scene/support
tolerances where applicable, but do not equate reaching a prescribed standing
target with learned standing success. Save every case, including failures, and
provide native motion files and animations. A deployment gate requires both
environments to load their actual checkpoints and generate finite conditioned
motions; task suitability remains a measured result, not a setup assertion.

Extend the existing Hydra inference dispatcher with a reusable external-bridge
component and one config fragment. Record external code versions and dependency
exports; reuse existing manifests for native input provenance. No new tools
script, hashing mechanism, extra smoke workload, core/expert edits or training.
Archive resolved configurations and machine preflight before reportable GPU
generation, use the standard experiment lifecycle, and run the full authority
suite plus registry validation. GPUs are shared with existing workloads;
allocate available headroom and report contention. Close this subphase with
setup instructions, all outputs and a phase summary.

The native adaptation fits the41free30Hzframes for240Adam updates (lr0.02),
with exact source/target poses in the20conditioned frames. Its fixed objective
is mean squared joint-target error +0.002rotation-matrix deviation from the
initializer +10framewise joint second-difference error. Kimodo initializes
from its decoded local rotations; CondMDI initializes its rotation-only fitting
from shortest-arc endpoint interpolation because its generated positions do not
specify native SMPL-X twists. Save the positional targets and fitting residuals;
these terms are kinematic conversion, with no scene/object optimization.
The reference pose ranks valid source frames by mean wrist height relative to
pelvis +0.25torso-tilt radians, with tilt<10deg, pelvis>0.65m, knee flexion<25deg,
both wrists below pelvis+0.10m and a foot marker within0.08m of worldY=0.
All thresholds are fixed before bridge sampling.
Kimodo's official MotionCorrection backend is a C++CPU implementation; record
its time separately. Model inference, native fitting and geometric evaluation
use CUDA.
Dispatch the two twelve-sample batches concurrently on shared RTX3090 devices
7(Kimodo) and6(CondMDI), each with four host threads. Native fitting/evaluation
uses device7 after both model jobs exit. Retain both exit codes and complete
logs even when one model job fails.

### Import-boundary correction

The initial run prepared all12inputs with maximum source-body reconstruction
error3.954e-7m, then both external jobs failed before checkpoint loading or
denoising. Python's package entry imported `mixer.__init__`, which eagerly loads
native PyTorch3D and the native `utils` module. Execute the external adapter as
its leaf file with only the external repository on PYTHONPATH. Keep the failed
manifest and logs. The replacement run reuses the immutable prepared inputs
and all prescribed settings; it begins at the generate stage with a fresh ID.

Runr1loaded both real checkpoints, then stopped before denoising on two upstream
API details. CondMDI overrides `_apply` and `train` without returning self;
call `to`, `eval`, and `requires_grad_` separately. Kimodo's end-effector
constructor leaves its joint-index tensors on CPU; use its supplied
`constraint.to(device)` before building GPU conditions. Retainr1as failed and
continue the same fixed protocol/inputs under a new identifier.

Before the first denoising call, make the HumanML observation mask reflect its
forward-difference definition: velocity/contact channels193–262at the last
prefix frame6depend on the unknown next frame and therefore remain unobserved.
All endpoint pose channels remain conditioned. The final static suffix defines
zero forward velocity by continued holding. This prevents the unobserved
interpolation scaffold from supplying a boundary velocity as if it were data.

Runr2completed all12Kimodo samples in6.2686s plus0.0718s official postprocessing,
with finite outputs. CondMDI stopped before denoising because its released
`000021.npy` has shape[180,52,3]; select its first22body joints as the author's
skeleton implementation does. Preserve allr2Kimodo raw outputs and use read-only
file references in the next run, which generates only the still-missing CondMDI
arm and then evaluates both. Synchronize CUDA before native-fit timing as well
as after it. No additional Kimodo samples or output-based source choices occur.

Runr3completed CondMDI body retargeting and GPU feature construction, then failed
before sampling at normalization-file lookup. The released absolute-root files
are `Mean_abs_3d.npy` and `Std_abs_3d.npy`; use those exact paths. Preserve this
failure and the existing Kimodo reference and continue the missing CondMDI arm
with a fresh run identifier.

### Phase5.2 completion

Runr4completed CondMDI's12fixed samples and reused all12r2Kimodo samples.
Both environments, native-body exports, full-motion concatenations and13videos
are complete. All24outputs are finite. Scene-geometry tolerances pass5/12Kimodo
and10/12CondMDI; Kimodo has smaller seam-velocity jumps. Both retain ground/contact
deficits, and shared target71itself penetrates the scene by10.316cm.
The deployment gate passes; physical suitability remains limited for these fixed
recipes. See [Phase5.2 summary](../phase_summaries/PHASE_5B_INBETWEEN.md) for all
results, endpoint attribution, uncertainty and retained failures.

## 2026-09-09 — Phase 5.3: adopt Kimodo and correct native contact

The user approved adopting Kimodo after the Phase5.2 comparison and the proposed
native contact postprocessing. Work on `phase/05c-kimodo-contact`. Kimodo is the
selected generator; use the existing twelve generated samples unchanged.
Hypothesis: joint optimization of native SMPL-X root/body trajectories against
floor, scene/object geometry and fixed predicted contact intervals reduces ground
penetration and planted-foot motion while retaining the smoother Kimodo seams.

Reuse all12Phase5.2conditions, source bodies, objects and Kimodo predictions.
Audit both boundary poses before editing with the original scene tolerances
(mean<=5mm/max<=5cm/in bounds). A failed boundary returns an explicit
`infeasible_endpoint` result and preserves its original motion; all12remain in
reports. Do not replace targets, resample, or silently count a rejected input as
corrected. Report the feasible subset separately and retain task71's known issue.

One fixed correction: optimize41freeframes10..50, retaining20contextframes and
all object transforms exactly. Use native SMPL-X surfaces, four foot patches
(heel/toe per side), and Kimodo's already saved four contact channels. Contact
patches use the32lowest target-pose vertices in the corresponding dominant
skinning region (7,10,8,11); constraints act on patch horizontal centroids and
lowest surface heights. Keep predicted contact labels fixed for before/after
metrics. Contact intervals shorter than3frames are omitted from optimization and
reported; intersecting fixed contexts and incompatible endpoint foot locations
are recorded rather than changing the given endpoint.

Use400Adamsteps, lr0.003, all22body-axis-angle rotations and root translation.
Objective terms (metres unless stated): body-joint deviation /0.05m; rotation
deviation /0.15rad; deviation of joint second differences /0.002m; seam second
differences /0.001m; per-frame maximum floor penetration /0.002m; maximum scene
penetration exceeding0.005m /0.005m; maximum object penetration exceeding0.005m
/0.005m; contact patch height /0.01m; fixed-contact horizontal displacement per
frame /0.001m. Coefficients respectively1,0.1,1,5,20,2,2,2,5. Mean squared terms;
all objectives logged. Original contexts contribute to seam evaluation, but only
freeframes contribute to geometry/contact height objectives. Report final fixed
budget iterate, no best-output selection or parameter sweep.

Save all12before/after native bridges and full source+bridge motions, endpoint
status, fixed-contact speed, foot height/contact coverage, original Phase5.2
metrics, joint/root/rotation correction size, iteration losses and synchronized
runtime/peak CUDA memory. Report shared GPU6allocation; parallel GPU calculations
inside each motion. Do not interrupt existing workloads. Source and geometry
identity reused by manifest reference; no new hashing mechanism or smoke workload.
Full authority tests and registry validation precede the standard manifest/start
and exact resolved Hydra config; paired10000seed42task bootstrap and CLI follow.

Adoption selects Kimodo independently of this diagnostic. A contact-correction
quality gate requires mean ground-depth and fixed-contact-speed reductions on
feasible cases, no scene-pass loss, no mean entrance/exit seam increase>0.02m/s,
exact contexts (<1e-5m), all finite. Report both improvements and deficits even
if this gate fails. No full task-chain rollout or further tuning follows here.
Deliver preregistration, implementation and completion commits with a phase
summary and before/after visualizations. CondMDI remains a historical reference.

Implementation makes Kimodo the default `inbetween.models` value. Historical
CondMDI comparison configs remain archived in their manifests; it is still
explicitly callable by model override. Native correction is an additional stage
of the existing dispatcher, and the existing renderer compares saved originals
with corrected motions. The fixed predicted contact intervals also define new
paired evaluation metrics; raising the foot cannot remove it from that metric.
No separate generation benchmark is needed because this stage reuses saved
predictions. Actual correction timing and memory are recorded in the formal run.
Assembly reporting uses51new30Hzframes (`bridge[10:]`); Phase5.2's numerical
exports already contain51but its narrative/assembly JSON said50. The source
prefix and actual exported frame arrays are retained exactly.

### Native gradient-path correction

The first correction run was stopped and sealed failed: the reused `decode_body`
helper is explicitly evaluation-only (`torch.no_grad`), so all geometric terms
were detached and the zero-initialized rotation-deviation term supplied zero
gradients. Saved completed cases have exactly zero motion changes. Use the native
`run_smplx_model` differentiable forward directly in optimization, retain the
evaluation decoder for measurements, and add a native-surface translation
Jacobian check. Keep all failed-run artifacts and use a fresh run identifier.
All budgets, objectives, masks, source motions and endpoints stay fixed.
The summary reduces scalar metrics; nested condition dictionaries remain in the
full per-case records and are excluded from numerical paired statistics.

### Phase5.3 completion

Kimodo is the default bridge generator. Native contact correction completed11
feasible cases; target71was rejected with its original motion retained. Scene
tolerances pass5/12to11/12 overall. On11feasible cases, mean per-clip free-frame
maximum ground depth42.665to0.180mm, fixed-contact surface speed0.063997to
0.012147m/s, and both seams improve. All registered correction gates pass.
Near-floor joint-speed remains0.178393to0.189588m/s, task29root planar drift
reaches11.55cm, and worst joint correction25.61cm. These remaining problems
qualify the prototype. All12motions, full stitches,13videos, paired statistics,
one retained failed run and1128-pass/4-skip suite are complete. Read
[Phase5.3 summary](../phase_summaries/PHASE_5C_KIMODO_CONTACT.md).


## 2026-09-09 — Phase 5.4: multi-task benchmark inputs and boundary contracts

The user approved extending HOSI-test with LINGO locomotion and static-object
interaction, checking composability at construction and using Kimodo at inference.
Work on `phase/05d-multitask-benchmark`; preserve all469 original tasks verbatim.
The full task-chain programme is split before implementation:5.4 builds and
checks task inputs, source boundaries and scene placements;5.5 executes the frozen
chains with actual generated histories, Kimodo and full-chain evaluation. This
session closes5.4. No expert training or new expert result transfer is involved.

Hypothesis: source-state, object-support and scene-geometry checks can construct
nonempty multi-task conditions independently of any evaluated model's outputs.
The benchmark has no complete recomposed motion GT. Source motion supplies
initial conditions, semantic timing and explicit target/contact references only.
Prior Phase5 generated standing outputs never select new benchmark membership.

Inventory the67HOSI files/469rows and all unique referenced OMOMO sequences.
Inventory unmirrored LINGO sources from the fixed seed42v3test partition, with
original language data_idx and sequence/frame bounds. First-version allowed
texts are walk, sit down on chair/office chair/sofa/couch and stand up from seat.
Hand-interaction==-1 is only one metadata check; explicit text semantics exclude
held props. Keep short clips and all exclusions in an audit instead of silently
turning them into viable16-frame stride3 windows. Record native30Hz, half-open
source intervals, full sequence terminal frame and separate model rollout timing.

Task schema records corpus-qualified source indices, target scene/geometry,
start_location, pelvis_goal, optional object_goal/scene_goal, source body identity,
heading, start/terminal context references, allowed duration, named contact target
and entry/exit requirements. Episode schema records ordered segments and explicit
transition edges, allowed contact changes and object persistence. Source scenes
and target scenes are distinct fields. Benchmark requirements stay independent of
Kimodo; the selected inference adapter consumes them. A chained successor uses
actual generated state rather than resetting to its source data_idx pose.

Audit first/last0.3s source contexts on CUDA: root/heading speed, feet support,
uprightness, hand/object distance, object lowest surface, speed and rotation.
Use the existing Phase5 support tolerances (floor<=5cm, object speed<=.10m/s,
angular speed<=.5rad/s) as explicitly labelled geometric proxies. Preserve raw
measures and per-reason counts. A suspended object requires a placement action;
release-only transitions cannot silently freeze it in mid-air.

Scene placement follows the user's scene-scope clarification. The default retains
the original67TRUMANS scenes and requires geometric support anchors for static
interaction. Native LINGO sources may verify source boundaries but cannot become
a cross-scene chain by copying their coordinates. Evaluate source/target boundary
body geometry at mean penetration<=5mm,max<=5cm,in bounds; report support/contact
separately. Keep all original469rows, accepted conditions and excluded candidates
with reasons. Freeze deterministic source ordering before any model output.
If target geometry cannot support a proposed chain, emit its explicit status;
do not replace the scene or action implicitly to meet a count.

Use one config fragment under existing Hydra `test_infbagel_hosi.py`, reusable
`mixer` benchmark components and component tests, no new tool script, hashing
mechanism, smoke workload or frozen core changes. Initial source scan and native
geometry use shared RTX3090CUDA with documented allocation. Archive exact resolved
config, machine preflight and standard manifest before the registered construction
run. Existing source identities are referenced from their manifests. No model
sampling is part of5.4; throughput micro-benchmarks and paired method bootstrap
are inapplicable to deterministic task construction.

Gate: all469original tasks preserved; all emitted source indices/bounds/splits
resolve; all published chains have explicit edges and scene/object state rules;
geometry and rejected-count records reconcile; at least one usable chain for each
published chain type; full authority tests and registry validation pass. A zero
usable count is a reported construction limitation, not permission to tune output
selection. Deliver tracked task manifests, aggregate audit, task-format guide,
geometry previews where placements are made, and a phase summary with5.5 entry.


### Fixed first construction recipe

Default scene scope is the original 67 TRUMANS scenes. The first published chain
is HOI -> walk -> sit. Keep at most 12 episodes from distinct scenes, visiting
scenes and original task ordinals in ascending order. Candidate eligibility uses
source conditions only: upright HOI terminal reference (tilt <=25deg, pelvis
>=.65m, foot marker within .08m) and a slow, floor-supported terminal object.
Sources that require placing a suspended object are recorded for a later chain
type. The terminal object orientation is the source terminal orientation transformed
by the task's initial-to-goal heading; record it as an added extension requirement,
never change the original HOSI task. Preserve its original object_goal exactly.

Use the first four eligible LINGO sit sources by ascending data_idx after requiring
an upright entry and seated exit (terminal pelvis <=.85m), plus the first eligible
walk source. Both are from the unmirrored fixed test partition. Transfer local
body poses to the HOI subject before target-geometry checks. The identity throughout
a chain is that HOI subject. Source motions are geometry witnesses, not output GT.

Search root XZ positions on the existing scene crop at 0.10m spacing and eight
45-degree headings. After body-identity transfer, one constant vertical shift
places the lowest context body vertex on the floor. Each context must retain a
foot marker within 8cm of the floor. A bilateral buttock surface patch must be within 3cm of
the scene SDF surface and have occupied scene 4cm below; this defines a geometric
support anchor, whose semantic seat identity is checked in the preview before
publication. Coarse joint and support queries precede full native endpoint-mesh
queries. At most 64 coarse candidates per source/task, ordered by straight-path
length then coordinates/heading, receive full checks. The walk path is a straight
segment of .75–3m, sampled at <=.10m spacing with a .25m horizontal body-clearance
proxy over heights .15–1.65m. Both static scene and persisted terminal object are
included. Failure of a straight path is an explicit first-version limitation.

The static interaction's entry/exit contexts use ten source frames each. Audit
native body penetration against scene and persisted object; require scene mean
<=5mm/max<=5cm, object maximum<=5cm, floor maximum<=1cm, and in-bounds. Source
endpoints failing these tolerances remain excluded with the measured reason.
A support anchor is checked independently from body penetration. No generated
motion, Kimodo result or visual aesthetic score participates in membership.
Publish geometry previews of accepted conditions for semantic support review;
retain all rejected candidate counts, cap exclusions and the original 469 rows.


### Input-reference correction after the first construction

The first run is retained as failed input construction. It emitted no candidate
chains: 147 source exclusions, 287 invalid HOI witnesses and 35 tasks whose four
static body transfers all failed foot support before any seating search.

All 469 supplied object-goal heights match raw frame
`language.start_idx[data_idx] + 3 * test_frames[-1]` (the current test frame is 15),
maximum difference 1.20e-7m. Their horizontal human-object separation also matches
that reference, maximum length difference 5.97e-7m. Full source-sequence ends
instead differ in object height by .31838m on average. Reusing the task-path
heading leaves a .33345m mean horizontal relation error even at the correct frame.

Correct the condition witness to that declared task reference frame and the
preceding ten-frame context. Keep the full sequence end as separate provenance
and a separate source-exit audit. Derive terminal yaw from the source human-object
XZ vector and the supplied pelvis/object-goal XZ vector, so both given goals are
reconstructed together. The initial HOSI task stays unchanged. This replaces the
incorrect full-sequence-end/path-heading assumption in the recipe above.

Static witnesses contain two independently prescribed endpoint contexts, not a
complete motion. One vertical shift across both contexts cannot ground both
states after body transfer. On the first accepted HOI body, the four fixed
sources' foot-distance maxima were .1074–.1264m under one shift; separate constant
shifts per ten-frame context give .0371–.0473m. Ground each context independently
by its native lowest surface. This preserves every local rotation and within-
context velocity; record both translations explicitly. It replaces the single-
shift rule above. The intermediate static action remains to be generated.

Retain the same four LINGO sit sources, walk source, candidate ordering, scene
scope, 12-episode cap, search budget, geometry/support thresholds and all original
469tasks. This is an input-contract repair, with no threshold tuning or output-
based source choice. The first native-body diagnostic was also retained after a
working-directory error; the corrected diagnostic uses the existing code/ cwd.
Use a new run ID and the complete authority suite after the source correction.

The support query also requires free space 4cm above each contact patch, paired
with occupied space 4cm below. This completes the directional support definition:
a point slightly inside a vertical wall must not qualify as a seating surface.
The existing 3cm contact-distance tolerance remains unchanged.


### Phase 5.4 completion

Run `p5-benchmark-multitask-inputs-r1-s42-20260909` publishes one input pilot in one
original scene: original HOSI task27 -> walk -> sit on the indicated low step.
The scene-mesh review names the actual low-step target in the task text and keeps
the LINGO office-chair text as source provenance. The frozen four-source pool,
geometry, candidate order and thresholds are unchanged by this annotation.

All 469 original task rows are preserved. Source audit resolves 150 OMOMO and
1214 eligible LINGO references, retaining 912 LINGO exclusions. The simple source
rule admits 7 original tasks; one fails the HOI witness check, five fail seating
placement, and one passes. The 462 source exclusions reflect this recipe's
floor-support/low-speed/upright requirements, not a proof of task impossibility.
Other support surfaces and explicit placement/settling tasks remain extensions.

The published manifest supplies three segments, two concrete bridge-target
contexts, persistent-object rules and actual-history inheritance. Native sitting
witness maximum scene penetration is 31.727mm near the right ankle, within the
registered 50mm tolerance; its bilateral support points are within 2.971mm of
the scene surface. Complete generated motion remains to be evaluated.

Full authority verification: 1146 passed,4 historical skips,203.48seconds.
All469 original rows,1364 source indices,469 goal reconstructions and both concrete
bridge contexts pass input validation. The maximum all-task object-goal reference
reconstruction error is 6.557e-7m. Construction took19.09seconds on shared GPU6,
with0.460GiB peak tensor allocation and0model samples. The first failed input
construction and all diagnostics are sealed and retained.

The input-pilot gate passes. No large-benchmark coverage or model-success claim is
made. Read [Phase5.4 summary](../phase_summaries/PHASE_5D_MULTITASK_BENCHMARK.md)
and `experiments/tasks/p5_multitask_pilot_s42_20260909.json` before Phase5.5.
This session ends at5.4; full actual-history expert/Kimodo execution is the next
subphase and has not started.

## 2026-09-10 - Phase 5.5.1: actual handoff and frame-budget audit

The user resumed session 01a084e8-a87e-7580-a4ac-324f907f7e5b and authorized
continuation. Read-only recovery found two upstream execution constraints in the
fixed pilot: the saved draw-0 HOI tail has object speed 0.616525 m/s against the
existing 0.10 m/s release limit, and its eight windows add 11.2 seconds against
the published 9.8-second HOI budget. These observations precede this diagnostic;
they are recorded here rather than presented as unseen experimental results.

Split 5.5 before implementation. Phase 5.5.1 on
`phase/05e1-handoff-audit` resolves actual handoff eligibility, native timing and
failure attribution. Phase 5.5.2 on `phase/05e2-multitask-execution` implements
the full actual-history expert/Kimodo chain after its entry contract is resolved.
This session completes only 5.5.1. Expert training and the published task set
retain their existing versions.

Hypothesis: evaluating the achieved body/object state with the published timing
and support contract identifies the concrete missing conditions for continuous
execution. A scene-feasible prescribed witness and an original goal-completion
flag alone are insufficient evidence for a release-and-walk transition.

Use the single published `multitask-hosi-027` episode, seed 42 and the existing
Phase 2.34 `correct_terminal_draw0` motion by reference. Keep the entire cached
motion and its original result. This is an import/handoff diagnostic; the cached
output was generated under the original native planner and is not a new rollout
under the extended benchmark budget. No new expert/Kimodo samples are drawn.

Native timing is explicit: a 16-frame stride-3 window with two coarse history
frames contributes 42 new 30-Hz samples. The first history spans 0.1 seconds.
Eight windows therefore contain 340 observed native frames, followed by two
held interpolation-padding frames in the legacy export. Seven windows permit
298 observed frames. Measure the fixed published-budget prefix and the full
observed cache separately; prefix inspection does not change the original
sampler's timing conditions or constitute a shorter-budget model run.

At both cutoffs measure the last ten actual frames (0.3 seconds between first
and last timestamps): pelvis/object goal errors, object orientation error to
the prescribed extension goal, translation/angular speed, lowest object surface,
native hand-object separation, body foot heights/speeds, uprightness, body/scene
and body/object penetration. Record whole-cache geometry and foot sliding too.
Use the existing thresholds: 0.10 m goal error, 0.05 m object support distance,
0.10 m/s object translation, 0.5 rad/s object rotation, 0.08 m hand release and
foot-marker support; scene mean/max 0.005/0.05 m, object max 0.05 m and floor max
0.01 m. Orientation-goal error is descriptive because the published manifest
does not define an angular-goal tolerance. Keep input-scene/body identity checks
and native reconstruction error (<=1e-4 m) alongside these measurements.

Load each of the two prescribed bridge contexts and recompute its body geometry
against the actual achieved persistent object at the matching cutoff. Measure
context position/rotation differences and foot support, retaining every result.
The first edge may proceed only when its source is within budget, at the stated
goals, supported/slow, geometrically admissible and natively reconstructed, and
its prescribed target is admissible and released. Failed prerequisites produce
named `blocked_by_predecessor` successor records. Do not synthesize missing-stage
metrics or report an unexecuted full-chain success rate.

Allowed diagnosis after a failed gate: save time-resolved object/root/foot
measurements, the full unchanged source and both cutoff contexts, and a trajectory
figure/scene preview identifying where the constraints fail. Preserve the original
completion flag and explicitly distinguish its goal-only protocol from the new
handoff checks. No endpoint replacement, frame holding, object settling, parameter
tuning, resampling or benchmark membership change follows this gate in 5.5.1.

Implementation extends the existing multitask Hydra stage, with one config
fragment and component tests for timing, padding and achieved-state guard logic.
GPU 0 runs native SMPL-X/geometry in the verified infbagel environment; record
concurrent host workloads and actual memory/time. Run the full authority suite,
registry validation and exact config resolution, then use the existing experiment
start/finish/register lifecycle. Reuse source provenance by reference. No new
hashing machinery, smoke workload, training benchmark or per-experiment script.
A paired-method bootstrap has no paired methods in this one-source deterministic
audit. Deliver compact results, raw diagnostics and a phase summary. The runtime
entry gate passes only if the fixed cached source meets all declared prerequisites;
an audited failure is a completed diagnostic with an unmet execution entry gate.

### Phase 5.5.1 completion

Run `p5-inference-handoff-audit-s42-20260910` completed the fixed-source audit.
All source identity checks passed and native reconstruction error was
`2.6656007889869215e-7 m`. Both prescribed bridge contexts pass membership geometry
checks against the achieved object transform. The published 9.8 s cutoff contains
298 observed frames; its source state misses pelvis/object goals by 12.285/24.843
cm and reaches object translation/angular speeds of 0.382 m/s and 0.905 rad/s.
The full cached source contains 340 observed frames plus two held padding frames,
or 42 frames beyond the budget; its goal errors improve to 2.566/2.668 cm but
object speeds remain 0.617 m/s and 0.520 rad/s. The original terminal repair was
not attempted (0 solver steps). The execution entry gate therefore fails and
walk/sit remain explicitly blocked with null metrics.

The run sampled zero models, used 0.425 GiB peak CUDA allocation and completed
the full authority suite with 1154 passed and 4 historical skips. The retained
environment failure from the restricted-CUDA first attempt is superseded by the
all-device rerun. Read [Phase 5.5.1 summary](../phase_summaries/PHASE_5E_HANDOFF_AUDIT.md)
and `experiments/results/p5_inference_handoff_audit_s42_20260910.json` before
the next subphase. The exact next entry is `phase/05e2-multitask-execution`,
which requires a new preregistered release/settling contract before any chain
sampling; no full-chain success claim is made here.

## 2026-09-10 - Phase 5.5.2a: source-only transition eligibility

The user approved a source-only membership rule for the next multi-task entry.
Split 5.5.2 before implementation: 5.5.2a builds and audits transition
eligibility from OMOMO/LINGO source actions and target-scene geometry; 5.5.2b
executes only the published source-eligible chains with actual history and
Kimodo. No model output participates in membership in either subphase.

Hypothesis: source endpoint contact state plus full source-motion SDF checks can
construct valid OMOMO/LINGO transition candidates without admitting chains that
require the evaluated model to repair an incompatible predecessor.

Build two ordered candidate families while retaining all original 469 HOSI rows:

- `omomo_to_lingo`: the OMOMO task-reference terminal must be upright, supported,
  object-supported and slow, and both hand markers must be released. The LINGO
  entry context is transformed to the target HOSI scene and its complete source
  interval is checked against scene/object SDFs before it can follow OMOMO.
- `lingo_to_omomo`: the OMOMO task initial context must keep both hands separated
  from the object throughout its entry context. The LINGO terminal context and
  complete source interval are transformed to the target scene and checked before
  OMOMO starts. The OMOMO initial object transform is preserved as the persistent
  target state.

Membership uses only source arrays, source text, source body identity, declared
HOSI goals, and target-scene SDF/object geometry. Source-scene coordinates remain
provenance and are never copied into the target scene. Every rejected pair keeps
its direction, source IDs, measured reasons, frame interval and geometry values.
The original benchmark and the source-only extension have separate tables.

The source audit uses a preregistered bounded pool for each direction: the first
8 eligible LINGO sources by `data_idx` for each action type are attempted. This
keeps the complete-frame SDF audit finite and reproducible while retaining the
full eligible LINGO catalogue and explicit `pool_not_attempted` records. The
pool is source-ordered and contains no model-output score.

The source path check evaluates every source frame after target placement, not a
straight-line proxy alone. It includes the persistent movable object whenever
the direction carries one. The registered scene thresholds remain mean/max
penetration 0.005/0.05 m, object max 0.05 m, floor max 0.01 m and no out-of-bounds;
entry/release uses the existing 0.08 m hand separation and support/speed proxies.

This subphase publishes candidate manifests and diagnostics only. It does not
reuse `correct_terminal_draw0`, Kimodo outputs or any other generated motion to
select candidates, and it does not claim a generated chain result. The next
subphase evaluates actual predecessor guards and all transition frames.

### Phase 5.5.2a completion

Run `p5-multitask-source-eligibility-r1-s42-20260910` audited all 469 OMOMO tasks,
1214 eligible LINGO sources and 912 LINGO exclusions using source state and full
target-scene geometry only. It accepted three `omomo_to_lingo` candidates and
zero `lingo_to_omomo` candidates. All 469 reverse-direction OMOMO initial states
failed the source hand-release guard; this zero is independent of the bounded
LINGO pool. Forward failures were 460 OMOMO state exclusions, 2 OMOMO geometry
failures and 67 full LINGO source-interval geometry failures.

The accepted source intervals pass the registered SDF/bounds checks. No model
output or generated motion was used, and no chain success is claimed. The first
run failed on a per-frame object-position broadcast error and is retained; the
corrected r1 run completed in 14.57 seconds on GPU 0. Full authority verification
after the fix is 1157 passed and 4 skipped. Read [Phase 5.5.2a summary](../phase_summaries/PHASE_5F_SOURCE_ELIGIBILITY.md)
and `experiments/results/p5_multitask_source_eligibility_s42_20260910.json`
before 5.5.2b actual-history execution.

## 2026-09-10 - Phase 5.5.2a.1: grasped OMOMO entry extension

The strict reverse family produced zero candidates because every audited OMOMO
initial context has a hand marker within 0.08 m of its object. The user approved
allowing this source state. This is a new source-only extension, not a rewrite of
the strict result: keep `lingo_to_omomo` with released initial hands as its own
empty table and add `lingo_to_omomo_grasped_entry` for the newly allowed state.

Hypothesis: an OMOMO initial context that already grasps its object can follow a
LINGO source action when the source contact is explicit, the object transform is
continuous, and a Kimodo endpoint bridge remains scene-feasible. Source
membership still cannot read model output; Kimodo is only the fixed runtime
bridge and its contact preservation is a measured execution guard.

For `lingo_to_omomo_grasped_entry`, require the OMOMO initial context to have at
least one hand marker within 0.08 m of the object in at least half of its ten
source frames and at its first frame. Source contact is a geometric proxy for
grasp, not a force or finger-closure measurement. Require scene/object/floor/bounds
checks and geometric support of the initial object by floor or scene. The target
is the first OMOMO native pose repeated for ten prescribed frames. This static
grasp target permits the subsequent expert to inherit actual history without
claiming to reproduce source entry velocities. Record the original moving context
separately. Use `bridge_contract=kimodo_acquire_contact` and
`object_policy=fixed_initial_transform`.

Read-only review found that all three old forward candidates are seated-start
stand-up actions whose full trajectories were vertically lifted 0.426-0.430 m
to align pelvis heights. SDF clearance alone accepted these unsupported motions.
The old results remain historical artifacts with this defect recorded. Replace
full XYZ alignment with yaw/XZ alignment and one whole-clip native-surface ground
translation. Check foot support in every source frame and bilateral scene seat
support throughout each seated context. Use the original native HOSI start yaw
for the OMOMO initial body/object transform. These representation fixes apply to
both strict families and the grasped-entry family. Preserve all 469 task rows and
record every cap/scene skip, source failure and full-geometry attempt explicitly.

At runtime, Kimodo receives the LINGO terminal as its prefix and the OMOMO
prescribed static grasp as its suffix. Contact may be acquired during the bridge;
the suffix must recover the source contacting hand(s). Require exact native body
contexts (<1e-5 m), fixed object transform, scene mean/max <=0.005/0.05 m,
body-object max <=0.05 m, floor max <=0.01 m and finite full-frame output. Record
foot sliding/contact coverage, joint/root/rotation seams, contact time series,
object support and every failure. These are kinematic proxies, not physical-grasp
success. Kimodo never chooses benchmark membership.

Use the same bounded pool of eight LINGO sources per action type and the existing
SDF thresholds, at most one accepted candidate per scene/direction and twelve per
direction. Publish all strict and grasped attempts and the fixed candidate list
before generating bridges. On `phase/05e2a1-grasp-entry`, this one subphase includes
source construction and a separate bridge diagnostic of every accepted grasped
candidate. Use the existing Kimodo 50-step empty-text recipe, seed42, one sample
per candidate, 61 frames, 10-frame prefix/suffix and 41 free frames; keep the
native 240-step fit and 400-step contact correction from Phase5.3. Preserve raw,
adapted and corrected motions and scene previews. Full expert-chain rollout is a
later subphase; bridge results describe source-endpoint feasibility only.

Use GPU1 for native construction/correction and GPU2 for Kimodo, recording live
contention and memory. Reuse existing Hydra/experiment entry points and source
provenance; add no tool script, hashing mechanism or smoke workload. Run the full
authority suite and registry validation. No paired-method bootstrap applies to
this single-method bridge diagnostic. A zero-candidate or failed-bridge result
retains its registered definitions and completes the diagnostic without tuning.

Implementation recovery on 2026-09-10 matches the native initial transform exactly:
remove the source root's extrinsic `zxy` heading, then apply
`atan2(-goal_delta_z, goal_delta_x) + pi/2` to the body and object together.
The interrupted draft omitted heading removal and used the opposite Z sign.
The existing `hand_distance_m=0.08` is the shared contact threshold for source
audits and bridge measurements. Component regression checks cover this placement,
whole-clip height preservation, source support and exact native bridge contexts.
Native contact markers are the left index/middle fingers (28-joint indices
24/25) and right index/middle fingers (26/27). The interrupted implementation
included indices 22/23, which map to SMPL-X eyes, and grouped a left finger with
the right hand. The source and bridge audits now share the corrected per-hand
distance calculation. This corrects marker identity at the registered 0.08 m
threshold; historical audit artifacts remain intact.

The source inventory completed before bridge dispatch with 5 forward and 12
grasped-entry candidates. A bridge interface review then corrected its object
SDF layout: native metrics/contact correction take `[1,D,H,W]`, while the source
geometry query takes `[1,1,D,H,W]`. A component regression exercises both distance
queries and the correction gradient on an analytic SDF. The bridge fix has its
own implementation commit and consumes the completed source manifest by reference.

### Phase 5.5.2a.1 completion

The source run `p5-multitask-source-eligibility-grasped-s42-20260910` retained all
469 original task rows and published 5 forward, 0 strict reverse and 12
grasped-entry reverse candidates. The grasped family reached the registered cap;
112 later rows are explicitly unattempted by that cap. All accepted LINGO inputs
come from two walking clips of 52 and 109 frames. The audit retains all 1407
task/direction rows and 45 complete-source pair attempts.

Run `p5-multitask-source-bridges-grasped-s42-20260910` generated the fixed twelve
Kimodo bridges once. All twelve pass the registered native-context, finite-output,
full-frame geometry, suffix-contact and fixed-object gates. Maximum body-scene,
body-object and floor penetration are 36.492, 13.661 and 3.432 mm; the maximum
required suffix hand distance is 38.434 mm. Context poses/translations, source
prefixes and object transforms are preserved exactly in the saved arrays.

Native correction changes fixed predicted foot-contact speed from 0.05158 to
0.01254 m/s and entry/exit joint-velocity jumps from 0.06448/0.04902 to
0.00353/0.00445 m/s. Local rotation jumps increase from 1.082/0.914 to
2.275/1.714 degrees; near-floor joint speed changes from 0.13210 to 0.13305 m/s.
These costs remain in the report. Full expert-chain execution is the next
subphase, with actual predecessor histories and its own preregistered guards.

Final authority verification is 1174 passed and 4 historical skips. All twelve
videos and 1521 native stitched frames were checked, and both reportable runs
completed. Read [the handoff summary](../phase_summaries/PHASE_5G_GRASPED_ENTRY.md)
and `experiments/results/p5_multitask_grasped_entry_s42_20260910.json` before
Phase 5.5.2b. This closes the approved source-endpoint diagnostic.

## 2026-09-10 — Phase 5.5.2b.1: expanded sources and actual-history pilot

The user accepts the current bridge quality, authorizes continued execution and
asks for substantially more than seventeen candidates. Work on
`phase/05e2b1-expanded-history`. Split the remaining execution work before coding:
5.5.2b.1 publishes the expanded source inventory and executes a fixed 24-chain
actual-history pilot; 5.5.2b.2 is the later evaluation of the full expanded table.
This session closes only 5.5.2b.1.

Hypothesis: the source cap and repeated first-success search conceal useful
coverage, and a source-diverse fixed pilot can exercise actual HSI → Kimodo → HOI
history inheritance without allowing generated outputs to choose membership.
Read-only counts are 1,214 eligible LINGO sources (1,045 walking, 169 static),
18 LINGO scene families; 1,199 sources are at most ten seconds long. The preceding
17 candidates use two walking sources. Its grasped reverse table skipped 112
tasks at the twelve-episode cap. Unsupported OMOMO objects remain an upstream
limitation; do not move them or relax support/penetration thresholds to grow the
table.

### Expanded input contract

Retain all 469 original task definitions and the previous 17 accepted source
pairs/witnesses by reference, then add distinct task/direction/LINGO-source pairs.
Within each of the two LINGO action types, take at most 128 sources of duration
<=10 s by round-robin source scene-family and exact source text; ascending
`data_idx` breaks ties. Preserve the complete catalog and explicit pool exclusions.
For each task/direction, try at most 48 additional source pairs, alternating the
two action types and preferring less-used source IDs and source families. Allow
three candidates per task/direction, nine per scene/direction, eight uses of a
new LINGO source per direction and 192 candidates per direction, including the
inherited candidates. These are fixed compute/diversity limits, not quality
targets. Keep every accepted/rejected pair and every unattempted reason.

Membership retains native yaw/XZ placement, whole-clip height preservation,
complete source foot/seat support, 8 cm initial hand contact and the same
scene/body/object/floor SDF gates. Source support failure can end a pair before
its full SDF pass; record that stage explicitly. A successful pair always has
every source frame checked. No model output participates in this construction.
Report unique LINGO/OMOMO sources, source families, target scenes, source texts,
object categories, durations and reuse counts alongside the raw candidate count.

Before sampling, select 24 grasped reverse episodes from the published table:
cycle object names alphabetically and prefer unused target scenes, LINGO source
IDs, then LINGO families; episode ID resolves ties. If fewer exist, execute all
and report the shortage. Publish this selection with the source manifest. All
other new candidates remain explicitly unevaluated by the pilot.

### Actual-history contract

Each episode starts from only the first ten native frames of its placed LINGO
source. HSIPrior R2 epoch222 generates the remaining predecessor, with frozen
500-step diffusion, CFG1 and posterior-coefficient scene guidance, original
source text and declared pelvis/static goal. The object stays at its supported
planned transform and appears in scene queries; its model channels use the
existing known-empty view. Each 16-sample window conditions on actual native
frames -4/-1 at stride3. Append its 42 new samples, retain the last partial window
only to the source's exact native frame count, and discard interpolation padding.
Progress and the local goal use this segment's actual frame budget and native
body reconstruction frame. Never reset a window to a source pose.

The predecessor guard checks finite output, actual goal error <=0.10 m, native
history reconstruction <=1e-4 m, complete-frame geometry, foot support and any
required seated support, and exact persistence of the supported object. A failed
predecessor blocks both successors with named reasons and null generated metrics.
Do not substitute a source terminal after a failure.

For each passing predecessor, construct the Kimodo prefix from its last ten
actual frames. Keep the prescribed ten-frame static OMOMO grasp target and the
existing 61-frame/50-step/empty-text/seed42/one-sample recipe, 240-step native fit
and 400-step contact correction. Append 51 new frames. Retain raw, adapted and
corrected motion, positional/rotational seams, foot sliding, contact traces and
all existing geometry/context/contact/object gates. A failed bridge blocks HOI.

For each passing acquisition, run frozen P15 HOIPrior plus its sealed Arm B
guidance from the actual body/object history, full original OMOMO text and
original object/pelvis goals. Use the original task's `episode_num` window budget:
each window appends 42 observed frames after the inherited context. The object
rotation reference, BPS conditioning and native contact/history channels must
describe this achieved entry; source `data_idx` supplies semantics/body identity,
never replacement poses. Save the raw native rollout and evaluate both original
goal completion and the complete-frame geometry/support/contact measures. This
pilot measures the frozen native expert execution path; the Phase2.34 offline
whole-motion editor is retained as an existing result, not replayed onto a changed
entry as though it were a matched result. This protocol difference is explicit
in the output and precludes a quality comparison to that benchmark.

Count the ten initial frames once, every HSI generated frame, all 51 appended
bridge frames and every HOI generated frame. Preserve segment boundaries,
conditioned-history errors, seam metrics and failed/blocked stages. Report stage
success numerators/denominators, longest completed prefix and whole-chain goal
and geometry success on all 24 registered episodes. Keep per-stage and combined
motion/video artifacts. No resampling, source replacement, extra window budget,
prompt tuning or new correction follows a failed gate in this pilot. Zero
successful chains is a complete negative execution diagnostic.

### Execution and gate

Extend the existing multitask/source/bridge components and Hydra dispatcher; one
config delta covers the source stage and execution stages. No new tool script,
hash machinery, smoke workload, expert/core edits, training or model selection.
Use GPU1 for source geometry, then GPU0–3 for four deterministic pilot lanes;
each lane's Kimodo subprocess uses its own GPU. Record contention, synchronized
generation/geometry times and peak allocation; leave existing jobs untouched.
Use the verified infbagel interpreter and the existing separate Kimodo environment.
Reuse sealed input provenance and the experiment start/finish/register lifecycle.
Archive exact resolved configs and clean committed code before reportable runs.

Verify full authority tests, registry and config resolution, unique membership,
unchanged original tasks, no source-terminal substitution, frame arithmetic,
native body/object roundtrips and saved artifact/video readability. Pilot failure
does not remove a candidate. No paired-method bootstrap applies to one execution
method; report bounded point estimates without population-generalization claims.
Deliver source coverage/failure analysis, all 24 execution records, review index,
compact result and phase summary. Freeze full-table execution until this pilot's
measured entry/continuation failures and costs have been reviewed.

Implementation verification before reportable sampling: 1,186 tests pass with
four historical skips (217.08 s); registry validation covers 444 records and the
expanded Hydra configuration resolves fully. Component checks exercise a real
SMPL-X body, a moving object/rotation/contact history, physical-to-native goal
conversion, partial-window counting and source-independent failure blocking.
The bridge adapter/correction is shared by source and actual-history execution.
The new native entry measures its first model windows in the reportable pilot;
no training/per-step optimization is changed, so a training micro-batch benchmark
does not apply.

### Native interpolation correction before completing the pilot

The first four lanes exposed a native context error: `utils.quaternion_slerp`
uses `q1*t + q2*(1-t)` in its small-angle linear branch, reversing the endpoints
while its spherical branch uses the correct time direction. Initial-context
joint discrepancies of approximately 0.7–2.7 mm exceed the 0.1 mm history gate.
All four runs are stopped and sealed failed; their 36 completed HSI windows,
15 completed episode records and all partial artifacts remain intact. No HOI
window completed in those runs. Body support failures remain separate measured
results (roughly 9–18 cm foot-marker gaps in the early completed cases).

Replace the reversed linear weights with `(1-t)*q1 + t*q2` in the shared native
interpolation utility and test both body and object endpoint interpolation.
This is an evaluator correction on this inference branch; preserve every
historical result and the frozen `core/` contract. No physical threshold,
candidate, text, diffusion budget, correction setting or goal changes.

Repeat the same 24 registered episodes under fresh r1 lane IDs. A completed
first HSI window from a failed lane may be consumed by reference: its denoising
preceded the defective interpolation and depends only on the unchanged ten-frame
source context. Publish an explicit first-window cache manifest before r1;
verify every actual body/object/contact input and progress value against the
current initial context. Recompute its native interpolation using the corrected
utility. All later windows sample anew from the corrected generated history;
there is no reuse of later-window outcomes or output-based candidate selection.
Record reused and newly sampled window counts and costs separately. The expanded
67-candidate source inventory uses direct source poses and remains unchanged.

Correction verification: 1,188 tests pass with four historical skips (208.18 s).
The previous surface-edit regression explicitly expected the old keyframe shift;
it now verifies preservation of the validated coarse poses. A direct audit of
all 15 reusable first windows finds a maximum corrected rotation-matrix error
of 5.96e-7 and maximum angular error of 6.71e-7 rad against the actual history.
Registry validation and the r1 resolved configuration pass. The old numerical
protocol remains documented in its immutable commits and artifacts.

### Kimodo output-reader correction and bounded resume

R1 lanes0–2 complete eighteen episodes. Lane3 retains three completed episodes
and one passing HSI predecessor with a completed Kimodo sample, then stops before
native adaptation: the NPZ reader attempts to index scalar `fps` as a batch.
Read only the four declared model-output arrays required by native adaptation;
use the same extraction for source and actual-history bridges. This changes no
motion generation or correction recipe. All four r1 manifests are sealed, with
lane3 retained as a failed operational run.

Publish an explicit resume manifest for the remaining three lane3 episode IDs.
The 21 completed episode records remain the authoritative results for their IDs.
For the interrupted episode, replay its completed r1 HSI denoising windows only
after exact actual-history/progress verification, and reuse its Kimodo prediction
only after verifying the native conditioned poses, translations and object
transforms. The interpolation utility and model conditions are unchanged from r1.
The remaining two episodes follow the same original pilot contract. Retain the
initial-window cache where applicable, report every reused window/sample and
new cost, and aggregate every one of the original24 IDs exactly once. Use a new
`p5-multitask-actual-r2lane3-s42-20260910` run. No result or candidate is replaced
because of its measured quality, and no sampling budget or threshold is changed.

Reader/resume verification: 1,189 tests pass with four historical skips
(198.52 s), 446 registry records validate and the r2 config resolves. The NPZ
regression includes scalar frame-rate metadata and selects the four declared
batched arrays. R1 lanes0–2 and lane3's three completed cases are sealed before
the reader correction is committed.

### Phase 5.5.2b.1 completion

The expanded source run publishes67 candidates (15 forward,52 grasped reverse),
retaining all17 prior pairs and all469 original task definitions. Coverage is
46LINGO sources,18LINGO test scene families,22original tasks and16target scenes.
All accepted LINGO actions remain walking. The fixed search retains1407 task/
direction rows and282 new pair attempts;423 reverse task placements fail object
support under the unchanged static-object contract.

All24 registered actual-history episodes now have unique retained results and
verified videos. HSI goal/geometry/history gates pass20/22/24 cases respectively;
only1passes foot support and the complete predecessor guard. That episode's
Kimodo acquisition bridge passes and HOIPrior generates four inherited-history
windows, reaching human/object goals within3.139/1.936cm. Its human/object scene
penetrations reach290.66/184.12mm and support also fails. Whole-chain goal
completion is1/24, full physical acceptance0/24; the other23 successors remain
explicitly blocked. The native pilot uses P15+ArmB, with the Phase2.34 offline
scene editor awaiting integration into this changed-history path.

The canonical results contain55 HSI windows (15 valid cached initial draws and
40new draws),1Kimodo sample,4HOI windows and2176 observed30Hz frames. The two
operational defects and all five failed manifests remain sealed; the21 complete
r1cases plus3r2cases cover every selected ID exactly once. Final verification is
1189passed,4historical skips; all24 source prefixes/window inputs and video frame
counts pass artifact verification. No training or paired-method bootstrap ran.

Read [Phase5.5.2b.1 summary](../phase_summaries/PHASE_5H_EXPANDED_HISTORY.md) and
`experiments/results/p5_multitask_expanded_history_s42_20260910.json` at the next
entry. Source expansion and the bounded execution diagnostic are complete.
Before5.5.2b.2, preregister actual HSI support and HOI scene-constraint integration;
the full67-table execution remains behind that unresolved quality entry gate.
