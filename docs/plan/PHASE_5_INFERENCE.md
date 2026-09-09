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
