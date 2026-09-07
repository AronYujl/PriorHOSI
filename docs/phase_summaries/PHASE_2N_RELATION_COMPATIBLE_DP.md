# Phase 2.14a — relation-compatible DP local gate (2026-09-07)

**NO-GO. Relation projection protects contact, with substantial retained motion;
correct-scene DP still provides no additional scene-quality benefit over the
same editor without HSI. Stop the current direct DP-gradient route.**

Engineering and the registered local diagnostic are complete. Development
closed-loop rollout and full469 were neither configured as promoted candidates
nor executed. The overall Phase2 quality gate remains open.

## Scope and implementation

User authorized direct execution of the latest handoff, found at
`/data/yujinlun/report/PriorHOSI_Phase2_14_Relation_Compatible_DP_Codex_Plan.md`;
the supplied `papers/` path was absent. Before implementation, split the proposed
work into2.14a fixed-source mechanism/asset audit and conditional2.14b development
rollout. This session closes2.14a. Base053eb13; preregistrationd89d597;
implementation6588b71; completion is sealed at `exp/p2n-relation-compatible-dp-v1`.
Branch `phase/02n-relation-compatible-dp`, integrated into `phase/02-mixer`.

`RelationalObjective.contact_residual()` exposes the existing world-coordinate
hand22/23 anchor vectors, using source contact>0.95 and future frames2:.
`relation_projection.py` computes6x67 frame-local Jacobians through actual FK
and tanh; tests verify equality with a full-window Jacobian, including nonzero
states. Thin float64 SVD projects the unscaled DP gradient under the unchanged
Euclidean parameter metric. Rank rtol1e-6/atol1e-10, no damping; float32 decoding
and returned gradients. History columns and inactive constraint rows contribute
no freedom/constraint respectively. Source anchors/masks stay fixed; Jacobians
refresh at every current editor state. Only the DP term is projected.

The optional parameter-linear DP proxy is frozen during each Armijo search.
G1 identity and G2/G3 projection share that proxy; original HOI motion-space
reference, six explicit terms, proximal term, bounds and domain guard remain.
Production defaults retain the legacy view/motion-space proxy. The existing
HOSI evaluator gains saved-context development replay; no new experiment runner,
expert training, core changes, native metric changes or new source generation.

## Complete paired workload and exact checks

All24 tasks/68 fixed windows from the sealed Phase2.13 r1 artifacts are reused.
Calibration004/006/055 and verification023/037/036 retain their roles. P15 online
ArmB500/R2 EMA, lambda26, beta1, eight levels300/264/229/193/157/121/86/50, seed42
and draw offset1000003 are unchanged. G0 lambda0; G1 correct environment B identity;
G2 B projected; G3 projected+2m local-X dynamic observation, with true geometry
and static conditions preserved. Source rays use3 draws/level; each short editor
uses the same8-refresh budget and draw0 seed, with its own evolving candidate.

68/68 raw sources, reconstructed references, masks and saved lambda0 parameters
match bitwise. All4896 source-arm records pass static/base pairing. RNG, scene
storage, source anchors, actual history and contact channels remain unchanged.
Every accepted trial passes the existing domain guard.9792 HSI and7616 HOI teacher
forwards match the budget; zero HOI source-generation calls. All6 lanes exit0.
There are272 independent short edits and9792 physical probes. No operational or
numerical solver failures; every1mm/5mm probe attains its amplitude. Initial
unit-test assertions were relaxed only to measured float32 precision (3e-6 to
1e-5 for common-motion coordinates); SVD constants and scientific gates were
unchanged, and both failed/passing numerical-check logs are retained.

## Results: contact mechanism succeeds, scene-evidence gate fails

All values below are **voxel/FK window proxies in centimetres**, except physical
RMS in millimetres. They are not native HS/OS, foot sliding or task completion.
Draws/levels average within window, windows within task, then24 equally weighted
tasks. Paired bootstrap uses10000 seed42 replicates, nominal95% intervals;
six-scene sensitivity reports are also saved.

| Same8-refresh editor | HS residual | OS residual | Anchor drift | Stance correction increment | Physical RMS,mm |
|---|---:|---:|---:|---:|---:|
| G0 no DP | 3.003147 | 5.739348 | 0.075110 | 0.139119 | 6.4082 |
| G1 B identity | 3.003075 | 5.737512 | 0.136820 | 0.169660 | 6.7210 |
| G2 B projected | 3.003183 | 5.738043 | 0.076387 | 0.160599 | 6.6331 |
| G3 shifted projected | 3.002310 | 5.735751 | 0.075566 | 0.181332 | 6.9266 |

Primary G2−G0 HS delta is+0.000035cm, task CI[-0.003099,+0.003635]. G2−G3 is
+0.000872cm,[-0.002718,+0.004926]. Neither establishes improvement; the registered
5% threshold requires approximately0.150cm against either comparator. G2−G0 OS
is−0.001305cm,[-0.005762,+0.002918]. G2−G3 OS is+0.002291cm,
[-0.002758,+0.006916], failing the preregistered nonpositive point-delta protection.
Scene-level intervals also cross zero for HS/OS. These are small unresolved
changes, not evidence that correctly conditioned DP adds useful scene quality.

G2−G1 short contact drift is−0.060433cm,[-0.084040,−0.038335]: relation protection
works with the fair new-proxy control. G2 remains close to G0 contact
(+0.001277cm,[-0.000216,+0.003568]). However, G2−G0 stance correction increment
is+0.021480cm,[+0.007859,+0.036833], exceeding the frozen+0.01cm point margin.
G2 improves this proxy over G1 and G3; those favorable comparisons are retained.
Contact and endpoint protections otherwise pass.

At matched5mm DP-only displacement, anchor drift is0.924533(G1),0.003768(G2),
0.003013(G3): G2 reduces it by99.59%. G2−G1 HS/OS deltas are only−0.000750 and
−0.000564cm, with intervals crossing zero. G2−G3 HS is+0.000214cm,
[-0.010644,+0.010857]; OS−0.018674cm,[-0.040553,+0.002184]. Stance increment
rises from0.078671(G1) to0.117844(G2), illustrating the finite-motion foot tradeoff.
The1mm results are fully retained in the JSON, including the unfavorable G2−G1
HS point change. Teacher-ray results and8-step editors have separate tables.

Mean retained parameter norm is0.820624, median0.828818, range0.406898–0.999303.
All projected source queries exceed the1e-3 weak-direction cutoff; no motion is
silently replaced or amplified from a negligible remainder. Maximum normalized
constraint residual across4352 projections is2.65e-9; max absolute Jv is1.34e-11.
This establishes numerical relation compatibility, not a useful physical score.
G2 source HS descent predicts495 decreases,681 increases and456 zero-gradient
cases; contact/stance scalar derivatives near source zero are roundoff-limited.

At either probe scale, domain-admissible counts/1632 areG1=949,G2=949,G3=1007.
The common all-arm feasible set has809 queries; its descriptive pooled means
are separate from task-balanced primary statistics. All68 sources have grasp;
38 already contain outside points. Search exhaustion remains visible:
G0/G1/G2/G3 accept284/329/329/328 of544 refreshes each; their exhausted searches
are252/215/215/216, andG0 has8 zero-gradient stops. Acceptance is not a quality gate.

## Evaluation assets, visual evidence and limits

All6 development raw scene meshes exist in `datasets/LINGO/Scene_mesh`.
Five are watertight;006 is open. Mesh bounds/topology are archived. Matching
native signed Scene_sdf files and verified development human-surface reconstruction
are absent, so this phase supplies no native surface penetration validation.
The raw meshes are usable starting assets for a separate evaluator validation;
the gap must not be described as absence of all scene geometry.

The July InfBaGel baseline retains469 per-task metrics, with469/469 identities
matching current benchmark tasks. Its sealed evaluator06086f4 has no motion
serialization or save_motion_params branch despite resolved flag=true. No motion
files exist in its sealed run. Historical scale3/world transforms/IK/SMPL-X code
is available, but motion-based unified re-evaluation is blocked by missing
trajectories. Preserve July as historical and paper Hybrid as external unpaired;
loading that checkpoint into the repaired representation would change its input.

Saved actual-scale animations cover preselected006/suitcase and036/clothesstand,
and036/floorlamp (task013), selected by largest task-mean short G2−G0 HS increase.
Every task retains first-level/draw1mm/5mm ray states and all four final short
motions. Skeleton/object illustrations show the small edits at their real scale;
they cannot establish scene penetration, native foot sliding or closed-loop
success. The paired-interval figure makes the contact/scene distinction explicit.

## Verification, artifacts and next entry

Final implementation suite: `INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python`
then `"$INFBAGEL_PYTHON" -m pytest tests -q`: **966 passed,4 historical skips,
174.41s**, exit0. Registry validation:368 rows. Core is unchanged from053eb13.
GPU paired statistics match the NumPy seed42 index plan to1e-12. No runtime source
changed during the reportable run. Numerical unit failures preceded the run;
there are zero failed reportable attempts in this phase.

Sampling wall640.06s on GPUs0–5; summed diagnostic window time2047.03s, peak
allocation504060928 bytes (480.71MiB). Short editor mean seconds/window:
G0=1.846,G1=2.168,G2=2.910,G3=2.979. Projected short refreshes average0.077s
Jacobian and0.015s solve/audit. These are concurrent batch1 diagnostic costs,
not isolated production throughput. Full timing/teacher/solver traces are saved.

- Compact tracked result: `experiments/results/p2_mixer_relation_compatible_s42_20260907.json`.
- All runtime, resolved configs, tests, input references, manifest and outputs:
  `results/experiments/p2-mixer-relation-compatible-s42-20260907/`.
- Analysis: `analysis/{summary,paired,means,projection,feasibility,strata,audit}.json`,
  raw window/ray/short rows and task/scene pairing inputs.
- Reproduce on a new output directory using the existing
  `mixer.scene_calibration.summarize_relation_compatible`; exact original analysis
  command is archived in `analyze.sh`. `render.py` and `visualizations/` preserve
  figure production and selection. Existing output paths are immutable.

**Next entry:** read this summary, OVERVIEW and the latest Phase2 plan section.
Stop the present DP difference as a direct collision-edit gradient; no lambda,
view or step-budget search follows this NO-GO. A separate proposal may test
geometry-proposed edits followed by HSI conditional-denoising repair and relation
reconstruction, comparing full dynamic/static/no-HSI repair under one fixed
proposal. That mechanism is unvalidated and was not implemented or launched.
