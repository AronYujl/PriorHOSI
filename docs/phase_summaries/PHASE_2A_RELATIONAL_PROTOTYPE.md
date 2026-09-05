# Phase 2.1 relational prototype handoff — 2026-09-06

The relational interface and registered window experiment completed. The
interface gate passed; the tested HSI dynamic-perception target failed to add
scene benefit over the same geometric objective. Phase 2's overall quality gate,
closed-loop composition and learned mixer training remain open.

## Scope and implementation

Development branch: `phase/02a-relational-prototype`; integration branch:
`phase/02-mixer`. Preregistration is in the dated Phase 2.1 section of
`docs/plan/PHASE_2_COMPOSITION.md` and the registry. The user approved continuing
from the completed HSI input diagnostic. Phase 2.2 is reserved for a later
`phase/02b-relational-rollout` session.

`KnownEmptyObjectView` generates a complete Gaussian forward trajectory from
clean zero with the canonical diffusion schedule and reads it at matching
reverse timesteps. This gives a defined temporal coupling for the HSI empty
object/contact modality. Its generator is independent, the first two frames
are zero, human motion remains shared, and object tokens are masked only at
the denoiser. Scene geometry uses the full world. The original fixed-epsilon
diagnostic remains available through its original interface.

`RelationalGeometry` decodes the HOI body and object into compatible physical
frames. It supports a common root-centred translation/yaw and increments to
all 21 non-root local rotations, with FK position reconstruction and exact
history/contact-channel restoration. The object uses the evaluator's world
prefix and BPS reference explicitly. Common motion preserves instantaneous
root-object and hand-object relations; articulated corrections are constrained
by a separate hand-anchor objective. The object-relative pose is fixed in this
prototype. Per-axis translations are bounded at 10 cm, angular components at
10 degrees. All 67 residual coordinates/frame have a differentiable path;
root and leg gradients at zero are finite and nonzero in component tests.

The named `RelationalPrototypeDiagnostic` runs four independent optimization
cells in one GPU batch for each source window. Every cell shares source
relation, floor, stance, endpoint and residual objectives. H adds tracking of
the conditional-minus-temporal-scene-masked HSI FK increment. G adds human
and object nearest-free-voxel objectives. All cells use 20 Adam steps at 0.05,
from zero residual, with fixed source masks and physical normalization scales.
This is an optimization-based mixer prototype; no neural mixer is trained.

Core and expert source/checkpoints were unchanged. The ordinary sampler keeps
its prediction arithmetic; the new HSI view is opt-in. The optimized states
were observed and saved, never fed into the G=0 carrier.

## Experiment and results

Run: `p2-mixer-relational-prototype-s42-20260905`, P15 online plus Arm B carrier,
R2 final EMA HSI, seed 42, HSI posterior guidance off. The same four scenes
selected from task start/goal metadata (bins 0,22,44,66), seven objects each,
provided 28 episodes, 124 windows and 372 states at reverse steps 10,1,0.
Four cells produced 1,488 optimized outputs and used 744 HSI forwards.

Values below are episode-first means on generated-history windows. They are
window proxies in cm, not native mesh penetration or rollout foot-sliding.

| Cell | Human-scene residual | Object-scene residual | Hand-anchor drift | Stance displacement |
|---|---:|---:|---:|---:|
| A00: shared source constraints | 0.34527 | 0.86793 | 0.61818 | 0.13731 |
| A10: add HSI target | 0.34550 | 0.87329 | 0.88084 | 0.13893 |
| A01: add geometry | 0.29507 | 0.71538 | 0.62976 | 0.13735 |
| A11: add both | 0.31527 | 0.77889 | 0.88613 | 0.13878 |

The primary A11-A01 comparison increases object-scene residual by 0.06351 cm
(episode 95% CI [0.01996,0.11765]; scene CI [0.03457,0.09245]) and hand-anchor
drift by 0.25637 cm (episode CI [0.19868,0.31352]; scene CI [0.21213,0.31220]).
Object occupied-point fraction also rises by 0.003267. Human-scene and stance
differences are inconclusive. Root/object endpoint deviations from HOI decrease
by 0.07014/0.07117 cm; those are reference deviations, not measured goal error.

Geometry alone (A01-A00) reduces object-scene residual by 0.15255 cm (episode
CI [-0.29148,-0.04426], scene CI [-0.20421,-0.10089]). Hand-anchor drift rises
slightly, by 0.01158 cm. The human-scene point estimate improves, but its scene
interval crosses zero. HSI alone has no resolved scene effect and increases
hand-anchor drift. All four-cell metrics, all three timesteps, initial-history
strata, factorial main effects and interactions remain in the full reports.

The relational interface is usable. The uniform HSI DP-displacement tracking
target is retained as a negative control. It should not be promoted into a
trained mixer or presumed beneficial in a closed loop. This outcome concerns
one bounded optimizer/target recipe on four scenes. It does not establish
that every HSI signal or learned composition mechanism is ineffective.

## Runtime verification and limits

All 372 current states, previous x0 states and HOI predictions are bitwise
equal to the corresponding saved G=0 states from the input diagnostic. All
1,488 optimized outputs preserve history and contact channels exactly, and
all cell gradients are finite. Shared-transform/frame/gradient component tests
pass. Four GPU process exits and both factorial-analysis exits are zero.

Four-cell optimization at a single observed state takes 0.805 seconds on
average (median 0.804, p95 0.828), with maximum recorded allocated memory
399.36 MiB including resident models. This timing covers the complete batched
20-step optimizer; it is not neural-mixer latency or production generation FPS.
The longest complete worker shard took 228.48 seconds.

Geometry observes 24 human joints and 128 fixed object vertices. Nearest-free
voxel residuals approximate scene feasibility; they do not certify full-mesh
collision freedom. Stance/contact sets come from the frozen HOI prediction;
contact labels remaining exact does not guarantee geometric contact. Bounded
corrections and the source reference do not prove motion naturalness. The
input trajectory has exact known-zero forward marginals/coupling, but that
alone does not identify a correct composed HOSI distribution.

All statistical reports use 10,000 paired replicates and seed 42 with one
resampling matrix across cells/metrics/contrasts. Episode intervals concern
these 28 tasks; the scene analysis has four units and limited generalization
scope. No closed-loop success, native sliding or learned-mixer claim is made.

## Verification, artifacts and integration

Preregistration commit: `c4c2ff3`; implementation/config/test commit: `1c09b38`.
The completion commit contains this handoff, result and registry record.
Integration uses the subphase's fast-forward into `phase/02-mixer`; tag:
`exp/p2a-relational-prototype-v1`, identifying the interface and negative
experiment together, not a Phase 2 quality success.

```text
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
"$INFBAGEL_PYTHON" code/test_infbagel_hosi.py --config-name config_sample_hosi_relational_prototype --cfg job --resolve
```

The final runtime-source authority suite passed 911 tests with 4 skips in
161.58 seconds. Its earlier two GPU-resume setup errors came from an
unexported interpreter variable and were corrected in the command. All new
component and Hydra/registry checks pass. The registered real-data experiment
provides runtime and compute/memory validation; no separate smoke was added.
No reportable GPU workload failed, restarted or overwrote an earlier result.

Worker source was published through Git. The four idle RTX 3090 GPUs ran
the committed object under the verified infbagel environment, archived
resolved configs and preflight, and worker-owned persistent sessions. The
74 MiB immutable return was recovered once and the checksum-only dry-run
reported no differences. Existing lifecycle provenance was reused.

- Compact result: `experiments/results/p2_mixer_relational_prototype_s42_20260905.json`.
- Complete authority return: `results/incoming/p2-mixer-relational-prototype-s42-20260905/`.
- Worker original: `/home/yujinlun/data/work/InfBaGel-mixer/results/experiments/p2-mixer-relational-prototype-s42-20260905/`.
- The return holds manifest/configs/preflight, raw and optimized states,
  every episode record, full factorial reports, runtime verification and completion.

## Exact next entry point

Read this handoff, the overview and latest Phase 2 plan sections. Reuse the
input process and relation geometry. Preserve A10/A11 as controls and review
how HSI information should enter the constraint-compatible correction before
registering Phase 2.2. A uniform FK-displacement target is the specific
negative here. Geometry-only A01 provides a necessary reference for any future
claim about HSI value, and A00 controls the shared source constraints.

A future closed-loop proposal must define where the correction enters the
posterior, preserve matched geometry/HOI guidance across cells, score native
quality and engagement, and distinguish prediction sensitivity from rollout
improvement. Joint training supervision and a source of useful HSI information
must be established before training a residual network. This session starts
neither Phase 2.2 nor the learned mixer.
