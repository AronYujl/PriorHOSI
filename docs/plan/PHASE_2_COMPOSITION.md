# Phase 2 — composing the two expert priors

Status: updated 2026-09-06. Phase 2.9 deliverable PASS; pilot quality PASS.
Armijo produces zero complete-objective increases in 744 corrections. All three
adjusted native primary comparisons and registered protections pass. A00 sliding
returns near reconstruction; A01 reduces OS mean by 13.20% against reconstruction,
while giving back significant HS/OS depth gains relative to the prior Adam recipe.
Retain reconstruction as the comparison anchor and Armijo as the passing pilot.

Experts remain **R2 final EMA + CG** and **P15 online + guidance Arm B**.
Full Phase2, useful HSI supervision, realism and learned training remain open.
Close only Phase 2.9. Review the scene objective's discontinuities and grid-boundary
semantics on recorded A01 solves before a separately approved next experiment.
[Phase2.9 handoff](../phase_summaries/PHASE_2I_ARMIJO.md).

## What is being composed, and why not by data mixing

The released InfBaGel trains one model on a mixture: synthesized pseudo-HOSI
(OMOMO motion plus voxelized free-space occupancy) plus real LINGO HSI. Its own
Table 1 and Table 6 report what that costs, and the numbers are the reason this
phase exists rather than a re-tuning of the ratio:

| trained on | Th | To | S% | FS | C% | Pbody | OS Pmean |
|---|--:|--:|--:|--:|--:|--:|--:|
| synth OMOMO only (1:0) | 4.75 | 8.14 | 83.16 | 0.13 | 78.18 | 3.96 | 16.62 |
| hybrid (1:0.5) | 4.37 | 7.94 | 81.45 | 0.15 | 76.96 | 5.05 | 12.45 |
| hybrid (1:1) | 4.80 | 9.44 | 69.72 | 0.18 | 76.48 | 4.01 | 16.00 |

Adding HSI data buys 25% of the object-scene penetration (16.62 → 12.45) and
pays 1.6% of the contact rate, 1.71 points of success and 27% of the body
penetration. Push to 1:1 and success collapses to 69.72 while the penetration
gain is given back. The paper states the trade-off directly: "too much HSI data
may compromise the model's ability to learn object manipulation from HOI data."

Data mixing has to pick one point on that curve for every episode, every frame
and every denoising step at once. A gated composition of two frozen experts does
not: the gate is a function, so it can spend scene-awareness where the scene is
the binding constraint and spend none where object manipulation is. The claim
Phase 2 has to establish is therefore not "beat 12.45" in isolation but
**dominate the trade-off frontier those three rows trace out**.

## The operator

    x0_h = G * x0_HSI + (1 - G) * x0_HOI

per denoising step, on one shared reverse chain. `code/mixer/composition.py`.

One anchor is exact and bitwise:

* `G == 0` → HOIPrior alone. Asserted against `HOIPriorSampler.p_sample_loop`
  itself at the production 500 steps (`tests/phase2/test_composed_sampler.py`,
  test C1), not argued from the shape of the code.

It short-circuits before any validation of the unused side. That is a
correctness property, not an optimization: `0 * nan == nan`, so the naive
arithmetic does **not** satisfy the anchor, and the tests assert it against a
sentinel tensor that raises on any access.

The second anchor was **withdrawn on 2026-08-30**; see the revision below.

### Per step, not per window

Averaging two independently sampled windows is wrong. A diffusion model's output
distribution is multimodal, and the mean of two independent samples from a
multimodal distribution is generally not a sample: averaging "walk left around
the table" with "walk right" yields a path through the table. priorMDM
(2303.01418) and MixerMDM (2504.01019) both compose per step, and the
preregistered operator is written on `x0_hat`, a per-step quantity. So there is
one chain, both experts see the same state at every step, and the shared
posterior (`priors/core/ddpm.py:posterior_sample`, which both experts already
consume) advances it from the blend.

### The gate is masked to the human channels

This is forced by measurement, not chosen. HSI is never supervised on channels
216:232 — `priors/hsi/data.py:253` calls `codec.encode` with no object arguments
and `core/window_codec.py:215` starts from `torch.zeros`, so every HSI training
target is exactly zero there. Blending against that zero is a pull toward the
origin of the normalized box:

* **object translation 216:219** — error is `G * |x| * half_range` per axis; the
  box half range is [3.0880, 1.0918, 3.0581] m, so up to **4.481 m** of L2
  object displacement per unit gate.
* **contact 228:232** — composed value is `contact_HOI * (1 - G)`, scaling every
  contact label down monotonically, directly into the metric the 15% contact
  budget is written against.
* **object rotation 219:228** — measured **invariant**: a uniform positive scale
  leaves the polar factor unchanged, so `project_to_so3((1-G) * R) == R` to
  9.99e-16 over 64x4 random cases. Masked for uniformity, not necessity.

So `human_gate_mask()` is 1 on 0:216 and 0 on 216:232. Object and contact always
come from HOI.

Measurements: `.claude/scratch/phase2-blend/blend_space.json`.

### Revision to the preregistration — 2026-08-30

On the user's instruction, two things changed together, because they are the same
defect seen from two sides.

**The mask is a hard requirement, not an option.** `channel_mask=None` is refused
by `compose_x0`, and `HOSIComposedSampler` refuses it at construction rather than
500 steps into the first window. A caller-supplied mask tensor is accepted only if
it is exactly zero on 216:232 — so an all-ones tensor, which is the same invalid row
`None` was, fails the same way. Any row produced with an open object channel is
invalid, and there is now no configuration that can produce one. The payload
records `composition.object_channels_from_hoi: true` as a claim, not as prose.

**The `G ≡ 1` = "HSIPrior alone" anchor is withdrawn.** It does not hold on
HOSI-test. HSIPrior receives no gradient on 216:232, so that row was never a scene
expert generating without object help; it was an object at the *centre* of the
normalized box, a zero 3×3 matrix with no polar factor at all, and every contact
label at 0. That is not a baseline for anything. The anchor now reads

    G ≡ 1  →  HSI on 0:216, HOI on 216:232, for channels only

which is exactly what the masked arithmetic gives, so **the operator is now
continuous at 1**: the value at 1 equals the limit from below, asserted at three
values of ε and at the level of the whole 500-step loop.

Why the old short-circuit was a defect and not a documented curiosity: it skipped
the mask, so a gate reaching exactly 1.0 anywhere hit a *different operator* than a
gate at 0.999. A learned gate is free to move through that value. The one gate value
that would genuinely degenerate — `G == 1`, where the object rotation's uniform scale
is 0 and the zero matrix has no polar factor — was the one value sitting outside the
mask's protection.

Consequence to state wherever it matters: **there is no HSI-alone row on this
benchmark.** `compose_x0` raises with a message naming this revision if asked for
one, rather than producing it.

### What each expert keeps

The two inference conventions are not interchangeable:

* **HOIPrior** — one model call per step, no classifier-free guidance at
  inference, then `prepare_clean_x0` to restore history and close object rotation
  on SO(3). Its preregistered P2 contact guidance is sampler state (sealed Arm B).
* **HSIPrior** — the released `models.infbagel.Sampler` architecture, so two
  model calls per step combined as `cond + w * (cond - uncond)`. Its "uncond"
  pass is not unconditional: `models/infbagel.py:1554` zeroes only the
  **temporal** scene embeddings and keeps the static scene, the text and the
  goals. Its CFG therefore amplifies dynamic scene perception specifically —
  precisely the term HOIPrior has no analogue for. Dropping it would silently
  de-scene the HSI expert.

HSIPrior needs no adapter: it *is* the released Sampler class. Only HOIPrior does,
because it is a separate implementation (`code/priors/hoi/`) with a different
`p_sample_loop` signature — the two agree on their first 19 positional
parameters and diverge at 20.

### One frozen contract, verified

Both experts speak the same 232-channel representation, and the mixer needs no
coordinate or normalization adapter. Checked rather than assumed:

* `data/train/norm.npy`, `data/test/norm.npy` and `data/dataset/norm.npy` are
  **byte-identical** (sha256 `6969c0c0…`), and `priors/core/contracts.py` pins
  that for both experts and forbids recomputing it.
* `code/priors/core/` is 7/7 byte-identical to both expert branches.
* `_global_rotations` is byte-identical between `priors/hoi/data.py` and
  `priors/hsi/data.py`; the world frame is decided by FK reproduction with ~6
  orders of magnitude separation.
* Both contracts declare the same 16-frame window, stride 3, 2 history frames and
  500 diffusion steps, and `core/diffusion_schedule.py` refuses any other count.

## Baselines

Decided by the user, 2026-08-29. Consolidated at `/data/yujinlun/report/HOSI_baseline.md`.

**Primary: InfBaGel (paper), Hybrid row.** It is the published number and the
right comparison for a method whose whole claim is a better way to combine HOI
and HSI knowledge than data mixing. Not locally reproduced, and two limits are
worth carrying explicitly rather than discovering later:

1. It cannot be paired. We have the paper's aggregate means, not its per-episode
   values, so a comparison against it is two numbers side by side with no
   confidence interval — weaker than the paired bootstrap used everywhere else in
   this project.
2. It was measured under the pre-repair representation. The released checkpoint
   is not re-runnable on current code (`contact_percent` 0.685 → 0.03 on the same
   checkpoint, localized to the rotation channel by `feet_height` surviving at
   3.66 vs 3.6484), so the paper's evaluator and ours are not the same evaluator.

The July released row (`p0-atomic-hosi-baseline-r2-s42-20260712`) tracks the
paper's 1:0 row to within 1-2% on nearly every metric, which is what makes the
paper rows credible as external calibration despite (2).

**Secondary and decision-relevant: G=0 P15+guide**
(`p2-hosi-hoi-alone-g0-p15-guided-armb-s42-20260829`, n=469). This is the only
row that is both reproducible here and paired, so it is what a composed row
should be significance-tested against. It is also an anchor of the operator, not
a neighbouring measurement.

One thing it is not: the model is scene-blind, but `code/astar.py:get_path` plans
on the scene occupancy and the evaluator feeds a point from that plan in as
`pelvis_goal` at every window. So the G=0 anchor is a scene-blind model under
**scene-aware waypoint supervision**, not a scene-blind row.

## Where the composed row has to improve

From the 469-episode paired comparison of G=0 against the July row, and from the
paper's own ablation:

| metric | G=0 | July rel. | paper hybrid | what the mixer owes |
|---|--:|--:|--:|---|
| C% | 69.15 | 78.05 | 76.96 | hold: 11% relative below July, inside the 15% budget |
| OS Pmean | 32.12 | 16.96 | 12.45 | **the target** |
| HS Pmean | 6.99 | 4.19 | 3.17 | the target |
| S% | 76.33 | 81.66 | 81.45 | recover |
| Th | 3.55 | 4.69 | 4.37 | already better |
| To | 7.59 | 8.13 | 7.94 | already better |

Penetration is where the scene expert is supposed to earn its place, and G=0 is
1.89x the July row on the object-penetration mean and 7.24x on its median.

### Why object identity is the first gate to try

The penetration mass concentrates by **object**, and the concentration is a
property of the episode rather than of the model:

* Spearman rho between G=0 and the July row is **+0.827** (object penetration)
  and **+0.825** (human), over all 469 paired episodes. The two models fail on
  the same episodes.
* Per-object means span **37x**: clothesstand 128.5, tripod 97.8, smalltable
  55.4, monitor 29.6, smallbox 19.4, floorlamp 10.0, suitcase 3.5.
* clothesstand and tripod are 29% of episodes and **65.7%** of the mass. A
  perfect fix on those two alone takes the mean 49.18 → 16.86.

Object identity is a task input known before the first denoising step, so the
gate can condition on it. That is the specific difference from the HSI
guidance-dose result, where the corresponding rank correlation was +0.056 and a
uniform intervention taxed 225 episodes that never needed it: here a per-object
dose is targetable, there it was not.

`code/mixer/gates.py` provides `ConstantGate`, `ScheduleGate`,
`ObjectConditionedGate` and `ChannelBlockGate` as reference gates. All are fixed
rules with no learned parameters; they bracket what a learned gate must beat.

The gate signature is `gate(step, current, hoi, hsi)`, keyword-only, asserted by
test. It sees the step index and both experts' `x0_hat` and nothing else — no
model, no weight, no internal feature. That is MixerMDM's modularity property
made mechanical: either expert can be swapped without retraining the gate.

## Reserved: the LLM state machine

Deferred by the user, with the interface reserved now. `compose_x0`,
`HOSIComposedSampler.__init__` and `HOSIComposedSampler.p_sample_loop` all accept
`state` and all raise `NotImplementedError` if anything is passed. Reserving the
name costs nothing; letting a caller believe state is honoured would not.

## 2026-08-30 — the HSI expert is wired in and running

The P17-OC arm finished on `phase/01c-hsi` at 589ac7f: epoch 222, sha256
`f64d956f88b8a81dddb160cb84fb5e9bdbe08f0606437a0e8b079cc92e8db5aa`. The composed
path now runs on real HOSI-test data with both checkpoints loaded
(`config_sample_hosi_composed.yaml`, `expert: composed`).

**One caveat on the checkpoint, stated because it is not visible from this
branch.** P17-OC's own Phase 1C verdict is not in. Its native LINGO evaluation
(guided and unguided, 8 shards) merged on 2026-08-30 but no aggregate or paired
bootstrap has been written, so whether the arm passed its own gate is unknown
here. The user's instruction was that the HSIPrior *framework* is settled and to
compose with the undistilled expert, which is what this does. The mixer code is
checkpoint-agnostic — `hsi_ckpt_path` is a config key — so if a different arm
becomes the official HSIPrior, composed rows need re-running but nothing needs
rewriting. No composed row should be cited as a main-table result until P17-OC's
own verdict exists.

### What real weights showed that the unit tests could not

* **`G == 0` still reproduces the sealed anchor, with the HSI expert loaded.** 7
  HOSI-test episodes, all 15 metrics identical to
  `p2-hosi-hoi-alone-g0-p15-guided-armb-s42-20260829`. This is what rules out the
  composed loop perturbing HOI's chain through a shared RNG stream, and it is
  strictly stronger than the stub-model C1 test.
* **The occupancy state was being passed wrong.** `_compute_occ_sample` takes both
  the noisy state and the previous step's `x_hat_0`, and they are not
  interchangeable: the second one places the three temporal occupancy queries. The
  composed loop was passing the noisy state for both, so the scene expert was
  querying the scene along a noise trajectory. It now carries the previous
  *composed* `x_hat_0` — the shared chain's own estimate, not either expert's
  private one.
* **The composed loop had no `@torch.no_grad()`.** Both single-expert loops do.
  Without it the HSI forward passes build a graph and the evaluator dies 500 steps
  later on `.cpu().numpy()`.

### The object-voxel decision, which is not free

The occupancy alphabet is 0 free / 1 occupied / 2 object. HSIPrior trained
LINGO-only, and under `lingo_only` every `object_points` tensor is the 999.0
sentinel (`datasets/infbagel_mix.py:471`) that falls out of bounds and clamps to
voxel 0 — so **the value 2 reached its scene ViT as at most one spurious corner
voxel per grid.** HOSI-test's object is real: measured on one episode, the three
temporal grids carry 225–239 voxels at 2.

`add_object_voxel: false` does *not* prevent this. The evaluator sets
`cfg.vis = True` unconditionally (`test_infbagel_hosi.py:442`) and
`_compute_occ_sample:706` then rebuilds the object from `obj_rest_verts`
regardless of the flag, so the key controls only the anchor grid `occ_list[0]`.
And the temporal grids are exactly the embeddings HSI's CFG amplifies
(`models/infbagel.py:1554` zeroes only those on the uncond pass).

Measured with the real weights on one window, remapping 2 → 1 moves HSI's
`x_hat_0` by **0.039–0.057 m mean and up to 0.158 m max** in joint position at
t ∈ {1, 100, 250, 400}, and by **exactly 0 at t = 499** — where those temporal
embeddings are zeroed on both passes, which independently confirms the whole
effect travels through the temporal channels. So `hsi_object_voxel_mode` is an
explicit knob: `occupied` keeps the input in distribution for a LINGO-trained
expert, `object` is the released arithmetic. Evidence:
`.claude/scratch/phase2-hsi-wiring/occ2_sensitivity.json`.

**Decided 2026-08-30 (user): `occupied` for every row; `object` is demoted to a
later ablation and must not be used for a reported row until that ablation runs.**
The grounds are the two facts above and nothing more — the input is in distribution
and the alternative is not, and the measured difference lives entirely in the
temporal channels where that shift is (exactly zero at t = 499, where those
embeddings are zeroed on both CFG passes). It is explicitly *not* a claim that
`occupied` generates better motion. Nothing measures that yet; the ablation is what
would.

### What a composed row costs

Skipping the HSI expert on steps whose gate is identically zero is worth **10x**,
not the 3x that counting network calls predicts: 58.69 s/episode → 5.89 s on the
same 7 episodes, still bitwise equal to the sealed row. So the HSI expert's
per-step cost here is dominated by `_compute_occ_sample`'s four 32,768-point scene
queries, not by its two forward passes. A `G > 0` row over the full 469 episodes
is therefore ≈7.8 h single-GPU, and the anchor row is already sealed so it does not
need re-running.

### Posterior identity, checked rather than assumed

The composed chain advances with `priors/core/ddpm.posterior_sample`. HOIPrior
already used exactly that; the HSI expert did not — `Sampler.p_sample` computes
its own mean from its own buffers. All four buffers are **bitwise identical**
(`betas`, `posterior_mean_coef1/2`, `posterior_log_variance`), and the operator
agrees bitwise at t ∈ {499, 250, 1, 0}. One wrinkle worth knowing: core always
adds `(0.5·log_var).exp()·noise`, and `posterior_variance[0]` is 0 but the *log*
clamps to log(1e-20), so that factor is 1e-10 rather than 0. It cancels only
because the caller passes a zero noise tensor at step 0 — which both the composed
loop and HOIPrior's own `sample()` do. Pass nonzero noise there and the two paths
diverge by 2.33e-10. Also, `posterior_mean_coef1[0]` is 0.9998340606689453, not 1,
because `1 − alpha_bar[0]` loses three digits in float32: the last reverse step is
*not* an identity on `x_hat_0`. That is shared with every InfBaGel row ever
produced, not introduced here. Evidence:
`.claude/scratch/phase2-hsi-wiring/posterior_identity.json`.

## Blocked on an HSI checkpoint

Unblocked as of 2026-08-30 by P17-OC, subject to the verdict caveat above. What
remains open is empirical rather than structural:

* Whether `ScheduleGate`'s `late` or `early` mode is right. The argument for
  `late` is that object manipulation is the harder constraint and should set the
  coarse trajectory; for `early`, that scene collision is decided by coarse
  structure and is expensive to fix afterwards. Neither is settled.
* Per-object gate values. The concentration says *where* to spend, not how much.
* Whether HSI should drive joint positions (0:84) but not rotations (84:216),
  since scene collision is a positional constraint. `ChannelBlockGate` exists to
  ask this — but see the body split below, which is the better-posed version.
* Where the body split's seam should sit: `BodyGroupGate`'s `torso` group defaults
  to HOI, putting the one unavoidable seam at the collars rather than inside the
  spine. Nothing measures which placement is better.
* `hsi_object_voxel_mode`: `occupied` is now the decision, `object` the ablation.

A first orientation on 7 episodes of one scene, G = 0.5 against G = 0 (a smoke,
**not** a result — 7 episodes of 469, one scene of 67, no uncertainty, and it
predates the object-voxel default): completion 57.1% → 71.4%, pelvis error
3.44 → 2.16, human penetration loss 7.45 → 3.97, scene-human penetration mean
1.00 → 0.26 and its frame ratio 0.503 → 0.269, object error flat at 8.08 → 8.10
exactly as the channel mask predicts, contact unchanged at 0.65 — and scene-object
penetration mean **worse**, 86.54 → 97.87. That last one is the direction to watch:
the human moves out of the scene while the object is still HOI's, so the pair can
be pulled apart. Whether any of it survives 469 episodes is unknown.

P17-OC also has a specific consequence for evaluation: it is the first checkpoint
in the project trained with the `occ_list[0]` X/Y permute applied, so any config
that evaluates it must set `occ_list_layout_repaired: true`. Every other
checkpoint predates the permute and must keep the default `false`.

## Open, and not blocked

* **Phase 1D / P15 closure.** `docs/plan/OVERVIEW.md` makes Phase 1D produce
  `PHASE_1D.md` and tag `exp/p1-priors-v1` before Phase 2. Neither exists, nor
  does a P15 phase summary, a P15 result JSON, or a P15 registry outcome row (the
  registry still says `status: preregistered`). The user has deferred closure
  until both experts are settled; it does not block mixer development, and the
  measurements are unaffected (the checkpoint sha256 matches both sealed P15 eval
  arms). What is missing is the citable record, not the evidence.
* Whether the object-point conditioning subset should be one fixed subset per
  object rather than the current per-window redraw.

## 2026-08-30 — the body split, per joint

`ChannelBlockGate` can only cut the representation at 84: positions against
rotations. That is the wrong axis. The two experts do not disagree along it — scene
collision is decided by where the **root and legs** go, object manipulation by where
the **arms and hands** go, and each of those spans both blocks. So the split has to
be per joint inside both, which was an implementation gap rather than a
configuration one. `mixer/body_groups.py` + `mixer.gates.BodyGroupGate`.

Groups, and the indices are taken from the repository's own code rather than assumed
from SMPL convention: `eval_metrics.py:107-119` reads ankles at 7/8 and feet at
10/11; `priors/hoi/losses.py:335-336` reads wrists at 20/21; `utils.py:300` gives
`SMPLX_JOINTS_28`, and `test_infbagel_hosi.py:379-380` plus `losses.py:335` read
hands at position slots 24/26 and 25/27.

| group | rotation joints | position joints | default |
|---|---|---|--:|
| `root` | 0 | 0 | HSI |
| `lower_body` | 1,2,4,5,7,8,10,11 | same | HSI |
| `torso` | 3,6,9,12,15 | + 22,23 (eyes) | HOI |
| `arms` | 13,14,16,17,18,19,20,21 | same | HOI |
| `hands` | — none — | 24,25,26,27 | HOI |

Both dicts are asserted at import to be a partition of 22 and 28 joints
respectively, and one weight drives a joint's positions *and* its rotations — a knee
whose position followed HSI while its rotation followed HOI is the defect the design
makes unrepresentable.

There are **no hand rotations**: the representation stops at 22 joints. "HOI drives
the hands" is a claim about four hand position channels and about the arm chain
carrying them, not about finger articulation this representation cannot express.

Two facts that shape the design and are easy to get wrong:

**Of the 84 position channels, only channels 0:3 reach the metrics.** The evaluator
takes `points_all[:, 0]` as the root translation
(`test_infbagel_hosi.py:885`), converts the 22 global rotations to locals through
`quat_ik_torch`, and runs SMPL-X; every geometric metric is computed on the vertices
and joints that come back. The other 81 act on the *rollout* instead — they are the
autoregressive history the next window is conditioned on, and what
`_compute_occ_sample` reads to place the temporal occupancy queries. Both matter, but
through different mechanisms, and a gate design that conflated them would be
reasoning about the wrong tensor. That is why `root` is its own group.

**A split cannot violate bone lengths, but it does create a one-joint seam.**
`quat_ik_torch` differences each global rotation against its parent, so a local
rotation is well defined however the globals were mixed and the rest template
supplies the lengths. What a split does create: if HSI owns Spine3's global frame and
HOI owns L_Collar's, the local collar rotation absorbs the whole disagreement between
the two experts' body headings as a shoulder twist. That seam is unavoidable in any
split; the group boundaries decide only where it lands. `torso` is a group of its own
so that placement is a knob rather than something folded silently into one side.

## 2026-08-30 — sharding the HOSI evaluation

A composed `G > 0` row costs 7.74 h on one GPU. `code/hosi_sharding.py` splits it;
`tools/launch_hosi_sharded.py` emits the launch plan. Four design points, each
decided by measurement, all in `.claude/scratch/phase2-sharding/`.

**The unit is the scene, not the episode.** `set_test_scene` reloads the scene mesh
and rebuilds the occupancy: 4.96 s, 67 scenes. An episode-level shard's episodes
scatter across nearly every scene, so each of four shards pays 60–63 switches; a
scene-level shard pays 17. Predicted slowest shard at four shards, switches
included: **1.97 h scene-level (3.94×) against 2.02 h episode-level (3.83×)**.

**Balance is by window count via a free proxy, and the exact plan is a net loss.**
The true per-episode window count is not in the data files. `test_item['episode_num']`
looks like it and is not: it matches the true count **164 of 461 times** (Spearman
0.853), and the evaluator never reads it — `seg_len = ceil(A* arc / 0.8) + 1`, with
`cond['is_loco']` true on all 469 episodes so that branch always fires. Computing the
true counts takes a **332 s** pre-pass. The straight-line xz chord from the JSON is
free and has **Spearman 0.971** against the true count (the A* arc is 1.13× the chord
on average, max 2.19×). At four shards the chord's packing costs **1.3 min** more wall
clock than the exact packing — so the 5.5 min pre-pass that would buy it back is a
loss. Measured totals: **469 episodes, 2086 windows**, min 2 max 11 per episode.

**No reseeding is needed, and that is measured.** Four facts, together:
`torch.initial_seed()` returns the seed rather than the live state, so drawing
numbers does not move it; `sample_calls` is per sampler instance and the evaluator
rebuilds `sampler_body` inside the scene loop, so it already resets per scene;
`__getitem__` consumes **no** global RNG at test time (all four of its draws are
gated off by `train=False`, `use_random_frame_bps=false`,
`use_object_keypoints=false`, or never fire — `np.random.randint` at
`datasets/infbagel.py:534` needs `not need_pi`, and `need_pi` is true on every
HOSI-test episode); and the three `randperm` sites already carry dedicated
generators (1c2d99b). Verified by hashing numpy, python-`random` and torch state
before and after every `__getitem__` over three scenes: **zero changes**. So a
scene-level shard reproduces the serial row bitwise, and **the sealed
`p2-hosi-hoi-alone-g0-…` anchor stays valid and pairable with no re-run.**

`hosi_per_episode_seeding` is implemented anyway, defaulting **off**, because it buys
a property scene sharding does not: an episode reproducible in *isolation*. Today
episode 3 of a scene can only be reproduced by running episodes 0–2 first — measured
offline, the same episode at scene position 0 versus 2 differs by up to **1.91** in
normalized units. Both halves are required: reseeding alone does *not* fix it
(`sample_calls` is the generator's other seed input) and resetting alone does; both
together reproduce the isolated episode exactly. Turning it on changes every number,
so a campaign that enables it must re-run its own anchor, and the merge refuses
shards that disagree on the flag.

**The merge anchors on canonical ordinals and raises on everything.** Every episode
record carries `canonical_ordinal` — its index in the full enumeration, never in its
shard. The merge refuses a missing shard, two shards claiming one index, a duplicated
or absent ordinal, a shard count that disagrees with the operator's own statement, a
different HOI *or* HSI checkpoint hash, a different gate/mask/voxel-mode, a different
seed, a different seeding regime, and a payload that predates sharding. Statistics
are recomputed over the union, never averaged from per-shard means — shards hold
unequal episode counts (119/112/119/119 at four shards), so averaging averages would
weight them wrongly. Wall-clock aggregates are nulled rather than deleted, so a
sharded payload diffs against a serial one as explicit nulls instead of missing keys.
Metrics are untouched: sharding invalidates the timing and nothing else.

**Operational, from the user (2026-08-30):** HSIPrior keeps iterating on the
authority host and mixer rows move to `infbagel-4gpu`. So the launcher arms the
return **before** the first shard starts — two detached tmux sessions per campaign,
one running the shards and writing `<name>.exitcode`, one blocking on all N and then
rsyncing — and proves the return path immediately with one tiny file rather than
discovering a key problem after the last shard. It transfers on failure too, since a
failed shard's log is the artifact most needed. It caps `OMP_NUM_THREADS=4`: uncapped,
the same protocol took 23 min against a capped 195 s, the cost being oversubscribed
BLAS in per-episode preprocessing, and capping is bitwise identical. Four concurrent
shards make that contention worse, not better. And every row records both
`checkpoint.sha256` and `hsi_checkpoint.sha256`, so a superseded HSIPrior means these
rows are **re-run, not rewritten** — the merge turns that into a mechanical check.

The launcher deliberately has no `--execute`: a row is a GPU workload needing the
user's explicit approval of one concrete experiment and a run id allocated through
`tools/experiment.py`.

## 2026-08-30 — P2-BG: the fixed body-group row, preregistered

**Approved by the user 2026-08-30**, run id
`p2-mixer-fixed-bodygroup-p15-p17oc-s42-20260830`, worker `node01`, 4 scene-level
shards on GPUs 0–3, seed 42. This section is written **before the result exists**.

### The arm

`BodyGroupGate` at its defaults: `root: 1.0`, `lower_body: 1.0` (HSIPrior),
`torso: 0.0`, `arms: 0.0`, `hands: 0.0` (HOIPrior), channels 216:232 from HOI at
every gate value. No `ScheduleGate` — the gate is constant over all 500 steps.
`mixer_hsi_object_voxel_mode: occupied`, `mixer_channel_mask: human`,
`hosi_per_episode_seeding: false`, `mixer_hsi_w: 1`.

HOIPrior is P15 + guidance Arm B, sha256 `ed8cf169…`. HSIPrior is P17-OC epoch 222,
sha256 `f64d956f…`. Both hashes are pinned in the config and the evaluator refuses a
mismatch; both are recorded in the manifest and in every shard payload.

### Criteria, as the user fixed them

| quantity | gate | G=0 anchor | note |
|---|--:|--:|---|
| `contact_percent` | ≥ 0.5878 | 0.69147 | 0.85× the anchor — the accepted 15% contact budget |
| `completion_rate` | ≥ 0.7433 | 0.76333 | 2 points below the anchor |
| `scene_human_penetration_s_mean` | ≤ ~6.288 | 6.98668 | ≈10% relative reduction |
| `scene_obj_penetration_s_mean` | reported separately | 32.11539 | **no threshold set** |

Primary comparison is **paired per episode against G=0**
(`p2-hosi-hoi-alone-g0-p15-guided-armb-s42-20260829`, n=469), which is the only row
on this benchmark that is both reproducible here and pairable. Pairing is by
`scene_name/object_name/test_idx` through `tools/hosi_per_sequence.py`; 15 metrics
carry sequence-level intervals and `completion_rate` does not — it is a proportion
over episodes, so it gets a proportion test, not a place in the same table.

**Immutable diagnostic.** The user's ruling: this row does not enter the main table
before P17-OC's own Phase 1C verdict exists. That verdict has since landed on
`phase/01c-hsi` as `f58d2b6` — **FAIL on both criteria, checkpoint not promoted**.
So P2-BG is a test of the composition mechanism, not a claim about a settled
HSIPrior, and its HSI half is an arm that failed its own gate.

### The risk the criteria do not cover

The threshold list constrains contact, completion and **human** penetration. The
metric the composed row most owes is **object** penetration: G=0 is 32.12 against
the July released row's 16.96 and the paper hybrid's 12.45, and that is the number
the phase's own "where the composed row has to improve" table marks as *the target*.
It has no threshold here, and it is the one the mechanism can be expected to move
the wrong way: 216:232 always comes from HOI, and HOI is scene-blind, so when HSI
moves the pelvis the object follows a root that no scene-aware expert chose.

Measured on the 7-episode smoke scene, same episodes, three arms:

| metric | G=0 | uniform G=0.5 | body-group | bg/G=0 |
|---|--:|--:|--:|--:|
| `contact_percent` | 0.6483 | 0.6474 | 0.5852 | 0.903 |
| `scene_human_penetration_s_mean` | 1.0020 | 0.2552 | 0.6553 | **0.654** |
| `scene_obj_penetration_s_mean` | 86.54 | 97.87 | 103.17 | **1.192** |
| `foot_sliding` | 0.1599 | 0.1057 | 0.0641 | 0.401 |
| `feet_height` | 4.0220 | 3.8731 | 3.2955 | 0.819 |
| `hand_pen_loss_omomo` | 0.4537 | 0.2414 | 0.2827 | 0.623 |
| `xy_points_err` | 3.4352 | 2.1592 | 3.1205 | 0.908 |

n=7 resolves nothing — HSI-side experience puts the sample needed for penetration
near 266 — and the smoke scene is harder than the benchmark (its G=0 object
penetration is 2.7× the full-set mean). Two things in it are still worth carrying
into the reading of the full row:

* Human penetration improves on **5 of 7** episodes, which is the mechanism working
  in the direction the arm claims.
* The object-penetration increase is **one episode**: clothesstand 527.5 → 656.8,
  +129.3, against 3 of 7 improving and 2 tied at exactly 0. clothesstand and tripod
  are 29% of episodes and 65.7% of the object-penetration mass, so the full row's
  verdict on this metric is mostly a verdict about those two objects.

The contact number needs the same care: 0.5852 looks like it fails the 0.5878 gate,
but 0.5878 is 0.85× the **full-set** anchor and this subset's anchor is 0.6483. The
subset-proportional floor is 0.551, which 0.5852 clears. Reading a full-set
threshold against a subset mean is the error to avoid here.

### What ran, operationally

Return armed before the first shard and verified with a probe file that reached the
authority at 23:43 — before any GPU work. `OMP_NUM_THREADS=MKL=OPENBLAS=4` on every
shard. The merge runs **inside** the return watcher, gated on all four exit codes
being 0, so the merged payload comes back in the same transfer; on a shard failure
the merge is skipped and the logs transfer anyway.

The worker needed provisioning first: it had no `hosi_test` at all. A second
immutable snapshot `InfBaGel-p2-hosi-v1` (test + object + hosi_test, 9.5 GB) was
pulled worker-initiated and verified by full-content `rsync --checksum` — zero
differing files — leaving the Phase 1B OMOMO-only snapshot untouched, since
`MULTI_SERVER_TRAINING.md`'s prohibition on `hosi_test` there is scoped to Phase 1B.

### Two defects this launch found, both mine

* `bb44621` put `@hydra.main` on `run_merge_shards` and left `main` undecorated, so
  **every** HOSI invocation died on `TypeError: main() missing 1 required positional
  argument`. The 856-test suite passed over it because no test goes through the CLI.
  Fixed in `0a1a26b`, with two assertions on the entry point.
* The `BodyGroupGate` override recipe in `sampler/hosi_composed.yaml` could not work:
  `gate` resolves to the scalar `${mixer_gate}`, so `_target_` needs `++`, and the
  `{root:1.0,…}` weights literal has its braces expanded by bash inside the
  launcher's tmux string. Fixed in `03e39cc`, with the recipe asserted against the
  real config tree.

Both were found by running the real evaluator on the worker. A one-scene preflight
through the actual CLI cost ~8 min and is now the thing that precedes a launch.

## 2026-08-31 — P2-BG result: FAIL, and the metric is the reason it reads as nothing

`p2-mixer-fixed-bodygroup-p15-p17oc-s42-20260830`, n=469, merged from 4 scene
shards, sealed and registered. HOI `ed8cf169` (P15 + guidance Arm B), HSI
`f64d956f` (P17-OC epoch 222), both hash-verified. Gate audit: `body_group` with the
five preregistered weights, mask `human`, `object_channels_from_hoi: true`, per-step
composition, timing nulled by design.

### The four criteria

| quantity | result | gate | verdict |
|---|--:|--:|---|
| `contact_percent` | 0.60395 | ≥ 0.5878 | PASS — but **significantly worse** |
| `completion_rate` | 0.75267 | ≥ 0.7433 | PASS |
| `scene_human_penetration_s_mean` | 6.97567 | ≤ 6.28801 | **FAIL** |
| `scene_obj_penetration_s_mean` | 32.79765 | (none) | 1.021×, null |

**The primary criterion fails as a null, not a shortfall.** Paired delta −0.011 with
CI [−0.782, +0.863] — a 0.16% drop against a 10% gate. The arm did not move the
metric it was built to move.

### Paired against G=0, 469 episodes, 10,000 replicates, seed 42

Significantly **better** (5): `scene_human_penetration_frame_ratio` −0.0719
(23.2% relative), `foot_sliding` −0.0459 (27.8%), `feet_height` −0.299,
`hand_pen_loss_omomo` −0.0254, `scene_obj_penetration_frame_ratio` −0.0125.

Significantly **worse** (2): `contact_percent` −0.0875 (12.7% relative),
`human_pen_ratio` +0.0147.

Null (9): `completed`, `end_obj_trans_err`, `hand_pen_ratio`,
`human_pen_loss_infbagel`, `scene_human_penetration_s_max`,
**`scene_human_penetration_s_mean`**, `scene_obj_penetration_s_max`,
`scene_obj_penetration_s_mean`, `xy_points_err`.

### Why the gated metric is a null while prevalence moves 23%

`penetration_s_mean` is `penetration_sum_per_frame.mean()` over **all** frames
(`test_infbagel_hosi.py:316`) and `frame_ratio` is the fraction of frames with any
penetrating vertex (`:320`). Both are then averaged over episodes, and the mass of
the first is extremely concentrated:

| | share of the G=0 `s_mean` total |
|---|--:|
| top 1% of episodes (5) | **52.9%** |
| top 5% (23) | 75.7% |
| top 10% (47) | **86.1%** |
| top 25% (117) | 95.8% |

The arm does not reach that tail:

| | `s_mean` G=0 → BG | ratio | `frame_ratio` ratio |
|---|--:|--:|--:|
| heavy decile (n=47, 86.1% of mass) | 60.008 → 61.122 | **1.019** | 0.929 |
| the other 422 episodes | 1.082 → 0.945 | **0.874** | 0.728 |

So penetration improves ~12.6% on 90% of the benchmark and the gated metric cannot
see it, because that metric is a mean over a distribution whose mass sits in 47
episodes. Per-episode `s_mean` is better on 266 and worse on 174; the four largest
regressions are +112.8, +73.5, +65.6 and +64.4, all on episodes already at 232–605.

Conditional depth per penetrating frame is itself a null (1.045×, CI
[−0.394, +2.871]), so this is not a prevalence-for-depth trade within episodes — it
is heterogeneity **between** them. Compare
[[hsi-prevalence-and-depth-disagree-on-deltas]]: the same two columns disagreeing,
for the same reason.

### The contact cost buys nothing

Spearman r(contact delta, penetration `frame_ratio` delta) = **+0.0177**. The
episodes that lost contact are not the episodes that gained on penetration. This is
the structure of the Phase 1C guidance-dose result, where +0.056 meant a uniform
intervention taxed 225 episodes that never needed it — here the correlation is
weaker still.

Both effects are object-conditioned, and **in different orders**:

| object | contact ratio | `s_mean` ratio |
|---|--:|--:|
| floorlamp | **0.705** | 1.025 |
| tripod | **0.639** | 1.053 |
| monitor | 0.864 | **1.096** |
| clothesstand | 0.868 | 0.804 |
| smalltable | 0.957 | 0.871 |
| smallbox | 0.963 | 0.926 |
| suitcase | 0.976 | **0.780** |

Contact loss is floorlamp and tripod (a 10× spread against suitcase). Penetration
regression is monitor and tripod. Penetration gain is suitcase and clothesstand.
14 episodes completed under G=0 and not here, carrying a larger contact loss
(−0.0945) than the 344 completed in both (−0.0589).

### What this says to do next

1. **Do not tune the body split against `s_mean`.** The metric is a mean whose mass
   is in 10% of episodes, and this arm demonstrably moves the other 90%. Tuning
   against it optimises a number that is structurally blind to what the gate does.
2. **The tail is the target, and it is probably not a body-split problem.** The
   large regressions land on episodes that were already catastrophic. The next
   diagnostic should ask what those 47 episodes share, not what weight the torso
   should take.
3. **`ObjectConditionedGate` already exists**, and the two effects being
   object-conditioned in different orders is the one intervention this data points
   at directly.
4. **One cheap separation first:** with r = +0.018 between the contact cost and the
   penetration gain, a gate keeping HSI on `lower_body` but returning `root` to HOI
   would test whether the contact loss is the root's doing. This arm confounds the
   two.

**Citation rule.** Quoting the `s_mean` null alone misreports this row. The finding
is a 23.2% significant reduction in penetration **prevalence** with no change in
mean depth, bought at a 12.7% contact cost that is uncorrelated with it.

### One operational fault, recorded not hidden

All four shard processes **exited 1**. The cause is a print block that formats a
timing key `invalidate_timing` had nulled; it runs after both `json.dump` calls
(`:1223`, `:1239`; crash at `:1267`), so all 469 episodes with ordinals 0..468
complete were already on disk and the measurement is unaffected. The watcher's
all-succeeded test correctly refused to merge, and the merge was run separately at
the same pinned commit — merge mode returns from `main` at `:509`, before the
defective block. Fixed in `27b4d5f` with two tests, one of them reading the source
so the guard itself is asserted. ~8 GPU-hours reported failure for work that had
succeeded; nothing was lost, and nothing was re-run.

## 2026-08-31 — the tail is a SCENE property, and it refutes my own next-step advice

Zero GPU, from the two sealed per-episode payloads and the scene SDF headers. The
P2-BG closure ended with four recommendations; this measurement kills the third and
sharpens the second.

### The 47 episodes holding 86.1% of `scene_human_penetration_s_mean`

| grouping | one-way R² on log1p(`s_mean`), 469 episodes |
|---|--:|
| **scene identity** (67 groups) | **0.271** |
| object identity (7 groups) | **0.019** |

Object identity explains **1.9%**. The P2-BG closure recommended
`ObjectConditionedGate` as "the one intervention this data points at directly" — on
`s_mean` that is wrong, and it is my error: I inferred it from a 7-row per-object
ratio table without ever asking how much variance object identity carries. It does
carry the *contact* effect; it does not carry the penetration mass.

Concentration is extreme even among scenes: **one scene (`0adb88db`) holds 33.58% of
the entire benchmark's `s_mean` total** in 3 of its 7 episodes, and 31 of 67 scenes
hold no tail episode at all. Object enrichment over the 10% base rate is at most
1.94× (smalltable 13/67), so the tail is not "clothesstand and tripod" — that pair is
the *object*-penetration story, not this one.

### The tail is a property of the episode, not of the model

| | |
|---|--:|
| \|G=0 tail ∩ P2-BG tail\| | **39 of 47** (83.0%) |
| Spearman ρ(G=0 `s_mean`, P2-BG `s_mean`) | **+0.896** |

A gate change moves the ranking almost not at all. And the tail is not made of failed
episodes — inside it, contact is 1.031× the rest, `xy_points_err` 0.928×,
`end_obj_trans_err` 0.955×, completion 1.004×. **These are normal episodes in
particular scenes.**

### The shape: 36× depth, 2.2× prevalence

| group | n | `s_mean` | `frame_ratio` | depth per penetrating frame |
|---|--:|--:|--:|--:|
| tail, G=0 | 47 | 60.008 | 0.6175 | **112.05** |
| rest, G=0 | 422 | 1.082 | 0.2757 | **3.14** |

Prevalence differs 2.2×; conditional depth differs **35.7×**. So the tail is not
penetrating more often, it is penetrating *enormously deeper* per frame — which is a
statement about how many vertices are how far inside something, not about how often
the body touches geometry.

### Where that depth comes from: `padding_mode='border'`

`compute_scene_sdf_penetration` (`test_infbagel_hosi.py:265-323`) normalises vertices
by `(v − centroid) / (extents.max()/2)` and samples the 256³ grid with
`padding_mode='border'`. Probing 12 scenes' grids directly:

| | 6 worst scenes | 6 cleanest |
|---|--:|--:|
| fraction of the volume negative | 90.1% | 94.6% |
| boundary shell negative | 99.5–100% | 99.5–100% |
| most negative boundary voxel | **−3.17 to −7.98 m** | −5.78 to −9.63 m |
| box max extent | 9.11 m | **12.94 m** |

Every scene's boundary shell is negative, at metre scale. Under
`padding_mode='border'`, **a vertex outside the box is charged the boundary value**,
so leaving the box costs metres per vertex. 10,475 SMPL-X vertices at ~1 cm each is
~105 units of summed depth, against the tail's measured 112.05 — the right order for
the mechanism, though this is a consistency check and not yet an attribution.

The worst scenes are also the **smallest** boxes (9.11 m vs 12.94 m max extent), which
is the direction this mechanism predicts: a smaller box is easier to walk out of.

### What this does and does not license

It does **not** yet prove the tail is an out-of-box artifact. Proving that needs the
per-frame vertex positions, and the evaluator saves no motion (`np.save` appears
nowhere in `test_infbagel_hosi.py` outside SDF *loading*), so it needs one re-run with
a vertex-range probe — GPU work, not authorized here, and not needed for the arm below.

It does establish three things that bind the next decisions:

1. **`ObjectConditionedGate` is not the lever for penetration mass.** R² 0.019.
2. **No gate can be expected to move `s_mean` much.** ρ = +0.896 across a gate change
   that altered 100 of 232 channels' provenance.
3. **A scene-conditioned quantity, or a bounded metric, is where the tail lives.**
   `frame_ratio` is bounded in [0,1] and is exactly the column that moved 23.2%.

## 2026-08-31 — P2-ROOT: returning the root to HOI, preregistered

**Approved by the user 2026-08-31.** Run id
`p2-mixer-rootsplit-p15-p17oc-s42-20260831`, worker `node01`, 4 scene-level shards on
GPUs 0–3, seed 42. Written **before the result exists**. Immutable diagnostic: does
not enter the main table before HSIPrior settles.

### The arm: a one-key diff from P2-BG

`BodyGroupGate` with `root: 0.0` instead of `1.0`. Everything else is byte-identical
to P2-BG:

| group | P2-BG | **P2-ROOT** |
|---|--:|--:|
| `root` | 1.0 (HSI) | **0.0 (HOI)** |
| `lower_body` | 1.0 (HSI) | 1.0 (HSI) |
| `torso` / `arms` / `hands` | 0.0 (HOI) | 0.0 (HOI) |

`mixer_hsi_object_voxel_mode: occupied`, `mixer_channel_mask: human`,
`hosi_per_episode_seeding: false`, `mixer_hsi_w: 1`, no `ScheduleGate`, 216:232 from
HOI. HOI `ed8cf169…` (P15 + guidance Arm B), HSI `f64d956f…` (P17-OC epoch 222); both
hashes pinned in the config, verified by the evaluator, recorded in the manifest.

### What it is expected to answer, and the half it cannot

**The contact half is structural and transfers past this HSI checkpoint.** Contact is
hand-to-object distance. With `root` at HOI, *every* joint on the chain
pelvis → Spine1 → Spine2 → Spine3 → Neck/Collar → Shoulder → Elbow → Wrist is HOI, and
216:232 is HOI, so hand-object registration is HOI's own — the pelvis no longer
displaces an HOI arm chain to where HSI put the body. Prediction, preregistered:
**`contact_percent` returns to ≈0.69** (G=0 is 0.69147; P2-BG lost 12.7% to 0.60395).
Residual coupling is second-order only: the shared `x_t` carries HSI leg content into
HOI's own forward pass, and the rollout history and the occupancy queries read all 28
position joints. If contact does *not* recover, pelvis displacement is not the
mechanism of the contact cost, and that is a finding about the operator rather than
about P17-OC.

**The penetration half is checkpoint-dependent AND newly confounded.** Two seams
move. P2-BG's split crossed the skeleton once (pelvis[HSI] → Spine1[HOI]); this arm
crosses it **three** times — pelvis[HOI] → Spine1[HOI] is now internal, but
pelvis[HOI] → L_Hip[HSI] and pelvis[HOI] → R_Hip[HSI] are new seams, and they sit
directly upstream of every foot and leg metric. `quat_ik_torch` differencing means the
hips absorb the whole disagreement between the two experts' body headings as hip
twist. So a disappearing penetration gain has **two** readings and the arm alone
cannot separate them.

### The discriminator that makes a null attributable

Preregistered before the result, on the leg-driven metrics `foot_sliding` and
`feet_height` (P2-BG: −27.8% and −0.299, both significant):

| if `foot_sliding`/`feet_height` … | reading |
|---|---|
| stay improved ≈ P2-BG | the **legs** carried the gain; the root carried the cost. Best case. |
| return to ≈ G=0 | the **root** carried the gain; a body split cannot have both. |
| go **worse than G=0** | the **hip seam is broken**; this arm's penetration column is uninterpretable and the split must move, not be re-weighted. |

### Criteria

P2-BG's four gates, carried forward unchanged so the two arms are read on one ruler:
`contact_percent` ≥ 0.5878, `completion_rate` ≥ 0.7433,
`scene_human_penetration_s_mean` ≤ ~6.288 (≈10% below the 6.98668 anchor),
`scene_obj_penetration_s_mean` reported separately with no threshold.

Two paired comparisons, both n=469, 10,000 replicates, seed 42, pairing by
`scene_name/object_name/test_idx`: **vs G=0** (the primary, as for P2-BG) and **vs
P2-BG** (the decomposition — a one-key contrast whose delta is attributable to the
root alone).

Given the tail measurement above, the `s_mean` gate is expected to be a null again,
and `frame_ratio` is the column carrying the mechanism. That expectation is recorded
here so it cannot be claimed as a prediction after the fact — and it does not license
moving the gate: the four thresholds stand as the user set them.

### Standing caveat

P17-OC failed its own Phase 1C gate (`f58d2b6`, FAIL on both criteria, not promoted).
Everything this row says about *penetration magnitude* is a statement about a
non-promoted checkpoint and must be re-run when HSIPrior settles. What survives a
checkpoint change is the contact-recovery structure and the seam reading.

## 2026-08-31 — the tail, continued: border padding is dead, and length is the second cause

Same zero-GPU sources as the section above, carried three steps further. **One claim in
that section is now refuted by my own follow-up measurement and is corrected here.**

### Correction: the border-padding mechanism does not hold

The section above called the boundary-shell arithmetic "the right order for the
mechanism" — 10,475 vertices × ~1 cm ≈ 105 units against the tail's 112.05. That was a
magnitude coincidence and I never tested it. Testing it kills it.

For each of the 67 scenes I located the floor by the lowest voxel on a 9-column grid
whose value crosses into the positive interior, and computed how far a foot may sink
before it leaves the grid entirely:

| | rank correlation with the scene's mean human penetration | p |
|---|--:|--:|
| margin from floor to grid's lower face | **−0.031** | 0.80 |
| most negative boundary voxel (cost of an exit) | +0.007 | 0.96 |
| box max extent (a smaller box is easier to leave) | −0.035 | 0.78 |
| vertical extent | +0.028 | 0.82 |
| normalisation divisor | −0.035 | 0.78 |
| box anisotropy | −0.098 | 0.43 |

**Six nulls.** And the reason is arithmetic I should have done first: because the
evaluator normalises all three axes by `extents.max()/2`, the grid covers a *cube* of
side `extents.max()`, while the room occupies only its own bbox. The measured margin
between the floor and the grid's lower face is **2.0–6.6 m**, not the 12 cm I inferred
from the bbox. Feet do not sink metres. Nothing exits the box downward.

So the tail's 112 units of depth per penetrating frame is **real geometry**: the model
drives the body into something that is actually there. The earlier section's three
binding conclusions are unaffected — they rest on the R² and ρ measurements, not on this
mechanism.

### The second cause: sequence length, independent of scene

`penetration_counts`, the benchmark builder's own field, is **0 for all 469 episodes** —
every episode was constructed collision-free, so the tail is not a specification defect.
Of the episode-specification scalars, the predictor is travel distance (ρ +0.312,
p 4.9e-12) — and travel is *length in disguise*: ρ(frames, travel) = **+0.971**, and at
fixed frame count travel stops predicting (mean within-decile ρ = **−0.097**).

Sequence length itself, from the 469 `frames: N` lines in the shard logs:

| | ρ with `s_mean` | p |
|---|--:|--:|
| frames (= windows, since `⌈(N−2)/14⌉`) | **+0.353** | 3.5e-15 |

| frame-count quartile | n | median `s_mean` | tail members |
|---|--:|--:|--:|
| Q1 90–132 | 118 | **0.033** | 11 |
| Q2 132–174 | 117 | 0.095 | 8 |
| Q3 174–258 | 117 | 0.537 | 8 |
| Q4 258–468 | 117 | **1.707** | 20 |

The medians are monotone across a **52× span**. (The quartile *means* are not monotone —
Q1 is 4.412 against Q2's 1.914 — because 11 tail episodes sit in Q1: a short episode in
a bad scene is still catastrophic. Read the medians for the length effect and the scene
R² for the other.)

Scene and length are **independent**, which is the useful part:

| | R² on log1p(`s_mean`), n=469 |
|---|--:|
| scene identity alone (67 groups) | 0.271 |
| scene identity **after** removing a cubic in log frames | **0.274** |
| length alone (cubic in log frames) | 0.107 |
| length **after** removing scene means | **0.091** |

Scene identity loses nothing to length. And length holds *inside* a scene, where
geometry is fixed: mean within-scene ρ = **+0.298**, positive in **52 of 67** scenes
(two-sided sign test **p = 6.5e-06**). Between scenes, ρ(mean frames, mean log1p
`s_mean`) = +0.393, p = 1.0e-03.

### What this means for the gate

Length predicting penetration at fixed geometry is autoregressive **drift**: the
evaluator generates 16-frame windows with 2 frames of history, each window conditioned
on the previous window's output, and error accumulates across that chain.

Be precise about what this does and does not say about a gate. The gate acts at every
denoising step of every window, so a better gate does slow the accumulation — that is
exactly what P2-BG's 23.2% prevalence reduction looks like. What a per-step gate has no
mechanism for is *correcting* error already accumulated: it sees the two experts' x̂₀ and
the step index, and nothing that tells it the rollout has drifted into a wall.

That points at two levers, neither of which is a body-split weight:

1. **A window-boundary correction** — something with access to the accumulated state,
   not just the per-step blend. This is where a scene-aware term can act on drift.
2. **A length-stratified reading of every future row.** Q4 carries 20 of the 47 tail
   episodes; a row evaluated only on short episodes will look far better than it is.

And it explains the P2-BG null more completely than the mass-concentration argument
alone did: the gate improved 90% of the benchmark by 12.6% while the two things that
actually produce the tail — which scene it is, and how many windows the rollout runs —
are both invariant to the gate.

## 2026-08-31 — the gate's benefit decays with rollout length; its cost does not

Third zero-GPU section, and the one that changes how a mixer row should be read. Same
paired protocol as every other row — pair by `scene/object/test_idx`, 10,000 replicates,
seed 42, 2.5/97.5 percentiles — but computed **within frame-count quartiles** instead of
over the whole benchmark. P2-BG against G=0, n=469, 117–118 per stratum.

| metric | Q1 90–132 | Q2 132–174 | Q3 174–258 | Q4 258–468 | all |
|---|--:|--:|--:|--:|--:|
| `frame_ratio` | −23.7% | −29.5% | −24.7% | **−15.7%** | −23.2% |
| `foot_sliding` | −35.2% | −36.0% | −31.7% | **−0.3% (null)** | −27.8% |
| `feet_height` | −9.8% | −7.5% | −8.2% | **−4.5% (null)** | −7.5% |
| `contact_percent` | −8.1% | −16.4% | −10.2% | **−16.1%** | −12.7% |
| `s_mean` | null | null | null | null | null |

Every entry except the nulls is significant at the 95% level.

**The three geometric gains decay with length and two of them die in Q4. The contact
cost does not decay — Q4 pays 16.1%, the joint-worst of the four.** On the longest
quartile the arm is nearly all cost: it gives up 16% of contact and buys a 15.7%
prevalence reduction, with foot sliding and foot height both indistinguishable from
zero (`foot_sliding` point estimate collapses from −0.0627 in Q1 to −0.0004, CI
[−0.0395, +0.0403]).

And Q4 is where the benchmark's penetration lives: **20 of the 47 tail episodes**, and a
median `s_mean` of 1.707 against Q1's 0.033.

### One honest confound, and why the reading survives it

Q4's G=0 `foot_sliding` baseline is 0.1271 against ~0.178 in Q1–Q3, so long episodes
slide less to begin with and there is less to win. That weakens the `foot_sliding` row
specifically. It does not explain the pattern: `feet_height` baselines are flat across
quartiles (3.99 / 4.02 / 3.93 / 3.99) and its gain still halves, and `frame_ratio`
baselines *rise* with length (0.285 / 0.287 / 0.338 / 0.331) — more to win in Q4, and
the arm wins less. The Q4 confidence intervals are also wide enough that "the effect is
zero in Q4" is not established; what is established is that the point estimates fall
monotonically and the contact cost does not.

### Why this is the useful form of the P2-BG null

The closure explained the `s_mean` null by mass concentration: 47 episodes hold 86.1%,
the arm moves the other 422. That is true, and this is the mechanism underneath it. The
gate acts per denoising step within a window. It makes each window better, which shows
up as a large gain on short rollouts. It has no signal for error the rollout has already
accumulated, so the gain erodes as windows compound — and the episodes with the most
windows are the ones holding the mass.

Two consequences for every future mixer row:

1. **Report length-stratified, always.** An arm evaluated on short episodes overstates
   itself by roughly 2× on prevalence and unboundedly on foot metrics.
2. **The tail needs a window-boundary mechanism, not a better per-step blend.** No
   choice of body-group weights changes the fact that the gate cannot see accumulated
   drift. This is the concrete form of what
   `docs/HSIPRIOR_DESIGN_PRIORS.md`-style negatives are for: a whole family of arms
   (re-weighting the split, per-object doses, schedules) shares one ceiling.

Recorded before P2-ROOT's result exists, so its own stratified table can be read against
this one rather than compared after the fact.

## 2026-08-31 — correction: the stratification above split ties, and the numbers move

The table in the section immediately above used **rank-based** frame-count quartiles. That
is unstable on this benchmark and I should have checked before committing it.

`test_infbagel_hosi.py:960` sets frames as `seg_len * (16 − 2) * 3 + 6 = 42·seg_len + 6`
with `seg_len` in 2..11, so there are only **ten distinct frame counts**: 90, 132, 174,
216, 258, 300, 342, 384, 426, 468. Every quartile boundary lands exactly on a heavily
populated one — p25 = 132 with **109** episodes at exactly 132, p50 = 174 with 85,
p75 = 258 with 51 — so a rank split cuts a tie group arbitrarily, and *which* episodes fall
either side depends on the sort algorithm. Two runs of my own script, one with `argsort`'s
default quicksort and one with `kind='stable'`, disagreed on the quartile medians and on
the tail-member counts (11/8/8/20 against 10/10/9/18).

Redone with strata defined by **value**, so episodes sharing a frame count are never split.
Sizes are unbalanced and that is the honest cost of not cutting ties.

| metric | S1 90/132 n=182 | S2 174/216 n=159 | S3 258/300 n=89 | S4 342+ n=39 | all n=469 |
|---|--:|--:|--:|--:|--:|
| `scene_human_pen frame_ratio` | −24.1% | −27.2% | −21.2% | **−10.1%** | −23.2% |
| `contact_percent` | −12.0% | −11.5% | −14.6% | **−16.3%** | −12.7% |
| `feet_height` | −7.6% | −9.3% | −5.5% (null) | −4.0% (null) | −7.5% |
| `foot_sliding` | −32.0% | −35.2% | −25.4% (null) | **+64.1% (null)** | −27.8% |
| `scene_human_pen s_mean` | null | null | null | null | null |
| `scene_obj_pen s_mean` | null | **+15.5% SIG worse** | null | −21.4% (null) | null |
| `completed` | null | null | null | null | null |

`frame_ratio` and `contact_percent` are significant in every stratum.

### What survives, and what changes

**Survives, strengthened.** The geometric gain decays with length while the contact cost
does not — and the cost is now *monotonically rising*: −12.0 / −11.5 / −14.6 / **−16.3**%.
On the longest episodes the arm pays its largest contact bill for its smallest penetration
return (−10.1% prevalence against −24.1% on the shortest). The tail-members column shows
why this matters: S4 is 39 episodes but holds **12 of the 47** tail episodes, a 3.7×
enrichment.

**Changes materially, three ways.**

1. **`foot_sliding` on the longest episodes is +64.1%, not −0.3%.** Null (CI [−0.016,
   +0.142], n=39), but the point estimate *flipped sign*. The S4 G=0 baseline is 0.0898
   against ~0.18 elsewhere, so long episodes barely slide to begin with and the arm makes
   them slide more. My earlier confound note was too gentle: this is not "less headroom",
   it is a sign flip on a small stratum.
2. **`scene_obj_penetration_s_mean` is +15.5% significantly WORSE on S2** (CI [+0.32,
   +9.96], n=159) — the only significant object-penetration regression anywhere, and the
   overall +2.1% null hides it entirely. This is the metric the phase marks as the target
   and the one the operator was always expected to threaten, since 216:232 is scene-blind
   HOI riding a pelvis HSI chose. S4 goes the other way at −21.4% (null, n=39).
3. **The `s_mean` nulls are nulls in every stratum**, including S4. Length does not rescue
   that metric; it never was going to.

### The per-value trend, which needs no boundary at all

G=0, 469 episodes, all ten frame counts:

| frames | 90 | 132 | 174 | 216 | 258 | 300 | 342 | 384 | 426 | 468 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| n | 73 | 109 | 85 | 74 | 51 | 38 | 26 | 8 | 4 | 1 |
| median `s_mean` | 0.034 | 0.039 | 0.341 | 0.478 | 0.503 | 1.652 | 3.464 | 2.957 | 3.434 | 9.880 |
| tail episodes | 7 | 8 | 6 | 6 | 5 | 3 | 7 | 3 | 1 | 1 |

The median rises ~100× from the shortest to the longest bucket, monotonically except for
the last three buckets which hold 13 episodes between them. **This is the length effect in
the form that has no analyst degrees of freedom**, and it is the version to cite.

### Standing rule, restated correctly

Stratify by frame-count **value**, never by rank. Four strata {90,132} / {174,216} /
{258,300} / {342+}, or the ten-value trend where n permits. Numbers and the per-episode
frame counts are sealed in
`experiments/results/p2_hosi_penetration_tail_diagnostic_s42_20260831.json`, since the
evaluator does not persist frame count and the shard logs it was parsed from live under
`results/incoming/`.

One further correction to the section above: it gives the floor-to-grid margin range as
"2.0–6.6 m" from the 12-scene probe. Over all 67 scenes the range is **0.37–6.57 m** — one
scene (`0aa05d5a`) has a 0.37 m margin, and it is among the *cleanest* at `s_mean` 0.28,
which is additional evidence against the border mechanism rather than for it.

### Amendment to the P2-ROOT preregistration — 2026-08-31, before the result exists

A third reading for a vanished penetration gain, which the preregistration above names
only two of. Found by reading the code while the shards ran, so it is recorded here
rather than after the fact.

**The occupancy window is pelvis-centred, and P2-ROOT hands the centring to a
scene-blind expert.** `models/infbagel.py:641-650`: HSIPrior's scene perception is a
32³ ego-crop, `mesh_grid: [-0.6, 0.6, 0.1, 1.2, -0.6, 0.6]` — **±0.6 m horizontally**
around a query frame whose translation is set from `x[:, :, :84]` at the pelvis joint,
with y zeroed (`mat_for_query[:, 1, 3] = 0`). So the expert only ever sees geometry
within 0.6 m of wherever the chain currently believes the pelvis is.

Under P2-BG that pelvis was HSI's own, scene-aware choice. Under P2-ROOT channels 0:3
come from HOIPrior, which has no scene input at all. If the two experts disagree about
the pelvis by 0.3 m, HSI's scene window shifts by half its radius, and the obstacle it
is being asked to avoid may not be inside the crop it is shown.

So the discriminator table has a third row:

| if `foot_sliding`/`feet_height` … | reading |
|---|---|
| stay improved ≈ P2-BG | the legs carried the gain |
| return to ≈ G=0 | the root carried the gain — **either** because pelvis placement *is* the mechanism, **or** because a scene-blind pelvis mis-centres HSI's only view of the scene. These two are not separable by this arm. |
| worse than G=0 | the hip seam is broken; the column is uninterpretable |

This does not change the arm, the criteria, or the run. It changes what a null in row 2
licenses: not "the root carries scene compliance" but "the root carries scene compliance
*and/or* the right to aim the scene sensor," and separating those needs a further cell
(root at HOI for the *output* channels while HSI's query frame keeps its own pelvis —
which the current sampler cannot express, since the query is read from the composed
`x0`). Related: the conditioning-box adequacy result was measured on HSI's own benchmark
and speaks to box *size*, not to centring by a foreign expert.

### Correction to the stratified table — the S2 object-penetration "regression" is four episodes

The section above calls `scene_obj_penetration_s_mean` **+15.5% significantly worse on
S2** "the only significant object-penetration regression anywhere." The interval is real
— CI [+0.32, +9.96] excludes zero — and the claim built on it is wrong. Checked properly:

| | S2, n=159 |
|---|--:|
| total delta over the stratum | **+740.0** |
| carried by the single worst episode | +256.6 = **34.7%** |
| carried by the top 2 | +494.3 = **66.8%** |
| carried by the top 5 | +768.9 = **103.9%** (the rest net negative) |
| episodes worse / better / **exactly tied** | 50 / 55 / **54** |
| two-sided sign test on the 105 non-tied | **p = 0.696** |
| median delta | **+0.0000** |
| trimmed mean (10% each end) | +0.066 |

The four largest regressions are **all clothesstand** (+256.6, +237.7, +129.3, +112.0),
and clothesstand's own median delta is exactly 0.000 with 12 of 26 worse. So this is a
mean shifted by four episodes of one object, not an effect on the stratum.

Two things make the metric especially unsuited to a mean-difference interval here: **54 of
159 episodes have exactly zero delta** — the object never penetrates at all in a third of
the stratum, so the distribution is zero-inflated — and the non-zero part is the same
heavy tail that made `s_mean` unreadable for human penetration.

**This is the P2-BG smoke failure mode reappearing inside my own stratified table**, three
hours after I wrote the memory about it. A stratum is a smaller sample, and slicing a
heavy-tailed zero-inflated metric four ways manufactures exactly the artefact that
sampling 7 episodes did. The `+15.5% SIG worse` cell should be read as **null**, and the
S4 `−21.4%` cell (n=39, one episode carrying −13.8% of a delta of the opposite sign)
carries no information either.

**Rule this adds, and it applies to P2-ROOT's table when it lands:** a stratified cell on
a heavy-tailed metric needs the sign test and the median beside the mean interval, and the
top-episode share of the stratum delta. Where they disagree, the mean interval loses. The
cells that survive this test in the stratified table are `frame_ratio` and
`contact_percent` — both bounded in [0,1], both significant in all four strata, and
neither zero-inflated. Those are the two columns the length finding actually rests on, and
they are unaffected.

### Refinement to the robustness rule — the median is the wrong third column

The rule two sections above says a stratified cell needs "the sign test and the median
beside the mean interval." The median half is wrong on this benchmark, and the reason is
worth recording because it changes which column to read.

Ties are pervasive. P2-BG against G=0, all 469 episodes:

| metric | exactly tied | better | worse | median |
|---|--:|--:|--:|--:|
| `completed` | **446** | 14 | 9 | 0 |
| `hand_pen_ratio` | 314 | 67 | 88 | 0 |
| `human_pen_ratio` | 307 | 60 | 102 | 0 |
| `feet_height` | **232** | 172 | 65 | 0 |
| `scene_obj_penetration_frame_ratio` | 189 | 165 | 115 | 0 |
| `scene_obj_penetration_s_max` | 167 | 147 | 155 | 0 |
| `scene_obj_penetration_s_mean` | 151 | 172 | 146 | 0 |
| `scene_human_penetration_s_max` | 90 | 233 | 146 | 0 |
| `scene_human_penetration_frame_ratio` | 82 | 269 | 118 | −0.0232 |
| `scene_human_penetration_s_mean` | 29 | 266 | 174 | −0.0050 |
| `contact_percent` | 24 | 318 | 127 | −0.0648 |
| `foot_sliding` / `xy_points_err` / `hand_pen_loss_omomo` / `end_obj_trans_err` | 0 | — | — | ≠0 |

When half the episodes are exactly tied the median is 0 **mechanically**, whatever the
effect is. `feet_height` is the clean case: 232 ties, and among the 237 episodes that move
at all it is 172 better against 65 worse — a sign test at p ≈ 1e-12. Its −7.5% is broad
and real, and its zero median says nothing against it.

So the third column is the **sign test on the non-tied episodes**, reported with the tie
count, plus the **top-episode share of the cell's delta**. Not the median.

Re-checked against the two calls this rule was written to make, and both stand:

* **The withdrawn S2 object cell is still withdrawn.** 54 tied, 50 worse, **55 better**,
  sign test p = 0.696 — the sign is not merely unresolved, it points the *opposite* way
  from the mean. That is not a tie artefact.
* **`feet_height`'s overall −7.5% is not withdrawn**, and my earlier phrasing that its
  median is zero "so most episodes don't change" was the wrong inference to draw.

And one cell the sign test upgrades rather than demotes: **`foot_sliding` on S4 is not a
"sign flip inside a null" — it is a broad worsening.** Mean interval [−0.0153, +0.1408]
(null, one episode carrying 35.7%), but sign test **p = 0.003** with median **+0.0260**
and zero ties. On the longest episodes most sequences slide *more* under the arm; the mean
simply cannot resolve it. That strengthens the length reading rather than weakening it,
and it is the one place where the arm is broadly harmful.

For symmetry, the same test on `scene_human_penetration_s_mean` overall: 29 tied, **266
better**, 174 worse, and the mean is a null. That is the mass-concentration finding in one
line, and it is why the citation rule for P2-BG exists.

## 2026-09-01 — P2-ROOT result: contact recovered, but the hip seam is broken

Run `p2-mixer-rootsplit-p15-p17oc-s42-20260831` completed on `node01`: all four
shards and the merge exited 0, and all 469 preregistered episodes are present. The
worker ran the clean, manifest-pinned commit `60e3bd1`; the worker-initiated recovery
landed 22 files, and one SHA-256 pass found the worker and authority trees identical.
The compact result is
`experiments/results/p2_mixer_rootsplit_p15_p17oc_s42_20260831.json`.

### Gate result: FAIL

| criterion | P2-ROOT | gate | result |
|---|---:|---:|---|
| `contact_percent` | **0.68659** | ≥ 0.58780 | PASS |
| completion | **0.75480** | ≥ 0.74330 | PASS |
| scene-human `s_mean` | **7.16626** | ≤ 6.28801 | **FAIL** |
| scene-object `s_mean` | 30.77058 | report only | — |

The main `s_mean` delta against G=0 is +2.6%, CI [−0.4272, +0.8817] in absolute
units: unresolved by the mean, and not a penetration improvement. The more stable
bounded prevalence metric is decisive in the harmful direction:
`scene_human_penetration_frame_ratio` rises from 0.30995 to **0.36270**, +17.0%,
95% CI [+0.0363, +0.0692]. Its sign is broad (270 worse, 157 better, 42 tied;
non-tied sign-test p=5.0e-8), and the largest episode is only 2.9% of the absolute
total delta. The regression is concentrated in S1 (+37.6%, n=182) and S2 (+14.8%,
n=159); S3 and S4 are null.

### The preregistered decomposition answered both halves

The contact prediction was right. P2-BG fell from G=0's 0.69147 to 0.60395;
P2-ROOT reaches **0.68659**, recovering **94.4%** of that loss. Against P2-BG the
paired increase is +0.08265, CI [+0.06769, +0.09739], with 310 positive, 130
negative and 29 tied episodes. Root disagreement was therefore the mechanism of
almost all P2-BG's contact cost.

That does not make this a usable split. `foot_sliding` rises from 0.16506 to
**0.25289** against G=0: **+53.2%**, CI [+0.06733, +0.10887], with 340 of 469
episodes worse and non-tied sign-test p=5.6e-23. Against P2-BG it is +112.1%.
`feet_height` still improves 10.8%, so HSI's legs retain one benefit while their
motion becomes incompatible with HOI's pelvis. This is exactly the preregistered
"hip seam is broken" cell: the split now crosses at both hips and its penetration
column cannot be credited to root placement alone. The pelvis-centred ±0.6 m HSI
scene window is an additional confound, not an alternative success reading.

**Decision:** reject the raw-channel root/lower-body split. Do not tune its weights.
The next mixer proposal must preserve kinematic coherence and separately specify
which pelvis centres HSI's scene query; it needs a new dated preregistration and user
approval before any GPU work. P17-OC remains non-promoted, so this row is an immutable
operator diagnostic, not a main-table result, and a settled HSIPrior will require a
fresh row rather than rewriting this one.

### Governance deviations retained, not repaired after the fact

The scientific hypothesis and discriminator were committed before launch in this
file at `60e3bd1`, but the matching registry hypothesis row was omitted. A post-result
row is not being backdated or represented as preregistered; the completion row records
the omission and points to the real timestamped evidence. The run also lacks a
separate pre-launch fully resolved Hydra config and machine-preflight file beside its
manifest. The start manifest does retain the base config content, exact overrides,
hardware snapshot, clean pinned Git state and both checkpoint hashes, which is enough
to audit this one-key diagnostic but does not retroactively satisfy those lifecycle
gates.

## 2026-09-01 — P2-KIN-API: kinematically coherent composition, preregistered

**Approved by the user 2026-09-01.** This is an implementation/API subphase, not a
checkpoint-selection experiment and not a 469-episode quality row. HSIPrior is still
improving; P17-OC may be loaded only to prove that the runtime interface works on real
data. No metric observed from it may select a weight, threshold, schedule, joint group
or future HSIPrior checkpoint.

### One manipulated factor

Hold P2-ROOT's ownership fixed: HOI owns the pelvis/root, torso, arms, hand markers,
object and contact; HSI owns the complete left and right leg rotation branches. Change
only the coordinate in which ownership is applied:

| | P2-ROOT | P2-KIN-API |
|---|---|---|
| rotation composition | select predicted **global** rotations, then run IK | convert each expert to its own **local** rotations, select branches, then run FK |
| position composition | independently select 28 predicted positions | reconstruct the articulated body; rigidly attach the six extra eye/hand markers |
| root | HOI | HOI, exact channel ownership |
| object/contact 216:232 | HOI | HOI, bitwise unchanged |
| HSI scene-query pelvis | shared chain state / previous composed prediction | the same; explicitly the actual composed pelvis, never HSI's private pelvis |

The 22-joint FK tree does not contain all 28 position markers. Slots 22/23 (eyes) are
attached to the composed head frame; slots 24/26 (index markers) to the corresponding
composed wrist; slots 25/27 use the two hand endpoints already present in the 24-joint
FK offsets. The attached local vectors come from HOI. This is fixed representation
plumbing, not a learned or tuned rule.

### Structural hypothesis and falsification

P2-ROOT's hip seam was created by differencing an HSI global hip rotation against an
HOI global pelvis rotation. Selecting HSI's hip-to-foot **local** rotations and running
FK below an HOI root removes that artificial disagreement while leaving the expert
roles unchanged. The implementation passes only if all of the following hold:

1. HOI root position, object pose and contact are exact invariants; history frames are
   restored exactly from `fixed_points`.
2. The composed local rotations equal HSI on joints `{1,2,4,5,7,8,10,11}` and HOI on
   every other rotation joint, within `1e-5` rad geodesic error.
3. FK bone lengths match the supplied rest offsets within `1e-5` m, every output is
   finite, and no one of the 84 position channels is independently averaged.
4. The six extra markers move rigidly with their declared HOI parent frame; 216:232
   remain bitwise equal to HOI.
5. The existing raw composer and the `G=0` bitwise anchor remain unchanged.
6. HSI occupancy receives `current` for its anchor query and the previous **composed**
   prediction for temporal queries. There is no private-HSI-pelvis configuration.

Any failed invariant blocks the subphase. Repairs may correct implementation defects
only; changing ownership, adding a blend weight or choosing a different scene-query
pelvis is a new direction requiring new approval.

### Validation and explicit non-claims

Run registry/config validation, the complete authority suite, and a full-window
batch-1 benchmark of the composer with CUDA synchronization because it adds IK/FK to
every reverse step. A real-data functional smoke uses the first canonical HOSI scene
(`hosi_scene_limit=1`, seven episodes) on the idle 4-GPU worker with one GPU, no run id,
`hosi_expected_episodes=null`, and the existing P15/P17-OC pair. It checks only finite
execution, all seven outputs, audit fields and the absence of API/shape failures. Its
quality metrics are non-reportable and forbidden as tuning evidence.

No formal HOSI row runs in this subphase. Once HSIPrior settles, a new preregistration
must compare the kinematic operator against G=0 and the raw P2-ROOT ownership-matched
row over all 469 episodes. Whether HSI-local legs improve scene compliance under an
HOI carrier is deliberately unanswered here.

## 2026-09-01 — P2-KIN-API completion: PASS for API, no quality claim

The fixed operator is implemented at runtime commit `1cc4961a7240f8a9a5626fccf94e0e130949d4c2`.
It converts both experts to parent-local rotations, assigns the two complete leg
branches to HSI, rebuilds the fixed 22/24-joint tree by level-vectorized matrix FK,
and transports the remaining four markers in their HOI-parent frames. HOI root and
216:232 ownership remain exact, and HSI receives the shared `current` pelvis for the
anchor occupancy query plus the previous composed x0 for temporal queries. There is
no alternative private-HSI pelvis switch and no learned or tuned parameter.

The first real production call caught an implementation defect before the smoke:
`InfBaGelDataset.quat_fk_torch` returns global quaternions `[N,22,4]`, whereas the
initial test stub returned matrices. Commit `d9be278` corrected that contract and a
new test invokes the production dataset implementation directly. The synchronized
benchmark then measured 25.546 ms per composer call because the dataset FK launches
one small CUDA kernel per joint. Commit `1cc4961` replaced only that execution with
algebraically equivalent, parent-tree-validated matrix IK and level-vectorized FK.
The production parity test is within the preregistered `1e-5` rotation/position
tolerances.

### Validation record

| check | result |
|---|---|
| focused structural/sampler suite | 60 passed |
| final authority suite | 891 passed, 4 skipped in 339.65 s |
| registry before completion row | 327 valid records |
| synchronized RTX 3090 batch-1 × 16 benchmark | 4.088 ms/call median (4.074–4.133), 2.044 s per 500-step window |
| pre-vectorization comparison | 25.546 ms/call; vectorized path 6.25× faster |
| first canonical-scene smoke | exit 0; 1 scene, 7/7 episodes, 0 skipped, 23 windows |
| smoke sampler audit | 11,500 compose calls; operator/query-pelvis fields exact; 0 sampler/guidance nonfinite values |
| artifact recovery | 9 files; worker/authority SHA-256 lists identical; list digest `8c6073ab7705776ea744157d14cdc5d3b059d7b63a6d6591a4c299e5324276cc` |

The worker's `infbagel` environment does not contain pytest, so no worker test pass is
claimed; the same committed object passed the authority suite, while the worker
proved the GPU/runtime/assets path and registry preflight. The compact record is
`experiments/results/p2_mixer_kinematic_api_s42_20260901.json`; recovered smoke files
remain untracked under `results/incoming/p2-kin-api-smoke-s42-20260901`.

The seven-episode evaluator quality values are intentionally absent from the compact
record and from this conclusion. They are non-reportable, may not select any mixer
parameter, and say nothing about whether P17-OC's legs improve composition. P2-KIN-API
therefore closes as an API PASS only. After HSIPrior settles, the exact next entry is
a new preregistration for the complete 469-episode comparison specified above.

## 2026-09-04 — P2-KIN-R2CG: settled-teacher operator comparison, preregistered

**Approved by the user 2026-09-04.** The checkpoint-frozen teacher for this comparison
is R2 final EMA
`hsi_b_r2_fullbody_seam_epoch222.pth`, SHA-256
`7a81a0a2627967a396e54aa08c0bad4612e294a4df33aac9ada4b063058740fe`, with the
Phase 1C R2-CG inference recipe `hsi_guidance_posterior_coef1=true`. The recipe was
implemented and passed its native HSI gate on `phase/01c-hsi` at `b9296ed`; carrying
that conclusion and its three-line sampler change onto the mixer branch is explicit
user-approved cross-branch communication. HOIPrior remains P15 online
`ed8cf16916f476349c53a9403c9a22415eeba7f8c9694ec91c44e55b70f6c11c` with guidance
Arm B. Neither expert is trained or tuned here.

R2 is the best available Diffusion Teacher at preregistration time, not a promise that
no later HSIPrior will supersede it. The two rows below may decide the composition
operator under this frozen expert pair; they may not select an HSI checkpoint or tune a
weight, schedule, joint group, threshold or scene-query pelvis. If a later HSIPrior is
promoted, the selected operator requires a fresh checkpoint-paired row rather than a
reinterpretation of these results.

### The guidance transfer is part of the operator contract

Merely setting `hsi_guidance_posterior_coef1=true` in the existing composed config
would be a silent no-op: `HOSIComposedSampler` calls the HSI denoiser directly to obtain
`x0_hat` and does not call `Sampler.p_sample`, where native R2-CG applies guidance. It
also rejects the evaluator's generic HOSI `guidance_fn`, correctly, because that
function includes hand-object and object-scene terms and would trespass on HOI-owned
object/contact behavior.

The composed interpretation is fixed before implementation:

1. Obtain both expert clean predictions and form the actual raw or kinematic composed
   clean body exactly as already specified.
2. Evaluate **only** `apply_hsi_guidance_loss` on the 24 FK joints reconstructed from
   that composed clean body. The energy is human-scene only; it has no dependency on
   channels 216:232 and does not use the HOSI object/contact guidance terms.
3. For reverse steps 499 through 1, add
   `posterior_mean_coef1(t) * guidance_weight * grad(-loss, composed_x0)` to the shared
   posterior sample, then restore the two history frames exactly. Step 0 is unguided,
   matching native diffusion. `guidance_weight=1`; cap, dose and alpha-decay remain off.
4. The energy is evaluated on the body the shared chain will actually follow, not on a
   private HSI pelvis/body. Anchor occupancy still uses shared `current`; temporal
   occupancy still uses previous composed `x0`. There is no private-HSI-pelvis option.

Adding the coefficient after the shared posterior is the same location and scaling as
native R2-CG. The noisy `x_{t-1}` need not itself satisfy FK; every following clean
prediction is reconstructed by the selected composer, just as a native guided sampler
denoises a guidance-shifted noisy state at its next step. This does not weaken the
P2-KIN-API clean-output invariants.

### Two formal rows, one controlled contrast

Both rows use 67 scenes × 7 objects = 469 canonical episodes, four scene-level shards
on `infbagel-4gpu` GPUs 0–3, seed 42, `hosi_per_episode_seeding=false`, 500 diffusion
steps, `mixer_hsi_w=1`, `mixer_hsi_object_voxel_mode=occupied`, repaired entry-0
occupancy layout, shared/previous-composed query pelvis, and the exact two checkpoint
hashes above.

| cell | run id | only operator difference |
|---|---|---|
| raw control | `p2-mixer-rootsplit-r2cg-s42-20260904` | P2-ROOT raw global-position/global-rotation channel ownership |
| candidate | `p2-mixer-kinematic-r2cg-s42-20260904` | P2-KIN local-rotation ownership plus FK position reconstruction |

Ownership is identical: HOI root/pelvis, torso, arms, hand markers, object and contact;
HSI complete leg branches `{1,2,4,5,7,8,10,11}`. The raw control must be rerun with R2-CG;
the old P17-OC P2-ROOT row is historical mechanism evidence, not a valid paired control
for this checkpoint/recipe.

### Frozen reading and gate

All 15 persisted per-episode metrics are compared with 10,000 paired bootstrap
replicates, seed 42, keyed by `scene_name/object_name/test_idx`. Completion uses its
episode-proportion comparison. Heavy-tailed penetration columns additionally report
better/worse/tied counts, a non-tied sign test and the largest-episode share of the
absolute total delta; a mean CI alone cannot establish breadth on this benchmark.

The candidate is an **operator PASS** only if:

1. Against the raw R2-CG control, `foot_sliding` has paired mean-difference CI upper
   bound below zero. This is the direct falsification of the raw hip-seam mechanism.
2. Against the same raw control, neither `contact_percent` nor completion falls by more
   than 0.02 absolute, and `scene_human_penetration_frame_ratio` is not significantly
   worse. These guards prevent a smoother result obtained by disengagement or by giving
   scene compliance back.
3. Against the sealed G=0 anchor, the historical P2 gates remain unchanged:
   `contact_percent >= 0.5878`, completion `>= 0.7433`, and
   `scene_human_penetration_s_mean <= 6.28801`. Object-scene penetration is reported
   without a threshold, as before.
4. Every episode is present and finite; both checkpoint hashes, the operator/query
   audit, R2-CG guidance call count, zero object-channel guidance dependency, and exact
   history restoration pass.

If criterion 1 fails, reject the kinematic operator as an empirical repair even though
its API invariants hold. If criterion 1 passes but a utility guard fails, record a
structural mechanism positive and a Phase 2 quality FAIL; do not tune this operator in
the same direction. No seven-episode smoke metric may alter these rules.

### Lifecycle and execution

Use one preregistration commit, one implementation/config/test commit and one completion
commit. The implementation adds two thin config fragments and the minimum default-off
R2-CG plumbing; it does not change `code/priors/core/`. Before the formal rows: registry
and Hydra resolution must pass, the complete authority suite must pass, the R2 file must
be transferred worker-initiated and hash-verified, and one canonical-scene foreground
smoke must prove finite R2-CG guidance plus exact audit fields. The smoke is functional
only and its quality values are non-reportable. Launch the raw control and candidate as
separate four-shard campaigns, with the raw control first; recover each once, merge only
after four zero exit codes, then run the frozen paired analysis and stop.

## 2026-09-04 — P2-R2CG-ENG1: inference-equivalent engineering pass, preregistered

**Approved by the user 2026-09-04.** The first raw R2-CG campaign
`p2-mixer-rootsplit-r2cg-s42-20260904` was stopped after 21/469 episode log records
because measured latency was roughly 67--82 seconds per generated window while each GPU
held only about 1.8--2.2 GiB and sustained roughly 18--24% SM utilization. All four
shards exited 143 after SIGTERM and the terminal manifest is registered as `aborted`;
none of its partial quality values may be read, reported or used for selection. The run
id is sealed and will not be reused.

This is not a second CPU-thread experiment. The launch already set
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=4`, each shard had
15 threads, and the host load was about 9 on 48 CPU cores. Phase 1C measured the same
cap improving diffusion evaluation only from 61.694 to 60.761 seconds/window (1.0154x).
The dominant fixed cost is the 499-step HSI scene-guidance/autograd path.

### Allowed implementation, with semantics frozen

The engineering candidate may make only these transformations:

1. Reuse the sampler's device-resident 32^3 meshgrid instead of recreating it on every
   reverse step, rebuilding it only if batch/device/dtype actually changes.
2. Transform the previous clean human trajectory to world coordinates once before the
   three temporal occupancy queries instead of repeating the identical transform in
   each loop iteration.
3. Cache the goal occupancy and its 2-D goal position for the duration of one sampling
   window. They depend only on that window's fixed matrix, goal, scene, object points
   and masks. The current-state anchor occupancy and all three previous-composed-x0
   temporal occupancies remain dynamic and are recomputed at every reverse step.
4. Cache the immutable posterior-mean coefficient schedule on the guidance gradient's
   device and gather there, invalidating the cache if the source tensor changes.
5. Replace empty-tensor-plus-repeated-`cat` construction of `occ_list`/`occ_pos` with a
   single ordered `cat` over the exact same tensors. No tensor arithmetic may change.
6. Repair `tools/launch_hosi_sharded.py` so every emitted Hydra override, including the
   last one, retains its shell continuation and evaluator stdout/stderr plus the true
   exit code remain inside the shard session.

The pass must not reduce 500 diffusion steps or 499 guidance applications, change
checkpoint/config/guidance weights, combine conditional and unconditional model calls,
change RNG draws or their order, change the raw/kinematic operator, change either
scene-query pelvis, enable a new occupancy backend, use compilation/CUDA graphs, or
touch `code/priors/core/`. R2 remains a frozen current teacher, not an HSIPrior tuning
target.

### Equivalence and promotion gate

On one fixed real HOSI window, seed 42, P15 online + Arm B and R2 final EMA +
`hsi_guidance_posterior_coef1=true`, run baseline and candidate in interleaved order
after warm-up. CUDA timing synchronizes before and after each measured region. The
candidate is eligible only if all of the following hold:

- final 232x16 output is bitwise equal;
- every retained per-step clean/posterior checkpoint used by the harness is bitwise
  equal, proving divergence was not hidden by the final step;
- sampler audit is exactly equal, including 499 R2-CG calls, gradient telemetry,
  object-channel independence and exact history restoration;
- RNG state after the window is bitwise equal;
- median end-to-end window wall time improves by at least 15% without increased peak
  GPU memory.

No quality metric is computed for this gate. Failure of any equivalence item rejects
the candidate. Passing equivalence but missing 15% records a correct engineering null;
the formal campaigns remain paused rather than being relaunched for a negligible gain.
If both gates pass, rerun the already frozen raw-versus-kinematic comparison under new
ids `p2-mixer-rootsplit-r2cg-eng1-s42-20260904` and
`p2-mixer-kinematic-r2cg-eng1-s42-20260904`; all scientific gates and pairings from
P2-KIN-R2CG remain unchanged. The aborted id is never aliased to either row.

## 2026-09-04 — P2-R2CG-ENG1 completion: equivalent, throughput gate failed

Implementation commit `fcfe3cb` added the default-off engineering path and repaired
the sharded launcher's final-override continuation. The authority suite collected 903
tests: 897 passed and 6 skipped in 158.64 seconds. The contract-freeze test passed and
no file under `code/priors/core/` changed.

The fixed real-window gate ran on `infbagel-4gpu`, RTX 3090 GPU 0, at that exact commit.
After one warm-up per mode, three interleaved baseline/candidate pairs measured:

| mode | synchronized seconds/window | median | peak sampling allocation |
|---|---:|---:|---:|
| baseline | 64.589, 64.887, 65.796 | 64.887 | 426.014 MiB |
| ENG1 | 62.377, 65.503, 62.524 | 62.524 | 425.389 MiB |

The speedup is 1.0378x, or 3.64%, against the preregistered 1.15x promotion threshold.
The numerical gate passed completely: all 500 retained posterior states and the final
232x16 tensor were bitwise equal; final tensor SHA-256 was
`1631afb61598ece7c421671493003edebcfd9bc19e8f0028bed03793f865b980`; sampler audits,
CPU RNG and CUDA RNG were exactly equal; R2 guidance ran 499 times; peak allocation did
not increase. A supporting real-geometry occupancy-only probe was also bitwise/RNG
equal and improved 100 calls from 0.941 to 0.595 seconds (1.582x), proving that the
transformation works but that occupancy is not the dominant full-chain cost.

**Verdict: equivalence PASS, performance FAIL; engineering null.** Keep the reusable
default-off implementation and the launcher correctness fix, but remove the opt-in from
both R2-CG formal configs. Do not launch either `eng1` 469-episode id.

A follow-up read-only CUDA-event profile on the same real window measured 65.996 seconds
total: HOI x0 2.374 seconds, HSI x0 20.557 seconds (6.413 occupancy and 13.937 across
the 1,000 conditional/unconditional forwards), and R2 guidance 42.494 seconds. Perfect
same-step overlap of HOI and HSI prediction could save at most 2.374 seconds, yielding
63.622 seconds or 1.0373x. A two-GPU shard would therefore halve four-way shard
concurrency for at most 3.60% lower per-window latency; its ideal whole-worker throughput
is only about 0.519x the current one. Single-GPU dual streams have the same upper bound
before contention. Reject both as the next primary optimization. The next proposal must
profile and optimize the 42.494-second R2 human-scene guidance path while preserving the
frozen 499-call posterior recipe and its exact gradient.

## 2026-09-04 — P2-KIN-R2CG-r1: resume original inference after ENG1 null

**Approved by the user 2026-09-04.** Further inference engineering is deferred because
the current HSIPrior will later be distilled. Resume the already frozen P2-KIN-R2CG
scientific comparison on the original numerical path with
`mixer_inference_engineering=false`. This is not a new mixer direction and changes no
checkpoint, guidance rule, operator, pelvis, RNG, sharding, metric or gate.

The aborted raw id and the unpromoted `eng1` ids remain unavailable. Allocate fresh
identities:

| cell | fresh run id | inference path |
|---|---|---|
| raw control | `p2-mixer-rootsplit-r2cg-r1-s42-20260904` | original P2-ROOT + R2-CG |
| candidate | `p2-mixer-kinematic-r2cg-r1-s42-20260904` | original P2-KIN + R2-CG |

Run the raw control first on four one-GPU scene-level shards. Only after it completes,
returns and merges with four zero exit codes may the kinematic row start. Partial output
from the aborted lifecycle is neither an input nor a baseline. The expected latency is
about 65 seconds/window, so the campaign is intentionally accepted as slow; no further
optimization or parallel topology change is authorized in this lifecycle.

## 2026-09-05 — P2-KIN-R2CG-r1 raw control completion

The fresh raw control `p2-mixer-rootsplit-r2cg-r1-s42-20260904` completed at commit
`687f3b5` on `infbagel-4gpu`. All four shard exit codes were zero and the immutable
worker return passed a checksum-only comparison against the authority staging tree.
The guarded merge recovered 469 distinct canonical ordinals `0..468`, with shard
counts `119/112/119/119`; every persisted numeric episode value is finite and both
checkpoint hashes match the frozen pair.

The raw P2-ROOT point estimates are: completion `0.761194`, foot sliding `0.329279`,
contact `0.692278`, scene-human penetration mean `7.619572`, and scene-human
penetration frame ratio `0.388053`. These are the paired control values, not an
operator verdict; no raw-only threshold was preregistered and the kinematic cell has
not yet run.

The run logs contain 1,617 completed windows, implying 806,883 R2-CG applications
under the fixed 499-through-1 sampler path. A pre-existing evaluator limitation is
recorded rather than hidden: `sampler_body` is rebuilt for each scene, so the device
counter persisted by each shard covers only that shard's terminal scene. All four
terminal-scene audits independently satisfy exactly 500 compose calls and 499 HSI
guidance calls per completed window, zero nonfinite guidance steps, posterior coef1,
zero object/contact dependency, exact history restoration and the frozen shared-current
plus previous-composed-x0 pelvis query. The full-run call count is therefore derived
from completed window records and the branch-free sampler contract; it is not labelled
as a cumulative device counter.

Tracked compact result:
`experiments/results/p2_mixer_rootsplit_r2cg_r1_s42_20260904.json`. Recovered artifact
anchors are manifest `40e99e0a...`, completion record `fe61fd0b...`, aggregate
`52a15289...`, and full merged summary `196709c1...`. The raw-first lifecycle gate is
met; after this completion record is committed, start only
`p2-mixer-kinematic-r2cg-r1-s42-20260904` under the unchanged original inference path.

## 2026-09-05 — P2-KIN-R2CG-r1 candidate completion and operator verdict

The kinematic candidate `p2-mixer-kinematic-r2cg-r1-s42-20260904` completed at
commit `eca2dc0` on `infbagel-4gpu`. Four shard exit codes are zero; the guarded
merge contains 469 unique canonical ordinals `0..468`, all persisted numeric values
are finite, both checkpoint hashes are exact, and the recovered worker/authority
trees pass a checksum-only comparison. The operator audit is
`kinematic_local_rotation_fk` with HOI root/carrier, HSI local leg rotations
`{1,2,4,5,7,8,10,11}`, HOI-attached markers and HOI object/contact. All four
terminal-scene audits satisfy the fixed 500 compose / 499 R2-CG relation with zero
nonfinite steps, posterior coef1, no object/contact dependency and exact history
restoration.

The preregistered 10,000-replicate episode-paired bootstrap (seed 42, one shared
resample-index matrix, key `scene_name/object_name/test_idx`) rejects the operator.
For the primary mechanism metric, foot sliding is `0.330981` against raw `0.329279`:
delta `+0.001702`, 95% CI `[-0.023549,+0.026544]`. The mean interval is a null, not
an improvement, so its upper bound does not clear zero. More importantly for this
heavy-tailed metric, zero episodes tie and the direction is broadly harmful: 295
worsen against 174 improve, exact two-sided sign-test `p=2.53e-8`, median delta
`+0.019587`. The local-rotation/FK replacement therefore does not remove the
empirical sliding failure attributed to the raw hip seam.

The utility guards do not explain the rejection. Contact improves significantly by
`+0.011451`, CI `[+0.004012,+0.018976]`; completion changes by `+0.012793`, CI
`[-0.004264,+0.029851]`; scene-human penetration frame ratio changes by `-0.004646`,
CI `[-0.016124,+0.006990]`, so it is not significantly worse. Against the frozen
G=0 anchor, contact `0.703729` and completion `0.773987` pass, but scene-human
penetration s-mean `7.632337` fails the unchanged `<=6.28801` gate. Its paired mean
against G=0 is a tail-dominated null (`+0.645659`, CI `[-0.541556,+2.224837]`), while
the non-tied direction is nevertheless broadly worse (265 worse, 194 better, 10 tied,
sign-test `p=0.00106`). Penetration prevalence is significantly worse than G=0 by
`+0.073456`, CI `[+0.056529,+0.090202]`.

**Verdict: operator FAIL; reject, do not tune in this direction.** It preserves or
improves engagement relative to the raw control, but fails both the direct sliding
mechanism criterion and the carried G=0 scene-compliance threshold. No checkpoint or
expert is selected by this result, and the current R2 teacher remains an interim
teacher as already recorded. The full 16-metric paired tables, sign tests, tail shares,
source hashes and resample-index hash are tracked in
`experiments/results/p2_mixer_kinematic_r2cg_r1_paired_s42_20260905.json`.

## 2026-09-05 — HSI input semantics on generated HOI histories

The user approved continuing the input-first sequence after the source/result
review. This session completes this Phase 2 diagnostic only. R2's archived
training config uses `lingo_only=true`, `load_object_goal=true`, and
`is_mix=false`: object/BPS condition embeddings are masked during training,
history object/contact channels are zero, and future empty channels receive
the ordinary forward noise. The composed caller instead exposes real object
conditions and motion. Occupancy remapping changes the label alphabet; it
does not establish that the complete conditional input is in distribution.

**Hypothesis.** Restoring the training-time HSI object input semantics changes
its human predictions on a fixed generated motion hypothesis. Separate object
condition tokens from the empty motion channels before attributing a quality
failure to the learned scene prior or training a residual mixer.

**Frozen carrier and sample.** P15 online plus Arm B drives the existing G=0
500-step chain; R2 final EMA is queried as a passive observer with CFG `w=1`.
Use the checkpoint pair and immutable worker snapshot of the completed
R2-CG comparison by reference. HSI posterior guidance is off; geometry still
queries the full shared state and previous carrier x0 with object voxels
mapped to occupied. No HSI probe prediction feeds the chain. Keep seed 42,
existing per-scene generator semantics and every generated window/history.

Choose four scenes using only the benchmark's start/goal metadata: bins
`0,22,44,66` of the existing 67-bin longest-first scene-chord partition.
These are respectively `a3df624b-0917-46e9-ac15-fab766276c72`,
`b1b053a9-b268-4f62-a06d-b9b9325c5092`,
`4abcb667-c57f-4d8f-940a-d964152329d5`, and
`0aa05d5a-81d5-497b-832c-c90c3fe73a36`. Include all seven objects per scene:
28 episodes, one scene per worker GPU. The selection uses no quality result.
Probe reverse steps `499,400,250,100,10,1,0` in every carrier window.

**Paired interventions.** Query occupancy once at each selected state, then
reuse exactly those tensors, human state, text/goals/progress and timestep:

| cell | object-condition tokens | motion channels 216:232 |
|---|---|---|
| legacy | real object | shared carrier |
| tokens | training-time masked tokens | shared carrier |
| motion | real object | training-time empty view |
| both | training-time masked tokens | training-time empty view |
| repeat | repeat legacy | repeat shared carrier |

The empty view pins the two history frames to zero and sets future channels
to `sqrt(1-alpha_bar[t]) * epsilon`. One independent seed-42 auxiliary noise
tensor per window is reused across cells and noise levels. This defines paired
one-step marginal probes, not a new reverse process for the missing modality.
The auxiliary generator never consumes the carrier/global RNG stream. Masking
tokens occurs only at the denoiser call, leaving the geometric context intact.

**Measurements and reading.** Preserve selected inputs/predictions and all
per-window/per-step records in the ignored run directory. Report displacement
in cm for raw human positions and 24-joint FK, separately for root, legs,
torso, arms and hands, plus global-rotation changes in degrees. Compute
denormalized positions with the dataset function, not a copied scale factor.
Compare tokens/legacy, motion/legacy, both/legacy, both/tokens,
both/motion, and repeat/legacy. Repeat defines the numerical reference.
Report initial-prefix and generated-history windows separately, and keep all
seven timesteps visible. Primary reading: mean future FK displacement for
both/legacy at steps `100,10,1,0` on generated-history windows. Count episodes
above 1 cm (one fifth of the evaluator's 5 cm hand-contact distance) as an
effect-size description, not a checkpoint or quality promotion threshold.

Aggregate within episode first. Use `tools/paired_bootstrap.py`, 10,000
replicates, seed 42, for each contrast against its repeat reference. Also
aggregate to four scenes and report scene-level intervals; episode intervals
describe these selected tasks, and four scenes do not establish generalization.
Material differences establish input sensitivity, not that the corrected view
improves HOSI quality. Small differences retain the source mismatch but weaken
its explanation of the measured failure. No gate/weight/teacher is selected.

**Implementation and completion gate.** Add one reusable named probe to
`code/mixer/diagnostics.py`, invoked by the existing Hydra evaluator and one
config fragment; add no tool script. Keep ordinary sampling arithmetic
unchanged and verify the passive probe preserves the carrier/RNG. Component
tests cover the real Unet token mask, missing-channel noise/history, physical
units, pairing, and generated-history aggregation. Run the complete authority
suite and metadata validation. The registered real-data diagnostic supplies
runtime verification; no separate smoke workload is added. A production
throughput benchmark is skipped because the production executed path is
unchanged; diagnostic overhead is recorded only as diagnostic runtime.

Create resolved configs and machine preflight beside the manifest before
worker execution, publish committed source through worker-initiated Git, and
run under a worker-owned persistent session. Retain every operational failure.
Completion requires four successful processes, exactly 28 unique episodes,
complete timestep coverage on every generated window, finite diagnostics,
repeat/reference results, paired reports and the compact conclusion. Reuse
existing lifecycle provenance and sealed asset references; introduce no new
hashing mechanism. The next direction is chosen from these results, with the
failed raw/KIN experiments preserved.

Implementation verification: the actual Unet object-goal/BPS mask, empty-channel
noise/history, cm conversion, FK-facing measurements, episode/scene aggregation,
and two complete 500-step carrier windows are covered by six component tests.
The passive query restores CPU/CUDA RNG, including the existing occupancy
function's CPU `randperm`. The final authority run was `pytest tests`:
903 passed, 6 skipped in 167.17 seconds. Fully resolved diagnostic config and
registry validation passed. The next action is the registered worker diagnostic.

## 2026-09-05 — HSI input diagnostic completion

`p2-mixer-hsi-input-s42-20260905` completed on committed implementation
`34b7331`: 28 episodes, 124 carrier windows and 868 paired probe states.
All four workers and ten bootstrap analyses exited successfully. Every recorded
metric is finite; repeated HSI predictions have exactly zero measured difference.
The immutable 124 MiB return passed its single checksum-only comparison.

The input-first hypothesis is refined by the result: token-only masking has a
submillimetre mean future-FK effect, while restoring missing motion channels
produces about a centimetre of change. This weakens the token-path explanation
of the old composition failure. These are passive prediction sensitivities;
adapted closed-loop quality and the cause of the raw/KIN sliding failure remain
unmeasured. No expert, production input mode or mixer is promoted.

Numbers: `experiments/results/p2_mixer_hsi_input_s42_20260905.json`.
Scope, interpretation, verification and the next entry point:
`docs/phase_summaries/PHASE_2_INPUT_DIAGNOSTIC.md`. This closes the registered
diagnostic; Phase 2's joint-composition gate remains open.

## 2026-09-05 — Phase 2.1 relational prototype, preregistered

The user approved advancing the shared-chain HSI view and joint relational
prototype. Split the work before implementation: **2.1**, on
`phase/02a-relational-prototype`, delivers the input process, differentiable
geometry and a controlled generated-window experiment; **2.2**, on a later
`phase/02b-relational-rollout` session, evaluates closed-loop four-cell quality.
Both integrate into `phase/02-mixer`. This session completes 2.1 only; the
Phase 2 quality gate and learned Phase 3 training remain open.

**Input process.** Model the known empty object/contact state as clean zero.
Generate one independent full forward-noise trajectory per window,
`u[t] = sqrt(alpha[t])*u[t-1] + sqrt(beta[t])*epsilon[t]`, starting at clean
zero, and expose it in reverse order at the corresponding denoising step.
This supplies both the training marginal and the correct known-zero temporal
coupling; it replaces the previous probe's fixed-epsilon marginal construction.
History is zero, the human hypothesis is shared, object tokens are masked at
the HSI call, and geometry continues to see the complete world. Use the HOI
window seed through a separate auxiliary generator; preserve global/carrier RNG.

**Geometry.** Decode HOI to root position, root/global/local rotations and
the object's physical pose in the same window frame. Apply a common root-centred
translation and yaw to human and object; apply local SO(3) increments to the
21 non-root body joints, then reconstruct positions with FK. Object-reference
encoding uses the evaluator's `mat`, `obj_rot_mat_prefix` and BPS reference
explicitly. Object-relative pose is fixed in this first prototype. Contact
channels remain HOI. Pin history. Common motion alone preserves instantaneous
root-object and hand-object relations; stance preservation requires joint motion.
Geometry and its objective must be differentiable from zero residual.

**Four cells, fixed optimization.** At generated G=0 states use P15 online +
Arm B and R2 final EMA from the sealed input diagnostic. Obtain conditional and
temporal-scene-masked HSI predictions through the new input view. Define the HSI
proposal as their future FK difference, added to the HOI FK body; it is a
dynamic-perception increment, with text/goals/static scene unchanged.

All cells optimize the same root translation/yaw and body-local variables for
20 Adam steps, learning rate 0.05, initialized at zero for each state. Bound
each translation axis to 0.10 m and each angular increment component to 10
degrees with tanh. Every cell includes residual regularization, HOI hand-object
anchor preservation under fixed HOI contact labels >0.95, support-foot floor
height, near-floor foot velocity, and root/object endpoint preservation. Energy
terms use mean squared errors normalized by explicit physical tolerances:
residual scales above; 0.05 m hand anchors and HSI proposal; 0.02 m floor/stance
displacement; 0.10 m endpoints. Each normalized term has weight one. No sweep.
The stance mask is fixed from HOI FK using the evaluator's 0.08 m ankle / 0.04 m
toe heights and applies to adjacent frames with contact in both frames.

Factor H adds the HSI proposal loss. Factor G adds human-scene and object-scene
nearest-free-voxel displacement losses, each normalized by 0.05 m. Human geometry
uses 24 FK joints; object geometry uses 128 evenly indexed rest-mesh vertices,
fixed across cells. Cells are A00 (neither), A10 (H), A01 (G), A11 (H+G).
This is an optimization-based mixer prototype, not trained network weights.

**Sample and measurements.** Reuse the previous metadata-selected bins
0,22,44,66 and all seven objects per scene: 28 episodes on four worker GPUs.
Observe reverse steps 10,1,0 in every G=0 window, including generated histories.
All four cells see identical state, masks, scene, object points and HSI outputs;
optimized outputs stay in the observer and never alter the carrier. Record all
objective terms before/after, human/object scene residuals and occupied-point
fractions, contact-anchor drift, stance displacement, endpoint shifts, applied
translation/angle magnitudes, optimizer gradient finiteness, and synchronized
optimization time/peak memory. Save final cell motions beside per-state records.

Use episode-first aggregation, initial/generated history strata, all three
timesteps, and the existing 10,000-replicate seed-42 factorial paired bootstrap.
Also report four-scene aggregation. Primary comparison is A11-A01 on generated
history: scene residuals, with contact/stance/endpoint changes beside them.
Read signs and uncertainty without a post-hoc scalar quality score. A negative
or inconclusive HSI increment is retained. These quantities measure constrained
window behavior, not native success, sliding, naturalness or closed-loop quality.

**Gate and lifecycle.** Geometry tests must establish shared-transform relations,
reference-frame round trips, exact history, and finite nonzero gradients through
root and articulated joints at zero residual. Input tests establish the forward
recurrence and marginal/covariance identities. The real-data gate requires 28
complete episodes, every registered state/cell, finite optimization, four zero
worker exits and complete paired reports. This gate permits an interface/probe
handoff even if HSI adds no measured value; it cannot promote the full method.
The registered workload includes batch-1 compute/memory timing for this changed
path and real-data runtime validation. Add no separate smoke workload or new
tool script. Use one config fragment, component modules/tests, the existing
Hydra evaluator and bootstrap tool. Reuse sealed assets by reference, keep
core/expert files unchanged, and use preregistration/implementation/completion
commits. Archive resolved configs and preflight before the worker-owned run;
recover once with the existing transfer/checksum procedure. Write
`docs/phase_summaries/PHASE_2A_RELATIONAL_PROTOTYPE.md` before integration and
tag the completed 2.1 interface deliverable. No 2.2 run starts in this session.

Implementation detail fixed before execution: the four cells are vectorized as
one four-cell GPU batch for each source window. Losses are reduced within each
cell and summed for backward; Adam moments remain independent per cell. The
recorded optimization time and peak allocation therefore describe one source
window's complete four-cell computation. The source/evaluator batch remains one.

Implementation verification: six new component tests cover the forward-noise
recurrence/covariance, RNG isolation, shared-transform invariants, reference
encoding, zero-residual gradients, four-cell optimization and probe serialization.
The final authority suite, with the verified interpreter exported for subprocess
tests, passed 911 tests with 4 skips in 161.58 seconds. Resolved Hydra config,
registry validation and whitespace checks passed. The earlier suite's two setup
errors were the unexported interpreter variable; they required a command fix.

## 2026-09-06 — Phase 2.1 handoff: interface passes, HSI target is negative

`p2-mixer-relational-prototype-s42-20260905` completed at `1c09b38`: 28 episodes,
124 windows, 372 observed states and 1,488 optimized cell outputs. All four GPU
processes and both factorial analyses completed. Every state matches the previous
G=0 carrier exactly in current, previous x0 and HOI prediction; all optimized
histories/contact channels are exact and all optimizer gradients are finite.
The single immutable recovery/checksum comparison passed.

The preregistered 2.1 interface gate passes. The scientific result does not
promote the HSI target recipe: A11 increases object-scene residual and contact
anchor drift against A01, while human-scene and stance differences are
inconclusive. Endpoint shifts relative to HOI become slightly smaller. A01's
geometric objective improves the measured object-scene proxy against A00.
These are window optimization outcomes, with all four outputs kept outside the
carrier; they do not establish native HOSI quality or a trained mixer.

Numbers and all-metric source pointers:
`experiments/results/p2_mixer_relational_prototype_s42_20260905.json`.
Implementation, interpretation, verification, limitations and the exact next
entry: `docs/phase_summaries/PHASE_2A_RELATIONAL_PROTOTYPE.md`.
Integrate this interface deliverable into `phase/02-mixer` and tag
`exp/p2a-relational-prototype-v1`. Before a Phase 2.2 rollout proposal, reconsider
the uniform full-body DP-displacement target in light of this negative result.
The current HSI target is a retained control, not a selected production recipe.

## 2026-09-06 — Phase 2.2 relational closed-loop experiment

The user settled R2+CG / P15+guide and requested continuation. This implements
the previously separated closed-loop subphase on `phase/02b-relational-rollout`.
The Phase 2.1 negative changes the question: first establish whether the shared
relation/geometry correction helps native rollout. Retain the tested HSI
increment as a negative control; its window loss supplies no positive training
target. Experts, the frozen core, and their inference weights stay fixed.

**Mechanism.** At reverse steps 10,1,0, apply the Phase 2.1 bounded relation
optimizer to the actual clean prediction before the shared DDPM posterior.
The corrected clean prediction also becomes the next temporal scene-query
reference. Use the exact known-empty HSI input process, 20 Adam steps at 0.05,
the existing 67 residual coordinates, physical scales and source masks. All
other denoising steps retain the HOI clean prediction. Common motion adjusts
both the human and HOI object; contact channels and history stay exact.
The sampler audit must describe this transformed object provenance explicitly.

**Fixed five rows.** R2-CG human-scene posterior guidance is active at all
499 nonzero reverse steps in every row, with coefficient1 and scale 1. P15
Arm B follows it with its sealed last-ten-step recipe. Every row has the same
A* goals, scene, seed 42, 500 steps, repaired occupancy and world geometry.

| Row | Clean correction at 10,1,0 |
|---|---|
| reference | HOI clean, with the matched CG/Arm B posterior; no relation optimizer |
| a00 | shared source relation, floor, stance, endpoint and residual objectives |
| a10 | a00 plus the retained HSI conditional-minus-temporal-masked FK target |
| a01 | a00 plus human/object nearest-free-voxel objectives |
| a11 | a00 plus both factors |

The R2 neural increment is queried in all four optimizer rows at the same three
steps; factors change only which objective contributes to optimization. The
reference and a00/a01 are controls for geometric guidance, not evidence that
the R2 learned scene prior has helped. The archived 469-case G=0 remains an
external context row because its CG state differs from these matched controls.

**Cohort and runtime.** Reuse metadata-selected scene bins 0,22,44,66 and all
seven objects: 28 complete episodes per row, 140 total. Use the eight idle
authority RTX 3090 GPUs for this mixer workload, batch 1; record the actual
allocation and any contention. Canonical episode/window seeds and posterior
noise stay paired. No weight/schedule search or neural training is included.
Archive resolved configurations, machine preflight and manifests before
persistent execution with the verified infbagel interpreter. The registered
experiment supplies real-data runtime verification and synchronized batch-1
compute/memory measurements; add no separate smoke or new tool script.

**Reading.** Persist all 15 native metrics and completion for every episode.
Use the existing 10,000-replicate seed-42 paired bootstrap: the four-cell
factorial, a01-reference, and a00-reference. Also report scene-mean versions
over the four scenes. Primary mechanism contrast is a01-a00; a11-a01 tests
whether the retained HSI target's negative transfers to rollout. Contact,
completion and feet height accompany every penetration/sliding comparison.
Native foot sliding observes the generated result, including newly planted
feet outside the optimizer's fixed source stance mask.

A pilot geometry benefit requires negative upper paired CI for object-scene
mean penetration in a01-a00 and a01-reference, contact/completion point drops
at most 0.02 against reference, and no significant worsening of native foot
sliding or human-scene frame penetration. An HSI benefit additionally requires
a11-a01 to improve scene metrics with the same engagement protections. Report
episode and scene uncertainty together; four scenes cannot establish the full
Phase 2 quality gate. Complete and retain all rows regardless of early signs.

**Deliverable gate.** Component tests establish single-cell/four-cell optimizer
equivalence, history/contact preservation, and that corrected clean predictions
reach the posterior and subsequent scene context. All authority tests, resolved
configs and registry validation pass. Complete 140 episodes with finite native
metrics, correction gradients and 499 CG calls/window, plus paired reports and
`docs/phase_summaries/PHASE_2B_RELATIONAL_ROLLOUT.md`. Classify the result even if
negative. Use preregistration, implementation and completion commits. This
subphase closes the pilot only; learned mixer training requires useful HSI
supervision and the outstanding full Phase 2 gate in a later session.

Implementation verification: the authority suite passed **914 tests with 4
skips in 172.44 seconds**. The new relation tests cover single-cell equivalence,
real optimizer history/contact restoration, and clean feedback through all 500
posterior steps. The test fixture uses valid SO(3) history and accounts for the
canonical step-zero posterior coefficient. Twenty exact job configurations
resolve completely. Each completed native episode also persists metrics,
generation timing and cumulative guidance/correction audit for detached-run
inspection. The eight physical GPUs are isolated with `CUDA_VISIBLE_DEVICES`;
four BLAS threads/process and a fixed scene-workload allocation are recorded
in the execution plan. Concurrent timing is throughput/context information.

## 2026-09-06 — Phase 2.2 completion: closed-loop quality FAIL

`p2-mixer-relational-rollout-r1-s42-20260906` completed on `3e17090`: all 20
GPU jobs and six paired analyses succeeded, covering 140 episodes, 620 windows,
309,380 CG applications and 1,488 relation corrections. Native metrics and
correction gradients are finite; correction histories/contact channels are
exact. The initial nohup launch consumed a separate id and failed before any
GPU job; its manifest/empty log remain retained. The fresh run used persistent
screen on authority GPUs 1–7 because GPU 0 was occupied at launch.

The deliverable gate passes; the pilot quality gate fails. A01's object-scene
depth sum improves against A00 by 8.909 (episode CI [-20.161,-2.025]), but its
improvement against the matched reference is unresolved at episode level
(-7.598, CI [-18.347,0.523]); the four-scene CI is [-12.852,-0.696]. Completion
falls from 22/28 to 19/28, exceeding the registered two-point protection budget.
Sliding rises from 0.17468 to 0.58421 and human-scene penetrating-frame prevalence
from 38.53% to 75.38%; both worsen at both resampling units. Contact stays near
the reference (67.34% versus 68.43%).

A00 already has sliding 0.55478 and completion 18/28. Therefore the common
reconstruction/objective bundle introduces the main cost before adding either
factor. This contrast does not identify which of its objectives causes it.
The floor objective and source-fixed stance mask remain candidates for a later
isolated diagnostic, especially given the feet-height change 4.012 to 1.670 cm.

The HSI result refines the window-only negative: A11-A01 reduces sliding by
0.21379 (episode CI [-0.26534,-0.16596]) and human-scene frame prevalence by
4.084 points, while increasing object-scene depth sum by 3.80666 (episode CI
[0.67010,8.07866]; scene CI [1.59575,5.64067]). A11 completes 18/28. Retain this
tradeoff; it supplies no net-quality promotion or training target for a mixer.

Full five-row values, both uncertainty units, gates, runtime and failures:
`experiments/results/p2_mixer_relational_rollout_r1_s42_20260906.json`.
Handoff and exact next entry:
`docs/phase_summaries/PHASE_2B_RELATIONAL_ROLLOUT.md`.
Integrate the completed pilot into `phase/02-mixer` and tag
`exp/p2b-relational-rollout-v1`, preserving the negative quality verdict.
The expert selection remains R2+CG / P15+Arm B. Start no new diagnostic or
learned-mixer training in this closing session.


## 2026-09-06 — Phase 2.3 approved common-correction diagnostic

The user explicitly approved the two-row diagnostic after the Phase 2.2 review.
Branch `phase/02c-common-diagnostic` isolates reconstruction feedback and the
floor objective. The archived A00 records contain 372 corrections: 309 have zero
initial stance energy, and the mean floor share of initial common energy is
96.803%. This is energy accounting, not gradient attribution or proof of an
empty stance mask. A00 worsens native sliding in 25/28 episodes and all four
scene means. Its fixed source stance mask uses absolute heights and two adjacent
frames; native sliding uses estimated-floor-relative heights on generated motion.

**Two new rows, no selection sweep.** `reconstruction` is A00 with zero optimizer
steps, retaining decode/encode and clean feedback at 10,1,0. `no_floor` is A00
with only the floor term excluded from the optimized sum, retaining 20 Adam steps
at 0.05. Floor energy is still measured. Residual/contact/stance/endpoint terms,
physical scales, bounds, masks and all sampler settings retain Phase 2.2 values.
R2 final EMA + CG and P15 online + Arm B remain fixed. Raw gate is zero; neither
new row adds an HSI learned or geometry objective. A zero-step result is an FK
projection of redundant source channels, not an identity requirement.

**Protocol.** Same bins 0,22,44,66, all seven objects per scene, seed 42:
28 episodes/124 windows per new row, 56 episodes/248 windows total. Reuse matched
`reference` and `a00` from `p2-mixer-relational-rollout-r1-s42-20260906` by reference.
Use the eight authority RTX 3090 GPUs, one independent scene job each, batch 1;
record actual availability, resolved configs and hardware before persistent screen
execution via the existing experiment/evaluator lifecycle. Reuse sealed input
provenance. No new tool script, smoke workload, expert/core edit or training.
The formal workload supplies real-data verification and synchronized batch-1
runtime/memory measurement; optimized compute changes with the zero-step arm.

**Analysis fixed before execution.** Report all 15 native metrics and completion
for all four rows. Use 10,000 seed-42 paired bootstrap replicates at episode and
scene units for reconstruction-reference, no_floor-a00, no_floor-reference and
no_floor-reconstruction. Primary diagnostic outcomes are native sliding and
human-scene penetrating-frame fraction; report contact, completion and feet
height beside them, plus object-scene depth. Negative upper 95% CI at both units
in no_floor-a00 supports an independent floor cost for that outcome. Positive
lower CI in reconstruction-reference identifies harmful reconstruction feedback.
Intervals crossing zero are unresolved. These contrasts permit interactions;
no sum-of-effects or individual stance-mask causal claim is implied. Lower
sliding with worse engagement/penetration is a tradeoff. Contact/completion point
losses above 0.02 against reference block a candidate promotion, as do significant
native sliding or human-scene prevalence regressions. Four scenes and this
diagnostic alone cannot satisfy full Phase 2 or authorize learned-mixer training.

**Deliverable gate.** Retain both rows and failures; finite native outputs, 499
finite CG calls/window, three correction calls/window, exact history/contact
preservation; actual optimizer gradients finite in no_floor (zero-step arm has
no optimizer gradients). Tests verify zero-step reconstruction and exact default
objective compatibility plus isolated floor exclusion; full authority suite and
registry validation pass. Write paired reports, compact result and
`docs/phase_summaries/PHASE_2C_COMMON_DIAGNOSTIC.md`; classify negative results.
Use preregistration, implementation and completion commits; integrate/tag the
completed diagnostic only. No further direction starts in its closing session.


Implementation verification: **916 passed, 4 skipped in 163.18 seconds** on the
final source. Zero-step reconstruction and isolated floor exclusion have component
coverage, including exact default-path compatibility. The evaluator entry point
now sets ROOT_DIR to its checkout's absolute root; this replaces its relative
reset and changes no data target. Eight fully resolved configs match sealed A00
except run/output locations and the registered interventions. The previous suite
also passed 916/4 before that path correction. Registry validation passes.


## 2026-09-06 — Phase 2.3 completion: floor cost identified

The approved `p2-mixer-common-diagnostic-s42-20260906` completed all 56 new
native episodes/248 windows on eight RTX 3090 GPUs in 34 minutes 17 seconds.
All eight jobs and eight paired reports passed; 123,752 CG calls, 744 corrections,
7,440 actual optimizer steps, finite outputs/gradients and exact history/contact.
The matched reference and A00 are reused from Phase 2.2.

Excluding only floor from A00 restores completion 18/28 to 22/28, lowers FS
0.55478 to 0.13113 and HS penetrating-frame prevalence 75.405% to 33.926%.
All three effects are significant at episode and scene units. Contact is 67.620%
versus A00 68.510%, with unresolved difference; native feet height returns from
1.670 to 3.895 cm. This identifies the floor cost within the fixed common bundle,
including its interactions, rather than a separate stance-mask effect.

Zero-step reconstruction reaches 22/28, contact 68.466%, FS 0.12928 and feet height
3.903 cm. Against reference, FS delta -0.04540 and HS prevalence delta -0.04331
are significant at both units. Thus reconstruction is a useful candidate anchor
on this pilot. Object-scene mean-depth change remains unresolved. No-floor adds
no significant FS/HS-prevalence improvement over reconstruction, decreases contact
0.8458 percentage points and increases HS maximum depth-sum 0.97438; both adverse
effects are significant at both units. Prefer the cheaper reconstruction anchor
for the next separately approved comparison. The full Phase 2 gate and useful
learned HSI composition remain open; previous factor effects must be retested on
the corrected anchor before transfer.

Full metrics, paired intervals, runtime and limits are in
`experiments/results/p2_mixer_common_diagnostic_s42_20260906.json` and
`docs/phase_summaries/PHASE_2C_COMMON_DIAGNOSTIC.md`. The diagnostic deliverable
passes. Integrate into `phase/02-mixer` and tag `exp/p2c-common-diagnostic-v1`.
Close this subphase without starting another workload.


## 2026-09-06 — Phase 2.4 approved floor-free factorial

User approved three new closed-loop rows: A01 geometry, A10 HSI increment,
A11 both, each with include_floor=false. Reuse sealed Phase 2.3 no_floor as
A00 and reconstruction as the practical anchor. Fixed R2 final EMA + CG and
P15 online + Arm B, seed 42, bins 0/22/44/66, seven objects each; 84 new
episodes/372 windows. Preserve 500 diffusion steps, CG at 499 steps, corrections
at 10/1/0, 20 Adam steps at LR 0.05, existing bounds/scales/masks and HSI input
view. No expert/core change, weight search or training. Branch
phase/02d-floor-free-factorial; run p2-mixer-floor-free-factorial-s42-20260906.

Hypothesis: removing forced floor allows geometry to improve scene penetration
against reconstruction; test whether HSI's previously measured sliding/OS-depth
tradeoff persists. A00 controls optimizer/common terms; reconstruction controls
practical utility. The retained HSI target is not presumed useful supervision.

Persist all 15 native metrics plus endpoint completion, engagement/feet height,
audits and synchronized runtime/memory. Paired 10,000 seed-42 bootstrap at episode
and four-scene units: factorial A00/A10/A01/A11 plus A01-reconstruction and
A11-reconstruction. Primary family has five OS s_mean contrasts: A01-A00,
A11-A01, A10-A00, A01-reconstruction, A11-reconstruction. In addition to nominal
95% intervals report Bonferroni 99% percentile intervals (five comparisons) at
each unit, using the same seed/resamples. Require negative upper adjusted CIs at
both units for a positive primary contrast. Report factorial interaction and all
other metrics with nominal CIs as secondary, explicitly without familywise claims.
Geometry promotion requires A01-A00 and A01-reconstruction OS gains. HSI promotion
requires A11-A01 and A11-reconstruction OS gains. Each promoted row must lose at
most .02 contact/completion against reconstruction and its matched factor control,
with no significant nominal worsening in FS or HS penetrating-frame prevalence
at either unit. Report HS/OS depth maxima and all adverse outcomes regardless of
gate. Four scenes and native endpoint completion do not establish full Phase 2,
state-machine success, motion realism or permission for learned training.

Use eight authority RTX 3090 lanes, one process/GPU, batch 1, four BLAS threads.
Schedule twelve scene jobs by known window counts; keep peak allocation below
20 GiB. Archive exact resolved configs and comparison to sealed no_floor, machine
preflight and inherited input references before persistent launch. Full authority
suite and registry validation required. Runtime code is unchanged: skip separate
functional/performance workloads; formal evaluation records batch-1 timing/memory.
Initial stability requires one completed episode per active lane with finite CG,
optimizer/native values and exact history/contact; episode artifacts provide
restart boundaries, retain any failed run and use a fresh id for an approved retry.

Deliverable: all 84 new episodes, matched 28-episode reused rows, complete paired
reports, retained failures, compact result and PHASE_2D_FLOOR_FREE_FACTORIAL.md.
Use preregistration, one config implementation and completion commits; integrate
and tag exp/p2d-floor-free-factorial-v1 after deliverable gate. Close only 2.4;
no subsequent experiment starts in its closing session.

Implementation verification: 916 passed, 4 skipped in 160.39 seconds; registry
valid with 345 records. All twelve exact resolved configs differ from sealed
no_floor only in run/output locations and the approved cell. The implementation
adds one inherited config fragment; runtime code and its tested path are unchanged.


## 2026-09-06 — Phase 2.4 completion: floor-free tradeoff retained

All 84 new episodes/372 windows and paired analyses completed successfully on
8x RTX 3090 in 40m39s. Reused reconstruction and no_floor A00 remain matched.
A01 reduces OS s_mean 30.53389 to 19.05300 against reconstruction; primary
Bonferroni 99% episode and scene intervals exclude zero. FS rises .12928 to
.20764, significant at both units; geometry quality gate fails its protection.
A11 raises OS s_mean to 25.38633 versus A01, a significant adjusted cost at both
units. Its sliding benefit versus A01 is unresolved at episode unit. A10/A11
complete 21/28 versus 22/28 controls; the same clothesstand endpoint exceeds
10 cm, violating the two-point completion budget. HSI quality gate fails.
All contact point protections pass; depth maxima and other adverse/positive
findings remain in the complete result. Secondary factorial interaction confirms
geometry-dependent HSI effects without identifying the underlying mechanism.

Deliverable PASS, both recipe promotion gates FAIL. Retain reconstruction anchor,
fixed experts and the earlier floor-active negative. Full Phase 2, realism and
learned-mixer training remain open. Compact result:
experiments/results/p2_mixer_floor_free_factorial_s42_20260906.json;
handoff: docs/phase_summaries/PHASE_2D_FLOOR_FREE_FACTORIAL.md.
Integrate/tag exp/p2d-floor-free-factorial-v1 after completion verification;
close only this subphase. Later review should localize the geometric sliding
cost from saved artifacts before a separately approved diagnostic.

Completion verification: **916 passed, 4 skipped in 157.04 seconds**; registry
valid with 346 records. Complete native/optimizer finiteness and history/contact
audits pass. All twelve jobs and all analysis reports succeeded.


## 2026-09-06 — Phase 2.5 approved stance recording diagnostic

User approved replaying floor-free A00/A01, 28 episodes each, with additional
recording only. Branch phase/02e-stance-recording; run
p2-mixer-stance-recording-s42-20260906. Fixed experts R2 final EMA + CG and P15
online + Arm B, seed 42, bins 0/22/44/66 and all seven objects; same 500-step
sampler, corrections 10/1/0, 20 Adam steps/.05, scales, masks and geometry.
A00 replays Phase 2.3 no_floor; A01 replays Phase 2.4 a01. No inference recipe,
objective, expert/core code or RNG change. Use eight authority RTX 3090 lanes.

Hypothesis: the sparse fixed world-height stance mask leaves corrected foot motion
outside its coverage; distinguish empty mask from stationary selected feet and
locate where native sliding differs from the correction-time surrogate. Existing
A01 audits have zero stance energy before/after at 338/372 calls; 19/28 episodes
have only zero stance energy and contribute 85.69% of net FS increase vs A00.
These are exploratory descriptive findings, not mask-count measurements.

Correct prior archival language: Phase 2.3/2.4 evaluate mode saved metrics and
scalar correction audits, not complete motion trajectories; save_motion_params
was not consumed by this entry point. Full motion retention starts with this
opt-in diagnostic. No prior result or negative finding is discarded.

Record per correction: exact source stance mask, raw optimizer parameters,
source and corrected world FK bodies/object poses, translation-only and common
translation+yaw decoded states (fixed-order descriptive decomposition), window
and reverse step. Also save each final sampled window, stitched pre-interpolation
positions/rotations/object poses, post-interpolation SMPL joints and native floor
height. Save detached CPU tensors per episode outside Git; flush buffers after
write. Observation must consume no RNG or modify returned tensors.

Gate: all 56 episodes/248 windows, 744 correction records and complete trajectory
artifacts; finite native/gradient values, exact history/contact, full authority
suite and registry validation. Every native metric and completion per episode
must equal its sealed row exactly; mismatch is a failed equivalence gate and
cannot support promoted inference or changed scientific findings. Compare existing
scalar optimization telemetry except timing/memory. New tests compare recorder
on/off tensors and RNG state and independent saved state/mask reconstruction.
Formal replay supplies real-data equivalence and recorded batch-1 latency/memory;
no separate smoke or benchmark workload. Peak allocation below 20 GiB, eight
persistent GPU lanes; initial stability requires first completed episode/lane.

Analysis uses GPU for tensor/frame calculations. First verify native FS exactly
from saved evaluated joints/floor. Decompose per-transition FS contributions by
source/corrected contact mask overlap on the correction grid, and native
predecessor-frame vs two-frame stance eligibility, world vs estimated-floor
thresholds. Separate fixed-mask empty/active corrections; report selected-foot
motion changes and outside-mask changes. Report root translation, added yaw and
added articulation contributions as order-dependent algebra, not isolated causal
interventions. Distinguish clean correction, final sampled window and stitched/
interpolated evaluated output; do not equate FK surrogate with SMPL native FS.
Episode-first aggregation, all four scenes, source vs generated histories and
three reverse steps. Use existing paired bootstrap (10,000 seed-42 replicates,
episode/scene units) for A01-A00 native replication and diagnostic quantities;
all new diagnostic CIs are exploratory nominal 95%, no recipe promotion gate.

Use one config fragment and existing component modules/tools. Preregistration,
implementation and completion commits; retain any failures, no run-id reuse.
Write PHASE_2E_STANCE_RECORDING.md and compact result, integrate/tag
exp/p2e-stance-recording-v1 after diagnostic gate. Full Phase 2 and learned
training remain open. Close only 2.5, with no new recipe in this session.

Implementation verification: **916 passed, 4 skipped in 159.36 seconds** on the
final source (earlier recorder suite also 916/4 in 159.11s). Component checks
establish exact recorder on/off outputs, RNG state and objective telemetry;
saved masks and decoded states match independent reconstruction. Eight resolved
job configs differ from their sealed row only in run/output paths and recording
flag. Registry validation passes with 347 records. New recording includes the
complete object rotations and evaluated joints before native metric processing.


## 2026-09-06 — Phase 2.5 completion: sparse stance coverage localized

All 56 episodes/248 windows and 56 trajectory files completed in 32m11s on eight
RTX3090 GPUs. Native per-episode metrics and optimizer scalars exactly reproduce
sealed A00/A01; saved SMPL joints reproduce native FS. Both rows have 338/372
empty masks; 19/28 episodes are entirely empty. Geometry's outside-source-mask
FS-change proxy increases .117005 with positive nominal episode/scene intervals;
inside-mask contrast is unresolved. The change is already present at clean
correction, with larger translation/articulation temporal increments. World-height
vs estimated-floor support selection is a measured mismatch; no mask intervention
has yet shown restored quality. Full statistical and representation limits are
in PHASE_2E_STANCE_RECORDING.md and the compact result
experiments/results/p2_mixer_stance_recording_s42_20260906.json.

Deliverable and exact replay PASS; retain prior recipe quality FAIL and frozen
experts/reconstruction anchor. Close this subphase and integrate/tag
exp/p2e-stance-recording-v1. Next separately approved experiment should test
support selection/height alignment before weight tuning or learned training.

Completion verification: **916 passed, 4 skipped in 161.25 seconds**; registry
valid with 348 records. All 16 native and 144 diagnostic metrics have complete
28-episode/4-scene paired coverage. Exact replay and all trajectory audits pass.


## 2026-09-06 — Phase 2.6 approved source-height intervention

User approved A00-height/A01-height, 28 episodes each. Branch
phase/02f-stance-height; run p2-mixer-stance-height-s42-20260906. Reuse sealed
Phase2.5 A00/A01 and Phase2.3 reconstruction. Same bins 0/22/44/66, seven
objects, seed42, R2 final EMA+CG/P15 online+Arm B, 500 diffusion steps,
499 CG calls/window, corrections 10/1/0, 20 Adam/.05, bounds/scales/objectives.
Only change stance height reference: world zero -> estimated source floor.
Retain two-adjacent-frame eligibility and squared horizontal stance displacement,
fixed source contact labels and absence of forced floor objective. No HSI factor,
expert/core change, weight search, evaluator change or neural training.

At each correction, detach its current pre-optimization 16-frame FK source.
Linearly interpolate toes on the native scale-3 grid, retaining the last real
sample but excluding its two artificial held duplicates. Select low-speed toe
samples by native 3D displacement <.005m; repeat last real velocity for its final
sample as the native estimator does. Cluster selected heights with native
DBSCAN eps=.005m/min_samples=3, preserving left-then-right input order for border
assignment and treating noise as a group as the existing evaluator does. Floor
is minimum group median, zero when no low-speed sample exists. Compute numeric
interpolation/clustering on GPU and verify against existing CPU native function
on matched nonpadded inputs. This is source-FK/native-rule alignment, not equality
to the future full-episode SMPL floor. No future realized trajectory or evaluation
floor is used. Freeze estimated height and resulting two-frame stance mask for
all 20 optimizer steps of that correction. Keep actual native evaluator unchanged.

Hypothesis: source-relative support coverage reduces A01 sliding while preserving
its scene-depth benefit. Risk: a short/noisy source may yield an unstable floor
or treat hovering feet as support. Persist estimates, low-speed sample counts,
absolute toe height, masks, source/corrected/sampled/stitched/evaluated trajectories,
new contacts, selected/outside-mask motion and all 15 native metrics/completion.
Summarize estimate changes across steps/windows and its differences from final
native floor descriptively. Coverage increase alone does not pass quality.

Primary family: three paired outcomes — A01-height minus sealed A01 foot_sliding;
A01-height minus A00-height OS s_mean; A01-height minus reconstruction OS s_mean.
10,000 seed42 paired replicates, episode and four-scene units; Bonferroni 98.3333%
percentile intervals (.833333,99.166667 percentiles) for three comparisons at each
unit. Require negative upper intervals at both units. Report all native metrics,
A00-height/A00 and A01-height/A01, interaction of height and geometry, plus both
new rows against reconstruction with nominal 95% secondary CIs. Contact and
completion point loss at most .02 for each new row against its sealed counterpart
and reconstruction; candidate must also pass against A00-height. Candidate may
have no significant nominal worsening of FS/HS penetrating-frame prevalence
against reconstruction or A00-height at either unit. Full Phase2/realism remains
open even if this pilot passes. Complete all rows regardless of early signs.

Gate: all 56 episodes/248 windows/744 corrections and complete recorded motions;
finite native/gradient/floor values and exact history/contact, exact default-path
compatibility, native-reference estimator tests and floor/mask frozen during
optimization. Full authority suite and registry validation. Archive eight resolved
configs compared to sealed rows, preflight, inherited input references and launch
artifacts before clean-worktree persistent execution on eight RTX3090 lanes.
Record synchronized batch1 timing/memory; formal run provides functional and
compute validation, no separate smoke/benchmark workload. Initial stability needs
one complete episode per lane and peak allocation below 20GiB. Preserve failures;
no mid-window resume or id reuse. Use one config fragment, existing component
modules/evaluator/bootstrap, no new tool script. Preregistration, implementation,
completion commits; PHASE_2F_STANCE_HEIGHT.md and compact result before integration
and tag exp/p2f-stance-height-v1. Close only 2.6; any further recipe needs approval.

Implementation verification: **919 passed, 4 skipped in 160.74 seconds**.
Native-reference tests cover CPU/CUDA source interpolation and DBSCAN height,
static/no-low-speed cases, no artificial terminal support, frozen source floor/
mask during optimizer steps, and exact default/explicit-world output agreement.
Eight resolved configs match Phase2.5 sealed rows except run/output paths and
source_floor=true. Registry valid with 349 records; diff check passed.


## 2026-09-06 — Phase 2.6 completion: coverage expands, quality FAIL

Both 28-episode rows completed on eight RTX3090 GPUs in35m56s. All motion/audits
and analyses pass. Empty masks fall338/372 to5/372 in each row. A01-height FS
.140129 vs old A01 .207636 is a32.51% point reduction but its adjusted episode
CI [-.197414,.045276] is unresolved. OS s_mean20.910681 improves against new
A00 and reconstruction under the three-comparison adjusted gate. It gives back
1.857684 of old A01's OS benefit, significant at nominal95% in both units.
HS penetrating frames rise4.5911 points against reconstruction, significant at
both units and failing the protection. Completion is22/28 with identical episode
outcomes throughout; all contact/completion point protections pass.

Source floors vary across windows and differ from final native estimates; these
are measured discrepancies, not ground-truth bias or proof of a causal failure
mechanism. The final classification is deliverable PASS, pilot quality FAIL.
Keep experts and reconstruction anchor fixed; full Phase2/learned training open.
Compact: experiments/results/p2_mixer_stance_height_s42_20260906.json;
handoff: docs/phase_summaries/PHASE_2F_STANCE_HEIGHT.md. Integrate/tag
exp/p2f-stance-height-v1 after final verification; close only this subphase.

Completion verification: **919 passed, 4 skipped in 165.41 seconds**; registry
valid with 350 records, complete native pairing at both units, saved-native FS
checks and all finiteness/history/contact audits pass.

## 2026-09-06 — Phase 2.7 approved source-stance-velocity experiment

User approved two 28-episode rows, A00-increment and A01-increment, following a
read-only review of the Phase 2.5/2.6 recordings. Branch
phase/02g-stance-increment; run p2-mixer-stance-increment-s42-20260906.
One config fragment inherits config_sample_hosi_stance_height and enables
source_stance_velocity. Fixed R2 final EMA+CG/P15 online+Arm B, seed 42,
bins 0/22/44/66, seven objects, 500 diffusion steps, 499 CG calls/window,
corrections 10/1/0, 20 Adam steps at .05, all bounds/scales and source contact
labels. Source-relative floor estimation and the two-adjacent-frame stance mask
remain frozen within each correction. Forced floor and HSI target factors stay
off. A01 adds the same human/object scene geometry as before.

**Mechanism and evidence.** A01-height selects 14,243 foot transitions; 4,042
exceed the existing low-speed criterion (.005 m per scale-3 interpolated sample,
equivalently .015 m per source segment). They contribute 97.989% of pooled
selected horizontal displacement squared. This identifies optimization pressure,
not whether those movements are gait or genuine sliding. A00's episode-first
initial stance energy is .010769 before height alignment and 1.679636 afterward;
initial residual/endpoint energies are zero and contact energy is about 4e-13.
The source motion itself therefore drives common optimization. On A01-height,
same-frozen-floor mask losses are 790/14,243 (5.55%); the previous review's
interpretation of mixed-reference lost_world_mask_count as lost support is
withdrawn. Its mean selected FS proxy contribution decreases, rather than
increases, overall. Between-window floor jumps have exploratory episode-level
Spearman rho .0066 with the FS change against old A01, without causal attribution.

Replace only the stance energy target for the approved rows. For frozen source
feet p_src and corrected feet p, define horizontal correction d = (p-p_src)_xz
and E_stance = mean_M ||d[t]-d[t-1]||^2 / (2*.02^2), on t=2..15 and joints
7/8/10/11. This equals the difference between corrected and source horizontal
displacements. The detached source stays fixed through optimization. The existing
zero-velocity recipe remains the default for sealed configurations. Preserve raw
stance_displacement_cm telemetry and add stance_increment_cm; energy_stance
records the actual optimized quantity. Record the target mode in the sampler audit.

Hypothesis: preserving source foot displacement removes an unnecessary drive to
rewrite gait, reducing HS penetrating-frame prevalence while retaining geometry's
OS-depth benefit. Risk: the source contains genuine sliding that this objective
preserves; geometry may also need to change foot motion. Saved joints and scalar
metrics do not locate native mesh-SDF penetrations, so neither the height review
nor this hypothesis claims that the extra HS penetration is at the feet.

**Comparisons and gates.** Reuse sealed Phase 2.6 A00-height/A01-height and
Phase 2.3 reconstruction. Primary family of three contrasts: A01-increment minus
A01-height HS penetrating-frame ratio; A01-increment minus A00-increment OS
s_mean; A01-increment minus reconstruction OS s_mean. Use 10,000 seed-42 paired
bootstrap replicates with Bonferroni 98.3333% percentile CIs at episode and
four-scene units; require all upper limits below zero at both units. Secondary
nominal 95% comparisons: each new row versus its sealed height counterpart and
reconstruction, A01-increment versus A00-increment, and the target-by-geometry
interaction. Report all 15 native metrics and completion, including negatives.
Each new row may lose at most .02 contact/completion points against its height
counterpart and reconstruction; A01 must also pass versus new A00. A01 may have
no significant nominal worsening in FS or HS frame prevalence against new A00
or reconstruction at either unit, or in FS against A01-height. Unresolved primary
comparisons fail promotion; absence of significance is not equivalence.

**Registered diagnostic.** Preserve source/corrected, transform-decomposed,
sampled-window, stitched and evaluated-joint snapshots. Verify that before
energy_stance and before stance_increment_cm are exactly zero in all corrections;
reconstruct the optimized increment and raw displacement from recorded motions.
Report both displacement measures, source floor/coverage, same-floor mask changes,
horizontal and vertical correction magnitudes, sparse-joint occupancy and scene
energies, episode-first and by step/initial-versus-generated history. Use each
episode's own frozen source for the within-correction comparison; different rows'
later source trajectories have diverged. Verify saved joints reproduce native FS.
These diagnostics do not replace native quality gates or measure motion realism.

**Execution and deliverable.** Same 56 episodes/248 windows/744 corrections and
14,880 optimizer steps, complete native pairing, finite values/gradients and
exact history/contact. Meaningful component checks cover zero loss/gradient at
the moving source, resistance to added foot motion, fixed source targets/masks,
horizontal-only loss and default output compatibility. Run the full authority
suite, registry validation and resolved-config comparison before formal sampling.
The formal batch-1 run supplies real-data functional and synchronized compute/
memory validation; no separate smoke or performance workload is added. Archive
all eight resolved configs, preflight, inherited sealed input references, command
and launch/analysis artifacts. Start from clean committed source through
tools/experiment.py start, then native Hydra evaluation and tools/paired_bootstrap.py.
Use a host-owned detached screen with eight GPU lanes and automatic analysis.
Initial stability requires one complete episode per lane and peak allocation below
20 GiB on each 24 GiB RTX 3090. Preserve every failure; no run-id reuse or automatic
restart. Source scope: existing mixer relation module, its component tests and one
config fragment; no expert/core/evaluator change or new tool script. Preregistration,
implementation and completion commits; PHASE_2G_STANCE_INCREMENT.md and compact
result before integration/tag exp/p2g-stance-increment-v1. Deliverable validity is
separate from pilot quality; full Phase 2/learned training remain open. Close only 2.7.

Implementation verification: **922 passed, 4 skipped in 163.29 seconds**. Moving
source has exact zero stance energy/gradient, added supported horizontal motion
is penalized, and spatial shifts/vertical motion preserve the horizontal target.
Source feet, mask and floor remain fixed during optimization; recorded/default
corrector outputs agree. All eight resolved configs match sealed Phase 2.6 except
run/output paths and source_stance_velocity=true; Hydra instantiation confirms
the selected objective. Registry valid with 351 records; diff check passes.

## 2026-09-06 — Phase 2.7 completion: HS primary unresolved, common optimizer cost

All eight GPU jobs and automatic analyses completed successfully: 56 episodes,
248 windows, 744 corrections, 14,880 optimizer steps and 123,752 CG calls.
Saved motions reconstruct the optimized increment and every native FS value;
all initial stance energies are exactly zero. Deliverable PASS, quality FAIL.

A01-increment HS frame prevalence falls 2.7963 percentage points versus
A01-height. The adjusted episode CI [-6.4085,+0.2678] points crosses zero;
the scene CI [-6.4548,-0.0209] points passes. Nominal 95% intervals exclude
zero at both units, but the registered adjusted primary gate remains unmet.
A01-increment OS mean 21.285703 improves versus new A00 30.645320 and
reconstruction 30.533886, with both adjusted comparisons passing at both units.
All registered candidate protections pass. All episode completion outcomes
match reconstruction (22/28), while candidate FS increases from reconstruction
.129277 to .160854 with unresolved nominal intervals.

The common control has a distinct negative: A00-increment FS rises .010686
(8.27%) versus reconstruction, significant at both units. Its entire active
objective increases in 336/372 corrections (36 tied), from an episode-first
initial mean 3.94e-13 to .047956. Exact zero initial stance energy therefore
does not guarantee that the 20-step optimizer preserves the source. The logged
energies establish this departure; its numerical/optimization cause remains open.
Source/corrected mask membership, absolute/incremental foot displacement,
vertical corrections and scene energies are retained with step/history strata.

GPU job wall time was 32m06s; manifest start through analysis completion was
32m39s. Full native/paired/diagnostic results, runtime and failures are in
`experiments/results/p2_mixer_stance_increment_s42_20260906.json` and
`docs/phase_summaries/PHASE_2G_STANCE_INCREMENT.md`. Retain reconstruction;
review the common optimizer before a separately approved new mechanism.
Integrate/tag exp/p2g-stance-increment-v1 after completion verification.

Completion verification: **922 passed, 4 skipped in 163.87 seconds**; registry
valid with 352 records. Native pairing, saved-native FS and increment audits pass.
Source and native evaluator stayed fixed; existing immutable input references
are retained.

## 2026-09-06 — Phase 2.8 approved A00 optimizer diagnosis

The user approved investigating why A00 correction raises its complete common
objective. Branch phase/02h-optimizer-diagnostic; run
p2-mixer-optimizer-diagnostic-s42-20260906. This is one diagnostic experiment:
reproduce all 28 Phase 2.7 A00-increment episodes, bins 0/22/44/66 and seven
objects, and observe all 372 corrections from their own frozen source states.
Retain R2 final EMA+CG/P15 online+Arm B, seed 42, 500 diffusion steps,
499 CG calls/window, corrections 10/1/0, 20 Adam steps at .05, default
betas/epsilon, all bounds/scales, source-height masks, source stance displacement
target, contact labels and native evaluation. Floor, HSI and scene objectives
remain inactive in A00. The observation and counterfactual never feed the chain.

**Hypothesis and causal diagnostic.** Contact anchors are transformed into
object coordinates with a rotation transpose and reconstructed in world
coordinates. The resulting floating-point source residual may supply the only
initial gradient, whose Adam-normalized update can increase an already near-zero
nonnegative objective. Before/after logs alone do not establish this mechanism.
Record states 0..20: residual coordinates, every energy and active total;
updates 1..20: actual total gradient and Adam first/second moments; record each
active term's initial gradient separately. Retain the original contact residual,
mask, source object rotation and anchor so the round trip can be reconstructed.
Evaluate the same round trip in float64 using the saved float32 inputs to separate
cached nonorthogonality from subsequent arithmetic error; this does not recreate
double-precision expert predictions. Check the first Adam update against
-lr*g/(abs(g)+eps). Evaluate one shadow first proposal -lr*g to measure the
effect of normalization at the same .05 LR, without a learning-rate search.

For each identical source, run one shadow 20-step Adam solve with only the
contact energy omitted. All other active terms and optimizer settings match.
Record its complete trajectory and evaluate its final state under the original
objective as well. This ablation isolates the trigger; it is not a contact-free
candidate recipe. Source state, target, mask and global RNG remain unchanged.
The diagnostic records source/global rotations and reconstruction inputs needed
to replay a correction independently, avoiding another full sampling run merely
to recover missing optimizer inputs.

**Evidence and gates.** Deliverable: 28 episodes/124 windows/372 corrections,
7,440 original and 7,440 shadow updates, finite traces, exact history/contact,
and exact replay of all 16 native outcomes and existing correction scalars/states
against sealed A00-increment. Recording must preserve optimizer output and RNG
in component tests. Independent trace reconstruction verifies active energies at every
step and Adam's first update (atol 1e-8, rtol 1e-6). Initial term-gradient
attribution and contact-off source preservation are measured over every correction,
including zero-contact and zero-gradient cases. A causal-positive classification
requires all original initial noncontact gradients to be exactly zero, the
contact gradient to reproduce the full gradient, the first update to satisfy
Adam's formula, and all contact-off trajectories to remain at zero residual.
Report any counterexample and leave its cause unresolved. Report first/last/best
objective, stepwise increases, initial gradient/update size, term contributions,
rotation/round-trip discrepancies and source movement by correction step and
initial/generated history. The initial iterate is included in best-objective
reporting; it is never selected for generated output.

Use episode-first and four-scene means with 10,000 seed-42 paired replicates and
nominal 95% intervals for original versus shadow common-objective increase and
motion displacement. Report native replay and the previously measured A00 cost;
there is no new native-quality claim for the shadow or a promotion gate.
An observed trigger does not by itself prove that a changed contact expression
or optimizer will improve A01's scene-constrained rollout.

**Execution and closure.** One config fragment enables observation, implemented
in existing relational modules and component tests, with no new tool script,
expert/core/evaluator change or production objective/optimizer replacement.
Run the full authority suite, registry validation and resolved-config comparison.
The formal diagnostic supplies real-data functionality and synchronized batch-1
timing/memory checks; no separate smoke or performance workload is added. Use
four RTX3090 lanes, one per existing scene shard, on the eight-GPU authority host;
four unused GPUs avoid duplicate sampling and preserve within-scene seed order.
Archive exact resolved configs, machine preflight, source/input references and
commands beside the clean-worktree tools/experiment.py start manifest. Run in a
host-owned detached screen with automatic native/trace/paired analysis. Initial
stability requires one episode per lane and peak allocation below 20 GiB.
Preserve all failures and artifacts, with no run-id reuse or automatic restart.
Use preregistration, logical implementation and completion commits, then write
PHASE_2H_OPTIMIZER_DIAGNOSTIC.md and a compact result before integration/tag
exp/p2h-optimizer-diagnostic-v1. Close only Phase 2.8; reconstruction and the
fixed experts remain the comparison anchor. Any production repair is a subsequent
concrete experiment informed by this diagnosis.

Implementation verification: **924 passed, 4 skipped in 163.62 seconds**.
CPU/CUDA component checks preserve original optimizer outputs, scalars and RNG
exactly while recording the full trajectory; the source contact gradient explains
the fixture's initial update, and its contact-off shadow stays exactly at zero.
These fixture findings are not substituted for the registered real-data cohort.
Saved tensor caches reproduce decoded geometry and contact residuals. Four fully
resolved configs match sealed A00-increment except run/output paths and
optimizer_diagnostic=true. Formal GPU execution supplies functional and timing/
memory validation; the native evaluator and production objective remain unchanged.

## 2026-09-06 — Phase 2.8 completion: contact trigger and optimizer departure identified

All four GPU jobs and automatic analysis exited zero: 28 episodes/124 windows,
372 original and 372 contact-off solves, 14,880 total optimizer steps and
61,876 CG calls. All 16 native outcomes, original correction scalars and saved
motion fields exactly reproduce Phase 2.7 A00-increment. Active energies reconstruct
at every recorded iterate; all registered causal checks pass with no counterexample.

Every initial residual/stance/endpoint gradient is zero and the contact gradient
exactly equals the total. All 336 nonzero-gradient sources increase their objective
at Adam step 1 and remain worse at step 20; the other 36 have exactly zero selected
contact residual and stay fixed. Every correction has contact labels, so those
36 are not empty-contact cases. The source is the best of all 21 recorded iterates
in every case. Contact-off shadows keep zero parameters/gradients throughout all
372 solves and preserve the original complete objective at its source value.

Episode-first initial contact residual RMS is 4.2822e-8 m, maximum gradient
1.1198e-7 and first maximum parameter update .0397644. The first articulation
component reaches .397383 degrees on the same mean-of-maxima basis; common
translation/yaw are initially negligible. Initial/first/final common objectives
are 3.9373e-13/.0451663/.0479565. The first loss is mainly contact .0449961;
the final loss is mainly stance .0429265. The mean curve peaks at .419932 at
iteration 3. The recorded steps follow Adam's formula; the finite 20-step recipe
fails to minimize this near-zero source objective. Finite gradients alone do not
validate the solve.

The single unnormalized -.05*g proposal has mean maximum parameter size 5.5990e-9
and objective 4.0248e-13. Its 329 tiny increases, 7 decreases and 36 ties remain
visible; it establishes the scale of Adam normalization, not a validated SGD
replacement. Float64 round-trip evaluation on the cached float32 geometry retains
4.4768e-8 m mean RMS error. Casting that arithmetic retains cached rotation error;
no full-float64 geometry or sampler intervention was tested.

Contact-off minus original objective increase is -.0479565, nominal 95% episode
CI [-.0537445,-.0416525] and scene CI [-.0538459,-.0438220]. Corresponding human
RMS motion shifts are -.283432 cm and object shifts -.174962 cm, with negative
intervals at both units. These are paired same-source correction diagnostics,
not native outcomes for a contact-free rollout. The original native FS remains
.139963, with the sealed +.010686 cost against reconstruction. A separate read-only
audit of archived A01 finds its complete objective (including both scene terms)
increases in335/372 corrections, .0537254 to .0813346 episode-first. Its cause was
not subjected to this contact-off experiment. Retain its evidence as the next
optimizer review target rather than assuming an A00-only numerical repair suffices.

GPU wall time was33m45s; manifest through analysis35m10s; peak allocated804.66MiB.
The 28 motion files total327,323,430 bytes. An initial exact-replay checker failed
on NumPy arrays; its original program and failure were retained, array equality
was corrected, and the final complete analysis passed without restarting sampling.
The original vectorized all-step Adam diagnostic uses float32 bias powers;
closure reconstruction with production Python-double powers agrees at the
registered1e-8/1e-6 tolerance (maximum absolute difference1.49e-8).

Implementation verification remains 924 passed/4 skipped at 4471699. Completion
verification: **924 passed, 4 skipped in 169.11 seconds**; registry valid with
354 records. Archive all raw traces, native/paired projections, strata, zero-gradient
cases, runtime and analysis revision. Integrate/tag exp/p2h-optimizer-diagnostic-v1
after completion verification. Full Phase2, learned training and realism remain
open; close this diagnostic subphase only.

## 2026-09-06 — Phase 2.9 approved complete-objective Armijo solve

The user's approval authorizes the solver repair following Phase 2.8. Branch
phase/02i-armijo; run p2-mixer-armijo-s42-20260906. One experiment runs
A00-armijo/A01-armijo, 28 episodes each, against sealed Phase 2.7 A00/A01-increment
and Phase 2.3 reconstruction. Fix seed 42, bins 0/22/44/66, seven objects,
R2 final EMA+CG/P15 online+Arm B, 500 diffusion steps, 499 CG calls/window,
corrections 10/1/0, representation, bounds, scales, weights, source contact
labels and source-height/stance-displacement targets. Floor and HSI factors
stay off. A01 retains both scene terms. Contact's existing formula stays active
and unchanged; this isolates the solver from an objective-definition intervention.

**Solver.** Replace the selected rows' fixed-step Adam with steepest descent
and Armijo backtracking on their complete active loss. At each iteration use
d=-g, initial step size 1, shrink factor .5 and c1=1e-4. Accept only a trial
with E_trial <= E_current + c1*alpha*(g dot d) and E_trial < E_current. The
strict decrease distinguishes a useful update from a rounded equality. Re-query
occupancy and nearest-free references at every current/trial/final evaluation.
Commit the accepted parameters as the next iterate. Stop at exactly zero
gradient, exhaustion of 20 backtracking trials, or 20 gradient iterations.
No ad hoc loss/gradient threshold, momentum, weight search or output selection
is added. A finite search limit is a recorded stopping reason, not a stationarity
certificate. Return the last accepted iterate. Default Adam remains available
unchanged for sealed configs and the diagnostic comparison.

The gradient-iteration ceiling stays 20; backtracking adds at most 400 objective-
only trials per correction, with source/current/final evaluations accounted
separately. Record actual gradient evaluations, objective evaluations, accepted
updates and line-search attempts. Initial per-term diagnostic derivatives are
separate from solver gradients. Initial step 1 is the registered line-search
proposal scale, not a tuned learning-rate sweep. The original solver is 20 Adam
steps at .05. Include synchronized compute/memory costs; equal diffusion budgets
do not imply equal optimizer work.

**Registered internal diagnostic.** Each new source also receives one original
20-step Adam/.05 shadow with the same complete objective, geometry, target and
frozen mask. The shadow never feeds sampling. Retain both parameter/energy/gradient
trajectories, every line-search trial's step size, energy and acceptance decision,
and accepted-state nearest-free references. Save cached geometry, scene grid,
contact anchors/mask and HSI target so accepted full losses can be reconstructed.
Keep source/corrected/transform-decomposed motions and sampled/stitched/evaluated
trajectories. Verify that the complete loss never rises at returned iterates or
within the accepted path, and that every accepted trial satisfies both conditions.
Report zero-step solves, search exhaustion, budget exhaustion and tiny source
movements; do not equate a motionless A01 with a useful solver.

The scene objective is piecewise because its integer voxel references change.
Its existing query also leaves zero displacement for invalid grid indices.
Record source/corrected/shadow invalid-query and geometric out-of-grid fractions,
including newly invalid points, so a boundary crossing is visible beside a scene-
energy reduction. These observations do not alter the loss or the native evaluator
and do not certify physical collision improvement. Preserve unresolved geometry
limitations alongside all native outcomes.

**Gates and comparisons.** Deliverable: all 56 episodes/248 windows/744 original
solves and matching Adam shadows, finite energies/gradients/recordings, exact
history/contact, frozen targets/masks and complete native pairing. Armijo trajectories
and their final complete objectives must be nonincreasing at every correction;
independent accepted-state reconstruction must agree at atol1e-8/rtol1e-6.
Component checks cover near-zero contact-driven sources, a nonzero scene objective,
a changing voxel reference, independent cells, trial rejection, accepted-state
return and recorded/default output/RNG compatibility. A failed technical gate is
an implementation/operational failure; a stalled or weak solver can pass technical
validity and fail the native quality gate.

Native primary family: A00-armijo minus A00-increment FS; A01-armijo minus
A00-armijo OS s_mean; A01-armijo minus reconstruction OS s_mean. Use 10,000 seed-42
paired replicates and Bonferroni 98.3333% percentile intervals at both 28-episode
and four-scene units, requiring all upper limits below zero. Report all 15 native
metrics and completion, both new rows against their sealed counterparts and
reconstruction, A01 versus new A00, and the solver-by-geometry interaction with
nominal 95% intervals. Each row's contact/completion point loss is at most .02
against its sealed counterpart and reconstruction; A01 also passes against new
A00. A00 may have no significant nominal FS/HS-frame harm versus reconstruction;
A01 may have none versus reconstruction, new A00 or sealed A01 at either unit.
An unresolved primary fails promotion; absence of significance is not equivalence.

Report episode-first and scene means, 10,000 seed-42 nominal paired intervals for
Armijo versus same-source Adam full-objective change and human/object motion RMS,
plus step10/1/0 and initial/generated-history strata. Separate within-source
optimizer diagnostics from the native comparison of diverging rollout sources.
Technical monotonicity alone does not establish realism or useful scene correction.

**Execution and closure.** One config fragment inherits source-stance-increment
and selects Armijo, initial step1 and the solver diagnostic. Scope existing mixer
relational modules/component tests and phase docs; no new tool script or expert/
core/evaluator change. Use one preregistration, one logical implementation and one
completion commit. Run the complete authority suite, registry validation, resolved
config comparison and diff check. The formal run supplies real-data functionality
and synchronized batch-1 timing/memory checks, without a separate smoke or benchmark.
Run eight RTX3090 lanes in a host-owned detached screen with automatic native,
trace and paired analysis. Initial stability requires one episode per lane, all
trace checks passing and peak allocation below20GiB. Keep all failures, raw data
and run identities; no automatic restart or overwriting of results. Archive
preflight, exact resolved configs, input identities by reference and commands before
clean-worktree tools/experiment.py start. Full solver counts and timings replace
the old assumption that every correction executes exactly20 gradient updates.
Write PHASE_2I_ARMIJO.md and a compact result before integration/tag
exp/p2i-armijo-v1. Close only Phase 2.9. Full Phase2, realism and learned training
remain open; any further intervention requires its own approval.

Implementation verification: **932 passed, 4 skipped in 171.78 seconds**.
Near-zero real-geometry fixtures and nonzero scene objectives pass on CPU/CUDA;
observed/default Armijo outputs, metrics and RNG agree exactly. Analytic tests
recover the complete-loss optimum, preserve independent cells, reject a trial
whose refreshed voxel reference raises its objective, and retain the last accepted
state on search exhaustion. Recorded accepted states retain their actual scene
references. All eight resolved configs match sealed Phase 2.7 except run/output
paths, solver=armijo, learning_rate=1, max_backtracks=20 and solver_diagnostic=true.
Registry validation passes with 355 records; diff check passes. Formal execution
will supply the registered functional and synchronized timing/memory validation.

## 2026-09-06 — Phase 2.9 completion: monotone solve and pilot quality PASS

All eight GPU jobs and automatic analysis exited zero: 56 episodes, 248 windows,
744 Armijo solves and same-source Adam shadows, 123,752 CG calls. Every accepted
iterate satisfies the registered full-objective Armijo condition and strict
decrease. All final objectives are nonincreasing, active energies independently
reconstruct, source masks/history/contact are exact and saved joints reproduce
native FS. The solver made 4,818 gradient evaluations, 28,333 line-search trials,
34,639 total objective evaluations and 4,246 accepted updates. Adam shadows made
14,880 gradient updates and never entered the generated chain.

The three primary contrasts pass Bonferroni 98.3333% intervals at both units:
A00-armijo FS versus A00-increment delta -.0106451 (episode [-.0178746,-.0046410],
scene [-.0162632,-.0050597]); A01-armijo OS mean versus new A00 delta -4.028411
(episode [-9.143940,-.651786], scene [-5.942986,-1.832115]); versus reconstruction
delta -4.029828 (episode [-9.147277,-.652113], scene [-5.946129,-1.833127]).
All point and nominal protections pass. Completion remains 22/28 with identical
episode outcomes for every row. Contact is 68.4657% for A00 and 68.3976% for A01;
the latter is .06812 percentage points below reconstruction with unresolved
nominal intervals. Reconstruction remains the comparison anchor.

A00 FS .129318 is 7.61% below old A00 and only .0000409 above reconstruction;
its source correction is at floating-point scale, with exactly zero stance and
endpoint energies. A01 FS .137591 is .008313 (6.43%) above reconstruction, with
nominal episode CI [-.007186,.033708] and scene [-.007046,.032112]. Passing the
registered absence-of-significant-harm protection does not establish equivalence.
A01 HS frames 34.5061% versus reconstruction 34.2028% is also unresolved. Its HS
mean 3.894199 versus 4.014550 is nominally better only at scene level. The full
Phase2 requirement for useful human-scene composition remains open.

The quality tradeoff against old A01 is material: OS mean rises 21.285703→26.504058
(delta +5.218355) and HS mean 2.212554→3.894199 (+1.681645), with positive nominal
95% intervals at both units. HS/OS maxima also worsen at both units. A01 retains
43.57% of the old OS-mean benefit versus reconstruction. Its FS improvement versus
old A01 is unresolved at both units. All 16 native metrics, nominal comparisons,
solver-by-geometry interactions and measured costs remain in the compact result;
the passing pilot is not a claim of dominance over the Adam recipe.

On identical A01 sources, mean complete objective falls .0535347→.0399240 under
Armijo and rises to .0808130 under Adam. Of 372 A01 corrections, 217 start with
positive scene energy; all 217 lower both complete and scene energy, 167 stop at
the 20-iteration budget and 50 exhaust line search after accepted updates. The
155 zero-scene sources account for all 125 zero-update solves; 30 take numerical
contact-improving steps. A00 has 109 tiny decreases and 263 ties, versus 339 increases
and 33 ties in its Adam shadows. A00/A01 total line-search exhaustion counts 334/192
include the near-zero source cases and must not be read as failed scene descent.

Boundary diagnostics identify seven newly invalid object-point observations in
four A01 corrections: floorlamp in scene b1b053a9, window 5, steps 10/1/0 (six points,
zero previous scene residual), and monitor in scene 0aa05d5a, window 7, step 0 (one point,
scene-energy drop .000139799 on that point). The cohort episode-first mean drop
on newly invalid points is 8.32136e-7. This leaves a real objective-domain issue;
these co-occurrences do not explain the full native gain. No new invalid human
query is recorded. Keep query validity, geometric out-of-grid status and native
mesh-SDF metrics distinct.

GPU job wall 36m52s; manifest through analysis 38m28s; peak allocated 804.43 MiB.
56 motion files total 710,829,890 bytes. Instrumented correction sums are 107.93s
A00 and 293.50s A01; generation sums 5375.08/5750.01s and shadow/archive sums
206.50/212.65s. Eight-lane timing includes brief initial trace verification on
GPU 7 and does not establish isolated production latency. No operational failure,
restart or result overwrite occurred. Implementation suite: 932 passed/4 skipped
in 171.78s. Completion suite: **932 passed, 4 skipped in 171.53 seconds**;
registry valid with 356 records. Native pairing, monotonicity, frozen-source and
saved-state audits all pass.

Integrate/tag exp/p2i-armijo-v1 after completion verification. Retain Armijo as a
passing pilot solver, fixed experts and reconstruction as the anchor. Read
PHASE_2I_ARMIJO.md before the next read-only review of scene-reference switching,
grid coverage and positive-scene solves that exhaust line search. No further
experiment is approved by this completion; close only 2.9.

## 2026-09-06 — Phase 2.10 DP-Edit implementation and integration diagnostic

Authorization: user requests advancing the new experiment in
/data/yujinlun/papers/PriorHOSI_SceneEvidenceEditing_Codex_Handoff.md. Branch
phase/02j-scene-evidence; deliver the handoff section 16 minimum implementation.
Hypothesis: raw HSI dynamic-perception evidence, queried on a clean candidate's
scene and shared noisy human view, can edit a fixed HOI source while HOI reference
and relation constraints preserve manipulation. This finite diagnostic establishes
implementation validity; quality and transfer remain hypotheses until 469 tasks.

Implement a default-off post-window editor, after unchanged 500-step P15 online
Arm B generation. R2 final EMA provides raw cond/static-only pairs. HSI posterior
CG, raw/body composition and old relational correction are off. Reuse the 67D
relational parameterization, 10cm/10degree component bounds, source-relative stance
mask/foot increments, contact anchors, endpoint and residual terms. Explicit
weights are 1 for residual/contact/stance/endpoint/human_scene/object_scene, using
RelationalObjective's existing physical scales; floor and old HSI displacement
tracking have weight zero. Reconstructed source is the fixed teacher reference.

Teacher: eight descending rounded integer levels linspace(300,50,8), one noise
sample/level; shared source/candidate noise; separate seeded editor and known-empty
forward trajectories. HOI beta=1; lambda=.1 versus 0; alpha/sigma weighting; no RMS
normalization or clipping. Future 0:228 channels participate, HSI only 0:216.
Freeze teachers and detach query conditions. One accepted steepest-descent step
per refreshed teacher, initial step1, shrink.5, c1=1e-4, ten backtracks, dimensionless
proximal weight1. Each trial refreshes explicit nearest-free references. Only
local frozen-teacher surrogate decrease is claimed. Nonfinite teacher/gradient
is an explicit run failure, with artifacts retained and no automatic fallback.

Domain: use floating scene bounds, reject newly exterior points and increased
per-point exterior Euclidean distance (tolerance zero) against the current accepted
state. Record source and final geometric and integer-query invalidity separately.
Original exterior points remain represented and evaluated. Contact/history exact;
fixed object point set; no random geometry resampling or global RNG consumption.

One config fragment supports disabled, reconstruct_only, lambda0 and lambda.1.
Formal finite integration diagnostic: these four rows on scene bin0, all seven
objects, seed42 (28 episodes), up to four independent RTX3090 lanes; all native
metrics and full saved trajectories retained. No hyperparameter selection or
quality promotion from this diagnostic. It supplies real-data functionality and
synchronized batch1 teacher/solver/total timing/memory; no separate smoke test or
performance workload. Use tools/experiment.py start from clean implementation,
archive exact resolved configs/preflight and reuse sealed asset identities by
reference. No new tools script, hash mechanism, expert/core or metric changes.
Evaluator changes may only connect the existing motion recorder to the editor.

Gate: component mathematics/sign, source cancellation, masks, detached teachers,
independent per-cell Armijo, domain rejection, geometry/history/contact and RNG;
complete authority suite and registry validation; paired resolved configurations;
real full episodes including consecutive edited-history windows; finite nonzero
DP evidence at an interior noise level; all trial decisions/terms/call counts and
full native outputs saved. Report 10,000 seed42 paired diagnostic intervals using
existing paired_bootstrap; seven episodes/one scene give no generalization claim.
No significant-improvement requirement for this implementation gate. Finish with
PHASE_2J_SCENE_EVIDENCE.md and compact result. Phase2.11 (branch
phase/02k-scene-evidence-benchmark) will freeze pilot settings and preregister the
four-row 469-task comparison with episode/scene uncertainty before execution;
this session closes only 2.10 and leaves that next entry concrete.

Implementation verification: **945 passed, 4 skipped in 163.99s**, including 13
new scene-evidence component cases. The existing skips concern unavailable sealed
historical evaluation artifacts. Registry validation passes at357 records. All four
Hydra jobs resolve. Train/test/LINGO norm.npy human bounds are exactly equal;
P15/R2 canonical beta equality is checked when wiring the real models. Raw heads
share existing once-prepared conditions; the editor consumes independent random
streams. The sampler restores actual fixed history after native object SO(3)
projection in enabled editor modes; the disabled path retains native arithmetic.
The existing evaluator changes only by selecting scene_editor for its motion
recorder. Per-window logs include all active explicit terms, teacher and explicit
parameter-gradient norms/dots, every local trial, boundary counts, calls and
synchronized timing. No expert, core, metric or tool implementation changes.

## 2026-09-06 — Phase 2.10 completion: editor works, HSI gain unresolved

All four rows completed28 episodes/180 windows in the registered single scene;
all GPU jobs and paired analysis exited zero. DP-Edit accepts329/360 updates and
preserves exact history/contact and geometric domain constraints. HSI signal is
nonzero; its mean weighted parameter-gradient norm is about589x smaller than the
explicit terms. DP-Edit versus lambda0 HS/OS means improve .0598%/.0209%, with
both nominal paired intervals crossing zero. Completion stays7/7 for every row.
Reconstruction changes FS/contact/penetration by itself and remains a required
control. Implementation gate passes; quality/transfer and full469 gates remain
open. No weights were selected or adjusted using this diagnostic.

See PHASE_2J_SCENE_EVIDENCE.md and
experiments/results/p2_mixer_scene_evidence_s42_20260906.json for all metrics,
contrasts, runtime and artifact references. Phase2.11 starts with an explicit
choice between the fixed-setting469 comparison and independent-development
signal-scale calibration; any calibration requires its own dated registration
and must preserve the test/development boundary. This session closes only2.10.

Completion verification: **945 passed, 4 skipped in169.67s**; the skips are two
historical HSI checkpoint-pair cases and two historical P8 bootstrap cases.
Registry valid with358 records and diff check passes. Disabled-row native metrics
match all seven corresponding episodes of the sealed469-task P15 Arm B baseline
exactly. Integrate by fast-forward and tag exp/p2j-scene-evidence-v1.

## 2026-09-07 — Phase 2.11 independent-development evidence-scale calibration

User approval: "批准" to calibrating HSI evidence versus explicit constraints on
independent development scenes before locking469-task settings. This session
implements and executes that calibration, preserving Phase2.10 results.

**Inputs.** Use only LINGO v3 seed42 TRAIN families and OMOMO internal-validation
sequences from data/train. From existing training episode metadata, take numeric
base scenes with an eligible first start/goal chord in[.4,2.0]m, then seed42
RandomState permutation of the sorted eligible scenes. Seven scenes qualify;
take six: calibration004/006/055, verification023/037/036;067 is unused. Each
scene pairs the first eligible LINGO path with four first-pi0 internal-validation
HOI sequences by ascending language-window index: clothesstand1509/seq16,
floorlamp5507/seq51, largebox10079/seq91, suitcase236335/seq1844. All24 tasks use
training-only initial motion/text and LINGO paths. Object goal is the path endpoint
plus the heading-rotated initial HOI root-object xz offset, preserving initial
object height. Source future motion is generated by unchanged P15 Arm B500-step
sampling. No calibration scene/task is read from HOSI test annotations. Archive
IDs, family membership, source asset paths, tasks and the construction command;
verify scene-ID disjointness and compare selected occupancy arrays exactly with
all67 benchmark occupancy arrays on GPU to reject duplicate grids. No hashes are
added. Motion assets are read-only links to data/train; Scene_vis links point to
the six original LINGO grids. Do not infer initial object feasibility from the
human-only LINGO path. Report all source domain/collision conditions.

**Known correctness repair.** HSIPrior's _compute_occ_sample uses CPU randperm for
object vertices. Phase2.10's new teacher did not isolate this global draw. Reuse
an independent per-window CPU generator inside fork_rng around teacher geometry
queries; preserve and test CPU/CUDA global RNG and fixed seed reproducibility on
the actual occupancy query path. Historical2.10 outputs remain immutable. This
fix is necessary for causal scale comparisons and introduces no new loss term.

**Measurement and fixed rule.** A passive calibrate editor mode queries the eight
registered levels on each reconstructed source, beta_ref1/lambda1, with no update;
return the original source so its future history remains HOI-generated. Record
unit-HSI, HOI-reference and explicit parameter gradients and all source terms.
For positive-scene-energy windows with nonzero HSI gradients, compute the median
across levels of ||g_explicit||/||g_HSI(unit)||. Compute a median per calibration
scene, then the median across the three scene values. Set a single global lambda
=0.1 times that result, rounded to two significant figures. This preregistered
10% target calibrates numerical influence; it is not an optimum or a quality
criterion. Require all three calibration scenes to contribute and at least six
positive-scene windows in total; retain/report all zero-signal or inactive cases.
No clipping, per-window normalization, weight sweep or test-based selection.
HOI reference beta, explicit terms,8 iterations, bounds, RNG levels and solver
settings remain2.10. Archive the three verification-scene scale ratios without
using them to select lambda.

**Verification.** Apply frozen calibrated lambda and lambda0 on all12 verification
tasks with the same editor and original generation/rebase/history path. This is
24 closed-loop development rollouts. Record all window proxies and complete
motions; these LINGO scenes have no native HOSI mesh-SDF asset bundle, so this
calibration does not substitute proxy metrics for native469 quality. Technical
checks: finite teacher/gradients/motions, exact history/contact, legal domains,
local Armijo decrease, reproducible seeded queries and unchanged passive sources.
Eligibility for benchmark candidate: mean hand-anchor drift and source stance
increment at most0.5cm each, HS and OS scene residual means each at most1.10 times
lambda0 (or exactlyzero when its reference iszero), and at least one accepted
update. Report all contact/stance/endpoint/residual/domain metrics, all failures
and source activity strata. If the gate fails, retain the estimated coefficient
as a negative candidate and propose a separately registered follow-up. Never
search another lambda using verification outcomes. Use10,000 seed42 paired
intervals at12-task and3-scene units as development uncertainty only.

**Lifecycle.** One inherited config, one prerequisite and one logical implementation
commit, one completion commit. Existing evaluator gains a development task-root
and a diagnostic-output mode; native evaluation/default paths remain intact.
Reuse existing mixer modules for the named scale estimator and scene editor;
no new tools script or expert/core change. Stage1 runs six GPU scene lanes
(24 passive tasks); stage2 runs six lanes (two cells x three scenes,24 tasks).
Use clean-worktree tools/experiment.py start, detached host screen, resolved
configs/preflight and existing sealed checkpoint identities by reference. Peak
allocation gate20GiB. These registered workloads provide functionality and
synchronized batch1 timing; no separate smoke/performance workload. Run authority
suite, registry validation, config comparisons and default native/RNG contracts.
Write PHASE_2K_SCENE_EVIDENCE_CALIBRATION.md plus compact result; integrate/tag
exp/p2k-scene-evidence-calibration-v1 only if its deliverable gate passes. Full469
and InfBaGel comparison belong to Phase2.12 after its own registration.

Implementation verification:949 passed/4 historical-asset skips in164.69s;
17 scene-evidence component cases pass, including CPU/CUDA execution of the
native object occupancy query with unchanged ambient RNG and seed-reproducible
sampled vertices. Passive calibration returns its raw source exactly. The
scene-balanced estimator excludes verification and zero-scene records. The
24-task manifest is tracked at experiments/tasks/scene_evidence_development_s42_20260907.json;
its read-only snapshot and construction command are under
results/evidence-scale-inputs-s42-20260907/. Six stage1 jobs resolve. Runtime uses
the existing evaluator's generator, A*, rebase and recorder; development mode
returns declared window proxies before the native mesh-SDF metric block.

## 2026-09-07 — Phase 2.11 completion: lambda26 calibrated, candidate gate PASS

All48 development executions/148 windows finish. The three calibration scenes
provide26 positive-scene windows plus2 inactive ones. Scene-balanced estimation
selects lambda26 (unrounded25.820766); verification data do not select or alter it.
Independent verification gives contact-anchor drift.177683cm and source stance
increment.222867cm, both below.5cm; HS/OS voxel residuals satisfy the1.10x bounds.
All solver/history/contact/domain/RNG checks pass. HS/OS proxy improvements are
only.0300%/.0653%; contact-anchor drift increases.041257cm with positive nominal
intervals at both task and scene units. Candidate eligibility is not a quality
claim. All source conditions, unfavorable outcomes and intervals are retained.

Use PHASE_2K_SCENE_EVIDENCE_CALIBRATION.md and
experiments/results/p2_mixer_evidence_scale_s42_20260907.json. Freeze lambda26,
original reference/explicit/solver settings and the repaired independent query
RNG for Phase2.12 registration. No469 native workload or new parameter search
starts in this session. Integrate/tag only2.11 after completion verification.

Completion authority suite:949 passed/4 historical-asset skips in178.79s;
registry valid with360 records, selected benchmark config resolves and diff check
passes. The candidate is sealed with lambda26; all native quality claims remain
open pending Phase2.12.

## 2026-09-07 — Phase 2.12 fixed DP-Edit full469 native benchmark

The user approved proceeding after the lambda26 calibration. Run the full native
Atomic-HOSI enumeration:67 scenes x7 objects, seed42, P15 online/Arm B500 steps,
R2 final EMA, post-window DP-Edit with lambda26, beta_ref1,8 levels/iterations,
unchanged source/explicit terms, bounds, local Armijo and isolated scene-query
RNG from2.11. No test-based tuning, new teacher, training or additional mechanism.
Branch phase/02l-scene-evidence-benchmark; one config fragment inherits
config_sample_hosi_scene_edit and fixes lambda26,8 shards and469 expected tasks.

**Rows and provenance.** Generate reconstruct_only, lambda0 and full lambda26,
469 tasks each (1407 new rollouts). Reuse the sealed native HOI469-task result
p2-hosi-hoi-alone-g0-p15-guided-armb-s42-20260829 by reference. Disabled source
identity is covered by the current full native-chain tests and the2.10 real
seven-episode exact native comparison; do not spend a duplicate469 baseline run.
All new rows use the same checkpoint pair and stage2.11 generator/geometry/RNG
recipe. Old A01/Armijo outputs are not the lambda0 control. Save full source,
reference, edited, stitched and evaluated trajectories and per-episode audits.
The previously observed28-task/four-scene subset remains exploratory; no lambda
was selected on it. Report the complete469 population without favorable subsets.

**Statistics and gates fixed before outcomes.** Keep all15 native fields plus
completed (both final errors<10cm). Pair by scene/object/test_idx with exactly469
keys and67 complete scenes, and also aggregate per scene. All resampling uses
10,000 seed42 paired replicates, sharing the index plan within each unit.
Six pairwise contrasts among the four local rows report nominal95% intervals for
every metric. Primary family has four comparisons: lambda26 minus lambda0 on HS
s_mean and OS s_mean, and lambda26 minus native HOI on those same two metrics.
Use Bonferroni98.75% two-sided intervals (quantiles.625/99.375) and require an upper
limit belowzero at both episode and scene units for each claimed improvement.
HSI evidence gate requires both lambda0 contrasts; total scene-quality gate
requires both native-HOI contrasts. A partial pass is reported as metric-specific,
not promoted into a joint success. Report relative magnitudes even when a tiny
difference is significant; no practical-effect threshold is inferred after seeing
results. All scalar units retain native meaning: s_mean is a frame-average sum
of penetrating vertex depths, and completion is endpoint success.

Protection gate: lambda26 versus lambda0 AND native HOI, at both units, must have
nominal95% lower bounds for contact/completion deltas at least-.02 (two percentage
points). The paired bootstrap ratio mean(FS_full)/mean(FS_reference) must have
nominal95% upper bound at most1.10. These are registered noninferiority margins;
absence of significant harm by itself does not pass. Joint promotion requires
both primary gates and all protections. A negative/unresolved scientific gate
still completes this fixed benchmark; preserve all failures and do not launch a
new mechanism or tune on test results. Reconstruction contrasts remain visible.

Report improved/worsened/tied task and scene counts for all metrics, object strata,
completion discordance, and the ten largest absolute HS/OS delta contributors
alongside full means. Tail contributions are descriptive and never define a
selected result. Original tools/paired_bootstrap.py produces the nominal reports;
GPU float64 resampling computes the registered adjusted and FS-ratio intervals
using the same NumPy seed42 index plan. All metric pairs must be finite; no
nonfinite pair is silently excluded in the benchmark gate.

**External comparison.** Include InfBaGel paper Table1 Hybrid1:0.5 as an explicitly
paper-reported, unpaired reference: success81.45%, FS.15, contact76.96%, HS Pmean3.17,
OS Pmean12.45 (handoff section14.5 / original paper Table1). Its pre-repair motion
representation/reconstruction differs from current local evaluation. Present
side-by-side context without paired uncertainty or a same-protocol superiority
claim. Keep the historical July reproduction separate if cited; never load its
released checkpoint into repaired representation and call that equivalent.

**Execution and technical gate.** Eight RTX3090 host-owned persistent lanes; each
lane runs its full/lambda0/reconstruct shard sequentially.24 shard jobs overall,
followed automatically by existing native merge and complete analysis. Resolve
and archive all24 configs, verify only registered row overrides, archive machine
preflight and sealed input/baseline/calibration references, then clean-worktree
tools/experiment.py start. No new runtime/evaluator/tool source is needed. Run the
full authority suite, registry and diff validation. Formal jobs provide observed
batch1 timing/memory; no separate smoke or benchmark is added.

Initial stability: every lane finishes one full native episode with finite
metrics/teacher/gradients, exact history/contact and peak allocation below20GiB;
its per-episode motion/audit files must exist. Keep all independent lane exits and
never automatically restart or overwrite a failed result. After that interval,
report observed throughput/ETA and yield while the persistent campaign continues;
continuous Codex polling is unnecessary. Sampling can survive a control-session
interruption. Episode files are durable outputs, not a claim that the native
evaluator supports resuming a partially written shard.

Merge only complete8-shard rows with469 ordinals and67 scenes; verify per-episode
mode/lambda, fixed configuration and all source/solver/domain invariants before
reporting. Keep source/corrected query-invalid and geometric-outside counts and
realized solver/teacher call counts. Sharded timing remains workload cost, not
isolated production latency. Write PHASE_2L_SCENE_EVIDENCE_BENCHMARK.md and compact
result before integration/tag exp/p2l-scene-evidence-benchmark-v1. Close only2.12;
learned training, new weights, state machine and subsequent phases stay deferred.

Implementation/config verification:949 passed/4 historical-asset skips in166.94s;
all24 jobs resolve and their differences match the registered rows/shards/outputs.
GPU adjusted-delta and paired-ratio intervals match the NumPy reference. The
initial Phase2.12 handoff records automatic analysis and exact completion entry.
Feet height directional counts are descriptive because no preferred direction
is registered. Runtime/core/expert/evaluator sources remain unchanged.

### Phase 2.12 completion — native scene-evidence benchmark (2026-09-07)

The frozen `lambda_dp=26` benchmark completed all 469 native tasks for full,
lambda0 and reconstruction rows, reusing the sealed native-HOI row. Protections
and the total scene-quality family pass; the HSI-evidence family and joint gate
fail because full versus lambda0 has HS delta -0.0021 [98.75% CI -0.0726, 0.0582]
and OS delta +0.0841 [-0.0465, 0.3213]. Full versus native HOI improves HS by
-0.6123 [-1.3295, -0.1273] and OS by -2.3490 [-3.9245, -0.9359]. The benchmark
is closed with the isolated HSI contribution unresolved; no test-set tuning or
new mechanism follows in this session. See
`docs/phase_summaries/PHASE_2L_SCENE_EVIDENCE_BENCHMARK.md` and the compact result
for all means, protections, audits and tail diagnostics.

## 2026-09-07 — Phase 2.13 fixed-source DP teacher views

User approved implementation and bounded execution of the Phase2.13 handoff,
located locally in /data/yujinlun/report/PriorHOSI_Codex_Handoff_Phase2_13_2e69eba.md.
Branch phase/02m-dp-evidence-views; baseline is the sealed Phase2.12 integration.
H1: dynamic manipulated-object overlay, despite known-empty HSI object modalities,
may bias DP evidence toward avoiding the held object. Requerying the original
surrounding environment may improve scene-directed corrections and contact.
H2 (execution-space limitation) and H3 (proxy/surface mismatch) remain distinct.

Keep P15 online/Arm B500, R2 EMA, lambda26, beta1, eight existing levels, 67D,
six explicit terms, bounds, local Armijo, static grids and query positions fixed.
Default teacher view remains legacy_occupied. Add environment_only_temporal only
to the teacher; query native geometry with obj_points=None, retaining underlying
walls. Preserve all world flags, object context, geometry and original RNG streams.
No shared core, expert, native metrics, planner or dataset/model source changes.

**Data and budget.** Replay the24 existing Phase2.11 development tasks (all68
passive windows) using their original pipeline to recover exact missing query
context; compare raw sources against sealed passive tensors. Save regenerated
context for reproducibility. Keep calibration004/006/055 and verification023/037/036
separate. Only reconstructed source candidates; no nonzero-candidate extension.
Three views A legacy, B environment-only, C fixed spatially shifted environment.
Choose the handoff's spatial alternative to avoid a donor-context generation pass:
C queries the same scene at +2m along window-local X, rotated by the unchanged
world mat; static grids/occ_pos/true geometry stay fixed. Shift is fixed before
outcomes; no donor search or official test input is used. Out-of-grid shifted
observations and distribution mismatch are recorded as limitations.
Per(window,level),3 deterministic draws use seed42/window seed +1000003*draw.
A/B/C share one noisy tensor, known-empty trajectory, sampled object query and
HOI pair. HSI budget68*8*3*3*2=9792 forwards; HOI pairs3264 forwards shared across
views. Additional independent lambda0 short edit at each source uses8 levels,
1088 HOI forwards and0 HSI, with outputs confined to diagnostics. Source replay
cost is24 native HOI development generations; six GPU lanes0..5, one scene each,
batch1. No full469, expert training, new seed or parameter selection.

**Registered diagnostic.** run_fixed_source_views in the editor's calibrate path
returns raw source exactly. Record raw x0/epsilon, alpha/sigma weighted direction,
unscaled/scaled VJP, six individual explicit gradients, HOI and total gradients,
group norms using existing body maps, dot/cosine and three-draw stability. Zero
norm cosine is null with reason. Fixed ray probes target1mm and5mm equal-weight
human/object future RMS: scalar bracket from1 with up to20 doublings then20
bisections, maximum scalar2**20. Keep the original ray through tanh decode;
record attained RMS, domain admissibility and saturation/weak-direction failure,
never replace a blocked direction by projection. Evaluate every probe including
inadmissible ones descriptively, and report feasible subsets with their counts
alongside all windows. Save source terms/domain/grasp activity, per-view grid
changes, world query centers/goals, metrics, realized calls/timing and selected
motion tensors (first window of every task, first level/draw, both scales).

Report all windows grouped by scene role/object/level, collision type, grasp,
source domain and overlay coverage. B-A and B-C comparisons first average draws
and levels within window, then windows within task; retain task and scene means
and95% paired uncertainty (10000 seed42 replicates). Primary diagnostic is HS
proxy change at5mm, with OS/contact/stance/endpoint,1mm and gradient alignment
fully reported. Evidence supporting H1 requires useful B versus A and C HS
changes at equal attainable displacement, without increased OS/contact damage;
otherwise classify not_supported or inconclusive. No native quality claim:
these development scenes lack the native mesh-SDF bundle, so H3 remains open.
A/B difference alone is input sensitivity, not useful transfer. Zero-size groups
and reverse responses remain in the report. Lambda0 short edits are the common
geometric reference; no diagnostic probe enters actual history.

**Verification and completion.** Exact legacy/default, static-base output,
source identity, world history, ambient CPU/CUDA RNG, backing scene storage,
query-layout/overlap and batch-contract tests; full authority suite and registry.
The first registered real task provides runtime functional/performance evidence
at the actual batch1; do not add a separate smoke or performance workload.
Archive six resolved jobs and machine/input references before clean-worktree
experiment.py start. Persistent host-owned lanes retain every independent exit.
Stop on correctness failures; retain operational failures without reuse. Complete
only2.13 with compact result and PHASE_2M_DP_EVIDENCE_VIEWS.md, then integrate/tag
exp/p2m-dp-evidence-views-v1. Scientific failure completes this diagnostic and
requires a separate next-phase decision; no automatic development rollout follows.

Implementation verification:955 passed/4 existing historical-asset skips in173.63s;
23 targeted scene-view/evidence cases pass. Six exact jobs resolve with lambda26,
CG off, gate0 and unchanged editor recipe; registry363 records validates. Runtime
source/context replay, actual R2 static-base equivalence and batch1 timings are
measured by the registered workload. Passive records carry all ray probes and
same-source lambda0 short edits; no probe changes actual history.

### Phase2.13 operational failure and same-protocol retry

The first run failed on all six first windows after queries/probes, while saving
optional local_bps=None with tensor.detach(). All six exits and failed manifest
remain at p2-mixer-evidence-views-s42-20260907; no episode was saved. Fix the
serialization to preserve None and extend the passive end-to-end test through
motion persistence. Rerun exactly the registered24 tasks/68 windows as
p2-mixer-evidence-views-r1-s42-20260907. First attempt adds six attempted windows
(864 HSI and384 HOI teacher forwards) of operational cost; these are outside the
9792/4352 complete retry budget. Settings and scientific selection remain fixed.

### Phase2.13 aggregate serialization failure and artifact-only recovery

The r1 campaign saved all24 tasks/68 windows, then all six final metrics.json
writers failed on Hydra ListConfig nested in diagnostics metadata. Retain six
exit1 statuses, truncated aggregates and the failed manifest. Per-episode JSON
and motion tensors are complete. Convert the configuration at the editor input
boundary to plain containers; extend the Hydra-configured end-to-end test through
json.dumps(audit_dict). This metadata-only fix preserves executed tensors and
adds no generation/probe workload. Recover statistics under the fresh
p2-mixer-evidence-views-analysis-s42-20260907 manifest, referencing r1 outputs;
retain all original failures. Bootstrap uses GPU7 with the existing pairing
rules and NumPy seed42 index plan; original source replay uses GPU0..5. All
source/legacy/static/RNG/coverage checks precede scientific interpretation.

### Phase2.13 completion — environment view hypothesis not supported

Artifact recovery validates all24 tasks/68 sources and4896 paired view records;
9792 HSI and4352 HOI teacher forwards match the registered complete budget.
B-A5mm HS delta is+0.000420cm (task95% CI[-0.000380,+0.001361]); OS and contact
are also unresolved. A/B direction median cosine is0.9966; B noise median0.9961.
Spatial mismatch changes direction but yields no resolved useful B advantage.
All probes attain the target; B is domain-admissible in949/1632 queries. Preserve
both serialization failures and the independent successful recovery. Engineering
is complete; H1 is not supported, H2/H3 remain inconclusive, and no development
rollout is promoted. See PHASE_2M_DP_EVIDENCE_VIEWS.md and the compact result.


## 2026-09-07 — Phase 2.14a relation-compatible DP, fixed-source gate

User approved direct execution of the latest handoff (actual path:
/data/yujinlun/report/PriorHOSI_Phase2_14_Relation_Compatible_DP_Codex_Plan.md).
Base exp/p2m-dp-evidence-views-v1 / 053eb13; branch phase/02n-relation-compatible-dp.
Split before implementation: 2.14a delivers the complete local mechanism gate,
replay, evaluation-asset audit and summary; 2.14b is conditional development
closed-loop rollout in a subsequent session, and full469 requires its own frozen
candidate gate. Close only 2.14a here. Prior negative findings remain unchanged.

Hypothesis: projecting only the unscaled DP parameter gradient into the active
source hand-anchor residual Jacobian nullspace may preserve manipulation while
providing useful scene evidence beyond the same editor without HSI. Extract the
existing world-coordinate hand residual (hands22/23, source contact>0.95, future
frames2:). Keep source anchors/masks fixed. Euclidean projection in unchanged
67D tanh parameter coordinates, no damping. Use per-frame6x67 Jacobian blocks
with six VJPs; verify no temporal dependencies against full-window derivatives.
Float32 decode/autograd, float64 SVD and solve, relative rank tolerance1e-6,
absolute tolerance1e-10; validate these numerical constants before performance
runs, never tune on scene outcomes. Cast projected directions back to float32.
Record numerical rank, active rows, free dimensions, residuals, norms and timings.
Zero/no-active/rank-zero directions have explicit reasons; nonfinite/solve errors
are technical failures. Recompute Jacobian at every nonzero editor refresh.

Only DP changes to a frozen parameter-linear proxy; G1 uses the same proxy with
identity projection. HOI reference retains its original motion-space proxy;
six explicit terms, proximal term, Armijo and domain guards are unchanged.
Default production uses the old proxy and legacy view. G0 skips HSI and must
match the saved Phase2.13 lambda0 short parameters bitwise on all68 windows.
No frozen core, expert, native metric, planner or source-generation changes.
A reusable replay dispatch in the existing evaluator consumes saved raw sources,
contexts, offsets and HOI arguments; no new experiment runner or HOI generation.

Data: all24 existing development tasks /68 saved windows from
p2-mixer-evidence-views-r1-s42-20260907. Keep calibration004/006/055 and
verification023/037/036 roles; four objects; same P15 online ArmB500/R2 EMA,
lambda26, beta1, eight levels300/264/229/193/157/121/86/50 and seed42 protocol.
G0 lambda0; G1 correct environment B identity; G2 B projected; G3 projected C
(+2m window-local X temporal view only). No static/world/context change.

A: DP-only rays G1/G2/G3 at source,3 draws/level with existing1000003 draw offset,
shared noisy tensors, known-empty trajectory, query stream and HOI pair. Match
1mm/5mm equal-weight human/object future RMS. Preserve existing bracket limit
2**20 and20 bisections. If projected norm/raw norm<1e-3, retain the direction in
logs but mark probes weak and return the source instead of amplifying it. All
failures/inadmissible rays remain in the full table. Common-feasible subsets
require all3 arms to attain amplitude and pass domain guards for a query/scale.
B: G0/G1/G2/G3 short editors each use the same8-refresh budget and draw0 seed;
queries are paired by refresh but candidates evolve separately. Report actual
RMS, all terms, explicit/HOI/DP norms and dots, group changes, decisions/timings.
Save all first-level/draw ray states and all final short states. Preselected
illustrations: first windows036/clothesstand and006/suitcase; failure illustration
is the task with largest task-mean G2-G0 short HS increase (ordinal breaks ties).
These are local edits and do not enter rollout history.
Budget:6528 HSI+3264 HOI source forwards;3264 HSI+4352 HOI short-edit forwards.
Total9792 HSI/7616 HOI,0 source generation. Six persistent GPU lanes0..5 by scene;
GPU7 float64 paired resampling. Real batch1 replay supplies functional and runtime
measurements; no additional smoke/performance run. Full authority suite required.

**Frozen decision rules (voxel/FK cm, never native HS/OS).** Average draws/levels
per window, windows per task; task means primary, six-scene means sensitivity.
10000 paired seed42 bootstrap replicates, nominal95% intervals, all metrics and
all comparisons published. This is a conjunctive development screen, not a
native benchmark claim. Technical failure/missing pairing => BLOCKED, preserve
all failure records. Otherwise GO requires every following condition:
1. G2 5mm ray task-mean contact drift <=20% G1, paired upper CI<0;
   >=95% of projected G2/G3 rays attain amplitude and >=95% retain norm ratio>=1e-3.
2. Primary short-edit HS: G2-G0 and G2-G3 each <=-max(0.01cm,5% comparator mean),
   task95% upper<0 and scene mean difference<=0. Absolute0.01cm rule is frozen
   for small denominators. G2-G1 HS and all other contrasts are reported.
3. For both short comparators G0/G3: OS point delta<=0 and upper CI<=0.02cm;
   contact and stance-increment point delta<=0.01cm and upper<=0.05cm;
   root/object endpoint point delta<=0.05cm and upper<=0.10cm.
4. G2 domain-feasible ray rate >=G1 rate-0.02; every short update obeys the
   existing domain guard, exact history/contact channels and finite outputs.
NO-GO if any scientific condition fails. No lambda/noise/step search follows.
Contact protection alone does not establish teacher utility. If GO, prepare
only a2.14b development rollout candidate/protocol (HS>=5% versus paired lambda0,
OS nondegradation plus registered native protections); no automatic full469.

Audit development surface assets and sealed InfBaGel outputs/decoder provenance
independently of scientific outcome. Use no full469 task for tuning. If assets
are missing, state the precise gap; paper Hybrid remains external unpaired.
Deliver aggregate JSON, all local per-window/paired inputs, animations, one-page
GO/NO-GO/BLOCKED summary PHASE_2N_RELATION_COMPATIBLE_DP.md, then integrate/tag
exp/p2n-relation-compatible-dp-v1 after the engineering gate. Negative science
closes the diagnostic. Suggested conditional-denoising repair stays a separate
unvalidated future mechanism, never implemented alongside this test.


Implementation verification before the reportable replay: initial full authority
suite965 passed/4 historical skips in174.58s; the added GPU paired-bootstrap check
matches the established NumPy seed42 plan to1e-12. Numerical tests fix SVD rank
rtol1e-6/atol1e-10 without changing the preregistration. Float32 common-motion
projection differences are bounded by1e-5 in parameter coordinates; direction
Jacobian finite differences use atol1e-4/rtol.005 and float64 nullspace residual
checks. Initial overly tight float32 test assertions (3e-6) exposed rounding at
3.10e-6 and common-motion components9.06e-6; the solver constants were unchanged.
These are pre-run numerical implementation checks, not omitted experiments.
The reusable saved-context replay is a dispatch in test_infbagel_hosi.py;
no HOI sampling occurs. Statistical summaries live in scene_calibration.py,
using established paired-bootstrap name/metric discovery and GPU float64 draws.

Evaluation audit: all six raw LINGO mesh_low.obj files exist under
/data/yujinlun/datasets/LINGO/Scene_mesh, distinct from the absent matching native
Scene_sdf bundle. Record topology/bounds in evaluation_assets.json; do not infer
signed native penetration from occupancy or unvalidated open surfaces. The sealed
July InfBaGel evaluator06086f4 has no save_motion_params branch or serialization
call despite its resolved flag=true. Its469 per-task metrics survive; no sealed
motion files exist in that run directory. Historical world transforms/scale3/IK
and pre-P12 conventions are readable in Git, but the requested motion-based
comparability check is blocked by missing sealed trajectories. Keep the July row
historical and paper Hybrid external; do not reload it into the repaired codec.

Final implementation suite:966 passed/4 historical skips in174.41s; all six
resolved configs differ from sealed views jobs only in output/run identity,
saved-replay source and registered diagnostic selector/weak-direction threshold.
Frozen core is unchanged from053eb13. No additional numerical/performance tuning.


### Phase2.14a completion — relation protection works; direct DP route NO-GO

All24 tasks/68 saved sources complete; all6 lanes exit0. Raw sources/references,
contact masks and lambda0 short parameters match bitwise68/68.9792 HSI/7616 HOI
forwards and9792 matched physical probes meet the budget; no source generation.
At5mm, G1/G2 contact drift0.924533/0.003768cm (99.59% reduction), with mean
r_keep0.820624 and max normalized Jv2.65e-9. Same8-step G2-G0 HS+0.000035cm,
95% task CI[-0.003099,+0.003635]; G2-G3 HS+0.000872[-0.002718,+0.004926].
Both fail the registered5%/significance gate. G2-G0 stance increment+0.021480cm
[+0.007859,+0.036833] also fails its point margin; G2-G3 OS point is positive.
Projection protects contact but supplies no demonstrated useful HSI scene gain.
Keep all favorable contact/stance contrasts and common-feasible subsets in the
full report.966 tests pass/4 historical skips;368 registry rows validate.

Close2.14a as engineering complete/scientific NO-GO at
exp/p2n-relation-compatible-dp-v1. No2.14b rollout candidate or full469 is promoted.
See PHASE_2N_RELATION_COMPATIBLE_DP.md and the compact result. Raw development
meshes exist (5 closed/006 open), while the validated native signed surface bundle
is missing. July InfBaGel469 task identities match, but its sealed evaluator
never serialized the trajectories needed for unified motion re-evaluation.
Any conditional-denoising-repair mechanism is a separately registered next
proposal; it remains unvalidated and is not executed in this session.

## 2026-09-07 — Phase 2.15 scene-prior conditional repair

User authorizes implementing the latest scene_prior_transfer handoff (actual file
/data/yujinlun/report/PriorHOSI_scene_prior_transfer_codex_plan.md), without further
approval. Base44a33a6; branch phase/02o-scene-prior-transfer. One candidate, frozen
P15 online ArmB500 and R2 EMA; no expert/core edits or old DP searches. This phase
includes implementation,24-task complete development rollouts, and conditional
full469 only if the frozen development conjunction below passes. Close once at
the development NO-GO or completed full469. No external baseline/asset engineering.

B0 is native HOI; B1 existing eight-refresh lambda0 geometry followed by the same
constrained pose fitter with proposal target; B2 substitutes full-condition HSI
clean targets. Development B0 must be newly evaluated with the same native motion
decoder; historical proxy-only development files cannot supply native FS. Full469
B0 may reuse p2-hosi-hoi-alone-g0-p15-guided-armb-s42-20260829 if unchanged.
Use existing24-task manifest scene_evidence_development_s42_20260907.json (six
scenes/four objects; previously used development, not untouched evaluation).
Every method generates its own future from actual submitted history. Independent
HOI episode/window seeds follow existing per-episode scheme; additional HSI noise
and CPU occupancy subsampling use private generators. Preserve all tasks/failures.

Configuration: canonical500-step linear beta[0.0001,0.02], x0 prediction;
existing DDIMSolver(500,25), start99, timesteps[99,79,59,39,19], eta0; finish with
clean x0 through solver's alpha_previous=1. KnownEmptyObjectView; original occupied
scene view; full condition only, CFG0, no cond-uncond query or posterior guidance.
Both geometry query states are the latest accepted clean candidate, with actual
object retained. q_sample initializes proposal; scheduler updates the full state
using constrained x0, so future locks remain at the appropriate noise level.
Each of five targets receives4 projected-gradient pose-fit iterations with local
Armijo(initial step0.25, shrink0.5,10 trials,c1=1e-4). Loss: summed local-rotation
chord distance/(2*10deg^2) +0.25/2 summed squared tanh residuals. The21 local
rotations use inherited10deg/component bounds. Translation/yaw parameters stay0;
root rotation, position, object and contact channels copy proposal exactly.
Source anchors/contact>0.95 remain from raw HOI, never proposal-reanchored.
Six-VJP hand Jacobian (common columns zero) + existing thin float64 SVD projects
the fitting direction; actual finite FK guards decide acceptance. No density-ratio
or DP-gradient interpretation. B1 executes the identical5x4 fit budget targeting
proposal and consumes no HSI RNG; exact zero gradient is a recorded stop.

Hard repair guards against fixed proposal: history/root/object/contact exact;
source active hand distance each <= max(proposal distance,0.005m)+0.001m;
HS voxel RMS <= proposal+0.01cm; stance correction RMS <=0.05cm;
foot displacement per frame <=2cm; no increased scene-domain violation per point.
Source anchors with proposal drift>5cm mark invalid proposal/task failure. Existing
outside-domain proposals stay visible and obey inherited nonexpansion rule, never
called collision-free. Nonfinite predictions/fit failures retain last feasible
state and explicit reason. Record every solver/teacher step, actual modification,
accepted/fallback windows, timing and complete source/proposal/submitted motions.

Development evaluator reuses native scale3/IK/SMPL-X and FS/contact/object-SDF/
endpoint/completion functions. Missing scene SDF gives explicit null native HS/OS;
complete-sequence FK/voxel proxies separately retain window protection scope.
SMPL-X/FK reconstruction agreement must be measured before native interpretation;
if incompatible, record unavailable native metrics and block promotion without
building a new asset system. No metric renaming. All15 native fields retained.

Frozen development gate (all conditions): no missing/failed tasks; immutable
history and common-motion checks all pass; >=50% B2 windows differ from proposal
by>=1mm future FK RMS, <=50% windows fall back unchanged. B2-B1 mean native FS
improvement >=0.10 native units (existing FS reports cm); nominal95% task-paired
upper CI<0 and scene mean delta<=0. Absolute0.10cm threshold exceeds rounding and
small decimal-only movement; no borrowed5%/10% improvement rule. Task/scene
bootstrap10000 seed42; all comparisons/metrics reported, never frames as samples.
Against BOTH B0/B1, contact/completion nominal95% lower bounds>=-0.02 at task/scene
units (inherited2.12), endpoint upper deltas<=1cm. Against B1 HS/OS proxy upper
<=0.02cm and point<=0; against B0 both scene proxies must have point<=0 and at least
one upper CI<0. Geometry gain retention and FS are read with engagement, root
travel, joint motion and boundary jumps. Preselected full-sequence videos tasks
007(006/suitcase),012(036/clothesstand), and all-task seam/displacement tables
must show no patterned freezing/release/collision regression. Native scene limits
are stated explicitly; development passes authorize full469 native confirmation.

Only one interface correction is permitted if a measured implementation defect
appears; retain failed manifests, use a new run id and identical scientific values.
No optional low-noise search is scheduled. If science fails, stop this candidate.
If passed, freeze and execute B1/B2 full469; use native HS/OS/contact/completion,
FS, all endpoints and paired task/scene uncertainties; no test-based changes.
Full469 HSI success requires same FS practical/statistical criterion and inherited
contact/completion protection, native HS/OS nonpositive point and upper<=0.02
native units versus B1, total scene benefit vs B0 upper<0 for at least one HS/OS
with other nonpositive point. Preserve all unfavorable contrasts and failure rows.

Use tools/experiment.py lifecycle and existing evaluator/bootstrap entry points;
no new experiment script. Full authority tests and registry validation required.
Actual complete development runs provide functional/performance evidence; user
prohibits separate smoke tests. GPUs0..5 scene lanes; GPU7 paired statistics.

### 2026-09-07 user scope correction before any reportable run

The user explicitly directs development to data/hosi_test, with the final aim of
exceeding InfBaGel on that benchmark. Replace the above24 LINGO development tasks
with the already-exploratory28 native benchmark tasks: sorted scene bins0,22,44,66,
all7 objects each. These bins are the same fixed subset used in prior diagnostics;
no outcome-based subset selection. The supplied InfBaGel-release/data/hosi_test/
Scene_sdf is the SAME directory as our linked native asset;67/67 benchmark scenes
have SDF,0/6 LINGO development scenes match. Use the unchanged complete native
motion/scene evaluator. No development asset engineering or partial-metric branch
is needed. The interrupted pre-run full suite and initial unit failures remain
implementation records, not experimental failures.

Keep the five-step candidate and physical guards unchanged. Development primary
native FS delta<=-0.10cm, task95% upper<0, scene delta<=0. Replace only the scene
proxy promotion endpoints by native scene_human_penetration_s_mean and
scene_obj_penetration_s_mean: against B1 each point<=0 and95% upper<=0.02 native
units; against B0 both point<=0 and at least one upper<0, at both task and scene
units. Contact/completion lower>=-.02 and endpoint upper<=1cm remain. The same
>=50%1mm modification and<=50%unchanged fallback rule applies, with all history/
common-motion guards and no task failure. Report proxies only as mechanism logs.
Four GPU scene lanes0..3; GPU7 statistics. All28 tasks receive B0/B1/B2 full
rollouts. Record fixed manifest of actual UUID scenes and canonical ordinals.
Per-episode seeding=true isolates HOI streams from differing previous episode
lengths; therefore the historical per_episode_seeding=false B0 cannot supply this
campaign's full469 paired baseline: regenerate B0/B1/B2 if promoted. Full469 then
contains the28 development-used tasks; explicitly report test-set development,
never untouched-test selection. Historical InfBaGel/paper values are external
reference rows with their original protocol and representation labels. This
scope correction does not authorize loading released weights into repaired code.

Implementation clarification before native workloads: the four indices refer to
existing `select_shard_scenes(...,67)` bins, whose exact UUIDs and28 canonical task
keys are tracked in experiments/tasks/conditional_repair_native_development_s42_20260907.json.
Replace the previous LINGO video identities with test_idx0 in each of these four
fixed native scenes (four full-sequence B0/B1/B2 videos). Reuse native interpolation,
SMPL-X and all15 metrics unchanged; add only task_failed for invalid proposals.
The reportable evaluator now applies its existing clean-worktree check to run_id
native runs as well. DDIMSolver previous-alpha buffers are float64; cast its result
back to the current float32 state before the next frozen expert call. Initial unit
failures identified that interface dtype and an invalid empty-budget test fixture;
all7 repair tests then pass without scientific parameter changes. The partial
LINGO metric branch was removed after the user's scope correction.

Pre-run scale correction (no candidate outcomes exist): sealed2.12 native FS is
0.1208 for lambda0. The provisional0.10 absolute criterion would demand82.78%
improvement and is inconsistent with a local coordination repair under strict
stance protection. Freeze0.01 native FS units instead (8.28% of that historical
scale), with the same negative paired95% upper CI and scene mean protection.
This supersedes0.10 in both development/full469 FS rules only. This is based on
sealed historical evidence, never the new28/469 outcomes; all other gates stay.

### Registered interface correction after first complete native attempt

r1 completes all84 episodes/372 windows with no nonfinite values or invalid
proposals. B2 makes620 HSI calls but accepts0/2480 fit steps (0/24800 trials),
returning B1 exactly. A post-run isolated implementation check on a zero-common-
column6x67 Jacobian and zero-common input produces56/56 nonzero common output
entries, max2.5213e-16, from float64 SVD roundoff. The common-motion guard requires
exact zero; including fixed columns in the SVD violates that representation
contract. Retain all r1 outcomes and this diagnosis; do not interpret r1 as evidence
that HSI targets are useless.

Use the handoff's ONE interface correction: SVD only the6x63 free-local Jacobian,
then concatenate exact common/history zeros. This is the intended constrained
parameter space, with unchanged solver, rank thresholds, schedule, objective,
step budget and hard/quality gates. Add per-trial guard reasons to the existing
trace for actual rejection attribution. No threshold relaxation or low-noise
search. Component test must expose the zero-column roundoff and verify exact
common/history locks for the corrected free-coordinate solve.

Rerun only B2 on all28 tasks using a new r2 manifest. Reuse r1 B0/B1 by reference:
B0 does not call this solver; all2480 B1 fit gradients were exactly zero and every
B1 output equalled its geometric proposal, so solving a zero63-vector preserves
its exact old output and RNG. Verify this from saved records and component tests.
The startup failure is operational orchestration, not the one model-interface
correction. A remaining scientific failure after this correction stops the
candidate. The r1 videos remain retained as identical B1/B2 evidence; render the
same four preselected task identities for r2.

### Phase2.15 completion — corrected native28 candidate NO-GO (2026-09-07)

The corrected free-coordinate B2 completes28 native tasks/124 windows, paired
with compatible r1 B0/B1. All4 corrected jobs exit0; all84 comparison episodes
have complete15-field native evaluation. Final suite974 passed/4 skipped167.34s.
HSI620 queries,331/2480 accepted fit steps,122/124 nonzero windows but only15
reach1mm; task-balanced RMS0.6283mm. There are24219 trials:331 admissible/accepted;
22828 rejections include stance,14727 fail stance alone; common/finite failures0.

B2-B1 FS=-0.001207, task95% CI[-0.005108,+0.001331]; HS=-0.093245
[-0.290010,+0.011178]; OS=-0.575203[-1.857078,+0.094102]. Scene-unit intervals
also cross zero. Repair-amplitude and FS gates fail, as does OS uncertainty
protection. Contact/completion/endpoint protections pass; all groups complete
identical22/28 tasks. B2-B0 total FS/HS/OS improvements pass task and scene checks,
but do not establish extra HSI efficacy. No full469 promotion or further search.

The failed initial launcher and r1 all-rejection interface result remain intact.
Only the registered interface correction was used; r1 B0/B1 reuse has exact
zero-gradient/output proof. Across r1/r2 all28 first sources/proposals/HSI fit
losses and124 seeds match; all96 later sources differ, verifying actual submitted
history propagation. No core/expert or native metric-definition changes.

Native469 SDF assets are complete; this phase needed no LINGO asset work after the
user's scope correction. Historical/paper InfBaGel values remain explicitly
protocol-mismatched reference rows. Four preselected complete videos and all
native/proxy metrics, object strata, failure traces and paired inputs are saved.
Summary: PHASE_2O_SCENE_PRIOR_TRANSFER.md; compact result:
experiments/results/p2_mixer_scene_prior_transfer_s42_20260907.json.
Seal engineering/development closure at exp/p2o-scene-prior-transfer-v1 and retain
unresolved scene-prior transfer quality. The next entry is review of the measured
pose-target/stance-constraint compatibility under the same native benchmark goal.


## 2026-09-07 — Phase2.16 quality-preserving foot guard (authorized)

User handoff: /data/yujinlun/report/PriorHOSI_next_experiment_Codex_af8f0d1.md.
Base af8f0d1; branch phase/02p-foot-quality-guard. One candidate, foot-quality-guard-v1.
Phase2.15 and its NO-GO remain sealed. This phase delivers native28 and its
quality decision. A successful candidate permits separately numbered2.17 static
condition ablation/441 confirmation, preregistered before those workloads; close
one phase per session. No expert training, new assets, root/object unlock or solver
search. Native data/hosi_test remains development-used (test_set_development=true).

Only replace the repair-local stance acceptance predicate, with explicit
foot_guard_mode=quality and foot_energy_epsilon_m2=1e-12. Legacy default remains
increment with RMS<=.05cm. Keep the geometry proposal, objectives, all other
constraints, P15/R2 EMA, ArmB500, five DDIM levels, seeds and actual-history
rollout unchanged. Do not AND the old predicate with the new one.

Reference mask is protection.stance, built from the current fixed proposal's
source-floor height: joints7/8 heights<.08m,10/11<.04m, both ends of each frame
pair active. Y is vertical, X/Z horizontal in world metres. Pairs are [1->2,...,
14->15], including the history/future boundary. Cache proposal energy once.
Compute squared X/Z displacement summed over2 axes then average over active
frame-foot pairs, in float64 from the same float32 FK coordinates. Empty masks
have energy0, count0 and a vacuous quality predicate; they convey no improvement
evidence. Nonfinite human FK remains rejected by the existing finite guard.
Epsilon1e-12m² is a numerical allowance (1micrometre RMS at a stationary
reference), not a physical relaxation. Identity must be exact; verify repeated
and reordered float64 reductions are within epsilon before native runs. No
outcome-based epsilon selection. Preserve legacy increment diagnostics and log
both predicates, proposal/candidate energies, difference, active count, other
rejections and accepted steps that legacy would disallow.

Fixed protocol before candidate outcomes:
- Reuse the exact native28 manifest from2.15, scene shards0/22/44/66, seven
  objects each. New B2_quality gets28 complete episodes. Reuse compatible B0/B1
  and corrected r2 B2_hsi_repair by reference after output/RNG regression.
- Primary: B2_quality minus B1_no_hsi, native
  scene_human_penetration_s_mean. Require >=5% reduction of the sealed B1
  mean4.104416241811123 (absolute reduction>=0.2052208120905562), task paired
  95% upper<0, scene mean delta<=0. Five percent is the declared minimum useful
  scene gain; old r2's2.27% gain is insufficient. No post-result metric choice.
- OS protection versus B1: point<=0 and95% upper<=.02 native s_mean, retaining
  the old margin for both task and scene units. Total HS/OS versus B0: both
  points<=0 and at least one upper<0, also both units.
- FS is now a protection metric:95% upper<=+.01 native cm versus B0 and B1
  for task and scene units. This explicitly replaces the previous primary
  FS improvement requirement for this new candidate only; .01cm is the same
  previous practical-effect scale, now a maximum tolerated deterioration.
- Retain contact/completion95% lower>=-.02 and root/object endpoint95%
  upper<=1cm versus B0/B1, task and scene units. Complete28 pairs, all hard
  final/history guards pass, no invalid proposals/nonfinite steps/task failures,
  world history discrepancy<=1e-5m. Repair amplitude/1mm coverage/fallback are
  diagnostics without quality gates. Report all original15 metrics.
- Comparisons: new-B1 primary, new-old mechanism, new-B0 total; also retain
  baseline/old contrasts. Paired10000 seed42 bootstrap, float64 CUDA7; task
  units plus four scene means, no frame/window pseudo-replication. Report all
  seven object strata and per-task deltas. Nominal95%, no cross-seed claim.
- All operational/scientific failures and missing tasks retained. Missing or
  nonfinite metrics block promotion; no complete-case pruning. Invalid proposals
  and fallback remain in denominators; native task_failed remains visible.
- Visualize preselected tasks014/329/371/420 in full, all four arms; add the
  largest new-minus-B1 HS worsening and any newly failed completion case.
  Review foot lift/penetration, hand loss, jitter, freezing and seams. Numerical
  passage also requires visual review; proxy improvement alone never promotes.

Tests cover identity, cancellation vs old rejection, worsening, frozen mask,
empty/nonfinite, metres/squared reduction/boundary, other guards, legacy and
B1 output/RNG. Full authority suite and registry validation required. Small
real-data interface checks use sealed inputs and the initial complete native
scene lane before remaining lanes; no separate unrequested smoke suite. Record
synchronized native editor timing/peak memory, explicitly sharded timing; no
training microbatch benchmark applies to batch1 inference. Preserve launchers
only in ignored run artifacts, use experiment.py start and archived resolved
Hydra configs on a clean committed worktree. Use screen for persistent workloads.

Stop with insufficient evidence if primary practical/statistical gain or any
protection fails. Do not relax more guards, change targets or run441 on failure.


Implementation verification before native outcomes:15 component tests pass.
All124 sealed B1 proposals equal outputs and all2480 fit gradients are exactly
zero; new-mode no-HSI/RNG regression passes. Float64 foot-energy identity delta
is exactly0 and maximum reverse-reduction discrepancy on those proposals is
8.673617379884035e-19m². Proposal energies range0..0.008176423223049857m²,
active counts0..56. Epsilon1e-12 is retained as registered. Four resolved configs
match sealed r2 after removing only run/output identity and the two new foot
options. No new sampling/noise/objective/checkpoint differences. The first full
scene lane supplies the requested real-data interface check, contributes its
seven actual episodes to native28, and must pass interface/finite/lock checks
before other scenes launch. It is not used to tune scientific settings.

Final implementation suite:981 passed,4 historical skips,169.63s. Registry374
rows valid; core and both expert trees unchanged. Component tests, baseline
numerical audit and full suite logs live in results/foot-quality-preflight-s42-20260907/.


### Phase2.16 completion — foot-quality guard NO-GO (2026-09-07)

Runtime ea161ca completes28 new native tasks/124 windows; all4 jobs exit0,
620 HSI calls,411 accepted fit steps,379 accepted steps that the legacy guard
would reject.88/124 windows reach1mm; task-balanced RMS2.7991mm;30 exact fallbacks.
All fixed/common/history and finite checks pass. Primary HS improves1.88%
(vs registered5%), delta−0.077019, task95%[-0.242772,+0.024704]. FS delta+0.009536
[-0.013094,+0.043544] fails uncertainty protection; OS delta−1.021186
[-3.727114,+0.519325] also fails its uncertainty protection. Completion21/28
vs22/28 loses task420 (object endpoint10.04149cm), failing2pp protection.
Contact/endpoints and total scene gains over B0 pass. Scene-unit checks agree.

The dominant rejection reason is now2cm foot displacement (16191 trials;
10850 sole-reason), followed by support energy9584. No further guard relaxation.
Task372 native FS worsening is highly sensitive to its candidate-specific floor
estimate; read-only counterfactual localization retains the original metric,
all tasks and NO-GO. Five complete videos cover the four preregistered examples,
worst HS/new completion failures and supplemental largest-FS case.

981 tests passed/4 historical skips169.63s, full native coverage, no operational
failure or rerun. Engineering closure is satisfied by the verified implementation
and complete positive/negative evidence; scientific promotion fails. No static
condition ablation,441 or469. Seal exp/p2p-foot-quality-guard-v1 and integrate
closure into phase/02-mixer. Detail/next entry: PHASE_2P_FOOT_QUALITY_GUARD.md;
compact result experiments/results/p2_mixer_foot_quality_guard_s42_20260907.json.

Closure bookkeeping: the hypothesis occupied the canonical run id, so the
register command refused a duplicate. Preserve it append-only and record a
unique completion event linked to the same completed manifest/run_id. Registry375
rows validates. No sampling rerun, result overwrite or scientific change.


## 2026-09-07 — Phase2.17 source-dynamics preservation (authorized)

Handoff: /data/yujinlun/report/PriorHOSI_next_experiment_Codex_90ea106_HTD_temporal.md
(the supplied papers path is absent). Base90ea106, clean worktree. Branch
phase/02q-temporal-preservation. User authorizes implementation/execution/publication.
H-T tests whether preserving same-window raw HOI dynamics protects edit quality;
H-S requires additional HSI utility under identical Temporal treatment. Neither
source error nor jitter alone establishes naturalness or HSI transfer.

Split before implementation:2.17 covers A0 asset/time recovery, component and full
suite, one frozen-source scale calibration and A1/A2 fixed-window repair/decision.
Conditional2.18 covers G1/H1 native28 closed rollouts, at most56 new episodes;
conditional2.19 covers additional441/full469. Close one subphase per session.
2.15/2.16 NO-GO remains sealed. No expert/evaluator/core changes, candidates,
training, new hash machinery, unrequested smoke suite or new tracked runner.

Inputs: exact committed native28 manifest, all124 B1 and124 B2-quality windows.
Each uses its own raw_source, original proposal, incoming edited parameters and
source anchors. Existing caches lack replay_context/rest_offsets; recover from
native task metadata and saved world motions with explicit replay tolerances;
never silently substitute another arm's history. Recover original frame transform
using evaluator formula and kinematic assets; require FK <=1e-5m and object/world
history consistency, exact original saved output for bypass. Record every missing
asset or mismatch. Native source30Hz, stride3 => dt=.1s, with original2-frame
history; verify dataset/evaluator sampling and metadata before physical loss.
No padding is generated inside the16-frame windows; interpolation tail is excluded.

Method: default-off post-editor Temporal; root-local world-consistent FK21 nonroot
joints, V/A stencils ending t2..15. Freeze source, incoming keep center and proposal
protection independently. Explicit frame/link validity; unequal timestamps rejected.
Scales: sqrt(task-balanced mean per-window mean square source derivative per
coordinate), pooled equally across G0/H0; minimum sV=.001m/s,sA=.01m/s².
Freeze before candidate outcomes; lambdaV=lambdaA=1, keep chordal SO3 squared
Frobenius/(2*(10deg)^2), mean over future frames/nonroot joints. Keep minimum
and derivative normalization explicit; no weight search. Same20 iterations,
initial step.25,10 backtracks,Armijo1e-4,halving; actual projected direction
slope uses original gradient dot direction. Keep63 free coordinates, original
proposal10deg component range,2cm feet,quality support1e-12m²,domain,contact,
HS and finite guards. Invalid input/no stencil => named no-op. Off/zero weights
return incoming tensors directly with no FK/random call. Loss always differentiable
at zero under enable_grad; old fit/solver paths unchanged.

A2 gate (fixed before new results): both arms task-balanced source V/A sum falls,
nonzero accepted changes, all hard/finite/history checks pass. Independent evidence
must be supplied by contact anchor error, world support slip, or reviewed motion;
optimized source error/boundary source error alone is insufficient. Quantitative
support: >=5% task-balanced reduction in mean active source-anchor distance or
fixed proposal-support horizontal speed, task95% upper<0,scene delta<=0 in at
least one arm, no significant opposite effect in the other. These are diagnostics
of geometry/feet, not native metrics. Require mean human scene voxel residual
increase<=.01cm in each arm and mean occupied fraction not higher; fixed source
floor toe-height rise<=.1cm; joint-speed retention>=95% of incoming, root/object
exactly fixed. All28 tasks remain in denominators; empty contact/support counts
are recorded and contribute no evidence. Visual review must not reveal loss of
contact, new freeze/lift or seam damage. If unavailable mark visual_review_pending
and withhold promotion. No editing quota. Preserve any failure and stop with
insufficient evidence if only directly optimized quantities improve.

Task-paired10000 seed42 bootstrap and four-scene sensitivity, CUDA float64.
Report G1-G0 andH1-H0 independently; offlineH1-G1 is descriptive because sources
and histories differ. No stitched replay success/native15 claim. All28 tasks get
fixed-view replay clips and V/A curves including372/420; independent-window
history resets explicitly labeled. No human perception score is claimed.

Prospective2.18 must register its concrete runs before execution. HSI primary
H1-G1 HS>=5% relative to G1 with task95% upper<0 and scene mean<=0. Protect FS
upper<=+.01cm,contact/completion lower>=-.02,root/object endpoint upper<=1cm,
OS point<=0 and upper<=.02 versus corresponding no-Temporal arm and H1vsG1,
both task/scene. Temporal quality needs independent native improvement with
protection of HS/OS/contact/completion, alongside source dynamics engagement.
No promotion based only on H1-B0 or source error. Remaining441 historically used,
not untouched; test_set_development=true throughout.

Engineering closure requires valid implementation/full suite, registry validation,
complete positive/negative fixed-window evidence and phase summary; scientific
promotion is separate. On a failed A2, no native28/441, no tuning guards/weights,
and no switch to candidate reranking. Publish negative results too.

A0 recovery/calibration:248/248 windows from56 cached episodes recovered without
expert calls. Source scales frozen at sV=0.2047062398122251m/s and
sA=1.6846152474941332m/s² using registered equal-arm/task/window/coordinate RMS.
Dataset stride3 and historical native30Hz establish .1s; all16 generated frames
are real coarse-grid samples and first2 are available history. Interpolated tail
padding never participates. Missing context recovered by exact recorded equations
Rworld=mat.R Rlocal and Robject_world=prefix Rrelative Rreference. Rest offsets
come from the task's data_idx/sequence mapping and repaired dataset loader.
All replay errors must remain<=1e-5 in world metres/rotation entries/encoded units;
saved original source hand anchors are restored verbatim. This is physical replay
tolerance, while disabled output returns the saved incoming tensor bitwise.
No cached motions or old results changed. Artifacts:results/temporal-preflight-s42-20260907/.

Governance correction before outcomes: initial hypothesis row at fe2ac5e omitted
created_at/config/results/conclusion and validator refused it. Add only those
metadata fields in implementation commit; original fields and original committed
row remain recoverable. This one explicit schema repair is an exception to bytewise
append-only history, not a hypothesis/gate change; future events append normally.

Recovered248-window audits before Temporal outcomes: max final/proposal FK error
4.7684e-7m, source-anchor error6.5565e-7m, object-world error4.7684e-7m,
rotation-entry error7.7486e-7, output reencoding error exactly0. All incoming
guards/contact/support masks pass.29 component tests pass, including both legacy
editor arms with zero Temporal weights and private-RNG checks. Source-time scale
calibration made no candidate optimization or new native evaluation. No training
microbatch benchmark applies; synchronized batch1 replay timing/memory will be
recorded. Full task replay serves the real-data interface execution, with no
separate smoke workload.

Final authority suite:995 passed,4 historical skips,171.96s. Registry376 rows
valid. Initial component fixture failures (strict binary decimal equality and
using a terminal head rotation that moves no selected FK joint) were corrected
in test inputs; no candidate parameters selected from outcomes. Full initial
suite989 passed before extra six zero-weight editor cases; final suite includes
all29 component cases. Runtime code/core/expert/evaluator scope verified before
committing. Loss derivative and actual-direction changes use the existing solver
with a Temporal-only option; legacy optimizer arithmetic remains unchanged.


### Phase2.17 completion — fixed-window Temporal NO-GO (2026-09-07)

Runtime b87bd8e completes248/248 cached windows across28 tasks,4 scenes; all4
jobs exit0,0 expert calls/new rollouts. G/H normalized source V+A errors fall
68.154%/53.900%,98/120 windows change; task-balanced RMS.024929/.534955mm
(window means.030018/.556178mm). All guards/finite/history/common/coverage and
scene/toe-height/motion protections pass. Independent quality gate fails:
G support-speed gain.03684% falls below5%; H speed worsens.2115% with uncertainty
crossing0; contact changes are unconfirmed. HS voxel means slightly regress by
.000306/.000544cm within.01cm protection. Direct objective gain does not establish
H-T or H-S. No2.18/441/full469,retuning or reranking follows.

995 tests pass/4 historical skips171.96s; registry377 records validates.
Controller393.29s including analysis; synchronized Temporal4.665/3.890s per
window in sharded runs, observed GPU use931–932MiB (not allocator peak).
All28 four-panel independent-window replay videos and physical curves saved.
Six task frame reviews show no clear additional quality improvement; independent
human blind review remains pending. No native15/completion scores claimed from
replay. Source/keep/proposal references, per-window seeds, times and stitching
intervals retained. Summary PHASE_2Q_TEMPORAL_PRESERVATION.md and compact result
p2_mixer_temporal_preservation_s42_20260907.json carry all comparisons/negatives.
Engineering closure passes; scientific promotion fails. Seal
exp/p2q-temporal-preservation-v1, integrate completed record into phase/02-mixer;
next entry requires new user handoff, not an automatic new phase.

Cache audit clarification:248 contact masks and124 H0 quality support masks compare
exactly. B1 lacks repair_stance_mask in all124 old snapshots; its proposal support
is reconstructed, and logged stance_exact=true is vacuous for that absent field.
No claim of an exact saved-B1-support comparison; no outcome or gate changes.


## 2026-09-07 — Phase2.20 fixed complete HOI candidate pool (authorized)

New handoff /data/yujinlun/report/PriorHOSI_dd993d7_Codex_Next_Experiment_Handoff.md
from base dd993d7; branch phase/02r-candidate-selection. This independent phase
implements A–D (protocol,112-slot pool,score/select,native28 conclusion) in one
session. It does not reopen failed Temporal2.18 or conditional2.19. User authorizes
execution and publication. No441/full469, expert/core/native evaluator edits,
training, posture edits, guard/weight/K searches or new tracked runner.

Frozen numerical protocol: experiments/protocols/p2_candidate_selection_s42_20260907.json.
All unspecified values there are this phase's implementation decisions, not
historical settings: generation offsets0/100000000/200000000/300000000, four
levels50/157/264/300 and two paired repeats, normalized84-coordinate future MSE,
10% geometric budget with fixed.006666667 dimensionless floor, fixed endpoint/
hand/foot protection, deterministic candidate-index ties and C0 common fallback.
Generation seed offset is scoped to sampler so task assets, object subsampling
and native metric RNG remain seed42; separate scoring streams. Full/static raw
heads share clean-candidate geometry and KnownEmptyObjectView. One negative
control rolls only dynamic voxel blocks+16 X cells, preserving static/anchor
blocks and coordinate embeddings and the occupied object-view convention.

CF-CG primary native HS requires>=5% reduction, task95% upper<0, scene point<=0.
OS point<=0/upper<=.02, FS upper<=.01cm, contact/completion lower>=-.02,
root/object endpoint upper<=1cm vs CG andC0 for task and scene units. Task-paired
10000 seed42 CUDA float64 bootstrap;112 candidates are not112 independent tasks.
All native15, failure/coverage/cost, CS and mismatch, one feasible oracle actual
candidate per task, seven object strata and fixed014/329/371/420 visuals retained.
Numerical pass requires visual review; otherwise review pending. Technical
failures or no multi-candidate budget set mean BLOCKED/inconclusive, not NO-GO.

Reuse sealed B1 only after input/config/seed/cached output compatibility; its
124 cached windows lack replay context, recovered from actual world transforms
and original task/path/condition equations, checked against new captured contexts.
Scorer uses final edited candidate, each window's own history, local2..15 mapped
to global14*w+2..14*w+15 (disjoint); no interpolation padding enters scores.
Hash checks requested by the handoff use existing manifest/file helpers only;
sealed checkpoint identities are referenced. No additional smoke suite; complete
first generation lane and scorer execution provide real-data interface evidence.
Full authority suite and registry validation precede reportable workloads.
Engineering closure requires full retained evidence and summary; scientific gate
is separate. Any failure preserves original outputs and allows registered
diagnostics only.

Pre-outcome implementation corrections: registry id key corrected to required
experiment_id (the initial validator rejection is retained in session log;
original b5eb005 row remains in Git history). This is a disclosed schema-only
append-only exception. Runtime inspection confirms first window adds0, not
1000003: protocol formula corrected to zero-based window_index before any
new candidate generation or scoring. Frozen offsets and all gates unchanged.

Implementation verification:17 new component tests passed; final full authority
1012 passed,4 historical skips,174.35s. An earlier full suite1012 passed173.23s
preceded GPU geometry placement and separate query/selection timing completion;
final suite covers those changes. All28 sealed B1 episodes/124 windows satisfy
identity/seed/representation and exact edited=proposal/zero repair parameters.
Resolved configs equal sealed B1 after removing only run/output identity, new
sampler target and offset/context-capture options. Native evaluator seed stays42.
Training mask source train_infbagel.get_mask(ind=-1,fixed_frame=2) and position
loss models.infbagel:1013 establish all84 future position coordinates valid.
Source checkpoint identities inherited verbatim; no full checkpoint audit added.
No training microbatch benchmark applies to batch1 inference. Runtime archives
synchronized query/geometry timing and allocator peaks; generation is sharded.
Artifacts:results/candidate-selection-preflight-s42-20260907/ and prospective
results/experiments/p2-mixer-candidate-selection-s42-20260907/. First full native
scene is an actual registered candidate lane; no separate smoke workload.


### Phase2.20 completion — fixed candidate pool NO-GO (2026-09-07)

Runtime38fcaa3 completes112/112 candidates (28 compatible B1 reused,84 new),
496 valid windows;12 generation and4 scoring jobs exit0, no technical retry.
All112 motion files immutable;372 captured contexts agree with reconstruction
within1.431e-6; legacy124 contexts recovered with previously disclosed mask
availability caveat.140 selected outputs link to exact existing candidate files.

CF-CG native HS +.00373478 (task95%[+.00003460,+.00927786]),.0935% worse;
OS +.10946594[+.02440425,+.22930376],.4027% worse; completion22/28→21/28.
HS primary and OS/completion protection fail. FS/contact/endpoints satisfy their
registered margins; contact nevertheless has a small resolved mean decline.
CF-CS HS improvement unconfirmed; correct-vs-mismatch HS improvement mostly
comes from task375.26 tasks permit multiple candidates, no empty acceptable set;
9 candidates excluded by support speed, geometry budget excludes no extra ones.
Protected oracle HS improvement overCG only.0414%, new candidates' mean human
coordinate RMS fromC0=3.633mm: limited proposal coverage and scoring utility.
Task420 CF object endpoint10.080928cm vsCG9.631516cm; registered coarse XZ
proxy6.420175cm passes while vertical error7.772160cm makes native3D fail.
This finite protection gap is retained, not revised after viewing outcomes.

1012 passed/4 historical skips174.35s; registry379 valid. Controller672.47s;
K4 cumulative generation2615.86s incl cachedC0, HSI3-branch153.26s/11904
forwards (3968 required CF,3968 each static/mismatch); no per-head time claim.
Five complete videos and40 reviewed frames show no clear quality gain;329
crouching persists. Human blind review remains pending, numerical NO-GO independent.
Full evidence and next entry: PHASE_2R_CANDIDATE_SELECTION.md; compact JSON
p2_mixer_candidate_selection_s42_20260907.json. Engineering closure passes;
scientific upgrade fails. Seal exp/p2r-candidate-selection-v1 and integrate
phase/02-mixer. Stop before441/469, tuning, post-editing or training. Await new
handoff on meaningful complete human-object proposal coverage and scene scoring.

## 2026-09-07 — Phase2.21 waypoint control and proposal attribution (authorized)

Handoff: /data/yujinlun/report/PriorHOSI_Codex_Handoff_after_ccd02dc.md.
Base ccd02dc, branch phase/02s-waypoint-control. Split before implementation:
2.21 covers A cached112-candidate attribution/three oracle levels, B actual
retained-frame native3D object endpoint protection, C paired first-window
waypoint control. Conditional2.22 covers D complete structured route pool;
conditional2.23 covers E same-pool HSI selection. Close one subphase per session;
old2.20 NO-GO remains sealed. User authorizes execution/publication.

Frozen complete protocol: experiments/protocols/p2_waypoint_control_s42_20260907.json.
28W0 first-window regenerations plus up to56 signed variants and4 exact repeats;
no new complete native rollout or HSI forward. Original waypoint must be an
intermediate path point; both signed.10m offsets require nonzero tangent, scene
bounds and free static occupancy at initial pelvis height. All28 task identities
remain visible, inapplicability is not a technical failure or substituted task.
Actual goals, clean fixed history, first model latent and progress are recorded.
BitwiseW0 against originalC0 and private-RNG matching precede interpretation.

Useful control requires >=14 applicable task pairs, both signs/raw and edited
mean directed root response>=1cm, positive response on>=75% tasks, task95% lower>0
and nonnegative per-scene means, mean object directed response>=0. Both signs
retain fixedW0 contact/feet/scene quality using registered margins and10000seed42
task-paired CUDA float64 intervals; at least50% task pairs meet per-task useful
response and quality. Contact anchor/surface1cm, contact5pp, support speed.01m/s,
human/object voxelRMS.1cm and occupied.5pp, world joint speed retention95%.
These new first-window practical thresholds are fixed decisions, not historical
native gates. Missing independent visual assessment remains pending and blocks
expansion; any response/quality failure stops route generation regardless.

CacheA reports common-world and root-local coordinateRMS at raw/proposal/edited;
first-window primary, later divergent histories descriptive. Ratio denominators
<1micrometre are undefined. Three oracle levels remain separately named and
never enter method selection. No additional ArmB-off diagnostic this phase.
EndpointB uses actual retained output tracks, humanY0 and full3D object, both
strict<.10m;112 sealed metrics only validate motion-derived values after computing
them. Legacy selector default remains frozen; new endpoint mode only prospective.

Full authority suite and registry validation precede clean-worktree manifests
via experiment.py; no new tracked experiment script, hash machinery, standalone
smoke, core/expert/evaluator edits, weights/guards/seed/offset sweeps or training.
FirstW0 interface execution belongs to the registered28, with all failures kept.
Full first-window evidence, all28 traces/animations and exact next entry required
for engineering closure. Scientific and engineering outcomes are separate.

Implementation gate before new outputs:32 component checks passed (15 endpoint/
waypoint cases plus17 existing candidate cases); full authority1027 passed,
4 historical skips169.78s. Registry380 rows valid. Hydra resolved config equals
sealed candidateB1 after removing the probe block/run identity and setting the
candidate offset0. Existing expert/core/native evaluator trees unchanged.
W0 and signed probes invoke the existing composed sampler with identical saved
first-window conditions; actual HOI model pre-hook captures goals/latent/text/
BPS/progress without random draws. Pelvis goals use local metres,Y0; object goal
normalization is unchanged. Endpointv2 is opt-in; native metric comparisons are
read after physical values are computed. No training microbatch benchmark applies;
synchronized window runtime and allocator peaks are recorded. Full first scene
supplies real-data interface evidence inside the registered workload.

Operational failure at runtime f439dbb: first scene66 reaches the firstW0
model-trace assertion; it wrongly expects only500 diffusion calls while the
unchanged B1 editor adds16 HOI reference calls. The run manifest is failed,
logs and task014 cacheA/endpointB records retained. No signed candidate or
quality outcome was emitted; no route/HSI work began. Correct the count to
500 plus actual geometry_edit.hoi_teacher_calls, preserve captures before
assertions, add an independent516-call regression. Use a new r2 run id and
full authority verification; this implementation-only restart exceeds the
nominal no-retry execution by one already-generatedW0 window, explicitly
retained as failed work. No offset/condition/quality/gate changes. The original
failure did not preserve that one generated tensor before assertion; its log,
manifest and source evidence remain, and it is never claimed as scored data.

Corrected runtime verification:1028 passed,4 historical skips173.91s; registry381
rows valid including retained failed run. New r2 manifest will start from the
committed instrumentation correction. Exact source conditions, ArmB/B1 calls,
offsets, hypothesis and numerical gates remain the preregistered ones.

### Phase2.21 completion — control response supported, quality NO-GO (2026-09-07)

Runtime05b781d completes A112 cached candidates/496windows, B112 exact native
endpoint/completion checks, C75 first windows:28W0+44 signed+3 exact repeats.
22 applicable pairs;6 tasks have original waypoint already at the path endpoint.
All28W0 raw/proposal/edited outputs match sealedC0 bitwise. Model latent/text/BPS/
progress/object goal pairing, repeat outputs, finite/history/editor guards pass.
No new full task rollout or HSI forward. Initial f439dbb count failure remains
failed with logs/partial A/B; one failedW0 tensor was not preserved before its
assertion. New runtime counts500diffusion+16B1reference calls and saves captures
first; no scientific change or implicit retry/overwriting.

A first-window raw/B1 world human coordinateRMS.98956/1.02092mm: B1 did not
collapse larger proposals. Later-window3.60183/3.60900mm is descriptive across
different histories. Registered oracle exact; completed-only/unprotected HS
3.96673 vs oldCG3.99304, only.65875% headroom and worse OS/contact means.
B native_retained_v2 aligns all112 root/object values exactly, including3D
object error, strict10cm classification and actual retained interpolation frame;
legacy Phase2.20 results remain sealed and unmodified.

C edited mean directed root4.6806/4.7030cm, object5.2787/5.2114cm forW+/W-.
All22 tasks have positive root response each sign; task95% and scene means
support direction. But anchor95% upper1.1533/1.1790cm exceeds1cm, plus-side
human/object voxelRMS and object occupancy upper bounds exceed registered
margins. Surface contact, foot and mean speed retention protections pass overall.
Only4/22 pairs meet both signs' per-task useful response and every quality
margin (9plus,8minus individually), below50%. Scientific NO-GO; do not mistake
it for absent waypoint controllability or universal contact failure. H2/H3
unopened;2.22route pool/2.23HSI selection/441/469 remain stopped.

Final authority1028 passed,4 historical skips173.91s; registry382 valid.
r2 controller245.45s;75-window generation386.73 cumulative GPU seconds,
37500diffusion+1200B1reference=38700HOI calls,0HSI. Allocator peaks422.68–427.18MB.
28 actual first-window animations/28 trajectories/112 frames rendered; five
fixed/failure tasks reviewed, human independent review pending. Raw/B1 visual
shapes similar;329 crouching persists. Summary PHASE_2S_WAYPOINT_CONTROL.md,
compact result p2_mixer_waypoint_control_s42_20260907.json, tag
exp/p2s-waypoint-control-v1. Engineering closure passes with retained failure;
next unique priority is jointly feasible waypoint conditions preserving grasp
relations and scene quality under a new bounded protocol, not larger offsets,
unregistered gate changes or automatic complete-route generation/training.
## 2026-09-07 — Phase 2.22 bounded waypoint joint-protection diagnostic

Phase 2.21 established signed waypoint response but only 4/22 applicable task
pairs passed all grasp and scene proxy margins. This subphase tests a deterministic
condition-side gate: retain a signed waypoint pair only when both generated first
windows pass the frozen W0-relative hand-object, contact, support-speed,
human-scene and object-scene margins. It reports rejection reasons and does not
alter generated motion after the fact.

The diagnostic replays the completed `p2-mixer-waypoint-control-r2-s42-20260907`
records for all 22 applicable pairs and retains the six inapplicable tasks.
Registered limits remain those of Phase 2.21, with both signs required to pass;
coverage uses paired task bootstrap (10,000 seed-42 replicates). No new waypoint,
offset search, HSI scoring, training, 441/469 evaluation or route-pool expansion
is permitted. A pass establishes protected coverage only, never full-task quality
or InfBaGel superiority.

## 2026-09-07 — Phase 2.23 single-side waypoint decisions with W0 fallback

User handoff `PriorHOSI_5557dbf_single_side_waypoint_codex_handoff.md` authorizes
independent signed proposals and at most G28/H28 closed-loop native episodes.
Phase2.21/2.22 remain sealed. A preliminary read of72 unique cached windows finds
17 accepted sides on13/28 tasks;333,372,376,421,424 have accepted strict geometric
improvements. This meets the handoff's entry condition; coverage alone is not the
criterion. Complete the cache table, runtime and both native rows in this phase.

Branch `phase/02u-single-side-waypoint`, protocol
`experiments/protocols/p2_single_side_waypoint_s42_20260907.json`, default-off config
`config_sample_hosi_single_side`. Reuse single-side margins unchanged. G minimizes
the existing normalized human24/object128 voxel energy; H minimizes existing
L_full inside G's original max(10%,.0066667) energy allowance. W0 has first tie
priority and remains available even when its scene quality is poor. H score errors
fall back to G and block promotion. All candidates use W0's original scoring
conditions and their own clean scene queries; generation goals alone vary.

Use isolated sampler/editor state and paired RNG; commit exactly one native
window. Save rejected motions and per-hand diagnostics. Formal first windows
must reproduce all72 available caches. Registered task372 interface checks include
candidate-order reversal and default-off B1 equivalence. Full authority suite,
registry validation, resolved configs and clean reportable runs are required.
Primary H-G target is at least5% native HS improvement with negative task upperCI,
scene-mean direction and frozen OS/FS/contact/completion/endpoint protections
against G and B1. Report all native15, all28 tasks, both resampling units, B0/B1
references, fallback and total generation/scoring costs. No new score, offset,
expert/core/evaluator change, training or441/469. The finite registered interface
workload is requested by the handoff and supplies real-data verification.

Implementation verification: authority1039 passed/4 historical skips178.03s;
28 focused component checks pass. Registered task372 interface completed13
windows (off1, G3, reversedG3, H3, reversedH3): disabled B1 exact, both candidate
orders and G/H candidate motions exact, H singleton-budget choice exact, CPU/CUDA RNG restored,
one logical sample call and editor record committed. Cached72 inputs/IDs/finite
and editor checks pass; all28 cache rows and per-hand traces are saved. Eight
resolved configs archive exact P15 online/R2/ArmB/B1 compatibility by reference.

## 2026-09-07 — Phase 2.23 completion: closed-loop decisions, quality NO-GO

Completed G28/H28,520 candidate windows/248 actual commits.144 first-window
cache checks exact;56 native retained endpoints exact;37 all-W0 episodes fully
reproduce B1. G-B1 HS point improves14.65% but taskCI crosses0 and contact/FS
protections fail. H-G HS/OS point worsen1.91%/4.29%; HSI added value is
unestablished. All B1/G/H completion outcomes remain22/28. G chooses12 offsets,
H14; H changes its contemporaneous geometric choice14 times. Benefits and costs
concentrate on a few tasks, particularly375 (HS/OS),20 (FS),333/376/421 (contact).

All8 jobs exit0,653.53s controller;268320 formal HOI and1584 HSI calls, plus13
registered interface windows/6708 HOI calls. Interface372 has singleton score
budget and does not exercise active score-repeat inference. Final authority1039
passed/4 skips175.78s; original registry enum error repaired with result preserved.
Five videos/20 frames and complete task/scene tables retained; human blind review
pending. Summary `PHASE_2U_SINGLE_SIDE_WAYPOINT.md`, compact result
`p2_mixer_single_side_waypoint_s42_20260907.json`, tag
`exp/p2u-single-side-waypoint-v1`. Engineering deliverable closes; method NO-GO.
Next discussion must address joint feasibility across generated histories;
do not tune current score/offset/margins or start further phases/441/469/training.

## 2026-09-07 — Phase 2.24 paired continuation outcomes (authorized)

Latest user handoff `PriorHOSI_f202825_next_experiment_plan_codex.md` authorizes
a bounded counterfactual diagnostic from sealed f202825, branch
`phase/02v-continuation-outcomes`. Read-only inventory verifies all26 adopted
decisions (G12/H14),12 tasks,4 scenes, all78 current candidates available.
Exact physical/condition/history and candidate equality yields22 states; four
duplicate source links remain visible. Plan132 new windows, hard ceiling156,
three cached candidates then at most two frozen B1 windows each. No current
regeneration, extra repeat calls, longer tails, training or441/469 is budgeted.

Protocol `experiments/protocols/p2_continuation_outcomes_s42_20260907.json` freezes
source identities, per-hand diagnostic margins, horizon labels, N/A, costs and
termination semantics before outcomes. Source fingerprints use existing lifecycle
helpers only; sealed expert/config assets are referenced. Recover the exact native
world decode/history update by extracting shared functions from the original
evaluator; use its sample_step for route/progress conditions. Original episodes
have fixed seg_len and no early-goal termination. Label budget censor separately
from genuine original terminal; native_retained_v2 and object3D stay frozen.

Current/next1/next2/cumulative separate actual retained frames; native-compatible
fragment surface metrics and FS stay distinct from whole-task native metrics
and FK/voxel proxies. Diagnose per-hand fixed-source relationships and predicted
contact transitions; ambiguous release remains explicit. Branches have fresh
sampler/editor/caches and paired private per-window RNG. Cached compatible old
B1 futures provide real replay checks within the132 formal windows. Full authority
suite and registry validation precede clean-worktree manifests and execution.
No separate smoke workload; synchronized timing/allocator measurements cover
batch1 inference, with no training microbatch benchmark path.

Gate is complete reproducible consequence evidence, all26 source/missing/error
records and state/task/scene tables, fixed animations/render entry and one-page
mechanism decision. It requires no favorable mean and cannot upgrade Phase2.23.
One preregistration, one logical implementation, one completion commit; close
only2.24. Prepare a future independent-domain learning protocol only if measured
feasible candidates and labels support it; no next-phase execution this session.

Implementation verification:21 continuation component tests pass; full authority
1060 passed/4 historical skips189.72s. Existing B1/G/H default paths share the
mechanically extracted native decode/next-history functions. Native metric formulas
and frozen core/expert code are unchanged. Eight fully resolved lane configs match
sealed B1 sampler, dataset and generation settings exactly after removing only old
waypoint-policy and output identity fields. Archive: results/continuation-preflight-
s42-20260907/ and prospective run results/experiments/p2-mixer-continuation-outcomes-
s42-20260907/. First lane is part of the132 formal windows; its captured B1 comparisons
must pass before the remaining seven lanes start. No extra repetition workload.

Descriptive paired state/component CSV, adopted-state task means and scene means
preserve nesting; no population-effect confidence interval or bootstrap gate is
claimed for these deliberately selected states. Native fragment FS applies the
unchanged floor estimator to each actual slice, with floor values retained.
Both native contact hands and fixed-source FK anchors are saved, including N/A.
Representative rendering selects the lexicographically first registered state for
each fixed task333/376/421/20/375, independently of the outcomes.

Operational failure retained at ecea024: lane00 fails before dataset/expert
construction because Hydra struct rejects the undeclared load_object_payload
option. Zero HOI/HSI calls or generated windows; other seven lanes unstarted.
Original manifest is failed and all logs retained. Correct the config schema
and pass dataset.vis, load_object_payload and test_scene_name explicitly in
the resolved job, removing the runtime assignments. Cached context already
provides the original BPS/text; payload materialization is a loader cost only.
The replacement r2 run uses the same22 states/132 windows and no extra calls.

Corrected preflight:1061 passed/4 historical skips174.73s, including the exact
Hydra compose/struct regression. Eight r2 jobs resolve explicit loader options;
B1 sampling and source physical conditions remain unchanged. The first failed
run consumed0 of132 planned windows. No runtime source edit occurred while its
controller was active; the controller had already exited before this correction.

Second operational failure at8c1b09e: six branches/two states generate12 finite
B1 windows; all4 compatible old B1 context/motion checks exact. Fragment
evaluation then fails because SMPLX_JOINTS_28 lives in utils, not constants.
No scientific metric or label emitted. All branch .pt files remain immutable.
Correct only the import and add native-fragment decode/censor coverage. r3
explicitly reuses state000/013 from r2, evaluates them with0 new HOI calls and
generates only the remaining120 windows. Across runs the planned132-window
generation budget is unchanged. Resolve resume inputs beside the new manifest;
no overwrite, automatic regeneration or changed metric/label threshold.

r3 verification:25 continuation checks; full authority1064 passed/4 historical
skips169.61s. Native-fragment execution covers the actual SMPL-X joint-index
import and7 observed vs9 terminal-held frames on a three-frame example.
A cost regression sums calls/work and takes the maximum memory peak. Eight r3
configs match r2 generation/dataset exactly; the generation function is unchanged.
Resume input identities and separate12 reused/120 new costs are archived beside
the new manifest. Scientific thresholds and the132 total generated-window budget
remain the original preregistration.

## 2026-09-07 — Phase2.24 completion: consequences recovered, result-head entry unmet

All22 states/66 branches complete with132 newly generated B1 windows total,
including12 retained from r2 and120 new in r3.42 same-state B1 motion/context
checks and22 current world-history checks are exact;132 editor records valid.
No missing state or final execution/evaluation error.18 branches reach the
original terminal,48 retain budget censoring.1064 tests pass/4 historical skips.

Original adopted-state categories16 immediate/5 delayed/1 recovery remain as
registered. Ten of17 current composite failure flags concern goal progress
only. Hand/relationship decomposition finds7 current submitted failures,
1 additional cached-tail failure,14 first detected in new B1 predictions.
All15 output-delayed states also fail hand protection in genuinely new
predicted frames after excluding carried history.14 adopted states have
future native hand-contact/distance failures;9 have future support-speed
failures.44 signed alternatives include27 locally eligible and17 rejected
diagnostic branches;18 have cumulative HS<0/OS<=0, but0 meet all protections.

Two operational failures retained: undeclared Hydra loader option (0 windows),
then fragment joint-index import (12 valid windows reused without regeneration).
68,112 HOI/0 HSI forwards,687.15 cumulative synchronized generation seconds,
307.20 controller seconds across all three runs.5 fixed comparison videos/20
frames and full state/task/scene tables retained. Native-compatible fragment
metrics stay separate from whole-task metrics; relationship deviation has
explicit grasp/release ambiguity and is not proof of physical contact loss.

Summary PHASE_2V_CONTINUATION_OUTCOMES.md and compact result
p2_mixer_continuation_outcomes_s42_20260907.json close this diagnostic, tag
exp/p2v-continuation-outcomes-v1. Priority is a new generation-side joint
feasibility proposal using actual histories and valid relationship semantics;
the result-head entry conditions are unmet. No PhaseB training protocol or
next-phase execution, old-score/guard tuning, native28 policy or441/469 launch.

Closure verification repeats the complete authority suite:1064 passed/4 skips,
240.41s;389 registry records valid. Runtime source remains5bf15b7. All work in
this session closes2.24; the next generation-side proposal remains unstarted.

## 2026-09-07 — Phase2.25 history-conditioned relational sampling (authorized)

Latest answer.txt authorizes a generation intervention from sealed7027ccb.
Branch phase/02w-relational-sampling; protocol p2_relational_sampling_s42_20260907.json.
C0 caches vs C1 locally refreshed and C2 persistent object-local anchors:22 actual
states, W0/adopted direction, regenerate current+two native future windows,264
new-window ceiling. P15/R2/ArmB/B1/seed42/waypoint10cm fixed. New module stays
in mixer; shared core/expert code and evaluator formulas remain frozen.

Trust requires both committed frames' full-surface FK proximity<=5cm, contact>.95
and <=2cm object-local motion. Use previous submitted12:14, not carried14:16;
initial window uses original2 source frames. Shared engagement transitions for
C1/C2; contact prediction alone cannot release. Native tasks lack explicit timed
per-hand release: retain zero confirmed releases and record uncertainty. Combined
far geometry/low contact suspends as ambiguous; re-establishment flagged. C1
refreshes references from executed history, C2 retains trusted reference. Same
energy/weights/posterior scale/nine steps, no tuning or new foot controller.

Protocol freezes numerical physical protections, distinct anchor/progress proxies,
state/task/scene pairing, and >=2-task direct-quality-plus-scene upgrade gate.
Evaluate all six branches against frozen C0 W0 activity/stance. Unknown states
stay in denominator. Preserve pre/post B1 output and term/update telemetry.
No separate smoke or hashes added; existing sealed provenance referenced, normal
manifest lifecycle retained. First formal lane validates actual path within264;
full pytest tests and registry validation before clean-worktree run, synchronized
batch1 generation/allocator timings. Fixed333/376/421/20/375 visual cases.

Scope split before implementation: this session closes2.25 bounded mechanism
comparison; conditional full-native28 policy is2.26, separately registered after
its concrete policy is determined. HSI learning/441/469 remain later stages.
One preregistration, logical implementation, completion commit; actual failures
retained. Historical2.24 gate and2.23 NO-GO remain their original definitions.

Implementation fixes the relation-only intervention as an optional composed-sampler
hook; existing CandidatePoolSampler remains the entry, with no extra sampler
class or tools script. Differentiable full-mesh object-coordinate palms match
existing native FK geometry, including the fixed object prefix. Fourteen component
checks cover trust/unknown/release, frame invariance, finite coupled gradients,
immutable history, quality/proxy separation and Hydra composition. All eight final
resolved sampler and dataset trees exactly equal sealed B1. C1/C2 current actions
have identical initial relation state; their first generated window is checked
pairwise, alongside exact cached arm conditions/history for all88 new branches.

All six outputs use one unchanged native fragment evaluator and fixed C0 W0
activity/stance. Legacy composite labels are explicitly archived as proxy labels.
Joint physical candidates are counted independently from the additional requirement
that trustworthy relation engagement occurred without suspension/regrasp ambiguity.
Intermediate goal distance and anchor error remain visible in paired tables.

Final implementation verification:1078 passed/4 historical skips,201.14s;
results/relational-sampling-preflight-s42-20260907/authority-final.log. Initial
suite1076/4 passed before the final two checks and rendering/generalization edit.
Registry390 records valid. Actual registered batch1 generation supplies functional
verification and synchronized performance; no training microbatch is executed.

Operational failure at35830de: four valid current windows generated (2064HOI,
0HSI); saved recorder was taken after SO3 projection but before original history
repin. Its history rotation differs by1.78814e-7; future source and all four final
motions exactly equal their C0 caches (both initial hands unknown). The assertion
stopped continuation and the controller stopped seven remaining lanes. Preserve
failed manifest/logs/four .pt outputs. Move recorder to the actual editor input,
after history repin; the reverse chain/editor output is unchanged. Add a real
500-step chain regression with nonorthonormal history. Explicit r2 resume reuses
these four current motions, corrects only their recorder's history bytes, restores
native absolute sampler counters and the same initial relation state, then runs
the remaining260 windows. Original264 total budget and all scientific parameters
remain fixed. Current source equality, stage metrics and evaluation are recovered
without another model call; resume source/costs remain visible per branch.

Recorder/resume correction verified:1079 passed/4 historical skips,193.18s;
authority-r2.log. Eight r2 resolved sampler/dataset configurations exactly match
the first run; archived resume_inputs.json references all four immutable motions.
First resumed lane generates20 new windows (four currents reused); remaining
seven lanes generate240, total260 new plus4 retained. No generation-output or
energy/threshold change was needed to correct the recorder boundary.

## 2026-09-07 — Phase2.25 completion: no added jointly protected candidate

264 unique new windows complete:4 retained currents plus260 new in r2,22states,
88new branches and44cached C0 branches. All8lanes exit0. C0/C1/C2 each have14/22
scene-beneficial states,18/22 with direct-quality failures, and the same1/22
state passing this phase's relative quality/scene protections (376/state008).
This candidate already exists in C0; new qualified alternatives0. Old2.24
0/27 and0/44 keep their original anchor/progress-inclusive definition.

Initial relation trust13/22states, ever-engaged17/22;5unknown remain in denominator.
C1/C2 each846 applied steps with0nonfinite gradients. No qualifying future
improvement task or extra C2 task; C2 adds task17 right-hand coverage regression
in next2 (C0/C1/C2 87.5/82.5/72.5%). Sole threshold-sized future benefit is C2
support speed on375, accompanied by worse native FS and higher HS/OS; joint gate
fails. W0 regressions0tasks for both versions. Absolute contact and N/A remain
visible; relative protection is not absolute physical certification.

Record timing failure retained at35830de;533b587 moves capture after history
repin and resumes four original currents without regeneration.1079tests/4skips,
88current context/history checks,44C1/C2 first-window equalities and1056exact
C0 native scalar values.136224HOI/0HSI calls;1490.284s cumulative generation;
608.364s controller wall including failure. Five fixed videos/20frames plus
mechanism plot and all task/scene tables retained.

Engineering gate closes2.25; fixed-relation method NO-GO. Summary
PHASE_2W_RELATIONAL_SAMPLING.md, compact p2_mixer_relational_sampling_s42_20260907.json,
tag exp/p2w-relational-sampling-v1. Stop additional weight search;2.26 complete
native28 policy, learning and441/469 unstarted. Future generation adaptation
requires a new concrete independent-domain supervision contract; HSI value
requires a matched no-HSI control. One subphase closed in this session.

Closure verification:1079passed/4skips171.42s;392registry records valid. Frozen
core/experts/native metric and original guidance loss files unchanged from7027ccb.
107persistent active-hand transitions retain C2 anchors exactly; C1 refreshes107;
264B1 editor records valid. Completion changes contain results/docs only.

## 2026-09-08 — Phase2.27 sustained-route continuation (authorized)

The user approved the same-current route experiment after7fe9fa4. Branch
phase/02x-route-continuation; protocol p2_route_continuation_s42_20260908.json.
Reserved2.26 remains the unstarted conditional relational policy. Close only2.27.

Reuse26 source links/22states and both cached W0/adopted-pulse branches. Commit
exactly the cached adopted current, then generate at most2 windows per state,
44 total. Freeze P15/R2/ArmB/B1/seed42/native16/2/final goals/time budget. The only
intervention is one persistent deformed A* polyline. Its cached signed10cm anchor
is fixed; quintic smoothstep enters over one original lookahead and exits over
up to two lookaheads, preserving endpoints. No curve, amplitude or weight search.
Actual-history native lookahead and all non-route conditions are recorded.

GPU fixed-pose sweeps carry current committed FK24 and full object mesh along
both routes, reporting occupancy/domain exposure without rejecting samples or
claiming exact joint-motion feasibility. Native surface/contact/FS evaluation
and fixed W0 support/activity remain shared; original anchor/progress labels are
separate. Report all22, the previously identified14new-prediction-risk subgroup,
state/task/scene pairs and original-terminal vs censored outcomes. The explicit
joint improvement and >=2-task discussion gate is in the frozen protocol.

Full authority suite, registry validation and exact resolved configs precede a
clean-worktree formal run. The first registered lane provides real-data checks
within44; no extra smoke or training microbatch path. Record synchronized timings,
allocator peaks and8RTX3090 allocation. Reuse sealed input provenance. No new tools
script, core/expert/evaluator-formula changes, training or full policy/441/469.
One preregistration, logical implementation and completion commit; preserve actual
operational failures and saved outputs. Deliver fixed five-case animations, paired
reports and a phase summary regardless of direction of results.

Implementation uses an optional route-control callback in the existing
continuation generator; the native model/sampler/history/evaluation functions
remain shared. The new default-off config dispatches one route lane. Registered
quintic polyline, GPU full-object fixed-pose envelope, actual model-input checks,
first-future non-route/history pairing, cached native scalar checks and descriptive
state/task/scene summaries are implemented. Eight resolved configs match sealed
B1 sampler/dataset/expert/native keys. All22 state inputs and44 baseline branches
are present; input identities are referenced. Initial component fixtures lacked
surface_mean_m; correcting those fixtures changes no physical metric or margin.
The final component suite has34 passing checks. Full authority verification is
archived under results/route-continuation-preflight-s42-20260908/.

## 2026-09-08 — Phase2.27 completion: sustained route joint gate unmet

All22states/44new windows complete;8jobs exit0. Cached current/non-route first
future histories match22/22;1408baseline native scalars exact;1088tests pass/4skips.
15states/8tasks show meaningful commands and28.35mm mean future root movement.
Scene-benefit coverage14->16;18states still violate direct protections. Two new
states pass W0-relative joint protection (both task375); only375/state005 also
passes protection vs the pulse, with no threshold-sized future gain. Qualifying
tasks0;8states across5tasks trigger direct regression. Task375 accounts for97.73%
of task-balanced HS net improvement. Six new branches reach original terminal,
five complete as before;16are censored. No HSI forwards or training.

Registered method gate fails; engineering deliverable closes. Stop this fixed
curve without tuning. See PHASE_2X_ROUTE_CONTINUATION.md and compact
p2_mixer_route_continuation_s42_20260908.json. Source4943a60, tag
exp/p2x-route-continuation-v1. One subphase only;2.26/full policy/441/469 remain
unstarted. Next proposal requires independent-domain valid-future supervision.

## 2026-09-08 — Phase2.28 native-surface whole-motion editing (authorized)

User approves the subsequent benchmark-focused experiments, explicitly allowing direct
InfBaGel paper HOSI-test metrics; generating its motions is optional. HSI-specific
benefit and universal per-window improvement are removed from this new phase's
entry criteria. Earlier registered failures remain unchanged.

Branch phase/02y-surface-edit; protocol
`experiments/protocols/p2_surface_edit_s42_20260908.json`. This single offline-editor
phase contains native28 development, matched independent/common relation arms at
fixed10/20cm bounds, separately reported terminal repair, and conditional fixed469
evaluation. P15/ArmB sources, core, experts, native metrics and duration stay fixed.
The editor replaces per-window geometry editing with complete native decoded motion
optimization, using full human/object surfaces and Scene_sdf. Six leg joints provide
support adjustment; source upper-body local pose and common human/object transforms
preserve manipulation. New edits never become subsequent generation history.

The protocol specifies all objective units/weights,20/40 source-risk-dependent steps,
0.5s cubic fields, fixed first6/final3 frames, same-budget independent-motion ablation,
a separate final1s/5cm goal correction and full task reporting. Source and every
iterate use the same objective; select its minimum. No weight/score/curve search
follows beyond the two registered bounds. Native28 eligibility uses OS>=10% reduction
and aggregate HS/contact/completion/FS protections, not confidence-bound perfection.
A qualifying relation arm automatically enters the already-authorized complete469
comparison; its formula is frozen before remaining task outcomes. All469 have prior
use, and development exposure is disclosed. Direct paper comparisons are unpaired.

Before execution: resolved configs, full authority suite, registry validation and
clean committed source. Use8RTX3090 for physical computation. Functional checks and
synchronized timing are contained in the formal workload; no extra smoke or hashing
scheme. Reuse sealed input identities. No core or expert modifications/training.
Deliver every negative candidate, native15, task/scene uncertainty, saved motion,
fixed visual evidence and phase summary. Phase2.27's adaptation-training suggestion
is superseded in priority by this newly approved deadline-focused experiment.

Implementation-stage metadata correction: the preregistration row lacked its required
next_action field; it is completed in the logical implementation commit. The first
commit retains the original schema error; hypothesis and scientific settings stay fixed.

Implementation ready: native decoded pose/translation reuse through an optional
body-parameter return; complete-surface GPU objective, cubic rigid/leg fields,
independent relation ablation, terminal candidate retention and native task summaries.
Fixed frames copy source pose/object/geometry directly; terminal holds the last3
frames.9 component checks pass, including analytic SDF gradients, articulated common
transforms and chunk-boundary reductions. Final authority:1097 passed/4 existing
skips,168.74s (results/surface-edit-preflight-s42-20260908/authority-final.log).
The initial suite had2 setup errors because INFBAGEL_PYTHON was not exported; the
corrected command passes.9 formal job configs resolve; all28 B0 source files exist.
Formal source-geometry/metric recovery and optimization stability remain to be
checked by the first registered task, followed by8GPU execution.

First formal start stopped before dataset loading or editing: CUDA allocator peak
reset preceded lazy context initialization. The failed manifest/trace are sealed in
p2-mixer-surface-edit-s42-20260908 (0 edits/0 native evaluations). The runner now
synchronizes/initializes its device before timed work and allocator reset. A fresh
r1 run retains identical task/arm/optimizer settings; no scientific retuning.

Recovery source also records terminal input metrics for its separate paired contrast,
allows the development-selected terminal input arm to stay fixed on full469, and
records synchronized arm time including evaluation. Final unchanged-science suite:
1097 passed/4 skipped,166.36s (authority-runtime-final.log).

r1 loaded and decoded its first source, then stopped before evaluation/editing:
the new object loader assumed name.npy while the official assets are name.ply.npy.
Use the native evaluator's actual first-dot object key and paired JSON filename;
a regression covers that shipped naming contract. All7 benchmark object SDF pairs
are present. The failed r1 manifest and12s execution are retained; r2 keeps the
scientific protocol unchanged and no completed edits require replay.

r2 stopped at source equivalence: GPU fragment rotation decoding differed from
saved full-native joints by1.257mm. Fixed-source diagnosis changes only rotation
decode/interpolation to the original CPU path: joint discrepancy becomes exactly0;
all native deltas vanish except a5.96e-8 object reduction difference. The full native
CPU pose conversion is now factored into utils.native_body_pose and shared by the
unchanged-formula full evaluator and replay helper. Object surface transformation
also uses the original repeated-vertex bmm order. Large geometry/optimization stays
on GPU. Earlier fragment results retain their original runtime and are not recomputed.
The failed r2 and its2 diagnostic source repeats are retained; no edits were made.

Native-source correction verification:45 focused checks pass; full authority
1099 passed/4 existing skips,167.63s (authority-native-final.log). The initial
verification was interrupted to add the new helper's local transform import, then
rerun on the final code. All9 r3 job configs resolve. Scientific settings stay fixed.

### Phase2.28 development decision and authorized full evaluation

Completed112 edits on28 unchanged B0 sources, all9 jobs exit0. Both relation bounds
pass the registered development point gates; select relation_20 by lower OS. Source
HS4.764332/OS31.093625 become1.177500/11.607790; contact.668160 and22/28 completion
are unchanged, FS.173899->.176571cm. Independent20 has HS1.165812/OS8.685489 but
contact.599414. Source/edited poses, all outcomes, task/scene paired intervals and
selection are sealed in p2-mixer-surface-edit-r3-s42-20260908. Native source joint
recovery is exact; all metric differences meet the registered tolerance.

Freeze20cm per horizontal coordinate,20deg yaw/leg bounds, source-risk20/40 steps
and every loss/selection rule before full469 outcomes. Recreate the full P15 B0 row
with per-episode seeding (the older full469 uses scene-level seeds), then edit the
same469 sources with matched independent20/relation20 and separately registered
terminal repair. The28 repeated source identities provide a compatibility check;
all469 remain a single benchmark denominator. InfBaGel stays the direct paper row.

Before terminal execution, its activity mask now uses direct coordinate-distance
reduction exactly as the native contact function, replacing cdist's matrix-product
distance approximation. A large-world-coordinate regression fixes the5cm predicate;
main relation/independent paths are unchanged. Terminal input metrics and the frozen
selected arm are recorded separately. The approved candidate/acceptance rules stay
fixed; no weight/bound search beyond10/20 is added.

Terminal-ready source: successful inputs reuse their exact saved metrics as well
as motion; attempted/recovered counts are reported separately.12 component checks
and the final full suite1100 passed/4 skipped,190.82s pass (authority-frozen-full.log).
Development visualization produced5 fixed full-motion clips,20 frames and the native
metric figure; the figure and a375 comparison frame were inspected. These skeletal
views support coarse comparison; no perceptual-quality claim is added.

### Phase2.28 complete motions and completion-interval correction

All469 source,938 main-arm outputs and469 terminal outputs completed and their
manifests are sealed at49ac532. Relation20 givesHS2.890980/OS12.102161 with unchanged
contact.692011 and357/469 completion; terminal recovers45/112 failures to402/469,
HS2.886076/OS12.070963/C.692397/FS.171446cm. All original outputs remain immutable.

During aggregation, completed was a JSON boolean and the existing generic metric
discovery excluded it from task-level bootstrap; scene-level means already made
it numeric. Point completion and all native15 metrics are correct. Normalize native
outcome tables, including terminal input, to floats before discovery. A regression
covers completion entering both task/scene contrasts. Recompute only the registered
10000-replicate seed42 summaries in a separate statistics addendum for development,
development-terminal,full469 andterminal469; preserve original summaries and verify
identical means and existing intervals. This is completion of the registered
uncertainty reporting, with zero new motions or optimizer changes. No separate
physical performance run is needed because its executed path is unchanged.

Statistics correction verified:13 component checks and full authority1101 passed,
4 existing skips,210.80s (authority-statistics-final.log). Motion-generation and
optimization settings remain the frozen49ac532 version.

### Phase2.28 closure — measured benchmark gains, 2026-09-08

All authorized work is complete: native28 relation ablation at10/20, separate
terminal28, fresh469 source, matched independent20/relation20 and terminal469.
The frozen relation20 passes full aggregate protections. SourceHS6.953108/OS32.001672
becomes2.890980/12.102161 with exact contact.692011 and357/469 completion preservation.
Independent20 reachesOS8.949761 butcontact.609641. Terminal restores45/112 failures,
producing402/469 (.857143),HS2.886076/OS12.070963/C.692397/FS.171446cm.

The final HS/OS/success point estimates improve upon the authorized external paper
Hybrid3.17/12.45/.8145; contact/FS/Pbody retain their measured deficits. Native15 plus
completion, all task/scene intervals, per-object outcomes and all failed candidates
are preserved. Task completion gain95% CI is[.070362,.123667], scene[.072495,.121535].
The registered statistics addendum at989eeb2 leaves all original means exact and
existing intervals within1.39e-17. Three earlier operational failures remain sealed.
No source/expert training, InfBaGel rerun or next-phase workload was added.

Full source compatibility:28 repeated tasks reproduce original joints and metrics
exactly. Main163/469 motions change; max hand-relative error7.7504e-7m and all locked
endpoint/initial position errors0. HS improves136 tasks and increases21; OS improves160
and increases0 at1e-6 tolerance. All469 have historical development use. Full SDF and
offline compute distinguish this setup from the paper's direct generation row.

Closure references: experiments/results/p2_mixer_surface_edit_s42_20260908.json and
[PHASE_2Y_SURFACE_EDIT](../phase_summaries/PHASE_2Y_SURFACE_EDIT.md); tag
exp/p2y-surface-edit-v1. The next entry is paper preparation from the actual measured
scope. Any new algorithmic direction requires a separate concrete proposal.

封存检查：1101 passed/4 existing skips，190.93s（authority-completion.log）；404条registry有效，9份manifest全部封存，完整469任务/67场景的每项对比均含全部16个结果。PNG/PDF主图已检查，工程交付按本phase完成门槛封存并快进整合至phase/02-mixer。

## 2026-09-08 — Phase2.29 full-HSI root/heading targets (approved)

The user restores HSIPrior scene-knowledge transfer as the core necessary claim,
and approves the proposed first-route experiment. Phase2.28 remains a strong
geometry-only baseline. Branchphase/02z-hsi-motion-target implements one fixed
full-condition root/heading target inside its relation-preserving native editor.
No expert training or shared reverse-chain redesign is included.

Protocol: experiments/protocols/p2_hsi_motion_target_s42_20260908.json. Reuse exact
Phase2.28 P15 B0 sources. Four main arms: same-budget relation geometry, correct
HSI relation, wrong-scene HSI relation, and correct HSI independent transforms.
Read R2 raw full x0 atlevel199 with2 paired draws on each source window; recover
its native conditions and training-compatible known-empty object view. The wrong
teacher rotates all world scene queries+90deg about the fixed task start while
preserving local motion/conditions/noise and the real evaluation scene.

World root XZ/yaw differences become bounded30Hz targets on the unchanged cubic
editing fields; HSI limbs and vertical/root/object channels are not copied.
Persistent target loss weight.25, scales5cm/10deg, original20/40 Adam steps and
all geometry/support settings fixed. Relation mapping moves human/object together.
Apply the same frozen terminal repair to every arm and report its separate gain.

Develop on fixed native28. Entry requires>=1% extra HS gain over no-HSI and>=.5%
of baseline HS advantage over wrong-scene HSI, plus registered OS/FS/contact/
completion/relationship protections. These are prospective point-estimate gates;
all task/scene10000-pair intervals remain reported. A passing candidate automatically
enters the already-approved fixed469 comparison. A failure closes this mechanism;
no new target strength, noise level, sign, or second route starts under this phase.

Read HSIPRIOR_DESIGN_PRIORS.md and Phase2Y/OVERVIEW before implementation. New
runtime work stays outsidecore/ andexpert directories, with one config fragment,
component tests and the existingHydra/manifest/bootstrap lifecycle. Full authority
suite is required; formal native tasks supply physical validation/timing. Input
identities are reused by reference; no additional smoke or hashing workflow.

Implementation ready: full x0 targets use paired level199 queries, circular heading
averages, native non-overlapping future ownership and30Hz interpolation. A consistent
wrong world query also rotates the carried-object prefix, keeping its relative
observation intact while changing the environment. Both query sets retain identical
non-scene arguments. Complete raw predictions, noisy inputs, observations, targets,
query coverage and synchronized teacher timings are saved. Targets remain detached;
only the existing joint-edit controls are optimized. HSI contribution is audited
against the executed same-source geometry output, including object displacement.

The standard native condition-recovery path is reused. Source caches lack their
original complete context, so recovered conditions are not advertised as a comparison
to absent captured fields. World/geometry recovery and same-budget native baseline
are checked against actual saved data. All four outputs receive the same terminal
repair with separate metrics, candidates, reasons and timing.

18 component checks pass; full authority1106 passed/4 existing skips,192.86s
(results/hsi-motion-target-preflight-s42-20260908/authority-final.log). One initial
unit fixture bypassing __init__ lacked the new optional field; corrected before
any formal run. Nine exact job configs resolve. Runtime physical validation and
performance measurements are contained in the registered workload. All8RTX3090
were idle at preparation. No expert/core code, extra smoke or hashing scheme added.

### Phase2.29 closure — HSI transfer executes; quality entry fails

All28 tasks/four main arms and shared terminal repairs completed atf4e7248,9jobs exit0,496HSI forwards and0newHOI windows. Correct HSI changes27/28 motions with source contact exact, root21.70mm andobject33.14mm mean changes versus geometry. HS1.177500→1.241234 andOS11.607790→12.737616; wrong-scene HS1.226711/OS12.734426. Extra-HS,scene-dependence andOS gates fail; other protections pass. All terminal arms recover the same1/6 task to23/28. Full469 is not started.

Correct/wrong target response24.01mm exceeds noise-repeat2.23mm, but final useful scene transfer is unestablished. Geometrically edited9 tasks carry most conflict; the19 unchanged tasks improve on geometry yet do not beat wrong-scene HSI. These strata are descriptive, not a new routed method. Raw full HSI predictions and all native outputs/intervals are retained. See [PHASE_2Z_HSI_MOTION_TARGET](../phase_summaries/PHASE_2Z_HSI_MOTION_TARGET.md) and experiments/results/p2_mixer_hsi_motion_target_s42_20260908.json. Engineering completion is sealed; method promotion is NO-GO, prior geometry baseline remains. No next phase, hyperparameter search or expert training is started.

### Phase2.29 input-contract correction before final closure

The preserved first run is complete and negative. Final readout review identifies
an inherited query/condition inconsistency:124/124 static goal patches use the object
goal, while124/124 model calls disable the object condition. Their centres differ
from the supplied human goal by1.147710m mean,5.358083m max. Native HSI with
is_mix=false/is_object=false selects the normalized0.8m human goal instead.

Align is_object=false at HSI scene-query construction as well as model input.
Actual carried-object geometry still participates in dynamic observations. Verify
same-noise legacy/new queries differ only in the static goal patch and its position;
all other inputs, dynamic observations, targets' noise/weights/bounds, optimization,
terminal logic and entry gates stay fixed. Retain f4e7248 results as the legacy-query
arm and execute one fresh corrected native28 campaign. This is the single evidence-
driven condition correction within the approved root/heading mechanism. No additional
view/parameter search or shared reverse-chain implementation follows a failure.

Human-goal query correction verification:19 component checks and full authority
1107 passed/4 existing skips,261.46s (authority-goal-query.log). The runtime records
the native human query-position error and verifies unchanged dynamic object/other
arguments against a same-seed legacy scene query. Noise, targets' weight/bounds,
optimizer, terminal rule and promotion thresholds remain fixed.

### Phase2.29 final closure after native human-goal query correction

Corrected runafc7712 completes28 tasks/9jobs with124/124 exact native human-goal query positions and unchanged dynamic object/other arguments. Geometry baseline native16 and source joints reproduce exactly. Correct HSI HS1.227326/OS12.766820/FS.179713/C.668160,wrongHS1.223961/OS12.738105. ExtraHS,scene-dependence andOS gates still fail; all other protections pass. Every terminal arm recovers the same1 task to23/28. No469 HSI expansion, additional tuning, expert training or next-phase workload follows.

HSI drives27/28 motions (root21.93mm,object32.76mm versus geometry) with exact task contact and hand-relative error7.11e-7m. The original legacy-goal negative and correction both remain sealed. The corrected run shares GPUs with existing HSI training; preflight/contention record retained and timings excluded from speed comparison. Final result: experiments/results/p2_mixer_hsi_motion_target_native_goal_s42_20260908.json; summary PHASE_2Z_HSI_MOTION_TARGET.md. Engineering completion passes; scientific promotion is NO-GO.

最终封存检查：1107 passed/4 existing skips，230.10s（authority-completion.log）；408条registry有效。修正前后manifest均完成，原始及修正后图表已检查。工程交付完成并快进整合至phase/02-mixer，固定HSI目标的科学升级门槛未通过，469扩展保持未启动。

## 2026-09-08 — Phase2.30 完整HSI预测与身体读出诊断（用户批准）

分支phase/02aa-hsi-body-readout。复用2.29修正轮28任务、124窗口、两次配对预测；新专家前向和优化均为0。协议experiments/protocols/p2_hsi_body_readout_s42_20260908.json固定7臂：source及正确/错配各自planar、residual、full。原生30Hz解码后，将完整身体精确分为root XZ/yaw刚体部分与剩余高度/倾斜/姿态，物体始终来自HOI并随planar部分共同移动。全部指标逐任务逐draw保存，再平均两个draw，做任务与场景配对区间。

这是无边界、无终点锁定的诊断反事实，不能作为采样或可部署组合改进；接触、源地面支撑、脚高和滑动与穿透同时报告。另查预测位置通道与原生FK的一致性及窗口接缝。full必须比source和planar各降低HS至少1%，并优于错配.5%source HS，才支持完整身体包含被丢弃的几何收益；进一步身体迁移建议还须满足协议中的交互与足部保护。条件不满足则停止身体自由度扩展，定位教师/条件域后再提新方案。不会自动开始训练、共同去噪或469。

当前8GPU与另一HSI训练共享，记录占用，短时GPU几何和统计不作速度比较。全套测试、原生来源复现和分解恒等验证；正式数据运行提供功能验证，生成/训练/优化路径保持原样，跳过额外性能基准。工程门槛是完整诊断和结论封存。

实现验证：20组件检查通过（4.40s）；全套1108 passed/4历史skips（273.42s，results/hsi-body-readout-preflight-s42-20260908/authority.log），409条registry有效，9份正式任务配置全部解析。诊断函数加入mixer/diagnostics.py，经原生HSI evaluator入口选择；原来的生成、训练和优化执行路径保持原样。所有物理反事实由原生解码后精确分解，组件检查验证完整表面重建、root XZ剥离和planar手物关系保持。

### Phase2.30 completion — 几何读出损失已定位，交互/支撑保护未通过

9作业全部完成，28任务/124窗口/两次配对预测/7臂，专家前向0。正确full HS2.260640，比planar3.801325低40.53%，几何点估计门槛通过；接触66.8160%→38.4038%，固定地面近地足部60.9153%→33.8809%，OS和终点完成同时退化，直接身体迁移门槛失败。正确full−错配full HS−.160666，任务区间[-.383322,.026828]，场景区间[-.405199,-.008832]。来源与缓存16项原生指标及关节误差0，分解误差<4.77e-7m。

下一入口是受交互、足部支撑及跨窗连续性约束后的身体残差是否仍有正确场景收益；不自动启动自由度扩展、共同去噪、训练或469。工程门槛完成；详见PHASE_2AA_HSI_BODY_READOUT总结和对应紧凑结果。原始HOI来源与强几何编辑基线在报告中明确区分。

最终封存检查：1108 passed/4历史skips，250.45s；410条registry有效，正式manifest已完成，图表已检查。Phase2.30诊断工程门槛完成；几何点估计信号通过、直接身体迁移保护条件失败。完成提交整合至本地phase/02-mixer并由exp/p2aa-hsi-body-readout-v1定位。

## 2026-09-08 — Phase2.31 接触/足部/连续性约束身体读出（用户批准）

分支phase/02ab-hsi-body-projection。复用2.30完整native预测和同28条HOI来源，登记一个固定方案：全序列15帧结点B-spline平滑身体残差，首6/尾3锁定，10cm/20°界限；原生SMPL-X形状FK中同时固定双手和四足部关节，全帧保留来源轨迹。69维身体增量采用80次投影拟合、相邻增量平滑与Newton锚点恢复；幅度通过固定可行性回溯，仅根据锚点/界限/连续性选择，场景指标不参与。实际native重建再次验证1e-5m锚点误差。此约束比仅保护接触帧更强，保留摆动足轨迹的限制明确报告。

五臂source/正确smooth/正确projected/错配smooth/错配projected，两draw各自评价后平均。协议experiments/protocols/p2_hsi_body_projection_s42_20260908.json固定所有投影和统计细节。几何收益要求正确projected HS比来源至少低1%，并优于错配至少.5%来源HS；同时保护接触、足部、OS、完成、首尾及物体。要求投影后平均身体位移>=.5cm且14/28条达到.5cm，区分约束压回原样和有效运动的负结果。任何失败都停止本固定设置，不搜索掩码/权重或自动开始469、训练、下一阶段。

先正式执行完整371任务，记录完整长度投影的同步耗时/显存作为新增计算路径的功能与性能验证，再8GPU并行余27。当前与既有HSI训练共享，记录争用，不作速度优势结论。全套测试、registry验证、所有候选/拒绝原因与负结果保留；无额外smoke、新hash或tools脚本。工程门槛是完整约束诊断和封存，不以科学正结果替代执行完整性。

实现验证：22组件检查通过（4.58s），全套1110 passed/4历史skips（253.66s），411条registry有效，9份完整任务配置解析通过。解析18×69锚点雅可比经有限差分验证，投影方向在锚点零空间，Newton恢复保留非锚点身体变化及首尾恒等。初始雅可比测试fixture使用2帧，改为符合完整序列约定的60帧后通过，属于正式运行前实现工作。

手部约束固定关节点位置，保留手掌朝向自由度；全体原生手—物体表面穿透指标必须与接触率一起报告。该诊断不把固定接触关节点解释为完整抓握表面几何恒定。

### Phase2.31 completion — 保护及点入口通过，场景收益集中于375

9作业全部完成，28任务/124窗口/五臂两draw，专家前向0。正确投影HS4.523796（来源4.764332，错配4.632435），登记几何/运动/保护条件全部通过；接触66.8160%、完成22/28、OS31.093625保持，原生手足锚点最大1.22153e-6m，首尾/物体恒等。28/28条身体变化>=.5cm，24点均值1.1722cm。

来源与错配两条HS任务/场景区间均跨零。正确比来源11胜/13负/4平，比错配8胜/16负/4平；任务375贡献超过净均值改善。将375单列的敏感性中其余27条HS差为+.017477（对来源）、+.005395（对错配），未改正式28分母或登记门槛。科学判断为约束机制与可行身体运动验证成立，广泛场景迁移证据尚不足；默认强几何路径保留。

下一入口是在强几何解上测同一受约束身体读出的额外收益并核对当前动作查询条件，预先报告任务覆盖与375敏感性。当前工程门槛完成，下一实验/469/训练均未自动启动。详见PHASE_2AB_HSI_BODY_PROJECTION总结、对应紧凑结果及原始全体记录。

最终封存检查：1110 passed/4历史skips，279.17s；412条registry有效，正式manifest已完成，图表已检查。Phase2.31工程门槛和登记点入口完成，保留375敏感性与区间跨零的科学限制。完成提交整合至本地phase/02-mixer，exp/p2ab-hsi-body-projection-v1定位本次交付。

## 2026-09-08 — Phase2.32 强几何解上的HSI增量（用户批准）

分支phase/02ac-hsi-geometry-increment。固定同28任务、relation_20终点修复前强解、2.31全部投影参数和五臂，重新以当前几何动作查询R2 EMA（level199，两配对draw，496前向）。当前native姿态重建位置/旋转通道、原生get_mat窗口框架与sample_step条件，保留实际物体几何和原生human-goal场景块；旧HOI原始位置预测通道与nativeFK的差异明确记录为输入契约变化。

先同时解码当前clean窗口与HSI输出，再将两者的身体位移/旋转差施加到精确30Hz强解，抵消重采样误差；不把coarse往返产生的差异当成HSI信号。所有原生基线、世界坐标/物体、native粗帧身体、零残差恒等与六锚点保护须验证。先正式执行非零几何编辑的375完整任务，记录功能/性能，再8GPU完成余27；其他HSI训练保留，耗时不作公平速度结论。

协议experiments/protocols/p2_hsi_geometry_increment_s42_20260908.json固定细节。点入口沿用2.31、参照改为强几何解；另登记覆盖条件：375单列后其余27对来源和错配的HS均值差均<=0，两项全28比较的改善数>=退化数（平局中立）。正式28分母及所有指标/候选保留，入口与覆盖分别报告；没有自动469、训练或下一阶段。当前nativeFK重新编码同时改变输入表示，不能把与2.31的差异全归因于源动作变强。工程门槛是本固定增量诊断完整封存。

实现检查：24组件检查通过（4.74s），413条registry有效，9份正式配置完全解析，旧/新均引用同一冻结R2 epoch222。世界坐标/物体旋转编码和零教师残差恒等已验证；投影组件保持字节原样。

验证资源事件：第一次全套检查在已有真实LINGO数据构造处耗时559.20s，466 passed/2 skips后主动中断，栈位于NumPy数组读取，进程128线程；原日志保留。当前以OMP/MKL/OpenBLAS各4线程重跑同一完整测试，未改变测试项或源码。新增接口已通过组件检查，因此固定实现提交后，登记中的完整375 GPU任务与该独立CPU数据检查并行；余27任务等待375与完整authority双通过。正式机器预检明确记录authority pending，完成记录补齐其结果；不把尚未完成的检查记作通过。正式前无额外GPU试跑。

### Phase2.32 execution-resource interruption and fixed-method retry

首轮ea4bcbf正式375在517.53s后因执行资源瓶颈主动中断，manifest failed已封存；40次当前HSI前向、全部查询/原生粗帧审计及目标保留，投影未完成，无科学结论。两次完整测试分别在559.20s（466/2）和1466.05s（469/2）于既有NumPy数组读取处中断，原日志均保留。采样证据：authority/375进程内核CPU时间占91.79%/87.74%，同一采样区间565次compact_stall全部失败；全局THP为madvise。

仅对本任务新进程设PR_SET_THP_DISABLE=1、NUMPY_MADVISE_HUGEPAGE=0，以定位/消除直接内存整理阻塞；不改全局内核或其他训练。新run id为p2-mixer-hsi-geometry-increment-r1-s42-20260908，算法、权重、模型、噪声、投影预算、阈值和测试项保持原样。本次额外提交用于实际运行失败与执行契约变更，属于governance-only；完整测试和375在此进程级内存设置下重试，余27仍等待双通过。依据与原始数据在hugepage_compaction_evidence.json及首轮operational_failure.json。内核进程级控制文档：https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html 。

### Phase2.32 coarse-grid audit correction

进程级THP关闭后，完整authority1112 passed/4 skips（231.44s），375完整任务98.49s；首轮40个HSI输入和预测与重跑逐项相同（误差0），内存策略未改变数值。r1随后完成375/372/377/329/15五任务，七lane在插值后粗网格验证处失败；12任务268次教师前向及全部部分结果保留，manifest failed封存。

根因是把原生时间插值后的粗索引当作编码输入：既有quaternion_slerp近同向线性分支在t=0混入相邻帧。编码的世界位置/旋转验证已通过；需在时间插值前直接解码当前姿态到原生SMPL-X，仍按1e-4m检查输入身体，而将插值后0.14–1.50mm误差作为记录项。原生插值定义与历史基线保持原样，clean/预测双路径及零残差还原机制也保持原样。该变更修正验证网格，不改变教师、目标、投影或科学门槛。新r2使用同一进程级内存策略，完整测试及正式全28重跑；原r1部分结果不用于挑选任务。

r2首作业包含375与373：分别覆盖非零几何编辑和原先最大1.50mm插值后检查误差；通过后8GPU运行余26。每个任务/噪声/算法预算保持原样，仅改变执行分组。

修正验证：25组件检查通过（4.75s），415条registry有效。新增测试以小角度序列复现原生插值后的粗索引变化，并验证插值前原生姿态直接解码逐帧正确；零残差还原继续严格恒等。仅新增直接粗帧解码审计，旧的插值后误差继续记录，所有教师与投影数值路径保持原样。完整authority和r2前两条正式任务在no-THP进程上下文中执行，余26等待双通过。


### Phase2.32 completion — 强几何增量与覆盖失败，保护及运动通过

r2在干净f75781e上完成9作业、28任务/124窗口/496教师前向/五臂两draw；16项来源原生指标及关节误差0，124当前查询通过，直接粗帧native身体最大5.96e-7m，零残差还原0。修正前后12任务2996个teacher张量完全一致，原5个完整任务全部指标误差0。两个失败manifest和部分产物保留，三次尝试实际前向共804。

正确投影HS1.197355，强几何G1.177500，错配W1.185904；C对G10胜/14负/4平，对W6胜/18负/4平。其余27的C−G/C−W为+.015664/+.005870，两项几何及四项覆盖条件均失败。全28的C−G任务区间[-.025401,.076329]、场景区间[.003321,.036388]；C−W分别[-.001906,.028061]/[-.000071,.022098]。entry=false，incremental_evidence=false。

正确平滑比错配平滑HS低.115230，任务及场景区间均低于0，但接触66.8160%→48.1125%、源地面近地足部61.4245%→29.7756%。固定投影恢复逐任务接触/支撑、物体与首尾，保留平均1.2178cm身体变化（28/28>=.5cm），原生六锚点最大1.25028e-6m；两项运动及全部八项保护通过。科学判断是该固定受约束读出未产生强基线上的增益；当前数据无法独立归因于目标方向或可行空间限制。

完整authority1113 passed/4历史skips（233.86s）后释放余26任务；8×3090争用记录保留。正式375完整426帧功能/性能、全部统计及失败封存，耗时仅描述实际环境。详见PHASE_2AC_HSI_GEOMETRY_INCREMENT总结及对应紧凑结果。工程门槛完成，停止本固定方法并保留强几何默认路径；下一入口先提具体诊断，469、训练、共同去噪和下一实验均未启动。


最终封存检查：1113 passed/4历史skips，217.34s（authority-completion.log）；416条registry有效，正式manifest在干净f75781e上完成，图表已核对。Phase2.32工程交付完成，科学增量及覆盖条件失败。完成提交快进整合至本地phase/02-mixer，由exp/p2ac-hsi-geometry-increment-v1定位。


## 2026-09-08 — Phase2.33 当前强解的可行方向诊断（用户批准）

分支phase/02ad-hsi-feasible-direction。复用2.32全部28强几何来源与正确/错配两draw平滑目标，固定比较当前HSI方向与真实完整nativeHS下降方向在相同六锚点约束中的局部作用。协议experiments/protocols/p2_hsi_feasible_direction_s42_20260908.json。69维规范化切空间、首6/尾3及固定物体沿用2.32；P=I-J^+J使用同一原生解析雅可比。正确/错配各2方向、几何1方向；几何梯度覆盖所有SMPL-X顶点和帧，与原生HS尺度一致，24帧分块GPU反传。该几何参照只检验当前可行机会，不能替代HSI贡献对照。

各非零投影方向先归一为原生24点线性化RMS1mm，再用五方向共用的单任务缩放满足5mm/2°单步及原30/10/10cm/s速度上界。固定主步1、副步.25，均做20次原有Newton锚点恢复；没有80步目标拟合、重算教师、反复梯度更新或得分挑选。负.25步只做中心方向导数审计。正步全部原生15指标及完成/保护/实际幅度保留，几何和source复用两draw同一结果；392条配对布局、308次独立正步原生评价。

首个正式375完成全426帧梯度和所有步幅的功能/性能核实，然后8GPU执行余27。完整测试与registry验证，当前进程树关闭THP、4CPU线程；既有HSI训练保持。专家前向、训练、HOI生成、终点修复及469均为0；不添加额外smoke或hash机制。

主判据固定在步1：正确HSI对G和错配的全28及预登记其余27 HS均值差都<0、改善数>=退化数，同时通过原生保护，才记录局部HSI信号；任务/场景10000seed42配对区间同时报告。纯几何在保护内均值改善且改善数>退化数表示局部机会。几何可改而HSI失败定位当前读出方向不适配；HSI小步有信号而原80步无收益支持后续有限步提案；全部缺少改善只表示本局部探测未展示机会。梯度/真实小步不一致、弱/零梯度和零HS任务完整报告，不能推断全局不可行。工程门槛是完整诊断封存；本轮结束后不自动启动新方案或全469接入。

实现验证：28组件检查通过（4.52s），全套1116 passed/4历史skips（207.07s），417条registry有效，9份配置完全解析。解析全24点切向速度经有限差分验证，投影梯度具有下降内积，零增量native姿态严格恒等；平面SDF夹具验证原生HS尺度、首尾梯度锁定及完整53帧分母。新增命名诊断复用原生body-projection数据/评价入口，原80步求解器保持原样；完整375正式任务承担新增反传和固定重建的功能/性能记录。


### Phase2.33 completion — 局部几何机会存在，HSI正确场景方向优势失败

干净bf0837c完成9作业、28任务/124缓存窗口、392条配对报告行/308次独立正步原生评价、140负步HS核对，专家前向0。G/C/W/几何主步HS为1.177500173/1.177469796/1.176936864/1.143398905。几何23改善/0退化/5相等，均值−2.90%、任务及场景区间均低于0；C对G12/11/5，对W8/15/5，主/副步均缺少正确场景优势。其余27C−G−.000262309、C−W+.000162997；局部HSI条件失败，几何机会通过。

原始C方向HS导数−1.824928，经投影为+.013954，去掉分量−1.838881；参数方向范数仍保留91.46%，与可行几何方向平均余弦−.002696。140方向解析/中心差分/主步变化符号全部一致。5个零梯度任务中4个仅锁定开头帧有穿透，1个来源HS0。数据支持当前读出方向与可行改善方向不适配，局部几何空间仍有机会；不推断全局不可行。

所有来源原生16和关节误差0，可微HS源误差<=1.06e-6；接触/支撑/物体/首尾保持，正确HSI和几何全部保护通过，错配3个内部FK1um阈值越界候选保留（原生均<10um）。控制器131.564s，主任务累计414.809s，单进程峰值.9373GiB，争用环境耗时仅描述实际运行。补充零梯度核对第一次工作目录命令失败及修正记录一并保留，正式9作业全部成功。详见PHASE_2AD_HSI_FEASIBLE_DIRECTION总结及紧凑结果。

工程诊断完成，停止本固定方向试验。下一入口按用户已接受的退化容忍偏好，具体登记现有HSI身体读出接入最优链的全469对照，分别报告参与程度和正确场景贡献；本轮没有自动启动该接入、额外几何优化或训练。


最终封存检查：1116 passed/4历史skips，207.39s（authority-completion.log）；418条registry有效，正式manifest已在干净bf0837c上完成，图表已核对。Phase2.33工程门槛完成，局部HSI场景信号失败、几何机会通过。完成提交快进整合至本地phase/02-mixer，标记exp/p2ad-hsi-feasible-direction-v1。


## 2026-09-08 — Phase2.34 最优链的HSI接入与全469对照（用户批准）

分支phase/02ae-hsi-integrated469。用户接受适度指标退化，批准在P15+guide→relation-preserving edit→terminal repair中加入固定HSI身体引导，完整评估469条。采用2.32受约束80步身体读出，放在关系编辑之后、终点修复之前；2.33的微小步诊断不作为本轮接入方法。固定R2 epoch222 EMA、noise199、正确/错配各2配对draw、当前native动作查询/零残差还原与全部投影参数。当前已有469条/67场景/2086窗口/90426帧来源，原28条强几何16指标与全量缓存误差0。

协议experiments/protocols/p2_hsi_integrated469_s42_20260908.json；六臂source/正确投影/错配投影及三者各自终点结果，共5628条task-draw-arm记录。来源和无HSI终点基线直接引用封存469数据，正确/错配重新查询全部2086窗口，8344次HSI前向；没有HOI生成、训练、新几何优化或参数搜索。每条源动作核对原生16/关节及terminal输入；正式首作业19/358/375分别覆盖终点恢复、最长468帧、已知几何编辑，并回放无HSI终点与封存基线比对。

终点阶段完整沿用2.28external路径：源姿态重建顶点，成功输入保持原样，失败输入使用最后一秒8cm目标裕量、5cm最大平移、20步原目标及既有验收；正确/错配两draw分别执行，保存所有候选、拒绝原因及前后指标。投影期物体/手足/首尾保护与终点阶段允许的目标修正分别报告。旧28兼容性、其余441历史使用分组及375其余468敏感性均保留，不进行任务筛选或best-of-two。

用户授权的接入交付与HSI收益判定分开：完成全469及工程契约即交付本固定链，指标下降照实报告；正确场景贡献仍单列最终C相对无HSI及错配的均值/保护条件、10000seed42任务/场景区间及任务覆盖。完整数据均有历史使用，不宣称独立泛化。科学负结果不触发权重/步数调参或训练。

9个持久作业，首3任务完成后8GPU按整场景帧预算均衡余466。其他HSI训练保留，进程局部THP关闭、4CPU线程；全长正式任务提供功能/性能记录，额外smoke和新hash机制均不添加。每任务先保存动作再写完成指标，全部部分结果保留。控制器完成后自动汇总并在干净源码封存；前置任务和持久输出验证后可报告吞吐/剩余时间并结束连续轮询，后续仍须完成报告/提交。当前阶段范围截至全469对照封存。

实现验证：29组件检查通过（4.69s），全套1117 passed/4历史skips（204.41s），419条registry有效，9份完整配置解析通过。新增终点组件检查验证成功输入恒等、固定20步、拒绝候选及原输入同时保留；原80步投影器保持原样。469条、2086窗口和90426帧按封存输入固定，先19/358/375回放，余466分配到8GPU。控制器已具备任务落盘回读、全量配对统计和干净源码自动封存，运行完成报告写入/data/yujinlun/report/PriorHOSI_hsi_integrated469_20260908.md。


### 2026-09-09 — Phase2.34 completion：完整接入交付，收益与代价分别封存

9作业全部exit0，469任务/67场景/2086窗口/8344HSI前向，5628条配对报告行全部完成。正式manifest在干净d806bd3自动封存。最终B/C/W的HS为2.886076/2.944940/2.966740；C相对B退化2.04%，手—物/人体—物穿透改善3.04%/2.84%。C两draw均401完成，B402，W401/402。C相对W的场景优势点条件通过，任务HS区间[-.056996,.011839]、场景[-.040532,-.005661]；相对B区间在两单位均高于0。额外HS改善与完成保护失败，整体scene_utility=false，授权接入交付继续按原约定完成。

C对B183改善/208退化/78相等，对W201/190/78；场景分别25/42和44/23。剩余468、旧28及余441统计完整保留。468/469条有>=.5cm投影身体变化，最终24点平均变化1.2694cm。原生来源16/关节、终点缓存输入、旧28三个主臂及三条终点回放误差均0。全部2086查询及零残差恒等检查通过，最大原生手足误差1.44e-6m。

完成损失为248/draw0正确与错配因FS、216/draw1正确因接触越过原终点验收阈值；所有候选已恢复终点却被质量规则拒绝，原候选和完整拒因均保留，未调整阈值。固定链交付配置config_sample_hosi_integrated469.yaml；完整总结PHASE_2AE_HSI_INTEGRATED469与对应紧凑结果记录全16指标、配对区间、运行成本和限制。当前阶段结束，保留无HSI质量参照，后续训练/调参/新方法均未启动。


最终封存检查（2026-09-09）：1117 passed/4历史skips，201.85s；420条registry有效，正式manifest已完成，图表已核对。Phase2.34按用户批准的质量取舍完成交付，科学整体收益条件保持失败记录。完成提交快进整合至phase/02-mixer，标记exp/p2ae-hsi-integrated469-v1。


## 2026-09-09 — Phase2.35 冻结R2的DNO动作编辑（用户批准）

分支phase/02af-hsi-dno。使用用户提供的本地官方DNO，冻结R2 epoch222 EMA，固定原28条强几何来源。协议experiments/protocols/p2_hsi_dno_s42_20260909.json锁定100步反演、10步可微DDIM、300步重建及300步正确/错配编辑，seed42单一共享初始latent。正确/错配共用真实SDF与全部源约束；各自零编辑解码也评价，避免把初始重建差异误当优化收益。

完整任务各窗口联合优化；生成历史跨窗反传，原生场景查询网格沿用封存当前源动作条件。该固定查询版本的局限明确报告。GPU可微原生身体和全表面SDF参与优化，源动作保持、全帧双手/四足1mm尺度损失及连续性共同约束。最后沿用现有平滑和80步精确锚点投影，逐阶段保留DNO原始/平滑/投影动作；投影后结果属于适配DNO，场景收益可能在此损失。纯几何300步参照使用相同物理目标及输出约束，计算预算差异明示。

首个完整375任务验证梯度、显存和可恢复阶段输出，再8GPU执行余27；既有训练保持，新增峰值显存限5GiB。完整authority及registry、干净提交与resolved配置、正式manifest和所有失败保留；不添加额外smoke、新hash或tools脚本。所有原生指标、重建误差、接触/支撑、任务及场景10000次配对区间、375敏感性均报告。有限预算重建失败不能证明全局不可表达；纯几何改善也不等价于HSI场景贡献。工程门槛是固定28完整封存，科学失败不触发搜索、训练或469。

正式执行前明确：旧目标读出的XZ/yaw剥离属于目标分解，六锚点可行域允许相应补偿。本轮以keep_planar=True把DNO完整XYZ/旋转目标送入原有平滑与80步投影，原有锚点、首尾、位移/角度/速度界限保持。正确/错配的零编辑重建均经过相同平滑/投影，配对报告各自编辑前后的变化之差。两种编辑重置相同CPU/CUDA随机状态，保证作者decorrelation补齐噪声对应。所有澄清发生在正式数据执行前，属于当前实现与对照契约。

实现验证：37项相关检查通过（4.77s），最终全套1125 passed/4历史skips（227.85s），421条registry有效，9份正式配置完全解析。检查覆盖DDIM正反更新/梯度、生成历史跨窗反传、原生插值值与相同旋转有限梯度、完整53帧分块SMPL目标的解析导数、实际DNO优化器检查点恢复逐项一致、完整XYZ/yaw目标与旧默认分解的区别。新增路径采用GPU可微原生解码与24帧分块表面反传；每50步保存优化器/噪声/随机状态。正式完整375任务承担新增运行路径与显存/耗时验证，随后按固定分组执行余27。

首次正式启动在创建manifest后因机器无tmux失败（exit127）；GPU任务、专家前向均为0，原manifest已failed封存。r1改用Python Popen(start_new_session=True)创建独立持久进程，训练/采样/目标/预算和分组全部保持；使用新run id保留首轮操作失败。该额外提交仅记录实际失败和执行契约，数值实现沿用f1b28b8。


### Phase2.35 — 源动作恒等修复与完整28恢复

r1运行12071.06s后failed封存：23条完成，371/421在source记录阶段因1.19209e-5的feet_height差及1.90735e-5的HS s_max差触发原1e-5检查；18/331/423随所属作业中断尚未执行。源动作在入口已验证通过，错误是记录source时又以24帧SMPL-X重新构建一次，引入约亚微米关节差；该记录应直接保留已验证的原生source。修复保持阈值、DNO反演/优化/损失、生成历史、所有约束和评价定义。

新r2以原生source直接记录；复用r1的23条完整编辑轨迹及各阶段原生结果，以封存2.32的同一source恢复精确参照行，原r1浮点差及完整结果继续保留。只补齐固定缺失18/331/371/421/423，全部采用原预算。新目录记录每条来源、source记录修正前后差和原manifest；完整28汇总使用原配对统计/判据。全套authority、registry、精确resolved配置和5个正式全长任务验证修复；沿用独立后台进程。该源参照序列化修复不改变23条的优化输入或编辑输出，完整结果仍限定人体编辑、固定物体轨迹的既定范围。

修复验证：完整authority1125 passed/4历史skips（214.02s），423条registry有效，5份配置完全解析；23条结果已按来源引用恢复。修复仅改变source记录，优化目标/梯度和每步计算保持，正式371/421完整任务验证原1e-5源契约；5条分配GPU2–6，实际硬件与原23条的资源差异记录。


### 2026-09-10 — Phase2.35 completion：完整28封存，重建与场景收益条件失败

r2补齐5条全部exit0，与r1原23条组成28任务/4场景/124窗口/5376帧、18阶段504条记录。源原生指标/关节、clean恒等误差0；504动作与84个最终检查点回读通过，物体及首6/末3帧精确保留。source修复仅恢复源参照，原编辑输出直接复用；两个失败manifest及逐项恢复记录保留。

源/C/W/G投影HS1.177500/1.209680/1.198180/.938860；C对源+2.73%，G对源−20.27%。C对源11改善/13退化/4平，对W13/11/4，对G1/23/4；几何23/0/5。全28 C−源/C−W区间跨零，C−G任务/场景区间均高于0。375驱动整体退化，预登记其余27C−源−.011539，区间仍跨零、正确场景优势仍失败。投影后编辑变化C−W为+.005513，两个单位区间跨零。

300步正确重建身体平均15.5004cm，范围9.4017–40.3576，0/28通过；接触66.816%→22.097%，支撑61.424%→33.679%。末步加权去相关/身体损失比中位数57.6835，28/28>1；仅损失尺度证据，梯度主导尚未测量。最终投影恢复交互/支撑，完成22/28保持，原生六锚点总体最大1.24458e−6m。科学收益两条件失败，全部六保护条件通过。

任务耗时合计58342.14s，2251840次实际HSI前向含梯度重计算；峰值1.32527GiB。r2五条控制器3823.49s，机器分配/争用记录保留，耗时仅描述实际环境。工程门槛完成，详见PHASE_2AF_HSI_DNO及紧凑结果。当前固定DNO配置停止；下一入口先提出源重建采样/反演一致性与分项梯度诊断。数据范围是固定物体的强几何来源人体编辑，人体物体联合改道、训练与469扩展均未启动。


最终封存检查（2026-09-10）：1125 passed/4历史skips，204.05s；424条registry有效，504阶段动作与84最终检查点核对通过，图表已核对。完成提交快进整合至phase/02-mixer，标记exp/p2af-hsi-dno-v1。工程诊断完成，科学重建与场景收益失败按原结果封存。


## 2026-09-10 — Phase2.36 源重建端点与分项梯度诊断（用户批准）

分支phase/02ag-hsi-dno-reconstruction。固定2.35全部28条强几何来源，冻结R2。原反演多做一步到alpha=0，却直接输入第499步解码；登记端点（原pure-noise/对齐499）×去相关（1000/0）的2×2对照，原臂直接复用，另三臂各300步。原生身体和特征目标、lr.05/warmup50、10步解码/100步反演、seed42及全部条件固定。零权重是因果诊断，有限预算结果不自动形成新的编辑默认。

初始、50、300步测身体/特征/加权去相关对latent的独立梯度范数、夹角和总梯度投影，同时记录噪声统计。诊断保存恢复CPU/CUDA RNG，不改变优化序列。初始latent另做100步解码以及10步源历史条件解码，与10步生成历史分别比较；报告编码恒等、alpha端点及最后纯噪声跳跃大小。原始clean作为t0的近似在两端点臂相同。

每臂全原生指标、身体重建误差和原接触/支撑/FS条件；全28、375与其余27，任务/场景10000次seed42配对区间和因子交互。新运行输出只含重建，无末端硬投影，以便直接测重建能力。先完整375，再8GPU余27；已有GPU4–7任务保留，新增显存5GiB上限、全长正式功能/性能记录，额外smoke/hash不添加。全suite、clean源码、resolved配置、正式manifest和阶段检查点沿用。

工程门槛是固定对照及梯度诊断完整封存；科学门槛按原<=1cm及接触/支撑/FS逐臂报告。达到重建条件只支持进入后续编辑提案。未通过只定位本固定预算，训练、编辑和469扩展均不自动启动。协议experiments/protocols/p2_hsi_dno_reconstruction_s42_20260910.json。

实现验证：10项DNO组件检查通过（1.46s）；全套1127 passed/4历史skips（196.47s），425条registry有效，9份配置完全解析。端点解析夹具验证保留第499步干净分量，梯度夹具区分损失大小与更新方向。新探针复用原生入口和DNO检查点，旧解码/反演默认保持；完整375正式任务承担功能/性能验证。执行前清理继承协议的旧编辑/投影描述，明确本轮仅重建。


### 2026-09-10 — Phase2.36 completion：去相关主导梯度，源重建仍未达标

9作业全部exit0，28任务/4场景/124窗口/5376帧固定2×2完成，三新臂各300步、原臂复用。原/reg0/对齐reg1000/对齐reg0身体误差15.5004/4.8680/15.1027/4.8007cm。去相关移除在两端点均28/28改善，任务及场景区间低于0；端点在reg0下额外−.0673cm，两单位区间跨零。四臂均0/28通过重建条件；aligned reg0接触63.170%/支撑49.496%，仍低于源66.816%/61.424%。

旧reg1000的正则/重建梯度比中位数初始1066.72、50步447.12、300步35.13，均28/28>1；总梯度与重建方向近正交，明确记录Adam预处理前的性质。初始10→100步解码18.259→15.812cm，源历史替换8.702cm，显示离散化/跨窗历史敏感性；剩余误差归因仍未唯一确定。

252新增动作、84最终检查点、336梯度snapshot回读通过，复用参照指标差0，源latent对应、物体及首尾精确保留。控制器18742.29s，2307516HSI前向含重计算，峰值1.11261GiB；GPU4–7争用导致尾段较慢，耗时描述实际环境。干净bdee7db开始/结束，完整结果PHASE_2AG_HSI_DNO_RECONSTRUCTION及紧凑JSON。工程诊断交付完成；科学重建失败，下一入口先提生成历史源拟合诊断，编辑、训练、469均未启动。

最终封存验证：1127 passed/4历史skips（198.16s），426条registry有效，git diff --check通过；图表已核对。预登记占用正式run id，追加完成行使用独立completion记录id并显式指向原run_id；原假设和正式manifest保持。


## 2026-09-10 — Phase2.37 拟合历史与解码历史的跨窗源重建诊断（用户批准）

分支phase/02ah-hsi-dno-history。固定2.36全部28条来源、对齐499的同一初始latent、R2epoch222EMA、10步DDIM与去相关0。复用生成历史拟合300步的aligned_reg0，新增源历史拟合300步，原生身体/特征目标、lr.05/warmup50/seed42保持。两种拟合latent均在生成历史与源历史下解码，组成GA/GS/SS/SA四格。没有额外拟合预算、课程或重启策略。

逐窗记录原生28点mean/RMS、root、216特征误差、输入历史偏差和reframe后的边界不一致，保存逐帧逐点误差与帧归属。相同latent首窗预测在两种历史解码下应精确相等；末3帧插值受下一窗影响，单列解释。对两个最终latent分别测两种历史下body+feature的逐窗梯度范数及方向夹角，分清拟合补偿与历史传递敏感性。clean源历史跨窗兼容性也核对。

四格完整native指标/交互/支撑/速度与原<=1cm条件；GA作为原可执行参照，SS只表示源条件下的窗口拟合，SA检验其生成历史转移，GS检验已有latent的历史补偿。全28、375/其余27及任务/场景10000次seed42配对区间完整报告。有限预算未达标只说明当前拟合不足，不能推断全局表达能力。

先完整375GPU0验证实际新路径、显存及可恢复检查点，再8GPU余27，保留已有GPU4–7作业、新增5GiB上限；共享资源耗时仅描述本次运行。完整authority、clean源码、精确resolved配置与正式manifest沿用，无额外smoke/hash/tools脚本。工程门槛是固定四格与逐窗诊断封存；下一方法、编辑、训练及469不自动启动。协议experiments/protocols/p2_hsi_dno_history_s42_20260910.json。

实现验证：12项DNO组件检查通过（1.73s），完整authority1129 passed/4历史skips（199.14s），427条registry有效，9份配置完全解析。历史干预的首窗恒等与跨窗梯度切断、原生90帧完整归属及输入历史/边界误差分离均通过；抽取共享native记录逻辑，旧路径默认保持。完整375正式任务验证实际新拟合路径、回放与显存性能；原先继承的历史条件说明在执行前明确为四格契约。


### 2026-09-10 — Phase2.37 completion：源条件拟合改善，生成历史传递失败

固定28任务/4场景/124窗口/5376帧，9作业全部exit0。GA/GS/SA/SS身体误差4.8007/6.1075/9.9903/2.4932cm，四格均0/28通过原重建条件。SS对GA27改善/1退化；SA对SS28/28退化（+7.4972cm），任务及场景区间均高于0；GS对GA24退化/4改善，显示已有latent的历史条件补偿。其余27结论保持。

SS后续窗解锁误差2.3345cm，SA14.4164cm；源历史reframe最大误差3.5763e−7，未发现此处的实现错位。SS接触65.495%、支撑49.062%，后者仍低于源61.424%；平均手—物漂移2.5442cm、接缝速度76.8846cm/s（源55.0803）。源历史在已知动作编辑中可用，但当前输出边界与交互仍需显式约束；源条件窗口拟合残差也保持开放。

168动作/168误差数组/28最终检查点/56双历史梯度块核对通过，物体及首尾精确保留，回放/clean恒等/首窗特征及前45原生帧误差0。控制器6163.79s，761360HSI前向含重计算，峰值1.16288GiB；原GPU4–7争用记录保留。干净b4fc08b开始/结束，详见PHASE_2AH_HSI_DNO_HISTORY及紧凑JSON。工程封存，科学重建失败；下一入口先提源条件编辑的输出交界与接触/支撑约束方案，当前没有新编辑或训练。

最终封存验证：1129 passed/4历史skips（200.12s），428条registry有效，git diff --check通过；原生source指标回放误差0，图表已核对。工程诊断完整交付，四格科学重建条件均失败。


## 2026-09-10 — Phase2.38 源条件DNO的输出交界与交互/支撑目标（用户批准）

分支phase/02ai-hsi-dno-constrained。固定原28条强几何来源、R2epoch222EMA和2.37源历史拟合latent，所有窗口使用已知源历史。C正确场景、W错配场景、Q正确场景但HS目标0均由同一latent起点各300步，真实场景SDF及物理条件相同；另G从原生source的69坐标零增量出发300步，作为同物理目标参照。C0/W0零编辑解码保留，报告绝对C−W及(C−C0)−(W−W0)，避免将正确场景拟合起点优势当编辑收益。

固定损失尺度：native28身体5cm；源5cm手—物接触掩码的手部位置1cm；源地面8/8/4/4cm阈值下的四足位置5mm；每个输出窗口交界前6原生帧的28点位置1cm；全帧及交界的source-relative速度分别10cm/s；完整原生HS除max(sourceHS,1)。各项权重1，C/W/Q另含216特征MSE；Q仅去掉HS项，G省去特征项。空活动掩码贡献0。正则保持0，lr.05/warmup50、10DDIM、seed42固定。初始C0/W0和末步C/W/Q逐项测真实latent梯度/损失，记录物理量与尺度，不在看到结果后调权。

这些是优化内的软目标。输出直接评价，物体和首6/末3严格固定；本轮不追加硬投影或按结果挑选候选。手部/足部只按source实际接触和近地状态约束，未接触手与摆动足仍可变化。所有native16、活动手/足误差、前缀位置、窗口特征差、速度、幅度及交互/支撑变化完整保留。

保护点条件：接触/支撑下降<=.002、FS增加<=.01cm，活动手mean<=1cm、近地足mean<=.5cm、交界前缀mean<=1cm，交界/全局平均correction速度<=10cm/s、最大<=30cm/s，root<=10cm/角度<=20deg、全身mean<=5cm及物体/首尾固定。场景信号要求C比W低至少.005sourceHS且编辑增量差<0；整体收益还要求C<=.99sourceHS且优于Q和G、保护通过。全28与375/其余27、10000seed42任务/场景配对区间报告；物体穿透完整列为附加代价。

先375GPU0全长验证，再8GPU余27，保留已有GPU4–7作业、额外峰值5GiB上限；全套authority、clean源码、resolved配置与manifest、每50步检查点沿用，无新增smoke/hash/tools脚本。科学失败不改变预算/阈值；工程门槛是全部28和固定对照封存。协议experiments/protocols/p2_hsi_dno_constrained_s42_20260910.json。

执行前明确：物理位移使用相同24帧（含1帧重叠）FK分块的source/current差，使几何零增量的位移损失和梯度精确为0。只移除源FK舍入差，记录相对原生source最大偏差；不会抵消DNO重建误差。原生评价仍对canonical source。保护逐任务/逐项报告，整体通过要求28条C全部保护通过。

实现验证：14项DNO组件检查通过（1.63s），完整authority1131 passed/4历史skips（193.65s），429条registry有效，9份配置完全解析。53帧全分块加权掩码损失及导数与完整计算匹配，空接触/近地/交界集合贡献0，源FK舍入差下的几何零增量损失/梯度精确为0。共享native物理分块和记录逻辑保持旧默认；完整375正式任务验证实际掩码、原生源HS、C0回放、分项梯度与显存性能。

### 2026-09-10 — Phase2.38 completion：软目标有效下降，保护与HSI额外收益失败

固定28任务/4场景/124窗口/5376帧，9作业均exit0，C/W/Q各300步。source/C0/C/W/Q/G的HS为1.177500/.908022/.537736/.609941/.856475/.296251。C比source改善54.33%，接触66.816%→70.874%，但源地面支撑61.424%→59.431%。C比Q的HS下降.318739，任务与场景区间低于0，支持真实SDF目标的作用；G的HS比C低.241486，两种区间均明确支持G。

C活动手/近地足/前缀误差较C0的2.217/3.060/4.146cm降至1.101/1.344/2.275cm，仍有14/27/28条超限。接缝修正速度、最大修正速度及局部角度28/28超限；C/W/Q完整保护均0/28，G为9/28。G另19条局部角度、16条最大速度超限，均值位置改善不足以保证所有运动保护。

正确/错配最终HS差−.072205，任务区间跨零、场景区间低于0；扣除各自零编辑起点后的差+.448226，任务[.015643,.994816]、场景[.125656,.878377]，登记的编辑增量条件失败。其余27条绝对C−W改善，C−G与保护失败保持。初始latent按正确场景拟合的不对称性明确报告，本协议额外HSI编辑收益尚未建立。

源历史全部窗口输入精确，C0/C输出边界特征MSE均值.007449→.006503。末步C约束与HS梯度20/23个有定义任务反向；Q也未通过保护，因此剩余残差同时涉及当前参数化、固定预算和软目标，不能单归因于HS梯度冲突或推断全局表达能力。

控制器19216.44s（5.338h），2250600次HSI前向含梯度重计算，新增峰值1.11195GiB；GPU4–7争用保留。干净cd77ddf开始/结束；G的23条300步、5条源零梯度按原stationary分支返回，全部输出保留。完整报告PHASE_2AI_HSI_DNO_CONSTRAINED及紧凑JSON记录指标、违反、梯度与执行细节。工程范围交付，科学utility=false；当前固定配置停止，下一方向先提具体方案，训练与469未启动。

最终封存验证：196动作、84最终优化器检查点、140梯度块与140逐窗数组核对通过，源/物体/首尾逐张量精确相等，源及C0回放指标差0。1131 passed/4历史skips（220.14s），430条registry有效，git diff --check通过；图表已核对。完成提交快进至phase/02-mixer，标记exp/p2ai-hsi-dno-constrained-v1。


## 2026-09-10 — Phase2.39a HOIPrior生成器的DNO场景编辑（用户批准）

用户明确并批准：HOIPrior始终生成候选人体—物体动作，HSIPrior提供当前动作的场景条件指导。分支phase/02aj-hoi-dno。本轮固定28条；2.39b完整469另设后续入口，须在另一session审阅通过后单独批准。

冻结P15 online及R2 EMA。新源由固定50步可微HOI DDIM生成并保存噪声，原500步DDPM+ArmB保留为采样参照；G纯几何、C正确HSI、W错配HSI从同一新源/噪声出发，各300步DNO，lr .05/warmup50/去相关0/latent保持.01。跨窗使用当前生成历史并全程反传；HOI固定条件来自既有任务/窗口，HSI在每轮读取当前候选的场景观测，level199、一个配对噪声、raw x0、未来FK24目标权重.25。错配仅旋转场景查询，真实物理场景保持。模型参数冻结，优化变量仅HOI噪声。

人体和物体中间路径均可调整。物理目标保护源物体坐标系手部关系、root局部动作与速度、支撑高度及滑动，约束根/物体轨迹加速度和终点目标，计入原生人体/物体表面SDF及查询域。首2全局粗帧历史保持，后续窗口历史由当前HOI输出传递；最终直接评价解码输出。源世界坐标手足固定和10cm根位移上限退出本新协议，旧协议结果保持原定义。

目标、物理尺度、判断条件、梯度/原生读出、采样差异、固定任务与执行契约见experiments/protocols/p2_hoi_dno_s42_20260910.json。主要判断C是否比同源G/W有额外HS收益，同时保留交互、支撑、连续性、完成数与物景代价；所有逐任务违反和10000次任务/场景配对区间完整报告。HSI刷新目标作为局部代理，额外场景价值由对照建立。既有局部HSI负结果仍有效，本轮检验HOI生成链中的新优化空间。

实现限定mixer组件、既有Hydra诊断入口、一个config override与组件测试；core和专家代码保持。完整authority及registry验证后提交实现，从干净源码运行。完整375任务承担实际功能/性能验证；先核对非零有限梯度、原生输出、源重放、显存和可恢复检查点，再运行其余27。初始分配GPU0–3，每卡任务batch1，额外峰值8GiB、保留至少2GiB实际余量；GPU4–7现有作业保留。持久会话启动，全部输入身份引用既有记录；无额外smoke、摘要机制或工具脚本。

### Phase2.39a implementation verification

23项DNO组件检查通过（2.82s）：人/物共同换帧、跨窗梯度、原生物体插值、刚体改道下的局部身体/手物关系不变、带两帧重叠的原生四输入导数、当前候选正确/错配查询、官方优化器中断恢复一致性。所有源/最终读出使用与优化相同的可微Transformer前向路径；模型权重冻结。物理源参考与当前FK分块保持一致，记录canonical差值；正式源将检验FK24与原生SMPL-X身体/手部的对齐。

完整authority 1140 passed/4历史skips/298既有warnings（205.43s），431条registry有效，Hydra配置完整解析，git diff --check通过。首轮两项恢复测试因调用命令未export INFBAGEL_PYTHON而初始化失败，修正环境后全套重跑通过；两个日志保留在results/hoi-dno-implementation-20260910/。合成插值夹具采用共享float32时间权重的1e-7精度，未改变原生插值。正式375任务承担真实数据功能与满任务batch1性能验证；结果未产生前保持登记的50/300步、权重和科学判据。

### Phase2.39a source-metric preflight failure and correction

首次p2-mixer-hoi-dno-s42-20260910在375源验证处failed封存（25.59s，0优化步）。HOI DDIM源与同噪声重放精确一致，原生FK24核对通过。OS源165.2218017578125，归一化目标误差1.1920928955078125e-7恰为一个float32 epsilon；还原原始量纲后为1.969597360584885e-5，被原始绝对1e-5阈值误判。根因是检查量纲与实际优化目标不同；检查改在归一化单位上执行1e-5，原始及归一化误差同时保存，所有目标、权重、采样、预算与科学门槛保持。失败源及日志保留，r1使用新编号。实现复核同时恢复旧run_body_projection入口的no_grad装饰器，使原入口保持原执行语义。

修正后完整authority再次通过：1140 passed/4历史skips/298既有warnings（203.45s），432条registry有效。日志results/hoi-dno-implementation-20260910/authority-r2.log。归一化源检查的量纲修正与旧入口装饰器恢复在同一实际失败修复提交中完成；r1继续原固定协议。


### 2026-09-11 — Phase2.39a completion：HOI生成器编辑完成，幅度与场景归因仍受限

固定28/4场景/124窗/5376帧，5作业exit0，G/C/W各300步。C的HS5.022809→4.533883（−9.734%），OS29.535301→25.454580（−13.816%），接触65.434%→65.652%，完成22/28保持。总体保护通过、逐任务26/28通过；329支撑及331足滑失败。C−G HS−.108417的任务/场景区间跨零，C−W HS+.000259，正确场景优势失败；OS错配比正确低.200962，两种区间均支持该差异。utility=false，完整469保持后续入口。

身体/物体平均修改3.81/3.74mm，C的root最大位移在329为8.65cm，第二大1.91cm。固定375路径基本重合。C的足滑改善集中于329，其支撑下降2.84pp；此失败分析保留全28主统计。新DDIM源相对DDPM的支撑代理下降6.89pp，单独报告采样差异。

140动作、84最终检查点、140梯度块、124场景查询配对核对通过，全部300步有限，源重放/初始历史精确，位移及范数归约误差0。补齐bool完成率的数值汇总与任务配对区间（全0），不改原封存输出。运行69a676b，控制器9.922h，峰值2.324GiB；首次源检查失败保留。详见PHASE_2AJ_HOI_DNO及紧凑JSON。下一次先审阅编辑幅度限制和HSI读出，保留HOI生成器分工；未启动下一实验。

最终封存验证：24组件检查通过，完整authority1141 passed/4历史skips/298既有warnings（184.24s），433条registry有效，git diff --check通过。全部C的root最大水平偏移2.234cm，逐任务最大水平偏移均值.407cm；方向分解保存在原生结果旁。工程固定28交付gate完成，科学utility=false；提交由exp/p2aj-hoi-dno-v1标记并整合至phase/02-mixer。


## 2026-09-11 — Phase2.40a 指标目标DNO与统一完整编辑链（用户批准）

分支phase/02ak-hoi-dno-metrics。用户在审阅完整469和2.39a后批准固定28对照。冻结P15/R2、原HOI噪声与条件、DDIM50、DNO300、lr.05/warmup50、HSIlevel199/weight.25；G/C/W共享新物理目标。移除latent距离惩罚，以动作空间的交互、支撑与连续性约束保持。完整契约见experiments/protocols/p2_hoi_dno_metrics_s42_20260911.json。

手部由源接触保持改为活动区间3cm表面目标，1cm超差尺度；主手由源接触数/平均距离选择，期间由源接触或物体速度>1cm/s界定，首3帧退出目标。保留另一手源接触，并以物体系手速10cm/s约束连续抓握。加入人体表面对物体原生SDF的负深度和（1m原生sum尺度）；源相邻支撑足速以0为目标，尺度5cm/s，源支撑高度1cm保护。HS/OS沿旧归一化，分别新增逐帧最深点2mm间隔、1cm尺度、权重.1。身体5cm、局部速度20cm/s、轨迹增量加速度.5m/s2、终点5cm、域外目标保留。所有新增目标逐步记录并独立记录源/终点噪声梯度。无按任务编号/物体类别的调参或GT目标。

DDPM_reference、同源DDIM source和G/C/W均执行相同relation20及terminal规则，保存raw/relation/final和终点拒绝候选。最终C主要对同源source_final比较，同时报告旧DDPM_reference_final；各阶段15原生指标及完成率、支撑/运动/交互诊断齐全。几何编辑允许选择既有注册目标最小迭代，DNO固定取300末步，无多噪声挑选。

进展条件为最终C接触至少+2pp、Pbody至少−10%、FS至少−10%同时满足；保护最终完成数、HS/OS深度与帧率、源地面支撑、局部动作和连续性，完整逐任务违反保留。HSI增量及正确场景条件单独报告，禁止用单项HSI失败覆盖指标改进结论。任务/场景10000次seed42配对区间；完整469留作另一session入口。

实现限mixer、既有Hydra入口、一个配置片段及组件测试，core/专家保持。全套authority与registry通过后从干净提交启动；首先GPU0完整375验证新增目标/真实导数/原生链及可恢复50步检查点，再按窗口数在8×3090执行余27。峰值8GiB/卡并保留2GiB实际余量。无新增smoke、哈希机制或tools脚本；正式运行提供功能及性能数据。一次预登记、一次逻辑实现、运行完成后一次封存提交；所有实际失败保留。

### Phase2.40a implementation verification

新增目标进入MetricMotionObjective，复用原生分块反传及HOIDDIM；sourceprediction/噪声须与2.39a逐张量精确一致。接触完整物体网格、人体—物体完整人体网格，HS/OS及间隔沿原生seed42最多10475物体点；后续relation/terminal仍完整物体网格。保留原源手接触95%条件，28源均有接触；1cm源手位置误差仅描述，允许修复抓握。

30项DNO组件检查包含新增掩码补接触、源零编辑的接触/滑动/穿透修复方向、53帧分块四输入导数、原生物体SDF坐标/数值/导数及raw/final基准隔离；官方优化器恢复分别覆盖latent保持0/.01。完整authority1147 passed/4历史skips/298既有warnings，188.81s，434条registry有效，9份正式配置完全解析。首次组件29项通过（4.28s）后补齐保持0的恢复覆盖，由最终全套执行。

日志results/hoi-dno-metrics-implementation-20260911/authority.log。运行路径涉及新增SDF、接触和后处理计算，正式全长375承担真实功能与性能验证，并记录同步耗时、显存和50步可恢复检查点；未添加独立smoke。实现只改mixer/组件测试/本阶段文档和协议，core与两个专家保持。

### 2026-09-11 — Phase2.40a completion：三项指标进展通过，HSI最终场景条件失败

固定28/4场景/124窗/5376帧，9作业exit0，84条优化均300步。C_final相对同源完整链接触65.434%→92.318%，Pbody8.406645→1.682672（−79.98%），FS.139874→.107714（−22.99%），HS1.068814→.394497、OS11.210390→5.971899，完成23→26。登记总体进展/保护通过；逐任务C全保护7/28，FS任务区间跨零及329支撑−18.94pp保留。

C_final−G_final HS+.074915、C_final−W_final+.081916，两种区间均高于0。421的正确HSI输入在20步关系编辑中选择源，第1步目标明显上升，G/W则进一步改善；该后处理差异贡献大部分最终HS反转。正确HSI的OS/FS点估计更好，场景gate仍失败。

按三键对齐July原资产同28条，C_final为11/12项点估计更好，人景帧率31.783%高于31.562%；这是收尾历史系统参照，未改变原门槛，完整469未运行。420动作/84最终检查点/504cadence/140梯度块/124教师查询对/140终点候选核对通过；原噪声/源动作精确，位移/梯度范数归约误差0。控制器6.801h，峰值2.4156GiB。详见PHASE_2AK_HOI_DNO_METRICS及紧凑JSON。下一session入口为冻结C配方的完整469协议与计算预算审阅，本轮保持固定28范围。

最终封存验证：完整authority1147 passed/4历史skips/298既有warnings（214.21s），435条registry有效，git diff --check通过。首次测试因进程内存整理耗时中断（469通过/2跳过、375.20s），用既有进程级THP设置重跑完整套件通过，两份日志保留。运行代码e8c651f保持；完成提交由exp/p2ak-hoi-dno-metrics-v1定位并快进整合phase/02-mixer。本轮固定28工程与指标进展已交付，HSI场景失败及完整469入口保留。


## 2026-09-11 — Phase2.40b 冻结DNO配方的完整469验证（用户批准）

分支phase/02al-hoi-dno469。用户在2.40a封存并建议完整469后批准。目标固定C（HOIPrior+DNO+正确HSI+relation20+terminal），同源G作全量消融；W保留旧28诊断。冻结2.40a全部50/300步、学习率/噪声/教师与物理尺度、掩码、后处理/终点/原生评价，不按开发结果调权。协议experiments/protocols/p2_hoi_dno469_s42_20260911.json。

复用完整28的源、G/C及全部对应阶段/原生指标，保留原e8c651f身份；余441各生成同一ordinal seed42 DDIM源，再执行G/C共882条300步优化。先在19/329/375/421回放旧源与最终latent并执行同一后处理，检验全部输出和指标兼容，随后8GPU分片余441。回放单列，不重复计入469、不重优化旧28。额外W源查询/梯度仍按原诊断保留，新441不生成W轨迹。

正式主表覆盖469/67场景，按scene_name/test_idx/object_name对齐July原469，逐项判断12原生指标点估计是否全部更好，并报告任务/场景10000seed42配对区间。C/G与同源链及DDPM链的对照、全部逐任务保护/失败/候选继续保留；空源接触任务的条件保护为null，手接触保留率按活动任务计算。HSI full469正确/错配轨迹差标为未测，沿用28的失败结论。旧28与余441分层描述，全469具有历史开发使用。

约两天的8卡预算基于旧累计设备时间估算；当前显存余量允许但GPU有其他负载，记录实际争用。每卡taskbatch1、峰值8GiB和2GiB余量，持久controller、50步恢复检查点与clean/resolved/manifest前检沿用；进程级THP设置覆盖测试及运行。全套authority与registry通过后提交实现启动，稳定初段后可交接后台。没有新增tools脚本、smoke、哈希机制或专家/core修改。一个预登记、一个逻辑实现、一个完成提交；实际失败按原生命周期保留。

### Phase2.40b implementation verification

运行与汇总显式支持G/C两臂；原W轨迹缺失时场景因果字段为null。collect_hoi_dno_records从旧运行读取28条且保留原提交/路径，拒绝重复计数；新441与旧28分开计时/分层。主表按三键对齐July469，固定判断C的12指标，不按G更优结果切换主方法。无源手接触任务单独标记并从条件保留率均值中剔除。

36项DNO组件检查通过（4.65s），覆盖两臂/三臂与空接触、缓存去重和不修改旧数据、回放偏移检测、基线身份与固定C判据。完整authority1153 passed/4历史skips/298既有warnings（207.08s），436条registry有效；12份正式配置完整解析。生成器、物理目标、优化器、后处理等10个组件的AST与f13bd81完全一致，登记数值配方逐项一致；修改限定调度/回放/汇总，core/专家保持。

旧4场景28条复用；余63场景441条/1962窗口/85050原生帧按整场景窗口数平衡至8卡，每卡226–249窗口。原生日志最大的任务358为468帧/11窗口。4条正式回放承担真实兼容性与完整链验证，新任务0的50步cadence提供当前共享环境吞吐/内存观察。每步计算路径保持，额外独立性能基准与smoke跳过；运行沿用进程级THP设置。日志results/hoi-dno469-implementation-20260911/。
