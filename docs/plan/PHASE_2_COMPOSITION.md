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
