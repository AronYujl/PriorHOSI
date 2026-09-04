# Phase 2 — composing the two expert priors

Status: implementation started 2026-08-29. No composed result exists yet, because
HSIPrior has no settled checkpoint (P17-OC is training on `phase/01c-hsi` from
589ac7f). Everything below that does not need an HSI checkpoint is built and
tested; everything that does is specified and blocked.

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
