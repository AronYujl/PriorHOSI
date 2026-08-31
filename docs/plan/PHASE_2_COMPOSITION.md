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
