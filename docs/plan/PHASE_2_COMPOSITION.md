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
