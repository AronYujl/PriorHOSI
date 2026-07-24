# Phase 1B D2-Z Immutable-GT Near-ground Gating Summary

## Scope and gate decision

D2-Z tested one controlled training change relative to D2-Y: the existing
1,024 multiplier on the same eight FK-routed foot x/z squared velocity
residuals was applied only when the immutable-GT previous sampled foot frame
was below a fixed joint-specific near-ground threshold. Inactive routed entries
used coefficient 1. The 232-D representation, residual target and construction,
architecture, conditions, global mean and velocity weight, other losses,
optimizer, effective batch, fixed 61.44M-window budget, and official evaluator
were unchanged.

The from-random training, fixed internal diagnostic, and single official-438
evaluation completed. D2-Z did not significantly improve paired official foot
sliding over D2-X. It also failed D2-X end-object and contact-F1 protection.
The preregistered classification is therefore
`immutable-gt-near-ground-joint-negative-stop`. All released-baseline absolute
diffusion checks passed, but no checkpoint was selected and consistency
distillation remains unauthorized.

## Implementation and configuration

- `2b818726093fc15ff819e76fd12119eb329343bf` preregistered the
  single-variable D2-Z mechanism.
- `3034510418fdc76e334cb8643c74414ded045726` implemented the isolated
  dataset-derived binary gate, gated velocity reduction, audit, diagnostic,
  target-only evaluator, fail-closed contracts, and tests.
- `1bb5ff5cd71eb67c0697094f7c89b0aed9c0643f` fixed only the observed raw
  aligned-joint shape contract from 24 to 28 after the retained CPU-audit r0
  failure.
- `f67a6437f1ca261ad78f5a8eceab6daabaeb40b5` fixed only smoke-tool model
  construction order after the retained no-forward r0 failure.
- `2634cea35e86cd054c9283fdfddb89fe507dc066` sealed the successful smoke
  result and was the exact clean commit used for formal training.
- `ff5148a7040cc6c9679393557a395e3f147a43b8` fixed only zero-support
  per-sequence reporting under a fresh internal r1 identity.
- `38d7e409a1b6208049d1b4ce358eacfef0dc9f3a` bound the sealed internal r1
  identity/hash into the otherwise unchanged official evaluator.

The gate used foot joints `[7,8,10,11]`, x/z routed components, immutable-GT
previous sampled frames, the full immutable aligned sequence's official floor,
and strict thresholds `0.08/0.08/0.04/0.04 m`. It was boolean and
stop-gradient. Active/inactive coefficients were 1,024/1; the original
87-slot global mean and velocity weight 0.1 remained.

Training stayed random-initialized, FP32, Adam at `1e-4`, effective batch
2,048, with no warmup, scheduler, clipping, weight decay, AMP, or EMA.

## Preserved operational failures

### CPU gate-audit r0

`p1-hoi-d2z-gate-audit-s42-20260724` failed before producing scientific gate
counts because the new helper incorrectly required raw `[16,24,3]` joints
instead of the locked OMOMO `[16,28,3]` source. It used zero checkpoints,
optimizer updates, and CUDA calls. The run is permanently retained; artifact
tree SHA-256:
`11553516ef660118e2e66b8ca9eb0f277c34aa21af10c8fca782a15b4b6cdc9f`.

The exact shape-only r1 completed. Its full-split gate audit had no nonfinite
floor/gate, and the sealed 32-sequence/96-window selection contained
`4,620/5,376` active joint entries, an active fraction of `0.859375`.
Per-joint active counts for joints 7/8/10/11 were
`1,096/1,081/1,211/1,232`. Gate-audit SHA-256:
`d56f1cbc5297b82d768cd396ab1a49c6e33d4101d156c0375501bf32ae055faa`.

### GPU smoke r0

`p1-hoi-d2z-gpu-smoke-s42-20260724` failed before forward/backward because the
smoke-only DataLoader iterator consumed Torch RNG before model construction.
It created no optimizer, performed no update, and loaded/wrote no checkpoint.
Artifact tree SHA-256:
`cdd6934b5cb68fc42a71feee1f165695256376b08044ff3e2ced76806193edd3`.

The exact model-before-iterator r1 reproduced the locked random initial state,
exercised real active/inactive gated data and production forward/backward, and
produced finite required gradients with no optimizer/update/checkpoint action.
Artifact tree SHA-256:
`3a36211aaa3143a965828f37c2cbbc53872188162905dd93fe3017f55a12f148`.

### Internal diagnostic r0

`p1-hoi-d2z-immutable-gt-near-ground-gating-internal-s42-20260724` failed
before any scientific diagnostic output because the reporting layer rejected
three valid fully-active sequences whose inactive per-sequence MSE was
mathematically undefined. It used no official-test sequence, optimizer,
update, checkpoint write/selection, or consistency. Artifact tree SHA-256:
`239fb2bf4a4ce7ff456bea8a0ebd265bcde8b12cef14a8ae0e81ba081b472544`.

The fresh r1 retained all 32 sequence positions, reporting undefined strata as
JSON `null` with exact count zero and leaving all defined/aggregate formulas
unchanged.

These three failures are implementation/operational defects in their stated
local scopes. None indicates a D2-Z training, checkpoint, or official evaluator
scientific defect.

## Experiments and results

### Formal training

- Run:
  `p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724`
- Windows/frames/updates:
  `61,440,000 / 983,040,000 / 30,000`
- Initial model-state SHA-256:
  `ad6980ce1e55a2b30420cb05993fa7b9f431ed674cea58c5795d4c885d52c14e`
- Loss and required gradients: finite/present; AMP overflow skips: 0
- Final validation total/reconstruction/FK/object-surface/velocity:
  `0.0502446 / 0.0419197 / 0.0102728 / 0.00830988 / 0.00381642`
- Throughput: `3,180.18 windows/s`
- Minimum per-rank memory headroom: `21,208,694,784 bytes`
- Cadence artifacts: 20 checkpoints and 80 rank RNG sidecars
- Early/mid/final checkpoint SHA-256:
  `dce01c06b58ac41120307f3fbf8fb4f4892d15140abe583e8202e4cf0a9c48ed` /
  `31ca1874e6da7105ca5a39196ca1f658492108d6695834fc28b89287b231189c` /
  `44c1ff8c8cf4abc2c7312923f64183e1a4a307166d187c9fcaff03abdcc162b6`
- Training tree: 113 files, 7,127,278,269 bytes, SHA-256
  `41de8a4a2b94b225d82d628ca3d074408b33619550f8809e1d6576ef2b1f4726`

The training loaded no released, author, D2-V, D2-X, D2-Y, prior, resume, or
EMA checkpoint/state. All restored-component/state lists were empty.

### Fixed internal diagnostic r1

- Run:
  `p1-hoi-d2z-immutable-gt-near-ground-gating-internal-r1-s42-20260724`
- Protocol: sealed 32-sequence/96-window selection, D2-X/D2-Y/D2-Z
  early/mid/final online checkpoints, timesteps 0/249/499, identical clean
  windows, noise, timestep, and condition-dropout replay
- Final active routed-residual RMS for D2-X at 0/249/499:
  `0.007179 / 0.008951 / 0.010113`
- Final active routed-residual RMS for D2-Y:
  `0.005561 / 0.007029 / 0.007667`
- Final active routed-residual RMS for D2-Z:
  `0.005675 / 0.007054 / 0.007912`
- Final D2-Z gated-vs-uniform gradient-norm retained fraction:
  `0.7767 / 0.7833 / 0.8587`
- Final D2-Z gated-vs-uniform gradient cosine:
  `0.9949 / 0.9592 / 0.9483`
- Final D2-Z gated-routed/FK cosine:
  `0.6908 / -0.5110 / -0.1521`
- Internal tree: 6 files, 515,725 bytes, SHA-256
  `039894526a9044865ea0fcfbdee1ba9c51f148a74860048ba16fdd6b1f31e960`

The internal artifact passed every record/finite/provenance check. It was
diagnostic-only: no official test, optimizer, update, checkpoint write/selection,
or consistency operation occurred.

### Official evaluation

- Run: `p1-hoi-d2z-native-eval-s42-20260724`
- Protocol: official 438 sequences, three windows/sequence, 500-step unguided
  diffusion, fixed final online weights
- Primary control: sealed D2-X r1 records, reused without regeneration
- D2-Y records: reported only as a non-selection mechanism comparator
- Target MPJPE/end-object/xy/object-translation:
  `12.2655 / 4.4567 / 3.8216 / 16.3945`
- Target foot sliding/contact F1:
  `0.363433 / 0.630798`
- Target hand/human penetration loss:
  `0.219640 / 3.447872`
- D2-X minus D2-Z foot-sliding mean difference:
  `-0.000423015`
- Paired bootstrap 95% CI:
  `[-0.0221524, 0.0209344]`
- D2-Z minus D2-X contact-F1 mean/95% CI:
  `-0.00662759 / [-0.0274279, 0.0137967]`
- D2-Z/D2-X end-object ratio 95% CI:
  `[1.12841, 1.25042]`
- MPJPE, xy, object-translation, and both penetration protection checks passed.
  End-object and contact-F1 protection failed.
- The fixed 181-sequence penetration mask matched exactly.
- Every released-baseline absolute MPJPE, end-object, xy, object-translation,
  foot-sliding, and contact check passed.
- Evaluation tree: 15 files, 389,634 bytes, SHA-256
  `9613bb8762dc1ba67e29c068dee966461cfa4ac4284a2d84c4dc671625e13bfe`

The foot CI lower bound was not greater than zero and protection was not fully
preserved, so the preregistered classifier returned
`immutable-gt-near-ground-joint-negative-stop`.

## Scientific interpretation

### Verified facts

- D2-Z completed the exact from-random fixed-budget training contract without
  prior-state load or numerical instability.
- On the fixed internal diagnostic, D2-Z retained most of D2-Y's routed
  residual improvement, but its gated gradient was highly collinear with the
  uniform D2-Y gradient and noisy-step routed/FK conflict remained.
- D2-Z did not produce a statistically certified official foot-sliding
  improvement over D2-X.
- It failed D2-X end-object and contact preservation while passing every
  released-baseline absolute diffusion check.
- All final lifecycle, provenance, normalization, finite-value, penetration-mask,
  and artifact contracts passed. No definite scientific implementation defect
  was found.

### Evidence-based inference

The sealed selection's 85.94% active occupancy made this binary gate a mild
change from uniform D2-Y amplification: it retained 78–86% of gradient norm at
cosine 0.95–0.99. Restricting the signal to immutable-GT near-ground support
therefore did not create a strong enough directional change to remove the
official-semantic transfer gap or the noisy-step FK conflict. The D2-Z point
estimates remain compatible with the registered sequence-level uncertainty.

### Unresolved questions

It remains unknown whether the leading limitation is immutable-GT
teacher-forcing versus rollout/predicted-state semantics, the official metric's
nonlinear near-ground/contact definition, or shared-objective gradient conflict.
D2-Z does not authorize a predicted gate, soft gate, zero-velocity target,
PCGrad, multiplier/threshold sweep, or any bundled intervention.

## Verification

- Authority full suite passed 286 tests before official workload publication.
- Worker D2-Z targeted suite passed 17 tests and registry validation at the
  exact evaluation commit.
- The training produced and independently hashed all 20 checkpoints and 80 RNG
  sidecars.
- `tools/experiment.py start`/`finish`, clean committed Git objects, fully
  resolved configs, same-context machine preflights, and run-local registrations
  were used for every reportable lifecycle.
- Worker-initiated `rsync -aH --partial` recovered all immutable artifacts.
  Worker and authority tree hashes matched for every returned tree.
- The failed gate-audit, smoke, and internal r0 identities remain registered and
  were never reused or overwritten.

## Tracked and external artifacts

- Compact aggregate:
  `experiments/results/p1_hoi_phase1b_d2z_immutable_gt_near_ground_gating_s42_20260724.json`
  (SHA-256
  `c2a0ff494784fac6d42485d189646b8b9618205f8fc8ed6608270ff871c16af4`)
- Training staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2z-immutable-gt-near-ground-gating-s42-20260724`
- Internal-r0 staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2z-immutable-gt-near-ground-gating-internal-s42-20260724`
- Internal-r1 staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2z-immutable-gt-near-ground-gating-internal-r1-s42-20260724`
- Evaluation staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2z-native-eval-s42-20260724`
- Training manifest/metrics/resolved/preflight/run-local-registry SHA-256:
  `39a400060c01056f03c03eff28a6ba83f0e0b88b520394a43a72eaf0903b28df` /
  `84b682c4ec78ce80402538c6304a419f8bfcf879b7bf3163ced68d8032d47d09` /
  `b81490a2941193679b9d9c1b9e85713884ef27ec2ed8076bb06c39cb6d202c26` /
  `a94293326f41243b89a9911d3d1c94a34755bd41cc6d70baf8a0f2c2dd83c38b` /
  `f5e2848696009542ad50353de8d572c9498c46945c8d4a69bf438410c8659188`
- Internal-r1 manifest/metrics/preflight/run-local-registry SHA-256:
  `d5f3d3bd04bdd290ec5dbd599d706a99daac78e1ec51259b88f9280fc9a0043d` /
  `0540afa33b485f3a893973d827fe0c48bfca08df3e0b3fdd54fa1f14ce9256e3` /
  `abca0349a533bf288f5a6ef80bfee447423c13563a7c9875ff2d098959e5626b` /
  `b09e80dcc370f31a2aa43f9a39d571ee01477bc5999236a99ea002dd0f1f4f50`
- Evaluation manifest/metrics/aggregate/per-sequence/resolved/preflight/
  run-local-registry SHA-256:
  `c134947136eee8a867222c55584ad744f6a8cffa72742b39a77980f541e50c6e` /
  `c20738824f0475294e42551121fd7796c2041fd48949e4630e012ad2d4959ae3` /
  `fb58a5ab3bd5ad0336ce02ff9a15cd7d97af8446599b147c9e2c806208a56162` /
  `9f0f0e65bd0eaa4fe3ec1f495f6e4a4489c88d842256dccc3a6b9b57a1e9113f` /
  `86610e7c646d08474072b67826ad2a7268b7c8a20cefe93e217f446da3f244ed` /
  `981c21f0d565e99ee1b9d9ae5e52667de650ddf0986e6f815448edf1a151b4af` /
  `80c908260c73d1c5e27e2f4da4c76e8927575ae329be71d0d0da9171b080c9b9`

No merge commit or immutable tag was created because the D2-Z gate failed.

## Exact next entry point

D2-Z is closed as a controlled negative result. Its checkpoint is not
selectable and must not initialize another run. A new session must begin by
reading this summary and `docs/EXPERIMENT_PLAN.md`, verifying the next available
Phase 1B identifier from the append-only registry, and performing a new
read-only mechanism audit before proposing one dated, single-variable,
from-random HOIPrior subphase.

Do not reuse or change the D2-Z gate, rerun its official evaluation, select an
intermediate/final D2-V/D2-X/D2-Y/D2-Z checkpoint, resume any of those runs,
bundle contact/penetration/sampler changes, start consistency distillation, or
enter HSIPrior or Mixer without a separately preregistered phase and explicit
user authorization.
