# Phase 1B D2-AD：human-local full-mesh BPS coordinate repair

## Scope and final outcome

D2-AD0 tested one fixed follow-up to the sealed D2-AC0 locality-negative result. It
kept the D2-AC architecture, parameter count, adapter placement, role queries, token
count, loss, optimizer, training budget, sampler and evaluator unchanged. The only
scientific change was to rebuild the adapter-only local-object BPS against each
immutable full rest mesh in the current Y-up human-window-local frame, rather than
mixing the raw BPS basis frame with the local human state.

The approved lifecycle is complete: implementation, authority CPU contract,
registered GPU smoke, one 61.44M-window from-random training, the fixed three-way
internal diagnostic, the fixed official 438-sequence native evaluation, artifact
recovery and hash verification. The final preregistered classification is:

`local-frame-interaction-adapter-locality-negative-stop`

The adapter was strongly used, but its fixed local correspondence was not causally
necessary. More importantly, the repaired representation transferred worse than the
sealed D2-X control on contact F1 and recall and violated several protection gates.
The fixed final-online checkpoint is not selectable. D2-AD1, a longer-budget run,
checkpoint selection, consistency, HSIPrior, Mixer, sweeps and any new HOIPrior
direction were not started.

## Commits and fixed mechanism

- D2-AD0 plan-only registration:
  `ccc023f44056a056131c730ff39a2dfae447505b`.
- Source/config/tests/lifecycle implementation:
  `07c41cfc4d07b75cf4e34d7628b54bd5d654cefe`.
- Authority CPU completion record:
  `636296967546a4507a45adf9fc53c186223eb4c9`.
- Registered GPU smoke record:
  `cff909188685f48b177c03de1c86964e4c43ec8b`.
- Formal training record:
  `eb61c3ac82cf162303e9ba8a2c43d98c0eed01b7`.
- Fixed internal diagnostic record:
  `82d8cbd453dbbfa2f5214eec8320628ea1fd2370`.
- Completion documentation/result/registry commit: identify the commit that adds
  this file with
  `git log --format=%H -- docs/phase_summaries/PHASE_1B_D2AD.md`.

The fixed implementation contract was:

- base/adapter/total parameters:
  `29,673,448 / 349,697 / 30,023,145`;
- no new trainable parameter relative to D2-AC0;
- 16 deterministic 10-D local tokens and the same three role queries;
- the same layer-4-to-layer-5 ReZero cross-attention adapter;
- immutable BPS SHA-256:
  `fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`;
- Y-up basis tensor SHA-256:
  `02b4f8f3510e723174010a823630f663ddda9875ad82a2f8de807d2bdccebd7d`;
- assignment SHA-256:
  `b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f33607959fddeecd`;
- 13-object rest-mesh manifest SHA-256:
  `ce8328ef2bf873a79d74fb5fd20cc488551a20d56fe5c5ecabf609824b0654d1`;
- object mapping SHA-256:
  `424fc96102c576a1d11b0824cc0ee616d52cd9e39524819f49b207d1598fe41b`;
- current first-frame human/object pose at training and generated two-frame history at
  rollout; no future GT, stored per-window local BPS, mesh subsample, SDF/voxel
  condition or category embedding;
- global BPS token, 232-D API, D2-X trunk/routing/losses and official evaluator
  preserved.

## Authority CPU contract and GPU smoke

Authority run `p1-hoi-d2ad-cpu-contract-s42-20260728` completed as
`cpu-contract-passed`:

- 352 tests passed;
- exact hashes, local-coordinate equivariance, worker-query determinism,
  dataset/evaluator builder parity and all 13 meshes passed;
- exact parameter count and `[B,16,232]` API passed;
- shared-trunk `eval()`, `alpha=0` parity max-abs difference was `0.0`;
- initial alpha gradient and all test-only activated adapter gradient groups were
  finite and nonzero;
- fixed correspondence permutation changed the CPU output
  (`max-abs 0.0004273951`);
- zero/constant/extreme inputs, provenance rejection, HSIPrior storage independence,
  Mixer clean-output contract and forbidden-path static scans passed;
- no CUDA, optimizer update, checkpoint load/write or selection occurred.

The CPU artifact tree has 3 files / 32,306 bytes and SHA-256
`514163cc45801253f19dbb6e1789464e791f59a00aa6f1b44cdadf9f348eb7ce`.

Registered smoke `p1-hoi-d2ad-gpu-smoke-s42-20260728` ran on
`infbagel-4gpu/node01`, `cuda:0`, real-data batch 8 and timesteps
`0/249/499`. Four RTX 3090 devices were visible; random-init losses, alpha gradient
and test-only adapter gradients were finite. Synchronized peak allocated/reserved and
headroom were `252,609,024 / 304,087,040 / 24,991,956,992` bytes. It created no
optimizer, update or checkpoint activity. Its 7-file / 60,861-byte recovered tree
matches worker and authority with SHA-256
`85ef57f3874ab113d4cac75b813259fb61ae5cff5d1b24ed9078b924223c621a`.

## Formal from-random training

Run: `p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728`.

- worker Git object:
  `cff909188685f48b177c03de1c86964e4c43ec8b`;
- initial model-state SHA-256:
  `260d12ed7e60774e9eb4280f3abc580a7af8b9d8a36d3c75b83a9bbf0021a1bc`;
- random initialization, no source/weight-init/resume/EMA checkpoint;
- fixed split SHA-256:
  `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`;
- 4×RTX 3090, per-GPU batch 512, effective batch 2,048, accumulation 1;
- exactly `61,440,000` windows / `983,040,000` frames / `30,000` updates;
- wall time `47,890.633 s`, throughput `1,282.923 windows/s` /
  `20,526.770 frames/s`;
- mean training total loss `0.0448805`;
- final validation total/reconstruction/FK/object-surface/velocity:
  `0.0483891 / 0.0403080 / 0.0109891 / 0.00818448 / 0.000111427`;
- learned alpha/gate:
  `0.10238598 / 0.10202970`;
- 20 cadence checkpoints and 80 rank RNG sidecars;
- first resumability checkpoint SHA-256:
  `7a27edd434490579e54852bfe3321406948762aed72b9987e02c163a6918bbd9`;
- fixed final-online checkpoint SHA-256:
  `f527d970243a42a1534b8db4437cd09dbc25334c832c3a13eb011f81db101c06`.

Manifest/metrics/resolved/preflight/training-state SHA-256:

`bccfd482...767f` / `3d30f83d...1ba6` / `fe6ae9e9...aa7` /
`7bce4a57...90e7` / `a82dfc6d...29f7`.

The complete recovered tree has 114 files / 7,211,938,158 bytes and matching
worker/authority SHA-256
`d694962309735ecae12f4480d4dcb52c8d191a9a453603fefd8e5f4bbd18b656`.
The training log emitted `terminate called without an active exception` during
post-completion process cleanup, after exit code 0, final state, metrics and all
artifacts had been durably written and rehashed. It is retained as cleanup noise, not
suppressed or treated as a scientific failure.

## Fixed internal causal diagnostic

The intended internal identity
`p1-hoi-d2ad-local-frame-interaction-adapter-internal-s42-20260728` stopped before
manifest/workload because preflight received the evaluator asset directory instead of
the pinned CHOIS Git checkout. The one-file failure tree remains preserved
unmodified: 5,348 bytes, SHA-256
`d0eda6ede4e692acb2ca52ed8286ba4e122b0fc1e4edc2845946d03714898a47`.

Retry `p1-hoi-d2ad-local-frame-interaction-adapter-internal-r1-s42-20260728`
reran from scratch with the corrected checkout path. It loaded only the fixed
final-online checkpoint and used the sealed D2-O 64-sequence/192-window cohort,
selection SHA-256
`1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`.
The three 500-step paths shared initial latent, every posterior-noise draw, condition,
ordering and history restoration. Statistics used sequence units, seed 42 and 10,000
paired bootstrap replicates.

Primary registered comparisons:

| comparison | paired point | sequence-bootstrap 95% CI | gate |
|---|---:|---:|---|
| full − gate-ablated direct-hand union 5-cm F1 | +0.6274720 | [0.5403343, 0.7116109] | pass |
| gate-ablated − full GT-contact distance (cm) | +97.93614 | [87.31619, 108.58261] | pass |
| full − locality-permuted direct-hand union 5-cm F1 | +0.0135183 | [-0.0062331, 0.0342150] | fail |
| locality-permuted − full GT-contact distance (cm) | +0.143274 | [-0.057880, 0.354987] | fail |

Representative full / gate-ablated / locality-permuted aggregates:

| metric | full | gate ablated | locality permuted |
|---|---:|---:|---:|
| direct 5-cm contact F1 | 0.76410 | 0.00753 | 0.74856 |
| MPJPE (cm) | 12.3010 | 220.9188 | 12.3401 |
| object goal (cm) | 93.2937 | 324.6356 | 93.3071 |
| pelvis goal (cm) | 5.1046 | 74.6775 | 5.1097 |
| FS | 0.82138 | 1.14343 | 0.81814 |

Thus the adapter is indispensable as a whole, but rotating the local delta-statistic
correspondence by eight clusters leaves behavior statistically indistinguishable.
Attention entropy remains descriptive and cannot override that causal failure.

The 13-file / 24,596,627-byte recovered tree matches worker and authority with
SHA-256
`4b80a78745de4d3fecc23399f023d736d4b5ff1f9e7d12e043e70e6bf27055e3`.
Metrics/manifest/paired-noise/attention/local-BPS appendix SHA-256:

`3fcb4835...fe9b` / `a69644a6...7647` / `775826e9...b2a7` /
`a317f58c...f7a` / `c55c69ae...6b59`.

## Fixed official native evaluation

Run: `p1-hoi-d2ad-native-eval-s42-20260728`.

The unchanged official evaluator ran 438 sequences × 3 windows with 500 unguided
steps, final-online weights, seed 42 and 10,000 paired sequence bootstraps. CFG,
dynamic perception, guidance, scene conditioning and consistency were off. The sealed
D2-X aggregate/per-sequence artifacts were reused without regeneration. D2-AC was
included only as a sealed descriptive comparison and did not affect selection.

Target and sealed D2-X point estimates:

| metric | D2-X control | D2-AD0 | D2-AD − D2-X / ratio |
|---|---:|---:|---:|
| end-object (cm) ↓ | 3.7402 | 4.2373 | 1.1290× |
| Txy (cm) ↓ | 4.0505 | 4.8036 | 1.1859× |
| FS ↓ | 0.36301 | 0.42539 | 1.1718× |
| contact precision ↑ | 0.78806 | 0.76795 | −0.02011 |
| contact recall ↑ | 0.59445 | 0.53300 | −0.06146 |
| contact F1 ↑ | 0.63743 | 0.58687 | −0.05055 |
| contact coverage | 0.47655 | 0.43497 | −0.04159 |
| Pbody ↓ | 3.8691 | 3.4625 | 0.8949× |
| hand penetration ↓ | 0.24536 | 0.21656 | 0.8826× |
| MPJPE (cm) ↓ | 12.0508 | 12.3847 | 1.0277× |
| Troot (cm) ↓ | 8.1701 | 9.2747 | 1.1352× |
| Tobj (cm) ↓ | 15.9940 | 16.4076 | 1.0259× |
| Oobj ↓ | 1.03094 | 1.01478 | 0.9843× |

Native gate evidence:

- contact F1 difference 95% CI:
  `[-0.0713216, -0.0293031]`;
- contact recall difference 95% CI:
  `[-0.0852768, -0.0377657]`;
- contact precision difference 95% CI:
  `[-0.0418402, 0.00175924]`;
- released contact-F1 gap closure:
  `-0.562760`, below the required `0.25`;
- end-object/Txy/FS/Troot ratio CI upper bounds:
  `1.19830 / 1.23695 / 1.25273 / 1.16017`, each above `1.10`;
- native transfer, protection and released-95% effectiveness gates all failed;
- the sealed 181-sequence penetration finite-mask contract passed, sequence-ID
  SHA-256
  `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`.

The evaluator did not generate FID, Matching, R-Precision or Diversity; those fields
were not synthesized or backfilled. Metrics/manifest/aggregate/per-sequence SHA-256:

`aac2919e...d59b` / `79127f64...cd4` / `dab67e52...c27` /
`c6e15ab7...460`.

The recovered 16-file / 2,416,785-byte tree matches worker and authority with
SHA-256
`6d0bcf47eac49aaf1a10341d81bc8d4f1a518ed86344fd145283b17c236c7d0c`.

## Interpretation

The registered evidence rejects the narrow D2-AD hypothesis: the earlier locality
failure was not primarily caused by the BPS/world/local coordinate mismatch. The
coordinate repair was real and passed equivariance/build parity tests, but it neither
made correspondence causal nor improved native transfer.

The strongest supported interpretation is that the adapter learned a high-leverage
global conditioning/residual pathway rather than a part-to-local-surface mechanism:

1. Setting the whole gate to zero destroys the generated trajectory, proving adapter
   use.
2. Permuting cluster delta correspondence leaves the trajectory nearly unchanged,
   disproving dependence on the registered local identity relation.
3. The global BPS token and object-token set remain available, so cross-attention can
   summarize object/pose state in a largely permutation-insensitive way.
4. Native contact and goal/locomotion transfer worsened despite stable training and
   good validation loss, so teacher-forced denoising fit is not evidence of rollout
   interaction quality.
5. The failure is not evidence that a different width, gate, token count, placement,
   budget or threshold would succeed; no such sweep was run, and the fixed result
   provides no basis for selecting one.

This closes the coordinate-contract direction. Any subsequent route toward a
baseline-level HOIPrior needs a new dated hypothesis whose intervention makes the
desired hand/object correspondence structurally identifiable and whose diagnostic can
distinguish local interaction from a generic conditioning shortcut. That is a new
research direction and is not authorized by this D2-AD0 closure.

## Verification and artifacts

All project Python commands used the verified `infbagel` environment. The final
authority closure runs:

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2ad -v
"$INFBAGEL_PYTHON" tools/experiment.py validate
git diff --check
```

Tracked compact result:

`experiments/results/p1_hoi_phase1b_d2ad_local_frame_interaction_adapter_s42_20260728.json`

Large checkpoints, RNG states, generated motions and per-sequence outputs remain
outside Git in the recovered staging trees named above. Nothing was overwritten or
deleted.

## Exact next entry point

Stop D2-AD0. Do not select or resume its checkpoint and do not start D2-AD1,
consistency, HSIPrior, Mixer, any parameter/token/placement/role/budget sweep or any
new HOIPrior mechanism. A next direction requires a new dated plan, an append-only
registry hypothesis and explicit user authorization.
