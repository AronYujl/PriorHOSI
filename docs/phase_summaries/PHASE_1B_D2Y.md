# Phase 1B D2-Y Routed-foot Residual Amplification Summary

## Scope and gate decision

D2-Y tested one controlled training change relative to D2-X: the same eight
FK-routed foot x/z squared velocity residuals received a fixed multiplier of
1,024 before the unchanged global mean. The 232-D representation, routed
residual construction and target, conditions, architecture, global velocity
weight, all other losses, optimizer, effective batch, 61.44M-window budget, and
official evaluator were fixed.

The from-random training, registered internal diagnostic, and single official-438
evaluation all completed. D2-Y significantly reduced the teacher-forced routed
residual at both registered noisy timesteps, so the internal mechanism gate
passed. It did not significantly improve paired official foot sliding, and the
D2-X end-object and contact protection checks failed. The strict classification
is therefore `routed-foot-amplification-transfer-negative-stop`. No checkpoint
was selected and consistency distillation remains unauthorized.

## Implementation and configuration

- `41767cf73d42d3b90a8aa343e84ffa1a597dbe14` introduced the D2-Y
  weighted reduction, config, fail-closed contracts, internal diagnostic,
  target-only evaluator, tests, plan, and registry preregistration.
- `adbad074cfae43e921a5ffa41f9850c130758404` bound the not-yet-started
  internal and evaluation lifecycles to their real 2026-07-24 date. It changed
  only those identities; the completed 2026-07-23 training and every scientific
  variable remained unchanged.
- Routed joints were `[7, 8, 10, 11]`; routed components were x/z. Their
  squared-residual coefficients were 1,024, the other 79 coefficients were 1,
  and the original 87-slot global mean and velocity weight 0.1 remained.
- Training stayed random-initialized, FP32, Adam at `1e-4`, effective batch
  2,048, with no warmup, scheduler, clipping, weight decay, AMP, or EMA.

## Experiments and results

### Operational provenance

Training legitimately started under
`p1-hoi-d2y-routed-foot-amplification-s42-20260723` on 2026-07-23 and crossed
midnight. Before either post-training manifest, checkpoint load, or GPU workload,
the real-date amendment replaced the internal/evaluation identities with
2026-07-24 IDs. The superseded
`p1-hoi-d2y-native-eval-s42-20260723` identity was never created or reused.

One evaluation-preparation shell command had a quoting error before manifest
creation and before any checkpoint load or GPU work; it produced only a
disposable resolved draft. This was not a reportable lifecycle, did not consume
the registered evaluation, and did not change its run ID or protocol.

### Training

- Run: `p1-hoi-d2y-routed-foot-amplification-s42-20260723`
- Windows/updates: 61,440,000 / 30,000
- Loss and required gradients: finite/present; AMP overflow skips: 0
- Cadence artifacts: 20 checkpoints and 80 rank RNG sidecars
- Final checkpoint SHA-256:
  `8734431f89cf8739283828d5fb683212ca43143ae3482ad0473f6ed5717eb7a7`
- Final validation total/reconstruction/FK/object-surface/velocity:
  `0.0508121 / 0.0420828 / 0.0103069 / 0.0084531 / 0.00570573`
- Throughput: 3,250.79 windows/s
- Minimum per-rank memory headroom: 21,212,889,088 bytes
- Training tree: 112 files, 7,128,453,002 bytes, SHA-256
  `177eb44fa53ee46518a714f04ee2fe864aa2a1d755f4377f3fd47fa8e40bf0f8`

The initial model-state SHA matched the registered random initialization, every
restored-component/state list was empty, and the final optimizer state was at
step 30,000 on all ranks.

### Internal mechanism diagnostic

- Run:
  `p1-hoi-d2y-routed-foot-amplification-internal-s42-20260724`
- Protocol: sealed 32-sequence/96-window selection, D2-X/D2-Y
  early/mid/final online checkpoints, timesteps 0/249/499, identical clean
  windows, noise, timesteps, and condition-dropout replay
- D2-X minus D2-Y routed-residual MSE at timestep 249:
  mean `2.85917e-05`, paired bootstrap 95% CI
  `[2.12425e-05, 3.62503e-05]`
- D2-X minus D2-Y routed-residual MSE at timestep 499:
  mean `3.13497e-05`, paired bootstrap 95% CI
  `[2.39972e-05, 3.95998e-05]`
- Both registered CI lower bounds were greater than zero; the internal gate
  passed.
- Final D2-Y routed/FK gradient cosine at timesteps 0/249/499:
  `0.8037 / -0.5001 / -0.4522`
- Internal tree: 6 files, 1,493,189 bytes, SHA-256
  `d57e48d5d4285b4f42ab090e2aef0b1ee641217ea960382ec0b60c6bf0e8d05f`

The diagnostic used no official test sequence, created no optimizer, performed
no update, wrote no checkpoint, and did not select a checkpoint.

### Official evaluation

- Run: `p1-hoi-d2y-native-eval-s42-20260724`
- Protocol: official 438 sequences, three windows/sequence, 500-step unguided
  diffusion, fixed final online weights; sealed D2-X records were reused without
  regeneration
- Target MPJPE/end-object/xy/object-translation:
  `12.1246 / 4.8506 / 3.9676 / 16.3290`
- Target foot sliding/contact F1: `0.3572 / 0.6351`
- Target hand/human penetration loss: `0.2184 / 3.4353`
- D2-X minus D2-Y foot-sliding mean difference: `0.00585353`
- Paired bootstrap 95% CI: `[-0.0162119, 0.0279164]`
- D2-Y minus D2-X contact-F1 95% CI:
  `[-0.0231844, 0.0184971]`; its lower bound missed the registered `-0.02`
  threshold.
- D2-Y/D2-X end-object ratio 95% CI:
  `[1.22629, 1.36180]`; its upper bound exceeded 1.10.
- MPJPE, xy, object-translation, and both penetration protection checks passed.
  The fixed 181-sequence penetration mask matched exactly.
- Every released-baseline absolute MPJPE, end-object, xy, object-translation,
  foot-sliding, and contact check passed.
- Evaluation tree: 15 files, 1,631,051 bytes, SHA-256
  `3e81a59fb1a97e7043e856fda5c502bae3683f24a179b0423143fe910837bdc0`

Because the internal gate passed but the official foot CI included zero, the
registered classifier selected
`routed-foot-amplification-transfer-negative-stop`. The additional end-object
and contact failures are retained as protection evidence; the classification
was not changed post hoc.

## Scientific interpretation

### Verified facts

- Fixed amplification was absorbed by optimization and significantly reduced
  the registered teacher-forced routed residual.
- It did not yield a statistically certified paired official foot-sliding
  improvement.
- It failed D2-X end-object and contact preservation while retaining all
  released-baseline absolute capability checks.
- All lifecycle, provenance, normalization, mask, finite-value, and artifact
  contracts passed. No definite implementation defect was found.

### Evidence-based inference

Global-mean dilution was real but is not a sufficient explanation of D2-X:
removing that weakness improved the surrogate without certified production
transfer. The remaining evidence favors a semantic gap between the temporal
surrogate and the nonlinear near-ground official metric, together with
late-noisy-timestep conflict between the amplified routed objective and
FK/object capability. The smaller D2-Y foot-sliding point estimate remains
compatible with the registered sequence-sampling uncertainty.

### Unresolved questions

It remains unknown whether an official-semantic near-ground training target can
improve foot sliding without the end-object/contact tradeoff, or whether a loss
geometry that resolves routed/FK conflict can transfer to rollout. D2-Y does not
authorize either direction.

## Verification

- Authority full suite passed 268 tests before workload publication.
- Worker full suite passed 268 tests with the two documented real-LINGO tests
  skipped under the HOI-worker role.
- The training produced all 20 registered checkpoints and 80 corresponding RNG
  sidecars; named checkpoint and artifact hashes were recomputed.
- `tools/experiment.py start`/`finish`, clean committed Git objects, captured
  resolved configs and machine preflights, and run-local registrations were used
  for all three lifecycles.
- Worker-initiated `rsync -aH --partial` recovered all immutable artifacts. The
  worker and authority tree hashes matched for training, internal diagnostic,
  and official evaluation.
- The tracked registry validated 129 records after completion, and the final
  authority suite passed all 269 tests.

## Tracked and external artifacts

- Compact aggregate:
  `experiments/results/p1_hoi_phase1b_d2y_routed_foot_amplification_s42_20260724.json`
  (SHA-256 `0bc2fffd4304bb3411176cf355dacddfe731e9f1d46eb01cc9bfefe3c215f875`)
- Training staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2y-routed-foot-amplification-s42-20260723`
- Internal staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2y-routed-foot-amplification-internal-s42-20260724`
- Evaluation staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2y-native-eval-s42-20260724`
- Training manifest/metrics/resolved/preflight/run-local-registry SHA-256:
  `487c08a62e3825930cbb1e19e3a6aadb17e3a6c5e9233e48ca8114aacdaa41cb` /
  `a424e11f36d659593e0ec65161e2ced9f71eebf852a5eba8b7d6eaa5fcf1eb16` /
  `5e215510a7552204ad2f44acab22aab4cc853136490a1afdd0c10c9172e56adc` /
  `f0bd247ecf771913fe453f103ffeb8d7ee5d5f66f0814aa128daed4c41577833` /
  `fe2c78389b29458e125864fbbed4fd9183bcedbdbe0c1401b0ad1ae138989eb7`
- Internal manifest/metrics/preflight/run-local-registry SHA-256:
  `9ebb4a5fd4805533c966dc7c2496cccad952a8ca02ad4ddfc651c63615b9c83a` /
  `4e8915993f9a87c31eabf3e861e30304d24f0dd74b6906e01280a561c8e2c8dd` /
  `85de60eb3930e61449a771898d102f072a3cb0800aebf5fe90c1c0b44c67d8fb` /
  `ffa0cb1fa517d8318bfaaed4ae5492ed0f2b48fe742ad351d9eeb8615baa54d2`
- Evaluation manifest/metrics/aggregate/per-sequence/resolved/preflight/
  run-local-registry SHA-256:
  `8bba4843be5f30c3eb30583a673801853aeb50ccd898eff84cf328981d9b14b9` /
  `a24223cd7e70dde41c49609f1c2eac0470d863ecba1c5866b19d9c4da5888537` /
  `776e6c35acdaa190ffcbab047b170ed4ab559c23f454714c31ad980db4dd8c70` /
  `ea2cde99372392c5f16446708e3acf3789a68be9f1b7cc95134fd45390b12c02` /
  `1d9a0db505eabccbd7a4184378075ac2dabc25323f9292677c06c0f0c21e0a0c` /
  `3d7c8a2aa791b7c6b08c3f3cfad742ce714495b6541b712afb69cc5e08efc6c9` /
  `a9fd6f13e5f58c0cec95238469c69ba54877e818982f94ac5b5a92159e66cfa7`

No merge commit or immutable tag was created because the D2-Y gate failed.

## Unresolved risks and exact next entry point

D2-Y rules out “the routed signal was merely too diluted” as a sufficient
mechanism, but it does not establish which semantic or gradient-conflict remedy
would work. The final checkpoint is not selectable and must not initialize
another run.

The next session must begin by reading this summary and
`docs/EXPERIMENT_PLAN.md`. It must perform a new read-only mechanism audit and
append a dated plan/registry hypothesis before any new HOIPrior code or workload.
It must not increase or sweep the D2-Y multiplier, resume D2-Y, post-hoc change a
gate, bundle penetration/contact/sampler changes, select any D2-V/D2-X/D2-Y
checkpoint, or start consistency distillation, HSIPrior, or Mixer without
separate user authorization.
