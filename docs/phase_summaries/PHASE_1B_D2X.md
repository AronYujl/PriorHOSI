# Phase 1B D2-X FK-foot Temporal Routing Summary

## Scope and gate decision

D2-X tested one controlled training change relative to D2-V: the four foot joints'
horizontal velocity residuals were routed through predicted root/rotations, FK, and
the existing position normalization. The 232-D representation, conditions, model,
loss tensor/reduction/weight, optimizer, effective batch, 61.44M-window budget, and
official evaluator were fixed.

The reportable r1 training and its single official-438 evaluation completed. All
artifact, provenance, normalization, sequence, and penetration-mask contracts passed.
All absolute diffusion thresholds passed, but the paired foot-sliding improvement CI
included zero. The preregistered decision is therefore
`fk-foot-temporal-routing-negative-stop`. No checkpoint was selected and consistency
distillation remains unauthorized.

## Implementation and configuration

- `48e2e6c31d281af8809d35b5c8ce2ac8123205d1` introduced the D2-X routing,
  config, evaluator, tests, plan, and registry preregistration.
- `3af3facaf73d3bbffcca6d6181bac1ff89909a24` fixed the pre-optimizer D2-X
  balanced-weight dispatch and registered the r1 lifecycle after the original run
  failed before any update.
- Routed joints are `[7, 8, 10, 11]`; routed components are x/z. The target remains
  clean direct normalized foot positions, and the first future residual uses the
  immutable final history frame.
- Training stayed random-initialized, FP32, Adam at `1e-4`, effective batch 2,048,
  with no warmup, scheduler, clipping, weight decay, AMP, or EMA.

## Experiments and results

### Preserved failures

The original run `p1-hoi-d2x-fk-foot-temporal-routing-s42-20260723` failed before
the first optimizer update because the old loss-weight mode dispatch omitted D2-X.
It recorded zero processed windows and zero checkpoints and remains permanently
registered as failed. Artifact tree SHA-256:
`4088e04b1b92d412d25aa1842cb6cd2d6d4191a48c79388fdc3c8229bf16ab95`.

The first r1 evaluation preflight used the wrong CHOIS Git-root argument and failed
before manifest creation, checkpoint loading, or GPU evaluation. Its SHA-256 is
`e28c64a4cd6ee1392a11d843f603378743f4a50d68ef8305fa248b63d55a62ed`.
The corrected preflight passed with SHA-256
`3262548e948038f3725978e645c4f2f14893253bddfa5349ec80e018b18cf76e`.

### Training r1

- Run: `p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723`
- Windows/updates: 61,440,000 / 30,000
- Loss and required gradients: finite/present
- Cadence artifacts: 20 checkpoints and 80 rank RNG sidecars
- Final checkpoint SHA-256:
  `b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51`
- Throughput: 3,243.04 windows/s
- Minimum per-rank memory headroom: 21,212,889,088 bytes
- Training tree: 112 files, 7,127,226,145 bytes, SHA-256
  `3f95773270e4701310daac9128c19d822d0a4e887ad7c5ddd40008f1b98a47c6`

### Official evaluation

- Run: `p1-hoi-d2x-native-eval-r1-s42-20260723`
- Protocol: official 438 sequences, three windows/sequence, 500-step unguided
  diffusion, final online weights
- Target MPJPE/end-object/xy/object-translation:
  `12.0508 / 3.7402 / 4.0505 / 15.9940`
- Target foot sliding/contact F1: `0.3630 / 0.6374`
- Target hand/human penetration loss: `0.2454 / 3.8691`
- D2-V minus D2-X foot-sliding mean difference: `0.0152733`
- Paired bootstrap 95% CI: `[-0.0037904, 0.0341575]`
- D2-X minus D2-V contact-F1 95% CI:
  `[-0.0014803, 0.0191332]`
- All six D2-V protection checks passed; the fixed 181-sequence penetration mask
  matched exactly.
- All absolute MPJPE, end-object, xy, object-translation, foot-sliding, and contact
  thresholds passed.
- Mechanism gate failed only because the foot-sliding improvement CI lower bound
  was not greater than zero.
- Evaluation tree: 16 files, 366,190 bytes, SHA-256
  `c4a853d99659ac92ac830621a0e8caf68aea3db9f3d954b3486d3aa4d3d3eb74`.

## Verification

- Authority D2-X targeted and surrounding regression tests passed before workload
  publication.
- Worker full suite passed 252 tests with the two real-LINGO file tests skipped under
  the documented HOI-worker role.
- Twenty checkpoint hashes were independently recomputed; all matched the training
  metrics. All 80 RNG sidecars were present.
- Worker and authority hashes matched for both returned artifact trees and all named
  manifests, metrics, configs, aggregates, per-sequence records, and registries.
- `tools/experiment.py finish`, run-local registration, authority tracked registration,
  and registry validation were used for both completed lifecycles.

## Tracked and external artifacts

- Compact aggregate:
  `experiments/results/p1_hoi_phase1b_d2x_fk_foot_temporal_routing_r1_s42_20260723.json`
  (SHA-256 `fdf21f8b0042d1d26ac2a3b4cf8a073a43cd20283f0292d55572aa66de6e42f6`)
- Training staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723`
- Evaluation staging:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2x-native-eval-r1-s42-20260723`
- Training manifest/metrics SHA-256:
  `2011ded7310f851d3a1278bd65fe2d19fdca8f7b859d289d7e240b3d8d347d85` /
  `0c99ac8b5880b1e7419cc6fe9c4be6388e7f806f066e9863da0ac320336693f3`
- Evaluation manifest/metrics/aggregate/per-sequence SHA-256:
  `fa19565eb96155f735a7b8c1569a95e1069267b8641ffe7968e878554fee4550` /
  `f2cb76d0c248c4d4b8ce4571758c1937e2e19170199c6e9631df21858dc1c807` /
  `3bfe1b62d9f282aa0c188e3ac43e27528ce993a62f5314caa0a4b290da77242b` /
  `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`.

No merge commit or immutable tag was created because the D2-X gate failed.

## Unresolved risks and next entry point

D2-X shows that evaluator-aligned FK-foot temporal routing is safe and descriptively
helpful, but it does not isolate a statistically certified foot-sliding mechanism at
the registered threshold. D2-V/D2-X still have weaker penetration than the released
baseline, and D2-X cannot be promoted despite passing the absolute point-estimate gate.

The next session must begin by reading this summary and `docs/EXPERIMENT_PLAN.md`.
Any new HOIPrior mechanism requires a new dated plan and registry amendment before
code or GPU work. It must not select an intermediate/D2-X checkpoint, resume D2-X,
bundle penetration/contact/sampler changes, or start CM without separate user
authorization.
