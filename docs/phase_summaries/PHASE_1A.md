# Phase 1A Handoff: Data Contracts, Representation, and Expert Scaffolds

## Scope and gate decision

Phase 1A implemented and verified only the independent expert data/model
scaffolds. It did not start full training, model selection, native-domain final
evaluation, a mixer, state machine, SDF guidance, or expert composition.

The Phase 1A gate **passed**. Both expert data contracts and hashes are fixed;
the LINGO split is leakage-free; schema/masks/initialization/parameter
independence are tested; and both experts completed single-GPU functional
smokes plus real 8-GPU optimizer updates at the locked global effective batch
1024.

## Implementation and API changes

- `code/priors/representation.py` is the only source of truth for the 232-D
  schema, 16-frame window, 2-frame history, 500 diffusion steps, coordinates,
  and domain loss masks.
- `code/priors/data.py` loads lean real OMOMO or LINGO windows without entering
  the legacy mixed-data path. HOI items have BPS/object/contact but no scene
  key. HSI items have real occupancy but no object-BPS key and fix the final 16
  representation channels empty.
- `HOIPrior.forward` has no scene argument or scene encoder. `HSIPrior.forward`
  requires real scene occupancy and has no object condition. The two types
  allocate fresh learnable modules; tests compare Parameter identities and
  storage pointers.
- `build_expert` accepts random initialization only and fails before model
  construction for every non-empty checkpoint request, including the released
  InfBaGel checkpoint.
- `tools/audit_prior_data.py` produces non-overwriting aggregate counts, source
  hashes, leakage/filter/text/missing-field/short-sequence/normalization and
  NaN/Inf audits. `code/train_prior_smoke.py` performs bounded real-data DDP
  optimizer updates and records per-rank loss/gradient/memory evidence.
- The human/object normalization bounds remain the existing OMOMO `norm.npy`.
  `data/dataset/norm.npy` is byte-identical. No statistics were recomputed.

The stable public contract is also documented in `docs/PHASE_1A_CONTRACT.md`.

## Representation schema and masks

| Field | Half-open indices | Width | HSI loss |
|---|---:|---:|:---:|
| 28 joint XYZ positions | `[0,84)` | 84 | included |
| 22 global joint 6-D rotations | `[84,216)` | 132 | included |
| object translation | `[216,219)` | 3 | empty/excluded |
| relative object 3×3 rotation | `[219,228)` | 9 | empty/excluded |
| four contact labels | `[228,232)` | 4 | empty/excluded |

Both losses exclude history frames `[0,2)`. The HSI gradient unit test and both
GPU smokes observed exactly zero gradient for output channels `[216,232)`.
Coordinates are Y-up, window-local in XZ, and aligned to initial root yaw.

## Dataset counts, filters, and split

### OMOMO-only HOI

| Partition | Raw sequences | Referenced sequences | Windows | Dynamic-object windows |
|---|---:|---:|---:|---:|
| train | 4,304 | 4,304 | 597,868 | 597,868 |
| validation/test | 482 | 438 | 1,314 | 1,314 |

There are no short windows, missing BPS sequences, missing contact sequences,
empty instructions, missing text features, or non-finite audited source values.
Instruction coverage is 100% (103 train and 88 validation unique strings). Scene
files/flags are not loaded or passed into the HOI model/loss.

### Filtered real-LINGO HSI

- Raw: 19,450 sequences, 2,275,973 windows, 111 scenes, 76 families.
- Required no-hand filter retains 1,939,898 windows and excludes 336,075.
- A validity diagnostic then excludes 21,856 windows with `seq_length <= 48`.
  Of these, 21,819 generated 48-frame windows cross their declared sequence
  end; the existing mixed loader already applies the same length rejection.
- Final: 1,918,042 windows. Train has 1,740,706 windows, 15,752 referenced
  sequences, 91 scenes, and 61 families. Validation has 177,336 windows, 1,690
  referenced sequences, 20 scenes, and 15 families.
- There is no scene or scene-family leakage. Mirror/new-locomotion/action scene
  variants remain grouped by the unchanged seed-42 family split.
- Text coverage is 100%; no empty/missing-feature instruction or non-finite
  audited source value was found.

The fixed OMOMO normalization was checked with 100,000 deterministic evenly
spaced windows per large partition. HOI train position/object tail rates were
`1.8601e-7` and `1.0e-5`; validation was zero. HSI position tail rate was
0.190635%, with maximum absolute normalized value 1.081419 and no NaN/Inf. This
is a mild, bounded cross-domain tail rather than a systematic failure; the
locked bounds were retained.

## Experiments and results

Data runs:

- `p1-data-hoi-contract-s42-20260713`: completed.
- `p1-data-hsi-contract-s42-20260713`: completed.

Smoke runs:

| Run | GPUs | Micro/GPU | Effective batch | Loss | Max reserved | Result |
|---|---:|---:|---:|---:|---:|:---|
| `p1-smoke-hoi-single-s42-20260713` | 1 | 2 | 2 | 1.460899 | 98 MiB | pass |
| `p1-smoke-hsi-single-s42-20260713` | 1 | 2 | 2 | 1.307895 | 74 MiB | pass |
| `p1-smoke-hoi-8gpu-s42-20260713` | 8 | 128 | 1,024 | 1.363620–1.389227 | 254 MiB | pass |
| `p1-smoke-hsi-8gpu-s42-20260713` | 8 | 128 | 1,024 | 1.341153–1.379167 | 262 MiB | pass |

Every run completed one optimizer update with finite rank losses and a nonzero,
finite key gradient. Both 8-GPU runs used accumulation 1 and the Phase 0 locked
effective batch 1024. The HSI masked-gradient maximum was zero on every rank.
Parameter counts are 3,918,888 for HOI and 3,591,208 for HSI. These are
functional smokes, not training or model-quality results.

## Failures and diagnostics retained

No data audit or GPU workload failed. One manifest-finalization attempt was
correctly refused because the audit command had written its aggregate directly
into tracked `experiments/results/`, making the worktree dirty during a running
manifest. The running manifests and run ids were not overwritten. The same
artifacts were moved to ignored run directories, both manifests were finished
from a clean unchanged HEAD, and byte-identical aggregates were then copied to
the tracked results directory and committed. This was a governance sequencing
fix, not a scientific rerun.

The short-sequence and normalization-tail findings above were handled only by
the preregistered filter/representation/normalization diagnostics. No new model
direction or split modification was introduced.

## Verification commands

```bash
/data/yujinlun/anaconda3/envs/infbagel/bin/python -m unittest discover -s tests -v
/data/yujinlun/anaconda3/envs/infbagel/bin/python tools/experiment.py validate
/data/yujinlun/anaconda3/envs/infbagel/bin/python -m py_compile \
  code/priors/*.py code/train_prior_smoke.py tools/audit_prior_data.py
git diff --check
```

Expected closure result: 29 tests pass; 18 registry records, one split, and two
evaluator definitions validate; compilation and diff checks pass.

## Artifacts and hashes

Tracked aggregates:

- HOI audit: `experiments/results/p1_data_hoi_contract_s42_20260713.json`,
  SHA-256 `1deea6a724a3319d4c5654da682d7f51af7e5c93b119d159bd2b37ad258f627f`.
- HSI audit: `experiments/results/p1_data_hsi_contract_s42_20260713.json`,
  SHA-256 `f67dffb3d8acfc89865e633ea3020616ce83bf0e3702382ff964fca1f39c1a92`.
- Smoke aggregate: `experiments/results/p1_prior_smokes_s42_20260713.json`,
  SHA-256 `7c2623e81fe96462198bdc36b04027d7f788624f9d8e3d5cfe0ee67e67147936`.
- Locked split SHA-256:
  `38677db82cb58071f146ffa679fba53a8bf3799d252f3a5b68eae975468a9ace`.

Ignored immutable manifests, logs, resolved configs, and raw metrics are under
`results/experiments/<run-id>/`. Exact per-run manifest, metrics, and resolved
config hashes are recorded in the smoke aggregate and registry. The HOI/HSI
data manifest hashes are respectively
`c152dde2c1049a51d26544018c838091b4adc7a9590a2bf7e48bcf6c6aa45d0d` and
`8274ef9f3e07d88530abaadaae06490b96801c97d0e48c7121c39b84492ea0a2`.

## Commits, integration, and tag

Key Phase 1A commits are:

- `1c4eab5`: independent contracts, schema, expert/data scaffolds, and tests.
- `aa97ee5`: reportable data-audit configuration.
- `ac696ba`: immutable data audits and registry records.
- `29131ca`: single-/eight-GPU smoke aggregates and registry records.

The phase summary commit is the closure/tag target. After final checks,
`phase/01a-data` is fast-forwarded into `research/state-compositional-priors`
and tagged `exp/p1a-data-v1`; no Phase 1B branch or workload is created here.

## Remaining limitations and risks

- Phase 1A validates interfaces and one optimizer update, not convergence,
  native-domain quality, or long-run memory fragmentation.
- The HSI OMOMO-normalization tail must remain visible in Phase 1C audits; it
  must not be clipped/recomputed silently.
- Full HSI training must continue to reject the diagnosed short sequences and
  use the immutable split/hash above.
- These lightweight independent scaffolds establish provenance and masking;
  Phase 1B/1C must still implement and validate their full training objectives
  without released-checkpoint initialization.

## Exact Phase 1B entry point (new session only)

1. Read this summary and the complete `docs/EXPERIMENT_PLAN.md`.
2. Verify `exp/p1a-data-v1` points to the clean Phase 1A closure commit on
   `research/state-compositional-priors`.
3. Create `phase/01b-hoi` from that tag. Do not use or copy commits/weights from
   `feature/independent-hoi-hsi-priors`.
4. Before any new implementation/training direction, append the Phase 1B
   hypothesis and exact optimizer-update budget/config to the plan and registry.
5. Use only the locked HOI contract/hash above, random initialization, micro
   batch 128 × 8 GPUs × accumulation 1, and global effective batch 1024.
