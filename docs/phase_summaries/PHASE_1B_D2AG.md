# Phase 1B D2-AG0: self-conditioned relation source

## Scope and final outcome

D2-AG0 spent the last authorized Phase 1B HOIPrior formal budget. It kept the
D2-AE0 GPU-native sparse current-state relation field structurally unchanged and
manipulated exactly one factor: the tensor from which the variable temporal
anchors `5/10/15` read their geometry. D2-AE/D2-AF read the current noisy state
`x_t`; D2-AG reads the model's own detached `x0_hat`, symmetrically on both
sides:

- training: per sample `m_i ~ Bernoulli(p=0.5)` from an independent
  `torch.Generator` seeded `cfg.seed*1_000_003 + processed_windows + rank`;
  selected samples take `s = sg[x0_hat]` from one same-timestep `inner.eval()`
  no-grad estimate forward that itself uses the D2-AE `x_t` source; unselected
  samples take `s = x_t`, i.e. bitwise D2-AE behaviour;
- sampling: `s = prev_x0` (the previous step's raw `x0_hat`, taken before
  `prepare_clean_x0`), with `s = x_t` at the first reverse step `t=499`;
- both sides pin `s[:, :2] = x_t[:, :2]`; anchor 0 and the two-frame history
  stay on the current noisy state; no SO(3) projection is applied to `s`.

Writeback stayed `H' = H + tanh(alpha) * routed_relation` with the D2-AF
`sqrt(alpha_bar)` attenuation disabled and no `sqrt_alpha_bar` buffer
registered. Relation exposure stayed exactly 1.0 with no relation-zero branch,
and the model stayed at `30,087,401` parameters (base `29,673,448`, relation
`413,953`) on the `[B,16,232]` clean-output contract. No clean target, future
GT, stored relation, stored per-frame BPS, contact label, scene asset, new loss,
SNR/timestep weighting, guidance or checkpoint initialization was introduced.

Final classification:

`selfcond-relation-source-transfer-negative-stop`

carrying the internal label `selfcond-relation-source-mechanism-negative-stop`.
The final-online checkpoint is not selectable, `d2ag1_authorized` is false and
`hoiprior_search_closed` is true. Phase 1B HOIPrior search is closed: no second
formal budget, resume, checkpoint selection, consistency, HSIPrior or Mixer was
started.

## Commits

- plan-only preregistration: `92c2ac0f16a87c8d28aadeea65ecb63cbec4fd4c`;
- source/config/tests/tools: `39e59c8ff7a38d0476e0ce2473460e16f8750f6b`;
- smoke parity probe determinism fix: `ada2d84223ecbf76f5ed9bbd313f5ac6dfce2cbb`;
- one-time performance waiver plan, validator and contract:
  `bc22b1d0f0b8ef3796e40752d0fd1916196c2268`,
  `c74bb7c5018f8ac6a99a2f7094528b131437c72e`,
  `905766eb5d1592e4e81692cf5a5293eeeea2145b`;
- evaluation-provenance semantic rebinding:
  `9d77a6f80d82f5738069fac358f2056757c14286`.

The formal run executed `905766e`; both evaluations executed `9d77a6f`. No tag
was created and no merge is authorized.

## Implementation and configuration changes

`39e59c8` added the mechanism and its whole variant-bound toolchain:
`code/priors/sparse_relation.py` (shared train/sample relation-source builder),
`code/priors/models.py` and `code/priors/diffusion.py` (variant
`d2ag_selfcond_relation_source` = `HOI_ARCHITECTURE_D2AG`, sampler `prev_x0`
state), `code/train_hoi_prior.py` (`_forward_losses` Bernoulli selection, the
eval-mode no-grad estimate forward with `try/finally` restore, variant/config/
run-id validation), the new `code/config/config_train_hoi_prior_d2ag.yaml`,
`code/priors/d2ag_diagnostic.py`, and `tools/{smoke,benchmark,diagnose}_hoi_d2ag.py`
plus `tools/run_hoi_d2ag_{internal,native_evaluation}.py`, with
`tests/test_hoi_d2ag*.py`. Checkpoint provenance is fail-closed in both
directions: released, author, base/D2-X, D2-AC, D2-AD, D2-AE and D2-AF schemas
are rejected by the D2-AG loader and D2-AG is rejected by the D2-AE/D2-AF
loaders. `bc22b1d`/`c74bb7c`/`905766e` added only the hash-bound waiver
validator, its config binding, tests and the immutable waiver contract
`experiments/contracts/p1_hoi_d2ag_performance_waiver_s42_20260731.json`
(SHA-256 `d91078a777cd2e54c5a6a6b3e77f6debe7b2acce1d299fee3b337d5401d28a97`).

### Evaluation-provenance amendment (`9d77a6f`)

After the formal run completed, both evaluation runners were blocked by two
conditions that could not be satisfied:

1. `tools/run_hoi_d2ag_internal.py::FORMAL_LINEAGE_SEALED` was a nine-key table
   whose values were all `None`, and `sealed_lineage_contract()` raised before
   the `--resolve-only` branch, so it could only be unlocked by hand-editing
   hashes into source and adding another governance commit;
2. both runners required `--resume-contract`, but D2-AG is a straight-through
   run and never produces `resume_contract.json` (`training_state.json` records
   `resume_checkpoint: null`). That artifact class only exists for a resumed run.

The root cause was that D2-AG copied the shell of the D2-AF hardening commit
`3d4ff1e`, and D2-AF was a resumed run while D2-AG, like D2-AE, is
straight-through. Within the file scope locked by the preregistration, the
binding was replaced with the D2-AE-style semantic `checkpoint_contract()`
(`tools/run_hoi_d2ag_internal.py:141`): hash the target file once against the
CLI-supplied value, require the fixed final-online basename, then read the
checkpoint and assert `run_id`, `seed=42`, `processed_windows=61,440,000`,
`processed_frames=983,040,000`, `optimizer_updates=30,000`, `world_size=4`,
`effective_batch_size=2048`, `architecture_variant`, `data_contract_sha256`,
`split_sha256`, from-random `weight_initialization` and the full
`selfcond_relation_source_contract`. This is strictly stronger than the removed
table, which could only prove byte-equality with a hand-typed constant. Only
no-op checks were deleted (same-file rehashing inside one process, constants
compared against themselves, a hard-coded `"asset_hashes_exact": True`, a
hex-format-only check, and an unreferenced constant). Sampler, metrics,
uncertainty, cohort, checkpoint-selection and failure rules were untouched.

## Retained failures

Every failed or aborted attempt is preserved and none was reused or overwritten.

| record | stage | nature |
|---|---|---|
| `p1-hoi-d2ag-cpu-contract-s42-20260731` | authority CPU gate | `selfcond-relation-source-contract-failure-stop`, `FileNotFoundError`: the gate requires a pre-archived resolved config. Rerun as `-r2-`. |
| `p1-hoi-d2ag-gpu-functional-smoke-{,-r2,-r3}-s42-20260731` | worker smoke | pre-allocation aborts leaving only a resolved config (the base directory is empty). Passed on `-r4-`. |
| `p1-hoi-d2ag-performance-benchmark-s42-20260731` | worker benchmark | operational preflight abort, resolved config only. Rerun as `-r2-`. |
| `p1-hoi-d2ag-performance-benchmark-r2-s42-20260731` | worker benchmark | completed and **scientifically failed**; see below. |
| `p1-hoi-d2ag-selfcond-relation-source-s42-20260731/operational_launch_failure_001.json` | formal training | pre-GPU launch-wrapper failure. |
| `p1-hoi-d2ag-selfcond-relation-source-internal-s42-20260801/operational_preflight_failure_001.json` | internal diagnostic | pre-allocation preflight failure. |

The **formal launch-wrapper failure** (SHA-256
`d92a4b224848e66415426d99878ab8f5c9284e424ee0d75c5ab1a0c67b5263c3`) occurred
after `tools/experiment.py start` wrote the manifest and before the detached
tmux wrapper was launched: the controlling interactive session terminated. No
training process, optimizer, checkpoint directory, `train.log` or GPU workload
ever existed (`gpu_workload_started: false`, `optimizer_updates: 0`,
`checkpoint_writes: 0`, GPU memory 15-102 MiB idle). Because nothing scientific
had happened, the same manifest and the same run id continued, exactly as in the
D2-AF precedent; `scientific_configuration_changed` is false.

The **internal preflight failure** (SHA-256
`2c3484dd1f6225abdf4967643313e2241950c64bfc1ec1e981bfce2f35368066`) failed the
single check `chois_pinned`: the completion chain passed `--chois-root` at
`third_party/chois_omomo_evaluator_assets`, which holds released weights and is
not a Git repository, so `git rev-parse HEAD` walked up to the InfBaGel checkout
and returned `9d77a6f` instead of the pinned CHOIS commit
`8ec585aa0200fd2a890ffb12897bcf69ae719463`. All 15 other preflight checks
passed, no run id was allocated and no manifest was created. The chain was
repointed at the evaluator checkout and the diagnostic reran under the new id
`-internal-r2-`; the aborted directory is retained per `EP:7073`.

## Authority CPU gate and worker smoke

`p1-hoi-d2ag-cpu-contract-r2-s42-20260731` passed as `cpu-contract-passed` in
`11.406160387210548 s` with zero CUDA, optimizer, checkpoint or evaluation
activity, covering the inherited D2-AE/D2-AF geometry, asset, SO(3),
invariance, permutation, dtype/device, parameter/API, checkpoint-provenance,
HSIPrior/Mixer-independence and forbidden-source contracts plus the new D2-AG
train/sample source parity, estimate-forward equivalence and detachment,
unselected-sample bitwise D2-AE equivalence, `s[:, :2]` pinning,
`prev_x0`/first-step, generator-isolation, global-RNG-independence and
eval-mode-restore conditions.

`p1-hoi-d2ag-gpu-functional-smoke-r4-s42-20260731` passed as
`functional-smoke-passed` on one RTX 3090 with real data, batch 8 and mixed
timesteps `0/249/499`, from seed-42 random initialization with no optimizer and
no checkpoint I/O. Peak allocated/reserved/headroom were
`270,394,880 / 325,058,560 / 24,970,985,472` bytes.

## Performance gate failure and the one-time user waiver

The registered 4-GPU full-micro-batch benchmark is a **failed scientific gate
and remains failed**. `p1-hoi-d2ag-performance-benchmark-r2-s42-20260731`
(summary SHA-256 `ad5b052d850a6978d2ce64022ecc0328ab73f32ffa7a72446a4a16bdf2c19cae`)
ran 64 warm-up + 256 measured updates at 4x512, measuring `524,288` windows:

- synchronized measured wall `241.36226116283797 s`;
- throughput `2,172.2037135137825 windows/s` against the registered minimum
  `2,756.580356467847 windows/s` (a `-21.199332774135327%` shortfall);
- `throughput_fraction_of_sealed_d2x = 0.6698056714198497`;
- extrapolated 61.44M-window ETA `7.8568444388944645 h` against the `6.20 h`
  limit;
- status `failed`, classification
  `selfcond-relation-source-performance-negative-stop`,
  `formal_training_authorized=false`, `sweep_authorized_on_failure=false`.

Exactly five checks failed (`classification`, `eta`, `formal_authorized`,
`status`, `throughput`). Every non-speed contract passed:
`all_rank_contract_pass`, `memory_headroom_pass`, `contention_pass`,
`losses_finite`, `gradients_finite`, `selfcond_estimate_forward_measured` and
`selfcond_graph_pass_instrumentation_pass`, with both contention samples empty.

The recorded attribution is that the gate was **not discriminating for this
mechanism**. Per-rank inclusive `backward` (including DDP critical-path wait)
spanned `105.08-206.66 s`, a factor of `1.966598891801045`, while the
manipulated mechanism was uniform across ranks: `estimate_trunk_forward`
`6.53-6.94 s`, mean `6.744397082805633 s`, only `2.794304731117614%` of wall.
Removing the estimate forward entirely would give `~234.62 s` and
`~2,234.65 windows/s`, still far below the threshold; the mechanism explains
about `10.685374742401925%` of the shortfall and the harness rank skew about
`89.31%`.

After that failure was preserved and reported, the user explicitly accepted the
measured `7.8568444388944645 h` ETA and authorized one run-id-bound waiver of
the execution stop rule. The waiver does **not** reclassify the benchmark: its
status stays `failed`, its classification stays
`selfcond-relation-source-performance-negative-stop`, and the formal lifecycle
state is expressed only as `failed-waived` /
`user-authorized-performance-waiver`, never as `performance-gate-passed`. No
batch, micro-batch, worker, thread, affinity, prefetch, architecture, point,
width, role, routing, `p` or budget sweep was run, and no second benchmark was
authorized. The waiver is one-time, binds only
`p1-hoi-d2ag-performance-benchmark-r2-s42-20260731` and
`p1-hoi-d2ag-selfcond-relation-source-s42-20260731`, and is explicitly not a
precedent for any later direction.

## Formal training

`p1-hoi-d2ag-selfcond-relation-source-s42-20260731` ran once, from seed-42
random initialization (initial model state
`b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`, zero
released/author/D2-X/D2-AC/D2-AD/D2-AE/D2-AF/EMA/consistency checkpoint loads),
on infbagel-4gpu/node01 with 4x RTX 3090 at 512 per GPU, effective batch 2048,
accumulation 1, FP32 Adam, LR `1e-4`, no warmup/scheduler/AMP/clipping/EMA:

- `61,440,000` windows, `983,040,000` frames, `30,000` optimizer updates,
  return code 0;
- wall `17,988.119868855923 s` = `4.99669996357109 h`;
- throughput `3,415.5876460649633 windows/s` (`54,649.40233703941 frames/s`);
- `amp_overflow_skips` 0 on every rank, losses finite, key gradients present;
- all 20 validation points finite, total validation loss
  `0.11909466094994059 -> 0.04894545498541447` (a `58.90205774548714%`
  reduction), final contact accuracy `0.9672710297854792`;
- 20 cadence checkpoints, `optimizer_step_min = optimizer_step_max = 30000`;
- peak allocated/reserved `5,193,363,968 / 6,352,273,408` bytes, minimum
  headroom `18,943,770,624` bytes;
- learned `alpha = -0.15308877825737`, `tanh(alpha) = -0.15190395712852478`.

The measured formal throughput `3,415.5876460649633 windows/s` is **`1.5724x`
the failed benchmark's `2,172.2037135137825 windows/s`, and above the registered
`2,756.580356467847 windows/s` minimum that the benchmark missed**; actual wall
`4.9967 h` beat the benchmark extrapolation `7.8568 h`. The benchmark's
CUDA-synchronized per-stage instrumentation over 320 updates surfaced a
rank-skewed DDP/backward wait that did not recur at the same magnitude under
the plain 30,000-update training loop. This reproduces the D2-AF precedent
(benchmark `2,089.8443630127094` -> formal `3,232.575359023025 windows/s`) and
confirms the benchmark understated deliverable throughput. It does **not**
retroactively pass the gate, and it says nothing about the mechanism.

Final-online checkpoint SHA-256
`f28af345254cb4884c64f0ddda799ebbb131e19b209583e47553bb601be4026f`; final model
state `94f05a8337603fe5c15094572121430066d886e3a1c9886b0fa5fc0716613033`;
metrics `27703648fc8e45ee3210e62975aa9500102c78ce2937be51807e21f5524497e0`;
manifest `4e561bfa3bb4054a16fadccc0e992ad9e9f91c78f34e41fdcb11c3fb10ae5d34`;
resolved config `81b55a0ec9e9877521415cf015f2aef7d5e31196dad589046bba700e53f5846b`;
training state `465ab8167ce2950fd54667d802a478646ea5466f4539ad87582d2c25e28c516f`.

## Fixed internal causal diagnostic

`p1-hoi-d2ag-selfcond-relation-source-internal-r2-s42-20260801` loaded only the
fixed final-online checkpoint and ran the sealed D2-O cohort (64 sequences x 3
windows, phase offsets `(14,56,98)`, selection SHA-256
`1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`, batch 8) as
six paired 500-step rollouts sharing initial latent, per-step posterior noise,
conditions, history and ordering, with 10,000 paired sequence bootstraps.
Runtime `708.4948208660353 s`. All 40 provenance, pairing and numerical
contracts passed; no optimizer, checkpoint write, checkpoint selection or
official-test use occurred.

Direct-hand union 5-cm F1, full minus other, 95% CI:

| perturbation | point | 95% CI | gate |
|---|---:|---:|---|
| left/right role swapped | 0.30495373 | [0.22151673496009272, 0.38789158704277416] | pass |
| temporal correspondence permuted | 0.18353939 | [0.11419948976023367, 0.2558505734245301] | pass |
| source substituted with `x_t` | -0.00411124 | [-0.01317133550839981, 0.0052449391460933584] | **fail** |
| object displaced counterfactual | -0.00656911 | [-0.015275483332368953, 0.0015924505161037208] | **fail** |
| high-t restricted (`t>=250` back to `x_t`) | -0.00127744 | [-0.003746313159783638, 0.0012043454632572222] | **fail** |

(The role-swap gate is judged on the left/right macro F1, point `0.25178018`,
CI `[0.18597224946534732, 0.31784488303567626]`.) Only 3 of 9 registered gate
checks passed, and all three passing checks belong to the two structural
properties inherited unchanged from D2-AE.

**The two significant effects are large while the three nulls are tight, so the
nulls are well powered rather than underpowered.** Role swap and temporal
permutation move union 5-cm F1 by `0.305` and `0.184`; the three failing
intervals have half-widths of at most about `0.009`, roughly an order of
magnitude smaller. The data therefore do not merely fail to detect an effect,
they bound it near zero: the trained model does not use the `x0_hat` provenance
of the relation source (substituting `x_t` back changes nothing), does not
follow a `0.10 m` counterfactual object displacement in `s`, and behaves
identically whether self-conditioning is active at high `t`. The learned gate is
`-0.15190395712852478`, but a nonzero parameter is not causal evidence.

`internal_status: source-provenance-negative`; classification
`selfcond-relation-source-internal-source-negative-continue-native`;
`native_evaluation_authorized: true` because the preregistration mandates the
native run regardless of the internal outcome.

Diagnostic artifacts: `full_self_conditioned`
`0f8b5ac06c6abcc44785f8fc513fa5d3430dacaaaa2254fd686d3b4f4853b517`,
`source_substituted_xt`
`6cefadc52b7337ce2001defbc2410178a0597329b81770e755a8abc6dfe80446`,
`high_t_restricted`
`c1fa65e23ba3abc93b7cd17d03b379d4638ed69c6d891c4553278f792c7753d9`,
`object_displaced_counterfactual`
`a7a392d42a55c140b882b6629af71721af16bb3218bcdb4646e600ed61410006`,
`temporal_correspondence_permuted`
`fe7b42d830ce6101221a9edb0f2af5d61813dde82e0240f5f203f63e6ffc3d21`,
`left_right_role_swapped`
`804d78a38c480c3151fdcfeb7315d56f5000c545ecf6b2ba4d6c95fe2a437dc9`,
`paired_noise` `6541cea3c5f2cd444add5f62e141934fa49ac91657742d9d5b496a1b05845c29`,
`paired_conditioning`
`d8ea7a40a7bde42f56479b256cf6aefdee5751e58853e985e75d0c425bf909db`;
metrics `d9bb2a92a0852e991f3021d73e8383f3ab6b5f0fdb0aee37ff8f814686f39aa1`,
manifest `37974a95d6a9ad439cd540d851db9209571881e0c94d2a128878f57eb3a069b2`.

## Fixed native evaluation

`p1-hoi-d2ag-native-eval-s42-20260801` ran the unchanged official evaluator over
438 sequences x 3 windows with 500 unguided steps, final-online weights, seed
42, sequence-paired units and 10,000 bootstraps. CFG, guidance, scene
conditioning, dynamic perception and consistency were off; no released or author
checkpoint was loaded; sealed D2-X aggregate/per-sequence outputs and the
released baseline aggregate were reused without regeneration. Runtime
`376.92146245902404 s`. All 29 training-contract and 44 internal-provenance
checks passed.

Target point estimates: end-object `3.6921996623277664 cm`, Txy
`3.9954081177711487 cm`, FS `0.40091825523376623`, contact precision/recall/F1
`0.8111961856520455 / 0.5984951273917594 / 0.6500867274276584`, Pbody
`2.9378708358181074`, hand penetration `0.1836719029758619`, MPJPE
`12.012922018766403 cm`, Troot `8.299961686134338 cm`, Tobj
`15.416969429094832 cm`, Oobj `1.00918565015388`.

**Transfer: all four registered checks failed.**

- contact F1 `0.6500867274276584` vs the registered minimum `0.6598838781`,
  short by `0.009797150672341659`;
- released gap closure `0.1409388939758756` vs the required `>=0.25`
  (control `0.6374259391059788`, released `0.7272576950146546`);
- target minus control contact F1 paired mean `+0.012660788321679566`, CI
  `[-0.00823759546954707, 0.034443270025997036]` - not significant;
- target minus control contact recall CI
  `[-0.019876465115143665, 0.028525400940548902]` - not significant.

The only significant contact gain is precision, `+0.023134310056835906`, CI
`[0.003597741570523799, 0.04268227842678902]`. Contact coverage therefore did
not improve; the model became slightly more conservative.

**Protection: 10 of 11 checks passed, one failed.** Foot sliding
`0.40091825523376623` is significantly **worse** than control
`0.36301002139393850`: control minus target paired mean
`-0.037908233839827776`, CI `[-0.06449495248837395, -0.011320260346009765]`,
with a ratio of `1.1044275133073798`, CI `[1.0299970463398207, 1.183784273570895]`
whose upper bound exceeds the `1.10` limit. All eight other protection ratios
and the contact-precision floor passed, and the sealed penetration finite mask
contract passed exactly at `181` finite of `438` official sequences (IDs SHA-256
`2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`). Hand and
human penetration actually improved (ratios `0.7485919504119813` and
`0.7593193542159281`).

**Released 95% effectiveness: 6 of 11 passed.** Failures were contact F1,
contact recall, end-object translation error, foot sliding and human
penetration.

Decision flags: `checkpoint_selected: false`, `d2ag1_authorized: false`,
`hoiprior_search_closed: true`, `internal_mechanism_passed: false`,
`native_transfer_passed: false`, `object_following_passed: false`,
`high_t_provenance_passed: false`, `contract_passed: true`,
`official_test_used: true`, `protection_passed: false`,
`released_95_percent_effectiveness_passed: false`,
`selectable_autonomous_diffusion_candidate: false`,
`consistency_authorized: false`, `role_binding_passed: true`,
`temporal_routing_passed: true`, `source_provenance_passed: false`.

Native artifacts: aggregate
`f51266f8eb81e87df9ff3a881613fc0917736c3ac7e184dfa5c7f8f091d8ae9b`,
per-sequence `eb701cf4e80a4a6c8198a0af5f914fab98c8bf26aabafee9971ce0827de2d835`,
metrics `f9d7cae8a343281db6529f0816e58a21a8ef5e6de51009cd8179bd9e5a052f69`,
manifest `2c46296bc723bbf450f5b1386db17a9dc0a15a34c2cdaaa085abea21f01f1d31`,
resolved target `417bede836f28e69609f0310af7773f149522c42f6a06295554fc46bfa028d75`.

## Scientific conclusion

D2-AG is a clean negative and it closes a real question rather than leaving it
open. The registered hypothesis was that D2-AE/D2-AF failed to transfer because
the relation path only ever saw high-noise `x_t` geometry during rollout. Giving
the same path low-noise self-conditioned geometry, symmetrically on both sides,
changed nothing measurable: the internal diagnostic bounds the source-provenance,
object-following and high-t effects near zero with well-powered intervals, and
the official evaluation shows no significant contact F1 or recall gain over
sealed D2-X, a contact F1 point estimate below the registered minimum, released
gap closure at `0.141` against `0.25`, and a significant foot-sliding
regression.

**Scope limit on the object-displaced null - do not read it as object
insensitivity.** The intervention (`code/priors/sparse_relation.py:680-697`)
adds `+0.10 m` to the object translation channels *of the self-conditioning
relation source only*, at variable anchors 5/10/15; anchor 0 is never displaced
(`:59`, `:32`), and per the registration (`EP:7143-7148`) every other channel
and the whole denoiser conditioning stack - including the global object BPS and
pose - is untouched. The source-substitution gate replaces those same three
frames' entire content with `x_t`, a far larger perturbation, and is itself
null. A 10 cm shift applied to frames already shown to be inert therefore
carries no information about whether the model uses object geometry, and
`object_following_passed: false` is a co-report of the source-provenance
failure, not an independent object-sensitivity measurement. The role-swap and
temporal-permutation effects do not settle the question either: the feature is
`delta = object_surface - role_joint` (`sparse_relation.py:530-537`), which
mixes hand and object, so a hand-only model would degrade identically. **This
diagnostic cannot answer the object-conditioning question in either
direction.** Independent evidence from the D2-O era points the other way:
generated-human x generated-object GT-contact distance `10.19/9.85/10.35 cm`
versus generated-human x GT-object `49.48/50.84/25.71 cm` (`EP:1502-1509`) -
the hand tracks its own generated object several times more closely than the
real one, which an object-blind model could not produce. The evidence now covers both plausible readings of D2-AE's
train-to-rollout gap - attenuation by signal reliability (D2-AF) and noise in
the source geometry (D2-AG) - and neither repairs contact coverage. The relation
path continues to behave as a generic conditioning residual whose structural
properties (role binding, temporal correspondence) are learnable and causal
while its interaction content is not.

## Recovery and verification

Recovery was worker-initiated and non-destructive, with no `--delete`, using
`rsync -aH --partial -e 'ssh -i $HOME/.ssh/id_ed25519_infbagel_8gpu -o
IdentitiesOnly=yes'` from `node01` to
`/data/yujinlun/InfBaGel-p1b-staging/` on the authority, followed by exactly one
`rsync -aHn --checksum --itemize-changes` pass per directory. All four passes
exited 0 with zero itemized differences, and `tools/experiment.py::sha256_path`
produced identical trees on both hosts:

| directory | files | bytes | tree SHA-256 |
|---|---:|---:|---|
| `p1-hoi-d2ag-selfcond-relation-source-s42-20260731` | 130 | 7,227,034,744 | `353c042b382d27b73b419b764320433f98e2c93e60458788167721a808ae7316` |
| `p1-hoi-d2ag-selfcond-relation-source-internal-r2-s42-20260801` | 19 | 256,597,169 | `3aa2f66d70a7d0153f1f809203c6b291d9ac169ed1c9b886440a161804f002a8` |
| `p1-hoi-d2ag-native-eval-s42-20260801` | 16 | 1,199,214 | `153a6b6fbb939545496deb950c3ddd18c7680e71297b25b59a033de96b05a738` |
| `p1-hoi-d2ag-selfcond-relation-source-internal-s42-20260801` (retained abort) | 3 | 12,236 | `70b3d2a0294b217da44f50451fc0aa7dcfad31dc464b3675219bced570d7446b` |

These tree hashes are the as-transferred values, taken before the run-local
`registry.jsonl` was appended on the authority. Run-local registry hashes are
`e21fa9c061c99d8569eb23e3d778d76f2f9b1664c31dcae1f9500ea6ca9ce4cd` (formal),
`d1c12614d0a8e9e9b7883b8bc5f8abe5766373cf655ae58b48a84eec0ba2c458` (internal)
and `753c544925cccb6c2857e1f3aacbde0d92b41d70a20768d658fa111400184018` (native).

Verification commands (authority, `INFBAGEL_PYTHON =
/data/yujinlun/anaconda3/envs/infbagel/bin/python`):

```bash
"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2ag tests.test_hoi_d2ag_eval \
  tests.test_hoi_d2ag_lifecycle_cpu          # 121 tests OK
"$INFBAGEL_PYTHON" tools/experiment.py validate
"$INFBAGEL_PYTHON" tools/experiment.py register --manifest <run>/manifest.json \
  --registry <run>/registry.jsonl --hypothesis ... --conclusion ... --next-action ...
```

`9d77a6f` additionally verified the new semantic binding with seven negative
tests (foreign `run_id`, an intermediate cadence checkpoint, a non-D2-AG
`architecture_variant`, a tampered self-conditioning contract, non-random
initialization, a wrong shape/budget, and a hash or basename mismatch), and
proved they are not vacuous by mutation testing: forcing the `run_id` and
`processed_windows` predicates true made the corresponding negative tests fail.

## Unresolved risks

1. The performance gate remains unusable in its current form. Both D2-AF and
   D2-AG failed it and both then trained faster than the gate's own minimum, so
   the benchmark harness measures its own rank skew rather than a mechanism's
   cost. Any future direction that inherits this gate will need it redesigned or
   will need a fresh explicit user authorization; the D2-AG waiver is not a
   precedent.
2. Two directions in a row (D2-AF, D2-AG) reached formal budget on a mechanism
   that turned out to be causally unused. Pre-training screening cannot
   currently distinguish "the path is wired correctly" from "the trained model
   will use it", and both CPU contracts and functional smokes passed in full.
3. The tracked compact result
   (`experiments/results/p1_hoi_phase1b_d2ag_selfcond_relation_source_s42_20260801.json`)
   and the append-only completion record are not yet written; this summary and
   the run-local registries are the current authoritative closure evidence.
4. The D2-AG smoke and benchmark run directories still exist only on the worker
   and have not been recovered to authority staging.
5. Contact recall/coverage relative to the released model remains the open HOI
   deficit, now with no authorized Phase 1B route to attack it.

## Exact next-phase entry point

Do not merge or tag this negative phase. Do not select, resume or reuse the
D2-AG final-online checkpoint, and do not start D2-AG1, a second formal budget,
a longer budget, a performance or `p` sweep, consistency, Mixer or any further
HOIPrior direction. Phase 1B HOIPrior search is closed.

The only authorized next step in this session is closure bookkeeping: append the
three completed runs to `experiments/registry.jsonl`, write the compact tracked
result and the completion record, and commit. A new session may then begin only
with a dated, plan-only Phase 1C HSIPrior preregistration on `phase/01c-hsi`,
which must initialize HSIPrior from random weights and must never load released,
author, D2-X, D2-AC, D2-AD, D2-AE, D2-AF or D2-AG checkpoints.
