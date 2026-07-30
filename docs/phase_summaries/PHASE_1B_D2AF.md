# Phase 1B D2-AF0：sqrt-alpha-bar current-state reliability routing

## Scope and final outcome

D2-AF0 was the final authorized HOIPrior direction in Phase 1B. It kept the
D2-AE0 GPU-native sparse current-state relation field unchanged and introduced
exactly one manipulated factor: a fixed, canonical, per-sample
`sqrt(alpha_bar[t])` multiplier on the relation writeback:

`H' = H + sqrt(alpha_bar[t]) * tanh(alpha) * routed_relation`.

The relation source remained the current noisy/generated state only. No clean
future state, predicted `x0`, scene/occupancy, stored relation, contact label,
new loss, SNR weighting, learned schedule, per-anchor gate, or evaluator change
was introduced. The model stayed at 30,087,401 parameters and the
`[B,16,232]` clean-output contract.

The complete authorized lifecycle is closed: source/provenance audit,
plan-only registration, implementation and CPU contracts, functional smoke,
clean-signal eligibility, the registered performance benchmark, one
user-authorized formal budget, fixed internal diagnostic, fixed native
evaluation, non-destructive recovery, hash verification and append-only
completion records.

Final classification:

`diffusion-reliability-ae-repair-negative-stop`

The final-online checkpoint is not selectable. Phase 1B HOIPrior search is
closed; no D2-AF1, second formal budget, resume/selection, consistency,
HSIPrior or Mixer was started.

## Commits and locked mechanism

The lifecycle commits, in order, were:

- plan-only: `cbf55ef2c5d667d28698597127767e0b14151f06`;
- source/config/tests: `cae7d4ed64fbc6c15b046c0d17b0cbdefd365b41`;
- CPU contract: `758d54897640e93cc60ac76050b9e769ddf4afbc`;
- functional smoke: `d12036e5e79d0e7142e8d163fc9a80a62fea317c`;
- clean-signal eligibility: `1c6c3058478411361bf3e73830f900f660ae516b`;
- performance failure: `f8d18e49cfa95542bfd66cd61a05bad509c75b0b`;
- one-time waiver plan/implementation/contract binding:
  `0bca540e44ff748089ec070a23c704a6a10ee8c1`,
  `9c908ad87dce8806eb052b2a2627160b0a1bbe72`,
  `69d8cb025c89c0e776d0a4c03a8c158bbd0a3265`,
  `7202d32a7375e7197886c4f873688fd472e2c803`;
- checkpoint-race continuation plan/implementation/binding:
  `3b9c0c5a0d1599fb2f80de8db36f3547f2f3f71c`,
  `b7248bba3e77234c8f2a5993d8bf3ee8a1db2757`,
  `044227fe512a9ee6d1c2a1bc898d3b8a2c6ca706`;
- formal completion: `d51057c35485d9b5e1abc846a55dc2f4324f9659`;
- evaluation provenance hardening plan/implementation/binding:
  `7a484fe18dc28e29e30b2966d966825823130c0b`,
  `3d4ff1eb5c57b1b08537859dca8e895bc428a26d`,
  `a4cdcf09f84553159be10c555ff8a6773b65d3aa`.

The canonical 500-step float32 schedule is hash-bound by beta
`496ec54f35af6fe7b92417f7da8b442f31c9c0070bfdd62dbb16fefc426c8f3e`,
alpha-bar `55f162cebbe109c67a75b00a10a1d23ea85fb1d18df9a372a3e237df5a8f48d4`,
and sqrt-alpha-bar
`5d25c63d6618c77cc31976ee9e2c5645aa41653030fca210594a05254323b440`.
The sparse relation assets are the immutable 100-point assets with mapping,
manifest and stacked-tensor hashes
`1af35119c1dd54e2ad44c99f3cb91b62c1b88f62ca80cddcc96f4b201ffe0f5b`,
`e88d74a7ee434f3e6320c95d1ebb74efdc8fe4740b70ff596e502666a096f7a7`, and
`793dad6a805d0a908087b273590bf171e7bce4c026297cf94d40f8c651fe4cab`.

## Authority CPU contract

`p1-hoi-d2af-cpu-contract-s42-20260729` completed as
`cpu-contract-passed` in 9.509392092 seconds. Exact parameter count,
canonical schedule/`GaussianDiffusion` parity, mixed-timestep scaling,
zero-gate D2-X parity, geometry and train/sample contracts, checkpoint
provenance rejection, HSIPrior/Mixer independence and forbidden-source scans
all passed. The mixed-timestep field error was
`2.384185791015625e-07`; zero-gate shared-trunk max-abs difference was exactly
`0.0`. No CUDA, optimizer, checkpoint load/write or evaluation occurred.

The recovered CPU tree is 3 files / 135,456 bytes,
`df730afb3685171099a7296fee87538e41cc64ae3ea61d50056eb87632221cd2`.

## Functional smoke and clean-signal eligibility

The real-data worker smoke
`p1-hoi-d2af-gpu-functional-smoke-s42-20260729` used batch 8 and mixed
timesteps `0/249/499`, with random initialization and no optimizer or
checkpoint I/O. Losses and all required gradients were finite; relation
construction was CUDA-only. Peak allocated/reserved/headroom were
270,197,248 / 325,058,560 / 24,970,985,472 bytes. The smoke tree is 10 files /
149,421 bytes,
`61fc8d844c68637e7cf34af4bb9e9b4dc969b71bb001af003fe22309247c0747`.

The no-model eligibility traversal covered 216 sequences and 29,382 windows.
The two registered corruption comparisons passed:

- `C249-C0`: 3.7088510435, CI `[3.6977504342, 3.7201192816]`;
- `C499-C249`: 0.7123014183, CI `[0.6981956675, 0.7267265609]`.

The immutable history anchor remained exactly unchanged across timesteps.
No model, optimizer, checkpoint, rollout or downstream metric was created.

## Performance gate, waiver and ETA explanation

The registered 4-GPU benchmark deliberately remains a failed scientific
performance gate. It ran 64 warm-up and 256 measured updates at 4×512,
measuring 524,288 windows:

- synchronized measured wall: 250.8741843551 s;
- throughput: 2,089.8443630 windows/s;
- fixed threshold: 3,179.6898630 windows/s;
- extrapolated 61.44M-window ETA: 8.1664773553 h;
- required ETA: 5.3673997785 h;
- minimum headroom: 18,993,577,984 bytes, so memory passed;
- finite losses/gradients, GPU-only relation and no external contention passed.

The timing increase was not caused by the reliability lookup or relation
mathematics. The complete relation module averaged 2.0323810 s across the
measured interval, close to the sealed D2-AE value 1.9570280 s; the main
descriptive anomaly was rank-skewed DataLoader wait
`53.4023972 / 154.4085614 / 9.5764329 / 7.5339022` seconds, followed by
inclusive DDP/backward waiting. No post-hoc worker/thread/batch/architecture
sweep was authorized.

The user then explicitly accepted the measured ETA and authorized one
hash-bound waiver. The waiver did not reclassify the benchmark as passed:
the original status/classification and speed values remain failed, and the
waiver contract is
`8a2d11c0febea603ac74328fbcd51622982740c4bef48597a0af71de7a53da97`.

## Formal training and checkpoint-race continuation

The sole formal run was
`p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-20260729`, from seed-42 random
initialization. A DDP checkpoint-sidecar existence race stopped the first
attempt at 9,216,000 attempted windows / 4,500 updates. The failed record
(`a66fec685afb5cbb4079619de9417b7171af7e29244723f1deac9d4ba306d1b1`) and
partial archive (`b5573764eceb388f6a28f10b4ed89b44bbbcdd430213dad490f6c8b5caa7f9dd`)
remain preserved. The same run continued from the exact 6,144,000-window
checkpoint (`3c94f7344991cb38aab37fd8356cabe83a84b449d10505e0e46341490605287e`)
under the tracked continuation contract
`1a4ddf3b220b96f7aea0f1de7c0b8fd3fd9458eb913d284aaacc85a7fa226424`.

The accepted lineage completed exactly 61,440,000 windows, 983,040,000
frames and 30,000 optimizer updates. Actual GPU cost was 31,500 updates
(4,500 failed-attempt/replay plus 27,000 continuation updates). The final
online checkpoint hash is
`483c63ecaeb6dbf5a0a54400e0eecec722ff6df6d72226ce263e7fe053e412e2`;
the final model-state hash is
`7b6e333724f21490c96a0599103cc7eb087b9452e64a8d3c2b9a5ce85ae704bb`.

The recovered accepted-lineage throughput was 3,218.215477 windows/s
(5.303 h equivalent), and continuation throughput was 3,232.575359
windows/s. Thus the actual training finished materially earlier than the
8.166 h benchmark extrapolation: the benchmark's rank-skewed loader stall did
not recur at the same magnitude. The serialized resume raw throughput is
explicitly not reported as full-lineage throughput. Accepted-lineage wall,
total GPU cost and both throughput definitions remain separately recorded.

Formal hashes include metrics
`25b172f21d78d97412cb4eeeb79b43566d7e488286c383127a4edf0272c11903`,
manifest `49371a577a037444aef47fd5fda64f5d147ecd712247308b99b675d1edee55d3`,
training state `8dcb3ea4e1e39d661bcef138de6ff347731db8eeb88213fe0b4e0ba83204f8a4`,
and formal-completion verification
`a6263835cf79c6b803275c3d9c96c269aa1c2e75b1c8fea3fce4b4b56f7f1ec1`.
The recovered tree is 156 files / 7,227,356,886 bytes,
`9bff3d9a182138ee30ca586b10d71f689e6aa0c7345d2b1052fe0ea15251dc6c`.

## Fixed internal causal diagnostic

The fixed internal run
`p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-s42-20260730` used the
sealed 64-sequence / 192-window D2-O cohort, phase offsets `(14,56,98)`,
five paired 500-step paths (`full_rho`, `unit_rho`, gate-ablated,
temporal-permuted and left/right-swapped), shared noise/conditions/history,
and 10,000 sequence-level bootstrap replicates.

The provenance and numerical contracts passed, but all seven preregistered
mechanism checks failed. The internal classification is therefore
`diffusion-reliability-internal-unused-continue-native`:

| comparison | point | 95% CI |
|---|---:|---:|
| full − unit-rho direct union 5-cm F1 | -0.00175350 | [-0.00873145, 0.00530813] |
| unit-rho − full GT-contact distance | 0.02683317 cm | [-0.08000149, 0.15226497] |
| full − gate-ablated direct union 5-cm F1 | 0.01289764 | [-0.02708867, 0.05341855] |
| gate-ablated − full distance | 0.47677635 cm | [-0.05841019, 1.05297888] |
| full − temporal-permuted direct union 5-cm F1 | 0.00216556 | [-0.03124416, 0.04387780] |
| temporal-permuted − full distance | 0.23349130 cm | [-0.31420467, 0.89456561] |
| full − role-swapped left/right macro-F1 | 0.01320221 | [-0.00803831, 0.03569142] |

The learned gate was `-0.0922407731`, but a nonzero parameter is not causal
evidence. The full-path relation appendix, raw five-path artifacts and paired
noise/conditioning artifacts remain preserved; the internal recovered tree is
17 files / 224,452,243 bytes,
`5d28e3abc02dcf62f781270fd0391e44f64f4172b7ff705257995be63faffeee`.

## Fixed native evaluation

The unchanged official evaluator ran 438 sequences × 3 windows with 500
unguided steps, final-online weights, seed 42 and 10,000 paired sequence
bootstraps. CFG, guidance, scene conditioning, dynamic perception and
consistency were off. D2-X and sealed D2-AE aggregate/per-sequence outputs
were reused without regenerating or loading their checkpoints.

Target point estimates:

| metric | D2-AF0 |
|---|---:|
| end-object | 5.57348 cm |
| Txy | 4.60799 cm |
| FS | 0.359582 |
| contact precision / recall / F1 | 0.790934 / 0.599041 / 0.641055 |
| Pbody | 3.58933 |
| hand penetration | 0.226889 |
| MPJPE | 12.42210 cm |
| Troot / Tobj | 8.44769 / 17.30380 cm |
| Oobj | 1.000725 |

Against D2-X, contact F1 was `+0.0036291`, CI
`[-0.0166731, 0.0241940]`; recall was `+0.0045862`, CI
`[-0.0184078, 0.0281108]`. Released gap closure was only 4.03985% and the
contact-F1 point estimate (0.641055) missed 0.6598838781.

The D2-AE single-factor repair failed for contact F1, recall and end-object
ratio; only its FS ratio subcheck passed. D2-X protection failed for
end-object (CI upper 1.57301), Txy (1.18020) and Tobj (1.10990). The fixed
181-sequence penetration mask contract passed exactly, with ID hash
`2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`.
Released-95%-effectiveness also failed. These failures determine the
headline classification; the internal negative result does not authorize a
retry or a new HOIPrior direction.

The native recovered tree is 16 files / 2,514,430 bytes,
`40a9925468e54966f726b2cccec4f55aa53caa92f2a0da188dccc435ebc5bd21`.
Metrics, manifest, aggregate and per-sequence hashes are
`94fc71cd3d3fbbe87ac6ec38246e39fb0c965d630fd7626c604f0983a1248f56`,
`10498fb42a02501859cfd0aaab484a7a606ee5f711134d7f009519723655de06`,
`417c245df047e4fd7724c7ddcc7f0884fffd5bda934fefe465fb904da400f488`, and
`7252931861dd2d4e60476a05cd7dd35d67aa7369995687de2dc9bcbc67c8acd5`.

## Recovery, operational noise and verification

Worker-initiated non-destructive recovery used no `--delete`. Formal,
internal and native worker/authority trees matched exactly, and all checksum
dry-runs exited 0 with zero itemized differences. Internal artifact-closure
entries and native raw aggregate/per-sequence/log/resolved-target entries
matched their embedded hashes.

Three wrapper issues are retained and classified as operational only:

1. an internal launch wrapper compared a Python executable to its unresolved
   symlink and exited before creating a run directory or workload;
2. an internal recovery tree-hash wrapper ran outside the repository root and
   raised `ModuleNotFoundError: tools`, after transfer had already completed;
3. a native recovery wrapper first created an empty staging directory without
   transferring files, after which the same directory was filled by compliant
   non-destructive rsync.

None changed a run id, source, metric, checkpoint, scientific classification or
artifact. The compact result is
`experiments/results/p1_hoi_phase1b_d2af_sqrt_alpha_bar_reliability_s42_20260730.json`.

## Exact next entry point

Do not merge or tag this negative phase, do not select or resume the D2-AF
checkpoint, and do not run D2-AF1, a longer budget, a performance sweep,
consistency, Mixer or any further HOIPrior direction. The next independent
session may begin only with a dated, plan-only Phase 1C HSIPrior
preregistration and must initialize HSIPrior from random weights, never from
released, author, D2-X, D2-AE or D2-AF checkpoints.
