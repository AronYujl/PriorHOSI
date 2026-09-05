# Phase 2 input diagnostic handoff — 2026-09-05

The approved HSI input diagnostic completed. It changes the priority of the
previous code-review diagnosis: activating the untrained object-condition
embeddings has a small measured effect; the missing-motion view matters about
ten times more on the same generated states. Phase 2 composition remains open.

## Scope and implementation

P15 online plus Arm B generated a G=0 carrier. R2 final EMA was queried with
its fixed CFG w=1, with HSI posterior guidance off. Four scenes were selected
from start/goal metadata through existing partition bins 0,22,44,66; all seven
objects in each scene were included. Every generated window was observed at
reverse steps 499,400,250,100,10,1,0. The four input cells and repeated legacy
reference shared the human hypothesis, geometric query, text, goals and timing.

`mixer.diagnostics.HSIInputDiagnostic` is a passive named probe invoked by
`test_infbagel_hosi.py` under one config fragment. The diagnostic masks tokens
only at the denoiser call; its empty motion view keeps zero history and the
training-time forward-noise marginal on future object/contact channels. The
scene query retains the complete shared world. The query's existing CPU
vertex subsampling runs inside restored CPU/CUDA RNG state. No observer
prediction changes the carrier. Raw positions and FK are measured separately
in physical units; initial and generated histories remain separate.

The ordinary production sampler retains its prediction arithmetic. The only
refactoring separates scene querying from HSI model prediction. Core, expert
source, checkpoint files, training objectives and fixed raw/KIN results were
unchanged. The new input views are diagnostic interventions, not a promoted
production sampler or a learned mixer.

## Results and limits

All 28 episodes completed: 124 windows, including 96 with generated history,
868 observed states, and 8,680 HSI forwards. All four process exits and ten
bootstrap exits were zero; all scalar records are finite. The repeated
prediction differs by exactly zero over every recorded metric in every state.

The primary quantity is future 24-joint FK displacement on generated-history
windows, averaged over reverse steps 100,10,1,0 within each episode first.

| Input intervention | Mean cm | Episode 95% CI | Scene 95% CI | Episodes above 1 cm |
|---|---:|---|---|---:|
| Mask object goal/BPS tokens | 0.09526 | [0.08934, 0.10138] | [0.08734, 0.10319] | 0/28 |
| Restore empty motion channels | 0.99914 | [0.89189, 1.12687] | [0.90806, 1.09125] | 11/28 |
| Both | 1.01234 | [0.90452, 1.13922] | [0.92799, 1.09850] | 13/28 |

After masking tokens, the additional motion-view effect remains 1.00222 cm;
after restoring motion channels, the additional token effect is 0.09371 cm.
Thus the ordering is stable in both conditional comparisons. Both changes
together affect root/legs/hands by 0.85052/0.92640/1.32054 cm respectively;
mean global-rotation change is 1.35258 degrees. The initial-prefix joint
effect is 0.82959 cm, below the 1.01234 cm generated-history effect.

The seven noise levels, all body groups, raw-position measurements and both
conditional comparisons are in the compact result; full per-metric bootstrap
tables and all states remain in the returned artifact directory. Intervals
use 10,000 replicates and seed 42. Episode intervals describe these 28 selected
tasks. Four scene units provide limited evidence about new scenes. The 1 cm
count is an effect-size description, not a quality threshold.

The source-level token mismatch remains real, but the measured submillimetre
effect weakens its proposed role as the main explanation of the old rollout
failure. Motion input sensitivity is larger. Neither intervention was fed
back into a reverse chain, so neither establishes quality improvement or
measures error accumulation. The empty-view auxiliary noise is one paired
marginal construction; its live reverse-process design is still open.
No new HOSI completion, contact, collision or sliding claim is made here.
The fixed root/leg split and human-only scene guidance remain unresolved
mechanisms, and the earlier operator rejection remains in force.

## Verification, provenance and failures

Preregistration commit: `7926e49`. Implementation/config/tests commit:
`34b7331`. The completion commit contains this handoff, the compact result and
the registry completion. This diagnostic has no integration tag; the whole
Phase 2 gate has not passed.

Executed verification:

```text
INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
"$INFBAGEL_PYTHON" code/test_infbagel_hosi.py --config-name config_sample_hosi_input_diagnostic --cfg job --resolve
```

Final runtime-source authority suite: 903 passed, 6 skipped, 167.17 seconds.
The registered GPU diagnostic supplies real-data runtime verification. A
production performance benchmark is skipped because the production arithmetic
is unchanged. Observed diagnostic runtimes by shard were 165.27, 123.26,
111.33 and 90.09 seconds; these measure the probe, not production latency.

Worker execution used four idle RTX 3090 GPUs, the verified machine-local
infbagel environment, clean committed source, four archived resolved configs,
and a machine preflight beside the manifest. Generation and bootstrap ran in
worker-owned persistent sessions. Code publication and the single immutable
artifact recovery were worker-initiated. One checksum-only dry-run comparison
reported no differences. Source/input identities are retained by the existing
manifest and checkpoint metadata; no hashing mechanism was added.

Artifacts:

- Compact result: `experiments/results/p2_mixer_hsi_input_s42_20260905.json`.
- Authority return: `results/incoming/p2-mixer-hsi-input-s42-20260905/` (124 MiB).
- Worker original: `/home/yujinlun/data/work/InfBaGel-mixer/results/experiments/p2-mixer-hsi-input-s42-20260905/`.
- The return contains the sealed manifest, resolved configs, preflight, logs,
  episode records, selected tensors, all paired reports and completion record.

No reportable workload failed. Local Git-index and SSH sandbox attempts were
retried with required permissions before the corresponding action executed.
There was no run restart or overwritten result.

## Exact next entry point

Read this handoff, `docs/plan/OVERVIEW.md`, and the latest dated sections of
`docs/plan/PHASE_2_COMPOSITION.md`. Keep this pilot separate from the frozen
469-episode raw/KIN results. Do not describe the token mismatch as an
established dominant root cause or the empty-view sensitivity as an observed
quality repair.

The next concrete proposal should define an HSI input view for a shared chain,
then a relational composition prototype whose root, legs and object can adjust
together. Its criterion must distinguish HSI denoiser information from the
same geometric guidance without HSI predictions. A same-state marginal probe
does not choose a reverse sampler for empty channels: state its temporal
coupling and demonstrate it before claiming adapted rollout quality.
Joint training supervision, object-scene feasibility and contact/stance
constraints must be specified before training a residual network. Further
weight tuning of the rejected fixed body split is not the next direction.
