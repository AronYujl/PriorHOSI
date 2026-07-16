# D2-L0 Fixed Auxiliary-Balance Diagnostic

`tools/diagnose_hoi_d2l.py` performs one frozen, zero-update comparison between
the production HOIPrior objective and a single preregistered gradient-balanced
auxiliary counterfactual. It does not sweep weights or modify the production
loss function.

The balanced FK and object-surface weights are derived only from the sealed
D2-I0 high-noise gradient records. Across both checkpoints, timesteps 250/499,
and all 32 registered blocks, each raw auxiliary gradient is matched to the
geometric mean of `sqrt(human_norm * object_norm)`. This fixes FK to
`0.3569973401779424` and object surface to `0.4772322188400037`; velocity,
terminal goal, and all reconstruction-field weights remain unchanged.
The source metrics file is hash-verified and the derivation is replayed before
the D2-L workload reads its fresh cohort.

The fresh cohort is D0 ranks 898--1025: 128 nonterminal internal-validation
windows with ordered global-index SHA-256
`b5faa79316c6bd7aa9df0687a2554d458a459bd331c94648a99380d5c3b43a75`.
It is disjoint from D2-H/I/J/K. Current and balanced candidates use identical
samples, conditions, and q-noise.

For each candidate the diagnostic reconstructs the complete field and
auxiliary gradient formula, applies the production global max-norm-1.0 clip,
and computes the exact next descent direction from cloned sealed AdamW states.
Both candidates report all fields, loss aggregates, directions, and parameter
groups. Model, raw optimizer, and device-mapped moment hashes are checked before
and after; no optimizer is created and no parameter or moment is written.

Generate the resolved config before `tools/experiment.py start`, archive live
preflight beside the manifest, and execute in a worker-owned persistent session.
After finish/register and immutable recovery, use
`tools/summarize_hoi_d2l.py` for the tracked compact aggregate. Positive and
negative classifications both stop at D2-L0 and do not authorize smoke,
D2-H1, training, or production loss changes.
