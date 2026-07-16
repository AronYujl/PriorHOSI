# D2-K0 Sealed-AdamW Routing Diagnostic

`tools/diagnose_hoi_d2k.py` performs one frozen, zero-update replay of the
next AdamW descent direction stored in each sealed online HOIPrior checkpoint.
It tests whether the terminal optimizer moments and per-coordinate second
moment preconditioning rescue the weak high-noise human routing observed in
D2-J0. It does not create an optimizer, call `backward` or `step`, populate
parameter `.grad` buffers, or write model, optimizer, or checkpoint state.

The primary cohort contains the first 128 nonterminal windows found by scanning
the deterministic D0 internal-validation ordering from rank 768. Terminal ranks
768 and 770 are skipped, so the selected ranks are 769--897 and their ordered
global-index SHA-256 is
`747c0b1c881e150a8ccdb8675044a877b1ab32f615169ea9e3577dcff0a3f90a`.
This cohort is disjoint from D2-H0, D2-I0, and D2-J0. Both checkpoints use the
same stable q-noise for every fixed 16-window block and timestep.

For every block the diagnostic reconstructs the locked weighted objective with
`torch.autograd.grad`, applies the production global max-norm-1.0 clipping
coefficient, and reads cloned AdamW `step`, `exp_avg`, and `exp_avg_sq` tensors.
It then reports the bias-corrected historical, current-gradient, decoupled
weight-decay, and full AdamW gradient-like descent directions. The sealed
optimizer mapping is required to cover all 119 model parameters in exact order
with the registered hyperparameters. Model, raw optimizer, and device-mapped
moment hashes are checked before and after the audit.

The sealed metrics contain all five representation fields, all registered loss
aggregates, all seven timesteps, both checkpoints, every block, and the same
eight parameter groups used by D2-I/J. The gate is the preregistered conjunction
over timesteps 250 and 499; the other measurements are descriptive and cannot
replace a failed gate.

Generate the exact resolved config before `tools/experiment.py start`, archive
the live preflight and hardware snapshot from the workload context, and run the
diagnostic in a worker-owned persistent session. After `finish`, `register`, and
immutable artifact recovery, use `tools/summarize_hoi_d2k.py` for the tracked
compact aggregate. Positive and negative classifications both stop at D2-K0;
neither authorizes optimizer reset or ablation, clipping or loss changes,
D2-H1, smoke testing, training, or any production behavior change.
