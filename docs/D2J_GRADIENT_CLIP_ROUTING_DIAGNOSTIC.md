# D2-J0 Global-Gradient-Clipping Routing Diagnostic

`tools/diagnose_hoi_d2j.py` audits the already locked Phase 1B training path
without training or changing inference. Formal HOIPrior training unscales AMP
gradients and then applies one global `clip_grad_norm_` call with max norm 1.0.
D2-J0 reconstructs that scalar clipping rule around frozen online checkpoints
and measures the direction efficiency delivered to each representation field.

The 128-window primary cohort is ranks 640--767 of the deterministic D0
internal-validation ordering. It is nonterminal and disjoint from D2-H0 ranks
0--511 and D2-I0 ranks 512--639. Its ordered global-index SHA-256 and every
16-window q-noise block are locked; both checkpoints use identical noise.

Every checkpoint, timestep, block, field, aggregate objective and parameter
group is written to the sealed metrics artifact. The record includes pre-clip
total-gradient norm, production clip coefficient, post-clip norm, a synthetic
PyTorch formula replay, direct-total/component-sum replay, and human/object
directional efficiencies. Model state is hashed before and after, model
`.grad` buffers remain empty, and no optimizer is created.

Generate the exact resolved config before `tools/experiment.py start`, archive
preflight and hardware from the workload context, and execute the diagnostic in
a worker-owned persistent session. After finish/register and immutable artifact
recovery, use `tools/summarize_hoi_d2j.py` to produce the tracked compact
aggregate. Positive and negative classifications both stop at D2-J0; neither
authorizes changing clipping, loss weights, training, D2-H1, or production
model/condition/sampler behavior.
