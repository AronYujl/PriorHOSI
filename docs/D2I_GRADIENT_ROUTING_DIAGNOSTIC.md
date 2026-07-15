# D2-I0 Frozen Gradient-Routing Diagnostic

`tools/diagnose_hoi_d2i.py` audits the geometry of the already locked Phase 1B
training objective without training or changing inference. It loads only the
two sealed online HOIPrior checkpoints, reconstructs the exact registered loss
sum, and uses `torch.autograd.grad` while leaving every parameter and `.grad`
buffer untouched.

The primary 128-window cohort is ranks 512--639 of the deterministic D0
internal-validation ordering. It is disjoint from D2-H0 ranks 0--511 and has no
terminal windows. A separate 64-window, D2-H0-disjoint terminal cohort records
the sparse terminal-goal component descriptively. Both selections and every
q-noise block are hash-locked.

For every checkpoint, cohort, timestep and fixed 16-window block, the artifact
contains loss values plus parameter-gradient norms and the complete cosine
matrix for all registered components and parameter groups. The direct total
gradient is replayed from its weighted components, and the model state dict is
hashed before and after the audit. The gate uses only all-parameter gradients
from primary-cohort timesteps 250 and 499; terminal and groupwise records cannot
substitute for a failed primary gate.

Generate the fully resolved config before `tools/experiment.py start`, then run
the exact archived command inside a worker-owned persistent session. Use
`tools/summarize_hoi_d2i.py` only after finish/register and immutable artifact
recovery. Positive and negative outcomes both stop at D2-I0: neither outcome
authorizes a loss-weight change, optimizer step, retraining, D2-H1, or a
production model/condition/sampler intervention.
