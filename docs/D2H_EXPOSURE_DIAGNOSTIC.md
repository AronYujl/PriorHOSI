# Phase 1B D2-H0 paired reverse-state exposure diagnostic

D2-H0 is a diagnostic-only comparison. It does not train, select, initialize,
resume, distill, project, clamp, or change the production reverse equation. The
only admissible checkpoints are the sealed online R-1024 and R-3072 artifacts
named in the dated preregistration.

For every locked internal-validation window and target timestep `t`, the tool
constructs a parent at `s=t+1` with a label-derived parent noise tensor. It then
uses the production posterior helper twice: once with true `x0` and once with
the checkpoint's parent prediction. Both calls receive the same parent state,
posterior noise, immutable two-frame history, condition, and registered
coefficients. The same checkpoint is evaluated again at target `t` on both
resulting states. True motion is used only by the oracle diagnostic posterior
and reference metrics; it is not a sampler condition.

The raw artifact retains paired per-window errors for all five representation
fields, physical object/pelvis/MPJPE measures, model-versus-oracle state
displacement, and all matched/text/BPS/pelvis/object-goal condition variants.
The compact tracked aggregate removes only per-window arrays. Its gate is the
exact conjunction registered in `docs/EXPERIMENT_PLAN.md`; implementation-parity
measurements are descriptive and cannot authorize a code intervention.

`tools/diagnose_hoi_d2h.py --resolve-only` must create the exact resolved JSON
before `tools/experiment.py start`. Runtime arguments must byte-semantically
match that archived configuration. The worker must run a clean exact committed
object with `INFBAGEL_WORKER_EXPERT=hoi` and its absolute verified Python. After
completion, `tools/summarize_hoi_d2h.py` creates the compact aggregate from the
immutable recovered artifact tree.

Regardless of whether the classification is
`reverse-state-exposure-positive-stop` or
`reverse-state-exposure-negative-stop`, D2-H1 remains unstarted until a later,
explicit user confirmation.
