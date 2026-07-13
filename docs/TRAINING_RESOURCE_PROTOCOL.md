# Expert Training Resource Protocol

This protocol supersedes the cross-expert batch constraint recorded during
Phase 0 and Phase 1A. Those immutable results remain valid smoke evidence.

## Objective

Train the strongest HOIPrior and HSIPrior independently. Batch size is not a
controlled variable across these different data domains. HSIPrior is assigned
to the 8×RTX 3090 server and HOIPrior to the 4×RTX 3090 server so they can train
concurrently.

## Batch selection

For each expert independently:

1. Find the largest stable per-GPU micro-batch that is arithmetically compatible
   with a candidate effective-batch tier on its allocated server, while retaining
   documented memory headroom and checking a multi-update stability soak, not
   only one update.
2. Select formal effective batch from `{512, 1024, 2048, 3072}`. Values such as
   1536 are invalid. Any additional tier requires a dated plan and registry
   amendment before execution.
3. Prefer accumulation 1. Increase accumulation only to reach the selected tier
   or for a preregistered optimization reason.
4. Jointly select/preregister learning rate and warmup with effective batch;
   larger batch is not assumed to produce a stronger expert automatically.

The invariant is:

`effective_batch = micro_batch_per_gpu × world_size × accumulation`

Thus the selected micro-batch must divide the selected effective batch after
multiplication by world size. Unused headroom should first support the intended
model/conditioning capacity or data pipeline; it must not be converted into a
nonstandard effective batch.

Memory and throughput auditing remains part of each expert-training phase:
Phase 1B audits HOI on four GPUs, and Phase 1C audits HSI on eight GPUs before
their respective reportable training starts. Phase 1A does not freeze the final
training micro-batch or run a cross-topology capacity audit.

With accumulation 1, the top registered tier corresponds to micro-batch 768 on
four GPUs and 384 on eight GPUs. These are candidates, not assumed-stable values.

## Budgets and comparisons

- Primary training budget: processed windows and frames, with epochs reported.
- Also record optimizer updates, but do not require equal update counts across
  HOI and HSI when their effective batches differ.
- Within one expert, every controlled architecture/loss/ablation comparison
  must keep hardware topology, effective batch, data budget, LR/warmup protocol,
  seed protocol, and evaluator fixed.
- Across experts, micro-batch, GPU count, effective batch, and optimizer updates
  may differ.

Every reportable manifest records allocated/visible GPUs, peak allocated and
reserved memory by rank, headroom, throughput, accumulation, effective batch,
processed windows/frames, optimizer updates, LR/warmup, and concurrent workload
or I/O contention.
