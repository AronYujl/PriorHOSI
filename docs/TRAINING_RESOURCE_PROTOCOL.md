# Expert Training Resource Protocol

This protocol supersedes the cross-expert batch constraint recorded during
Phase 0 and Phase 1A. Those immutable results remain valid smoke evidence.

## Objective

Train the strongest HOIPrior and HSIPrior independently. Batch size is not a
controlled variable across these different data domains. The larger-memory
expert may use the 8×RTX 3090 server while the smaller expert trains
concurrently on the 4×RTX 3090 server.

## Batch selection

For each expert independently:

1. Find the largest stable per-GPU micro-batch that is arithmetically compatible
   with a candidate effective-batch tier on its allocated server, while retaining
   documented memory headroom and checking a multi-update stability soak, not
   only one update.
2. Select formal effective batch from `{512, 1024, 2048, 3072}`. Values such as 1536
   are invalid. A larger tier must be a power of two and requires a dated plan
   and registry amendment before execution.
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

For accumulation 1, the conventional topology-aligned micro-batches are:

| GPUs | EB 512 | EB 1024 | EB 2048 | EB 3072 |
|---:|---:|---:|---:|---:|
| 8 | 64 | 128 | 256 | 384 |
| 4 | 128 | 256 | 512 | 768 |

Smaller compatible micro-batches may reach the same tier with integer gradient
accumulation. The resource audit reports all compatible choices rather than
inventing a nonregistered effective batch.

## Phase 1A resource gate

Resource selection is part of Phase 1A, before expert training. The addendum
must freeze the exact formal HOI and HSI model/loss/config shapes, then audit
each on 8-GPU and 4-GPU topologies. Each candidate performs a real data
forward/backward; the selected configuration additionally completes at least
30 optimizer updates to expose fragmentation and transient peaks. Results record
peak memory, headroom, throughput, data-loading contention, and every failed/OOM
candidate. Lightweight scaffolds and checkpoints or manifests without locked
research-branch provenance cannot satisfy this gate.

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
