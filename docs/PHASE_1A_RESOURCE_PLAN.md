# Phase 1A Resource Audit Addendum

This addendum reopens only the Phase 1A resource gate. Phase 1B and Phase 1C
training remain blocked.

## Deliverables

1. Freeze independently implemented formal HOIPrior and HSIPrior architecture,
   condition, representation, loss, optimizer-state, and precision configs.
2. Record parameter counts and config/source hashes. Reject released InfBaGel
   initialization and any checkpoint from outside the locked research lineage.
3. Audit both experts on 8-GPU and 4-GPU RTX 3090 topologies with real domain
   batches. Search compatible micro-batches for effective batch tiers 512, 1024,
   2048, and 3072.
4. Preserve every stable, failed, and OOM candidate. For each selected topology,
   complete at least 30 optimizer updates and report peak allocated/reserved
   memory, headroom, windows/s, frames/s, loss, gradients, and I/O contention.
5. Recommend which server runs each expert and preregister the remaining
   LR/warmup choices before Phase 1B/1C.

## Arithmetic candidates with accumulation 1

| World size | EB 512 | EB 1024 | EB 2048 | EB 3072 |
|---:|---:|---:|---:|---:|
| 8 | 64 | 128 | 256 | 384 |
| 4 | 128 | 256 | 512 | 768 |

If a value is not stable, use a smaller divisor and integer accumulation. Do
not substitute an unregistered effective batch.

## Provenance boundary

The existing ignored `results/priors/benchmarks/` artifacts identify commit
`dd3ec896...`, which is not the locked research lineage and whose implementation
is absent from the current branch. They are not reportable inputs, cannot freeze
the formal architecture, and must not initialize either expert. They remain
untouched as user artifacts.
