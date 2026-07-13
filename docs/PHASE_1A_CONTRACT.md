# Phase 1A Expert Contract

This contract is limited to Phase 1A scaffolding. It does not authorize full
training, model selection, a mixer, expert composition, state machines, or SDF
guidance.

## Shared representation

Both experts expose 16 frames, use the first two frames as a fixed history
prefix, and use a 500-step linear diffusion schedule. Coordinates remain Y-up,
window-local in XZ, and aligned to the initial root yaw. Position normalization
uses the existing OMOMO bounds in `data/train/norm.npy`. The author-replaced
`data/dataset/norm.npy` is byte-identical and is the only normalization used for
LINGO; no statistics are recomputed.

| Field | Half-open indices | Width | Semantics |
|---|---:|---:|---|
| joint positions | `[0, 84)` | 84 | 28 joints × XYZ |
| joint rotations | `[84, 216)` | 132 | 22 global rotations × 6D |
| object translation | `[216, 219)` | 3 | normalized dynamic-object XYZ |
| object rotation | `[219, 228)` | 9 | relative 3×3 matrix |
| contact | `[228, 232)` | 4 | human-object contact labels |

HSI fixes `[216, 232)` to empty and its loss mask is exactly `[0, 216)`. The
history prefix is excluded from both expert losses. Automated tests verify that
HSI output gradients are exactly zero in `[216, 232)`.

## HOIPrior

HOI uses OMOMO only. Its dataset item exposes motion, full instruction, BPS,
object goal/pose/contact, and `pi/end_pi/seq_length`; it exposes no scene field.
Its model API has no scene argument or scene encoder. Released InfBaGel
checkpoint initialization is rejected before model construction.

## HSIPrior

HSI uses real LINGO only and the immutable seed-42 scene-family split. It first
requires both hand-interaction frames to equal `-1`, then rejects source
sequences of length at most 48 because their generated 48-frame windows can
cross the declared sequence boundary. This second validity filter does not
alter the locked split. Dataset items expose real occupancy, text, and human
goals but no object BPS; object pose/contact representation channels remain
empty and masked from loss.

## Independence and initialization

`build_expert("hoi")` and `build_expert("hsi")` allocate distinct model types
and fresh learnable modules. Tests compare both Parameter object identities and
storage pointers. Any non-empty `init_checkpoint` fails fast for both experts.
