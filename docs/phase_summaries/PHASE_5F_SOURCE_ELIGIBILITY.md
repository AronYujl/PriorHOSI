# Phase 5.5.2a - Source-only transition eligibility

Date: 2026-09-10
Branch: `phase/05e2-multitask-execution`
Run: `p5-multitask-source-eligibility-r1-s42-20260910`
Implementation: `ed1a4c261f638e2f8825fb3eb4a7ef2d208ffa46`

This subphase applies the approved source-only membership rule. It keeps all 469
original HOSI tasks and all fixed seed-42 LINGO source exclusions. It does not
load a checkpoint, use a generated motion, or claim an episode success result.

## Contract

`omomo_to_lingo` requires the OMOMO task-reference terminal to be upright,
foot-supported, object-supported and slow, with hand markers released.
`lingo_to_omomo` requires both OMOMO entry hands to remain separated from the
object over the initial context. The candidate LINGO source interval is retargeted
to the OMOMO body and target HOSI scene. Every source frame is checked against
scene SDF, persistent-object SDF, floor and bounds. Source scenes remain
provenance only.

The source audit uses the first eight eligible LINGO records by `data_idx` for
each action type as a bounded, source-ordered pool. The complete catalogue and
the 1198 pool-not-attempted records remain in the run artifacts. No selection
score is derived from a model output.

## Results

The inventory contains 469 OMOMO tasks, 1214 eligible LINGO sources and 912
LINGO exclusions. It produced three `omomo_to_lingo` candidates and no
`lingo_to_omomo` candidates. All 469 reverse-direction OMOMO initial contexts
failed the source hand-release guard, so the zero is independent of the bounded
LINGO pool. In the forward direction, 460 OMOMO source states were ineligible,
two failed OMOMO target-scene geometry, 67 LINGO full-interval geometries failed,
and three passed.

The three accepted candidates are source-only records for `hosi-027`, `hosi-327`
and `hosi-397`. Their complete LINGO intervals contain 116, 91 and 91 frames.
Across accepted records, source body scene penetration maximum is 0, source body
object penetration maximum is 1.430 cm, source object scene penetration maximum
is 1.724 mm, and source object floor penetration is 0. These are source-motion
geometry witnesses, not generated-chain results.

## Failure and verification

The first reportable run `p5-multitask-source-eligibility-s42-20260910` failed
before candidate construction because the per-frame object position lacked its
vertex dimension in the geometry query. The failure manifest and diagnostic are
retained. The corrected r1 run completed in 14.57 seconds on GPU 0 with no model
samples. The full authority suite after the fix passed **1157 tests** with 4
historical skips. The registry contains both the failed operational run and the
completed source inventory.

## Next entry

The exact next subphase is 5.5.2b actual-history execution on the three accepted
forward candidates. It must evaluate the runtime predecessor guard, use actual
human/object history, generate Kimodo transitions where required, and score every
transition frame. The reverse direction has no source-eligible task in this fixed
source catalogue and is not to be synthesized by relaxing the hand-release rule.
