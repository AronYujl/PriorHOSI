# Phase 2.13 — fixed-source DP temporal teacher views (2026-09-07)

**Engineering complete; H1 is not supported by this diagnostic.** Removing the
manipulated-object overlay preserves the input contract but provides no resolved
HS/OS or contact improvement over the legacy teacher. Keep the view opt-in.
The next entry is teacher reconsideration, with the full Phase2 gate still open.

## Scope and implementation

The user approved the latest handoff, found at
`/data/yujinlun/report/PriorHOSI_Codex_Handoff_Phase2_13_2e69eba.md` (the supplied
`papers/` path was absent), and authorized direct execution. Base: sealed
Phase2.12 integration `2e69eba`; branch: `phase/02m-dp-evidence-views`.

`mixer/scene_views.py` requeries the native temporal lattice with object points
set to None. This preserves underlying walls, goal occupancy, anchor occupancy,
query-position conditioning and the real world/context. It supports the current
visual batch1 contract explicitly. The production default remains
`legacy_occupied`; `environment_only_temporal` is optional. The shifted observation
exists only inside passive diagnostics. Frozen core, expert networks, native
metrics, A*, source generation and the original object-voxel enumeration are
unchanged.

`scene_evidence_diagnostics.py:run_fixed_source_views` shares one noisy input,
known-empty trajectory, legacy geometric query and HOI pair across A/B/C. It
records raw x0/epsilon response, weighted motion directions, unscaled/scaled
relation gradients, six explicit gradients, group norms, dots/cosines, matched
physical rays and a separate same-source lambda0 short edit. All diagnostics return
the raw source exactly. Replay context, HOI arguments, offsets, optional BPS and
probe motions are saved with the existing development episode records. New config:
`config_sample_hosi_evidence_views.yaml`; tests remain component-organized.

## Conditions and exact replay

All24 existing development tasks/68 passive windows are retained: four OMOMO
internal-validation objects paired with LINGO training scenes. Calibration scenes
are004/006/055; verification scenes023/037/036. Replaying the original pipeline
recovers the query context absent from the old tensor records. Every raw source
and reconstructed reference is **bitwise equal in68/68 windows** to Phase2.11.

P15 online/Arm B500 and R2 final EMA remain frozen. Lambda26, beta1, levels
300/264/229/193/157/121/86/50, 67D parameterization, explicit terms and Armijo
settings are fixed. Candidates are reconstructed sources only. Three realizations
per window/level use deterministic window-seed offsets1000003*draw, all under
seed42. C is the preregistered spatial alternative: environment observations at
+2m along window-local X, keeping static inputs and true geometry fixed. There
is no donor search or calibration/verification exchange.

The complete diagnostic used9792 HSI forwards and4352 HOI teacher forwards,
including1088 HOI forwards for the independent lambda0 short edits. All4896
view records pass static-input/base-output equality and zero HOI reference delta.
Ambient CPU/CUDA RNG, scene backing storage and context remain unchanged. Actual
history/contact output is exact; maximum world-history discrepancy is7.15e-7.
Legacy unscaled gradient norm differs from the old record by at most8.75e-8
relative, from the float64 norm summary. No new candidate enters actual history.

## Direction and local physical results

The added object occupancy changes63/68 windows, averaging0.1102% of temporal
voxels. B and A directions are very similar: cosine mean0.9654, median0.9966.
B versus spatially shifted C has mean0.1498, median0.2544. B is stable across
noise draws (mean0.9926, median0.9961). Thus the teacher responds to the environment,
but removing the object overlay barely changes its usual direction. Average
unscaled parameter norms are0.000846735(A),0.000846283(B),0.001014818(C).
Torso and shared yaw have the largest mean scaled group norms; all group values
and separate position/rotation responses are retained in the analysis.

At source states with a nonzero HS gradient, B has460 positive versus716 negative
gradient dots;456 queries have zero HS gradient. Since updates use -g, negative
dots predict increased local HS energy. Stability and scene sensitivity therefore
do not establish a useful manipulation correction.

The following are **changes from reconstructed source in voxel/FK proxies, cm**,
averaged over draws/levels, then windows per task, then24 tasks. They are not native
mesh penetration metrics. The five-millimetre probe is an independent HSI-only
descent ray through the unchanged tanh parameterization.

| view,5mm physical RMS | HS residual delta | OS residual delta | contact-anchor drift delta | stance increment delta |
|---|---:|---:|---:|---:|
| A legacy | -0.003401 | -0.009362 | +0.926305 | +0.078213 |
| B environment | -0.002981 | -0.008821 | +0.924527 | +0.078671 |
| C shifted environment | -0.000964 | +0.004232 | +0.958492 | +0.081670 |

The primary5mm B-A HS delta is+0.000420cm, nominal95% paired task CI
[-0.000380,+0.001361]; scene CI[-0.000370,+0.001290]. OS is+0.000541cm, task
CI[-0.000555,+0.001671]; contact is-0.001778cm, task CI[-0.015635,+0.011087].
Both units are unresolved for these comparisons. B-C HS is-0.002016cm,
task CI[-0.011394,+0.007165]; OS-0.013054cm,[-0.030612,+0.003533]. Correct
environment has no resolved useful advantage over this negative control either.

At1mm, B-A has a favorable HS point delta-0.000625cm but task CI
[-0.001933,+0.000069] and scene CI[-0.001918,+0.000113] still cross zero.
OS and contact remain unresolved. The separate calibration/verification analyses
are retained; at5mm both roles have positive B-A HS point deltas. All intervals use
10000 seed42 paired replicates with shared NumPy index plans and GPU float64
arithmetic, using the existing paired-bootstrap pairing/metric-discovery rules.

Across68 windows at5mm, B versus A improves/worsens/ties HS in25/21/22, OS in
27/30/11 and contact in26/37/5. All preselected first-window examples are saved,
including favorable036/clothesstand (HS B-A -0.005776cm averaged across draws/levels)
and unfavorable006/suitcase (+0.020423cm). Their source/probe tensors are in
`views-shard02of06/episode-012.pt` and `views-shard00of06/episode-007.pt` respectively;
saved visual probe states use the preregistered first level/draw. These examples
do not define the aggregate or select a candidate.

## Feasibility, shared anchor and limitations

All9792 physical probes across both scales reach their registered amplitude;
the maximum5mm amplitude error is2.74e-8m. At each scale, domain-admissible
query counts are951/1632(A),949/1632(B),1007/1632(C). The common all-view subset
has802/1632 queries. On that subset at5mm, B also has smaller scene improvements
than A (HS -0.007927 versus-0.008995cm; OS -0.009704 versus-0.010647cm), with
contact drift+0.9963 versus+1.0116cm. These conditional summaries are descriptive;
every excluded query remains in the full report.

Sources contain46 joint collisions,16 object-only,3 human-only and3 collision-free
windows;38/68 have existing outside points. All68 have active grasp, so the
no-grasp stratum has zero samples. C introduces intentional static/dynamic mismatch;
mean geometrically exterior query fraction is0.3815%, versus0.0321% for B.

The same-source lambda0 eight-step editor changes physical RMS by6.408mm on
average, reducing HS/OS proxies by0.1463/0.2160cm, with contact drift0.0751cm and
stance increment0.1391cm. This is a descriptive geometric reference with different
solver budget and attained displacement, not an equal-amplitude comparison. It
does not establish a new rollout result or authorize promoting a view.

H1 is **not supported as a useful repair**; this does not rule out every object
overlay effect. H2 remains inconclusive: many rays are blocked, while useful
environment direction has not been demonstrated even in the common feasible
subset. H3 remains untested because these development scenes lack the native
mesh/SDF bundle. No native HS/OS/FS/completion or complete-rollout claim follows.

## Failures, fixes and verification

1. `p2-mixer-evidence-views-s42-20260907`: all six first-window saves failed on
   optional local_bps=None treated as a tensor. No episode was saved. All exits1,
   logs and the failed manifest remain. Wall57.44s; six attempted diagnostic
   windows add864 HSI/384 HOI teacher calls of operational overhead, inferred
   from the completed query routine preceding the failed save.
2. `p2-mixer-evidence-views-r1-s42-20260907`: all24 episodes/68 windows were saved,
   then all six final shard summaries failed on a nested Hydra ListConfig.
   Wall509.41s. Retain all exits1, truncated summaries and the failed manifest.
3. `p2-mixer-evidence-views-analysis-s42-20260907`: recover complete per-episode
   artifacts and complete metadata fields preceding the truncated sampler audit.
   The independent analysis succeeds. Original aggregates are never overwritten.

The first fix preserves None in replay records; the second converts diagnostics
configuration to plain containers at initialization. The end-to-end test now
uses a Hydra configuration, exercises optional BPS, saves motion records and
JSON-serializes the full editor audit. The final metadata conversion leaves the
executed numerical path unchanged; no GPU motion is generated again for recovery.

Full authority verification on the final code: **955 passed,4 historical-asset
skips,171.10s**. The skips are two historical HSI checkpoint-pair and two P8 asset
cases. Completion verification:955 passed/4 skipped in173.48s; registry366 records valid.
Earlier implementation/fix suites also passed; both real serialization
failures remain part of this record. Six resolved source configs and their retry
equivalence are archived. Registry and diff validation pass. Registered real
batch1 diagnostics provide functional and performance evidence; no separate smoke
or performance workload is added. Peak allocation504060928 bytes (480.71MiB).
Summed concurrent editor cost1480.65s includes170.49s teacher/query,1235.81s probes
and64.99s lambda0 short edits. These diagnostic costs are not production FPS.

## Artifacts, commits and next entry

- Compact: `experiments/results/p2_mixer_evidence_views_s42_20260907.json`.
- Source/config/context/motion/logs: `results/experiments/p2-mixer-evidence-views-r1-s42-20260907/`.
- Recovery manifest, exact analysis command and reports:
  `results/experiments/p2-mixer-evidence-views-analysis-s42-20260907/`.
- Analysis includes all paired units/roles, means, strata, feasibility, raw-response
  and gradient summaries, noise stability, lambda0 anchors, representative windows,
  and recovered checkpoint/shard metadata. Sealed input/checkpoint identities are
  referenced through the original manifests; no new identity mechanism is added.
- Preregistration5cdf98d; implementationc3e5c54; executed source/None fixf97c9b9;
  audit metadata fix and recovery source5823cdb. The completion commit contains
  this summary and compact result. Integration tag: `exp/p2m-dp-evidence-views-v1`.

Read this summary, OVERVIEW.md and the Phase2.13 section before a new session.
Keep P15/R2, lambda26 and all current source/geometry settings fixed. The precise
next entry is a review of the DP teacher's useful direction on these saved
contexts, using the response/gradient/physical hierarchy to distinguish a teacher
definition issue from execution constraints. Register one new mechanism before
implementation. Environment-only development rollout, full469, new dose/steps,
expert training and learned mixer remain deferred. This session closes only2.13.
