# Worker2 checkpoint-to-artifact workflow

This workflow turns a trusted checkpoint on the authority host into returned
motion parameters without modifying either expert worktree.  Its current
registered scope is post-repair P12 HOIPrior inference with sealed Arm B
guidance; HSI and mixer profiles will be added only after their exporters and
inference contracts stabilize.

## One-input start

Run from the clean `visualization/renderer` worktree with the verified
authority environment:

```bash
INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
CHECKPOINT=/absolute/authority/path/to/checkpoint.pth

"$INFBAGEL_PYTHON" tools/worker2_inference.py start "$CHECKPOINT"
```

`CHECKPOINT` is the only required scientific argument.  The tool reads trusted
checkpoint metadata and SHA256, auto-selects a compatible profile, derives a
non-reusable run id, checks an idle worker2 GPU, resolves the full Hydra config,
and starts a persistent worker-owned systemd unit.  It returns immediately;
the unit performs generation, verification, and artifact return even if the
interactive control tunnel later disconnects.

Inspect a returned run id with:

```bash
"$INFBAGEL_PYTHON" tools/worker2_inference.py status RUN_ID
```

If generation completed but the final network return failed, retry only the
idempotent return stage—never generation—with:

```bash
"$INFBAGEL_PYTHON" tools/worker2_inference.py retry-return RUN_ID
```

Use `start ... --dry-run` to inspect the selected profile, exact paths, command,
source commit, and destination without SSH, transfer, artifact creation, or GPU
execution.

## Why a profile is required

A checkpoint's `git_commit` describes its training start, not necessarily the
correct later evaluator.  The P12 checkpoint records training commit
`25931627a7a5668598e3120f57762546306c13a7`; its validated frame-repaired
inference source is `8742d1a3b88800161324a8e45c597ffafdcbb607`.

[`worker2_profiles.json`](../../tools/visualization/worker2_profiles.json)
therefore pins:

- checkpoint metadata match rules;
- exact inference commit and Hydra config;
- deployable guidance overrides;
- expected sequence count and required assets;
- post-export coordinate-frame interpretation.

Auto-selection fails closed if no profile matches.  A new HOI architecture,
HSIPrior, or PriorHOSI output requires one tested profile addition before the
one-input interface can accept it; the workflow never guesses a source commit.

## Fixed topology and transfer ownership

```text
authority control
  -> ssh -F /dev/null, 127.0.0.1:22216 reverse endpoint
  -> worker2 persistent systemd user unit

worker2 bulk data
  -> ssh/rsync -F /dev/null, direct campus route to 10.184.17.253
  -> checkpoint/Git pull and artifact push
```

- Worker2 root: `/data2/yujinlun/infbagel-inference`.
- Code: one detached checkout per pinned inference commit under
  `work/checkouts/<commit>`.
- Checkpoints: content-addressed under `checkpoints/by-sha256/<sha>/`.
- Worker jobs: `artifacts/jobs/<run-id>`.
- Authority results:
  `/data/yujinlun/InfBaGel-visualization-artifacts/worker2-inference/<expert>/<run-id>`.

Windows is a bootstrap client only.  Neither Windows nor its proxy participates
in checkpoint, Git, data, or artifact transport.

## Completion and failure contract

Before launch the workflow requires:

- a clean visualization worktree and clean pinned worker checkout;
- exact checkpoint SHA256 after worker2 pulls it;
- matching data/SMPL-X links and required assets;
- no compute process on the selected GPU;
- a fully resolved config containing no `${...}` interpolation;
- absent worker output, systemd unit, staging, and final destinations.

After generation it requires exactly the profile's sequence count for motion,
CHOIS predictions, and CHOIS ground truth.  It hashes every motion pickle,
records the checkpoint/config/Git state, and lets worker2 push raw motion,
Hydra/evaluation outputs, and provenance to an authority `.incoming` directory.
The authority verifies the returned motion hash manifest before the worker
atomically renames `.incoming` to the final directory.

Failures are retained.  Existing run ids, worker outputs, staging directories,
and final directories are never overwritten or reused.
