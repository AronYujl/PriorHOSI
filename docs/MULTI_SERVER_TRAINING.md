# Multi-server expert training

This is the operational contract for concurrent expert work. The 8-GPU host
`10.184.17.253` is authoritative for development, review, the append-only
registry, and Phase 1C HSI execution. The 4-GPU host `10.181.9.214` is a Phase
1B HOI execution worker. Windows remains an SSH client; it is not a source of
training inputs or reportable artifacts.

The worker is not provisioned merely by copying the repository directory. It
needs a committed Git revision, a verified Linux environment, an immutable
OMOMO-only data snapshot, evaluator assets, and an explicit return path for
manifests/checkpoints/logs. Do not start Phase 1B until Steps 1-8 pass.

Connectivity was measured on 2026-07-13: ICMP works both ways, inbound TCP/22
from the 8-GPU host to the worker times out, and TCP/22 from the worker to the
8-GPU host succeeds. Therefore every server-to-server transfer is initiated by
the worker. Windows is used only to copy the worker's public key text during
bootstrap; it must not relay bulk datasets, environments, or run artifacts.

## 1. Collect the four-GPU machine facts

The worker account is `yujinlun`, its home is `/home/yujinlun`, and project
storage is rooted at `/home/yujinlun/data`. The recorded preflight is Ubuntu
20.04.6, kernel 5.15, glibc 2.31, Python 3.8.10, 125 GiB RAM, synchronized NTP,
4.0 TiB free and 235 million free inodes under `/home`. It has four RTX 3090
24GB GPUs with driver 580.126.09. The authoritative host is also Ubuntu 20.04.6
with glibc 2.31, so a verified `conda-pack` relocation is appropriate.

The following commands remain the reproducible preflight and must be archived
again beside the first reportable Phase 1B manifest:

```bash
id
hostname -f
uname -a
cat /etc/os-release
nvidia-smi
df -h /home
df -i /home
free -h
timedatectl status
command -v git rsync tar sha256sum
python3 --version
```

At the initial check, GPU 2 had an unrelated Python process using about 3.5 GiB
and 35% utilization. All four GPUs must be idle for the reportable memory audit
and training, or the contention must be recorded and the affected run cannot be
used as the clean capacity decision.

Reserve substantially more than the roughly 14-16 GiB initial HOI training
snapshot: evaluation assets, the unpacked environment (currently about 9 GiB),
checkpoints, optimizer state, and immutable run directories need headroom. A
practical preflight target is at least 150 GiB free, with 250 GiB preferred for
multiple retained runs.

## 2. Use a fixed worker layout

Fixed paths, all below the worker account's existing `~/data` directory:

```text
/home/yujinlun/data/work/InfBaGel-release             # clean execution checkout
/home/yujinlun/data/envs/infbagel                     # relocated packed environment
/home/yujinlun/data/datasets/InfBaGel-p1b-omomo-v1    # immutable data snapshot
/home/yujinlun/data/transfer                          # staging only
/home/yujinlun/data/results                           # optional checkpoint backup root
```

In the checkout, `data` is a local symlink to the immutable snapshot. Never
copy the absolute `/data/...` symlink from the 8-GPU host.

## 3. Establish worker-to-authority SSH

The authoritative host intentionally rejects passwords. Create a dedicated key
on the worker; do not reuse or transmit its private half:

```bash
install -d -m 700 "$HOME/.ssh"
ssh-keygen -t ed25519 -a 100 -N '' \
  -f "$HOME/.ssh/id_ed25519_infbagel_8gpu" \
  -C 'node01-hoi-worker-to-10.184.17.253'
cat "$HOME/.ssh/id_ed25519_infbagel_8gpu.pub"
```

Copy only that one `.pub` line through the Windows clipboard. On the 8-GPU host,
append it to `~/.ssh/authorized_keys` with the restrictions below, replacing
`<WORKER_PUBLIC_KEY_LINE>` with the complete `ssh-ed25519 ...` line:

```bash
install -d -m 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"
WORKER_KEY='<WORKER_PUBLIC_KEY_LINE>'
ENTRY="from=\"10.181.9.214\",restrict $WORKER_KEY"
grep -qxF "$ENTRY" "$HOME/.ssh/authorized_keys" || \
  printf '%s\n' "$ENTRY" >> "$HOME/.ssh/authorized_keys"
unset WORKER_KEY ENTRY
```

The ECDSA host fingerprint independently verified on the authoritative host is
`SHA256:TAndo0bQIctobfT4jyGeuPBnLRNbxgbuVtho2JBz35A`. On the worker, test the
dedicated key explicitly:

```bash
ssh -i "$HOME/.ssh/id_ed25519_infbagel_8gpu" \
  -o IdentitiesOnly=yes yujinlun@10.184.17.253 'hostname; pwd'
```

An empty passphrase is intentional only for this restricted automation key. If
local policy requires a passphrase, use `ssh-agent` and document its lifecycle.
Do not add a forced command: Git, rsync, and hash checks require remote exec.

## 4. Pull committed code with Git

Phase 1B implementation is performed and committed on the authoritative host.
The worker reads that repository directly; no worker bare repository and no
source-tree rsync are needed:

```bash
mkdir -p "$HOME/data/work" "$HOME/data/transfer"
GIT_SSH_COMMAND="ssh -i $HOME/.ssh/id_ed25519_infbagel_8gpu -o IdentitiesOnly=yes" \
  git clone --branch research/state-compositional-priors \
  yujinlun@10.184.17.253:/data/yujinlun/InfBaGel-release \
  "$HOME/data/work/InfBaGel-release"
```

After `phase/01b-hoi` exists on the authority, the worker fetches and switches
to it using the same `GIT_SSH_COMMAND`. Before every reportable run, both hosts
must print the same `git rev-parse HEAD`, and the worker worktree must be clean.
Never commit source changes or force-push from the worker.

## 5. Replicate the verified environment

The current authoritative environment is Python 3.8, PyTorch 1.13.1+cu117, and
PyTorch3D 0.7.8. `requirements.txt` alone is insufficient because it does not
fully reproduce CUDA/PyTorch3D and compiled extensions. Prefer a one-time
`conda-pack` transfer when the worker OS/architecture and NVIDIA driver are
compatible.

On the 8-GPU host:

```bash
conda-pack -p /data/yujinlun/anaconda3/envs/infbagel \
  -o /tmp/infbagel-linux-x86_64.tar.gz
sha256sum /tmp/infbagel-linux-x86_64.tar.gz
```

Then pull from the worker:

```bash
rsync -avP -e 'ssh -i /home/yujinlun/.ssh/id_ed25519_infbagel_8gpu -o IdentitiesOnly=yes' \
  yujinlun@10.184.17.253:/tmp/infbagel-linux-x86_64.tar.gz \
  "$HOME/data/transfer/"
mkdir -p "$HOME/data/envs/infbagel"
tar -xzf "$HOME/data/transfer/infbagel-linux-x86_64.tar.gz" \
  -C "$HOME/data/envs/infbagel"
export INFBAGEL_PREFIX="$HOME/data/envs/infbagel"
export PATH="$INFBAGEL_PREFIX/bin:$PATH"
hash -r
command -v python
python --version
conda-unpack
export INFBAGEL_PYTHON="$INFBAGEL_PREFIX/bin/python"
"$INFBAGEL_PYTHON" -c \
  'import torch,pytorch3d; print(torch.__version__, torch.version.cuda, pytorch3d.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Record the tarball SHA-256, interpreter path, import output, `pip freeze`, and
`nvidia-smi` in the worker preflight. If `conda-pack` relocation or a compiled
extension fails, stop and build a separately pinned environment; do not fall
back to system Python or silently reinstall newer packages.

## 6. Transfer an OMOMO-only immutable snapshot

For Phase 1B, do not transfer `data/dataset` (LINGO), `data/hosi_test`, or any
`data/train/Scene*` / `data/test/Scene*` synthesized-scene directory. The initial
snapshot consists of the scene-free OMOMO train/test fields needed by the
Phase 1A contract plus `data/object` geometry/SDF for native-domain metrics.
Phase 1B may add a missing scene-free asset only by recording its path and hash.

Create the destination once:

```bash
mkdir -p "$HOME/data/datasets/InfBaGel-p1b-omomo-v1/data/train" \
         "$HOME/data/datasets/InfBaGel-p1b-omomo-v1/data/test" \
         "$HOME/data/datasets/InfBaGel-p1b-omomo-v1/data/object"
```

Use `rsync -aH --partial --info=progress2` for the required top-level files and
the following directories in both `train` and `test`:

```text
betas.npy, gender.pkl, human_joints_aligned.npy, human_orient.npy,
human_pose.npy, transl_aligned.npy, rest_human_offsets_aligned.npy,
start_idx.npy, end_idx.npy, norm.npy, object_name.pkl, object_trans.npy,
object_rot_mat.npy, scene_name.pkl, clip_features.npy, text2features_idx.pkl,
language_motion_dict/, cano_object_bps_npy_files_joints24_120/,
contact_label_npy_files/, rest_object_geo/
```

For `test`, also copy `seq_id.pkl`. Copy the complete `data/object/` tree. Do not
copy `object_points.npy` unless the final Phase 1B scene-free implementation
explicitly requires and registers it; the current loader treats it as optional.

After the first transfer, repeat the same rsync command with `--checksum
--dry-run`; it must report no differences. Then link the snapshot:

```bash
cd "$HOME/data/work/InfBaGel-release"
ln -s "$HOME/data/datasets/InfBaGel-p1b-omomo-v1/data" data
```

Run the Phase 1A HOI audit on the worker and compare every `source_hashes` entry
and `contract_sha256` with
`experiments/results/p1_data_hoi_contract_s42_20260713.json`. Store the new audit
under an ignored preflight run directory; do not overwrite the tracked aggregate.
The expected tracked aggregate SHA-256 is
`1deea6a724a3319d4c5654da682d7f51af7e5c93b119d159bd2b37ad258f627f`.

## 7. Transfer evaluation-only dependencies

Full Phase 1B evaluation also needs the pinned CHOIS evaluator assets and SMPL
models. Transfer these separately after their existing hashes/commits are
checked:

```text
third_party/text-to-motion/                         pinned Git checkout
third_party/chois_omomo_evaluator_assets/           about 6.2 GiB
smpl_models/                                        about 2.2 GiB
```

`smpl_models` is required before the HOI dataset can derive its kinematic
parent table, so sync it before unit tests or a forward smoke. It is a static
kinematic asset, not scene supervision. The CHOIS assets may wait until native
evaluation is implemented, but must be present before the Phase 1B evaluation
gate.

These assets may evaluate HOIPrior outputs; they must never initialize HOIPrior.
Do not transfer `checkpoint/checkpoint.pth` as a training initializer.

## 8. Worker preflight before Phase 1B runs

In the worker checkout:

```bash
export ROOT_DIR="$(git rev-parse --show-toplevel)"
export INFBAGEL_PYTHON="$HOME/data/envs/infbagel/bin/python"
export INFBAGEL_WORKER_EXPERT=hoi
test -L data
git status --short
git rev-parse HEAD
git rev-parse exp/p1a-data-v1^{}
"$INFBAGEL_PYTHON" -m unittest discover -s tests -v
"$INFBAGEL_PYTHON" tools/experiment.py validate
```

Also archive hostname, time, `nvidia-smi`, `df -h`, environment hashes, data
audit comparison, and a one-GPU import/forward smoke. The actual four-GPU memory
audit remains Phase 1B work and must use `tools/experiment.py start/finish/register`
from a clean worktree.

With `INFBAGEL_WORKER_EXPERT=hoi`, only tests that require real LINGO files may
be skipped. HSI representation/mask/model API tests still run, and HOI real-data
tests must pass. Never satisfy the suite by copying `data/dataset` to this worker.

## 9. Artifact ownership and return flow

Each run id has one writer. While a worker run is active:

- the worker alone writes `results/experiments/<run-id>/` and its checkpoint tree;
- the 8-GPU host may continue development only on a different branch/run id;
- neither host syncs into the other's live run directory;
- the tracked append-only registry has one integrator at a time.

After `finish`, stop the workload and copy its entire immutable run directory to
a staging path on the 8-GPU host using `rsync -aH --partial`. Verify a tree hash,
then register/commit the aggregate metadata on the owning phase branch. Large
checkpoints stay ignored and are copied to redundant storage; registry records
contain their hashes and worker-local provenance. Never use `rsync --delete` for
results, checkpoints, repositories, or registries.

For future concurrent HOI and HSI training, use distinct phase branches, run ids,
result directories, and machine roles. Integration remains sequential: fetch a
completed worker branch, verify its manifests/hashes/tests on the authoritative
host, and merge only after that phase gate passes.

## Remaining provisioning checks

- install and test the dedicated worker-to-authority public key;
- confirm whether the worker can reach GitHub/Conda (not required for the
  preferred packed-environment path, but useful for recovery);
- decide whether checkpoints need a second storage location beyond `/home`;
- confirm all four GPUs are idle before reportable capacity/training runs.

Passwords, private keys, and tokens are never required in the repository or chat.
