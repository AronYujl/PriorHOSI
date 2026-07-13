# Multi-server expert training

This is the operational contract for concurrent expert work. The 8-GPU host
`10.184.17.253` is authoritative for development, review, the append-only
registry, and Phase 1C HSI execution. The 4-GPU host `10.181.9.214` is a Phase
1B HOI execution worker. Windows remains an SSH client; it is not a source of
training inputs or reportable artifacts.

The worker is not provisioned merely by copying the repository directory. It
needs a committed Git revision, a verified Linux environment, an immutable
OMOMO-only data snapshot, evaluator assets, and an explicit return path for
manifests/checkpoints/logs. Do not start Phase 1B until Steps 1-7 pass.

## 1. Collect the four-GPU machine facts

Run the following on `10.181.9.214` and retain the output. Replace
`<FOUR_USER>` below with the actual account name.

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

Also confirm whether the 8-GPU host can SSH directly to `<FOUR_USER>@10.181.9.214`
and whether the four-GPU host can reach GitHub/Conda. Do not share passwords or
private keys in chat. A dedicated SSH key may be installed in the worker's
`~/.ssh/authorized_keys`; retain the private key only on the initiating host.

Reserve substantially more than the roughly 14-16 GiB initial HOI training
snapshot: evaluation assets, the unpacked environment (currently about 9 GiB),
checkpoints, optimizer state, and immutable run directories need headroom. A
practical preflight target is at least 150 GiB free, with 250 GiB preferred for
multiple retained runs.

## 2. Use a fixed worker layout

Recommended paths, all below the worker account's `/home`:

```text
/home/<FOUR_USER>/git/InfBaGel-release.git          # bare transfer repository
/home/<FOUR_USER>/work/InfBaGel-release             # clean execution checkout
/home/<FOUR_USER>/envs/infbagel                     # relocated packed environment
/home/<FOUR_USER>/datasets/InfBaGel-p1b-omomo-v1    # immutable data snapshot
/home/<FOUR_USER>/transfer                          # staging only
```

In the checkout, `data` is a local symlink to the immutable snapshot. Never
copy the absolute `/data/...` symlink from the 8-GPU host.

## 3. Publish code with Git, not rsync

After direct SSH from the authoritative host works, initialize a bare repository
once on the worker:

```bash
ssh <FOUR_USER>@10.181.9.214 \
  'mkdir -p "$HOME/git" "$HOME/work" "$HOME/transfer" && git init --bare "$HOME/git/InfBaGel-release.git"'
```

On the 8-GPU host, add a narrowly named remote and publish the current research
branch and immutable Phase 1A tag:

```bash
git remote add hoi-worker \
  <FOUR_USER>@10.181.9.214:/home/<FOUR_USER>/git/InfBaGel-release.git
git push hoi-worker research/state-compositional-priors
git push hoi-worker exp/p1a-data-v1
```

Clone locally on the worker. Phase 1B implementation is performed and committed
on the authoritative host; only then is its exact `phase/01b-hoi` revision pushed
and checked out on the worker.

```bash
ssh <FOUR_USER>@10.181.9.214 \
  'git clone --branch research/state-compositional-priors \
     "$HOME/git/InfBaGel-release.git" "$HOME/work/InfBaGel-release"'
git push hoi-worker phase/01b-hoi
ssh <FOUR_USER>@10.181.9.214 \
  'cd "$HOME/work/InfBaGel-release" && git fetch origin && git switch phase/01b-hoi'
```

Before every reportable run, the two hosts must print the same `git rev-parse
HEAD`, and the worker worktree must be clean. Never force-push a run commit.

## 4. Replicate the verified environment

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
rsync -avP /tmp/infbagel-linux-x86_64.tar.gz \
  <FOUR_USER>@10.181.9.214:/home/<FOUR_USER>/transfer/
```

On the worker:

```bash
mkdir -p "$HOME/envs/infbagel"
tar -xzf "$HOME/transfer/infbagel-linux-x86_64.tar.gz" \
  -C "$HOME/envs/infbagel"
"$HOME/envs/infbagel/bin/conda-unpack"
export INFBAGEL_PYTHON="$HOME/envs/infbagel/bin/python"
"$INFBAGEL_PYTHON" -c \
  'import torch,pytorch3d; print(torch.__version__, torch.version.cuda, pytorch3d.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Record the tarball SHA-256, interpreter path, import output, `pip freeze`, and
`nvidia-smi` in the worker preflight. If `conda-pack` relocation or a compiled
extension fails, stop and build a separately pinned environment; do not fall
back to system Python or silently reinstall newer packages.

## 5. Transfer an OMOMO-only immutable snapshot

For Phase 1B, do not transfer `data/dataset` (LINGO), `data/hosi_test`, or any
`data/train/Scene*` / `data/test/Scene*` synthesized-scene directory. The initial
snapshot consists of the scene-free OMOMO train/test fields needed by the
Phase 1A contract plus `data/object` geometry/SDF for native-domain metrics.
Phase 1B may add a missing scene-free asset only by recording its path and hash.

Create the destination once:

```bash
ssh <FOUR_USER>@10.181.9.214 \
  'mkdir -p "$HOME/datasets/InfBaGel-p1b-omomo-v1/data/train" \
             "$HOME/datasets/InfBaGel-p1b-omomo-v1/data/test" \
             "$HOME/datasets/InfBaGel-p1b-omomo-v1/data/object"'
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
cd "$HOME/work/InfBaGel-release"
ln -s "$HOME/datasets/InfBaGel-p1b-omomo-v1/data" data
```

Run the Phase 1A HOI audit on the worker and compare every `source_hashes` entry
and `contract_sha256` with
`experiments/results/p1_data_hoi_contract_s42_20260713.json`. Store the new audit
under an ignored preflight run directory; do not overwrite the tracked aggregate.
The expected tracked aggregate SHA-256 is
`1deea6a724a3319d4c5654da682d7f51af7e5c93b119d159bd2b37ad258f627f`.

## 6. Transfer evaluation-only dependencies

Full Phase 1B evaluation also needs the pinned CHOIS evaluator assets and SMPL
models. Transfer these separately after their existing hashes/commits are
checked:

```text
third_party/text-to-motion/                         pinned Git checkout
third_party/chois_omomo_evaluator_assets/           about 6.2 GiB
smpl_models/                                        about 2.2 GiB
```

These assets may evaluate HOIPrior outputs; they must never initialize HOIPrior.
Do not transfer `checkpoint/checkpoint.pth` as a training initializer.

## 7. Worker preflight before Phase 1B runs

In the worker checkout:

```bash
export ROOT_DIR="$(git rev-parse --show-toplevel)"
export INFBAGEL_PYTHON="$HOME/envs/infbagel/bin/python"
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

## 8. Artifact ownership and return flow

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

## Information still required before remote provisioning

- the four-GPU account name and exact `$HOME`;
- SSH reachability from `10.184.17.253` to `10.181.9.214` and desired key policy;
- OS, kernel, NVIDIA driver, four exact GPU records, RAM, free disk/inodes;
- whether Conda/GitHub access is available on the worker;
- whether training outputs need a second storage location beyond `/home`.

Passwords, private keys, and tokens are never required in the repository or chat.
