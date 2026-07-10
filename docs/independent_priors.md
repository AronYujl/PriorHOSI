# Independent HOI and HSI Priors

This document defines the training and evaluation protocol for the two frozen
experts that will later be composed into a HOSI model.  It is part of the
experimental contract: changing a state layout, split seed, normalization, or
diffusion schedule creates a different prior and must be reported as such.

## Causal boundary

The HOI prior is trained only from OMOMO human-object data.  Its architecture
does not instantiate a scene encoder and cannot receive scene occupancy or a
scene-interaction goal.  In particular, the synthetic OMOMO bounding-box scene
is not loaded.

The HSI prior is trained only from LINGO human-scene data.  It receives real
scene occupancy but has no object state, object BPS encoder input, object goal,
or contact-label output.  This avoids learning the constant dummy object
channels used by the original mixed-dataset implementation.

Both priors retain the same 16-frame, stride-3 human representation and the
same 500-step linear diffusion schedule.  The distributed InfBaGel data uses
the OMOMO human normalization in both `data/train/norm.npy` and
`data/dataset/norm.npy`.  Every checkpoint stores a SHA-256 digest of the
normalization so a future composition stage can reject incompatible priors.

## State layouts

| Slice | Meaning | HOI | HSI |
|---|---|---:|---:|
| `0:84` | 28 global joint positions | yes | yes |
| `84:216` | 22 global joint rotations in 6D | yes | yes |
| `216:219` | object translation | yes | no |
| `219:228` | relative object rotation matrix | yes | no |
| `228:232` | four contact labels | yes | no |
| Total |  | 232 | 216 |

The contracts live in `code/models/priors.py`.  `HOIPrior` and `HSIPrior`
validate their input dimensionality at every forward call.

## Data splits

Window-level random splits are prohibited because adjacent windows overlap by
47 of 48 source frames.

- HOI validation holds out complete OMOMO source sequences using a stable hash
  with seed 2027.  Final evaluation uses the official `data/test` split.
- HSI validation holds out complete LINGO scenes using the same seed.  No
  sequence from a held-out scene is visible during training.
- A capped validation/evaluation set is sampled round-robin across held-out
  groups rather than taking the first contiguous windows.

The resolved split type, seed, counts, source folder, and normalization hash
are embedded in every structured checkpoint.

## Training

Run commands from `code/`.  The default configs use four GPUs and AMP.  Batch
size is per GPU.

```bash
cd code

# OMOMO, scene-free, 232-dimensional HOI prior
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_prior.py --config-name config_train_hoi_prior

# LINGO, real scenes, 216-dimensional HSI prior
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_prior.py --config-name config_train_hsi_prior
```

Resume all optimizer and AMP state from the last structured checkpoint:

```bash
python train_prior.py --config-name config_train_hoi_prior \
  resume_from=../results/priors/hoi_prior/checkpoints/last.pth
```

Useful debugging overrides are:

```bash
python train_prior.py --config-name config_train_hsi_prior \
  num_gpus=1 per_device_batch_size=1 num_workers=0 epochs=1 \
  max_steps_per_epoch=1 validation.max_batches=1 tensorboard=false
```

`max_steps_per_epoch` is only a smoke-test/debug control and should remain zero
for reported experiments.  Report the global batch size, number of optimizer
updates, best validation epoch, and total source windows seen, rather than only
the nominal epoch count, because LINGO contains substantially more windows.

### Losses

Both priors optimize x0 reconstruction with position MSE and rotation L1.
The HOI prior additionally uses object translation MSE, object rotation L1,
contact L1, transformed object keypoint loss, and hand/foot FK loss.  The HSI
loss cannot instantiate any object term; these entries are `None` in logs and
checkpoints rather than zero-valued dummy losses.

## Evaluation

The deterministic diagnostic evaluates identical Gaussian corruption at
timesteps `[0, 50, 100, 250, 499]`.  This is useful for selecting checkpoints
and detecting condition collapse.

```bash
# Official OMOMO test split
python evaluate_prior.py --config-name config_eval_hoi_prior \
  checkpoint=../results/priors/hoi_prior/checkpoints/best.pth

# Scene-held-out LINGO validation split
python evaluate_prior.py --config-name config_eval_hsi_prior \
  checkpoint=../results/priors/hsi_prior/checkpoints/best.pth
```

Set `sampling.enabled=true` for full 500-step conditional generation metrics.
This is disabled by default because it is much slower than the fixed-noise
diagnostic.

```bash
python evaluate_prior.py --config-name config_eval_hsi_prior \
  checkpoint=../results/priors/hsi_prior/checkpoints/best.pth \
  sampling.enabled=true sampling.max_batches=32
```

### Reported metrics

Common metrics:

- human MPJPE in meters;
- human 6D rotation L1;
- pelvis/scene goal error in meters;
- sampled and denoised metrics under separate prefixes.

HOI-specific metrics:

- object translation error in meters;
- SO(3)-projected object rotation geodesic error in degrees;
- object goal error;
- contact MAE and F1;
- contact-conditioned hand-object relative displacement error.

HSI-specific metrics:

- scene joint penetration rate;
- scene-condition effect, measured by comparing the same noisy input with the
  scene condition enabled and ablated.

The evaluator writes the complete resolved config, checkpoint epoch, selected
window counts, and all metrics to JSON.  Denoising metrics are diagnostics,
not substitutes for final HOSI task evaluation or a human study.

## Checkpoint schema

New checkpoints are dictionaries with:

- schema and prior type;
- exact state specification;
- model, optimizer, and AMP scaler states;
- epoch and optimizer step;
- resolved Hydra config;
- dataset/split/normalization contract;
- validation metrics and RNG states.

Legacy raw state dictionaries remain loadable by the existing InfBaGel utility
functions.  Independent-prior training uses strict loading and rejects a HOI
checkpoint when an HSI checkpoint is requested, and vice versa.

## Recommended paper ablations

At minimum, preserve the following checkpoints and evaluation JSON files:

1. HOI prior with synthetic OMOMO scene, reproducing the original treatment.
2. HOI prior without any scene, as implemented here.
3. HSI prior with scene occupancy ablated.
4. HSI prior with real scene occupancy.
5. HSI sequence-level split versus the required scene-level split.
6. Composition with denoising-only metrics versus full ancestral sampling.

Use at least three training seeds.  Select checkpoints using only the specified
validation partition and evaluate the OMOMO test/held-out LINGO scenes once per
seed.  Future HOSI composition experiments should additionally test scene
interventions: keep text, object, and initialization fixed while modifying an
obstacle, then measure whether the global route changes while the local
human-object relation remains stable.
