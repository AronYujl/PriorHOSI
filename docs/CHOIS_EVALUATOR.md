# CHOIS OMOMO evaluator integration

The official evaluator is pinned in
`experiments/evaluators/chois_omomo.json`. Verify a checkout with:

```bash
python tools/chois_evaluator.py verify-upstream --upstream third_party/chois_release
```

InfBaGel exports one NPZ per sequence when `save_chois_eval_npz: true`. Each
contains `seq_name` and Z-up `global_jpos[T,24,3]`, matching the official loader.
Validate prediction/GT identity and shape before evaluation:

```bash
python tools/chois_evaluator.py validate-inputs \
  --predictions results/chois_npz/predictions \
  --ground-truth results/chois_npz/ground_truth
```

`prepare-run` additionally hashes the processed OMOMO annotations, GloVe files,
feature checkpoint, all inputs, and an explicit `--options-module` obtained from
the official bundle (the pinned Git release omits this imported file).
Do not recreate that module from guessed defaults: doing so would no longer be the
official metric protocol. Store the resulting run specification beside raw logs
and register aggregate FID and R-Precision@1/2/3 through `tools/experiment.py`.
