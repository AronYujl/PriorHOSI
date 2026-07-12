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

The public CHOIS release is not self-contained: it imports
`options/train_options.py` and `utils/plot_script.py`, neither of which it ships,
and its evaluation files contain author-local absolute paths. The pinned public
[`text-to-motion`](https://github.com/EricGuo5513/text-to-motion) dependency is
recorded in `experiments/evaluators/text_to_motion.json`. Do not copy those files
into the CHOIS checkout or invent parser defaults.

After placing the official Drive assets under
`third_party/chois_omomo_evaluator_assets/for_t2m_eval`, execute the supplied
CHOIS-versus-GT regression run below. It leaves both third-party checkouts
unchanged and writes a non-overwritable JSON result.

```bash
python tools/run_chois_evaluator.py \
  --chois-root /data/yujinlun/chois_release \
  --text-to-motion-root third_party/text-to-motion \
  --predictions third_party/chois_omomo_evaluator_assets/for_t2m_eval/chois_eval_related/res_npz_files/chois \
  --ground-truth third_party/chois_omomo_evaluator_assets/for_t2m_eval/chois_eval_related/res_npz_files/gt \
  --data-root third_party/chois_omomo_evaluator_assets/for_t2m_eval/chois_eval_related \
  --glove-root third_party/chois_omomo_evaluator_assets/for_t2m_eval/chois_eval_related/glove_840B \
  --checkpoints-dir third_party/chois_omomo_evaluator_assets/for_t2m_eval/checkpoints \
  --checkpoint third_party/chois_omomo_evaluator_assets/for_t2m_eval/checkpoints/omomo/text_motion_features/model/finest.tar \
  --output results/p0-chois-official-regression-s42-20260712.json
```

The official released `chois` directory contains 482 sequences and `gt`
contains 480 (the additional IDs are `sub17_woodchair_055` and
`sub17_woodchair_056`). This is accepted because it reproduces the original
script: it computes prediction text/motion embeddings independently and FID
against GT embeddings. For a new InfBaGel export, use `validate-inputs` first to
enforce the stricter same-ID protocol, then archive the adapter JSON beside the
raw log and register FID/R-Precision@1/2/3 through `tools/experiment.py`.
