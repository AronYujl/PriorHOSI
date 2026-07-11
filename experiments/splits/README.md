# Dataset splits

Generate the fixed LINGO split only from official local assets:

```bash
python tools/make_lingo_split.py \
  --dataset-root data/dataset \
  --compact \
  --output experiments/splits/lingo_scene_disjoint_seed42.json
```

The generator groups scene variants, uses seed 42 and an 80/20 scene-family
split, records input hashes, and fails if train/validation scenes or families
overlap. The generated manifest should be reviewed and committed unchanged.
