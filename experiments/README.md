# Experiment records

`registry.jsonl` is append-only and tracked. Each line is one JSON object with a
unique `experiment_id`. Raw manifests and per-sample outputs live under
`results/experiments/` and remain ignored.

Typical lifecycle:

```bash
python tools/experiment.py start \
  --id p0-atomic-baseline-s42-20260711 --phase p0 --seed 42 \
  --config code/config/config_sample_infbagel.yaml \
  --asset checkpoint=checkpoint/checkpoint.pth \
  --asset dataset=data/hosi_test

# Run the experiment, writing aggregate metrics.json.

python tools/experiment.py finish \
  --manifest results/experiments/p0-atomic-baseline-s42-20260711/manifest.json \
  --metrics results/experiments/p0-atomic-baseline-s42-20260711/metrics.json \
  --status completed

python tools/experiment.py register \
  --manifest results/experiments/p0-atomic-baseline-s42-20260711/manifest.json \
  --hypothesis "The released baseline is reproducible within tolerance." \
  --conclusion "..." --next-action "..."
```

Use `python tools/experiment.py validate` before commits and merges.
