# Handmade GPT Step 5 Validation

Run after a fresh clone and dependency installation:

```bash
python scripts/validate_clean.py \
  --output evidence/setup/clean-static.json

python scripts/validate_clean.py --execute --mode learning \
  --output evidence/setup/clean-learning.json

python scripts/validate_clean.py --execute --mode evidence \
  --output evidence/setup/clean-full.json
```

The validator prints each script name, streams the script's ordinary output,
and records elapsed time and return code. It writes the requested JSON only
after the run stops or completes.

## Modes

| Mode | Training steps per model | Evaluation batches per split | Purpose |
|---|---:|---:|---|
| Static | 0 | 0 | Dataset, compilation, architecture markers, PyTorch/device |
| Learning | 200 | 20 | Rapid end-to-end execution check |
| Evidence | 10,000 | 200 | Full practicum evidence |

`--train-steps` and `--eval-iters` may override these values for a
documented experiment. The same model code is used in both modes.

Device selection is CUDA, then Apple MPS, then CPU. A static pass is not a
training pass, and a learning-mode pass is not full evidence-mode validation.

## Current validation status

Static validation passed on the instructor Mac with PyTorch 2.8.0 and MPS
available. The original silent evidence run was interrupted safely and produced
no `clean-full.json`. That run exposed the hidden-output/long-runtime defect
corrected by the current harness. Learning and evidence execution remain to be
recorded.
