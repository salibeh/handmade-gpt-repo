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

All three validation levels passed on the instructor Apple Silicon Mac with
PyTorch 2.8.0 and MPS selected:

- [clean-static.json](evidence/setup/clean-static.json): static/runtime pass
- [clean-learning.json](evidence/setup/clean-learning.json): six-stage,
  200-step learning execution pass
- [clean-full.json](evidence/setup/clean-full.json): six-stage, 10,000-step
  evidence execution pass

The full evidence run used 200 evaluation batches per split, completed every
stage with return code 0, and took approximately 429.122 seconds. The evidence
artifact was committed as `cd6f82f`. Model metrics and qualified analysis are
recorded in [DEVLOG.md](DEVLOG.md). Multi-seed and accuracy work remains outside
this clean-execution result.
