# Handmade GPT Step 5 Validation

Run after a fresh clone and dependency installation:

```bash
python scripts/validate_clean.py --output evidence/setup/clean-static.json
python scripts/validate_clean.py --execute \
  --output evidence/setup/clean-full.json
```

The first command verifies the dataset, compiles every script, checks required
architecture markers, and confirms PyTorch availability. The second also runs
the entropy program and every training stage. A static pass is not a training
pass. Commit compact JSON evidence, not complete virtual environments or model
checkpoints.

Step 5 development inspection confirmed that the repository contains the
frozen bigram-to-minimal-GPT path and a clean validation harness. Full clean
execution remains pending on a host with the pinned PyTorch environment; the
current automation worker does not have PyTorch installed.

