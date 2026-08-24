# From Bigram Prediction to a Minimal GPT

A character-level language-model learning project inspired by Nikhil Bajpai’s
Medium article, [“I Built a GPT From Scratch on a MacBook — Days 1–5: From a
Bigram to a Working Self-Attention
Head”](https://medium.com/@nikhil.cse16/i-built-a-gpt-from-scratch-on-a-macbook-days-1-5-from-a-bigram-to-a-working-self-attention-head-0d3082ac417c).

The project reconstructs the progression:

```text
bigram → uniform causal context → one attention head → multi-head attention → decoder-only Transformer
```

The article-driven stages stop at attention. This repository now adds a
minimal decoder-only GPT extension with positional embeddings, projected
multi-head causal attention, residual connections, pre-LayerNorm,
feedforward sublayers, and stacked Transformer blocks.

- [HANDMADE_GPT_DESIGN_BASELINE.md](HANDMADE_GPT_DESIGN_BASELINE.md): authoritative scope, sequence, outcomes, artifacts, and completion gates
- [LAB.md](LAB.md): student-facing execution, evidence, and reasoning tasks
- [DEVLOG.md](DEVLOG.md): focused technical development history
- [VALIDATION_REPORT.md](VALIDATION_REPORT.md): clean-check commands and validation boundary
- [SOURCES.md](SOURCES.md): article, dataset, and attribution record

## Implementation status

The final 17-section instructional structure is frozen in the design baseline.
The core bigram-to-minimal-GPT execution path is implemented, including
positional embeddings and complete Transformer blocks. Evidence-mode
automation, multi-seed metrics, diagrams, and the required report remain
pending. The external-foundations bridge is now defined below. Remaining implementation
work is tracked in the design baseline.

## Relationship to AI Enterprise Lab 0

The AI Enterprise course’s
[Lab 0 Foundations Baseline](https://github.com/salibeh/ai-enterprise-course/blob/course-redesign-v2/documentation/LAB0_FOUNDATIONS_BASELINE.md)
is the required, independently completable preparation for its enterprise labs.
It covers pretrained-model inference, retrieval embeddings, vector similarity,
RAG, agents, MCP, and evidence boundaries without requiring model training.

This Handmade GPT practicum is optional deeper study of model internals:
training, loss, token embeddings, attention, positional embeddings, and
Transformer blocks. Completing it does not replace Lab 0 or satisfy the
[Lab 0 → Lab 1 dependency contract](https://github.com/salibeh/ai-enterprise-course/blob/course-redesign-v2/documentation/LAB0_LAB1_DEPENDENCY_CONTRACT.md).

## Quick start

```bash
git clone https://github.com/salibeh/handmade-gpt-repo.git
cd handmade-gpt-repo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

wc -c input.txt
python scripts/bigram.py
python scripts/loss-limit.py
python scripts/uniform-context-model.py
python scripts/context-model.py
python scripts/multi_head_model.py
python scripts/gpt_model.py
```

The dataset check should report 1,115,394 bytes.

Every training script:

- Selects MPS when available and otherwise uses CPU
- Prints periodic training-batch loss for progress only
- Reports averaged training and validation loss over 200 batches
- Reports validation perplexity

Do not compare models using the last randomly sampled training batch.

## Clean-check validation

After a clean clone and dependency installation:

```bash
python scripts/validate_clean.py --output evidence/setup/clean-static.json
python scripts/validate_clean.py --execute --mode learning \
  --output evidence/setup/clean-learning.json
python scripts/validate_clean.py --execute --mode evidence \
  --output evidence/setup/clean-full.json
```

The static check does not prove training completed. Learning mode runs 200
steps per model for rapid end-to-end confirmation. Only evidence mode runs the
full 10,000 steps per model. The validator displays each script and elapsed
time while preserving MPS acceleration on Apple Silicon.

## Architecture boundary

`scripts/multi_head_model.py` is the article-aligned attention endpoint and
is not a complete GPT. `scripts/gpt_model.py` is the repository’s subsequent
minimal GPT extension. It is decoder-only and structurally complete for this
teaching scale, but remains a tiny character model with an eight-token context,
two blocks, and a small corpus; architecture completeness does not imply
frontier-model capability.

## Reproducibility

Dependencies are pinned in [requirements.txt](requirements.txt). Training uses
seed 1337, but exact cross-platform equality is not promised because PyTorch
kernels and MPS/CPU execution may differ. Compare averaged validation results,
record the device and package versions, and use multiple seeds before making a
general performance claim.

## License status

No repository license has yet been selected. See [SOURCES.md](SOURCES.md)
before reusing or redistributing material.
