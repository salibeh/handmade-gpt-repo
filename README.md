# From Bigram Prediction to Causal Attention

A character-level language-model learning project inspired by Nikhil Bajpai’s
Medium article, [“I Built a GPT From Scratch on a MacBook — Days 1–5: From a
Bigram to a Working Self-Attention
Head”](https://medium.com/@nikhil.cse16/i-built-a-gpt-from-scratch-on-a-macbook-days-1-5-from-a-bigram-to-a-working-self-attention-head-0d3082ac417c).

The project reconstructs the progression:

```text
bigram → uniform causal context → one attention head → multi-head attention
```

It deliberately stops short of calling the final model a complete GPT.
Positional embeddings, feedforward layers, residual connections, LayerNorm,
and stacked Transformer blocks remain future work.

- [LAB.md](LAB.md): student-facing execution, evidence, and reasoning tasks
- [DEVLOG.md](DEVLOG.md): focused technical development history
- [SOURCES.md](SOURCES.md): article, dataset, and attribution record

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
```

The dataset check should report 1,115,394 bytes.

Every training script:

- Selects MPS when available and otherwise uses CPU
- Prints periodic training-batch loss for progress only
- Reports averaged training and validation loss over 200 batches
- Reports validation perplexity

Do not compare models using the last randomly sampled training batch.

## Current model boundary

The final implementation contains causal multi-head attention but no positional
encoding or complete Transformer block. It is therefore an attention-based
character language model, not yet a full GPT implementation.

## Reproducibility

Dependencies are pinned in [requirements.txt](requirements.txt). Training uses
seed 1337, but exact cross-platform equality is not promised because PyTorch
kernels and MPS/CPU execution may differ. Compare averaged validation results,
record the device and package versions, and use multiple seeds before making a
general performance claim.

## License status

No repository license has yet been selected. See [SOURCES.md](SOURCES.md)
before reusing or redistributing material.
