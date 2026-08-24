# Development Log — Bigram to Causal Attention

This technical log records the construction and correction of a
character-level language-model learning project on a MacBook Pro with Apple
Silicon. The effort was driven by Nikhil Bajpai’s Medium article,
[“I Built a GPT From Scratch on a MacBook — Days 1–5: From a Bigram to a
Working Self-Attention
Head”](https://medium.com/@nikhil.cse16/i-built-a-gpt-from-scratch-on-a-macbook-days-1-5-from-a-bigram-to-a-working-self-attention-head-0d3082ac417c),
published June 28, 2026.

The article supplies the pedagogical progression. This repository independently
reproduces, measures, extends, and corrects the implementation. Full attribution
is maintained in [SOURCES.md](SOURCES.md).

## 1. Scope decision

The original repository called the endpoint “GPT From Scratch.” Audit found
that the implementation stops at causal multi-head attention and lacks
positional embeddings, feedforward sublayers, residual connections, LayerNorm,
and stacked Transformer blocks. The project is therefore now described as
“bigram to causal attention.” A complete GPT remains future work.

This terminology correction is not cosmetic: students should be able to name
which mechanism each result actually demonstrates.

## 2. Initial environment and dataset

The initial work used a MacBook Pro M1 with 16 GB unified memory, macOS 13.2,
Python 3.9.6, and PyTorch. The Tiny Shakespeare corpus contains 1,115,394
characters and produces a 65-character vocabulary.

A missing-NumPy warning after installing PyTorch demonstrated that successful
package installation does not prove a usable environment. The corrected
repository now pins PyTorch and NumPy in `requirements.txt` and requires an
actual import/device check.

## 3. Data-module extraction

Early scripts assumed variables from an interactive Python session would remain
available in later processes. They do not: each script begins with a new
interpreter state. Importing a training script to reuse data would also execute
its training loop as an import side effect.

Shared corpus preparation was therefore extracted into `scripts/data.py`,
which contains no training. Audit later found a second defect: it opened
`input.txt` relative to the process working directory, while the README told
users to run from `scripts/` and keep the dataset in the repository root.

Correction: `data.py` now resolves the dataset relative to its own file:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "input.txt"
```

The scripts can now run from the repository root or from `scripts/`.

## 4. Bigram baseline and empirical entropy

The bigram model uses a `vocab_size × vocab_size` embedding table. For each
current-character ID, its selected row directly supplies the logits for the
next character.

An independent frequency calculation estimates empirical conditional entropy:

```text
H(next | current) = -Σ P(current) Σ P(next | current) ln P(next | current)
```

The initial project observed empirical training entropy near 2.4519 nats and a
last-batch training loss near 2.4488. This was a useful clue but was described
too strongly as proof that optimization reached the entropy floor.

Audit correction:

- One random batch is not comparable with a whole-corpus statistic.
- Training entropy and held-out validation performance answer different
  questions.
- AdamW regularization and finite optimization affect the fitted model.
- The corrected scripts average 200 training and 200 validation batches.
- `loss-limit.py` reports empirical train and validation entropy separately.

The appropriate claim is that empirical training entropy estimates the
unregularized bigram optimum for the observed training distribution; averaged
measurements are needed to determine how closely training approached it.

## 5. Batch size and context length

`block_size=8` controls the maximum context in each sampled sequence.
`batch_size=32` controls how many sequences contribute to one optimizer step.
A larger batch generally reduces gradient-estimate noise but increases memory
use.

The original log also argued that batch 32 used the M1 GPU more effectively.
Audit found that no tensor or model was moved to MPS, so the scripts actually
ran on CPU. That hardware claim was unsupported by the implementation.

Correction: `scripts/common.py` selects MPS when available and moves batches
to the selected device; every model is also moved to that device and prints it.
Batch-size effects on utilization remain hypotheses unless measured.

## 6. Uniform-context negative result

The uniform model replaces one-character prediction with a normalized
lower-triangular average of all available token embeddings. The causal mask
prevents future-token access, but the weights cannot distinguish relevant from
irrelevant allowed positions.

The original run produced a worse final sampled-batch loss than the bigram.
This remains an instructive negative result, but the corrected interpretation
is limited to the tested configuration and must use averaged validation loss.
It does not prove that uniform aggregation is universally worse.

## 7. Single-head causal attention

The single-head model applies learned Query, Key, and Value projections:

- Query represents what the current position seeks.
- Key represents how each allowed position can be matched.
- Value provides the payload combined after relevance scoring.

Scaled dot-product scores are causally masked, normalized by softmax, and used
to blend Value vectors. A learned output layer maps the head result to
next-character logits.

Randomly initialized parameters contain numeric variation but no learned
corpus-specific organization. Corpus-derived predictive structure emerges only
through optimization. This replaces the imprecise earlier statement that
random parameters contain “no information.”

## 8. Multi-head extension

The project extended beyond the article’s single-head endpoint by creating four
parallel attention heads and concatenating their outputs. The canonical file is
now `scripts/multi_head_model.py`; the inconsistent
`scripts/4-context-model.py` name was retired.

A lower validation loss would establish better predictive performance for the
tested run. It would not, by itself, prove that every head learned a distinct,
human-interpretable linguistic role. Attention inspection or controlled
ablation would be required for that claim.

## 9. Evaluation redesign

The most important audit finding was that every results claim used only the
last randomly sampled training batch. Although `val_data` existed, aggregate
validation loss was never calculated.

`scripts/common.py` now provides one evaluation contract for every model:

1. Switch the model to evaluation mode.
2. Disable gradient construction.
3. Sample 200 batches from training and validation partitions separately.
4. Average their cross-entropy losses.
5. Restore training mode.
6. Report validation perplexity as `exp(validation_loss)`.

Periodic training-batch loss remains useful for progress monitoring but is not
used for model comparison. The old numeric table is historical and must be
replaced only after executing the corrected scripts.

Top-three predictions and generated text remain qualitative diagnostics, not
aggregate performance evidence.

## 10. Reproducibility and repository corrections

The audit implemented:

- Pinned dependencies in `requirements.txt`
- Path-independent dataset loading
- Shared MPS/CPU device selection
- Shared averaged evaluation
- Canonical multi-head filename
- Explicit source and dataset attribution
- Evidence/checkpoint ignore rules
- Removal of the temporary `Emil` access-test file
- Student tasks with immediate evidence and reasoning questions
- Separation of technical history from unrelated personal reflections

A fixed seed improves repeatability, but exact equality across MPS and CPU is
not promised. Broader claims require multiple seeds with mean and variance.

## 11. Git and credential lessons retained

A virtual environment was initially committed before `.gitignore`, causing a
GitHub large-file rejection. The durable lesson is to create ignore rules
before staging and to use explicit paths rather than `git add .`.

Credentials were also exposed during early repository work. They were revoked.
The public technical record retains only the general security lesson: never
place tokens in chat, source files, remote URLs, evidence, or committed shell
transcripts. Use a configured GitHub integration or credential manager.

## 12. Open work

The next architecture milestone is a minimal decoder-only Transformer block:

1. Add positional embeddings.
2. Add an attention output projection.
3. Add residual connections.
4. Add pre-normalization.
5. Add a per-position feedforward network.
6. Stack blocks.
7. Evaluate across multiple seeds.
8. Save checkpoints and machine-readable metrics.
9. Select a repository license after reviewing source terms.

Only after those mechanisms are present should the implementation be labeled a
small GPT architecture.
