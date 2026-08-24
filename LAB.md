# Lab — From Bigram Prediction to a Minimal GPT

**Source-driven origin:** This lab was initiated by Nikhil Bajpai’s Medium
article, [“I Built a GPT From Scratch on a MacBook — Days 1–5: From a Bigram
to a Working Self-Attention
Head”](https://medium.com/@nikhil.cse16/i-built-a-gpt-from-scratch-on-a-macbook-days-1-5-from-a-bigram-to-a-working-self-attention-head-0d3082ac417c).
The lab independently executes, measures, and qualifies the article’s
bigram-to-attention progression. See [SOURCES.md](SOURCES.md).

> No evidence, no credit. A script completing is execution evidence; it is not
> proof that one architecture generalizes better than another.

## 0. Frozen structure and current coverage

[HANDMADE_GPT_DESIGN_BASELINE.md](HANDMADE_GPT_DESIGN_BASELINE.md) is
authoritative for this practicum's final scope, sequence, outcomes, artifacts,
and completion gates. The existing procedures below exercise the implemented
core, but they do not yet constitute the complete frozen practicum.

| Frozen section | Current coverage | Status |
|---|---|---|
| 1. Purpose, provenance, and boundary | Lab introduction and Section 1 | Implemented |
| 2. Minimal applied mathematics | Entropy and tensor-shape questions | Partial |
| 3. Text, tokens, and token IDs | Step 2 | Implemented |
| 4. Token embeddings | Steps 2, 3, and 8 | Partial |
| 5. Training versus inference | Model execution tasks | Partial |
| 6. Bigram baseline | Step 3 | Implemented |
| 7. Empirical entropy | Step 3 | Implemented |
| 8. Uniform causal context | Step 4 | Implemented |
| 9. Single-head attention | Step 5 | Partial; hand calculation pending |
| 10. Multi-head attention | Step 6 | Implemented |
| 11. Positional embeddings | Step 8 | Implemented |
| 12. Complete Transformer block | Step 8 | Implemented |
| 13. Stacked minimal GPT | Step 8 | Implemented |
| 14. Logits, softmax, temperature, sampling | Generation only | Partial |
| 15. Evaluation and repeatability | Averaged loss/perplexity | Partial |
| 16. External-foundations boundary | Section immediately below | Implemented |
| 17. Required report | No report template yet | Pending |

Remaining work includes complete applied-math exercises, explicit
training-versus-inference parameter evidence, command-line learning/evidence
modes, greedy and temperature comparisons, top-1/top-3 accuracy, multi-seed
summaries, JSON metrics, architecture diagrams, and
`HANDMADE_GPT_REPORT.md`.

The current `S1-T1`/`S1-Q1` identifiers are transitional. As the missing
sections are implemented, all student work must be normalized to the frozen
`HGPT-S<section>-T<task>` and `HGPT-S<section>-Q<question>` convention.
Questions must remain adjacent to the task that produces their evidence.

### External-foundations bridge

This practicum explains how representations are learned and used inside a
small language model. The AI Enterprise course’s
[Lab 0 Foundations Baseline](https://github.com/salibeh/ai-enterprise-course/blob/course-redesign-v2/documentation/LAB0_FOUNDATIONS_BASELINE.md)
uses pretrained services to teach a different operational boundary:

| This practicum | AI Enterprise Lab 0 |
|---|---|
| Character tokenization and token IDs | Text, tokens, request context, and inference |
| Internal token embeddings | Internal token embeddings versus retrieval embeddings |
| Training, loss, and parameter updates | Training-versus-inference distinction; no training required |
| Attention and positional embeddings | Observable generation behavior |
| Minimal Transformer blocks | Retrieval, RAG, agent, MCP, and service-stack boundaries |
| Model evaluation | Execution evidence versus semantic correctness |

Handmade GPT is optional enrichment. It neither replaces Lab 0 nor satisfies
Lab 0’s vector-retrieval, RAG, MCP, architecture, or evidence gates. Students
preparing for AI Enterprise should complete Lab 0 or its approved
equivalent-competency check. See the
[Lab 0 → Lab 1 dependency contract](https://github.com/salibeh/ai-enterprise-course/blob/course-redesign-v2/documentation/LAB0_LAB1_DEPENDENCY_CONTRACT.md).

This bridge is deliberately conceptual; it adds no vector database, RAG,
agent, or MCP implementation to the Handmade GPT practicum.

## 1. Purpose and architecture boundary

This lab opens the language-model “black box” by constructing increasingly
capable next-character predictors. Students first observe what a one-character
bigram model can learn, calculate its empirical information limit, deliberately
try a weak way of using more context, and then replace fixed context weights
with learned causal attention.

The article-aligned attention stage is not a complete GPT. The final extension
in this lab adds positional embeddings, attention output projection,
feedforward sublayers, residual connections, pre-LayerNorm, final LayerNorm,
and stacked decoder blocks. It is a complete minimal decoder-only GPT
architecture, not a large or production-capable language model.

## 2. Learning objectives

After completing the lab, students should be able to:

1. Explain token IDs, learned embeddings, logits, cross-entropy, and perplexity.
2. Distinguish batch size from context length.
3. Calculate the empirical conditional entropy of a character bigram corpus.
4. Explain why one random training-batch loss is not an evaluation.
5. Compare averaged training and held-out validation loss.
6. Implement causal context aggregation using a lower-triangular mask.
7. Explain the distinct Query, Key, and Value roles.
8. Trace tensor shapes through single-head and multi-head attention.
9. Interpret a negative experimental result without treating it as failure.
10. Distinguish the article-aligned attention endpoint from the complete minimal GPT extension.
11. Trace positional embeddings, residual paths, LayerNorm, attention, and feedforward computation through a Transformer block.

## 3. Required equipment and files

- macOS Apple Silicon or another host with Python 3.9+
- Approximately 2 GB free disk recommended for the environment and artifacts
- Git and Internet access for the initial clone
- Repository files:
  - `input.txt`
  - `requirements.txt`
  - `scripts/data.py`
  - `scripts/common.py`
  - the six executable model/evaluation scripts

The scripts use MPS when `torch.backends.mps.is_available()` is true and CPU
otherwise. MPS availability alone does not prove a script used MPS; each script
prints its selected device.

## Step 1 — Establish a reproducible environment

This step creates an isolated environment, verifies the dataset, and captures
the execution platform before model results are generated.

### S1-T1 — Clone and inspect the repository

```bash
git clone https://github.com/salibeh/handmade-gpt-repo.git
cd handmade-gpt-repo
git status
find scripts -maxdepth 1 -type f -print | sort
```

**S1-Q1:** Which command proves the repository was cloned cleanly, and which
command merely lists files? Explain why these are different evidence claims.

### S1-T2 — Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p evidence/setup
python -m pip freeze | tee evidence/setup/python-packages.txt
```

**S1-Q2:** Why is `.venv/` excluded from Git while
`requirements.txt` is retained?

### S1-T3 — Verify host, device capability, and dataset

```bash
{
  sw_vers 2>/dev/null || true
  uname -a
  python --version
  python -c "import torch; print('torch', torch.__version__); print('mps_available', torch.backends.mps.is_available())"
  wc -c input.txt
} | tee evidence/setup/environment.txt
```

The dataset size must be `1115394` bytes.

**S1-Q3:** Does `mps_available True` prove training ran on MPS? Identify the
additional evidence required.

## Step 2 — Inspect the data pipeline

The dataset is converted from characters to integer token IDs. The IDs are
lookup keys, not semantic values. A learned embedding table later maps each ID
to trainable numbers.

### S2-T1 — Verify path-independent data loading

```bash
python -c "from scripts.data import DATASET_PATH, vocab_size, train_data, val_data; print(DATASET_PATH); print(vocab_size, len(train_data), len(val_data))" \
  | tee evidence/setup/data-check.txt
(cd scripts && python -c "from data import DATASET_PATH; print(DATASET_PATH)")
```

Expected values are vocabulary 65, training length 1,003,854, and validation
length 111,540.

**S2-Q1:** Why did the earlier `open("input.txt")` implementation fail when
the process ran from `scripts/`, and how does `Path(__file__)` fix it?

### S2-T2 — Trace a shifted training example

```bash
python - <<'PY'
from scripts.data import decode, train_data
block = train_data[:8]
target = train_data[1:9]
print("x:", repr(decode(block.tolist())))
print("y:", repr(decode(target.tolist())))
for index in range(len(block)):
    print(index, repr(decode(block[:index+1].tolist())), "->", repr(decode([target[index].item()])))
PY
```

**S2-Q2:** How can one block of eight positions provide eight next-character
prediction targets without becoming eight independent sequences?

## Step 3 — Build and evaluate the bigram baseline

The bigram model uses only the current character to predict the next one.
Periodic losses show optimization progress; the final comparison uses averaged
training and validation loss.

### S3-T1 — Train and capture the bigram result

```bash
mkdir -p evidence/results
python scripts/bigram.py | tee evidence/results/bigram.txt
```

**S3-Q1:** Identify the selected device, averaged training loss, averaged
validation loss, and validation perplexity. Why is the periodic
`training-batch loss` unsuitable for the final comparison?

### S3-T2 — Calculate empirical bigram entropy

```bash
python scripts/loss-limit.py | tee evidence/results/entropy.txt
```

**S3-Q2:** Compare empirical training entropy with averaged training loss.
Why should neither be compared with one random batch? Why can validation
entropy and validation model loss differ?

## Step 4 — Test a deliberately weak use of additional context

Uniform causal averaging gives every available prior position equal weight.
This is a controlled negative design: it tests whether “more context” is
automatically helpful.

### S4-T1 — Train the uniform-context model

```bash
python scripts/uniform-context-model.py | tee evidence/results/uniform-context.txt
```

**S4-Q1:** Did averaged validation loss improve over the bigram run? State only
what this run supports. Do not claim the result will hold for every seed,
dataset, context length, or optimizer.

### S4-T2 — Explain the causal averaging matrix

For a four-position example, write the normalized lower-triangular weight
matrix by hand.

**S4-Q2:** Why does a lower-triangular matrix prevent future-token access, and
why can equal weighting still discard useful distinctions among allowed
tokens?

## Step 5 — Replace fixed weights with one learned attention head

Query and Key projections determine relevance; Value supplies the payload
combined according to those learned relevance weights.

### S5-T1 — Train the single-head model

```bash
python scripts/context-model.py | tee evidence/results/single-head.txt
```

**S5-Q1:** Record the averaged validation result. Does one run establish that
attention always beats a bigram? What additional repetitions would support a
broader claim?

### S5-T2 — Trace attention tensor shapes

Using batch 32, time 8, embedding width 32, and head width 16, determine the
shapes of `x`, `key`, `query`, the attention-score matrix, `value`, and
the head output.

**S5-Q2:** Why are Query and Key insufficient by themselves to provide a
separate retrieved payload?

## Step 6 — Run causal multi-head attention

Four heads learn independent projections and concatenate four eight-value head
outputs. The result returns to embedding width 32.

### S6-T1 — Train and inspect the multi-head model

```bash
python scripts/multi_head_model.py | tee evidence/results/multi-head.txt
```

**S6-Q1:** Compare its averaged validation loss with the other three models.
Does a lower loss prove that each head learned a different interpretable
linguistic function? Explain.

### S6-T2 — Separate examples from aggregate evidence

The script prints generated text and an illustrative top-three prediction
trace after the averaged evaluation.

**S6-Q2:** Why are selected top-three predictions and “English-shaped” output
weaker evidence than an aggregate held-out metric? What can they still reveal
that one aggregate number cannot?

## Step 7 — Form a qualified conclusion

### S7-T1 — Build the evidence table

Create `evidence/results/summary.md`:

| Model | Device | Averaged train loss | Averaged validation loss | Validation perplexity |
|---|---|---:|---:|---:|
| Bigram | | | | |
| Uniform context | | | | |
| Single head | | | | |
| Multi-head | | | | |

**S7-Q1:** Which observed ordering is supported by this seed? Distinguish
“observed in this execution” from “expected to generalize.”

### S7-T2 — State the architecture boundary

**S7-Q2:** List the missing components that prevent
`multi_head_model.py` from being called a complete GPT. Explain what
positional embeddings would add that token embeddings and a causal mask do not
fully provide.

## Step 8 — Complete the minimal decoder-only GPT

The article-driven progression ends with attention. This extension supplies the
remaining decoder-only Transformer mechanisms:

```text
token embedding + position embedding
        ↓
pre-LayerNorm → causal multi-head attention → residual addition
        ↓
pre-LayerNorm → feedforward network → residual addition
        ↓
repeat block → final LayerNorm → vocabulary logits
```

### S8-T1 — Inspect the architecture

```bash
rg -n 'position_embedding|TransformerBlock|LayerNorm|FeedForward|projection|x = x \+' scripts/gpt_model.py
```

**S8-Q1:** What information does the positional embedding supply that token
identity alone does not? Why does the causal mask restrict visibility without
fully representing absolute token position?

### S8-T2 — Train and evaluate the minimal GPT

```bash
python scripts/gpt_model.py | tee evidence/results/gpt-model.txt
```

Record its device, parameter count, configuration, averaged training loss,
averaged validation loss, and validation perplexity.

**S8-Q2:** Trace one block’s two residual paths. Why must the attention output
projection and feedforward output both return width `N_EMBD` before their
residual additions?

### S8-T3 — Qualify the result

**S8-Q3:** Does having all principal decoder-only Transformer components make
this model comparable in capability to a production GPT? Discuss corpus size,
tokenization, context length, embedding width, number of layers, training
budget, and alignment.

## Clean-environment validation

From a fresh clone, install `requirements.txt`, then run:

```bash
python scripts/validate_clean.py --output evidence/setup/clean-static.json
python scripts/validate_clean.py --execute --mode learning \
  --output evidence/setup/clean-learning.json
python scripts/validate_clean.py --execute --mode evidence \
  --output evidence/setup/clean-full.json
```

The first command verifies source, dataset, architecture markers, and PyTorch.
Learning mode executes every stage with 200 training steps and 20 evaluation
batches per split. Evidence mode executes the full 10,000 steps and 200
batches. Both display progress and use CUDA, then MPS, then CPU. See
[VALIDATION_REPORT.md](VALIDATION_REPORT.md). A static pass must not be
reported as a full training pass.

## 4. Submission checklist

Submit:

- `evidence/setup/environment.txt`
- `evidence/setup/python-packages.txt`
- `evidence/setup/data-check.txt`
- Six model-output files plus entropy and summary under `evidence/results/`
- Completed `evidence/results/summary.md`
- Answers S1-Q1 through S8-Q3

Do not submit `.venv/`, model-provider credentials, or unrelated personal
files.

## 5. Historical results notice

Earlier repository versions recorded approximate losses of 2.4488, 2.8604,
2.4439, and 2.2479 from the final sampled training batch. Those values motivated
the project but are not retained as validated comparison results. Rerun the
corrected scripts and use averaged validation measurements before publishing a
new results table.
