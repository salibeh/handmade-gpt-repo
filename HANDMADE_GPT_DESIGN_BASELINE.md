# Handmade GPT Model-Internals Design Baseline

## 1. Status and authority

**Artifact:** Handmade GPT — Model Internals Practicum  
**Baseline version:** 1.0  
**Status:** Final instructional structure frozen; implementation partially complete  
**Role:** Optional enrichment, honors work, graduate practicum, or instructor reference  
**Primary inspiration:** Nikhil Bajpai’s article recorded in [SOURCES.md](SOURCES.md)

This document is authoritative for the practicum’s scope, sequence, learning
outcomes, artifacts, and completion criteria. [LAB.md](LAB.md) contains the
student procedures. [DEVLOG.md](DEVLOG.md) records implementation history.

This practicum is not a required prerequisite for the AI Enterprise course.
The required enterprise preparation is defined by the
[Lab 0 Foundations Baseline](https://github.com/salibeh/ai-enterprise-course/blob/course-redesign-v2/documentation/LAB0_FOUNDATIONS_BASELINE.md) and its
[Lab 0 → Lab 1 dependency contract](https://github.com/salibeh/ai-enterprise-course/blob/course-redesign-v2/documentation/LAB0_LAB1_DEPENDENCY_CONTRACT.md).
The short boundary in Section 16 preserves that distinction.

## 2. Purpose

The practicum develops a strong internal model of how a small decoder-only
language model is trained and executed:

```text
text → tokens → token IDs → embeddings → attention → Transformer blocks
     → logits → probabilities → next token → autoregressive generation
```

Students build from a bigram baseline to a structurally complete minimal GPT
and use evidence to distinguish:

- Execution progress from model quality
- Training from inference
- Token identity from learned representation
- More context from useful context
- Qualitative output from aggregate evaluation
- Architectural completeness from production capability

## 3. Scope boundary

### Included

- Applied mathematical foundations
- Character tokenization
- Token embeddings
- Training and inference
- Cross-entropy, entropy, and perplexity
- Uniform causal context as a negative experiment
- Query, Key, Value, and causal attention
- Multi-head attention
- Positional embeddings
- Decoder-only Transformer blocks
- Sampling and temperature
- Averaged train/validation evaluation
- Multiple-seed comparison
- Structured report and evidence

### Excluded

- Retrieval embedding implementation
- Vector databases
- RAG pipelines
- Agents
- MCP
- A2A
- Fine-tuning a production model
- Distributed training
- Production serving and observability
- Claims of production-scale language capability

A short conceptual boundary with external AI-system foundations is reserved for
Step 4 and is not part of this Step 3 implementation.

## 4. Final instructional structure

The practicum must use this sequence.

### Section 1 — Purpose, provenance, and architecture boundary

Explain why the model is built, its origin in the Medium article, which work
extends beyond the article, and why a tiny complete architecture is not a
frontier model.

### Section 2 — Minimal applied mathematics

Students exercise:

- Scalar, vector, matrix, and tensor
- Shape and dimension
- Dot product
- Matrix multiplication
- Probability distribution
- Softmax
- Natural logarithm
- Cross-entropy
- Perplexity
- Gradient

Every item must be attached to a small numerical example or tensor used later.

### Section 3 — Text, characters, tokens, and token IDs

Students build the vocabulary, encode and decode text, verify round-trip
correctness, and explain why token IDs are lookup keys rather than semantic
magnitudes.

Character tokenization must be identified as a transparency choice, not the
usual tokenization strategy of production GPT systems.

### Section 4 — Token embeddings

Students inspect embedding-table dimensions, retrieve rows, and compare a
selected embedding before and after training.

The lab must say precisely that random initialization contains numeric
variation but no learned corpus-specific organization.

### Section 5 — Training versus inference

Students trace:

```text
training:
input + target → forward pass → loss → backward pass → optimizer update

inference:
input + fixed parameters → logits → probabilities → selected token
```

At least one task must prove that an optimizer step changes a parameter.

### Section 6 — Bigram baseline

Students train and evaluate the one-character model, record parameter count,
and report averaged training loss, averaged validation loss, and validation
perplexity.

### Section 7 — Empirical entropy

Students calculate training and validation conditional bigram entropy and
explain why a whole-corpus statistic must not be equated with one random batch.

### Section 8 — Uniform causal context

Students predict, execute, and interpret the deliberately weak equal-weight
context design. The conclusion must be limited to the tested configuration.

### Section 9 — Single-head causal attention

Students trace Query, Key, Value, scores, mask, softmax weights, and output
shapes. They must calculate at least one tiny attention example by hand.

### Section 10 — Multi-head attention

Students trace head width, parallel projections, concatenation, and output
projection. A lower loss must not be presented as proof that heads learned
distinct interpretable linguistic roles.

### Section 11 — Positional embeddings

Students compare the same token at different positions and distinguish:

- Causal mask: which positions are visible
- Positional embedding: where a token occurs

### Section 12 — Complete Transformer block

Students trace pre-LayerNorm attention and feedforward residual paths and
explain why each residual branch returns the same embedding width.

### Section 13 — Stacked minimal GPT

Students inspect the complete teaching architecture:

- Token embeddings
- Positional embeddings
- Projected causal multi-head attention
- Feedforward network
- Residual connections
- Pre-LayerNorm
- Stacked blocks
- Final LayerNorm
- Vocabulary head
- Context-cropped generation

### Section 14 — Logits, softmax, temperature, and sampling

Students compare greedy selection and sampling at multiple temperatures.
Output preference alone is not sufficient evidence; settings and repeated
results must be recorded.

### Section 15 — Evaluation and repeatability

Students report:

- Averaged training loss
- Averaged validation loss
- Validation perplexity
- Top-1 accuracy
- Top-3 accuracy
- At least three seeds
- Mean and standard deviation
- Device and dependency metadata

Machine-readable JSON is the canonical metrics artifact.

### Section 16 — External-foundations boundary

This practicum provides optional depth in learned token representations,
training, attention, positional embeddings, and Transformer blocks. AI
Enterprise Lab 0 provides required breadth in pretrained inference, retrieval
embeddings, vector comparison, RAG, agents, MCP, and evidence discipline.

Completion of Handmade GPT does not replace Lab 0 or waive its competency
gates. The practicum links to the
[Lab 0 Foundations Baseline](https://github.com/salibeh/ai-enterprise-course/blob/course-redesign-v2/documentation/LAB0_FOUNDATIONS_BASELINE.md) and
[dependency contract](https://github.com/salibeh/ai-enterprise-course/blob/course-redesign-v2/documentation/LAB0_LAB1_DEPENDENCY_CONTRACT.md) but does not add RAG, vector-store,
agent, or MCP implementation.

### Section 17 — Required report

Students submit `HANDMADE_GPT_REPORT.md` containing:

1. Environment
2. Applied mathematics
3. Tokenization
4. Token embeddings
5. Training behavior
6. Bigram and entropy comparison
7. Uniform-context negative result
8. Single-head attention
9. Multi-head attention
10. Positional embeddings
11. Transformer block
12. Minimal GPT
13. Sampling experiment
14. Multi-seed evaluation
15. Limitations
16. External-foundations boundary

## 5. Required learning outcomes

A completing student must be able to:

1. Trace text into tokens, token IDs, and target IDs.
2. Calculate the small vector/matrix operations used by attention.
3. Explain how token embeddings acquire corpus-specific organization.
4. Distinguish training from inference using observed parameter evidence.
5. Interpret cross-entropy and perplexity without treating them as correctness.
6. Compare a trained bigram with empirical conditional entropy responsibly.
7. Explain why additional context can harm when aggregated poorly.
8. Trace Query, Key, Value, causal masking, and attention output.
9. Trace multi-head concatenation and output projection.
10. Distinguish position representation from causal visibility.
11. Trace residual, LayerNorm, attention, and feedforward paths.
12. Explain autoregressive generation, softmax, sampling, and temperature.
13. Compare models with held-out aggregate metrics and multiple seeds.
14. State the tested model’s architectural and capability limitations.

## 6. Stable task numbering

Student tasks and questions must use:

```text
HGPT-S<section>-T<task>
HGPT-S<section>-Q<question>
```

Examples:

```text
HGPT-S4-T2
HGPT-S12-Q1
```

Questions must appear immediately after the task that produces their evidence.

## 7. Execution modes

### Learning mode

Purpose: rapid inspection and debugging.

Target:

- Reduced training steps
- One seed
- Short generation
- Full tensor-shape and mechanism checks
- Expected completion within one instructional session

### Evidence mode

Purpose: assess architecture and repeatability.

Target:

- Full configured training steps
- At least three seeds
- Averaged validation metrics
- Accuracy metrics
- JSON output
- Report-ready evidence

The same code path should support both modes through command-line arguments;
separate, drifting implementations are prohibited.

## 8. Canonical artifacts

Planned canonical outputs:

```text
evidence/
  setup/environment.json
  setup/packages.txt
  results/<model>-seed-<n>.json
  results/summary.json
  results/summary.md
  sampling/temperature-<value>-seed-<n>.txt
HANDMADE_GPT_REPORT.md
```

Secrets and complete virtual environments must not be committed.

## 9. Current implementation status

| Capability | Status |
|---|---|
| Source attribution | Implemented |
| Path-independent dataset loading | Implemented |
| Bigram model | Implemented |
| Training/validation entropy | Implemented |
| Uniform-context model | Implemented |
| Single-head attention | Implemented |
| Multi-head attention | Implemented |
| Positional embeddings | Implemented in `gpt_model.py` |
| Complete minimal Transformer blocks | Implemented in `gpt_model.py` |
| MPS/CPU device selection | Implemented |
| CUDA device selection | Pending |
| Command-line learning/evidence modes | Pending |
| Greedy and temperature experiments | Pending |
| Top-1/top-3 aggregate accuracy | Pending |
| Multi-seed runner | Pending |
| JSON metrics | Pending |
| Architecture diagrams | Pending |
| Report template | Pending |
| Clean-clone execution validation | Pending |
| Repository license decision | Pending owner decision |
| External-foundations bridge | Implemented in Step 4 |

## 10. Completion gates

The practicum is complete only after:

1. Every structural section is present in `LAB.md`.
2. Every required task has a stable identifier.
3. Learning mode runs from a clean clone.
4. Evidence mode runs from a clean clone.
5. CPU and one accelerated backend are tested.
6. JSON metrics validate against one documented schema.
7. Three-seed summaries are generated.
8. Report instructions resolve to real evidence.
9. Diagrams render and match the code.
10. Source attribution is complete.
11. The owner selects or explicitly declines a repository license.
12. Technical claims are reconciled with observed results.

## 11. Freeze rule

Changing the section sequence, learning outcomes, artifact contract, or
completion gates requires:

- A baseline-version increment
- Development-log rationale
- Student-effort impact assessment
- Evidence-schema impact assessment

Implementation details may change without reopening the baseline if they
preserve the frozen structure and outcomes.

## 12. Step 3 completion record

Step 3 of the six-step integration plan is complete when:

- This baseline exists on `main`
- README identifies it as authoritative
- LAB maps its current content to the frozen structure
- DEVLOG records the structure decision
- All references are independently re-fetched

Step 4 is complete: the short external-foundations bridge and reciprocal
cross-repository links are present. The practicum remains optional and does not
replace Lab 0 competency gates.
