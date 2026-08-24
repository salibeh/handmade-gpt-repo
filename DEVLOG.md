# Development Log — GPT From Scratch

A record of building a character-level GPT from the ground up on a MacBook Pro
(M1, 16GB), following the "bigram → self-attention" progression, with each
step measured and verified rather than taken on faith.

Reference article: *"I Built a GPT from Scratch on a MacBook: Days 1-5, from a
Bigram to a Working Self-Attention Head"* by Nikhil.

---

## Environment

- MacBook Pro, Apple M1, 16GB unified memory, macOS 13.2
- Python 3.9.6, venv
- PyTorch with MPS (Metal) backend confirmed available and built
- Dataset: tiny Shakespeare (`input.txt`, 1,115,394 characters)

Setup issue hit and resolved: `pip install torch` did not pull in `numpy`,
causing a `UserWarning` on import. Fixed with `pip install numpy`. PyTorch
depends on numpy for some internal tensor conversions but doesn't declare it
as a hard dependency in all versions.

---

## Day 1 — Data pipeline

- Loaded `input.txt`, confirmed length: 1,115,394 characters.
- Built character-level vocabulary: `vocab_size = 65` (uppercase, lowercase,
  punctuation, space, newline).
- Built `stoi`/`itos` mappings and `encode`/`decode` functions; verified
  round-trip correctness (`decode(encode("hello")) == "hello"`).
- Converted full text to a tensor of integers; split 90/10 into
  `train_data` (1,003,854 chars) / `val_data` (111,540 chars), preserving
  original order (order matters — a language model's whole job is
  predicting what comes next, so shuffling would destroy the signal).
- Built `get_batch(split)`: randomly samples `batch_size` starting positions,
  slices `block_size`-length chunks. `y` is `x` shifted by one character —
  every position in a block is simultaneously training example: predict the
  next character given everything before it, up to `block_size`.
- Confirmed by hand-decoding actual batch contents (e.g. `xb[1]` = "for
  that", `yb[1]` = "or that ") that the shift mechanism works as intended.

## Day 2 — Bigram baseline

- Built `BigramLanguageModel`: a single `nn.Embedding(vocab_size, vocab_size)`
  — each character's row directly *is* its 65 next-character logits. No
  context beyond the current character.
- Untrained loss ≈ 5.04 (single noisy batch); theoretical untrained baseline
  for uniform random guessing over 65 classes is `-ln(1/65) ≈ 4.174`.
- Trained 10,000 steps, AdamW, `lr=1e-3`, `batch_size=32`. Final loss
  converged to **2.4488**, perplexity ≈ **11.57**.
- Computed the **theoretical entropy floor** directly from real
  character-pair statistics in `train_data` (conditional entropy,
  `-Σ P(c2|c1) log P(c2|c1)`, weighted by `P(c1)`): **2.4519**.
- Result: trained loss (2.4488) essentially matches the computed floor
  (2.4519) — the bigram model is optimal for the amount of context it's
  structurally allowed to use (exactly one character). The residual loss is
  irreducible uncertainty (Shannon entropy of the bigram distribution), not
  a training deficiency.

## Day 4 — Context via averaging (negative result, intentionally)

- Built the causal running-average trick (`xbow`): for each position `t`,
  average the vectors of positions `0..t` inclusive. Verified two
  equivalent implementations produce identical output:
  1. Double for-loop, direct slicing and `.mean(0)`.
  2. Matrix multiply: `wei = tril / tril.sum(1, keepdim=True)`,
     `xbow2 = wei @ x`. Confirmed via `torch.allclose(xbow, xbow2)` → `True`.
     The matrix form is the one that generalizes to real attention.
- Wired this into a real model: embedding → uniform-weighted average over up
  to 8 characters of context → linear head (`nn.Linear(n_embd, vocab_size)`)
  → cross-entropy loss.
- Trained 10,000 steps. Final loss **2.8604**, perplexity ≈ **17.47** —
  *worse* than the 1-character bigram (2.4488).
- Conclusion (predicted before running, then confirmed): uniform averaging
  treats every past character as equally relevant, diluting useful recent
  signal with irrelevant distant signal. More context, used dumbly, is not
  automatically better. This result directly motivates learned,
  content-dependent weighting (attention).

## Day 5 — Single-head self-attention

- Added `Key`, `Query`, `Value` as three separate `nn.Linear(n_embd,
  head_size, bias=False)` projections of the same embedding.
- `wei = q @ k.transpose(-2,-1) * head_size**-0.5` (scaled dot-product,
  scaling prevents softmax saturation at larger head sizes).
- Causal mask via `tril`: future positions set to `-inf` before softmax, so
  each position can only attend to itself and earlier positions.
- `out = softmax(wei) @ v` — blends **Values** (not raw embeddings),
  weighted by learned relevance between Query and Key.
- Bridged `out` (16-dim) to 65 logits via `nn.Linear(head_size,
  vocab_size)`, trained end-to-end (embedding + K/Q/V + output head jointly).
- Trained 10,000 steps. Final loss **2.4439**, perplexity ≈ **11.52** — a
  small but real improvement over the bigram (2.4488), and a clear
  improvement over uniform averaging (2.8604), using the identical 8-token
  context window uniform averaging had access to.

## Multi-head attention (extension beyond the source article)

- Refactored single-head logic into a reusable `Head(nn.Module)` class, and
  added `MultiHeadAttention`: runs `n_head=4` independent heads in parallel
  (`head_size = n_embd // n_head = 8` each), concatenates outputs back to
  `n_embd=32`.
- Reorganized the whole model into a proper `nn.Module` (`SimpleGPT`) —
  embedding → multi-head attention → linear head — trained via
  `model.parameters()` rather than manually listing tensors.
- Trained 10,000 steps. Final loss **2.2479**, perplexity ≈ **9.47** — the
  clearest improvement in the whole progression.

### Results summary

| Model                      | Context           | Final loss | Perplexity |
|-----------------------------|--------------------|-----------:|-----------:|
| Bigram                      | 1 character        | 2.4488     | ~11.57     |
| Uniform averaging            | 8 chars, equal wt  | 2.8604     | ~17.47     |
| Single self-attention head   | 8 chars, learned wt| 2.4439     | ~11.52     |
| 4-head multi-head attention  | 8 chars, 4 patterns| 2.2479     | ~9.47      |

Theoretical bigram entropy floor (computed independently from data):
**2.4519** — matches the trained bigram loss almost exactly.

## Generation

- Implemented `generate()`: autoregressive sampling. Crops input to the last
  `block_size` tokens each step (model was only ever trained on
  fixed-length context, so this avoids shape errors as generated text grows
  beyond `block_size`), takes only the last position's logits, samples via
  `torch.multinomial` (not greedy argmax, to avoid repetitive output), and
  appends the sampled token before repeating.
- Assessment of output at this stage: expected to be "English-shaped" but
  not coherent — right local statistics (letter frequency, some short-word
  fragments), no real semantic structure. This is expected at this
  architectural depth (no MLP, no residual connections, no LayerNorm, only
  8 tokens of context, single layer).

## Verification exercise: top-3 prediction check against real validation text

- Ran the trained 4-head model against a real (never-trained-on) 8-character
  validation snippet, printing top-3 predicted next-characters with
  probabilities at every position, alongside the actual next character.
- Findings: several exact hits with high confidence at predictable
  transitions (e.g., punctuation → newline); confident, categorically
  correct misses (predicting a vowel when a vowel was needed, just not the
  specific one; predicting a letter when a letter was needed); and
  appropriately *low* confidence at genuinely ambiguous positions (start of
  an unfamiliar proper noun). This is direct evidence of real, learned
  statistical structure — not memorization, and not random noise.

## Outstanding (not yet built, per source article's later sections)

- Residual connections (`x = x + sublayer(x)`)
- LayerNorm
- Feedforward MLP after attention (extra per-position computation)
- Stacking multiple transformer blocks
- Larger `block_size` / `n_embd` / more training steps at scale

## Key conceptual threads

- **Cross-entropy loss is a direct application of Shannon information
  theory** — it measures the same quantity (entropy / conditional entropy)
  that governs the achievable floor. This was demonstrated empirically,
  not just asserted: the independently-computed entropy floor (2.4519)
  matched the trained bigram model's loss almost exactly.
- **Perplexity (`e^loss`) is entropy re-expressed as "effective number of
  equally-likely choices."** Useful as an interpretable companion metric to
  raw loss, and this is a standard metric quoted in real model papers/cards.
- **More context is not automatically better** — only *useful, correctly
  weighted* context is. This was demonstrated as a negative result
  (uniform averaging underperforming the simpler bigram), not just stated.
- **Embeddings and attention weights carry no information at
  initialization** — meaning is entirely a product of gradient descent
  shaping initially-random parameters against real training data, over many
  steps. This applies identically to the embedding table, Q/K/V matrices,
  and the output head.
