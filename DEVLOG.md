# Development Log — GPT From Scratch

A comprehensive record of building a character-level GPT from the ground up
on a MacBook Pro (M1, 16GB) — not just the modeling results, but the real
debugging, engineering decisions, and conceptual detours that were part of
actually doing this.

Reference article: *"I Built a GPT from Scratch on a MacBook: Days 1-5, from
a Bigram to a Working Self-Attention Head"* by Nikhil. This log covers
everything actually done in the process — including where it diverged from,
extended past, or hit friction not covered by the source article.

---

## Environment setup

- MacBook Pro, Apple M1, 16GB unified memory, macOS 13.2
- Python 3.9.6, venv
- PyTorch with MPS (Metal) backend confirmed available and built:
  `torch.backends.mps.is_available()` → `True`, `torch.backends.mps.is_built()`
  → `True`. This means PyTorch can place tensors on the M1's GPU cores
  (`.to('mps')`) instead of only CPU — relevant for speed once models scale
  up, though these scripts are small enough to run fine on CPU too.

### Debugging encountered: missing numpy dependency

`import torch` produced:
```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```
`pip install torch` did not pull in `numpy` as a hard dependency in this
environment, even though PyTorch uses it internally for some tensor
conversions. Fixed with `pip install numpy`. Lesson: a clean `pip install
torch` is not guaranteed to be a complete working environment — verify by
actually importing and exercising the library, not just installing it.

---

## Day 1 — Data pipeline

- Downloaded tiny Shakespeare (`input.txt`), confirmed length matches the
  source article exactly: 1,115,394 characters.
- Built character-level vocabulary via `sorted(set(text))`: `vocab_size = 65`.
  Reasoned through *why* character-level (not word-level) is the easier
  starting point before being told: vocabulary size directly determines
  the size of the first embedding table (`vocab_size × vocab_size` for the
  bigram model) — ~65×65 (4,225 numbers) vs. 10,000+×10,000+ for word-level,
  a difference of orders of magnitude, unrelated to model "quality," purely
  about keeping the mechanism small enough to reason about by hand.
- Built `stoi`/`itos` and `encode`/`decode`; verified round-trip correctness
  by hand (`encode("hello") == [46, 43, 50, 50, 53]`, decodes back to
  "hello").
- Converted full text to a tensor of integers; split 90/10 into
  `train_data` / `val_data`, **preserving original order** — explicitly
  reasoned through why: shuffling characters before splitting would destroy
  the sequential structure a language model depends on (it predicts *next*
  given *previous*; frequency alone, decoupled from order, isn't the target).
- Built `get_batch(split)`: randomly samples `batch_size` starting positions,
  slices `block_size`-length chunks. `y` is `x` shifted by one character.
  Traced this by hand on real decoded batch content (e.g. `xb[1]` = "for
  that", `yb[1]` = "or that ") to directly verify the shift mechanism,
  rather than trusting the code from the shape alone.

### Debugging encountered: variables not persisting across scripts

Hit repeated `NameError: name 'vocab_size' is not defined` and
`NameError: name 'xb' is not defined` when moving from interactive `python3`
sessions to standalone `.py` scripts. Root cause understood explicitly:
each `.py` file execution is its own blank slate — variables from a prior
interactive session, or from a *different* script file, do not carry over.
Also hit a case where importing a training script (`bigram.py`) to reuse
its `train_data` would have re-run its entire 10,000-step training loop as
a side effect of `import` executing the whole file top-to-bottom — this
was caught and avoided *before* being run, by reasoning through what
`import` actually does.

**Fix, and a real engineering decision made as a result:** extracted the
shared setup (reading text, building vocab, encoding, splitting data) into
a standalone `data.py` module. Every other script imports from it
(`from data import vocab_size, train_data, val_data, ...`), so the
expensive/duplicated setup logic runs once, cleanly, without re-triggering
unrelated training loops.

## Batching parameters — batch_size and block_size reasoning

- `block_size = 8`: how many characters of context a single training
  example spans.
- `batch_size`: initially set to `4` for *inspection* purposes (small
  enough to manually decode and trace by hand, e.g. tracing `xb[0]`,
  `xb[1]` back to real words). Later increased to `32` for actual
  *training*.
- **Explicit reasoning for why 32, not 4, for training:** a small batch
  (4 sequences × 8 positions = 32 individual predictions) gives a noisy,
  less representative gradient estimate at each step. A larger batch (32
  sequences × 8 = 256 predictions) gives a smoother, more reliable gradient
  direction per step. There's also a hardware angle: the M1 GPU is built
  for parallel work, and a batch of 4 leaves most of that parallel capacity
  idle — same "wasted parallelism" issue the source article raises when
  explaining why RNNs (inherently sequential) lose to attention (fully
  parallelizable) on GPU hardware.
- **Why not 64 or higher:** reasoned through the tradeoffs rather than
  picking a number arbitrarily — bigger batches use more memory (a real
  constraint on 16GB unified memory once models get deeper), and give
  diminishing returns on gradient smoothness past a certain point (32 → 64
  helps much less than 4 → 32 did). 32 was kept for consistency with the
  source article, explicitly noted as a tunable hyperparameter, not a fixed
  rule.
- Also explicitly worked through a Python-scoping gotcha: `get_batch()`
  reads `batch_size` from the enclosing scope *at call time*, not at
  function-definition time — so reassigning `batch_size = 32` after
  `get_batch` is defined, but before it's called in the training loop,
  correctly changes what every subsequent call uses. This was verified by
  reasoning it through explicitly, not just observed as a side effect.

## Day 2 — Bigram baseline

- Built `BigramLanguageModel`: single `nn.Embedding(vocab_size, vocab_size)`
  — each character's row directly *is* its 65 next-character logits.
- **Manually traced what `nn.Embedding` returns**, distinguishing the
  integer `xb` value (a fixed, meaningless lookup key/ID — analogous to a
  student ID number) from the embedding vector it retrieves (the actual
  learned content — analogous to a school directory profile). Explicitly
  worked through where that "content" originates: pure random
  initialization at first, then reshaped by `loss.backward()` +
  `optimizer.step()` over many training steps, based purely on what
  reduces prediction error across the real training data — no information
  is hand-coded in.
- Untrained loss ≈ 5.04 (single noisy batch); theoretical untrained
  baseline for uniform random guessing over 65 classes is
  `-ln(1/65) ≈ 4.174`. Discussed *why* a single-batch loss reading (5.04)
  can land above the theoretical untrained baseline — small-sample noise,
  same idea as why averaging over many batches (later, `estimate_loss`-
  style evaluation) gives a more trustworthy reading than any one batch.
- Trained 10,000 steps, AdamW, `lr=1e-3`, `batch_size=32`. Final loss
  converged to **2.4488**, perplexity ≈ **11.57**.

### Independent verification: computing the entropy floor from real data

- Computed the **theoretical entropy floor** directly from real
  character-pair statistics in `train_data` — a from-scratch conditional
  entropy calculation (`-Σ P(c2|c1) log P(c2|c1)`, weighted by `P(c1)`),
  completely independent of the trained model, using only counted
  frequencies.
- Hit a real bug running this standalone (`loss-limit.py`): same
  "variables don't persist across scripts" issue as above, fixed the same
  way, via the shared `data.py` import.
- Result: **2.4519** — matches the trained bigram loss (2.4488) almost
  exactly.
- **This was treated as a genuine empirical checkpoint, not a formula to
  accept on faith** — the explicit conclusion drawn: the bigram model's
  loss floor is not a training deficiency (more steps/tuning would not
  meaningfully improve it), but a real information-theoretic limit — one
  character of context genuinely cannot resolve more uncertainty than
  this, no matter how good the model gets.

## Day 4 — Context via uniform averaging (deliberate negative result)

- Built the causal running-average trick (`xbow`) two ways and proved them
  numerically identical:
  1. **Double for-loop**, direct slicing (`x[b, :t+1]`) and `.mean(0)`.
     Traced by hand against real printed tensor values before running
     code, confirming e.g. `xbow[0][1] = [-0.0894, -0.4926]` matches
     manual arithmetic on `x[0]`'s first two rows.
  2. **Matrix multiply form**: `tril = torch.tril(torch.ones(T,T))`,
     `wei = tril / tril.sum(1, keepdim=True)`, `xbow2 = wei @ x`.
     Confirmed via `torch.allclose(xbow, xbow2, atol=1e-6)` → `True`
     — with explicit discussion of what `atol` means (floating-point
     arithmetic doesn't guarantee bit-identical results across different
     computational paths to the same mathematical answer; `1e-6` tolerance
     absorbs harmless rounding noise, not a real discrepancy).
- **Explicit conceptual grounding built before wiring this into a real
  model**: worked through, via a "students in a row holding number-cards"
  analogy, precisely what each of `position`, `vector`, `embedding`,
  `sequence`, `batch`, and `channels (C)` represents, and how `xbow`
  differs from `x` (isolated per-position data vs. context-blended data).
  Also explicitly worked through, and self-corrected via Socratic
  back-and-forth, the actual relationship between `get_batch` (selection
  of raw integer chunks — no math) and the embedding table (converts
  those integers into vectors — a separate, later step) — this distinction
  was initially conflated and had to be untangled across several turns.
- Wired the uniform-average mechanism into a real, trainable model:
  embedding → uniform-weighted average over up to 8 characters of context
  → `nn.Linear(n_embd, vocab_size)` bridging the blended context vector to
  65 logits → cross-entropy loss. Explicit discussion of *why* this bridge
  layer is needed at all (blended vector is `n_embd`-dimensional, e.g. 32;
  cross-entropy needs exactly `vocab_size`, 65, scores — a learned,
  differentiable projection is required to connect the two, as opposed to
  a naive idea like "just repeat the 16/32 numbers to fill 65 slots,"
  which was explicitly proposed, tested against reasoning, and rejected:
  repetition has no adjustable parameters and no way to represent that
  different specific characters have independently different likelihoods).
- Trained 10,000 steps. Final loss **2.8604**, perplexity ≈ **17.47** —
  *worse* than the 1-character bigram (2.4488). **This result was
  predicted before running the code**, based on reasoning that uniform
  averaging can't distinguish relevant recent context from irrelevant
  distant context, and diluting the former with the latter should hurt
  more than help. The prediction was confirmed, not just observed after
  the fact.

## Day 5 — Single-head self-attention

- Added `Key`, `Query`, `Value` as three separate `nn.Linear(n_embd,
  head_size, bias=False)` projections of the same embedding, grounded in
  the search-engine analogy from the source article (Query = "what am I
  looking for," Key = "what do I advertise about myself," Value = "what do
  I actually hand over").
- Explicitly reasoned through **why three roles specifically, not two or
  four**: two (Query/Key only) would let you compute relevance scores but
  leaves nothing separate to retrieve as payload; conflating Key and Value
  into one thing (or reusing the raw embedding for everything) forces
  "what makes me findable" and "what I contribute" to be identical,
  and forces relevance to be symmetric (`x[t]·x[s] = x[s]·x[t]`), which is
  unrealistic (a word caring about another word isn't necessarily
  reciprocal). "More heads" (the eventual multi-head extension) means more
  parallel *copies* of this same three-role unit, not a fourth role within
  one unit.
- Explicitly reasoned through **where the actual "information gain" comes
  from**: not from the Q/K/V linear transformation itself at any single
  forward pass (which, on randomly-initialized weights, adds no new
  information — a repeat/copy operation and a random linear projection are
  equally "meaningless" at initialization). The real gain comes from
  training: gradient descent shapes the three projections, over many
  steps on real data, to extract and encode statistical regularities from
  the *entire training corpus* — compressed into the weight matrices —
  which is not present in any single input's raw embedding.
- `wei = q @ k.transpose(-2,-1) * head_size**-0.5` (scaled dot-product;
  scaling explicitly connected to preventing softmax saturation — if
  raw scores get too large/spread, softmax pushes nearly all probability
  onto one option, killing the gradient signal for adjusting the rest).
- Causal mask via `tril`: future positions set to `-inf` before softmax.
- `out = softmax(wei) @ v` — blends **Values**, weighted by learned Q·K
  relevance.
- Bridged `out` (16-dim) to 65 logits via `nn.Linear(head_size,
  vocab_size)`, trained end-to-end (embedding + K/Q/V + output head jointly).
- Trained 10,000 steps. Final loss **2.4439**, perplexity ≈ **11.52** — a
  small but real improvement over the bigram (2.4488), and a clear
  improvement over uniform averaging (2.8604), using the identical 8-token
  context window uniform averaging had access to. **Explicitly connected
  this result back to entropy/information theory**: learned,
  content-dependent weighting lets the model pull in only the genuinely
  predictive context for a given situation (e.g., weighting "c" and "i"
  heavily when predicting the next letter of "city," and down-weighting
  irrelevant distant characters) rather than being forced into fixed,
  input-independent proportions — which is precisely what a lower loss
  (lower cross-entropy, i.e., less remaining uncertainty) means.

## Multi-head attention (extension beyond the source article)

- Refactored single-head logic into a reusable `Head(nn.Module)` class,
  added `MultiHeadAttention` running `n_head=4` heads in parallel
  (`head_size = n_embd // n_head = 8` each), concatenating outputs back to
  `n_embd=32`. Predicted the concatenated output shape (`(B, T, 32)`)
  correctly before running, reasoning from "4 heads × 8 numbers each,
  concatenated along the last dimension."
- Reorganized the whole model into a proper `nn.Module` (`SimpleGPT`),
  trained via `model.parameters()` rather than manually listing tensors —
  an explicit code-quality improvement, not just a numerical experiment.
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

- Implemented `generate()`: autoregressive sampling. Crops input to the
  last `block_size` tokens each step, takes only the last position's
  logits, samples via `torch.multinomial` (explicitly chosen over greedy
  argmax, reasoned through: greedy argmax tends to produce repetitive,
  boring text; sampling according to the probability distribution allows
  variety while still favoring higher-probability characters).
- Explicit assessment of output at this stage, stated plainly rather than
  overclaimed: expected to be "English-shaped" but not coherent —
  right local statistics, no real semantic structure — and explicitly
  reasoned through *why* (no MLP, no residual connections, no LayerNorm,
  only 8 tokens of context, single attention layer applied once).
- Extended discussion, explicitly distinguishing **real, learned
  statistical patterning** (what this model does) from **intelligence**
  (a loaded term) — landed on the framing that this is genuine,
  measurable pattern-extraction (evidenced by perplexity dropping
  step-by-step across the four models), just narrow in scope (short
  context, single layer, no extra per-position computation, character-level
  not concept-level) — not a categorically different phenomenon from what
  larger, more capable models do, just far more scaled up.

## Verification exercise: top-3 prediction check against real validation text

- Ran the trained 4-head model against a real (never-trained-on) 8-
  character validation snippet, printing top-3 predicted next-characters
  with probabilities at every position, alongside the actual next
  character.
- **Findings, read in detail, not just summarized**: several exact hits
  with high confidence at predictable transitions (e.g., punctuation →
  newline, ~76% and ~30% confidence); confident, categorically correct
  misses (predicting a vowel when a vowel was needed — top-3 were E/A/I,
  all vowels, at a position where the true next-letter was also a vowel,
  just the wrong specific one — same pattern for consonant-following-
  consonant contexts); and appropriately *low* confidence at genuinely
  ambiguous positions (start of an unfamiliar proper noun, "GREMIO," which
  the model had not seen often enough to spell confidently, correctly
  reflected in a flatter, lower-confidence probability distribution rather
  than false certainty).
- This was explicitly chosen as a **more convincing and more checkable**
  demonstration of real learned structure than either (a) reading raw
  generated text, or (b) trusting the aggregate loss number alone —
  because it lets you verify, position by position, whether the model's
  confidence tracks the real predictability of each specific situation.

## Outstanding (not yet built, per source article's later sections)

- Residual connections (`x = x + sublayer(x)`)
- LayerNorm
- Feedforward MLP after attention (extra per-position computation)
- Stacking multiple transformer blocks
- Larger `block_size` / `n_embd` / more training steps at scale

## Conceptual threads developed beyond direct code, through extended discussion

These were substantive learning threads pursued through discussion, not
just code — included here because they materially shaped understanding of
*why* the code behaves the way it does, and were treated with the same
rigor (defining terms precisely, correcting imprecise framings, grounding
in concrete examples) as the coding work itself.

- **Cross-entropy loss as a direct application of Shannon information
  theory.** Not asserted — demonstrated empirically: the independently-
  computed conditional-entropy floor (2.4519) matched the trained bigram
  model's loss (2.4488) almost exactly. Explicitly distinguished
  *randomness* (a property of the underlying process), *uncertainty* (an
  observer's state of not-knowing, arising from randomness), *probability
  distribution* (the mathematical object quantifying both), and *entropy*
  (a single summary number computed from that distribution) — as a
  precise chain, not interchangeable synonyms.
- **Perplexity (`e^loss`)** developed as an interpretable companion metric
  — "effective number of equally-likely choices" — computed and
  interpreted at each stage of the model progression (untrained ≈ 67-70;
  bigram ≈ 11.57; uniform averaging ≈ 17.47; single-head ≈ 11.52;
  multi-head ≈ 9.47).
- **Explicit correction of an overreach**: initially proposed that
  "patterning has zero entropy" — corrected through worked examples
  (untrained uniform baseline 4.174 nats → bigram 2.4519 nats floor →
  near-zero only in the limiting case of a fully deterministic
  relationship, e.g., "q" almost always followed by "u") to the precise
  claim: pattern *reduces* entropy relative to a less-informed baseline;
  it does not, in general, eliminate it. Only a perfectly deterministic
  mapping reaches exactly zero.
- **Rate of return / CAGR / IRR** — a brief, separate personal-finance
  detour (simple return, annualized/CAGR, real return, and where IRR
  becomes necessary for irregular cash flows), kept clearly labeled as
  mechanics rather than financial advice.
- **Connection to *The Drunkard's Walk* (Mlodinow), chapter 1** — explicit
  reasoning through the relationship between the book's informal theme
  (humans misjudge randomness, see false patterns) and Shannon entropy as
  the rigorous, computable version of the same underlying question
  ("is there really a pattern here, and how much genuine uncertainty
  remains if not?"). Extended to a discussion of where this bias shows up
  not just in daily-life judgment (gambler's fallacy, streak illusions)
  but in professional/scientific contexts (regression to the mean
  misread as causation, small-sample noise mistaken for real effects,
  and why rigorous fields build in statistical safeguards — significance
  testing, replication, large-sample averaging — as an explicit corrective
  for the same cognitive bias, directly paralleling why this project
  moved from trusting single-batch loss readings to averaging over many
  batches and to computing an independent theoretical floor).
- **Career-direction reflection** ("From Strength to Strength" framing) —
  a separate, personal discussion connecting observed patterns in this
  session (independently verifying claims, pushing until reasoning felt
  earned) to a tension between a stated motivation ("stay in sync with
  how the new generation thinks") and a stated regret-test answer
  ("math/probability") — flagged as not automatically pointing to the
  same career option, with data/teaching/travel weighed against each,
  kept separate from the technical thread but recorded here since it
  occurred in the same working session.

## GitHub / repository management — real project history, including mistakes

This section is included deliberately, because the process of getting this
project onto GitHub involved real mistakes and lessons, not just a clean
`git push`.

- **First mistake: committing `venv/` before creating `.gitignore`.**
  A `git push` attempt failed with GitHub rejecting a 203MB file
  (`libtorch_cpu.dylib`) — exceeding GitHub's 100MB single-file limit.
  Root cause: `git add .` was run before a `.gitignore` excluding `venv/`
  existed, so the entire virtual environment (including large compiled
  PyTorch libraries) got committed into git history. Because the oversized
  file was already baked into a past commit, simply adding `.gitignore`
  afterward did not fix it — the fix required either purging history
  (`git filter-repo`, overkill for a personal repo) or, since the push had
  already been rejected and nothing was live on GitHub yet, wiping local
  git history entirely (`rm -rf .git`) and reinitializing cleanly with the
  correct `.gitignore` in place *before* the first `git add`.
- **Second mistake, more serious: a live GitHub Personal Access Token was
  pasted directly into chat.** Treated immediately and explicitly as a
  security incident — a PAT is equivalent to a password, and pasting it
  into a chat transcript exposes it in stored conversation history
  regardless of whether it's ever "used." Explicit lesson recorded and
  repeated multiple times through the rest of the project: PATs should be
  revoked immediately upon exposure, never reused after being pasted in
  chat, and only ever entered directly into a local terminal's
  credential prompt (or, if used with an AI agent's sandboxed environment
  for a one-off task, treated as single-use and revoked immediately after,
  since it still passes through the conversation transcript either way).
  This lesson was applied twice more in practice: two subsequent tokens
  were pasted for actual push operations, and both were explicitly
  revoked immediately after use, as agreed.
- Repo was ultimately renamed/recreated as **`handmade-gpt-repo`**
  (public) after an earlier private repo (`GPT-from_Scratch`) was
  deleted and recreated during cleanup, and a naming mismatch
  (`homemade-gpt-repo` vs. `homemde-gpt-repo` vs. the actual
  `handmade-gpt-repo`) briefly caused "Repository not found" errors —
  resolved by checking the actual repo page directly rather than guessing
  spelling variants.
- **Token-permission lesson**: an initial fine-grained PAT could clone
  (read) the repo but was denied (403) on push — read and write access
  are separate, explicitly-grantable permissions on a fine-grained token,
  and cloning successfully does not imply write access. A second token,
  generated with explicit read/write ("Contents: Read and write")
  permission, resolved this.
- Final, correct project structure: `scripts/` folder containing all six
  Python files (`data.py`, `bigram.py`, `loss-limit.py`,
  `uniform-context-model.py`, `context-model.py`, `4-context-model.py`),
  plus root-level `README.md`, `DEVLOG.md`, `LAB.md`, `.gitignore`, and
  `input.txt`.
