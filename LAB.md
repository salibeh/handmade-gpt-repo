# Lab: Building a Character-Level GPT From Scratch (macOS, Apple Silicon)

A hands-on, reproducible lab for building up from a bigram model to
multi-head self-attention, with each step measured against a theoretical
baseline rather than taken on faith. Written for macOS on Apple Silicon
(M1/M2/M3), but the code is portable.

## Prerequisites

- macOS with Python 3.9+ (`python3 --version` to check)
- ~500MB free disk space (PyTorch + dependencies + dataset)
- Terminal access

## 0. Environment setup

```bash
mkdir handmade-gpt && cd handmade-gpt
python3 -m venv venv
source venv/bin/activate

pip install torch
pip install numpy   # required — pip install torch does not always pull this in
```

Verify Apple Silicon GPU (MPS / Metal) support:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available()); print(torch.backends.mps.is_built())"
```

Both should print `True`. `is_available()` being `True` means PyTorch can
place tensors on the M-series GPU cores (via `.to('mps')`) instead of only
the CPU — meaningfully faster for the matrix multiplications a neural net
does constantly. This lab's scripts are small enough to run fine on CPU too;
MPS becomes more valuable once you scale up embedding size, depth, or batch
size.

### `.gitignore` (set this up *before* your first commit if using git)

```bash
cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
EOF
```

Committing `venv/` accidentally will blow past GitHub's 100MB single-file
limit (PyTorch's compiled libraries are 100-200MB+) — if this has already
happened and a push was rejected, the clean fix for a fresh repo is:
`rm -rf .git`, recreate `.gitignore`, then `git init` again from scratch,
rather than trying to surgically remove the file from existing history.

## 1. Get the dataset

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Confirm: `wc -c input.txt` should show `1115394`.

## 2. Shared data module — `scripts/data.py`

All other scripts import from this rather than duplicating the setup.
Each `.py` file is its own blank slate — variables from a previous
interactive session or a different script do **not** carry over.

```python
import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

Sanity check: `vocab_size` should be `65`; `len(train_data)` /
`len(val_data)` should be `1003854` / `111540`.

## 3. Bigram baseline — `scripts/bigram.py`

The simplest possible model: predicts the next character using only the
current character (one row of a 65×65 lookup table).

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from data import vocab_size, train_data, val_data

torch.manual_seed(1337)
batch_size = 32
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(10000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 1000 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")
print(f"final perplexity: {torch.exp(loss).item():.4f}")
```

**Expect:** final loss ≈ 2.45, perplexity ≈ 11.5-11.6, plateauing hard after
roughly step 5000 — the model has converged to its structural ceiling.

## 4. Verify the ceiling is real — `scripts/loss-limit.py`

Computes the theoretical minimum loss directly from real character-pair
statistics (conditional entropy), independent of any trained model. This
confirms whether the bigram's plateau is a training limitation or a genuine
information limitation.

```python
from data import vocab_size, train_data
from collections import Counter
import math

pairs = Counter()
counts = Counter()
for i in range(len(train_data) - 1):
    c1 = train_data[i].item()
    c2 = train_data[i+1].item()
    pairs[(c1, c2)] += 1
    counts[c1] += 1

total_entropy = 0.0
total_count = sum(counts.values())

for c1 in counts:
    c1_count = counts[c1]
    entropy_c1 = 0.0
    for c2 in range(vocab_size):
        p = pairs.get((c1, c2), 0) / c1_count
        if p > 0:
            entropy_c1 -= p * math.log(p)
    weight = c1_count / total_count
    total_entropy += weight * entropy_c1

print(f"Theoretical minimum loss (entropy floor): {total_entropy:.4f}")
```

**Expect:** ≈ 2.4519 — matching the trained bigram's loss almost exactly.
This is the core empirical checkpoint of the whole lab: it proves the
bigram isn't undertrained, it's *information-limited* — one character of
context genuinely cannot resolve more uncertainty than this, no matter how
long you train.

## 5. Context via uniform averaging (deliberately a negative result)

Build the causal running-average mechanism (`tril`/`wei` matrix multiply)
and wire it into a full model exactly like the bigram, but blending up to
8 characters of context with **equal weight** per position, via
`nn.Linear(n_embd, vocab_size)` bridging the blended context vector to 65
logits.

**Expect:** final loss ≈ 2.86, *worse* than the bigram. This is intentional
and instructive — uniform weighting dilutes useful nearby signal with
irrelevant distant signal. It directly motivates the next step.

## 6. Single self-attention head — `scripts/context-model.py`

Replace uniform weights with **learned, content-dependent** weights via
Query/Key/Value projections:

```python
head_size = 16
key   = nn.Linear(n_embd, head_size, bias=False)
query = nn.Linear(n_embd, head_size, bias=False)
value = nn.Linear(n_embd, head_size, bias=False)

k, q, v = key(x), query(x), value(x)
wei = q @ k.transpose(-2, -1) * head_size ** -0.5
wei = wei.masked_fill(tril == 0, float('-inf'))   # causal mask
wei = F.softmax(wei, dim=-1)
out = wei @ v
```

Bridge `out` (16-dim) to 65 logits via `nn.Linear(head_size, vocab_size)`,
train end-to-end.

**Expect:** final loss ≈ 2.44 — a small win over the bigram, and a clear
win over uniform averaging, using the same 8-token context window.

## 7. Multi-head attention — `scripts/multi-head-model.py`

Refactor into reusable classes; run several attention heads in parallel and
concatenate their outputs.

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
```

Wrap in a full `SimpleGPT(nn.Module)` (embedding → `MultiHeadAttention` →
`lm_head`), train the same way via `model.parameters()`.

**Expect:** final loss ≈ 2.25, perplexity ≈ 9.5 — the clearest improvement
in the whole lab. Multiple independently-learned relevance patterns capture
more useful structure than any single pattern alone.

## 8. Generation

```python
def generate(model, idx, max_new_tokens, block_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Cropping to the last `block_size` tokens each step is required — the model
was only ever trained on fixed-length context and will error on longer
input otherwise.

**Expect:** English-shaped but largely incoherent output at this
architectural depth. This is expected, not a bug — residual connections,
LayerNorm, and a feedforward MLP (not yet built in this lab) are what push
output toward genuine coherence.

## 9. Verification: inspect real predictions, don't just trust the loss number

Run the trained multi-head model against a real (held-out) validation
snippet and print top-3 predicted next-characters with probabilities at
each position, alongside the actual next character. Look for: high
confidence at genuinely predictable transitions, categorically-sensible
misses (vowel-for-vowel, letter-for-letter), and appropriately low
confidence at ambiguous points. This is a more convincing, checkable
demonstration of real learned structure than eyeballing generated text.

## Results checkpoint table

| Model                      | Context             | Final loss | Perplexity |
|-----------------------------|----------------------|-----------:|-----------:|
| Bigram                      | 1 character          | ~2.4488    | ~11.57     |
| Uniform averaging            | 8 chars, equal weight| ~2.8604    | ~17.47     |
| Single self-attention head   | 8 chars, learned wt  | ~2.4439    | ~11.52     |
| 4-head multi-head attention  | 8 chars, 4 patterns  | ~2.2479    | ~9.47      |
| Entropy floor (computed)     | 1 character (theory) | 2.4519     | —          |

Exact numbers will vary slightly run-to-run due to random initialization
and batch sampling, but the *ordering* (uniform < bigram < single-head <
multi-head) should reproduce consistently.

## Next steps (not covered in this lab)

- Residual connections: `x = x + sublayer(x)`
- LayerNorm before each sublayer
- Feedforward MLP after attention, per position
- Stacking multiple transformer blocks
- Scaling up `n_embd`, `block_size`, training steps
