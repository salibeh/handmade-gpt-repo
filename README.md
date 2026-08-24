# GPT From Scratch

A character-level GPT built from the ground up — bigram model → causal
averaging → single self-attention head → multi-head attention — with each
step measured against a theoretical baseline (Shannon entropy) rather than
taken on faith.

See `LAB.md` for a full reproducible walkthrough (macOS setup included),
and `DEVLOG.md` for a narrative log of what was built, what was tried and
failed on purpose, and why.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy

curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

cd scripts
python3 bigram.py            # baseline: 1 char of context
python3 loss-limit.py        # theoretical entropy floor, computed from data
python3 uniform-context-model.py   # 8 chars, uniform weight (negative result)
python3 context-model.py     # 8 chars, single learned attention head
python3 4-context-model.py   # 8 chars, 4-head attention + generation demo
```

Note: `input.txt` should be downloaded into the repo root (one level above
`scripts/`), and scripts should be run from inside `scripts/` since they
import from `data.py` via a relative import.

## Results

| Model                      | Context             | Final loss | Perplexity |
|-----------------------------|----------------------|-----------:|-----------:|
| Bigram                      | 1 character          | ~2.4488    | ~11.57     |
| Uniform averaging            | 8 chars, equal weight| ~2.8604    | ~17.47     |
| Single self-attention head   | 8 chars, learned wt  | ~2.4439    | ~11.52     |
| 4-head multi-head attention  | 8 chars, 4 patterns  | ~2.2479    | ~9.47      |
| Entropy floor (computed)     | 1 character (theory) | 2.4519     | —          |
