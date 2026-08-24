# Sources and Attribution

## Primary inspiration

This repository was initiated by and follows the conceptual progression in:

- Nikhil Bajpai, [“I Built a GPT From Scratch on a MacBook — Days 1–5:
  From a Bigram to a Working Self-Attention
  Head”](https://medium.com/@nikhil.cse16/i-built-a-gpt-from-scratch-on-a-macbook-days-1-5-from-a-bigram-to-a-working-self-attention-head-0d3082ac417c),
  Medium, June 28, 2026.

The article’s stated scope is a from-first-principles progression from a
bigram model to a working self-attention head. This repository reproduces,
audits, and extends that progression with multi-head attention, averaged
train/validation evaluation, an empirical entropy calculation, explicit
evidence requirements, and troubleshooting records.

This repository does not claim that the article’s endpoint is a complete GPT
architecture. Positional embeddings, feedforward sublayers, residual
connections, layer normalization, and stacked Transformer blocks remain later
work.

## Dataset

- Tiny Shakespeare corpus from Andrej Karpathy’s
  [char-rnn repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).
- Expected file: `input.txt`
- Expected size: 1,115,394 bytes

## Additional conceptual reference

- Andrej Karpathy,
  [“Let’s build GPT: from scratch, in code, spelled
  out.”](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Reuse and licensing

Source attribution does not itself grant a license to reuse third-party text or
code. Before redistribution as formal course material, the repository owner
must select a repository license and confirm that copied or adapted material is
used consistently with each source’s terms.
