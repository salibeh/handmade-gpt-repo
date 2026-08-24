import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import make_batch, print_evaluation, select_device
from data import decode, train_data, val_data, vocab_size

SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 8
N_EMBD = 32
N_HEAD = 4
HEAD_SIZE = N_EMBD // N_HEAD
TRAIN_STEPS = int(os.getenv("HANDMADE_GPT_TRAIN_STEPS", "10000"))

torch.manual_seed(SEED)
device = select_device()


def get_batch(split):
    source = train_data if split == "train" else val_data
    return make_batch(source, BLOCK_SIZE, BATCH_SIZE, device)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        time = x.shape[1]
        key = self.key(x)
        query = self.query(x)
        weights = query @ key.transpose(-2, -1) * HEAD_SIZE**-0.5
        weights = weights.masked_fill(
            self.causal_mask[:time, :time] == 0, float("-inf")
        )
        weights = F.softmax(weights, dim=-1)
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(N_HEAD)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadCharacterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.attention = MultiHeadAttention()
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.lm_head(self.attention(self.token_embedding(idx)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        if was_training:
            self.train()
        return idx


model = MultiHeadCharacterModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"device: {device}")
for step in range(TRAIN_STEPS):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 1000 == 0:
        print(f"step {step}: training-batch loss {loss.item():.4f}")

print_evaluation(model, get_batch)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200)
print("\n--- Sample generation ---")
print(decode(generated[0].detach().cpu().tolist()))

model.eval()
sample = val_data[:BLOCK_SIZE].unsqueeze(0).to(device)
with torch.no_grad():
    logits, _ = model(sample)
    probabilities = F.softmax(logits, dim=-1)

print("\n--- Illustrative top-3 predictions (not an aggregate metric) ---")
print("Input:", decode(sample[0].detach().cpu().tolist()))
for position in range(BLOCK_SIZE):
    top3 = torch.topk(probabilities[0, position], 3)
    characters = [decode([index.item()]) for index in top3.indices.cpu()]
    scores = [f"{score.item():.3f}" for score in top3.values.cpu()]
    actual = decode([val_data[position + 1].item()])
    prefix = decode(sample[0, : position + 1].detach().cpu().tolist())
    print(
        f'position {position} ("{prefix}") -> '
        f"top-3: {list(zip(characters, scores))} | actual: {actual!r}"
    )
