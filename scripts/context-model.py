import torch
import torch.nn as nn
from torch.nn import functional as F

from common import make_batch, print_evaluation, select_device
from data import train_data, val_data, vocab_size

SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 8
N_EMBD = 32
HEAD_SIZE = 16
TRAIN_STEPS = 10_000

torch.manual_seed(SEED)
device = select_device()


def get_batch(split):
    source = train_data if split == "train" else val_data
    return make_batch(source, BLOCK_SIZE, BATCH_SIZE, device)


class SingleHeadAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.key = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.lm_head = nn.Linear(HEAD_SIZE, vocab_size)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        weights = query @ key.transpose(-2, -1) * HEAD_SIZE**-0.5
        time = idx.shape[1]
        weights = weights.masked_fill(
            self.causal_mask[:time, :time] == 0, float("-inf")
        )
        weights = F.softmax(weights, dim=-1)

        logits = self.lm_head(weights @ value)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1)
            )
        return logits, loss


model = SingleHeadAttentionModel().to(device)
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
