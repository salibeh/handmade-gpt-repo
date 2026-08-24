import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import make_batch, print_evaluation, select_device
from data import train_data, val_data, vocab_size

SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 8
N_EMBD = 32
TRAIN_STEPS = int(os.getenv("HANDMADE_GPT_TRAIN_STEPS", "10000"))

torch.manual_seed(SEED)
device = select_device()


def get_batch(split):
    source = train_data if split == "train" else val_data
    return make_batch(source, BLOCK_SIZE, BATCH_SIZE, device)


class UniformContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
        lower = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("weights", lower / lower.sum(1, keepdim=True))

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        x = self.weights[: idx.shape[1], : idx.shape[1]] @ x
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1)
            )
        return logits, loss


model = UniformContextModel().to(device)
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
