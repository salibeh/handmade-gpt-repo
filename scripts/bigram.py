import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import make_batch, print_evaluation, select_device
from data import train_data, val_data, vocab_size

SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 8
TRAIN_STEPS = int(os.getenv("HANDMADE_GPT_TRAIN_STEPS", "10000"))

torch.manual_seed(SEED)
device = select_device()


def get_batch(split):
    source = train_data if split == "train" else val_data
    return make_batch(source, BLOCK_SIZE, BATCH_SIZE, device)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch * time, channels),
                targets.reshape(batch * time),
            )
        return logits, loss


model = BigramLanguageModel().to(device)
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
