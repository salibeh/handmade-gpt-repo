import torch
import torch.nn as nn
from torch.nn import functional as F
from data import vocab_size, train_data, val_data

torch.manual_seed(1337)
batch_size = 32
block_size = 8
n_embd = 32

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

token_embedding_table = nn.Embedding(vocab_size, n_embd)
lm_head = nn.Linear(n_embd, vocab_size)

tril = torch.tril(torch.ones(block_size, block_size))
wei_fixed = tril / tril.sum(1, keepdim=True)

optimizer = torch.optim.AdamW(
    list(token_embedding_table.parameters()) + list(lm_head.parameters()),
    lr=1e-3
)

for step in range(10000):
    xb, yb = get_batch('train')
    x = token_embedding_table(xb)
    xbow = wei_fixed @ x
    logits = lm_head(xbow)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")
print(f"final perplexity: {torch.exp(loss).item():.4f}")
