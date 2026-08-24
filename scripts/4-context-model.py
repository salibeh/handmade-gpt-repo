import torch
import torch.nn as nn
from torch.nn import functional as F
from data import vocab_size, train_data, val_data, decode

torch.manual_seed(1337)
batch_size = 32
block_size = 8
n_embd = 32
n_head = 4
head_size = n_embd // n_head  # 32 // 4 = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class SimpleGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)
        x = self.sa_heads(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = SimpleGPT()
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

# --- Generation demo ---
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=200)
print("\n--- Sample generation ---")
print(decode(generated[0].tolist()))

# --- Top-3 next-character prediction check against real validation text ---
model.eval()
sample = val_data[:block_size].unsqueeze(0)
logits, _ = model(sample)
probs = F.softmax(logits, dim=-1)

print("\n--- Top-3 prediction check ---")
print("Input:", decode(sample[0].tolist()))
for t in range(block_size):
    top3 = torch.topk(probs[0, t], 3)
    top3_chars = [decode([i.item()]) for i in top3.indices]
    top3_probs = [f"{p.item():.3f}" for p in top3.values]
    actual_next = decode([val_data[t+1].item()]) if t+1 < len(val_data) else "?"
    print(f"position {t} (\"{decode(sample[0][:t+1].tolist())}\") -> "
          f"predicted top-3: {list(zip(top3_chars, top3_probs))} | actual next: '{actual_next}'")
