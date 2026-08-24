import torch
import torch.nn as nn
from torch.nn import functional as F

from common import make_batch, print_evaluation, select_device
from data import decode, train_data, val_data, vocab_size

SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 8
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.0
TRAIN_STEPS = 10_000

if N_EMBD % N_HEAD != 0:
    raise ValueError("N_EMBD must be divisible by N_HEAD")

torch.manual_seed(SEED)
device = select_device()


def get_batch(split):
    source = train_data if split == "train" else val_data
    return make_batch(source, BLOCK_SIZE, BATCH_SIZE, device)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.query = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.value = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.projection = nn.Linear(N_EMBD, N_EMBD)
        self.attention_dropout = nn.Dropout(DROPOUT)
        self.residual_dropout = nn.Dropout(DROPOUT)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

    def forward(self, x):
        batch, time, channels = x.shape
        head_size = channels // N_HEAD

        key = self.key(x).view(batch, time, N_HEAD, head_size).transpose(1, 2)
        query = self.query(x).view(batch, time, N_HEAD, head_size).transpose(1, 2)
        value = self.value(x).view(batch, time, N_HEAD, head_size).transpose(1, 2)

        weights = query @ key.transpose(-2, -1) * head_size**-0.5
        weights = weights.masked_fill(
            self.causal_mask[:time, :time] == 0, float("-inf")
        )
        weights = self.attention_dropout(F.softmax(weights, dim=-1))

        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.residual_dropout(self.projection(output))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_attention = nn.LayerNorm(N_EMBD)
        self.attention = CausalSelfAttention()
        self.layer_norm_feedforward = nn.LayerNorm(N_EMBD)
        self.feedforward = FeedForward()

    def forward(self, x):
        x = x + self.attention(self.layer_norm_attention(x))
        x = x + self.feedforward(self.layer_norm_feedforward(x))
        return x


class CharacterGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(N_LAYER)]
        )
        self.final_layer_norm = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        time = idx.shape[1]
        if time > BLOCK_SIZE:
            raise ValueError(
                f"Input length {time} exceeds BLOCK_SIZE={BLOCK_SIZE}"
            )

        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.final_layer_norm(x))

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


model = CharacterGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
parameter_count = sum(parameter.numel() for parameter in model.parameters())

print(f"device: {device}")
print(f"parameters: {parameter_count:,}")
print(
    f"configuration: block={BLOCK_SIZE}, embedding={N_EMBD}, "
    f"heads={N_HEAD}, layers={N_LAYER}, dropout={DROPOUT}"
)

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
generated = model.generate(context, max_new_tokens=300)
print("\n--- Sample generation ---")
print(decode(generated[0].detach().cpu().tolist()))
