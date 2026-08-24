from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "input.txt"

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_PATH}. "
        "Run the dataset-download command from the repository root."
    )

text = DATASET_PATH.read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(value):
    return [stoi[character] for character in value]


def decode(indices):
    return "".join(itos[index] for index in indices)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
