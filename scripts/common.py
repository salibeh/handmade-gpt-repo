import math
import os

import torch


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batch(data, block_size, batch_size, device):
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_losses(model, batch_fn, eval_iters=None):
    if eval_iters is None:
        eval_iters = int(os.getenv("HANDMADE_GPT_EVAL_ITERS", "200"))
    was_training = model.training
    model.eval()
    result = {}

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            x, y = batch_fn(split)
            _, loss = model(x, y)
            losses[index] = loss.detach().cpu()
        result[split] = losses.mean().item()

    if was_training:
        model.train()
    return result


def print_evaluation(model, batch_fn, eval_iters=200):
    losses = estimate_losses(model, batch_fn, eval_iters)
    print(f"averaged train loss: {losses['train']:.4f}")
    print(f"averaged validation loss: {losses['val']:.4f}")
    print(f"validation perplexity: {math.exp(losses['val']):.4f}")
