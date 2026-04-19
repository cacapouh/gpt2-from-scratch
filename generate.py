"""Text generation utilities."""
from __future__ import annotations

import torch

from data import get_tokenizer


@torch.no_grad()
def generate_token_ids(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Autoregressively extend idx by max_new_tokens tokens."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # last position

        if top_k is not None and top_k > 0:
            top_vals, _ = torch.topk(logits, top_k)
            min_val = top_vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

        if temperature is not None and temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (next_id == eos_id).all():
            break

        idx = torch.cat([idx, next_id], dim=1)
    return idx


def text_to_ids(text: str, tokenizer) -> torch.Tensor:
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def ids_to_text(ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(ids.squeeze(0).tolist())


def generate(
    model,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = 50,
    device: torch.device | str = "cpu",
) -> str:
    tokenizer = get_tokenizer()
    idx = text_to_ids(prompt, tokenizer).to(device)
    context_size = model.cfg["context_length"]
    out_ids = generate_token_ids(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
    )
    return ids_to_text(out_ids, tokenizer)
