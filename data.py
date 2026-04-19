"""Dataset / DataLoader for GPT-2 style next-token prediction."""
from __future__ import annotations

from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


class GPTDataset(Dataset):
    """Sliding-window next-token-prediction dataset."""

    def __init__(self, text: str, tokenizer, max_length: int = 256, stride: int | None = None):
        if stride is None:
            stride = max_length  # non-overlapping by default
        if stride <= 0:
            raise ValueError("stride must be positive")

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        # Need at least max_length + 1 tokens to make one (input, target) pair.
        for i in range(0, len(token_ids) - max_length, stride):
            chunk = token_ids[i : i + max_length + 1]
            self.input_ids.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.target_ids.append(torch.tensor(chunk[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    batch_size: int = 8,
    max_length: int = 256,
    stride: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = get_tokenizer()
    dataset = GPTDataset(text, tokenizer, max_length=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
