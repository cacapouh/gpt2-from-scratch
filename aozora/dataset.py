"""スライディングウィンドウの PyTorch Dataset（rinna T5Tokenizer 用）。"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MeijiDataset(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int = 512, stride: int | None = None):
        if stride is None:
            stride = max_length
        if stride <= 0:
            raise ValueError("stride must be positive")

        # rinna の T5Tokenizer。特殊トークンは挟まない（自前で </s> を入れている）
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        for i in range(0, len(token_ids) - max_length, stride):
            chunk = token_ids[i : i + max_length + 1]
            self.input_ids.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.target_ids.append(torch.tensor(chunk[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]
