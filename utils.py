"""Small utility helpers."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> str:
    d = get_device()
    if d.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"CUDA: {name} ({total:.1f} GB) | torch {torch.__version__}"
    return f"Device: {d} | torch {torch.__version__}"


if __name__ == "__main__":
    print(device_info())
