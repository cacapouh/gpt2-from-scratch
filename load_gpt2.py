"""Load GPT-2 pretrained weights from Hugging Face into our PyTorch model.

We fetch the model.safetensors file from the openai-community/gpt2 repo on the
Hugging Face Hub (mirror of OpenAI's official GPT-2 weights). This avoids
needing TensorFlow to read the original TF checkpoint.

Key names in HF GPT-2 safetensors (GPT-2 uses Conv1D, weights stored as (in, out)):
  wte.weight                          -> tok_emb.weight
  wpe.weight                          -> pos_emb.weight
  h.{i}.ln_1.weight / bias            -> blocks.{i}.norm1.scale / shift
  h.{i}.attn.c_attn.weight / bias     -> split 3 ways -> W_q/W_k/W_v (transpose)
  h.{i}.attn.c_proj.weight / bias     -> W_out (transpose)
  h.{i}.ln_2.weight / bias            -> blocks.{i}.norm2.scale / shift
  h.{i}.mlp.c_fc.weight / bias        -> ff.linear1 (transpose)
  h.{i}.mlp.c_proj.weight / bias      -> ff.linear2 (transpose)
  ln_f.weight / bias                  -> final_norm.scale / shift
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm

from config import GPT_CONFIGS
from model import GPTModel

HF_REPOS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-small": "openai-community/gpt2",
    "124M": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "355M": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "774M": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    "1558M": "openai-community/gpt2-xl",
}

CFG_KEYS = {
    "gpt2": "gpt2-small",
    "gpt2-small": "gpt2-small",
    "124M": "gpt2-small",
    "gpt2-medium": "gpt2-medium",
    "355M": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "774M": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "1558M": "gpt2-xl",
}


def download_safetensors(model_size: str, models_dir: str | Path = "gpt2_weights") -> Path:
    """Download model.safetensors for the given GPT-2 size to models_dir/<repo>/."""
    if model_size not in HF_REPOS:
        raise ValueError(f"unknown model size: {model_size}")
    repo = HF_REPOS[model_size]
    target_dir = Path(models_dir) / repo.split("/")[-1]
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / "model.safetensors"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="model.safetensors"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return dest


def _assign(param: torch.nn.Parameter, value: torch.Tensor | np.ndarray) -> None:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.to(dtype=param.dtype).contiguous()
    if param.shape != value.shape:
        raise ValueError(
            f"shape mismatch: param {tuple(param.shape)} vs value {tuple(value.shape)}"
        )
    with torch.no_grad():
        param.copy_(value)


def load_weights_into_gpt(model: GPTModel, sd: dict[str, torch.Tensor]) -> None:
    """Copy a HuggingFace GPT-2 state dict (from safetensors) into our model."""
    _assign(model.tok_emb.weight, sd["wte.weight"])
    _assign(model.pos_emb.weight, sd["wpe.weight"])

    n_layers = len(model.blocks)
    for i in range(n_layers):
        block = model.blocks[i]
        p = f"h.{i}."

        # c_attn: weight shape (emb, 3*emb) in HF (Conv1D style)
        c_attn_w = sd[p + "attn.c_attn.weight"]  # (emb, 3*emb)
        c_attn_b = sd[p + "attn.c_attn.bias"]    # (3*emb,)
        q_w, k_w, v_w = torch.chunk(c_attn_w, 3, dim=-1)
        q_b, k_b, v_b = torch.chunk(c_attn_b, 3, dim=-1)

        # Conv1D (in, out) -> nn.Linear (out, in) -> transpose
        _assign(block.attn.W_q.weight, q_w.T)
        _assign(block.attn.W_k.weight, k_w.T)
        _assign(block.attn.W_v.weight, v_w.T)
        _assign(block.attn.W_q.bias, q_b)
        _assign(block.attn.W_k.bias, k_b)
        _assign(block.attn.W_v.bias, v_b)

        _assign(block.attn.W_out.weight, sd[p + "attn.c_proj.weight"].T)
        _assign(block.attn.W_out.bias, sd[p + "attn.c_proj.bias"])

        _assign(block.ff.linear1.weight, sd[p + "mlp.c_fc.weight"].T)
        _assign(block.ff.linear1.bias, sd[p + "mlp.c_fc.bias"])
        _assign(block.ff.linear2.weight, sd[p + "mlp.c_proj.weight"].T)
        _assign(block.ff.linear2.bias, sd[p + "mlp.c_proj.bias"])

        _assign(block.norm1.scale, sd[p + "ln_1.weight"])
        _assign(block.norm1.shift, sd[p + "ln_1.bias"])
        _assign(block.norm2.scale, sd[p + "ln_2.weight"])
        _assign(block.norm2.shift, sd[p + "ln_2.bias"])

    _assign(model.final_norm.scale, sd["ln_f.weight"])
    _assign(model.final_norm.shift, sd["ln_f.bias"])

    # GPT-2 ties the output head to the token embedding.
    _assign(model.out_head.weight, sd["wte.weight"])


def build_openai_gpt(
    model_size: str = "gpt2", models_dir: str | Path = "gpt2_weights"
) -> GPTModel:
    """Download (if needed) and load GPT-2 weights into a new GPTModel."""
    from safetensors.torch import load_file  # lazy import

    if model_size not in CFG_KEYS:
        raise ValueError(f"unknown model size: {model_size}")

    cfg = dict(GPT_CONFIGS[CFG_KEYS[model_size]])
    cfg["qkv_bias"] = True   # HF/OpenAI weights include QKV bias
    cfg["drop_rate"] = 0.0   # inference: disable dropout

    path = download_safetensors(model_size, models_dir=models_dir)
    sd = load_file(str(path))

    model = GPTModel(cfg)
    load_weights_into_gpt(model, sd)
    model.eval()
    return model


# Backwards-compatible alias
download_gpt2 = download_safetensors
