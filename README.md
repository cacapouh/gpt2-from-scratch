# GPT-2 from Scratch

A minimal, from-scratch PyTorch implementation of GPT-2 (124M) following
Sebastian Raschka's *"Build a Large Language Model (from Scratch)"*.

## Layout
- [config.py](config.py) - model size presets
- [data.py](data.py) - sliding-window dataset + DataLoader (tiktoken GPT-2 BPE)
- [model.py](model.py) - `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel`
- [generate.py](generate.py) - autoregressive sampling (temperature + top-k)
- [train.py](train.py) - training loop with periodic eval / sample printing
- [load_gpt2.py](load_gpt2.py) - download HuggingFace GPT-2 safetensors and map into our model
- [main.py](main.py) - CLI (`train` / `generate`)
- [utils.py](utils.py) - device helpers

## Install

Use Python 3.12 (PyTorch 2.11 has CUDA wheels for cp310-cp313 but not cp314 yet).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# CUDA 12.8 build (for RTX 50-series / Blackwell sm_120):
pip install --index-url https://download.pytorch.org/whl/cu128 torch

# Or CPU-only:
# pip install torch

pip install -r requirements.txt
```

Verify the GPU is detected:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Usage

**Train on a short text (e.g. "The Verdict" by Edith Wharton):**

```powershell
python main.py train --data the-verdict.txt --epochs 10
```

**Generate with OpenAI's pretrained weights (downloads on first run):**

```powershell
python main.py generate --weights gpt2 --prompt "The meaning of life is"
```

**Generate with your own checkpoint:**

```powershell
python main.py generate --weights checkpoints/model.pt --prompt "Hello"
```

Use `--temperature` and `--top-k` to control sampling.

## Hardware

Tuned for a single RTX 5070 (12 GB). Defaults: `max_length=256`, `batch_size=8`.
GPT-2 small (~124M params) fits easily; enlarge batch size if VRAM allows.
