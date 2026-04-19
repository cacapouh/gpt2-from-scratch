# CLI Reference

Entry: `python main.py <subcommand> [flags]`  — see [../main.py](../main.py).

Three subcommands: [train](#train), [finetune](#finetune), [generate](#generate).

## train

Train a GPT-2 from random init.

```powershell
python main.py train --data the-verdict.txt --epochs 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | **required** | Path to a UTF-8 text file |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `8` | Mini-batch size |
| `--max-length` | `256` | Sequence length *and* positional-embedding size |
| `--lr` | `4e-4` | AdamW learning rate |
| `--eval-freq` | `20` | Eval every N steps |
| `--sample-every` | `100` | Sample text every N steps |
| `--prompt` | `"Every effort moves you"` | Prompt used for sampling |
| `--checkpoint` | `checkpoints/model.pt` | Where to save |

## finetune

Warm-start from OpenAI pretrained weights.

```powershell
python main.py finetune --data wilde.txt --epochs 3 --prompt "Marriage is"
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | **required** | Path to a UTF-8 text file |
| `--base-model` | `gpt2` | `gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl` |
| `--epochs` | `3` | |
| `--batch-size` | `4` | |
| `--max-length` | `256` | |
| `--lr` | `1e-5` | 40× lower than `train` on purpose |
| `--eval-freq` | `20` | |
| `--sample-every` | `50` | |
| `--prompt` | `"Marriage is"` | |
| `--checkpoint` | `checkpoints/wilde.pt` | |
| `--models-dir` | `gpt2_weights` | Cache for downloaded safetensors |

After training, the CLI prints the matching `generate` command.

## generate

Sample text from either pretrained weights or a checkpoint.

```powershell
# OpenAI pretrained
python main.py generate --weights gpt2 --prompt "The meaning of life is"

# local checkpoint
python main.py generate --weights checkpoints/wilde.pt --prompt "Marriage is"
```

| Flag | Default | Meaning |
|---|---|---|
| `--weights` | **required** | `gpt2`/`gpt2-medium`/… **or** path to a `.pt` checkpoint |
| `--prompt` | **required** | Text to condition on |
| `--max-new-tokens` | `50` | How many tokens to generate |
| `--temperature` | `1.0` | Sampling temperature. `0` → greedy |
| `--top-k` | `50` | Top-k filter. `0` → disabled |
| `--models-dir` | `gpt2_weights` | Cache dir for HF weights |

## Exit behavior

All three subcommands return exit code `0` on success and propagate Python
exceptions (non-zero) on failure. No custom error handling on top.

## Environment

- Python 3.12 venv (see [../README.md](../README.md) for install steps).
- PyTorch 2.11 + CUDA 12.8 for RTX 50-series. CPU fallback works automatically via [../utils.py](../utils.py) `get_device()`.
- First `generate --weights gpt2` call downloads ~548 MB to `gpt2_weights/gpt2/model.safetensors` and caches it.
