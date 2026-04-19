# Architecture Overview

This repository is a minimal PyTorch re-implementation of **GPT-2 small (124M)** with
enough infrastructure to train from scratch, fine-tune, and run inference with
OpenAI's pretrained weights — all from a single CLI.

## 10-second summary

| Aspect | Value |
|---|---|
| Model family | Decoder-only Transformer (GPT-2 small) |
| Parameters | ~124M (untied) / ~162.4M when positional embeddings are shrunk to `max_length=256` |
| Tokenizer | GPT-2 BPE via `tiktoken` (vocab 50,257) |
| Training objective | Next-token prediction (causal LM, cross-entropy) |
| Precision | fp32 |
| Hardware target | Single RTX 5070 (12 GB VRAM), CUDA 12.8 |
| Weight tying | `out_head.weight` is tied to `tok_emb.weight` when loading OpenAI weights |

## Module map

```mermaid
graph LR
    subgraph CLI
        MAIN[main.py]
    end
    subgraph Model
        MODEL[model.py<br/>GPTModel]
        CFG[config.py<br/>GPT_CONFIGS]
    end
    subgraph Data
        DATA[data.py<br/>GPTDataset + loader]
        TOK[tiktoken<br/>GPT-2 BPE]
    end
    subgraph Runtime
        TRAIN[train.py<br/>train_model]
        GEN[generate.py<br/>generate]
    end
    subgraph Weights
        LOAD[load_gpt2.py<br/>build_openai_gpt]
        HF[(HuggingFace<br/>safetensors)]
    end
    subgraph Scripts
        DLW[download_wilde.py]
        GUT[(Project<br/>Gutenberg)]
    end

    MAIN --> CFG
    MAIN --> DATA
    MAIN --> TRAIN
    MAIN --> GEN
    MAIN --> LOAD
    DATA --> TOK
    TRAIN --> MODEL
    TRAIN --> GEN
    GEN --> MODEL
    LOAD --> MODEL
    LOAD --> HF
    DLW --> GUT
```

## High-level data flow

```mermaid
flowchart TD
    T[Text file] -->|tiktoken encode| IDS[token ids]
    IDS -->|sliding window stride=max_length| PAIRS[input_ids, target_ids]
    PAIRS --> DL[DataLoader batches]
    DL --> FW[GPTModel forward]
    FW --> LOGITS[logits b,t,vocab]
    LOGITS -->|cross_entropy vs target_ids| LOSS
    LOSS -->|backward, AdamW| FW
    LOGITS -->|last-position only| SAMPLE[temperature / top-k]
    SAMPLE -->|append| IDS2[new token]
    IDS2 -->|decode| OUT[generated text]
```

## Three execution modes

All entered through [../main.py](../main.py):

```mermaid
flowchart LR
    U([user]) --> M[main.py]
    M -->|subcommand| T{mode}
    T -->|train| TRAIN_MODE[fresh GPTModel<br/>random init<br/>lr=4e-4]
    T -->|finetune| FT_MODE[load pretrained<br/>re-enable dropout<br/>lr=1e-5]
    T -->|generate| GEN_MODE[load ckpt or HF<br/>autoregressive decode]
    TRAIN_MODE --> CKPT1[checkpoints/model.pt]
    FT_MODE --> CKPT2[checkpoints/wilde.pt]
    GEN_MODE --> STDOUT[stdout]
```

- **train**: cold start. Use to observe loss curves and to verify the architecture is wired correctly.
- **finetune**: warm start from OpenAI weights. Cheap, fast, and produces high-quality stylistic transfer.
- **generate**: inference from either HuggingFace weights (`gpt2`, `gpt2-medium`, …) or any saved `.pt` checkpoint.

## Why GPT-2 small?

- Fits in 12 GB VRAM with `batch_size=8`, `max_length=256`, fp32, AdamW.
- The published TF checkpoint and its HF mirror are simple (no GQA, no RoPE, no MoE).
- Pre-Norm decoder-only is the direct ancestor of every modern open-weights LLM; everything you learn here transfers.

## Design decisions worth knowing

- **Custom `LayerNorm` / `GELU`**: reimplemented to match `nn.LayerNorm` / `nn.GELU(approximate='tanh')` so the math is visible. Verified numerically equivalent.
- **Pre-Norm residuals**: `x = x + f(norm(x))` inside each [TransformerBlock](../model.py). This is what GPT-2 uses; it stabilizes training of deep stacks compared to Post-Norm.
- **Separate `W_q`, `W_k`, `W_v`**: clearer than a fused `c_attn` matrix. The weight loader splits OpenAI's fused `c_attn` into three via `torch.chunk(..., 3, dim=-1)`.
- **Boolean causal mask as a `register_buffer`**: no gradients, travels with `.to(device)`, but `persistent=False` keeps it out of `state_dict`.
- **Positional embeddings are learned, shape `(context_length, emb_dim)`**: the CLI shrinks `context_length` to `max_length` during `train` to save memory. The pretrained loader keeps the full 1024.

See [Model Internals](model.md) for the per-layer tour.
