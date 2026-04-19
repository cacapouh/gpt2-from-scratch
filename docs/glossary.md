# Glossary

Short definitions for the jargon that shows up in the other pages.

## BPE (Byte-Pair Encoding)
Subword tokenizer. Starts from raw bytes and greedily merges the most frequent
adjacent pairs into new tokens until a target vocab size is reached. GPT-2's
vocab is 50,257 tokens.

## Causal mask
A boolean matrix that prevents position *t* from attending to any position
> *t*. Implemented as an upper-triangular mask that turns disallowed scores
into `-inf` before softmax.

## Context length / `context_length`
Maximum number of tokens the model can see at once. GPT-2 small = 1024. We
sometimes shrink this (e.g. to 256) during `train` to save memory on the
positional embedding table.

## Cross-entropy loss
$-\log p(\text{target})$ averaged over every predicted position. Equivalent
to the negative log-likelihood of the true token under the model's softmax.

## Decoder-only
A Transformer that only uses causal self-attention (no encoder-style bidirectional attention). All modern open LLMs are decoder-only.

## Dropout
Randomly zeroes a fraction of activations during training. Disabled at
inference by calling `model.eval()` (which flips every `nn.Dropout` to pass-through).

## GELU
Gaussian Error Linear Unit, the activation function inside the MLP. GPT-2
uses the **tanh approximation**: $0.5 x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$.

## Head / multi-head attention
Splitting the `d`-dimensional attention into `H` parallel "heads" of size
`d/H`, each with its own Q/K/V projection (in practice fused into one linear
and split), so the model can attend to multiple types of relationships at once.

## Layer norm (Pre-Norm)
Per-token normalization applied **before** each sublayer in GPT-2:
`x = x + f(LayerNorm(x))`. Stabilizes very deep stacks.

## Logits
Raw unnormalized scores output by the final linear layer. Convert to
probabilities with `softmax`. Shape `(batch, seq_len, vocab_size)`.

## Positional embedding
Learned lookup table `(context_length, emb_dim)` added to the token embedding
to give the model a sense of position. GPT-2 uses **absolute learned**
positional embeddings (not rotary/RoPE, not ALiBi).

## QKV bias
Whether the linear projections producing Q, K, V include an additive bias.
GPT-2 pretrained weights include it (`qkv_bias=True`); training from scratch
doesn't need it.

## Safetensors
A file format for storing tensors. Mmap-friendly, safe to load (no pickle
code execution), and now the default on HuggingFace. We load GPT-2 this way.

## Softmax / temperature
Softmax turns logits into probabilities. Dividing logits by `temperature`
before softmax either sharpens (`T<1`) or flattens (`T>1`) the distribution.
`T=0` is shorthand for "just argmax".

## Top-k sampling
Before softmax, set every logit not in the top `k` to `-inf`, so sampling
only chooses among the `k` most likely next tokens.

## Weight tying
Sharing the token embedding matrix with the output head:
`out_head.weight is tok_emb.weight`. Halves the effective parameter count of
those two matrices. GPT-2 does this; `load_gpt2.py` enforces it via copy.

## `<|endoftext|>` (token id 50256)
GPT-2's document-boundary special token. `download_wilde.py` inserts it
between concatenated Wilde works so the model doesn't try to smooth over the
transition.
