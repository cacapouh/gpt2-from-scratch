# Documentation

A DeepWiki-style guide to this repository.

## Pages
- [Architecture Overview](architecture.md) — system diagram, module map, data flow
- [Model Internals](model.md) — `GPTModel`, attention, layer norm, GELU, feed-forward
- [Data Pipeline](data.md) — BPE tokenization, sliding-window dataset
- [Training Loop](training.md) — loss, evaluation, sampling hooks
- [Text Generation](generation.md) — autoregressive decoding, temperature, top-k
- [Weight Loading](weight-loading.md) — HuggingFace safetensors → our `GPTModel`
- [Fine-tuning](finetuning.md) — adapting pretrained GPT-2 to new corpora
- [CLI Reference](cli.md) — all subcommands and flags
- [Glossary](glossary.md) — BPE, Pre-Norm, causal mask, weight tying, etc.

## Reading order
1. [Architecture Overview](architecture.md) — what and why
2. [Model Internals](model.md) — how each layer works
3. [Data Pipeline](data.md) + [Training Loop](training.md) — how the model learns
4. [Text Generation](generation.md) — how the model speaks
5. [Weight Loading](weight-loading.md) + [Fine-tuning](finetuning.md) — practical recipes
