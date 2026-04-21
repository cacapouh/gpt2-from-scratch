"""素の rinna またはファインチューン済みチェックポイントでテキスト生成。"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, T5Tokenizer


def generate(
    weights: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(weights)
    tokenizer.do_lower_case = True

    model = AutoModelForCausalLM.from_pretrained(weights).to(device)
    model.eval()

    ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)
