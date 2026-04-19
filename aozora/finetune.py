"""rinna/japanese-gpt2-medium を明治文学でファインチューニング。"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, T5Tokenizer

from dataset import MeijiDataset


def _load_tokenizer(name: str) -> T5Tokenizer:
    tok = T5Tokenizer.from_pretrained(name)
    tok.do_lower_case = True  # rinna 推奨
    return tok


def _sample(model, tokenizer, prompt: str, device, max_new_tokens: int = 60) -> str:
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if was_training:
        model.train()
    return text


def finetune(
    data_path: str = "data/combined.txt",
    base_model: str = "rinna/japanese-gpt2-medium",
    output_dir: str = "checkpoints/meiji_gpt2",
    epochs: int = 3,
    batch_size: int = 2,
    max_length: int = 512,
    lr: float = 5e-5,
    weight_decay: float = 0.1,
    eval_freq: int = 50,
    sample_every: int = 200,
    sample_prompt: str = "文明の進歩は、",
    grad_accum_steps: int = 1,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = _load_tokenizer(base_model)
    print(f"loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    text = Path(data_path).read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} chars")
    dataset = MeijiDataset(text, tokenizer, max_length=max_length)
    print(f"windows: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"batches/epoch: {len(loader)}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    for epoch in range(epochs):
        running = 0.0
        optimizer.zero_grad()
        for step, (input_ids, target_ids) in enumerate(loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # transformers の CausalLM は labels を渡すと内部で 1 つシフト済み CE を計算する。
            # ここでは既に shift 済み target を渡しているので labels=target_ids でそのまま OK
            # （実体は同じ Cross-Entropy）。
            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running += outputs.loss.item()
            if global_step % eval_freq == 0:
                avg = running / max(1, (global_step % eval_freq) + 1)
                print(f"ep {epoch+1} step {global_step} | loss {outputs.loss.item():.3f} (avg {avg:.3f})")

            if global_step > 0 and global_step % sample_every == 0:
                sample = _sample(model, tokenizer, sample_prompt, device)
                print(f"  sample: {sample[:200]}")

            global_step += 1
        print(f"--- epoch {epoch+1} done ---")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved to {out}")
