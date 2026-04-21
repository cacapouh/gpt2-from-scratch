"""rinna/japanese-gpt2-medium を明治文学でファインチューニング。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, T5Tokenizer, get_cosine_schedule_with_warmup

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
    stride: int | None = None,
    lr: float = 5e-5,
    weight_decay: float = 0.1,
    eval_freq: int = 50,
    sample_every: int = 200,
    sample_prompt: str = "文明の進歩は、",
    grad_accum_steps: int = 1,
    warmup_ratio: float = 0.05,
    max_grad_norm: float = 1.0,
    bf16: bool = False,
    label_smoothing: float = 0.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = _load_tokenizer(base_model)
    print(f"loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.to(device)
    if bf16 and device.type == "cuda":
        model.to(dtype=torch.bfloat16)
        print("bf16: enabled (model cast to bfloat16)")
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    text = Path(data_path).read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} chars")
    dataset = MeijiDataset(text, tokenizer, max_length=max_length, stride=stride)
    print(f"windows: {len(dataset)} (max_length={max_length}, stride={stride or max_length})")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"batches/epoch: {len(loader)}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_optim_steps = (len(loader) // grad_accum_steps) * epochs
    warmup_steps = max(1, int(total_optim_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps
    )
    print(f"lr schedule: cosine, warmup {warmup_steps} / total {total_optim_steps} optim steps")

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"label smoothing: {label_smoothing}")

    global_step = 0
    ema_loss = None  # 指数移動平均（smoothing 0.98）
    for epoch in range(epochs):
        optimizer.zero_grad()
        for step, (input_ids, target_ids) in enumerate(loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # label_smoothing を使うため labels を渡さず自前 CE。既に shift 済み target 。
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (B, T, V)
            loss_raw = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss = loss_raw / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            raw = loss_raw.item()
            ema_loss = raw if ema_loss is None else 0.98 * ema_loss + 0.02 * raw
            if global_step % eval_freq == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"ep {epoch+1} step {global_step} | loss {raw:.3f} "
                    f"(ema {ema_loss:.3f}) lr {cur_lr:.2e}"
                )

            if global_step > 0 and global_step % sample_every == 0:
                sample = _sample(model, tokenizer, sample_prompt, device)
                # cp932 \u7b49\u306b encode \u3067\u304d\u306a\u3044\u7d50\u5408\u6587\u5b57\u3067\u843d\u3061\u306a\u3044\u3088\u3046\u30d5\u30a9\u30fc\u30eb\u30d0\u30c3\u30af
                try:
                    print(f"  sample: {sample[:200]}")
                except UnicodeEncodeError:
                    enc = (sys.stdout.encoding or "utf-8")
                    safe = sample[:200].encode(enc, errors="replace").decode(enc, errors="replace")
                    print(f"  sample: {safe}")

            global_step += 1
        print(f"--- epoch {epoch+1} done ---")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved to {out}")
