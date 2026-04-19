"""Training loop for GPT-2."""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from generate import generate


def calc_loss_batch(input_ids, target_ids, model, device):
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), target_ids.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches: int | None = None):
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= num_batches:
                break
            total_loss += calc_loss_batch(x, y, model, device).item()
    model.train()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter: int):
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter) if val_loader is not None else float("nan")
    return train_loss, val_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs: int,
    eval_freq: int = 50,
    eval_iter: int = 5,
    sample_prompt: str = "Every effort moves you",
    sample_every: int | None = None,
    checkpoint_path: str | Path | None = None,
):
    """Basic GPT-2 training loop. Prints losses and periodic samples."""
    if sample_every is None:
        sample_every = eval_freq * 4

    model.to(device)
    model.train()

    tokens_seen = 0
    global_step = -1
    history = {"train": [], "val": [], "step": []}

    for epoch in range(num_epochs):
        epoch_start = time.time()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += x.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                history["train"].append(train_loss)
                history["val"].append(val_loss)
                history["step"].append(global_step)
                print(
                    f"ep {epoch+1} step {global_step:06d} | "
                    f"train {train_loss:.3f} | val {val_loss:.3f} | tokens {tokens_seen}"
                )

            if sample_every and global_step % sample_every == 0 and global_step > 0:
                sample = generate(
                    model, sample_prompt, max_new_tokens=30, temperature=1.0, top_k=25,
                    device=device,
                )
                print("  sample:", sample.replace("\n", " "))

        print(f"epoch {epoch+1} done in {time.time() - epoch_start:.1f}s")

        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model.cfg,
                },
                checkpoint_path,
            )
            print(f"  checkpoint saved -> {checkpoint_path}")

    return history
