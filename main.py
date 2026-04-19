"""Command-line entry point: train or generate with GPT-2."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import GPT_CONFIG_124M
from data import create_dataloader, load_text_file
from generate import generate
from model import GPTModel
from train import train_model
from utils import device_info, get_device


def cmd_train(args: argparse.Namespace) -> None:
    device = get_device()
    print(device_info())

    text = load_text_file(args.data)
    split_idx = int(len(text) * 0.9)
    train_text, val_text = text[:split_idx], text[split_idx:]

    cfg = dict(GPT_CONFIG_124M)
    cfg["context_length"] = args.max_length  # shrink positional embeddings to save memory

    train_loader = create_dataloader(
        train_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.max_length,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.max_length,
        shuffle=False,
        drop_last=False,
    )

    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    torch.manual_seed(123)
    model = GPTModel(cfg).to(device)
    print(f"model params: {model.num_parameters()/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=5,
        sample_prompt=args.prompt,
        sample_every=args.sample_every,
        checkpoint_path=args.checkpoint,
    )


def cmd_finetune(args: argparse.Namespace) -> None:
    from load_gpt2 import build_openai_gpt

    device = get_device()
    print(device_info())

    text = load_text_file(args.data)
    split_idx = int(len(text) * 0.9)
    train_text, val_text = text[:split_idx], text[split_idx:]

    print(f"loading pretrained base model: {args.base_model}")
    model = build_openai_gpt(args.base_model, models_dir=args.models_dir)
    # build_openai_gpt() sets drop_rate=0.0 for inference; re-enable dropout for fine-tuning.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.1
    model.cfg["drop_rate"] = 0.1
    model.to(device)
    model.train()
    print(f"model params: {model.num_parameters()/1e6:.1f}M")

    train_loader = create_dataloader(
        train_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.max_length,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.max_length,
        shuffle=False,
        drop_last=False,
    )
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=5,
        sample_prompt=args.prompt,
        sample_every=args.sample_every,
        checkpoint_path=args.checkpoint,
    )

    print("\nFine-tuning complete. Generate with:")
    print(
        f'  python main.py generate --weights {args.checkpoint} '
        f'--prompt "{args.prompt}" --temperature 0.8 --top-k 50'
    )


def cmd_generate(args: argparse.Namespace) -> None:
    device = get_device()
    print(device_info())

    weights = args.weights
    if weights in {"gpt2", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl",
                   "124M", "355M", "774M", "1558M"}:
        from load_gpt2 import build_openai_gpt

        model = build_openai_gpt(weights, models_dir=args.models_dir)
    else:
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", dict(GPT_CONFIG_124M))
        model = GPTModel(cfg)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    out = generate(
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print("\n" + "=" * 60)
    print(out)
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPT-2 from scratch")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train from a text file")
    pt.add_argument("--data", required=True, help="Path to a plain text file")
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--batch-size", type=int, default=8)
    pt.add_argument("--max-length", type=int, default=256)
    pt.add_argument("--lr", type=float, default=4e-4)
    pt.add_argument("--eval-freq", type=int, default=20)
    pt.add_argument("--sample-every", type=int, default=100)
    pt.add_argument("--prompt", default="Every effort moves you")
    pt.add_argument("--checkpoint", default="checkpoints/model.pt")
    pt.set_defaults(func=cmd_train)

    pf = sub.add_parser("finetune", help="Fine-tune a pretrained GPT-2 on a text file")
    pf.add_argument("--data", required=True, help="Path to a plain text file")
    pf.add_argument("--base-model", default="gpt2",
                    help="Pretrained model size (gpt2/gpt2-medium/gpt2-large/gpt2-xl)")
    pf.add_argument("--epochs", type=int, default=3)
    pf.add_argument("--batch-size", type=int, default=4)
    pf.add_argument("--max-length", type=int, default=256)
    pf.add_argument("--lr", type=float, default=1e-5)
    pf.add_argument("--eval-freq", type=int, default=20)
    pf.add_argument("--sample-every", type=int, default=50)
    pf.add_argument("--prompt", default="Every effort moves you")
    pf.add_argument("--checkpoint", default="checkpoints/finetuned.pt")
    pf.add_argument("--models-dir", default="gpt2_weights")
    pf.set_defaults(func=cmd_finetune)

    pg = sub.add_parser("generate", help="Generate text with pretrained or saved weights")
    pg.add_argument(
        "--weights",
        required=True,
        help="'gpt2'/'gpt2-medium'/... to use OpenAI weights, or path to a .pt checkpoint",
    )
    pg.add_argument("--prompt", required=True)
    pg.add_argument("--max-new-tokens", type=int, default=50)
    pg.add_argument("--temperature", type=float, default=1.0)
    pg.add_argument("--top-k", type=int, default=50)
    pg.add_argument("--models-dir", default="gpt2_weights")
    pg.set_defaults(func=cmd_generate)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
