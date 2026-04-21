"""aozora プロジェクトの CLI エントリポイント。"""
from __future__ import annotations

import argparse


def cmd_download(_: argparse.Namespace) -> None:
    import download
    download.main()


def cmd_clean(_: argparse.Namespace) -> None:
    import clean
    clean.main()


def cmd_prepare(args: argparse.Namespace) -> None:
    import prepare_data
    prepare_data.main(total_chars=args.total_chars, seed=args.seed)


def cmd_data(args: argparse.Namespace) -> None:
    """download → clean → prepare を一括実行（コーパス再現用）。"""
    import download
    import clean
    import prepare_data
    download.main()
    clean.main()
    prepare_data.main(total_chars=args.total_chars, seed=args.seed)


def cmd_finetune(args: argparse.Namespace) -> None:
    from finetune import finetune
    finetune(
        data_path=args.data,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        lr=args.lr,
        eval_freq=args.eval_freq,
        sample_every=args.sample_every,
        sample_prompt=args.prompt,
        grad_accum_steps=args.grad_accum_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        label_smoothing=args.label_smoothing,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    from generate import generate
    out = generate(
        weights=args.weights,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print("\n" + "=" * 60)
    print(out)
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="aozora meiji finetune")
    sub = p.add_subparsers(dest="command", required=True)

    sd = sub.add_parser("download", help="青空文庫から作品をダウンロード")
    sd.set_defaults(func=cmd_download)

    sc = sub.add_parser("clean", help="青空文庫マークアップを除去")
    sc.set_defaults(func=cmd_clean)

    sp = sub.add_parser("prepare", help="配合比率で combined.txt を作成")
    sp.add_argument("--total-chars", type=int, default=7_000_000)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_prepare)

    sa = sub.add_parser("data", help="download + clean + prepare を一括実行")
    sa.add_argument("--total-chars", type=int, default=7_000_000)
    sa.add_argument("--seed", type=int, default=42)
    sa.set_defaults(func=cmd_data)

    sf = sub.add_parser("finetune", help="ファインチューニング実行")
    sf.add_argument("--data", default="data/combined.txt")
    sf.add_argument("--base-model", default="rinna/japanese-gpt2-medium")
    sf.add_argument("--output-dir", default="checkpoints/meiji_gpt2")
    sf.add_argument("--epochs", type=int, default=3)
    sf.add_argument("--batch-size", type=int, default=2)
    sf.add_argument("--max-length", type=int, default=512)
    sf.add_argument("--stride", type=int, default=None,
                    help="スライディングウィンドウのストライド。デフォルトは max_length（重なり無し）。")
    sf.add_argument("--lr", type=float, default=5e-5)
    sf.add_argument("--eval-freq", type=int, default=50)
    sf.add_argument("--sample-every", type=int, default=200)
    sf.add_argument("--prompt", default="文明の進歩は、")
    sf.add_argument("--grad-accum-steps", type=int, default=1)
    sf.add_argument("--warmup-ratio", type=float, default=0.05)
    sf.add_argument("--max-grad-norm", type=float, default=1.0)
    sf.add_argument("--bf16", action="store_true", help="bfloat16 で学習（RTX 30/40/50 系推奨）")
    sf.add_argument("--label-smoothing", type=float, default=0.0)
    sf.set_defaults(func=cmd_finetune)

    sg = sub.add_parser("generate", help="テキスト生成")
    sg.add_argument("--weights", "--model", dest="weights", required=True,
                    help="'rinna/japanese-gpt2-medium' または checkpoints/xxx")
    sg.add_argument("--prompt", required=True)
    sg.add_argument("--max-new-tokens", type=int, default=100)
    sg.add_argument("--temperature", type=float, default=0.9)
    sg.add_argument("--top-k", type=int, default=50)
    sg.add_argument("--top-p", type=float, default=1.0)
    sg.add_argument("--repetition-penalty", type=float, default=1.1)
    sg.add_argument("--no-repeat-ngram-size", type=int, default=0)
    sg.set_defaults(func=cmd_generate)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
