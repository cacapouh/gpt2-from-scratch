"""作家ごとの配合比率で data/cleaned/*.txt を 1 本に連結する。

配合比率（文字数ベース）:
  福沢諭吉 (296):  40%
  森鷗外   (129):  25%
  夏目漱石 (148):  20%
  樋口一葉 (20):   10%
  幸田露伴 (51):    5%
"""
from __future__ import annotations

import random
from pathlib import Path

CLEANED_DIR = Path(__file__).parent / "data" / "cleaned"
OUT_PATH = Path(__file__).parent / "data" / "combined.txt"

# 人物番号 -> 目標配合比率
RATIOS: dict[int, float] = {
    296: 0.40,
    129: 0.25,
    148: 0.20,
    64:  0.10,
    51:  0.05,
}

EOS = "</s>"  # rinna の T5Tokenizer が認識する文末トークン


def author_texts() -> dict[int, str]:
    """人物番号ごとに、その作家の全テキストを連結して返す。"""
    by_author: dict[int, list[str]] = {}
    for p in sorted(CLEANED_DIR.glob("*.txt")):
        # ファイル名は "{person_id}_{title}.txt"
        prefix = p.name.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        pid = int(prefix)
        by_author.setdefault(pid, []).append(p.read_text(encoding="utf-8"))
    return {pid: ("\n\n" + EOS + "\n\n").join(chunks) for pid, chunks in by_author.items()}


def sample_paragraphs(text: str, target_chars: int, rng: random.Random) -> str:
    """段落単位でランダムサンプリングして target_chars に近づける。"""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""
    if sum(len(p) for p in paragraphs) <= target_chars:
        # 全部使っても足りない -> 繰り返しで埋める
        out: list[str] = []
        total = 0
        while total < target_chars:
            shuf = paragraphs[:]
            rng.shuffle(shuf)
            for p in shuf:
                out.append(p)
                total += len(p)
                if total >= target_chars:
                    break
        return "\n\n".join(out)

    # 足りている場合: シャッフルして target に到達するまで詰める
    shuf = paragraphs[:]
    rng.shuffle(shuf)
    out = []
    total = 0
    for p in shuf:
        out.append(p)
        total += len(p)
        if total >= target_chars:
            break
    return "\n\n".join(out)


def main(total_chars: int = 3_000_000, seed: int = 42) -> None:
    rng = random.Random(seed)
    authors = author_texts()
    print(f"loaded authors: {list(authors.keys())}")
    for pid, t in authors.items():
        print(f"  person{pid}: {len(t):,} chars")

    parts: list[str] = []
    for pid, ratio in RATIOS.items():
        if pid not in authors:
            print(f"  [warn] person{pid} の資料が無い -> スキップ")
            continue
        target = int(total_chars * ratio)
        sampled = sample_paragraphs(authors[pid], target, rng)
        print(f"  person{pid}: target={target:,} actual={len(sampled):,}")
        parts.append(sampled)

    # 作家の境目には EOS を挟む
    combined = ("\n\n" + EOS + "\n\n").join(parts)
    OUT_PATH.write_text(combined, encoding="utf-8")
    print(f"\nwrote {OUT_PATH} ({len(combined):,} chars)")


if __name__ == "__main__":
    main()
