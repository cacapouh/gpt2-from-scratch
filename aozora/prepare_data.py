"""作家ごとの配合比率で data/cleaned/*.txt を 1 本に連結する。

配合比率（文字数ベース）:
  福沢諭吉 (296):  25%  論説の要
  森鷗外   (129):  18%  明治口語小説・歴史小説
  夏目漱石 (148):  13%
  二葉亭四迷 (6):    8%  言文一致小説の源流
  正岡子規 (305):  8%  歌論・病牀随筆
  国木田独歩 (38): 8%  明治浪漫短編
  内村鑑三 (34):   6%  キリスト教論説
  北村透谷 (157):  5%  明治文学評論
  樋口一葉 (64):   4%
  幸徳秋水 (261):  3%  社会羻電・書簡
  徳冨蘆花 (280):  3%
  幸田露伴 (51):   2%
  岡倉天心 (238):  2%  茶の本
  中江兆民 (1212): 1%  朝野論説

各作家のテキスト先頭に『【作家名】』タグを付けて文体制御を育てる。
"""
from __future__ import annotations

import random
from pathlib import Path

CLEANED_DIR = Path(__file__).parent / "data" / "cleaned"
OUT_PATH = Path(__file__).parent / "data" / "combined.txt"

# 人物番号 -> 目標配合比率
RATIOS: dict[int, float] = {
    296:  0.25,
    129:  0.18,
    148:  0.13,
    6:    0.08,
    305:  0.08,
    38:   0.08,
    34:   0.06,
    157:  0.05,
    64:   0.04,
    261:  0.03,
    280:  0.03,
    51:   0.02,
    238:  0.02,
    1212: 0.01,
}

# 人物番号 -> スタイルタグに使う作家名
AUTHOR_TAGS: dict[int, str] = {
    296:  "福沢諭吉",
    129:  "森鷗外",
    148:  "夏目漱石",
    6:    "二葉亭四迷",
    305:  "正岡子規",
    38:   "国木田独歩",
    34:   "内村鑑三",
    157:  "北村透谷",
    64:   "樋口一葉",
    261:  "幸徳秋水",
    280:  "徳冨蘆花",
    51:   "幸田露伴",
    238:  "岡倉天心",
    1212: "中江兆民",
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
        body = p.read_text(encoding="utf-8")
        # 作品先頭に『【作家名】』タグを付けてスタイルと紐付ける
        tag = AUTHOR_TAGS.get(pid)
        if tag:
            body = f"【{tag}】\n\n{body}"
        by_author.setdefault(pid, []).append(body)
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


def main(total_chars: int = 5_000_000, seed: int = 42) -> None:
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
