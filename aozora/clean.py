"""青空文庫マークアップを除去して本文だけ残す。"""
from __future__ import annotations

import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUT_DIR = Path(__file__).parent / "data" / "cleaned"


def clean_aozora(text: str) -> str:
    # 1) 冒頭の区切り線 (------------) で囲まれた凡例ブロックを除去
    #    青空文庫は最初の区切り線と 2 つ目の区切り線の間に凡例を置く
    text = re.sub(r"-{10,}.*?-{10,}\s*", "", text, count=1, flags=re.DOTALL)

    # 2) 底本情報以降を削除（"底本：" から末尾まで）
    text = re.sub(r"底本[:：].*", "", text, flags=re.DOTALL)

    # 3) 基準点マーカー ｜ を削除
    text = text.replace("｜", "")

    # 4) ルビ《...》を削除（漢字《かんじ》→ 漢字）
    text = re.sub(r"《[^》]*》", "", text)

    # 5) 注記 ［＃...］を削除
    text = re.sub(r"［＃[^］]*］", "", text)

    # 6) ※［＃...］ のような派生表記も除去
    text = re.sub(r"※[［\[][^］\]]*[］\]]", "", text)

    # 7) 連続空行を 1 つの空行に正規化
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8) 前後の空白を整える
    text = text.strip()

    return text


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(RAW_DIR.glob("*.txt")):
        raw = src.read_text(encoding="utf-8")
        cleaned = clean_aozora(raw)
        dst = OUT_DIR / src.name
        dst.write_text(cleaned, encoding="utf-8")
        print(f"  {src.name}: {len(raw):,} -> {len(cleaned):,} chars")


if __name__ == "__main__":
    main()
