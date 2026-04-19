"""青空文庫から指定作品をダウンロードして Shift-JIS → UTF-8 変換する。

戦略:
  1. 各作家の人物ページ (person{id}.html) を取得
  2. 作品一覧テーブルから、タイトルがキーワードに一致する行を拾う
  3. その作品ページを辿り、ルビ付きテキストの zip リンクを取得
  4. zip をダウンロードして中の *.txt を Shift-JIS で読んで UTF-8 で保存
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

AOZORA_BASE = "https://www.aozora.gr.jp/"
HEADERS = {"User-Agent": "Mozilla/5.0 (aozora-meiji-finetune bot)"}


# 人物番号 → ダウンロードしたい作品タイトルの部分一致キーワード
# （青空文庫の実際のタイトル表記に合わせている。見つからない候補はスキップされる）
TARGETS: dict[int, list[str]] = {
    296:  ["学問のすすめ", "福翁自伝", "修身要領", "学問の独立"],   # 福沢諭吉
    129:  ["舞姫", "青年", "高瀬舟"],                                # 森鷗外
    148:  ["草枕", "虞美人草", "吾輩は猫である"],                    # 夏目漱石
    64:   ["たけくらべ", "にごりえ", "十三夜", "大つごもり"],       # 樋口一葉
    51:   ["五重塔", "運命", "風流仏"],                              # 幸田露伴
}

OUT_DIR = Path(__file__).parent / "data" / "raw"


def _get(url: str, *, encoding: str | None = None) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if encoding:
        r.encoding = encoding
    return r


def _safe_filename(title: str) -> str:
    # Windows で使えない文字を置換
    return re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("_")


def list_person_cards(person_id: int) -> list[tuple[str, str]]:
    """人物ページから (作品タイトル, カードURL) のリストを返す。"""
    url = f"{AOZORA_BASE}index_pages/person{person_id}.html"
    # 人物ページは UTF-8（meta charset=utf-8）
    html = _get(url, encoding="utf-8").text
    soup = BeautifulSoup(html, "html.parser")

    results: list[tuple[str, str]] = []
    # 「公開中の作品」以下にある ol > li > a[href*="cards/"]
    for a in soup.select('a[href*="/cards/"]'):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or not title:
            continue
        card_url = urljoin(url, href)
        # ファイル一覧ページ (cards/NNNNNN/cardNNNNN.html) だけを対象にする
        if re.search(r"/cards/\d+/card\d+\.html$", card_url):
            results.append((title, card_url))
    return results


def find_zip_url(card_url: str) -> str | None:
    """作品カードページからルビ付きテキストの zip URL を返す。"""
    # card ページは Shift-JIS のことが多い。meta を見て判断するのが確実だが、
    # zip リンク抽出だけなら encoding に依らない（href は ASCII）。utf-8 で十分。
    html = _get(card_url, encoding="utf-8").text
    soup = BeautifulSoup(html, "html.parser")
    # ダウンロードテーブル: <table class="download"> 内の .zip リンク
    for a in soup.select("a[href$='.zip']"):
        href = a.get("href", "")
        # ルビ付きを優先（_ruby_ を含むもの）。無ければ最初の zip
        if "ruby" in href:
            return urljoin(card_url, href)
    # fallback: 最初に見つかった zip
    first = soup.select_one("a[href$='.zip']")
    if first:
        return urljoin(card_url, first.get("href", ""))
    return None


def extract_text_from_zip(data: bytes) -> str:
    """zip バイト列から Shift-JIS の .txt を読み出して UTF-8 文字列にする。"""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_names:
            raise RuntimeError("zip に .txt が無い")
        with zf.open(txt_names[0]) as f:
            raw = f.read()
    # 青空文庫の txt は Shift-JIS (cp932)
    try:
        return raw.decode("shift_jis")
    except UnicodeDecodeError:
        return raw.decode("cp932", errors="replace")


def download_one(person_id: int, keyword: str, out_dir: Path) -> Path | None:
    """人物ページから keyword に部分一致する作品を 1 本ダウンロード。"""
    try:
        cards = list_person_cards(person_id)
    except Exception as e:
        print(f"  [skip] person{person_id}: {e}")
        return None

    match = next(((t, u) for t, u in cards if keyword in t), None)
    if match is None:
        titles = [t for t, _ in cards]
        hint = ", ".join(titles[:5])
        print(f"  [miss] person{person_id}: '{keyword}' が見つからない（候補例: {hint} ...）")
        return None

    title, card_url = match
    out_path = out_dir / f"{person_id}_{_safe_filename(keyword)}.txt"
    if out_path.exists():
        print(f"  [cache] {out_path.name}")
        return out_path

    try:
        zip_url = find_zip_url(card_url)
        if not zip_url:
            print(f"  [skip] {title}: zip リンクが無い")
            return None
        r = _get(zip_url)
        text = extract_text_from_zip(r.content)
    except Exception as e:
        print(f"  [fail] {title}: {e}")
        return None

    out_path.write_text(text, encoding="utf-8")
    print(f"  [ok] {out_path.name} ({len(text):,} chars)")
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for person_id, keywords in TARGETS.items():
        print(f"\n=== person {person_id} ===")
        for kw in keywords:
            download_one(person_id, kw, OUT_DIR)
            time.sleep(0.5)  # 負荷をかけない


if __name__ == "__main__":
    main()
