"""Download Oscar Wilde works from Project Gutenberg and concatenate to wilde.txt.

Usage:
    python download_wilde.py
"""
from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

WORKS = [
    ("The Importance of Being Earnest", "https://www.gutenberg.org/files/844/844-0.txt"),
    ("The Picture of Dorian Gray",       "https://www.gutenberg.org/files/174/174-0.txt"),
    ("An Ideal Husband",                 "https://www.gutenberg.org/files/885/885-0.txt"),
    ("Lady Windermere's Fan",            "https://www.gutenberg.org/files/790/790-0.txt"),
    ("A Woman of No Importance",         "https://www.gutenberg.org/files/854/854-0.txt"),
]

START_MARK = "*** START OF"
END_MARK = "*** END OF"
EOT = "<|endoftext|>"


def fetch(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (gpt2-from-scratch download script)"}
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunks = []
        with tqdm(total=total, unit="B", unit_scale=True, desc=url.rsplit("/", 1)[-1]) as bar:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    chunks.append(chunk)
                    bar.update(len(chunk))
    raw = b"".join(chunks)
    # Project Gutenberg files are UTF-8 with BOM
    return raw.decode("utf-8-sig", errors="replace")


def strip_gutenberg(text: str) -> str:
    """Strip Project Gutenberg header/footer using the *** START/END markers."""
    lines = text.splitlines()
    start_idx = 0
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if START_MARK in line:
            start_idx = i + 1
            break
    for i in range(start_idx, len(lines)):
        if END_MARK in lines[i]:
            end_idx = i
            break
    body = "\n".join(lines[start_idx:end_idx]).strip()
    return body


def main() -> None:
    out_path = Path("wilde.txt")
    pieces: list[str] = []
    for title, url in WORKS:
        print(f"-- {title}")
        raw = fetch(url)
        body = strip_gutenberg(raw)
        print(f"   cleaned length: {len(body):,} chars")
        pieces.append(body)

    combined = f"\n\n{EOT}\n\n".join(pieces)
    out_path.write_text(combined, encoding="utf-8")
    print(f"\nwrote {out_path} ({len(combined):,} chars, {len(WORKS)} works)")


if __name__ == "__main__":
    main()
