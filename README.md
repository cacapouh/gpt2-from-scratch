# GPT-2 from Scratch

Sebastian Raschka 著『Build a Large Language Model (from Scratch)』に沿って、
GPT-2 small (124M) を PyTorch でゼロから実装した最小構成のプロジェクトです。

**ドキュメント**: [docs/README.md](docs/README.md) に DeepWiki 風のガイド
（アーキテクチャ、モデル内部、データパイプライン、学習／生成ループ、
重みロード、ファインチューニング、CLI リファレンス、用語集）があります。

## 構成
- [config.py](config.py) — モデルサイズのプリセット
- [data.py](data.py) — スライディングウィンドウの Dataset と DataLoader（tiktoken GPT-2 BPE）
- [model.py](model.py) — `MultiHeadAttention`、`LayerNorm`、`GELU`、`FeedForward`、`TransformerBlock`、`GPTModel`
- [generate.py](generate.py) — 自己回帰サンプリング（temperature + top-k）
- [train.py](train.py) — 定期 eval ／サンプル生成付きの学習ループ
- [load_gpt2.py](load_gpt2.py) — HuggingFace の GPT-2 safetensors をダウンロードして本実装に読み込む
- [main.py](main.py) — CLI（`train` / `finetune` / `generate`）
- [utils.py](utils.py) — デバイスヘルパ
- [download_wilde.py](download_wilde.py) — Project Gutenberg から Oscar Wilde 5 作品を取得して連結

## インストール

Python 3.12 を使ってください（PyTorch 2.11 の CUDA ホイールは cp310–cp313 にはありますが、
cp314 にはまだありません）。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# CUDA 12.8 ビルド（RTX 50 シリーズ / Blackwell sm_120 向け）:
pip install --index-url https://download.pytorch.org/whl/cu128 torch

# CPU のみの場合:
# pip install torch

pip install -r requirements.txt
```

GPU が認識されているかを確認：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 使い方

**短いテキストで学習（例: Edith Wharton 『The Verdict』）:**

```powershell
python main.py train --data the-verdict.txt --epochs 10
```

**OpenAI の事前学習済み重みで生成（初回はダウンロードが走ります）:**

```powershell
python main.py generate --weights gpt2 --prompt "The meaning of life is"
```

**自分で学習したチェックポイントで生成:**

```powershell
python main.py generate --weights checkpoints/model.pt --prompt "Hello"
```

`--temperature` と `--top-k` でサンプリング挙動を調整できます。

## ハードウェア

シングル RTX 5070（12 GB）を想定。既定値は `max_length=256`、`batch_size=8`。
GPT-2 small (~124M) は余裕で載るので、VRAM に余裕があればバッチを増やしてください。

## 関連サブプロジェクト

- [aozora/](aozora/README.md) — `rinna/japanese-gpt2-medium` (336M) を
  青空文庫の明治文学（福沢諭吉・森鷗外・夏目漱石・樋口一葉・幸田露伴）で
  ファインチューンし、現代的なプロンプトに対して**明治論説文調**の出力を得る実験。
  HuggingFace Transformers 上で PyTorch 生ループ学習を行う最小構成。

