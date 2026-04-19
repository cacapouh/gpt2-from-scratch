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
- [the-verdict.txt](the-verdict.txt) — スクラッチ学習の動作確認用の短い英語テキスト

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

## ファインチューン

`train` がスクラッチ学習なのに対し、`finetune` は **OpenAI の事前学習済み GPT-2**
をベースにして任意の英語テキストに適応させるモードです。[load_gpt2.py](load_gpt2.py)
が HuggingFace から safetensors を落として本実装の `GPTModel` にマッピングし、
そこから学習ループを再開します。

```powershell
# 任意の英語テキスト（UTF-8）でファインチューン
# RTX 5070 12GB: batch_size=4, max_length=256, lr=1e-5 で数分
python main.py finetune --data your_corpus.txt --epochs 3 --checkpoint checkpoints/mine.pt

# 生成
python main.py generate --weights checkpoints/mine.pt --prompt "Once upon a time" --temperature 0.8 --top-k 50
```

### `train` と `finetune` の違い

| 項目 | `train` | `finetune` |
| --- | --- | --- |
| 初期重み | ランダム | OpenAI 事前学習済み（`gpt2` / `gpt2-medium` 等） |
| 学習率 | `4e-4`（大きめ） | `1e-5`（小さめ、壊さないため） |
| 既定 epochs | 10 | 3 |
| Dropout | 0.1 | 推論用の 0.0 から 0.1 に再有効化 |
| 典型用途 | アーキテクチャ検証 | 文体・ドメインへの適応 |

### 主要オプション

```text
python main.py finetune
    --data           学習テキスト（必須、UTF-8）
    --base-model     gpt2 / gpt2-medium / gpt2-large / gpt2-xl （既定: gpt2）
    --epochs         エポック数 （既定: 3）
    --batch-size     バッチ （既定: 4）
    --max-length     文脈長 （既定: 256）
    --lr             学習率 （既定: 1e-5）
    --checkpoint     保存先 （既定: checkpoints/finetuned.pt）
    --prompt         学習中に定期サンプリングするプロンプト
    --models-dir     事前学習済み重みのキャッシュ先 （既定: gpt2_weights/）
```

### 日本語でファインチューンしたい場合

OpenAI 公開の GPT-2（本実装のベース）は英語コーパスで事前学習されており、
GPT-2 BPE は日本語を byte 単位で細切れに扱うため、**日本語テキストを
このルートの `finetune` に渡しても意味のある結果は得られません**。

日本語は専用のサブプロジェクト [aozora/](aozora/README.md) を参照してください。
青空文庫の明治文学コーパスと日本語事前学習モデル
`rinna/japanese-gpt2-medium` を使い、HuggingFace Transformers 上で
ファインチューンする一式が用意されています。

## ハードウェア

シングル RTX 5070（12 GB）を想定。既定値は `max_length=256`、`batch_size=8`。
GPT-2 small (~124M) は余裕で載るので、VRAM に余裕があればバッチを増やしてください。

## 関連サブプロジェクト

- [aozora/](aozora/README.md) — `rinna/japanese-gpt2-medium` (336M) を
  青空文庫の明治文学（福沢諭吉・森鷗外・夏目漱石・樋口一葉・幸田露伴）で
  ファインチューンし、現代的なプロンプトに対して**明治論説文調**の出力を得る実験。
  HuggingFace Transformers 上で PyTorch 生ループ学習を行う最小構成。

