# CLI リファレンス

エントリ: `python main.py <サブコマンド> [フラグ]` ―― 詳細は [../main.py](../main.py)。

サブコマンドは 3 つ: [train](#train)、[finetune](#finetune)、[generate](#generate)。

## train

GPT-2 をランダム初期化から学習する。

```powershell
python main.py train --data the-verdict.txt --epochs 10
```

| フラグ | 既定値 | 意味 |
|---|---|---|
| `--data` | **必須** | UTF-8 テキストファイルへのパス |
| `--epochs` | `10` | 学習エポック数 |
| `--batch-size` | `8` | ミニバッチサイズ |
| `--max-length` | `256` | シーケンス長 *および* 位置埋め込みサイズ |
| `--lr` | `4e-4` | AdamW の学習率 |
| `--eval-freq` | `20` | N ステップごとに評価 |
| `--sample-every` | `100` | N ステップごとにサンプル生成 |
| `--prompt` | `"Every effort moves you"` | サンプリング用プロンプト |
| `--checkpoint` | `checkpoints/model.pt` | 保存先 |

## finetune

OpenAI 事前学習済み重みからウォームスタート。

```powershell
python main.py finetune --data wilde.txt --epochs 3 --prompt "Marriage is"
```

| フラグ | 既定値 | 意味 |
|---|---|---|
| `--data` | **必須** | UTF-8 テキストファイルへのパス |
| `--base-model` | `gpt2` | `gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl` |
| `--epochs` | `3` | |
| `--batch-size` | `4` | |
| `--max-length` | `256` | |
| `--lr` | `1e-5` | `train` より 40 倍低いのは意図的 |
| `--eval-freq` | `20` | |
| `--sample-every` | `50` | |
| `--prompt` | `"Marriage is"` | |
| `--checkpoint` | `checkpoints/wilde.pt` | |
| `--models-dir` | `gpt2_weights` | ダウンロード safetensors のキャッシュ |

学習後、対応する `generate` コマンドを CLI が出力する。

## generate

事前学習済み重み、または自作の checkpoint からテキスト生成。

```powershell
# OpenAI 事前学習済み
python main.py generate --weights gpt2 --prompt "The meaning of life is"

# ローカル checkpoint
python main.py generate --weights checkpoints/wilde.pt --prompt "Marriage is"
```

| フラグ | 既定値 | 意味 |
|---|---|---|
| `--weights` | **必須** | `gpt2`／`gpt2-medium`／… **または** `.pt` checkpoint へのパス |
| `--prompt` | **必須** | 条件となるテキスト |
| `--max-new-tokens` | `50` | 生成するトークン数 |
| `--temperature` | `1.0` | サンプリング温度。`0` で greedy |
| `--top-k` | `50` | top-k フィルタ。`0` で無効 |
| `--models-dir` | `gpt2_weights` | HF 重みのキャッシュディレクトリ |

## 終了挙動

3 つのサブコマンドは成功時 exit code `0` を返し、失敗時は Python 例外
（非ゼロ）を伝播します。上にかぶせたエラー処理はありません。

## 実行環境

- Python 3.12 の venv（インストール手順は [../README.md](../README.md) 参照）。
- RTX 50 系向けに PyTorch 2.11 + CUDA 12.8。CPU フォールバックは [../utils.py](../utils.py) の `get_device()` が自動で行います。
- 初回の `generate --weights gpt2` で約 548 MB を `gpt2_weights/gpt2/model.safetensors` にダウンロードしてキャッシュします。
