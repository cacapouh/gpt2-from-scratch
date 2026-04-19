# ファインチューニング

ソース: [../main.py](../main.py) の `cmd_finetune`、および [../download_wilde.py](../download_wilde.py)

## ここで言う「ファインチューニング」の意味

新しいコーパス上で言語モデルの事前学習を **継続する** こと。損失は同じ
（次トークンのクロスエントロピー）、アーキテクチャも同じ、学習率だけを
劇的に小さくします。モデルは OpenAI 事前学習で得た一般的な英語能力を保ったまま、
分布を新しいテキストに寄せていきます。

## フロー

```mermaid
flowchart TD
    CLI[python main.py finetune --data wilde.txt] --> LOAD[build_openai_gpt gpt2]
    LOAD -->|未取得なら DL| HF[(HF safetensors)]
    LOAD --> M[GPTModel drop_rate=0, qkv_bias=True]
    M --> FIX[Dropout.p=0.1<br/>cfg drop_rate=0.1<br/>model.train]
    FIX --> SPLIT[テキストを 9:1 に分割]
    SPLIT --> DL1[train DataLoader]
    SPLIT --> DL2[val DataLoader]
    DL1 --> TL[train_model]
    DL2 --> TL
    M --> TL
    OPT[AdamW lr=1e-5 wd=0.1] --> TL
    TL --> CKPT[checkpoints/wilde.pt<br/>state_dict + cfg]
    CKPT --> HINT[generate コマンドを出力]
```

## なぜ 40 倍小さい学習率か

事前学習済み重みはすでに強い最小値の近くにいます。`4e-4` のような新規学習向きの
ステップ幅ではそこを吹き飛ばして、GPT-2 が知っていることを上書きしてしまいます。
`1e-5` は GPT-2 スケールのファインチューニングでよく使われるデフォルトです。

## dropout の再有効化

`build_openai_gpt()` は推論が主目的なので `drop_rate=0.0` に設定します。
ファインチューニング前に dropout を有効化し直します:

```python
for m in model.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.1
model.cfg["drop_rate"] = 0.1
model.train()
```

`model.cfg["drop_rate"]` を更新するのは重要 ―― checkpoint に保存されるからです。
次の `generate` 呼び出しは `cfg` からモデルを再構築し、そこで `.eval()` されるため
推論時に dropout は発火しません。つまり保存した値は **継続学習時** だけ意味を持ちます。

## checkpoint の互換性

[../train.py](../train.py) はモデル構築に使った `cfg` をそのまま保存するため、
`cmd_generate` は完全に同じアーキテクチャを再構築できます:

```python
ckpt = torch.load(path, weights_only=False)
cfg  = ckpt["config"]           # qkv_bias=True も含む
model = GPTModel(cfg)           # アーキテクチャ一致
model.load_state_dict(ckpt["model_state_dict"])
```

これが `python main.py generate --weights checkpoints/wilde.pt` がそのまま動く理由です。

## 実例: Oscar Wilde

`download_wilde.py` は Project Gutenberg から Wilde 5 作品を取得し、
冒頭／末尾のボイラープレートを除いて `<|endoftext|>` で連結します:

```
The Importance of Being Earnest
The Picture of Dorian Gray
An Ideal Husband
Lady Windermere's Fan
A Woman of No Importance
```

できあがりの `wilde.txt` は約 970 KB。

RTX 5070 で:

- 1 エポック、`batch_size=4`、`max_length=256`、学習バッチ 255。
- 実時間 約 34 秒。
- train loss 3.67 → 2.69、val loss 3.48 → 2.69。

1 エポック後にプロンプト `"Marriage is"` で生成したサンプル:

> Marriage is a manly act. And if it is not allowed to be, it is not allowed to be at all. It is no longer acceptable, and it is no longer safe either for those who love me, or for those who do not. It is an obligation as much as a duty; and

Wilde らしい逆説と韻律が既に出ていて、事前学習の文法もそのまま保たれています。

## 触る価値のあるノブ

- `--epochs 3` が既定。このコーパスでは train loss は下がり続けますが val loss は 2 エポック目で頭打ち ―― 典型的なファインチューニングの過学習カーブ。
- `--lr 1e-5` は保守的。`5e-5` まで上げる実践者もいますが、上げすぎると「忘却」のリスク。
- `--max-length` は 1024 まで可（事前学習モデルが `context_length=1024` だから）。長くすると VRAM を急激に消費します。
