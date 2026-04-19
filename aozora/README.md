# 明治文学で日本語 GPT-2 をファインチューニング

`rinna/japanese-gpt2-medium` (336M) を青空文庫の明治文学でファインチューンし、
現代的なプロンプト（例: 「電子計算機とは、」）に対しても
**明治知識人風の論説文**を生成させる実験。

親プロジェクト（[../README.md](../README.md)）は GPT-2 をゼロから実装するのに対し、
こちらは HuggingFace Transformers を使った**実用的なファインチューン最小構成**。

## データとチェックポイントの扱い

このリポジトリには**青空文庫の原文のみ**が含まれ、加工済みコーパスと
学習済み重みは含まれていません。

| パス | git | 備考 |
| --- | --- | --- |
| `data/raw/*.txt` | **含まれる** | 青空文庫の原文そのまま。入力・校正者クレジットと底本情報を保持 |
| `data/cleaned/` | `.gitignore` | `clean.py` が底本・クレジット等の付帯情報を除去してしまうため非公開 |
| `data/combined.txt` | `.gitignore` | 加工後コーパス。同上の理由で非公開 |
| `checkpoints/` | `.gitignore` | 学習済み重み（1.3GB）。自分で `finetune` して生成 |
| `finetune.log` | `.gitignore` | 学習ログ |

### 著作権と再配布について

対象 5 作家（福沢諭吉・樋口一葉・森鷗外・夏目漱石・幸田露伴）はいずれも
没後 50 年以上経過しており、作品自体の著作権は消滅しています。ただし
青空文庫ファイルには**入力・校正ボランティアへのクレジット**と**底本情報**が
含まれており、これらを保持することが青空文庫の
[利用条件](https://www.aozora.gr.jp/guide/kijyunn.html) となっています。

- `data/raw/*.txt` はファイル末尾の「入力：○○／校正：○○／青空文庫作成ファイル：…」
  を**そのまま保持している**ため再配布可能。このリポジトリでも原文を含めて公開しています。
- `clean.py` はこれらの付帯情報を除去するため、その出力（`cleaned/`, `combined.txt`）は
  再配布を避けて `.gitignore` しています。自分で再生成してください。

青空文庫様のご尽力に感謝します。

## セットアップ

```powershell
cd aozora
# torch は親プロジェクト側で既に入っている想定（CUDA 12.8 ビルド、Python 3.12 .venv を共有）
pip install -r requirements.txt
```

`requirements.txt` に `protobuf` を含めている点に注意。`T5Tokenizer`
（SentencePiece 変換）の読み込みで必要。

> **VS Code + Copilot ユーザー向け**: チャットで `/setup-aozora` と入力すると、
> [../.github/prompts/setup-aozora.prompt.md](../.github/prompts/setup-aozora.prompt.md)
> が呼び出され、下記の再現手順をエージェントが順次実行します。

## パイプライン

```powershell
# 一括実行（推奨）: download → clean → prepare を順に走らせる
python main.py data

# または個別に
python main.py download   # 青空文庫から 17 作品をダウンロード（UTF-8 作家ページから zip を辿る）
python main.py clean      # 青空文庫マークアップを除去（ルビ《》、｜、［＃注記］、底本以降、区切り線）
python main.py prepare    # 配合比率どおりに連結して combined.txt を作成
```

**再現の期待値**（2026-04 時点）:

- `data/raw/` に 17 ファイル
- `data/cleaned/` に 17 ファイル
- `data/combined.txt` が約 **3,037,319 字**（±数万字、seed=42 固定）

字数が大きくズレる場合、青空文庫側で底本や表記が変わった可能性があるので
`download.py` の `TARGETS` とログを確認してください。

作家別配合比率（`prepare_data.py` の `RATIOS`、約 3,000,000 字になるよう
先頭から切り出し）:

| 作家 | person_id | 比率 | 作品 |
| --- | --- | --- | --- |
| 福沢諭吉 | 296 | 40% | 学問のすすめ・福翁自伝・修身要領・学問の独立 |
| 森鷗外 | 129 | 25% | 舞姫・青年・高瀬舟 |
| 夏目漱石 | 148 | 20% | 草枕・虞美人草・吾輩は猫である |
| 樋口一葉 | 64  | 10% | たけくらべ・にごりえ・十三夜・大つごもり |
| 幸田露伴 | 51  | 5%  | 五重塔・運命・風流仏 |

```powershell
# 4. ファインチューン（RTX 5070 12GB で 3 epoch ≈ 33 分）
python main.py finetune --epochs 3

# 5. 生成
python main.py generate --weights checkpoints/meiji_gpt2 --prompt "電子計算機とは、"
python main.py generate --weights rinna/japanese-gpt2-medium --prompt "電子計算機とは、"  # 比較用
```

## 学習設定

| 項目 | 値 |
| --- | --- |
| ベースモデル | `rinna/japanese-gpt2-medium` (336M, GPT-2 medium, T5Tokenizer SentencePiece) |
| コーパス | 3,037,319 字の連結テキスト |
| window | 512 token スライディングウィンドウ、3,907 windows |
| batch size | 2（grad_accum_steps=1）→ 1,953 batches / epoch |
| optimizer | AdamW, lr=5e-5, weight_decay=0.1 |
| epochs | 3（全 5,859 steps） |
| device | 単一 RTX 5070、ピーク VRAM 約 8.6 GB / 200W |

損失は 3 エポックでおおよそ 5〜6 → 1.5〜3 付近まで低下（バッチサイズが
小さいためステップごとの揺れは大きい）。

## 実測：ファインチューン前後の比較

素の rinna はブログ／Wikipedia 風の現代口語を生成するのに対し、
ファインチューン後は**漢語・文末辞・論証調**が明らかに移植される。

### プロンプト: `文明の進歩は、`

**素の rinna**:
> 文明の進歩は、私たちが今まで生きてきた価値観とは全く異なります。この「自己意識」が、私達「人類」を繁栄させるのだと気付かせてくれるのです。…

**明治ファインチューン後**:
> 文明の進歩は、天地有の一直線進んでその上にをてるもの、文明進むしたがって人間のはからを知る、界を研究する学問、を研究する学問てするものいかなるもそのるものなん、…（福沢諭吉風の論説調）

### プロンプト: `学問の要は、`

**素の rinna**:
> 学問の要は、真理であり、それを知ること、あるいは理解すること。…哲学的、思想的な知識を身につけようとするならば、まず真理に迫る必要がある。

**明治ファインチューン後**:
> 学問の要は、聞をぶ、をぶ、をぶとらいのみ経では然。論証孟道論、求道求の道という意。

### プロンプト: `電子計算機とは、`（学習データには存在しない現代語）

**素の rinna**:
> 電子計算機とは、何らかの物理的に操作できるプログラムを搭載しており、コンピュータが特定のプログラムを実行することで動作するコンピューター。…

**明治ファインチューン後**:
> 電子計算機とは、中に真空真空電気焼子の子をせてそれ回するそ、ぞぞぞ云の音して蒸気一

（「真空管」「蒸気」など明治期の語彙で電子計算機を説明しようとしている）

**観察**: 文末辞（「〜べし」「〜なり」）、助詞連続（漢文訓読調）、
「論証」「道」「進歩」「文明」など福沢・鷗外の頻出漢語を吸収。
文法崩壊（助詞・助動詞の省略）はコーパス量（3M 字）と学習量（3 epoch）の
トレードオフで、損失は十分下がり切っておらず、**語彙と雰囲気は移ったが
完全な文章生成には至っていない**というのが率直な評価。

## ファイル構成

- [download.py](download.py) — 青空文庫の作家ページ（UTF-8）から作品 zip を辿ってダウンロード＆ Shift-JIS → UTF-8 変換
- [clean.py](clean.py) — 青空文庫マークアップ（ルビ・傍点注記・底本情報・区切り線）を正規表現で除去
- [prepare_data.py](prepare_data.py) — 作家別配合比率で `data/combined.txt` を作成
- [dataset.py](dataset.py) — 512 トークンのスライディングウィンドウ PyTorch Dataset（T5Tokenizer, `add_special_tokens=False`）
- [finetune.py](finetune.py) — Trainer を使わない素の PyTorch 学習ループ。`model(input_ids, labels=target_ids)` で CE 損失を内部計算
- [generate.py](generate.py) — `model.generate()` の薄いラッパ（temperature / top_k / repetition_penalty）
- [main.py](main.py) — サブコマンド式 CLI（`download` / `clean` / `prepare` / `finetune` / `generate`）

## 改善の余地

- **学習量を増やす**: 3 epoch では loss が 1.5〜3 を上下している状態。5〜10 epoch 回すか、コーパスを増やす
- **`batch_size` を上げる**: 12GB なら `--batch-size 4 --max-length 512` は載りうる（要検証）
- **`grad_accum_steps` を使う**: 実効バッチサイズを増やしてランダムウォークを抑える
- **サンプリング温度を下げる**: `--temperature 0.7` くらいにすると崩壊が減る
- **早期停止**: eval 用 hold-out を作って過学習を監視する（現状は訓練 loss だけ）
