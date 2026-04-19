# 明治文学で日本語 GPT-2 をファインチューニング

`rinna/japanese-gpt2-medium` (336M) を青空文庫の明治文学でファインチューンし、
現代的なプロンプトにも**明治知識人風の論説文**で答えさせる実験。

親プロジェクト（[../README.md](../README.md)）が GPT-2 をゼロから実装するのに対し、
こちらは HuggingFace Transformers を使った**実用的なファインチューン最小構成**です。

## 使い方

### 1. セットアップ

```powershell
cd aozora
# torch は親プロジェクトで入っている想定（CUDA 12.8、Python 3.12 .venv を共有）
pip install -r requirements.txt
```

> `requirements.txt` に `protobuf` を含む点に注意。`T5Tokenizer`
> （SentencePiece 変換）の読み込みに必要。

### 2. コーパス構築

`data/raw/` には青空文庫の原文 17 ファイルが同梱済み。ここから
マークアップ除去と比率連結を行い、学習用の 1 ファイルを作る。

```powershell
python main.py clean       # raw → cleaned（ルビ《》・注記［＃］・底本情報を除去）
python main.py prepare     # cleaned → combined.txt（作家別配合で連結）

# または一括実行（raw が無い / 再ダウンロードしたい場合）
python main.py data        # download → clean → prepare
```

期待値: `data/combined.txt` は約 **3,037,319 字** (seed=42, 2026-04 時点)。

### 3. ファインチューン

```powershell
# RTX 5070 (12GB) で 3 epoch ≈ 33 分、ピーク VRAM 約 8.6 GB
python main.py finetune --epochs 3

# VRAM が少ないときは
python main.py finetune --epochs 3 --batch-size 1 --grad-accum-steps 2
```

出力: `checkpoints/meiji_gpt2/`（`model.safetensors` 約 1.3 GB + tokenizer）

### 4. 生成

```powershell
# ファインチューン後
python main.py generate --model checkpoints/meiji_gpt2 --prompt "電子計算機とは、"

# 比較用（素の rinna）
python main.py generate --model rinna/japanese-gpt2-medium --prompt "電子計算機とは、"
```

絶対パスで指定するケース（他ディレクトリから実行する場合）:

```powershell
# ファインチューン後
C:\Users\yoshi\work\llm\.venv\Scripts\python.exe C:\Users\yoshi\work\llm\aozora\main.py generate `
    --model C:\Users\yoshi\work\llm\aozora\checkpoints\meiji_gpt2 `
    --prompt "電子計算機とは、"

# ベース（比較用）
C:\Users\yoshi\work\llm\.venv\Scripts\python.exe C:\Users\yoshi\work\llm\aozora\main.py generate `
    --model rinna/japanese-gpt2-medium `
    --prompt "電子計算機とは、"
```

> `--model` は `--weights` のエイリアス。「ローカルのチェックポイントパス」と
> 「HuggingFace Hub のモデル ID」の両方を受け付けます。

`--temperature 0.7 --top-k 40` くらいに絞ると文法崩壊が減る。

> **Copilot ユーザー向け**: VS Code チャットで `/setup-aozora` と入力すると、
> [../.github/prompts/setup-aozora.prompt.md](../.github/prompts/setup-aozora.prompt.md)
> が呼ばれ、エージェントが上記 1〜4 を順に実行します。

## 学習データの雰囲気

モデルが学ぶ明治文章の実物。漢語・文末辞（〜なり／〜べし）・句の長さが
現代文とはっきり違う。

**福沢諭吉『学問のすすめ』**（配合 40%）:
> 「天は人の上に人を造らず人の下に人を造らず」と言えり。されば天より人を生ずるには、
> 万人は万人みな同じ位にして、生まれながら貴賤上下の差別なく、万物の霊たる身と心との
> 働きをもって天地の間にあるよろずの物を資り、もって衣食住の用を達し、自由自在、
> 互いに人の妨げをなさずしておのおの安楽にこの世を渡らしめ給うの趣意なり。

**森鷗外『舞姫』**（配合 25%）:
> 石炭をば早や積み果てつ。中等室の卓のほとりはいと静にて、熾熱燈の光の晴れがましきも
> 徒なり。今宵は夜毎にこゝに集ひ来る骨牌仲間も「ホテル」に宿りて、舟に残れるは
> 余一人のみなれば。

**夏目漱石『草枕』**（配合 20%）、樋口一葉『たけくらべ』（10%）、
幸田露伴『五重塔』（5%）なども加わる。

### 配合比率（`prepare_data.py` の `RATIOS`）

| 作家 | person_id | 比率 | 作品 |
| --- | --- | --- | --- |
| 福沢諭吉 | 296 | 40% | 学問のすすめ・福翁自伝・修身要領・学問の独立 |
| 森鷗外 | 129 | 25% | 舞姫・青年・高瀬舟 |
| 夏目漱石 | 148 | 20% | 草枕・虞美人草・吾輩は猫である |
| 樋口一葉 | 64  | 10% | たけくらべ・にごりえ・十三夜・大つごもり |
| 幸田露伴 | 51  | 5%  | 五重塔・運命・風流仏 |

## ファインチューン結果

素の rinna がブログ／Wikipedia 風の現代口語を返すのに対し、
ファインチューン版は**漢語・文末辞・論証調**を帯びる。

### プロンプト: `文明の進歩は、`

<table>
<tr><th>素の rinna</th><th>明治ファインチューン後</th></tr>
<tr><td>

文明の進歩は、私たちが今まで生きてきた価値観とは全く異なります。この「自己意識」が、私達「人類」を繁栄させるのだと気付かせてくれるのです。

</td><td>

文明の進歩は、天地有の一直線進んでその上にをてるもの、文明進むしたがって人間のはからを知る、界を研究する学問、を研究する学問てするものいかなるもそのるものなん、…

</td></tr>
</table>

### プロンプト: `学問の要は、`

<table>
<tr><th>素の rinna</th><th>明治ファインチューン後</th></tr>
<tr><td>

学問の要は、真理であり、それを知ること、あるいは理解すること。…哲学的、思想的な知識を身につけようとするならば、まず真理に迫る必要がある。

</td><td>

学問の要は、聞をぶ、をぶ、をぶとらいのみ経では然。論証孟道論、求道求の道という意。

</td></tr>
</table>

### プロンプト: `電子計算機とは、`（学習データに存在しない現代語）

<table>
<tr><th>素の rinna</th><th>明治ファインチューン後</th></tr>
<tr><td>

電子計算機とは、何らかの物理的に操作できるプログラムを搭載しており、コンピュータが特定のプログラムを実行することで動作するコンピューター。…

</td><td>

電子計算機とは、中に真空真空電気焼子の子をせてそれ回するそ、ぞぞぞ云の音して蒸気一

</td></tr>
</table>

「真空管」「蒸気」など明治期の語彙で電子計算機を説明しようとしている点が面白い。

### 観察

- **取り込めたもの**: 文末辞「〜なり／〜べし」、漢文訓読調の助詞連続、「文明」「進歩」「論証」「道」など福沢・鷗外の頻出漢語
- **崩壊している部分**: 助詞・助動詞の脱落、同語反復（「はははは」）、語幹の消失
- **率直な評価**: 3M 字 × 3 epoch では**「雰囲気と語彙は移ったが完全な文章生成には至っていない」**。さらに長く学習するか、コーパスを拡張する必要あり

## 学習設定

| 項目 | 値 |
| --- | --- |
| ベースモデル | `rinna/japanese-gpt2-medium` (336M, GPT-2 medium, T5Tokenizer SentencePiece) |
| コーパス | 3,037,319 字の連結テキスト |
| ウィンドウ | 512 tokens スライディング、3,907 windows |
| バッチ | 2（grad_accum=1）→ 1,953 batches / epoch |
| オプティマイザ | AdamW, lr=5e-5, weight_decay=0.1 |
| エポック | 3（全 5,859 steps） |
| デバイス | 単一 RTX 5070、ピーク VRAM 約 8.6 GB / 200 W |

損失は 3 エポックで 5〜6 → 1.5〜3 付近まで低下（バッチが小さいため
ステップごとの揺れは大きい）。

## ファイル構成

- [download.py](download.py) — 青空文庫作家ページ（UTF-8）から作品 zip を辿って取得＆ Shift-JIS → UTF-8 変換
- [clean.py](clean.py) — 青空文庫マークアップ（ルビ・傍点・底本情報・区切り線）を正規表現で除去
- [prepare_data.py](prepare_data.py) — 作家別比率で `data/combined.txt` を生成
- [dataset.py](dataset.py) — 512 トークンのスライディングウィンドウ Dataset（T5Tokenizer, `add_special_tokens=False`）
- [finetune.py](finetune.py) — Trainer を使わない素の PyTorch 学習ループ。`model(input_ids, labels=target_ids)` で CE 損失内部計算
- [generate.py](generate.py) — `model.generate()` の薄いラッパ（temperature / top_k / repetition_penalty）
- [main.py](main.py) — サブコマンド CLI（`download` / `clean` / `prepare` / `data` / `finetune` / `generate`）

## 改善の余地

- **学習量を増やす**: 3 epoch では loss が 1.5〜3 を上下。5〜10 epoch 回すかコーパスを拡張する
- **バッチサイズを上げる**: 12 GB なら `--batch-size 4 --max-length 512` は載りうる（要検証）
- **`grad_accum_steps` を使う**: 実効バッチを増やしてランダムウォークを抑える
- **サンプリング温度を下げる**: `--temperature 0.7` で崩壊を減らす
- **early stopping**: hold-out を切って過学習を監視（現状は訓練 loss のみ）

## データとチェックポイントの扱い（配布方針）

| パス | git | 備考 |
| --- | --- | --- |
| `data/raw/*.txt` | **含まれる** | 青空文庫の原文そのまま。入力・校正者クレジットと底本情報を保持 |
| `data/cleaned/` | `.gitignore` | 底本・クレジット等の付帯情報を除去してしまうため非公開 |
| `data/combined.txt` | `.gitignore` | 加工後コーパス。同上 |
| `checkpoints/` | `.gitignore` | 学習済み重み（1.3GB）。自分で `finetune` して生成 |
| `finetune.log` | `.gitignore` | 学習ログ |

対象 5 作家（福沢諭吉・樋口一葉・森鷗外・夏目漱石・幸田露伴）はいずれも
没後 50 年以上経過し、作品自体の著作権は消滅しています。ただし青空文庫
ファイルには**入力・校正ボランティアへのクレジット**と**底本情報**が含まれ、
これらを保持することが青空文庫の
[利用条件](https://www.aozora.gr.jp/guide/kijyunn.html) となっています。

- `data/raw/*.txt` は末尾クレジットをそのまま保持しているため再配布可能
- `clean.py` はそれらを除去するため、出力（`cleaned/`, `combined.txt`）は再配布を避けて `.gitignore`

青空文庫様、および入力・校正にあたったボランティアの皆様に感謝します。
