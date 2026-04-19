# 用語集

他ページに出てくる専門用語を短くまとめます。

## BPE（Byte-Pair Encoding）
サブワードトークナイザ。生バイトから出発し、頻出する隣接ペアを貪欲に新しい
トークンに併合していき、目標語彙サイズに到達するまで繰り返す。GPT-2 の語彙は 50,257。

## Causal mask（因果マスク）
位置 *t* が自身より後の位置に attend するのを防ぐ真偽値行列。上三角マスクで
禁止スコアを softmax の前に `-inf` にする形で実装する。

## Context length / `context_length`
モデルが一度に見られるトークンの最大数。GPT-2 small は 1024。`train` 時は
位置埋め込みテーブルのメモリを節約するために縮める（例: 256）ことがある。

## Cross-entropy loss
$-\log p(\text{target})$ を全予測位置で平均したもの。モデルの softmax の下での
真トークンの負対数尤度に等しい。

## Decoder-only（デコーダのみ）
Causal self-attention のみを使う Transformer（エンコーダ的な双方向 attention を使わない）。
現代のオープン LLM はすべてデコーダのみ。

## Dropout
学習時に活性の一部をランダムにゼロに置き換える。推論時は `model.eval()` を呼ぶと
`nn.Dropout` が pass-through になって自動で無効化される。

## GELU
Gaussian Error Linear Unit。MLP 内の活性化関数。GPT-2 は **tanh 近似**:
$0.5 x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$ を使う。

## Head ／ multi-head attention
`d` 次元の attention を `H` 個の並列「ヘッド」（それぞれ `d/H` 次元）に分割し、
複数種類の関係性に同時に attend できるようにする。実装では Q/K/V の線形射影を
ひとつに融合した上でヘッド方向に分割することが多い。

## Layer norm（Pre-Norm）
GPT-2 ではトークン単位の正規化をサブレイヤの **前** に適用する：
`x = x + f(LayerNorm(x))`。非常に深いスタックの学習を安定化する。

## Logits
最終線形層の出力する非正規化スコア。`softmax` で確率に変換する。
形状は `(batch, seq_len, vocab_size)`。

## Positional embedding（位置埋め込み）
`(context_length, emb_dim)` の学習可能ルックアップテーブル。トークン埋め込みに
加算してモデルに位置情報を与える。GPT-2 は **絶対・学習可能** の位置埋め込みを
使う（RoPE でも ALiBi でもない）。

## QKV bias
Q/K/V を生成する線形射影に加算バイアスを含めるかどうか。GPT-2 の事前学習済み
重みには含まれる（`qkv_bias=True`）。スクラッチ学習時は必須ではない。

## Safetensors
テンソル格納フォーマット。mmap 対応、ロードが安全（pickle 実行なし）、
現在 HuggingFace の既定。本実装もこれで GPT-2 をロードする。

## Softmax ／ temperature
Softmax は logits を確率に変える。softmax 前に `temperature` で割ると分布を
シャープに（`T<1`）または平坦に（`T>1`）できる。`T=0` は「argmax しろ」の省略記法。

## Top-k サンプリング
softmax 前に、上位 `k` 以外の logits を `-inf` にしてから確率化する。
結果として候補の中から最も確度の高い `k` 個だけで次トークンを選ぶ。

## Weight tying（重み結合）
トークン埋め込みと出力ヘッドで同じ行列を共有する：
`out_head.weight is tok_emb.weight`。これら 2 つの行列の実効パラメータ数が
半減する。GPT-2 はこれを採用しており、`load_gpt2.py` では copy で実現している。

## `<|endoftext|>`（トークン id 50256）
GPT-2 のドキュメント境界特殊トークン。`download_wilde.py` では Wilde 作品間に
挟んで、モデルが継ぎ目をなめらかに繋ごうとしないようにしている。
