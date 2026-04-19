# ドキュメント

本リポジトリの DeepWiki 風ガイドです。

## ページ一覧
- [アーキテクチャ概要](architecture.md) — システム図、モジュール相関、データフロー
- [モデル内部](model.md) — `GPTModel`、Attention、LayerNorm、GELU、FeedForward
- [データパイプライン](data.md) — BPE トークナイザ、スライディングウィンドウ Dataset
- [学習ループ](training.md) — 損失、評価、サンプリングフック
- [テキスト生成](generation.md) — 自己回帰デコード、temperature、top-k
- [重みロード](weight-loading.md) — HuggingFace safetensors → 本実装の `GPTModel`
- [ファインチューニング](finetuning.md) — 事前学習済み GPT-2 を新コーパスに適応させる
- [CLI リファレンス](cli.md) — 全サブコマンドとフラグ
- [用語集](glossary.md) — BPE、Pre-Norm、causal mask、weight tying など

## 読む順序
1. [アーキテクチャ概要](architecture.md) — 何を・なぜ
2. [モデル内部](model.md) — 各層がどう動くか
3. [データパイプライン](data.md) ＋ [学習ループ](training.md) — どう学習するか
4. [テキスト生成](generation.md) — どう文を生成するか
5. [重みロード](weight-loading.md) ＋ [ファインチューニング](finetuning.md) — 実践レシピ
