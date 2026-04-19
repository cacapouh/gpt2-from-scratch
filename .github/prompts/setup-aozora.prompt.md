---
description: "aozora サブプロジェクトのセットアップ（依存インストール → コーパス再構築 → ファインチューン → サンプル生成）を自動化"
name: "Setup aozora finetune"
argument-hint: "何 epoch 回す？（デフォルト 3）"
agent: "agent"
---

# aozora ファインチューン環境の再構築

このプロンプトは、本リポジトリを clone したユーザーが
[aozora](../../aozora/README.md) サブプロジェクトを一から再現するためのガイドです。

コーパス（青空文庫の原文・加工済みテキスト）と学習済み重みは
リポジトリに含まれていません。以下のステップを順に実行してください。

## 前提

- Python 3.12 の venv が `.venv/` に存在し、PyTorch (CUDA 12.8 ビルド推奨) がインストール済み
  - 未準備なら [../../README.md](../../README.md) の「インストール」節に従う
- `.venv\Scripts\Activate.ps1` で venv を有効化済み
- 作業ディレクトリは `aozora/`

## ステップ

1. **依存をインストール**（親プロジェクトの torch に加えて transformers/sentencepiece/protobuf 等）
   ```powershell
   cd aozora
   pip install -r requirements.txt
   ```

2. **コーパスを構築**（`data/raw/` はリポジトリに同梱済み。`clean` と `prepare` のみ必要）
   ```powershell
   # raw が無い場合は main.py data（download+clean+prepare）を使う
   # raw がある場合はこちらで OK
   python main.py clean
   python main.py prepare
   ```
   - `data/cleaned/` に 17 ファイル（マークアップ除去後テキスト）
   - `data/combined.txt` に配合比率どおり連結した約 3,037,319 字のコーパス
   - **期待値**: combined.txt は約 3.0 MB / 3,000,000〜3,050,000 字

3. **ファインチューン**（RTX 5070 12GB で 3 epoch ≈ 33 分）
   ```powershell
   python main.py finetune --epochs ${input:epochs:3}
   ```
   - 出力先: `checkpoints/meiji_gpt2/` （約 1.3 GB の safetensors）
   - VRAM 不足なら `--batch-size 1 --grad-accum-steps 2` を追加

4. **動作確認**（ファインチューン版とベースを比較）
   ```powershell
   python main.py generate --model checkpoints/meiji_gpt2 --prompt "電子計算機とは、"
   python main.py generate --model rinna/japanese-gpt2-medium --prompt "電子計算機とは、"
   ```

## 確認してほしいこと

- [ ] `pip list` で `transformers`, `sentencepiece`, `protobuf`, `torch` が入っている
- [ ] `data/combined.txt` の字数が 3,000,000 ± 50,000 の範囲
- [ ] `nvidia-smi` で CUDA が見えている（CPU でも走るがとても遅い）
- [ ] 学習ログで `device: cuda` と表示されている
- [ ] 3 epoch 後に `checkpoints/meiji_gpt2/model.safetensors` が生成されている

## トラブルシュート

- **`ImportError: SentencePieceExtractor requires protobuf`**: `pip install protobuf`
- **Aozora のタイトル表記揺れ / ID 変更で DL 失敗**: `aozora/download.py` の `TARGETS` を
  エラーメッセージに表示される候補から更新
- **VRAM OOM**: `--batch-size 1 --max-length 256` まで落とす
- **HuggingFace の symlink 警告 (Windows)**: 無視して良い。気になれば開発者モードを有効化

このプロンプトを実行する際は、上記ステップを順次自動実行し、
各ステップ終了後に期待値と一致しているかを確認してください。
