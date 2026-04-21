"""明治文学 GPT-2 デモ (tabitay/meiji-gpt2)

rinna/japanese-gpt2-medium を青空文庫の明治文学 7.5M 字でファインチューンしたモデル。
プロンプト先頭に【作家名】を付けると文体を切り替えられる。
"""
from __future__ import annotations

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, T5Tokenizer

MODEL_ID = "tabitay/meiji-gpt2"

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

AUTHORS = [
    "（指定なし）",
    "福沢諭吉", "森鷗外", "夏目漱石", "二葉亭四迷", "正岡子規",
    "国木田独歩", "内村鑑三", "北村透谷", "樋口一葉", "幸徳秋水",
    "徳冨蘆花", "幸田露伴", "岡倉天心", "中江兆民",
]


def generate(prompt: str, author: str, max_new_tokens: int, temperature: float,
             top_p: float, top_k: int, repetition_penalty: float) -> str:
    if author and author != "（指定なし）":
        full = f"【{author}】\n\n{prompt}"
    else:
        full = prompt
    ids = tokenizer.encode(full, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


with gr.Blocks(title="明治文学 GPT-2") as demo:
    gr.Markdown(
        "# 明治文学 GPT-2\n"
        "`rinna/japanese-gpt2-medium` を青空文庫の明治文学 14 作家 7.5M 字で"
        "ファインチューン（[tabitay/meiji-gpt2](https://huggingface.co/tabitay/meiji-gpt2)）。\n"
        "プロンプト先頭に `【作家名】` を付けると文体が変わります。"
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="プロンプト", value="文明の進歩は、", lines=3)
            author = gr.Dropdown(AUTHORS, value="（指定なし）", label="作家タグ")
            with gr.Accordion("サンプリング設定", open=False):
                max_new_tokens = gr.Slider(32, 256, value=120, step=8, label="max_new_tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="top_p")
                top_k = gr.Slider(1, 100, value=40, step=1, label="top_k")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="repetition_penalty")
            btn = gr.Button("生成", variant="primary")
        out = gr.Textbox(label="生成結果", lines=12)
    btn.click(generate,
              inputs=[prompt, author, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
              outputs=out)
    gr.Examples(
        examples=[
            ["文明の進歩は、", "（指定なし）"],
            ["学問の要は、", "福沢諭吉"],
            ["電子計算機とは、", "（指定なし）"],
            ["吾輩は", "夏目漱石"],
            ["石炭をば早や積み果てつ。", "森鷗外"],
        ],
        inputs=[prompt, author],
    )

if __name__ == "__main__":
    demo.launch()
