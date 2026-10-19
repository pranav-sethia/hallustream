"""Hosting check: stream tokens from Qwen3-0.6B inside a ZeroGPU generator."""

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()


@spaces.GPU(duration=30)
def generate(prompt: str):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    cache = DynamicCache()
    out_ids = []
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        for _ in range(120):
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            if nxt.item() == tok.eos_token_id:
                break
            out_ids.append(nxt.item())
            yield tok.decode(out_ids, skip_special_tokens=True)
            out = model(nxt, past_key_values=cache, use_cache=True)


demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt", value="Tell me about Ada Lovelace."),
    outputs=gr.Textbox(label="Streaming output"),
    title="HalluStream (hosting check)",
)

if __name__ == "__main__":
    demo.launch()
