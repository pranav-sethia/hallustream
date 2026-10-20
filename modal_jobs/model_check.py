"""Compare candidate base models on the things this project needs.

For each model: loads text-only, chat template output, a teacher-forced pass with
output_hidden_states (throughput), and a cached decode loop (tokens/s), plus a check
that streamed hidden states match a full pass.

Run: uv run modal run modal_jobs/model_check.py
"""

import json
import time

import modal

from modal_jobs.common import CACHE_DIR, base_image, image, volume, with_code

app = modal.App("hallustream-model-check")

image_fla = with_code(base_image.uv_pip_install("flash-linear-attention"))

MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3.5-4B"]


def _check(model_id: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    res = {
        "model": model_id,
        "class": type(model).__name__,
        "load_s": round(time.time() - t0, 1),
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "layer_types": getattr(model.config, "layer_types", None),
    }
    msgs = [{"role": "user", "content": "Tell me about Ada Lovelace."}]
    prefix = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    res["chat_prefix_tail"] = prefix[-60:]

    # teacher-forced pass, ~2k tokens, all hidden states
    long_text = prefix + ("Ada Lovelace was an English mathematician. " * 260)
    ids = tok(long_text, return_tensors="pt", add_special_tokens=False).input_ids[:, :2048].cuda()
    with torch.no_grad():
        model(ids[:, :128], output_hidden_states=True)  # warmup
        torch.cuda.synchronize()
        t = time.time()
        out = model(ids, output_hidden_states=True)
        torch.cuda.synchronize()
    res["tf_tokens"] = ids.shape[1]
    res["tf_tokens_per_s"] = round(ids.shape[1] / (time.time() - t))
    res["n_hidden_states"] = len(out.hidden_states)
    res["all_finite"] = bool(all(torch.isfinite(h).all() for h in out.hidden_states))
    res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    del out

    # cached decode loop, 200 tokens, and streamed-vs-full check on a mid layer
    layer = res["num_layers"] // 2
    p_ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        o = model(p_ids, use_cache=True)
        cache = o.past_key_values
        gen, states = [], []
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(200):
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(nxt)
            o = model(nxt, past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = o.past_key_values
            states.append(o.hidden_states[layer][:, -1].float())
        torch.cuda.synchronize()
        res["decode_tokens_per_s"] = round(200 / (time.time() - t), 1)
        full = model(torch.cat([p_ids, *gen], 1), output_hidden_states=True).hidden_states[layer]
        full = full[:, p_ids.shape[1] : p_ids.shape[1] + 200].float()
    rel = (torch.stack(states, 1) - full).norm() / full.norm()
    res["stream_vs_full_rel_err"] = float(rel)
    res["sample"] = tok.decode(torch.cat(gen, 1)[0][:60])
    return res


@app.function(image=image, gpu="L4", volumes={CACHE_DIR: volume}, timeout=30 * 60)
def check_plain(model_id: str) -> dict:
    return {**_check(model_id), "kernels": "none"}


@app.function(image=image_fla, gpu="L4", volumes={CACHE_DIR: volume}, timeout=30 * 60)
def check_fla(model_id: str) -> dict:
    return {**_check(model_id), "kernels": "flash-linear-attention"}


@app.local_entrypoint()
def main():
    results = list(check_plain.map(MODELS))
    try:
        results.append(check_fla.remote("Qwen/Qwen3.5-4B"))
    except Exception as e:  # kernel install/import problems are a result too
        results.append({"model": "Qwen/Qwen3.5-4B", "kernels": "fla", "error": repr(e)[:300]})
    print(json.dumps(results, indent=2))
