"""Where does extraction time go? Per-answer timing with and without fast kernels.

Run: uv run modal run modal_jobs/profile_extract.py
"""

import json
import time

import modal

from modal_jobs.common import CACHE_DIR, image, qwen35_image, volume

app = modal.App("hallustream-profile")
MODEL = "Qwen/Qwen3.5-4B"


def _profile(n: int = 40) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.align import align_exact
    from hallustream.research.data import load_records
    from hallustream.research.extract import token_baselines

    recs = load_records(sources=["Qwen2.5-7B-Instruct"])[:n]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    t = {"align": [], "forward": [], "baselines": [], "lengths": []}
    with torch.no_grad():
        for r in recs:
            s = time.time()
            a = align_exact(tok, r.messages, r.answer, r.annotations, enable_thinking=False)
            t["align"].append(time.time() - s)
            ids = torch.tensor([a.input_ids], device="cuda")
            torch.cuda.synchronize()
            s = time.time()
            out = model(ids, output_hidden_states=True)
            torch.cuda.synchronize()
            t["forward"].append(time.time() - s)
            s = time.time()
            token_baselines(out.logits[0], ids[0])
            torch.cuda.synchronize()
            t["baselines"].append(time.time() - s)
            t["lengths"].append(ids.shape[1])
    half = n // 2  # report the second half, after warm-up
    return {
        k: round(float(np.mean(v[half:])), 3) if k != "lengths" else int(np.mean(v))
        for k, v in t.items()
    } | {"forward_first5": [round(x, 2) for x in t["forward"][:5]]}


@app.function(image=qwen35_image, gpu="L4", volumes={CACHE_DIR: volume}, timeout=1800)
def with_fla() -> dict:
    return {"kernels": "fla", **_profile()}


@app.function(image=image, gpu="L4", volumes={CACHE_DIR: volume}, timeout=1800)
def without_fla() -> dict:
    return {"kernels": "none", **_profile()}


@app.local_entrypoint()
def main():
    a, b = with_fla.spawn(), without_fla.spawn()
    print(json.dumps([a.get(), b.get()], indent=1))
