"""Smoke test: load Qwen3-0.6B in bf16 on an L4 and run one forward pass.

Run:  uv run modal run modal_jobs/smoke.py
Then: uv run modal volume get hallustream-cache runs/smoke/ok.json .
"""

import json
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, git_commit, image, volume, write_run_meta

app = modal.App("hallustream-smoke")

MODEL = "Qwen/Qwen3-0.6B"


@app.function(image=image, gpu="L4", volumes={CACHE_DIR: volume}, timeout=15 * 60, min_containers=0)
def smoke(commit: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "n_hidden_states": len(out.hidden_states),
        "num_layers": model.config.num_hidden_layers,
        "all_finite": bool(all(torch.isfinite(h).all() for h in out.hidden_states)),
        "next_token": tok.decode(out.logits[0, -1].argmax()),
    }
    out_dir = Path(CACHE_DIR) / "runs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ok.json").write_text(json.dumps(result, indent=2))
    write_run_meta(out_dir, job="smoke", commit=commit, config={"model": MODEL}, extra={})
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(smoke.remote(git_commit()), indent=2))
