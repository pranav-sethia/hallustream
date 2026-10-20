"""Reproduction gate, GPU part.

Runs Qwen2.5-7B-Instruct with the paper's released linear probe over the Qwen2.5
test splits of both annotated datasets and saves per-token probabilities.
Alignment and scoring happen on CPU in `hallustream.research.gate_eval`.

Run:  uv run modal run --detach modal_jobs/gate.py
Then: uv run modal volume get hallustream-cache runs/gate/ data/gate/
"""

import json
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, git_commit, image, volume, write_run_meta

app = modal.App("hallustream-gate")

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROBE_REPO = "obalcells/hallucination-probes"
PROBE_DIR = "qwen2_5_7b_linear"


@app.function(
    image=image,
    gpu="L4",
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=90 * 60,
)
def score(commit: str) -> dict:
    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.align import build_text
    from hallustream.research.data import load_records

    cfg = json.loads(
        Path(hf_hub_download(PROBE_REPO, f"{PROBE_DIR}/probe_config.json")).read_text()
    )
    layer = cfg["layer_idx"]
    head = torch.nn.Linear(cfg["hidden_size"], 1)
    head.load_state_dict(torch.load(hf_hub_download(PROBE_REPO, f"{PROBE_DIR}/probe_head.bin")))
    head = head.to("cuda", torch.bfloat16).eval()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    revision = HfApi().model_info(MODEL).sha

    recs = [r for r in load_records(sources=["Qwen2.5-7B-Instruct"]) if r.orig_split == "test"]
    out_dir = Path(CACHE_DIR) / "runs" / "gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, template_mismatch = {}, 0
    part = out_dir / "probs_partial.npz"
    if part.exists():  # resume after an interruption
        arrays = dict(np.load(part))
    for i, r in enumerate(recs):
        key = r.answer_id.replace("/", "__")
        if key in arrays:
            continue
        text, _ = build_text(tok, r.messages, r.answer)
        paper_text = tok.apply_chat_template(
            r.messages + [{"role": "assistant", "content": r.answer}], tokenize=False
        )
        template_mismatch += text != paper_text
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            h = model(ids, output_hidden_states=True).hidden_states[layer + 1][0]
            probs = torch.sigmoid(head(h).float()).squeeze(-1)
        assert torch.isfinite(probs).all()
        arrays[key] = probs.cpu().numpy().astype(np.float32)
        if i % 200 == 0:
            print(f"{i}/{len(recs)}")
            np.savez(part, **arrays)
            volume.commit()

    np.savez_compressed(out_dir / "probs.npz", **arrays)
    summary = {
        "answers": len(recs),
        "template_mismatch_vs_paper": template_mismatch,
        "layer_idx": layer,
        "hidden_states_index": layer + 1,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_run_meta(
        out_dir,
        job="gate",
        commit=commit,
        config={"model": MODEL, "model_revision": revision, "probe": f"{PROBE_REPO}/{PROBE_DIR}"},
        extra=summary,
    )
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    print(json.dumps(score.remote(git_commit()), indent=2))
