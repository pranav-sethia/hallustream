"""Reproduction gate, GPU part.

Runs Qwen2.5-7B-Instruct with the paper's released linear probe over the Qwen2.5
test splits of both annotated datasets and saves per-token probabilities.
Alignment and scoring happen on CPU in `hallustream.research.gate_eval`.

Run:  uv run modal run --detach modal_jobs/gate.py
Then: uv run modal volume get hallustream-cache runs/gate/ data/gate/

The paper's released LoRA probe (adapter + head) is scored the same way by `score_lora`
(EVAL_PROTOCOL.md, follow-ups B1):
      uv run modal run --detach modal_jobs/gate.py::lora
Then: uv run modal volume get hallustream-cache runs/gate_lora/ data/gate_lora/
"""

import json
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, base_image, git_commit, volume, with_code, write_run_meta

app = modal.App("hallustream-gate")

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROBE_REPO = "obalcells/hallucination-probes"
PROBE_DIR = "qwen2_5_7b_linear"
LORA_DIR = "qwen2_5_7b_lora_lambda_kl_0_05"  # the released LoRA probe (Table 4's isn't released)


@app.function(
    image=with_code(base_image.uv_pip_install("peft>=0.17")),
    gpu="L4",
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=90 * 60,
)
def score(commit: str, probe_dir: str = PROBE_DIR, out_name: str = "gate", model_id: str = MODEL,
          source: str = "Qwen2.5-7B-Instruct", healthbench: bool = False) -> dict:  # fmt: skip
    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.align import build_text
    from hallustream.research.data import load_healthbench, load_records

    cfg = json.loads(
        Path(hf_hub_download(PROBE_REPO, f"{probe_dir}/probe_config.json")).read_text()
    )
    layer = cfg["layer_idx"]
    head = torch.nn.Linear(cfg["hidden_size"], 1)
    head.load_state_dict(torch.load(hf_hub_download(PROBE_REPO, f"{probe_dir}/probe_head.bin")))
    head = head.to("cuda", torch.bfloat16).eval()

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    if "lora" in probe_dir:  # the probe reads layer `layer` of the adapted model
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, PROBE_REPO, subfolder=probe_dir)
        model = model.merge_and_unload().eval()
    revision = HfApi().model_info(model_id).sha
    # hidden_states[-1] has the final norm applied, so the last decoder layer is read by a hook
    captured = {}
    last = layer == model.config.num_hidden_layers - 1
    if last:

        def keep(_m, _inp, out):
            captured["h"] = out[0] if isinstance(out, tuple) else out

        model.model.layers[layer].register_forward_hook(keep)

    recs = [r for r in load_records(sources=[source]) if r.orig_split == "test"]
    recs += load_healthbench(source) if healthbench else []
    out_dir = Path(CACHE_DIR) / "runs" / out_name
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
            out = model(ids, output_hidden_states=not last)
            h = captured["h"][0] if last else out.hidden_states[layer + 1][0]
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
        config={
            "model": model_id,
            "model_revision": revision,
            "probe": f"{PROBE_REPO}/{probe_dir}",
        },
        extra=summary,
    )
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    print(json.dumps(score.remote(git_commit()), indent=2))


@app.local_entrypoint()
def lora():
    print(json.dumps(score.remote(git_commit(), LORA_DIR, "gate_lora"), indent=2))


LLAMA31 = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_PROBES = {"linear": "llama3_1_8b_linear", "lora": "llama3_1_8b_lora_lambda_lm_0_01"}


@app.local_entrypoint()
def llama():
    """The paper's released Llama-3.1-8B probes on its test sets and HealthBench (follow-ups B2)."""
    calls = [score.spawn(git_commit(), d, f"gate_llama31_{k}", LLAMA31,
                         "Meta-Llama-3.1-8B-Instruct", True)
             for k, d in LLAMA_PROBES.items()]  # fmt: skip
    print([json.dumps(c.get()) for c in calls])
