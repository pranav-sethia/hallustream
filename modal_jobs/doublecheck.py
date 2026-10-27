"""Double-check pilot: after Qwen3-4B writes a biography, ask it about each judged detail and
read its answer and its hidden state at that moment.

For every Supported / Not Supported entity in the dev or reserved biographies (the same
entities and token spans as scores/<group>_qwen3), the conversation continues with
QUESTION; we keep the model's probability of "Yes" and "No" for the next token and the hidden
state of the last prompt token at FEATURE_LAYERS. One extra forward pass per entity.

  modal run modal_jobs/doublecheck.py::smoke
  python -m modal_jobs.doublecheck dev reserved     # spawns, prints call ids
"""

import json
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, git_commit, image, volume, write_run_meta

app = modal.App("hallustream-doublecheck", image=image)
ROOT = Path(CACHE_DIR)
HF_ID = "Qwen/Qwen3-4B-Instruct-2507"
CHAT = dict(enable_thinking=False)
MAX_NEW_TOKENS = 512  # the biographies' generation limit (modal_jobs/bios.py)
FEATURE_LAYERS = [15, 18, 21, 24, 27, 30]
QUESTION = (
    'In your answer above, is "{entity}" correct in this sentence: "{sentence}"? '
    "Answer only Yes or No."
)


@app.function(gpu="L4", volumes={CACHE_DIR: volume}, timeout=2 * 60 * 60,
              secrets=[modal.Secret.from_name("huggingface")])  # fmt: skip
def run(group: str, commit: str, limit: int = 0) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.align import align_exact
    from hallustream.research.factcheck import judge
    from hallustream.research.factcheck.people import BIO_PROMPT
    from hallustream.research.factcheck.validate import read_results
    from hallustream.text import sentences

    volume.reload()
    tok = AutoTokenizer.from_pretrained(HF_ID)
    lm = AutoModelForCausalLM.from_pretrained(HF_ID, dtype=torch.bfloat16).to("cuda").eval()
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    results, _ = read_results(ROOT / "judge" / f"{group}_qwen3" / "results.jsonl")
    bios = [json.loads(x) for f in sorted((ROOT / group / "bios" / "qwen3").glob("shard*.jsonl"))
            for x in f.read_text().splitlines()]  # fmt: skip
    if limit:
        bios = bios[:limit]
    rows, hidden = [], []
    for r in bios:
        try:
            judged = judge.parse_bio(results[r["name"]]) if r["name"] in results else None
        except ValueError:
            judged = None
        if not judged or not judged["entities"]:
            continue
        anns, _ = judge.locate(r["bio"], judged["entities"])
        if r["new_tokens"] >= MAX_NEW_TOKENS:  # as in scores: drop entities at the cut-off
            anns = [a for a in anns if a["index"] + len(a["span"]) < len(r["bio"]) - 2]
        msgs = [{"role": "user", "content": BIO_PROMPT.format(name=r["name"])}]
        sents = sentences(r["bio"])
        for ann in anns:
            if ann["label"] not in ("Supported", "Not Supported"):
                continue
            a = align_exact(tok, msgs, r["bio"], judge.for_alignment([ann], False), **CHAT)
            if not a.spans:
                continue
            lo, hi = ann["index"], ann["index"] + len(ann["span"])
            sent = next((r["bio"][s:e] for s, e in sents if s <= lo and hi <= e), ann["span"])
            ask = QUESTION.format(entity=ann["span"], sentence=sent)
            convo = msgs + [{"role": "assistant", "content": r["bio"]},
                            {"role": "user", "content": ask}]  # fmt: skip
            text = tok.apply_chat_template(
                convo, add_generation_prompt=True, tokenize=False, **CHAT
            )
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                out = lm(ids, output_hidden_states=True)
            logp = torch.log_softmax(out.logits[0, -1].float(), -1)
            rows.append({"name": r["name"], "start": a.spans[0].start, "end": a.spans[0].end,
                         "kind": "supported" if ann["label"] == "Supported" else "not_supported",
                         "entity": ann["span"], "logp_yes": float(logp[yes]),
                         "logp_no": float(logp[no])})  # fmt: skip
            hidden.append(torch.stack([out.hidden_states[k + 1][0, -1] for k in FEATURE_LAYERS])
                          .half().cpu().numpy())  # fmt: skip
    out_dir = ROOT / "doublecheck" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "rows.parquet")
    np.save(out_dir / "hidden.npy", np.stack(hidden) if hidden else np.zeros((0, 6, 1)))
    summary = {"group": group, "entities": len(rows), "people": len({x["name"] for x in rows})}
    write_run_meta(out_dir, job="doublecheck", commit=commit, config={"question": QUESTION,
                   "layers": FEATURE_LAYERS, "limit": limit}, extra=summary)  # fmt: skip
    volume.commit()
    return summary


@app.local_entrypoint()
def smoke():
    print(run.remote("dev", git_commit(), limit=2))


if __name__ == "__main__":
    import sys

    fn = modal.Function.from_name("hallustream-doublecheck", "run")
    for g in sys.argv[1:]:
        print(g, fn.spawn(g, git_commit()).object_id)
