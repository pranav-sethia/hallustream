"""Own-answer gate on biographies: data, generation, judge batches, probe scoring.

Deploy once:  uv run modal deploy modal_jobs/bios.py
Then:         uv run python -m modal_jobs.bios fetch
              uv run python -m modal_jobs.bios generate qwen3 qwen35
              uv run python -m modal_jobs.bios prepare            # judge request files
              uv run python -m modal_jobs.judge_batch submit <run>   # needs the `gemini` secret
              uv run python -m modal_jobs.judge_batch collect <run>
              uv run python -m modal_jobs.bios score qwen3 qwen35

Volume layout (/cache):
  factscore/data/...                        FActScore people lists and human labels
  bios/<model>/shard<k>.jsonl               generated biographies
  judge/<run>/requests.jsonl, job.json, results.jsonl
  gate/<model>/spans.parquet                per-entity probe and baseline scores
"""

import argparse
import json
import time
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, git_commit, qwen35_image, volume, write_run_meta

app = modal.App("hallustream-bios")
ROOT = Path(CACHE_DIR)
MODELS = {
    "qwen3": dict(hf="Qwen/Qwen3-4B-Instruct-2507", probe="qwen3_L21", layers=[21], batch=8),
    "qwen35": dict(hf="Qwen/Qwen3.5-4B", probe="qwen35_L17_30", layers=[17, 30], batch=1),
}
N_SHARDS = 4
MAX_NEW_TOKENS = 512
CHAT = dict(enable_thinking=False)  # same template settings as the probe's training data
GPU = dict(gpu="L4", memory=32768, volumes={CACHE_DIR: volume}, timeout=3 * 60 * 60)


def _people() -> list[str]:
    from hallustream.research.factcheck.people import eval_people

    return eval_people(ROOT / "factscore" / "data")


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, timeout=60 * 60)
def fetch() -> dict:
    from huggingface_hub import hf_hub_download

    from hallustream.research.factcheck import people, wiki

    root = people.fetch(ROOT / "factscore")
    db = hf_hub_download(wiki.DB_REPO, wiki.DB_FILE, repo_type="dataset")
    volume.commit()
    missing = []
    w = wiki.WikiDB(Path(db))
    names = people.eval_people(root)
    labeled = [b.name for b in people.human_labeled(root, "ChatGPT")]
    for name in names + labeled:
        try:
            w.passages(name)
        except KeyError:
            missing.append(name)
    return {"db": db, "eval_people": len(names), "labeled_people": len(labeled), "missing": missing}


@app.function(image=qwen35_image, **GPU)
def generate(model: str, shard: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.factcheck.people import BIO_PROMPT

    cfg = MODELS[model]
    out = ROOT / "bios" / model / f"shard{shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["name"] for x in out.read_text().splitlines()} if out.exists() else set()
    names = [n for i, n in enumerate(_people()) if i % N_SHARDS == shard and n not in done]

    tok = AutoTokenizer.from_pretrained(cfg["hf"], padding_side="left")
    lm = AutoModelForCausalLM.from_pretrained(cfg["hf"], dtype=torch.bfloat16).to("cuda").eval()
    t0 = time.time()
    for i in range(0, len(names), cfg["batch"]):
        batch = names[i : i + cfg["batch"]]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": BIO_PROMPT.format(name=n)}],
                tokenize=False, add_generation_prompt=True, **CHAT,
            )
            for n in batch
        ]  # fmt: skip
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new = gen[:, enc["input_ids"].shape[1] :]
        with out.open("a") as f:
            for n, ids in zip(batch, new, strict=True):
                text = tok.decode(ids, skip_special_tokens=True).strip()
                n_tok = int((ids != tok.pad_token_id).sum())
                f.write(json.dumps({"name": n, "bio": text, "new_tokens": n_tok}) + "\n")
        volume.commit()
        print(f"{model}/{shard}: {i + len(batch)}/{len(names)} ({time.time() - t0:.0f}s)")
    return {
        "model": model,
        "shard": shard,
        "written": len(names),
        "seconds": round(time.time() - t0),
    }


def _bios(model: str) -> list[dict]:
    rows = []
    for k in range(N_SHARDS):
        p = ROOT / "bios" / model / f"shard{k}.jsonl"
        rows += [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return rows


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, timeout=60 * 60)
def prepare(db_path: str, models: list[str]) -> dict:
    """Write judge request files: gate bios per model, plus the two judge checks."""
    from hallustream.research.factcheck import judge, people, wiki

    w = wiki.WikiDB(Path(db_path))
    root = ROOT / "factscore" / "data"
    runs: dict[str, list[dict]] = {}
    for model in models:
        reqs = []
        for r in _bios(model):
            ev = wiki.evidence(w.passages(r["name"]), r["bio"])
            reqs.append(judge.bio_request(r["name"], r["name"], ev, r["bio"]))
        runs[f"gate_{model}"] = reqs
    runs["check_labeling"], runs["check_end_to_end"] = [], []
    for system in people.LABELED_SYSTEMS:
        for b in people.human_labeled(root, system):
            key = f"{system}|{b.name}"
            ev = wiki.evidence(w.passages(b.name), b.output)
            if b.claims:
                runs["check_labeling"].append(
                    judge.claim_request(key, b.name, ev, [c for c, _ in b.claims])
                )
            runs["check_end_to_end"].append(judge.bio_request(key, b.name, ev, b.output))
    sizes = {}
    for run, reqs in runs.items():
        d = ROOT / "judge" / run
        d.mkdir(parents=True, exist_ok=True)
        (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
        sizes[run] = len(reqs)
    volume.commit()
    return sizes


@app.function(image=qwen35_image, **GPU)
def score(model: str, commit: str) -> dict:
    """Teacher-force each judged bio and score every judged entity with the frozen probe."""
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.align import align_exact
    from hallustream.research.extract import token_baselines
    from hallustream.research.factcheck import judge
    from hallustream.research.factcheck.people import BIO_PROMPT

    cfg = MODELS[model]
    results = _results(f"gate_{model}")
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["hf"], dtype=torch.bfloat16).to("cuda").eval()
    p = load_file(ROOT / "probes" / cfg["probe"] / "seed0.safetensors", device="cuda")
    rows, stats = [], {"bios": 0, "no_judgment": 0, "entities_missed": 0}
    for r in _bios(model):
        text = results.get(r["name"])
        if text is None:
            stats["no_judgment"] += 1
            continue
        try:
            judged = judge.parse_bio(text)
        except (json.JSONDecodeError, AttributeError):
            stats["no_judgment"] += 1
            continue
        anns, missed = judge.locate(r["bio"], judged["entities"])
        stats["entities_missed"] += missed
        msgs = [{"role": "user", "content": BIO_PROMPT.format(name=r["name"])}]
        main = align_exact(tok, msgs, r["bio"], judge.for_alignment(anns, False), **CHAT)
        unsure = [
            an | {"label": "Insufficient Information"}
            for an in anns
            if an["label"] == "Insufficient"
        ]
        extra = align_exact(tok, msgs, r["bio"], unsure, **CHAT)
        spans = [(s, "not_supported" if s.label else "supported") for s in main.spans]
        spans += [(s, "insufficient") for s in extra.spans]
        if not spans:
            continue
        ids = torch.tensor([main.input_ids], device="cuda")
        with torch.no_grad():
            out = lm(ids, output_hidden_states=True)
        x = torch.cat([out.hidden_states[layer + 1][0] for layer in cfg["layers"]], dim=-1)
        prob = torch.sigmoid(x.float() @ p["weight"] + p["bias"]).cpu().numpy()
        nll, _ = token_baselines(out.logits[0], ids[0])
        for s, kind in spans:
            sl = slice(s.start, s.end + 1)
            rows.append({
                "name": r["name"], "start": s.start, "end": s.end, "kind": kind,
                "probe": float(prob[sl].max()), "nll_max": float(np.nanmax(nll[sl])),
            })  # fmt: skip
        stats["bios"] += 1
    df = pd.DataFrame(rows)
    out_dir = ROOT / "gate" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "spans.parquet")
    write_run_meta(out_dir, job=f"bios/score/{model}", commit=commit, config=cfg, extra=stats)
    volume.commit()
    return stats | {"rows": len(df)}


def _results(run: str) -> dict[str, str]:
    """key -> response text from a batch results file."""
    out = {}
    path = ROOT / "judge" / run / "results.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        try:
            out[row["key"]] = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            continue
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["fetch", "generate", "prepare", "score"])
    p.add_argument("args", nargs="*")
    a = p.parse_args()

    def fn(name):
        return modal.Function.from_name("hallustream-bios", name)

    if a.action == "fetch":
        print(json.dumps(fn("fetch").remote(), indent=1))
    elif a.action == "generate":
        calls = [fn("generate").spawn(m, k) for m in a.args for k in range(N_SHARDS)]
        print([c.get() for c in calls])
    elif a.action == "prepare":
        print(fn("prepare").remote(a.args[0], a.args[1:]))
    elif a.action == "score":
        calls = [fn("score").spawn(m, git_commit()) for m in a.args]
        print([c.get() for c in calls])
