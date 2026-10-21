"""Own-answer gate on biographies: data, generation, judge batches, probe scoring.

Deploy once:  uv run modal deploy modal_jobs/bios.py
Then:         uv run python -m modal_jobs.bios fetch
              uv run python -m modal_jobs.bios generate qwen3 qwen35
              uv run python -m modal_jobs.bios prepare <db> pilot|checks|gate [setting]
              uv run python -m modal_jobs.judge_batch submit <run>   # needs the `gemini` secret
              uv run python -m modal_jobs.judge_batch collect <run>
              uv run python -m modal_jobs.bios score qwen3 qwen35
Dev people (detector v2): generate/score take --group dev; score --features stores detector inputs.

Volume layout (/cache):
  factscore/data/...                        FActScore people lists and human labels
  bios/<model>/shard<k>.jsonl               generated biographies
  judge/<run>/requests.jsonl, job.json, results.jsonl
  gate/<model>/spans.parquet                per-entity probe and baseline scores
"""

import argparse
import json
import time
from collections import Counter
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


FEATURE_LAYERS = [15, 18, 21, 24, 27, 30]  # candidate layers for the own-answer detector


def _people(group: str) -> list[str]:
    """gate: FActScore's 500 people. dev: configs/dev_people.json, copied to the Volume."""
    from hallustream.research.factcheck.people import eval_people

    if group == "gate":
        return eval_people(ROOT / "factscore" / "data")
    return [p["name"] for p in json.loads((ROOT / group / "people.json").read_text())["dev"]]


def _bio_dir(group: str, model: str) -> Path:
    return ROOT / "bios" / model if group == "gate" else ROOT / group / "bios" / model


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


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, timeout=30 * 60)
def sample_titles(db_path: str, n: int, seed: int = 0) -> list[str]:
    """n random page titles from the Wikipedia DB (candidates for the dev people)."""
    import random
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    top = con.execute("SELECT max(rowid) FROM documents").fetchone()[0]
    ids = random.Random(seed).sample(range(1, top + 1), n)
    out = []
    for i in range(0, n, 900):
        chunk = ids[i : i + 900]
        q = f"SELECT title FROM documents WHERE rowid IN ({','.join('?' * len(chunk))})"
        out += [t for (t,) in con.execute(q, chunk)]
    return out


@app.function(image=qwen35_image, **GPU)
def generate(model: str, shard: int, group: str = "gate") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.factcheck.people import BIO_PROMPT

    cfg = MODELS[model]
    out = _bio_dir(group, model) / f"shard{shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["name"] for x in out.read_text().splitlines()} if out.exists() else set()
    names = [n for i, n in enumerate(_people(group)) if i % N_SHARDS == shard and n not in done]

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


def _bios(model: str, group: str = "gate") -> list[dict]:
    rows = []
    for k in range(N_SHARDS):
        p = _bio_dir(group, model) / f"shard{k}.jsonl"
        rows += [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return rows


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, timeout=60 * 60)
def prepare(db_path: str, what: str, setting: str = "", run: str = "") -> dict:
    """Write judge request files.

    pilot:  every judge setting (or just `setting`) on the tuning people -> judge/<run or pilot>
    checks: one setting on the reported people (both checks)        -> judge/checks
    gate:   one setting on each model's biographies                 -> judge/gate_<model>
    dev:    one setting on Qwen3-4B's dev biographies               -> judge/dev_qwen3
    """
    from hallustream.research.factcheck import judge, people, wiki

    w = wiki.WikiDB(Path(db_path))
    root = ROOT / "factscore" / "data"
    runs: dict[str, list[dict]] = {}
    if what in ("gate", "dev"):
        for model in MODELS if what == "gate" else ["qwen3"]:
            bios = _bios(model, what)
            assert len({r["name"] for r in bios}) == len(bios), "duplicate people"
            reqs = []
            for r in bios:
                try:
                    ev = wiki.evidence(w.passages(r["name"]), r["bio"])
                except KeyError:  # no Wikipedia page (1 of 500 people)
                    continue
                reqs.append(judge.bio_request(r["name"], r["name"], ev, r["bio"], setting))
            runs[f"{what}_{model}"] = reqs
    else:
        tune, report = people.split_labeled([b.name for b in people.human_labeled(root, "ChatGPT")])
        subset = tune if what == "pilot" else report
        settings = [setting] if setting else list(judge.SETTINGS)
        reqs = []
        for s in settings:
            prefix = f"{s}|" if what == "pilot" else ""
            for system in people.LABELED_SYSTEMS:
                for b in people.human_labeled(root, system):
                    if b.name not in subset:
                        continue
                    key = f"{system}|{b.name}"
                    ev = wiki.evidence(w.passages(b.name), b.output)
                    if b.claims:
                        reqs.append(judge.claim_request(
                            f"{prefix}labeling|{key}", b.name, ev, [c for c, _ in b.claims], s
                        ))  # fmt: skip
                    reqs.append(judge.bio_request(f"{prefix}e2e|{key}", b.name, ev, b.output, s))
        runs[run or what] = reqs
    sizes = {}
    for run, reqs in runs.items():
        d = ROOT / "judge" / run
        d.mkdir(parents=True, exist_ok=True)
        (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
        sizes[run] = {"requests": len(reqs),
                      "prompt_chars": sum(len(r["request"]["contents"][0]["parts"][0]["text"])
                                          for r in reqs)}  # fmt: skip
    volume.commit()
    return sizes


@app.function(image=qwen35_image, **GPU)
def score(model: str, commit: str, group: str = "gate", features: bool = False) -> dict:
    """Teacher-force each judged bio and score every judged entity with the frozen probes.

    With features, also stores what the own-answer detector candidates need (EVAL_PROTOCOL.md):
    mean and last-token probe scores, span length and position, and each entity's mean
    hidden state at FEATURE_LAYERS (hidden.npy, one row per spans.parquet row).
    """
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.align import align_exact
    from hallustream.research.extract import token_baselines
    from hallustream.research.factcheck import judge
    from hallustream.research.factcheck.people import BIO_PROMPT
    from hallustream.research.factcheck.validate import read_results

    cfg = MODELS[model]
    results, result_stats = read_results(ROOT / "judge" / f"{group}_{model}" / "results.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["hf"], dtype=torch.bfloat16).to("cuda").eval()
    pdir = ROOT / "probes" / cfg["probe"]
    probes = {f"p_seed{k}": load_file(pdir / f"seed{k}.safetensors", "cuda") for k in range(5)}
    rows, greedy, hidden = [], [], []
    bios = _bios(model, group)
    stats = Counter(bios=len(bios))
    for r in bios:
        text = results.get(r["name"])
        try:
            judged = judge.parse_bio(text) if text is not None else None
        except ValueError:  # includes json.JSONDecodeError
            judged = None
        if judged is None:
            stats["no_judgment"] += 1
            continue
        if not judged["entities"]:
            stats["no_entities"] += 1  # mostly "I don't know this person"
            continue
        anns, missed = judge.locate(r["bio"], judged["entities"])
        stats["entities_judged"] += len(judged["entities"])
        stats["entities_missed"] += missed
        truncated = r["new_tokens"] >= MAX_NEW_TOKENS
        if truncated:  # drop entities that touch the cut-off end
            anns = [a for a in anns if a["index"] + len(a["span"]) < len(r["bio"]) - 2]
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
            stats["no_spans"] += 1
            continue
        ids = torch.tensor([main.input_ids], device="cuda")
        with torch.no_grad():
            out = lm(ids, output_hidden_states=True)
        # Greedy decoding picked every answer token, so teacher forcing should reproduce it;
        # a low match rate would mean re-tokenizing the text changed the tokens.
        a, b = main.answer_start, len(main.input_ids) - 2  # answer tokens, without the end marker
        pred = out.logits[0, a - 1 : b - 1].argmax(-1)
        greedy.append(float((pred == ids[0, a:b]).float().mean()))
        x = torch.cat([out.hidden_states[layer + 1][0] for layer in cfg["layers"]], dim=-1).float()
        prob = {
            k: torch.sigmoid(x @ p["weight"] + p["bias"]).cpu().numpy() for k, p in probes.items()
        }
        nll, ent = token_baselines(out.logits[0], ids[0])
        mean_prob = np.mean(list(prob.values()), axis=0)
        for s, kind in spans:
            sl = slice(s.start, s.end + 1)
            row = {
                "name": r["name"], "start": s.start, "end": s.end, "kind": kind,
                "truncated": truncated,
                **{k: float(v[sl].max()) for k, v in prob.items()},
                "nll_max": float(np.nanmax(nll[sl])), "nll_mean": float(np.nanmean(nll[sl])),
                "entropy_max": float(np.nanmax(ent[sl])),
                "entropy_mean": float(np.nanmean(ent[sl])),
            }  # fmt: skip
            if features:
                row |= {
                    "v1_mean": float(mean_prob[sl].mean()), "v1_last": float(mean_prob[s.end]),
                    "n_tokens": s.end - s.start + 1,
                    "position": (s.start - a) / max(b - a, 1),
                }  # fmt: skip
                hidden.append(torch.stack([
                    out.hidden_states[layer + 1][0, sl].float().mean(0) for layer in FEATURE_LAYERS
                ]).half().cpu().numpy())  # fmt: skip
            rows.append(row)
        stats["scored"] += 1
    df = pd.DataFrame(rows)
    out_dir = ROOT / "gate" / model if not features else ROOT / "scores" / f"{group}_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "spans.parquet")
    if features:
        np.save(out_dir / "hidden.npy", np.stack(hidden))
    extra = dict(stats) | {
        "results": result_stats,
        "missed_fraction": round(stats["entities_missed"] / max(stats["entities_judged"], 1), 4),
        "greedy_match_mean": round(float(np.mean(greedy)), 4) if greedy else None,
        "greedy_match_below_0.95": int(sum(g < 0.95 for g in greedy)),
        "truncated_bios": int(sum(r["new_tokens"] >= MAX_NEW_TOKENS for r in bios)),
    }
    config = cfg | {"group": group, "feature_layers": FEATURE_LAYERS if features else None}
    write_run_meta(out_dir, job=f"bios/score/{model}", commit=commit, config=config, extra=extra)
    volume.commit()
    return extra | {"rows": len(df)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["fetch", "generate", "prepare", "score"])
    p.add_argument("args", nargs="*")
    p.add_argument("--group", default="gate", help="gate (FActScore's 500 people) or dev")
    p.add_argument("--features", action="store_true", help="score: also store detector features")
    a = p.parse_args()

    def fn(name):
        return modal.Function.from_name("hallustream-bios", name)

    if a.action == "fetch":
        print(json.dumps(fn("fetch").remote(), indent=1))
    elif a.action == "generate":
        calls = [fn("generate").spawn(m, k, a.group) for m in a.args for k in range(N_SHARDS)]
        print([c.get() for c in calls])
    elif a.action == "prepare":
        print(json.dumps(fn("prepare").remote(*a.args), indent=1))
    elif a.action == "score":
        calls = [fn("score").spawn(m, git_commit(), a.group, a.features) for m in a.args]
        print([c.get() for c in calls])
