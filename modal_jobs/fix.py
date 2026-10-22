"""Detect-and-fix experiment on FActScore biographies (EVAL_PROTOCOL.md, detect-and-fix).

Deploy once:  uv run modal deploy modal_jobs/fix.py
Pilot (dev people):
              uv run python -m modal_jobs.fix pilot_triggers
              uv run python -m modal_jobs.fix retrieve_once dev
              uv run python -m modal_jobs.fix prepare_judge dev A1   # then judge_batch
              uv run python -m modal_jobs.fix pilot_rag

Volume layout (/cache):
  configs/detector_v3.json            frozen detector (uploaded from configs/)
  fix/<split>/<arm>/bios.jsonl        biographies per arm
  fix/pilot/*.parquet                 pilot measurements
  judge/fix_<split>_<arm>/            judge requests and results
"""

import argparse
import json
import time
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, git_commit, qwen35_image, volume, write_run_meta

app = modal.App("hallustream-fix")
ROOT = Path(CACHE_DIR)
HF = "Qwen/Qwen3-4B-Instruct-2507"
PROBE = "qwen3_L21"
CHAT = dict(enable_thinking=False)
MAX_NEW_TOKENS = 512
DB = (
    "/cache/hf/hub/datasets--danielli2003--enwiki-20230401.db/snapshots/"
    "f15343f7209459595eb338bf54ae80b7bfdb86d2/enwiki-20230401.db"
)
GPU = dict(gpu="L4", memory=32768, volumes={CACHE_DIR: volume}, timeout=3 * 60 * 60)


def _split(which: str) -> list[str]:
    from hallustream.fix.spans import split_people
    from hallustream.research.factcheck.people import eval_people

    dev, test = split_people(eval_people(ROOT / "factscore" / "data"))
    return dev if which == "dev" else test


def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.fix.score import Detector

    tok = AutoTokenizer.from_pretrained(HF, padding_side="left")
    lm = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16).to("cuda").eval()
    probes = [ROOT / "probes" / PROBE / f"seed{k}.safetensors" for k in range(5)]
    return tok, lm, Detector(probes, ROOT / "configs" / "detector_v3.json")


def _greedy(name: str) -> dict:
    rows = [json.loads(x) for k in range(4) for x in (ROOT / "bios" / "qwen3" / f"shard{k}.jsonl")
            .read_text().splitlines()]  # fmt: skip
    return {r["name"]: r for r in rows}[name] if name else {r["name"]: r for r in rows}


def _judged(run: str) -> dict:
    from hallustream.research.factcheck.validate import read_results

    return read_results(ROOT / "judge" / run / "results.jsonl")[0]


@app.function(image=qwen35_image, **GPU)
def pilot_triggers(commit: str) -> dict:
    """Per-sentence trigger signals on the dev people's greedy biographies, with labels."""
    import pandas as pd

    from hallustream.fix import score, spans
    from hallustream.research.factcheck import judge
    from hallustream.research.factcheck.people import BIO_PROMPT

    tok, lm, det = _load()
    bios, results = _greedy(""), _judged("gate_qwen3")
    rows = []
    for name in _split("dev"):
        if name not in results:
            continue
        bio = bios[name]["bio"]
        try:
            ents = judge.parse_bio(results[name])["entities"]
        except ValueError:
            continue
        if not ents:
            continue
        anns, _ = judge.locate(bio, ents)
        sents = spans.sentences(bio)
        found = spans.entities(bio, sents, threshold=0.3)
        msgs = [{"role": "user", "content": BIO_PROMPT.format(name=name)}]
        fw = score.forward(lm, tok, det, msgs, bio, **CHAT)
        for i, (lo, hi) in enumerate(sents):
            sig = score.sentence_signals(fw, det, (lo, hi), found)
            if not sig:
                continue
            inside = [a for a in anns if lo <= a["index"] < hi]
            rows.append({
                "name": name, "sentence": i, "start": lo, "end": hi,
                "has_not_supported": any(a["label"] == "Not Supported" for a in inside),
                "has_judged_entity": any(a["label"] != "Insufficient" for a in inside),
                **sig,
            })  # fmt: skip
    df = pd.DataFrame(rows)
    out = ROOT / "fix" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "sentences.parquet")
    write_run_meta(out, job="fix/pilot_triggers", commit=commit, config={"hf": HF}, extra={})
    volume.commit()
    return {"sentences": len(df), "people": int(df["name"].nunique())}


def evidence_message(name: str, passages: list[str]) -> str:
    from hallustream.research.factcheck.people import BIO_PROMPT

    listed = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return (f"{BIO_PROMPT.format(name=name)}\n\n"
            f"Here are passages from Wikipedia that may help:\n\n{listed}")  # fmt: skip


@app.function(image=qwen35_image, **GPU)
def retrieve_once(split: str, commit: str, k: int = 5, batch: int = 8) -> dict:
    """Arm A1: the top-k BM25 passages of the person's page (query = name) in the user turn."""
    import torch

    from hallustream.research.factcheck import wiki

    tok, lm, _ = _load()
    w = wiki.WikiDB(Path(DB))
    out = ROOT / "fix" / split / "A1" / "bios.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    names = [n for n in _split(split) if _has_page(w, n)]
    for i in range(0, len(names), batch):
        chunk = names[i : i + batch]
        ev = {n: [w.passages(n)[j] for j in wiki.bm25_rank(n, w.passages(n))[:k]] for n in chunk}
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": evidence_message(n, ev[n])}],
                                    tokenize=False, add_generation_prompt=True, **CHAT)
            for n in chunk
        ]  # fmt: skip
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        for n, ids in zip(chunk, gen[:, enc["input_ids"].shape[1] :], strict=True):
            rows.append({"name": n, "bio": tok.decode(ids, skip_special_tokens=True).strip(),
                         "new_tokens": int((ids != tok.pad_token_id).sum()),
                         "evidence": ev[n]})  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {"seconds": round(time.time() - t0)}
    job = f"fix/retrieve_once/{split}"
    write_run_meta(out.parent, job=job, commit=commit, config={"hf": HF, "k": k}, extra=meta)
    volume.commit()
    return {"bios": len(rows), "seconds": round(time.time() - t0)}


def _has_page(w, name: str) -> bool:
    try:
        w.passages(name)
        return True
    except KeyError:
        return False


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, timeout=30 * 60)
def prepare_judge(split: str, arm: str, setting: str = "t1_minimal") -> dict:
    from hallustream.research.factcheck import judge, wiki

    w = wiki.WikiDB(Path(DB))
    rows = [json.loads(x) for x in (ROOT / "fix" / split / arm / "bios.jsonl").read_text()
            .splitlines()]  # fmt: skip
    reqs = [judge.bio_request(r["name"], r["name"], wiki.evidence(w.passages(r["name"]), r["bio"]),
                              r["bio"], setting) for r in rows]  # fmt: skip
    d = ROOT / "judge" / f"fix_{split}_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    volume.commit()
    return {"run": d.name, "requests": len(reqs)}


@app.function(image=qwen35_image, **GPU)
def pilot_rag(commit: str) -> dict:
    """Probe scores on dev retrieve-once biographies, with and without the evidence in context."""
    import pandas as pd

    from hallustream.fix import score
    from hallustream.research.factcheck import judge
    from hallustream.research.factcheck.people import BIO_PROMPT

    tok, lm, det = _load()
    results = _judged("fix_dev_A1")
    rows = []
    bios = [json.loads(x) for x in (ROOT / "fix" / "dev" / "A1" / "bios.jsonl").read_text()
            .splitlines()]  # fmt: skip
    for r in bios:
        try:
            ents = judge.parse_bio(results[r["name"]])["entities"]
        except (KeyError, ValueError):
            continue
        anns, _ = judge.locate(r["bio"], ents)
        anns = [a for a in anns if a["label"] != "Insufficient"]
        if not anns:
            continue
        for ctx in ("plain", "evidence"):
            content = (evidence_message(r["name"], r["evidence"]) if ctx == "evidence"
                       else BIO_PROMPT.format(name=r["name"]))  # fmt: skip
            fw = score.forward(lm, tok, det, [{"role": "user", "content": content}], r["bio"],
                               **CHAT)  # fmt: skip
            toks = [score.char_to_tokens(fw, a["index"], a["index"] + len(a["span"])) for a in anns]
            keep = [(a, t) for a, t in zip(anns, toks, strict=True) if t]
            v1, v3 = det.span_scores(fw, [t for _, t in keep])
            for (a, _), x1, x3 in zip(keep, v1, v3, strict=True):
                rows.append({"name": r["name"], "context": ctx, "span": a["span"],
                             "not_supported": a["label"] == "Not Supported",
                             "v1": float(x1), "v3": float(x3)})  # fmt: skip
    df = pd.DataFrame(rows)
    out = ROOT / "fix" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "rag_entities.parquet")
    write_run_meta(out, job="fix/pilot_rag", commit=commit, config={"hf": HF}, extra={})
    volume.commit()
    return {"rows": len(df)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    fn = modal.Function.from_name("hallustream-fix", a.action)
    if a.action == "prepare_judge":
        print(fn.remote(*a.args))
    elif a.action == "retrieve_once":
        print(fn.remote(a.args[0], git_commit()))
    else:
        print(fn.remote(git_commit()))
