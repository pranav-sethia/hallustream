"""Detect-and-fix experiment on FActScore biographies (EVAL_PROTOCOL.md, detect-and-fix).

Deploy once:  uv run modal deploy modal_jobs/fix.py
Pilot (dev people):
              uv run python -m modal_jobs.fix pilot_triggers
              uv run python -m modal_jobs.fix retrieve_once dev
              uv run python -m modal_jobs.fix prepare_judge dev A1   # then judge_batch
              uv run python -m modal_jobs.fix pilot_rag
Arms:         uv run python -m modal_jobs.fix score_drafts <split> <A0|A1> <plain|evidence>
              uv run python -m hallustream.fix.experiment flags ... > flags.json (on the laptop)
              uv run python -m modal_jobs.fix rewrite <split> <base> <arm> <flags.json> [--policy]

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
    from hallustream.fix.engine import evidence_message as message
    from hallustream.research.factcheck.people import BIO_PROMPT

    return message(BIO_PROMPT.format(name=name), passages)


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

    volume.reload()  # a warm container may not see bios written by another function
    w = wiki.WikiDB(Path(DB))
    rows = [json.loads(x) for x in (ROOT / "fix" / split / arm / "bios.jsonl").read_text()
            .splitlines()]  # fmt: skip
    # an empty answer is an abstention and is not judged
    reqs = [judge.bio_request(r.get("key", r["name"]), r["name"],
                              wiki.evidence(w.passages(r["name"]), r["bio"]), r["bio"], setting)
            for r in rows if r["bio"].strip()]  # fmt: skip
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


@app.function(image=qwen35_image, **GPU)
def sample_pilot(commit: str, n_people: int = 60, n: int = 8, temperature: float = 0.8) -> dict:
    """Probe-reward pilot: n sampled biographies per dev person, each scored by the probe."""
    import numpy as np
    import pandas as pd
    import torch

    from hallustream.fix import score, spans
    from hallustream.research.factcheck.people import BIO_PROMPT

    tok, lm, det = _load()
    names = _split("dev")[:n_people]
    rows, t0 = [], time.time()
    torch.manual_seed(0)
    for name in names:
        msgs = [{"role": "user", "content": BIO_PROMPT.format(name=name)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT)
        enc = tok([prompt] * n, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                              temperature=temperature, top_p=0.8, top_k=20)  # fmt: skip
        for k, ids in enumerate(gen[:, enc["input_ids"].shape[1] :]):
            bio = tok.decode(ids, skip_special_tokens=True).strip()
            sents = spans.sentences(bio)
            found = spans.entities(bio, sents, threshold=0.3)
            fw = score.forward(lm, tok, det, msgs, bio, **CHAT)
            toks = [t for lo, hi in found if (t := score.char_to_tokens(fw, lo, hi))]
            _, v3 = det.span_scores(fw, toks)
            p = 1 / (1 + np.exp(-v3))
            surprise = [float(np.nanmax(fw.nll[a : b + 1])) for a, b in toks]
            entropy = [float(np.nanmax(fw.entropy[a : b + 1])) for a, b in toks]
            rows.append({
                "key": f"{name}#{k}", "name": name, "sample": k, "bio": bio,
                "new_tokens": int((ids != tok.pad_token_id).sum()), "entities": len(toks),
                "probe_p": p.tolist(), "surprise": surprise, "entropy": entropy,
            })  # fmt: skip
    out = ROOT / "fix" / "dev" / "pilot_samples"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bios.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pd.DataFrame(rows).drop(columns=["bio"]).to_parquet(out / "scores.parquet")
    meta = {"people": len(names), "samples": len(rows), "seconds": round(time.time() - t0)}
    write_run_meta(out, job="fix/sample_pilot", commit=commit,
                   config={"hf": HF, "n": n, "temperature": temperature}, extra=meta)  # fmt: skip
    volume.commit()
    return meta


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, timeout=30 * 60)
def prepare_writer(run: str, names: list[str], thinking: str = "minimal") -> dict:
    """Gemini batch requests that ask another model to write the biographies (reader study)."""
    from hallustream.research.factcheck.people import BIO_PROMPT

    config = {"maxOutputTokens": 4096, "thinkingConfig": {"thinkingLevel": thinking}}
    reqs = [{"key": n, "request": {
        "contents": [{"role": "user", "parts": [{"text": BIO_PROMPT.format(name=n)}]}],
        "generationConfig": config,
    }} for n in names]  # fmt: skip
    d = ROOT / "judge" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    volume.commit()
    return {"run": run, "requests": len(reqs)}


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, timeout=30 * 60)
def collect_writer(run: str, split: str, arm: str) -> dict:
    """Turn a writer batch's results into fix/<split>/<arm>/bios.jsonl."""
    from hallustream.research.factcheck.validate import read_results

    volume.reload()
    texts, stats = read_results(ROOT / "judge" / run / "results.jsonl")
    out = ROOT / "fix" / split / arm / "bios.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"name": n, "bio": t.strip()}) + "\n"
                           for n, t in texts.items()))  # fmt: skip
    volume.commit()
    return {"bios": len(texts), **stats}


@app.function(image=qwen35_image, **GPU)
def read_external(split: str, arm: str, commit: str) -> dict:
    """Reader mode: Qwen3-4B teacher-forces another model's biographies; v1 and baselines
    are scored on every entity the judge labeled."""
    import numpy as np
    import pandas as pd

    from hallustream.fix import score
    from hallustream.research.factcheck import judge
    from hallustream.research.factcheck.people import BIO_PROMPT

    volume.reload()
    tok, lm, det = _load()
    results = _judged(f"fix_{split}_{arm}")
    rows = []
    for x in (ROOT / "fix" / split / arm / "bios.jsonl").read_text().splitlines():
        r = json.loads(x)
        try:
            ents = judge.parse_bio(results[r["name"]])["entities"]
        except (KeyError, ValueError):
            continue
        anns, _ = judge.locate(r["bio"], ents)
        if not anns:
            continue
        msgs = [{"role": "user", "content": BIO_PROMPT.format(name=r["name"])}]
        fw = score.forward(lm, tok, det, msgs, r["bio"], **CHAT)
        toks = [score.char_to_tokens(fw, a["index"], a["index"] + len(a["span"])) for a in anns]
        keep = [(a, t) for a, t in zip(anns, toks, strict=True) if t]
        v1, _ = det.span_scores(fw, [t for _, t in keep])
        for (a, (lo, hi)), x1 in zip(keep, v1, strict=True):
            rows.append({"name": r["name"], "label": a["label"], "span": a["span"], "v1": float(x1),
                         "nll_max": float(np.nanmax(fw.nll[lo : hi + 1])),
                         "entropy_max": float(np.nanmax(fw.entropy[lo : hi + 1]))})  # fmt: skip
    df = pd.DataFrame(rows)
    out = ROOT / "fix" / split / arm
    df.to_parquet(out / "reader_entities.parquet")
    write_run_meta(out, job=f"fix/read_external/{split}/{arm}", commit=commit,
                   config={"reader": HF}, extra={"rows": len(df)})  # fmt: skip
    volume.commit()
    return {"rows": len(df), "people": int(df["name"].nunique()) if len(df) else 0}


def _drafts(split: str, base: str) -> dict:
    """Draft biographies an arm starts from: A0 = greedy, A1 = retrieve-once."""
    if base == "A0":
        bios = _greedy("")
        return {n: bios[n] for n in _split(split) if n in bios}
    path = ROOT / "fix" / split / base / "bios.jsonl"
    return {(r := json.loads(x))["name"]: r for x in path.read_text().splitlines()}


@app.function(image=qwen35_image, **GPU)
def score_drafts(split: str, base: str, context: str, commit: str) -> dict:
    """Trigger signals for every sentence of every draft (scored once, shared by all triggers).

    context: "plain" scores with the question only; "evidence" also puts the draft's
    retrieve-once passages in the prompt (A1 drafts only).
    """
    import pandas as pd

    from hallustream.fix import score, spans
    from hallustream.fix.engine import Draft
    from hallustream.research.factcheck.people import BIO_PROMPT

    tok, lm, det = _load()
    rows = []
    for name, r in _drafts(split, base).items():
        d = Draft.of(name, r["bio"])
        text = d.text()
        sents = spans.sentences(text)
        found = spans.entities(text, sents, threshold=0.3)
        content = (evidence_message(name, r["evidence"]) if context == "evidence"
                   else BIO_PROMPT.format(name=name))  # fmt: skip
        fw = score.forward(lm, tok, det, [{"role": "user", "content": content}], text, **CHAT)
        for i, (lo, hi) in enumerate(sents):
            sig = score.sentence_signals(fw, det, (lo, hi), found)
            rows.append({"name": name, "sentence": i, "start": lo, "end": hi,
                         "text": text[lo:hi], **sig})  # fmt: skip
    df = pd.DataFrame(rows)
    out = ROOT / "fix" / split / base
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"sentences_{context}.parquet")
    write_run_meta(out, job=f"fix/score_drafts/{split}/{base}/{context}", commit=commit,
                   config={"hf": HF}, extra={"sentences": len(df)})  # fmt: skip
    volume.commit()
    return {"sentences": len(df), "people": int(df["name"].nunique())}


@app.function(image=qwen35_image, **GPU)
def rewrite(
    split: str,
    base: str,
    arm: str,
    flags: dict,
    commit: str,
    k: int = 3,
    policy: str = "accept",
    threshold: float = 0.0,
    batch: int = 8,
) -> dict:
    """Rewrite each draft's flagged sentences in order, with top-k passages for that sentence.

    flags: name -> sentence indices to fix, or to drop if arm starts with "drop".
    policy: "accept" keeps every rewrite; "rescore" scores the new sentence with v3 and,
    if it is still >= threshold, retries once with 5 passages and drops it if still flagged.
    """
    import torch

    from hallustream.fix import score, spans
    from hallustream.fix.engine import REWRITE_MAX_TOKENS, Draft, first_sentence
    from hallustream.research.factcheck import wiki
    from hallustream.research.factcheck.people import BIO_PROMPT

    tok, lm, det = _load()
    w = wiki.WikiDB(Path(DB))
    drafts = {n: Draft.of(n, r["bio"]) for n, r in _drafts(split, base).items()}
    todo = {n: sorted(v) for n, v in flags.items() if n in drafts and v}
    stats = {"rewrites": 0, "retries": 0, "dropped": 0, "extra_tokens": 0}
    t0 = time.time()

    def passages(name, sentence, n):
        ps = w.passages(name)
        return [ps[j] for j in wiki.bm25_rank(f"{name} {sentence}", ps)[:n]]

    def generate(jobs):  # jobs: (name, index, passages) -> new sentence per job
        out = []
        for i in range(0, len(jobs), batch):
            chunk = jobs[i : i + batch]
            texts = [
                tok.apply_chat_template([{"role": "user", "content": evidence_message(n, ps)}],
                                        tokenize=False, add_generation_prompt=True, **CHAT)
                + drafts[n].text(upto=j)
                for n, j, ps in chunk
            ]  # fmt: skip
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to("cuda")
            with torch.no_grad():
                gen = lm.generate(**enc, max_new_tokens=REWRITE_MAX_TOKENS, do_sample=False)
            stats["extra_tokens"] += int(enc["attention_mask"].sum())
            for ids in gen[:, enc["input_ids"].shape[1] :]:
                out.append(first_sentence(tok.decode(ids, skip_special_tokens=True)))
        return out

    def still_flagged(name, j):
        d = drafts[name]
        text = d.text(upto=j) + d.fixed[j]
        lo = len(d.text(upto=j))
        found = spans.entities(text, [(lo, len(text))], threshold=0.3)
        msgs = [{"role": "user", "content": BIO_PROMPT.format(name=name)}]
        fw = score.forward(lm, tok, det, msgs, text, **CHAT)
        sig = score.sentence_signals(fw, det, (lo, len(text)), found)
        return bool(sig) and sig["v3_max"] >= threshold

    rounds = max((len(v) for v in todo.values()), default=0)
    for r in range(rounds):
        names = [n for n, v in todo.items() if len(v) > r]
        if arm.startswith("drop"):
            for n in names:
                drafts[n].fixed[todo[n][r]] = ""
                stats["dropped"] += 1
            continue
        jobs = [(n, todo[n][r], passages(n, drafts[n].sents[todo[n][r]], k)) for n in names]
        for (n, j, _), new in zip(jobs, generate(jobs), strict=True):
            drafts[n].fixed[j] = new
            stats["rewrites"] += 1
        if policy == "rescore":
            retry = [(n, j) for n, j, _ in jobs if still_flagged(n, j)]
            jobs = [(n, j, passages(n, drafts[n].sents[j], 5)) for n, j in retry]
            for (n, j, _), new in zip(jobs, generate(jobs), strict=True):
                drafts[n].fixed[j] = new
                stats["retries"] += 1
            for n, j, _ in jobs:
                if still_flagged(n, j):
                    drafts[n].fixed[j] = ""
                    stats["dropped"] += 1

    out = ROOT / "fix" / split / arm / "bios.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(
        json.dumps({"name": n, "bio": d.text(), "fixed": {str(i): t for i, t in d.fixed.items()},
                    "flags": todo.get(n, [])}) + "\n"
        for n, d in drafts.items()
    ))  # fmt: skip
    stats["seconds"] = round(time.time() - t0)
    config = {"hf": HF, "base": base, "k": k, "policy": policy, "threshold": threshold}
    write_run_meta(out.parent, job=f"fix/rewrite/{split}/{arm}", commit=commit, config=config,
                   extra=stats)  # fmt: skip
    volume.commit()
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action")
    p.add_argument("args", nargs="*")
    p.add_argument("--policy", default="accept")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args()
    fn = modal.Function.from_name("hallustream-fix", a.action)
    if a.action == "prepare_judge":
        print(fn.remote(*a.args))
    elif a.action == "retrieve_once":
        print(fn.remote(a.args[0], git_commit()))
    elif a.action == "score_drafts":  # <split> <base> <context>
        print(fn.remote(*a.args, git_commit()))
    elif a.action == "rewrite":  # <split> <base> <arm> <flags.json>
        split, base, arm, path = a.args
        flags = json.loads(Path(path).read_text())["flags"]
        print(fn.remote(split, base, arm, flags, git_commit(), k=a.k, policy=a.policy,
                        threshold=a.threshold))  # fmt: skip
    elif a.action == "prepare_writer":  # <run> <thinking> (people: FActScore dev + first 200 test)
        names = json.loads(Path(a.args[2]).read_text()) if len(a.args) > 2 else None
        print(fn.remote(a.args[0], names, a.args[1]))
    elif a.action == "collect_writer":  # <run> <split> <arm>
        print(fn.remote(*a.args))
    elif a.action == "read_external":  # <split> <arm>
        print(fn.remote(*a.args, git_commit()))
    else:
        print(fn.remote(git_commit()))
