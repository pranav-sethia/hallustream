"""Checks of the library on Qwen3-4B-Instruct-2507 (docs/LIBRARY.md).

  parity     the library's scores equal the stored research scores on the topic texts
  automatic  how well it ranks entities it finds itself (GLiNER), on held-out texts whose
             entities the judge labeled: 150 topic texts and 150 greedy SHARS-test biographies
  speed      ms per generated token: plain decoding, with the live probe, and with the probe
             plus GLiNER at each sentence end

Run: uv run python -m modal_jobs.library_check
"""

import json
import time
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, git_commit, image, volume, write_run_meta

app = modal.App("hallustream-library")
ROOT = Path(CACHE_DIR)
GPU = dict(gpu="L4", memory=32768, volumes={CACHE_DIR: volume}, timeout=60 * 60)
# SHARS's system prompt and query (modal_jobs/shars.py), for its greedy test biographies
SHARS_SYST = (
    "You are given a query from the user asking for information. Write the answer in English "
    "characters. Output plain text only. Do not use formatting styles, symbols, or headings. "
    "Each sentence should provide different information and must not repeat the same content. "
    "Output only the final answer to the query. Do not generate explanations, reasoning, or "
    "commentary for the answer. Do not ask follow-up questions."
)
SHARS_QUERY = "Tell me about {entity}."


def _plain(det, prompt: str, max_new_tokens: int) -> int:
    """Greedy decoding with a KV cache and nothing else: the loop stream() adds the probe to."""
    import torch
    from transformers import DynamicCache

    tok, lm = det.tokenizer, det.model
    msgs = [{"role": "user", "content": prompt}]
    prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **det.chat)
    ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(lm.device)
    eos = lm.generation_config.eos_token_id
    stop = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    cache = DynamicCache()
    with torch.no_grad():
        out = lm(ids, past_key_values=cache, use_cache=True)
        n = 0
        for _ in range(max_new_tokens):
            nxt = int(out.logits[0, -1].argmax())
            if nxt in stop:
                break
            n += 1
            out = lm(torch.tensor([[nxt]], device=lm.device), past_key_values=cache, use_cache=True)
    return n


@app.function(image=image, **GPU)
def check(commit: str, n_parity: int = 60, n_auto: int = 150, n_speed: int = 8,
          max_new_tokens: int = 200) -> dict:  # fmt: skip
    import random

    import pandas as pd
    import torch

    from hallustream import Detector
    from hallustream.probe import forward
    from hallustream.research.factcheck import judge
    from hallustream.research.factcheck.topics import TOPIC_PROMPT
    from hallustream.research.factcheck.validate import read_results

    volume.reload()
    det = Detector.from_pretrained("qwen3-4b", device="cuda")
    rng = random.Random(0)

    # 1. Parity on the topic texts, using the stored token spans of every judged entity.
    topic_bios = {}
    for f in sorted((ROOT / "topics" / "bios" / "qwen3").glob("shard*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            topic_bios[r["name"]] = r["bio"]
    spans = pd.read_parquet(ROOT / "scores" / "topics_qwen3" / "spans.parquet")
    parity = []
    for name in rng.sample(sorted(spans["name"].unique()), n_parity):
        msgs = [{"role": "user", "content": TOPIC_PROMPT.format(name=name)}]
        fw = forward(det.model, det.tokenizer, det.probe, msgs, topic_bios[name], **det.chat)
        for r in spans[spans["name"] == name].itertuples():
            v1, v3 = det.probe.span_scores(fw, [(int(r.start), int(r.end))])  # pyright: ignore
            parity.append({"name": name, "start": int(r.start), "end": int(r.end),  # pyright: ignore
                           "lib_v1": float(v1[0]), "lib_v3": float(v3[0])})  # fmt: skip

    # 2. Automatic mode: GLiNER entities, labeled by the judge's entities they overlap.
    sets = {}
    topic_results, _ = read_results(ROOT / "judge" / "topics_qwen3" / "results.jsonl")
    names = rng.sample(sorted(n for n in topic_bios if n in topic_results), n_auto)
    sets["topics"] = [(n, [{"role": "user", "content": TOPIC_PROMPT.format(name=n)}],
                       topic_bios[n], topic_results[n]) for n in names]  # fmt: skip
    bio_results, _ = read_results(ROOT / "judge" / "fix_shars_naive" / "results.jsonl")
    lines = (ROOT / "fix" / "shars" / "naive" / "bios.jsonl").read_text().splitlines()
    shars_bios = {(r := json.loads(x))["name"]: r["bio"] for x in lines}
    pool = sorted(n for n, b in shars_bios.items() if b.strip() and n in bio_results)
    names = rng.sample(pool, min(n_auto, len(pool)))
    sets["biographies"] = [(n, [{"role": "system", "content": SHARS_SYST},
                                {"role": "user", "content": SHARS_QUERY.format(entity=n)}],
                            shars_bios[n], bio_results[n]) for n in names]  # fmt: skip
    auto, coverage = [], []
    for kind, items in sets.items():
        for name, msgs, text, judged in items:
            try:
                anns, _ = judge.locate(text, judge.parse_bio(judged)["entities"])
            except ValueError:
                continue
            ents = det.score(messages=msgs, text=text).entities
            for e in ents:
                over = [a for a in anns if a["index"] < e.end and
                        e.start < a["index"] + len(a["span"])]  # fmt: skip
                labels = {a["label"] for a in over}
                auto.append({"set": kind, "name": name, "start": e.start, "end": e.end,
                             "risk": e.risk, "labels": sorted(labels)})  # fmt: skip
            for a in anns:
                if a["label"] in ("Supported", "Not Supported"):
                    hit = any(e.start < a["index"] + len(a["span"]) and e.end > a["index"]
                              for e in ents)  # fmt: skip
                    coverage.append({"set": kind, "label": a["label"], "found": hit})

    # 3. Speed: the same greedy tokens three ways; warm up each first.
    quiet = Detector(det.model, det.tokenizer, det.probe, det.threshold, det.chat,
                     det.thresholds, entity_finder=lambda text, sents: [])  # fmt: skip

    def streamed(d):
        return lambda p: len(list(d.stream(p, max_new_tokens=max_new_tokens))[-1].ids)

    modes = {"plain": lambda p: _plain(det, p, max_new_tokens),
             "probe": streamed(quiet), "probe_gliner": streamed(det)}  # fmt: skip
    prompts = [TOPIC_PROMPT.format(name=n) for n in sorted(topic_bios)[:n_speed]]
    speed = {}
    for mode, fn in modes.items():
        fn(prompts[0])
        torch.cuda.synchronize()
        t0, toks = time.perf_counter(), 0
        for p in prompts:
            toks += fn(p)
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / toks
        speed[mode] = {"tokens": toks, "ms_per_token": round(ms, 2)}

    out = ROOT / "library_check"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(parity).to_parquet(out / "parity.parquet")
    pd.DataFrame(auto).to_parquet(out / "auto.parquet")
    pd.DataFrame(coverage).to_parquet(out / "coverage.parquet")
    (out / "speed.json").write_text(json.dumps(speed, indent=1))
    write_run_meta(out, job="library_check", commit=commit,
                   config={"n_parity": n_parity, "n_auto": n_auto, "n_speed": n_speed,
                           "max_new_tokens": max_new_tokens},
                   extra={"parity_rows": len(parity), "auto_rows": len(auto)})  # fmt: skip
    volume.commit()
    return {"parity_rows": len(parity), "auto_rows": len(auto), "speed": speed}


@app.function(image=image, **GPU)
def showcase(prompts: list[str], max_new_tokens: int = 220) -> list[dict]:
    """Record real stream() runs (text, checked position, entities, time) for the README GIF."""
    from hallustream import Detector

    det = Detector.from_pretrained("qwen3-4b", device="cuda")
    list(det.stream("Hi.", max_new_tokens=5))  # warm up
    runs = []
    for prompt in prompts:
        t0, events = time.perf_counter(), []
        for ev in det.stream(prompt, max_new_tokens=max_new_tokens):
            ents = [e.__dict__ for e in ev.entities]
            events.append({"t": round(time.perf_counter() - t0, 3), "text": ev.text,
                           "checked": ev.checked, "entities": ents})  # fmt: skip
        runs.append({"prompt": prompt, "events": events})
    return runs


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["showcase"]:
        runs = modal.Function.from_name("hallustream-library", "showcase").remote(sys.argv[2:])
        Path("results/library/showcase.json").write_text(json.dumps(runs, indent=1))
        sys.exit()
    fn = modal.Function.from_name("hallustream-library", "check")
    small = sys.argv[1:] == ["--smoke"]  # 3 texts per check and 20 tokens, to test the job
    kwargs = dict(n_parity=3, n_auto=3, n_speed=1, max_new_tokens=20) if small else {}
    print(fn.spawn(git_commit(), **kwargs).object_id)
