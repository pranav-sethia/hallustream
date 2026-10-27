"""SHARS (ICML 2026) with its own detector vs our probe, on Qwen3-4B-Instruct-2507.

SHARS writes a biography sentence by sentence; each sentence is split into (entity, claim)
propositions by the model itself, every proposition gets an uncertainty, and the sentence is
kept, rewritten from its trusted claims, or rejected. Only the uncertainty step differs:
  halluse  SHARS's detector: per proposition, write a question, sample answers, cluster with NLI
  probe    our v3 probe on the proposition's entity tokens (one forward pass per sentence)
  probe_v1 the probe before adaptation to the model's own answers (the paper's recipe)
  logprob  the model's own token surprise on the entity tokens (free baseline)

Their code is vendored in third_party/shars (see PATCHES.md) and runs on transformers 4.x.

Deploy:  uv run modal deploy modal_jobs/shars.py
Dev:     uv run python -m modal_jobs.shars dev_props      # detector comparison on fixed bios
Test:    uv run python -m modal_jobs.shars run <arm> <threshold> [--people N] [--shards K]
         uv run python -m modal_jobs.shars merge <arm><tag>       # after a sharded run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, PROJECT_ROOT, git_commit, volume, write_run_meta

app = modal.App("hallustream-shars")
ROOT = Path(CACHE_DIR)
MODEL = "qwen3-4b-instruct"  # Qwen/Qwen3-4B-Instruct-2507 in SHARS's LLM.MODELS
SHARS_DIR = "/root/shars"
# SHARS's own system prompt and query format (their main.py), used for every arm.
SYST = (
    "You are given a query from the user asking for information. Write the answer in English "
    "characters. Output plain text only. Do not use formatting styles, symbols, or headings. "
    "Each sentence should provide different information and must not repeat the same content. "
    "Output only the final answer to the query. Do not generate explanations, reasoning, or "
    "commentary for the answer. Do not ask follow-up questions."
)
QUERY = "Tell me about {entity}."
# Qwen3's recommended decoding (their paper): first attempt greedy, then the Following policy.
TEMPERATURES = [0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1",
        "transformers>=4.51,<5",
        "accelerate",
        "nltk",
        "addict",
        "wandb",
        "sentencepiece",
        "tiktoken",
        "protobuf",
        "safetensors",
        "numpy",
        "pandas",
        "pyarrow",
    )  # fmt: skip
    .env(
        {"HF_HOME": f"{CACHE_DIR}/hf", "TOKENIZERS_PARALLELISM": "false", "NLTK_DATA": "/root/nltk"}
    )
    .run_commands(
        "python -c \"import nltk; [nltk.download(p, download_dir='/root/nltk') "
        "for p in ('punkt', 'punkt_tab')]\""
    )  # fmt: skip
    .add_local_dir(str(PROJECT_ROOT / "third_party" / "shars"), SHARS_DIR)
    .add_local_python_source("hallustream", "modal_jobs", ignore=["**/__pycache__/**", "**/*.pyc"])
)
GPU = dict(gpu="L4", memory=32768, volumes={CACHE_DIR: volume}, timeout=6 * 60 * 60)


def _shars():
    if SHARS_DIR not in sys.path:
        sys.path.insert(0, SHARS_DIR)
    import generator
    import LLM
    import uncertainty
    import utils

    return LLM, uncertainty, generator, utils


def _probe_estimators():
    """Our detectors as SHARS uncertainty estimators (same decomposition, different scores),
    from the library, with the probe files of the runs reported in docs/RESULTS.md."""
    from hallustream.integrations.shars import estimator_class
    from hallustream.probe import Probe

    _, uncertainty, _, _ = _shars()
    probes = [ROOT / "probes" / "qwen3_L21" / f"seed{k}.safetensors" for k in range(5)]
    probe = Probe(probes, ROOT / "configs" / "detector_v3.json")
    return estimator_class(uncertainty, probe=probe, system_prompt=SYST)


def _halluse_answers(est, query, text_sofar, prop, n_answers):
    """SHARS's HalluSE uncertainty for one proposition, also returned for the first 3 answers.

    Runs their get_proposition_uncertainty with n_answers answers (Q=1) and recomputes the
    3-answer entropy from the first 3 of those answers with their own clustering and rules.
    """
    import numpy as np

    est.n_answers = n_answers
    u5, data = est.get_proposition_uncertainty(query, text_sofar, prop.fact_claim, prop.entity)
    out = {"halluse_a5": float(u5)}
    for q in data.values():
        answers = list(q.answers)[:3]
        _, u3 = est.get_semantic_uncertainty(answers)
        stop = ["not available", "not provided", "unknown", "unclear"]
        unknown = sum(any(w in a.lower() for w in stop) for a in answers)
        if unknown >= len(answers) // 2:
            u3 = -np.log(1 / len(answers))
        out["halluse_a3"] = float(u3)
        out["answers"] = list(q.answers)
    return out


@app.function(image=image, **GPU)
def dev_props(names: list[str], commit: str, shard: int = 0) -> dict:
    """Detector comparison on fixed greedy biographies (SHARS prompt), per proposition."""
    import pandas as pd

    LLM, uncertainty, generator, utils = _shars()
    model = LLM.get_model(MODEL, "cuda")
    common = dict(model=MODEL, n_questions=1, n_answers=5, entailment_type="lax",
                  temperature=0, top_k=20, top_p=0.8, thinking=False, device="cuda")  # fmt: skip
    halluse = uncertainty.QADebertaEntailment(se_w_expected_answer=False, **common)
    probe = _probe_estimators()(signal="v3", context="native", **common)
    naive = _generator("naive", model, generator)
    rows, bios, t_halluse, t_probe = [], [], 0.0, 0.0
    for name in names:
        query = QUERY.format(entity=name)
        bio = _generate(naive, query, name)
        bios.append({"name": name, "bio": bio, "non_answer": bool(utils.is_non_answer(bio))})
        if utils.is_non_answer(bio):
            continue
        sofar = ""
        for i, sent in enumerate(utils.split_sentences(bio)):
            props = halluse.get_sent_props(sent, name)
            t0 = time.time()
            nat = probe.sentence_scores(query, sofar, sent, name, props)
            probe.context = "shars"
            sh = probe.sentence_scores(query, sofar, sent, name, props)
            probe.context = "native"
            t_probe += time.time() - t0
            for prop, a, b in zip(props, nat, sh, strict=True):
                t0 = time.time()
                try:
                    h = _halluse_answers(halluse, query, sofar.strip(), prop, 5)
                except Exception as e:  # SHARS raises if no question could be generated
                    h = {"halluse_error": str(e)}
                t_halluse += time.time() - t0
                rows.append({
                    "name": name, "sentence_idx": i, "sentence": sent, "entity": prop.entity,
                    "fact_claim": prop.fact_claim, **h,
                    **{f"native_{k}": v for k, v in (a or {}).items()},
                    **{f"shars_{k}": v for k, v in (b or {}).items()},
                })  # fmt: skip
            sofar += sent if sofar.endswith((" ", "\n")) or not sofar else " " + sent
    out = ROOT / "shars" / "dev"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / f"props_{shard}.parquet")
    (out / f"bios_{shard}.jsonl").write_text("".join(json.dumps(b) + "\n" for b in bios))
    meta = {"people": len(names), "props": len(rows), "seconds_halluse": round(t_halluse),
            "seconds_probe": round(t_probe)}  # fmt: skip
    write_run_meta(out, job=f"shars/dev_props/{shard}", commit=commit, config={"model": MODEL},
                   extra=meta)  # fmt: skip
    volume.commit()
    return meta


@app.function(image=image, **GPU)
def parity(rows: list[dict]) -> dict:
    """Code check: the current estimator against the dev proposition scores stored by
    dev_props (native context). Each row carries the text written before its sentence."""
    from types import SimpleNamespace

    _shars()
    common = dict(model=MODEL, n_questions=1, n_answers=5, entailment_type="lax",
                  temperature=0, top_k=20, top_p=0.8, thinking=False, device="cuda")  # fmt: skip
    probe = _probe_estimators()(signal="v3", context="native", **common)
    worst, n = {"v1": 0.0, "v3": 0.0, "surprise": 0.0}, 0
    for r in rows:
        prop = SimpleNamespace(entity=r["entity"], fact_claim=r["fact_claim"])
        (s,) = probe.sentence_scores(QUERY.format(entity=r["name"]), r["sofar"], r["sentence"],
                                     r["name"], [prop])  # fmt: skip
        if s is None:
            continue
        n += 1
        for k in worst:
            worst[k] = max(worst[k], abs(s[k] - r[f"native_{k}"]))
    return {"props": n, "rows": len(rows), "max_abs_diff": worst}


@app.function(image=image, **GPU)
def rescore_halluse(commit: str) -> dict:
    """HalluSE's optional variants on the stored dev answers (no new sampling): entropy with the
    claimed entity added as an answer (se_w_expected_answer), and SHARS's entailment check
    (is the claimed entity in the largest answer cluster?), each for 3 and 5 answers."""
    import numpy as np
    import pandas as pd

    LLM, uncertainty, _, _ = _shars()
    common = dict(model=MODEL, n_questions=1, n_answers=5, entailment_type="lax",
                  temperature=0, top_k=20, top_p=0.8, thinking=False, device="cuda")  # fmt: skip
    est = uncertainty.QADebertaEntailment(se_w_expected_answer=False, **common)
    volume.reload()
    d = ROOT / "shars" / "dev"
    props = pd.read_parquet(d / "props_all.parquet")
    stop = ["not available", "not provided", "unknown", "unclear"]

    def entropy(answers):
        ids, u = est.get_semantic_uncertainty(answers)
        unknown = sum(any(w in a.lower() for w in stop) for a in answers)
        return (float(-np.log(1 / len(answers))) if unknown >= len(answers) // 2 else float(u)), ids

    out = []
    for _, r in props.iterrows():
        row = {}
        answers = list(r["answers"]) if r["answers"] is not None else []
        for n in (3, 5):
            a = answers[:n]
            if len(a) < n:
                row |= {f"exp_a{n}": np.nan, f"entailed_a{n}": np.nan, f"entailed_exp_a{n}": np.nan}
                continue
            row[f"exp_a{n}"], ids_exp = entropy([*a, r["entity"]])
            _, ids = entropy(a)
            entailed, _ = est.is_entailed_largest_cluster(r["entity"], a, ids)
            row[f"entailed_a{n}"] = float(entailed)
            # With the entity among the answers, SHARS checks against those clusters instead
            entailed, _ = est.is_entailed_largest_cluster(r["entity"], a, ids_exp)
            row[f"entailed_exp_a{n}"] = float(entailed)
        out.append(row)
    out = pd.DataFrame(out)
    props = props.drop(columns=[c for c in out.columns if c in props]).reset_index(drop=True)
    props = pd.concat([props, out], axis=1)
    props.to_parquet(d / "props_all.parquet")
    volume.commit()
    return {"props": len(props)}


DB = (
    "/cache/hf/hub/datasets--danielli2003--enwiki-20230401.db/snapshots/"
    "f15343f7209459595eb338bf54ae80b7bfdb86d2/enwiki-20230401.db"
)


@app.function(image=image, volumes={CACHE_DIR: volume}, timeout=30 * 60)
def prepare_prop_judge(setting: str = "t1_minimal") -> dict:
    """Claim-judge requests for every dev proposition: one request per person, claims numbered
    in table order (sentence, then proposition)."""
    import pandas as pd

    from hallustream.research.factcheck import judge, wiki

    volume.reload()
    d = ROOT / "shars" / "dev"
    shards = sorted(d.glob("props_[0-9]*.parquet"))  # the shards, not props_all itself
    props = pd.concat([pd.read_parquet(f) for f in shards])
    bios = {(r := json.loads(x))["name"]: r["bio"] for f in sorted(d.glob("bios_*.jsonl"))
            for x in f.read_text().splitlines()}  # fmt: skip
    w = wiki.WikiDB(Path(DB))
    reqs = []
    for name, g in props.groupby("name", sort=True):
        ev = wiki.evidence(w.passages(str(name)), bios[str(name)])
        reqs.append(
            judge.claim_request(str(name), str(name), ev, g["fact_claim"].tolist(), setting)
        )
    out = ROOT / "judge" / "shars_dev_props"
    out.mkdir(parents=True, exist_ok=True)
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    props.to_parquet(d / "props_all.parquet")
    volume.commit()
    return {"people": len(reqs), "props": len(props)}


@app.function(image=image, **GPU)
def run(
    arm: str,
    threshold: float,
    names: list[str],
    commit: str,
    tag: str = "",
    n_answers: int = 3,
    context: str = "native",
    se_expected: bool = False,
    entail_check: bool = False,
    part: int = -1,
    rewrite: bool = True,
) -> dict:
    """SHARS end to end (Following policy, N = 10, rewrite on unless turned off) with one
    detector. With part >= 0, writes only that slice of the people (combine with `merge`)."""
    LLM, uncertainty, generator, utils = _shars()
    model = LLM.get_model(MODEL, "cuda")
    import contextlib
    import io

    import torch

    if arm != "naive":
        uncertainty.ESTIMATORS["probe"] = _probe_estimators()

    timer = {"detect": 0.0, "decompose": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        gen = _generator(arm, model, generator, threshold, n_answers, context, se_expected,
                         entail_check, timer, rewrite)  # fmt: skip
    est = getattr(gen, "uct_estimator", None)
    rows = []
    for name in names:
        query = QUERY.format(entity=name)
        t0 = time.time()
        log = io.StringIO()
        timer.update(detect=0.0, decompose=0.0)
        if hasattr(est, "n_props"):
            est.n_props = est.n_unlocated = 0
        with contextlib.redirect_stdout(log):
            text = _generate(gen, query, name)
        if utils.is_non_answer(text):
            text = ""
        aborted = "unexpected error" in log.getvalue()
        rows.append({"name": name, "bio": text, "seconds": round(time.time() - t0, 2),
                     "detect_seconds": round(timer["detect"] - timer["decompose"], 2),
                     "decompose_seconds": round(timer["decompose"], 2),
                     "aborted_on_error": aborted, "log": log.getvalue(),
                     "props": getattr(est, "n_props", None),
                     "unlocated": getattr(est, "n_unlocated", None),
                     "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2)})  # fmt: skip
        print(f"{name}: {rows[-1]['seconds']}s, {len(text)} chars, error={aborted}, "
              f"detect {rows[-1]['detect_seconds']}s, gpu {rows[-1]['gpu_gb']} GB")  # fmt: skip
    run_name = f"{arm}{tag}"
    config = {"model": MODEL, "arm": arm, "threshold": threshold, "n_answers": n_answers,
              "context": context, "se_expected": se_expected,
              "entail_check": entail_check, "rewrite": rewrite}  # fmt: skip
    if part >= 0:
        out = ROOT / "shars" / "runs" / run_name / "parts"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{part}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / f"{part}.json").write_text(json.dumps({"commit": commit, "config": config}))
        volume.commit()
        return {"run": run_name, "part": part, "people": len(rows)}
    return _write_run(run_name, rows, commit, config)


def _write_run(run_name: str, rows: list[dict], commit: str, config: dict) -> dict:
    for out in (ROOT / "shars" / "runs" / run_name, ROOT / "fix" / "shars" / run_name):
        out.mkdir(parents=True, exist_ok=True)
        (out / "bios.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = ROOT / "shars" / "runs" / run_name
    errors = sum(r["aborted_on_error"] for r in rows)
    write_run_meta(out, job=f"shars/run/{run_name}", commit=commit, config=config,
                   extra={"people": len(rows), "aborted_on_error": errors})  # fmt: skip
    volume.commit()
    return {"run": run_name, "people": len(rows), "errors": errors,
            "seconds": round(sum(r["seconds"] for r in rows))}  # fmt: skip


@app.function(image=image, volumes={CACHE_DIR: volume}, timeout=10 * 60)
def merge(run_name: str, names: list[str]) -> dict:
    """Combine a run's slices into one bios.jsonl, in the order of `names`."""
    volume.reload()
    parts = ROOT / "shars" / "runs" / run_name / "parts"
    by_name, meta = {}, None
    for f in sorted(parts.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            by_name[r["name"]] = r
        meta = json.loads(f.with_suffix(".json").read_text())
    missing = [n for n in names if n not in by_name]
    if missing:
        return {"run": run_name, "missing": len(missing), "first_missing": missing[:5]}
    return _write_run(run_name, [by_name[n] for n in names], meta["commit"], meta["config"])


def _timed(obj, method, timer, key):
    """Add the time spent in obj.method to timer[key]."""
    f = getattr(obj, method)

    def wrapped(*args, **kwargs):
        t0 = time.time()
        try:
            return f(*args, **kwargs)
        finally:
            timer[key] += time.time() - t0

    setattr(obj, method, wrapped)


def _generator(arm, model, generator, threshold=0.0, n_answers=3, context="native",
               se_expected=False, entail_check=False, timer=None, rewrite=True):  # fmt: skip
    """One SHARS generator per run, reused for every person (as in SHARS's own main.py)."""
    if arm == "naive":
        return generator.NaiveGenerator(model, device="cuda")
    est = "DeBERTa" if arm == "halluse" else "probe"
    signal = {"probe": "v3", "probe_v1": "v1"}.get(arm, "surprise")
    extra = ({"se_w_expected_answer": se_expected} if arm == "halluse"
             else {"signal": signal, "context": context})  # fmt: skip
    gen = generator.UncertaintyGenerator(
        model, uncertainty_estimator=est, uncertainty_base_model=MODEL,
        uncertainty_model_thinking=False, n_questions=1, n_answers=n_answers,
        entailment_type="lax", uncertainty_threshold=threshold, max_resample_attempts=10,
        max_repetition_attempts=10, rewrite=rewrite, resample_policy="natural",
        entail_check=entail_check and arm == "halluse",
        device="cuda", temperatures=TEMPERATURES, top_k=20, top_p=0.8, **extra,
    )  # fmt: skip
    if timer is not None:
        # Detection = proposition scoring (incl. the entailment check); the shared
        # decomposition into propositions is timed separately and subtracted.
        est = gen.uct_estimator
        _timed(est, "get_sent_props", timer, "decompose")
        _timed(est, "get_sent_uncertainty", timer, "detect")
        if hasattr(est, "is_entailed_largest_cluster"):
            _timed(est, "is_entailed_largest_cluster", timer, "detect")
    return gen


def _generate(gen, query, name):
    temps = [0] if type(gen).__name__ == "NaiveGenerator" else TEMPERATURES
    return gen.generate(user_prompt=query, syst_prompt=SYST, entity=name, thinking=False,
                        max_new_tokens=1024, temperatures=temps, top_k=20, top_p=0.8)  # fmt: skip


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "action",
        choices=["dev_props", "run", "merge", "prepare_prop_judge", "rescore_halluse", "parity"],
    )
    p.add_argument("args", nargs="*")
    p.add_argument("--people", type=int, default=0, help="first N people only")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--n-answers", type=int, default=3)
    p.add_argument("--context", default="native")
    p.add_argument("--people-file", default="", help="JSON list of names (default: SHARS's 182)")
    p.add_argument("--se-expected", action="store_true")
    p.add_argument("--entail-check", action="store_true")
    p.add_argument("--no-rewrite", action="store_true")
    a = p.parse_args()
    if a.action == "parity":  # args: the local dev props table, number of people
        import pandas as pd

        props = pd.read_parquet(a.args[0]).drop_duplicates(["name", "sentence_idx", "entity"])
        rows = []
        for name in sorted(props["name"].unique())[: int(a.args[1])]:
            g = props[props["name"] == name].sort_values("sentence_idx")
            sents = dict(zip(g["sentence_idx"], g["sentence"], strict=False))
            for _, r in g.iterrows():
                k = int(r["sentence_idx"])
                if any(i not in sents for i in range(k)):
                    continue  # an earlier sentence had no propositions, so its text is unknown
                sofar = ""
                for i in range(k):  # the joining rule of dev_props
                    glue = "" if sofar.endswith((" ", "\n")) or not sofar else " "
                    sofar += glue + sents[i]
                rows.append(
                    {
                        c: r[c]
                        for c in [
                            "name",
                            "sentence",
                            "entity",
                            "fact_claim",
                            "native_v1",
                            "native_v3",
                            "native_surprise",
                        ]
                    }
                    | {"sofar": sofar}
                )
        print(modal.Function.from_name("hallustream-shars", "parity").remote(rows))
    elif a.action == "rescore_halluse":
        fn = modal.Function.from_name("hallustream-shars", "rescore_halluse")
        print(fn.spawn(git_commit()).object_id)
    elif a.action == "prepare_prop_judge":
        print(modal.Function.from_name("hallustream-shars", "prepare_prop_judge").remote())
    elif a.action == "dev_props":
        from hallustream.fix.spans import split_people
        from hallustream.research.factcheck.people import eval_people

        local = Path(a.args[0]) if a.args else None
        dev, _ = split_people(eval_people(local))
        names = dev[: a.people or 60]
        fn = modal.Function.from_name("hallustream-shars", "dev_props")
        calls = [fn.spawn(names[k :: a.shards], git_commit(), k) for k in range(a.shards)]
        print([c.get() for c in calls])
    else:
        if a.people_file:
            names = json.loads(Path(a.people_file).read_text())
        else:
            names = (PROJECT_ROOT / "third_party" / "shars" / "factscore_people.txt").read_text()
            names = [n.strip() for n in names.splitlines() if n.strip()]
        names = names[: a.people] if a.people else names
        if a.action == "merge":
            print(modal.Function.from_name("hallustream-shars", "merge").remote(a.args[0], names))
            sys.exit()
        arm, threshold = a.args[0], float(a.args[1])
        fn = modal.Function.from_name("hallustream-shars", "run")
        for k in range(a.shards):
            call = fn.spawn(arm, threshold, names[k :: a.shards], git_commit(), tag=a.tag,
                            n_answers=a.n_answers, context=a.context, se_expected=a.se_expected,
                            entail_check=a.entail_check, part=k if a.shards > 1 else -1,
                            rewrite=not a.no_rewrite)  # fmt: skip
            print(call.object_id)  # poll with modal.FunctionCall.from_id(...).get()
