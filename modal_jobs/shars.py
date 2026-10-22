"""SHARS (ICML 2026) with its own detector vs our probe, on Qwen3-4B-Instruct-2507.

SHARS writes a biography sentence by sentence; each sentence is split into (entity, claim)
propositions by the model itself, every proposition gets an uncertainty, and the sentence is
kept, rewritten from its trusted claims, or rejected. Only the uncertainty step differs:
  halluse  SHARS's detector: per proposition, write a question, sample answers, cluster with NLI
  probe    our v3 probe on the proposition's entity tokens (one forward pass per sentence)
  logprob  the model's own token surprise on the entity tokens (free baseline)

Their code is vendored in third_party/shars (see PATCHES.md) and runs on transformers 4.x.

Deploy:  uv run modal deploy modal_jobs/shars.py
Dev:     uv run python -m modal_jobs.shars dev_props      # detector comparison on fixed bios
Test:    uv run python -m modal_jobs.shars run <arm> <threshold> [--people N]
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
    """Our detectors as SHARS uncertainty estimators (same decomposition, different scores)."""
    import numpy as np

    from hallustream.fix import score
    from hallustream.research.factcheck.people import BIO_PROMPT

    _, uncertainty, _, _ = _shars()

    class ProbeEstimator(uncertainty.UncertaintyEstimator):
        """Uncertainty of a proposition = probe probability that its entity is hallucinated.

        context "native": the probe reads the text as it was trained (user turn
        `Tell me a bio of X.`, no system prompt); "shars": SHARS's own messages.
        signal "v3", "v1", or "surprise" (max token surprise, the free logprob baseline).
        """

        def __init__(self, signal="v3", context="native", **kwargs):
            super().__init__(**kwargs)
            self.signal, self.context = signal, context
            probes = [ROOT / "probes" / "qwen3_L21" / f"seed{k}.safetensors" for k in range(5)]
            self.det = score.Detector(probes, ROOT / "configs" / "detector_v3.json")

        def sentence_scores(self, user_query, text_sofar, sentence, query_entity, props):
            if self.context == "native":
                msgs = [{"role": "user", "content": BIO_PROMPT.format(name=query_entity)}]
            else:
                msgs = [
                    {"role": "system", "content": SYST},
                    {"role": "user", "content": user_query},
                ]
            text = text_sofar + sentence
            fw = score.forward(self.model.model, self.model.tokenizer, self.det, msgs, text,
                               enable_thinking=False)  # fmt: skip
            out = []
            for prop in props:
                lo = sentence.find(prop.entity)
                span = score.char_to_tokens(fw, len(text_sofar) + lo,
                                            len(text_sofar) + lo + len(prop.entity))  # fmt: skip
                if lo < 0 or span is None:
                    out.append(None)
                    continue
                v1, v3 = self.det.span_scores(fw, [span])
                a, b = span
                out.append({"v1": float(v1[0]), "v3": float(v3[0]),
                            "surprise": float(np.nanmax(fw.nll[a : b + 1])),
                            "entropy": float(np.nanmax(fw.entropy[a : b + 1]))})  # fmt: skip
            return out

        def get_sent_uncertainty(self, user_query, text_sofar, sentence, query_entity):
            props = self.get_sent_props(sentence, query_entity)
            scores = self.sentence_scores(user_query, text_sofar, sentence, query_entity, props)
            kept = []
            for prop, s in zip(props, scores, strict=True):
                if s is None:
                    continue
                x = s[self.signal]
                prop.uncertainty = x if self.signal == "surprise" else float(1 / (1 + np.exp(-x)))
                prop.qas = {}
                kept.append(prop)
            return kept

    return ProbeEstimator


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
    rows, bios, t_halluse, t_probe = [], [], 0.0, 0.0
    for name in names:
        query = QUERY.format(entity=name)
        bio = _generate("naive", model, generator, query, name, 0.0, 3, "native")
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
def run(
    arm: str,
    threshold: float,
    names: list[str],
    commit: str,
    tag: str = "",
    n_answers: int = 3,
    context: str = "native",
) -> dict:
    """SHARS end to end (Following policy, rewrite on, N = 10) with one detector."""
    LLM, uncertainty, generator, utils = _shars()
    model = LLM.get_model(MODEL, "cuda")
    import contextlib
    import io

    if arm != "naive":
        uncertainty.ESTIMATORS["probe"] = _probe_estimators()

    rows = []
    for name in names:
        query = QUERY.format(entity=name)
        t0 = time.time()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            text = _generate(arm, model, generator, query, name, threshold, n_answers, context)
        if utils.is_non_answer(text):
            text = ""
        aborted = "unexpected error" in log.getvalue()
        rows.append({"name": name, "bio": text, "seconds": round(time.time() - t0, 2),
                     "aborted_on_error": aborted, "log": log.getvalue()[-4000:]})  # fmt: skip
        print(f"{name}: {rows[-1]['seconds']}s, {len(text)} chars, error={aborted}")
    run_name = f"{arm}{tag}"
    for out in (ROOT / "shars" / "runs" / run_name, ROOT / "fix" / "shars" / run_name):
        out.mkdir(parents=True, exist_ok=True)
        (out / "bios.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = ROOT / "shars" / "runs" / run_name
    config = {"model": MODEL, "arm": arm, "threshold": threshold, "n_answers": n_answers,
              "context": context}  # fmt: skip
    errors = sum(r["aborted_on_error"] for r in rows)
    write_run_meta(out, job=f"shars/run/{run_name}", commit=commit, config=config,
                   extra={"people": len(rows), "aborted_on_error": errors})  # fmt: skip
    volume.commit()
    return {"run": run_name, "people": len(rows), "errors": errors,
            "seconds": round(sum(r["seconds"] for r in rows))}  # fmt: skip


def _generate(arm, model, generator, query, name, threshold, n_answers, context):
    if arm == "naive":
        gen = generator.NaiveGenerator(model, device="cuda")
        return gen.generate(user_prompt=query, syst_prompt=SYST, entity=name, thinking=False,
                            max_new_tokens=1024, temperatures=[0], top_k=20, top_p=0.8)  # fmt: skip
    est = "DeBERTa" if arm == "halluse" else "probe"
    signal = "v3" if arm == "probe" else "surprise"
    extra = ({"se_w_expected_answer": False} if arm == "halluse"
             else {"signal": signal, "context": context})  # fmt: skip
    gen = generator.UncertaintyGenerator(
        model, uncertainty_estimator=est, uncertainty_base_model=MODEL,
        uncertainty_model_thinking=False, n_questions=1, n_answers=n_answers,
        entailment_type="lax", uncertainty_threshold=threshold, max_resample_attempts=10,
        max_repetition_attempts=10, rewrite=True, resample_policy="natural", entail_check=False,
        device="cuda", temperatures=TEMPERATURES, top_k=20, top_p=0.8, **extra,
    )  # fmt: skip
    return gen.generate(
        user_prompt=query,
        syst_prompt=SYST,
        entity=name,
        thinking=False,
        max_new_tokens=1024,
        temperatures=TEMPERATURES,
        top_k=20,
        top_p=0.8,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["dev_props", "run"])
    p.add_argument("args", nargs="*")
    p.add_argument("--people", type=int, default=0, help="first N people only")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--n-answers", type=int, default=3)
    p.add_argument("--context", default="native")
    a = p.parse_args()
    if a.action == "dev_props":
        from hallustream.fix.spans import split_people
        from hallustream.research.factcheck.people import eval_people

        local = Path(a.args[0]) if a.args else None
        dev, _ = split_people(eval_people(local))
        names = dev[: a.people or 60]
        fn = modal.Function.from_name("hallustream-shars", "dev_props")
        calls = [fn.spawn(names[k :: a.shards], git_commit(), k) for k in range(a.shards)]
        print([c.get() for c in calls])
    else:
        arm, threshold = a.args[0], float(a.args[1])
        names = (PROJECT_ROOT / "third_party" / "shars" / "factscore_people.txt").read_text()
        names = [n.strip() for n in names.splitlines() if n.strip()]
        names = names[: a.people] if a.people else names
        fn = modal.Function.from_name("hallustream-shars", "run")
        print(fn.remote(arm, threshold, names, git_commit(), a.tag, a.n_answers, a.context))
