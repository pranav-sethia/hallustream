"""LoRA probe runs (research/lora_probe.py) on Obeso et al.'s answers, with our splits.

Each run trains on a fixed random subset of the train split and scores a fixed random subset
of the validation split; the frozen v1 probe's scores on the same spans are already stored
(results/probes/qwen3_L21/spans_val.parquet), so the comparison is like for like.

  modal run modal_jobs/lora.py::smoke            # a few answers, end to end
  python -m modal_jobs.lora pilot                # spawns the pilot configs, prints call ids
"""

import json
import random
import time
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, git_commit, image, volume, write_run_meta

app = modal.App("hallustream-lora", image=image)
ROOT = Path(CACHE_DIR)
HF_ID = "Qwen/Qwen3-4B-Instruct-2507"
GPU = dict(gpu="L4", volumes={CACHE_DIR: volume}, timeout=3 * 60 * 60,
           secrets=[modal.Secret.from_name("huggingface")])  # fmt: skip
PILOT = {  # name -> config overrides; the pilot's three arms
    "head_L21": {"layer": 21, "lora": False},
    "lora_L21": {"layer": 21},
    "lora_L34": {"layer": 34},
}


def _subset(ids: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(ids), min(n, len(ids))))


@app.function(**GPU)
def run(name: str, overrides: dict, n_train: int, n_val: int, commit: str) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.align import align_exact
    from hallustream.research.data import load_records
    from hallustream.research.lora_probe import LoraConfig, LoraProbe, score, train
    from hallustream.research.metrics import auroc

    cfg = LoraConfig(**overrides)
    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    want = {"train": set(_subset(splits["train"], n_train, 0)),
            "val": set(_subset(splits["val"], n_val, 1))}  # fmt: skip
    records = {r.answer_id: r for r in load_records() if r.answer_id in want["train"] | want["val"]}
    tok = AutoTokenizer.from_pretrained(HF_ID)
    lm = AutoModelForCausalLM.from_pretrained(HF_ID, dtype=torch.bfloat16).to("cuda")
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    probe = LoraProbe(lm, cfg)

    def aligned(split):
        out = []
        for aid in sorted(want[split]):
            r = records[aid]
            a = align_exact(tok, r.messages, r.answer, r.annotations, max_length=cfg.max_length,
                            enable_thinking=False)  # fmt: skip
            out.append((aid, a))
        return out

    train_data = aligned("train")
    random.Random(cfg.seed).shuffle(train_data)
    tokens = sum(len(a.input_ids) for _, a in train_data)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    history = train(probe, [a for _, a in train_data], log_every=25)
    train_s = time.time() - t0
    out = ROOT / "lora" / name
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in probe.adapter_state().items()},
              str(out / "probe.safetensors"))  # fmt: skip

    rows, drift = [], []
    t1 = time.time()
    for aid, a in aligned("val"):
        s = score(probe, a)
        rows += [{"answer_id": aid, "span_id": i, "prob": p, "label": y} for i, p, y in s["spans"]]
        if "kl" in s:
            drift.append((s["kl"], s["same_top1"]))
    val_s = time.time() - t1
    df = pd.DataFrame(rows)
    df.to_parquet(out / "spans_val.parquet")
    summary = {
        "name": name, "config": cfg.dict(), "n_train": len(train_data),
        "n_val_answers": len(want["val"]), "val_spans": len(df),
        "val_auroc": round(auroc(df["label"].to_numpy(), df["prob"].to_numpy()), 4),
        "train_tokens": tokens, "train_seconds": round(train_s, 1),
        "train_tokens_per_s": round(tokens / train_s, 1), "val_seconds": round(val_s, 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "kl_mean": round(float(np.mean([d[0] for d in drift])), 5) if drift else None,
        "same_top1": round(float(np.mean([d[1] for d in drift])), 4) if drift else None,
    }  # fmt: skip
    (out / "history.json").write_text(json.dumps(history))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_run_meta(out, job="lora", commit=commit, config=summary["config"], extra=summary)
    volume.commit()
    return summary


@app.local_entrypoint()
def smoke():
    print(run.remote("smoke", {"layer": 21, "batch": 2}, 4, 3, git_commit()))


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["pilot"]:
        fn = modal.Function.from_name("hallustream-lora", "run")
        for name, ov in PILOT.items():
            print(name, fn.spawn(name, ov, 1000, 300, git_commit()).object_id)
