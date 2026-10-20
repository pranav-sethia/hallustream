"""Step 4: teacher-forced feature extraction on Modal (sharded, resumable).

Deploy once:  uv run modal deploy modal_jobs/extract.py
Launch:       uv run python -m modal_jobs.launch extract <run>     (see RUNS below)
Wait:         uv run python -m modal_jobs.launch wait <run>

Each shard writes chunks of CHUNK answers to /cache/features/<run>/shard<k>/:
  feat_L<layer>_c<j>.npy (float16), rows_c<j>.parquet, spans_c<j>.parquet
Finished chunks are skipped on re-run, so an interrupted job just resumes.
"""

import hashlib
import json
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, git_commit, qwen35_image, volume, write_run_meta

app = modal.App("hallustream-extract")

CHUNK = 250
N_NEG = 50
QWEN35, QWEN3 = "Qwen/Qwen3.5-4B", "Qwen/Qwen3-4B-Instruct-2507"
RUNS = {
    # layer sweep: 1,200 train + 800 val answers, several layers
    "sweep_qwen35": dict(model=QWEN35, layers=[15, 18, 19, 22, 23, 26, 27, 30], subset="sweep"),
    "sweep_qwen3": dict(model=QWEN3, layers=[18, 22, 26, 30, 34], subset="sweep"),
    # main cache: every train/val/test answer at the chosen layer (filled in after the sweep)
    "main_qwen35": dict(model=QWEN35, layers=None, subset="all"),
    "main_qwen3": dict(model=QWEN3, layers=None, subset="all"),
    # tiny end-to-end check
    "smoke": dict(model=QWEN35, layers=[26, 30], subset="smoke"),
}
N_SHARDS = {"sweep": 4, "all": 8, "smoke": 1}


def select(subset: str):
    """Deterministic (record, split) list for a subset."""
    import random

    from hallustream.research.data import SOURCES, load_records, make_splits

    recs = load_records()
    split_of = {}
    res = make_splits(recs, train_sources=SOURCES)
    for name, ids in (("train", res.train), ("val", res.val), ("test", res.test)):
        split_of.update({i: name for i in ids})
    chosen = [(r, split_of[r.answer_id]) for r in recs if r.answer_id in split_of]
    if subset == "all":
        return chosen
    rng = random.Random(0)
    train = [x for x in chosen if x[1] == "train"]
    val = [x for x in chosen if x[1] == "val"]
    if subset == "sweep":
        return rng.sample(train, 1200) + rng.sample(val, 800)
    return rng.sample(train, 10) + rng.sample(val, 10)


def answer_seed(answer_id: str) -> int:
    return int(hashlib.sha1(answer_id.encode()).hexdigest()[:8], 16)


@app.function(
    image=qwen35_image,
    gpu="L4",
    memory=32768,
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=4 * 60 * 60,
)
def extract_shard(run: str, shard: int, n_shards: int, layers: list[int], commit: str) -> dict:
    import time

    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hallustream.research.extract import extract_answer

    cfg = RUNS[run]
    items = [x for i, x in enumerate(select(cfg["subset"])) if i % n_shards == shard]
    out = Path(CACHE_DIR) / "features" / run / f"shard{shard:02d}"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    t0, done = time.time(), 0
    for j in range(0, len(items), CHUNK):
        c = j // CHUNK
        if (out / f"spans_c{c:03d}.parquet").exists():
            continue  # finished in an earlier attempt
        rows, spans, feats = [], [], {layer: [] for layer in layers}
        for rec, split in items[j : j + CHUNK]:
            rng = np.random.default_rng(answer_seed(rec.answer_id))
            f = extract_answer(model, tok, rec, layers, N_NEG, rng, split, enable_thinking=False)
            rows.append(pd.DataFrame(f.rows))
            spans.extend(f.spans)
            for layer in layers:
                feats[layer].append(f.features[layer])
        for layer in layers:
            np.save(out / f"feat_L{layer}_c{c:03d}.npy", np.concatenate(feats[layer]))
        pd.concat(rows).to_parquet(out / f"rows_c{c:03d}.parquet")
        pd.DataFrame(spans).to_parquet(out / f"spans_c{c:03d}.parquet")  # written last = done
        volume.commit()
        done += len(items[j : j + CHUNK])
        print(f"shard {shard}: {j + CHUNK}/{len(items)} answers, {time.time() - t0:.0f}s")

    write_run_meta(
        out,
        job=f"extract/{run}",
        commit=commit,
        config={**cfg, "layers": layers, "n_neg": N_NEG, "shard": shard, "n_shards": n_shards},
        extra={"answers": len(items), "answers_this_attempt": done},
    )
    volume.commit()
    return {"shard": shard, "answers": len(items), "seconds": round(time.time() - t0)}


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, timeout=10 * 60)
def run_status(run: str) -> dict:
    root = Path(CACHE_DIR) / "features" / run
    shards = sorted(root.glob("shard*")) if root.exists() else []
    return {
        s.name: {
            "chunks_done": len(list(s.glob("spans_c*.parquet"))),
            "finished": (s / "run_meta.json").exists(),
        }
        for s in shards
    }


@app.function(image=qwen35_image, volumes={CACHE_DIR: volume}, memory=16384, timeout=60 * 60)
def check_run(run: str) -> dict:
    """Sanity-check a finished cache: counts per split, label balance, finite features."""
    import numpy as np
    import pandas as pd

    root = Path(CACHE_DIR) / "features" / run
    n_rows = n_spans = 0
    answers: dict[str, set] = {}
    pos_rows = pos_spans = bad = 0
    mismatched = []
    for shard in sorted(root.glob("shard*")):
        layers = json.loads((shard / "run_meta.json").read_text())["config"]["layers"]
        for sp in sorted(shard.glob("spans_c*.parquet")):
            c = sp.stem.split("_c")[1]
            rows = pd.read_parquet(shard / f"rows_c{c}.parquet")
            spans = pd.read_parquet(sp)
            for layer in layers:
                X = np.load(shard / f"feat_L{layer}_c{c}.npy", mmap_mode="r")
                if X.shape[0] != len(rows):
                    mismatched.append(f"{shard.name}/c{c}/L{layer}")
                bad += int((~np.isfinite(X)).any(axis=1).sum())
            n_rows += len(rows)
            n_spans += len(spans)
            pos_rows += int((rows["label"] == 1).sum())
            pos_spans += int((spans["label"] == 1).sum())
            for split, ids in rows.groupby("split")["answer_id"]:
                answers.setdefault(split, set()).update(ids)
    return {
        "answers": {k: len(v) for k, v in answers.items()},
        "rows": n_rows,
        "spans": n_spans,
        "row_pos_rate": round(pos_rows / n_rows, 4),
        "span_pos_rate": round(pos_spans / n_spans, 4),
        "rows_with_nonfinite": bad,
        "shape_mismatches": mismatched,
    }


if __name__ == "__main__":
    print(
        json.dumps({k: {**v, "shards": N_SHARDS[v["subset"]]} for k, v in RUNS.items()}, indent=1)
    )
    print(git_commit())
