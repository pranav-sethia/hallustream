"""Train probes on the main feature caches (GPU), tune on validation, predict.

Deploy once:  uv run modal deploy modal_jobs/train.py
Then:         uv run python -m modal_jobs.train tune <variant>
              uv run python -m modal_jobs.train fit <variant> --config '{...}'
              uv run python -m modal_jobs.train predict <variant> --split test

Outputs go to /cache/probes/<variant>/ on the Volume:
  seed<k>.safetensors (weight/bias acting on raw hidden states), and
  spans_<split>.parquet / ordinary_<split>.parquet with one probability column per seed.
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, image, volume

app = modal.App("hallustream-train")

VARIANTS = {
    "qwen35_L17": dict(run="main2_qwen35", layers=[17]),
    "qwen35_L17_30": dict(run="main2_qwen35", layers=[17, 30]),
    "qwen3_L21": dict(run="main2_qwen3", layers=[21]),
    # head-to-head with the paper's Qwen2.5-7B probe: the sweep's best layer and the paper's layer
    "qwen25_L18": dict(run="main_qwen25", layers=[18]),
    "qwen25_L26": dict(run="main_qwen25", layers=[26]),
}
SEEDS = [0, 1, 2, 3, 4]
# Paper recipe plus the knobs the sweep showed matter (regularization), on validation only.
GRID = [
    dict(loss=loss, lr=lr, epochs=ep, l2=l2)
    for loss, lr, ep, l2 in itertools.product(
        ["annealed", "token"], [1e-3, 5e-3], [1, 3], [0.0, 1e-4, 1e-3, 1e-2]
    )
]
# The paper's annealed loss on 1,625 answers: more passes may be needed (EVAL_PROTOCOL.md).
GRID_QWEN25 = [
    dict(loss="annealed", lr=lr, epochs=ep, l2=l2)
    for lr, ep, l2 in itertools.product([1e-3, 5e-3], [4, 8, 16, 32], [0.0, 1e-3, 1e-2])
]
GPU = dict(gpu="L40S", memory=131072, volumes={CACHE_DIR: volume}, timeout=4 * 60 * 60)


def _load(variant: str, splits: list[str], exclude: frozenset[str]):
    """Rows, spans and features for the given splits, read chunk by chunk to save memory."""
    import numpy as np
    import pandas as pd

    from hallustream.research.layer_sweep import _chunks

    v = VARIANTS[variant]
    root = Path(CACHE_DIR) / "features" / v["run"]
    rows, spans, feats = [], [], []
    for shard, c in _chunks(root):
        r = pd.read_parquet(shard / f"rows_c{c}.parquet")
        s = pd.read_parquet(shard / f"spans_c{c}.parquet")
        keep = (r["split"].isin(splits) & ~r["answer_id"].isin(exclude)).to_numpy()
        x = np.hstack([np.load(shard / f"feat_L{layer}_c{c}.npy") for layer in v["layers"]])
        rows.append(r[keep])
        spans.append(s[s["split"].isin(splits) & ~s["answer_id"].isin(exclude)])
        feats.append(x[keep])
    rows = pd.concat(rows, ignore_index=True)
    return rows, pd.concat(spans, ignore_index=True), np.concatenate(feats)


def _data(variant, split, exclude, entity_weight=10.0):
    from hallustream.research.probe_train import build_data

    t0 = time.time()
    rows, spans, X = _load(variant, [split], exclude)
    d = build_data(rows, spans, X, entity_weight, device="cuda")
    print(f"{variant}/{split}: {len(rows):,} rows, {len(spans):,} spans ({time.time() - t0:.0f}s)")
    return d, rows, spans


def _scores(probe, d):
    from hallustream.research.metrics import auroc, recall_at_fpr
    from hallustream.research.probe_train import predict

    _, span_p, y = predict(probe, d)
    return {
        "val_auroc": round(auroc(y, span_p), 4),
        "val_recall": round(recall_at_fpr(y, span_p), 4),
    }


@app.function(image=image, **GPU)
def tune(variant: str, grid: list[dict], exclude: list[str]) -> list[dict]:
    from hallustream.research.probe_train import TrainConfig, feature_stats, train

    tr, _, _ = _data(variant, "train", frozenset(exclude))
    va, _, _ = _data(variant, "val", frozenset(exclude))
    stats = feature_stats(tr.X)
    out = []
    for g in grid:
        cfg = TrainConfig(**g)
        t0 = time.time()
        res = asdict(cfg) | _scores(train(tr, cfg, stats), va)
        res["seconds"] = round(time.time() - t0)
        print(json.dumps(res))
        out.append(res)
    return out


def _save_predictions(probes, d, rows, spans, out: Path, split: str):
    import numpy as np

    from hallustream.research.probe_train import predict

    keep = ["answer_id", "split", "source", "dataset", "subset", "span_id", "label", "n_tokens"]
    keep += ["nll_max", "nll_mean", "entropy_max", "entropy_mean"]  # confidence baselines
    span_df = spans[keep].copy()
    ord_mask = (rows["span_id"] < 0).to_numpy()
    ord_df = rows.loc[ord_mask, ["answer_id", "source", "dataset", "weight"]].copy()
    for k, probe in probes.items():
        row_p, span_p, y = predict(probe, d)
        assert np.array_equal(y, span_df["label"].to_numpy())
        span_df[f"p_seed{k}"] = span_p
        ord_df[f"p_seed{k}"] = row_p[ord_mask]
    span_df.to_parquet(out / f"spans_{split}.parquet")
    ord_df.to_parquet(out / f"ordinary_{split}.parquet")


@app.function(image=image, **GPU)
def fit(variant: str, config: dict, seeds: list[int], exclude: list[str], commit: str) -> dict:
    from safetensors.torch import save_file

    from hallustream.research.probe_train import TrainConfig, feature_stats, train

    out = Path(CACHE_DIR) / "probes" / variant
    out.mkdir(parents=True, exist_ok=True)
    tr, _, _ = _data(variant, "train", frozenset(exclude))
    va, va_rows, va_spans = _data(variant, "val", frozenset(exclude))
    stats = feature_stats(tr.X)
    probes, results = {}, []
    for k in seeds:
        cfg = TrainConfig(**(config | {"seed": k}))
        probes[k] = train(tr, cfg, stats)
        w, b = probes[k].export()
        save_file({"weight": w.contiguous(), "bias": b.reshape(1)}, out / f"seed{k}.safetensors")
        results.append({"seed": k} | _scores(probes[k], va))
        print(json.dumps(results[-1]))
    _save_predictions(probes, va, va_rows, va_spans, out, "val")
    meta = {"variant": VARIANTS[variant], "config": config, "seeds": results, "commit": commit}
    (out / "fit_meta.json").write_text(json.dumps(meta, indent=1))
    volume.commit()
    return meta


@app.function(image=image, **GPU)
def predict_split(variant: str, split: str, seeds: list[int]) -> str:
    import torch
    from safetensors.torch import load_file

    out = Path(CACHE_DIR) / "probes" / variant
    d, rows, spans = _data(variant, split, frozenset())

    class Raw(torch.nn.Module):  # shipped form: raw hidden states in, logits out
        def __init__(self, path):
            super().__init__()
            t = load_file(path, device="cuda")
            self.w, self.b = t["weight"], t["bias"]

        def forward(self, x):
            return x.float() @ self.w + self.b

    probes = {k: Raw(out / f"seed{k}.safetensors") for k in seeds}
    _save_predictions(probes, d, rows, spans, out, split)
    volume.commit()
    return str(out / f"spans_{split}.parquet")


def _get(call):
    while True:
        try:
            return call.get(timeout=60)
        except TimeoutError:
            continue
        except (modal.exception.RemoteError, modal.exception.ExecutionError):
            raise
        except Exception as e:  # local network blip
            print(f"retrying after {type(e).__name__}")
            time.sleep(10)


if __name__ == "__main__":
    from modal_jobs.common import git_commit

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["tune", "fit", "predict"])
    p.add_argument("variants", nargs="+", choices=list(VARIANTS))
    p.add_argument("--config", type=json.loads)
    p.add_argument("--split", default="test")
    a = p.parse_args()
    exclude = json.loads(Path("data/leak_check.json").read_text())["exclude"]
    fn = modal.Function.from_name(
        "hallustream-train", {"predict": "predict_split"}.get(a.action, a.action)
    )
    if a.action == "tune":
        calls = {v: fn.spawn(v, GRID_QWEN25 if v.startswith("qwen25") else GRID, exclude)
                 for v in a.variants}  # fmt: skip
    elif a.action == "fit":
        calls = {v: fn.spawn(v, a.config, SEEDS, exclude, git_commit()) for v in a.variants}
    else:
        calls = {v: fn.spawn(v, a.split, SEEDS) for v in a.variants}
    Path("results").mkdir(exist_ok=True)
    for v, c in calls.items():
        res = _get(c)
        Path(f"results/{a.action}_{v}.json").write_text(json.dumps(res, indent=1))
        print(v, "done")
