"""Probe evaluation, in two phases (EVAL_PROTOCOL.md).

Phase "val": everything that is chosen is chosen here, on validation only:
  layer choice for Qwen3.5 (rule fixed before training), Platt calibration,
  the fpr10 and guard thresholds, and the best confidence baseline.
  Writes results/frozen_choices.json, which is committed before the test set is scored.

Phase "test": scores the test set once with the frozen choices.
  Writes results/test_results.json.

Inputs are the prediction files written by modal_jobs/train.py, downloaded to
results/probes/<variant>/.

Run: uv run python -m hallustream.research.evaluate val
     uv run python -m hallustream.research.evaluate test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.metrics import auroc, paired_bootstrap, recall_at_fpr

PROBES = Path("results/probes")
FROZEN = Path("results/frozen_choices.json")
BASELINES = ["nll_max", "nll_mean", "entropy_max", "entropy_mean"]
SHIP_SEED = 0  # decided in advance: ship seed 0, report the spread over all seeds
GUARD_FPR = 0.02


def load(variant: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = PROBES / variant
    spans = pd.read_parquet(d / f"spans_{split}.parquet")
    ordinary = pd.read_parquet(d / f"ordinary_{split}.parquet")
    return spans, ordinary


def seeds(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("p_seed"))


def question_groups(answer_ids: pd.Series) -> np.ndarray:
    from hallustream.research.data import load_records, question_key

    q = {r.answer_id: question_key(r.question) for r in load_records()}
    return answer_ids.map(q).to_numpy()


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6).fit(_logit(p)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    return 1 / (1 + np.exp(-(ab[0] * _logit(p) + ab[1])))


def threshold_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false-alarm rate on negatives is at most `fpr`."""
    neg = np.sort(scores[y == 0])
    return float(neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))])


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
            for b in range(bins) if (idx == b).any())
    )  # fmt: skip


def operating_point(p, y, thr) -> dict:
    flagged = p >= thr
    tp = int((flagged & (y == 1)).sum())
    return {
        "recall": round(tp / max(int((y == 1).sum()), 1), 4),
        "precision": round(tp / max(int(flagged.sum()), 1), 4),
        "fpr": round(float((flagged & (y == 0)).sum() / max(int((y == 0).sum()), 1)), 4),
    }


# --- phase 1: validation choices -------------------------------------------------------


def choose_qwen35_layers(groups: np.ndarray) -> dict:
    a, _ = load("qwen35_L17", "val")
    b, _ = load("qwen35_L17_30", "val")
    assert (a[["answer_id", "span_id"]].to_numpy() == b[["answer_id", "span_id"]].to_numpy()).all()
    y, col = a["label"].to_numpy(), f"p_seed{SHIP_SEED}"
    bs = paired_bootstrap(y, b[col].to_numpy(), a[col].to_numpy(), groups)
    rec_a, rec_b = recall_at_fpr(y, a[col].to_numpy()), recall_at_fpr(y, b[col].to_numpy())
    use_b = bs["ci_low"] > 0 and rec_b >= rec_a
    return {
        "auroc_A_L17": round(auroc(y, a[col].to_numpy()), 4),
        "auroc_B_L17_30": round(auroc(y, b[col].to_numpy()), 4),
        "B_minus_A": {k: round(v, 4) for k, v in bs.items()},
        "recall10_A": round(rec_a, 4),
        "recall10_B": round(rec_b, 4),
        "chosen": "qwen35_L17_30" if use_b else "qwen35_L17",
    }


def freeze(variant: str, spans: pd.DataFrame) -> dict:
    y = spans["label"].to_numpy()
    p = spans[f"p_seed{SHIP_SEED}"].to_numpy()
    platt = fit_platt(p, y)
    pc = apply_platt(p, platt)
    base = {c: round(auroc(y, spans[c].to_numpy()), 4) for c in BASELINES}
    return {
        "variant": variant,
        "platt": platt,
        "threshold_fpr10": threshold_at_fpr(pc, y, 0.10),
        "threshold_guard": threshold_at_fpr(pc, y, GUARD_FPR),
        "best_baseline": max(base, key=lambda k: base[k]),
        "val": {
            "auroc_by_seed": {s: round(auroc(y, spans[s].to_numpy()), 4) for s in seeds(spans)},
            "ece_raw": round(ece(p, y), 4),
            "ece_calibrated": round(ece(pc, y), 4),
            "baselines": base,
        },
    }


def phase_val() -> dict:
    a, _ = load("qwen35_L17", "val")
    groups = question_groups(a["answer_id"])
    layer = choose_qwen35_layers(groups)
    out = {"layer_choice": layer, "models": {}}
    for v in (layer["chosen"], "qwen3_L21"):
        spans, _ = load(v, "val")
        out["models"][v] = freeze(v, spans)
    # copied to configs/ and committed before phase "test"
    FROZEN.write_text(json.dumps(out, indent=1))
    return out


# --- phase 2: test, once --------------------------------------------------------------


def score_block(spans: pd.DataFrame, ordinary: pd.DataFrame, fz: dict) -> dict:
    y = spans["label"].to_numpy()
    col = f"p_seed{SHIP_SEED}"
    pc = apply_platt(spans[col].to_numpy(), tuple(fz["platt"]))
    from sklearn.metrics import average_precision_score

    ord_p = apply_platt(ordinary[col].to_numpy(), tuple(fz["platt"]))
    w = ordinary["weight"].to_numpy()
    alarm = float((w * (ord_p >= fz["threshold_fpr10"])).sum() / w.sum())
    by_seed = [auroc(y, spans[s].to_numpy()) for s in seeds(spans)]
    return {
        "n_spans": int(len(y)),
        "auroc": round(auroc(y, pc), 4),
        "auroc_seeds_mean": round(float(np.mean(by_seed)), 4),
        "auroc_seeds_std": round(float(np.std(by_seed)), 4),
        "auprc": round(float(average_precision_score(y, pc)), 4),
        "ece": round(ece(pc, y), 4),
        "at_fpr10_threshold": operating_point(pc, y, fz["threshold_fpr10"]),
        "at_guard_threshold": operating_point(pc, y, fz["threshold_guard"]),
        "ordinary_token_alarm_rate": round(alarm, 4),
        "baseline_auroc": round(auroc(y, spans[fz["best_baseline"]].to_numpy()), 4),
    }  # fmt: skip


def phase_test() -> dict:
    frozen = json.loads(FROZEN.read_text())
    out: dict = {"frozen_from": str(FROZEN), "models": {}}
    data = {v: load(v, "test") for v in frozen["models"]}
    first = next(iter(data.values()))[0]
    groups = question_groups(first["answer_id"])
    for v, fz in frozen["models"].items():
        spans, ordinary = data[v]
        y = spans["label"].to_numpy()
        res = {"overall": score_block(spans, ordinary, fz)}
        for key in ("dataset", "source"):
            for val in sorted(spans[key].unique()):
                m_s, m_o = spans[key] == val, ordinary[key] == val
                res[f"{key}={val}"] = score_block(spans[m_s], ordinary[m_o], fz)
        probe, base = spans[f"p_seed{SHIP_SEED}"].to_numpy(), spans[fz["best_baseline"]].to_numpy()
        bs = paired_bootstrap(y, probe, base, groups)
        res["probe_minus_best_baseline"] = {k: round(x, 4) for k, x in bs.items()}
        out["models"][v] = res
    (v35, s35), (v3, s3) = [(v, data[v][0]) for v in frozen["models"]]
    assert (
        s35[["answer_id", "span_id"]].to_numpy() == s3[["answer_id", "span_id"]].to_numpy()
    ).all()
    col = f"p_seed{SHIP_SEED}"
    out["qwen35_minus_qwen3"] = {
        k: round(x, 4)
        for k, x in paired_bootstrap(
            s35["label"].to_numpy(), s35[col].to_numpy(), s3[col].to_numpy(), groups
        ).items()
    }
    Path("results/test_results.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = phase_val() if sys.argv[1] == "val" else phase_test()
    print(json.dumps(res, indent=1))
