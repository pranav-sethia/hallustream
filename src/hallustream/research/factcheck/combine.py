"""Own-answer detector candidates, chosen by cross-validation on dev people (EVAL_PROTOCOL.md).

v1 is the frozen probe: mean of the 5 seeds, max over the entity's tokens. Candidates:
  A  logistic regression on v1's logit and the model's confidence signals
  D  v1 with mean or last-token aggregation instead of max (nothing fitted)
  B  v1's logit plus a learned linear term on the entity's mean hidden state at one layer
  C  A plus the hidden-state term of B
Every model is a logistic regression with L2 penalties, fitted with L-BFGS.

Run: uv run python -m hallustream.research.factcheck.combine choose <dev_dir> <out.json>
     uv run python -m hallustream.research.factcheck.combine apply <detector.json> <scores_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from hallustream.research.metrics import auroc, paired_bootstrap

CONFIDENCE = ["nll_max", "nll_mean", "entropy_max", "entropy_mean"]
LAYERS = [15, 18, 21, 24, 27, 30]  # same order as modal_jobs/bios.FEATURE_LAYERS
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]
ORDER = ["A", "D_mean", "D_last", "B", "C"]  # simplest first
MIN_GAIN = 0.01
EPS = 1e-6


def load(d: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Supported / Not Supported rows of a scores dir, with their hidden states."""
    df = pd.read_parquet(d / "spans.parquet")
    hidden = np.load(d / "hidden.npy")
    keep = (df["kind"] != "insufficient").to_numpy()
    return df[keep].reset_index(drop=True), hidden[keep].astype(np.float32)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def v1_logit(df: pd.DataFrame, agg: str = "max") -> np.ndarray:
    if agg == "max":
        return _logit(df[[c for c in df.columns if c.startswith("p_seed")]].mean(axis=1).to_numpy())
    return _logit(df[f"v1_{agg}"].to_numpy())


def tabular(df: pd.DataFrame) -> np.ndarray:
    """Candidate A's inputs: v1 logit, confidence signals, log span length, position."""
    x = np.column_stack(
        [v1_logit(df), df[CONFIDENCE].to_numpy(), np.log(df["n_tokens"]), df["position"]]
    )
    return np.nan_to_num(x).astype(np.float32)


class Model:
    """Logistic regression: offset + bias + a . x + w . h, L2 on a and w (inputs standardized)."""

    def __init__(self, lam_x: float, lam_h: float, offset: bool):
        self.lam_x, self.lam_h, self.offset = lam_x, lam_h, offset

    @staticmethod
    def _std(z: np.ndarray | None):
        if z is None:
            return None
        return z.mean(0), z.std(0) + EPS

    def fit(self, y, x=None, h=None, off=None) -> Model:
        self.sx, self.sh = self._std(x), self._std(h)
        xs = self._scale(x, self.sx)
        hs = self._scale(h, self.sh)
        n = len(y)
        a = torch.zeros(0 if xs is None else xs.shape[1], requires_grad=True)
        w = torch.zeros(0 if hs is None else hs.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        yt = torch.tensor(y, dtype=torch.float32)
        ot = torch.tensor(off if self.offset else np.zeros(n), dtype=torch.float32)
        opt = torch.optim.LBFGS([a, w, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = ot + b + (xs @ a if xs is not None else 0) + (hs @ w if hs is not None else 0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt)
            loss = loss + self.lam_x * (a**2).sum() + self.lam_h * (w**2).sum() / max(len(w), 1)
            loss.backward()
            return loss

        opt.step(closure)
        self.a, self.w, self.b = a.detach(), w.detach(), b.detach()
        return self

    @staticmethod
    def _scale(z, s):
        return None if z is None else torch.tensor((z - s[0]) / s[1], dtype=torch.float32)

    def predict(self, x=None, h=None, off=None) -> np.ndarray:
        n = len(off if off is not None else (x if x is not None else h))
        z = torch.tensor(off if self.offset else np.zeros(n), dtype=torch.float32) + self.b
        if x is not None:
            z = z + self._scale(x, self.sx) @ self.a
        if h is not None:
            z = z + self._scale(h, self.sh) @ self.w
        return z.numpy()

    def to_json(self, **extra) -> dict:
        def arr(t):
            return [round(float(v), 6) for v in t]

        out = {"lam_x": self.lam_x, "lam_h": self.lam_h, "offset": self.offset,
               "a": arr(self.a), "w": arr(self.w), "b": float(self.b[0])}  # fmt: skip
        for name, s in (("x", self.sx), ("h", self.sh)):
            if s is not None:
                out[f"{name}_mean"], out[f"{name}_std"] = arr(s[0]), arr(s[1])
        return out | extra


def _inputs(cand: str, df: pd.DataFrame, hidden: np.ndarray, layer: int | None):
    h = hidden[:, LAYERS.index(layer)] if layer is not None else None
    if cand == "A":
        return tabular(df), None, None
    if cand == "B":
        return None, h, v1_logit(df)
    if cand == "C":
        return tabular(df), h, None
    raise ValueError(cand)


def _grid(cand: str):
    if cand == "A":
        return [(None, lx, 0.0) for lx in LAMBDAS]
    if cand == "B":
        return [(L, 0.0, lh) for L in LAYERS for lh in LAMBDAS]
    return [(L, lx, lh) for L in LAYERS for lx in LAMBDAS[1:3] for lh in LAMBDAS]


def _fit(cand, df, hidden, y, layer, lam_x, lam_h) -> Model:
    x, h, off = _inputs(cand, df, hidden, layer)
    return Model(lam_x, lam_h, offset=cand == "B").fit(y, x, h, off)


def _predict(m: Model, cand, df, hidden, layer) -> np.ndarray:
    x, h, off = _inputs(cand, df, hidden, layer)
    return m.predict(x, h, off)


def _tune(cand, df, hidden, y, groups, folds: int = 4) -> tuple:
    """Best (layer, lam_x, lam_h) by inner cross-validated AUROC."""
    best, best_auc = None, -1.0
    splits = list(GroupKFold(folds).split(df, y, groups))
    for params in _grid(cand):
        pred = np.zeros(len(y))
        for tr, te in splits:
            m = _fit(cand, df.iloc[tr], hidden[tr], y[tr], *params)
            pred[te] = _predict(m, cand, df.iloc[te], hidden[te], params[0])
        a = auroc(y, pred)
        if a > best_auc:
            best, best_auc = params, a
    return best


def out_of_fold(cand, df, hidden, y, groups, folds: int = 5) -> tuple[np.ndarray, list]:
    """Out-of-fold scores, with hyperparameters tuned inside each training fold."""
    if cand.startswith("D_"):
        return v1_logit(df, cand[2:]), []
    pred, chosen = np.zeros(len(y)), []
    for tr, te in GroupKFold(folds).split(df, y, groups):
        params = _tune(cand, df.iloc[tr], hidden[tr], y[tr], groups[tr])
        m = _fit(cand, df.iloc[tr], hidden[tr], y[tr], *params)
        pred[te] = _predict(m, cand, df.iloc[te], hidden[te], params[0])
        chosen.append(params)
    return pred, chosen


def choose(df: pd.DataFrame, hidden: np.ndarray) -> dict:
    y = (df["kind"] == "not_supported").to_numpy().astype(int)
    groups = df["name"].to_numpy()
    base = v1_logit(df)
    res = {"people": int(len(set(groups))), "entities": int(len(y)), "positives": int(y.sum()),
           "v1_auroc": round(auroc(y, base), 4),
           "baselines": {c: round(auroc(y, df[c].to_numpy()), 4) for c in CONFIDENCE},
           "candidates": {}}  # fmt: skip
    for cand in ORDER:
        pred, chosen = out_of_fold(cand, df, hidden, y, groups)
        gain = paired_bootstrap(y, pred, base, groups, n_boot=2000)
        res["candidates"][cand] = {
            "auroc": round(auroc(y, pred), 4),
            "gain": {k: round(v, 4) for k, v in gain.items()},
            "fold_params": [list(p) for p in chosen],
        }
    winner = next(
        (c for c in ORDER
         if res["candidates"][c]["gain"]["delta"] >= MIN_GAIN
         and res["candidates"][c]["gain"]["ci_low"] > 0),
        None,
    )  # fmt: skip
    res["chosen"] = winner or "v1"
    if winner and not winner.startswith("D_"):
        params = _tune(winner, df, hidden, y, groups, folds=5)
        m = _fit(winner, df, hidden, y, *params)
        res["detector"] = m.to_json(candidate=winner, layer=params[0])
    elif winner:
        res["detector"] = {"candidate": winner}
    return res


def apply(det: dict, df: pd.DataFrame, hidden: np.ndarray) -> np.ndarray:
    """Scores (logits) of a frozen detector from configs/detector_v2.json."""
    cand = det["candidate"]
    if cand.startswith("D_"):
        return v1_logit(df, cand[2:])
    m = Model(det["lam_x"], det["lam_h"], det["offset"])
    m.a, m.w, m.b = (torch.tensor(det[k], dtype=torch.float32) for k in ("a", "w", "b"))
    m.b = m.b.reshape(1)
    m.sx = (np.array(det["x_mean"]), np.array(det["x_std"])) if "x_mean" in det else None
    m.sh = (np.array(det["h_mean"]), np.array(det["h_std"])) if "h_mean" in det else None
    return _predict(m, cand, df, hidden, det.get("layer"))


if __name__ == "__main__":
    if sys.argv[1] == "choose":
        out = choose(*load(Path(sys.argv[2])))
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "detector"}, indent=1))
    else:
        det = json.loads(Path(sys.argv[2]).read_text())["detector"]
        df, hidden = load(Path(sys.argv[3]))
        df["v2"] = apply(det, df, hidden)
        df.to_parquet(Path(sys.argv[3]) / "spans_v2.parquet")
        print(f"wrote {len(df)} rows")
