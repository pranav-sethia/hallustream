import numpy as np
import pandas as pd

from hallustream.research.factcheck import combine


def _dev(n_people=60, per=8, seed=0):
    """Synthetic dev scores: v1 is weakly informative, entropy adds signal, hidden dim 0 too."""
    rng = np.random.default_rng(seed)
    rows, hidden = [], []
    for p in range(n_people):
        for _ in range(per):
            y = rng.integers(0, 2)
            h = rng.normal(size=(len(combine.LAYERS), 8)).astype(np.float32)
            h[2, 0] += 1.5 * y  # signal only at layer 21
            p_v1 = 1 / (1 + np.exp(-(0.8 * y + rng.normal())))
            rows.append({
                "name": f"p{p}", "kind": "not_supported" if y else "supported",
                **{f"p_seed{k}": p_v1 for k in range(5)},
                "v1_mean": p_v1, "v1_last": 1 / (1 + np.exp(-rng.normal())),
                "nll_max": rng.normal() + y, "nll_mean": rng.normal(), "entropy_max": rng.normal(),
                "entropy_mean": rng.normal(), "n_tokens": int(rng.integers(1, 5)),
                "position": rng.random(),
            })  # fmt: skip
            hidden.append(h)
    return pd.DataFrame(rows), np.stack(hidden)


def test_choose_finds_the_extra_signal_and_apply_reproduces_the_fit():
    df, hidden = _dev()
    res = combine.choose(df, hidden)
    assert res["chosen"] in {"A", "B", "C"}
    assert res["candidates"][res["chosen"]]["auroc"] > res["v1_auroc"] + 0.05
    s = combine.apply(res["detector"], df, hidden)
    y = (df["kind"] == "not_supported").to_numpy()
    assert s[y].mean() > s[~y].mean()


def test_choose_keeps_v1_when_nothing_helps():
    df, hidden = _dev(seed=1)
    df[["nll_max"]] = np.random.default_rng(2).normal(size=(len(df), 1))
    hidden[:, 2, 0] = np.random.default_rng(3).normal(size=len(df))
    res = combine.choose(df, hidden)
    assert res["chosen"] == "v1" and "detector" not in res
