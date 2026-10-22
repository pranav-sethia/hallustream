"""Probe-reward pilot analysis (EVAL_PROTOCOL.md, 'Probe-reward pilot').

Run: uv run python -m hallustream.fix.pilot_reward <dir with bios.jsonl and results.jsonl>
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.factcheck import judge
from hallustream.research.factcheck.validate import read_results

d = Path(sys.argv[1])
rows = [json.loads(x) for x in (d / "bios.jsonl").read_text().splitlines()]
res, stats = read_results(d / "results.jsonl")
# confidence signals mapped to [0,1] by rank over all entities in the pilot
for sig in ("surprise", "entropy"):
    allv = np.concatenate([r[sig] for r in rows if r[sig]])
    for r in rows:
        r[f"{sig}_p"] = [float(np.searchsorted(np.sort(allv), v) / len(allv)) for v in r[sig]]


def reward(p):
    p = np.asarray(p, float)
    T = len(p)
    F = float((1 - p).sum())
    return F / (T + 1) + 0.5 * np.log1p(F)


recs = []
for r in rows:
    try:
        claims = judge.parse_bio(res[r["key"]])["claims"]
    except (KeyError, ValueError):
        continue
    sup = sum(l == "Supported" for _, l in claims)
    n = len(claims)
    recs.append(
        {
            "name": r["name"],
            "key": r["key"],
            "claims": n,
            "supported": sup,
            "unsupported": n - sup,
            "precision": sup / n if n else np.nan,
            "entities": r["entities"],
            "tokens": r["new_tokens"],
            "r_probe": reward(r["probe_p"]),
            "r_surprise": reward(r["surprise_p"]),
            "r_entropy": reward(r["entropy_p"]),
            "m_probe": -np.mean(r["probe_p"]) if r["probe_p"] else 0.0,
            "m_surprise": -np.mean(r["surprise"]) if r["surprise"] else 0.0,
            "m_entropy": -np.mean(r["entropy"]) if r["entropy"] else 0.0,
        }
    )
df = pd.DataFrame(recs)
print(
    "judge results:",
    stats,
    "| samples judged:",
    len(df),
    "| with claims:",
    int((df.claims > 0).sum()),
)
ans = df[df.claims > 0]
# 1. diversity
g = ans.groupby("name").precision.agg(["max", "min", "size"])
g = g[g["size"] >= 2]
div = float(((g["max"] - g["min"]) >= 0.2).mean())
print(
    f"1. diversity: {div:.3f} of {len(g)} people have best-worst precision gap >= 0.2  (pass >= 0.50)"
)
# 2. within-person pair accuracy
sigs = ["r_probe", "r_surprise", "r_entropy", "m_probe", "m_surprise", "m_entropy"]
acc = {s: [] for s in sigs}
for _, grp in ans.groupby("name"):
    for (_, a), (_, b) in itertools.combinations(grp.iterrows(), 2):
        gap = a.precision - b.precision
        if abs(gap) < 0.2:
            continue
        for s in sigs:
            dd = a[s] - b[s]
            acc[s].append(0.5 if dd == 0 else float(np.sign(dd) == np.sign(gap)))
print(
    f"2. within-person pair accuracy ({len(acc['r_probe'])} pairs):",
    {s: round(np.mean(v), 3) for s, v in acc.items()},
)
best_conf = max(np.mean(acc[s]) for s in ["r_surprise", "r_entropy", "m_surprise", "m_entropy"])
print(
    f"   probe reward {np.mean(acc['r_probe']):.3f} vs best confidence {best_conf:.3f}  (pass >= 0.70 and +0.05)"
)
# 3. best-of-8 by probe reward vs a random sample (all samples, abstentions count as 0 correct claims)
bo, rnd = [], []
for name, grp in df.groupby("name"):
    best = grp.loc[grp.r_probe.idxmax()]
    bo.append((best.precision, best.supported, best.unsupported))
    rnd.append((grp[grp.claims > 0].precision.mean(), grp.supported.mean(), grp.unsupported.mean()))
bo, rnd = np.array(bo, float), np.array(rnd, float)
print(
    f"3. best-of-8 by probe: precision {np.nanmean(bo[:, 0]):.3f} vs random {np.nanmean(rnd[:, 0]):.3f}; "
    f"correct claims {bo[:, 1].mean():.2f} vs {rnd[:, 1].mean():.2f} ({bo[:, 1].mean() / rnd[:, 1].mean() - 1:+.1%}); "
    f"incorrect {bo[:, 2].mean():.2f} vs {rnd[:, 2].mean():.2f} ({bo[:, 2].mean() / rnd[:, 2].mean() - 1:+.1%})"
)
df.to_csv(d / "pilot_table.csv", index=False)
