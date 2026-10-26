"""How well risk (sigmoid of the v3 score) matches the share of entities that are false.

Run: uv run python -m hallustream.research.calibration results/reserved results/topics/scores
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from hallustream.research.factcheck import combine

V3 = Path(__file__).parents[3] / "configs" / "detector_v3.json"


def calibration(d: Path, bins: int = 10) -> dict:
    spans, hidden = combine.load(d)
    risk = 1 / (1 + np.exp(-combine.apply(json.loads(V3.read_text())["detector"], spans, hidden)))
    false = (spans["kind"] != "supported").to_numpy().astype(float)
    idx = np.minimum((risk * bins).astype(int), bins - 1)
    rows = [{"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int((idx == b).sum()),
             "mean_risk": round(float(risk[idx == b].mean()), 3),
             "false_share": round(float(false[idx == b].mean()), 3)}
            for b in range(bins) if (idx == b).any()]  # fmt: skip
    ece = sum(r["n"] * abs(r["mean_risk"] - r["false_share"]) for r in rows) / len(risk)
    return {"entities": len(risk), "expected_calibration_error": round(ece, 3), "bins": rows}


if __name__ == "__main__":
    print(json.dumps({d: calibration(Path(d)) for d in sys.argv[1:]}, indent=1))
