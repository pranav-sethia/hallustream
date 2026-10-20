"""Score layer candidates (single layers or combinations) on a sweep cache, in parallel on CPU.

Deploy once:  uv run modal deploy modal_jobs/sweep.py
Run:          uv run python -m modal_jobs.sweep <run> singles
              uv run python -m modal_jobs.sweep <run> combos --spec combos.json
Results land in results/<run>_<name>.json (local, gitignored).
"""

import argparse
import json
import time
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, image, volume

app = modal.App("hallustream-sweep")
LEAK_CHECK = Path("data/leak_check.json")


@app.function(image=image, volumes={CACHE_DIR: volume}, cpu=4, memory=32768, timeout=2 * 60 * 60)
def score_candidate(
    run: str, layers: list[int], combine: str, exclude: list[str], C: float = 1.0
) -> dict:
    from hallustream.research.layer_sweep import evaluate

    t0 = time.time()
    res = evaluate(Path(CACHE_DIR) / "features" / run, layers, combine, frozenset(exclude), C)
    return res | {"seconds": round(time.time() - t0)}


@app.function(image=image, volumes={CACHE_DIR: volume}, memory=16384, timeout=30 * 60)
def score_baselines(run: str, exclude: list[str]) -> dict:
    from hallustream.research.layer_sweep import baselines

    return baselines(Path(CACHE_DIR) / "features" / run, frozenset(exclude))


def _get(call: modal.FunctionCall) -> dict:
    while True:  # tolerate local network blips; real job errors still raise
        try:
            return call.get(timeout=60)
        except TimeoutError:
            continue
        except (modal.exception.RemoteError, modal.exception.ExecutionError):
            raise
        except Exception as e:
            print(f"retrying after {type(e).__name__}")
            time.sleep(10)


def run_candidates(run: str, candidates: list[dict], name: str) -> Path:
    exclude = json.loads(LEAK_CHECK.read_text())["exclude"]
    fn = modal.Function.from_name("hallustream-sweep", "score_candidate")
    calls = [
        fn.spawn(run, c["layers"], c.get("combine", "concat"), exclude, c.get("C", 1.0))
        for c in candidates
    ]
    base = modal.Function.from_name("hallustream-sweep", "score_baselines").spawn(run, exclude)
    print(f"{run}: {len(calls)} candidates launched")
    results = []
    for c in calls:
        r = _get(c)
        print(f"  {r['combine']:6s} {r['layers']} C={r['C']}: {r['val_auroc']}  ({r['seconds']}s)")
        results.append(r)
    out = Path("results") / f"{run}_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"run": run, "baselines": _get(base), "candidates": results}))
    return out


if __name__ == "__main__":
    from modal_jobs.extract import RUNS

    p = argparse.ArgumentParser()
    p.add_argument("run", choices=list(RUNS))
    p.add_argument("name", help="'singles' scores every layer; anything else needs --spec")
    p.add_argument("--spec", type=Path, help="JSON list of {layers, combine, C}")
    a = p.parse_args()
    if a.name == "singles":
        cands = [{"layers": [layer]} for layer in RUNS[a.run]["layers"]]
    else:
        cands = json.loads(a.spec.read_text())
    print(run_candidates(a.run, cands, a.name))
