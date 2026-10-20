"""Start deployed Modal jobs server-side and wait for them, tolerant of local network drops.

uv run python -m modal_jobs.launch extract <run> [--layers 26,30]
uv run python -m modal_jobs.launch wait <run>
uv run python -m modal_jobs.launch status <run>
"""

import argparse
import json
import time
from pathlib import Path

import modal

from modal_jobs.common import git_commit
from modal_jobs.extract import N_SHARDS, RUNS

CALLS_DIR = Path("data/calls")


def launch(run: str, layers: list[int] | None) -> None:
    cfg = RUNS[run]
    layers = layers or cfg["layers"]
    if not layers:
        raise SystemExit(f"{run} needs --layers (chosen from the sweep)")
    fn = modal.Function.from_name("hallustream-extract", "extract_shard")
    n = N_SHARDS[cfg["subset"]]
    commit = git_commit()
    calls = [fn.spawn(run, k, n, layers, commit).object_id for k in range(n)]
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    (CALLS_DIR / f"{run}.json").write_text(json.dumps({"layers": layers, "calls": calls}))
    print(f"launched {n} shards for {run} with layers {layers}")


def wait(run: str) -> None:
    calls = json.loads((CALLS_DIR / f"{run}.json").read_text())["calls"]
    pending = dict(enumerate(calls))
    while pending:
        for k, cid in list(pending.items()):
            try:
                print(json.dumps(modal.FunctionCall.from_id(cid).get(timeout=30)))
                del pending[k]
            except TimeoutError:
                pass
            except (modal.exception.RemoteError, modal.exception.ExecutionError) as e:
                print(f"shard {k} FAILED: {str(e)[:200]}")  # a real job failure
                del pending[k]
            except Exception as e:  # local network blip: retry
                print(f"shard {k}: retrying after {type(e).__name__}")
                time.sleep(10)
    print("all shards returned")


def status(run: str) -> None:
    fn = modal.Function.from_name("hallustream-extract", "run_status")
    print(json.dumps(fn.remote(run), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["extract", "wait", "status"])
    p.add_argument("run", choices=list(RUNS))
    p.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")])
    a = p.parse_args()
    {
        "extract": lambda: launch(a.run, a.layers),
        "wait": lambda: wait(a.run),
        "status": lambda: status(a.run),
    }[a.action]()
