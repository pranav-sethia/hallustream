"""Gemini Batch API jobs for the biography judge (needs the `gemini` Modal secret).

Deploy once:  uv run modal deploy modal_jobs/judge_batch.py
Then:         uv run python -m modal_jobs.judge_batch submit <run> [model] [--first N]
              uv run python -m modal_jobs.judge_batch collect <run>
--first N submits only the first N requests, as run "<run>_first<N>" (a cheap format check).
Request files are written by modal_jobs/bios.py (prepare).
"""

import argparse
import json
from pathlib import Path

import modal

from modal_jobs.common import CACHE_DIR, image, volume

app = modal.App("hallustream-judge")
ROOT = Path(CACHE_DIR)
JUDGE_MODEL = "gemini-3.1-flash-lite"


@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("gemini")],
    timeout=30 * 60,
)
def submit(run: str, model: str = JUDGE_MODEL, first: int = 0) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    d = ROOT / "judge" / run
    if first:
        lines = (d / "requests.jsonl").read_text().splitlines()[:first]
        run, d = f"{run}_first{first}", ROOT / "judge" / f"{run}_first{first}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "requests.jsonl").write_text("".join(x + "\n" for x in lines))
    up = client.files.upload(
        file=str(d / "requests.jsonl"),
        config=types.UploadFileConfig(display_name=run, mime_type="jsonl"),
    )
    job = client.batches.create(model=model, src=up.name, config={"display_name": run})
    (d / "job.json").write_text(json.dumps({"job": job.name, "model": model}))
    volume.commit()
    return job.name


@app.function(
    image=image,
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("gemini")],
    timeout=30 * 60,
)
def collect(run: str) -> dict:
    from google import genai

    client = genai.Client()
    volume.reload()  # a warm container may not see job.json written by submit
    d = ROOT / "judge" / run
    job = client.batches.get(name=json.loads((d / "job.json").read_text())["job"])
    state = job.state.name
    if state != "JOB_STATE_SUCCEEDED":
        return {"run": run, "state": state, "error": str(job.error) if job.error else None}
    content = client.files.download(file=job.dest.file_name).decode("utf-8")
    (d / "results.jsonl").write_text(content)
    volume.commit()
    return {"run": run, "state": state, "lines": len(content.splitlines())}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["submit", "collect"])
    p.add_argument("run")
    p.add_argument("model", nargs="?", default=JUDGE_MODEL)
    p.add_argument("--first", type=int, default=0)
    a = p.parse_args()
    fn = modal.Function.from_name("hallustream-judge", a.action)
    print(fn.remote(a.run, a.model, a.first) if a.action == "submit" else fn.remote(a.run))
