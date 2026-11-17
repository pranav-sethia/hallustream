"""The headline numbers in the docs match results/public/numbers.json, which
research/public.py recomputes from the public results bundle."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "results" / "public" / "numbers.json"
if not NUMBERS.exists():
    pytest.skip("no results/public/numbers.json", allow_module_level=True)
N = json.loads(NUMBERS.read_text())
DOCS = {n: (ROOT / n).read_text().replace("**", "") for n in ("README.md", "docs/RESULTS.md")}


def f3(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float) -> str:
    return f"{'+' if x >= 0 else '−'}{abs(x):.3f}"


def ci(c: list[float]) -> str:
    return f"{c[0]:.3f} to {c[1]:.3f}"


def _h2h(model: str, ds: str) -> dict:
    return N[f"h2h_{model}_{ds}"]


OURS = {"qwen25": "qwen25_L18", "llama31": "llama31_L12"}
CASES = []
for model in ("qwen25", "llama31"):
    for ds in ("longfact", "longfact_aug") + (("healthbench",) if model == "llama31" else ()):
        r = _h2h(model, ds)
        CASES += [
            ("docs/RESULTS.md", f3(r[OURS[model]]["auroc"])),
            ("docs/RESULTS.md", f3(r["paper"]["auroc"])),
            ("docs/RESULTS.md", f3(r["lora"]["auroc"])),
            ("docs/RESULTS.md", signed(r["ours_minus_paper"]["delta"])),
        ]
# README: the reproduction, the gains over the linear probe and the gaps to the LoRA probe
qw = [_h2h("qwen25", ds) for ds in ("longfact", "longfact_aug")]
ll = [_h2h("llama31", ds) for ds in ("longfact", "longfact_aug")]
for doc in ("README.md", "docs/RESULTS.md"):
    CASES.append((doc, f"{qw[0]['paper']['auroc']:.4f} / {qw[1]['paper']['auroc']:.4f}"))
for rows in (qw, ll):
    d = sorted(r["ours_minus_paper"]["delta"] for r in rows)
    CASES.append(("README.md", f"{d[0]:.3f}–{d[1]:.3f}"))
    CASES.append(("README.md", " / ".join(signed(r["ours_minus_lora"]["delta"]) for r in rows)))
for part in ("people", "topics"):
    r = N[f"own_fresh_{part}"]
    CASES += [
        ("docs/RESULTS.md", f3(r["v4"]["auroc"])),
        ("docs/RESULTS.md", f3(r["v1"]["auroc"])),
        ("docs/RESULTS.md", f3(r["entropy_max"]["auroc"])),
        ("README.md", f3(r["v4"]["auroc"])),
        ("docs/RESULTS.md", f"{signed(r['v4_minus_entropy_max']['delta'])} "
         f"({ci(r['v4_minus_entropy_max']['ci'])})"),
    ]  # fmt: skip
CASES.append(("docs/RESULTS.md", f"{signed(N['fresh_v4_minus_v3']['delta'])} "
              f"({ci(N['fresh_v4_minus_v3']['ci'])})"))  # fmt: skip
for arm in ("naive", "halluse", "halluse_07", "probe_30", "probe_45", "probe_60"):
    a = N["shars_shared"][arm]  # the SHARS table's adjacent true and false cells
    cells = f"| {a['true_per_person']:.2f} | {a['false_per_person']:.2f} |"
    CASES.append(("docs/RESULTS.md", cells))


@pytest.mark.parametrize("doc, text", CASES)
def test_doc_states_the_recomputed_number(doc, text):
    assert text in DOCS[doc], f"{doc} does not state {text!r} (results/public/numbers.json)"
