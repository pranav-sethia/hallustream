import json

from hallustream.research import shars_test
from hallustream.research.shars_log import parse

LOG = """Verified text:\t
Attempting 1/10 with T=0.00: \tAda Lovelace was a mathematician and a painter.
Prop-0 (Uncertainty:  0.200, Factual) :\tAda Lovelace was a mathematician.\t(mathematician)
Prop-1 (Uncertainty:  0.900, Hallued) :\tAda Lovelace was a painter.\t(painter)
Factual propositions exist, but not all.
Rewrite the original sentence:
    Ada Lovelace was a mathematician and a painter.
to the new sentence:
    Ada Lovelace was a mathematician.
Verified text:\tAda Lovelace was a mathematician.
Attempting 1/10 with T=0.00: \t She lived in Paris.
Prop-0 (Uncertainty:  0.950, Hallued) :\tShe lived in Paris.\t(Paris)
All propositions are hallucinated.
Discard the current sentence and resample.
Attempting 2/10 with T=0.00: \t She wrote notes on engines and poems.
Prop-0 (Uncertainty:  0.100, Factual) :\tShe wrote notes on engines.\t(engines)
Prop-1 (Uncertainty:  0.800, Hallued) :\tShe wrote poems.\t(poems)
Factual propositions exist, but not all.
Rewrite the original sentence:
    She wrote notes on engines and poems.
Failed to rewrite into a valid sentence
Discard the current sentence and resample.
[END] Generation complete due to EOS.
"""


def test_parse_counts_outcomes_and_checks_rewrites():
    p = parse(LOG)
    assert p["counts"] == {"attempts": 3, "props": 5, "props_flagged": 3, "rewritten": 1,
                           "all_flagged": 1, "rewrite_failed": 1}  # fmt: skip
    assert p["end"] == "eos"
    (w,) = p["rewrites"]
    assert w["new"] == "Ada Lovelace was a mathematician."
    assert w["flagged_kept"] == [] and w["trusted_missing"] == []
    assert w["trusted_claims"] == ["Ada Lovelace was a mathematician."]


def _result(name, claims):
    text = json.dumps({"entities": [], "claims": [{"text": t, "label": lab} for t, lab in claims]})
    return {"key": name, "response": {"candidates": [{"content": {"parts": [{"text": text}]},
                                                      "finishReason": "STOP"}]}}  # fmt: skip


def _arm(d, bios, results):
    d.mkdir()
    (d / "bios.jsonl").write_text("".join(json.dumps(b) + "\n" for b in bios))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))


def test_report_counts_abstentions_and_pairs_with_greedy(tmp_path):
    S, N = "Supported", "Not Supported"
    _arm(
        tmp_path / "naive",
        [{"name": "a", "bio": "x", "seconds": 1.0}, {"name": "b", "bio": "y", "seconds": 1.0}],
        [_result("a", [("1", S), ("2", N), ("3", N)]), _result("b", [("1", N), ("2", N)])],
    )
    _arm(tmp_path / "probe", [{"name": "a", "bio": "x", "seconds": 2.0, "log": LOG,
                               "props": 5, "unlocated": 0},
                              {"name": "b", "bio": "", "seconds": 2.0, "log": "", "props": 2,
                               "unlocated": 1}],
         [_result("a", [("1", S), ("2", N)])])  # fmt: skip
    arms = {k: shars_test.arm_table(tmp_path / k) for k in ("naive", "probe")}
    out = shars_test.report(arms)
    probe = out["arms"]["probe"]
    assert probe["response_rate"] == 0.5 and probe["precision"] == 0.5
    assert probe["supported_per_bio"] == 0.5 and probe["unsupported_per_bio"] == 0.5
    assert probe["log"]["rewrites"] == 1 and probe["log"]["unlocated_share"] == round(1 / 7, 4)
    d = out["paired"]["probe-naive"]
    assert d["unsupported"]["delta"] == -1.5  # a: 1 - 2, b: 0 - 2
    assert d["precision"]["people"] == 1  # only "a" answered in both


def test_length_penalty():
    assert shars_test.length_penalized(0.8, 20) == 0.8
    assert shars_test.length_penalized(0.8, 5) < 0.8 * 0.4
