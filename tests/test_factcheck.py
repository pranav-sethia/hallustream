import json
import sqlite3

import pytest
from transformers import AutoTokenizer

from hallustream.research.align import align_exact
from hallustream.research.factcheck import judge, wiki
from hallustream.research.factcheck.people import HumanLabeledBio

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
BIO = "Marie Curie was born in 1867 in Warsaw. She won three Nobel Prizes. Curie died in 1934."


def test_parse_bio_keeps_valid_labels_and_strips_fences():
    text = (
        "```json\n"
        + json.dumps(
            {
                "entities": [
                    {"span": "1867", "label": "Supported"},
                    {"span": "x", "label": "Maybe"},
                ],
                "claims": [{"text": "She won three Nobel Prizes.", "label": "Not Supported"}],
            }
        )
        + "\n```"
    )
    out = judge.parse_bio(text)
    assert out["entities"] == [("1867", "Supported")]
    assert out["claims"] == [("She won three Nobel Prizes.", "Not Supported")]


def test_parse_claims_pads_missing_labels():
    assert judge.parse_claims('{"labels": ["Supported", "??"]}', 3) == ["Supported", None, None]


def test_locate_matches_repeated_spans_in_reading_order():
    bio = "Curie met Curie in Paris. Later, Curie left Paris."
    ents = [
        ("Curie", "Supported"),
        ("Curie", "Supported"),
        ("Paris", "Supported"),
        ("Curie", "Not Supported"),
        ("Paris", "Supported"),
        ("Lyon", "Supported"),
    ]
    anns, missed = judge.locate(bio, ents)
    assert [a["index"] for a in anns] == [0, 10, 19, 33, 44]
    assert missed == 1


def test_located_entities_align_to_the_right_tokens():
    tok = AutoTokenizer.from_pretrained(TINY)
    anns, _ = judge.locate(
        BIO, [("1867", "Supported"), ("three", "Not Supported"), ("1934", "Insufficient")]
    )
    msgs = [{"role": "user", "content": "Tell me a bio of Marie Curie."}]
    primary = align_exact(tok, msgs, BIO, judge.for_alignment(anns, False))
    secondary = align_exact(tok, msgs, BIO, judge.for_alignment(anns, True))
    assert len(primary.spans) == 2 and len(secondary.spans) == 3
    three = [s for s in primary.spans if s.label == 1][0]
    assert tok.decode(primary.input_ids[three.start : three.end + 1]).strip() == "three"


def test_requests_fill_both_prompts():
    r = judge.bio_request("k1", "Marie Curie", ["Passage one."], BIO)
    text = r["request"]["contents"][0]["parts"][0]["text"]
    assert (
        r["key"] == "k1"
        and "[1] Passage one." in text
        and BIO in text
        and "{" not in text.split("Return only JSON")[0]
    )
    c = judge.claim_request("k2", "Marie Curie", ["P."], ["A.", "B."])
    assert "1. A.\n2. B." in c["request"]["contents"][0]["parts"][0]["text"]


def test_wiki_db_and_bm25(tmp_path):
    db = tmp_path / "w.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE documents (title PRIMARY KEY, text)")
    page = wiki.SEPARATOR.join(
        [
            "<s>Curie was born in Warsaw.</s>",
            "<s>She won two Nobel Prizes.</s>",
            "<s>Unrelated passage about gardening.</s>",
        ]
    )
    con.execute("INSERT INTO documents VALUES (?, ?)", ("Marie Curie", page))
    con.commit()
    ps = wiki.WikiDB(db).passages("Marie Curie")
    assert ps[0] == "Curie was born in Warsaw."
    assert wiki.bm25_rank("Nobel Prizes", ps)[0] == 1
    assert wiki.evidence(ps, "Nobel Warsaw", limit=2) == ps[:2]
    with pytest.raises(KeyError):
        wiki.WikiDB(db).passages("Nobody")


def test_human_factscore():
    bio = HumanLabeledBio(
        "ChatGPT", "X", [], "...", [("a", "S"), ("b", "NS"), ("c", "S"), ("d", "S")]
    )
    assert bio.factscore == 0.75


def test_balanced_accuracy_weights_both_classes_equally():
    from hallustream.research.factcheck.validate import balanced_accuracy

    human = ["S", "S", "S", "NS"]
    assert balanced_accuracy(human, ["Supported"] * 4) == 0.5
    assert balanced_accuracy(human, ["Supported", "Supported", "Supported", "Not Supported"]) == 1.0


def test_gate_drops_insufficient_in_primary_only():
    import numpy as np
    import pandas as pd

    from hallustream.research.factcheck.validate import gate

    rng = np.random.default_rng(0)
    rows = []
    for p in range(40):
        for kind, mu in (("supported", 0.3), ("not_supported", 0.7), ("insufficient", 0.5)):
            rows.append({"name": f"p{p}", "kind": kind, "probe": rng.normal(mu, 0.1),
                         "nll_max": rng.normal(0, 1)})  # fmt: skip
    res = gate(pd.DataFrame(rows))
    assert res["primary"]["entities"] == 80 and res["secondary"]["entities"] == 120
    assert res["primary"]["auroc"] > 0.95 and res["passes"]
    assert res["primary"]["ci"][0] <= res["primary"]["auroc"] <= res["primary"]["ci"][1]
