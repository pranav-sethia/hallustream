import json
import sqlite3

import numpy as np
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


def test_parse_claims_joins_on_id_and_rejects_gaps():
    ok = {"labels": [{"id": 2, "label": "Not Supported"}, {"id": 1, "label": "Supported"}]}
    assert judge.parse_claims(json.dumps(ok), 2) == ["Supported", "Not Supported"]
    for bad in (
        {"labels": [{"id": 1, "label": "Supported"}]},  # claim 2 skipped
        {"labels": [{"id": 1, "label": "Supported"}, {"id": 1, "label": "Supported"}]},
        {"labels": [{"id": 1, "label": "Supported"}, {"id": 2, "label": "Maybe"}]},
    ):
        with pytest.raises(ValueError):
            judge.parse_claims(json.dumps(bad), 2)


def test_parse_bio_drops_non_string_spans_and_rejects_non_objects():
    out = judge.parse_bio('{"entities": [{"span": 1867, "label": "Supported"}], "claims": []}')
    assert out["entities"] == []
    with pytest.raises(ValueError):
        judge.parse_bio("[1, 2]")


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


def test_locate_needs_word_boundaries_and_never_reuses_text():
    anns, missed = judge.locate("Parisian art in Paris.", [("Paris", "Supported")])
    assert anns[0]["index"] == 16 and missed == 0
    anns, missed = judge.locate("Born in Paris.", [("Paris", "Supported"), ("Paris", "Supported")])
    assert len(anns) == 1 and missed == 1


def test_locate_ignores_markdown_and_quote_styles():
    bio = "**Jean d’Arc** won the *Prix   Goncourt* in 1990."
    anns, missed = judge.locate(bio, [("Jean d'Arc", "Supported"), ("Prix Goncourt", "Supported")])
    assert missed == 0
    assert [a["span"] for a in anns] == ["Jean d’Arc", "Prix   Goncourt"]
    assert all(bio[a["index"] : a["index"] + len(a["span"])] == a["span"] for a in anns)


def test_response_text_skips_thoughts_and_errors():
    parts = [{"text": "thinking...", "thought": True}, {"text": '{"a": '}, {"text": "1}"}]
    cand = {"finishReason": "STOP", "content": {"parts": parts}}
    row = {"key": "k", "response": {"candidates": [cand]}}
    assert judge.response_text(row) == ('{"a": 1}', "STOP")
    assert judge.response_text({"key": "k", "error": {"code": 500}}) == (None, "ERROR")


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
    r = judge.bio_request("k1", "Marie Curie", ["Passage one."], BIO, "t1_low")
    text = r["request"]["contents"][0]["parts"][0]["text"]
    assert r["key"] == "k1" and "[1] Passage one." in text and BIO in text and "Marie Curie" in text
    cfg = r["request"]["generationConfig"]
    assert cfg["thinkingConfig"] == {"thinkingLevel": "low"} and cfg["temperature"] == 1.0
    assert cfg["responseSchema"]["properties"]["entities"]["items"]["properties"]["label"]["enum"]
    c = judge.claim_request("k2", "Marie Curie", ["P."], ["A.", "B."], "t0_minimal")
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


def test_human_factscore_counts_irrelevant_facts_as_unsupported():
    bio = HumanLabeledBio(
        "ChatGPT", "X", [], "...", [("a", "S"), ("b", "NS"), ("c", "S"), ("d", "IR")]
    )
    assert bio.factscore == 0.5
    assert bio.claims == [("a", "S"), ("b", "NS"), ("c", "S")]


def test_split_labeled_is_fixed_and_disjoint():
    from hallustream.research.factcheck.people import split_labeled

    names = [f"p{i}" for i in range(183)]
    tune, report = split_labeled(names)
    assert len(tune) == 40 and len(report) == 143 and not tune & report
    assert split_labeled(list(reversed(names))) == (tune, report)


def test_balanced_accuracy_weights_both_classes_equally():
    from hallustream.research.factcheck.validate import balanced_accuracy

    human = ["S", "S", "S", "NS"]
    assert balanced_accuracy(human, ["Supported"] * 4) == 0.5
    assert balanced_accuracy(human, ["Supported", "Supported", "Supported", "Not Supported"]) == 1.0
    assert np.isnan(balanced_accuracy(["S", "S"], ["Supported", "Supported"]))


def test_gate_drops_insufficient_in_primary_only():
    import pandas as pd

    from hallustream.research.factcheck.validate import gate

    rng = np.random.default_rng(0)
    rows = []
    for p in range(40):
        for kind, mu in (("supported", 0.3), ("not_supported", 0.7), ("insufficient", 0.5)):
            seeds = {f"p_seed{k}": rng.normal(mu, 0.1) for k in range(2)}
            rows.append({"name": f"p{p}", "kind": kind, "entropy_max": rng.normal(0, 1)} | seeds)
    res = gate(pd.DataFrame(rows), "entropy_max")
    assert res["primary"]["entities"] == 80 and res["secondary"]["entities"] == 120
    assert res["primary"]["auroc"] > 0.95 and res["passes"]
    assert res["primary"]["ci"][0] <= res["primary"]["auroc"] <= res["primary"]["ci"][1]
    assert set(res["primary"]["auroc_by_seed"]) == {"p_seed0", "p_seed1"}
    assert gate(pd.DataFrame(rows), "entropy_max", {"p0", "p1"})["people"] == 2


def test_audit_request_lists_entities_with_their_sentences():
    r = judge.audit_request("k", "Marie Curie", ["P."], BIO, [("1867", "Born in 1867.")], "t1_low")
    text = r["request"]["contents"][0]["parts"][0]["text"]
    assert '1. "1867" in: Born in 1867.' in text
    enum = r["request"]["generationConfig"]["responseSchema"]["properties"]["labels"]
    assert enum["items"]["properties"]["label"]["enum"] == list(judge.ENTITY_LABELS)
    ok = '{"labels": [{"id": 1, "label": "Insufficient"}]}'
    assert judge.parse_ids(ok, 1, judge.ENTITY_LABELS) == ["Insufficient"]


def test_sentence_of_handles_sentences_and_list_lines():
    t = "Born in 1901 in Paris. He studied at Oxford.\n- Award: Nobel"
    got = [judge.sentence_of(t, t.index(x)) for x in ("1901", "Oxford", "Nobel")]
    assert got == ["Born in 1901 in Paris.", "He studied at Oxford.", "- Award: Nobel"]


def test_audit_compares_labels_per_person_in_order():
    from hallustream.research.factcheck.validate import audit

    items = [
        {"name": "A", "span": "1867", "label": "Supported"},
        {"name": "A", "span": "Paris", "label": "Not Supported"},
        {"name": "B", "span": "Oxford", "label": "Insufficient"},
    ]
    results = {
        "A": json.dumps(
            {"labels": [{"id": 1, "label": "Supported"}, {"id": 2, "label": "Supported"}]}
        ),  # fmt: skip
        "B": "not json",
    }
    out = audit(items, results)
    assert out["entities"] == 2 and out["failed"] == 1
    assert out["agreement_supported_vs_not"] == 0.5
    assert out["confusion"]["Not Supported -> Supported"] == 1
