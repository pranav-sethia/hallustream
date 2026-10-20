import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import AutoTokenizer

from hallustream.research.align import (
    IGNORE,
    Span,
    align_exact,
    align_paper,
    build_text,
    find_string_in_tokens,
)
from hallustream.research.metrics import auroc, recall_at_fpr, span_scores

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
MSGS = [{"role": "user", "content": "Tell me about Paris."}]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TINY)


def ann(answer, text, label, nth=0):
    idx = -1
    for _ in range(nth + 1):
        idx = answer.index(text, idx + 1)
    return {"index": idx, "span": text, "label": label}


def decoded(tok, aligned, sp):
    return tok.decode(aligned.input_ids[sp.start : sp.end + 1])


def test_labels_the_right_mention_of_a_repeated_entity(tok):
    answer = "Paris is in France. Paris was founded in 1832."
    a = align_exact(tok, MSGS, answer, [ann(answer, "Paris", "Not Supported", nth=1)])
    (sp,) = a.spans
    # the second "Paris" sits after "France." — check the tokens just before it
    before = tok.decode(a.input_ids[a.answer_start : sp.start])
    assert "France." in before and "Paris" in decoded(tok, a, sp)


def test_prompt_is_never_labeled_and_answer_defaults_to_zero(tok):
    answer = "Paris was founded in 1832."
    a = align_exact(tok, MSGS, answer, [ann(answer, "1832", "Not Supported")])
    assert all(x == IGNORE for x in a.labels[: a.answer_start])
    assert set(a.labels[a.answer_start :]) == {0, 1}
    assert tok.decode(a.input_ids[: a.answer_start]).endswith("assistant\n")


def test_positive_wins_on_overlap_and_unlabeled_are_ignored(tok):
    answer = "The Eiffel Tower opened in 1889."
    anns = [
        ann(answer, "Eiffel Tower", "Supported"),
        ann(answer, "Tower", "Insufficient Information"),
        {"index": None, "span": "1889", "label": None},
    ]
    a = align_exact(tok, MSGS, answer, anns)
    assert [s.label for s in a.spans] == [0, 1]
    tower = a.spans[1]
    assert all(a.labels[t] == 1 for t in range(tower.start, tower.end + 1))
    assert a.stats["ignored_label"] == 1


def test_wrong_index_is_recovered_to_nearest_copy(tok):
    answer = "Born in 1901. Died in 1901."
    bad = {"index": 20, "span": "1901", "label": "Supported"}  # off by a few chars
    a = align_exact(tok, MSGS, answer, [bad])
    assert a.stats["recovered"] == 1 and "1901" in decoded(tok, a, a.spans[0])


def test_build_text_ends_with_answer(tok):
    text, start = build_text(tok, MSGS, "Hello.")
    assert text[start:] == "Hello.<|im_end|>\n"


def test_find_string_in_tokens_minimal_first_window(tok):
    ids = tok("aa Paris bb Paris cc", add_special_tokens=False).input_ids
    s, e = find_string_in_tokens("Paris", ids, tok)
    assert "Paris" in tok.decode(ids[s:e])
    assert "Paris" not in tok.decode(ids[s + 1 : e]) and "Paris" not in tok.decode(ids[s : e - 1])
    assert "bb" not in tok.decode(ids[:e])  # first occurrence


def test_paper_mode_takes_first_occurrence(tok):
    answer = "Paris is in France. Paris was founded in 1832."
    text, _ = build_text(tok, MSGS, answer)
    ids = tok(text, add_special_tokens=False).input_ids
    spans, _ = align_paper(tok, ids, answer, [ann(answer, "Paris", "Not Supported", nth=1)])
    before = tok.decode(ids[: spans[0].start])
    assert "France" not in before  # the paper's search lands on the FIRST "Paris"


@settings(max_examples=40, deadline=None)
@given(
    words=st.lists(
        st.sampled_from(["alpha", "Paris", "1999", "Zürich", "😀", "the", "C++", "  "]),
        min_size=3,
        max_size=25,
    ),
    pick=st.integers(min_value=0, max_value=24),
)
def test_aligned_tokens_always_cover_the_span(tok, words, pick):
    answer = " ".join(words)
    target = words[pick % len(words)]
    if not target.strip():
        return
    a = align_exact(tok, MSGS, answer, [ann(answer, target, "Supported")])
    assert len(a.spans) == 1
    assert target in decoded(tok, a, a.spans[0])


def test_metrics_match_hand_computed_values():
    probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    spans = [Span(0, 1, 1), Span(2, 2, 0), Span(3, 4, 1), Span(4, 4, 0)]
    scores, labels = span_scores(probs, spans)
    assert scores.tolist() == [0.9, 0.2, 0.8, 0.3]
    assert auroc(labels, scores) == 1.0
    assert recall_at_fpr(labels, scores) == 1.0


def test_prompt_longer_than_max_length_fails_loudly(tok):
    long_msgs = [{"role": "user", "content": "word " * 200}]
    with pytest.raises(ValueError, match="longer than max_length"):
        align_exact(tok, long_msgs, "Paris is in France.", [], max_length=50)
