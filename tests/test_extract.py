import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hallustream.research.data import Record
from hallustream.research.extract import choose_rows, extract_answer, token_baselines

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    return model, AutoTokenizer.from_pretrained(TINY)


def record():
    answer = "Ada Lovelace was born in 1815 in London. She worked with Charles Babbage."
    anns = [
        {"index": answer.index("1815"), "span": "1815", "label": "Supported"},
        {"index": answer.index("London"), "span": "London", "label": "Not Supported"},
        {"index": None, "span": "x", "label": None},
    ]
    return Record(
        answer_id="ds/src/test/0",
        dataset="ds",
        source="src",
        orig_split="test",
        subset="s",
        messages=[{"role": "user", "content": "Tell me about Ada Lovelace."}],
        answer=answer,
        annotations=anns,
    )


def test_token_baselines_match_manual_computation():
    torch.manual_seed(0)
    logits = torch.randn(5, 11)
    ids = torch.tensor([1, 4, 2, 9, 3])
    nll, ent = token_baselines(logits, ids)
    logp = torch.log_softmax(logits, -1)
    assert np.isnan(nll[0])
    for t in range(1, 5):
        assert nll[t] == pytest.approx(-logp[t - 1, ids[t]].item(), rel=1e-5)
        assert ent[t] == pytest.approx(-(logp[t - 1].exp() * logp[t - 1]).sum().item(), rel=1e-5)


def test_extract_answer_rows_features_and_spans(tiny):
    model, tok = tiny
    rng = np.random.default_rng(0)
    layers = [0, 1]
    f = extract_answer(model, tok, record(), layers, n_neg=5, rng=rng, split="test")
    n = len(f.rows["token_pos"])
    assert set(f.features) == set(layers)
    assert all(
        v.shape == (n, model.config.hidden_size) and v.dtype == np.float16
        for v in f.features.values()
    )
    # every entity token is saved, with its span's label; ordinary samples have label 0
    ent = [i for i, s in enumerate(f.rows["span_id"]) if s >= 0]
    labels = {s["span_id"]: s["label"] for s in f.spans}
    assert all(f.rows["label"][i] == labels[f.rows["span_id"][i]] for i in ent)
    assert sum(1 for s in f.rows["span_id"] if s < 0) == 5
    assert [s["label"] for s in f.spans] == [0, 1]
    # features equal the model's hidden state at that position
    with torch.no_grad():
        from hallustream.research.align import align_exact

        a = align_exact(tok, record().messages, record().answer, record().annotations)
        hs = model(torch.tensor([a.input_ids]), output_hidden_states=True).hidden_states
    p = f.rows["token_pos"][0]
    np.testing.assert_allclose(f.features[1][0], hs[2][0, p].numpy(), rtol=1e-2, atol=1e-2)


def test_sampled_weights_reweight_to_the_full_count(tiny):
    model, tok = tiny
    from hallustream.research.align import align_exact

    a = align_exact(tok, record().messages, record().answer, record().annotations)
    pos, w = choose_rows(a, n_neg=4, rng=np.random.default_rng(1))
    n_entity = sum(s.end - s.start + 1 for s in a.spans)
    n_ordinary = len(a.input_ids) - a.answer_start - n_entity
    assert sum(w[n_entity:]) == pytest.approx(n_ordinary)
    assert len(set(pos)) == len(pos)
