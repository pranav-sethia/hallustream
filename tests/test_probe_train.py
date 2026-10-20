import numpy as np
import pandas as pd
import torch

from hallustream.research.metrics import auroc
from hallustream.research.probe_train import (
    LinearProbe,
    TrainConfig,
    batch_loss,
    build_data,
    predict,
    span_max,
    train,
)


def toy(n_answers=60, d=8, seed=0):
    """Answers with 3 entities of 2 tokens each plus 4 sampled ordinary tokens.
    Feature 0 carries the label signal."""
    rng = np.random.default_rng(seed)
    rows, spans, X = [], [], []
    for a in range(n_answers):
        aid = f"a{a}"
        for s in range(3):
            label = int(rng.random() < 0.3)
            start = 10 + 5 * s
            spans.append(dict(answer_id=aid, start=start, end=start + 1, label=label))
            for t in (start, start + 1):
                rows.append(dict(answer_id=aid, token_pos=t, label=label, weight=1.0, span_id=s))
                X.append(rng.normal(size=d) + np.eye(d)[0] * (2.0 * label))
        for t in (1, 3, 5, 7):
            rows.append(dict(answer_id=aid, token_pos=t, label=0, weight=5.0, span_id=-1))
            X.append(rng.normal(size=d))
    return pd.DataFrame(rows), pd.DataFrame(spans), np.array(X, dtype=np.float16)


def test_span_max_takes_each_spans_riskiest_token():
    logits = torch.tensor([0.1, 2.0, -1.0, 0.5])
    out = span_max(logits, torch.tensor([0, 1, 2, 3]), torch.tensor([0, 0, 1, 1]), 2)
    assert out.tolist() == [2.0, 0.5]


def test_sampled_ordinary_row_counts_like_the_tokens_it_stands_for():
    y, ones = torch.zeros(3), torch.ones(3)
    full = batch_loss(torch.tensor([0.3, 0.3, 0.3]), y, ones, ones, None, None, 0.0)
    one = torch.ones(1)
    sampled = batch_loss(torch.tensor([0.3]), y[:1], one * 3, one * 3, None, None, 0.0)
    assert torch.isclose(full, sampled)


def test_membership_links_every_span_token():
    rows, spans, X = toy(n_answers=2)
    d = build_data(rows, spans, X, entity_weight=10.0)
    assert len(d.mem_row) == 2 * len(spans)
    assert (d.y[d.mem_row] == d.span_y[d.mem_span]).all()
    assert d.w[rows["span_id"].to_numpy() >= 0].eq(10.0).all()


def test_exported_weights_act_on_raw_features():
    mean, std = torch.randn(8), torch.rand(8) + 0.5
    probe = LinearProbe(mean, std)
    x = torch.randn(5, 8)
    w, b = probe.export()
    assert torch.allclose(probe(x), x @ w + b, atol=1e-5)


def test_both_losses_learn_the_signal():
    rows, spans, X = toy()
    d = build_data(rows, spans, X, entity_weight=10.0)
    for loss in ("annealed", "token"):
        probe = train(d, TrainConfig(loss=loss, lr=5e-2, epochs=3))
        _, span_p, span_y = predict(probe, d)
        assert auroc(span_y, span_p) > 0.9
