import torch
from transformers import AutoModelForCausalLM

from hallustream.research.align import IGNORE, Aligned, Span
from hallustream.research.lora_probe import LoraConfig, LoraProbe, kl_to_base, score, train

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def example(seed: int) -> Aligned:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, 500, (40,), generator=g).tolist()
    labels = [IGNORE] * 8 + [0] * 32
    spans = [Span(12, 14, 1), Span(20, 21, 0), Span(30, 33, 1)]
    for s in spans:
        for t in range(s.start, s.end + 1):
            labels[t] = s.label
    return Aligned(ids, labels, 8, spans)


def tiny(cfg: LoraConfig) -> LoraProbe:
    torch.manual_seed(0)
    lm = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    return LoraProbe(lm, cfg)


def test_adapters_start_neutral_and_switch_off_exactly():
    probe = tiny(LoraConfig(layer=0, dropout=0.0))
    a = example(0)
    ids = torch.tensor([a.input_ids])
    _, lm = probe(ids)
    assert float(kl_to_base(probe, ids, lm, a)) < 1e-8  # B = 0: same model as before
    for ad in probe.adapters:
        ad.B.data.normal_()
    _, changed = probe(ids)
    with probe.adapters_off():
        _, base = probe(ids)
    assert not torch.allclose(changed, base)
    assert torch.allclose(base, lm, atol=1e-6)
    n = probe.model.config.num_hidden_layers
    assert len(probe.adapters) == 7 and n > 1  # only the layers up to the head's layer


def test_training_lowers_the_loss_and_keeps_kl_small():
    probe = tiny(LoraConfig(layer=0, batch=2, lora_lr=1e-2, head_lr=1e-2))
    data = [example(s) for s in range(8)] * 4
    hist = train(probe, data, log_every=0)
    assert hist[-1]["probe"] < hist[0]["probe"]
    assert all(h["kl"] >= -1e-6 for h in hist)  # KL >= 0, up to float rounding (B starts at 0)
    out = score(probe, example(0))
    assert len(out["spans"]) == 3 and 0 <= out["same_top1"] <= 1


def test_head_only_mode_leaves_the_model_frozen():
    probe = tiny(LoraConfig(layer=0, lora=False, batch=2))
    before = [p.clone() for p in probe.model.parameters()]
    train(probe, [example(s) for s in range(4)], log_every=0)
    assert not probe.adapters
    assert all(torch.equal(a, b) for a, b in zip(before, probe.model.parameters(), strict=True))
