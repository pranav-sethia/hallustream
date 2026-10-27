"""LoRA probe (Obeso et al. 2025): a linear head on one layer, trained together with low-rank
adapters on the layers up to it. A KL term keeps the adapted model's next-token distribution
on the answer close to the original, so the model writes the same text.

Recipe from the paper's released code (configs/train_config.yaml, probe/loss.py,
probe/trainer.py): rank 16, alpha 32, dropout 0.05 on every attention and MLP projection;
loss = 0.5 * probe loss + 0.5 * KL; learning rates 1e-4 (adapters) and 1e-3 (head); one
epoch, batch 4, answers cut at 1,536 tokens; the token loss anneals into the span (max) loss
over training; entity tokens weighted 10x. The adapters are written here (a few lines) rather
than taken from a library, so turning them off for the KL reference pass is explicit.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hallustream.research.align import IGNORE, Aligned

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class LoraConfig:
    layer: int = 21  # the head reads the output of this decoder layer
    lora: bool = True  # False: a plain linear probe on the frozen model (same code path)
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    lambda_kl: float = 0.5
    lora_lr: float = 1e-4
    head_lr: float = 1e-3
    batch: int = 4  # answers per optimizer step (one at a time, gradients accumulated)
    max_length: int = 1536
    entity_weight: float = 10.0
    seed: int = 0

    def dict(self) -> dict:
        return asdict(self)


class LoraLinear(nn.Module):
    """y = W x + (alpha / r) * B A dropout(x); B starts at zero, so training starts from the
    original model."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / rank, True
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            return y
        z = self.drop(x).to(self.A.dtype) @ self.A.T @ self.B.T
        return y + (self.scale * z).to(y.dtype)


class LoraProbe(nn.Module):
    def __init__(self, model, cfg: LoraConfig):
        super().__init__()
        self.model, self.cfg = model, cfg
        for p in model.parameters():
            p.requires_grad_(False)
        self.adapters: list[LoraLinear] = []
        if cfg.lora:
            for i, block in enumerate(model.model.layers):
                if i > cfg.layer:  # layers above the head cannot change what it reads
                    break
                for parent in (block.self_attn, block.mlp):
                    for name in PROJECTIONS:
                        if isinstance(getattr(parent, name, None), nn.Linear):
                            ad = LoraLinear(getattr(parent, name), cfg.rank, cfg.alpha,
                                            cfg.dropout).to(model.device)  # fmt: skip
                            setattr(parent, name, ad)
                            self.adapters.append(ad)
        self.head = nn.Linear(model.config.hidden_size, 1, dtype=torch.float32).to(model.device)

    @contextmanager
    def adapters_off(self):
        for ad in self.adapters:
            ad.enabled = False
        try:
            yield
        finally:
            for ad in self.adapters:
                ad.enabled = True

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(probe logits per token, next-token logits) for one sequence (1, T)."""
        out = self.model(ids, output_hidden_states=True)
        h = out.hidden_states[self.cfg.layer + 1][0].float()
        return self.head(h).squeeze(-1), out.logits[0]

    def trainable(self) -> list[dict]:
        groups = [{"params": list(self.head.parameters()), "lr": self.cfg.head_lr}]
        if self.adapters:
            params = [p for ad in self.adapters for p in (ad.A, ad.B)]
            groups.append({"params": params, "lr": self.cfg.lora_lr})
        return groups

    def adapter_state(self) -> dict[str, torch.Tensor]:
        out = {f"{i}.{k}": v.detach().cpu() for i, ad in enumerate(self.adapters)
               for k, v in (("A", ad.A), ("B", ad.B))}  # fmt: skip
        out["head.weight"] = self.head.weight.detach().cpu()
        out["head.bias"] = self.head.bias.detach().cpu()
        return out


def token_targets(a: Aligned, entity_weight: float, device) -> tuple[torch.Tensor, ...]:
    """Labels and weights per token: answer tokens 1x, entity tokens 10x, prompt ignored."""
    y = torch.tensor([max(lab, 0) for lab in a.labels], dtype=torch.float32, device=device)
    w = torch.tensor([0.0 if lab == IGNORE else 1.0 for lab in a.labels], device=device)
    for s in a.spans:
        w[s.start : s.end + 1] = entity_weight
    return y, w


def probe_loss(logits: torch.Tensor, a: Aligned, omega: float, entity_weight: float):
    """(1 - omega) * weighted token BCE (mean over answer tokens) + omega * span-max BCE."""
    y, w = token_targets(a, entity_weight, logits.device)
    answer = w > 0
    token = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w)[answer].mean()
    if omega == 0 or not a.spans:
        return token
    span_logit = torch.stack([logits[s.start : s.end + 1].max() for s in a.spans])
    span_y = torch.tensor([float(s.label) for s in a.spans], device=logits.device)
    return (1 - omega) * token + omega * F.binary_cross_entropy_with_logits(span_logit, span_y)


def kl_to_base(probe: LoraProbe, ids: torch.Tensor, logits: torch.Tensor, a: Aligned,
               chunk: int = 256) -> torch.Tensor:  # fmt: skip
    """Mean KL(base || adapted) of the next-token distribution over answer positions."""
    with torch.no_grad(), probe.adapters_off():
        ref = probe.model(ids).logits[0]
    pos = torch.arange(a.answer_start - 1, len(a.input_ids) - 1, device=logits.device)
    total = logits.new_zeros((), dtype=torch.float32)
    for i in range(0, len(pos), chunk):
        p = pos[i : i + chunk]
        log_q = torch.log_softmax(logits[p].float(), -1)
        p_ref = torch.softmax(ref[p].float(), -1)
        total = total + F.kl_div(log_q, p_ref, reduction="none").sum(-1).sum()
    return total / max(len(pos), 1)


def train(probe: LoraProbe, data: list[Aligned], log=print, log_every: int = 25) -> list[dict]:
    """One epoch over `data`; returns the logged losses."""
    cfg = probe.cfg
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(probe.trainable(), weight_decay=0.01)
    steps = math.ceil(len(data) / cfg.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    params = [p for g in probe.trainable() for p in g["params"]]
    history, step = [], 0
    probe.train()
    for start in range(0, len(data), cfg.batch):
        omega = step / steps
        batch = data[start : start + cfg.batch]
        parts = {"probe": 0.0, "kl": 0.0}
        for a in batch:
            ids = torch.tensor([a.input_ids], device=probe.model.device)
            logits, lm_logits = probe(ids)
            loss = probe_loss(logits, a, omega, cfg.entity_weight)
            parts["probe"] += loss.item() / len(batch)
            if probe.adapters and cfg.lambda_kl > 0:
                kl = kl_to_base(probe, ids, lm_logits, a)
                parts["kl"] += kl.item() / len(batch)
                loss = (1 - cfg.lambda_kl) * loss + cfg.lambda_kl * kl
            (loss / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        history.append({"step": step, "omega": round(omega, 3), **parts})
        if log_every and step % log_every == 0:
            log(f"step {step}/{steps} probe {parts['probe']:.4f} kl {parts['kl']:.5f}")
    probe.eval()
    return history


@torch.no_grad()
def score(probe: LoraProbe, a: Aligned) -> dict:
    """Span scores (max probability over each span) and how much the adapted model's next-token
    choice differs from the original on this answer."""
    ids = torch.tensor([a.input_ids], device=probe.model.device)
    logits, lm = probe(ids)
    prob = torch.sigmoid(logits)
    out = {"spans": [(i, float(prob[s.start : s.end + 1].max()), s.label)
                     for i, s in enumerate(a.spans)]}  # fmt: skip
    if probe.adapters:
        with probe.adapters_off():
            ref = probe.model(ids).logits[0]
        sl = slice(a.answer_start - 1, len(a.input_ids) - 1)
        out["same_top1"] = float((lm[sl].argmax(-1) == ref[sl].argmax(-1)).float().mean())
        out["kl"] = float(kl_to_base(probe, ids, lm, a))
    return out
