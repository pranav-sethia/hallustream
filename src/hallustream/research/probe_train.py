"""Train a linear probe on cached hidden states with Obeso et al.'s recipe.

The paper's loss (checked against their code):
- token loss: BCE on every answer token, entity tokens weighted 10x, ordinary tokens 1x,
  prompt ignored, averaged over answer tokens;
- span loss: BCE on each entity's riskiest token (max logit over the span);
- training blends linearly from the token loss to the span loss (omega = progress).

Our cache holds every entity token but only a sample of ordinary tokens, each with a
weight standing for the ordinary tokens not sampled. Using those weights in the token
loss gives the same expected loss as training on every token.

Features are standardized with training statistics; the scaling is folded back into
the weights on export, so the shipped probe reads raw hidden states.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrainConfig:
    loss: str = "annealed"  # "annealed" (paper) or "token"
    lr: float = 1e-3
    epochs: int = 1
    l2: float = 0.0  # extra L2 penalty on the (standardized) weights
    weight_decay: float = 0.01  # torch AdamW default, as in the paper
    entity_weight: float = 10.0
    batch_answers: int = 4
    seed: int = 0

    def name(self) -> str:
        return f"{self.loss}_lr{self.lr:g}_ep{self.epochs}_l2{self.l2:g}_s{self.seed}"


@dataclass
class ProbeData:
    """Rows, spans and span membership for a set of answers, laid out by answer."""

    X: torch.Tensor  # (rows, d) float16
    y: torch.Tensor  # (rows,) token labels
    w: torch.Tensor  # (rows,) token-loss weights
    count: torch.Tensor  # (rows,) how many answer tokens each row stands for
    row_ptr: np.ndarray  # answer a owns rows row_ptr[a]:row_ptr[a+1]
    span_ptr: np.ndarray  # answer a owns spans span_ptr[a]:span_ptr[a+1]
    mem_ptr: np.ndarray  # answer a owns membership entries mem_ptr[a]:mem_ptr[a+1]
    mem_row: torch.Tensor  # membership: global row index ...
    mem_span: torch.Tensor  # ... belongs to this global span index
    span_y: torch.Tensor  # (spans,) span labels
    answer_ids: list[str]

    @property
    def n_answers(self) -> int:
        return len(self.answer_ids)


def _ptr(keys: pd.Series, order: list[str]) -> np.ndarray:
    counts = keys.value_counts().reindex(order, fill_value=0).to_numpy()
    return np.concatenate([[0], np.cumsum(counts)])


def build_data(
    rows: pd.DataFrame,
    spans: pd.DataFrame,
    X: np.ndarray,
    entity_weight: float,
    device: str = "cpu",
) -> ProbeData:
    """rows/spans must list answers in the same order, each answer's rows contiguous."""
    order = list(dict.fromkeys(rows["answer_id"]))
    span_order = list(dict.fromkeys(spans["answer_id"]))
    has_spans = set(span_order)
    if span_order != [a for a in order if a in has_spans]:
        raise ValueError("rows and spans list answers in different orders")
    row_ptr, span_ptr = _ptr(rows["answer_id"], order), _ptr(spans["answer_id"], order)
    if not (rows["answer_id"].to_numpy() == np.repeat(order, np.diff(row_ptr))).all():
        raise ValueError("each answer's rows must be contiguous")

    pos = rows["token_pos"].to_numpy()
    starts, ends = spans["start"].to_numpy(), spans["end"].to_numpy()
    mem_row, mem_span, mem_ptr = [], [], [0]
    for a in range(len(order)):
        r0, r1 = row_ptr[a], row_ptr[a + 1]
        row_of = dict(zip(pos[r0:r1].tolist(), range(r0, r1), strict=True))
        for s in range(span_ptr[a], span_ptr[a + 1]):
            for t in range(starts[s], ends[s] + 1):
                mem_row.append(row_of[t])  # every entity token is cached
                mem_span.append(s)
        mem_ptr.append(len(mem_row))

    entity = rows["span_id"].to_numpy() >= 0
    sample_w = rows["weight"].to_numpy(dtype=np.float32)
    w = np.where(entity, entity_weight, sample_w).astype(np.float32)

    def t(a, dtype=torch.float32):
        return torch.tensor(np.asarray(a), dtype=dtype, device=device)

    return ProbeData(
        X=torch.as_tensor(X, device=device),
        y=t(rows["label"].to_numpy()),
        w=t(w),
        count=t(sample_w),
        row_ptr=row_ptr,
        span_ptr=span_ptr,
        mem_ptr=np.array(mem_ptr),
        mem_row=t(mem_row, torch.long),
        mem_span=t(mem_span, torch.long),
        span_y=t(spans["label"].to_numpy()),
        answer_ids=order,
    )


def span_max(logits: torch.Tensor, mem_row: torch.Tensor, mem_span: torch.Tensor, n: int):
    """Max over each span's tokens. `logits` is indexed by mem_row; spans are 0..n-1."""
    out = torch.full((n,), -torch.inf, device=logits.device, dtype=logits.dtype)
    return out.scatter_reduce(0, mem_span, logits[mem_row], reduce="amax")


def batch_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    count: torch.Tensor,
    span_logit: torch.Tensor,
    span_y: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """(1 - omega) * token loss + omega * span loss, as in the paper."""
    token = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum()
    token = token / count.sum()
    if omega == 0:
        return token
    span = F.binary_cross_entropy_with_logits(span_logit, span_y)
    return (1 - omega) * token + omega * span


class LinearProbe(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.linear = torch.nn.Linear(mean.shape[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear((x.float() - self.mean) / self.std).squeeze(-1)

    def export(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Weight and bias that act on raw (unstandardized) hidden states."""
        w = self.linear.weight[0] / self.std
        b = self.linear.bias[0] - (w * self.mean).sum()
        return w.detach().cpu(), b.detach().cpu()


def feature_stats(X: torch.Tensor, chunk: int = 20_000) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    ss = torch.zeros_like(s)
    for i in range(0, X.shape[0], chunk):
        x = X[i : i + chunk].double()
        s += x.sum(0)
        ss += (x * x).sum(0)
    mean = s / X.shape[0]
    std = (ss / X.shape[0] - mean**2).clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


def _answer_batch(d: ProbeData, answers: np.ndarray):
    rows = torch.cat([torch.arange(d.row_ptr[a], d.row_ptr[a + 1]) for a in answers])
    mem = torch.cat([torch.arange(d.mem_ptr[a], d.mem_ptr[a + 1]) for a in answers])
    spans = torch.cat([torch.arange(d.span_ptr[a], d.span_ptr[a + 1]) for a in answers])
    return rows.to(d.X.device), mem.to(d.X.device), spans.to(d.X.device)


def train(d: ProbeData, cfg: TrainConfig, stats=None, log_every: int = 0) -> LinearProbe:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    mean, std = stats if stats is not None else feature_stats(d.X)
    probe = LinearProbe(mean, std).to(d.X.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(d.n_answers / cfg.batch_answers)
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / total)  # HF default
    remap = torch.empty(d.X.shape[0], dtype=torch.long, device=d.X.device)
    span_remap = torch.empty(d.span_y.shape[0], dtype=torch.long, device=d.X.device)
    step = 0
    for _ in range(cfg.epochs):
        perm = rng.permutation(d.n_answers)
        for i in range(0, d.n_answers, cfg.batch_answers):
            rows, mem, spans = _answer_batch(d, perm[i : i + cfg.batch_answers])
            logits = probe(d.X[rows])
            omega = step / total if cfg.loss == "annealed" else 0.0
            span_logit = None
            if omega > 0:
                remap[rows] = torch.arange(len(rows), device=rows.device)
                span_remap[spans] = torch.arange(len(spans), device=rows.device)
                span_logit = span_max(
                    logits, remap[d.mem_row[mem]], span_remap[d.mem_span[mem]], len(spans)
                )
            loss = batch_loss(
                logits, d.y[rows], d.w[rows], d.count[rows],
                span_logit, d.span_y[spans] if omega > 0 else None, omega,
            )  # fmt: skip
            if cfg.l2:
                loss = loss + cfg.l2 * probe.linear.weight.pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)  # paper: max_grad_norm 1
            opt.step()
            sched.step()
            step += 1
            if log_every and step % log_every == 0:
                print(f"  step {step}/{total} loss {loss.item():.4f} omega {omega:.2f}")
    return probe


@torch.no_grad()
def predict(probe, d: ProbeData, chunk: int = 50_000) -> tuple[np.ndarray, ...]:
    """(row probabilities, span max-probabilities, span labels)."""
    logits = torch.cat([probe(d.X[i : i + chunk]) for i in range(0, d.X.shape[0], chunk)])
    span_logit = span_max(logits, d.mem_row, d.mem_span, d.span_y.shape[0])
    return (
        torch.sigmoid(logits).cpu().numpy(),
        torch.sigmoid(span_logit).cpu().numpy(),
        d.span_y.cpu().numpy(),
    )


def config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
