from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cox_partial_likelihood_loss(log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Negative Cox partial log-likelihood (Breslow).
    log_risk: shape [B] (higher => higher hazard)
    time: shape [B]
    event: shape [B] in {0,1}
    """
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]

    log_cum_sum_exp = torch.logcumsumexp(log_risk, dim=0)
    neg_log_lik = -(log_risk - log_cum_sum_exp) * event
    denom = event.sum().clamp_min(1.0)
    return neg_log_lik.sum() / denom


def mtm_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked tissue modeling (MTM): mean squared error on masked positions.

    pred, target: [B, G] or [N, G]; mask: same leading shape, 1 where supervised.
    """
    if mask.dtype != torch.float32:
        mask = mask.float()
    diff = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def domain_classification_loss(logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for domain adversarial head (HTAN vs TCGA)."""
    return F.cross_entropy(logits, domain_labels.long())


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        lam = ctx.lambda_
        return grad_output.neg() * lam, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambda_)  # type: ignore[return-value]


class DomainHead(nn.Module):
    """Small MLP discriminator on pooled slide embedding (Stage 2)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, n_domains: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
