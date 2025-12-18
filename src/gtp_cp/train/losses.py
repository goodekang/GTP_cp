from __future__ import annotations

import torch


def cox_partial_likelihood_loss(log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Negative Cox partial log-likelihood (Breslow).
    log_risk: shape [B] (higher => higher hazard)
    time: shape [B]
    event: shape [B] in {0,1}
    """
    # sort by time descending (risk set: those with time >= current)
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]

    # logsumexp over risk set is cumulative since sorted descending
    log_cum_sum_exp = torch.logcumsumexp(log_risk, dim=0)
    # contributions only where event==1
    neg_log_lik = -(log_risk - log_cum_sum_exp) * event
    denom = event.sum().clamp_min(1.0)
    return neg_log_lik.sum() / denom





