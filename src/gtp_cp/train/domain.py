from __future__ import annotations

from dataclasses import dataclass

from gtp_cp.train.losses import DomainHead, domain_classification_loss, gradient_reversal


@dataclass(frozen=True)
class DomainAdaptConfig:
    lambda_grl: float = 0.1
    hidden_dim: int = 256
    n_domains: int = 2


def domain_step(
    *,
    pooled: torch.Tensor,
    domain_labels: torch.Tensor,
    head: DomainHead,
    cfg: DomainAdaptConfig,
) -> torch.Tensor:
    """
    Forward pass for domain adversarial loss with gradient reversal on ``pooled`` features.
    """
    z = gradient_reversal(pooled, cfg.lambda_grl)
    logits = head(z)
    return domain_classification_loss(logits, domain_labels)
