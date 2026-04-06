"""Training loops, losses, and perturbation utilities."""

from gtp_cp.train.domain import DomainAdaptConfig, domain_step
from gtp_cp.train.engine import FitResult, evaluate, fit
from gtp_cp.train.losses import (
    DomainHead,
    GradientReversalFn,
    cox_partial_likelihood_loss,
    domain_classification_loss,
    gradient_reversal,
    mtm_mse_loss,
)
from gtp_cp.train.perturbation import ablate_edges_by_index, ablate_node_features, delta_risk, zero_edge_subset

__all__ = [
    "FitResult",
    "evaluate",
    "fit",
    "cox_partial_likelihood_loss",
    "mtm_mse_loss",
    "domain_classification_loss",
    "DomainHead",
    "GradientReversalFn",
    "gradient_reversal",
    "DomainAdaptConfig",
    "domain_step",
    "delta_risk",
    "zero_edge_subset",
    "ablate_edges_by_index",
    "ablate_node_features",
]
