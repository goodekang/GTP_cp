from __future__ import annotations

import torch
from torch_geometric.data import Data


def clone_graph(data: Data) -> Data:
    """Shallow-safe clone for perturbation (tensors copied)."""
    return data.clone()


def zero_edge_subset(data: Data, edge_mask_drop: torch.Tensor) -> Data:
    """
    Set ``edge_bio`` or edge weights to zero for edges where ``edge_mask_drop`` is True.

    If ``edge_bio`` is absent, it is created as ones then masked.
    """
    out = data.clone()
    e = out.edge_index.size(1)
    if edge_mask_drop.numel() != e:
        raise ValueError("edge_mask_drop must have length equal to number of edges.")
    keep = (~edge_mask_drop).float()
    if hasattr(out, "edge_bio") and out.edge_bio is not None:
        out.edge_bio = out.edge_bio * keep
    else:
        out.edge_bio = keep
    return out


def delta_risk(
    model: torch.nn.Module,
    data: Data,
    perturbed: Data,
) -> torch.Tensor:
    """
    In silico perturbation: ΔRisk = f(G) - f(G') (scalar per graph or batch).

    Positive values (for higher risk = worse prognosis) mean the ablated component
    was associated with increased risk in the baseline model.
    """
    model.eval()
    with torch.no_grad():
        r0 = model(data)
        r1 = model(perturbed)
    return r0 - r1


def ablate_edges_by_index(data: Data, drop_indices: torch.Tensor) -> Data:
    """Convenience: build edge_mask_drop from flat indices in [0, E)."""
    e = data.edge_index.size(1)
    m = torch.zeros((e,), dtype=torch.bool, device=data.edge_index.device)
    m[drop_indices.long()] = True
    return zero_edge_subset(data, m)


def ablate_node_features(data: Data, node_indices: torch.Tensor, *, fill: float = 0.0) -> Data:
    """Set selected node feature rows to ``fill`` (node ablation, Methods)."""
    out = data.clone()
    x = out.x.clone()
    x[node_indices.long()] = fill
    out.x = x
    return out
