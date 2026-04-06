from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphBuildSpec:
    """Parameters for hierarchical (L0/L1/L2) tissue graph construction."""

    feature_dim: int = 64
    spatial_radius_cells: float = 50.0
    motif_radius_um: float = 100.0
    sigma_um: float = 25.0


def gaussian_spatial_weight(dist_um: np.ndarray, sigma_um: float) -> np.ndarray:
    """Edge weight w_ij = exp(-d_ij^2 / (2 sigma^2)) as in Methods."""
    return np.exp(-(dist_um.astype(np.float64) ** 2) / (2.0 * float(sigma_um) ** 2)).astype(np.float32)


class GraphBuilder:
    """
    Build a PyG ``Data`` object from tabular features and edge indices.

    Expected node order (optional L2 tissue nodes at the end):
    ``[patches | cells | genes | tissue_motifs]`` with ``node_type`` in
    ``{0: patch, 1: cell, 2: gene, 3: tissue}``.
    """

    def __init__(self, spec: GraphBuildSpec):
        self.spec = spec

    def build(self, raw: object) -> Data:
        if isinstance(raw, dict):
            return self.build_from_arrays(**raw)
        raise TypeError("Pass a dict of arrays to ``build`` or call ``build_from_arrays``.")

    def build_from_arrays(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        node_type: np.ndarray,
        *,
        edge_attr: np.ndarray | None = None,
        edge_bio: np.ndarray | None = None,
        pos: np.ndarray | None = None,
        batch_index: int | None = None,
        **extra: Any,
    ) -> Data:
        ei = np.asarray(edge_index, dtype=np.int64)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")

        d: dict[str, Any] = {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "edge_index": torch.from_numpy(ei.contiguous()),
            "node_type": torch.from_numpy(np.asarray(node_type, dtype=np.int64)),
        }
        if edge_attr is not None:
            ea = np.asarray(edge_attr, dtype=np.float32)
            if ea.ndim == 1:
                ea = ea.reshape(-1, 1)
            d["edge_attr"] = torch.from_numpy(ea)
        if edge_bio is not None:
            d["edge_bio"] = torch.from_numpy(np.asarray(edge_bio, dtype=np.float32).reshape(-1))
        if pos is not None:
            d["pos"] = torch.from_numpy(np.asarray(pos, dtype=np.float32))
        if batch_index is not None:
            d["slide_id"] = torch.tensor([int(batch_index)], dtype=torch.long)
        for k, v in extra.items():
            if isinstance(v, torch.Tensor):
                d[k] = v
            elif isinstance(v, np.ndarray):
                d[k] = torch.from_numpy(v)
            else:
                d[k] = v
        return Data(**d)


def concatenate_graphs(graphs: list[Data]) -> Data:
    """Concatenate disjoint graphs into one (single-slide or batch preprocessing)."""
    from torch_geometric.data import Batch

    return Batch.from_data_list(graphs)
