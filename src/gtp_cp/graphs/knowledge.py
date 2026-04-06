from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class KnowledgeEdgeSpec:
    """Metadata for PPI / pathway / ligand-receptor priors."""

    n_nodes: int
    undirected: bool = True


def edges_to_adjacency(n_nodes: int, edge_index: np.ndarray, undirected: bool = True) -> np.ndarray:
    """Dense 0/1 adjacency for gene--gene or entity--entity priors."""
    ei = np.asarray(edge_index, dtype=np.int64)
    a = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if ei.size == 0:
        return a
    src, dst = ei[0], ei[1]
    a[src, dst] = 1.0
    if undirected:
        a[dst, src] = 1.0
    np.fill_diagonal(a, 1.0)
    return a


def attention_bias_from_adjacency(adj: torch.Tensor, fill_value: float = -1e4) -> torch.Tensor:
    """
    Map adjacency to additive attention bias: allowed edges -> 0, disallowed -> fill_value.

    For softmax(QK^T/sqrt(d) + M), use ``fill_value`` large negative to zero unsupported pairs.
    """
    if adj.ndim != 2:
        raise ValueError("adj must be [N, N].")
    mask = adj > 0.5
    bias = torch.full_like(adj, fill_value, dtype=torch.float32)
    bias[mask] = 0.0
    return bias


def edge_bio_from_gene_adjacency(
    edge_index: np.ndarray,
    gene_index: np.ndarray,
    adj_gene: np.ndarray,
) -> np.ndarray:
    """
    Per-edge biological plausibility: 1 if both endpoints map to genes with adj[i_g, j_g] > 0.

    ``gene_index`` maps node id -> gene id in ``[0, G)`` or -1 if not a gene node.
    """
    ei = np.asarray(edge_index, dtype=np.int64)
    gmap = np.asarray(gene_index, dtype=np.int64)
    e = ei.shape[1]
    out = np.ones((e,), dtype=np.float32)
    src, dst = ei[0], ei[1]
    for k in range(e):
        gs, gd = int(gmap[src[k]]), int(gmap[dst[k]])
        if gs < 0 or gd < 0:
            continue
        if gs >= adj_gene.shape[0] or gd >= adj_gene.shape[1]:
            continue
        out[k] = float(adj_gene[gs, gd] > 0.5)
    return out


def random_ppi_edges(
    n_genes: int,
    n_edges: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetic PPI edge list for demos when STRING/Reactome files are absent."""
    e = max(1, min(n_edges, n_genes * (n_genes - 1) // 2))
    seen: set[tuple[int, int]] = set()
    rows: list[tuple[int, int]] = []
    while len(rows) < e:
        i = int(rng.integers(0, n_genes))
        j = int(rng.integers(0, n_genes))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append((a, b))
    arr = np.array(rows, dtype=np.int64).T
    both = np.concatenate([arr, arr[::-1]], axis=1)
    return both
