from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from gtp_cp.graphs.knowledge import random_ppi_edges


@dataclass(frozen=True)
class SyntheticSpec:
    n_patch: int
    n_cell_per_patch: int
    n_gene_per_cell: int
    feature_dim: int
    patch_radius: int
    n_motif: int = 0
    use_gene_ppi: bool = False
    ppi_edges_per_graph: int = 32


def _make_patch_grid_edges(n_patch: int, radius: int) -> np.ndarray:
    side = int(np.sqrt(n_patch))
    if side * side != n_patch:
        edges = []
        for i in range(n_patch):
            for d in range(1, radius + 1):
                j = (i + d) % n_patch
                edges.append((i, j))
                edges.append((j, i))
        return np.array(edges, dtype=np.int64).T

    coords = [(i // side, i % side) for i in range(n_patch)]
    edges = []
    for i, (r1, c1) in enumerate(coords):
        for j, (r2, c2) in enumerate(coords):
            if i == j:
                continue
            if abs(r1 - r2) + abs(c1 - c2) <= radius:
                edges.append((i, j))
    return np.array(edges, dtype=np.int64).T


def generate_patient_graph(rng: np.random.Generator, spec: SyntheticSpec) -> tuple[Data, float, int]:
    """
    Returns ``(graph, time, event)`` for one synthetic patient.

    Node types:
    - 0: patch (WSI tile)
    - 1: cell / spot
    - 2: gene
    - 3: tissue motif (L2), optional when ``n_motif > 0``
    """
    n_patch = spec.n_patch
    n_cell = spec.n_patch * spec.n_cell_per_patch
    n_gene = n_cell * spec.n_gene_per_cell
    n_motif = int(spec.n_motif)
    n_total = n_patch + n_cell + n_gene + n_motif

    x = rng.standard_normal((n_total, spec.feature_dim)).astype(np.float32)
    node_type = np.zeros((n_total,), dtype=np.int64)
    node_type[n_patch : n_patch + n_cell] = 1
    node_type[n_patch + n_cell : n_patch + n_cell + n_gene] = 2
    if n_motif > 0:
        node_type[n_patch + n_cell + n_gene :] = 3

    patch_edges = _make_patch_grid_edges(n_patch, spec.patch_radius)

    pc_edges = []
    for p in range(n_patch):
        for k in range(spec.n_cell_per_patch):
            c = p * spec.n_cell_per_patch + k
            c_global = n_patch + c
            pc_edges.append((p, c_global))
            pc_edges.append((c_global, p))
    pc_edges = np.array(pc_edges, dtype=np.int64).T

    cg_edges = []
    for c in range(n_cell):
        c_global = n_patch + c
        for g in range(spec.n_gene_per_cell):
            g_global = n_patch + n_cell + (c * spec.n_gene_per_cell + g)
            cg_edges.append((c_global, g_global))
            cg_edges.append((g_global, c_global))
    cg_edges = np.array(cg_edges, dtype=np.int64).T

    edge_parts = [patch_edges, pc_edges, cg_edges]

    if n_motif > 0:
        base_m = n_patch + n_cell + n_gene
        mc_edges = []
        for m in range(n_motif):
            m_global = base_m + m
            picks = rng.choice(n_cell, size=max(1, n_cell // 8), replace=False)
            for c in picks:
                c_global = n_patch + int(c)
                mc_edges.append((m_global, c_global))
                mc_edges.append((c_global, m_global))
        edge_parts.append(np.array(mc_edges, dtype=np.int64).T)

    edge_index = np.concatenate(edge_parts, axis=1)

    driver_patch = int(rng.integers(0, n_patch))
    driver_cells = [
        n_patch + driver_patch * spec.n_cell_per_patch + k for k in range(spec.n_cell_per_patch)
    ]
    signal = x[driver_cells].mean(axis=0)
    true_log_risk = float(0.8 * signal[0] - 0.6 * signal[1] + 0.3 * signal[2])

    rate = float(np.exp(np.clip(true_log_risk, -3.0, 3.0)))
    time = float(rng.exponential(scale=1.0 / rate))
    censor_time = float(rng.exponential(scale=1.2))
    event = int(time <= censor_time)
    observed_time = min(time, censor_time)

    gene_start = n_patch + n_cell
    gene_index = np.full((n_total,), -1, dtype=np.int64)
    for gi in range(n_gene):
        gene_index[gene_start + gi] = gi

    edge_bio = np.ones((edge_index.shape[1],), dtype=np.float32)
    if spec.use_gene_ppi and n_gene > 1:
        ppi = random_ppi_edges(
            n_genes=min(n_gene, 256),
            n_edges=min(spec.ppi_edges_per_graph, n_gene * 2),
            rng=rng,
        )
        g_src = ppi[0] + gene_start
        g_dst = ppi[1] + gene_start
        extra = np.stack([g_src, g_dst], axis=0)
        edge_index = np.concatenate([edge_index, extra], axis=1)
        edge_bio = np.concatenate(
            [edge_bio, np.ones((extra.shape[1],), dtype=np.float32)],
            axis=0,
        )

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        node_type=torch.from_numpy(node_type),
        gene_index=torch.from_numpy(gene_index),
        edge_bio=torch.from_numpy(edge_bio),
        driver_patch=torch.tensor([driver_patch], dtype=torch.long),
    )
    return data, observed_time, event


def generate_dataset(seed: int, n: int, spec: SyntheticSpec) -> list[tuple[Data, float, int]]:
    rng = np.random.default_rng(seed)
    return [generate_patient_graph(rng, spec) for _ in range(n)]
