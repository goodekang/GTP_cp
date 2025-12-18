from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class SyntheticSpec:
    n_patch: int
    n_cell_per_patch: int
    n_gene_per_cell: int
    feature_dim: int
    patch_radius: int


def _make_patch_grid_edges(n_patch: int, radius: int) -> np.ndarray:
    """
    Create edges on a sqrt(n_patch) x sqrt(n_patch) grid.
    For simplicity, require perfect square; if not, fall back to ring graph.
    """
    side = int(np.sqrt(n_patch))
    if side * side != n_patch:
        # ring
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
    Returns (graph, time, event) for one synthetic patient.

    Graph contains 3 node types encoded in node_type:
    - 0: patch
    - 1: cell
    - 2: gene
    """
    n_patch = spec.n_patch
    n_cell = spec.n_patch * spec.n_cell_per_patch
    n_gene = n_cell * spec.n_gene_per_cell
    n_total = n_patch + n_cell + n_gene

    # Node features
    x = rng.standard_normal((n_total, spec.feature_dim)).astype(np.float32)
    node_type = np.zeros((n_total,), dtype=np.int64)
    node_type[n_patch : n_patch + n_cell] = 1
    node_type[n_patch + n_cell :] = 2

    # Edges:
    # patch<->patch (spatial)
    patch_edges = _make_patch_grid_edges(n_patch, spec.patch_radius)

    # patch<->cell (cells belong to patches)
    pc_edges = []
    for p in range(n_patch):
        for k in range(spec.n_cell_per_patch):
            c = p * spec.n_cell_per_patch + k
            c_global = n_patch + c
            pc_edges.append((p, c_global))
            pc_edges.append((c_global, p))
    pc_edges = np.array(pc_edges, dtype=np.int64).T

    # cell<->gene (genes belong to cells)
    cg_edges = []
    for c in range(n_cell):
        c_global = n_patch + c
        for g in range(spec.n_gene_per_cell):
            g_global = n_patch + n_cell + (c * spec.n_gene_per_cell + g)
            cg_edges.append((c_global, g_global))
            cg_edges.append((g_global, c_global))
    cg_edges = np.array(cg_edges, dtype=np.int64).T

    edge_index = np.concatenate([patch_edges, pc_edges, cg_edges], axis=1)

    # --- "Causal" ground-truth mechanism (toy) ---
    # Choose one patch as driver; its mean cell features influence hazard.
    driver_patch = int(rng.integers(0, n_patch))
    driver_cells = [
        n_patch + driver_patch * spec.n_cell_per_patch + k for k in range(spec.n_cell_per_patch)
    ]
    signal = x[driver_cells].mean(axis=0)
    true_log_risk = float(0.8 * signal[0] - 0.6 * signal[1] + 0.3 * signal[2])

    # event time ~ exponential with rate exp(true_log_risk)
    rate = float(np.exp(np.clip(true_log_risk, -3.0, 3.0)))
    time = float(rng.exponential(scale=1.0 / rate))
    # censoring
    censor_time = float(rng.exponential(scale=1.2))
    event = int(time <= censor_time)
    observed_time = min(time, censor_time)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        node_type=torch.from_numpy(node_type),
        driver_patch=torch.tensor([driver_patch], dtype=torch.long),
    )
    return data, observed_time, event


def generate_dataset(seed: int, n: int, spec: SyntheticSpec) -> list[tuple[Data, float, int]]:
    rng = np.random.default_rng(seed)
    return [generate_patient_graph(rng, spec) for _ in range(n)]






