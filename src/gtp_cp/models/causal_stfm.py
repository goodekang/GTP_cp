from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool


@dataclass(frozen=True)
class ModelSpec:
    feature_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    use_causal_mask: bool
    causal_mask_strength: float


class CausalSTFM(nn.Module):
    """
    Minimal Causal-STFM-style model:
    - Node type embedding (patch/cell/gene)
    - Stacked graph transformer conv
    - Patch-level pooling -> survival risk head

    Notes:
    - The "causal mask" here is a *demo hook*. In the full model it should be
      derived from biological priors (PPI/Reactome/ligand-receptor) and applied
      as an attention constraint. This scaffold keeps the interface clean.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec

        self.type_emb = nn.Embedding(3, spec.feature_dim)  # 0 patch, 1 cell, 2 gene
        self.in_proj = nn.Linear(spec.feature_dim, spec.hidden_dim)

        convs = []
        for _ in range(spec.num_layers):
            convs.append(
                TransformerConv(
                    in_channels=spec.hidden_dim,
                    out_channels=spec.hidden_dim // spec.num_heads,
                    heads=spec.num_heads,
                    dropout=spec.dropout,
                    edge_dim=1,
                    beta=True,
                )
            )
        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList([nn.LayerNorm(spec.hidden_dim) for _ in range(spec.num_layers)])
        self.ffn = nn.Sequential(
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(spec.hidden_dim),
            nn.Linear(spec.hidden_dim, spec.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim // 2, 1),
        )

    def _edge_weight(self, node_type: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Default: uniform edges.
        w = torch.ones((edge_index.size(1), 1), device=edge_index.device, dtype=torch.float32)
        if not self.spec.use_causal_mask:
            return w

        # Demo "mask": downweight same-type edges slightly, keep cross-scale edges.
        src, dst = edge_index[0], edge_index[1]
        same = (node_type[src] == node_type[dst]).float().unsqueeze(-1)
        w = w * (1.0 - 0.3 * self.spec.causal_mask_strength * same).clamp_min(0.1)
        return w

    def forward(self, data) -> torch.Tensor:
        x = data.x
        node_type = data.node_type
        edge_index = data.edge_index
        batch = getattr(data, "batch", torch.zeros(x.size(0), device=x.device, dtype=torch.long))

        x = x + self.type_emb(node_type)
        x = self.in_proj(x)

        edge_attr = self._edge_weight(node_type, edge_index)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + h)
            x = norm(x + self.ffn(x))

        patch_mask = node_type == 0
        x_patch = x[patch_mask]
        batch_patch = batch[patch_mask]
        pooled = global_mean_pool(x_patch, batch_patch)
        log_risk = self.head(pooled).squeeze(-1)
        return log_risk






