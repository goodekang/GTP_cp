from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool


@dataclass(frozen=True)
class ModelSpec:
    """
    Bio-STFM / Causal-STFM configuration aligned with the manuscript (Methods).

    - Dual-stream encoder: ViT patch embeddings (768) and Geneformer gene embeddings (256)
      are projected to a common ``hidden_dim`` when ``use_dual_stream`` is True.
    - ``use_causal_mask`` controls a demo-style structural bias; supply ``edge_bio`` on
      ``Data`` for knowledge-guided edge gating (PPI / pathway priors).
    """

    feature_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    use_causal_mask: bool = True
    causal_mask_strength: float = 1.0
    patch_dim: int | None = None
    gene_dim: int | None = None
    use_dual_stream: bool = False
    use_mil_pooling: bool = True
    num_node_types: int = 4
    pool_hidden_dim: int | None = None


class BioSTFM(nn.Module):
    """
    Biology-constrained graph transformer (Bio-STFM).

    Stack of ``TransformerConv`` layers with optional dual-stream input projection,
    knowledge-guided edge weights, and attention-weighted (MIL) or mean pooling over
    patch nodes for Cox log-risk prediction.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        pdim = spec.patch_dim if spec.patch_dim is not None else spec.feature_dim
        gdim = spec.gene_dim if spec.gene_dim is not None else spec.feature_dim
        self._pdim = pdim
        self._gdim = gdim

        self.type_emb = nn.Embedding(spec.num_node_types, spec.hidden_dim)
        if spec.use_dual_stream:
            self.proj_patch = nn.Linear(pdim, spec.hidden_dim)
            self.proj_gene = nn.Linear(gdim, spec.hidden_dim)
            self.in_proj = None
        else:
            self.proj_patch = None
            self.proj_gene = None
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

        self.norm1 = nn.ModuleList([nn.LayerNorm(spec.hidden_dim) for _ in range(spec.num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(spec.hidden_dim) for _ in range(spec.num_layers)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(spec.hidden_dim, spec.hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(spec.dropout),
                    nn.Linear(spec.hidden_dim * 4, spec.hidden_dim),
                    nn.Dropout(spec.dropout),
                )
                for _ in range(spec.num_layers)
            ]
        )

        pool_h = spec.pool_hidden_dim or max(spec.hidden_dim // 2, 32)
        self.mil_attn = nn.Sequential(
            nn.Linear(spec.hidden_dim, pool_h),
            nn.Tanh(),
            nn.Linear(pool_h, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(spec.hidden_dim),
            nn.Linear(spec.hidden_dim, spec.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim // 2, 1),
        )

    def _embed_nodes(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        if not self.spec.use_dual_stream:
            return self.in_proj(x) + self.type_emb(node_type)

        # 0=patch, 3=tissue motif: patch stream; 1=cell, 2=gene: gene stream
        patch_like = (node_type == 0) | (node_type == 3)
        gene_like = (node_type == 1) | (node_type == 2)
        h = torch.zeros(x.size(0), self.spec.hidden_dim, device=x.device, dtype=x.dtype)
        if patch_like.any():
            xp = x[patch_like]
            if xp.size(1) != self._pdim:
                xp = xp[..., : self._pdim]
            h[patch_like] = self.proj_patch(xp)
        if gene_like.any():
            xg = x[gene_like]
            if xg.size(1) != self._gdim:
                xg = xg[..., : self._gdim]
            h[gene_like] = self.proj_gene(xg)
        return h + self.type_emb(node_type)

    def _edge_weight(
        self,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_bio: torch.Tensor | None,
    ) -> torch.Tensor:
        e = edge_index.size(1)
        w = torch.ones((e, 1), device=edge_index.device, dtype=torch.float32)
        if edge_bio is not None:
            w = w * edge_bio.view(-1, 1).clamp(0.0, 1.0)

        if not self.spec.use_causal_mask:
            return w

        src, dst = edge_index[0], edge_index[1]
        same = (node_type[src] == node_type[dst]).float().unsqueeze(-1)
        w = w * (1.0 - 0.3 * self.spec.causal_mask_strength * same).clamp_min(0.1)
        return w

    def _pool_patches(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        patch_mask = node_type == 0
        x_patch = x[patch_mask]
        batch_patch = batch[patch_mask]
        if not self.spec.use_mil_pooling:
            return global_mean_pool(x_patch, batch_patch)

        # Attention-weighted MIL pooling (TransMIL-style) per graph.
        logits = self.mil_attn(x_patch).squeeze(-1)
        max_g = int(batch_patch.max().item()) + 1 if batch_patch.numel() else 0
        out = []
        for g in range(max_g):
            sel = batch_patch == g
            if not torch.any(sel):
                out.append(torch.zeros(self.spec.hidden_dim, device=x.device, dtype=x.dtype))
                continue
            lp = logits[sel]
            a = F.softmax(lp, dim=0)
            out.append((a.unsqueeze(-1) * x_patch[sel]).sum(dim=0))
        return torch.stack(out, dim=0)

    def encode(self, data) -> torch.Tensor:
        x = data.x
        node_type = data.node_type
        edge_index = data.edge_index
        batch = getattr(
            data,
            "batch",
            torch.zeros(x.size(0), device=x.device, dtype=torch.long),
        )
        edge_bio = getattr(data, "edge_bio", None)

        x = self._embed_nodes(x, node_type)
        edge_attr = self._edge_weight(node_type, edge_index, edge_bio)

        for conv, n1, n2, ffn in zip(self.convs, self.norm1, self.norm2, self.ffn):
            h = conv(x, edge_index, edge_attr=edge_attr)
            x = n1(x + h)
            x = n2(x + ffn(x))
        return x, node_type, batch

    def forward(self, data) -> torch.Tensor:
        x, node_type, batch = self.encode(data)
        pooled = self._pool_patches(x, node_type, batch)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def predict_log_risk(self, data) -> torch.Tensor:
        self.eval()
        return self.forward(data)


class CausalSTFM(BioSTFM):
    """Backward-compatible alias for ``BioSTFM``."""

    pass


def mtm_gene_head(hidden_dim: int, n_genes: int) -> nn.Module:
    """Linear decoder for masked-tissue modeling (gene expression reconstruction)."""
    return nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_genes))

