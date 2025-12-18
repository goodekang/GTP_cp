from __future__ import annotations

from dataclasses import dataclass

from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphBuildSpec:
    """
    Placeholder for real HTAN/TCGA ingestion.
    In production, this would include:
    - WSI tiling params
    - superpixel/patch graph construction
    - cell mapping + spatial coordinates
    - gene selection / GRN edges
    """

    feature_dim: int = 64


class GraphBuilder:
    """
    Interface for converting raw multi-modal inputs into a PyG `Data` object.
    """

    def __init__(self, spec: GraphBuildSpec):
        self.spec = spec

    def build(self, raw: object) -> Data:
        raise NotImplementedError("Implement HTAN/TCGA graph construction here.")





