from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BRCAPaths:
    root: Path

    @property
    def registry(self) -> Path:
        return self.root / "registry.yaml"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def manifest_wsi(self) -> Path:
        return self.manifest_dir / "tcga_brca_wsi_manifest.csv"

    @property
    def clinical_dir(self) -> Path:
        return self.root / "clinical"

    @property
    def clinical_csv(self) -> Path:
        return self.clinical_dir / "tcga_brca_clinical.csv"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    @property
    def split_csv(self) -> Path:
        return self.splits_dir / "patient_split.csv"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def graphs_dir(self) -> Path:
        return self.root / "graphs"

    @property
    def graphs_index(self) -> Path:
        return self.graphs_dir / "graphs_index.csv"





