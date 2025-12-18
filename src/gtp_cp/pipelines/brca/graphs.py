from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


def _knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    # coords: [N,2]
    n = coords.shape[0]
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
    # exclude self
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = nn.reshape(-1)
    edges = np.stack([src, dst], axis=0)
    # make undirected
    rev = np.stack([dst, src], axis=0)
    return np.concatenate([edges, rev], axis=1).astype(np.int64)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Build PyG graphs for TCGA-BRCA from pre-extracted patch features."
    )
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument(
        "--features-csv",
        type=str,
        default=None,
        help=(
            "CSV with per-patch features. Required columns: "
            "patient_id, slide_id, x, y, feat_path. "
            "feat_path points to a .npy feature vector file."
        ),
    )
    p.add_argument("--k", type=int, default=8, help="kNN for patch graph edges.")
    args = p.parse_args(argv)

    paths = BRCAPaths(Path(args.root))
    ensure_dir(paths.graphs_dir)

    if args.features_csv is None:
        raise ValueError(
            "--features-csv is required. Produce it from your tiling+feature extractor step.\n"
            "Tip: you can start by generating patch features with any backbone (UNI/CONCH/CLIP) and export to .npy."
        )

    feat_df = pd.read_csv(args.features_csv)
    required = {"patient_id", "slide_id", "x", "y", "feat_path"}
    missing = required - set(feat_df.columns)
    if missing:
        raise ValueError(f"features_csv missing columns: {sorted(missing)}")

    label_df = pd.read_csv(paths.split_csv)  # patient_id, os_days, os_event, split
    label_df = label_df.set_index("patient_id")

    graph_rows = []
    for patient_id, gdf in feat_df.groupby("patient_id"):
        if patient_id not in label_df.index:
            continue

        coords = gdf[["x", "y"]].to_numpy(dtype=np.float32)
        edge_index = _knn_edges(coords, k=int(args.k))

        feats = []
        for fp in gdf["feat_path"].astype(str).tolist():
            v = np.load(fp).astype(np.float32)
            feats.append(v)
        x = np.stack(feats, axis=0)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            node_type=torch.zeros((x.shape[0],), dtype=torch.long),  # patch-only MVP
        )

        out_path = paths.graphs_dir / f"{patient_id}.pt"
        torch.save(
            {
                "graph": data,
                "patient_id": patient_id,
                "os_days": float(label_df.loc[patient_id, "os_days"]),
                "os_event": float(label_df.loc[patient_id, "os_event"]),
                "split": str(label_df.loc[patient_id, "split"]),
            },
            out_path,
        )
        graph_rows.append({"patient_id": patient_id, "graph_path": str(out_path)})

    idx = pd.DataFrame(graph_rows).sort_values("patient_id")
    idx.to_csv(paths.graphs_index, index=False)
    print(f"[ok] wrote graphs: {paths.graphs_dir} (patients={len(idx)})")
    print(f"[ok] wrote index:  {paths.graphs_index}")


if __name__ == "__main__":
    main()





