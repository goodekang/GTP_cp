from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch.utils.data import Dataset

from gtp_cp.models.causal_stfm import CausalSTFM, ModelSpec
from gtp_cp.train.engine import evaluate, fit
from gtp_cp.utils.io import ensure_dir
from gtp_cp.utils.seed import seed_everything


class GraphFileDataset(Dataset):
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        rec = torch.load(self.rows.loc[idx, "graph_path"], map_location="cpu")
        g = rec["graph"]
        t = torch.tensor(float(rec["os_days"]), dtype=torch.float32)
        e = torch.tensor(float(rec["os_event"]), dtype=torch.float32)
        return g, t, e


def _collate(batch):
    graphs, times, events = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.stack(times), torch.stack(events)


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train survival model on TCGA-BRCA graphs.")
    p.add_argument("--graphs-index", type=str, default="data/brca/graphs/graphs_index.csv")
    p.add_argument("--out", type=str, default="artifacts/brca")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args(argv)

    seed_everything(int(args.seed))
    device = _device(str(args.device))

    idx = pd.read_csv(args.graphs_index)
    # Need split column; stored in each .pt record, but we can re-read quickly by sampling.
    splits = []
    for gp in idx["graph_path"].astype(str).tolist():
        rec = torch.load(gp, map_location="cpu")
        splits.append(rec.get("split", "train"))
    idx["split"] = splits

    train_df = idx[idx["split"] == "train"].copy()
    val_df = idx[idx["split"] == "val"].copy()
    test_df = idx[idx["split"] == "test"].copy()

    def make_loader(df: pd.DataFrame, shuffle: bool):
        ds = GraphFileDataset(df)
        return torch.utils.data.DataLoader(
            ds, batch_size=int(args.batch_size), shuffle=shuffle, num_workers=0, collate_fn=_collate
        )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    # infer feature_dim from first graph
    sample = torch.load(train_df.iloc[0]["graph_path"], map_location="cpu")["graph"]
    feature_dim = int(sample.x.size(-1))

    model = CausalSTFM(
        ModelSpec(
            feature_dim=feature_dim,
            hidden_dim=int(args.hidden),
            num_layers=int(args.layers),
            num_heads=int(args.heads),
            dropout=float(args.dropout),
            use_causal_mask=False,  # patch-only MVP; causal mask comes with biology priors later
            causal_mask_strength=0.0,
        )
    )

    out_dir = ensure_dir(Path(args.out))
    # Wrap loaders so they are re-iterable across epochs (engine expects an iterable each epoch).
    train_iter = _LoaderAdapter(train_loader)
    val_iter = _LoaderAdapter(val_loader)
    test_iter = _LoaderAdapter(test_loader)
    result = fit(
        model=model,
        train_loader=train_iter,
        val_loader=val_iter,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.wd),
        grad_clip=1.0,
        log_every=10,
        out_dir=out_dir,
        early_stop_patience=0,
    )

    metrics = evaluate(model.to(device), test_iter, device)
    summary = {
        "best_val_cindex": result.best_val_cindex,
        "test_cindex": metrics["cindex"],
        "test_loss": metrics["loss"],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "feature_dim": feature_dim,
    }
    pd.Series(summary).to_csv(out_dir / "metrics_summary.csv")
    print(f"[ok] wrote: {out_dir / 'metrics_summary.csv'}")

    # Export per-patient predictions (Figure-ready CSV)
    _export_predictions(model, idx, out_dir, device, batch_size=int(args.batch_size))


class _B:
    def __init__(self, graph, time, event):
        self.graph = [d for d in graph.to_data_list()]
        self.time = time
        self.event = event


class _LoaderAdapter:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for graph, time, event in self.loader:
            yield _B(graph, time, event)


@torch.no_grad()
def _export_predictions(model, index_df: pd.DataFrame, out_dir: Path, device: torch.device, batch_size: int):
    model.eval()
    rows = []
    for gp in index_df["graph_path"].astype(str).tolist():
        rec = torch.load(gp, map_location="cpu")
        g = rec["graph"]
        batch = Batch.from_data_list([g]).to(device)
        log_risk = float(model(batch).detach().cpu().item())
        rows.append(
            {
                "patient_id": rec.get("patient_id"),
                "split": rec.get("split"),
                "os_days": float(rec.get("os_days")),
                "os_event": float(rec.get("os_event")),
                "log_risk": log_risk,
            }
        )
    pred = pd.DataFrame(rows).sort_values(["split", "patient_id"])
    pred.to_csv(out_dir / "predictions.csv", index=False)
    print(f"[ok] wrote: {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()


