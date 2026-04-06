from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Create patient-level train/val/test split for TCGA-BRCA.")
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    args = p.parse_args(argv)

    if not np.isclose(args.train + args.val + args.test, 1.0):
        raise ValueError("train+val+test must sum to 1.0")

    paths = BRCAPaths(Path(args.root))
    ensure_dir(paths.splits_dir)

    clinical = pd.read_csv(paths.clinical_csv)
    clinical = clinical.dropna(subset=["patient_id", "os_days", "os_event"])
    patients = clinical["patient_id"].astype(str).unique().tolist()

    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * float(args.train))
    n_val = int(n * float(args.val))
    train_ids = set(patients[:n_train])
    val_ids = set(patients[n_train : n_train + n_val])
    test_ids = set(patients[n_train + n_val :])

    def split_of(pid: str) -> str:
        if pid in train_ids:
            return "train"
        if pid in val_ids:
            return "val"
        if pid in test_ids:
            return "test"
        return "excluded"

    out = clinical[["patient_id", "os_days", "os_event"]].copy()
    out["split"] = out["patient_id"].astype(str).map(split_of)
    out = out[out["split"] != "excluded"].sort_values("patient_id")
    out.to_csv(paths.split_csv, index=False)
    print(f"[ok] wrote split: {paths.split_csv} (train={len(train_ids)} val={len(val_ids)} test={len(test_ids)})")


if __name__ == "__main__":
    main()













