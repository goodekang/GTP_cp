from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gtp_cp.pipelines.brca.gdc_api import query_tcga_brca_diagnostic_slides
from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate TCGA-BRCA WSI manifest from GDC API.")
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument("--size", type=int, default=20000, help="Max number of records to fetch from GDC.")
    args = p.parse_args(argv)

    paths = BRCAPaths(Path(args.root))
    ensure_dir(paths.manifest_dir)

    records = query_tcga_brca_diagnostic_slides(size=int(args.size))
    df = pd.DataFrame([r.__dict__ for r in records])

    # Normalize expected columns
    # submitter_id is the case-level TCGA barcode; we treat it as patient key (first 12 chars).
    df["patient_id"] = df["submitter_id"].fillna("").astype(str).str.slice(0, 12)
    df["is_svs"] = df["file_name"].fillna("").astype(str).str.lower().str.endswith(".svs")

    df = df.sort_values(["patient_id", "file_name"])
    df.to_csv(paths.manifest_wsi, index=False)
    print(f"[ok] wrote manifest: {paths.manifest_wsi} (rows={len(df)})")
    print("Next: download SVS via gdc-client using file_id list (see docs/BRCA_PIPELINE.md).")


if __name__ == "__main__":
    main()






