from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Convert tcga_brca_wsi_manifest.csv into a gdc-client manifest.tsv."
    )
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument(
        "--in-csv",
        type=str,
        default=None,
        help="Input CSV (default: data/brca/manifests/tcga_brca_wsi_manifest.csv).",
    )
    p.add_argument(
        "--out-tsv",
        type=str,
        default=None,
        help="Output TSV path (default: data/brca/manifests/gdc_manifest.tsv).",
    )
    p.add_argument(
        "--svs-only",
        action="store_true",
        help="Keep only .svs files (recommended).",
    )
    args = p.parse_args(argv)

    paths = BRCAPaths(Path(args.root))
    ensure_dir(paths.manifest_dir)

    in_csv = Path(args.in_csv) if args.in_csv else paths.manifest_wsi
    out_tsv = Path(args.out_tsv) if args.out_tsv else (paths.manifest_dir / "gdc_manifest.tsv")

    df = pd.read_csv(in_csv)
    if args.svs_only:
        df = df[df["file_name"].astype(str).str.lower().str.endswith(".svs")]

    # gdc-client expects: id filename md5 size
    out = pd.DataFrame(
        {
            "id": df["file_id"].astype(str),
            "filename": df["file_name"].astype(str),
            "md5": df.get("md5sum", pd.Series([None] * len(df))).astype(str),
            "size": df.get("file_size", pd.Series([None] * len(df))),
        }
    )
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"[ok] wrote gdc-client manifest: {out_tsv} (rows={len(out)})")


if __name__ == "__main__":
    main()













