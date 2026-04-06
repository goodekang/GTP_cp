"""
Complete Preprocessing Pipeline for TCGA-BRCA.

This script orchestrates the full data preparation workflow:
1. Generate WSI manifest from GDC API
2. Fetch clinical survival data
3. Create train/val/test split
4. Tile WSI slides into patches (requires downloaded SVS files)
5. Extract features from patches
6. Build graphs for training

Usage:
    # Full pipeline (after downloading WSI files)
    gtp-brca-preprocess --root data/brca --wsi-dir data/brca/raw/gdc_wsi

    # Skip to specific step
    gtp-brca-preprocess --root data/brca --start-from features
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


STEPS = ["manifest", "clinical", "split", "tile", "features", "graphs"]


def _run_cmd(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[step] {description}")
    print(f"[cmd]  {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"[error] Step failed: {description}")
        return False
    
    print(f"[ok] Completed: {description}")
    return True


def run_pipeline(
    root: Path,
    wsi_dir: Path | None,
    model: str,
    k: int,
    seed: int,
    start_from: str,
    device: str,
) -> None:
    """Run the full preprocessing pipeline."""
    paths = BRCAPaths(root)
    ensure_dir(root)
    
    # Determine which steps to run
    if start_from not in STEPS:
        raise ValueError(f"Invalid start_from: {start_from}. Choose from: {STEPS}")
    
    start_idx = STEPS.index(start_from)
    steps_to_run = STEPS[start_idx:]
    
    print(f"\n[info] Pipeline root: {root}")
    print(f"[info] Steps to run: {steps_to_run}")
    
    python = sys.executable
    
    # Step 1: Generate manifest
    if "manifest" in steps_to_run:
        if not _run_cmd(
            [python, "-m", "gtp_cp.pipelines.brca.manifest", "--root", str(root)],
            "Generate WSI manifest from GDC API"
        ):
            return
    
    # Step 2: Fetch clinical data
    if "clinical" in steps_to_run:
        if not _run_cmd(
            [python, "-m", "gtp_cp.pipelines.brca.clinical", "--root", str(root)],
            "Fetch clinical survival data"
        ):
            return
    
    # Step 3: Create split
    if "split" in steps_to_run:
        if not _run_cmd(
            [python, "-m", "gtp_cp.pipelines.brca.split", 
             "--root", str(root), "--seed", str(seed)],
            "Create train/val/test split"
        ):
            return
    
    # Step 4: Tile WSI
    if "tile" in steps_to_run:
        if wsi_dir is None:
            wsi_dir = root / "raw" / "gdc_wsi"
        
        if not wsi_dir.exists():
            print(f"\n[warn] WSI directory not found: {wsi_dir}")
            print("[info] Please download WSI files first:")
            print(f"       1. Download gdc-client: .\\scripts\\brca\\download_gdc_client.ps1")
            print(f"       2. Download WSIs: .\\scripts\\brca\\download_gdc_wsi.ps1 -Root {root}")
            print(f"\n[info] Or use demo mode with synthetic data: gtp-train")
            print(f"\n[skip] Skipping tiling step...")
        else:
            if not _run_cmd(
                [python, "-m", "gtp_cp.pipelines.brca.tiling",
                 "--root", str(root), "--wsi-dir", str(wsi_dir)],
                "Tile WSI slides into patches"
            ):
                return
    
    # Step 5: Extract features
    if "features" in steps_to_run:
        patches_csv = root / "patches" / "patches_index.csv"
        if not patches_csv.exists():
            print(f"\n[warn] Patches index not found: {patches_csv}")
            print("[skip] Skipping feature extraction step...")
        else:
            if not _run_cmd(
                [python, "-m", "gtp_cp.pipelines.brca.features",
                 "--root", str(root),
                 "--model", model,
                 "--device", device],
                f"Extract features using {model}"
            ):
                return
    
    # Step 6: Build graphs
    if "graphs" in steps_to_run:
        features_csv = paths.features_dir / "patch_features.csv"
        if not features_csv.exists():
            print(f"\n[warn] Features CSV not found: {features_csv}")
            print("[skip] Skipping graph building step...")
        else:
            if not _run_cmd(
                [python, "-m", "gtp_cp.pipelines.brca.graphs",
                 "--root", str(root),
                 "--features-csv", str(features_csv),
                 "--k", str(k)],
                f"Build kNN graphs (k={k})"
            ):
                return
    
    # Summary
    print("\n" + "="*60)
    print("[done] Preprocessing pipeline complete!")
    print("="*60)
    
    graphs_index = paths.graphs_index
    if graphs_index.exists():
        print(f"\n[ready] You can now train the model:")
        print(f"        gtp-brca-train --graphs-index {graphs_index} --out artifacts/brca --device {device}")
    else:
        print(f"\n[info] Graph index not found. Complete all preprocessing steps first.")
        print(f"[info] Or use synthetic data demo: gtp-train")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Run complete TCGA-BRCA preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (requires downloaded WSI files)
  gtp-brca-preprocess --root data/brca --wsi-dir data/brca/raw/gdc_wsi

  # Only run metadata steps (no WSI needed)
  gtp-brca-preprocess --root data/brca --start-from manifest

  # Resume from feature extraction
  gtp-brca-preprocess --root data/brca --start-from features --model resnet50

Available steps: manifest -> clinical -> split -> tile -> features -> graphs
"""
    )
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument("--wsi-dir", type=str, default=None, help="Directory with downloaded WSI files.")
    p.add_argument("--model", type=str, default="resnet50",
                   choices=["resnet50", "uni", "dinov2"],
                   help="Feature extractor model.")
    p.add_argument("--k", type=int, default=8, help="kNN for graph construction.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for split.")
    p.add_argument("--start-from", type=str, default="manifest",
                   choices=STEPS, help="Start from this step.")
    p.add_argument("--device", type=str, default="cuda", help="Device for feature extraction.")
    args = p.parse_args(argv)
    
    run_pipeline(
        root=Path(args.root),
        wsi_dir=Path(args.wsi_dir) if args.wsi_dir else None,
        model=str(args.model),
        k=int(args.k),
        seed=int(args.seed),
        start_from=str(args.start_from),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()











