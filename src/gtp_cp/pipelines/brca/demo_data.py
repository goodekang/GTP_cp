"""
Demo Data Generator for TCGA-BRCA Pipeline.

This module generates synthetic patches and features for testing the pipeline
without requiring actual WSI downloads (which can be ~500GB+).

Usage:
    gtp-brca-demo --root data/brca --n-patients 50 --n-patches 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


def generate_synthetic_patch(rng: np.random.Generator, size: int = 256) -> Image.Image:
    """Generate a synthetic tissue-like patch image."""
    # Create a base pink/purple tissue-like background
    base_color = rng.integers(180, 230, size=3)
    img = np.ones((size, size, 3), dtype=np.uint8) * base_color
    
    # Add some random "nuclei" (darker spots)
    n_nuclei = rng.integers(20, 100)
    for _ in range(n_nuclei):
        cx, cy = rng.integers(10, size-10, size=2)
        r = rng.integers(3, 8)
        y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
        mask = x*x + y*y <= r*r
        nuclei_color = rng.integers(80, 150, size=3)
        img[mask] = nuclei_color
    
    # Add some "cytoplasm" regions (lighter areas)
    n_cyto = rng.integers(5, 20)
    for _ in range(n_cyto):
        cx, cy = rng.integers(20, size-20, size=2)
        r = rng.integers(15, 40)
        y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
        mask = x*x + y*y <= r*r
        cyto_color = rng.integers(200, 250, size=3)
        img[mask] = img[mask] * 0.7 + cyto_color * 0.3
    
    return Image.fromarray(img.astype(np.uint8))


def generate_demo_data(
    root: Path,
    n_patients: int,
    n_patches_per_patient: int,
    feature_dim: int,
    seed: int,
) -> None:
    """Generate complete demo dataset for testing the pipeline."""
    paths = BRCAPaths(root)
    rng = np.random.default_rng(seed)
    
    # Load split to get patient IDs
    split_csv = paths.split_csv
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_csv}\n"
            "Run first: gtp-brca-split --root {root}"
        )
    
    split_df = pd.read_csv(split_csv)
    patient_ids = split_df["patient_id"].astype(str).tolist()
    
    # Sample patients if we have more than requested
    if len(patient_ids) > n_patients:
        patient_ids = rng.choice(patient_ids, size=n_patients, replace=False).tolist()
    
    print(f"[info] Generating demo data for {len(patient_ids)} patients")
    
    # Create directories
    patches_dir = root / "patches"
    features_dir = paths.features_dir
    ensure_dir(patches_dir)
    ensure_dir(features_dir)
    
    all_patches = []
    
    for patient_id in tqdm(patient_ids, desc="Generating patients"):
        patient_patches_dir = patches_dir / patient_id
        patient_features_dir = features_dir / patient_id
        ensure_dir(patient_patches_dir)
        ensure_dir(patient_features_dir)
        
        # Generate synthetic slide_id
        slide_id = f"{patient_id}-01Z-00-DX1"
        
        # Generate patches for this patient
        for patch_idx in range(n_patches_per_patient):
            # Random coordinates (simulating WSI coordinates)
            x = int(rng.integers(0, 50000))
            y = int(rng.integers(0, 50000))
            
            # Generate and save patch image
            patch_img = generate_synthetic_patch(rng)
            patch_path = patient_patches_dir / f"{patch_idx:06d}.png"
            patch_img.save(patch_path)
            
            # Generate and save feature
            feat = rng.standard_normal(feature_dim).astype(np.float32)
            feat_path = patient_features_dir / f"{patch_idx:06d}.npy"
            np.save(feat_path, feat)
            
            all_patches.append({
                "patient_id": patient_id,
                "slide_id": slide_id,
                "patch_idx": patch_idx,
                "x": x,
                "y": y,
                "patch_path": str(patch_path),
                "feat_path": str(feat_path),
            })
    
    # Save patches index
    patches_df = pd.DataFrame(all_patches)
    patches_csv = patches_dir / "patches_index.csv"
    patches_df.to_csv(patches_csv, index=False)
    print(f"[ok] wrote patches index: {patches_csv} (rows={len(patches_df)})")
    
    # Save patch_features.csv (the file needed by gtp-brca-graphs)
    features_csv = features_dir / "patch_features.csv"
    features_df = patches_df[["patient_id", "slide_id", "x", "y", "feat_path"]].copy()
    features_df.to_csv(features_csv, index=False)
    print(f"[ok] wrote patch_features.csv: {features_csv} (rows={len(features_df)})")
    
    print(f"\n[done] Demo data generation complete!")
    print(f"[next] Build graphs: gtp-brca-graphs --root {root} --features-csv {features_csv}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Generate demo data for TCGA-BRCA pipeline testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This creates synthetic patches and features for testing without downloading
actual WSI files (which can be hundreds of GB).

Example workflow:
    1. gtp-brca-manifest --root data/brca
    2. gtp-brca-clinical --root data/brca
    3. gtp-brca-split --root data/brca --seed 7
    4. gtp-brca-demo --root data/brca --n-patients 50  # <-- this script
    5. gtp-brca-graphs --root data/brca --features-csv data/brca/features/patch_features.csv
    6. gtp-brca-train --graphs-index data/brca/graphs/graphs_index.csv
"""
    )
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument("--n-patients", type=int, default=50, help="Number of patients to generate.")
    p.add_argument("--n-patches", type=int, default=100, help="Patches per patient.")
    p.add_argument("--feature-dim", type=int, default=2048, help="Feature dimension (e.g., 2048 for ResNet50).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = p.parse_args(argv)
    
    generate_demo_data(
        root=Path(args.root),
        n_patients=int(args.n_patients),
        n_patches_per_patient=int(args.n_patches),
        feature_dim=int(args.feature_dim),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()











