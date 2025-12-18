"""
WSI Tiling Module for TCGA-BRCA Pipeline.

This module extracts non-overlapping patches from whole slide images (SVS files)
and saves them as individual image files for downstream feature extraction.

Requirements:
    - openslide-python (conda install -c conda-forge openslide-python)
    - pillow
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import openslide
except ImportError:
    openslide = None
    print("[warn] openslide not installed. Run: conda install -c conda-forge openslide-python")

from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


@dataclass(frozen=True)
class TilingConfig:
    patch_size: int = 256  # patch size at target magnification
    target_mag: float = 20.0  # target magnification (20x is common for pathology)
    tissue_threshold: float = 0.7  # minimum tissue ratio to keep a patch
    min_tissue_area: int = 1000  # minimum tissue pixels
    max_white_ratio: float = 0.9  # maximum white pixel ratio
    num_workers: int = 4


def _get_best_level(slide: "openslide.OpenSlide", target_mag: float) -> tuple[int, float]:
    """Find the best level for target magnification."""
    try:
        base_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
    except (ValueError, TypeError):
        base_mag = 40.0

    target_downsample = base_mag / target_mag
    level_downsamples = slide.level_downsamples
    
    best_level = 0
    best_diff = float("inf")
    for level, ds in enumerate(level_downsamples):
        diff = abs(ds - target_downsample)
        if diff < best_diff:
            best_diff = diff
            best_level = level
    
    actual_downsample = level_downsamples[best_level]
    return best_level, actual_downsample


def _is_tissue(patch: Image.Image, config: TilingConfig) -> bool:
    """Check if patch contains sufficient tissue (not background/white)."""
    arr = np.array(patch)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    
    # Convert to grayscale for analysis
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    
    # White/background detection (high intensity, low saturation)
    white_mask = gray > 220
    white_ratio = white_mask.sum() / white_mask.size
    
    if white_ratio > config.max_white_ratio:
        return False
    
    # Tissue detection: not too bright, not too dark
    tissue_mask = (gray > 20) & (gray < 220)
    tissue_ratio = tissue_mask.sum() / tissue_mask.size
    
    return tissue_ratio >= config.tissue_threshold


def _tile_single_slide(
    svs_path: Path,
    output_dir: Path,
    config: TilingConfig,
) -> list[dict]:
    """Extract patches from a single WSI."""
    if openslide is None:
        raise ImportError("openslide is required for tiling")
    
    slide_id = svs_path.stem
    slide_out = output_dir / slide_id
    ensure_dir(slide_out)
    
    try:
        slide = openslide.OpenSlide(str(svs_path))
    except Exception as e:
        print(f"[error] Failed to open {svs_path}: {e}")
        return []
    
    level, downsample = _get_best_level(slide, config.target_mag)
    level_dims = slide.level_dimensions[level]
    
    # Calculate step size at level 0
    step_at_level0 = int(config.patch_size * downsample)
    
    patches = []
    patch_idx = 0
    
    # Iterate over grid
    for y in range(0, slide.dimensions[1] - step_at_level0 + 1, step_at_level0):
        for x in range(0, slide.dimensions[0] - step_at_level0 + 1, step_at_level0):
            try:
                region = slide.read_region((x, y), level, (config.patch_size, config.patch_size))
                patch = region.convert("RGB")
            except Exception:
                continue
            
            if not _is_tissue(patch, config):
                continue
            
            # Save patch
            patch_name = f"{patch_idx:06d}.png"
            patch_path = slide_out / patch_name
            patch.save(patch_path)
            
            patches.append({
                "slide_id": slide_id,
                "patch_idx": patch_idx,
                "x": x,
                "y": y,
                "level": level,
                "patch_path": str(patch_path),
            })
            patch_idx += 1
    
    slide.close()
    return patches


def tile_wsi_directory(
    wsi_dir: Path,
    output_dir: Path,
    manifest_df: pd.DataFrame,
    config: TilingConfig,
) -> pd.DataFrame:
    """Tile all WSIs in directory based on manifest."""
    ensure_dir(output_dir)
    
    # Build file_id -> svs_path mapping
    # GDC downloads create subdirectories by file_id
    svs_files = {}
    for p in wsi_dir.rglob("*.svs"):
        svs_files[p.stem] = p
    for p in wsi_dir.rglob("*.SVS"):
        svs_files[p.stem] = p
    
    # Also check if files are directly in the directory
    for p in wsi_dir.glob("*/*.svs"):
        svs_files[p.stem] = p
    
    all_patches = []
    
    # Filter manifest to only SVS files we have
    manifest_df = manifest_df[manifest_df["is_svs"] == True].copy()
    
    found_slides = []
    for _, row in manifest_df.iterrows():
        file_name = str(row.get("file_name", ""))
        slide_stem = Path(file_name).stem
        
        # Try to find the file
        svs_path = None
        if slide_stem in svs_files:
            svs_path = svs_files[slide_stem]
        else:
            # Check by file_id directory
            file_id = str(row.get("file_id", ""))
            possible = wsi_dir / file_id / file_name
            if possible.exists():
                svs_path = possible
        
        if svs_path and svs_path.exists():
            found_slides.append((svs_path, row["patient_id"]))
    
    print(f"[info] Found {len(found_slides)} slides to tile")
    
    for svs_path, patient_id in tqdm(found_slides, desc="Tiling slides"):
        patches = _tile_single_slide(svs_path, output_dir, config)
        for p in patches:
            p["patient_id"] = patient_id
        all_patches.extend(patches)
    
    return pd.DataFrame(all_patches)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tile TCGA-BRCA WSI slides into patches.")
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument("--wsi-dir", type=str, default=None, help="Directory containing downloaded WSI files.")
    p.add_argument("--patch-size", type=int, default=256, help="Patch size in pixels.")
    p.add_argument("--mag", type=float, default=20.0, help="Target magnification.")
    p.add_argument("--tissue-thresh", type=float, default=0.7, help="Minimum tissue ratio.")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = p.parse_args(argv)
    
    paths = BRCAPaths(Path(args.root))
    
    wsi_dir = Path(args.wsi_dir) if args.wsi_dir else paths.root / "raw" / "gdc_wsi"
    if not wsi_dir.exists():
        raise FileNotFoundError(
            f"WSI directory not found: {wsi_dir}\n"
            "Download WSIs first using: .\\scripts\\brca\\download_gdc_wsi.ps1"
        )
    
    manifest = pd.read_csv(paths.manifest_wsi)
    
    config = TilingConfig(
        patch_size=int(args.patch_size),
        target_mag=float(args.mag),
        tissue_threshold=float(args.tissue_thresh),
        num_workers=int(args.workers),
    )
    
    output_dir = paths.root / "patches"
    patches_df = tile_wsi_directory(wsi_dir, output_dir, manifest, config)
    
    # Save patches index
    patches_csv = paths.root / "patches" / "patches_index.csv"
    patches_df.to_csv(patches_csv, index=False)
    print(f"[ok] wrote patches index: {patches_csv} (patches={len(patches_df)})")


if __name__ == "__main__":
    main()


