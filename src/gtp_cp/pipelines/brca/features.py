"""
Feature Extraction Module for TCGA-BRCA Pipeline.

This module extracts deep features from patch images using pretrained models.
Supports: ResNet50 (ImageNet), UNI (pathology foundation model), CONCH, etc.

Requirements:
    - torch, torchvision
    - timm (for UNI/CONCH models)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError:
    timm = None

from gtp_cp.pipelines.brca.paths import BRCAPaths
from gtp_cp.utils.io import ensure_dir


@dataclass(frozen=True)
class FeatureConfig:
    model_name: str = "resnet50"  # resnet50, uni, conch, dinov2
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PatchDataset(Dataset):
    """Dataset for loading patch images."""
    
    def __init__(self, patch_paths: list[str], transform=None):
        self.patch_paths = patch_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    def __len__(self) -> int:
        return len(self.patch_paths)
    
    def __getitem__(self, idx: int):
        img = Image.open(self.patch_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, idx


def _load_resnet50_encoder() -> nn.Module:
    """Load ResNet50 pretrained on ImageNet, remove classification head."""
    from torchvision.models import resnet50, ResNet50_Weights
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final FC layer
    model.fc = nn.Identity()
    model.eval()
    return model


def _load_uni_encoder() -> nn.Module:
    """
    Load UNI pathology foundation model.
    UNI: https://github.com/mahmoodlab/UNI
    
    Note: UNI requires downloading weights from HuggingFace.
    """
    if timm is None:
        raise ImportError("timm is required for UNI. Install with: pip install timm")
    
    # UNI is based on ViT-L/16 trained on pathology images
    # You need to download weights from HuggingFace: MahmoodLab/UNI
    try:
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,  # remove head
        )
        # Try to load UNI weights if available
        uni_weights = Path.home() / ".cache" / "uni" / "uni_weights.pth"
        if uni_weights.exists():
            state = torch.load(uni_weights, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print("[info] Loaded UNI weights")
        else:
            print("[warn] UNI weights not found, using random init ViT-L/16")
            print(f"       Download from HuggingFace and save to: {uni_weights}")
    except Exception as e:
        print(f"[warn] Failed to create UNI model: {e}, falling back to ResNet50")
        return _load_resnet50_encoder()
    
    model.eval()
    return model


def _load_dinov2_encoder() -> nn.Module:
    """Load DINOv2 ViT-B/14 model."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model.eval()
    return model


def load_encoder(model_name: str) -> tuple[nn.Module, int]:
    """
    Load feature encoder by name.
    Returns (model, feature_dim).
    """
    model_name = model_name.lower()
    
    if model_name == "resnet50":
        model = _load_resnet50_encoder()
        return model, 2048
    
    elif model_name == "uni":
        model = _load_uni_encoder()
        return model, 1024  # ViT-L output dim
    
    elif model_name == "dinov2":
        model = _load_dinov2_encoder()
        return model, 768  # ViT-B/14 output dim
    
    elif model_name == "conch":
        # CONCH: another pathology foundation model
        # Similar to UNI, requires separate download
        print("[warn] CONCH not implemented, falling back to ResNet50")
        model = _load_resnet50_encoder()
        return model, 2048
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: resnet50, uni, dinov2, conch")


@torch.no_grad()
def extract_features(
    patches_df: pd.DataFrame,
    output_dir: Path,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Extract features for all patches and save as .npy files.
    Returns updated dataframe with feat_path column.
    """
    ensure_dir(output_dir)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    
    # Load model
    print(f"[info] Loading encoder: {config.model_name}")
    model, feat_dim = load_encoder(config.model_name)
    model = model.to(device)
    model.eval()
    
    # Prepare dataset
    patch_paths = patches_df["patch_path"].astype(str).tolist()
    dataset = PatchDataset(patch_paths)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Extract features
    all_features = []
    all_indices = []
    
    for batch, indices in tqdm(loader, desc="Extracting features"):
        batch = batch.to(device)
        features = model(batch)
        
        # Handle different output shapes
        if features.dim() > 2:
            features = features.mean(dim=(2, 3))  # Global average pooling for CNNs
        
        all_features.append(features.cpu().numpy())
        all_indices.extend(indices.tolist())
    
    features_array = np.concatenate(all_features, axis=0)
    print(f"[info] Extracted features shape: {features_array.shape}")
    
    # Save features per patient for efficient loading
    feat_paths = []
    for patient_id, group in patches_df.groupby("patient_id"):
        patient_dir = output_dir / str(patient_id)
        ensure_dir(patient_dir)
        
        for _, row in group.iterrows():
            idx = patches_df.index.get_loc(row.name)
            feat = features_array[idx]
            
            # Save individual feature file
            feat_name = f"{row['patch_idx']:06d}.npy"
            feat_path = patient_dir / feat_name
            np.save(feat_path, feat)
            feat_paths.append(str(feat_path))
    
    patches_df = patches_df.copy()
    patches_df["feat_path"] = feat_paths
    
    return patches_df


def generate_patch_features_csv(
    patches_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate the patch_features.csv required by gtp-brca-graphs.
    
    Required columns: patient_id, slide_id, x, y, feat_path
    """
    required = ["patient_id", "slide_id", "x", "y", "feat_path"]
    
    # Ensure all required columns exist
    for col in required:
        if col not in patches_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    out_df = patches_df[required].copy()
    out_df.to_csv(output_path, index=False)
    print(f"[ok] wrote patch_features.csv: {output_path} (rows={len(out_df)})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract features from TCGA-BRCA patches.")
    p.add_argument("--root", type=str, default="data/brca", help="Pipeline root directory.")
    p.add_argument("--patches-csv", type=str, default=None, help="Patches index CSV from tiling step.")
    p.add_argument("--model", type=str, default="resnet50", 
                   choices=["resnet50", "uni", "dinov2", "conch"],
                   help="Feature extractor model.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction.")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu).")
    args = p.parse_args(argv)
    
    paths = BRCAPaths(Path(args.root))
    
    # Load patches index
    patches_csv = Path(args.patches_csv) if args.patches_csv else paths.root / "patches" / "patches_index.csv"
    if not patches_csv.exists():
        raise FileNotFoundError(
            f"Patches index not found: {patches_csv}\n"
            "Run tiling first: gtp-brca-tile --root data/brca"
        )
    
    patches_df = pd.read_csv(patches_csv)
    print(f"[info] Loaded {len(patches_df)} patches from {patches_csv}")
    
    config = FeatureConfig(
        model_name=str(args.model),
        batch_size=int(args.batch_size),
        num_workers=int(args.workers),
        device=str(args.device),
    )
    
    # Extract features
    output_dir = paths.features_dir
    patches_df = extract_features(patches_df, output_dir, config)
    
    # Generate the final CSV for graph building
    patch_features_csv = paths.features_dir / "patch_features.csv"
    generate_patch_features_csv(patches_df, patch_features_csv)
    
    print("\n[done] Feature extraction complete!")
    print(f"Next step: gtp-brca-graphs --root {args.root} --features-csv {patch_features_csv}")


if __name__ == "__main__":
    main()











