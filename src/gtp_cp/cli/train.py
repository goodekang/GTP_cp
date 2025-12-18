from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from gtp_cp.data.dataloaders import make_loader
from gtp_cp.data.synthetic import SyntheticSpec, generate_dataset
from gtp_cp.models.causal_stfm import CausalSTFM, ModelSpec
from gtp_cp.train.engine import fit
from gtp_cp.utils.io import ensure_dir
from gtp_cp.utils.seed import seed_everything


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))
    device = _device(str(cfg.device))

    spec = SyntheticSpec(
        n_patch=int(cfg.graph.n_patch),
        n_cell_per_patch=int(cfg.graph.n_cell_per_patch),
        n_gene_per_cell=int(cfg.graph.n_gene_per_cell),
        feature_dim=int(cfg.graph.feature_dim),
        patch_radius=int(cfg.graph.patch_radius),
    )

    train_items = generate_dataset(int(cfg.seed), int(cfg.data.n_train), spec)
    val_items = generate_dataset(int(cfg.seed) + 1, int(cfg.data.n_val), spec)
    test_items = generate_dataset(int(cfg.seed) + 2, int(cfg.data.n_test), spec)

    train_loader = make_loader(
        train_items,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
    )
    val_loader = make_loader(
        val_items,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
    )
    test_loader = make_loader(
        test_items,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
    )

    model = CausalSTFM(
        ModelSpec(
            feature_dim=int(cfg.graph.feature_dim),
            hidden_dim=int(cfg.model.hidden_dim),
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            dropout=float(cfg.model.dropout),
            use_causal_mask=bool(cfg.model.use_causal_mask),
            causal_mask_strength=float(cfg.model.causal_mask_strength),
        )
    )

    run_dir = ensure_dir(Path(str(cfg.output.dir)) / hydra.core.hydra_config.HydraConfig.get().run.dir)
    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(cfg.train.epochs),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        grad_clip=float(cfg.train.grad_clip),
        log_every=int(cfg.train.log_every),
        out_dir=run_dir,
    )

    # quick test
    from gtp_cp.train.engine import evaluate

    metrics = evaluate(model.to(device), test_loader, device)
    print(
        f"best_val_cindex={result.best_val_cindex:.4f} test_cindex={metrics['cindex']:.4f} "
        f"checkpoint={result.best_checkpoint}"
    )


if __name__ == "__main__":
    main()





