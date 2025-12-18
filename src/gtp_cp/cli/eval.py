from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from gtp_cp.data.dataloaders import make_loader
from gtp_cp.data.synthetic import SyntheticSpec, generate_dataset
from gtp_cp.models.causal_stfm import CausalSTFM, ModelSpec
from gtp_cp.train.engine import evaluate
from gtp_cp.utils.io import latest_checkpoint
from gtp_cp.utils.seed import seed_everything


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@hydra.main(version_base=None, config_path="../../../configs", config_name="eval")
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

    # regenerate the same synthetic splits as train.py
    val_items = generate_dataset(int(cfg.seed) + 1, int(cfg.data.n_val), spec)
    test_items = generate_dataset(int(cfg.seed) + 2, int(cfg.data.n_test), spec)
    items = val_items if str(cfg.split) == "val" else test_items

    loader = make_loader(
        items,
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
    ).to(device)

    ckpt = cfg.checkpoint_path
    if ckpt is None:
        found = latest_checkpoint(Path(str(cfg.output.dir)))
        if found is None:
            raise FileNotFoundError(
                f"No checkpoint found under output.dir={cfg.output.dir}. Run `gtp-train` first."
            )
        ckpt_path = found
    else:
        ckpt_path = Path(str(ckpt))

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])

    metrics = evaluate(model, loader, device)
    print(f"split={cfg.split} loss={metrics['loss']:.4f} cindex={metrics['cindex']:.4f} ckpt={ckpt_path}")


if __name__ == "__main__":
    main()






