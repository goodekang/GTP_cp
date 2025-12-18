import torch
from torch_geometric.data import Batch

from gtp_cp.data.synthetic import SyntheticSpec, generate_dataset
from gtp_cp.models.causal_stfm import CausalSTFM, ModelSpec


def test_forward_smoke():
    spec = SyntheticSpec(
        n_patch=4,
        n_cell_per_patch=2,
        n_gene_per_cell=4,
        feature_dim=16,
        patch_radius=1,
    )
    items = generate_dataset(0, 3, spec)
    graphs = [g for (g, _, _) in items]
    batch = Batch.from_data_list(graphs)

    model = CausalSTFM(
        ModelSpec(
            feature_dim=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            use_causal_mask=True,
            causal_mask_strength=1.0,
        )
    )
    out = model(batch)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()





