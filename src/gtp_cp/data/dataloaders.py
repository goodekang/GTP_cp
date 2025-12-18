from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


@dataclass(frozen=True)
class SurvivalBatch:
    graph: object  # torch_geometric.data.Batch
    time: torch.Tensor
    event: torch.Tensor


class SurvivalGraphDataset(Dataset):
    def __init__(self, items: list[tuple[object, float, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[object, torch.Tensor, torch.Tensor]:
        g, t, e = self.items[idx]
        return g, torch.tensor(t, dtype=torch.float32), torch.tensor(e, dtype=torch.float32)


def _collate(batch: list[tuple[object, torch.Tensor, torch.Tensor]]) -> SurvivalBatch:
    graphs, times, events = zip(*batch)
    # PyG DataLoader handles batching graphs; we just need to stack labels.
    return SurvivalBatch(graph=graphs, time=torch.stack(times), event=torch.stack(events))


def make_loader(items: list[tuple[object, float, int]], batch_size: int, shuffle: bool, num_workers: int):
    ds = SurvivalGraphDataset(items)
    # Use PyG DataLoader for graphs, but keep our labels aligned.
    # We'll pass a collate that returns a list of Data and stacked tensors; we then batch graphs inside train loop.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=_collate)






