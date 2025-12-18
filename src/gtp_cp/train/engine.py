from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from torch.optim import AdamW
from torch_geometric.data import Batch

from gtp_cp.train.losses import cox_partial_likelihood_loss
from gtp_cp.utils.metrics import concordance_index


@dataclass(frozen=True)
class FitResult:
    best_val_cindex: float
    best_checkpoint: Path


def _to_device(batch, device: torch.device):
    graphs = [g.to(device) for g in batch.graph]
    g_batch = Batch.from_data_list(graphs)
    return g_batch, batch.time.to(device), batch.event.to(device)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    risks = []
    times = []
    events = []
    losses = []
    for b in loader:
        g_batch, t, e = _to_device(b, device)
        log_risk = model(g_batch)
        loss = cox_partial_likelihood_loss(log_risk, t, e)
        losses.append(float(loss.item()))
        risks.append(log_risk.detach().cpu().numpy())
        times.append(t.detach().cpu().numpy())
        events.append(e.detach().cpu().numpy())

    risk = np.concatenate(risks, axis=0)
    time = np.concatenate(times, axis=0)
    event = np.concatenate(events, axis=0)
    cindex = concordance_index(time, risk, event)
    return {"loss": float(np.mean(losses)), "cindex": float(cindex)}


def fit(
    *,
    model,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    log_every: int,
    out_dir: Path,
) -> FitResult:
    console = Console()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for step, b in enumerate(train_loader, start=1):
            g_batch, t, e = _to_device(b, device)
            log_risk = model(g_batch)
            loss = cox_partial_likelihood_loss(log_risk, t, e)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            running.append(float(loss.item()))
            if step % log_every == 0:
                console.log(f"epoch={epoch} step={step} train_loss={np.mean(running):.4f}")

        val = evaluate(model, val_loader, device)
        console.log(
            f"[val] epoch={epoch} loss={val['loss']:.4f} cindex={val['cindex']:.4f}"
        )
        if val["cindex"] > best_val:
            best_val = float(val["cindex"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val_cindex": best_val,
                },
                ckpt_path,
            )

    return FitResult(best_val_cindex=best_val, best_checkpoint=ckpt_path)





