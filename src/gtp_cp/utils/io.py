from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_checkpoint(root: str | Path) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    ckpts = list(root.rglob("*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]













