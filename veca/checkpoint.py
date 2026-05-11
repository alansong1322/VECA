from __future__ import annotations

import torch
import torch.nn as nn

from veca.config import cfg_to_dict


def save_ckpt(
    path: str,
    *,
    student_ddp,
    optimizer_adam,
    optimizer_muon,
    scheduler_adam,
    scheduler_muon,
    cfg,
    extra: dict,
) -> None:
    payload = {
        "student": student_ddp.module.state_dict(),
        "optimizer": optimizer_adam.state_dict(),
        "optimizer_muon": optimizer_muon.state_dict(),
        "scheduler": scheduler_adam.state_dict(),
        "scheduler_muon": scheduler_muon.state_dict(),
        "config": cfg_to_dict(cfg),
        **extra,
    }
    torch.save(payload, path)
    print(f"[CKPT] Saved -> {path}")


@torch.no_grad()
def load_state_dict_forgiving(module: nn.Module, state_dict: dict):
    current = module.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in current and current[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped
