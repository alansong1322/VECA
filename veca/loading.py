from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from veca.config import MultiResFinetuneConfig, normalize_nested_settings
from veca.model import VECA


def _torch_load_cpu(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("student."):
            key = key[len("student."):]
        out[key] = value
    return out


def _config_from_checkpoint(ckpt: Any, **overrides) -> MultiResFinetuneConfig:
    cfg = MultiResFinetuneConfig()
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    for key, value in ckpt_cfg.items():
        if key == "matryoshka_chunk_size":
            cfg.nested_chunk_size = int(value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown VECA config override: {key}")
        setattr(cfg, key, value)

    normalize_nested_settings(cfg)
    cfg.init_patch_from_dinov3 = False
    return cfg


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    eval: bool = True,
    strict: bool = True,
    return_config: bool = False,
    **config_overrides,
):
    """Load a VECA checkpoint in one call."""
    ckpt = _torch_load_cpu(checkpoint_path)
    state = ckpt["student"] if isinstance(ckpt, dict) and "student" in ckpt else ckpt
    cfg = _config_from_checkpoint(ckpt, **config_overrides)

    model = VECA(cfg)
    model.load_state_dict(_clean_state_dict(state), strict=strict)
    model = model.to(device)

    if eval:
        model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    if return_config:
        return model, cfg
    return model
