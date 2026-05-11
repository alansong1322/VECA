from __future__ import annotations

import math

import torch


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    min_lr_ratio: float = 0.0,
):
    min_lr_ratio = float(min_lr_ratio)
    assert 0.0 <= min_lr_ratio <= 1.0

    def lr_lambda(step: int):
        if step < num_warmup_steps:
            warm = step / max(1, num_warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * warm
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cos = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def autocast_ctx(cfg):
    mp_ = str(cfg.mixed_precision).lower()
    if mp_ == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mp_ == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    assert mp_ == "fp32"
    return torch.autocast(device_type="cuda", enabled=False)


def configure_backend_precision() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "tf32"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    if hasattr(torch.backends.cudnn, "fp32_precision"):
        torch.backends.cudnn.fp32_precision = "tf32"
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    if hasattr(torch.backends.cudnn, "rnn") and hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
        torch.backends.cudnn.rnn.fp32_precision = "tf32"
