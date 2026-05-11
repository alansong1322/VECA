from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cls_cosine_loss(student_cls: torch.Tensor, teacher_cls: torch.Tensor, norm_eps: float = 1e-6) -> torch.Tensor:
    s = F.normalize(student_cls.float(), dim=-1, eps=norm_eps)
    t = F.normalize(teacher_cls.float(), dim=-1, eps=norm_eps)
    cos = (s * t).sum(dim=-1)
    return (1.0 - cos).mean()


class DenseCosineMSELoss(nn.Module):
    def __init__(self, norm_eps: float = 1e-6, mse_weight: float = 1.0):
        super().__init__()
        self.norm_eps = float(norm_eps)
        self.mse_weight = float(mse_weight)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = student.float()
        t = teacher.float()
        cos_dist = 1.0 - F.cosine_similarity(s, t, dim=-1, eps=self.norm_eps)
        loss_cos = cos_dist.mean()
        loss_mse = F.mse_loss(s, t)
        loss = loss_cos + self.mse_weight * loss_mse
        return loss, loss_cos, loss_mse
