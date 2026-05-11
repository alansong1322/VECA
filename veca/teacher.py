from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from veca.data import DEFAULT_MEAN, DEFAULT_STD
from veca.losses import DenseCosineMSELoss, cls_cosine_loss
from veca.model import VECA


def teacher_amp_dtype(cfg) -> Optional[torch.dtype]:
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
    return mapping[str(cfg.teacher_dtype).lower()]


@torch.no_grad()
def teacher_forward(
    teacher: nn.Module,
    images: torch.Tensor,
    *,
    n_reg: int,
    microbatch: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = images.shape[0]
    mb = microbatch or B
    use_amp = images.is_cuda and amp_dtype is not None
    cls_list, patch_list = [], []
    for s in range(0, B, mb):
        x = images[s:s + mb]
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = teacher(pixel_values=x, return_dict=True)
        else:
            out = teacher(pixel_values=x, return_dict=True)
        tokens = out.last_hidden_state
        cls_list.append(tokens[:, 0, :].contiguous())
        patch_list.append(tokens[:, 1 + n_reg:, :].contiguous())
    return torch.cat(cls_list, dim=0), torch.cat(patch_list, dim=0)


class DistillVECA(nn.Module):
    def __init__(self, cfg, device: torch.device, *, verbose: bool = True):
        super().__init__()
        self.cfg = cfg
        self.device_ = device
        self.teacher = AutoModel.from_pretrained(cfg.teacher_id).to(self.device_)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        t_cfg = self.teacher.config
        self.teacher_dim = int(getattr(t_cfg, "hidden_size"))
        n_reg = cfg.teacher_num_register_tokens
        if n_reg is None:
            n_reg = getattr(t_cfg, "num_register_tokens", 0) or 0
        self.n_reg = int(n_reg)
        cfg.distill_output_dim = self.teacher_dim
        self.student = VECA(cfg).to(self.device_)
        self.dense_loss = DenseCosineMSELoss(norm_eps=cfg.dense_norm_eps, mse_weight=cfg.dense_mse_weight)
        self.teacher_amp_dtype = teacher_amp_dtype(cfg)
        if cfg.teacher_compile:
            self.teacher = torch.compile(self.teacher, mode=cfg.teacher_compile_mode)
        if verbose:
            print(f"[Teacher] id={cfg.teacher_id} | dim={self.teacher_dim} | n_reg={self.n_reg} | device={self.device_}")
            print(
                f"[Student] family={cfg.model_family} | hidden_dim={cfg.hidden_dim} | output_dim={self.student.output_dim} "
                f"| device={self.device_} | query_chunks={self.student.num_query_chunks} | chunk={self.student.chunk_size}"
            )
            print(f"[Norm] mean={DEFAULT_MEAN} std={DEFAULT_STD}")
            print(f"[Loss] cls=cosine | dense=cosine+{cfg.dense_mse_weight}*mse | dense_weight={cfg.weight_dense}")
            print(f"[Nested] budgets={cfg.nested_budgets} | weights={cfg.nested_budget_weights}")

    def distill_step(
        self,
        images: torch.Tensor,
        *,
        active_k: int,
        teacher_microbatch_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        assert active_k in self.cfg.nested_budgets
        B = images.shape[0]
        microbatch = teacher_microbatch_override
        if microbatch is None:
            microbatch = getattr(self.cfg, "teacher_microbatch", B)
        t_cls, t_patches = teacher_forward(
            self.teacher,
            images,
            n_reg=self.n_reg,
            microbatch=microbatch,
            amp_dtype=self.teacher_amp_dtype,
        )
        s_cls, s_patches = self.student(images, active_k=active_k)
        assert s_patches.shape[1] == t_patches.shape[1]
        loss_cls = cls_cosine_loss(s_cls, t_cls, norm_eps=self.cfg.dense_norm_eps)
        loss_dense, loss_dense_cos, loss_dense_mse = self.dense_loss(s_patches, t_patches)
        loss = loss_cls + self.cfg.weight_dense * loss_dense
        logs = {
            "active_k": torch.tensor(float(active_k), device=images.device),
            "loss_cls": loss_cls.detach(),
            "loss_dense": loss_dense.detach(),
            "loss_dense_cos": loss_dense_cos.detach(),
            "loss_dense_mse": loss_dense_mse.detach(),
        }
        return loss, logs
