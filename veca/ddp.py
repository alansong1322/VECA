from __future__ import annotations

import os
from typing import List, Tuple

import torch
import torch.distributed as dist


def ddp_setup_from_torchrun() -> Tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    return local_rank, rank, world_size


def ddp_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


@torch.no_grad()
def ddp_allreduce_mean_multi(vals: List[torch.Tensor], world_size: int) -> List[torch.Tensor]:
    if world_size <= 1:
        return vals
    x = torch.stack([v.detach() for v in vals], dim=0)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= float(world_size)
    return [x[i] for i in range(x.shape[0])]


@torch.no_grad()
def sample_active_k_ddp(cfg, device: torch.device, rank: int) -> int:
    if rank == 0:
        weights = torch.tensor(cfg.nested_budget_weights, device=device, dtype=torch.float32)
        idx = torch.multinomial(weights, num_samples=1).item()
        k = int(cfg.nested_budgets[idx])
        x = torch.tensor([k], device=device, dtype=torch.int64)
    else:
        x = torch.zeros(1, device=device, dtype=torch.int64)
    dist.broadcast(x, src=0)
    k = int(x.item())
    assert k in cfg.nested_budgets
    return k
