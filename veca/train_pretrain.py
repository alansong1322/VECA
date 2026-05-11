from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from veca.checkpoint import save_ckpt
from veca.config import PretrainConfig, cfg_to_dict, normalize_nested_settings
from veca.data import create_object365_dataloaders_ddp, ensure_index_file, require_data_paths
from veca.ddp import ddp_allreduce_mean_multi, ddp_cleanup, ddp_setup_from_torchrun, is_main_process, sample_active_k_ddp
from veca.optim import build_optimizers, count_params
from veca.schedules import autocast_ctx, configure_backend_precision, get_cosine_schedule_with_warmup
from veca.teacher import DistillVECA

MODEL_ARCH_VERSION = "veca_obj365_unfused_qkv_nested_chunkparams_v3"
OPTIMIZER_LAYOUT_VERSION = "muon_clean_split_v1"
TRAINING_LAYOUT_VERSION = "obj365_full_batch_nested_v1"


def train_one_epoch_ddp(
    model: DistillVECA,
    train_loader,
    train_sampler,
    optimizer_adam,
    optimizer_muon,
    scheduler_adam,
    scheduler_muon,
    epoch: int,
    cfg: PretrainConfig,
    *,
    rank: int,
    world_size: int,
    enable_wandb: bool,
):
    model.student.train()
    model.teacher.eval()
    train_sampler.set_epoch(epoch)
    steps = len(train_loader)
    assert steps > 0
    meters = {
        "loss": 0.0,
        "cls": 0.0,
        "dense": 0.0,
        "dense_cos": 0.0,
        "dense_mse": 0.0,
        "active_k": 0.0,
    }
    budget_hist = {int(k): 0 for k in cfg.nested_budgets}
    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"E{epoch + 1:03d}", ncols=150, mininterval=5.0, dynamic_ncols=True)
        data_iter = enumerate(pbar)
    else:
        pbar = None
        data_iter = enumerate(train_loader)
    for step, imgs_cpu in data_iter:
        assert imgs_cpu.shape[0] == cfg.batch_size
        imgs = imgs_cpu.to(model.device_, non_blocking=True)
        active_k = sample_active_k_ddp(cfg, model.device_, rank=rank)
        budget_hist[active_k] += 1
        optimizer_adam.zero_grad(set_to_none=True)
        optimizer_muon.zero_grad(set_to_none=True)
        with autocast_ctx(cfg):
            loss, logs = model.distill_step(imgs, active_k=active_k)
        loss.backward()
        optimizer_adam.step()
        optimizer_muon.step()
        scheduler_adam.step()
        scheduler_muon.step()
        vals = ddp_allreduce_mean_multi(
            [
                loss.detach(),
                logs["loss_cls"],
                logs["loss_dense"],
                logs["loss_dense_cos"],
                logs["loss_dense_mse"],
                logs["active_k"],
            ],
            world_size,
        )
        loss_m, cls_m, dense_m, dcos_m, dmse_m, k_m = [float(v.item()) for v in vals]
        meters["loss"] += loss_m
        meters["cls"] += cls_m
        meters["dense"] += dense_m
        meters["dense_cos"] += dcos_m
        meters["dense_mse"] += dmse_m
        meters["active_k"] += k_m
        if is_main_process(rank):
            lr_now = float(scheduler_adam.get_last_lr()[0])
            if pbar is not None and ((step % 50 == 0) or (step == steps - 1)):
                n = step + 1
                pbar.set_postfix({
                    "K": f"{active_k}",
                    "Kavg": f"{meters['active_k'] / n:.1f}",
                    "loss": f"{meters['loss'] / n:.4f}",
                    "cls": f"{meters['cls'] / n:.4f}",
                    "dense": f"{meters['dense'] / n:.4f}",
                    "lr": f"{lr_now:.1e}",
                    "bs": f"{cfg.batch_size}",
                })
            if enable_wandb and ((step % int(cfg.wandb_log_every) == 0) or (step == steps - 1)):
                global_step = epoch * steps + step + 1
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/lr": lr_now,
                        "train/active_k": active_k,
                        "train/loss": loss_m,
                        "train/loss_cls": cls_m,
                        "train/loss_dense": dense_m,
                        "train/loss_dense_cos": dcos_m,
                        "train/loss_dense_mse": dmse_m,
                    },
                    step=global_step,
                )
    if is_main_process(rank) and pbar is not None:
        pbar.close()
    train_avg = {k: v / steps for k, v in meters.items()}
    if is_main_process(rank):
        hist_msg = " ".join([f"K{k}:{budget_hist[k]}" for k in cfg.nested_budgets])
        print(f"[BUDGET][E{epoch + 1:03d}] {hist_msg}")
        if enable_wandb:
            payload = {
                "epoch": epoch + 1,
                "train_epoch/loss": float(train_avg["loss"]),
                "train_epoch/loss_cls": float(train_avg["cls"]),
                "train_epoch/loss_dense": float(train_avg["dense"]),
                "train_epoch/loss_dense_cos": float(train_avg["dense_cos"]),
                "train_epoch/loss_dense_mse": float(train_avg["dense_mse"]),
                "train_epoch/active_k_avg": float(train_avg["active_k"]),
            }
            for k in cfg.nested_budgets:
                payload[f"budget_count/k_{k}"] = int(budget_hist[k])
                payload[f"budget_frac/k_{k}"] = float(budget_hist[k]) / float(steps)
            wandb.log(payload, step=(epoch + 1) * steps)
    return train_avg, budget_hist


@torch.no_grad()
def validate_ddp(
    model: DistillVECA,
    val_loader,
    val_sampler,
    epoch: int,
    cfg: PretrainConfig,
    *,
    rank: int,
    world_size: int,
    enable_wandb: bool,
    active_k: int,
):
    assert active_k in cfg.nested_budgets
    model.student.eval()
    model.teacher.eval()
    val_sampler.set_epoch(epoch)
    steps = len(val_loader)
    assert steps > 0
    meters = {
        "loss": 0.0,
        "cls": 0.0,
        "dense": 0.0,
        "dense_cos": 0.0,
        "dense_mse": 0.0,
    }
    for imgs_cpu in val_loader:
        imgs = imgs_cpu.to(model.device_, non_blocking=True)
        with autocast_ctx(cfg):
            loss, logs = model.distill_step(imgs, active_k=active_k)
        vals = ddp_allreduce_mean_multi(
            [
                loss.detach(),
                logs["loss_cls"],
                logs["loss_dense"],
                logs["loss_dense_cos"],
                logs["loss_dense_mse"],
            ],
            world_size,
        )
        loss_m, cls_m, dense_m, dcos_m, dmse_m = [float(v.item()) for v in vals]
        meters["loss"] += loss_m
        meters["cls"] += cls_m
        meters["dense"] += dense_m
        meters["dense_cos"] += dcos_m
        meters["dense_mse"] += dmse_m
    meters = {k: v / steps for k, v in meters.items()}
    if is_main_process(rank):
        print(
            f"[VAL][E{epoch + 1:03d}] K={active_k} "
            f"loss={meters['loss']:.4f} cls={meters['cls']:.4f} "
            f"dense={meters['dense']:.4f} dcos={meters['dense_cos']:.4f} dmse={meters['dense_mse']:.4f}"
        )
        if enable_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    f"val_k{active_k}/loss": float(meters["loss"]),
                    f"val_k{active_k}/loss_cls": float(meters["cls"]),
                    f"val_k{active_k}/loss_dense": float(meters["dense"]),
                    f"val_k{active_k}/loss_dense_cos": float(meters["dense_cos"]),
                    f"val_k{active_k}/loss_dense_mse": float(meters["dense_mse"]),
                },
                step=(epoch + 1) * cfg.steps_per_epoch,
            )
    return meters


@record
def main(cfg: PretrainConfig | None = None) -> None:
    cfg = cfg or PretrainConfig()
    normalize_nested_settings(cfg)
    assert torch.cuda.is_available()
    assert hasattr(torch, "compile")
    assert hasattr(F, "scaled_dot_product_attention")
    assert cfg.hidden_dim % cfg.num_heads == 0
    assert (cfg.hidden_dim // cfg.num_heads) % 4 == 0
    configure_backend_precision()
    local_rank, rank, world_size = ddp_setup_from_torchrun()
    mainp = is_main_process(rank)
    torch.manual_seed(cfg.base_seed + rank)
    random.seed(cfg.base_seed + rank)
    np.random.seed(cfg.base_seed + rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cudnn.benchmark = True
    if mainp:
        print(f"[DDP] rank={rank}/{world_size} local_rank={local_rank} device={device} mp={cfg.mixed_precision}")
        print(f"[Nested] chunk={cfg.nested_chunk_size} budgets={cfg.nested_budgets} val_active_k={cfg.val_active_k}")
        require_data_paths(cfg)
        ensure_index_file(cfg.train_dir, cfg.train_index)
        ensure_index_file(cfg.val_dir, cfg.val_index)
    torch.distributed.barrier()
    train_loader, val_loader, train_sampler, val_sampler = create_object365_dataloaders_ddp(cfg, rank=rank, world_size=world_size)
    cfg.steps_per_epoch = len(train_loader)
    if mainp:
        print(f"[Epoch] steps_per_epoch={cfg.steps_per_epoch} val_steps={len(val_loader)}")
        print(f"[Batch] per-rank Object365 batch={cfg.batch_size}")
    enable_wandb = mainp
    if not enable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    run = None
    if enable_wandb:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config=cfg_to_dict(cfg),
        )
    model = DistillVECA(cfg, device=device, verbose=mainp)
    model.student = DDP(model.student, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
    optimizer, optimizer_muon, muon_names, adamw_names = build_optimizers(model.student, cfg)
    if mainp:
        student_module = model.student.module
        total = count_params([p for p in student_module.parameters() if p.requires_grad])
        adam_n = count_params(optimizer.param_groups[0]["params"])
        muon_n = count_params(optimizer_muon.param_groups[0]["params"])
        print(f"[Params] total={total:,} AdamW={adam_n:,} ({adam_n / total:.2%}) NorMuon={muon_n:,} ({muon_n / total:.2%})")
        print("[Muon] Parameter groups:")
        for n in sorted(muon_names):
            print(f"  {n}")
        print("[AdamW] Parameter groups:")
        for n in sorted(adamw_names):
            print(f"  {n}")
        if enable_wandb:
            wandb.log({"params/total": total, "params/adamw": adam_n, "params/normuon": muon_n}, step=0)
    total_steps = cfg.steps_per_epoch * cfg.epochs
    warmup_steps = cfg.steps_per_epoch * cfg.warmup_epochs
    min_ratio = float(cfg.lr_min) / float(cfg.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=min_ratio)
    scheduler_muon = get_cosine_schedule_with_warmup(optimizer_muon, warmup_steps, total_steps, min_lr_ratio=min_ratio)
    start_epoch = 0
    best_val = float("inf")
    if os.path.exists(cfg.ckpt_name):
        map_loc = {"cuda:0": f"cuda:{local_rank}"}
        ckpt = torch.load(cfg.ckpt_name, map_location=map_loc)
        assert ckpt.get("model_arch_version") == MODEL_ARCH_VERSION
        assert ckpt.get("optimizer_layout_version") == OPTIMIZER_LAYOUT_VERSION
        assert ckpt.get("training_layout_version") == TRAINING_LAYOUT_VERSION
        model.student.module.load_state_dict(ckpt["student"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_muon.load_state_dict(ckpt["optimizer_muon"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scheduler_muon.load_state_dict(ckpt["scheduler_muon"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("best_val", float("inf"))
        if mainp:
            print(f"[CKPT] Resumed: {cfg.ckpt_name}")
            print(f"[CKPT] start_epoch={start_epoch} best_val={best_val:.6f}")
    torch.distributed.barrier()
    for epoch in range(start_epoch, cfg.epochs):
        train_meters, budget_hist = train_one_epoch_ddp(
            model=model,
            train_loader=train_loader,
            train_sampler=train_sampler,
            optimizer_adam=optimizer,
            optimizer_muon=optimizer_muon,
            scheduler_adam=scheduler,
            scheduler_muon=scheduler_muon,
            epoch=epoch,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            enable_wandb=enable_wandb,
        )
        if mainp:
            print(
                f"[TRAIN][E{epoch + 1:03d}] Kavg={train_meters['active_k']:.2f} "
                f"loss={train_meters['loss']:.4f} cls={train_meters['cls']:.4f} "
                f"dense={train_meters['dense']:.4f} dcos={train_meters['dense_cos']:.4f} dmse={train_meters['dense_mse']:.4f}"
            )
            alphas = model.student.module.pos_alpha.detach().float().cpu().tolist()
            print(f"[ALPHA][E{epoch + 1:03d}] {' '.join(f'{a:+.4f}' for a in alphas)}")
            if enable_wandb:
                wandb.log({f"alpha/layer_{i}": float(a) for i, a in enumerate(alphas)}, step=(epoch + 1) * cfg.steps_per_epoch)
        do_val = ((epoch + 1) % cfg.validate_every_epochs == 0) or (epoch == 0)
        do_save = ((epoch + 1) % cfg.save_every_epochs == 0) or (epoch == 0)
        extra = {
            "epoch": epoch,
            "best_val": float(best_val),
            "best_val_budget": int(cfg.val_active_k),
            "last_epoch_budget_hist": {int(k): int(v) for k, v in budget_hist.items()},
            "model_arch_version": MODEL_ARCH_VERSION,
            "optimizer_layout_version": OPTIMIZER_LAYOUT_VERSION,
            "training_layout_version": TRAINING_LAYOUT_VERSION,
        }
        if do_val:
            val_meters = validate_ddp(
                model=model,
                val_loader=val_loader,
                val_sampler=val_sampler,
                epoch=epoch,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                enable_wandb=enable_wandb,
                active_k=cfg.val_active_k,
            )
            if mainp:
                extra.update({f"val_k{cfg.val_active_k}_{k}": v for k, v in val_meters.items()})
                if val_meters["loss"] < best_val:
                    best_val = float(val_meters["loss"])
                    extra["best_val"] = best_val
                    extra["best_epoch"] = epoch
                    extra["pos_alpha"] = model.student.module.pos_alpha.detach().float().cpu().tolist()
                    if enable_wandb:
                        wandb.log({"best/val_loss": float(best_val)}, step=(epoch + 1) * cfg.steps_per_epoch)
                    save_ckpt(
                        cfg.best_ckpt_name,
                        student_ddp=model.student,
                        optimizer_adam=optimizer,
                        optimizer_muon=optimizer_muon,
                        scheduler_adam=scheduler,
                        scheduler_muon=scheduler_muon,
                        cfg=cfg,
                        extra=extra,
                    )
        if mainp:
            extra["best_val"] = float(best_val)
        if do_save and mainp:
            extra["pos_alpha"] = model.student.module.pos_alpha.detach().float().cpu().tolist()
            save_ckpt(
                cfg.ckpt_name,
                student_ddp=model.student,
                optimizer_adam=optimizer,
                optimizer_muon=optimizer_muon,
                scheduler_adam=scheduler,
                scheduler_muon=scheduler_muon,
                cfg=cfg,
                extra=extra,
            )
        torch.distributed.barrier()
    if mainp:
        print("Done.")
        if run is not None:
            run.finish()
    ddp_cleanup()


if __name__ == "__main__":
    main()
