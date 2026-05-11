from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from veca.checkpoint import load_state_dict_forgiving, save_ckpt
from veca.config import MultiResFinetuneConfig, cfg_to_dict, normalize_nested_settings
from veca.data import create_multires_object365_loaders_ddp, ensure_index_file, require_data_paths
from veca.ddp import ddp_allreduce_mean_multi, ddp_cleanup, ddp_setup_from_torchrun, is_main_process, sample_active_k_ddp
from veca.optim import build_optimizers, count_params
from veca.schedules import autocast_ctx, configure_backend_precision, get_cosine_schedule_with_warmup
from veca.teacher import DistillVECA

MODEL_ARCH_VERSION = "veca_obj365_unfused_qkv_multires_nested_chunkparams_v3"
OPTIMIZER_LAYOUT_VERSION = "muon_clean_split_v1"
TRAINING_LAYOUT_VERSION = "multires_obj365_only_nested_v1"


def normalize_prob_dict(prob_dict: Dict[int, float]) -> Tuple[List[int], np.ndarray]:
    keys = sorted(prob_dict.keys())
    probs = np.array([float(prob_dict[k]) for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    return keys, probs


def sample_resolution_for_step(cfg: MultiResFinetuneConfig, global_step: int, *, rank: int, device: torch.device, rng) -> int:
    if global_step < cfg.phase_a_steps:
        keys, probs = normalize_prob_dict(cfg.phase_a_probs)
    else:
        keys, probs = normalize_prob_dict(cfg.phase_b_probs)
    if rank == 0:
        assert rng is not None
        chosen = int(rng.choice(keys, p=probs))
    else:
        chosen = 0
    t = torch.tensor([chosen], device=device, dtype=torch.int64)
    torch.distributed.broadcast(t, src=0)
    return int(t.item())


def set_coord_path_trainable(student_module, trainable: bool) -> None:
    for m in student_module.pos_linears:
        for p in m.parameters():
            p.requires_grad_(trainable)
    student_module.pos_alpha.requires_grad_(trainable)


def coord_path_is_trainable(student_module) -> bool:
    if student_module.pos_alpha.requires_grad:
        return True
    return any(p.requires_grad for m in student_module.pos_linears for p in m.parameters())


@torch.no_grad()
def validate_one_resolution_budget_ddp(
    model: DistillVECA,
    val_mgr,
    resolution: int,
    active_k: int,
    global_step: int,
    cfg: MultiResFinetuneConfig,
    *,
    rank: int,
    world_size: int,
    enable_wandb: bool,
):
    assert active_k in cfg.val_budgets
    model.student.eval()
    model.teacher.eval()
    meters = {
        "loss": 0.0,
        "cls": 0.0,
        "dense": 0.0,
        "dense_cos": 0.0,
        "dense_mse": 0.0,
    }
    steps = int(cfg.val_steps_by_res[resolution])
    for _ in range(steps):
        imgs_cpu = val_mgr.next_batch(resolution)
        imgs = imgs_cpu.to(model.device_, non_blocking=True)
        with autocast_ctx(cfg):
            loss, logs = model.distill_step(
                imgs,
                active_k=active_k,
                teacher_microbatch_override=cfg.teacher_microbatch_by_res[resolution],
            )
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
            f"[VAL][step {global_step:06d}][res {resolution}][K {active_k}] "
            f"loss={meters['loss']:.4f} cls={meters['cls']:.4f} "
            f"dense={meters['dense']:.4f} dcos={meters['dense_cos']:.4f} dmse={meters['dense_mse']:.4f}"
        )
        if enable_wandb:
            wandb.log(
                {
                    f"val/{resolution}/k{active_k}/loss": float(meters["loss"]),
                    f"val/{resolution}/k{active_k}/loss_cls": float(meters["cls"]),
                    f"val/{resolution}/k{active_k}/loss_dense": float(meters["dense"]),
                    f"val/{resolution}/k{active_k}/loss_dense_cos": float(meters["dense_cos"]),
                    f"val/{resolution}/k{active_k}/loss_dense_mse": float(meters["dense_mse"]),
                },
                step=global_step,
            )
    return meters


@torch.no_grad()
def validate_all_resolutions_budgets_ddp(
    model: DistillVECA,
    val_mgr,
    global_step: int,
    cfg: MultiResFinetuneConfig,
    *,
    rank: int,
    world_size: int,
    enable_wandb: bool,
):
    grid_metrics = {}
    for res in cfg.resolutions:
        grid_metrics[res] = {}
        for active_k in cfg.val_budgets:
            grid_metrics[res][active_k] = validate_one_resolution_budget_ddp(
                model=model,
                val_mgr=val_mgr,
                resolution=res,
                active_k=active_k,
                global_step=global_step,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                enable_wandb=enable_wandb,
            )
    avg_grid_loss = float(np.mean([grid_metrics[r][k]["loss"] for r in cfg.resolutions for k in cfg.val_budgets]))
    avg_grid_cls = float(np.mean([grid_metrics[r][k]["cls"] for r in cfg.resolutions for k in cfg.val_budgets]))
    avg_grid_dense = float(np.mean([grid_metrics[r][k]["dense"] for r in cfg.resolutions for k in cfg.val_budgets]))
    avg_loss_by_res = {
        int(r): float(np.mean([grid_metrics[r][k]["loss"] for k in cfg.val_budgets]))
        for r in cfg.resolutions
    }
    avg_loss_by_budget = {
        int(k): float(np.mean([grid_metrics[r][k]["loss"] for r in cfg.resolutions]))
        for k in cfg.val_budgets
    }
    if is_main_process(rank):
        by_res_msg = " ".join([f"r{r}:{avg_loss_by_res[r]:.4f}" for r in cfg.resolutions])
        by_budget_msg = " ".join([f"k{k}:{avg_loss_by_budget[k]:.4f}" for k in cfg.val_budgets])
        print(f"[VAL][step {global_step:06d}][grid avg] loss={avg_grid_loss:.4f} cls={avg_grid_cls:.4f} dense={avg_grid_dense:.4f}")
        print(f"[VAL][step {global_step:06d}][by_res] {by_res_msg}")
        print(f"[VAL][step {global_step:06d}][by_budget] {by_budget_msg}")
        if enable_wandb:
            payload = {
                "val/avg_grid_loss": avg_grid_loss,
                "val/avg_grid_cls": avg_grid_cls,
                "val/avg_grid_dense": avg_grid_dense,
            }
            for r in cfg.resolutions:
                payload[f"val/avg_loss_res_{r}"] = avg_loss_by_res[r]
            for k in cfg.val_budgets:
                payload[f"val/avg_loss_k_{k}"] = avg_loss_by_budget[k]
            wandb.log(payload, step=global_step)
    return grid_metrics, avg_grid_loss, avg_loss_by_res, avg_loss_by_budget


@record
def main(cfg: MultiResFinetuneConfig | None = None) -> None:
    cfg = cfg or MultiResFinetuneConfig()
    normalize_nested_settings(cfg)
    assert torch.cuda.is_available()
    assert hasattr(torch, "compile")
    assert hasattr(F, "scaled_dot_product_attention")
    assert cfg.hidden_dim % cfg.num_heads == 0
    assert (cfg.hidden_dim // cfg.num_heads) % 4 == 0
    assert cfg.total_steps > cfg.phase_a_steps >= 0
    assert 0 < cfg.warmup_steps <= cfg.total_steps
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
        require_data_paths(cfg)
        ensure_index_file(cfg.train_dir, cfg.train_index)
        ensure_index_file(cfg.val_dir, cfg.val_index)
    torch.distributed.barrier()
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
    if cfg.init_ckpt and os.path.exists(cfg.init_ckpt):
        map_loc = {"cuda:0": f"cuda:{local_rank}"}
        ckpt = torch.load(cfg.init_ckpt, map_location=map_loc)
        missing, unexpected, skipped = load_state_dict_forgiving(model.student, ckpt["student"])
        if mainp:
            print(f"[INIT CKPT] Loaded student init from: {cfg.init_ckpt}")
            print(f"[INIT CKPT] missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}")
    elif mainp:
        print(f"[INIT CKPT] Not found: {cfg.init_ckpt}")
    model.student = DDP(model.student, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
    optimizer, optimizer_muon, muon_names, adamw_names = build_optimizers(model.student, cfg)
    min_ratio = float(cfg.lr_min) / float(cfg.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.total_steps, min_lr_ratio=min_ratio)
    scheduler_muon = get_cosine_schedule_with_warmup(optimizer_muon, cfg.warmup_steps, cfg.total_steps, min_lr_ratio=min_ratio)
    start_step = 0
    best_val_avg_grid = float("inf")
    if cfg.resume_if_exists and os.path.exists(cfg.ft_ckpt_name):
        map_loc = {"cuda:0": f"cuda:{local_rank}"}
        ckpt = torch.load(cfg.ft_ckpt_name, map_location=map_loc)
        if ckpt.get("model_arch_version") is not None:
            assert ckpt.get("model_arch_version") == MODEL_ARCH_VERSION
        if ckpt.get("optimizer_layout_version") is not None:
            assert ckpt.get("optimizer_layout_version") == OPTIMIZER_LAYOUT_VERSION
        if ckpt.get("training_layout_version") is not None:
            assert ckpt.get("training_layout_version") == TRAINING_LAYOUT_VERSION
        missing, unexpected = model.student.module.load_state_dict(ckpt["student"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "optimizer_muon" in ckpt:
            optimizer_muon.load_state_dict(ckpt["optimizer_muon"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scheduler_muon" in ckpt:
            scheduler_muon.load_state_dict(ckpt["scheduler_muon"])
        start_step = int(ckpt.get("global_step", 0))
        best_val_avg_grid = float(ckpt.get("best_val_avg_grid", float("inf")))
        if mainp:
            print(f"[RESUME CKPT] Loaded: {cfg.ft_ckpt_name}")
            print(f"[RESUME CKPT] missing={len(missing)} unexpected={len(unexpected)}")
            print(f"[RESUME CKPT] start_step={start_step} best_val_avg_grid={best_val_avg_grid:.6f}")
    if start_step < cfg.freeze_pos_steps:
        set_coord_path_trainable(model.student.module, trainable=False)
        if mainp:
            print(f"[Freeze] coord residual path frozen for first {cfg.freeze_pos_steps} steps")
    else:
        set_coord_path_trainable(model.student.module, trainable=True)
    train_mgr, val_mgr = create_multires_object365_loaders_ddp(cfg, rank=rank, world_size=world_size)
    train_mgr.set_epoch_all(0)
    val_mgr.set_epoch_all(0)
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
        for res in cfg.resolutions:
            print(f"[Batch][res {res}] per-rank Object365 batch={cfg.batch_size_by_res[res]}")
        print(f"[ValGrid] resolutions={cfg.resolutions} budgets={cfg.val_budgets}")
        if enable_wandb:
            wandb.log({"params/total": total, "params/adamw": adam_n, "params/normuon": muon_n}, step=0)
    torch.distributed.barrier()
    resolution_rng = np.random.default_rng(cfg.resolution_seed + start_step) if mainp else None
    model.student.train()
    model.teacher.eval()
    running = {
        "loss": 0.0,
        "cls": 0.0,
        "dense": 0.0,
        "dense_cos": 0.0,
        "dense_mse": 0.0,
        "active_k": 0.0,
    }
    running_count = 0
    budget_hist = {int(k): 0 for k in cfg.nested_budgets}
    pbar = None
    if mainp:
        pbar = tqdm(total=cfg.total_steps - start_step, desc="multires-nested-ft", ncols=150, mininterval=5.0, dynamic_ncols=True)
    for global_step in range(start_step, cfg.total_steps):
        if global_step == cfg.freeze_pos_steps and not coord_path_is_trainable(model.student.module):
            set_coord_path_trainable(model.student.module, trainable=True)
            if mainp:
                print(f"[Freeze] Unfroze coord residual path at step {global_step}")
        res = sample_resolution_for_step(cfg, global_step=global_step, rank=rank, device=device, rng=resolution_rng)
        active_k = sample_active_k_ddp(cfg, device=device, rank=rank)
        budget_hist[active_k] += 1
        imgs_cpu = train_mgr.next_batch(res)
        assert imgs_cpu.shape[0] == cfg.batch_size_by_res[res]
        imgs = imgs_cpu.to(model.device_, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        optimizer_muon.zero_grad(set_to_none=True)
        with autocast_ctx(cfg):
            loss, logs = model.distill_step(
                imgs,
                active_k=active_k,
                teacher_microbatch_override=cfg.teacher_microbatch_by_res[res],
            )
        loss.backward()
        optimizer.step()
        optimizer_muon.step()
        scheduler.step()
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
        running["loss"] += loss_m
        running["cls"] += cls_m
        running["dense"] += dense_m
        running["dense_cos"] += dcos_m
        running["dense_mse"] += dmse_m
        running["active_k"] += k_m
        running_count += 1
        if mainp:
            lr_now = float(scheduler.get_last_lr()[0])
            coord_trainable = int(coord_path_is_trainable(model.student.module))
            if pbar is not None:
                pbar.update(1)
                if (global_step % 25 == 0) or (global_step == cfg.total_steps - 1):
                    pbar.set_postfix({
                        "step": f"{global_step + 1}/{cfg.total_steps}",
                        "res": str(res),
                        "K": str(active_k),
                        "Kavg": f"{running['active_k'] / running_count:.1f}",
                        "loss": f"{running['loss'] / running_count:.4f}",
                        "cls": f"{running['cls'] / running_count:.4f}",
                        "dense": f"{running['dense'] / running_count:.4f}",
                        "lr": f"{lr_now:.1e}",
                        "bs": f"{cfg.batch_size_by_res[res]}",
                        "coord": coord_trainable,
                    })
            if enable_wandb and (((global_step + 1) % int(cfg.wandb_log_every) == 0) or (global_step == cfg.total_steps - 1)):
                wandb.log(
                    {
                        "train/resolution": res,
                        "train/active_k": active_k,
                        "train/lr": lr_now,
                        "train/loss": loss_m,
                        "train/loss_cls": cls_m,
                        "train/loss_dense": dense_m,
                        "train/loss_dense_cos": dcos_m,
                        "train/loss_dense_mse": dmse_m,
                        "train/coord_path_trainable": coord_trainable,
                        "train/batch_size": cfg.batch_size_by_res[res],
                        "train/teacher_microbatch": cfg.teacher_microbatch_by_res[res],
                    },
                    step=global_step + 1,
                )
        do_eval = ((global_step + 1) % cfg.eval_every_steps == 0) or (global_step == cfg.total_steps - 1)
        if do_eval:
            torch.distributed.barrier()
            grid_val, avg_grid_val, avg_loss_by_res, avg_loss_by_budget = validate_all_resolutions_budgets_ddp(
                model=model,
                val_mgr=val_mgr,
                global_step=global_step + 1,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                enable_wandb=enable_wandb,
            )
            model.student.train()
            model.teacher.eval()
            if mainp and cfg.save_best_by_avg_val and avg_grid_val < best_val_avg_grid:
                best_val_avg_grid = avg_grid_val
                extra = {
                    "global_step": global_step + 1,
                    "best_val_avg_grid": best_val_avg_grid,
                    "best_val_per_res_budget": {
                        str(r): {str(k): float(grid_val[r][k]["loss"]) for k in cfg.val_budgets}
                        for r in cfg.resolutions
                    },
                    "best_val_avg_by_res": {str(r): float(avg_loss_by_res[r]) for r in cfg.resolutions},
                    "best_val_avg_by_budget": {str(k): float(avg_loss_by_budget[k]) for k in cfg.val_budgets},
                    "pos_alpha": model.student.module.pos_alpha.detach().float().cpu().tolist(),
                    "last_budget_hist": {str(k): int(v) for k, v in budget_hist.items()},
                    "model_arch_version": MODEL_ARCH_VERSION,
                    "optimizer_layout_version": OPTIMIZER_LAYOUT_VERSION,
                    "training_layout_version": TRAINING_LAYOUT_VERSION,
                }
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
                if enable_wandb:
                    wandb.log({"best/val_avg_grid_loss": float(best_val_avg_grid)}, step=global_step + 1)
            torch.distributed.barrier()
        do_save = ((global_step + 1) % cfg.save_every_steps == 0) or (global_step == cfg.total_steps - 1)
        if do_save and mainp and cfg.always_save_latest:
            extra = {
                "global_step": global_step + 1,
                "best_val_avg_grid": best_val_avg_grid,
                "pos_alpha": model.student.module.pos_alpha.detach().float().cpu().tolist(),
                "last_budget_hist": {str(k): int(v) for k, v in budget_hist.items()},
                "model_arch_version": MODEL_ARCH_VERSION,
                "optimizer_layout_version": OPTIMIZER_LAYOUT_VERSION,
                "training_layout_version": TRAINING_LAYOUT_VERSION,
            }
            save_ckpt(
                cfg.ft_ckpt_name,
                student_ddp=model.student,
                optimizer_adam=optimizer,
                optimizer_muon=optimizer_muon,
                scheduler_adam=scheduler,
                scheduler_muon=scheduler_muon,
                cfg=cfg,
                extra=extra,
            )
        if mainp and (global_step + 1) in cfg.milestone_steps:
            milestone_name = cfg.ft_ckpt_name.replace(".pt", f"_step{global_step + 1}.pt")
            extra = {
                "global_step": global_step + 1,
                "best_val_avg_grid": best_val_avg_grid,
                "pos_alpha": model.student.module.pos_alpha.detach().float().cpu().tolist(),
                "last_budget_hist": {str(k): int(v) for k, v in budget_hist.items()},
                "model_arch_version": MODEL_ARCH_VERSION,
                "optimizer_layout_version": OPTIMIZER_LAYOUT_VERSION,
                "training_layout_version": TRAINING_LAYOUT_VERSION,
            }
            save_ckpt(
                milestone_name,
                student_ddp=model.student,
                optimizer_adam=optimizer,
                optimizer_muon=optimizer_muon,
                scheduler_adam=scheduler,
                scheduler_muon=scheduler_muon,
                cfg=cfg,
                extra=extra,
            )
        torch.distributed.barrier()
        if mainp and (((global_step + 1) % 200 == 0) or (global_step == cfg.total_steps - 1)):
            hist_msg = " ".join([f"K{k}:{budget_hist[k]}" for k in cfg.nested_budgets])
            print(
                f"[TRAIN][step {global_step + 1:06d}] res={res} K={active_k} "
                f"loss={running['loss'] / running_count:.4f} cls={running['cls'] / running_count:.4f} "
                f"dense={running['dense'] / running_count:.4f} dcos={running['dense_cos'] / running_count:.4f} "
                f"dmse={running['dense_mse'] / running_count:.4f} Kavg={running['active_k'] / running_count:.2f}"
            )
            print(f"[BUDGET][step {global_step + 1:06d}] {hist_msg}")
            if coord_path_is_trainable(model.student.module):
                alphas = model.student.module.pos_alpha.detach().float().cpu().tolist()
                print(f"[ALPHA][step {global_step + 1:06d}] {' '.join(f'{a:+.4f}' for a in alphas)}")
                if enable_wandb:
                    wandb.log({f"alpha/layer_{i}": float(a) for i, a in enumerate(alphas)}, step=global_step + 1)
            running = {
                "loss": 0.0,
                "cls": 0.0,
                "dense": 0.0,
                "dense_cos": 0.0,
                "dense_mse": 0.0,
                "active_k": 0.0,
            }
            running_count = 0
    if mainp:
        if pbar is not None:
            pbar.close()
        print("Done.")
        if run is not None:
            run.finish()
    ddp_cleanup()


if __name__ == "__main__":
    main()
