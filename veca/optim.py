from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from dion import NorMuon

from veca.polar import polar_express_orthogonalize


def build_optimizer_param_groups(student_module: nn.Module, cfg) -> Tuple[list, list, list, list]:
    muon_params = []
    adamw_params = []
    muon_names = []
    adamw_names = []
    module_dict = dict(student_module.named_modules())
    last_fc2_name = f"blocks.{cfg.num_layers - 1}.fc2.weight"
    for par_name, par_val in student_module.named_parameters():
        if not par_val.requires_grad:
            continue
        parent_module = None
        leaf_name = None
        if "." in par_name:
            parent_name, leaf_name = par_name.rsplit(".", 1)
            parent_module = module_dict[parent_name]
        is_linear_weight = (
            parent_module is not None
            and isinstance(parent_module, nn.Linear)
            and leaf_name == "weight"
            and par_val.ndim == 2
        )
        is_bias = par_name.endswith(".bias")
        is_query_tokens = par_name.startswith("query_token_chunks.")
        is_query_coords = par_name.startswith("qcoord_chunks.")
        is_pos_alpha = par_name == "pos_alpha"
        is_coord_output_linear = par_name.startswith("pos_linears.") and par_name.endswith(".weight")
        is_final_output_linear = par_name == last_fc2_name
        use_muon = (
            is_linear_weight
            and not is_bias
            and not is_query_tokens
            and not is_query_coords
            and not is_pos_alpha
            and not is_coord_output_linear
            and not is_final_output_linear
        )
        if use_muon:
            muon_params.append(par_val)
            muon_names.append(par_name)
        else:
            adamw_params.append(par_val)
            adamw_names.append(par_name)
    muon_ids = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    all_ids = {id(p) for p in student_module.parameters() if p.requires_grad}
    assert len(muon_ids & adamw_ids) == 0
    assert (muon_ids | adamw_ids) == all_ids
    assert len(muon_params) > 0
    assert len(adamw_params) > 0
    return muon_params, adamw_params, muon_names, adamw_names


def count_params(params) -> int:
    return sum(p.numel() for p in params)


def build_optimizers(student_ddp, cfg):
    student_module = student_ddp.module
    nor_muon_params, adamw_params, muon_names, adamw_names = build_optimizer_param_groups(student_module, cfg)
    optimizer_adam = torch.optim.AdamW(
        adamw_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eps=cfg.adamw_eps,
    )
    optimizer_muon = NorMuon(
        nor_muon_params,
        distributed_mesh=student_ddp.process_group,
        lr=cfg.lr,
        mu=cfg.normuon_mu,
        muon_beta2=cfg.normuon_beta2,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.normuon_nesterov,
        adjust_lr=cfg.normuon_adjust_lr,
        flatten=cfg.normuon_flatten,
        use_triton=cfg.normuon_use_triton,
        newton_schulz_func=polar_express_orthogonalize,
        cautious_wd=cfg.normuon_cautious_wd,
    )
    return optimizer_adam, optimizer_muon, muon_names, adamw_names
