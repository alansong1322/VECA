from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


def _env_path(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _index_dir() -> str:
    return _env_path("OBJECT365_INDEX_DIR", "indices")


def _train_dir() -> str:
    root = _env_path("OBJECT365_ROOT")
    return _env_path("OBJECT365_TRAIN_DIR", os.path.join(root, "train") if root else "")


def _val_dir() -> str:
    root = _env_path("OBJECT365_ROOT")
    return _env_path("OBJECT365_VAL_DIR", os.path.join(root, "val") if root else "")


def _train_index() -> str:
    return _env_path("OBJECT365_TRAIN_INDEX", os.path.join(_index_dir(), "object365_train_paths.txt"))


def _val_index() -> str:
    return _env_path("OBJECT365_VAL_INDEX", os.path.join(_index_dir(), "object365_val_paths.txt"))


FAMILY_PRESETS = {
    "small": {
        "num_query_tokens": 64,
        "hidden_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "mlp_ratio": 2.66667,
        "dropout": 0.0,
    },
    "splus": {
        "num_query_tokens": 64,
        "hidden_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
    },
    "base": {
        "num_query_tokens": 64,
        "hidden_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "mlp_ratio": 2.66667,
        "dropout": 0.0,
    },
    "large": {
        "num_query_tokens": 64,
        "hidden_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "mlp_ratio": 2.66667,
        "dropout": 0.0,
    },
}


@dataclass
class SharedConfig:
    model_family: str = "base"
    num_query_tokens: int = 64
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 2.66667
    dropout: float = 0.0
    patch_size: int = 16
    in_channels: int = 3
    patch_embed_use_ln: bool = False
    init_patch_from_dinov3: bool = True
    dinov3_timm_name: str = "vit_base_patch16_dinov3.lvd1689m"
    rope_shift_coords: float | None = 0.08
    rope_jitter_coords: float | None = None
    rope_rescale_coords: float | None = 2.0
    rope_p_zero_freqs: float = 0.125
    rope_zero_freqs_from: str = "low"
    fps_pool_size: int = 1000
    fps_coord_range: float = 0.95
    nested_chunk_size: int = 8
    nested_budgets: Tuple[int, ...] = (8, 16, 24, 32, 40, 48, 56, 64)
    nested_budget_weights: Tuple[float, ...] = (1, 1, 2, 2, 3, 3, 4, 4)
    teacher_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    teacher_num_register_tokens: int | None = None
    teacher_compile: bool = False
    teacher_compile_mode: str = "max-autotune"
    teacher_dtype: str = "bf16"
    distill_output_dim: int | None = None
    weight_dense: float = 1.0
    dense_norm_eps: float = 1e-6
    dense_mse_weight: float = 1.0
    lr: float = 4e-4
    lr_min: float = 5e-5
    adamw_eps: float = 1e-8
    weight_decay: float = 0.015
    mixed_precision: str = "bf16"
    train_dir: str = field(default_factory=_train_dir)
    val_dir: str = field(default_factory=_val_dir)
    index_dir: str = field(default_factory=_index_dir)
    train_index: str = field(default_factory=_train_index)
    val_index: str = field(default_factory=_val_index)
    wandb_project: str = "veca"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_log_every: int = 50
    normuon_mu: float = 0.95
    normuon_beta2: float = 0.95
    normuon_nesterov: bool = True
    normuon_adjust_lr: str = "rms_norm"
    normuon_use_triton: bool = False
    normuon_flatten: bool = False
    normuon_cautious_wd: bool = True
    base_seed: int = 42


@dataclass
class PretrainConfig(SharedConfig):
    image_size: int = 256
    resize_short: int = 288
    val_active_k: int = 64
    teacher_microbatch: int = 128
    batch_size: int = 128
    epochs: int = 135
    warmup_epochs: int = 25
    steps_per_epoch: int = 10000
    num_workers: int = 8
    save_every_epochs: int = 5
    validate_every_epochs: int = 5
    ckpt_name: str = "veca_pretrain_dinov3vitb16_256_q64_nested.pt"
    best_ckpt_name: str = "veca_pretrain_dinov3vitb16_256_q64_nested_best.pt"


@dataclass
class MultiResFinetuneConfig(SharedConfig):
    lr: float = 8e-5
    lr_min: float = 3e-5
    weight_decay: float = 0.01
    resolutions: Tuple[int, ...] = (256, 384, 512, 768)
    resize_short_by_res: Dict[int, int] = field(default_factory=lambda: {256: 288, 384: 432, 512: 576, 768: 864})
    batch_size_by_res: Dict[int, int] = field(default_factory=lambda: {256: 128, 384: 128, 512: 128, 768: 128})
    teacher_microbatch_by_res: Dict[int, int] = field(default_factory=lambda: {256: 128, 384: 64, 512: 32, 768: 8})
    phase_a_steps: int = 7000
    phase_a_probs: Dict[int, float] = field(default_factory=lambda: {256: 0.35, 384: 0.35, 512: 0.30})
    total_steps: int = 50000
    phase_b_probs: Dict[int, float] = field(default_factory=lambda: {256: 0.15, 384: 0.25, 512: 0.30, 768: 0.30})
    freeze_pos_steps: int = 0
    warmup_steps: int = 7000
    eval_every_steps: int = 10000
    save_every_steps: int = 10000
    milestone_steps: Tuple[int, ...] = (10000, 20000, 30000, 40000)
    val_steps_by_res: Dict[int, int] = field(default_factory=lambda: {256: 100, 384: 100, 512: 100, 768: 50})
    val_budgets: Tuple[int, ...] = (8, 16, 24, 32, 40, 48, 56, 64)
    save_best_by_avg_val: bool = True
    always_save_latest: bool = True
    train_num_workers: int = 2
    val_num_workers: int = 2
    prefetch_factor: int = 2
    init_ckpt: str = "veca_pretrain_dinov3vitb16_256_q64_nested_best.pt"
    ft_ckpt_name: str = "veca_multires_finetune_dinov3vitb16_q64_nested.pt"
    best_ckpt_name: str = "veca_multires_finetune_dinov3vitb16_q64_nested_best.pt"
    resume_if_exists: bool = True
    wandb_run_name: str | None = "veca-multires-nested-finetune"
    resolution_seed: int = 424242


def cfg_to_dict(cfg: SharedConfig) -> dict:
    return asdict(cfg)


def apply_family_preset(cfg: SharedConfig) -> None:
    family = str(cfg.model_family).lower()
    assert family in FAMILY_PRESETS, f"Unknown VECA model_family={cfg.model_family}. Available: {tuple(FAMILY_PRESETS)}"
    cfg.model_family = family
    for key, value in FAMILY_PRESETS[family].items():
        setattr(cfg, key, value)


def normalize_nested_settings(cfg: SharedConfig) -> None:
    apply_family_preset(cfg)
    budgets = tuple(int(k) for k in cfg.nested_budgets)
    weights = tuple(float(w) for w in cfg.nested_budget_weights)
    assert cfg.num_query_tokens > 0
    assert cfg.nested_chunk_size > 0
    assert cfg.num_query_tokens % cfg.nested_chunk_size == 0
    assert len(budgets) > 0
    assert budgets == tuple(sorted(budgets))
    assert len(set(budgets)) == len(budgets)
    assert budgets[-1] == cfg.num_query_tokens
    for k in budgets:
        assert 1 <= k <= cfg.num_query_tokens
        assert k % cfg.nested_chunk_size == 0
    assert len(weights) == len(budgets)
    assert all(w > 0 for w in weights)
    if hasattr(cfg, "val_active_k"):
        assert int(cfg.val_active_k) in budgets
        cfg.val_active_k = int(cfg.val_active_k)
    if hasattr(cfg, "val_budgets"):
        val_budgets = tuple(int(k) for k in cfg.val_budgets)
        assert len(val_budgets) > 0
        for k in val_budgets:
            assert k in budgets
        cfg.val_budgets = val_budgets
    cfg.nested_budgets = budgets
    cfg.nested_budget_weights = weights
