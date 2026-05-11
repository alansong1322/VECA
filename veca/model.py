from __future__ import annotations

import gc
import math
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def load_timm_patch_embed_proj(model_name: str):
    m = timm.create_model(model_name, pretrained=True)
    assert hasattr(m, "patch_embed") and hasattr(m.patch_embed, "proj")
    proj = m.patch_embed.proj
    w = proj.weight.detach().cpu()
    b = proj.bias.detach().cpu() if (hasattr(proj, "bias") and proj.bias is not None) else None
    del m
    gc.collect()
    return w, b


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        init_from_pretrained: bool,
        pretrained_timm_name: str,
        use_ln: bool,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.LayerNorm(embed_dim) if use_ln else nn.Identity()
        self.loaded_pretrained = False
        if init_from_pretrained:
            assert pretrained_timm_name
            self._load_weights(pretrained_timm_name)

    @torch.no_grad()
    def _load_weights(self, name: str) -> None:
        w, b = load_timm_patch_embed_proj(name)
        if w.shape != self.proj.weight.shape:
            print(f"[PatchEmbed] Skipped timm patch_embed.proj from {name}: expected {tuple(self.proj.weight.shape)}, got {tuple(w.shape)}")
            return
        self.proj.weight.copy_(w)
        if b is not None and self.proj.bias is not None:
            assert b.shape == self.proj.bias.shape
            self.proj.bias.copy_(b)
        self.loaded_pretrained = True
        print(f"[PatchEmbed] Loaded patch_embed.proj from timm: {name}")
        gc.collect()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (Hp, Wp)


@torch.no_grad()
def farthest_point_sampling_2d(points: torch.Tensor, k: int) -> torch.Tensor:
    M = points.shape[0]
    if k >= M:
        return torch.arange(M, device=points.device)
    dist0 = (points ** 2).sum(dim=1)
    first = torch.argmin(dist0)
    selected = torch.empty(k, dtype=torch.long, device=points.device)
    selected[0] = first
    d = ((points - points[first]) ** 2).sum(dim=1)
    for t in range(1, k):
        idx = torch.argmax(d)
        selected[t] = idx
        d = torch.minimum(d, ((points - points[idx]) ** 2).sum(dim=1))
    return selected


@torch.no_grad()
def init_query_coords_fps(num_queries: int, pool_size: int, coord_range: float) -> torch.Tensor:
    pool_tanh = torch.empty(pool_size, 2).uniform_(-coord_range, coord_range)
    idx = farthest_point_sampling_2d(pool_tanh, num_queries)
    q_tanh = pool_tanh[idx].clone()
    return torch.atanh(q_tanh.clamp(-0.999, 0.999))


class Rotary2D(nn.Module):
    def __init__(
        self,
        head_dim: int,
        *,
        base: float = 100.0,
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        p_zero_freqs: float = 0.0,
        zero_freqs_from: str = "low",
    ):
        super().__init__()
        assert head_dim % 4 == 0
        assert zero_freqs_from in ("low", "high")
        self.head_dim = head_dim
        self.base = base
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.p_zero_freqs = float(p_zero_freqs)
        self.zero_freqs_from = zero_freqs_from
        self.register_buffer("periods", torch.empty(head_dim // 4), persistent=False)
        self._init_periods()

    @torch.no_grad()
    def _init_periods(self) -> None:
        Hd = self.head_dim
        periods = self.base ** (2 * torch.arange(Hd // 4, dtype=torch.float32) / (Hd // 2))
        n_zero = int(round(self.p_zero_freqs * periods.numel()))
        if n_zero > 0:
            if self.zero_freqs_from == "low":
                periods[-n_zero:] = float("inf")
            else:
                periods[:n_zero] = float("inf")
        self.periods.copy_(periods)

    def augment_coords_patch(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return coords
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if self.shift_coords is not None:
            shift = coords.new_empty(2).uniform_(-self.shift_coords, self.shift_coords)
            coords = coords + shift.view(1, 1, 2)
        if self.jitter_coords is not None:
            jitter_max = math.log(float(self.jitter_coords))
            jitter = coords.new_empty(2).uniform_(-jitter_max, jitter_max).exp()
            coords = coords * jitter.view(1, 1, 2)
        if self.rescale_coords is not None:
            rescale_max = math.log(float(self.rescale_coords))
            s = coords.new_empty(1).uniform_(-rescale_max, rescale_max).exp()
            coords = coords * s.view(1, 1, 1)
        return coords.squeeze(0) if squeeze else coords

    def cos_sin(self, coords: torch.Tensor):
        coords_f = coords.float()
        periods = self.periods.float().to(coords.device)
        x = coords_f[..., 0:1]
        y = coords_f[..., 1:2]
        ang_x = (2.0 * math.pi) * x / periods.view(1, 1, -1)
        ang_y = (2.0 * math.pi) * y / periods.view(1, 1, -1)
        ang_x = ang_x.repeat_interleave(2, dim=-1)
        ang_y = ang_y.repeat_interleave(2, dim=-1)
        angles = torch.cat([ang_x, ang_y], dim=-1)
        return angles.cos(), angles.sin()

    @staticmethod
    def rotate_half(x: torch.Tensor):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return x * cos + self.rotate_half(x) * sin


class MaskedBiDirAttentionSDPA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.out_drop = nn.Identity() if float(dropout) == 0.0 else nn.Dropout(dropout)
        self.attn_drop_p = float(dropout)

    def forward(self, x_all: torch.Tensor, rope: Rotary2D, cos, sin, k: int) -> torch.Tensor:
        B, N, D = x_all.shape
        H, Hd = self.num_heads, self.head_dim
        assert 1 <= k < N
        q = self.q_proj(x_all)
        kk = self.k_proj(x_all)
        v = self.v_proj(x_all)
        q = q.view(B, N, H, Hd).transpose(1, 2)
        kk = kk.view(B, N, H, Hd).transpose(1, 2)
        v = v.view(B, N, H, Hd).transpose(1, 2)
        cos_ = cos.to(dtype=q.dtype)
        sin_ = sin.to(dtype=q.dtype)
        q = rope.apply(q, cos_, sin_)
        kk = rope.apply(kk, cos_, sin_)
        drop_p = self.attn_drop_p if self.training else 0.0
        out_q = F.scaled_dot_product_attention(
            q[:, :, :k, :],
            kk,
            v,
            dropout_p=drop_p,
            is_causal=False,
        )
        out_p = F.scaled_dot_product_attention(
            q[:, :, k:, :],
            kk[:, :, :k, :],
            v[:, :, :k, :],
            dropout_p=drop_p,
            is_causal=False,
        )
        out = torch.cat([out_q, out_p], dim=2)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return self.out_drop(out)


class QueryBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = MaskedBiDirAttentionSDPA(dim, num_heads, dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, 2 * hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        self.drop = nn.Identity() if float(dropout) == 0.0 else nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope: Rotary2D, cos, sin, k: int) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), rope=rope, cos=cos, sin=sin, k=k)
        y = self.norm_ffn(x)
        u, v = self.fc1(y).chunk(2, dim=-1)
        y = F.silu(u) * v
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class VECA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Q_total = cfg.num_query_tokens
        self.chunk_size = int(cfg.nested_chunk_size)
        assert self.Q_total % self.chunk_size == 0
        self.num_query_chunks = self.Q_total // self.chunk_size
        self.patch_embed = PatchEmbed(
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.hidden_dim,
            init_from_pretrained=cfg.init_patch_from_dinov3,
            pretrained_timm_name=cfg.dinov3_timm_name,
            use_ln=cfg.patch_embed_use_ln,
        )
        self.query_token_chunks = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.chunk_size, cfg.hidden_dim) / math.sqrt(cfg.hidden_dim))
            for _ in range(self.num_query_chunks)
        ])
        q0 = init_query_coords_fps(self.Q_total, cfg.fps_pool_size, cfg.fps_coord_range).unsqueeze(0)
        self.qcoord_chunks = nn.ParameterList([
            nn.Parameter(q0[:, i * self.chunk_size:(i + 1) * self.chunk_size, :].contiguous())
            for i in range(self.num_query_chunks)
        ])
        self.rope = Rotary2D(
            head_dim=cfg.hidden_dim // cfg.num_heads,
            base=100.0,
            shift_coords=cfg.rope_shift_coords,
            jitter_coords=cfg.rope_jitter_coords,
            rescale_coords=cfg.rope_rescale_coords,
            p_zero_freqs=cfg.rope_p_zero_freqs,
            zero_freqs_from=cfg.rope_zero_freqs_from,
        )
        self.pos_linears = nn.ModuleList([
            nn.Linear(cfg.hidden_dim, 2, bias=True)
            for _ in range(cfg.num_layers - 1)
        ])
        self.pos_alpha = nn.Parameter(torch.full((cfg.num_layers - 1,), 0.01))
        self.blocks = nn.ModuleList([
            QueryBlock(cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output_dim = int(cfg.distill_output_dim or cfg.hidden_dim)
        if self.output_dim == cfg.hidden_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(cfg.hidden_dim, self.output_dim, bias=True)
        self._grid_cache: dict = {}
        self._init_weights()

    def _init_weights(self) -> None:
        if not self.patch_embed.loaded_pretrained:
            nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_active_query_tensors(self, B: int, K: int):
        n_chunks = K // self.chunk_size
        q_tok = torch.cat([self.query_token_chunks[i] for i in range(n_chunks)], dim=1).expand(B, -1, -1)
        q_raw = torch.cat([self.qcoord_chunks[i] for i in range(n_chunks)], dim=1).expand(B, -1, -1)
        return q_tok, q_raw

    @torch.no_grad()
    def _get_patch_grid(self, Hp: int, Wp: int, device: torch.device) -> torch.Tensor:
        key = (str(device), Hp, Wp)
        if key in self._grid_cache:
            return self._grid_cache[key]
        y = (torch.arange(Hp, device=device, dtype=torch.float32) + 0.5) / Hp * 2.0 - 1.0
        x = (torch.arange(Wp, device=device, dtype=torch.float32) + 0.5) / Wp * 2.0 - 1.0
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
        self._grid_cache[key] = coords
        return coords

    def forward(self, images: torch.Tensor, active_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = images.shape[0]
        K = self.Q_total if active_k is None else int(active_k)
        assert 1 <= K <= self.Q_total
        assert K % self.chunk_size == 0
        patches, (Hp, Wp) = self.patch_embed(images)
        queries, p_q_raw = self._build_active_query_tensors(B, K)
        p_img = self._get_patch_grid(Hp, Wp, patches.device).unsqueeze(0).expand(B, -1, -1)
        p_img = self.rope.augment_coords_patch(p_img)
        p_q = torch.tanh(p_q_raw)
        x = torch.cat([queries, patches], dim=1)
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx > 0:
                q = x[:, :K, :]
                delta = self.pos_linears[layer_idx - 1](q)
                alpha = self.pos_alpha[layer_idx - 1].to(dtype=delta.dtype)
                p_q_raw = p_q_raw + alpha * delta
                p_q = torch.tanh(p_q_raw)
            coords = torch.cat([p_q, p_img], dim=1)
            cos, sin = self.rope.cos_sin(coords)
            x = block(x, self.rope, cos, sin, K)
        x = self.final_norm(x)
        queries = x[:, :K, :]
        patches = x[:, K:, :]
        queries = self.output_proj(queries)
        patches = self.output_proj(patches)
        return queries[:, 0, :], patches
