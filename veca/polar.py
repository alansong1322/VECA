from __future__ import annotations

from itertools import repeat
from math import inf, sqrt

import numpy as np
import torch


def optimal_quintic(l, u):
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u ** 3), (3 / 8) / (u ** 5)
    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    E, old_E = inf, None
    while (old_E is None) or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l ** 3, l ** 5, 1],
            [q, q ** 3, q ** 5, -1],
            [r, r ** 3, r ** 5, 1],
            [u, u ** 3, u ** 5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt(
            (-3 * b + np.array([-1, 1]) * sqrt(9 * b ** 2 - 20 * a * c)) / (10 * c)
        )
    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, safety_factor_eps=0, cushion=0):
    u = 1
    assert 0 <= l <= u
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for iter_idx in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion * u), u)
        if cushion * u > l:
            pl = a * l + b * l ** 3 + c * l ** 5
            pu = a * u + b * u ** 3 + c * u ** 5
            rescaler = 2 / (pl + pu)
            a *= rescaler
            b *= rescaler
            c *= rescaler
        if iter_idx < num_iters - 1:
            a /= safety_factor
            b /= safety_factor ** 3
            c /= safety_factor ** 5
        coefficients.append((a, b, c))
        l = a * l + b * l ** 3 + c * l ** 5
        u = 2 - l
    return coefficients


coeffs_list = optimal_composition(l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.02)


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def polar_express_orthogonalize(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    _ = epsilon
    return PolarExpress(x, steps=5)
