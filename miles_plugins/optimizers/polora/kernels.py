"""Eigendecomposition-free matrix kernels for Polora.

Vendored from https://github.com/nikhilgsh/polora (``polora/utils.py``).
"""

from __future__ import annotations

import torch

__all__ = ["power_iter_top", "polar_express_gram_batched", "ns_inv_sqrt"]


def _normalize_or_zero(X, dim):
    """Normalize over ``dim``, mapping zero-norm inputs to zero."""
    norm = X.norm(dim=dim, keepdim=True)
    is_zero = norm == 0
    safe_norm = torch.where(is_zero, torch.ones_like(norm), norm)
    return torch.where(is_zero, torch.zeros_like(X), X / safe_norm)


def power_iter_top(M, symmetric=False, v_init=None, n_iters=8):
    """Estimate the largest singular value or eigenvalue by power iteration.

    Args:
        M: Matrix batch ``(..., m, n)``.
        symmetric: Treat ``M`` as symmetric positive semidefinite.
        v_init: Optional warm-start vector ``(..., k)``.
        n_iters: Number of iterations.

    Returns:
        The estimate and the final vector, which can be reused as ``v_init``.
    """
    if M.numel() == 0:
        return torch.zeros(M.shape[:-2], device=M.device, dtype=torch.float32), None
    Mf = M.float() if M.dtype != torch.float32 else M
    if symmetric:
        Mf = 0.5 * (Mf + Mf.transpose(-2, -1))
    Mt = Mf.transpose(-2, -1)
    *batch, m, n = Mf.shape
    left = symmetric or m <= n

    if left:
        v_ones = (Mf @ torch.ones(*batch, n, 1, device=Mf.device, dtype=torch.float32)).squeeze(-1)
        scores = Mf.square().sum(dim=-1)
    else:
        v_ones = (Mt @ torch.ones(*batch, m, 1, device=Mf.device, dtype=torch.float32)).squeeze(-1)
        scores = Mf.square().sum(dim=-2)
    floor = scores.amax(dim=-1).sqrt()

    v_fallback = v_ones

    if v_init is None:
        v = v_fallback
    else:
        v_norm = v_init.norm(dim=-1, keepdim=True)
        restart = (~torch.isfinite(v_norm)) | (v_norm == 0)
        v = torch.where(restart, v_fallback, v_init)
    v = _normalize_or_zero(v, dim=-1).unsqueeze(-1)

    for _ in range(n_iters):
        if symmetric:
            v = Mf @ v
        elif left:
            v = Mf @ (Mt @ v)
        else:
            v = Mt @ (Mf @ v)
        v = _normalize_or_zero(v, dim=-2)

    if symmetric:
        value = (v * (Mf @ v)).sum(dim=(-2, -1))
    elif left:
        value = (Mt @ v).norm(dim=(-2, -1))
    else:
        value = (Mf @ v).norm(dim=(-2, -1))
    return torch.maximum(value, floor), v.squeeze(-1)


# PolarExpress quintic fits from Amsel et al. (arXiv:2505.16932).
_POLAR_EXPRESS_RAW = (
    (8.31968561540051, -23.85945031896673, 17.53144504181025),
    (4.123266419055485, -2.980709974760902, 0.5520797360741728),
    (3.9656114721772022, -2.941248486368672, 0.5589295013119786),
    (3.3312009004415994, -2.4980080275937966, 0.5111272954169234),
    (2.320007312889811, -1.6862169729967622, 0.42068027340235137),
    (1.8951443404954809, -1.2722050191923813, 0.377227344813122),
    (1.875006772051659, -1.250007524486259, 0.37500075245392533),
    (1.8750008025096776, -1.2500016050034328, 0.375000802493755),
    (1.8749954784656357, -1.249990956953079, 0.374995478487443),
    (1.8749954775662068, -1.2499909551542292, 0.3749954775880226),
)
_SAFETY = 1.01


def _polar_coeffs(nsteps):
    """Return ``nsteps`` coefficients from the PolarExpress schedule."""
    take = _POLAR_EXPRESS_RAW[:nsteps]
    if len(take) < nsteps:
        take = take + (_POLAR_EXPRESS_RAW[-1],) * (nsteps - len(take))
    s, s3, s5 = _SAFETY, _SAFETY**3, _SAFETY**5
    return tuple((a, b, c) if i == nsteps - 1 else (a / s, b / s3, c / s5) for i, (a, b, c) in enumerate(take))


def polar_express_gram_batched(X, nsteps=8):
    """Compute batched polar factors with the Gram-form PolarExpress iteration.

    The iteration operates on the smaller Gram matrix and accumulates ``Q`` so
    that ``Q @ X`` approaches the polar factor.

    Args:
        X: Matrix batch ``(..., m, n)``.
        nsteps: Number of polynomial iterations.

    Returns:
        The polar factor of ``X`` in float32, same shape as ``X``.
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return X
    X = X.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-2, -1)
    orig_leading = X.shape[:-2]
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    r = X.shape[-2]
    X_normed = _normalize_or_zero(X, dim=(-2, -1))

    coeffs = _polar_coeffs(nsteps)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if X.is_cuda else None
    if X.is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        eye = torch.eye(r, dtype=X_normed.dtype, device=X_normed.device)
        R = X_normed @ X_normed.transpose(-2, -1)
        Q = eye.expand(*X_normed.shape[:-2], r, r).clone()
        for a, b, c in coeffs:
            R2 = R @ R
            M = b * R + c * R2
            M.diagonal(dim1=-2, dim2=-1).add_(a)
            Q = M @ Q
            R = M @ R @ M
    finally:
        if X.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    out = (Q @ X_normed).reshape(*orig_leading, r, X_normed.shape[-1])
    return out.transpose(-2, -1) if tall else out


def ns_inv_sqrt(S, nsteps=8, eps=1e-4, eps_relative=False, floor=1e-12):
    """Compute a damped inverse square root with Newton-Schulz iteration.

    Args:
        S: Symmetric positive-semidefinite matrix batch ``(..., r, r)``.
        nsteps: Number of iterations.
        eps: Diagonal damping.
        eps_relative: Scale damping by ``lambda_max(S)``.
        floor: Minimum relative damping.

    Returns:
        ``(S + delta_eff*I)^{-1/2}``, same shape and dtype as ``S``.
    """
    S = 0.5 * (S + S.transpose(-2, -1))
    n = S.shape[-1]
    dev, dt = S.device, S.dtype
    eye = torch.eye(n, dtype=dt, device=dev).expand_as(S)
    batch_shape = (*S.shape[:-2], 1, 1)

    if eps_relative:
        lam_max, _ = power_iter_top(S, symmetric=True, n_iters=8)
        eps_eff = (eps * lam_max.reshape(batch_shape)).clamp_min(floor).to(dt)
    else:
        eps_eff = torch.as_tensor(float(eps), device=dev, dtype=dt)
    S_d = S + eps_eff * eye
    tr = S_d.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(batch_shape)
    scale = tr
    R0 = S_d / scale

    coeffs = _polar_coeffs(nsteps)

    R = R0.clone()
    Q = eye.clone()
    use_tf32_guard = (dt == torch.float32) and S.is_cuda
    prev_tf32 = None
    if use_tf32_guard:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for a, b, c in coeffs:
            R2 = R @ R
            M = (b * R) + (c * R2)
            M.diagonal(dim1=-2, dim2=-1).add_(a)
            Q = M @ Q
            R = M @ R @ M
            R = 0.5 * (R + R.transpose(-2, -1))
    finally:
        if use_tf32_guard:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return Q / scale.sqrt()
