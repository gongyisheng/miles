"""PoLoRA -- spectral-preconditioned LoRA optimizer, ported for Megatron.

Vendored from https://github.com/nikhilgsh/polora (Apache-2.0, see
LICENSE.upstream). The update math is unchanged; see ``kernels.py`` and the
upstream module docstring for the derivation. What differs here:

  * Pair discovery walks Megatron-Bridge adapters (``.adapter.linear_in`` /
    ``.adapter.linear_out``) rather than PEFT's ``lora_A`` / ``lora_B``. The
    weight layouts already agree: ``linear_in.weight`` is ``(r, d_in)`` and
    ``linear_out.weight`` is ``(d_out, r)``.
  * Gradients are read from Megatron DDP's ``main_grad`` bucket view when
    present, falling back to ``.grad`` for plain-torch use and tests.
  * ``step()`` does not zero gradients. Megatron owns that lifecycle through
    ``optimizer.zero_grad()``; zeroing the bucket view here would clobber
    gradient accumulation across microbatches.
  * Per-pair state lives in torch's own ``self.state[A]`` instead of a private
    ``pair_state`` dict, so ``state_dict``/``load_state_dict`` serialize it by
    param index and miles' LoRA-adapter resume works unmodified.

State is fp32 regardless of parameter dtype: ``M_A``/``M_B`` (momentum) plus the
``Q``/``P`` diagonals, roughly half of Adam's footprint.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.optim import Optimizer

from .kernels import ns_inv_sqrt, polar_express_gram_batched, power_iter_top


def _grad_of(param):
    """Returns the gradient Megatron actually populated for ``param``.

    Megatron's DDP accumulates into ``main_grad`` (a view into the bucket
    buffer) and leaves ``.grad`` as None, so ``.grad`` alone would see nothing.
    """
    grad = getattr(param, "main_grad", None)
    return param.grad if grad is None else grad


def collect_lora_pairs(model):
    """Discovers trainable LoRA ``(A, B)`` weight pairs on a model.

    Handles both naming conventions in play: PEFT exposes ``lora_A`` / ``lora_B``
    (a ``ModuleDict`` keyed by adapter name, or a bare linear), while
    Megatron-Bridge adapters name the same two factors ``linear_in`` /
    ``linear_out``. Both lay the weights out identically, so the pair is
    ``(r, d_in)`` and ``(d_out, r)`` either way.

    Frozen pairs are skipped, so an adapter sharing the model but not being
    trained (e.g. a DPO reference adapter) is not collected.

    Args:
        model: A model, or a sequence of model chunks (Megatron builds one per
            virtual pipeline stage).

    Returns:
        List of ``(A, B)`` parameter pairs in module-discovery order.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    pairs = []
    for chunk in chunks:
        for _, mod in chunk.named_modules():
            lora_A = getattr(mod, "lora_A", None)
            lora_B = getattr(mod, "lora_B", None)
            if lora_A is None or lora_B is None:
                lora_A = getattr(mod, "linear_in", None)
                lora_B = getattr(mod, "linear_out", None)
            if lora_A is None or lora_B is None:
                continue
            if hasattr(lora_A, "keys"):  # PEFT ModuleDict: one entry per adapter name
                pairs.extend((lora_A[name].weight, lora_B[name].weight) for name in lora_A if name in lora_B)
            elif hasattr(lora_A, "weight") and hasattr(lora_B, "weight"):
                pairs.append((lora_A.weight, lora_B.weight))
    return [(A, B) for A, B in pairs if A.requires_grad and B.requires_grad]


class Polora(Optimizer):
    """Spectral-preconditioned LoRA optimizer.

    Each LoRA pair is whitened in a diagonal-Kronecker curvature metric, passed
    through the matrix sign, unwhitened, and rescaled to spectral norm ``rho``.

    Args:
        model: Megatron model chunk(s); adapter pairs are discovered
            automatically. Provide either ``model`` or ``pairs``. (Default: None)
        lr: Learning rate; each factor update is rescaled to spectral norm
            ``rho = lr / (sigma_max(A) + sigma_max(B))``. (Default: 2e-4)
        beta1: Momentum coefficient. (Default: 0.9)
        epsilon: Numerical floor for preconditioner init, inverse-sqrt damping,
            and the spectral-norm normalizations. (Default: 1e-12)
        delta: Relative damping for the ``C_A``/``C_B`` and ``Q``/``P`` inverse
            square roots. (Default: 1e-4)
        curvature_beta: EMA coefficient for the diagonal preconditioners.
            (Default: 0.99)
        ns_steps: Iterations of the PolarExpress matrix-sign solver. (Default: 8)
        higham_iters: Newton-Schulz iterations for the inverse square roots.
            (Default: 8)
        compile: If True, wraps the two spectral kernels in ``torch.compile``;
            the update is unchanged. (Default: False)
        pairs: Explicit ``[(A, B), ...]`` list, bypassing discovery.
            (Default: None)
    """

    def __init__(
        self,
        model=None,
        lr=2e-4,
        beta1=0.9,
        epsilon=1e-12,
        delta=1e-4,
        curvature_beta=0.99,
        ns_steps=8,
        higham_iters=8,
        compile=False,
        *,
        pairs=None,
    ):
        if pairs is None:
            if model is None:
                raise ValueError("Polora requires either `model` or `pairs=[(A,B),...]`.")
            pairs = collect_lora_pairs(model)
        else:
            pairs = list(pairs)
        if not pairs:
            raise ValueError("No trainable LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.beta1 = beta1
        self.curvature_beta = float(curvature_beta)
        self.ns_steps = int(ns_steps)
        self.higham_iters = int(higham_iters)
        # dynamic=True lets the LoRA shape groups share one compiled kernel.
        if compile:
            self._polar_fn = torch.compile(polar_express_gram_batched, dynamic=True, fullgraph=False)
            self._invsqrt_fn = torch.compile(ns_inv_sqrt, dynamic=True, fullgraph=False)
        else:
            self._polar_fn = polar_express_gram_batched
            self._invsqrt_fn = ns_inv_sqrt

    def _pair_state(self, A, B):
        """Per-pair state, allocated fp32 on first use regardless of param dtype."""
        st = self.state[A]
        if not st:
            st["M_A"] = torch.zeros_like(A, dtype=torch.float32)
            st["M_B"] = torch.zeros_like(B, dtype=torch.float32)
            st["Q"] = torch.full((A.shape[1],), self.epsilon, dtype=torch.float32, device=A.device)
            st["P"] = torch.full((B.shape[0],), self.epsilon, dtype=torch.float32, device=B.device)
        return st

    def load_state_dict(self, state_dict):
        """Restores state without letting torch downcast it.

        torch's base implementation casts every floating state tensor to the
        owning parameter's dtype as it loads. Under bf16 LoRA factors that
        rounds the fp32 momentum and preconditioners to bf16, and re-casting to
        fp32 afterwards cannot recover the lost mantissa bits -- so this maps
        the saved state onto parameters directly instead of calling super().
        """
        params = [p for group in self.param_groups for p in group["params"]]
        saved_groups = state_dict["param_groups"]
        saved_ids = [pid for group in saved_groups for pid in group["params"]]
        if len(saved_ids) != len(params):
            raise ValueError(
                f"polora state_dict covers {len(saved_ids)} params but this optimizer holds "
                f"{len(params)}; the adapter layout changed between save and resume"
            )
        id_to_param = dict(zip(saved_ids, params, strict=True))

        self.state = defaultdict(dict)
        for saved_id, saved_state in state_dict["state"].items():
            param = id_to_param[saved_id]
            restored = {}
            for key, value in saved_state.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    value = value.detach().to(device=param.device, dtype=torch.float32)
                restored[key] = value
            self.state[param] = restored

        for group, saved_group in zip(self.param_groups, saved_groups, strict=True):
            group.update({k: v for k, v in saved_group.items() if k != "params"})

    @staticmethod
    def _sym(M):
        return 0.5 * (M + M.transpose(-2, -1))

    def _rdinv(self, x):
        """Relative-damped inverse square root along the last dim:
        ``(x / x_max + delta)^{-1/2}``. ``x_max > 0`` from step 1 because the
        preconditioner EMA is seeded at ``epsilon``."""
        xmax = x.amax(dim=-1, keepdim=True)
        return (x / xmax + self.delta).rsqrt()

    def _smax_warm(self, M, states, key):
        """Batched ``sigma_max`` with per-pair warm-start caching, floored at
        ``max(row L2, col L2)`` -- a lower bound on ``sigma_max`` -- so a cold
        or degenerate start vector cannot under-estimate the top value."""
        cached = [st.get(key) for st in states]
        vi = torch.stack(cached) if all(c is not None for c in cached) else None
        s, v = power_iter_top(M, v_init=vi, n_iters=8)
        for j, st in enumerate(states):
            st[key] = v[j].detach()
        with torch.no_grad():
            Mf = M.detach().float()
            rn = Mf.pow(2).sum(dim=-1).amax(dim=-1).sqrt()
            cn = Mf.pow(2).sum(dim=-2).amax(dim=-1).sqrt()
            s = torch.maximum(s, torch.maximum(rn, cn).reshape(s.shape).to(s.dtype))
        return s

    def _polar_batched(self, X):
        """Matrix sign via PolarExpress (Frobenius-prescaled internally)."""
        return self._polar_fn(X, nsteps=self.ns_steps)

    @torch.no_grad()
    def step(self, closure=None):
        """Applies one Polora update to every LoRA pair.

        Gradients must already be populated on every ``(A, B)`` factor, either
        as Megatron ``main_grad`` bucket views or plain ``.grad``. Unlike
        upstream this does not zero them; Megatron's ``zero_grad()`` owns that.

        Args:
            closure: Callable that re-evaluates the model and returns the loss
                (standard ``torch.optim`` protocol). (Default: None)

        Returns:
            The loss returned by ``closure``, or None if no closure is given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        lr = self.param_groups[0]["lr"]
        cb = self.curvature_beta
        b1, eps = self.beta1, self.epsilon
        pairs = self.pairs

        for A, B in pairs:
            if _grad_of(A) is None or _grad_of(B) is None:
                raise ValueError("Gradients are required for Polora update.")
        S = [self._pair_state(A, B) for A, B in pairs]

        # One batched bmm / Newton-Schulz / sigma_max per (r, d_in, d_out, device)
        # shape group; rank and device must match for torch.stack.
        groups = defaultdict(list)
        for i, (A, B) in enumerate(pairs):
            groups[(A.shape[0], A.shape[1], B.shape[0], A.device)].append(i)
        for idxs in groups.values():
            G_A = torch.stack([_grad_of(pairs[i][0]).float() for i in idxs])
            G_B = torch.stack([_grad_of(pairs[i][1]).float() for i in idxs])
            Aw = torch.stack([pairs[i][0].detach().float() for i in idxs])
            Bw = torch.stack([pairs[i][1].detach().float() for i in idxs])
            M_A = torch.stack([S[i]["M_A"] for i in idxs]).mul_(b1).add_(G_A, alpha=1.0 - b1)
            M_B = torch.stack([S[i]["M_B"] for i in idxs]).mul_(b1).add_(G_B, alpha=1.0 - b1)
            Q = torch.stack([S[i]["Q"] for i in idxs])  # d_in metric diagonal
            P = torch.stack([S[i]["P"] for i in idxs])  # d_out metric diagonal

            Q_isqrt = self._rdinv(Q)
            P_isqrt = self._rdinv(P)
            Q_dmp = (Q_isqrt * Q_isqrt).reciprocal()
            P_dmp = (P_isqrt * P_isqrt).reciprocal()
            # Curvature factors, rebuilt each step (not stored).
            C_B = Bw.transpose(-2, -1) @ (P_dmp.unsqueeze(-1) * Bw)
            C_A = (Aw * Q_dmp.unsqueeze(1)) @ Aw.transpose(-2, -1)
            C_B_isqrt = self._invsqrt_fn(
                self._sym(C_B), nsteps=self.higham_iters, eps=self.delta, eps_relative=True, floor=self.epsilon
            )
            C_A_isqrt = self._invsqrt_fn(
                self._sym(C_A), nsteps=self.higham_iters, eps=self.delta, eps_relative=True, floor=self.epsilon
            )
            C_B_inv = C_B_isqrt @ C_B_isqrt
            C_A_inv = C_A_isqrt @ C_A_isqrt

            # Nesterov look-ahead.
            M_hat_A = M_A.mul(b1).add(G_A, alpha=1.0 - b1)
            M_hat_B = M_B.mul(b1).add(G_B, alpha=1.0 - b1)

            grp = [S[i] for i in idxs]
            sigma_A = self._smax_warm(Aw, grp, "v_sigma_A")
            sigma_B = self._smax_warm(Bw, grp, "v_sigma_B")
            rho = lr / (sigma_B + sigma_A)

            # Whiten -> polar -> unwhiten.
            zA = (C_B_isqrt @ M_hat_A) * Q_isqrt.unsqueeze(1)
            zB = (M_hat_B @ C_A_isqrt) * P_isqrt.unsqueeze(-1)
            zA = self._polar_batched(zA)
            zB = self._polar_batched(zB)
            D_A = (C_B_isqrt @ zA) * Q_isqrt.unsqueeze(1)
            D_B = (zB @ C_A_isqrt) * P_isqrt.unsqueeze(-1)
            sigma_DA = self._smax_warm(D_A, grp, "v_sigma_DA")
            sigma_DB = self._smax_warm(D_B, grp, "v_sigma_DB")
            dA = -(rho / sigma_DA.clamp_min(eps)).view(-1, 1, 1) * D_A
            dB = -(rho / sigma_DB.clamp_min(eps)).view(-1, 1, 1) * D_B

            # Preconditioner update: each diagonal accumulates the gradient
            # whitened by the inverse of the other side's r x r matrix.
            r = G_A.shape[1]
            Q.mul_(cb).add_((G_A * (C_B_inv @ G_A)).sum(dim=1), alpha=(1.0 - cb) / r)
            P.mul_(cb).add_((G_B * (G_B @ C_A_inv)).sum(dim=2), alpha=(1.0 - cb) / r)

            for j, i in enumerate(idxs):
                A_, B_ = pairs[i]
                A_.add_(dA[j].to(dtype=A_.dtype, device=A_.device))
                B_.add_(dB[j].to(dtype=B_.dtype, device=B_.device))
                S[i]["M_A"].copy_(M_A[j])
                S[i]["M_B"].copy_(M_B[j])
                S[i]["Q"].copy_(Q[j])
                S[i]["P"].copy_(P[j])

        return loss
