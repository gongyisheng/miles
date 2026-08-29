"""Polora optimizer adapted for Megatron LoRA training.

The update follows https://github.com/nikhilgsh/polora. This version supports
Megatron-Bridge adapters and ``main_grad``, and stores optimizer state in fp32,
including an fp32 master copy of every LoRA factor.

Master weights are not an optional refinement here. Polora rescales each factor
update to spectral norm ``rho = lr / (sigma_max(A) + sigma_max(B))``, so with
lr=5e-4 and a rank-32 adapter the per-entry step lands around 1e-5 while the
entries of ``A`` sit around 1.5e-2 -- a relative step of ~5e-4, well under
bf16's ~4e-3 resolution. Added straight to a bf16 factor, most of that update
rounds away: over 200 steps of a fixed gradient direction, a bf16 ``A`` travels
less than half as far as the same run in fp32. The optimizer therefore owns
fp32 masters, applies the update there, and casts back into the (bf16) module
parameter, which stays the tensor the forward pass and the weight sync read.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.optim import Optimizer

from .kernels import ns_inv_sqrt, polar_express_gram_batched, power_iter_top


def _grad_of(param):
    """Return ``main_grad`` when Megatron DDP populated it, else ``grad``."""
    grad = getattr(param, "main_grad", None)
    return param.grad if grad is None else grad


def collect_lora_pairs(model):
    """Find trainable LoRA ``(A, B)`` pairs in PEFT or Megatron-Bridge models.

    Args:
        model: A model or sequence of Megatron model chunks.

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
            if hasattr(lora_A, "keys"):
                pairs.extend((lora_A[name].weight, lora_B[name].weight) for name in lora_A if name in lora_B)
            elif hasattr(lora_A, "weight") and hasattr(lora_B, "weight"):
                pairs.append((lora_A.weight, lora_B.weight))
    return [(A, B) for A, B in pairs if A.requires_grad and B.requires_grad]


class Polora(Optimizer):
    """Spectral-preconditioned LoRA optimizer.

    Each LoRA pair is whitened in a diagonal-Kronecker curvature metric, passed
    through the matrix sign, unwhitened, and rescaled to spectral norm ``rho``.

    The update is applied to fp32 master copies of the factors (``W_A``/``W_B``
    in the per-pair state) and then cast into the module parameters, so bf16
    rounding never eats the step. The masters ride along in ``state_dict()``.

    Args:
        model: Megatron model chunk(s). Provide either ``model`` or ``pairs``.
        lr: Learning rate; each factor update is rescaled to spectral norm
            ``rho = lr / (sigma_max(A) + sigma_max(B))``.
        beta1: Momentum coefficient.
        epsilon: Numerical floor for preconditioning and normalization.
        delta: Relative damping for the ``C_A``/``C_B`` and ``Q``/``P`` inverse
            square roots.
        curvature_beta: EMA coefficient for the diagonal preconditioners.
        ns_steps: PolarExpress iterations.
        higham_iters: Newton-Schulz iterations.
        compile: Compile the spectral kernels.
        pairs: Explicit ``[(A, B), ...]`` list.
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
        if compile:
            self._polar_fn = torch.compile(polar_express_gram_batched, dynamic=True, fullgraph=False)
            self._invsqrt_fn = torch.compile(ns_inv_sqrt, dynamic=True, fullgraph=False)
        else:
            self._polar_fn = polar_express_gram_batched
            self._invsqrt_fn = ns_inv_sqrt

    def _pair_state(self, A, B):
        """Allocate fp32 state for a parameter pair on first use."""
        st = self.state[A]
        if not st:
            st["M_A"] = torch.zeros_like(A, dtype=torch.float32)
            st["M_B"] = torch.zeros_like(B, dtype=torch.float32)
            st["Q"] = torch.full((A.shape[1],), self.epsilon, dtype=torch.float32, device=A.device)
            st["P"] = torch.full((B.shape[0],), self.epsilon, dtype=torch.float32, device=B.device)
        # Seeded separately from the block above so a checkpoint written before
        # masters existed resumes by seeding them from the saved bf16 factors
        # rather than losing its momentum and curvature.
        if "W_A" not in st:
            st["W_A"] = A.detach().float().clone()
            st["W_B"] = B.detach().float().clone()
        return st

    @torch.no_grad()
    def sync_masters_from_params(self):
        """Re-seed the fp32 masters from the module parameters.

        Call after anything writes the factors behind the optimizer's back (a
        checkpoint load into the live model, for instance). Pairs whose masters
        have not been allocated yet are left alone: they seed from the params on
        the next step anyway.
        """
        for A, B in self.pairs:
            st = self.state.get(A)
            if st and "W_A" in st:
                st["W_A"].copy_(A.detach().float())
                st["W_B"].copy_(B.detach().float())

    def load_state_dict(self, state_dict):
        """Restore state without torch downcasting it to the parameter dtype."""
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
        """Return ``(x / x_max + delta)^{-1/2}`` along the last dimension."""
        xmax = x.amax(dim=-1, keepdim=True)
        return (x / xmax + self.delta).rsqrt()

    def _smax_warm(self, M, states, key):
        """Estimate batched ``sigma_max`` with cached warm starts."""
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
        """Compute the matrix sign with PolarExpress."""
        return self._polar_fn(X, nsteps=self.ns_steps)

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one Polora update to every LoRA pair.

        Gradients must already be populated on every ``(A, B)`` factor, either
        as Megatron ``main_grad`` bucket views or plain ``.grad``.

        Args:
            closure: Callable that re-evaluates the model and returns the loss
                (standard ``torch.optim`` protocol).

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

        # Batch pairs with matching shapes and devices.
        groups = defaultdict(list)
        for i, (A, B) in enumerate(pairs):
            groups[(A.shape[0], A.shape[1], B.shape[0], A.device)].append(i)
        for idxs in groups.values():
            G_A = torch.stack([_grad_of(pairs[i][0]).float() for i in idxs])
            G_B = torch.stack([_grad_of(pairs[i][1]).float() for i in idxs])
            # The masters, not the (bf16) parameters, are the weights Polora
            # differentiates around: the parameters are a rounded view of them.
            Aw = torch.stack([S[i]["W_A"] for i in idxs])
            Bw = torch.stack([S[i]["W_B"] for i in idxs])
            M_A = torch.stack([S[i]["M_A"] for i in idxs]).mul_(b1).add_(G_A, alpha=1.0 - b1)
            M_B = torch.stack([S[i]["M_B"] for i in idxs]).mul_(b1).add_(G_B, alpha=1.0 - b1)
            Q = torch.stack([S[i]["Q"] for i in idxs])
            P = torch.stack([S[i]["P"] for i in idxs])

            Q_isqrt = self._rdinv(Q)
            P_isqrt = self._rdinv(P)
            Q_dmp = (Q_isqrt * Q_isqrt).reciprocal()
            P_dmp = (P_isqrt * P_isqrt).reciprocal()
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

            # Update each diagonal from the opposite factor's curvature.
            r = G_A.shape[1]
            Q.mul_(cb).add_((G_A * (C_B_inv @ G_A)).sum(dim=1), alpha=(1.0 - cb) / r)
            P.mul_(cb).add_((G_B * (G_B @ C_A_inv)).sum(dim=2), alpha=(1.0 - cb) / r)

            for j, i in enumerate(idxs):
                A_, B_ = pairs[i]
                W_A, W_B = S[i]["W_A"], S[i]["W_B"]
                W_A.add_(dA[j])
                W_B.add_(dB[j])
                # copy_ casts, and writes through the flat-buffer view Megatron
                # may have re-mapped the parameter onto.
                A_.copy_(W_A)
                B_.copy_(W_B)
                S[i]["M_A"].copy_(M_A[j])
                S[i]["M_B"].copy_(M_B[j])
                S[i]["Q"].copy_(Q[j])
                S[i]["P"].copy_(P[j])

        return loss
