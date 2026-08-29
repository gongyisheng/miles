#!/usr/bin/env python3
"""Compare Miles' Polora kernels and optimizer against upstream.

Usage: ``python scripts/polora_numerical_diff.py [--upstream DIR] [--steps N]``
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/nikhilgsh/polora"

FAILURES: list[str] = []


def load_upstream(upstream_dir: Path):
    """Clone upstream if needed and import its ``polora`` package."""
    if not (upstream_dir / "polora" / "optim.py").exists():
        print(f"cloning {UPSTREAM_URL} -> {upstream_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", UPSTREAM_URL, str(upstream_dir)],
            check=True,
            capture_output=True,
        )
    sha = subprocess.run(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"upstream: {upstream_dir} @ {sha}")

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg_spec = importlib.util.spec_from_file_location(
        "up_polora", upstream_dir / "polora" / "__init__.py", submodule_search_locations=[str(upstream_dir / "polora")]
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["up_polora"] = pkg
    utils = _load("up_polora.utils", upstream_dir / "polora" / "utils.py")
    pkg.utils = utils
    optim = _load("up_polora.optim", upstream_dir / "polora" / "optim.py")
    return optim, utils


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Return absolute and relative maximum differences in float64."""
    a64, b64 = a.detach().double(), b.detach().double()
    max_abs = (a64 - b64).abs().max().item() if a64.numel() else 0.0
    scale = b64.abs().max().item() if b64.numel() else 0.0
    return max_abs, (max_abs / scale if scale > 0 else max_abs)


def report(label: str, a: torch.Tensor, b: torch.Tensor, tol: float = 0.0, kind: str = "abs") -> None:
    """Print a comparison and record failures."""
    max_abs, rel = rel_diff(a, b)
    metric = max_abs if kind == "abs" else rel
    ok = metric <= tol
    status = "OK  " if ok else "FAIL"
    exact = " (bit-exact)" if max_abs == 0.0 else ""
    print(f"  [{status}] {label:<58} max_abs={max_abs:.3e}  rel={rel:.3e}{exact}")
    if not ok:
        FAILURES.append(f"{label}: {kind}={metric:.3e} > tol={tol:.3e}")


def report_bool(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


def check_kernels(mine, up, device: str) -> None:
    print(f"\n== kernels vs upstream ({device}) ==")
    g = torch.Generator(device="cpu").manual_seed(0)

    shapes = [(3, 8, 64), (3, 64, 8), (2, 16, 16), (4, 4, 128)]
    for b, m, n in shapes:
        X = torch.randn(b, m, n, generator=g).to(device)
        for nsteps in (5, 8, 12):
            report(
                f"polar_express_gram_batched({b},{m},{n}) nsteps={nsteps}",
                mine.polar_express_gram_batched(X, nsteps=nsteps),
                up.polar_express_gram_batched(X, nsteps=nsteps),
            )

    X0 = torch.zeros(2, 8, 32, device=device)
    v = torch.randn(2, 8, 1, generator=g).to(device)
    X1 = v @ torch.randn(2, 1, 32, generator=g).to(device)
    for name, X in [("zeros", X0), ("rank1", X1), ("1e8*randn", torch.randn(2, 8, 32, generator=g).to(device) * 1e8)]:
        report(
            f"polar_express degenerate[{name}]", mine.polar_express_gram_batched(X), up.polar_express_gram_batched(X)
        )

    for b, r in [(3, 8), (2, 16), (1, 4)]:
        F = torch.randn(b, r, r + 4, generator=g).to(device)
        S = F @ F.transpose(-2, -1)
        for kwargs in ({}, {"eps_relative": True, "eps": 1e-4, "floor": 1e-12}, {"eps": 1e-2}):
            report(
                f"ns_inv_sqrt({b},{r},{r}) {kwargs or 'defaults'}",
                mine.ns_inv_sqrt(S, **kwargs),
                up.ns_inv_sqrt(S, **kwargs),
            )
    Sz = torch.zeros(2, 6, 6, device=device)
    report(
        "ns_inv_sqrt degenerate[zeros, relative]",
        mine.ns_inv_sqrt(Sz, eps_relative=True),
        up.ns_inv_sqrt(Sz, eps_relative=True),
    )

    for b, m, n in [(3, 8, 64), (3, 64, 8), (2, 16, 16)]:
        M = torch.randn(b, m, n, generator=g).to(device)
        for sym in (False, True):
            Msym = M @ M.transpose(-2, -1) if sym else M
            s_mine, v_mine = mine.power_iter_top(Msym, symmetric=sym)
            s_up, v_up = up.power_iter_top(Msym, symmetric=sym)
            report(f"power_iter_top({b},{m},{n}) symmetric={sym} value", s_mine, s_up)
            report(f"power_iter_top({b},{m},{n}) symmetric={sym} vector", v_mine, v_up)


def check_kernels_vs_dense(mine, device: str) -> None:
    """Compare the iterative kernels with dense linear algebra."""
    print(f"\n== kernels vs dense ground truth ({device}) ==")
    g = torch.Generator(device="cpu").manual_seed(1)

    for b, m, n in [(3, 8, 64), (3, 64, 8), (2, 16, 16)]:
        X = torch.randn(b, m, n, generator=g).to(device).double()
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
        polar = U @ Vh
        got = mine.polar_express_gram_batched(X.float(), nsteps=8)
        report(f"polar_express({b},{m},{n}) vs SVD polar", got.double(), polar, tol=2e-2, kind="rel")

    for b, r in [(3, 8), (2, 16)]:
        F = torch.randn(b, r, r + 4, generator=g).to(device).double()
        S = F @ F.transpose(-2, -1)
        lam, Vv = torch.linalg.eigh(S)
        lam_max = lam[..., -1:]
        S_d = S + (1e-4 * lam_max).unsqueeze(-1) * torch.eye(r, dtype=S.dtype, device=S.device)
        lam_d, Vd = torch.linalg.eigh(S_d)
        ref = (Vd * lam_d.clamp_min(0).rsqrt().unsqueeze(-2)) @ Vd.transpose(-2, -1)
        got = mine.ns_inv_sqrt(S.float(), nsteps=8, eps=1e-4, eps_relative=True)
        report(f"ns_inv_sqrt({b},{r}) vs eigh", got.double(), ref, tol=5e-2, kind="rel")

    # Power iteration estimates sigma_max from below.
    for b, m, n in [(3, 8, 64), (2, 16, 16)]:
        M = torch.randn(b, m, n, generator=g).to(device)
        s, _ = mine.power_iter_top(M)
        ref = torch.linalg.matrix_norm(M, ord=2)
        err = ((ref - s) / ref).double()
        report_bool(
            f"power_iter_top({b},{m},{n}) is a lower bound on ||M||_2, within 5%",
            bool((err >= -1e-6).all() and (err <= 5e-2).all()),
            f"relative shortfall per batch = {[f'{e:.2e}' for e in err.tolist()]}",
        )
        v = None
        for _ in range(4):
            s_warm, v = mine.power_iter_top(M, v_init=v)
        report_bool(
            f"power_iter_top({b},{m},{n}) warm-started is tighter",
            bool((((ref - s_warm) / ref) <= ((ref - s) / ref) + 1e-6).all()),
            f"warm shortfall = {[f'{e:.2e}' for e in ((ref - s_warm) / ref).double().tolist()]}",
        )


class PeftLike(nn.Module):
    """Adapter shaped like a PEFT LoRA layer."""

    def __init__(self, r, d_in, d_out, dtype):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False, dtype=dtype)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False, dtype=dtype)})


class BridgeLike(nn.Module):
    """Adapter shaped like a Megatron-Bridge LoRA layer."""

    def __init__(self, r, d_in, d_out, dtype):
        super().__init__()
        self.linear_in = nn.Linear(d_in, r, bias=False, dtype=dtype)
        self.linear_out = nn.Linear(r, d_out, bias=False, dtype=dtype)


def build_models(shapes, dtype, device, seed=0):
    """Build PEFT- and Bridge-shaped models with identical weights."""
    torch.manual_seed(seed)
    peft = nn.ModuleList([PeftLike(r, d_in, d_out, dtype) for r, d_in, d_out in shapes]).to(device)
    bridge = nn.ModuleList([BridgeLike(r, d_in, d_out, dtype) for r, d_in, d_out in shapes]).to(device)
    with torch.no_grad():
        for p, b in zip(peft, bridge, strict=True):
            b.linear_in.weight.copy_(p.lora_A["default"].weight)
            b.linear_out.weight.copy_(p.lora_B["default"].weight)
    return peft, bridge


def make_grads(pairs, dtype, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for A, B in pairs:
        gA = (torch.randn(A.shape, generator=g) * 0.01).to(device=device, dtype=dtype)
        gB = (torch.randn(B.shape, generator=g) * 0.01).to(device=device, dtype=dtype)
        out.append((gA, gB))
    return out


def check_discovery(mine, up, device) -> None:
    print("\n== collect_lora_pairs discovery ==")
    shapes = [(8, 64, 32), (8, 64, 32), (4, 32, 96)]
    peft, bridge = build_models(shapes, torch.float32, device)

    up_pairs = up.collect_lora_pairs(peft)
    mine_peft = mine.collect_lora_pairs(peft)
    mine_bridge = mine.collect_lora_pairs(bridge)

    report_bool(
        "PEFT tree: miles finds the same pairs as upstream",
        len(mine_peft) == len(up_pairs) and all(a is c and b is d for (a, b), (c, d) in zip(mine_peft, up_pairs)),
        f"miles={len(mine_peft)} upstream={len(up_pairs)}",
    )
    report_bool(
        "Bridge tree (linear_in/linear_out): same count and shapes/order",
        [(a.shape, b.shape) for a, b in mine_bridge] == [(a.shape, b.shape) for a, b in up_pairs],
        f"{[(tuple(a.shape), tuple(b.shape)) for a, b in mine_bridge]}",
    )

    bridge[0].linear_in.weight.requires_grad_(False)
    peft[0].lora_A["default"].weight.requires_grad_(False)
    report_bool(
        "frozen pair skipped identically",
        len(mine.collect_lora_pairs(bridge)) == len(up.collect_lora_pairs(peft)) == len(shapes) - 1,
        f"miles={len(mine.collect_lora_pairs(bridge))} upstream={len(up.collect_lora_pairs(peft))}",
    )


def check_trajectory(mine, up, device, dtype, steps, grad_source, seed=0) -> None:
    """Run both optimizers on identical gradients.

    Under bf16 the two are *meant* to differ: upstream adds each update straight
    into the bf16 factor, while miles keeps fp32 masters, updates those, and casts
    the result back into the parameter. The reference for the miles masters is
    therefore upstream running the same recipe in fp32 from the same (bf16-rounded)
    initial weights and gradients -- which it must still match bit for bit.
    """
    label = f"{device}/{dtype_name(dtype)}/grads-via-{grad_source}"
    print(f"\n== optimizer trajectory: {label} ({steps} steps) ==")
    shapes = [(8, 64, 32), (8, 64, 32), (4, 32, 96), (16, 128, 128)]
    peft, bridge = build_models(shapes, dtype, device, seed=seed)
    masters = dtype is not torch.float32
    if masters:
        # Exact widening of the same weights: the runs still start from one point.
        peft.float()

    up_pairs = up.collect_lora_pairs(peft)
    mine_pairs = mine.collect_lora_pairs(bridge)

    kwargs = dict(lr=3e-4, beta1=0.9, curvature_beta=0.99, ns_steps=8, higham_iters=8)
    opt_up = up.Polora(pairs=up_pairs, **kwargs)
    opt_mine = mine.Polora(pairs=mine_pairs, **kwargs)

    def mine_weights(i, A_m):
        """The weights miles actually updates: masters once they exist."""
        st = opt_mine.state[A_m]
        A_w, B_w = mine_pairs[i]
        return (st["W_A"], st["W_B"]) if "W_A" in st else (A_w, B_w)

    init_w = [(A.detach().clone(), B.detach().clone()) for A, B in mine_pairs]
    worst = 0.0
    worst_round = 0.0
    for step in range(steps):
        grads = make_grads(up_pairs, dtype, device, seed=1000 + step)
        for (A_u, B_u), (A_m, B_m), (gA, gB) in zip(up_pairs, mine_pairs, grads, strict=True):
            A_u.grad, B_u.grad = gA.to(A_u.dtype), gB.to(B_u.dtype)
            if grad_source == "main_grad":
                A_m.main_grad, B_m.main_grad = gA.clone(), gB.clone()
                A_m.grad = B_m.grad = None
            else:
                A_m.grad, B_m.grad = gA.clone(), gB.clone()

        opt_up.step()
        opt_mine.step()

        for i, ((A_u, B_u), (A_m, B_m)) in enumerate(zip(up_pairs, mine_pairs, strict=True)):
            W_A, W_B = mine_weights(i, A_m)
            for nm, t_m, t_u in (("A", W_A, A_u), ("B", W_B, B_u)):
                worst = max(worst, rel_diff(t_m, t_u)[0])
                if step == steps - 1:
                    report(f"step {step + 1} pair{i} {nm} weight", t_m, t_u)
            if masters:
                # The parameter is the master rounded down to its own dtype.
                worst_round = max(worst_round, rel_diff(A_m, W_A.to(dtype))[0], rel_diff(B_m, W_B.to(dtype))[0])

        if step == steps - 1:
            for i, ((A_u, _), (A_m, _)) in enumerate(zip(up_pairs, mine_pairs, strict=True)):
                su, sm = opt_up.pair_state[i], opt_mine.state[A_m]
                for key in ("M_A", "M_B", "Q", "P"):
                    report(f"step {step + 1} pair{i} state[{key}]", sm[key], su[key])

    weights = "master weights" if masters else "weights"
    report_bool(f"trajectory {label}: worst {weights} deviation over all steps", worst == 0.0, f"max_abs={worst:.3e}")
    if masters:
        report_bool(
            f"trajectory {label}: params are exactly the masters cast to {dtype_name(dtype)}",
            worst_round == 0.0,
            f"max_abs={worst_round:.3e}",
        )

    moves = [max(rel_diff(A, A0)[0], rel_diff(B, B0)[0]) for (A, B), (A0, B0) in zip(mine_pairs, init_w, strict=True)]
    report_bool(
        "every pair actually moved (comparison is non-vacuous)",
        min(moves) > 0.0,
        f"per-pair |delta w| = {[f'{m:.2e}' for m in moves]}",
    )


def check_state_dict_roundtrip(mine, device) -> None:
    """Check that fp32 state survives a save and resume."""
    print("\n== miles state_dict/load_state_dict resume (bf16 params) ==")
    shapes = [(8, 64, 32), (4, 32, 96)]
    _, bridge_a = build_models(shapes, torch.bfloat16, device, seed=3)
    _, bridge_b = build_models(shapes, torch.bfloat16, device, seed=3)
    pairs_a = mine.collect_lora_pairs(bridge_a)
    pairs_b = mine.collect_lora_pairs(bridge_b)
    opt_a = mine.Polora(pairs=pairs_a, lr=3e-4)
    opt_b = mine.Polora(pairs=pairs_b, lr=3e-4)

    def run(pairs, opt, steps, offset=0):
        for s in range(steps):
            for (A, B), (gA, gB) in zip(
                pairs, make_grads(pairs, torch.bfloat16, device, 500 + offset + s), strict=True
            ):
                A.grad, B.grad = gA.clone(), gB.clone()
            opt.step()

    run(pairs_a, opt_a, 3)
    sd = copy.deepcopy(opt_a.state_dict())
    run(pairs_b, opt_b, 3)

    opt_c = mine.Polora(pairs=pairs_b, lr=3e-4)
    opt_c.load_state_dict(sd)
    dtypes = {
        k: v.dtype
        for st in opt_c.state.values()
        for k, v in st.items()
        if torch.is_tensor(v) and v.is_floating_point()
    }
    report_bool("restored state stays fp32 under bf16 params", set(dtypes.values()) == {torch.float32}, str(dtypes))

    run(pairs_a, opt_a, 2, offset=100)
    run(pairs_b, opt_c, 2, offset=100)
    for i, ((A_a, B_a), (A_b, B_b)) in enumerate(zip(pairs_a, pairs_b, strict=True)):
        report(f"post-resume pair{i} A", A_b, A_a)
        report(f"post-resume pair{i} B", B_b, B_a)


def dtype_name(dt):
    return str(dt).replace("torch.", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="/tmp/polora_upstream", type=Path)
    ap.add_argument("--steps", default=5, type=int)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from miles_plugins.optimizers.polora import kernels as mine_kernels
    from miles_plugins.optimizers.polora import optimizer as mine_optim

    up_optim, up_kernels = load_upstream(args.upstream)
    print(f"miles:    {Path(mine_optim.__file__).relative_to(REPO_ROOT)}")
    print(f"torch:    {torch.__version__}")

    devices = ["cpu"] + ([] if args.cpu_only or not torch.cuda.is_available() else ["cuda"])
    for dev in devices:
        check_kernels(mine_kernels, up_kernels, dev)
        check_kernels_vs_dense(mine_kernels, dev)

    check_discovery(mine_optim, up_optim, devices[-1])

    for dev in devices:
        for dtype in (torch.float32, torch.bfloat16):
            for src in ("grad", "main_grad"):
                check_trajectory(mine_optim, up_optim, dev, dtype, args.steps, src)

    check_state_dict_roundtrip(mine_optim, devices[-1])

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed: miles polora matches upstream numerically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
