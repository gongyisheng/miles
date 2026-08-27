#!/usr/bin/env python3
"""Check that Miles' Polora optimizer actually converges.

Fits a frozen two-layer MLP whose teacher differs from the student by an exactly
rank-``r`` delta per layer, so the reachable optimum is 0 and the loss is a
meaningful progress signal.

The learning rate is decayed to zero because polora's step has a fixed spectral
length -- it discards gradient magnitude, so at constant lr it limit-cycles at an
amplitude set by lr instead of settling. The second section measures that floor
directly by holding lr constant from an already-converged solution.

For the same reason the budget is a path length: a run covers roughly
``lr * steps / 2`` of spectral distance under cosine decay, and it converges only
if that reaches the optimum. Lower ``--lr`` and ``--steps`` has to go up.

Usage: ``python scripts/polora_converge.py [--steps N ...] [--device cpu|cuda]``
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

FAILURES: list[str] = []


def report_bool(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


def tail_level(history, frac=0.5):
    """Geometric mean of the last ``frac`` of the losses.

    A constant-lr run ends inside a limit cycle, so its final loss is one draw from
    that cycle and swings by an order of magnitude between adjacent steps. The
    geometric mean over a window is the stable readout of the cycle's amplitude.
    """
    tail = history[int((1 - frac) * len(history)) :]
    return math.exp(statistics.fmean(math.log(max(x, 1e-300)) for x in tail))


def cosine(lr):
    """Cosine decay from ``lr`` to 0 over the length of the run."""
    return lambda i, steps: lr * 0.5 * (1 + math.cos(math.pi * i / steps))


def constant(lr):
    return lambda i, steps: lr


class LoRALinear(nn.Module):
    """Frozen base weight plus a rank-r adapter, named so collect_lora_pairs finds it."""

    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Linear(d_in, r, bias=False)  # (r, d_in)
        self.lora_B = nn.Linear(r, d_out, bias=False)  # (d_out, r)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


class Fit:
    """A rank-r LoRA fit of a frozen MLP, whose optimum the adapter can reach exactly."""

    def __init__(self, make_opt, device, d=32, r=4, samples=256, seed=1):
        torch.manual_seed(seed)
        self.net = nn.Sequential(LoRALinear(d, d, r), nn.Tanh(), LoRALinear(d, d, r)).to(device)
        teacher = nn.Sequential(nn.Linear(d, d, bias=False), nn.Tanh(), nn.Linear(d, d, bias=False)).to(device)
        with torch.no_grad():
            for t, s in ((teacher[0], self.net[0]), (teacher[2], self.net[2])):
                delta = torch.randn(d, r, device=device) @ torch.randn(r, d, device=device) / d
                t.weight.copy_(s.base.weight + delta)
            self.x = torch.randn(samples, d, device=device)
            self.y = teacher(self.x)
        self.opt = make_opt(self.net)

    def run(self, steps, lr_fn):
        """Train for ``steps`` more steps and return the per-step loss history."""
        history = []
        for i in range(steps):
            self.opt.param_groups[0]["lr"] = lr_fn(i, steps)
            self.opt.zero_grad()
            loss = F.mse_loss(self.net(self.x), self.y)
            loss.backward()
            self.opt.step()
            history.append(loss.item())
        return history


def check_convergence(polora, adamw, make_fit, step_counts, lr, adamw_lr, ratio):
    """With lr decayed to zero the loss should collapse toward the rank-r optimum.

    Returns ``(fit, loss)`` from the longest budget, for the floor probe to reuse.
    """
    print("\n== convergence, lr -> 0 ==")

    # each optimizer gets its own lr: polora's is a spectral distance per step,
    # adamw's a per-coordinate size, so the two scales are not comparable
    finals, first, fit = [], None, None
    for steps in step_counts:
        fit = make_fit(polora)
        hist = fit.run(steps, cosine(lr))
        first, final = hist[0], hist[-1]  # same seed, so `first` matches across budgets
        adamw_loss = make_fit(adamw).run(steps, cosine(adamw_lr))[-1]
        finals.append(final)
        print(
            f"         {steps:>5} steps: initial {first:.3e} -> polora {final:.3e} ({first / final:.3g}x)"
            f", adamw(lr={adamw_lr:g}) {adamw_loss:.3e} ({first / adamw_loss:.3g}x)"
        )

    report_bool(
        f"{step_counts[-1]} steps drop the loss by >{ratio:g}x",
        finals[-1] < first / ratio,
        f"{first:.3e} -> {finals[-1]:.3e} = {first / finals[-1]:.3g}x",
    )
    report_bool(
        "a longer budget lands lower (progress is not an early plateau)",
        all(a > b for a, b in zip(finals, finals[1:])),
        f"steps={step_counts} finals={[f'{x:.3e}' for x in finals]}",
    )
    return fit, finals[-1]


def check_constant_lr_floor(fit, converged, lrs, steps) -> None:
    """A constant lr cannot settle: the step has a fixed spectral length.

    Probing outward from the converged solution reads the limit cycle in a few tens
    of steps -- the iterate is already at the optimum, so the only thing left to
    measure is how far each step throws it back out.
    """
    print(f"\n== constant-lr floor, probed from the converged solution ({steps} steps each) ==")
    floors = []
    for lr in lrs:
        floors.append(tail_level(fit.run(steps, constant(lr))))
        print(f"         constant lr={lr:<8g} floor {floors[-1]:.3e} ({floors[-1] / converged:.2g}x the converged loss)")

    report_bool(
        "every constant lr is thrown back off the optimum",
        all(f > converged * 10 for f in floors),
        f"converged {converged:.3e} vs floors {[f'{f:.3e}' for f in floors]}",
    )
    report_bool(
        "the floor rises with lr (step length, not gradient, sets it)",
        all(a < b for a, b in zip(floors, floors[1:])),
        f"lrs={list(lrs)} floors={[f'{f:.3e}' for f in floors]}",
    )
    # A fixed step length around a locally quadratic optimum parks the iterate at a
    # weight-space radius proportional to lr, so the loss floor should scale as lr^2.
    slope, _ = fit_loglog(lrs, floors)
    report_bool(
        "the floor scales as lr^2 (fixed step length around a quadratic optimum)",
        1.5 <= slope <= 2.5,
        f"fitted exponent {slope:.2f}, expected ~2",
    )


def fit_loglog(xs, ys):
    """Least-squares slope and intercept of ``log y`` against ``log x``."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    var = sum((a - mx) ** 2 for a in lx)
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True)) / var
    return slope, my - slope * mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=[100, 800], type=int, nargs="+", help="step counts for the decayed runs")
    ap.add_argument("--lr", default=0.2, type=float, help="polora lr (a spectral distance per step)")
    ap.add_argument("--adamw-lr", default=0.01, type=float, help="baseline adamw lr")
    ap.add_argument("--ratio", default=1e5, type=float, help="required loss reduction for the longest run")
    ap.add_argument("--floor-lrs", default=[0.003, 0.01, 0.03, 0.1], type=float, nargs="+")
    ap.add_argument("--floor-steps", default=60, type=int, help="steps per constant-lr probe")
    ap.add_argument("--dim", default=32, type=int, help="hidden width of the toy MLP")
    ap.add_argument("--rank", default=4, type=int, help="LoRA rank, and the rank of the teacher delta")
    ap.add_argument("--seed", default=1, type=int, help="seed for the base weights, teacher delta and data")
    ap.add_argument("--threads", default=8, type=int, help="cap on torch cpu threads")
    ap.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"), help="auto runs cpu, then cuda when visible"
    )
    ap.add_argument("--cpu-only", action="store_true", help="alias for --device cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from miles_plugins.optimizers.polora.optimizer import Polora

    # the matrices here are tiny and the step is launch-bound; spreading it over every
    # core of a big host costs far more in synchronization than it saves
    torch.set_num_threads(max(1, min(args.threads, torch.get_num_threads())))

    device = "cpu" if args.cpu_only else args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device visible")
        return 1
    devices = [device] if device != "auto" else ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    print(f"miles:    {Path(sys.modules[Polora.__module__].__file__).relative_to(REPO_ROOT)}")
    print(f"torch:    {torch.__version__} ({torch.get_num_threads()} cpu threads)")
    print(f"model:    2-layer MLP, width {args.dim}, rank-{args.rank} adapter fitting a rank-{args.rank} teacher delta")
    print(f"devices:  {', '.join(devices)}")

    def polora(net):
        return Polora(model=net, lr=args.lr)

    def adamw(net):
        return torch.optim.AdamW(
            [p for p in net.parameters() if p.requires_grad], lr=args.adamw_lr, weight_decay=0.0
        )

    for dev in devices:
        print(f"\n{'-' * 78}\ndevice: {dev}")

        def make_fit(make_opt, dev=dev):
            return Fit(make_opt, dev, d=args.dim, r=args.rank, seed=args.seed)

        fit, converged = check_convergence(
            polora, adamw, make_fit, sorted(args.steps), args.lr, args.adamw_lr, args.ratio
        )
        check_constant_lr_floor(fit, converged, sorted(args.floor_lrs), args.floor_steps)

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed: miles polora converges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
