"""Megatron facade for :class:`Polora`.

``Polora`` is a plain ``torch.optim.Optimizer``, but miles' training loop drives
Megatron's optimizer contract: ``step()`` returns
``(update_successful, grad_norm, num_zeros_in_grad)``, gradients arrive as DDP
``main_grad`` bucket views, and ``OptimizerParamScheduler`` drives ``param_groups``.
This wrapper supplies that contract and nothing else.

No fp32 master weights: LoRA factors stay bf16 and the optimizer's own state is
fp32, matching upstream. That is why this subclasses ``MegatronOptimizer``
directly instead of reusing ``Float16OptimizerWithFloat16Params``, whose entire
job is master creation, grad copy, and copy-back.

No gradient clipping: the update is already rescaled to spectral norm
``rho = lr / (sigma_max(A) + sigma_max(B))``, so clipping the gradient would not
change the update magnitude -- it would only perturb the ``Q``/``P``
preconditioner EMAs, which accumulate the raw gradient. ``grad_norm`` is still
computed and returned so logging and CI's ``check_grad_norm`` keep working.
"""

from __future__ import annotations

import logging

import torch
from megatron.core.optimizer.optimizer import MegatronOptimizer

logger = logging.getLogger(__name__)


class PoloraMegatronOptimizer(MegatronOptimizer):
    """Adapts :class:`Polora` to Megatron's optimizer interface.

    Args:
        optimizer: The wrapped ``Polora`` instance.
        config: Megatron ``OptimizerConfig``.
        init_state_fn: Retained for interface compatibility; unused.
    """

    def __init__(self, optimizer, config, init_state_fn=lambda x: None):
        super().__init__(optimizer, config, init_state_fn)
        self.is_stub_optimizer = False
        # Deliberately leave `grad_stats_parallel_group` unset: the base class
        # returns it verbatim when present, and `all_reduce(group=None)` reduces
        # over WORLD, not over nothing -- which would sum the same DDP-averaged
        # gradient once per DP rank and inflate the reported grad norm by
        # sqrt(world_size). Unset, the base falls back to the model-parallel
        # group, matching what Megatron's own factory assigns to the
        # non-distributed optimizers; TP/PP are both 1 here, so that group has
        # size 1 and the reduction is the no-op it should be.
        self.grad_norms_by_group = {}
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self._scale_one = torch.ones(1, dtype=torch.float32, device=device)

    def get_loss_scale(self):
        """Always 1.0: bf16 training uses no dynamic loss scaler."""
        return self._scale_one

    def prepare_grads(self) -> bool:
        """Publishes DDP's ``main_grad`` buckets as ``.grad``.

        Mirrors ``FP32Optimizer.prepare_grads``. The base class's grad-norm and
        zero-count helpers read ``param.grad``, so they need this aliasing even
        though ``Polora`` reads ``main_grad`` directly. Returns False: with no
        loss scaler there is no overflow to detect.
        """
        for param in self.get_parameters():
            param.grad = param.main_grad
        return False

    def step_with_ready_grads(self) -> bool:
        self.optimizer.step()
        return True

    @torch.no_grad()
    def step(self):
        """Runs one optimizer step.

        Returns:
            ``(update_successful, grad_norm, num_zeros_in_grad)``. ``grad_norm``
            is reported for logging only -- no clipping is applied.
        """
        self.grad_norms_by_group = {}
        if self.prepare_grads():
            return False, None, None
        grad_norm = self.get_grad_norm()
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        return self.step_with_ready_grads(), grad_norm, num_zeros_in_grad

    def zero_grad(self, set_to_none: bool = True):
        """Drops the ``.grad`` aliases; DDP owns the ``main_grad`` buffers and
        clears them via ``model.zero_grad_buffer()``."""
        for param in self.get_parameters():
            param.grad = None

    def reload_model_params(self, state_dict=None):
        """No-op: there are no master params to re-seed from the model."""

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, metadata=None):
        raise NotImplementedError(
            "polora does not implement Megatron distributed checkpointing. LoRA runs save "
            "through save_checkpoint_with_lora(), which uses optimizer.state_dict(); reaching "
            "here means a non-LoRA checkpoint path was taken, which polora does not support."
        )


def build_polora_optimizer(args, config, model):
    """Builds the polora optimizer for a LoRA-adapted model.

    Args:
        args: Training arguments namespace.
        config: Megatron ``OptimizerConfig``.
        model: DDP-wrapped model chunks carrying LoRA adapters.

    Returns:
        A :class:`PoloraMegatronOptimizer` ready for the training loop.
    """
    from .optimizer import Polora

    assert not config.use_distributed_optimizer, (
        "polora requires use_distributed_optimizer=False: the update is coupled across "
        "each (A, B) pair and needs whole factor matrices, which the distributed "
        "optimizer's flat sharded buckets do not preserve"
    )
    assert not config.fp16, "polora requires bf16 (no dynamic loss scaler)"

    optimizer = Polora(
        model=model,
        lr=config.lr,
        beta1=args.polora_beta1,
        curvature_beta=args.polora_curvature_beta,
        ns_steps=args.polora_ns_steps,
        higham_iters=args.polora_higham_iters,
        compile=args.polora_compile,
    )
    logger.info(
        "Built polora optimizer over %d LoRA (A, B) pairs (beta1=%s, curvature_beta=%s, "
        "ns_steps=%d, higham_iters=%d, compile=%s)",
        len(optimizer.pairs),
        args.polora_beta1,
        args.polora_curvature_beta,
        args.polora_ns_steps,
        args.polora_higham_iters,
        args.polora_compile,
    )
    return PoloraMegatronOptimizer(optimizer, config)
