"""Adapt :class:`Polora` to Megatron's optimizer interface.

The module's LoRA factors remain bf16; the optimizer holds fp32 master copies
of them plus fp32 state, and gradients are not clipped because Polora rescales
each update by its spectral norm.
"""

from __future__ import annotations

import logging

import torch
from megatron.core.optimizer.optimizer import MegatronOptimizer

logger = logging.getLogger(__name__)


class PoloraMegatronOptimizer(MegatronOptimizer):
    """Wrap :class:`Polora` with Megatron's optimizer interface.

    Args:
        optimizer: The wrapped ``Polora`` instance.
        config: Megatron ``OptimizerConfig``.
        init_state_fn: Unused interface argument.
    """

    def __init__(self, optimizer, config, init_state_fn=lambda x: None):
        super().__init__(optimizer, config, init_state_fn)
        self.is_stub_optimizer = False
        # Leave grad_stats_parallel_group unset so the base class uses the
        # model-parallel group. Setting it to None would reduce over WORLD.
        self.grad_norms_by_group = {}
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self._scale_one = torch.ones(1, dtype=torch.float32, device=device)

    def get_loss_scale(self):
        """Return the fixed bf16 loss scale."""
        return self._scale_one

    def prepare_grads(self) -> bool:
        """Expose ``main_grad`` to Megatron's gradient statistics helpers."""
        for param in self.get_parameters():
            # Under --accumulate-allreduce-grads-in-fp32 main_grad is fp32 while
            # the LoRA factors stay bf16, and torch rejects a .grad whose dtype
            # differs from its parameter's unless grad_dtype opts out. Opt out
            # rather than casting: get_grad_norm/count_zeros read .grad, and
            # narrowing the fp32 buffer to bf16 just to report on it would
            # throw away the precision the flag was set to keep.
            if getattr(param, "grad_dtype", None) is not None:
                param.grad_dtype = None
            param.grad = param.main_grad
        return False

    def step_with_ready_grads(self) -> bool:
        self.optimizer.step()
        return True

    @torch.no_grad()
    def step(self):
        """Run one optimizer step.

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
        """Drop ``.grad`` aliases; DDP owns the ``main_grad`` buffers."""
        for param in self.get_parameters():
            param.grad = None

    def reload_model_params(self, state_dict=None):
        """Re-seed the fp32 masters after the model params were loaded into.

        Megatron calls this once the base checkpoint has been read into the live
        model. Polora's masters are the weights it updates, so a load that
        rewrites the factors has to be mirrored into them; masters not allocated
        yet (the usual case here, since nothing has stepped) seed from the params
        on the first step.
        """
        self.optimizer.sync_masters_from_params()

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
    """Build a Megatron-compatible Polora optimizer."""
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
