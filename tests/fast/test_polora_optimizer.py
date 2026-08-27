"""Tests for the Polora optimizer's Miles integration."""

import pytest
import torch

from miles_plugins.optimizers.polora import Polora, collect_lora_pairs


def _make_pairs(shapes=((8, 32, 16),), dtype=torch.float32):
    """Build ``(A, B)`` parameters for ``(r, d_in, d_out)`` shapes."""
    pairs = []
    for r, d_in, d_out in shapes:
        A = torch.nn.Parameter(torch.randn(r, d_in, dtype=dtype) * 0.02)
        B = torch.nn.Parameter(torch.zeros(d_out, r, dtype=dtype))
        pairs.append((A, B))
    return pairs


def _set_grads(pairs, attr="grad", seed=0):
    generator = torch.Generator().manual_seed(seed)
    for A, B in pairs:
        setattr(A, attr, torch.randn(A.shape, generator=generator, dtype=torch.float32))
        setattr(B, attr, torch.randn(B.shape, generator=generator, dtype=torch.float32))


class TestGradientSource:
    def test_reads_megatron_main_grad_when_grad_is_none(self):
        torch.manual_seed(0)
        pairs = _make_pairs()
        _set_grads(pairs, attr="main_grad")
        before = [A.detach().clone() for A, _ in pairs]

        Polora(pairs=pairs, lr=1e-2).step()

        assert pairs[0][0].grad is None
        assert not torch.equal(pairs[0][0].detach(), before[0])

    def test_step_does_not_zero_gradients(self):
        torch.manual_seed(0)
        pairs = _make_pairs()
        _set_grads(pairs, attr="main_grad")
        grad_before = pairs[0][0].main_grad.clone()

        Polora(pairs=pairs, lr=1e-2).step()

        assert torch.equal(pairs[0][0].main_grad, grad_before)

    def test_missing_gradients_raise(self):
        pairs = _make_pairs()
        with pytest.raises(ValueError, match="Gradients are required"):
            Polora(pairs=pairs, lr=1e-2).step()


class TestStatePrecision:
    def test_state_is_fp32_under_bf16_params(self):
        torch.manual_seed(0)
        pairs = _make_pairs(dtype=torch.bfloat16)
        _set_grads(pairs, attr="main_grad")
        optimizer = Polora(pairs=pairs, lr=1e-2)
        optimizer.step()

        state = optimizer.state[pairs[0][0]]
        assert pairs[0][0].dtype is torch.bfloat16
        for key in ("M_A", "M_B", "Q", "P"):
            assert state[key].dtype is torch.float32, key

    def test_state_dict_round_trip_keeps_fp32(self):
        torch.manual_seed(0)
        pairs = _make_pairs(dtype=torch.bfloat16)
        _set_grads(pairs, attr="main_grad")
        source = Polora(pairs=pairs, lr=1e-2)
        source.step()
        saved = source.state_dict()

        restored = Polora(pairs=pairs, lr=1e-2)
        restored.load_state_dict(saved)

        original_state = source.state[pairs[0][0]]
        restored_state = restored.state[pairs[0][0]]
        for key in ("M_A", "M_B", "Q", "P"):
            assert restored_state[key].dtype is torch.float32, key
            torch.testing.assert_close(restored_state[key], original_state[key])


class TestUpdate:
    def test_descends_on_a_factored_least_squares_problem(self):
        torch.manual_seed(0)
        r, d_in, d_out = 8, 32, 16
        pairs = _make_pairs(((r, d_in, d_out),))
        A, B = pairs[0]
        target = torch.randn(d_out, d_in) * 0.1
        optimizer = Polora(pairs=pairs, lr=1e-2)

        def loss_fn():
            return ((B @ A) - target).pow(2).sum()

        initial = loss_fn().item()
        for _ in range(20):
            loss = loss_fn()
            A.grad, B.grad = torch.autograd.grad(loss, [A, B])
            optimizer.step()
        final = loss_fn().item()

        assert final < initial

    def test_shape_groups_batch_independently(self):
        torch.manual_seed(0)
        pairs = _make_pairs(((8, 32, 16), (4, 16, 8), (8, 32, 16)))
        _set_grads(pairs, attr="main_grad")
        before = [(A.detach().clone(), B.detach().clone()) for A, B in pairs]

        Polora(pairs=pairs, lr=1e-2).step()

        for (A, B), (A0, B0) in zip(pairs, before, strict=True):
            assert not torch.equal(A.detach(), A0)
            assert not torch.equal(B.detach(), B0)

    def test_rejects_empty_pairs(self):
        with pytest.raises(ValueError, match="No trainable LoRA"):
            Polora(pairs=[])


class _PeftLoraLinear(torch.nn.Module):
    def __init__(self, r, d_in, d_out, adapters=("default",)):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(d_out, d_in))
        self.lora_A = torch.nn.ModuleDict({a: torch.nn.Linear(d_in, r, bias=False) for a in adapters})
        self.lora_B = torch.nn.ModuleDict({a: torch.nn.Linear(r, d_out, bias=False) for a in adapters})


class _Adapter(torch.nn.Module):
    def __init__(self, r, d_in, d_out):
        super().__init__()
        self.linear_in = torch.nn.Linear(d_in, r, bias=False)
        self.linear_out = torch.nn.Linear(r, d_out, bias=False)


class _AdaptedLinear(torch.nn.Module):
    def __init__(self, r, d_in, d_out):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(d_out, d_in))
        self.adapter = _Adapter(r, d_in, d_out)


class TestCollectLoraPairsPeft:
    def _model(self, adapters=("default",)):
        model = torch.nn.Module()
        model.layer0 = _PeftLoraLinear(8, 32, 16, adapters=adapters)
        model.layer1 = _PeftLoraLinear(4, 16, 8, adapters=adapters)
        model.plain = torch.nn.Linear(4, 4)
        return model

    def test_finds_every_adapter_pair_in_order(self):
        pairs = collect_lora_pairs(self._model())

        assert [(A.shape, B.shape) for A, B in pairs] == [
            (torch.Size([8, 32]), torch.Size([16, 8])),
            (torch.Size([4, 16]), torch.Size([8, 4])),
        ]

    def test_collects_each_named_adapter_in_the_module_dict(self):
        pairs = collect_lora_pairs(self._model(adapters=("default", "reference")))

        assert len(pairs) == 4

    def test_skips_frozen_adapters(self):
        model = self._model(adapters=("default", "reference"))
        model.layer0.lora_A["reference"].weight.requires_grad_(False)
        model.layer0.lora_B["reference"].weight.requires_grad_(False)

        pairs = collect_lora_pairs(model)

        assert len(pairs) == 3

    def test_optimizes_a_peft_shaped_model_end_to_end(self):
        torch.manual_seed(0)
        model = self._model()
        optimizer = Polora(model=model, lr=1e-2)
        for A, B in optimizer.pairs:
            A.main_grad = torch.randn(A.shape)
            B.main_grad = torch.randn(B.shape)
        before = optimizer.pairs[0][0].detach().clone()

        optimizer.step()

        assert len(optimizer.pairs) == 2
        assert not torch.equal(optimizer.pairs[0][0].detach(), before)


class TestCollectLoraPairsMegatron:
    def _model(self):
        model = torch.nn.Module()
        model.layer0 = _AdaptedLinear(8, 32, 16)
        model.layer1 = _AdaptedLinear(4, 16, 8)
        model.plain = torch.nn.Linear(4, 4)
        return model

    def test_finds_every_adapter_pair_in_order(self):
        model = self._model()
        pairs = collect_lora_pairs(model)

        assert [(A.shape, B.shape) for A, B in pairs] == [
            (torch.Size([8, 32]), torch.Size([16, 8])),
            (torch.Size([4, 16]), torch.Size([8, 4])),
        ]

    def test_accepts_a_list_of_model_chunks(self):
        chunks = [self._model(), self._model()]
        assert len(collect_lora_pairs(chunks)) == 4

    def test_skips_frozen_adapters(self):
        model = self._model()
        model.layer0.adapter.linear_in.weight.requires_grad_(False)

        pairs = collect_lora_pairs(model)

        assert len(pairs) == 1
        assert pairs[0][0].shape == torch.Size([4, 16])


class TestMegatronAdapter:
    def _optimizer(self):
        pytest.importorskip("megatron.core.optimizer.optimizer")
        from megatron.core.optimizer import OptimizerConfig

        from miles_plugins.optimizers.polora.megatron_adapter import PoloraMegatronOptimizer

        return PoloraMegatronOptimizer(Polora(pairs=_make_pairs()), OptimizerConfig())

    def test_does_not_shadow_the_grad_stats_parallel_group(self):
        """An explicit None would reduce the gradient norm over WORLD."""
        assert not hasattr(self._optimizer(), "grad_stats_parallel_group")

    def test_loss_scale_is_one(self):
        assert self._optimizer().get_loss_scale().item() == 1.0
