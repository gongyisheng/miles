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
        for key in ("M_A", "M_B", "Q", "P", "W_A", "W_B"):
            assert restored_state[key].dtype is torch.float32, key
            torch.testing.assert_close(restored_state[key], original_state[key])


class TestMasterWeights:
    def test_masters_are_fp32_and_hold_more_than_the_bf16_params(self):
        torch.manual_seed(0)
        pairs = _make_pairs(dtype=torch.bfloat16)
        _set_grads(pairs, attr="main_grad")
        optimizer = Polora(pairs=pairs, lr=1e-2)
        optimizer.step()

        A, B = pairs[0]
        state = optimizer.state[A]
        assert state["W_A"].dtype is torch.float32
        assert state["W_B"].dtype is torch.float32
        # The parameter is the master rounded to bf16, and rounding lost something.
        torch.testing.assert_close(A.detach(), state["W_A"].bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(B.detach(), state["W_B"].bfloat16(), rtol=0, atol=0)
        assert not torch.equal(state["W_A"], state["W_A"].bfloat16().float())

    def test_bf16_params_follow_the_same_trajectory_as_fp32_params(self):
        """The update is applied in fp32, so the parameter dtype must not steer it."""
        torch.manual_seed(0)
        bf16_pairs = _make_pairs(dtype=torch.bfloat16)
        # Seed the fp32 run from the *rounded* weights so the only remaining
        # difference between the runs would be bf16 arithmetic in the update.
        fp32_pairs = [
            (torch.nn.Parameter(A.detach().float()), torch.nn.Parameter(B.detach().float()))
            for A, B in bf16_pairs
        ]
        bf16_opt = Polora(pairs=bf16_pairs, lr=1e-2)
        fp32_opt = Polora(pairs=fp32_pairs, lr=1e-2)

        for step in range(5):
            _set_grads(bf16_pairs, attr="main_grad", seed=step)
            _set_grads(fp32_pairs, attr="main_grad", seed=step)
            bf16_opt.step()
            fp32_opt.step()

        for (A, B), (A32, B32) in zip(bf16_pairs, fp32_pairs, strict=True):
            torch.testing.assert_close(bf16_opt.state[A]["W_A"], A32.detach(), rtol=0, atol=0)
            torch.testing.assert_close(bf16_opt.state[A]["W_B"], B32.detach(), rtol=0, atol=0)

    def test_sub_ulp_updates_accumulate_into_the_bf16_params(self):
        """Steps smaller than a bf16 ULP must not round away step after step."""
        torch.manual_seed(0)
        pairs = _make_pairs(dtype=torch.bfloat16)
        A = pairs[0][0]
        before = A.detach().clone()
        # lr is small enough that one step moves most of A by less than a bf16 ULP.
        optimizer = Polora(pairs=pairs, lr=1e-5)

        _set_grads(pairs, attr="main_grad", seed=0)
        optimizer.step()
        unchanged = (A.detach() == before).float().mean()
        assert unchanged > 0.75, "the single-step update should be sub-ULP for most entries"
        assert not torch.equal(optimizer.state[A]["W_A"], before.float()), "the master must still move"

        # Repeating the same sub-ULP step accumulates in the master and eventually
        # carries the bf16 parameter with it; without a master it would never move.
        for _ in range(200):
            _set_grads(pairs, attr="main_grad", seed=0)
            optimizer.step()
        assert (A.detach() == before).float().mean() < 0.1

    def test_legacy_state_without_masters_seeds_them_from_the_params(self):
        torch.manual_seed(0)
        pairs = _make_pairs(dtype=torch.bfloat16)
        _set_grads(pairs, attr="main_grad")
        source = Polora(pairs=pairs, lr=1e-2)
        source.step()
        saved = source.state_dict()
        for entry in saved["state"].values():
            entry.pop("W_A", None)
            entry.pop("W_B", None)

        restored = Polora(pairs=pairs, lr=1e-2)
        restored.load_state_dict(saved)
        A, B = pairs[0]
        assert "W_A" not in restored.state[A]

        _set_grads(pairs, attr="main_grad")
        restored.step()
        state = restored.state[A]
        assert state["W_A"].dtype is torch.float32
        # Momentum survived the load rather than being reset alongside the masters.
        torch.testing.assert_close(state["M_A"], source.state[A]["M_A"], rtol=1e-3, atol=0)

    def test_sync_masters_from_params_picks_up_an_external_write(self):
        torch.manual_seed(0)
        pairs = _make_pairs(dtype=torch.bfloat16)
        _set_grads(pairs, attr="main_grad")
        optimizer = Polora(pairs=pairs, lr=1e-2)
        optimizer.step()

        A, B = pairs[0]
        with torch.no_grad():
            A.copy_(torch.full_like(A, 0.125))
            B.copy_(torch.full_like(B, 0.25))
        optimizer.sync_masters_from_params()

        assert torch.equal(optimizer.state[A]["W_A"], A.detach().float())
        assert torch.equal(optimizer.state[A]["W_B"], B.detach().float())

    def test_sync_masters_is_a_noop_before_the_first_step(self):
        pairs = _make_pairs(dtype=torch.bfloat16)
        optimizer = Polora(pairs=pairs, lr=1e-2)

        optimizer.sync_masters_from_params()

        assert not optimizer.state[pairs[0][0]]


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
    def _optimizer(self, dtype=torch.float32):
        pytest.importorskip("megatron.core.optimizer.optimizer")
        from megatron.core.optimizer import OptimizerConfig

        from miles_plugins.optimizers.polora.megatron_adapter import PoloraMegatronOptimizer

        return PoloraMegatronOptimizer(Polora(pairs=_make_pairs(dtype=dtype)), OptimizerConfig())

    def test_does_not_shadow_the_grad_stats_parallel_group(self):
        """An explicit None would reduce the gradient norm over WORLD."""
        assert not hasattr(self._optimizer(), "grad_stats_parallel_group")

    def test_loss_scale_is_one(self):
        assert self._optimizer().get_loss_scale().item() == 1.0

    def test_prepare_grads_aliases_an_fp32_main_grad_onto_a_bf16_param(self):
        """--accumulate-allreduce-grads-in-fp32 pairs bf16 factors with fp32 main_grad.

        torch rejects a .grad whose dtype differs from its parameter's unless
        grad_dtype opts out, and Megatron's grad-norm helpers read .grad.
        """
        optimizer = self._optimizer(dtype=torch.bfloat16)
        params = optimizer.get_parameters()
        for param in params:
            param.main_grad = torch.ones(param.shape, dtype=torch.float32)

        assert optimizer.prepare_grads() is False

        for param in params:
            assert param.dtype is torch.bfloat16
            assert param.grad is param.main_grad
            assert param.grad.dtype is torch.float32
