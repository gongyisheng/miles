"""Unit tests for --optimizer polora argument validation."""

from argparse import Namespace

import pytest

from miles.utils.arguments import _validate_polora_args


def _args(**overrides) -> Namespace:
    base = dict(
        optimizer="polora",
        lora_rank=32,
        multi_lora=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        fp16=False,
        reset_optimizer_states=False,
        optimizer_cpu_offload=False,
        use_precision_aware_optimizer=False,
    )
    base.update(overrides)
    return Namespace(**base)


class TestPoloraValidation:
    def test_valid_configuration_passes(self):
        _validate_polora_args(_args())

    @pytest.mark.parametrize("optimizer", ["adam", "muon", None])
    def test_other_optimizers_are_untouched(self, optimizer):
        """Constraints must not leak onto runs that never asked for polora."""
        _validate_polora_args(_args(optimizer=optimizer, lora_rank=0, tensor_model_parallel_size=8))

    def test_requires_lora(self):
        with pytest.raises(AssertionError, match="lora-rank"):
            _validate_polora_args(_args(lora_rank=0))

    def test_rejects_multi_lora(self):
        with pytest.raises(AssertionError, match="multi-LoRA"):
            _validate_polora_args(_args(multi_lora=True))

    def test_rejects_tensor_parallelism(self):
        with pytest.raises(AssertionError, match="tensor-model-parallel-size 1"):
            _validate_polora_args(_args(tensor_model_parallel_size=2))

    def test_rejects_pipeline_parallelism(self):
        with pytest.raises(AssertionError, match="pipeline-model-parallel-size 1"):
            _validate_polora_args(_args(pipeline_model_parallel_size=2))

    def test_rejects_fp16(self):
        with pytest.raises(AssertionError, match="bf16"):
            _validate_polora_args(_args(fp16=True))

    def test_rejects_reset_optimizer_states(self):
        with pytest.raises(AssertionError, match="chained_optimizers"):
            _validate_polora_args(_args(reset_optimizer_states=True))

    def test_rejects_optimizer_cpu_offload(self):
        with pytest.raises(AssertionError, match="optimizer-cpu-offload"):
            _validate_polora_args(_args(optimizer_cpu_offload=True))

    def test_rejects_precision_aware_optimizer(self):
        with pytest.raises(AssertionError, match="precision-aware-optimizer"):
            _validate_polora_args(_args(use_precision_aware_optimizer=True))
