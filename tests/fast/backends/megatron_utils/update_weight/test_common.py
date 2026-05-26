"""Tests for helpers in miles/backends/megatron_utils/update_weight/common.py.

Covers:
- _check_weight_sync_results: engine-result shape handling and failure
  detection for both base and LoRA weight sync.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from dataclasses import dataclass

import pytest

from miles.backends.megatron_utils.update_weight.common import _check_weight_sync_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeEngineResult:
    """Mimics sglang's LoRAUpdateOutput / weight-sync result."""

    success: bool
    error_message: str | None = None


# ---------------------------------------------------------------------------
# _check_weight_sync_results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_lora", [pytest.param(False, id="base"), pytest.param(True, id="lora")])
class TestCheckWeightSyncResults:
    """Validate that _check_weight_sync_results raises on engine failures.

    Parametrized for base + LoRA. Test IDs include [base]/[lora] so any
    failure pinpoints which mode regressed.
    """

    def test_success_dataclass_result_passes(self, is_lora):
        results = [_FakeEngineResult(success=True)]
        _check_weight_sync_results(results, is_lora=is_lora)

    def test_failure_dataclass_result_raises(self, is_lora):
        results = [_FakeEngineResult(success=False, error_message="incompatible format")]
        with pytest.raises(RuntimeError, match="weight sync failed"):
            _check_weight_sync_results(results, is_lora=is_lora)

    def test_plain_tuple_result_passes(self, is_lora):
        """Non-dataclass results (e.g. (True, 'Success') tuples) should not raise."""
        results = [(True, "Success")]
        _check_weight_sync_results(results, is_lora=is_lora)

    def test_mixed_dataclass_results_raises_on_first_failure(self, is_lora):
        results = [
            _FakeEngineResult(success=True),
            _FakeEngineResult(success=False, error_message="oops"),
        ]
        with pytest.raises(RuntimeError):
            _check_weight_sync_results(results, is_lora=is_lora)

    def test_dict_result_with_success_true_passes(self, is_lora):
        """Engine results shaped as Mapping (dict) with success=True must not raise."""
        results = [{"success": True}]
        _check_weight_sync_results(results, is_lora=is_lora)

    def test_dict_result_with_success_false_raises(self, is_lora):
        results = [{"success": False, "error_message": "checkpoint mismatch"}]
        with pytest.raises(RuntimeError, match="weight sync failed"):
            _check_weight_sync_results(results, is_lora=is_lora)

    def test_dict_result_uses_error_key_when_no_error_message(self, is_lora):
        """Engines that report failure with 'error' instead of 'error_message'
        should still produce a useful error string."""
        results = [{"success": False, "error": "fallback message"}]
        with pytest.raises(RuntimeError):
            _check_weight_sync_results(results, is_lora=is_lora)
