"""Tests for the colocate-mode weight-sync logic in update_weight_from_tensor.py.

Covers:
- _send_lora_params direct calls (success kwargs, no-LoRA raise, lora_loaded flag)
- _send_base_params direct calls (success kwargs, empty chunk)
- _send_to_colocated_engine (placeholder rank, base/lora branches, lora_loaded toggle,
  rank-0 RPC ordering)
- update_weights() per-RL-mode behavior (TestUpdateWeightFor{FullParamRL,LoRARL})
- FlattenedTensorBucket round-trip (sglang dependency smoke test)

Engine-result validation tests (_check_weight_sync_results) live in
test_update_weight_common.py since that helper is defined in common.py.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")


from argparse import Namespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, is_lora_weight_name
from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
    UpdateWeightFromTensor,
    _send_to_colocated_engine,
)

_UW_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_LORA_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.randn(4, 2)),
    ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.randn(2, 4)),
    ("model.layers.0.mlp.gate_proj.lora_A.weight", torch.randn(8, 2)),
    ("model.layers.0.mlp.gate_proj.lora_B.weight", torch.randn(2, 8)),
]

SAMPLE_BASE_ONLY_WEIGHTS = [
    ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4)),
    ("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 4)),
]


def _make_args(**overrides):
    defaults = dict(
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["linear_qkv", "linear_proj"],
        megatron_to_hf_mode="bridge",
        rollout_num_gpus_per_engine=1,
        hf_checkpoint="/fake/path",
        update_weight_buffer_size=1 << 30,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        pause_generation_mode="retract",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _build_updater(*, is_lora: bool, quantization_config=None, iterator=None, args_overrides=None):
    """Construct an UpdateWeightFromTensor for colocate-mode tests.

    Patches dist + HfWeightIteratorBase locally for __init__ only. Tests
    whose body itself calls into dist (e.g. update_weights tests) should
    additionally apply @patch decorators at the method level — those
    remain in effect after this helper's local patches expire.

    Args:
        is_lora: passed through to the constructor.
        quantization_config: passed through (None for non-quant tests,
            {"quant_method": "compressed-tensors"} for the quant path).
        iterator: optional mock iterator to inject as the HfWeightIterator
            instance. Use this when the test needs a specific filtering
            behavior (e.g. _make_filtering_iterator()) or an empty iterator
            for zero-chunk tests.
        args_overrides: optional dict to override fields in the Namespace
            passed to __init__ (e.g. actor_num_gpus_per_node for
            connect_rollout_engines tests).
    """
    with patch(f"{_UW_MODULE}.dist") as mock_dist, patch(f"{_UW_MODULE}.HfWeightIteratorBase") as mock_iter_base:
        mock_dist.get_world_size.return_value = 1
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock()
        mock_iter_base.create.return_value = iterator if iterator is not None else MagicMock()
        updater = UpdateWeightFromTensor(
            args=_make_args(**(args_overrides or {})),
            model=[MagicMock()],
            weights_getter=lambda: {},
            model_name="qwen",
            quantization_config=quantization_config,
            is_lora=is_lora,
        )
    updater._ipc_engine = MagicMock()
    updater._ipc_gather_src = 0
    updater._ipc_gather_group = MagicMock()
    updater.rollout_engines = [MagicMock()]
    updater.use_distribute = False
    return updater


def _make_filtering_iterator():
    """Mock iterator whose get_hf_weight_chunks yields a subset by weight_type
    (mimicking HfWeightIteratorBridge's real filtering behavior)."""
    iterator = MagicMock()

    def _filter(_weights, weight_type):
        if weight_type == "base":
            yield SAMPLE_BASE_ONLY_WEIGHTS
        elif weight_type == "lora":
            yield SAMPLE_LORA_WEIGHTS

    iterator.get_hf_weight_chunks.side_effect = _filter
    return iterator


def _make_empty_iterator():
    """Mock iterator that yields no chunks for any weight_type."""
    iterator = MagicMock()
    iterator.get_hf_weight_chunks.return_value = iter([])
    return iterator


def _setup_mock_dist_for_gather(mock_dist, *, world_size=1, rank=0):
    """Configure a mocked dist module so gather_object writes the calling
    rank's payload into the gather list (single-rank semantics).

    Lets us drive _send_to_colocated_engine end-to-end without booting a
    real process group; FlattenedTensorBucket and MultiprocessingSerializer
    stay real (both CPU-safe)."""
    mock_dist.get_world_size.return_value = world_size
    mock_dist.get_rank.return_value = rank

    def _fake_gather_object(obj, *, object_gather_list=None, dst=None, group=None):
        if object_gather_list is not None:
            object_gather_list[0] = obj

    mock_dist.gather_object.side_effect = _fake_gather_object


# ---------------------------------------------------------------------------
# UpdateWeightFromTensor.__init__ wiring (LoRA config)
# ---------------------------------------------------------------------------


class TestUpdateWeightFromTensorLoraConfig:
    """Verify _lora_config is built only when is_lora=True."""

    def test_lora_true_sets_config(self):
        updater = _build_updater(is_lora=True)
        assert updater._lora_config is not None
        assert updater._lora_config["peft_type"] == "LORA"
        assert updater._lora_config["r"] == 32

    def test_lora_false_no_config(self):
        updater = _build_updater(is_lora=False)
        assert not hasattr(updater, "_lora_config")


# ---------------------------------------------------------------------------
# connect_rollout_engines (colocate-mode branches only)
# ---------------------------------------------------------------------------


class TestConnectRolloutEngines:
    """Tests for UpdateWeightFromTensor.connect_rollout_engines.

    Covers the colocate-mode branches: default vs custom engine_gpu_offsets,
    current rank's _ipc_engine assignment, placeholder ranks (rank reserved
    but no engine covers it), and IPC group recreation when the layout
    changes.

    Distributed-mode branches (use_distribute=True, _is_distributed_src_rank,
    connect_rollout_engines_from_distributed) are deferred to
    test_weight_sync_distributed.py.
    """

    @patch(f"{_UW_MODULE}.dist")
    def test_default_offsets_dense_packing(self, mock_dist):
        """When engine_gpu_offsets=None, compute dense packing from
        engine_gpu_counts and assign current rank to the matching engine."""
        mock_dist.get_rank.return_value = 0

        updater = _build_updater(
            is_lora=False,
            args_overrides={"actor_num_gpus_per_node": 4, "actor_num_nodes": 1},
        )
        engines = [MagicMock(name="engine0"), MagicMock(name="engine1")]
        updater.connect_rollout_engines(
            rollout_engines=engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            # engine_gpu_offsets=None → dense packing produces [0, 2]
        )

        assert updater.use_distribute is False
        assert updater._ipc_engine is engines[0]  # rank 0 in [0, 2)
        assert updater.rollout_engines == engines

    @patch(f"{_UW_MODULE}.dist")
    def test_custom_offsets_with_gap(self, mock_dist):
        """Explicit engine_gpu_offsets are honored, including non-dense gaps."""
        mock_dist.get_rank.return_value = 5  # last rank in engine 1's range

        updater = _build_updater(
            is_lora=False,
            args_overrides={"actor_num_gpus_per_node": 6, "actor_num_nodes": 1},
        )
        engines = [MagicMock(name="engine0"), MagicMock(name="engine1")]
        updater.connect_rollout_engines(
            rollout_engines=engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 4],  # ranks 2-3 are placeholders (no engine)
        )

        assert updater.use_distribute is False
        assert updater._ipc_engine is engines[1]  # rank 5 in [4, 6)

    @patch(f"{_UW_MODULE}.dist")
    def test_rank_in_first_engine(self, mock_dist):
        """Rank inside engine 0's GPU range → _ipc_engine is engines[0]."""
        mock_dist.get_rank.return_value = 1

        updater = _build_updater(
            is_lora=False,
            args_overrides={"actor_num_gpus_per_node": 4, "actor_num_nodes": 1},
        )
        engines = [MagicMock(name="engine0"), MagicMock(name="engine1")]
        updater.connect_rollout_engines(
            rollout_engines=engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
        )

        assert updater._ipc_engine is engines[0]  # rank 1 in [0, 2)

    @patch(f"{_UW_MODULE}.dist")
    def test_rank_in_second_engine(self, mock_dist):
        """Rank inside engine 1's GPU range → _ipc_engine is engines[1]."""
        mock_dist.get_rank.return_value = 3

        updater = _build_updater(
            is_lora=False,
            args_overrides={"actor_num_gpus_per_node": 4, "actor_num_nodes": 1},
        )
        engines = [MagicMock(name="engine0"), MagicMock(name="engine1")]
        updater.connect_rollout_engines(
            rollout_engines=engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
        )

        assert updater._ipc_engine is engines[1]  # rank 3 in [2, 4)

    @patch(f"{_UW_MODULE}.dist")
    def test_placeholder_rank_resets_ipc_state(self, mock_dist):
        """Rank not covered by any engine → _ipc_engine, _ipc_gather_group,
        and _ipc_gather_src all become None. Catches the case where a worker
        reserved a GPU slot but no engine claims those GPUs."""
        mock_dist.get_rank.return_value = 3

        updater = _build_updater(
            is_lora=False,
            args_overrides={"actor_num_gpus_per_node": 4, "actor_num_nodes": 1},
        )
        engines = [MagicMock(name="engine0")]
        updater.connect_rollout_engines(
            rollout_engines=engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2],  # single engine covering [0, 2); rank 3 is a placeholder
        )

        assert updater.use_distribute is False
        assert updater._ipc_engine is None
        assert updater._ipc_gather_group is None
        assert updater._ipc_gather_src is None

    @patch(f"{_UW_MODULE}.dist")
    def test_ipc_group_recreated_when_was_none(self, mock_dist):
        """If _ipc_gather_group is None on entry (e.g. previous call left this
        rank as a placeholder) and the new layout DOES cover this rank,
        connect_rollout_engines recreates the IPC group via dist.new_group."""
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.return_value = MagicMock(name="new_gather_group")

        updater = _build_updater(
            is_lora=False,
            args_overrides={"actor_num_gpus_per_node": 4, "actor_num_nodes": 1},
        )
        # Simulate a prior placeholder state: gather group reset to None.
        updater._ipc_gather_group = None
        updater._ipc_gather_src = None

        engines = [MagicMock(name="engine0"), MagicMock(name="engine1")]
        updater.connect_rollout_engines(
            rollout_engines=engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
        )

        # New IPC group created for rank 0's engine (covering [0, 2)).
        mock_dist.new_group.assert_called()
        assert updater._ipc_gather_group is mock_dist.new_group.return_value
        assert updater._ipc_gather_src == 0


# ---------------------------------------------------------------------------
# _send_base_params
# ---------------------------------------------------------------------------


class TestSendBaseParams:
    """Tests for UpdateWeightFromTensor._send_base_params (direct calls only).

    End-to-end behavior in update_weights() is covered by
    TestUpdateWeightForFullParamRL and TestUpdateWeightForLoRARL.
    """

    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    def test_passes_base_weights_to_colocated_engine(self, mock_send):
        """Happy path: base-only chunk is forwarded with base-specific kwargs."""
        updater = _build_updater(is_lora=False)
        refs, _ = updater._send_base_params(SAMPLE_BASE_ONLY_WEIGHTS)
        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["hf_named_tensors"] == SAMPLE_BASE_ONLY_WEIGHTS
        assert kwargs["ipc_engine"] is updater._ipc_engine
        assert kwargs["ipc_gather_src"] == updater._ipc_gather_src
        assert kwargs["ipc_gather_group"] is updater._ipc_gather_group
        assert kwargs["weight_version"] == updater.weight_version
        assert refs == []

    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    def test_does_not_raise_on_empty_chunk(self, mock_send):
        """Empty chunk through base path is valid (degenerate but not an error)."""
        updater = _build_updater(is_lora=False)
        updater._send_base_params([])
        mock_send.assert_called_once()
        assert mock_send.call_args.kwargs["hf_named_tensors"] == []


# ---------------------------------------------------------------------------
# _send_lora_params
# ---------------------------------------------------------------------------


class TestSendLoraParams:
    """Tests for UpdateWeightFromTensor._send_lora_params (direct calls only).

    End-to-end behavior in update_weights() is covered by
    TestUpdateWeightForLoRARL.
    """

    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    def test_passes_lora_weights_to_colocated_engine(self, mock_send):
        """Happy path: lora chunk is forwarded with lora-specific kwargs."""
        updater = _build_updater(is_lora=True)
        updater._send_lora_params(SAMPLE_LORA_WEIGHTS)
        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["hf_named_tensors"] == SAMPLE_LORA_WEIGHTS
        assert kwargs["ipc_engine"] is updater._ipc_engine
        assert kwargs["ipc_gather_src"] == updater._ipc_gather_src
        assert kwargs["ipc_gather_group"] is updater._ipc_gather_group
        assert kwargs["lora_config"] is updater._lora_config
        assert kwargs["lora_name"] == LORA_ADAPTER_NAME

    def test_raises_when_chunk_has_no_lora_weights(self):
        updater = _build_updater(is_lora=True)
        with pytest.raises(RuntimeError, match="no LoRA weights"):
            updater._send_lora_params(SAMPLE_BASE_ONLY_WEIGHTS)

    @patch(f"{_UW_MODULE}._send_to_colocated_engine", return_value=([], []))
    def test_lora_loaded_flag_flips_after_first_call(self, mock_send):
        """Contract: first _send_lora_params forwards lora_loaded=False (no prior
        adapter to unload). After the call, self._lora_loaded flips to True so
        the next call forwards lora_loaded=True (triggers unload-then-load)."""
        updater = _build_updater(is_lora=True)
        assert updater._lora_loaded is False

        # First call
        updater._send_lora_params(SAMPLE_LORA_WEIGHTS)
        assert mock_send.call_args_list[0].kwargs["lora_loaded"] is False
        assert updater._lora_loaded is True  # flipped after first call

        # Second call
        updater._send_lora_params(SAMPLE_LORA_WEIGHTS)
        assert mock_send.call_args_list[1].kwargs["lora_loaded"] is True
        assert updater._lora_loaded is True  # stays True


# ---------------------------------------------------------------------------
# _send_to_colocated_engine
# ---------------------------------------------------------------------------


class TestSendToColocatedEngine:
    """Tests for the colocated-engine send path.

    Direct unit tests for the module-level _send_to_colocated_engine helper,
    plus the orchestration-level RPC sequencing in update_weights() that
    wraps it (pause/flush/post_process/continue must hit the engine in a
    specific order around the weight-send call).
    """

    @patch(f"{_UW_MODULE}.dist")
    def test_base_sync_calls_update_weights_from_tensor(self, mock_dist):
        """Base path: ipc_engine.update_weights_from_tensor.remote is invoked
        with the serialized payload and a stringified weight_version. The LoRA
        adapter APIs must NOT be touched."""
        _setup_mock_dist_for_gather(mock_dist)
        ipc_engine = MagicMock()

        refs, long_lived = _send_to_colocated_engine(
            hf_named_tensors=SAMPLE_BASE_ONLY_WEIGHTS,
            ipc_engine=ipc_engine,
            ipc_gather_src=0,
            ipc_gather_group=MagicMock(),
            weight_version=7,
        )

        ipc_engine.update_weights_from_tensor.remote.assert_called_once()
        ipc_engine.load_lora_adapter_from_tensors.remote.assert_not_called()
        ipc_engine.unload_lora_adapter.remote.assert_not_called()

        kwargs = ipc_engine.update_weights_from_tensor.remote.call_args.kwargs
        assert kwargs["load_format"] == "flattened_bucket"
        assert kwargs["weight_version"] == "7"
        # One serialized payload (single bucket for the one float32 dtype)
        assert len(kwargs["serialized_named_tensors"]) == 1
        # refs contains the one .remote() return; long_lived holds the bucket
        # data we must keep alive until ray.get on the caller side.
        assert refs == [ipc_engine.update_weights_from_tensor.remote.return_value]
        assert len(long_lived) == 1

    @patch(f"{_UW_MODULE}.dist")
    def test_lora_sync_when_adapter_not_loaded(self, mock_dist):
        """LoRA path with lora_loaded=False: load_lora_adapter_from_tensors is
        invoked, unload_lora_adapter is NOT (no prior adapter to drop)."""
        _setup_mock_dist_for_gather(mock_dist)
        ipc_engine = MagicMock()
        lora_config = {"peft_type": "LORA", "r": 32}

        refs, _ = _send_to_colocated_engine(
            hf_named_tensors=SAMPLE_LORA_WEIGHTS,
            ipc_engine=ipc_engine,
            ipc_gather_src=0,
            ipc_gather_group=MagicMock(),
            lora_config=lora_config,
            lora_name="test_adapter",
            lora_loaded=False,
        )

        ipc_engine.unload_lora_adapter.remote.assert_not_called()
        ipc_engine.load_lora_adapter_from_tensors.remote.assert_called_once()
        ipc_engine.update_weights_from_tensor.remote.assert_not_called()

        kwargs = ipc_engine.load_lora_adapter_from_tensors.remote.call_args.kwargs
        assert kwargs["lora_name"] == "test_adapter"
        assert kwargs["config_dict"] is lora_config
        assert kwargs["load_format"] == "flattened_bucket"
        assert refs == [ipc_engine.load_lora_adapter_from_tensors.remote.return_value]

    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    def test_lora_sync_when_adapter_loaded(self, mock_dist, mock_ray):
        """LoRA path with lora_loaded=True: unload_lora_adapter must be called
        BEFORE load_lora_adapter_from_tensors. The ray.get on the unload ref
        is awaited synchronously, so we mock ray to make it a no-op."""
        _setup_mock_dist_for_gather(mock_dist)

        # Track invocation order across the two ipc_engine methods.
        ipc_engine = MagicMock()
        call_order = []
        ipc_engine.unload_lora_adapter.remote.side_effect = (
            lambda **kw: call_order.append("unload") or MagicMock()
        )
        ipc_engine.load_lora_adapter_from_tensors.remote.side_effect = (
            lambda **kw: call_order.append("load") or MagicMock()
        )

        _send_to_colocated_engine(
            hf_named_tensors=SAMPLE_LORA_WEIGHTS,
            ipc_engine=ipc_engine,
            ipc_gather_src=0,
            ipc_gather_group=MagicMock(),
            lora_config={"peft_type": "LORA", "r": 32},
            lora_name="test_adapter",
            lora_loaded=True,
        )

        ipc_engine.unload_lora_adapter.remote.assert_called_once_with(lora_name="test_adapter")
        ipc_engine.load_lora_adapter_from_tensors.remote.assert_called_once()
        # Unload first, then load.
        assert call_order == ["unload", "load"]
        # And ray.get was awaited on the unload ref before issuing load.
        mock_ray.get.assert_called_once()

    @patch(f"{_UW_MODULE}.post_process_weights")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_rank_zero_rpc_order_for_base(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_pp
    ):
        """Full-param RL: on rank 0, RPCs must follow
            pause_generation → flush_cache → _send_base_params
                → post_process_weights → continue_generation
        The LoRA send path is never invoked."""
        mock_dist.get_rank.return_value = 0
        mock_ray.get.return_value = []

        updater = _build_updater(is_lora=False, iterator=_make_filtering_iterator())
        engine = MagicMock(name="engine")
        updater.rollout_engines = [engine]

        with (
            patch.object(updater, "_send_base_params", return_value=([], [])) as spy_base,
            patch.object(updater, "_send_lora_params", return_value=([], [])) as spy_lora,
        ):
            order = MagicMock()
            order.attach_mock(engine.pause_generation.remote, "pause")
            order.attach_mock(engine.flush_cache.remote, "flush")
            order.attach_mock(spy_base, "send_base")
            order.attach_mock(mock_pp, "post_process_weights")
            order.attach_mock(engine.continue_generation.remote, "continue_generation")

            updater.update_weights()

        method_order = [c[0] for c in order.mock_calls]
        assert method_order == [
            "pause",
            "flush",
            "send_base",
            "post_process_weights",
            "continue_generation",
        ]
        spy_lora.assert_not_called()

    @patch(f"{_UW_MODULE}.post_process_weights")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_rank_zero_rpc_order_for_lora(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_pp
    ):
        """LoRA RL (colocated): on rank 0, RPCs must follow
            pause_generation → flush_cache → _send_base_params → _send_lora_params
                → post_process_weights → continue_generation
        Base sync must complete BEFORE lora sync so the adapter loads against
        fresh base weights."""
        mock_dist.get_rank.return_value = 0
        mock_ray.get.return_value = []

        updater = _build_updater(is_lora=True, iterator=_make_filtering_iterator())
        engine = MagicMock(name="engine")
        updater.rollout_engines = [engine]

        with (
            patch.object(updater, "_send_base_params", return_value=([], [])) as spy_base,
            patch.object(updater, "_send_lora_params", return_value=([], [])) as spy_lora,
        ):
            order = MagicMock()
            order.attach_mock(engine.pause_generation.remote, "pause")
            order.attach_mock(engine.flush_cache.remote, "flush")
            order.attach_mock(spy_base, "send_base")
            order.attach_mock(spy_lora, "send_lora")
            order.attach_mock(mock_pp, "post_process_weights")
            order.attach_mock(engine.continue_generation.remote, "continue_generation")

            updater.update_weights()

        method_order = [c[0] for c in order.mock_calls]
        assert method_order == [
            "pause",
            "flush",
            "send_base",
            "send_lora",
            "post_process_weights",
            "continue_generation",
        ]


    @pytest.mark.parametrize(
        "lora_config",
        [
            pytest.param(None, id="base"),
            pytest.param({"peft_type": "LORA", "r": 32}, id="lora"),
        ],
    )
    def test_placeholder_rank_returns_empty(self, lora_config):
        """When ipc_gather_group is None (rank reserved a GPU slot but no
        engine covers it), _send_to_colocated_engine must short-circuit and
        return ([], None) without touching the ipc engine.

        Parametrized to lock in that the short-circuit is mode-agnostic — it
        runs BEFORE the `is_lora = lora_config is not None` check in the
        production code. If someone ever inserts a mode-specific branch above
        the short-circuit, the [base] or [lora] case will catch it."""
        ipc_engine = MagicMock()
        refs, long_lived = _send_to_colocated_engine(
            hf_named_tensors=SAMPLE_BASE_ONLY_WEIGHTS,
            ipc_engine=ipc_engine,
            ipc_gather_src=None,
            ipc_gather_group=None,
            weight_version=0,
            lora_config=lora_config,
            lora_name="x" if lora_config is not None else None,
        )
        assert refs == []
        assert long_lived is None
        # No engine RPCs because we short-circuited before serialization.
        ipc_engine.update_weights_from_tensor.remote.assert_not_called()
        ipc_engine.load_lora_adapter_from_tensors.remote.assert_not_called()
        ipc_engine.unload_lora_adapter.remote.assert_not_called()

    @patch(f"{_UW_MODULE}.dist")
    def test_non_gather_src_rank_returns_empty_refs(self, mock_dist):
        """Ranks that are NOT the gather src still participate in
        dist.gather_object (their serialized payload goes to the src), but
        they must return refs=[] — only the src rank issues .remote() RPCs
        to the engine actor."""
        _setup_mock_dist_for_gather(mock_dist, world_size=2, rank=1)
        ipc_engine = MagicMock()

        refs, long_lived = _send_to_colocated_engine(
            hf_named_tensors=SAMPLE_BASE_ONLY_WEIGHTS,
            ipc_engine=ipc_engine,
            ipc_gather_src=0,  # rank 1 is NOT the src
            ipc_gather_group=MagicMock(),
            weight_version=0,
        )

        assert refs == []
        # Local serialization still ran on this rank (long_live_tensors holds
        # the bucket data we must keep alive until gather_object completes).
        assert long_lived is not None
        assert len(long_lived) >= 1
        # No engine RPCs from this rank.
        ipc_engine.update_weights_from_tensor.remote.assert_not_called()
        ipc_engine.load_lora_adapter_from_tensors.remote.assert_not_called()
        ipc_engine.unload_lora_adapter.remote.assert_not_called()
        # And gather_object was still called (this rank contributed its payload).
        mock_dist.gather_object.assert_called_once()


# ---------------------------------------------------------------------------
# update_weights() — per-RL-mode behavior
# ---------------------------------------------------------------------------


class TestUpdateWeightForFullParamRL:
    """Tests for update_weights() behavior in full-param RL (is_lora=False, colocated).

    Contract: every update_weights() round syncs only base weights; the LoRA
    send path is never invoked.
    """

    @patch(f"{_UW_MODULE}.post_process_weights")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_colocate_mode_weight_sync_behavior(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_pp
    ):
        """Drive update_weights() twice; verify base sent both rounds, lora never."""
        mock_dist.get_rank.return_value = 0
        mock_ray.get.return_value = []

        updater = _build_updater(is_lora=False, iterator=_make_filtering_iterator())

        with (
            patch.object(updater, "_send_base_params", return_value=([], [])) as spy_base,
            patch.object(updater, "_send_lora_params", return_value=([], [])) as spy_lora,
        ):
            updater.update_weights()  # Round 1
            updater.update_weights()  # Round 2

        assert spy_base.call_args_list == [call(SAMPLE_BASE_ONLY_WEIGHTS)] * 2
        spy_lora.assert_not_called()

    @patch("miles.backends.megatron_utils.update_weight.common.ray")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_colocate_mode_no_raise_on_zero_chunks(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_common_ray
    ):
        """Empty iterator output is valid (e.g. empty model state)."""
        mock_dist.get_rank.return_value = 0

        updater = _build_updater(is_lora=False, iterator=_make_empty_iterator())
        updater.update_weights()

    @patch(f"{_UW_MODULE}.post_process_weights")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_colocate_mode_non_rank_zero_skips_rank0_rpcs(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_pp
    ):
        """On non-rank-0 workers, update_weights skips pause/flush/continue
        engine RPCs and the final post_process_weights (those are gated on
        rank == 0). The weight-send loop still runs on every rank."""
        mock_dist.get_rank.return_value = 1  # non-rank-0
        mock_ray.get.return_value = []

        updater = _build_updater(is_lora=False, iterator=_make_filtering_iterator())
        engine = MagicMock(name="engine")
        updater.rollout_engines = [engine]

        with (
            patch.object(updater, "_send_base_params", return_value=([], [])) as spy_base,
            patch.object(updater, "_send_lora_params", return_value=([], [])),
        ):
            updater.update_weights()

        # Rank-0-only RPCs not invoked
        engine.pause_generation.remote.assert_not_called()
        engine.flush_cache.remote.assert_not_called()
        engine.continue_generation.remote.assert_not_called()
        mock_pp.assert_not_called()
        # The weight-send loop still runs across all ranks
        spy_base.assert_called_once_with(SAMPLE_BASE_ONLY_WEIGHTS)

    @patch(f"{_UW_MODULE}.post_process_weights")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_colocate_mode_compressed_tensors_quant_pre_and_post_process(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_pp
    ):
        """When quantization_config['quant_method'] == 'compressed-tensors',
        post_process_weights is invoked TWICE per round:
        1) pre-sync, with restore_weights_before_load=True, post_process_quantization=False
        2) post-sync, with restore_weights_before_load=False, post_process_quantization=True
        """
        mock_dist.get_rank.return_value = 0
        mock_ray.get.return_value = []

        updater = _build_updater(
            is_lora=False,
            quantization_config={"quant_method": "compressed-tensors"},
            iterator=_make_filtering_iterator(),
        )

        with patch.object(updater, "_send_base_params", return_value=([], [])):
            updater.update_weights()

        assert mock_pp.call_count == 2
        pre, post = mock_pp.call_args_list
        assert pre.kwargs["restore_weights_before_load"] is True
        assert pre.kwargs["post_process_quantization"] is False
        assert post.kwargs["restore_weights_before_load"] is False
        assert post.kwargs["post_process_quantization"] is True


class TestUpdateWeightForLoRARL:
    """Tests for update_weights() behavior in LoRA RL (is_lora=True, colocated).

    Contract: every update_weights() round syncs both base and LoRA weights.
    The distributed-mode optimization that skips base after the first round
    lives in test_weight_sync_distributed.py (out of scope here).
    """

    @patch(f"{_UW_MODULE}.post_process_weights")
    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_colocate_mode_weight_sync_behavior(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo, mock_pp
    ):
        """Drive update_weights() twice; verify each round sends base BEFORE
        lora (the LoRA adapter must reference fresh base weights), and that
        the iterator's weight_type filtering routes the right chunk to each
        method."""
        mock_dist.get_rank.return_value = 0
        mock_ray.get.return_value = []

        updater = _build_updater(is_lora=True, iterator=_make_filtering_iterator())

        with (
            patch.object(updater, "_send_base_params", return_value=([], [])) as spy_base,
            patch.object(updater, "_send_lora_params", return_value=([], [])) as spy_lora,
        ):
            # Attach both spies to a parent mock so mock_calls records the
            # chronological interleaving across the two methods.
            order = MagicMock()
            order.attach_mock(spy_base, "send_base")
            order.attach_mock(spy_lora, "send_lora")

            updater.update_weights()  # Round 1
            updater.update_weights()  # Round 2

        assert order.mock_calls == [
            call.send_base(SAMPLE_BASE_ONLY_WEIGHTS),
            call.send_lora(SAMPLE_LORA_WEIGHTS),
            call.send_base(SAMPLE_BASE_ONLY_WEIGHTS),
            call.send_lora(SAMPLE_LORA_WEIGHTS),
        ]

    @patch(f"{_UW_MODULE}.get_gloo_group", return_value=MagicMock())
    @patch(f"{_UW_MODULE}.ray")
    @patch(f"{_UW_MODULE}.dist")
    @patch(f"{_UW_MODULE}.HfWeightIteratorBase")
    def test_colocate_mode_raise_on_zero_chunks(
        self, mock_iter_base, mock_dist, mock_ray, mock_gloo
    ):
        """LoRA sync with zero chunks signals a real incompatibility and must raise."""
        mock_dist.get_rank.return_value = 0

        updater = _build_updater(is_lora=True, iterator=_make_empty_iterator())
        with pytest.raises(RuntimeError, match="zero chunks"):
            updater.update_weights()


# ---------------------------------------------------------------------------
# FlattenedTensorBucket round-trip correctness
# ---------------------------------------------------------------------------


class TestFlattenedTensorBucketRoundTrip:
    """Verify serialize -> reconstruct preserves tensor values exactly."""

    def _get_bucket_class(self):
        try:
            from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
        except ImportError:
            pytest.skip("sglang FlattenedTensorBucket not available")
        return FlattenedTensorBucket

    def test_roundtrip_single_dtype(self):
        FlattenedTensorBucket = self._get_bucket_class()
        tensors = [
            ("a", torch.randn(4, 4, dtype=torch.bfloat16)),
            ("b", torch.randn(2, 8, dtype=torch.bfloat16)),
        ]

        bucket = FlattenedTensorBucket(named_tensors=tensors)
        reconstructed = bucket.reconstruct_tensors()

        assert len(reconstructed) == len(tensors)
        for (orig_name, orig_t), (rec_name, rec_t) in zip(tensors, reconstructed, strict=True):
            assert orig_name == rec_name
            assert orig_t.shape == rec_t.shape
            assert orig_t.dtype == rec_t.dtype
            assert torch.equal(orig_t, rec_t), f"Tensor {orig_name} values differ after round-trip"

    @pytest.mark.xfail(
        reason="SGLang FlattenedTensorBucket.reconstruct_tensors() fails with mixed dtypes "
        "due to PyTorch view() alignment requirements (storage_offset not divisible by "
        "element size). In practice LoRA weights are typically uniform dtype so this is safe.",
        raises=RuntimeError,
        strict=False,
    )
    def test_roundtrip_mixed_dtypes(self):
        FlattenedTensorBucket = self._get_bucket_class()

        if not getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
            pytest.skip("FlattenedTensorBucket does not support multi-dtypes")

        tensors = [
            ("a_bf16", torch.randn(3, 3, dtype=torch.bfloat16)),
            ("b_fp32", torch.randn(2, 2, dtype=torch.float32)),
            ("c_fp16", torch.randn(5, dtype=torch.float16)),
        ]

        bucket = FlattenedTensorBucket(named_tensors=tensors)
        reconstructed = bucket.reconstruct_tensors()

        assert len(reconstructed) == len(tensors)
        for (orig_name, orig_t), (rec_name, rec_t) in zip(tensors, reconstructed, strict=True):
            assert orig_name == rec_name
            assert orig_t.dtype == rec_t.dtype
            assert torch.equal(orig_t, rec_t), f"Tensor {orig_name} values differ after round-trip"

    def test_roundtrip_from_flattened_data(self):
        """Simulate the receiver side: reconstruct from flattened_tensor + metadata."""
        FlattenedTensorBucket = self._get_bucket_class()

        original = [
            ("lora_A", torch.randn(8, 2, dtype=torch.bfloat16)),
            ("lora_B", torch.randn(2, 8, dtype=torch.bfloat16)),
        ]

        sender_bucket = FlattenedTensorBucket(named_tensors=original)
        flat_tensor = sender_bucket.get_flattened_tensor()
        metadata = sender_bucket.get_metadata()

        receiver_bucket = FlattenedTensorBucket(flattened_tensor=flat_tensor, metadata=metadata)
        reconstructed = receiver_bucket.reconstruct_tensors()

        for (orig_name, orig_t), (rec_name, rec_t) in zip(original, reconstructed, strict=True):
            assert orig_name == rec_name
            assert torch.equal(orig_t, rec_t)

    def test_lora_only_tensors_filtered_correctly(self):
        """Verify that after filtering, only LoRA tensors survive and round-trip intact."""
        FlattenedTensorBucket = self._get_bucket_class()

        mixed = [
            ("model.layers.0.q_proj.weight", torch.randn(4, 4)),
            ("model.layers.0.q_proj.lora_A.weight", torch.randn(4, 2)),
            ("model.layers.0.q_proj.lora_B.weight", torch.randn(2, 4)),
        ]

        lora_only = [(n, t) for n, t in mixed if is_lora_weight_name(n)]
        assert len(lora_only) == 2

        bucket = FlattenedTensorBucket(named_tensors=lora_only)
        reconstructed = bucket.reconstruct_tensors()

        for (orig_name, orig_t), (rec_name, rec_t) in zip(lora_only, reconstructed, strict=True):
            assert orig_name == rec_name
            assert torch.equal(orig_t, rec_t)
