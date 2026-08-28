"""Unit tests for miles.backends.megatron_utils.lora_utils.

Tests cover module name conversion, LoRA detection helpers, parameter identification,
exclude-module parsing, and LoRA sync config building — all without GPU.
"""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.megatron_utils.lora_utils import (
    _adapter_param_groups,
    _get_lora_class_name,
    _is_adapter_param_name,
    _repack_params_into,
    build_lora_sync_config,
    convert_target_modules_to_hf,
    convert_target_modules_to_megatron,
    is_lora_enabled,
    is_lora_weight_name,
    parse_exclude_modules,
)
from miles.utils.lora import LORA_ADAPTER_NAME

# ---------------------------------------------------------------------------
# _get_lora_class_name
# ---------------------------------------------------------------------------


class TestGetLoraClassName:
    def test_none_returns_canonical(self):
        assert _get_lora_class_name(None) == "CanonicalLoRA"

    def test_type_returns_class_name(self):
        class FakeLoRA:
            pass

        assert _get_lora_class_name(FakeLoRA) == "FakeLoRA"

    def test_instance_returns_class_name(self):
        class FakeLoRA:
            pass

        assert _get_lora_class_name(FakeLoRA()) == "FakeLoRA"


# ---------------------------------------------------------------------------
# convert_target_modules_to_megatron
# ---------------------------------------------------------------------------


def _make_lora_type(name: str):
    """Helper to create a mock lora_type whose class name matches *name*."""
    mock = MagicMock()
    type(mock).__name__ = name
    return mock


class TestConvertTargetModulesToMegatron:
    def test_gdn_hf_names_collapse_to_fused_in_proj(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(["in_proj_qkvz", "in_proj_ba", "out_proj"], lora_type=lora)
        assert result == ["in_proj", "out_proj"]

    # --- "all-linear" variants ------------------------------------------------

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_string_canonical(self, shorthand):
        result = convert_target_modules_to_megatron(shorthand, lora_type=None)
        assert result == [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_string_standard(self, shorthand):
        lora_type = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(shorthand, lora_type=lora_type)
        assert result == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_single_element_list(self, shorthand):
        result = convert_target_modules_to_megatron([shorthand], lora_type=None)
        assert len(result) == 7  # CanonicalLoRA has 7 modules

    # --- HF -> Megatron conversion (standard LoRA) ----------------------------

    def test_hf_to_megatron_standard_dedup(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj"], lora_type=lora)
        assert result == ["linear_qkv"]

    def test_hf_to_megatron_standard_all_modules(self):
        lora = _make_lora_type("LoRA")
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        result = convert_target_modules_to_megatron(modules, lora_type=lora)
        assert result == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    # --- HF -> Megatron conversion (CanonicalLoRA) ----------------------------

    def test_hf_to_megatron_canonical_split(self):
        result = convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj"], lora_type=None)
        assert result == ["linear_q", "linear_k", "linear_v"]

    def test_hf_to_megatron_canonical_gate_up(self):
        result = convert_target_modules_to_megatron(["gate_proj", "up_proj"], lora_type=None)
        assert result == ["linear_fc1_gate", "linear_fc1_up"]

    # --- Already in Megatron format -------------------------------------------

    def test_megatron_format_passthrough(self):
        modules = ["linear_qkv", "linear_proj"]
        result = convert_target_modules_to_megatron(modules, lora_type=None)
        assert result == modules

    def test_megatron_format_passthrough_canonical(self):
        modules = ["linear_q", "linear_k", "linear_v"]
        result = convert_target_modules_to_megatron(modules, lora_type=None)
        assert result == modules

    # --- Single string input --------------------------------------------------

    def test_single_hf_string_input(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron("o_proj", lora_type=lora)
        assert result == ["linear_proj"]


# ---------------------------------------------------------------------------
# convert_target_modules_to_hf
# ---------------------------------------------------------------------------


class TestConvertTargetModulesToHf:
    def test_standard_linear_qkv(self):
        assert convert_target_modules_to_hf(["linear_qkv"]) == ["q_proj", "k_proj", "v_proj"]

    def test_standard_linear_proj(self):
        assert convert_target_modules_to_hf(["linear_proj"]) == ["o_proj"]

    def test_standard_linear_fc1(self):
        assert convert_target_modules_to_hf(["linear_fc1"]) == ["gate_proj", "up_proj"]

    def test_standard_linear_fc2(self):
        assert convert_target_modules_to_hf(["linear_fc2"]) == ["down_proj"]

    def test_gdn_in_proj_expands_to_sglang_modules(self):
        assert convert_target_modules_to_hf(["in_proj"]) == ["in_proj_qkvz", "in_proj_ba"]

    @pytest.mark.parametrize(
        "module,expected",
        [
            ("out_proj", ["out_proj"]),  # same-name passthrough
            ("language_model.decoder.layers.*.self_attention.out_proj", ["out_proj"]),  # wildcard path
            ("language_model.decoder.layers.0.self_attention.out_proj", ["out_proj"]),  # dotted path, passthrough
            (
                "language_model.decoder.layers.0.self_attention.linear_qkv",  # dotted path, table-mapped
                ["q_proj", "k_proj", "v_proj"],
            ),
        ],
    )
    def test_paths_reduce_to_leaf_before_mapping(self, module, expected):
        assert convert_target_modules_to_hf([module]) == expected

    def test_canonical_split_modules(self):
        result = convert_target_modules_to_hf(["linear_q", "linear_k", "linear_v"])
        assert result == ["q_proj", "k_proj", "v_proj"]

    def test_canonical_fc1_gate_up(self):
        result = convert_target_modules_to_hf(["linear_fc1_gate", "linear_fc1_up"])
        assert result == ["gate_proj", "up_proj"]

    def test_unknown_module_passthrough(self):
        assert convert_target_modules_to_hf(["some_custom_module"]) == ["some_custom_module"]

    def test_roundtrip_canonical_all_linear(self):
        megatron = convert_target_modules_to_megatron("all-linear", lora_type=None)
        hf = convert_target_modules_to_hf(megatron)
        assert set(hf) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def test_roundtrip_standard_all_linear(self):
        lora = _make_lora_type("LoRA")
        megatron = convert_target_modules_to_megatron("all-linear", lora_type=lora)
        hf = convert_target_modules_to_hf(megatron)
        assert set(hf) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ---------------------------------------------------------------------------
# is_lora_enabled
# ---------------------------------------------------------------------------


class TestIsLoraEnabled:
    def test_enabled_by_rank(self):
        args = Namespace(lora_rank=32, lora_adapter_path=None)
        assert is_lora_enabled(args) is True

    def test_enabled_by_adapter_path(self):
        args = Namespace(lora_rank=0, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_enabled_by_both(self):
        args = Namespace(lora_rank=16, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_disabled(self):
        args = Namespace(lora_rank=0, lora_adapter_path=None)
        assert is_lora_enabled(args) is False

    def test_disabled_missing_attrs(self):
        args = Namespace()
        assert is_lora_enabled(args) is False


# ---------------------------------------------------------------------------
# is_lora_weight_name / _is_adapter_param_name
# ---------------------------------------------------------------------------


class TestIsLoraWeightName:
    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_A.default.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_B.default.weight",
        ],
    )
    def test_positive(self, name):
        assert is_lora_weight_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )
    def test_negative(self, name):
        assert is_lora_weight_name(name) is False


class TestIsAdapterParamName:
    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.lora_A.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
        ],
    )
    def test_positive(self, name):
        assert _is_adapter_param_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.weight",
            "module.decoder.layers.0.mlp.linear_fc1.weight",
            "module.embedding.word_embeddings.weight",
        ],
    )
    def test_negative(self, name):
        assert _is_adapter_param_name(name) is False


# ---------------------------------------------------------------------------
# parse_exclude_modules
# ---------------------------------------------------------------------------


class TestParseExcludeModules:
    def test_none(self):
        args = Namespace(exclude_modules=None)
        assert parse_exclude_modules(args) == []

    def test_single_module_string(self):
        args = Namespace(exclude_modules="o_proj")
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert result == ["linear_proj"]

    def test_comma_separated(self):
        args = Namespace(exclude_modules="o_proj, down_proj")
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert set(result) == {"linear_proj", "linear_fc2"}

    def test_list_input(self):
        args = Namespace(exclude_modules=["o_proj", "down_proj"])
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert set(result) == {"linear_proj", "linear_fc2"}

    def test_missing_attr(self):
        args = Namespace()
        assert parse_exclude_modules(args) == []


# ---------------------------------------------------------------------------
# build_lora_sync_config
# ---------------------------------------------------------------------------


class TestBuildLoraSyncConfig:
    def test_basic_config(self):
        args = Namespace(
            lora_rank=32,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        )
        config = build_lora_sync_config(args)
        assert config["peft_type"] == "LORA"
        assert config["r"] == 32
        assert config["lora_alpha"] == 32
        assert config["lora_dropout"] == 0.0
        assert config["bias"] == "none"
        assert config["task_type"] == "CAUSAL_LM"
        assert set(config["target_modules"]) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_no_target_modules_uses_default(self):
        args = Namespace(lora_rank=16, lora_alpha=16, lora_dropout=0.0, target_modules=None)
        config = build_lora_sync_config(args)
        assert len(config["target_modules"]) == 7

    def test_canonical_target_modules(self):
        args = Namespace(
            lora_rank=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["linear_q", "linear_k"],
        )
        config = build_lora_sync_config(args)
        assert config["target_modules"] == ["q_proj", "k_proj"]
        assert config["r"] == 8


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    assert LORA_ADAPTER_NAME == "miles_lora"


# ---------------------------------------------------------------------------
# Adapter param re-mapping out of the pausable memory-saver region
# ---------------------------------------------------------------------------


def _add_adapter(parent, name, dtype, in_shape=(2, 3), out_shape=(3, 2)):
    """Attach a `<name>.adapter.linear_{in,out}.weight` pair, as megatron-bridge PEFT names them."""
    wrapped = torch.nn.Module()
    wrapped.weight = torch.nn.Parameter(torch.ones(2, 3, dtype=dtype))  # frozen base
    adapter = torch.nn.Module()
    for side, shape, value in (("linear_in", in_shape, 2.0), ("linear_out", out_shape, 3.0)):
        leaf = torch.nn.Module()
        leaf.weight = torch.nn.Parameter(torch.full(shape, value, dtype=dtype))
        setattr(adapter, side, leaf)
    wrapped.adapter = adapter
    setattr(parent, name, wrapped)


def _chunk_with_adapter(use_distributed_optimizer=None, dtype=torch.float32):
    """A module shaped like a DDP-wrapped chunk: frozen base + two adapter params."""
    chunk = torch.nn.Module()
    chunk.decoder = torch.nn.Module()
    _add_adapter(chunk.decoder, "linear_qkv", dtype)
    if use_distributed_optimizer is not None:
        chunk.ddp_config = SimpleNamespace(use_distributed_optimizer=use_distributed_optimizer)
    return chunk


class TestAdapterParamGroups:
    def test_selects_only_adapter_params(self):
        chunk = _chunk_with_adapter()
        groups = _adapter_param_groups(chunk)

        assert len(groups) == 1
        [params] = groups.values()
        assert {tuple(p.shape) for p in params} == {(2, 3), (3, 2)}
        # The frozen base weight sits next to the adapter and must not be re-mapped.
        assert all(p is not chunk.decoder.linear_qkv.weight for p in params)

    def test_skipped_when_distributed_optimizer_already_remapped(self):
        assert _adapter_param_groups(_chunk_with_adapter(use_distributed_optimizer=True)) == {}

    def test_not_skipped_without_distributed_optimizer(self):
        assert _adapter_param_groups(_chunk_with_adapter(use_distributed_optimizer=False)) != {}

    def test_missing_ddp_config_is_treated_as_not_remapped(self):
        # Anything not demonstrably re-mapped must be re-mapped, or it faults on pause.
        assert _adapter_param_groups(_chunk_with_adapter()) != {}

    def test_grouped_by_dtype(self):
        chunk = _chunk_with_adapter(dtype=torch.float32)
        _add_adapter(chunk.decoder, "linear_fc1", torch.bfloat16)

        groups = _adapter_param_groups(chunk)
        assert {dtype for dtype, _ in groups} == {torch.float32, torch.bfloat16}
        assert all(len(params) == 2 for params in groups.values())


class TestRepackParamsInto:
    def test_values_preserved_and_storage_shared(self):
        chunk = _chunk_with_adapter()
        params = list(_adapter_param_groups(chunk).values())[0]
        expected = [p.detach().clone() for p in params]

        flat = torch.empty(sum(p.numel() for p in params))
        _repack_params_into(flat, params)

        for param, want in zip(params, expected, strict=True):
            assert torch.equal(param.data, want)
            assert param.shape == want.shape
            assert param.requires_grad
            assert param.data_ptr() >= flat.data_ptr()
            assert param.data.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()

    def test_already_registered_grad_hook_still_fires(self):
        # Megatron's DDP registers its AccumulateGrad hooks during __init__, before
        # this runs. set_data must not detach the param from them.
        param = torch.nn.Parameter(torch.ones(2, 2))
        fired = []
        tmp = param.expand_as(param)
        grad_acc = tmp.grad_fn.next_functions[0][0]
        grad_acc.register_hook(lambda *_: fired.append(1))

        _repack_params_into(torch.empty(4), [param])
        (param * 2).sum().backward()

        assert fired
        assert torch.equal(param.grad, torch.full((2, 2), 2.0))
