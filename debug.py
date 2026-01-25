# Debug file for matmul_persistent and related functions
# Extracted from batch_invariant_ops.py (standalone, no sglang imports)

import os
from collections.abc import Callable
from typing import Any, Dict

import torch
import triton
import triton.language as tl


# inlined from sglang.srt.utils.common
def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()
    truthy_values = ("true", "1")
    return value in truthy_values


# inlined from sglang.srt.utils.common
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# inlined from sglang.srt.layers.deep_gemm_wrapper.configurer
def _compute_enable_deep_gemm():
    if not torch.cuda.is_available():
        return False
    # check SM version >= 90 (Hopper+)
    props = torch.cuda.get_device_properties(0)
    sm_version = props.major * 10 + props.minor
    if sm_version < 90:
        return False
    try:
        import deep_gemm  # noqa: F401
        return True
    except ImportError:
        return False


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm

_ENABLE_MM_DEEPGEMM = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "1"
)
_ENABLE_MM_FALLBACK_VARIANT = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT", "0"
)
_ENABLE_MM_COMPARISON_TEST = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_COMPARISON_TEST"
)

if not _ENABLE_MM_DEEPGEMM:
    print("Disable DeepGEMM in batch invariant ops. Performance may be suboptimal.")


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        elif c_ptr.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float32:
            c = accumulator.to(tl.float32)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def _matmul_persistent_triton(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert (
        bias is None or bias.dim() == 1
    ), "Currently assuming bias is 1D, let Horace know if you run into this"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


def _matmul_persistent_deepgemm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    out = torch.empty((M, N), device=a.device, dtype=dtype)

    try:
        deep_gemm.bf16_gemm_nn(a, b, out)
    except RuntimeError as e:
        raise RuntimeError(
            f"DeepGEMM failed for matrix shapes M={M}, N={N}, K={K}. "
            f"This typically occurs when dimensions are too small for DeepGEMM's TMA descriptors. "
            f"Consider increasing MIN_DEEPGEMM_DIM in matmul_persistent() or disabling DeepGEMM "
            f"for small matrices. Original error: {e}"
        ) from e

    if bias is not None:
        out += bias

    return out


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    K, N = b.shape

    # DeepGEMM has minimum dimension requirements for TMA descriptors
    MIN_DEEPGEMM_DIM = 16

    if (
        _ENABLE_MM_DEEPGEMM
        and ENABLE_JIT_DEEPGEMM
        and (a.dtype == torch.bfloat16)
        and (b.dtype == torch.bfloat16)
        and a.is_contiguous()
        and b.transpose(0, 1).is_contiguous()
        and N >= MIN_DEEPGEMM_DIM
    ):
        if _ENABLE_MM_COMPARISON_TEST:
            out_triton = _matmul_persistent_triton(a=a, b=b, bias=bias)
            out_deepgemm = _matmul_persistent_deepgemm(a=a, b=b, bias=bias)
            diff = calc_diff(out_triton, out_deepgemm)
            assert diff < 0.0001, f"{diff=} {out_triton=} {out_deepgemm=}"
            return out_deepgemm

        return _matmul_persistent_deepgemm(a=a, b=b, bias=bias)

    if _ENABLE_MM_FALLBACK_VARIANT:
        out = torch.einsum("ik,kj->ij", a, b)
        if bias is not None:
            out += bias
        return out

    return _matmul_persistent_triton(a=a, b=b, bias=bias)


if __name__ == "__main__":
    M, K, N = 256, 512, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    # test without bias
    out = matmul_persistent(a, b)
    ref = torch.mm(a, b)
    diff = (out - ref).abs().max().item()
    print(f"Test shape: M={M}, K={K}, N={N}")
    print(f"Without bias - max diff: {diff}")

    # test with bias
    out_bias = matmul_persistent(a, b, bias=bias)
    ref_bias = torch.mm(a, b) + bias
    diff_bias = (out_bias - ref_bias).abs().max().item()
    print(f"With bias - max diff: {diff_bias}")

    print(f"Result shape: {out.shape}")
    print(f"ENABLE_JIT_DEEPGEMM: {ENABLE_JIT_DEEPGEMM}")
    print(f"_ENABLE_MM_DEEPGEMM: {_ENABLE_MM_DEEPGEMM}")
