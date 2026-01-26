import math
from typing import Optional, Tuple

import torch


def is_triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        return True
    except Exception:
        return False


@torch.no_grad()
def pack_w4_rowwise(w_int4: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Pack signed 4-bit weights [-8, 7] row-wise into int8.

    Args:
        w_int4: [N, K] int8 tensor where each element is in [-8, 7].

    Returns:
        packed: [N, K_packed] int8 where K_packed = ceil(K / 2).
        K: original column size.
    """
    assert w_int4.dtype == torch.int8
    if w_int4.numel() == 0:
        return torch.empty_like(w_int4), 0

    N, K = w_int4.shape
    K_packed = (K + 1) // 2

    # Map signed [-8, 7] to unsigned [0, 15] for packing.
    w = w_int4.to(torch.int16)
    w = torch.where(w < 0, w + 16, w)
    # even positions (low nibble) and odd positions (high nibble)
    low = w[:, 0::2]
    high = w[:, 1::2]
    if high.shape[1] < low.shape[1]:
        # pad last high column with zeros
        pad = torch.zeros((N, 1), dtype=low.dtype, device=low.device)
        high = torch.cat([high, pad], dim=1)
    packed = (high << 4) | low
    return packed.to(torch.int8), K


if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _w4a4_mm_kernel(
        A_ptr,  # fp16/bf16 activation, [M, K]
        Bq_ptr,  # int8 packed weights, [N, K_packed]
        Sw_ptr,  # fp16 weight scales, [N, 1]
        Sa_ptr,  # fp16 activation scales, [M, 1]
        C_ptr,  # fp16/bf16 output, [M, N]
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        K_packed: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_sw_n: tl.constexpr,
        stride_sa_m: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE_FP16: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k  # [BLOCK_K]
            mask_k = k < K

            # Load A block [BLOCK_M, BLOCK_K]
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
            a = a.to(tl.float16)

            # Quantize activations to 4-bit sym around zero per-row using Sa
            sa_ptrs = Sa_ptr + offs_m * stride_sa_m
            sa = tl.load(sa_ptrs, mask=offs_m < M, other=1.0)  # [BLOCK_M]
            # avoid div by 0
            sa = tl.where(sa == 0, 1.0, sa)
            # simple symmetric quant: a_int4 ~ round(a / sa)
            a_scaled = a / sa[:, None]
            qmax = 7.0
            a_q = tl.clamp(tl.round(a_scaled), -qmax, qmax)

            # Load packed weights and dequantize on the fly.
            # For each k index, find packed column index and whether it is low/high nibble.
            k_packed = k // 2
            is_low = (k % 2) == 0

            # [BLOCK_N, K_packed_block]
            b_ptrs = Bq_ptr + offs_n[:, None] * stride_bn + k_packed[None, :] * stride_bk
            b_packed = tl.load(
                b_ptrs,
                mask=(offs_n[:, None] < N)
                & (k_packed[None, :] < K_packed)
                & mask_k[None, :],
                other=0,
            ).to(tl.uint8)

            # unpack 4-bit signed
            low = b_packed & 0xF
            high = (b_packed >> 4) & 0xF
            # choose low/high based on is_low
            is_low_b = is_low[None, :]
            val = tl.where(is_low_b, low, high)
            # map unsigned [0,15] -> signed [-8,7]
            val = val.to(tl.int8)
            val = tl.where(val > 7, val - 16, val)
            b_int4 = val.to(tl.float16)

            # apply weight scale per row
            sw_ptrs = Sw_ptr + offs_n * stride_sw_n
            sw = tl.load(sw_ptrs, mask=offs_n < N, other=1.0)  # [BLOCK_N]
            b_deq = b_int4 * sw[:, None]

            # dequantized activations: a_q * sa
            a_deq = a_q * sa[:, None]

            acc += tl.dot(a_deq, tl.trans(b_deq))

        c = acc.to(tl.float16 if OUT_DTYPE_FP16 else tl.bfloat16)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def _w4a16_mm_kernel(
        A_ptr,  # fp16/bf16 activation, [M, K]
        Bq_ptr,  # int8 packed weights, [N, K_packed]
        Sw_ptr,  # fp16 weight scales, [N, 1]
        C_ptr,  # fp16/bf16 output, [M, N]
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        K_packed: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_sw_n: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUT_DTYPE_FP16: tl.constexpr,
    ):
        """
        Weight-only 4-bit GEMM: A is kept in fp16/bf16, B is 4-bit packed.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k  # [BLOCK_K]
            mask_k = k < K

            # Load A block [BLOCK_M, BLOCK_K] and keep in fp16
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
            a = a.to(tl.float16)

            # Load packed weights and dequantize on the fly.
            k_packed = k // 2
            is_low = (k % 2) == 0

            b_ptrs = Bq_ptr + offs_n[:, None] * stride_bn + k_packed[None, :] * stride_bk
            b_packed = tl.load(
                b_ptrs,
                mask=(offs_n[:, None] < N)
                & (k_packed[None, :] < K_packed)
                & mask_k[None, :],
                other=0,
            ).to(tl.uint8)

            low = b_packed & 0xF
            high = (b_packed >> 4) & 0xF
            is_low_b = is_low[None, :]
            val = tl.where(is_low_b, low, high)
            val = val.to(tl.int8)
            val = tl.where(val > 7, val - 16, val)
            b_int4 = val.to(tl.float16)

            sw_ptrs = Sw_ptr + offs_n * stride_sw_n
            sw = tl.load(sw_ptrs, mask=offs_n < N, other=1.0)
            b_deq = b_int4 * sw[:, None]

            acc += tl.dot(a, tl.trans(b_deq))

        c = acc.to(tl.float16 if OUT_DTYPE_FP16 else tl.bfloat16)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def w4a4_mm(
    A: torch.Tensor,
    Bq_packed: torch.Tensor,
    Sw: torch.Tensor,
    Sa: torch.Tensor,
    *,
    K_orig: int,
    out_dtype: torch.dtype = torch.float16,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> Optional[torch.Tensor]:
    """
    Compute Y = (Q4(A, Sa) * Sa) @ (Q4(B, Sw) * Sw)^T using Triton if available.

    - A: [M, K] activations (fp16/bf16)
    - Bq_packed: [N, K_packed] int8 packed weights (2 weights per byte)
    - Sw: [N, 1] fp16 weight scales
    - Sa: [M, 1] fp16 activation scales
    - K_orig: original K dimension before packing.

    Returns:
        Y: [M, N] tensor or None if Triton unavailable / CPU.
    """
    if not is_triton_available():
        return None
    if not A.is_cuda:
        return None

    import triton  # type: ignore

    assert A.dim() == 2
    assert Bq_packed.dim() == 2
    assert Sw.dim() in (1, 2)
    assert Sa.dim() in (1, 2)

    M, K = A.shape
    N, Kp = Bq_packed.shape
    assert K == K_orig, "A K dim must equal original K"

    # Move tensors to same device
    dev = A.device
    if Bq_packed.device != dev:
        Bq_packed = Bq_packed.to(dev, non_blocking=True)
    if Sw.device != dev:
        Sw = Sw.to(dev, dtype=torch.float16, non_blocking=True)
    if Sa.device != dev:
        Sa = Sa.to(dev, dtype=torch.float16, non_blocking=True)

    if Sw.dim() == 1:
        Sw = Sw.view(N, 1)
    if Sa.dim() == 1:
        Sa = Sa.view(M, 1)

    OUT_DTYPE_FP16 = out_dtype == torch.float16
    Y = torch.empty((M, N), device=dev, dtype=out_dtype)

    grid = (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    _w4a4_mm_kernel[grid](
        A,
        Bq_packed,
        Sw,
        Sa,
        Y,
        M,
        N,
        K_orig,
        Kp,
        A.stride(0),
        A.stride(1),
        Bq_packed.stride(0),
        Bq_packed.stride(1),
        Sw.stride(0),
        Sa.stride(0),
        Y.stride(0),
        Y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        OUT_DTYPE_FP16=OUT_DTYPE_FP16,
        num_warps=4,
        num_stages=2,
    )

    return Y


def w4a16_mm(
    A: torch.Tensor,
    Bq_packed: torch.Tensor,
    Sw: torch.Tensor,
    *,
    K_orig: int,
    out_dtype: torch.dtype = torch.float16,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> Optional[torch.Tensor]:
    """
    Weight-only 4-bit GEMM:
    Y = A @ (Q4(B, Sw) * Sw)^T

    - A: [M, K] activations (fp16/bf16), 不做激活量化
    - Bq_packed: [N, K_packed] int8 packed weights (2 weights per byte)
    - Sw: [N, 1] fp16 weight scales
    - K_orig: original K dimension before packing.
    """
    if not is_triton_available():
        return None
    if not A.is_cuda:
        return None

    import triton  # type: ignore

    assert A.dim() == 2
    assert Bq_packed.dim() == 2
    assert Sw.dim() in (1, 2)

    M, K = A.shape
    N, Kp = Bq_packed.shape
    assert K == K_orig, "A K dim must equal original K"

    dev = A.device
    if Bq_packed.device != dev:
        Bq_packed = Bq_packed.to(dev, non_blocking=True)
    if Sw.device != dev:
        Sw = Sw.to(dev, dtype=torch.float16, non_blocking=True)

    if Sw.dim() == 1:
        Sw = Sw.view(N, 1)

    OUT_DTYPE_FP16 = out_dtype == torch.float16
    Y = torch.empty((M, N), device=dev, dtype=out_dtype)

    grid = (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    _w4a16_mm_kernel[grid](
        A,
        Bq_packed,
        Sw,
        Y,
        M,
        N,
        K_orig,
        Kp,
        A.stride(0),
        A.stride(1),
        Bq_packed.stride(0),
        Bq_packed.stride(1),
        Sw.stride(0),
        Y.stride(0),
        Y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        OUT_DTYPE_FP16=OUT_DTYPE_FP16,
        num_warps=4,
        num_stages=2,
    )

    return Y


