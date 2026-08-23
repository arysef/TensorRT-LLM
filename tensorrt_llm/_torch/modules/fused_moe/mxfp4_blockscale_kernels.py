# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenAI-Triton kernels for packed-MXFP4 experts with block-scaled FP8 acts.

The weights are the checkpoint's own packed MXFP4: two E2M1 nibbles per byte
along K, low nibble first, with one UE8M0 exponent per 32 logical K values.
The activations are quantized *per token and per 128 K values* to FP8 E4M3
with a power-of-two scale, which is the W4A8 contract a producer that wrote
``scale_fmt="ue8m0"`` used when it trained and evaluated the checkpoint.

Every arithmetic decision here mirrors one in the reference implementation
rather than approximating it, because the whole point of the path is to
reproduce it:

* the scale is ``2 ** ceil(log2(amax / 448))`` computed by exponent
  arithmetic, so an ``amax`` that is already a power of two keeps its own
  exponent instead of being pushed up by a floating-point ``log2``;
* ``amax`` is floored at ``1e-4`` before the scale is taken, so an all-zero
  block quantizes to zeros instead of dividing by zero;
* the GEMM walks K in groups of 32 -- one weight-scale group -- accumulating
  each group's dot product in FP32 and scaling it by ``act_scale * w_scale``
  before adding it to a second FP32 accumulator;
* both GEMM outputs are rounded to BF16, because the reference GEMM returns
  the default dtype and the SwiGLU/FC2 stages consume that rounded value.

Nothing here is model-specific: the entry points take packed weights, scales
and a routing plan, and know nothing about which checkpoint produced them.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

#: FP8 E4M3 saturation bound, and the reciprocal the reference multiplies by.
FP8_MAX = 448.0
#: Activation block: one FP8 scale per this many K values.
ACT_BLOCK_SIZE = 128
#: Weight group: one UE8M0 exponent per this many logical K values.
WEIGHT_GROUP_SIZE = 32
#: Floor applied to a block's absolute maximum before the scale is derived.
AMAX_FLOOR = 1e-4

# Triton only lets a kernel read a global that is already a ``constexpr``, so
# the three constants the kernels use are re-exported in that form. They are
# defined from the plain values above rather than duplicated, so the Python
# entry points and the kernels cannot drift apart.
_TL_FP8_MAX = tl.constexpr(FP8_MAX)
_TL_FP8_MAX_INV = tl.constexpr(1.0 / FP8_MAX)
_TL_AMAX_FLOOR = tl.constexpr(AMAX_FLOOR)
_TL_ACT_BLOCK_SIZE = tl.constexpr(ACT_BLOCK_SIZE)


@triton.jit
def _round_scale(amax, max_inv):
    """``2 ** ceil(log2(amax * max_inv))`` by IEEE-754 exponent arithmetic.

    The reference computes this with bit tricks rather than ``log2``/``ceil``
    (``fast_log2_ceil`` + ``fast_pow2``). Reproduced rather than approximated:
    a floating-point ``log2`` of an exact power of two can land a hair above
    the integer and push ``ceil`` one exponent too high, which changes every
    FP8 code in the block.
    """
    scaled = (amax * max_inv).to(tl.float32)
    bits = scaled.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF).to(tl.int32) - 127
    mantissa = bits & 0x7FFFFF
    ceil_log2 = exponent + tl.where(mantissa != 0, 1, 0)
    return (((ceil_log2 + 127) << 23).to(tl.uint32)).to(tl.float32, bitcast=True)


@triton.jit
def _decode_e2m1(nibble):
    """Decode E2M1 nibbles to FP32, branchlessly.

    Bit 3 is the sign, bits 2-1 the exponent, bit 0 the mantissa. Subnormals
    (``exponent == 0``) are ``mantissa * 0.5``; everything else is
    ``(1 + mantissa/2) * 2**(exponent-1)``. ``exp2`` of a small integer is
    exact, so the decode is exact.
    """
    sign = (nibble >> 3) & 1
    exponent = ((nibble >> 1) & 3).to(tl.int32)
    mantissa = (nibble & 1).to(tl.float32)
    magnitude = tl.where(
        exponent == 0,
        mantissa * 0.5,
        (1.0 + 0.5 * mantissa) * tl.exp2((exponent - 1).to(tl.float32)),
    )
    return tl.where(sign == 1, -magnitude, magnitude)


@triton.jit
def _ue8m0_to_f32(bits):
    """A UE8M0 byte is a float32 exponent field with an empty mantissa."""
    return ((bits.to(tl.uint32)) << 23).to(tl.float32, bitcast=True)


# ---------------------------------------------------------------------------
# Activation quantization.
# ---------------------------------------------------------------------------


@triton.jit
def _quant_rows_kernel(
    X,
    Y,
    S,
    stride_xm,
    stride_ym,
    stride_sm,
    num_rows,
    BLOCK: tl.constexpr,
):
    """One program per (row, K-block): quantize BLOCK values to FP8 E4M3."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    if row >= num_rows:
        return
    offs = block * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_xm + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), _TL_AMAX_FLOOR)
    scale = _round_scale(amax, _TL_FP8_MAX_INV)
    q = tl.clamp(x / scale, -_TL_FP8_MAX, _TL_FP8_MAX)
    tl.store(Y + row * stride_ym + offs, q.to(tl.float8e4nv))
    tl.store(S + row * stride_sm + block, scale)


def quantize_blockwise_ue8m0(
    x: torch.Tensor, block_size: int = ACT_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dim of ``x`` to FP8 E4M3 with power-of-two scales.

    Returns ``(q, scale)`` with ``q`` the FP8 codes and ``scale`` one FP32
    power of two per ``block_size`` values, i.e. ``x ~= q * scale``.
    """
    assert x.dim() == 2, f"expected a 2-D activation, got {tuple(x.shape)}"
    rows, cols = x.shape
    assert cols % block_size == 0, f"{cols} is not a multiple of {block_size}"
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, cols // block_size), dtype=torch.float32, device=x.device)
    if rows == 0:
        return q, scale
    _quant_rows_kernel[(rows, cols // block_size)](
        x,
        q,
        scale,
        x.stride(0),
        q.stride(0),
        scale.stride(0),
        rows,
        BLOCK=block_size,
    )
    return q, scale


@triton.jit
def _swiglu_quant_kernel(
    FC1,
    ROUTE,
    Y,
    S,
    stride_fc1,
    stride_ym,
    stride_sm,
    inter_size,
    swiglu_limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Clamped SwiGLU, routing weight, BF16 round-trip, then FP8 quantization.

    Fused because the reference's intermediate never leaves its expert: it is
    produced in FP32, multiplied by the routing weight, rounded to BF16 by the
    cast in ``Expert.forward``, and only then quantized by the FC2 ``linear``.
    Splitting those steps into separate torch ops would materialize two
    tensors the size of the padded token-expert grid for no numerical gain.

    ``FC1`` holds ``[up | gate]``: the loader concatenates ``w3`` before
    ``w1``, matching the fused-MoE weight layout, so the first half is the
    linear branch and the second half is the one SiLU is applied to.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    up = tl.load(FC1 + row * stride_fc1 + offs).to(tl.float32)
    gate = tl.load(FC1 + row * stride_fc1 + inter_size + offs).to(tl.float32)
    if HAS_LIMIT:
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)
        gate = tl.minimum(gate, swiglu_limit)
    h = (gate * tl.sigmoid(gate)) * up
    h = h * tl.load(ROUTE + row)
    # The reference casts the FP32 intermediate back to the activation dtype
    # before FC2 quantizes it, so the quantizer sees BF16-rounded values.
    h = h.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(h)), _TL_AMAX_FLOOR)
    scale = _round_scale(amax, _TL_FP8_MAX_INV)
    q = tl.clamp(h / scale, -_TL_FP8_MAX, _TL_FP8_MAX)
    tl.store(Y + row * stride_ym + offs, q.to(tl.float8e4nv))
    tl.store(S + row * stride_sm + block, scale)


def swiglu_and_quantize(
    fc1_out: torch.Tensor,
    routing_weight: torch.Tensor,
    inter_size: int,
    swiglu_limit: Optional[float],
    block_size: int = ACT_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``silu(gate) * up * routing_weight`` then blockwise FP8 quantization."""
    rows = fc1_out.shape[0]
    assert fc1_out.shape[1] == 2 * inter_size
    assert inter_size % block_size == 0
    q = torch.empty((rows, inter_size), dtype=torch.float8_e4m3fn, device=fc1_out.device)
    scale = torch.empty(
        (rows, inter_size // block_size), dtype=torch.float32, device=fc1_out.device
    )
    if rows == 0:
        return q, scale
    _swiglu_quant_kernel[(rows, inter_size // block_size)](
        fc1_out,
        routing_weight,
        q,
        scale,
        fc1_out.stride(0),
        q.stride(0),
        scale.stride(0),
        inter_size,
        0.0 if swiglu_limit is None else float(swiglu_limit),
        HAS_LIMIT=swiglu_limit is not None,
        BLOCK=block_size,
    )
    return q, scale


# ---------------------------------------------------------------------------
# Grouped W4A8 GEMM.
# ---------------------------------------------------------------------------


@triton.jit
def _moe_w4a8_gemm_kernel(
    AQ,
    AS,
    AROW,
    BQ,
    BS,
    C,
    EXPERT_IDS,
    stride_aq_m,
    stride_as_m,
    stride_bq_e,
    stride_bq_n,
    stride_bs_e,
    stride_bs_n,
    stride_c_m,
    K,
    N,
    local_experts,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``C[m, n] = sum_k A[row(m), k] * B[expert(m), n, k]`` with block scales.

    K is walked in groups of 32 -- one weight-scale group -- so each group's
    dot product can be scaled by its own pair of scales before it joins the
    output accumulator. That is the reference kernel's structure, and keeping
    it means the two disagree only by FP32 accumulation order rather than by
    where the scales are applied.

    ``AROW`` maps a padded output row to its activation row, or to a negative
    value for a padding slot; ``EXPERT_IDS`` names one local expert per
    ``BLOCK_M`` rows, or ``local_experts`` for a block that owns nothing.
    Both cases store zeros, so the caller never reads uninitialized memory.
    """
    GROUP_K: tl.constexpr = 32
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_row = tl.load(AROW + offs_m)
    row_valid = a_row >= 0
    a_row = tl.maximum(a_row, 0)

    expert = tl.load(EXPERT_IDS + pid_m)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # A block with no live row still has to store its zeros, but it must not
    # walk K to produce them. Every expert's row count is padded up to
    # BLOCK_M, so at single-token decode almost every block is empty and this
    # is the difference between touching one expert's weights and all of them.
    live = tl.max(row_valid.to(tl.int32)) > 0
    if live and expert < local_experts:
        bq_row = BQ + expert.to(tl.int64) * stride_bq_e + offs_n[:, None] * stride_bq_n
        bs_row = BS + expert.to(tl.int64) * stride_bs_e + offs_n * stride_bs_n
        offs_byte = tl.arange(0, GROUP_K // 2)
        offs_k = tl.arange(0, GROUP_K)
        for k0 in range(0, K, GROUP_K):
            a = tl.load(
                AQ + a_row[:, None] * stride_aq_m + (k0 + offs_k)[None, :],
                mask=row_valid[:, None],
                other=0.0,
            )
            packed = tl.load(bq_row + (k0 // 2 + offs_byte)[None, :])
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            # Nibbles are packed along K with the low nibble first, so the
            # logical row is the two halves interleaved, not concatenated.
            nibbles = tl.interleave(low, high)
            b = _decode_e2m1(nibbles).to(tl.float8e4nv)

            group = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
            act_scale = tl.load(
                AS + a_row * stride_as_m + k0 // _TL_ACT_BLOCK_SIZE, mask=row_valid, other=0.0
            )
            weight_scale = _ue8m0_to_f32(tl.load(bs_row + k0 // GROUP_K))
            acc += group * act_scale[:, None] * weight_scale[None, :]

    tl.store(
        C + offs_m[:, None] * stride_c_m + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=offs_n[None, :] < N,
    )


def moe_w4a8_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    a_row: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    local_experts: int,
    block_m: int,
    block_n: int = 128,
) -> torch.Tensor:
    """Run the grouped W4A8 GEMM and return a BF16 ``[rows, N]`` result."""
    rows = a_row.shape[0]
    num_experts, n, packed_k = b_packed.shape
    k = packed_k * 2
    assert a_q.shape[1] == k, f"activation K {a_q.shape[1]} != weight K {k}"
    assert b_scale.shape == (num_experts, n, k // WEIGHT_GROUP_SIZE), (
        f"weight scale {tuple(b_scale.shape)} does not match one UE8M0 exponent "
        f"per {WEIGHT_GROUP_SIZE} of {k} K values for {n} rows"
    )
    assert rows % block_m == 0, f"{rows} padded rows is not a multiple of {block_m}"
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=a_q.device)
    _moe_w4a8_gemm_kernel[(rows // block_m, triton.cdiv(n, block_n))](
        a_q,
        a_scale,
        a_row,
        b_packed,
        b_scale,
        out,
        expert_ids,
        a_q.stride(0),
        a_scale.stride(0),
        b_packed.stride(0),
        b_packed.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        out.stride(0),
        k,
        n,
        local_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=3,
    )
    return out


# ---------------------------------------------------------------------------
# Deterministic combine.
# ---------------------------------------------------------------------------


@triton.jit
def _combine_kernel(
    FC2,
    PAIR_POS,
    OUT,
    stride_fc2,
    stride_out,
    top_k,
    hidden_size,
    BLOCK_N: tl.constexpr,
):
    """Sum a token's ``top_k`` expert rows in FP32, in a fixed order.

    A scatter-add would be shorter, but the CUDA implementations of both
    ``index_add_`` and ``atomic_add`` sum in nondeterministic order, so two
    identical requests could differ in the last bit. Greedy decoding has to be
    reproducible, so the reduction runs per token over a fixed ``top_k`` loop
    instead.
    """
    token = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < hidden_size
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(top_k):
        row = tl.load(PAIR_POS + token * top_k + k)
        acc += tl.load(FC2 + row * stride_fc2 + offs_n, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + token * stride_out + offs_n, acc, mask=mask)


def combine_expert_rows(
    fc2_out: torch.Tensor,
    pair_pos: torch.Tensor,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
) -> torch.Tensor:
    """Reduce sorted per-pair FC2 rows back to one FP32 row per token."""
    out = torch.empty((num_tokens, hidden_size), dtype=torch.float32, device=fc2_out.device)
    if num_tokens == 0:
        return out
    block_n = 256
    _combine_kernel[(num_tokens, triton.cdiv(hidden_size, block_n))](
        fc2_out,
        pair_pos,
        out,
        fc2_out.stride(0),
        out.stride(0),
        top_k,
        hidden_size,
        BLOCK_N=block_n,
    )
    return out
