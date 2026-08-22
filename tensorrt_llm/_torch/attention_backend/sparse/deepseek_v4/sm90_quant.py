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
"""Source-faithful SM90 numerics for the DeepSeek-V4 Compressor and Indexer.

Two things live here: the activation-quantization simulation the checkpoint's
QAT makes semantic, and the Indexer's index-score reduction, whose *dtype
chain* is equally semantic because its output feeds a discrete top-k.

The checkpoint was quantization-aware trained, so three of its activation
quantizations are *semantics*, not storage decisions --- the model expects to
see the rounded values:

``inference/model.py``, ``Attention.forward``::

    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)

``inference/model.py``, ``Compressor.forward``::

    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)  # main compressor
    fp4_act_quant(kv, fp4_block_size, True)  # indexer compressor

``inference/model.py``, ``Indexer.forward``::

    fp4_act_quant(q, fp4_block_size, True)

All four calls pass ``inplace=True``, i.e. they quantize *and dequantize* and
leave the tensor in BF16. Nothing downstream reads a narrow container; what
changes is the numeric value.

TensorRT-LLM has no cache preset that expresses "round-trip through FP8, keep
BF16" (``KVCacheDtype.NONE`` stores the unrounded row), and the one native
kernel that produces true E2M1 codes cannot run here: ``packE2M1x2`` in
``compressorKernels.cu`` is guarded by ``__CUDA_ARCH__ >= 1000`` and returns
``0`` below it, so ``KVCacheDtype.MXFP4_BLOCKWISE`` silently writes zeros on
Hopper. This module supplies both in OpenAI Triton, which is what the plan
enumerates for the SM90 indexer ("software E2M1/UE8M0 ... within
``DeepseekV4Indexer``").

The indexer keeps the existing ``FP8_BLOCKWISE`` cache *container* rather than
switching the pool to packed FP4. That is a measured choice, not a shortcut: an
FP4-simulated row is **bit-exact** through an FP8 E4M3 round trip whose scale is
a power of two, because every E2M1 level (0, ±0.5, 1, 1.5, 2, 3, 4, 6) needs at
most two significand bits and E4M3 has four. So the container is lossless for
these values, and the whole existing DeepGEMM chunking, paging and top-k
machinery keeps working unchanged. The C++ scatter's own FP8 scale is
``amax / 448``, which is *not* a power of two and does lose bits, which is why
the SM90 indexer writes its cache rows from
:func:`quantize_indexer_rows_to_fp8_cache` here instead.

What the DeepGEMM *logits* kernels cannot supply is the reduction dtype.
``inference/model.py``, ``Indexer.forward``::

    index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[...])
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

runs under ``torch.set_default_dtype(torch.bfloat16)`` with ``weights_proj``
declared ``dtype=torch.bfloat16``, so the score is rounded to BF16 three times.
``fp8_mqa_logits`` and ``fp8_fp4_mqa_logits`` both reduce in FP32 --- finer, but
different, and top-k is a discrete decision: measured on identical FP4-simulated
inputs at 2304 tokens / 576 slots / top-k 512, the FP32 chain selects a
different slot set than the source on 33 of 253 deciding rows.
:func:`source_index_scores` and :func:`source_index_scores_paged` are the
plan's "paged MQA logits in OpenAI Triton within ``DeepseekV4Indexer``" and
supply that chain for the context and cached-decode phases respectively.

They reproduce it exactly rather than approximately, which is possible only
because of what the operands are. Both sides of the GEMM hold E2M1 levels
scaled by a power of two, so every product carries at most four significant
bits and the FP32 accumulation of a 128-wide dot is exact --- independent of
the order the hardware sums in, hence identical to what cuBLAS computes for the
source's ``einsum``. Dequantizing after the dot rather than before is exact for
the same reason: the container scales are powers of two, so they commute with
both the FP32 sum and the BF16 rounding.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: E2M1 magnitude levels and the source's block size for ``fp4_act_quant``.
FP4_MAX = 6.0
FP4_BLOCK = 32
#: ``act_quant``'s block size in the two ``model.py`` call sites above.
FP8_BLOCK = 64
FP8_MAX = 448.0

# Triton only reads module globals that are already `tl.constexpr`, so the
# device-side copies are declared separately from the Python-side constants
# above rather than duplicating the numbers at every use site.
_FP4_MAX = tl.constexpr(FP4_MAX)
_FP8_MAX = tl.constexpr(FP8_MAX)
#: Smallest positive normal FP32 times the FP4 maximum: the source's
#: ``T.max(amax, 6 * (2 ** -126))`` floor in ``fp4_quant_kernel``.
_FP4_MIN_AMAX = tl.constexpr(FP4_MAX * 1.1754943508222875e-38)
#: ``act_quant_kernel``'s own floor, ``T.max(amax, 1e-4)``
#: (``inference/kernel.py:79``). It is not a divide-by-zero guard: at seven
#: orders of magnitude above FP32's smallest normal it deliberately pins the
#: scale for any low-magnitude block, so everything below ~2**-22 * 0.5
#: flushes to zero instead of being preserved at a finer scale. Dropping it
#: makes TensorRT-LLM *more* precise than the checkpoint, which is still a
#: divergence.
FP8_MIN_AMAX = 1e-4
_FP8_MIN_AMAX = tl.constexpr(FP8_MIN_AMAX)


@triton.jit
def _pow2_scale(amax, max_inv):
    """``fast_round_scale``: ``2 ** ceil(log2(amax * max_inv))``.

    ``inference/kernel.py`` computes this with IEEE-754 exponent bit tricks;
    for finite positive inputs the float form is the same value.

    Deliberately *no* floor of its own, matching ``fast_round_scale``: each
    caller applies the floor its own source kernel applies, so an omitted floor
    shows up as a wrong answer instead of being absorbed here. An earlier
    version clamped internally at ``1e-38``, which silently reproduced the FP4
    kernel's ``6 * 2 ** -126`` guard and made removing that guard undetectable.
    Callers must pass a strictly positive ``amax``.
    """
    # Pinned to fp32: a bare Python float literal widens the expression to
    # fp64 in Triton, and fp64 has no E4M3 conversion.
    return tl.exp2(tl.ceil(tl.log2((amax * max_inv).to(tl.float32)))).to(tl.float32)


@triton.jit
def _round_e2m1(a):
    """Round a non-negative magnitude to the nearest E2M1 level, ties to even.

    Codes 0..7 are (0, 0.5, 1, 1.5, 2, 3, 4, 6); the midpoints are 0.25, 0.75,
    1.25, 1.75, 2.5, 3.5 and 5.0. The comparison operators below encode
    ties-to-even directly: ``<=`` on a midpoint keeps the even code, ``<``
    pushes to it. Ties are common rather than negligible here because the
    inputs are BF16 values divided by a power of two, i.e. dyadic.
    """
    return tl.where(
        a <= 0.25,
        0.0,
        tl.where(
            a < 0.75,
            0.5,
            tl.where(
                a <= 1.25,
                1.0,
                tl.where(
                    a < 1.75,
                    1.5,
                    tl.where(a <= 2.5, 2.0, tl.where(a < 3.5, 3.0, tl.where(a <= 5.0, 4.0, 6.0))),
                ),
            ),
        ),
    )


@triton.jit
def _fp4_quant_dequant(v):
    """``fp4_act_quant(x, 32, inplace=True)`` on a ``[groups, 32]`` FP32 tile."""
    amax = tl.max(tl.abs(v), axis=1, keep_dims=True)
    amax = tl.maximum(amax, _FP4_MIN_AMAX)
    scale = _pow2_scale(amax, 1.0 / _FP4_MAX)
    a = tl.minimum(tl.abs(v) / scale, _FP4_MAX).to(tl.float32)
    return (tl.where(v < 0, -1.0, 1.0) * _round_e2m1(a) * scale).to(tl.float32)


@triton.jit
def _fp8_quant_dequant(v):
    """``act_quant(x, block, "ue8m0", ..., inplace=True)`` on a ``[groups, block]`` tile."""
    amax = tl.max(tl.abs(v), axis=1, keep_dims=True)
    amax = tl.maximum(amax, _FP8_MIN_AMAX)
    scale = _pow2_scale(amax, 1.0 / _FP8_MAX)
    q = tl.clamp(v / scale, -_FP8_MAX, _FP8_MAX).to(tl.float32).to(tl.float8e4nv)
    return q.to(tl.float32) * scale


# ---------------------------------------------------------------------------
# Dense helpers: the indexer's Q and the main attention's window latent row.
# ---------------------------------------------------------------------------


@triton.jit
def _dense_fp4_kernel(
    x_ptr, stride_r, NGROUP: tl.constexpr, NGROUP_P2: tl.constexpr, GROUP: tl.constexpr
):
    row = tl.program_id(0)
    groups = tl.arange(0, NGROUP_P2)[:, None]
    live = groups < NGROUP
    ptr = x_ptr + row * stride_r + groups * GROUP + tl.arange(0, GROUP)[None, :]
    v = _fp4_quant_dequant(tl.load(ptr, mask=live, other=0.0).to(tl.float32))
    tl.store(ptr, v.to(x_ptr.dtype.element_ty), mask=live)


def fp4_quant_dequant_(x: torch.Tensor) -> torch.Tensor:
    """In-place ``fp4_act_quant(x, 32, inplace=True)`` over the last dimension."""
    assert x.is_cuda and x.stride(-1) == 1, "fp4 simulation needs unit-stride rows"
    width = x.shape[-1]
    assert width % FP4_BLOCK == 0, f"width {width} is not a multiple of {FP4_BLOCK}"
    flat = x.reshape(-1, width)
    if flat.shape[0] == 0:
        return x
    _dense_fp4_kernel[(flat.shape[0],)](
        flat,
        flat.stride(0),
        NGROUP=width // FP4_BLOCK,
        NGROUP_P2=triton.next_power_of_2(width // FP4_BLOCK),
        GROUP=FP4_BLOCK,
        num_warps=4,
    )
    return x


@triton.jit
def _dense_fp8_kernel(
    x_ptr, stride_r, NGROUP: tl.constexpr, NGROUP_P2: tl.constexpr, GROUP: tl.constexpr
):
    row = tl.program_id(0)
    groups = tl.arange(0, NGROUP_P2)[:, None]
    live = groups < NGROUP
    ptr = x_ptr + row * stride_r + groups * GROUP + tl.arange(0, GROUP)[None, :]
    v = _fp8_quant_dequant(tl.load(ptr, mask=live, other=0.0).to(tl.float32))
    tl.store(ptr, v.to(x_ptr.dtype.element_ty), mask=live)


def source_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``RMSNorm.forward`` from ``inference/model.py``, rounding where it rounds.

    The checkpoint's norm is

        x = x.float(); var = x.square().mean(-1, keepdim=True)
        return (self.weight * x * torch.rsqrt(var + eps)).to(dtype)

    -- FP32 throughout, an FP32 weight, and exactly one rounding, at the end.
    TensorRT-LLM's :class:`RMSNorm` instead computes
    ``self.weight * hidden_states.to(input_dtype)``: it rounds the normalised
    value to BF16 *before* multiplying, and multiplies by a BF16 weight. Both
    are reasonable norms; they are not the same numbers, and the compressor
    feeds a blockwise-64 FP8 quantiser whose levels are ~6% apart, so a value
    near a decision boundary moves a whole FP8 step. Measured on the real
    checkpoint at layer 3, driving this chain with TensorRT-LLM's rounding
    order scored rel_max_abs 7.4e-02 against the source's cached rows, while
    the order below is bit-exact.

    The weight is upcast rather than assumed FP32: the checkpoint stores it in
    BF16 and the source widens it at load, so ``.float()`` here reproduces the
    source's parameter exactly either way.
    """
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    return (weight.float() * xf * torch.rsqrt(var + eps)).to(dtype)


def fp8_quant_dequant_(x: torch.Tensor, block_size: int = FP8_BLOCK) -> torch.Tensor:
    """In-place ``act_quant(x, block_size, "ue8m0", ..., inplace=True)``.

    ``x`` is typically a *slice* of the latent row (the source quantizes only
    the non-RoPE prefix), so the last dimension may be shorter than the row
    stride; that is why the stride is read rather than assumed.
    """
    assert x.is_cuda and x.stride(-1) == 1, "fp8 simulation needs unit-stride rows"
    width = x.shape[-1]
    assert width % block_size == 0, f"width {width} is not a multiple of {block_size}"
    # A column slice keeps its parent's row stride, so it cannot be reshaped;
    # it is already 2-D and the kernel reads the stride anyway.
    flat = x if x.dim() == 2 else x.reshape(-1, width)
    if flat.shape[0] == 0:
        return x
    _dense_fp8_kernel[(flat.shape[0],)](
        flat,
        flat.stride(0),
        NGROUP=width // block_size,
        NGROUP_P2=triton.next_power_of_2(width // block_size),
        GROUP=block_size,
        num_warps=4,
    )
    return x


# ---------------------------------------------------------------------------
# Paged helpers: the compressed rows the fused postprocess+scatter just wrote.
# ---------------------------------------------------------------------------


@triton.jit
def _owning_batch(cu_ptr, token_idx, bsz, BSZ_P2: tl.constexpr):
    """Batch that owns ``token_idx``, from the exclusive prefix sum ``cu``.

    The C++ scatter binary-searches ``cu_kv_comp``; the same answer is one
    vectorized comparison here, which keeps the kernel branch-free and
    graph-safe.
    """
    slots = tl.arange(0, BSZ_P2)
    cu = tl.load(cu_ptr + slots + 1, mask=slots < bsz, other=2**30)
    return tl.sum(tl.where(cu <= token_idx, 1, 0).to(tl.int32))


@triton.jit
def _compressed_cache_write_kernel(
    src_ptr,
    cache_ptr,
    mask_ptr,
    cu_ptr,
    num_outputs_ptr,
    start_pos_ptr,
    block_table_ptr,
    src_stride,
    bt_stride_r,
    block_stride,
    tokens_per_block,
    max_blocks,
    bsz,
    HEAD_DIM: tl.constexpr,
    NGROUP: tl.constexpr,
    NGROUP_P2: tl.constexpr,
    GROUP: tl.constexpr,
    ROPE_P2: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BSZ_P2: tl.constexpr,
):
    """Write one already-postprocessed compressed row into the paged cache.

    The source ends its compressor with ``act_quant(kv[..., :-rd], 64, ...,
    inplace=True)`` -- a blockwise-64 FP8 round trip that stays BF16 -- applied
    to the non-RoPE prefix only, and then stores the row. Both halves are done
    here, so the caller hands over the row exactly as the source's
    ``self.norm -> apply_rotary_emb`` produced it and this kernel supplies the
    quantisation and the paging.
    """
    token_idx = tl.program_id(0)
    if tl.load(mask_ptr + token_idx) == 0:
        return
    batch = _owning_batch(cu_ptr, token_idx, bsz, BSZ_P2)
    local = token_idx - tl.load(cu_ptr + batch)
    if local >= tl.load(num_outputs_ptr + batch):
        return

    cache_pos = tl.load(start_pos_ptr + batch) + local
    logical = cache_pos // tokens_per_block
    if logical >= max_blocks:
        return
    phys = tl.load(block_table_ptr + batch * bt_stride_r + logical)
    if phys < 0:
        return

    row = phys.to(tl.int64) * block_stride + (cache_pos % tokens_per_block) * HEAD_DIM
    groups = tl.arange(0, NGROUP_P2)[:, None]
    live = groups < NGROUP
    offs = groups * GROUP + tl.arange(0, GROUP)[None, :]
    src = src_ptr + token_idx.to(tl.int64) * src_stride
    v = _fp8_quant_dequant(tl.load(src + offs, mask=live, other=0.0).to(tl.float32))
    tl.store(cache_ptr + row + offs, v.to(cache_ptr.dtype.element_ty), mask=live)

    # The RoPE suffix is stored verbatim: the source quantises only the prefix.
    rope = tl.arange(0, ROPE_P2)
    rope_live = rope < ROPE_DIM
    nope = NGROUP * GROUP
    tl.store(
        cache_ptr + row + nope + rope,
        tl.load(src + nope + rope, mask=rope_live, other=0.0),
        mask=rope_live,
    )


def write_source_compressed_rows(
    rows: torch.Tensor,
    kv_cache: torch.Tensor,
    compressed_mask: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    num_comp_tokens: torch.Tensor,
    start_pos: torch.Tensor,
    block_table: torch.Tensor,
    *,
    total_tokens: int,
    tokens_per_block: int,
    nope_dim: int,
    block_size: int = FP8_BLOCK,
) -> None:
    """Scatter postprocessed compressed rows, FP8-simulating the prefix.

    Addresses the destination exactly as ``postProcessScatterKernel`` does ---
    owning batch from ``cu_new_comp_kv``, ``start_pos[batch] + local`` as the
    compressed slot, then the block table --- so a row is written if and only
    if the native scatter would have written it. That equality is the reason
    this can replace the fused kernel's store rather than sit beside it.
    """
    if total_tokens == 0:
        return
    assert kv_cache.dim() == 3, f"kv_cache must be [blocks, tokens, dim], got {kv_cache.shape}"
    head_dim = kv_cache.shape[-1]
    assert nope_dim % block_size == 0, f"nope_dim {nope_dim} is not a multiple of {block_size}"
    assert nope_dim <= head_dim
    assert rows.shape[0] >= total_tokens and rows.shape[-1] == head_dim, (
        f"rows must be [>={total_tokens}, {head_dim}], got {tuple(rows.shape)}"
    )
    assert rows.dtype == kv_cache.dtype, (
        f"row dtype {rows.dtype} does not match the compressed cache {kv_cache.dtype}"
    )
    assert rows.stride(-1) == 1, "compressed rows must be unit-stride"
    bsz = num_comp_tokens.shape[0]
    rope_dim = head_dim - nope_dim
    _compressed_cache_write_kernel[(total_tokens,)](
        rows,
        kv_cache,
        compressed_mask,
        cu_new_comp_kv,
        num_comp_tokens,
        start_pos,
        block_table,
        rows.stride(0),
        block_table.stride(0),
        kv_cache.stride(0),
        tokens_per_block,
        block_table.shape[1],
        bsz,
        HEAD_DIM=head_dim,
        NGROUP=nope_dim // block_size,
        NGROUP_P2=triton.next_power_of_2(nope_dim // block_size),
        GROUP=block_size,
        ROPE_P2=triton.next_power_of_2(max(rope_dim, 1)),
        ROPE_DIM=rope_dim,
        BSZ_P2=triton.next_power_of_2(bsz),
        num_warps=4,
    )


@triton.jit
def _indexer_fp4_to_fp8_cache_kernel(
    src_ptr,
    quant_out_ptr,
    scale_out_ptr,
    cache_u8_ptr,
    cache_f32_ptr,
    mask_ptr,
    cu_ptr,
    num_outputs_ptr,
    start_pos_ptr,
    block_table_ptr,
    src_stride_r,
    bt_stride_r,
    block_stride_bytes,
    tokens_per_block,
    max_blocks,
    bsz,
    HEAD_DIM: tl.constexpr,
    NGROUP: tl.constexpr,
    GROUP: tl.constexpr,
    BSZ_P2: tl.constexpr,
):
    """FP4-simulate one postprocessed indexer row, then store it as FP8 + pow2 scale.

    ``src`` is the ``kv_out`` the fused postprocess produced: normalised,
    RoPE'd and Hadamard-rotated, in BF16 --- exactly the tensor the source
    hands to ``fp4_act_quant``. The FP8 container is lossless because the
    scale is a power of two (see the module docstring), so the bytes in the
    cache dequantize back to the FP4 levels bit-for-bit.

    Cache layout per physical block, matching ``postProcessScatterKernel``::

        [tokens_per_block * HEAD_DIM fp8 bytes][tokens_per_block * 1 fp32 scale]
    """
    token_idx = tl.program_id(0)
    offs = tl.arange(0, NGROUP)[:, None] * GROUP + tl.arange(0, GROUP)[None, :]

    v = tl.load(src_ptr + token_idx * src_stride_r + offs).to(tl.float32)
    # The source stores the dequantized value back into a bf16 tensor. Every
    # E2M1 level times a power of two is exactly representable in bf16, so the
    # cast is information-preserving; it is here for fidelity, not rounding.
    v = _fp4_quant_dequant(v).to(tl.bfloat16).to(tl.float32)
    # Container scale, not a source contract: the checkpoint keeps these rows in
    # BF16 and never picks an FP8 scale for them. Any power of two large enough
    # carries the E2M1 levels exactly, so the only requirement is that an
    # all-zero row (a padded slot) does not produce `2 ** -inf`. The floor is
    # therefore FP32's smallest normal, not `act_quant`'s 1e-4 --- using that
    # here would zero small rows the source keeps.
    scale = _pow2_scale(tl.maximum(tl.max(tl.abs(v)), 1.1754943508222875e-38), 1.0 / _FP8_MAX)
    q = tl.clamp(v / scale, -_FP8_MAX, _FP8_MAX).to(tl.float32).to(tl.float8e4nv)

    tl.store(quant_out_ptr + token_idx * HEAD_DIM + offs, q)
    tl.store(scale_out_ptr + token_idx, scale)

    if tl.load(mask_ptr + token_idx) == 0:
        return
    batch = _owning_batch(cu_ptr, token_idx, bsz, BSZ_P2)
    local = token_idx - tl.load(cu_ptr + batch)
    if local >= tl.load(num_outputs_ptr + batch):
        return

    cache_pos = tl.load(start_pos_ptr + batch) + local
    logical = cache_pos // tokens_per_block
    if logical >= max_blocks:
        return
    phys = tl.load(block_table_ptr + batch * bt_stride_r + logical)
    if phys < 0:
        return

    slot = cache_pos % tokens_per_block
    base = phys.to(tl.int64) * block_stride_bytes
    tl.store(cache_u8_ptr + base + slot * HEAD_DIM + offs, q.to(tl.uint8, bitcast=True))
    # The data region is a whole number of 4-byte words, so the scale region is
    # fp32-aligned and can be addressed in word units.
    tl.store(cache_f32_ptr + (base + tokens_per_block * HEAD_DIM) // 4 + slot, scale)


def quantize_indexer_rows_to_fp8_cache(
    postprocessed: torch.Tensor,
    kv_cache: torch.Tensor,
    compressed_mask: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    num_comp_tokens: torch.Tensor,
    start_pos: torch.Tensor,
    block_table: torch.Tensor,
    *,
    tokens_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-FP4-simulate the indexer's compressed rows and page them as FP8.

    Returns the dense ``(fp8, fp32 scale)`` pair the indexer hands to
    ``fp8_mqa_logits``; the same values are written into the paged indexer K
    cache that ``fp8_paged_mqa_logits`` and the chunked-context gather read.
    """
    total_tokens, head_dim = postprocessed.shape
    assert head_dim % FP4_BLOCK == 0, f"indexer head_dim {head_dim} is not a multiple of 32"
    quant = torch.empty(
        total_tokens, head_dim, dtype=torch.float8_e4m3fn, device=postprocessed.device
    )
    scale = torch.empty(total_tokens, 1, dtype=torch.float32, device=postprocessed.device)
    if total_tokens == 0:
        return quant, scale

    assert kv_cache.dim() == 3, f"kv_cache must be [blocks, tokens, dim], got {kv_cache.shape}"
    assert kv_cache.element_size() == 1, "the indexer FP8 pool is byte-addressed"
    # [data | scale] per token row; one fp32 scale per 128-wide row.
    assert kv_cache.shape[-1] == head_dim + 4, (
        f"indexer pool row is {kv_cache.shape[-1]} bytes, expected {head_dim} + 4"
    )
    block_stride_bytes = kv_cache.stride(0) * kv_cache.element_size()
    storage = kv_cache.view(torch.uint8).reshape(-1)
    bsz = num_comp_tokens.shape[0]
    _indexer_fp4_to_fp8_cache_kernel[(total_tokens,)](
        postprocessed,
        quant,
        scale,
        storage,
        storage.view(torch.float32),
        compressed_mask,
        cu_new_comp_kv,
        num_comp_tokens,
        start_pos,
        block_table,
        postprocessed.stride(0),
        block_table.stride(0),
        block_stride_bytes,
        tokens_per_block,
        block_table.shape[1],
        bsz,
        HEAD_DIM=head_dim,
        NGROUP=head_dim // FP4_BLOCK,
        GROUP=FP4_BLOCK,
        BSZ_P2=triton.next_power_of_2(bsz),
        num_warps=4,
    )
    return quant, scale


# ---------------------------------------------------------------------------
# The Indexer's index-score reduction, in the source's own dtype chain.
# ---------------------------------------------------------------------------

#: Compressed slots scored per program. Unlike the sparse-attention tile size
#: this is *not* part of the numerics --- the reduction runs over heads, not
#: over slots, so every slot is independent and the tile only trades Q re-reads
#: against occupancy.
INDEX_SCORE_BLOCK = 128


def _index_score_block(num_slots: int) -> int:
    return max(16, min(INDEX_SCORE_BLOCK, triton.next_power_of_2(max(num_slots, 1))))


@triton.jit
def _source_index_score(q, k, k_scale, w, H: tl.constexpr, BLOCK_S: tl.constexpr):
    """``sum_h bf16(relu(bf16(q_h . k)) * w_h)``, rounded once more at the end.

    ``q`` is ``[H, D]`` BF16, ``k`` is ``[BLOCK_S, D]`` in the FP8 container and
    ``k_scale`` its ``[BLOCK_S]`` power-of-two scale. Returns ``[BLOCK_S]`` FP32
    holding BF16-exact values.

    ReLU is applied before the BF16 rounding rather than after; the two commute
    because rounding preserves sign, and doing it here also removes the sign
    from the scale multiply so the two can be reordered freely.
    """
    acc = tl.dot(q, tl.trans(k.to(tl.bfloat16)), out_dtype=tl.float32)
    acc = tl.maximum(acc * k_scale[None, :], 0.0)
    # `index_score.relu_() * weights.unsqueeze(-1)`: a BF16 x BF16 product, i.e.
    # the exact product rounded once. Spelling it in FP32 and rounding at the
    # end is the same value and avoids depending on Triton's promotion rules.
    scored = (acc.to(tl.bfloat16).to(tl.float32) * w.to(tl.float32)[:, None]).to(tl.bfloat16)
    return tl.sum(scored.to(tl.float32), axis=0).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _index_scores_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    w_ptr,
    out_ptr,
    ks_ptr,
    ke_ptr,
    q_stride_r,
    w_stride_r,
    out_stride_r,
    num_slots,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    token = tl.program_id(0)
    slots = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    live = slots < num_slots

    heads = tl.arange(0, H)
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + token * q_stride_r + heads[:, None] * D + dims[None, :])
    k = tl.load(k_ptr + slots[:, None] * D + dims[None, :], mask=live[:, None], other=0.0)
    k_scale = tl.load(k_scale_ptr + slots, mask=live, other=0.0)
    w = tl.load(w_ptr + token * w_stride_r + heads)

    row = _source_index_score(q, k, k_scale, w, H, BLOCK_S)

    start = tl.load(ks_ptr + token)
    end = tl.load(ke_ptr + token)
    inside = (slots >= start) & (slots < end)
    tl.store(
        out_ptr + token * out_stride_r + slots, tl.where(inside, row, float("-inf")), mask=live
    )


def source_index_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Context-phase index scores with the source's BF16 reduction.

    Drop-in for ``fp8_mqa_logits``: ``q`` is ``[tokens, heads, dim]`` BF16 (the
    source keeps Q in BF16 --- it only FP4-*simulates* it), ``k``/``k_scale``
    are the FP8 container the compressor produced, ``weights`` is
    ``[tokens, heads]`` BF16, and the result is ``[tokens, slots]`` FP32 with
    ``-inf`` outside each row's ``[ks, ke)`` window. Logits are always cleaned;
    the caller's ``clean_logits`` knob only exists to skip work the DeepGEMM
    kernel would otherwise do in a second pass.
    """
    assert q.dim() == 3 and weights.dim() == 2, (
        f"expected q [tokens, heads, dim] and weights [tokens, heads], got {tuple(q.shape)} "
        f"and {tuple(weights.shape)}"
    )
    assert q.dtype == torch.bfloat16 and weights.dtype == torch.bfloat16, (
        f"the source's index score is a BF16 chain; got q={q.dtype}, weights={weights.dtype}"
    )
    num_tokens, num_heads, head_dim = q.shape
    num_slots = k.shape[0]
    assert k.shape[-1] == head_dim, f"k row is {k.shape[-1]} wide, expected {head_dim}"
    assert q.stride(-1) == 1 and q.stride(-2) == head_dim, "q heads must be contiguous rows"
    assert k.is_contiguous(), "the compressed K block must be contiguous"

    out = torch.empty(num_tokens, num_slots, dtype=torch.float32, device=q.device)
    if num_tokens == 0 or num_slots == 0:
        return out
    block = _index_score_block(num_slots)
    _index_scores_kernel[(num_tokens, triton.cdiv(num_slots, block))](
        q,
        k,
        k_scale.reshape(-1),
        weights,
        out,
        cu_seqlen_ks,
        cu_seqlen_ke,
        q.stride(0),
        weights.stride(0),
        out.stride(0),
        num_slots,
        H=num_heads,
        D=head_dim,
        BLOCK_S=block,
        num_warps=4,
    )
    return out


@triton.jit
def _index_scores_paged_kernel(
    q_ptr,
    cache_fp8_ptr,
    cache_f32_ptr,
    w_ptr,
    out_ptr,
    ctx_len_ptr,
    block_table_ptr,
    q_stride_r,
    w_stride_r,
    out_stride_r,
    ctx_stride_b,
    bt_stride_r,
    block_stride_bytes,
    tokens_per_block,
    max_blocks,
    next_n,
    max_seq_len,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // next_n
    offset = row % next_n
    slots = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    live = slots < max_seq_len

    # Same window the eager fallback in `sparse_attn_indexer` applies:
    # `positions <= context_lens[b] - next_n + next_n_offset`.
    end = tl.load(ctx_len_ptr + batch * ctx_stride_b + offset) - next_n + offset
    logical = slots // tokens_per_block
    addressable = live & (slots <= end) & (logical < max_blocks)
    page = tl.load(block_table_ptr + batch * bt_stride_r + logical, mask=addressable, other=-1)
    inside = addressable & (page >= 0)

    # Page layout, as `postProcessScatterKernel` writes it: the whole block's
    # FP8 rows first, then one FP32 scale per row.
    base = page.to(tl.int64) * block_stride_bytes
    slot = slots % tokens_per_block
    heads = tl.arange(0, H)
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + row * q_stride_r + heads[:, None] * D + dims[None, :])
    k = tl.load(
        cache_fp8_ptr + base[:, None] + slot[:, None] * D + dims[None, :],
        mask=inside[:, None],
        other=0.0,
    )
    k_scale = tl.load(
        cache_f32_ptr + (base + tokens_per_block * D) // 4 + slot, mask=inside, other=0.0
    )
    w = tl.load(w_ptr + row * w_stride_r + heads)

    scores = _source_index_score(q, k, k_scale, w, H, BLOCK_S)
    tl.store(
        out_ptr + row * out_stride_r + slots, tl.where(inside, scores, float("-inf")), mask=live
    )


def source_index_scores_paged(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Decode-phase index scores against the paged indexer K cache.

    Drop-in for ``fp8_paged_mqa_logits``: ``q`` is ``[batch, next_n, heads,
    dim]`` BF16, ``kv_cache`` the byte-addressed indexer pool, ``weights``
    ``[batch * next_n, heads]`` BF16 and ``context_lens`` ``[batch, next_n]``
    indexer KV lengths. Returns ``[batch * next_n, max_seq_len]`` FP32.
    """
    assert q.dim() == 4, f"expected q [batch, next_n, heads, dim], got {tuple(q.shape)}"
    assert q.dtype == torch.bfloat16 and weights.dtype == torch.bfloat16, (
        f"the source's index score is a BF16 chain; got q={q.dtype}, weights={weights.dtype}"
    )
    batch, next_n, num_heads, head_dim = q.shape
    tokens_per_block = kv_cache.shape[1]
    assert kv_cache.element_size() == 1, "the indexer K pool is byte-addressed"
    assert kv_cache.shape[-1] == head_dim + 4, (
        f"indexer pool row is {kv_cache.shape[-1]} bytes, expected {head_dim} + 4"
    )
    q = q.reshape(batch * next_n, num_heads, head_dim)
    assert q.stride(-1) == 1 and q.stride(-2) == head_dim, "q heads must be contiguous rows"

    out = torch.empty(batch * next_n, max_seq_len, dtype=torch.float32, device=q.device)
    if out.numel() == 0:
        return out
    storage = kv_cache.reshape(-1)
    block = _index_score_block(max_seq_len)
    _index_scores_paged_kernel[(batch * next_n, triton.cdiv(max_seq_len, block))](
        q,
        storage.view(torch.float8_e4m3fn),
        storage.view(torch.float32),
        weights,
        out,
        context_lens,
        block_table,
        q.stride(0),
        weights.stride(0),
        out.stride(0),
        context_lens.stride(0),
        block_table.stride(0),
        kv_cache.stride(0) * kv_cache.element_size(),
        tokens_per_block,
        block_table.shape[1],
        next_n,
        max_seq_len,
        H=num_heads,
        D=head_dim,
        BLOCK_S=block,
        num_warps=4,
    )
    return out


def q_norm_source_dtype(
    q_proj: torch.Tensor, num_heads: int, head_dim: int, eps: float
) -> torch.Tensor:
    """Per-head RMS scaling of Q in the source's own dtype.

    ``inference/model.py`` writes::

        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

    with ``q`` in BF16, so the square, the mean, the epsilon add, the
    reciprocal square root and the final multiply all round to BF16. The
    native ``deepseek_v4_q_norm`` kernel computes the same expression in FP32
    and rounds once at the end; the two differ by roughly two BF16 steps at the
    peak element, which is above the registered ``q_projection_and_norm``
    tolerance. There is no learned gain, so this is a two-line elementwise
    expression rather than an RMSNorm module.
    """
    assert q_proj.dim() == 2 and q_proj.shape[1] == num_heads * head_dim, (
        f"q must be [tokens, {num_heads * head_dim}], got {tuple(q_proj.shape)}"
    )
    q = q_proj.view(-1, num_heads, head_dim)
    scale = torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    return (q * scale).view_as(q_proj)


_HADAMARD_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _hadamard_matrix(width: int, device: torch.device) -> torch.Tensor:
    """Sylvester-order +/-1 Hadamard matrix, cached per (width, device)."""
    key = (width, device)
    matrix = _HADAMARD_CACHE.get(key)
    if matrix is None:
        assert width > 0 and width & (width - 1) == 0, (
            f"Hadamard rotation needs a power-of-two width, got {width}"
        )
        matrix = torch.ones(1, 1, device=device, dtype=torch.float32)
        while matrix.shape[0] < width:
            top = torch.cat([matrix, matrix], dim=-1)
            matrix = torch.cat([top, torch.cat([matrix, -matrix], dim=-1)], dim=0)
        _HADAMARD_CACHE[key] = matrix
    return matrix


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """``fast_hadamard_transform(x, scale=width ** -0.5)`` without the extension.

    ``rotate_activation`` in the shared DSA indexer *silently returns its input*
    when ``fast-hadamard-transform`` is not installed, which drops the source's
    rotation from Q while the compressor's fused postprocess keeps applying its
    own butterfly to K. Rotating one side and not the other is worse than
    rotating neither, and the rotation is not optional in the source: it is
    what conditions the FP4 quantisation that follows. The transform is
    orthogonal and small (128 wide), so a matrix multiply is a complete
    implementation rather than a stand-in.
    """
    width = x.shape[-1]
    matrix = _hadamard_matrix(width, x.device)
    flat = x.reshape(-1, width).float() @ matrix
    return (flat * width**-0.5).to(x.dtype).view_as(x)
