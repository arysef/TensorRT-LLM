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
"""SM90 execution for DeepSeek-V4 sparse MLA.

The native sparse-MLA path is Blackwell-only, and not because of a Python
guard: ``AttentionOp::useSparseMLA()`` requires ``mUseTllmGen``, which is false
on SM90, so removing the ``get_sm_version() < 100`` rejection alone would run
dense MLA against a sparse index table. Hopper therefore needs an actual
implementation, not a relaxed check.

This module is that implementation. It keeps every ownership boundary the
Blackwell path already established --- the same ``DeepseekV4TrtllmAttention``
backend, the same ``DeepseekV4TrtllmAttentionMetadata``, the same
``DeepseekV4CacheManager`` on top of ``KVCacheManagerV2``, the same dual-pool
global index table produced by
:func:`~.kernels.deepseek_v4_local_to_global_indices` --- and replaces only the
three things SM90 cannot execute inside the fused native op: the RoPE that op
applies on the way in, the paged latent write it performs, and the sparse-MLA
kernel itself. Splitting a fused op into its three externally visible steps is
the same shape the DSA sparse module already uses (`sparse/dsa/module.py`
calls `sparse_attn_predict`, then appends, then runs its own kernel).

The index table is what makes a single attention kernel possible. Each row
already holds ``window_size`` slots addressing the SWA pool followed by
``max_compressed_indices`` slots addressing the compressed pool, with ``-1``
marking padding, so the gather needs no request bookkeeping, no per-token
Python loop, and no second index format: the slot's *position* in the row picks
the pool, and its value is the token index within that pool, measured from that
pool's base pointer.

Numerics follow ``inference/kernel.py``'s ``sparse_attn_kernel`` rather than a
single-pass softmax, because the difference is observable in BF16 output:
per tile of :data:`SPARSE_ATTN_BLOCK` selected slots the scores accumulate in
FP32, the row maximum extends across tiles, the output accumulator and running
denominator are rescaled by ``exp(prev_max - new_max)``, the numerators are
materialised in the activation dtype before the value GEMM, and the attention
sink is added to the denominator once at the end against the global maximum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

from ..params import SparseBackendForwardArgs
from .params import DeepseekV4AttentionType
from .sm90_quant import fp8_quant_dequant_

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.mla import MLA

    from .metadata import DeepseekV4TrtllmAttentionMetadata

#: Selected slots processed per tile. This matches ``block = 64`` in the
#: source kernel and is part of the numerics rather than a tuning knob: the
#: running maximum, the accumulator rescale and the activation-dtype
#: materialisation of the attention weights all happen once per tile.
SPARSE_ATTN_BLOCK = 64

#: Incremented once per SM90 launch, per step. Evidence only --- these exist so
#: a test or an evidence run can prove this path executed instead of a dense or
#: Blackwell fallback, which is otherwise indistinguishable from plausible
#: output. Deliberately not user-facing flags and not dispatch inputs.
_DISPATCH_COUNTER = {
    "sparse_mla_dual_pool": 0,
    "append_latent_to_paged_cache": 0,
    "context": 0,
    "generation": 0,
}

#: Device-side count of latent rows the append kernel refused to write because
#: the block table had no page for them. Kept on the device so the hot path
#: never synchronizes; :func:`dispatch_counts` pays the one sync needed to read
#: it, and it is evidence rather than control flow. A dropped row is a runtime
#: allocation bug that would otherwise surface only as a subtly wrong token.
_DROPPED_ROWS: dict[torch.device, torch.Tensor] = {}


def _dropped_rows_counter(device: torch.device) -> torch.Tensor:
    counter = _DROPPED_ROWS.get(device)
    if counter is None:
        counter = torch.zeros(1, dtype=torch.int32, device=device)
        _DROPPED_ROWS[device] = counter
    return counter


def dispatch_counts() -> dict[str, int]:
    """Return a snapshot of the SM90 dispatch counters.

    ``append_rows_dropped`` is read back from the device, so this synchronizes.
    It is an evidence hook, never called from the forward path.
    """
    counts = dict(_DISPATCH_COUNTER)
    counts["append_rows_dropped"] = sum(int(c.item()) for c in _DROPPED_ROWS.values())
    return counts


def reset_dispatch_counts() -> None:
    """Zero the SM90 dispatch counters."""
    for key in _DISPATCH_COUNTER:
        _DISPATCH_COUNTER[key] = 0
    for counter in _DROPPED_ROWS.values():
        counter.zero_()


@triton.jit
def _sparse_mla_dual_pool_kernel(
    q_ptr,
    swa_pool_ptr,
    cmp_pool_ptr,
    idx_ptr,
    sink_ptr,
    out_ptr,
    softmax_scale,
    q_stride_t,
    q_stride_h,
    idx_stride_t,
    out_stride_t,
    out_stride_h,
    num_heads,
    num_selected,
    num_swa_indices,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """One program per token; heads are the GEMM's M dimension.

    That shape is deliberate rather than incidental. MLA keeps a single latent
    row per position serving as both key and value, so every head of a token
    reads exactly the same gathered rows --- making the token the natural unit
    of work and letting one gather feed all heads. It also mirrors the source
    kernel, whose grid is ``(m, b)`` with ``q_shared`` of shape ``(h, d)``.
    """
    token = tl.program_id(0)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    head_mask = offs_h < num_heads

    q = tl.load(
        q_ptr + token * q_stride_t + offs_h[:, None] * q_stride_h + offs_d[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_H], dtype=tl.float32)
    run_max = tl.full([BLOCK_H], float("-inf"), tl.float32)

    for start in range(0, num_selected, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        idx = tl.load(idx_ptr + token * idx_stride_t + offs_n, mask=offs_n < num_selected, other=-1)
        valid = idx >= 0
        # A slot's position in the row picks the pool; its value is the token
        # index inside that pool. 64-bit offsets because a large KV pool times
        # a 512-wide row overflows int32.
        rows = tl.where(valid, idx, 0)[:, None].to(tl.int64) * HEAD_DIM + offs_d[None, :]

        if HAS_COMPRESS:
            from_swa = offs_n < num_swa_indices
            kv_swa = tl.load(swa_pool_ptr + rows, mask=(valid & from_swa)[:, None], other=0.0)
            kv_cmp = tl.load(cmp_pool_ptr + rows, mask=(valid & ~from_swa)[:, None], other=0.0)
            kv = tl.where(from_swa[:, None], kv_swa, kv_cmp)
        else:
            kv = tl.load(swa_pool_ptr + rows, mask=valid[:, None], other=0.0)

        # Padded rows are zero, so they contribute nothing to either GEMM even
        # before the -inf mask takes effect --- exactly as the source zeroes
        # `kv_shared` and pre-seeds `acc_s` at -inf.
        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(valid[None, :], scores, float("-inf"))

        prev_max = run_max
        run_max = tl.maximum(run_max, tl.max(scores, 1))
        # `libdevice.exp`, not `tl.exp`. Triton lowers `tl.exp` to the hardware
        # `ex2.approx` sequence, which carries up to ~15 FP32 ulp on this range
        # and disagrees with `torch.exp` on 86% of the exponents this softmax
        # produces; libdevice's is correctly rounded and bit-identical to it on
        # 2^22 sampled values. Both GEMMs already agree with the reference bit
        # for bit, so the exponential was the whole of the residual: the source
        # kernel's attention weights are the correctly-rounded ones, and an
        # approximate exp here is a faithfulness gap, not a free speedup.
        rescale = tl.where(run_max == float("-inf"), 0.0, libdevice.exp(prev_max - run_max))
        probs = tl.where(valid[None, :], libdevice.exp(scores - run_max[:, None]), 0.0)

        sum_exp = sum_exp * rescale + tl.sum(probs, 1)
        acc = acc * rescale[:, None]
        # `acc_s_cast`: the numerators are materialised at the activation dtype
        # before the value GEMM, and the tile accumulates *into* the rescaled
        # output accumulator rather than being summed separately first.
        acc = tl.dot(probs.to(kv.dtype), kv, acc)

    if HAS_SINK:
        # Denominator mass only: the sink adds no value, so it can shrink the
        # output but never steer it. Added once, against the global maximum.
        sink = tl.load(sink_ptr + offs_h, mask=head_mask, other=float("-inf"))
        sum_exp = sum_exp + libdevice.exp(sink - run_max)

    out = acc / sum_exp[:, None]
    tl.store(
        out_ptr + token * out_stride_t + offs_h[:, None] * out_stride_h + offs_d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=head_mask[:, None],
    )


def sparse_mla_dual_pool(
    q: torch.Tensor,
    swa_pool: torch.Tensor,
    compress_pool: torch.Tensor | None,
    global_indices: torch.Tensor,
    num_swa_indices: int,
    softmax_scale: float,
    attn_sink: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse MLA over the DeepSeek-V4 dual-pool KV cache.

    Args:
        q: ``[num_tokens, num_heads, head_dim]`` absorbed queries, already
            carrying V4 RoPE on the trailing ``qk_rope_head_dim`` dimensions.
        swa_pool: the sliding-window pool as ``[pool_tokens, head_dim]``,
            addressed from the pool base pointer the index table was built
            against.
        compress_pool: the compressed pool with the same row width. Required
            whenever the index table has a compressed region; only a ratio-1
            layer, whose table is SWA-only, may pass ``None``.
        global_indices: ``[num_tokens, num_selected]`` int32 from
            :func:`~.kernels.deepseek_v4_local_to_global_indices`. Slots below
            ``num_swa_indices`` index ``swa_pool``, the rest index
            ``compress_pool``, and ``-1`` marks padding.
        num_swa_indices: width of the SWA region, i.e. the window size.
        softmax_scale: the source's ``1 / sqrt(head_dim)``.
        attn_sink: per-head FP32 sink contributing denominator mass only.
        out: optional destination; allocated when omitted.

    Returns:
        ``[num_tokens, num_heads, head_dim]`` in ``q``'s dtype.
    """
    assert q.dim() == 3, f"q must be [tokens, heads, dim], got {tuple(q.shape)}"
    num_tokens, num_heads, head_dim = q.shape
    assert swa_pool.dim() == 2 and swa_pool.shape[1] == head_dim, (
        f"swa_pool must be [pool_tokens, {head_dim}], got {tuple(swa_pool.shape)}"
    )
    assert swa_pool.dtype == q.dtype, (
        f"KV pool dtype {swa_pool.dtype} does not match q dtype {q.dtype}. "
        "The SM90 sparse-MLA path reads the latent rows directly, so a quantized "
        "pool would need its dequant scales threaded through; the DeepSeek-V4-Flash "
        "checkpoint quantizes weights, not the KV cache."
    )
    assert global_indices.dtype == torch.int32, (
        f"global_indices must be int32, got {global_indices.dtype}"
    )
    assert global_indices.shape[0] == num_tokens, (
        f"global_indices has {global_indices.shape[0]} rows for {num_tokens} tokens"
    )
    num_selected = global_indices.shape[1]
    assert 0 <= num_swa_indices <= num_selected, (
        f"num_swa_indices {num_swa_indices} outside [0, {num_selected}]"
    )

    # Presence of a compressed region is a property of the index table alone.
    # Deriving it from `compress_pool is not None` instead would let a caller
    # that forgot the pool fall through to the SWA branch, where compressed
    # slots silently address the SWA pool -- a wiring bug that still returns
    # plausible attention output.
    has_compress = num_selected > num_swa_indices
    if has_compress:
        if compress_pool is None:
            raise ValueError(
                f"global_indices has {num_selected - num_swa_indices} compressed slots "
                f"(width {num_selected}, SWA region {num_swa_indices}) but compress_pool "
                "is None. Those slots address the compressed pool; running without it "
                "would read them from the SWA pool and return plausible but wrong "
                "attention. Pass the compressed pool, or an SWA-only index table."
            )
        assert compress_pool.dim() == 2 and compress_pool.shape[1] == head_dim, (
            f"compress_pool must be [pool_tokens, {head_dim}], got {tuple(compress_pool.shape)}"
        )
        assert compress_pool.dtype == q.dtype, (
            f"compressed pool dtype {compress_pool.dtype} does not match q dtype {q.dtype}"
        )

    if out is None:
        out = torch.empty_like(q)
    else:
        assert out.shape == q.shape, f"out must be {tuple(q.shape)}, got {tuple(out.shape)}"

    if num_tokens == 0:
        return out

    # `tl.dot` needs at least 16 rows, and the source pads to 16 heads for the
    # same reason. Padded heads are masked out of both the load and the store.
    block_h = max(16, triton.next_power_of_2(num_heads))

    _DISPATCH_COUNTER["sparse_mla_dual_pool"] += 1
    _sparse_mla_dual_pool_kernel[(num_tokens,)](
        q,
        swa_pool,
        compress_pool if has_compress else swa_pool,
        global_indices,
        attn_sink,
        out,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        global_indices.stride(0),
        out.stride(0),
        out.stride(1),
        num_heads,
        num_selected,
        num_swa_indices,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        BLOCK_N=SPARSE_ATTN_BLOCK,
        HAS_COMPRESS=has_compress,
        HAS_SINK=attn_sink is not None,
        num_warps=8,
        num_stages=2,
    )
    return out


@triton.jit
def _append_latent_kernel(
    latent_ptr,
    pool_ptr,
    req_ptr,
    pos_ptr,
    block_table_ptr,
    dropped_ptr,
    latent_stride_t,
    bt_stride_r,
    bt_stride_b,
    pool_offset_in_tokens,
    max_blocks,
    TOKENS_PER_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Scatter one normalized, RoPE'd latent row per token into the paged pool.

    The destination arithmetic is deliberately identical to the SWA half of
    ``_deepseek_v4_local_to_global_kernel``: same block-ordinal split, same
    block table, same ``pool_offset_in_tokens`` rebasing. Write and read agree
    because they compute the same address from the same inputs, and the
    integration tests assert that agreement rather than assuming it.
    """
    token = tl.program_id(0)
    req = tl.load(req_ptr + token)
    pos = tl.load(pos_ptr + token)

    block_ordinal = pos // TOKENS_PER_BLOCK
    token_in_block = pos % TOKENS_PER_BLOCK
    in_range = (pos >= 0) & (block_ordinal < max_blocks)
    page = tl.load(
        block_table_ptr + req * bt_stride_r + block_ordinal * bt_stride_b,
        mask=in_range,
        other=-1,
    )
    # A negative page is BAD_PAGE_INDEX: the runtime allocated no block for
    # this position. Never write there; count it so evidence can prove zero.
    writable = in_range & (page >= 0)
    if not writable:
        tl.atomic_add(dropped_ptr, 1)

    offs_d = tl.arange(0, HEAD_DIM)
    row = pool_offset_in_tokens + page.to(tl.int64) * TOKENS_PER_BLOCK + token_in_block
    value = tl.load(latent_ptr + token * latent_stride_t + offs_d)
    # `offs_d < HEAD_DIM` is uniformly true; it is there to give the scalar
    # `writable` predicate the vector shape the store's mask needs.
    tl.store(pool_ptr + row * HEAD_DIM + offs_d, value, mask=(offs_d < HEAD_DIM) & writable)


def append_latent_to_paged_cache(
    latent: torch.Tensor,
    pool: torch.Tensor,
    req_idx: torch.Tensor,
    token_positions: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    pool_offset_in_tokens: int = 0,
) -> None:
    """Write the current tokens' latent rows into the paged SWA pool.

    Args:
        latent: ``[num_tokens, head_dim]`` normalized latent rows carrying V4
            RoPE on their trailing ``qk_rope_head_dim`` dimensions.
        pool: ``[pool_tokens, head_dim]`` flat view of the SWA pool, addressed
            from the same base pointer the index table was built against.
        req_idx: ``[num_tokens]`` int32 batch-local request index per token.
        token_positions: ``[num_tokens]`` int32 absolute position of each token
            within its own sequence.
        block_table: ``[num_requests, max_blocks]`` int32 SWA block table.
        tokens_per_block: SWA page size, a power of two.
        pool_offset_in_tokens: rebases the layer's buffer onto the pool base.
    """
    assert latent.dim() == 2, f"latent must be [tokens, dim], got {tuple(latent.shape)}"
    num_tokens, head_dim = latent.shape
    assert pool.dim() == 2 and pool.shape[1] == head_dim, (
        f"pool must be [pool_tokens, {head_dim}], got {tuple(pool.shape)}"
    )
    assert pool.dtype == latent.dtype, (
        f"pool dtype {pool.dtype} does not match latent dtype {latent.dtype}"
    )
    assert pool.is_contiguous(), "pool must be a contiguous [pool_tokens, head_dim] view"
    assert latent.stride(1) == 1, "latent rows must be unit-stride"
    assert req_idx.dtype == torch.int32 and req_idx.shape == (num_tokens,), (
        f"req_idx must be int32 [{num_tokens}], got {req_idx.dtype} {tuple(req_idx.shape)}"
    )
    assert token_positions.dtype == torch.int32 and token_positions.shape == (num_tokens,), (
        f"token_positions must be int32 [{num_tokens}], "
        f"got {token_positions.dtype} {tuple(token_positions.shape)}"
    )
    assert block_table.dtype == torch.int32 and block_table.dim() == 2, (
        f"block_table must be 2D int32, got {block_table.dtype} {tuple(block_table.shape)}"
    )
    assert tokens_per_block > 0 and tokens_per_block & (tokens_per_block - 1) == 0, (
        f"tokens_per_block must be a power of two, got {tokens_per_block}"
    )
    if num_tokens == 0:
        return

    _DISPATCH_COUNTER["append_latent_to_paged_cache"] += 1
    _append_latent_kernel[(num_tokens,)](
        latent,
        pool,
        req_idx.contiguous(),
        token_positions.contiguous(),
        block_table,
        _dropped_rows_counter(latent.device),
        latent.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        pool_offset_in_tokens,
        block_table.shape[1],
        TOKENS_PER_BLOCK=tokens_per_block,
        HEAD_DIM=head_dim,
        num_warps=4,
    )


def token_positions(
    metadata: "DeepseekV4TrtllmAttentionMetadata", start_idx: int, end_idx: int
) -> torch.Tensor:
    """Absolute in-sequence position of each token in ``[start_idx, end_idx)``.

    Rebuilt from the same three metadata buffers
    ``prepare_for_deepseek_v4_indices`` uses to build the SWA index table
    (``req_idx_per_token``, ``cu_seq_lens_cuda``, ``cached_token_lens_cuda``),
    so the position a row is *written* at is derived from the same state as the
    positions the table selects. Device-only and shape-static, so it neither
    synchronizes nor breaks CUDA-graph capture.

    Deliberately not ``position_ids``: the native RoPE kernels derive positions
    from the cache lengths too, and the cache address must follow the cache's
    own accounting. The two agreeing is an invariant the integration tests
    check, not something assumed here.
    """
    num_tokens = end_idx - start_idx
    req = metadata.req_idx_per_token[start_idx:end_idx].to(torch.int64)
    token_idx = torch.arange(
        start_idx, end_idx, dtype=torch.int32, device=metadata.req_idx_per_token.device
    )
    seq_start = metadata.cu_seq_lens_cuda[req].to(torch.int32)
    positions = metadata.cached_token_lens_cuda[req].to(torch.int32) + (token_idx - seq_start)
    assert positions.shape == (num_tokens,)
    return positions


def _flat_pool(
    mla: "MLA",
    metadata: "DeepseekV4TrtllmAttentionMetadata",
    attn_type: DeepseekV4AttentionType,
    expected_base_ptr: int,
) -> torch.Tensor:
    """A ``[pool_tokens, head_dim]`` view of a pool, based where indices count from.

    The global index table counts token rows from ``expected_base_ptr``, which
    is the pointer :class:`DeepseekV4TrtllmAttention` hands the index kernel and
    the one the Blackwell native op receives as ``aux_kv_cache_pool_ptr``. So
    the view has to start exactly there --- but ``get_buffers`` returns *this
    layer's* region, and layers sharing a compressed pool sit at different
    offsets inside it (``get_mem_pool_base_address`` bakes the layer offset in
    for ``PageIndexMode.SHARED``). The index kernel already adds that offset to
    every index, so the view is extended backwards to the pool origin instead
    of being rebased on the layer, which would double-count it.

    The offset is required to be a whole number of token rows and to be
    forward: a view based *after* the index origin would still run and still
    produce attention-shaped output, just reading the wrong rows.
    """
    cache = getattr(mla, "_sm90_pool_views", None)
    if cache is None:
        cache = {}
        mla._sm90_pool_views = cache
    key = (attn_type, expected_base_ptr)
    view = cache.get(key)
    if view is not None:
        return view

    buffers = metadata.kv_cache_manager.get_buffers(mla.layer_idx, attn_type)
    row_dim = buffers.shape[-1]
    token_bytes = row_dim * buffers.element_size()
    layer_offset_bytes = buffers.data_ptr() - expected_base_ptr
    assert layer_offset_bytes >= 0 and layer_offset_bytes % token_bytes == 0, (
        f"{attn_type.name} buffer for layer {mla.layer_idx} starts at "
        f"{buffers.data_ptr():#x}, but the index table is built against pool base "
        f"{expected_base_ptr:#x}; that is {layer_offset_bytes} bytes away, which is not "
        f"a whole number of {token_bytes}-byte token rows at or after the base."
    )
    rows = layer_offset_bytes // token_bytes + buffers.numel() // row_dim
    view = convert_to_torch_tensor(TensorWrapper(expected_base_ptr, buffers.dtype, (rows, row_dim)))
    cache[key] = view
    return view


def forward_sparse_attn_sm90(
    mla: "MLA",
    q: torch.Tensor,
    attn_metadata: "DeepseekV4TrtllmAttentionMetadata",
    output: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: Optional[torch.Tensor],
    attention_input_type: AttentionInputType,
) -> torch.Tensor:
    """Run one DeepSeek-V4 sparse-MLA phase on SM90 and fill ``output``.

    Performs, explicitly, the three steps the Blackwell native op fuses:
    external V4 RoPE on Q and on the latent row, the paged latent append, and
    the dual-pool sparse attention. Index production, the block tables, the KV
    pools and the cache manager are all the existing ones.
    """
    backend = mla.mqa
    num_tokens = q.shape[0]
    num_heads = mla.num_heads_tp
    head_dim = mla.qk_head_dim
    nope_dim = mla.qk_nope_head_dim
    rope_dim = mla.qk_rope_head_dim

    if getattr(mla, "_fused_kv_norm_active", False):
        raise NotImplementedError(
            "The SM90 DeepSeek-V4 path expects an already-normalized BF16 latent and a "
            "BF16 KV cache. The fused kv-norm prologue only engages with an FP8 KV "
            "cache, whose paged rows this path cannot read without threading dequant "
            "scales through the sparse kernel."
        )
    assert num_tokens > 0, "SM90 sparse attention called with an empty token range"
    assert q.shape == (num_tokens, num_heads * head_dim), (
        f"q must be [{num_tokens}, {num_heads * head_dim}], got {tuple(q.shape)}"
    )
    assert latent_cache.shape == (num_tokens, head_dim), (
        f"latent_cache must be [{num_tokens}, {head_dim}], got {tuple(latent_cache.shape)}"
    )
    assert output.shape == (num_tokens, num_heads * mla.v_head_dim), (
        f"output must be [{num_tokens}, {num_heads * mla.v_head_dim}], got {tuple(output.shape)}"
    )

    is_context = attention_input_type == AttentionInputType.context_only
    _DISPATCH_COUNTER["context" if is_context else "generation"] += 1

    if is_context:
        start_idx, end_idx = 0, attn_metadata.num_ctx_tokens
    else:
        start_idx, end_idx = attn_metadata.num_ctx_tokens, attn_metadata.num_tokens
    assert end_idx - start_idx == num_tokens, (
        f"phase {attention_input_type} covers {end_idx - start_idx} metadata tokens "
        f"but received {num_tokens}"
    )

    positions = token_positions(attn_metadata, start_idx, end_idx)

    # `inverse_rotary_emb` holds the layer's own RoPE table -- per-ratio theta
    # and YaRN from `_deepseek_v4_pos_embd_params` -- and `inverse` is applied
    # at call time, not baked into the table. Using it for the forward
    # direction makes the output-side inverse RoPE an exact inverse by
    # construction rather than by two tables happening to match.
    cos_sin = mla.inverse_rotary_emb.rotary_cos_sin
    is_neox = mla.inverse_rotary_emb.is_neox

    q3 = q.view(num_tokens, num_heads, head_dim)
    torch.ops.trtllm.mla_rope_inplace(
        q3, positions, cos_sin, num_heads, nope_dim, rope_dim, False, is_neox
    )
    # Cloned rather than rotated in place: `latent_cache` is the caller's
    # [compressed_kv | k_pe] concat, and the generation half of a mixed batch
    # must not observe a half-rotated latent.
    latent = latent_cache.clone()
    torch.ops.trtllm.mla_rope_inplace(
        latent.view(num_tokens, 1, head_dim),
        positions,
        cos_sin,
        1,
        nope_dim,
        rope_dim,
        False,
        is_neox,
    )
    # `Attention.forward` in inference/model.py follows kv_norm + RoPE with
    #   act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
    # -- a blockwise-64 FP8 round trip that stays BF16, applied to the window
    # latent before it enters the cache. The checkpoint is QAT'd for it, and
    # the compressed rows get the same treatment inside `Compressor`, so
    # applying it to one pool and not the other would make the two halves of
    # the same attention disagree about how a latent row is represented.
    fp8_quant_dequant_(latent[:, :nope_dim])

    compress_ratio = backend.compress_ratio
    swa_pool = _flat_pool(
        mla, attn_metadata, DeepseekV4AttentionType.SWA, attn_metadata.sparse_mla_base_ptrs[1]
    )
    token_bytes = head_dim * swa_pool.element_size()
    swa_buffer_delta = attn_metadata.swa_buffer_ptrs[mla.layer_idx] - swa_pool.data_ptr()
    assert swa_buffer_delta >= 0 and swa_buffer_delta % token_bytes == 0, (
        f"SWA buffer pointer is {swa_buffer_delta} bytes past the pool base, which is not "
        f"a whole number of {token_bytes}-byte token rows"
    )

    local_layer_idx = attn_metadata.kv_cache_manager.layer_offsets[mla.layer_idx]
    block_table_swa = attn_metadata.sliding_block_tables[
        local_layer_idx, DeepseekV4AttentionType.SWA.value
    ]
    append_latent_to_paged_cache(
        latent,
        swa_pool,
        attn_metadata.req_idx_per_token[start_idx:end_idx],
        positions,
        block_table_swa,
        attn_metadata.kv_cache_manager.tokens_per_block,
        pool_offset_in_tokens=swa_buffer_delta // token_bytes,
    )

    # The current token's own latent has to be visible to its own attention, so
    # the append precedes index consumption; the table already includes each
    # token's own SWA slot.
    global_indices, _ = backend.sparse_attn_predict(
        q,
        None,
        attn_metadata,
        AttentionForwardArgs(
            attention_input_type=attention_input_type,
            sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk_indices),
        ),
    )
    assert global_indices is not None, "DeepSeek-V4 sparse index prediction returned nothing"

    compress_pool = None
    if compress_ratio > 1:
        compress_pool = _flat_pool(
            mla,
            attn_metadata,
            DeepseekV4AttentionType.COMPRESS,
            attn_metadata.sparse_mla_base_ptrs[compress_ratio],
        )

    attn_sink = getattr(backend, "attn_sink", None)
    sparse_mla_dual_pool(
        q3,
        swa_pool,
        compress_pool,
        global_indices,
        backend.sparse_attention_config.window_size,
        # The reference spells this `head_dim ** -0.5`. `1 / (q_scaling *
        # sqrt(head_dim))` is one ULP away from it in FP64 --- invisible today
        # because Triton passes the scalar as FP32, but there is no reason to
        # carry a needless difference from the expression under test.
        (float(head_dim) ** -0.5) / backend.q_scaling,
        attn_sink=None if attn_sink is None else attn_sink.data,
        out=output.view(num_tokens, num_heads, mla.v_head_dim),
    )
    return output
