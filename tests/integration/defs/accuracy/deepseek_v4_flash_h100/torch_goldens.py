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
"""Independent pure-Torch goldens for DeepSeek-V4-Flash module semantics.

This module is the *middle rung* of the reference ladder. Everything here is
written directly from the mathematical contract in the checkpoint's official
``inference/model.py`` / ``inference/kernel.py`` using nothing but plain
PyTorch: no tilelang kernel, no official ``model.py`` class, and --- most
importantly --- no TensorRT-LLM import. A bug shared between the code under
test and its reference produces excellent false parity, so the reference has
to be derived independently rather than borrowed.

The guarantee is enforced, not just documented: :func:`assert_independent`
checks that ``tensorrt_llm`` was never imported into the running interpreter,
and the evidence driver calls it immediately before computing goldens.

Coverage mirrors the Stage-1 acceptance item: Q/KV norm, RoPE (forward and the
inverse applied to attention output), Compressor gated pooling, Indexer
selection, sparse attention with the FP32 attention sink, MoE routing and the
clamped-SwiGLU expert, and mHC (Sinkhorn split, pre-mix and post-mix).
"""

from __future__ import annotations

import math
import sys
from typing import Any

import torch
import torch.nn.functional as F

# FP8 E4M3 finite range used by the official activation quantiser.
FP8_MAX = 448.0
# FP4 E2M1 magnitude range and its eight representable magnitudes.
FP4_MAX = 6.0
FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def assert_independent() -> None:
    """Fail if TensorRT-LLM has been imported into this interpreter.

    The reference ladder is only trustworthy if it cannot inherit a bug from
    the implementation it is used to judge. Being *installed* is fine; being
    *imported* while goldens are computed is not.
    """
    leaked = sorted(m for m in sys.modules if m == "tensorrt_llm" or m.startswith("tensorrt_llm."))
    if leaked:
        raise AssertionError(
            "pure-Torch goldens must not share helpers with the code under test, "
            f"but tensorrt_llm is imported: {leaked[:5]}"
        )


# ---------------------------------------------------------------------------
# Quantisation simulation.
#
# The official model does not merely *store* weights quantised --- it also
# round-trips several activations through FP8/FP4 at inference time to match
# how the network was trained (`act_quant(..., inplace=True)` and
# `fp4_act_quant(..., inplace=True)`). A golden that skips those round trips
# diverges from the source for reasons that have nothing to do with the module
# under test, so they are reproduced here in plain Torch.
# ---------------------------------------------------------------------------


def _round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """Power-of-two scale: ``2 ** ceil(log2(amax * max_inv))``.

    The kernel computes this with IEEE-754 bit tricks (``fast_round_scale``);
    exponent arithmetic reproduces it exactly for finite positive inputs.
    """
    scaled = amax.float() * max_inv
    exp = torch.ceil(torch.log2(scaled))
    return torch.exp2(exp)


def fp8_quant_dequant(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Round-trip ``x`` through blockwise FP8 E4M3 with UE8M0 (power-of-2) scales.

    Mirrors ``act_quant(x, block_size, "ue8m0", float8_e8m0fnu, inplace=True)``.
    """
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).float()
    blocks = flat.unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = _round_scale(amax, 1.0 / FP8_MAX)
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).flatten(-2).reshape(shape).to(x.dtype)


def _fp4_round_to_nearest_even(v: torch.Tensor) -> torch.Tensor:
    """Round magnitudes to the nearest E2M1 level, ties to the even code.

    E2M1 codes 0..7 map to FP4_LEVELS. Exact ties land on the even code, which
    matters because the inputs are dyadic (bf16 divided by a power of two), so
    ties are common rather than a measure-zero curiosity.
    """
    levels = torch.tensor(FP4_LEVELS, dtype=torch.float32, device=v.device)
    dist = (v.unsqueeze(-1) - levels).abs()
    # Bias strictly smaller than the closest distinct-level gap (0.25), so it
    # only ever decides exact ties.
    even_bonus = torch.tensor(
        [0.0 if i % 2 else 1e-6 for i in range(len(FP4_LEVELS))],
        dtype=torch.float32,
        device=v.device,
    )
    return levels[(dist - even_bonus).argmin(dim=-1)]


def fp4_quant_dequant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Round-trip ``x`` through blockwise MXFP4 with UE8M0 scales.

    Mirrors ``fp4_act_quant(x, block_size, inplace=True)``.
    """
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).float()
    blocks = flat.unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2.0**-126)
    scale = _round_scale(amax, 1.0 / FP4_MAX)
    scaled = (blocks / scale).clamp(-FP4_MAX, FP4_MAX)
    q = torch.sign(scaled) * _fp4_round_to_nearest_even(scaled.abs())
    return (q * scale).flatten(-2).reshape(shape).to(x.dtype)


def dequant_fp8_blockwise(
    weight: torch.Tensor, scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Expand a checkpoint FP8 weight with its 128x128 UE8M0 block scales."""
    out_f, in_f = weight.shape
    w = weight.float().unflatten(1, (-1, block)).unflatten(0, (-1, block))
    return (w * scale.float()[:, None, :, None]).flatten(2, 3).flatten(0, 1)[:out_f, :in_f]


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Expand packed MXFP4 routed-expert weights to float.

    ``packed`` is the checkpoint's I8 byte container holding two E2M1 nibbles
    per byte along K, low nibble first; ``scale`` carries one UE8M0 exponent
    per 32 logical K values.
    """
    levels = torch.tensor(FP4_LEVELS, dtype=torch.float32, device=packed.device)
    raw = packed.view(torch.uint8)
    low, high = raw & 0x0F, (raw >> 4) & 0x0F

    # Bit 3 is the sign; bits 0-2 select the magnitude level.
    def decode(nib: torch.Tensor) -> torch.Tensor:
        mag = levels[(nib & 0x07).long()]
        return torch.where((nib & 0x08) != 0, -mag, mag)

    vals = torch.stack([decode(low), decode(high)], dim=-1).flatten(-2)
    return vals * scale.float().repeat_interleave(group, dim=-1)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalised Walsh-Hadamard transform over the last dim (power-of-two).

    The source calls ``fast_hadamard_transform(x, scale=d**-0.5)`` to spread
    information across dims before FP4 quantisation. The butterfly below is
    the same orthogonal transform in natural (Hadamard) order, written out so
    the golden does not depend on the same CUDA extension the source uses.
    """
    d = x.shape[-1]
    if d & (d - 1):
        raise ValueError(f"Hadamard transform needs a power-of-two width, got {d}")
    y = x.float().clone()
    step = 1
    while step < d:
        y = y.unflatten(-1, (-1, 2, step))
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).flatten(-3)
        step *= 2
    return (y * d**-0.5).to(x.dtype)


# ---------------------------------------------------------------------------
# Norms.
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Weighted RMS norm, computed in FP32 and cast back (source contract)."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(dtype)


def per_head_rms_scale(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Unweighted per-head RMS scaling applied to Q after ``wq_b``.

    The source multiplies in place without a learned gain and without an
    FP32 round trip, so this stays in the input dtype.
    """
    return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Positional encoding.
# ---------------------------------------------------------------------------


def yarn_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """YaRN *frequency interpolation only* --- there is no amplitude term.

    ``original_seq_len == 0`` disables YaRN entirely, which is what the
    SWA-only (ratio-0) layers use with the base theta.
    """

    def correction_dim(num_rotations: float) -> float:
        return (
            dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
        )

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the trailing RoPE dims of ``x``; ``inverse`` de-rotates.

    ``x`` is ``[b, s, rope_dim]`` or ``[b, s, heads, rope_dim]`` and
    ``freqs_cis`` is ``[s, rope_dim // 2]``. The source broadcasts by an
    explicit ``view`` rather than trailing-dim alignment, so the head axis
    (when present) has to be inserted before the frequency axis, not after
    the sequence axis.
    """
    dtype = x.dtype
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if z.ndim == 3:
        freqs_cis = freqs_cis.view(1, z.size(1), z.size(-1))
    elif z.ndim == 4:
        freqs_cis = freqs_cis.view(1, z.size(1), 1, z.size(-1))
    else:
        raise ValueError(f"unsupported RoPE input rank {z.ndim}")
    return torch.view_as_real(z * freqs_cis).flatten(-2).to(dtype)


# ---------------------------------------------------------------------------
# Compressor: learned gated pooling over `ratio` consecutive tokens.
# ---------------------------------------------------------------------------


def compressor_prefill(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    ratio: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    rotate: bool,
    hadamard: Any = None,
    stages: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Whole-prefill compressed KV rows for one layer.

    Ratio 4 pools *overlapping* windows (hence the doubled projection width
    and the shift-by-one transform); ratio 128 pools disjoint windows. Pooling
    is a softmax over the learned gate plus an absolute position embedding,
    and it runs in FP32 exactly as the source does.

    Pass ``stages`` to also collect the intermediate tensors under the names
    ``pooled_fp32``, ``pooled``, ``normed``, ``roped`` and ``quantized``. The
    chain is short but it crosses three different roundings and a quantiser,
    so when the whole-chain result disagrees with an implementation the only
    cheap way to say *where* is to compare the same boundaries.
    """
    bsz, seqlen, _ = x.shape
    xf = x.float()
    kv = F.linear(xf, wkv.float())
    score = F.linear(xf, wgate.float())

    cutoff = seqlen - seqlen % ratio
    if cutoff == 0:
        raise ValueError(f"prefill of {seqlen} tokens has no complete ratio-{ratio} group")

    pooled_fp32 = compressor_pool(
        kv[:, :cutoff], score[:, :cutoff], ape, ratio=ratio, head_dim=head_dim
    )
    pooled = pooled_fp32.to(x.dtype)
    normed = rms_norm(pooled, norm_weight, eps)
    rope_part = apply_rope(normed[..., -rope_dim:], freqs_cis[:cutoff:ratio])
    roped = torch.cat([normed[..., :-rope_dim], rope_part], dim=-1)

    if rotate:
        quantized = fp4_quant_dequant(hadamard(roped), 32)
    else:
        head = fp8_quant_dequant(roped[..., :-rope_dim], 64)
        quantized = torch.cat([head, roped[..., -rope_dim:]], dim=-1)

    if stages is not None:
        stages.update(
            pooled_fp32=pooled_fp32,
            pooled=pooled,
            normed=normed,
            roped=roped,
            quantized=quantized,
        )
    return quantized


def compressor_pool(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    *,
    ratio: int,
    head_dim: int,
) -> torch.Tensor:
    """FP32 gated pooling of one compressor's projected KV and gate rows.

    Split out of :func:`compressor_prefill` so the same reduction can be
    driven from someone else's projection: that is what separates a
    projection-GEMM difference from a reduction difference when an
    implementation disagrees with the source.

    ``kv``/``score`` are ``[b, cutoff, coff * head_dim]`` FP32; the softmax is
    over the ``ratio`` (or ``2 * ratio``, overlapping) slots of each group and
    is taken *per feature*, not per slot.
    """
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + ape.float()
    if ratio == 4:
        kv = _overlap_transform(kv, 0.0, ratio, head_dim)
        score = _overlap_transform(score, float("-inf"), ratio, head_dim)
    return (kv * score.softmax(dim=2)).sum(dim=2)


def _overlap_transform(t: torch.Tensor, fill: float, ratio: int, head_dim: int) -> torch.Tensor:
    """Interleave the previous group's first half with this group's second half."""
    b, s = t.shape[0], t.shape[1]
    out = t.new_full((b, s, 2 * ratio, head_dim), fill)
    out[:, :, ratio:] = t[..., head_dim:]
    out[:, 1:, :ratio] = t[:, :-1, :, :head_dim]
    return out


# ---------------------------------------------------------------------------
# Indexer: learned top-k selection over compressed slots (ratio-4 layers only).
# ---------------------------------------------------------------------------


def indexer_scores_and_topk(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    weights: torch.Tensor,
    *,
    seqlen: int,
    ratio: int,
    topk: int,
    offset: int,
    kv_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query index scores and the selected compressed slots.

    Scores are a ReLU'd per-head dot product reduced by the learned per-head
    weight; the causal mask forbids any compressed slot whose window is not
    fully in the past. Invalid selections are marked ``-1`` rather than
    dropped, which is the padding convention the attention kernel expects.

    ``kv_len`` selects the source's other branch. ``Indexer.forward`` masks and
    marks only when ``start_pos == 0``; on a decode step every slot the cache
    already holds is visible to the single query and nothing is marked ``-1``,
    which is what passing ``kv_len`` expresses. The whole difference is the
    per-query limit, so the two branches share one expression here.

    The reduction stays in the operands' own dtype, which for this checkpoint
    means BF16 end to end. ``Indexer.forward`` writes::

        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[...])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

    under ``torch.set_default_dtype(torch.bfloat16)``, with ``weights_proj``
    declared ``dtype=torch.bfloat16``. So the value is rounded to BF16 three
    times --- once at the einsum epilogue, once at the per-head weight
    multiply, and once after the 64-head sum --- while the dot product and the
    head sum accumulate in FP32 inside the GEMM and the reduction. An earlier
    version of this function called ``.float()`` on all three operands, which
    computes the same *mathematical* quantity in a strictly finer grid; that is
    not a harmless improvement, because top-k is a discrete decision. Measured
    at the checkpoint's top-k of 512 over 576 slots, the FP32 chain selects a
    different slot set than the source on 33 of 253 deciding rows.

    Widening is therefore never done here. Passing FP32 operands still gives an
    all-FP32 chain, which is what the CPU unit tests want; passing the source's
    BF16 gives the source's rounding.
    """
    score = torch.einsum("bshd,btd->bsht", q, compressed_kv)
    score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)

    slots = torch.arange(score.shape[-1], device=score.device)
    if kv_len is None:
        limit = (torch.arange(1, seqlen + 1, device=score.device) // ratio).unsqueeze(1)
    else:
        limit = torch.full((seqlen, 1), kv_len, device=score.device)
    invalid = slots.unsqueeze(0) >= limit
    # `index_score += torch.where(mask, -inf, 0)` in the source. Adding zero is
    # a no-op on the valid entries, so this is the same values without
    # promoting the tensor to the dtype of the mask constant.
    score = score.masked_fill(invalid, float("-inf"))

    idxs = score.topk(min(topk, compressed_kv.shape[1]), dim=-1)[1]
    return score, torch.where(idxs >= limit, -1, idxs + offset)


# ---------------------------------------------------------------------------
# Sparse attention with an FP32 denominator-only sink.
# ---------------------------------------------------------------------------


#: Selected slots the source kernel processes per pipelined tile. The block
#: size is part of the numerics, not a tuning knob: the running maximum, the
#: rescaling of the accumulator and the BF16 materialisation of the attention
#: weights all happen once per tile, so a single-pass softmax lands on
#: measurably different values.
SPARSE_ATTN_BLOCK = 64


def _accumulating_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched ``a @ b`` with the operands' own dtype and an FP32 accumulator.

    The source kernel declares ``q_shared`` / ``kv_shared`` / ``acc_s_cast`` as
    BF16 shared memory and every ``T.gemm`` accumulates into an FP32 fragment,
    i.e. BF16 x BF16 -> FP32 on tensor cores. Widening the operands to FP32
    first and running an FP32 GEMM computes the same *mathematical* quantity
    but sums the products in a different order, and that ordering is
    observable: on H100 it disagrees with the kernel on 94% of the score
    elements and 14% of the value elements, versus bit-exact agreement when the
    operands stay BF16. So the reference keeps the storage dtype and asks Torch
    for the FP32 accumulator instead of casting up.

    Operands are promoted to a common dtype rather than assumed equal, so the
    same code path serves an all-FP32 CPU unit test and the BF16 CUDA replay.
    When that common dtype is already FP32 the accumulator request is a no-op
    and a plain ``bmm`` is bit-identical, which keeps the CPU tests working;
    ``bmm``'s ``out_dtype`` overload is registered for CUDA only. Narrow
    operands on a backend without it raise rather than silently widening,
    because widening is the very thing this helper exists to avoid.
    """
    dtype = torch.promote_types(a.dtype, b.dtype)
    a, b = a.to(dtype), b.to(dtype)
    if dtype in (torch.float32, torch.float64):
        return torch.bmm(a, b)
    return torch.bmm(a, b, out_dtype=torch.float32)


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    block: int = SPARSE_ATTN_BLOCK,
) -> torch.Tensor:
    """Independent reference for the sparse MLA kernel's online softmax.

    MLA stores one latent row per position that serves as both key and value,
    so ``kv`` is indexed once. ``topk_idxs`` selects the visible rows for each
    query and ``-1`` marks a padded slot. The sink contributes ``exp(sink)`` to
    the softmax *denominator only* --- it adds no value mass, so it can only
    shrink the output, never steer it.

    The arithmetic follows the source's FlashAttention-style recurrence rather
    than a single-pass softmax, because that ordering is observable in the
    result. Per tile of ``block`` selected slots the source:

    1. seeds the scores at ``0`` for live slots and ``-inf`` for padded ones,
       accumulates ``q @ kv^T`` into them in FP32, then applies the scale;
    2. extends a *running* row maximum across tiles (``clear=False``) and
       rescales the FP32 output accumulator and the running denominator by
       ``exp(prev_max - new_max)``;
    3. reduces the denominator from the FP32 numerators, but **casts those
       numerators to BF16** (``acc_s_cast``) before the value GEMM, so every
       attention weight carries ~2^-8 of relative error into the weighted sum;
    4. adds ``exp(sink - final_max)`` to the denominator once, at the end,
       against the *global* maximum, and only then divides.

    Both GEMMs keep the operands in their storage dtype and accumulate in FP32
    (see :func:`_accumulating_matmul`); widening to FP32 first would compute
    the same quantity in a different summation order and disagree with the
    kernel on most elements. The recurrence is reproduced here; the tile
    arithmetic itself is plain Torch, with no tilelang, no CUDA and no shared
    code with the source.
    """
    bsz, seqlen, heads, dim = q.shape
    n_sel = topk_idxs.shape[-1]
    rows = bsz * seqlen

    acc_o = torch.zeros(bsz, seqlen, heads, dim, dtype=torch.float32, device=q.device)
    sum_exp = torch.zeros(bsz, seqlen, heads, dtype=torch.float32, device=q.device)
    scores_max = torch.full(
        (bsz, seqlen, heads), float("-inf"), dtype=torch.float32, device=q.device
    )

    for start in range(0, n_sel, block):
        idx = topk_idxs[..., start : start + block]
        valid = idx >= 0
        sel = torch.gather(
            kv.unsqueeze(1).expand(bsz, seqlen, kv.shape[1], dim),
            2,
            idx.clamp(min=0).long().unsqueeze(-1).expand(*idx.shape, dim),
        )
        # The source zeroes padded rows in shared memory, so they contribute
        # nothing to either GEMM even before the -inf mask takes effect.
        sel = torch.where(valid.unsqueeze(-1), sel, sel.new_zeros(()))
        sel_rows = sel.reshape(rows, -1, dim)

        acc_s = _accumulating_matmul(q.reshape(rows, heads, dim), sel_rows.transpose(1, 2))
        acc_s = acc_s.reshape(bsz, seqlen, heads, -1) * softmax_scale
        acc_s = torch.where(valid.unsqueeze(2), acc_s, float("-inf"))

        prev_max, scores_max = scores_max, torch.maximum(scores_max, acc_s.amax(dim=-1))
        # -inf - -inf is a NaN, and it only arises before any live slot has
        # been seen; both accumulators are still exactly zero there.
        rescale = torch.nan_to_num(torch.exp(prev_max - scores_max), nan=0.0)
        probs = torch.nan_to_num(torch.exp(acc_s - scores_max.unsqueeze(-1)), nan=0.0)

        sum_exp = sum_exp * rescale + probs.sum(dim=-1)
        # ``acc_s_cast``: the numerators are materialised at the activation
        # dtype before the value GEMM, so each weight carries ~2^-8 of
        # relative error into the weighted sum exactly as the source does.
        tile = _accumulating_matmul(probs.to(q.dtype).reshape(rows, heads, -1), sel_rows)
        acc_o = acc_o * rescale.unsqueeze(-1) + tile.reshape(bsz, seqlen, heads, dim)

    sum_exp = sum_exp + torch.exp(attn_sink.float().view(1, 1, heads) - scores_max)
    return (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)


# ---------------------------------------------------------------------------
# MoE routing and experts.
# ---------------------------------------------------------------------------


def moe_route(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    *,
    topk: int,
    route_scale: float,
    bias: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(routing_weights, expert_ids, raw_scores)``.

    Two variants coexist. Layers 0-2 route by a checkpoint token-id table, so
    the hidden state picks the *weights* but not the *experts*. Later layers
    select by score plus a correction bias --- and the bias shifts selection
    only: the returned weights come from the unbiased scores.
    """
    scores = F.softplus(F.linear(x.float(), gate_weight.float())).sqrt()
    selection = scores if bias is None else scores + bias.float()

    if tid2eid is not None:
        if input_ids is None:
            raise ValueError("hash-routed layers need input_ids")
        indices = tid2eid[input_ids.flatten().long()].long()
    else:
        indices = selection.topk(topk, dim=-1)[1]

    weights = scores.gather(1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * route_scale, indices, scores


def expert_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    *,
    swiglu_limit: float,
    weights: torch.Tensor | None = None,
    quantize_act: bool = True,
) -> torch.Tensor:
    """Clamped SwiGLU expert over dequantised MXFP4 weights.

    The clamp is asymmetric on purpose: the gate is bounded above only, while
    the up projection is bounded on both sides.

    ``quantize_act`` reproduces what the source's ``linear()`` does before a
    quantised GEMM --- it rounds the activation through blockwise FP8 first.
    Leaving it out is not a harmless simplification: the activation round trip
    is part of the arithmetic the network was trained against, and skipping it
    shows up as a large max-abs outlier even while cosine stays near 1.
    """
    dtype = x.dtype

    def project(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.linear(fp8_quant_dequant(v, 128) if quantize_act else v, w)

    gate = project(x, w1).float()
    up = project(x, w3).float()
    if swiglu_limit > 0:
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h
    return project(h.to(dtype), w2)


# ---------------------------------------------------------------------------
# mHC (hyper-connections).
# ---------------------------------------------------------------------------


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the mix vector into pre-weights, post-weights and a Sinkhorn matrix.

    Layout of the ``(2 + hc) * hc`` mix vector: ``[0:hc)`` are the pre
    weights, ``[hc:2*hc)`` the post weights, and the remaining ``hc * hc``
    entries the combination matrix, row-major. The matrix is softmaxed over
    rows, column-normalised, then alternately row/column normalised for
    ``iters - 1`` further rounds.
    """
    m = mixes.float()
    scale, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(m[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult])

    comb = m[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]
    comb = comb.unflatten(-1, (hc_mult, hc_mult))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce ``hc_mult`` residual streams to one. ``x`` is ``[b, s, hc, d]``."""
    shape, dtype = x.shape, x.dtype
    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, hc_fn.float()) * rsqrt
    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, iters=iters, eps=hc_eps
    )
    y = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
    return y.to(dtype), post, comb


def hc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """Expand one stream back to ``hc_mult`` and mix in the residual streams."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def hc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Final head reduction: a plain sigmoid gate, no Sinkhorn."""
    shape, dtype = x.shape, x.dtype
    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    return torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2).to(dtype)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def compare(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Scale-free comparison metrics for a golden check.

    ``rel_max_abs`` divides by the RMS of the reference so a tolerance cannot
    be passed simply by both tensors being small.
    """
    g, r = got.detach().float().flatten(), ref.detach().float().flatten()
    diff = (g - r).abs()
    rms = r.square().mean().sqrt().clamp(min=1e-12)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_max_abs": (diff.max() / rms).item(),
        "cosine": F.cosine_similarity(g, r, dim=0).item(),
        "ref_rms": rms.item(),
        "finite": bool(torch.isfinite(g).all()),
    }
