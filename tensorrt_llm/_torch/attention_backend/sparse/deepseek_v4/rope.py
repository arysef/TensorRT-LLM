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
"""DeepSeek-V4's RoPE frequency table, built the way the checkpoint builds it.

The rotary table is a *model constant*: the checkpoint's ``inference/model.py``
defines it exactly, in ``precompute_freqs_cis``, as

    freqs = 1.0 / (base ** (arange(0, dim, 2, float32) / dim))     # float32 pow
    freqs = freqs / factor * (1 - smooth) + freqs * smooth          # YaRN ramp
    freqs_cis = polar(ones, outer(arange(seqlen), freqs))           # CUDA sincos

``RopeEmbeddingUtils`` computes the same mathematical constant by a different
numerical recipe --- a float64 ``**`` rounded to float32, then NumPy's
single-precision ``cos``/``sin`` on the host. Both are accurate; they are not
the same float32 values. Measured on the real checkpoint at
``max_positions=4096``, ``dim=64``, they differ by up to 9.5e-07, which is 816
of 8224 cosines and 1000 of 8224 sines over a 257-token prompt.

That difference is invisible in isolation and observable end to end. Replaying
real checkpoint activations through the SM90 sparse path on eight ranks, the
NumPy table left 1 to 4 of 1,052,672 BF16 query elements one storage step away
from the source's; on one rank that step landed on an element of magnitude 4.5,
which is 0.089 of the tensor RMS and three times the registered tolerance. With
the table below the same comparison is *bit-exact* on every rank --- the
divergence was entirely the table, never the RoPE kernel: driving the native
``mla_rope_inplace`` with this table and with the source's own tensor produce
identical results, and so does the source's own complex multiply.

This is deliberately not a change to ``RopeEmbeddingUtils``. Every other model
is entitled to its own reference's convention, and the shared builder is the
right default; DeepSeek-V4 simply has a reference that pins the recipe. It is
also deliberately not restricted to SM90: a positional-encoding table that
depended on which GPU executed the layer would make one model mean two things.
"""

from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm.functional import RotaryScalingType

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import RopeParams

#: ``(RopeParams, interleave) -> weakref to the built cos/sin table``.
#:
#: The table is a constant of the model, and at checkpoint scale it is large:
#: ``max_position_embeddings`` is 1,048,576, so one interleaved float32 table is
#: 512 MiB. All 43 decoder layers of a compression class share one ``RopeParams``
#: *value*, so without this cache every layer would build and hold its own copy
#: and the model would not fit in HBM. ``RopeParams`` is ``unsafe_hash=True``, so
#: equal parameters hash equal --- the same keying the shared builder's own cache
#: in ``RopeParams.create_rope_const_params`` uses. Values are weak, so a table
#: is freed once the last module referencing it goes away.
_TABLE_CACHE: dict[tuple, "weakref.ReferenceType[torch.Tensor]"] = {}


def _checkpoint_inv_freq(rope: "RopeParams") -> torch.Tensor:
    """``precompute_freqs_cis``'s inverse frequencies, ``dim // 2`` of them.

    Ordering is load-bearing rather than stylistic: ``base ** x`` in float32
    and the same expression in float64 rounded to float32 differ in the last
    place for most entries, and every such difference is multiplied by the
    position index before the cosine sees it.
    """
    dim = rope.dim
    base = float(rope.theta)
    device = torch.device("cuda")

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    # `precompute_freqs_cis` gates the YaRN ramp on `original_seq_len > 0`, and
    # `_deepseek_v4_pos_embd_params` expresses that same choice as the scaling
    # type: SWA-only layers get `none` and base theta, compressed layers get
    # `yarn` and the compression theta.
    if rope.scale_type == RotaryScalingType.yarn:
        original = float(rope.original_max_positions)
        factor = float(rope.scale)

        def correction_dim(rotations: float) -> float:
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction_dim(rope.beta_fast)), 0)
        high = min(math.ceil(correction_dim(rope.beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low),
            0,
            1,
        )
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    return freqs


def _checkpoint_angles(rope: "RopeParams", inv_freq: torch.Tensor) -> torch.Tensor:
    """``outer(arange(seqlen), freqs)``, the per-position angles."""
    positions = torch.arange(rope.max_positions, device=inv_freq.device)
    return torch.outer(positions, inv_freq)


def deepseek_v4_rope_const_params(
    rope: "RopeParams", interleave: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(inv_freq, cos_sin)`` in ``create_rope_const_params``'s two layouts.

    ``interleave=True`` returns the flat buffer the native attention op reads.
    ``mlaKernels.cu`` addresses it as ``float2 const*`` and strides one position
    by ``ROPE_DIM`` float2 entries, so a position occupies ``2 * dim`` floats:
    the ``dim // 2`` ``(cos, sin)`` pairs, written twice. That second copy is
    what ``RopeEmbeddingUtils`` calls ``duplicate_data``, and
    ``RopeParams.from_config`` turns it on for every model carrying
    ``qk_rope_head_dim`` --- DeepSeek-V4 included. Emitting a single copy here
    would halve the position stride the kernel assumes and silently rotate every
    token past the first by the wrong angle, so the flag is honoured rather than
    asserted away.

    ``interleave=False`` returns the ``[1, max_positions * dim]`` buffer
    :class:`RotaryEmbedding` reshapes to ``[max_positions, 2, dim // 2]``, all
    cosines then all sines. The shared builder drops the duplicate in that
    layout (it slices ``[:, :dim // 2, :]``), so this one never duplicates.
    """
    inv_freq = _checkpoint_inv_freq(rope)
    angles = _checkpoint_angles(rope, inv_freq)
    # `torch.polar`, not `cos`/`sin`: it is the call the checkpoint makes, and
    # matching the call is the whole point of this module.
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    cos, sin = freqs_cis.real.float(), freqs_cis.imag.float()

    if interleave:
        cos_sin = torch.stack([cos, sin], dim=-1)  # [pos, dim/2, 2]
        if rope.duplicate_data:
            cos_sin = torch.cat([cos_sin, cos_sin], dim=1)  # [pos, dim, 2]
    else:
        cos_sin = torch.stack([cos, sin], dim=1)  # [pos, 2, dim/2]
    return inv_freq.contiguous(), cos_sin.reshape(1, -1).contiguous()


def _cached_rope_table(rope: "RopeParams", interleave: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`deepseek_v4_rope_const_params`, memoized by parameter value.

    Inside a model context the table is held *strongly*, in the same
    ``extra_attrs`` dict the shared builder caches into, so it lives exactly as
    long as the model does. A weak cache is not enough here: the installer hands
    each module a view of the table, and a view keeps the storage alive without
    keeping the original Python object alive, so weak entries were observed
    dying between layers and every layer rebuilt its own 512 MiB copy.

    Outside a model context --- focused tests, the evidence replay --- there is
    no dict whose lifetime bounds the cache, so the fallback stays weak. Those
    callers size the table by their own short sequence length, where a miss
    costs milliseconds.

    ``inv_freq`` is small and cheap, so only the table is cached; rebuilding the
    frequencies costs a 32-element pow.
    """
    from tensorrt_llm._torch.utils import get_model_extra_attrs

    key = (rope, interleave)
    attrs = get_model_extra_attrs()
    store = _TABLE_CACHE if attrs is None else attrs.setdefault("deepseek_v4_rope_tables", {})
    cached = store.get(key)
    if isinstance(cached, weakref.ref):
        cached = cached()
    if cached is not None:
        return _checkpoint_inv_freq(rope).contiguous(), cached
    inv_freq, cos_sin = deepseek_v4_rope_const_params(rope, interleave=interleave)
    store[key] = weakref.ref(cos_sin) if attrs is None else cos_sin
    return inv_freq, cos_sin


def install_deepseek_v4_rope_table(
    module: torch.nn.Module,
    rope: Optional["RopeParams"],
    *,
    interleave: bool,
    attr: str = "rotary_cos_sin",
    inv_freq_attr: Optional[str] = None,
) -> None:
    """Replace ``module``'s rotary table with the checkpoint's own.

    A shared tensor is assigned rather than an in-place copy: the shared builder
    memoizes its result per ``(RopeParams, interleave)`` in the model's extra
    attributes, so writing through the tensor would reach every other module
    that resolved to the same cache entry. This side memoizes on the same key
    for the same reason it matters at checkpoint scale --- see
    :data:`_TABLE_CACHE`.

    The element count is checked against whatever the shared builder already
    installed, because the two must agree on the position stride the native
    kernel reads. A mismatch is a layout bug in this module (most likely
    ``duplicate_data``), not a condition to paper over, so it raises.
    """
    if rope is None or rope.dim == 0:
        return
    inv_freq, cos_sin = _cached_rope_table(rope, interleave)
    existing = getattr(module, attr, None)
    if existing is not None:
        if existing.numel() != cos_sin.numel():
            raise RuntimeError(
                f"DeepSeek-V4 rotary table for {type(module).__name__}.{attr} would have "
                f"{cos_sin.numel()} elements but the shared builder produced "
                f"{existing.numel()} (max_positions={rope.max_positions}, dim={rope.dim}, "
                f"duplicate_data={rope.duplicate_data}, interleave={interleave}); "
                "the layouts have diverged."
            )
        cos_sin = cos_sin.reshape(existing.shape)
    setattr(module, attr, cos_sin)
    if inv_freq_attr is not None and getattr(module, inv_freq_attr, None) is not None:
        setattr(module, inv_freq_attr, inv_freq)


def deepseek_v4_rotary_embedding(rope: "RopeParams", **kwargs) -> torch.nn.Module:
    """:class:`RotaryEmbedding` carrying the checkpoint's table."""
    from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding

    embedding = RotaryEmbedding(rope, **kwargs)
    install_deepseek_v4_rope_table(embedding, rope, interleave=False)
    return embedding
