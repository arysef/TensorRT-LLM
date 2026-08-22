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
"""Hopper (SM90) semantics for the DeepSeek-V4 Compressor, Indexer and selection.

The sparse-attention *kernel* and its cache wiring are covered by
`test_deepseek_v4_sm90.py` / `test_deepseek_v4_sm90_runtime.py`. This file
covers the operators that feed it: the Compressor's gated pooling and its
retained decode state, the Indexer's scoring and exact top-k, and the
ratio-128 selection rule --- each driven on a real H100 and compared against
the independent source ladder goldens in
`tests/integration/defs/accuracy/deepseek_v4_flash_h100/torch_goldens.py`,
which were written from the checkpoint's own `inference/model.py` and
`inference/kernel.py` rather than from TensorRT-LLM.

Tolerances are the pre-registered ones in `manifests/tolerances.json`; they are
read from that file rather than restated, so this file cannot silently loosen
a gate. Discrete decisions (Indexer top-k, selection rules) are exact.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import (
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
    DeepseekV4AttentionType,
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttentionMetadata,
    sm90_quant,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.compressor import Compressor
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.indexer import DeepseekV4Indexer
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

_EVIDENCE = (
    Path(__file__).resolve().parents[5]
    / "integration"
    / "defs"
    / "accuracy"
    / "deepseek_v4_flash_h100"
)


def _load_goldens():
    name = "deepseek_v4_flash_h100_torch_goldens"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _EVIDENCE / "torch_goldens.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tg = _load_goldens()
#: The golden's own Walsh-Hadamard butterfly, used as the reference rotation
#: whenever the compressor applies its native one.
_hadamard = tg.hadamard_transform
TOL = json.loads((_EVIDENCE / "manifests" / "tolerances.json").read_text())["modules"]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# Checkpoint geometry: 512-wide latent row (448 non-RoPE + 64 RoPE), 128-token
# window and KV page, ratio-4 CSA / ratio-128 HCA, indexer top-k 512 over
# 128-wide indexer heads.
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
HIDDEN = 1024
WINDOW = 128
TOKENS_PER_BLOCK = 128
INDEX_HEAD_DIM = 128
#: The source's `act_quant` block size at both `model.py` call sites.
FP8_BLOCK = 64
INDEX_TOPK = 512
INDEX_HEADS = 64
COMPRESS_RATIOS = [1, 4, 128]
RATIO_TO_LAYER = {1: 0, 4: 1, 128: 2}
MAX_SEQ_LEN = 1024
EPS = 1e-6


def _assert_within(metrics: dict, module: str, label: str) -> None:
    """Gate on the pre-registered tolerance for `module`, read from the manifest."""
    limits = TOL[module]
    assert metrics["finite"], f"{label}: non-finite output"
    assert metrics["cosine"] >= limits["cosine_min"], (
        f"{label}: cosine {metrics['cosine']:.6f} < {limits['cosine_min']} ({module})"
    )
    assert metrics["rel_max_abs"] <= limits["rel_max_abs_max"], (
        f"{label}: rel_max_abs {metrics['rel_max_abs']:.4f} > "
        f"{limits['rel_max_abs_max']} ({module})"
    )


def _pos_embd(compress_ratio: int, max_seq_len: int = MAX_SEQ_LEN) -> PositionalEmbeddingParams:
    """The per-ratio RoPE contract `_deepseek_v4_pos_embd_params` applies."""
    rope = RopeParams(
        dim=ROPE_DIM,
        max_positions=max_seq_len,
        original_max_positions=65536,
        max_seq_len=max_seq_len,
        beta_fast=32,
        beta_slow=1,
    )
    if compress_ratio > 1:
        rope.theta = 160000.0
        rope.scale_type = RotaryScalingType.yarn
        rope.scale = 16.0
        rope.mscale = 0.0
        rope.mscale_all_dim = 0.0
        pos_type = PositionEmbeddingType.yarn
    else:
        rope.theta = 10000.0
        rope.scale_type = RotaryScalingType.none
        rope.scale = 1.0
        pos_type = PositionEmbeddingType.rope_gptj
    return PositionalEmbeddingParams(type=pos_type, rope=rope, is_neox=False)


def _golden_freqs(compress_ratio: int, max_seq_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    if compress_ratio > 1:
        return tg.yarn_freqs_cis(ROPE_DIM, max_seq_len, 65536, 160000.0, 16.0, 32, 1).cuda()
    return tg.yarn_freqs_cis(ROPE_DIM, max_seq_len, 0, 10000.0, 1.0, 32, 1).cuda()


def _sparse_config():
    return DeepSeekV4SparseAttentionConfig(
        index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        window_size=WINDOW,
        compress_ratios=list(COMPRESS_RATIOS),
        index_topk=INDEX_TOPK,
        skip_indexer_for_short_seqs=False,
    )


def _cache_manager(sparse_config, max_seq_len: int = MAX_SEQ_LEN):
    return DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            max_tokens=max_seq_len * 2, enable_block_reuse=False, event_buffer_max_size=0
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=len(COMPRESS_RATIOS),
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        max_input_len=max_seq_len,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=max_seq_len + 1,
        sparse_attn_config=sparse_config,
    )


def _metadata(cache_manager, sparse_config, seq_len, cached_len, num_contexts):
    metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor([seq_len], dtype=torch.int),
        request_ids=[0],
        max_num_requests=1,
        num_contexts=num_contexts,
        prompt_lens=[cached_len + seq_len],
        max_num_tokens=seq_len,
        kv_cache_manager=cache_manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached_len]),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=sparse_config,
    )
    metadata.prepare()
    return metadata


def _compressor(layer_idx, ratio, seed):
    """A Compressor with reproducible weights, plus the raw wkv/wgate halves.

    The module fuses the source's two projections into one `wkv_gate`; the
    golden takes them separately, so the split order is part of the contract
    being tested (a swap is caught by `test_..._weight_layout` below).
    """
    comp = Compressor(
        MLAParams(hidden_size=HIDDEN, qk_rope_head_dim=ROPE_DIM, qk_nope_head_dim=NOPE_DIM),
        layer_idx,
        ratio,
        EPS,
        False,
        _pos_embd(ratio),
        dtype=torch.bfloat16,
        kv_cache_dtype="default",
        rotate_activation=False,
        # What `DeepseekV4TrtllmAttention` asks for on SM90: the checkpoint's
        # FP32 gate/KV projection and its post-pool FP8 simulation.
        simulate_source_act_quant=True,
    ).cuda()
    torch.manual_seed(seed)
    state_dim = comp.state_dim
    wkv = (torch.randn(state_dim, HIDDEN, device="cuda") * (HIDDEN**-0.5)).bfloat16()
    wgate = (torch.randn(state_dim, HIDDEN, device="cuda") * (HIDDEN**-0.5)).bfloat16()
    with torch.no_grad():
        comp.wkv_gate.weight.copy_(torch.cat([wkv, wgate], dim=0))
        comp.norm.weight.copy_((torch.randn(HEAD_DIM, device="cuda") * 0.05 + 1.0).bfloat16())
        comp.ape.copy_(torch.randn(ratio, state_dim, device="cuda", dtype=torch.float32) * 0.1)
    return comp, wkv, wgate


def _golden_compressed(
    comp,
    wkv,
    wgate,
    x,
    ratio,
    *,
    stage="cache",
    head_dim=HEAD_DIM,
    hadamard=None,
    max_seq_len=MAX_SEQ_LEN,
):
    """Source-golden compressed rows for a whole prefill.

    ``stage="cache"`` is the source's complete `Compressor.forward`, ending
    with its quantisation simulation; nothing is substituted for it. Only
    ``stage="pool"`` alters the golden, and it does so by *removing* later
    stages so a pooling error cannot hide behind them.
    """
    saved = (tg.rms_norm, tg.apply_rope, tg.fp8_quant_dequant, tg.fp4_quant_dequant)
    if stage == "pool":
        tg.rms_norm = lambda t, w, eps: t
        tg.apply_rope = lambda t, f, inverse=False: t
        tg.fp8_quant_dequant = lambda t, block_size=128: t
        tg.fp4_quant_dequant = lambda t, block_size=32: t
    else:
        assert stage == "cache", f"unknown golden stage {stage!r}"
    try:
        return tg.compressor_prefill(
            x.unsqueeze(0),
            wkv,
            wgate,
            comp.ape,
            comp.norm.weight,
            _golden_freqs(ratio, max_seq_len),
            ratio=ratio,
            head_dim=head_dim,
            rope_dim=ROPE_DIM,
            eps=EPS,
            rotate=hadamard is not None,
            hadamard=hadamard,
        ).squeeze(0)
    finally:
        tg.rms_norm, tg.apply_rope, tg.fp8_quant_dequant, tg.fp4_quant_dequant = saved


def _unsimulated_reference(comp, wkv, wgate, x, ratio):
    """The source golden with only its quantisation simulation removed.

    Diagnostic contrast for `test_..._carry_the_sources_fp8_simulation`: it
    measures how far the *old* SM90 behaviour was from the source. It is never
    a pass reference --- every gate in this file compares against the complete
    golden.
    """
    saved = tg.fp8_quant_dequant
    tg.fp8_quant_dequant = lambda t, block_size=128: t
    try:
        return _golden_compressed(comp, wkv, wgate, x, ratio, stage="cache")
    finally:
        tg.fp8_quant_dequant = saved


def _cached_rows(cache_manager, layer_idx, count):
    """Read the paged COMPRESS pool rows a prefill just produced.

    `Compressor.forward` returns the *pre*-postprocess pooled tensor in BF16
    cache mode; the normalised, RoPE'd row it scatters lives in the cache, so
    that is what has to be compared against the source.
    """
    buffers = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS)
    block = cache_manager.compressed_block_sizes[layer_idx]
    pages = cache_manager.get_cache_indices(0, layer_idx, DeepseekV4AttentionType.COMPRESS)
    return torch.stack([buffers[pages[i // block], i % block] for i in range(count)])


def _prefill(cache_manager, sparse_config, seq_len):
    request = LlmRequest(
        request_id=0,
        max_new_tokens=16,
        input_tokens=list(range(seq_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    assert cache_manager.prepare_context(request)
    assert cache_manager.resize_context(request, request.context_chunk_size)
    return request, _metadata(cache_manager, sparse_config, seq_len, 0, num_contexts=1)


def _advance_to_generation(cache_manager, request, prompt_len):
    scheduled = ScheduledRequests()
    scheduled.context_requests_last_chunk = [request]
    request.context_current_position = prompt_len
    request.add_new_token(prompt_len, 0)
    cache_manager.update_context_resources(scheduled)
    assert cache_manager.try_allocate_generation(request)


# ---------------------------------------------------------------------------
# Activation-quantisation primitives.
#
# Every SM90 source-faithfulness claim above the Compressor rests on these two
# helpers, so they are pinned to *bit* equality with the independent golden
# rather than to a tolerance: a tolerance on a quantiser cannot distinguish
# "the grid is right" from "the grid is wrong but the inputs were benign".
# ---------------------------------------------------------------------------


def _fp8_without_the_source_floor(x: torch.Tensor, block: int) -> torch.Tensor:
    """The FP8 round trip with `act_quant_kernel`'s ``amax`` floor removed.

    Present only so the tests below can prove the floor is load-bearing. If
    this produced the same answer as the real helper on the swept magnitudes,
    the parity assertions would pass with the floor missing --- which is
    exactly the false pass this file is guarding against.
    """
    flat = x.reshape(-1, x.shape[-1]).float()
    blocks = flat.unflatten(-1, (-1, block))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * scale).flatten(-2).reshape(x.shape).to(x.dtype)


@pytest.mark.parametrize("exponent", [-30, -22, -18, -14, -10, -6, -2, 0, 4, 8])
@pytest.mark.parametrize("width", [FP8_BLOCK, NOPE_DIM])
def test_sm90_fp8_activation_simulation_is_bit_equal_to_the_source(exponent, width):
    """`act_quant(x, 64, "ue8m0", ..., inplace=True)`, swept across its floor.

    ``inference/kernel.py:79`` pins the block maximum with
    ``amax = max(amax, 1e-4)`` *before* choosing the power-of-two scale. That
    is seven orders of magnitude above FP32's smallest normal, so it is not a
    divide-by-zero guard: it fixes the scale for every low-magnitude block and
    flushes anything under roughly ``2 ** -23`` to zero. Omitting it leaves
    TensorRT-LLM strictly *more* precise than the checkpoint, which is still a
    divergence, and it is invisible on realistic activations --- normalised
    latent rows have ``amax`` of order 1 --- so only a deliberate sweep of the
    underflow band catches it.

    The sweep spans ``2 ** -30`` (deep under the floor) to ``2 ** 8`` (well
    over it) at both the 64-wide block and the full 448-wide non-RoPE row the
    Compressor and the window latent actually quantise.
    """
    torch.manual_seed(101 + exponent)
    x = (torch.randn(37, width, device="cuda") * (2.0**exponent)).bfloat16()

    got = sm90_quant.fp8_quant_dequant_(x.clone(), FP8_BLOCK)
    ref = tg.fp8_quant_dequant(x, FP8_BLOCK)
    assert torch.equal(got, ref), (
        f"2**{exponent}, width {width}: {int((got != ref).sum())} of {x.numel()} values "
        f"differ from the source's FP8 grid (max_abs {float((got.float() - ref.float()).abs().max()):.3e})"
    )

    # Under the floor the answer must actually depend on it.
    no_floor = _fp8_without_the_source_floor(x, FP8_BLOCK)
    below_floor = (
        float(x.float().abs().reshape(-1, FP8_BLOCK).amax(dim=-1).max()) < sm90_quant.FP8_MIN_AMAX
    )
    assert below_floor == (not torch.equal(no_floor, ref)), (
        f"2**{exponent}: block amax {'below' if below_floor else 'above'} the 1e-4 floor, "
        "but dropping the floor "
        f"{'did not change' if below_floor else 'changed'} the result; this case cannot "
        "detect a missing floor"
    )


@pytest.mark.parametrize("exponent", [-135, -128, -126, -20, -4, 0, 6])
def test_sm90_fp4_activation_simulation_is_bit_equal_to_the_source(exponent):
    """`fp4_act_quant(x, 32, inplace=True)`, swept across its own floor.

    ``fp4_quant_kernel`` floors ``amax`` at ``6 * 2 ** -126`` --- a genuine
    denormal guard rather than a precision decision, unlike the FP8 one --- and
    the sweep reaches below it so the guard is exercised rather than assumed.
    E2M1 has eight levels, so bit equality here also pins the
    round-to-nearest-even tie rule at every level boundary.
    """
    torch.manual_seed(211 + exponent)
    x = (torch.randn(11, INDEX_HEAD_DIM, device="cuda") * (2.0**exponent)).bfloat16()
    got = sm90_quant.fp4_quant_dequant_(x.clone())
    ref = tg.fp4_quant_dequant(x, 32)
    assert torch.equal(got, ref), (
        f"2**{exponent}: {int((got != ref).sum())} of {x.numel()} values differ from the "
        "source's E2M1 grid"
    )


def test_sm90_fp8_simulation_pins_the_scale_in_the_underflow_band():
    """The named low-magnitude regression, stated as positive facts.

    A 64-wide BF16 block whose maximum is ~1e-8 sits entirely under the
    source's ``amax = max(amax, 1e-4)`` floor, which pins the scale at
    ``2 ** -22`` no matter how small the block is. Two consequences are
    asserted directly rather than inferred from a metric: every surviving
    value lands on that scale's E4M3 lattice, and values a few decades further
    down --- the source's own example is 5.009e-13 --- reach exactly zero
    because they fall below half an FP8 subnormal step.

    The previous SM90 build chose a finer scale instead, preserved those
    values, and measured ``rel_max_abs = 0.0573`` against the golden, over the
    registered `compressor` limit of 0.03.
    """
    torch.manual_seed(7)
    x = (torch.randn(1, FP8_BLOCK, device="cuda") * 1e-8).bfloat16()
    # A few decades further down, inside the band the pinned scale zeroes.
    x[0, ::8] = (torch.randn(FP8_BLOCK // 8, device="cuda") * 5e-13).bfloat16()
    assert float(x.abs().max()) < sm90_quant.FP8_MIN_AMAX, "the block must sit under the floor"

    got = sm90_quant.fp8_quant_dequant_(x.clone(), FP8_BLOCK)
    ref = tg.fp8_quant_dequant(x, FP8_BLOCK)
    assert torch.equal(got, ref), (
        f"{int((got != ref).sum())} of {FP8_BLOCK} underflow-band values differ from the source"
    )
    metrics = tg.compare(got, ref)
    assert metrics["rel_max_abs"] <= TOL["compressor"]["rel_max_abs_max"], (
        f"underflow band: rel_max_abs {metrics['rel_max_abs']:.6g} > "
        f"{TOL['compressor']['rel_max_abs_max']}"
    )

    # The scale is 2**ceil(log2(1e-4 / 448)) = 2**-22, and every finite E4M3
    # value is an integer multiple of its subnormal ulp 2**-9, so every output
    # must be an integer multiple of 2**-31. A finer (unfloored) scale would
    # put values off that lattice, so this pins the floor rather than merely
    # observing agreement.
    lattice = 2.0**-31
    quotient = got.float() / lattice
    assert torch.equal(quotient, quotient.round()), (
        "an underflow-band value is not on the 2**-22-scaled E4M3 lattice, so the "
        "source's amax floor did not pin the scale"
    )
    assert (got[0, ::8] == 0).all(), (
        f"values ~5e-13 must fall below half an FP8 subnormal step at this scale and "
        f"reach zero, got {got[0, ::8][:4].tolist()}"
    )
    assert _fp8_without_the_source_floor(x, FP8_BLOCK)[0, ::8].any(), (
        "without the floor those values survive; if that is no longer true this test "
        "cannot detect the floor going missing"
    )


# ---------------------------------------------------------------------------
# Compressor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [4, 128])
def test_sm90_compressor_pooling_matches_the_source_golden(ratio):
    """`compressor_prefill_reduction` on H100 vs the source's gated pooling.

    Ratio 4 pools *overlapping* windows (doubled projection width plus the
    shift-by-one transform); ratio 128 pools disjoint ones. Both are compared
    against the independent golden with the norm, RoPE and quantisation stages
    disabled, so a pooling error cannot hide behind a later stage.
    """
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        seq_len = 512
        _, metadata = _prefill(cache_manager, sparse_config, seq_len)
        comp, wkv, wgate = _compressor(layer_idx, ratio, seed=11 + ratio)
        torch.manual_seed(3)
        x = (torch.randn(seq_len, HIDDEN, device="cuda") * 0.5).bfloat16()

        pooled, _ = comp(x, metadata)
        assert pooled.shape == (seq_len // ratio, HEAD_DIM)
        ref = _golden_compressed(comp, wkv, wgate, x, ratio, stage="pool")
        _assert_within(tg.compare(pooled, ref), "compressor", f"ratio {ratio} pooling")
    finally:
        cache_manager.shutdown()


@pytest.mark.parametrize("ratio", [4, 128])
def test_sm90_compressor_projects_in_the_sources_two_gemm_shape(ratio):
    """The FP32 projection must be issued as the source issues it: two GEMMs.

    The source runs `self.wkv(x)` and `self.wgate(x)` as separate
    [state_dim, dim] projections. Fusing them into one [2 * state_dim, dim]
    GEMM computes the same quantity, and cuBLAS answers it with a different
    accumulation: measured on the real checkpoint at layer 40, that moved 17
    of 32768 pooled values by one BF16 ULP, and the blockwise-64 FP8 quantiser
    two steps later turned two of them into a whole FP8 level.

    Rounding differences that small are data-dependent -- they vanish on
    random inputs at these shapes -- so this pins the *shape of the call*,
    which is the contract that fix established, rather than a numeric
    difference that would not reproduce here.
    """
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        seq_len = 256
        _, metadata = _prefill(cache_manager, sparse_config, seq_len)
        comp, _, _ = _compressor(layer_idx, ratio, seed=29)
        torch.manual_seed(9)
        x = (torch.randn(seq_len, HIDDEN, device="cuda") * 0.5).bfloat16()

        real_linear = F.linear
        shapes = []

        def spy(inp, weight, bias=None):
            shapes.append(tuple(weight.shape))
            return real_linear(inp, weight, bias)

        with patch.object(F, "linear", spy):
            comp(x, metadata)

        state_dim = comp.state_dim
        assert shapes.count((state_dim, HIDDEN)) == 2, (
            f"expected two [{state_dim}, {HIDDEN}] projections as the source issues "
            f"them; saw {shapes}"
        )
        assert (2 * state_dim, HIDDEN) not in shapes, (
            "the fused wkv_gate GEMM is back; it rounds differently from the "
            f"source's two projections. Saw {shapes}"
        )
    finally:
        cache_manager.shutdown()


@pytest.mark.parametrize("ratio", [4, 128])
def test_sm90_compressor_weight_layout_is_kv_then_gate(ratio):
    """The fused `wkv_gate` must be [wkv; wgate], not the other way round.

    Both halves have identical shapes, so a swap runs perfectly happily and
    produces plausible compressed rows.
    """
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        seq_len = 256
        _, metadata = _prefill(cache_manager, sparse_config, seq_len)
        comp, wkv, wgate = _compressor(layer_idx, ratio, seed=17)
        torch.manual_seed(4)
        x = (torch.randn(seq_len, HIDDEN, device="cuda") * 0.5).bfloat16()
        ref = _golden_compressed(comp, wkv, wgate, x, ratio, stage="pool")

        with torch.no_grad():
            comp.wkv_gate.weight.copy_(torch.cat([wgate, wkv], dim=0))
        swapped, _ = comp(x, metadata)
        assert tg.compare(swapped, ref)["cosine"] < 0.5, (
            "swapping the kv/gate halves produced the same pooling; the layout "
            "contract is not actually being exercised"
        )
    finally:
        cache_manager.shutdown()


@pytest.mark.parametrize("ratio", [4, 128])
def test_sm90_compressed_cache_rows_match_the_source_postprocess(ratio):
    """Norm + RoPE + FP8 simulation + paged scatter, read back out of the pool.

    Compared against the *complete* source golden --- including the
    ``act_quant(kv[..., :-rope_dim], 64, ..., inplace=True)`` round trip that
    ends `Compressor.forward` --- at the registered `compressor` tolerance,
    with nothing substituted or derived.

    On SM90 the main compressor does not take the fused postprocess kernel at
    all: `_source_postprocess` reproduces the checkpoint's own BF16 rounding
    chain (norm, then RoPE, then Hadamard, rounding at each step rather than
    once at the store) and `sm90_quant.write_source_compressed_rows` scatters
    that already-postprocessed row, applying the blockwise-64 FP8 round trip at
    the destination. TensorRT-LLM's `KVCacheDtype.NONE` preset would otherwise
    store the unrounded row, and without that step the cached row is
    systematically *closer* to an FP32 recomputation than the checkpoint
    expects, which is a semantic gap rather than a tolerance question.
    """
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        seq_len = 512
        _, metadata = _prefill(cache_manager, sparse_config, seq_len)
        comp, wkv, wgate = _compressor(layer_idx, ratio, seed=11 + ratio)
        torch.manual_seed(3)
        x = (torch.randn(seq_len, HIDDEN, device="cuda") * 0.5).bfloat16()
        comp(x, metadata)

        rows = seq_len // ratio
        cached = _cached_rows(cache_manager, layer_idx, rows)
        ref = _golden_compressed(comp, wkv, wgate, x, ratio, stage="cache")
        _assert_within(tg.compare(cached, ref), "compressor", f"ratio {ratio} cache rows")

        # The compressed position used for RoPE is the window's first token.
        positions = metadata.compressed_position_ids_cuda[ratio][:rows]
        expected = torch.arange(0, rows * ratio, ratio, dtype=positions.dtype, device="cuda")
        assert torch.equal(positions, expected), (
            f"ratio {ratio} compressed RoPE positions {positions[:4].tolist()} are not the "
            f"window-first positions the source uses ({expected[:4].tolist()})"
        )
    finally:
        cache_manager.shutdown()


@pytest.mark.parametrize("ratio", [4, 128])
def test_sm90_compressed_cache_rows_carry_the_sources_fp8_simulation(ratio):
    """The simulation is actually present, and only where the source puts it.

    A test that merely passes a tolerance cannot tell "the FP8 round trip ran"
    from "the tolerance is loose enough to cover its absence". So this asserts
    the positive fact directly: every cached non-RoPE value is a fixed point of
    a blockwise-64 FP8 round trip, the RoPE tail is *not* (the source excludes
    it via ``kv[..., :-rd]``), and the un-simulated golden is measurably
    further away than the simulated one.
    """
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        seq_len = 512
        _, metadata = _prefill(cache_manager, sparse_config, seq_len)
        comp, wkv, wgate = _compressor(layer_idx, ratio, seed=11 + ratio)
        torch.manual_seed(3)
        x = (torch.randn(seq_len, HIDDEN, device="cuda") * 0.5).bfloat16()
        comp(x, metadata)
        cached = _cached_rows(cache_manager, layer_idx, seq_len // ratio)

        # Idempotence: quantising an already-quantised row changes nothing.
        assert torch.equal(
            tg.fp8_quant_dequant(cached[..., :NOPE_DIM], 64), cached[..., :NOPE_DIM]
        ), (
            f"ratio {ratio}: cached non-RoPE values are not on the FP8 blockwise-64 grid, "
            "so the source's quantisation simulation did not run"
        )
        rope_tail = cached[..., NOPE_DIM:]
        assert not torch.equal(tg.fp8_quant_dequant(rope_tail, 64), rope_tail), (
            f"ratio {ratio}: the RoPE tail is also on the FP8 grid; the source applies "
            "act_quant to kv[..., :-rope_dim] only"
        )

        # And the step is not free. A row that skipped it -- what the previous
        # SM90 build stored -- sits measurably further from the source golden
        # than the registered tolerance allows, so this is a real semantic
        # difference rather than rounding either way.
        ref = _golden_compressed(comp, wkv, wgate, x, ratio, stage="cache")
        limit = TOL["compressor"]["rel_max_abs_max"]
        assert tg.compare(cached, ref)["rel_max_abs"] <= limit
        skipped = tg.compare(_unsimulated_reference(comp, wkv, wgate, x, ratio), ref)
        assert skipped["rel_max_abs"] > limit, (
            f"ratio {ratio}: a compressed row without the FP8 simulation measures "
            f"rel_max_abs {skipped['rel_max_abs']:.4f} against the source, inside the "
            f"registered {limit}; this test would then pass with the step missing"
        )
    finally:
        cache_manager.shutdown()


@pytest.mark.parametrize("ratio", [4, 128])
def test_sm90_compressor_state_completes_a_group_started_in_prefill(ratio):
    """Retained kv_state / score_state across an incomplete compression group.

    The prefill deliberately stops part-way through a group, so the row that
    completes it during decode can only be right if the leftover pooling state
    survived the phase change. Prefill passing while decode diverges is the
    exact failure this covers.
    """
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        remainder = ratio // 2
        prompt_len = 4 * ratio - remainder
        request, metadata = _prefill(cache_manager, sparse_config, prompt_len)
        comp, wkv, wgate = _compressor(layer_idx, ratio, seed=23 + ratio)
        torch.manual_seed(9)
        total = 4 * ratio
        x_all = (torch.randn(total, HIDDEN, device="cuda") * 0.5).bfloat16()

        pooled_ctx, _ = comp(x_all[:prompt_len], metadata)
        assert pooled_ctx.shape[0] == prompt_len // ratio
        _advance_to_generation(cache_manager, request, prompt_len)

        # Feed the remaining tokens one at a time; the group completes on the last.
        produced = None
        for step in range(remainder):
            cached_len = prompt_len + step
            if step > 0:
                request.add_new_token(cached_len, 0)
                assert cache_manager.try_allocate_generation(request)
            gen_md = _metadata(cache_manager, sparse_config, 1, cached_len, num_contexts=0)
            out, _ = comp(x_all[cached_len : cached_len + 1], gen_md)
            if out.shape[0] > 0:
                produced = out
        assert produced is not None and produced.shape[0] == 1, (
            f"ratio {ratio}: decode never completed the group left open by prefill"
        )

        # The completing row must equal the whole-sequence golden's last row.
        ref_all = _golden_compressed(comp, wkv, wgate, x_all, ratio, stage="pool")
        _assert_within(
            tg.compare(produced[0], ref_all[total // ratio - 1]),
            "compressor_state",
            f"ratio {ratio} group completed during decode",
        )
    finally:
        cache_manager.shutdown()


# ---------------------------------------------------------------------------
# Indexer and selection.
# ---------------------------------------------------------------------------


def _assert_indexer_cache_matches(cache_manager, layer_idx, values, scales):
    """The paged indexer K rows must hold what the dense return says they do.

    `fp8_paged_mqa_logits` reads decode scores straight out of these pages, so
    a dense/paged divergence would pass every context-phase test and only show
    up as wrong generation.
    """
    buffers = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS)
    block = cache_manager.compressed_block_sizes[layer_idx]
    pages = cache_manager.get_cache_indices(0, layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS)
    raw = buffers.view(torch.uint8)
    for i in range(values.shape[0]):
        page, slot = pages[i // block], i % block
        row = raw[page].reshape(-1)
        data = row[slot * INDEX_HEAD_DIM : (slot + 1) * INDEX_HEAD_DIM].view(torch.float8_e4m3fn)
        base = block * INDEX_HEAD_DIM + slot * 4
        scale = row[base : base + 4].view(torch.float32)
        assert torch.equal((data.float() * scale).bfloat16(), values[i]), (
            f"paged indexer K row {i} (page {page}, slot {slot}) does not match the "
            "dense tensor the context phase consumes"
        )
        assert torch.equal(scale.reshape(()), scales[i].reshape(())), (
            f"paged indexer K scale at row {i} differs from the dense scale"
        )


def _indexer(layer_idx, ratio, sparse_config, max_seq_len=MAX_SEQ_LEN):
    """A real `DeepseekV4Indexer`, the module the SM90 path actually runs."""
    return DeepseekV4Indexer(
        None,
        _pos_embd(ratio, max_seq_len),
        MLAParams(hidden_size=HIDDEN, qk_rope_head_dim=ROPE_DIM, qk_nope_head_dim=NOPE_DIM),
        False,
        sparse_config.to_sparse_params(),
        torch.bfloat16,
        compress_ratio=ratio,
        layer_idx=layer_idx,
    ).cuda()


def _seed_indexer_compressor(indexer, ratio, seed):
    """Reproducible compressor weights, plus the raw wkv/wgate the golden takes."""
    torch.manual_seed(seed)
    state_dim = indexer.compressor.state_dim
    wkv = (torch.randn(state_dim, HIDDEN, device="cuda") * (HIDDEN**-0.5)).bfloat16()
    wgate = (torch.randn(state_dim, HIDDEN, device="cuda") * (HIDDEN**-0.5)).bfloat16()
    with torch.no_grad():
        indexer.compressor.wkv_gate.weight.copy_(torch.cat([wkv, wgate], dim=0))
        indexer.compressor.norm.weight.copy_(
            (torch.randn(INDEX_HEAD_DIM, device="cuda") * 0.05 + 1.0).bfloat16()
        )
        indexer.compressor.ape.copy_(
            torch.randn(ratio, state_dim, device="cuda", dtype=torch.float32) * 0.1
        )
    return wkv, wgate


#: A prefill long enough that the checkpoint's top-k of 512 is a *decision*:
#: 2304 tokens give 576 ratio-4 compressed slots, so 253 query rows have more
#: valid slots than the selection width and the ranking has to choose. A short
#: prompt would select every valid slot on both sides and prove nothing.
DECIDING_SEQ_LEN = 2304
DECIDING_MAX_SEQ_LEN = 3072


def test_sm90_indexer_q_and_k_carry_the_sources_fp4_simulation():
    """The values the index GEMM sees must be the source's FP4-rounded ones.

    `Indexer.forward` in the checkpoint's `inference/model.py` does
    ``fp4_act_quant(q, fp4_block_size, True)`` on Q, and its Compressor ends
    with the same call on K: the checkpoint is QAT'd for FP4 index scores, so
    the rounding is semantics, not storage. Both sides are dequantized and
    required to equal the independent golden's `fp4_quant_dequant` *bitwise*.

    Bit equality is the right rule rather than a tolerance. The FP8 container
    carries E2M1 levels exactly when its scale is a power of two, which is the
    property the whole SM90 indexer design rests on; if that ever stopped
    holding this fails here instead of drifting into a top-k mismatch far
    downstream, where it would look like a selection bug.
    """
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        ratio, layer_idx = 4, RATIO_TO_LAYER[4]
        indexer = _indexer(layer_idx, ratio, sparse_config)
        assert indexer.simulate_source_q_fp4, "the SM90 FP4 simulation is not enabled"
        assert indexer.compressor.rotate_activation, (
            "the source always rotates the indexer's compressed K; skipping it would "
            "change which E2M1 levels the values land on"
        )

        torch.manual_seed(29)
        q = (torch.randn(96, INDEX_HEADS, INDEX_HEAD_DIM, device="cuda") * 0.4).bfloat16()
        q_sim, q_scale = indexer._quantize_q(q.clone())
        # `fp4_act_quant(q, ..., inplace=True)` quantizes *and dequantizes*: the
        # source leaves Q in BF16 and never puts it in a narrow container, so
        # neither does SM90. A scale here would mean it did.
        assert q_scale is None and q_sim.dtype == torch.bfloat16, (
            f"SM90 must hand the score kernel the source's BF16 Q; got {q_sim.dtype} "
            f"with scale {None if q_scale is None else tuple(q_scale.shape)}"
        )
        got_q = q_sim.reshape(-1, INDEX_HEAD_DIM)
        # The reference rotation is the golden's own butterfly, not the shared
        # `rotate_activation` helper --- that one silently returns its input
        # when `fast-hadamard-transform` is absent, which is exactly the hole
        # the SM90 Torch rotation fills.
        want_q = tg.fp4_quant_dequant(_hadamard(q).view(-1, INDEX_HEAD_DIM), 32)
        assert torch.equal(got_q, want_q), (
            "the indexer's simulated Q does not equal the source's FP4 levels"
        )

        seq_len = 512
        _, metadata = _prefill(cache_manager, sparse_config, seq_len)
        wkv, wgate = _seed_indexer_compressor(indexer, ratio, seed=31)
        torch.manual_seed(37)
        x = (torch.randn(seq_len, HIDDEN, device="cuda") * 0.5).bfloat16()
        k_fp8, k_scale = indexer.compressor(x, metadata)
        got_k = (k_fp8.float() * k_scale).bfloat16()
        want_k = _golden_compressed(
            indexer.compressor,
            wkv,
            wgate,
            x,
            ratio,
            stage="cache",
            head_dim=INDEX_HEAD_DIM,
            hadamard=_hadamard,
        )
        # K must sit on the E2M1 grid exactly -- that is the positive proof the
        # simulation ran, and it cannot be satisfied by a loose tolerance.
        assert torch.equal(tg.fp4_quant_dequant(got_k, 32), got_k), (
            "the indexer's compressed K rows are not on the E2M1 grid, so the source's "
            "FP4 simulation did not run"
        )
        # Against the golden's values, near-exact rather than exact. The two
        # sides use independent Hadamard implementations (a matrix multiply
        # here, a recursive butterfly in the golden) that agree to ~1e-7, and a
        # quantiser turns any disagreement at a decision boundary into a whole
        # level. Measured: 1 of 16384 values at this width, 0 of 73728 at the
        # 2304-token width the selection test uses. `rel_max_abs` is therefore
        # not a meaningful gate on a quantised tensor -- one adjacent-level
        # crossing already saturates it -- so the count is bounded directly and
        # the binding gate is exact top-k in the next test.
        differing = int((got_k != want_k).sum())
        assert differing <= max(4, got_k.numel() // 4096), (
            f"{differing} of {got_k.numel()} compressed K values differ from the source's "
            "FP4 levels; that is more than boundary rounding between two independent "
            "Hadamard implementations"
        )
        assert tg.compare(got_k, want_k)["cosine"] >= TOL["indexer_scores"]["cosine_min"]
        # The paged rows the decode kernel reads must hold the same values.
        _assert_indexer_cache_matches(cache_manager, layer_idx, got_k, k_scale)
    finally:
        cache_manager.shutdown()


def _selection_mismatches(logits, ref_topk, rows):
    """Rows whose selected slot set differs from the source's.

    Both sides are ranked from BF16 scores by the same `torch.topk` call, so an
    exact tie between the last selected and the first rejected slot --- which
    BF16's 8-bit significand makes common at this width --- is broken the same
    way on both. A difference here is a difference in the *scores*.
    """
    got = logits.bfloat16().topk(INDEX_TOPK, dim=-1).indices
    return got, [
        i for i in rows if set(got[i].tolist()) != {v for v in ref_topk[i].tolist() if v >= 0}
    ]


def _assert_selection_matches(logits, ref_topk, rows, label):
    got, mismatches = _selection_mismatches(logits, ref_topk, rows)
    if mismatches:
        i = mismatches[0]
        diff = sorted(set(got[i].tolist()) ^ {v for v in ref_topk[i].tolist() if v >= 0})
        raise AssertionError(
            f"{label}: selected a different slot set than the source on {len(mismatches)} of "
            f"{len(rows)} deciding rows; first at query {i}, symmetric difference {diff}"
        )
    return got


def _context_prefill_inputs(indexer, cache_manager, sparse_config, seq_len, ratio):
    """Run a real prefill through the real compressor and Q simulation.

    Returns everything both sides of the comparison need: the module outputs
    (`k_fp8`/`k_scale`, the BF16 Q the source scores, the BF16 per-head
    weights) and the raw inputs the independent golden takes.
    """
    _, metadata = _prefill(cache_manager, sparse_config, seq_len)
    wkv, wgate = _seed_indexer_compressor(indexer, ratio, seed=31)

    torch.manual_seed(29)
    x = (torch.randn(seq_len, HIDDEN, device="cuda") * 0.5).bfloat16()
    q = (torch.randn(seq_len, INDEX_HEADS, INDEX_HEAD_DIM, device="cuda") * 0.4).bfloat16()
    # BF16, as `weights_proj` is in the checkpoint. Handing this path an FP32
    # weight would quietly widen one of the three roundings under test.
    weights = (torch.randn(seq_len, INDEX_HEADS, device="cuda") * 0.2).bfloat16()

    k_fp8, k_scale = indexer.compressor(x, metadata)
    assert k_fp8.shape[0] == seq_len // ratio
    q_sim, q_scale = indexer._quantize_q(q.clone())
    scaled = indexer._apply_weight_scale(weights, q_scale)
    assert scaled.dtype == torch.bfloat16, (
        f"the source's per-head index weight is bf16; got {scaled.dtype}"
    )
    golden_k = _golden_compressed(
        indexer.compressor,
        wkv,
        wgate,
        x,
        ratio,
        stage="cache",
        head_dim=INDEX_HEAD_DIM,
        hadamard=_hadamard,
        max_seq_len=DECIDING_MAX_SEQ_LEN,
    )
    return metadata, q, q_sim, q_scale, weights, scaled, k_fp8, k_scale, golden_k


def test_sm90_indexer_selection_matches_the_source_at_the_checkpoint_topk():
    """The SM90 index-score path vs the source's own reduction, exact top-512.

    End to end through the real modules: a 2304-token prefill drives the real
    `DeepseekV4Indexer`'s compressor to produce K and its `_quantize_q` to
    produce Q, and the scores come from `_call_mqa_logits` --- the method the
    context phase of `sparse_attn_indexer` actually calls. The reference is the
    independent golden fed the *source's* FP4-simulated inputs, so this is not
    the kernel being compared against its own dequantized bytes.

    The selected set is held to exact equality, as the manifest's
    `indexer_topk` rule demands; there is no float tolerance to fall back on.
    The scores themselves are additionally held to near-*bit* equality with the
    source's BF16 chain, which a float tolerance cannot express: the point of
    this path is that it lands on the same BF16 grid the checkpoint does, and
    `indexer_scores`' 0.05 `rel_max_abs` is satisfied by an FP32 reduction that
    selects the wrong slots (see the next test).

    On these inputs the measurement is 0 of 662,976 scores differing --- bit
    equality with the source's own reduction. The bound is nevertheless not
    written as zero, because only half of the chain is guaranteed. The dot
    product is: every operand is an E2M1 level times a power of two, so each
    product carries at most four significant bits and the FP32 accumulation of
    a 128-wide row is order-independent (measured: 0 of 9,437,184 per-head dots
    differ between an FP32 recomputation and the BF16 `einsum`). The 64-head
    sum is not: its terms span enough exponents that FP32 addition is
    order-sensitive at the last bit, and Triton's tree reduction is not
    Torch's. A stray one-ulp score is therefore allowed; a systematic
    divergence is not, and the selection is exact either way.
    """
    ratio, seq_len = 4, DECIDING_SEQ_LEN
    num_slots = seq_len // ratio
    assert num_slots > INDEX_TOPK, "the prompt must be long enough for top-k to decide"

    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config, DECIDING_MAX_SEQ_LEN)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        indexer = _indexer(layer_idx, ratio, sparse_config, DECIDING_MAX_SEQ_LEN)
        assert indexer.source_faithful_scores, "the SM90 score reduction is not enabled"
        metadata, q, q_sim, q_scale, weights, scaled, k_fp8, k_scale, golden_k = (
            _context_prefill_inputs(indexer, cache_manager, sparse_config, seq_len, ratio)
        )

        limit = (torch.arange(1, seq_len + 1, device="cuda") // ratio).to(torch.int32)
        starts = torch.zeros(seq_len, dtype=torch.int32, device="cuda")
        logits = indexer._call_mqa_logits(q_sim, k_fp8, k_scale, scaled, starts, limit, q_scale)

        # Chunked prefill feeds the same call from `indexer_k_cache_gather_op`
        # rather than the compressor's dense return, so a layout error in the
        # paged gather would otherwise only surface under chunking. Same rows,
        # therefore bit-identical scores.
        gathered, gathered_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
            cache_manager.get_indexer_k_cache_buffers(layer_idx),
            metadata.slot_mapping_fp8_fullkv,
            metadata.slot_mapping_scale_fullkv,
            0,
            num_slots,
            INDEX_HEAD_DIM,
        )
        assert torch.equal(
            indexer._call_mqa_logits(
                q_sim, gathered, gathered_scale, scaled, starts, limit, q_scale
            ),
            logits,
        ), "the chunked-prefill gather does not reproduce the compressor's own K rows"

        ref_score, ref_topk = tg.indexer_scores_and_topk(
            tg.fp4_quant_dequant(_hadamard(q), 32).unsqueeze(0),
            golden_k.unsqueeze(0),
            (weights * indexer.weight_scale_factor).unsqueeze(0),
            seqlen=seq_len,
            ratio=ratio,
            topk=INDEX_TOPK,
            offset=0,
        )
        ref_score, ref_topk = ref_score.squeeze(0), ref_topk.squeeze(0)

        # Compare only inside each row's valid window; the kernel fills the
        # rest with -inf, which no metric can average over.
        valid = torch.arange(num_slots, device="cuda").unsqueeze(0) < limit.unsqueeze(1)
        got_valid, ref_valid = logits[valid].bfloat16(), ref_score[valid]
        # `index_score` is a BF16 tensor in the source, and the runtime top-k
        # ranks whatever this kernel stores. Carrying more precision than the
        # checkpoint's own reduction would rank values the source never
        # computes, so the grid itself is asserted rather than inferred.
        assert torch.equal(got_valid.float(), logits[valid]), (
            "the SM90 index scores are not on the BF16 grid the source reduces onto"
        )
        _assert_within(
            tg.compare(got_valid.float(), ref_valid.float()), "indexer_scores", "indexer scores"
        )
        differing = int((got_valid != ref_valid).sum())
        assert differing <= max(4, got_valid.numel() // 65536), (
            f"{differing} of {got_valid.numel()} index scores differ from the source's BF16 "
            "chain; that is more than last-bit reduction ordering in the 64-head sum"
        )

        rows = [i for i in range(seq_len) if int(limit[i]) > INDEX_TOPK]
        assert len(rows) > 32, "no row has more valid slots than the selection width"
        _assert_selection_matches(logits, ref_topk, rows, "SM90 context indexer")
    finally:
        cache_manager.shutdown()


def test_sm90_indexer_reduction_width_changes_the_selection():
    """Why the Triton score path exists, stated as a measurement.

    The FP8 container is lossless for these values, so DeepGEMM's
    `fp8_mqa_logits` sees exactly the same numbers as the source. It still
    picks different slots, because it reduces in FP32 while the checkpoint
    reduces in BF16. This test pins both halves of that claim:

      * the FP32 scores comfortably pass the registered `indexer_scores` float
        tolerance, so the divergence is not a numerical-accuracy problem and
        cannot be closed by tightening one; and
      * they nonetheless select a different slot set on a substantial number of
        deciding rows, which the manifest's `exact` rule makes a failure.

    Measured on these inputs: `rel_max_abs` 0.0346 against a registered limit
    of 0.05, cosine 0.999994 --- and 31 of 253 deciding rows choose differently.
    So if someone deletes the Triton path and restores DeepGEMM, this fails
    rather than the change looking free.
    """
    from tensorrt_llm.deep_gemm import fp8_mqa_logits

    ratio, seq_len = 4, DECIDING_SEQ_LEN
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config, DECIDING_MAX_SEQ_LEN)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        indexer = _indexer(layer_idx, ratio, sparse_config, DECIDING_MAX_SEQ_LEN)
        _, q, q_sim, _, weights, _, k_fp8, k_scale, golden_k = _context_prefill_inputs(
            indexer, cache_manager, sparse_config, seq_len, ratio
        )

        # Put the source's BF16 Q into the FP8 container DeepGEMM needs, and
        # prove that step loses nothing before blaming the reduction.
        amax = (
            q_sim.abs()
            .amax(dim=-1, keepdim=True)
            .float()
            .clamp(min=torch.finfo(torch.float32).tiny)
        )
        q_scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        q_fp8 = (q_sim.float() / q_scale).to(torch.float8_e4m3fn)
        assert torch.equal((q_fp8.float() * q_scale).bfloat16(), q_sim), (
            "the FP8 container is not lossless for these values, so this test would be "
            "measuring quantisation rather than reduction width"
        )

        limit = (torch.arange(1, seq_len + 1, device="cuda") // ratio).to(torch.int32)
        fp32_logits = fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scale.reshape(-1)),
            weights.float() * q_scale.squeeze(-1) * indexer.weight_scale_factor,
            torch.zeros(seq_len, dtype=torch.int32, device="cuda"),
            limit,
        )

        ref_score, ref_topk = tg.indexer_scores_and_topk(
            tg.fp4_quant_dequant(_hadamard(q), 32).unsqueeze(0),
            golden_k.unsqueeze(0),
            (weights * indexer.weight_scale_factor).unsqueeze(0),
            seqlen=seq_len,
            ratio=ratio,
            topk=INDEX_TOPK,
            offset=0,
        )
        ref_score, ref_topk = ref_score.squeeze(0), ref_topk.squeeze(0)

        num_slots = seq_len // ratio
        valid = torch.arange(num_slots, device="cuda").unsqueeze(0) < limit.unsqueeze(1)
        _assert_within(
            tg.compare(fp32_logits[valid].float(), ref_score[valid].float()),
            "indexer_scores",
            "fp32-reduced indexer scores",
        )

        rows = [i for i in range(seq_len) if int(limit[i]) > INDEX_TOPK]
        _, mismatches = _selection_mismatches(fp32_logits, ref_topk, rows)
        assert len(mismatches) > 4, (
            f"the FP32 reduction changed only {len(mismatches)} of {len(rows)} deciding rows; "
            "if it now agrees with the source, the Triton reduction is no longer justified "
            "and this decision has to be re-made rather than silently inherited"
        )
    finally:
        cache_manager.shutdown()


def _cached_indexer_rows(cache_manager, layer_idx, count):
    """Dequantize `count` paged indexer K rows back to the source's BF16 values."""
    buffers = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS)
    block = cache_manager.compressed_block_sizes[layer_idx]
    pages = cache_manager.get_cache_indices(0, layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS)
    raw = buffers.view(torch.uint8)
    rows = []
    for i in range(count):
        page, slot = pages[i // block], i % block
        row = raw[page].reshape(-1)
        data = row[slot * INDEX_HEAD_DIM : (slot + 1) * INDEX_HEAD_DIM].view(torch.float8_e4m3fn)
        base = block * INDEX_HEAD_DIM + slot * 4
        scale = row[base : base + 4].view(torch.float32)
        rows.append((data.float() * scale).bfloat16())
    return torch.stack(rows)


def test_sm90_indexer_decode_selection_matches_the_source_over_the_paged_cache():
    """Cached decode: the same reduction, reading K through the block table.

    Context scoring gets K as a dense tensor the compressor just returned;
    decode gets it from the paged pool, one page indirection per slot, with the
    scale living in a separate region of the same page. A kernel can be right
    about the arithmetic and wrong about the addressing, and only generation
    would show it --- so this drives a real prefill, advances the request to
    the generation phase through the cache manager, and scores the *cached*
    rows.

    The prompt is long enough that the decode step still has to choose: 2304
    prompt tokens leave 576 compressed slots against a selection width of 512.
    Both the paged score kernel and the full `sparse_attn_indexer` decode
    dispatch (including the C++ top-k) are compared against the source.
    """
    ratio, seq_len = 4, DECIDING_SEQ_LEN
    num_slots = seq_len // ratio
    assert num_slots > INDEX_TOPK, "the decode step must still have a selection to make"

    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config, DECIDING_MAX_SEQ_LEN)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        indexer = _indexer(layer_idx, ratio, sparse_config, DECIDING_MAX_SEQ_LEN)
        request, metadata = _prefill(cache_manager, sparse_config, seq_len)
        _seed_indexer_compressor(indexer, ratio, seed=31)
        torch.manual_seed(29)
        x = (torch.randn(seq_len, HIDDEN, device="cuda") * 0.5).bfloat16()
        assert indexer.compressor(x, metadata)[0].shape[0] == num_slots

        _advance_to_generation(cache_manager, request, seq_len)
        metadata = _metadata(cache_manager, sparse_config, 1, seq_len, num_contexts=0)

        torch.manual_seed(5)
        step_x = (torch.randn(1, HIDDEN, device="cuda") * 0.5).bfloat16()
        k_fp8, k_scale = indexer.compressor(step_x, metadata)
        q = (torch.randn(1, INDEX_HEADS, INDEX_HEAD_DIM, device="cuda") * 0.4).bfloat16()
        weights = (torch.randn(1, INDEX_HEADS, device="cuda") * 0.2).bfloat16()
        q_sim, q_scale = indexer._quantize_q(q.clone())
        scaled = indexer._apply_weight_scale(weights, q_scale)

        # The source scores every completed slot on a decode step, whether or
        # not this token closed a new compression group.
        kv_len = int(metadata.kv_lens_cuda_2d[0, 0])
        assert kv_len == num_slots, (
            f"the decode step sees {kv_len} indexer slots, expected {num_slots}"
        )
        ref_score, ref_topk = tg.indexer_scores_and_topk(
            tg.fp4_quant_dequant(_hadamard(q), 32).unsqueeze(0),
            _cached_indexer_rows(cache_manager, layer_idx, kv_len).unsqueeze(0),
            (weights * indexer.weight_scale_factor).unsqueeze(0),
            seqlen=1,
            ratio=ratio,
            topk=INDEX_TOPK,
            offset=0,
            kv_len=kv_len,
        )
        ref_score, ref_topk = ref_score.squeeze(0), ref_topk.squeeze(0)

        block_table = metadata.indexer_k_cache_block_offsets[:1]
        logits = indexer._call_paged_mqa_logits(
            q_sim.view(1, 1, INDEX_HEADS, INDEX_HEAD_DIM),
            cache_manager.get_indexer_k_cache_buffers(layer_idx),
            scaled,
            metadata.kv_lens_cuda_2d[:1, :1].contiguous(),
            block_table,
            None,
            metadata.get_indexer_max_seq_len(),
            q_scale,
        )
        got_valid = logits[:, :kv_len].bfloat16()
        assert torch.equal(got_valid.float(), logits[:, :kv_len]), (
            "the paged index scores are not on the BF16 grid the source reduces onto"
        )
        _assert_within(
            tg.compare(got_valid.float(), ref_score.float()),
            "indexer_scores",
            "paged indexer scores",
        )
        differing = int((got_valid != ref_score).sum())
        assert differing <= max(4, got_valid.numel() // 65536), (
            f"{differing} of {got_valid.numel()} paged index scores differ from the source"
        )
        assert torch.isneginf(logits[0, kv_len:]).all(), (
            "slots past the cached length must be -inf, not stale page content"
        )
        _assert_selection_matches(logits[:, :kv_len], ref_topk, [0], "SM90 paged indexer")

        # Page-table addressing, proved independently of the arithmetic: this
        # prompt spans 28 pages of 32 slots, so a kernel that mixed up the page
        # indirection or the data/scale split would still produce plausible
        # scores. Zeroing one cached row must move exactly one score.
        target = kv_len - 76
        block = cache_manager.compressed_block_sizes[layer_idx]
        pages = cache_manager.get_cache_indices(
            0, layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS
        )
        raw = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS).view(
            torch.uint8
        )
        assert len(pages) > 2 and block < kv_len, "the prompt must span several pages"
        row = raw[pages[target // block]].reshape(-1)
        lo, hi = (target % block) * INDEX_HEAD_DIM, (target % block + 1) * INDEX_HEAD_DIM
        row[lo:hi] = 0
        perturbed = indexer._call_paged_mqa_logits(
            q_sim.view(1, 1, INDEX_HEADS, INDEX_HEAD_DIM),
            cache_manager.get_indexer_k_cache_buffers(layer_idx),
            scaled,
            metadata.kv_lens_cuda_2d[:1, :1].contiguous(),
            block_table,
            None,
            metadata.get_indexer_max_seq_len(),
            q_scale,
        )
        moved = (logits[0, :kv_len] != perturbed[0, :kv_len]).nonzero().flatten().tolist()
        assert moved == [target], (
            f"zeroing cached slot {target} moved scores {moved}; the paged read does not "
            "address one slot per score"
        )

        # And through the real decode dispatch, which reaches the same kernel
        # via the metadata's block table and finishes with the C++ top-k.
        selected = indexer.sparse_attn_indexer(
            metadata, step_x, q_sim, k_fp8, k_scale, scaled, q_scale=q_scale
        )
        assert set(selected[0].tolist()) == {v for v in ref_topk[0].tolist() if v >= 0}, (
            "the decode dispatch selected a different slot set than the source"
        )
    finally:
        cache_manager.shutdown()


def test_sm90_indexer_fp4_kernels_are_blackwell_only_and_the_config_falls_back():
    """Why SM90 runs the FP8 indexer: the FP4 kernel refuses to launch.

    The plan treats the FP8 blockwise indexer as a *starting* fallback that has
    to be justified rather than assumed. The justification is this: DeepGEMM's
    FP4 MQA-logits kernel hard-asserts the architecture, and the LLM API config
    already downgrades the DeepSeek-V4 `fp4` default to `fp8` on pre-Blackwell
    rather than failing at kernel-launch time.
    """
    assert get_sm_version() < 100, "this Hopper expectation only holds pre-Blackwell"

    from tensorrt_llm.deep_gemm import fp8_fp4_mqa_logits

    seq_len, num_slots = 32, 64
    with pytest.raises(RuntimeError, match="arch_major"):
        fp8_fp4_mqa_logits(
            (
                torch.zeros(
                    seq_len, INDEX_HEADS, INDEX_HEAD_DIM // 2, dtype=torch.int8, device="cuda"
                ),
                torch.ones(seq_len, INDEX_HEADS, dtype=torch.int32, device="cuda"),
            ),
            (
                torch.zeros(num_slots, INDEX_HEAD_DIM // 2, dtype=torch.int8, device="cuda"),
                torch.ones(num_slots, dtype=torch.int32, device="cuda"),
            ),
            torch.randn(seq_len, INDEX_HEADS, device="cuda", dtype=torch.float32),
            torch.zeros(seq_len, dtype=torch.int32, device="cuda"),
            torch.arange(1, seq_len + 1, dtype=torch.int32, device="cuda"),
        )

    # The default config resolves itself to the supported path on this device.
    assert _sparse_config().indexer_k_dtype == "fp8"
    with pytest.raises(ValueError, match="requires SM>=100"):
        DeepSeekV4SparseAttentionConfig(
            index_head_dim=INDEX_HEAD_DIM,
            compress_ratios=list(COMPRESS_RATIOS),
            indexer_k_dtype="fp4",
        )


def test_sm90_indexer_decode_logits_kernel_dispatches_for_the_supported_widths():
    """The decode-phase indexer kernel, and where its SM90 support stops.

    `fp8_paged_mqa_logits` serves one or two query positions per step on
    Hopper. Target-only greedy decoding needs one; the limit is recorded here
    so that enabling speculative decoding later fails loudly at this test
    rather than inside a kernel.
    """
    from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits, get_paged_mqa_logits_metadata

    block, batch, kv_len = 64, 2, 64
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    def run(next_n):
        q = torch.randn(batch, next_n, INDEX_HEADS, INDEX_HEAD_DIM, device="cuda").to(
            torch.float8_e4m3fn
        )
        k_cache = torch.zeros(8, block, 1, INDEX_HEAD_DIM + 4, dtype=torch.uint8, device="cuda")
        weights = torch.randn(batch * next_n, INDEX_HEADS, device="cuda", dtype=torch.float32)
        context_lens = torch.full((batch, next_n), kv_len, dtype=torch.int32, device="cuda")
        block_table = torch.zeros(batch, 4, dtype=torch.int32, device="cuda")
        schedule = get_paged_mqa_logits_metadata(context_lens, block, num_sms)
        return fp8_paged_mqa_logits(
            q, k_cache, weights, context_lens, block_table, schedule, kv_len
        )

    for next_n in (1, 2):
        out = run(next_n)
        assert out.shape == (batch * next_n, kv_len)
        assert torch.isfinite(out).all()
    with pytest.raises(RuntimeError):
        run(4)


@pytest.mark.parametrize("kv_len", [1, 127, 128, 129, 255, 256, 257, 512])
def test_sm90_ratio128_selection_matches_the_source_rule(kv_len):
    """Ratio-128 layers take every compressed slot whose window has closed.

    There is no learned selection here, so the rule is exact: slot ``j`` is
    valid iff ``j < kv_len // 128``. The metadata builds this table on device;
    the source computes it as a mask, and the two must agree exactly at and
    around every compression boundary.
    """
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        request, _ = _prefill(cache_manager, sparse_config, max(kv_len - 1, 1))
        _advance_to_generation(cache_manager, request, max(kv_len - 1, 1))
        metadata = _metadata(cache_manager, sparse_config, 1, kv_len - 1, num_contexts=0)

        width = metadata.max_compressed_indices[128]
        got = metadata.compressed_local_indices_cuda[:1, :width].flatten()
        num_valid = kv_len // 128
        expected = torch.full((width,), -1, dtype=torch.int32, device="cuda")
        expected[:num_valid] = torch.arange(num_valid, dtype=torch.int32, device="cuda")
        assert torch.equal(got, expected), (
            f"kv_len {kv_len}: compressed selection {got[: num_valid + 2].tolist()} does not "
            f"match the source rule {expected[: num_valid + 2].tolist()}"
        )
    finally:
        cache_manager.shutdown()


def _checkpoint_freqs_cis_on(device: str, compress_ratio: int) -> torch.Tensor:
    """`precompute_freqs_cis` from `inference/model.py`, on a chosen device.

    `tg.yarn_freqs_cis` is the same formula but is CPU-bound by construction,
    and a float32 `cos`/`sin` on the host is not bit-identical to one on the
    device. Both are used below: this one for the bit-exactness claim, the
    existing independent golden to prove *this* function is right.
    """
    dim, seqlen = ROPE_DIM, MAX_SEQ_LEN
    base = 160000.0 if compress_ratio > 1 else 10000.0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if compress_ratio > 1:
        original, factor = 65536, 16.0

        def correction_dim(rot):
            return dim * math.log(original / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction_dim(32)), 0)
        high = min(math.ceil(correction_dim(1)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen, device=device), freqs)
    return torch.polar(torch.ones_like(angles), angles)


@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_sm90_v4_rope_table_is_the_checkpoints_own(ratio):
    """DeepSeek-V4 must rotate with the table the checkpoint defines.

    `RopeEmbeddingUtils` computes the same mathematical constant by a different
    numerical recipe -- a float64 `**` rounded to float32, then NumPy's
    single-precision `cos`/`sin` on the host -- and the last-place difference is
    not academic: driving `mla_rope_inplace` with it left query elements one
    BF16 step away from the source's on real checkpoint activations, which on
    one rank exceeded the registered `sparse_attention_output` tolerance
    end to end. Bit equality against the checkpoint's own recipe is therefore
    the contract, not closeness.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.rope import (
        deepseek_v4_rotary_embedding,
    )

    pos_embd = _pos_embd(ratio)
    emb = deepseek_v4_rotary_embedding(
        pos_embd.rope, head_dim=ROPE_DIM, is_neox=pos_embd.is_neox, inverse=True
    )
    table = emb.rotary_cos_sin
    assert table.shape == (MAX_SEQ_LEN, 2, ROPE_DIM // 2), (
        f"unexpected table layout {tuple(table.shape)}"
    )

    # The local CUDA recomputation is only trustworthy if it agrees with the
    # independent CPU golden that the reference ladder already validated.
    local = _checkpoint_freqs_cis_on("cuda", ratio)
    golden = _golden_freqs(ratio)
    host_gap = max(
        float((local.real - golden.real).abs().max()), float((local.imag - golden.imag).abs().max())
    )
    # Not a bit check: the golden's `cos`/`sin` run on the host and this one on
    # the device, and the two float32 libraries differ in the last places --- by
    # more at large angles, where the argument reduction does the work. This
    # only has to show the two compute the same formula.
    assert host_gap < 1e-5, (
        f"ratio {ratio}: the local checkpoint-recipe table disagrees with the independent "
        f"golden by {host_gap:.3e}; one of the two no longer follows precompute_freqs_cis"
    )

    assert torch.equal(table[:, 0, :], local.real) and torch.equal(table[:, 1, :], local.imag), (
        f"ratio {ratio}: the installed table is not the checkpoint's own "
        f"(cos max {float((table[:, 0, :] - local.real).abs().max()):.3e}, "
        f"sin max {float((table[:, 1, :] - local.imag).abs().max()):.3e})"
    )

    # And it is genuinely a different tensor from the shared builder's, so this
    # test fails if someone reverts to a bare `RotaryEmbedding`.
    shared = (
        pos_embd.rope.create_rope_const_params(interleave=False)[1]
        .reshape(MAX_SEQ_LEN, 2, -1)
        .cuda()
    )
    assert not torch.equal(table, shared), (
        f"ratio {ratio}: the V4 table is bit-identical to RopeEmbeddingUtils'. Either the "
        "shared builder changed recipe or the V4 override was dropped; the second is a "
        "silent parity regression."
    )


@pytest.mark.parametrize("ratio", [4, 128])
def test_sm90_v4_rope_table_reaches_the_compressor_and_indexer(ratio):
    """The override has to hold for *every* V4 module that rotates.

    The Compressor rotates pooled rows and the Indexer rotates its own Q/K, so
    a table that only reached attention would rotate the compressed pool and
    the index keys against a different constant than the queries that read
    them. Ratio-1 layers are absent because they have neither module.
    """
    pos_embd = _pos_embd(ratio)
    expected = _checkpoint_freqs_cis_on("cuda", ratio)

    comp, _, _ = _compressor(RATIO_TO_LAYER[ratio], ratio, seed=11)
    tables = {"compressor": comp.rotary_emb.rotary_cos_sin}
    if ratio == 4:
        indexer = DeepseekV4Indexer(
            None,
            pos_embd,
            MLAParams(hidden_size=HIDDEN, qk_rope_head_dim=ROPE_DIM, qk_nope_head_dim=NOPE_DIM),
            False,
            _sparse_config().to_sparse_params(),
            torch.bfloat16,
            ratio,
            RATIO_TO_LAYER[4],
            None,
        ).cuda()
        tables["indexer"] = indexer.rotary_emb.rotary_cos_sin

    for name, table in tables.items():
        flat = table.reshape(-1, 2, ROPE_DIM // 2)
        rows = flat.shape[0]
        assert torch.equal(flat[:, 0, :], expected.real[:rows]) and torch.equal(
            flat[:, 1, :], expected.imag[:rows]
        ), f"ratio {ratio}: the {name}'s rotary table is not the checkpoint's own"
