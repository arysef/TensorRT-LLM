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
"""SM90 DeepSeek-V4 runtime and KV-cache contract tests.

The companion file `test_deepseek_v4_sm90.py` pins the sparse-MLA kernel's
arithmetic against the source-anchored golden using synthetic flat pools. That
proves the math and nothing about the runtime: flat synthetic indices cannot
show whether the pool views are rebased on the pointer the index table was
built against, whether a latent row lands where the table will look for it, or
whether `KVCacheManagerV2` page allocation across a prefill/decode boundary
lines up with either.

This file closes that gap. Every test here drives a real
`DeepseekV4CacheManager` on top of `KVCacheManagerV2`, real
`DeepseekV4TrtllmAttentionMetadata`, a real `DeepseekV4TrtllmAttention`, and the
real `deepseek_v4_local_to_global_indices` kernel, against pool tensors taken
from the manager's own base addresses.

The one thing not real is the `MLA` container: `forward_sparse_attn_sm90` reads
a fixed set of geometry attributes plus `mqa` and `inverse_rotary_emb` off it,
and constructing a full `MLA` would drag in a `ModelConfig`, weights and the
non-attention prologue without changing a single line this code executes. Real
model-level replay against checkpoint activations is the next Goal.
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
    DeepseekV4AttentionType,
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttention,
    DeepseekV4TrtllmAttentionMetadata,
    sm90,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.cache_manager import get_token_bytes
from tensorrt_llm._torch.attention_backend.sparse.params import SparseBackendForwardArgs
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

_GOLDENS_PATH = (
    Path(__file__).resolve().parents[5]
    / "integration"
    / "defs"
    / "accuracy"
    / "deepseek_v4_flash_h100"
    / "torch_goldens.py"
)


def _load_goldens():
    name = "deepseek_v4_flash_h100_torch_goldens"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _GOLDENS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tg = _load_goldens()

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# Checkpoint geometry at TP8: 512-wide latent row (448 non-RoPE + 64 RoPE),
# 8 of the 64 Q heads per rank, 128-token window, 128-token KV pages.
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
LOCAL_HEADS = 8
WINDOW = 128
TOKENS_PER_BLOCK = 128
INDEX_TOPK = 512
INDEX_HEAD_DIM = 128
# One layer of each schedule the checkpoint uses, in the same order
# `compress_ratios` lists them.
COMPRESS_RATIOS = [1, 4, 128]
RATIO_TO_LAYER = {1: 0, 4: 1, 128: 2}
# The checkpoint repeats its schedule down 61 layers, so every compressed pool
# is shared by many layers and only the first of them sits at the pool base.
# One layer per ratio cannot show that, which is why this second geometry
# exists.
SHARED_POOL_RATIOS = [1, 4, 128, 4, 128]
MAX_SEQ_LEN = 512


def _sparse_config(compress_ratios=COMPRESS_RATIOS):
    return DeepSeekV4SparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=INDEX_HEAD_DIM,
        window_size=WINDOW,
        compress_ratios=list(compress_ratios),
        index_topk=INDEX_TOPK,
        skip_indexer_for_short_seqs=False,
    )


def _cache_manager(sparse_config, batch_size=1, max_input_len=MAX_SEQ_LEN):
    return DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            max_tokens=MAX_SEQ_LEN * batch_size * 4,
            enable_block_reuse=False,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=len(sparse_config.compress_ratios),
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=batch_size,
        max_input_len=max_input_len,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=max_input_len * batch_size + batch_size,
        sparse_attn_config=sparse_config,
    )


def _pos_embd_params(compress_ratio: int) -> PositionalEmbeddingParams:
    """Mirror `_deepseek_v4_pos_embd_params`'s per-ratio RoPE contract.

    Ratio 0/1 layers use theta 10000 with no YaRN; compressed layers use theta
    160000 with YaRN factor 16 over an original length of 65536 and no
    amplitude scaling. Built from `RopeParams` directly rather than through a
    `PretrainedConfig`, so the test states the contract it depends on.
    """
    rope = RopeParams(
        dim=ROPE_DIM,
        max_positions=MAX_SEQ_LEN,
        original_max_positions=65536,
        max_seq_len=MAX_SEQ_LEN,
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


def _backend(layer_idx: int, sparse_config, pos_embd) -> DeepseekV4TrtllmAttention:
    mla_params = MLAParams(
        q_lora_rank=1024,
        kv_lora_rank=NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=NOPE_DIM,
        v_head_dim=HEAD_DIM,
        rope_append=False,
        predicted_tokens_per_seq=1,
        hidden_size=4096,
    )
    layer = DeepseekV4TrtllmAttention(
        layer_idx=layer_idx,
        num_heads=LOCAL_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=1,
        q_scaling=1.0,
        pos_embd_params=pos_embd,
        mla_params=mla_params,
        sparse_attention_config=sparse_config,
        skip_create_weights_in_init=True,
    )
    layer.update_quant_config(None)
    return layer


def _mla_host(backend, layer_idx, pos_embd):
    """The attribute surface `forward_sparse_attn_sm90` reads off an `MLA`."""
    return SimpleNamespace(
        mqa=backend,
        layer_idx=layer_idx,
        num_heads_tp=LOCAL_HEADS,
        qk_head_dim=HEAD_DIM,
        qk_nope_head_dim=NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        v_head_dim=HEAD_DIM,
        inverse_rotary_emb=RotaryEmbedding(
            pos_embd.rope, head_dim=ROPE_DIM, is_neox=pos_embd.is_neox, inverse=True
        ),
    )


def _make_request(request_id: int, prompt_len: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _metadata(cache_manager, sparse_config, request_ids, seq_lens, cached_lens, num_contexts):
    metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=list(request_ids),
        max_num_requests=max(len(request_ids), 1),
        num_contexts=num_contexts,
        prompt_lens=[c + s for c, s in zip(cached_lens, seq_lens)],
        max_num_tokens=sum(seq_lens),
        kv_cache_manager=cache_manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached_lens)),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=sparse_config,
    )
    metadata.prepare()
    return metadata


def _prefill(cache_manager, sparse_config, prompt_len, request_id=0):
    request = _make_request(request_id, prompt_len)
    assert cache_manager.prepare_context(request)
    assert cache_manager.resize_context(request, request.context_chunk_size)
    metadata = _metadata(
        cache_manager, sparse_config, [request_id], [prompt_len], [0], num_contexts=1
    )
    return request, metadata


def _advance_to_generation(cache_manager, request, prompt_len):
    scheduled = ScheduledRequests()
    scheduled.context_requests_last_chunk = [request]
    request.context_current_position = prompt_len
    request.add_new_token(prompt_len, 0)
    cache_manager.update_context_resources(scheduled)
    assert cache_manager.try_allocate_generation(request)


def _swa_pool_view(cache_manager, layer_idx, metadata):
    buffers = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.SWA)
    assert buffers.data_ptr() == metadata.sparse_mla_base_ptrs[1]
    return buffers.view(-1, buffers.shape[-1])


def _global_indices(backend, metadata, topk_indices, attention_input_type):
    indices, _ = backend.sparse_attn_predict(
        torch.empty(0, device="cuda"),
        None,
        metadata,
        AttentionForwardArgs(
            attention_input_type=attention_input_type,
            sparse_backend_args=SparseBackendForwardArgs(topk_indices=topk_indices),
        ),
    )
    return indices


def _local_indices(positions: torch.Tensor, num_cmp_indices: int, ratio: int) -> torch.Tensor:
    """The local (in-sequence) index layout the metadata builds, restated here.

    Written from the documented layout rather than read back from the metadata,
    so a bug in index construction cannot make both sides agree.
    """
    device = positions.device
    swa = (positions.unsqueeze(1) - WINDOW + 1).clamp(min=0) + torch.arange(WINDOW, device=device)
    swa = torch.where(swa > positions.unsqueeze(1), -1, swa)
    if num_cmp_indices == 0:
        return swa.int().contiguous()
    col = torch.arange(num_cmp_indices, device=device)
    num_valid = (positions + 1) // ratio
    cmp = torch.where(col.unsqueeze(0) < num_valid.unsqueeze(1), col.unsqueeze(0), -1)
    return torch.cat([swa, cmp], dim=1).int().contiguous()


def _torch_rope_interleaved(
    x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor
) -> torch.Tensor:
    """Independent interleaved (non-neox) RoPE over the trailing ROPE_DIM dims.

    `x` is `[tokens, heads, HEAD_DIM]`. Pairs `(x[2i], x[2i+1])` rotate by
    `(cos[i], sin[i])`, which is the GPT-J convention `is_neox=False` selects.
    Written in plain Torch so it can disagree with the native kernel.
    """
    out = x.float().clone()
    table = cos_sin[positions.long()].float()
    cos, sin = table[:, 0, :], table[:, 1, :]
    seg = out[..., NOPE_DIM:].reshape(x.shape[0], x.shape[1], ROPE_DIM // 2, 2)
    even, odd = seg[..., 0], seg[..., 1]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotated = torch.stack([even * cos - odd * sin, odd * cos + even * sin], dim=-1)
    out[..., NOPE_DIM:] = rotated.reshape(x.shape[0], x.shape[1], ROPE_DIM)
    return out.to(x.dtype)


def _metrics(got: torch.Tensor, ref: torch.Tensor):
    got_f, ref_f = got.float(), ref.float()
    rms = float(ref_f.square().mean().sqrt())
    rel_max_abs = float((got_f - ref_f).abs().max()) / rms
    cosine = float(
        torch.nn.functional.cosine_similarity(
            ref_f.flatten().double(), got_f.flatten().double(), dim=0
        )
    )
    return rel_max_abs, cosine


# ---------------------------------------------------------------------------
# The cache-append contract, against the real index table.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_len", [3, 128, 129, 257])
def test_append_lands_where_the_real_index_table_reads_prefill(prompt_len):
    """Every valid window slot must read back the latent of its own position.

    This is the write/read agreement the whole SM90 path rests on, checked
    against `deepseek_v4_local_to_global_indices` output computed from real
    `KVCacheManagerV2` page allocations rather than a flat synthetic pool. The
    prompt lengths straddle the 128-token page boundary in both directions.
    """
    device = torch.device("cuda")
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[1]
        _, metadata = _prefill(cache_manager, sparse_config, prompt_len)
        pool = _swa_pool_view(cache_manager, layer_idx, metadata)
        pool.zero_()

        # Row p carries the value p, so a misrouted write is identifiable
        # rather than merely different.
        latent = torch.arange(prompt_len, device=device, dtype=torch.float32)
        latent = latent.unsqueeze(1).expand(prompt_len, HEAD_DIM).contiguous().bfloat16()

        positions = sm90.token_positions(metadata, 0, prompt_len)
        assert torch.equal(positions, torch.arange(prompt_len, device=device, dtype=torch.int32))

        sm90.reset_dispatch_counts()
        sm90.append_latent_to_paged_cache(
            latent,
            pool,
            metadata.req_idx_per_token[:prompt_len],
            positions,
            metadata.sliding_block_tables[
                cache_manager.layer_offsets[layer_idx], DeepseekV4AttentionType.SWA.value
            ],
            cache_manager.tokens_per_block,
        )
        assert sm90.dispatch_counts()["append_rows_dropped"] == 0

        indices = _global_indices(
            _backend(layer_idx, sparse_config, _pos_embd_params(1)),
            metadata,
            None,
            AttentionInputType.context_only,
        )
        assert indices.shape == (prompt_len, WINDOW)

        local = _local_indices(positions, 0, 1)
        valid = local >= 0
        # Every slot the table marks valid must hold that absolute position's
        # row; every slot it pads must stay padded.
        assert torch.equal(valid, indices >= 0)
        rows = pool[indices.clamp(min=0).long()][..., 0].float()
        expected = local.clamp(min=0).float()
        assert torch.equal(rows[valid], expected[valid]), (
            "a window slot read back a different token's latent row"
        )
        # The current token is always visible to itself.
        own_slot = torch.clamp(positions.long(), max=WINDOW - 1)
        own = rows[torch.arange(prompt_len, device=device), own_slot]
        assert torch.equal(own, positions.float())
    finally:
        cache_manager.shutdown()


def test_cached_decode_reuses_context_pages_across_a_boundary():
    """Prefill, then decode tokens that cross the 128-token page boundary.

    Decode is where the cache contract actually gets tested: the window has to
    span pages allocated during prefill plus a page allocated at generation
    time, and the newly appended row has to be visible to its own step.
    """
    device = torch.device("cuda")
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[1]
        backend = _backend(layer_idx, sparse_config, _pos_embd_params(1))
        prompt_len = 126
        request, metadata = _prefill(cache_manager, sparse_config, prompt_len)
        pool = _swa_pool_view(cache_manager, layer_idx, metadata)
        pool.zero_()

        def block_table_of(md):
            return md.sliding_block_tables[
                cache_manager.layer_offsets[layer_idx], DeepseekV4AttentionType.SWA.value
            ]

        def marker(count, start=0):
            values = torch.arange(start, start + count, device=device, dtype=torch.float32)
            return values.unsqueeze(1).expand(count, HEAD_DIM).contiguous().bfloat16()

        sm90.append_latent_to_paged_cache(
            marker(prompt_len),
            pool,
            metadata.req_idx_per_token[:prompt_len],
            sm90.token_positions(metadata, 0, prompt_len),
            block_table_of(metadata),
            cache_manager.tokens_per_block,
        )

        # Steps 126..130 walk across the page-0/page-1 boundary at 128.
        for step in range(5):
            cached = prompt_len + step
            if step == 0:
                _advance_to_generation(cache_manager, request, cached)
            else:
                request.add_new_token(cached, 0)
                assert cache_manager.try_allocate_generation(request)
            gen_md = _metadata(cache_manager, sparse_config, [0], [1], [cached], num_contexts=0)
            positions = sm90.token_positions(gen_md, 0, 1)
            assert int(positions[0]) == cached, "decode position lost the cached length"

            sm90.reset_dispatch_counts()
            sm90.append_latent_to_paged_cache(
                marker(1, start=cached),
                pool,
                gen_md.req_idx_per_token[:1],
                positions,
                block_table_of(gen_md),
                cache_manager.tokens_per_block,
            )
            assert sm90.dispatch_counts()["append_rows_dropped"] == 0

            indices = _global_indices(backend, gen_md, None, AttentionInputType.generation_only)
            local = _local_indices(positions, 0, 1)
            valid = local >= 0
            assert torch.equal(valid, indices >= 0)
            rows = pool[indices.clamp(min=0).long()][..., 0].float()
            assert torch.equal(rows[valid], local.clamp(min=0).float()[valid]), (
                f"decode step {step} (position {cached}) read a wrong cached row"
            )
    finally:
        cache_manager.shutdown()


def test_append_refuses_pages_the_runtime_never_allocated():
    """An unallocated page must not be written and must not pass silently."""
    device = torch.device("cuda")
    pool = torch.zeros(4 * TOKENS_PER_BLOCK, HEAD_DIM, device=device, dtype=torch.bfloat16)
    latent = torch.ones(3, HEAD_DIM, device=device, dtype=torch.bfloat16)
    block_table = torch.tensor([[0, -1]], dtype=torch.int32, device=device)

    sm90.reset_dispatch_counts()
    sm90.append_latent_to_paged_cache(
        latent,
        pool,
        torch.zeros(3, dtype=torch.int32, device=device),
        # 5 lands in the allocated page 0; 200 hits the BAD_PAGE_INDEX column;
        # 400 is past the end of the block table entirely.
        torch.tensor([5, 200, 400], dtype=torch.int32, device=device),
        block_table,
        TOKENS_PER_BLOCK,
    )
    assert sm90.dispatch_counts()["append_rows_dropped"] == 2
    assert float(pool[5].float().max()) == 1.0
    # Nothing outside the one valid destination was touched.
    written = (pool.float().abs().sum(dim=1) > 0).nonzero().flatten().tolist()
    assert written == [5], f"append wrote outside the allocated page: rows {written}"


def test_metadata_positions_agree_with_model_position_ids():
    """The cache address and the RoPE angle must come from the same position.

    `forward_sparse_attn_sm90` derives positions from the cache accounting, the
    way the native RoPE kernels do, while the output-side inverse RoPE uses the
    model's `position_ids`. If those ever disagreed the forward and inverse
    rotations would not cancel, so the agreement is checked rather than assumed
    --- for a whole prefill, a resumed (chunked) prefill, and a decode step.
    """
    device = torch.device("cuda")
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        prompt_len = 200
        request, metadata = _prefill(cache_manager, sparse_config, prompt_len)
        expected = torch.arange(prompt_len, device=device, dtype=torch.int32)
        assert torch.equal(sm90.token_positions(metadata, 0, prompt_len), expected)

        # Resumed prefill: 64 tokens already cached, 40 new ones this pass.
        resumed = _metadata(cache_manager, sparse_config, [0], [40], [64], num_contexts=1)
        assert torch.equal(
            sm90.token_positions(resumed, 0, 40),
            torch.arange(64, 104, device=device, dtype=torch.int32),
        )

        _advance_to_generation(cache_manager, request, prompt_len)
        gen_md = _metadata(cache_manager, sparse_config, [0], [1], [prompt_len], num_contexts=0)
        assert int(sm90.token_positions(gen_md, 0, 1)[0]) == prompt_len
    finally:
        cache_manager.shutdown()


def test_external_rope_matches_an_independent_torch_rotation():
    """Pin the external-RoPE step's convention, table and position mapping.

    `mla_rope_inplace` is the same native binding the Blackwell path uses for
    the output-side inverse rotation, but nothing else in this path would
    notice if it were fed the wrong positions or the wrong half of the row.
    """
    device = torch.device("cuda")
    pos_embd = _pos_embd_params(128)
    rotary = RotaryEmbedding(pos_embd.rope, head_dim=ROPE_DIM, is_neox=False, inverse=True)
    torch.manual_seed(31)
    x = (torch.randn(9, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    positions = torch.tensor(
        [0, 1, 127, 128, 129, 255, 256, 257, 384], dtype=torch.int32, device=device
    )

    reference = _torch_rope_interleaved(x, positions, rotary.rotary_cos_sin)
    got = x.clone()
    torch.ops.trtllm.mla_rope_inplace(
        got, positions, rotary.rotary_cos_sin, LOCAL_HEADS, NOPE_DIM, ROPE_DIM, False, False
    )
    assert torch.equal(got[..., :NOPE_DIM], x[..., :NOPE_DIM]), "RoPE touched the non-RoPE half"
    rel_max_abs, cosine = _metrics(got, reference)
    assert rel_max_abs < 5e-3 and cosine > 0.9999, (
        f"external RoPE disagrees with the Torch rotation: {rel_max_abs=} {cosine=}"
    )

    # Position 0 is the identity rotation, and a different position must differ.
    assert torch.equal(got[0], x[0])
    assert not torch.equal(got[1], x[1])


# ---------------------------------------------------------------------------
# The whole SM90 phase, end to end.
# ---------------------------------------------------------------------------


def _run_phase(mla, metadata, q, latent, output, topk_indices, input_type):
    return sm90.forward_sparse_attn_sm90(mla, q, metadata, output, latent, topk_indices, input_type)


def _reference_output(mla, q, latent_by_position, positions, local_indices, sink, cmp_rows):
    """Source-anchored golden over rows selected by absolute position."""
    cos_sin = mla.inverse_rotary_emb.rotary_cos_sin
    q_roped = _torch_rope_interleaved(q.view(q.shape[0], LOCAL_HEADS, HEAD_DIM), positions, cos_sin)
    kv = _torch_rope_interleaved(
        latent_by_position.unsqueeze(1),
        torch.arange(latent_by_position.shape[0], dtype=torch.int32, device=q.device),
        cos_sin,
    ).squeeze(1)
    # `Attention.forward` in inference/model.py FP8-simulates the non-RoPE half
    # of the window latent before it enters the cache:
    #   act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
    # The compressed rows this test seeds directly are not subject to it --- in
    # the source they come out of the Compressor's own quantization.
    kv = torch.cat([tg.fp8_quant_dequant(kv[:, :NOPE_DIM], 64), kv[:, NOPE_DIM:]], dim=-1)
    shifted = local_indices
    if cmp_rows is not None:
        slot = torch.arange(local_indices.shape[1], device=q.device)
        shifted = torch.where(
            (local_indices >= 0) & (slot.unsqueeze(0) >= WINDOW),
            local_indices + kv.shape[0],
            local_indices,
        )
        kv = torch.cat([kv, cmp_rows], dim=0)
    return tg.sparse_attention(
        q_roped.unsqueeze(0),
        kv.unsqueeze(0),
        sink,
        shifted.unsqueeze(0),
        1.0 / math.sqrt(float(HEAD_DIM)),
    ).squeeze(0)


def _seed_compress_pool(cache_manager, layer_idx, request_id, num_compressed, device):
    """Write known rows into the COMPRESS pool at compressed-token ordinals.

    Addressed through `get_cache_indices` and the compressed page size, which
    is the layout `deepseek_v4_local_to_global_indices` reads back --- computed
    here independently rather than by calling that kernel.
    """
    buffers = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS)
    block_size = cache_manager.compressed_block_sizes[layer_idx]
    torch.manual_seed(41)
    rows = (torch.randn(num_compressed, HEAD_DIM, device=device) * 0.5).bfloat16()
    pages = cache_manager.get_cache_indices(request_id, layer_idx, DeepseekV4AttentionType.COMPRESS)
    for ordinal in range(num_compressed):
        buffers[pages[ordinal // block_size], ordinal % block_size, :] = rows[ordinal]
    return rows


@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_the_sm90_phase_matches_the_golden_through_the_real_runtime(ratio):
    """Prefill then decode through `forward_sparse_attn_sm90`, end to end.

    Real cache manager, real metadata, real index kernel, real backend, real
    pool base pointers. The golden selects the same rows by absolute position
    from an independently RoPE'd latent, so a wrong page, a wrong pool base, a
    wrong rotation or a wrong slot-to-pool split all show up as a numeric
    failure rather than as plausible output.
    """
    device = torch.device("cuda")
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[ratio]
        pos_embd = _pos_embd_params(ratio)
        backend = _backend(layer_idx, sparse_config, pos_embd)
        torch.manual_seed(53)
        sink = torch.randn(LOCAL_HEADS, device=device, dtype=torch.float32) * 0.5
        backend.attn_sink = nn.Parameter(sink, requires_grad=False)
        mla = _mla_host(backend, layer_idx, pos_embd)

        prompt_len = 257
        request, metadata = _prefill(cache_manager, sparse_config, prompt_len)
        pool = _swa_pool_view(cache_manager, layer_idx, metadata)
        pool.zero_()

        num_cmp_indices = metadata.max_compressed_indices[ratio]
        cmp_rows = None
        if ratio > 1:
            # One decode step follows, so seed through the decode token's count.
            cmp_rows = _seed_compress_pool(
                cache_manager, layer_idx, 0, (prompt_len + 1) // ratio, device
            )

        torch.manual_seed(59)
        total = prompt_len + 1
        latent_all = (torch.randn(total, HEAD_DIM, device=device) * 0.5).bfloat16()
        q_all = (torch.randn(total, LOCAL_HEADS * HEAD_DIM, device=device) * 0.5).bfloat16()

        sm90.reset_dispatch_counts()

        # --- prefill ---
        positions = sm90.token_positions(metadata, 0, prompt_len)
        local = _local_indices(positions, num_cmp_indices, ratio)
        topk = local[:, WINDOW:].contiguous() if ratio == 4 else None
        output = torch.empty(
            prompt_len, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        got = _run_phase(
            mla,
            metadata,
            q_all[:prompt_len].clone(),
            latent_all[:prompt_len].clone(),
            output,
            topk,
            AttentionInputType.context_only,
        )
        assert got.data_ptr() == output.data_ptr()
        counts = sm90.dispatch_counts()
        assert counts["context"] == 1 and counts["generation"] == 0
        assert counts["sparse_mla_dual_pool"] == 1
        assert counts["append_latent_to_paged_cache"] == 1
        assert counts["append_rows_dropped"] == 0
        assert torch.isfinite(got.float()).all()

        ref = _reference_output(
            mla, q_all[:prompt_len], latent_all[:prompt_len], positions, local, sink, cmp_rows
        )
        rel_max_abs, cosine = _metrics(got.view(prompt_len, LOCAL_HEADS, HEAD_DIM), ref)
        assert rel_max_abs <= 0.03 and cosine >= 0.999, (
            f"prefill ratio {ratio}: {rel_max_abs=:.4e} {cosine=:.6f}"
        )

        # --- decode, reusing the pages prefill wrote ---
        _advance_to_generation(cache_manager, request, prompt_len)
        gen_md = _metadata(cache_manager, sparse_config, [0], [1], [prompt_len], num_contexts=0)
        gen_positions = sm90.token_positions(gen_md, 0, 1)
        gen_local = _local_indices(gen_positions, num_cmp_indices, ratio)
        gen_topk = gen_local[:, WINDOW:].contiguous() if ratio == 4 else None
        gen_out = torch.empty(1, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16)
        _run_phase(
            mla,
            gen_md,
            q_all[prompt_len:].clone(),
            latent_all[prompt_len:].clone(),
            gen_out,
            gen_topk,
            AttentionInputType.generation_only,
        )
        counts = sm90.dispatch_counts()
        assert counts["generation"] == 1 and counts["append_rows_dropped"] == 0
        assert counts["sparse_mla_dual_pool"] == 2

        gen_ref = _reference_output(
            mla, q_all[prompt_len:], latent_all, gen_positions, gen_local, sink, cmp_rows
        )
        rel_max_abs, cosine = _metrics(gen_out.view(1, LOCAL_HEADS, HEAD_DIM), gen_ref)
        assert rel_max_abs <= 0.03 and cosine >= 0.999, (
            f"decode ratio {ratio}: {rel_max_abs=:.4e} {cosine=:.6f}"
        )
    finally:
        cache_manager.shutdown()


def _poison_compress_pool(cache_manager, layer_idx, request_id, num_compressed, device):
    """Fill another layer's compressed region with values no answer may contain.

    A view based on the wrong layer reads *some* valid pool memory, so a zeroed
    neighbour would still average out to something plausible. Large values make
    the wrong rows dominate the softmax instead.
    """
    buffers = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS)
    block_size = cache_manager.compressed_block_sizes[layer_idx]
    pages = cache_manager.get_cache_indices(request_id, layer_idx, DeepseekV4AttentionType.COMPRESS)
    for ordinal in range(num_compressed):
        buffers[pages[ordinal // block_size], ordinal % block_size, :] = torch.full(
            (HEAD_DIM,), 8.0, device=device, dtype=torch.bfloat16
        )


@pytest.mark.parametrize("ratio", [4, 128])
def test_a_layer_sharing_a_compressed_pool_reads_its_own_rows(ratio):
    """The second layer of a ratio sits *inside* the pool, not at its base.

    Compressed pools are shared: ``get_mem_pool_base_address`` bakes the layer
    offset into the pointer for every layer after the first, while the index
    table --- and the Blackwell native op's ``aux_kv_cache_pool_ptr`` --- still
    count rows from the pool origin. A one-layer-per-ratio geometry has a zero
    offset and cannot tell the two apart; the real checkpoint has 61 layers and
    cannot avoid it. The first layer of the same ratio is poisoned so that
    reading at the wrong origin is a numeric failure rather than plausible
    output.
    """
    device = torch.device("cuda")
    sparse_config = _sparse_config(SHARED_POOL_RATIOS)
    cache_manager = _cache_manager(sparse_config)
    try:
        first_layer = SHARED_POOL_RATIOS.index(ratio)
        layer_idx = len(SHARED_POOL_RATIOS) - 1 - SHARED_POOL_RATIOS[::-1].index(ratio)
        assert layer_idx != first_layer, "geometry must have two layers with this ratio"

        pos_embd = _pos_embd_params(ratio)
        backend = _backend(layer_idx, sparse_config, pos_embd)
        torch.manual_seed(53)
        sink = torch.randn(LOCAL_HEADS, device=device, dtype=torch.float32) * 0.5
        backend.attn_sink = nn.Parameter(sink, requires_grad=False)
        mla = _mla_host(backend, layer_idx, pos_embd)

        prompt_len = 257
        request, metadata = _prefill(cache_manager, sparse_config, prompt_len)
        del request

        # The property under test, stated before it is exercised: this layer's
        # compressed region is a whole number of token rows past the pool base
        # the index table counts from, and it is not zero.
        pool_base = metadata.sparse_mla_base_ptrs[ratio]
        own = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS)
        offset_bytes = own.data_ptr() - pool_base
        assert offset_bytes > 0, "expected this layer to sit past the shared pool base"
        assert offset_bytes % (HEAD_DIM * own.element_size()) == 0
        assert (
            cache_manager.get_buffers(first_layer, DeepseekV4AttentionType.COMPRESS).data_ptr()
            == pool_base
        )

        pool = _swa_pool_view(cache_manager, layer_idx, metadata)
        pool.zero_()
        num_cmp_indices = metadata.max_compressed_indices[ratio]
        num_compressed = prompt_len // ratio
        cmp_rows = _seed_compress_pool(cache_manager, layer_idx, 0, num_compressed, device)
        _poison_compress_pool(cache_manager, first_layer, 0, num_compressed, device)

        torch.manual_seed(59)
        latent = (torch.randn(prompt_len, HEAD_DIM, device=device) * 0.5).bfloat16()
        q = (torch.randn(prompt_len, LOCAL_HEADS * HEAD_DIM, device=device) * 0.5).bfloat16()

        positions = sm90.token_positions(metadata, 0, prompt_len)
        local = _local_indices(positions, num_cmp_indices, ratio)
        topk = local[:, WINDOW:].contiguous() if ratio == 4 else None
        output = torch.empty(
            prompt_len, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        got = _run_phase(
            mla,
            metadata,
            q.clone(),
            latent.clone(),
            output,
            topk,
            AttentionInputType.context_only,
        )
        assert torch.isfinite(got.float()).all()

        ref = _reference_output(mla, q, latent, positions, local, sink, cmp_rows)
        rel_max_abs, cosine = _metrics(got.view(prompt_len, LOCAL_HEADS, HEAD_DIM), ref)
        assert rel_max_abs <= 0.03 and cosine >= 0.999, (
            f"shared-pool ratio {ratio} on layer {layer_idx}: {rel_max_abs=:.4e} {cosine=:.6f}"
        )
    finally:
        cache_manager.shutdown()


def test_a_mixed_batch_keeps_the_two_phases_on_their_own_token_ranges():
    """One context request and one decode request in a single forward.

    A mixed batch is where the phase slicing can go wrong quietly: both halves
    read the same metadata but must take disjoint token ranges, different
    request rows of the block table, and different position bases. Getting the
    split wrong still returns output of the right shape.
    """
    device = torch.device("cuda")
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config, batch_size=2)
    try:
        layer_idx = RATIO_TO_LAYER[1]
        pos_embd = _pos_embd_params(1)
        backend = _backend(layer_idx, sparse_config, pos_embd)
        torch.manual_seed(67)
        sink = torch.randn(LOCAL_HEADS, device=device, dtype=torch.float32) * 0.5
        backend.attn_sink = nn.Parameter(sink, requires_grad=False)
        mla = _mla_host(backend, layer_idx, pos_embd)

        # Request 1 decodes at position 200; request 0 prefills 96 tokens in
        # the same batch. Their windows overlap in index value but not in page.
        decode_len = 200
        ctx_len = 96
        decoder, decoder_md = _prefill(cache_manager, sparse_config, decode_len, request_id=1)
        pool = _swa_pool_view(cache_manager, layer_idx, decoder_md)
        pool.zero_()

        torch.manual_seed(71)
        latent_decoder = (torch.randn(decode_len + 1, HEAD_DIM, device=device) * 0.5).bfloat16()
        latent_ctx = (torch.randn(ctx_len, HEAD_DIM, device=device) * 0.5).bfloat16()

        # Seed the decoder's history through the ordinary prefill path.
        ctx_out = torch.empty(
            decode_len, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        _run_phase(
            mla,
            decoder_md,
            torch.zeros(decode_len, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16),
            latent_decoder[:decode_len].clone(),
            ctx_out,
            None,
            AttentionInputType.context_only,
        )
        _advance_to_generation(cache_manager, decoder, decode_len)

        prefiller = _make_request(0, ctx_len)
        assert cache_manager.prepare_context(prefiller)
        assert cache_manager.resize_context(prefiller, prefiller.context_chunk_size)

        # Contexts come first in the flattened batch, generations after.
        mixed = _metadata(
            cache_manager,
            sparse_config,
            [0, 1],
            [ctx_len, 1],
            [0, decode_len],
            num_contexts=1,
        )
        assert mixed.num_ctx_tokens == ctx_len and mixed.num_tokens == ctx_len + 1
        assert torch.equal(
            sm90.token_positions(mixed, 0, mixed.num_tokens),
            torch.cat(
                [
                    torch.arange(ctx_len, device=device, dtype=torch.int32),
                    torch.tensor([decode_len], device=device, dtype=torch.int32),
                ]
            ),
        )

        torch.manual_seed(73)
        q_mixed = (torch.randn(ctx_len + 1, LOCAL_HEADS * HEAD_DIM, device=device) * 0.5).bfloat16()
        out = torch.empty(ctx_len + 1, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16)
        latent_mixed = torch.cat([latent_ctx, latent_decoder[decode_len:]], dim=0)

        sm90.reset_dispatch_counts()
        _run_phase(
            mla,
            mixed,
            q_mixed[:ctx_len].clone(),
            latent_mixed[:ctx_len].clone(),
            out[:ctx_len],
            None,
            AttentionInputType.context_only,
        )
        _run_phase(
            mla,
            mixed,
            q_mixed[ctx_len:].clone(),
            latent_mixed[ctx_len:].clone(),
            out[ctx_len:],
            None,
            AttentionInputType.generation_only,
        )
        counts = sm90.dispatch_counts()
        assert counts["context"] == 1 and counts["generation"] == 1
        assert counts["append_rows_dropped"] == 0
        assert torch.isfinite(out.float()).all()

        # The context half sees only its own fresh sequence...
        ctx_positions = torch.arange(ctx_len, device=device, dtype=torch.int32)
        ctx_ref = _reference_output(
            mla,
            q_mixed[:ctx_len],
            latent_ctx,
            ctx_positions,
            _local_indices(ctx_positions, 0, 1),
            sink,
            None,
        )
        rel_max_abs, cosine = _metrics(out[:ctx_len].view(ctx_len, LOCAL_HEADS, HEAD_DIM), ctx_ref)
        assert rel_max_abs <= 0.03 and cosine >= 0.999, (
            f"mixed-batch context half: {rel_max_abs=:.4e} {cosine=:.6f}"
        )

        # ...and the decode half sees only the 200 rows of its own history.
        gen_positions = torch.tensor([decode_len], device=device, dtype=torch.int32)
        gen_ref = _reference_output(
            mla,
            q_mixed[ctx_len:],
            latent_decoder,
            gen_positions,
            _local_indices(gen_positions, 0, 1),
            sink,
            None,
        )
        rel_max_abs, cosine = _metrics(out[ctx_len:].view(1, LOCAL_HEADS, HEAD_DIM), gen_ref)
        assert rel_max_abs <= 0.03 and cosine >= 0.999, (
            f"mixed-batch decode half: {rel_max_abs=:.4e} {cosine=:.6f}"
        )
    finally:
        cache_manager.shutdown()


def test_a_rebased_pool_view_is_rejected_rather_than_silently_offset():
    """Global indices count tokens from the pool base; a shifted view is wrong.

    Every index would be off by a constant, which still lands inside the pool
    and still produces attention-shaped output, so the equality is asserted.
    """
    device = torch.device("cuda")
    sparse_config = _sparse_config()
    cache_manager = _cache_manager(sparse_config)
    try:
        layer_idx = RATIO_TO_LAYER[1]
        pos_embd = _pos_embd_params(1)
        backend = _backend(layer_idx, sparse_config, pos_embd)
        mla = _mla_host(backend, layer_idx, pos_embd)
        _, metadata = _prefill(cache_manager, sparse_config, 8)

        # A table built against a base one token row past the real pool is
        # exactly the failure mode: every index still lands inside the pool.
        buffers = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.SWA)
        metadata.sparse_mla_base_ptrs = dict(metadata.sparse_mla_base_ptrs)
        metadata.sparse_mla_base_ptrs[1] = buffers.data_ptr() + HEAD_DIM * buffers.element_size()

        with pytest.raises(AssertionError, match="index table is built against pool base"):
            sm90.forward_sparse_attn_sm90(
                mla,
                torch.zeros(8, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16),
                metadata,
                torch.zeros(8, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16),
                torch.zeros(8, HEAD_DIM, device=device, dtype=torch.bfloat16),
                None,
                AttentionInputType.context_only,
            )
    finally:
        cache_manager.shutdown()


def test_the_swa_token_stride_the_index_table_uses_matches_the_pool_row():
    """The index unit is one SWA token row; the pool view must use the same."""
    assert (
        get_token_bytes(HEAD_DIM, INDEX_HEAD_DIM, 1, DeepseekV4AttentionType.SWA, False)
        == HEAD_DIM * torch.finfo(torch.bfloat16).bits // 8
    )
    assert (
        get_token_bytes(HEAD_DIM, INDEX_HEAD_DIM, 128, DeepseekV4AttentionType.COMPRESS, False)
        == HEAD_DIM * torch.finfo(torch.bfloat16).bits // 8
    )


# ---------------------------------------------------------------------------
# The protected Blackwell branch.
# ---------------------------------------------------------------------------


def test_sm100_still_selects_the_native_branch_and_sm90_does_not():
    """The SM90 branch must be entered only below SM100, in both phases.

    A unit-level dispatch check, deliberately: the point is which branch the
    architecture predicate selects, and the Blackwell native op cannot run on
    this machine to be observed directly.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import module as v4_module

    calls = []

    def fake_sm90(mla, q, metadata, output, latent, topk, input_type):
        calls.append(("sm90", input_type))
        return output

    def fake_native(*args, **kwargs):
        calls.append(("native", kwargs.get("attention_input_type")))
        return kwargs["output"]

    stub = SimpleNamespace(
        num_heads_tp=LOCAL_HEADS,
        qk_head_dim=HEAD_DIM,
        qk_nope_head_dim=NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        kv_lora_rank=NOPE_DIM,
        mqa=SimpleNamespace(has_fp8_kv_cache=False),
        kv_a_layernorm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        mapping=SimpleNamespace(has_cp_helix=lambda: False),
        out_scale=None,
        _attn_forward_gen=fake_native,
        inverse_rotary_emb=SimpleNamespace(rotary_cos_sin=None, is_neox=False),
    )
    q = torch.zeros(2, LOCAL_HEADS * HEAD_DIM)
    out = torch.zeros(2, LOCAL_HEADS * HEAD_DIM)

    original_sm90 = v4_module.forward_sparse_attn_sm90
    original_sm_version = v4_module.get_sm_version
    try:
        v4_module.forward_sparse_attn_sm90 = fake_sm90

        v4_module.get_sm_version = lambda: 90
        v4_module.forward_context_sparse_attn(stub, q, None, None, None, out)
        v4_module.forward_generation_sparse_attn(stub, q, None, None, None, out)
        assert [c[0] for c in calls] == ["sm90", "sm90"]
        assert calls[0][1] == AttentionInputType.context_only
        assert calls[1][1] == AttentionInputType.generation_only

        calls.clear()
        for sm in (100, 103):
            v4_module.get_sm_version = lambda sm=sm: sm
            v4_module.forward_context_sparse_attn(stub, q, None, None, None, out)
            assert calls and calls[-1][0] == "native", f"SM{sm} left the Blackwell branch"
            assert calls[-1][1] == AttentionInputType.context_only
    finally:
        v4_module.forward_sparse_attn_sm90 = original_sm90
        v4_module.get_sm_version = original_sm_version
