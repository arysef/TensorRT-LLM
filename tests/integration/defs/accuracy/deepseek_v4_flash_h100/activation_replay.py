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
"""``source_activation_replay``: real checkpoint activations through the SM90 path.

Every earlier rung of the ladder judges TensorRT-LLM against a *reference*
--- a pure-Torch golden, or the source's own kernel driven by synthetic
Gaussian inputs. This one drives the real ``DeepseekV4TrtllmAttention`` SM90
branch, the real ``DeepseekV4CacheManager`` (a ``KVCacheManagerV2`` subclass)
and the real index tables with the activations the *official* model produced
from the real checkpoint on this rank, and compares against what the official
kernel produced from those same activations.

The replay boundary is chosen so no weight mapping stands between the two
sides. ``Attention.forward`` in ``inference/model.py`` reaches its sparse
kernel through

    qr = q = self.q_norm(self.wq_a(x))
    q = self.wq_b(q).unflatten(-1, (n_local_heads, head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)   # <- replay input
    apply_rotary_emb(q[..., -rd:], freqs_cis)
    kv = self.kv_norm(self.wkv(x))                              # <- replay input
    apply_rotary_emb(kv[..., -rd:], freqs_cis)
    act_quant(kv[..., :-rd], 64, ..., True)
    ...
    o = sparse_attn(q, kv, self.attn_sink, topk_idxs, softmax_scale)   # <- reference

so the two marked tensors are captured *before* the in-place RoPE, handed to
``forward_sparse_attn_sm90``, and its output is compared with the source
kernel's own ``o``. Both sides therefore start from identical real activations
and identical real weights (the sink is copied from the official parameter),
and every step in between --- the external RoPE, the FP8 window-latent
simulation, the paged append, the dual-pool gather, the FP32 online softmax
and the denominator-only sink --- is TensorRT-LLM's.

Head geometry is the production one: the official model is loaded MP-sharded
across the same eight ranks, so each rank replays its own 8 of the 64 query
heads at the checkpoint's 512-wide head dim, 128-token window, ratio-4 top-512
selection and ratio-128 all-valid selection.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils._device

#: Nesting depth of :func:`_host_default_device`; see its docstring.
_DEVICE_SCOPE_DEPTH = 0


@contextlib.contextmanager
def _host_default_device():
    """Build RoPE tables with the default device back on the host.

    ``OfficialSource`` ends its load with ``torch.set_default_device("cuda")``,
    which the official model's own index helpers need --- they allocate with a
    bare ``torch.arange``. The TensorRT-LLM runtime wants the opposite: its
    YaRN table builder ends in ``tensor.numpy()`` and its attention metadata
    pins host-side bookkeeping, neither of which works on a CUDA tensor. Both
    sides are right about their own code, so the default is scoped rather than
    fought over, and every tensor that must live on the GPU inside this block
    is allocated with an explicit device.

    The override is *removed* rather than set to ``"cpu"``. ``set_default_device``
    installs a ``__torch_function__`` mode that rewrites every factory call, and
    the cache manager reaches its pools through ``convert_to_torch_tensor``,
    which wraps an existing device pointer and then checks the pointer survived.
    A "cpu" mode fails that check just as surely as a "cuda" one; only no mode
    at all leaves those calls alone.
    """
    global _DEVICE_SCOPE_DEPTH
    if _DEVICE_SCOPE_DEPTH:
        # Re-entrant by design. `get_default_device()` reports "cpu" both when a
        # cpu mode is installed and when no mode is, so a nested exit would
        # *install* a cpu mode the outer block had deliberately removed --- and
        # a cpu mode breaks `convert_to_torch_tensor` exactly like a cuda one.
        yield
        return
    # `torch.utils._device.CURRENT_DEVICE` is `None` when no mode is installed,
    # which `get_default_device()` cannot distinguish from an explicit "cpu".
    previous = torch.utils._device.CURRENT_DEVICE
    _DEVICE_SCOPE_DEPTH += 1
    torch.set_default_device(None)
    try:
        yield
    finally:
        _DEVICE_SCOPE_DEPTH -= 1
        torch.set_default_device(previous)


def _pos_embd_params(compress_ratio: int, cfg: Any, max_seq_len: int) -> Any:
    """The per-ratio RoPE contract ``_deepseek_v4_pos_embd_params`` applies.

    Restated from the checkpoint config rather than imported so that a change
    to the model-side helper shows up as a numeric failure here instead of
    being adopted by both sides at once.
    """
    from tensorrt_llm._torch.attention_backend.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

    scaling = cfg.rope_scaling or {}
    rope = RopeParams(
        dim=cfg.qk_rope_head_dim,
        max_positions=max_seq_len,
        original_max_positions=scaling.get("original_max_position_embeddings", 65536),
        max_seq_len=max_seq_len,
        beta_fast=scaling.get("beta_fast", 32),
        beta_slow=scaling.get("beta_slow", 1),
    )
    if compress_ratio > 1:
        rope.theta = float(cfg.compress_rope_theta)
        rope.scale_type = RotaryScalingType.yarn
        rope.scale = float(scaling.get("factor", 16))
        rope.mscale = 0.0
        rope.mscale_all_dim = 0.0
        pos_type = PositionEmbeddingType.yarn
    else:
        # A ratio-0 layer is pure sliding-window attention: the source disables
        # YaRN and falls back to the base theta.
        rope.theta = float(cfg.rope_theta)
        rope.scale_type = RotaryScalingType.none
        rope.scale = 1.0
        pos_type = PositionEmbeddingType.rope_gptj
    return PositionalEmbeddingParams(type=pos_type, rope=rope, is_neox=False)


class _Replay:
    """One rank's TensorRT-LLM side: cache manager, backends and metadata."""

    def __init__(self, cfg: Any, layer_ids: tuple[int, ...], max_seq_len: int, local_heads: int):
        from tensorrt_llm._torch.attention_backend.interface import MLAParams
        from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import DeepseekV4CacheManager
        from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.backend import (
            DeepseekV4TrtllmAttention,
        )
        from tensorrt_llm.bindings import DataType
        from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
        from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
        from tensorrt_llm.mapping import Mapping

        self.cfg = cfg
        self.layer_ids = layer_ids
        self.max_seq_len = max_seq_len
        self.local_heads = local_heads
        self.head_dim = cfg.v_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.nope_dim = self.head_dim - self.rope_dim
        # The cache manager indexes its pools by *position in its own ratio
        # list*, so the three layers under replay are given a dense local
        # numbering. Every dimension that the kernel and the index tables see
        # --- head width, window, ratios, top-k --- stays the checkpoint's.
        self.ratios = [cfg.compress_ratios[lid] for lid in layer_ids]
        self.slot = {lid: i for i, lid in enumerate(layer_ids)}

        self.sparse_config = DeepSeekV4SparseAttentionConfig(
            index_n_heads=cfg.index_n_heads,
            index_head_dim=cfg.index_head_dim,
            index_topk=cfg.index_topk,
            window_size=cfg.window_size,
            compress_ratios=list(self.ratios),
            skip_indexer_for_short_seqs=False,
        )
        self.cache_manager = DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                max_tokens=max_seq_len * 2, enable_block_reuse=False, event_buffer_max_size=0
            ),
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(self.ratios),
            num_kv_heads=1,
            head_dim=self.head_dim,
            tokens_per_block=128,
            max_seq_len=max_seq_len,
            max_batch_size=1,
            max_input_len=max_seq_len,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=cfg.vocab_size,
            max_num_tokens=max_seq_len + 1,
            sparse_attn_config=self.sparse_config,
        )

        self.mla: dict[int, Any] = {}
        for lid in layer_ids:
            ratio = cfg.compress_ratios[lid]
            pos_embd = _pos_embd_params(ratio, cfg, max_seq_len)
            mla_params = MLAParams(
                q_lora_rank=cfg.q_lora_rank,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_rope_head_dim=self.rope_dim,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                v_head_dim=self.head_dim,
                rope_append=False,
                predicted_tokens_per_seq=1,
                hidden_size=cfg.hidden_size,
            )
            with _host_default_device():
                backend = DeepseekV4TrtllmAttention(
                    layer_idx=self.slot[lid],
                    num_heads=local_heads,
                    head_dim=self.head_dim,
                    num_kv_heads=1,
                    q_scaling=1.0,
                    pos_embd_params=pos_embd,
                    mla_params=mla_params,
                    sparse_attention_config=self.sparse_config,
                    # The Compressor's projections are real parameters here,
                    # not placeholders: the replay loads the official layer's
                    # own `wkv`/`wgate`/`norm`/`ape` into them and produces the
                    # compressed rows rather than copying the source's.
                    skip_create_weights_in_init=False,
                )
                backend.update_quant_config(None)
                # `_host_default_device` has the default device back on the
                # host, which is what the metadata and the RoPE table builder
                # need, so the parameters just created landed on the CPU.
                for sub in ("compressor", "indexer"):
                    module = getattr(backend, sub, None)
                    if module is not None:
                        module.cuda()
                self.mla[lid] = _MlaHost(backend, self.slot[lid], pos_embd, local_heads, cfg)

    def shutdown(self) -> None:
        self.cache_manager.shutdown()


class _MlaHost:
    """The attribute surface ``forward_sparse_attn_sm90`` reads off an ``MLA``."""

    def __init__(self, backend: Any, layer_idx: int, pos_embd: Any, local_heads: int, cfg: Any):
        from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.rope import (
            deepseek_v4_rotary_embedding,
        )

        self.mqa = backend
        self.layer_idx = layer_idx
        self.num_heads_tp = local_heads
        self.qk_head_dim = cfg.v_head_dim
        self.qk_nope_head_dim = cfg.v_head_dim - cfg.qk_rope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.compressor = None
        self.indexer = getattr(backend, "indexer", None)
        # Built exactly as `_deepseek_v4_mla_post_init` builds it, including the
        # checkpoint's own rotary table, so the replay cannot pass with a table
        # the production module would not have used.
        self.inverse_rotary_emb = deepseek_v4_rotary_embedding(
            pos_embd.rope, head_dim=cfg.qk_rope_head_dim, is_neox=pos_embd.is_neox, inverse=True
        )


def _metadata(replay: _Replay, seq_len: int, cached_len: int, num_contexts: int) -> Any:
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm.mapping import Mapping

    metadata = DeepseekV4TrtllmAttentionMetadata(
        # Explicitly host-side: the metadata setter pins this buffer, and the
        # official model leaves the default device on CUDA.
        seq_lens=torch.tensor([seq_len], dtype=torch.int, device="cpu"),
        request_ids=[0],
        max_num_requests=1,
        num_contexts=num_contexts,
        prompt_lens=[cached_len + seq_len],
        max_num_tokens=max(seq_len, 1),
        kv_cache_manager=replay.cache_manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached_len]),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=replay.sparse_config,
    )
    metadata.prepare()
    return metadata


def _open_request(replay: _Replay, prompt_len: int) -> Any:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import SamplingConfig

    request = LlmRequest(
        request_id=0,
        max_new_tokens=8,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    assert replay.cache_manager.prepare_context(request)
    assert replay.cache_manager.resize_context(request, request.context_chunk_size)
    return request


def _advance_to_generation(replay: _Replay, request: Any, prompt_len: int) -> None:
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    scheduled = ScheduledRequests()
    scheduled.context_requests_last_chunk = [request]
    request.context_current_position = prompt_len
    request.add_new_token(prompt_len, 0)
    replay.cache_manager.update_context_resources(scheduled)
    assert replay.cache_manager.try_allocate_generation(request)


def load_source_compressor(backend: Any, attn: Any, compressor: Any = None) -> None:
    """Copy the official Compressor's parameters into TensorRT-LLM's.

    The two modules hold the same three tensors in a different shape: the
    source keeps ``wkv`` and ``wgate`` as separate FP32 projections, and
    TensorRT-LLM fuses them into one ``wkv_gate`` whose first ``state_dim``
    rows are the KV half. That split order is a contract rather than a
    convention --- swapping it produces a differently-pooled but entirely
    plausible compressed row --- and it is asserted here by shape rather than
    assumed.

    The checkpoint stores both projections in BF16 and the source widens them
    to FP32 at load, so narrowing back is exact.
    """
    src = attn.compressor if compressor is None else compressor
    dst = backend.compressor
    state_dim = dst.state_dim
    assert src.wkv.weight.shape == (state_dim, src.dim), (
        f"official wkv is {tuple(src.wkv.weight.shape)}, TensorRT-LLM expects "
        f"({state_dim}, {src.dim}); the fused wkv_gate split would be wrong"
    )
    with torch.no_grad():
        dst.wkv_gate.weight.copy_(
            torch.cat([src.wkv.weight, src.wgate.weight], dim=0).to(dst.wkv_gate.weight.dtype)
        )
        dst.norm.weight.copy_(src.norm.weight.to(dst.norm.weight.dtype))
        dst.ape.copy_(src.ape.to(dst.ape.dtype))


def load_source_indexer(backend: Any, attn: Any, world: int) -> dict[str, Any]:
    """Copy the official Indexer's parameters into TensorRT-LLM's.

    The two implementations shard this module differently, and that difference
    is the whole reason a collective appears here. The source declares
    ``wq_b`` and ``weights_proj`` as ``ColumnParallelLinear``, so rank *r*
    holds index heads ``8r .. 8r+7`` and its ``index_score`` is a partial sum
    that ``Indexer.forward`` finishes with ``dist.all_reduce``. TensorRT-LLM
    declares both as plain replicated ``Linear``s over all 64 heads and needs
    no runtime collective; the two compute the same quantity.

    So the collective moves from every token to once at load: the eight
    rank-local slices are gathered into the full 64-head parameters. Column
    parallelism shards the *output* dimension by rank, so concatenating the
    shards in rank order reproduces the unsharded weight exactly.

    The indexer's own Compressor is replicated on both sides, so it is copied
    rank-locally like the main one.
    """
    import torch.distributed as dist

    src, dst = attn.indexer, backend.indexer
    gathered: dict[str, torch.Tensor] = {}
    for name in ("wq_b", "weights_proj"):
        shard = getattr(src, name).weight.detach().contiguous()
        parts = [torch.empty_like(shard) for _ in range(world)]
        if world > 1:
            dist.all_gather(parts, shard)
        else:
            parts = [shard]
        gathered[name] = torch.cat(parts, dim=0)

    full_heads = gathered["weights_proj"].shape[0]
    assert full_heads == dst.n_heads, (
        f"gathered {full_heads} index heads from {world} ranks but TensorRT-LLM's "
        f"replicated indexer expects {dst.n_heads}"
    )
    with torch.no_grad():
        dst.wq_b.weight.copy_(gathered["wq_b"].to(dst.wq_b.weight.dtype))
        dst.weights_proj.weight.copy_(gathered["weights_proj"].to(dst.weights_proj.weight.dtype))
    load_source_compressor(dst, attn, compressor=src.compressor)
    return {
        "index_heads_per_rank": int(gathered["weights_proj"].shape[0] // world),
        "index_heads_total": int(full_heads),
        "collective": "torch.distributed.all_gather of wq_b/weights_proj shards",
    }


def _compressed_pool_rows(replay: _Replay, lid: int, num_rows: int) -> torch.Tensor:
    """The first ``num_rows`` compressed rows, read back through the block table.

    Read the way the sparse kernel reads them --- page ordinal then offset ---
    so a compressor that wrote to the right buffer through the wrong page
    mapping fails here instead of passing on a flat view.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import DeepseekV4AttentionType

    slot = replay.slot[lid]
    buffers = replay.cache_manager.get_buffers(slot, DeepseekV4AttentionType.COMPRESS)
    block = replay.cache_manager.compressed_block_sizes[slot]
    pages = replay.cache_manager.get_cache_indices(0, slot, DeepseekV4AttentionType.COMPRESS)
    ordinals = torch.arange(num_rows, device=buffers.device)
    page = torch.as_tensor(list(pages), device=buffers.device)[ordinals // block]
    return buffers[page.long(), (ordinals % block).long(), :]


class _SparseKernelArgs:
    """Capture the arguments the production SM90 launch actually receives.

    ``forward_sparse_attn_sm90`` resolves ``sparse_mla_dual_pool`` through its
    module globals, so replacing the name here intercepts the real call without
    changing what runs.
    """

    def __init__(self, sm90: Any):
        self._sm90 = sm90
        self._orig = sm90.sparse_mla_dual_pool
        self.captured: dict[str, Any] | None = None

    def __enter__(self) -> "_SparseKernelArgs":
        orig = self._orig

        def wrapper(q, swa_pool, compress_pool, global_indices, num_swa_indices, scale, **kw):
            self.captured = {
                "q": q.detach().clone(),
                "swa_pool": swa_pool,
                "compress_pool": compress_pool,
                "global_indices": global_indices.detach().clone(),
                "num_swa_indices": int(num_swa_indices),
                "softmax_scale": float(scale),
                "attn_sink": None if kw.get("attn_sink") is None else kw["attn_sink"].clone(),
            }
            return orig(q, swa_pool, compress_pool, global_indices, num_swa_indices, scale, **kw)

        self._sm90.sparse_mla_dual_pool = wrapper
        return self

    def __exit__(self, *exc: Any) -> None:
        self._sm90.sparse_mla_dual_pool = self._orig


def _bitwise(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    """Are these the same numbers, bit for bit?"""
    if got.shape != ref.shape or got.dtype != ref.dtype:
        return {"comparable": False, "got_shape": list(got.shape), "ref_shape": list(ref.shape)}
    gi = got.contiguous().view(torch.int16)
    ri = ref.contiguous().view(torch.int16)
    # An empty region is vacuously identical --- a ratio-0 layer has no
    # compressed slots at all --- but `max()` has no answer for it.
    diff = (got.float() - ref.float()).abs()
    return {
        "comparable": True,
        "elements": int(got.numel()),
        "differing_values": int((gi != ri).sum()),
        "bit_exact": bool(torch.equal(gi, ri)),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
    }


def _gather(cap: dict[str, Any]) -> torch.Tensor:
    """The KV rows the SM90 kernel reads, in the slot order it reads them."""
    idx = cap["global_indices"]
    num_swa = cap["num_swa_indices"]
    swa, cmp_ = cap["swa_pool"], cap["compress_pool"]
    from_swa = torch.arange(idx.shape[1], device=idx.device) < num_swa
    # A slot's position picks its pool, so an index only has to be in range for
    # that pool; clamping it against the other one's length would fault.
    rows_swa = torch.where(from_swa[None, :], idx, 0).clamp(0, swa.shape[0] - 1).long()
    if cmp_ is None:
        gathered = swa[rows_swa]
    else:
        rows_cmp = torch.where(from_swa[None, :], 0, idx).clamp(0, cmp_.shape[0] - 1).long()
        gathered = torch.where(from_swa[None, :, None], swa[rows_swa], cmp_[rows_cmp])
    return torch.where((idx >= 0)[:, :, None], gathered, torch.zeros_like(gathered))


def _sparse_attention_provenance(
    cap: dict[str, Any], call: dict[str, Any], sm90: Any, tg: Any
) -> dict[str, Any]:
    """Where does the residual attention disagreement actually live?

    ``sparse_attention`` compares two whole kernels, so a difference in it can
    be the kernel's own arithmetic or the numbers it was handed. These two
    diagnostics separate those: the first asks whether TensorRT-LLM's Q/RoPE,
    Compressor and paged-cache path reproduce the source's ``q``/``kv``/sink
    bit for bit, and the second re-runs the shipped SM90 kernel on the
    *source's own* arguments, where any remaining difference can only be the
    kernel's association.
    """
    src_q = call["q"].reshape(cap["q"].shape)
    src_kv = call["kv"][0].contiguous()
    src_idx = call["topk_idxs"][0].contiguous().to(torch.int32)
    src_sink = call["attn_sink"].contiguous()
    src_out = call["out"].reshape(cap["q"].shape).contiguous()

    idx, num_swa = cap["global_indices"], cap["num_swa_indices"]
    got_kv = _gather(cap)
    ref_kv = torch.where(
        (src_idx >= 0)[:, :, None],
        src_kv[src_idx.clamp_min(0).long()],
        torch.zeros(1, dtype=src_kv.dtype, device=src_kv.device),
    )
    # What the kernel consumes is a *slot position -> row* map, so the two are
    # compared position by position over the width they share. TensorRT-LLM's
    # table is the wider of the two (it pads the compressed region to the
    # configured top-k); the extra slots have to be padding, which is asserted
    # rather than assumed, because a live row past the source's width would be
    # an extra key the source never attended to.
    width = min(idx.shape[1], src_idx.shape[1])
    tail_live = int((idx[:, width:] >= 0).sum()) + int((src_idx[:, width:] >= 0).sum())

    isolated = sm90.sparse_mla_dual_pool(
        src_q.contiguous(),
        src_kv,
        None,
        src_idx,
        src_idx.shape[1],
        float(call["softmax_scale"]),
        src_sink,
    )
    metrics = tg.compare(isolated, src_out)

    # Third diagnostic: the *independent* pure-Torch golden on the same source
    # arguments. `kernel_on_source_inputs` localises a residual to the kernel's
    # association, but it cannot say whether any implementation could avoid it.
    # This can: the golden shares no code with either side, so when it disagrees
    # with the source at the same element, the source's own BF16 output is not a
    # reproducible target there and no candidate can be bit-exact with it.
    # `reference_ladder` makes this comparison the independent way, in a process
    # with no TensorRT-LLM import, but only for prefill; this covers the decode
    # steps that suite does not reach.
    golden = tg.sparse_attention(
        src_q.unsqueeze(0),
        src_kv.unsqueeze(0),
        src_sink,
        src_idx.unsqueeze(0),
        float(call["softmax_scale"]),
    ).reshape(src_out.shape)
    golden_metrics = tg.compare(golden, src_out)
    return {
        "input_identity": {
            "q": _bitwise(cap["q"], src_q),
            "kv_swa_region": _bitwise(got_kv[:, :num_swa], ref_kv[:, :num_swa]),
            # A ratio-4 layer selects its compressed slots through the Indexer,
            # whose contract fixes the top-k *set* --- asserted exactly by the
            # co-located ``indexer_topk`` check --- and not the order they are
            # listed in. So a difference here on a ratio-4 layer is a slot
            # ordering difference, which retiles the online softmax without
            # changing which keys are attended to; on ratio-0/128 layers, where
            # the slot rule is positional, it would be a real disagreement.
            "kv_compressed_region": _bitwise(got_kv[:, num_swa:width], ref_kv[:, num_swa:width]),
            "attn_sink": _bitwise(cap["attn_sink"].reshape(src_sink.shape), src_sink),
            "softmax_scale_equal": cap["softmax_scale"] == float(call["softmax_scale"]),
            "live_slots_beyond_shared_width": tail_live,
            "compared_slot_width": width,
        },
        "kernel_on_source_inputs": {
            k: (round(v, 9) if isinstance(v, float) else v) for k, v in metrics.items()
        },
        "independent_golden_on_source_inputs": {
            k: (round(v, 9) if isinstance(v, float) else v) for k, v in golden_metrics.items()
        },
        "trtllm_table_width": int(idx.shape[1]),
        "source_table_width": int(src_idx.shape[1]),
    }


def _record(
    out: dict[str, Any],
    name: str,
    module: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    tol: dict[str, Any],
    tg: Any,
    judge: Any,
    tolerance: Any,
    ulp_report: Any,
    ranks: Any,
    context: dict[str, Any],
) -> None:
    """Measure one check against its pre-registered tolerance and log it."""
    metrics = tg.compare(got, ref)
    limits = tolerance(tol, module)
    storage = ulp_report(got, ref)
    passed, problems = judge(metrics, limits, storage)
    out[name] = {
        "module": module,
        "metrics": {k: (round(v, 9) if isinstance(v, float) else v) for k, v in metrics.items()},
        "storage_resolution": storage,
        "tolerance": limits,
        "passed": passed,
        "problems": problems,
        "context": context,
    }
    ranks.log(
        f"  replay {name:34s} cos={metrics['cosine']:.6f} "
        f"rel_max_abs={metrics['rel_max_abs']:.3e} "
        f"{'PASS' if passed else 'FAIL ' + str(problems)}"
    )


def build_mhc(block: Any, which: str, cfg: Any) -> Any:
    """TensorRT-LLM's :class:`mHC` carrying the official layer's parameters.

    The source keeps the two hyper-connection sites as six loose parameters on
    the ``Block`` (``hc_{attn,ffn}_{fn,base,scale}``); TensorRT-LLM keeps each
    site as a module with ``fn``/``base``/``scale``. The mapping is one to one
    and the shapes are asserted rather than trusted, because ``mix_hc`` and
    ``hc_dim`` are both derived quantities and a silent transpose here would
    still Sinkhorn to something plausible.
    """
    from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

    # Constructed exactly as `DeepseekV4DecoderLayer` constructs `hc_attn` /
    # `hc_ffn`, `post_mult_value=2.0` included. That factor lives in the
    # pre-mapping's `post_mix` output, so getting it wrong leaves `layer_input`
    # correct and silently rescales everything the post-mapping produces --
    # measured as a passing mHC check next to a decoder layer off by 12.7.
    module = mHC(
        mult=cfg.hc_mult,
        hidden_size=cfg.dim,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        dtype=torch.float32,
        post_mult_value=2.0,
    ).cuda()
    fn = getattr(block, f"hc_{which}_fn")
    base = getattr(block, f"hc_{which}_base")
    scale = getattr(block, f"hc_{which}_scale")
    assert fn.shape == module.fn.shape, (
        f"official hc_{which}_fn is {tuple(fn.shape)}, TensorRT-LLM expects "
        f"{tuple(module.fn.shape)}"
    )
    with torch.no_grad():
        module.fn.copy_(fn.float())
        module.base.copy_(base.float())
        module.scale.copy_(scale.float())
    return module


def _mhc_tactics(mhc: Any, x: torch.Tensor, ref: torch.Tensor, tg: Any) -> dict[str, Any]:
    """Every tactic the SM90 autotuner may pick, measured on the real input.

    The autotuner selects on latency, so a tactic it is *allowed* to choose is
    a tactic this layer can run in production. Measuring only the one it
    happens to return leaves the rest unproven, which matters here because the
    DeepGEMM tactics run the mix GEMM in TF32 --- 10 mantissa bits against the
    FMA ladder's 23 --- and score 1.71e-01 on ``layer_input`` where every FMA
    tactic scores 1.07e-02. They are withheld below SM100 for exactly that
    reason; this check is what proves the withholding took effect.
    """
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.modules.mhc import mhc_cuda

    runner = mhc_cuda.MhcPreMappingRunner(
        n=mhc.mult,
        hidden_size=mhc.hidden_size,
        rms_eps=mhc.norm_eps,
        hc_pre_eps=mhc.eps,
        hc_sinkhorn_eps=mhc.sinkhorn_eps,
        hc_post_mult_value=mhc.post_mult_value,
        sinkhorn_repeat=mhc.sinkhorn_iters,
    )
    residual = x.reshape(-1, mhc.mult, mhc.hidden_size).contiguous()
    inputs = [
        residual.view(-1, mhc.hc_dim),
        mhc.fn.contiguous(),
        residual,
        mhc.scale.contiguous(),
        mhc.base.contiguous(),
    ]
    _, selected = AutoTuner.get().choose_one(
        "trtllm::mhc_pre_mapping",
        [runner],
        mhc_cuda.MhcPreMappingRunner.tuning_config,
        inputs,
    )
    per_tactic: dict[str, float] = {}
    worst: tuple[float, Any, dict[str, float]] | None = None
    for tactic in runner.get_valid_tactics(inputs, None):
        _, _, layer_input = runner(inputs=inputs, tactic=tactic)
        metrics = tg.compare(layer_input.reshape(ref.shape), ref)
        per_tactic[str(tactic)] = round(metrics["rel_max_abs"], 9)
        if worst is None or metrics["rel_max_abs"] > worst[0]:
            worst = (metrics["rel_max_abs"], tactic, metrics)
    assert worst is not None, "the mHC pre_mapping runner offered no valid tactic"
    return {
        "selected_tactic": str(selected),
        "worst_tactic": str(worst[1]),
        "worst_metrics": worst[2],
        "tactics_measured": len(per_tactic),
        "rel_max_abs_per_tactic": per_tactic,
        "tf32_deepgemm_tactics_offered": sorted(t for t in per_tactic if t.startswith("('dg")),
    }


def source_attention_epilogue(
    attn: Any, core: torch.Tensor, start_pos: int, num_tokens: int
) -> Any:
    """``Attention.forward``'s tail: inverse RoPE, grouped ``wo_a``, ``wo_b``.

    Taken from the official module rather than reimplemented. Loading these
    projections into TensorRT-LLM is Goal 2.1's weight-loader work, so at this
    Goal the decoder layer keeps the source's own O path and substitutes
    TensorRT-LLM only where this Goal owns the implementation -- the sparse
    attention core and, below, both hyper-connection sites.
    """
    import sys

    model = sys.modules["model"]
    rd = attn.rope_head_dim
    o = core.reshape(1, num_tokens, attn.n_local_heads, attn.head_dim).clone()
    model.apply_rotary_emb(o[..., -rd:], attn.freqs_cis[start_pos : start_pos + num_tokens], True)
    o = o.view(1, num_tokens, attn.n_local_groups, -1)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return attn.wo_b(o.flatten(2))


def _topk_agreement(got: torch.Tensor, ref: torch.Tensor | None) -> dict[str, Any]:
    """Compare two selections as *sets* per query row, which is the contract.

    ``tolerances.json`` registers ``indexer_topk`` as ``exact``: the selected
    index set must equal the source's. Position within the row is deliberately
    not compared --- both sides take a ``topk`` over scores that can tie, and
    CUDA's tie order is not part of the model's semantics --- but membership
    is, because a different slot is a different set of keys.

    The two tables can also differ in width: the source emits
    ``min(index_topk, kv_len // ratio)`` columns while the runtime table is a
    fixed ``index_topk`` wide, so ``-1`` padding is dropped before comparing.
    """
    if ref is None:
        return {"exact": False, "rows": 0, "rows_differing": 0, "reason": "no source selection"}
    g = got.reshape(got.shape[0], -1)
    r = ref.reshape(ref.shape[0], -1)
    assert g.shape[0] == r.shape[0], (
        f"selection has {g.shape[0]} query rows, source has {r.shape[0]}"
    )
    rows_differing = 0
    first_bad: dict[str, Any] | None = None
    got_sets = [set(row[row >= 0].tolist()) for row in g.cpu()]
    ref_sets = [set(row[row >= 0].tolist()) for row in r.cpu()]
    for i, (a, b) in enumerate(zip(got_sets, ref_sets)):
        if a != b:
            rows_differing += 1
            if first_bad is None:
                first_bad = {
                    "row": i,
                    "missing": sorted(b - a)[:8],
                    "extra": sorted(a - b)[:8],
                    "selected": len(a),
                    "source_selected": len(b),
                }
    return {
        "exact": rows_differing == 0,
        "rows": len(ref_sets),
        "rows_differing": rows_differing,
        "selected_slots_total": sum(len(s) for s in got_sets),
        "source_slots_total": sum(len(s) for s in ref_sets),
        "first_mismatch": first_bad,
    }


def _per_head_rms(q: torch.Tensor, heads: int, head_dim: int, eps: float) -> torch.Tensor:
    """``q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)``, source dtype.

    Written out rather than imported from either side: it is the *input* to the
    replay, so computing it with a TensorRT-LLM helper would let a bug in that
    helper cancel between the two sides.
    """
    view = q.reshape(-1, heads, head_dim)
    return (view * torch.rsqrt(view.square().mean(-1, keepdim=True) + eps)).reshape(
        -1, heads * head_dim
    )


def capture(
    src: Any, token_ids: list[int], layer_ids: tuple[int, ...], recorder_cls: Any, capture_fn: Any
) -> dict[str, Any]:
    """Run the official model over one prompt and one decode step, recording.

    Returns per-phase, per-layer: the pre-RoPE Q and latent the source's sparse
    kernel is about to consume, the kernel's own arguments and output, and the
    compressed-slot half of the source's index table.
    """
    phases: dict[str, Any] = {}
    src.reset_cache()
    for phase, (tokens, start_pos) in (
        ("prefill", (token_ids, 0)),
        ("decode", ([token_ids[-1]], len(token_ids))),
    ):
        store: dict[str, Any] = {}
        handles = []
        for lid in layer_ids:
            # `Block.forward(x, start_pos, input_ids)` -- the hook records the
            # [b, s, hc, d] residual stream entering the layer and the one
            # leaving it, which is what a decoder-layer replay has to reproduce.
            handles.append(capture_fn(src.model.layers[lid], store, f"l{lid}.block"))
            attn = src.model.layers[lid].attn
            handles.append(capture_fn(attn.wq_b, store, f"l{lid}.wq_b"))
            handles.append(capture_fn(attn.kv_norm, store, f"l{lid}.kv_norm"))
            if attn.compress_ratio:
                # `Compressor.forward(x, start_pos)` -- the hook records `x`,
                # which is the same hidden state the whole attention block
                # consumes, and the compressed rows it returns.
                handles.append(capture_fn(attn.compressor, store, f"l{lid}.compressor"))
            if getattr(attn, "indexer", None) is not None:
                # `qr = self.q_norm(self.wq_a(x))` is the indexer's own query
                # input, and `Indexer.forward` returns the selected slots the
                # source's attention then uses.
                handles.append(capture_fn(attn.q_norm, store, f"l{lid}.q_norm"))
                handles.append(capture_fn(attn.indexer, store, f"l{lid}.indexer"))
        toks = torch.tensor([tokens], dtype=torch.long, device="cuda")
        recorder = recorder_cls(tuple(layer_ids), len(src.model.layers))
        with recorder, torch.inference_mode():
            src.model.forward(toks, start_pos)
        torch.cuda.synchronize()
        for h in handles:
            h.remove()
        phases[phase] = {
            "start_pos": start_pos,
            "num_tokens": len(tokens),
            "store": store,
            "calls": {lid: recorder.record(lid) for lid in layer_ids},
        }
    return phases


def _source_compressed_indices(
    call: dict[str, Any], window: int, offset: int, width: int
) -> torch.Tensor | None:
    """The compressed half of the source's index table, in local slot numbers.

    ``Attention.forward`` concatenates a ``window_size``-wide sliding-window
    block with the compressed block and shifts the latter by ``offset`` --- the
    number of window rows it prepended to ``kv`` --- so subtracting that offset
    recovers the compressed-slot ordinal TensorRT-LLM's ``topk_indices``
    contract uses. ``-1`` padding is preserved.

    The source's ratio-4 block is ``min(index_topk, kv_len // ratio)`` wide, so
    on a prompt shorter than ``index_topk * ratio`` it is *narrower* than the
    fixed ``index_topk`` table the indexer hands the backend. Padding to that
    width with ``-1`` is the same statement --- no more slots are selectable ---
    in the layout the backend reads.
    """
    idxs = call["topk_idxs"]
    if idxs.shape[-1] <= window:
        return None
    cmp = idxs[..., window:].reshape(idxs.shape[-2], -1)
    cmp = torch.where(cmp < 0, cmp.new_full((), -1), cmp - offset).int()
    if cmp.shape[-1] < width:
        pad = cmp.new_full((cmp.shape[0], width - cmp.shape[-1]), -1)
        cmp = torch.cat([cmp, pad], dim=-1)
    return cmp.contiguous()


def _mhc_site(
    out: dict[str, Any],
    stem: str,
    site: str,
    block: Any,
    mhc: Any,
    x: torch.Tensor,
    ctx: dict[str, Any],
    tol: dict[str, Any],
    tg: Any,
    judge: Any,
    tolerance: Any,
    ulp_report: Any,
    ranks: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One hyper-connection site's ``pre_mapping``, against the source's.

    ``pre_mapping`` returns three things and the decoder layer consumes all
    three, so all three are compared: ``layer_input`` feeds the next norm,
    ``post_mix`` and ``comb_mix`` feed ``post_mapping``. A single check on
    ``layer_input`` would have left the two mix matrices ungated, and it is
    ``post_mix`` that carries the ``post_mult_value`` factor --- the one that
    was silently wrong for a whole iteration.

    Returns TensorRT-LLM's three outputs followed by the source's ``post`` and
    ``comb``, which the caller needs for the identical-operand ``post_mapping``
    check once it has that site's layer output.
    """
    fn = getattr(block, f"hc_{site}_fn")
    base = getattr(block, f"hc_{site}_base")
    scale = getattr(block, f"hc_{site}_scale")
    ref_input, ref_post, ref_comb = block.hc_pre(x, fn, scale, base)
    post_mix, comb_mix, layer_input = mhc.pre_mapping(x)

    for what, got, ref in (
        ("layer_input", layer_input.reshape(ref_input.shape), ref_input),
        ("post_mix", post_mix.reshape(ref_post.shape), ref_post),
        ("comb_mix", comb_mix.reshape(ref_comb.shape), ref_comb),
    ):
        _record(
            out,
            f"{stem}.mhc_{site}_{what}",
            "mhc",
            got,
            ref,
            tol,
            tg,
            judge,
            tolerance,
            ulp_report,
            ranks,
            {**ctx, "site": f"{site} {what}"},
        )

    sweep = _mhc_tactics(mhc, x, ref_input, tg)
    limits = tolerance(tol, "mhc")
    passed, problems = judge(sweep["worst_metrics"], limits)
    name = f"{stem}.mhc_{site}_tactics"
    out[name] = {
        "module": "mhc",
        "metrics": {
            k: (round(v, 9) if isinstance(v, float) else v)
            for k, v in sweep.pop("worst_metrics").items()
        },
        "tolerance": limits,
        "passed": passed,
        "problems": problems,
        "context": {**ctx, "site": f"{site} pre_mapping tactic sweep", **sweep},
    }
    ranks.log(
        f"  replay {name:38s} tactics={sweep['tactics_measured']} "
        f"worst={sweep['worst_tactic']} {'PASS' if passed else 'FAIL ' + str(problems)}"
    )
    return post_mix, comb_mix, layer_input, ref_post, ref_comb


def _post_mapping_check(
    out: dict[str, Any],
    stem: str,
    site: str,
    block: Any,
    mhc: Any,
    layer_out: torch.Tensor,
    residual: torch.Tensor,
    ref_post: torch.Tensor,
    ref_comb: torch.Tensor,
    ctx: dict[str, Any],
    tol: dict[str, Any],
    tg: Any,
    judge: Any,
    tolerance: Any,
    ulp_report: Any,
    ranks: Any,
) -> None:
    """``post_mapping`` against ``hc_post`` on *identical* operands.

    Both sides get the source's own ``post``/``comb`` and the same layer
    output, so any difference is this kernel's --- association order in
    ``post*x + sum_k comb[k]*residual[k]`` and the single BF16 store --- rather
    than one inherited from ``pre_mapping``. Without holding the operands fixed
    the two effects are indistinguishable in the composed result.

    Because the operands are identical the expression has one *exact* value, so
    the check also records which side's BF16 store is the correctly rounded one
    wherever they disagree. That turns "these two differ by a last bit" into a
    statement about which implementation is right, which a relative-error
    metric on its own cannot make.
    """
    got = mhc.post_mapping(layer_out, residual, ref_post, ref_comb)
    ref = block.hc_post(layer_out, residual, ref_post, ref_comb)
    exact = (
        ref_post.double().unsqueeze(-1) * layer_out.double().unsqueeze(-2)
        + (ref_comb.double().unsqueeze(-1) * residual.double().unsqueeze(-2)).sum(dim=-3)
    ).to(got.dtype)
    disagree = got != ref
    _record(
        out,
        f"{stem}.mhc_{site}_post_mapping",
        "mhc",
        got,
        ref,
        tol,
        tg,
        judge,
        tolerance,
        ulp_report,
        ranks,
        {
            **ctx,
            "site": f"{site} post_mapping",
            "operands": "source post/comb, identical",
            "exactness": {
                "elements": int(disagree.numel()),
                "elements_disagreeing": int(disagree.sum()),
                "trtllm_is_correctly_rounded": int((disagree & (got == exact)).sum()),
                "source_is_correctly_rounded": int((disagree & (ref == exact)).sum()),
                "neither_is_correctly_rounded": int(
                    (disagree & (got != exact) & (ref != exact)).sum()
                ),
                "reference": "float64 evaluation of hc_post on the same operands",
            },
        },
    )


def decoder_layer(
    src: Any,
    cfg_cls: Any,
    prompt: dict[str, Any],
    phases: dict[str, Any],
    lid: int,
    local_heads: int,
    eps: float,
    out: dict[str, Any],
    tol: dict[str, Any],
    tg: Any,
    ranks: Any,
    judge: Any,
    tolerance: Any,
    ulp_report: Any,
) -> None:
    """One complete decoder layer, wired end to end through TensorRT-LLM.

    The point of this replay is that nothing downstream is fed from the
    source's copy of an upstream quantity. TensorRT-LLM's ``pre_mapping``
    produces ``layer_input``; the source's own norm and projections consume
    *that*; TensorRT-LLM's Compressor, Indexer and sparse attention consume
    *those*; the O path consumes the attention core; both ``post_mapping``
    sites consume what the stage before them produced. An error anywhere in the
    chain therefore reaches the measured output instead of being absorbed by a
    source-derived operand.

    It runs in its own ``_Replay``, so no compressed row or KV entry written
    while the module-level checks were being fed source activations is visible
    here --- prefill fills this cache from the connected inputs and decode
    reads back what prefill wrote.

    Reference-owned, and named as such in the artifact: the Q/KV projections
    and norms, the inverse-RoPE + grouped O-LoRA epilogue (loading those is
    Goal 2.1) and the MoE (production MoE is Goal 2.2). Everything this Goal
    owns is TensorRT-LLM's.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import module as v4_module

    cfg = src.args
    block = src.model.layers[lid]
    attn = block.attn
    ratio = cfg.compress_ratios[lid]
    seq_len = len(prompt["token_ids"])
    head_dim = cfg.head_dim
    mhc_attn = build_mhc(block, "attn", cfg)
    mhc_ffn = build_mhc(block, "ffn", cfg)

    replay = _Replay(cfg_cls, (lid,), max_seq_len=max(1024, seq_len + 128), local_heads=local_heads)
    try:
        request = _open_request(replay, seq_len)
        mla = replay.mla[lid]
        mla.mqa.attn_sink = torch.nn.Parameter(attn.attn_sink.detach().clone(), requires_grad=False)
        load_source_compressor(mla.mqa, attn)
        indexer_report = load_source_indexer(mla.mqa, attn, ranks.world)

        for phase in ("prefill", "decode"):
            info = phases[phase]
            num_tokens = info["num_tokens"]
            start_pos = info["start_pos"]
            if phase == "prefill":
                metadata = _metadata(replay, seq_len, 0, num_contexts=1)
            else:
                _advance_to_generation(replay, request, seq_len)
                metadata = _metadata(replay, 1, seq_len, num_contexts=0)

            x_in = info["store"][f"l{lid}.block"]["inputs"][0]
            block_ref = info["store"][f"l{lid}.block"]["output"]
            stem = f"layer{lid}.{phase}"
            ctx = {
                "layer": lid,
                "ratio": ratio,
                "phase": phase,
                "prompt": prompt["id"],
                "num_tokens": num_tokens,
                "hc_mult": cfg.hc_mult,
                "sinkhorn_iters": cfg.hc_sinkhorn_iters,
                "post_mult_value": mhc_attn.post_mult_value,
            }

            with torch.no_grad():
                # --- attention-site hyper-connection ----------------------
                post_mix, comb_mix, layer_input, ref_post1, ref_comb1 = _mhc_site(
                    out,
                    stem,
                    "attn",
                    block,
                    mhc_attn,
                    x_in,
                    ctx,
                    tol,
                    tg,
                    judge,
                    tolerance,
                    ulp_report,
                    ranks,
                )

                # --- the source's norm and projections, on that input -----
                hidden = block.attn_norm(layer_input).reshape(num_tokens, cfg.dim).contiguous()
                qr = attn.q_norm(attn.wq_a(hidden)).contiguous()
                q = _per_head_rms(
                    attn.wq_b(qr).reshape(num_tokens, -1), local_heads, head_dim, eps
                ).contiguous()
                latent = attn.kv_norm(attn.wkv(hidden)).reshape(num_tokens, head_dim).contiguous()

                # --- TensorRT-LLM Compressor, Indexer, sparse attention ---
                mla.mqa.compressor(hidden, metadata)
                positions = torch.arange(
                    start_pos, start_pos + num_tokens, dtype=torch.int32, device=qr.device
                )
                topk = mla.mqa.indexer(qr, hidden, metadata, positions)
                core = torch.empty(
                    num_tokens, local_heads * head_dim, dtype=q.dtype, device=q.device
                )
                production_entry = (
                    v4_module.forward_context_sparse_attn
                    if phase == "prefill"
                    else v4_module.forward_generation_sparse_attn
                )
                production_entry(
                    mla, q, None, None, metadata, core, latent_cache=latent, topk_indices=topk
                )

                # --- O path, FFN-site hyper-connection, reference MoE -----
                attn_out = source_attention_epilogue(
                    attn, core.reshape(num_tokens, local_heads, head_dim), start_pos, num_tokens
                )
                _post_mapping_check(
                    out,
                    stem,
                    "attn",
                    block,
                    mhc_attn,
                    attn_out,
                    x_in,
                    ref_post1,
                    ref_comb1,
                    ctx,
                    tol,
                    tg,
                    judge,
                    tolerance,
                    ulp_report,
                    ranks,
                )
                x_mid = mhc_attn.post_mapping(attn_out, x_in, post_mix, comb_mix)

                post2, comb2, layer_input2, ref_post2, ref_comb2 = _mhc_site(
                    out,
                    stem,
                    "ffn",
                    block,
                    mhc_ffn,
                    x_mid,
                    ctx,
                    tol,
                    tg,
                    judge,
                    tolerance,
                    ulp_report,
                    ranks,
                )
                ids = torch.tensor(
                    [prompt["token_ids"][-num_tokens:]], dtype=torch.long, device=x_mid.device
                )
                moe_out = block.ffn(block.ffn_norm(layer_input2), ids)
                _post_mapping_check(
                    out,
                    stem,
                    "ffn",
                    block,
                    mhc_ffn,
                    moe_out,
                    x_mid,
                    ref_post2,
                    ref_comb2,
                    ctx,
                    tol,
                    tg,
                    judge,
                    tolerance,
                    ulp_report,
                    ranks,
                )
                x_out = mhc_ffn.post_mapping(moe_out, x_mid, post2, comb2)

            _record(
                out,
                f"{stem}.decoder_layer",
                "decoder_layer",
                x_out,
                block_ref,
                tol,
                tg,
                judge,
                tolerance,
                ulp_report,
                ranks,
                {
                    **ctx,
                    "dataflow": (
                        "trtllm mHC pre -> source attn_norm/wq_a/q_norm/wq_b/wkv/kv_norm -> "
                        "trtllm Compressor -> trtllm Indexer -> trtllm sparse attention -> "
                        "source inverse-RoPE + O-LoRA -> trtllm mHC post -> trtllm mHC pre -> "
                        "source ffn_norm -> reference MoE -> trtllm mHC post"
                    ),
                    "independent_cache": "own _Replay; no row written from source activations",
                    "reference_owned": [
                        "Q/KV projections and norms",
                        "inverse RoPE + grouped O-LoRA (Goal 2.1 loader)",
                        "MoE router and experts (plan-allowed reference path,"
                        " production MoE is Goal 2.2)",
                    ],
                    "compressed_slots": None if topk is None else int(topk.shape[-1]),
                    **indexer_report,
                },
            )
    finally:
        replay.shutdown()


def _compressor_stages(
    attn: Any,
    backend: Any,
    hidden: torch.Tensor,
    kv_comp: torch.Tensor | None,
    ref_rows: torch.Tensor,
    eps: float,
    tg: Any,
    phase: str,
) -> dict[str, Any]:
    """Bisect the compressor chain: pooling, then postprocess-and-quantise.

    The whole-chain metric says a compressed row disagrees with the source; it
    cannot say which of the four roundings between the projection and the
    paged store moved. This splits the chain at the one boundary TensorRT-LLM
    exposes -- ``Compressor.forward`` returns the pooled ``kv_comp`` before
    postprocessing -- and drives the *second* half from TensorRT-LLM's own
    pooled rows, so each half is attributed independently:

    ``pooling``
        TensorRT-LLM's pooled BF16 rows against the independent golden's.
    ``pooling_from_fused_projection`` / ``pooling_from_split_projection``
        the same golden reduction, driven once from the single fused
        ``wkv_gate`` GEMM TensorRT-LLM issues and once from the two separate
        ``wkv``/``wgate`` GEMMs the source issues, both against TensorRT-LLM's
        pooled rows. These separate "the reduction rounds differently" from
        "the projection rounds differently because the GEMM has a different
        shape", which need different fixes.
    ``postprocess_and_quant``
        the golden's norm/RoPE/FP8 chain applied to TensorRT-LLM's pooled rows,
        against the source's cached rows. Bit-exact here with a failing
        whole-chain metric means the deviation is in TensorRT-LLM's own
        ``_source_postprocess``/``write_source_compressed_rows``.
    ``golden_end_to_end``
        the control: the golden's whole chain against the source's rows. This
        is the number that says whether *any* Torch reimplementation can hit
        the source at this layer, which is what separates "TensorRT-LLM is
        wrong" from "the reference is not exact at this depth".
    """
    ratio = attn.compress_ratio
    if phase != "prefill" or kv_comp is None or kv_comp.shape[0] == 0:
        # Decode compresses only every `ratio`-th step; when this step did not
        # produce a row the comparison above is re-reading prefill's rows and
        # there is no new arithmetic to attribute.
        return {"skipped": f"{phase} produced no new compressed row"}

    rd = attn.rope_head_dim
    x_in = hidden.reshape(1, hidden.shape[0], -1)
    stages: dict[str, torch.Tensor] = {}
    golden = tg.compressor_prefill(
        x_in,
        attn.compressor.wkv.weight,
        attn.compressor.wgate.weight,
        attn.compressor.ape,
        attn.compressor.norm.weight,
        attn.compressor.freqs_cis,
        ratio=ratio,
        head_dim=attn.head_dim,
        rope_dim=rd,
        eps=eps,
        rotate=False,
    )[0]
    tg.compressor_prefill(
        x_in,
        attn.compressor.wkv.weight,
        attn.compressor.wgate.weight,
        attn.compressor.ape,
        attn.compressor.norm.weight,
        attn.compressor.freqs_cis,
        ratio=ratio,
        head_dim=attn.head_dim,
        rope_dim=rd,
        eps=eps,
        rotate=False,
        stages=stages,
    )
    rows = min(kv_comp.shape[0], golden.shape[0], ref_rows.shape[0])

    # Second half of the chain, driven from TensorRT-LLM's own pooled rows.
    hybrid = tg.rms_norm(kv_comp[:rows], attn.compressor.norm.weight, eps)
    rope_part = tg.apply_rope(
        hybrid[..., -rd:].unsqueeze(0), attn.compressor.freqs_cis[: rows * ratio : ratio]
    )[0]
    hybrid = torch.cat([hybrid[..., :-rd], rope_part], dim=-1)
    hybrid = torch.cat([tg.fp8_quant_dequant(hybrid[..., :-rd], 64), hybrid[..., -rd:]], dim=-1)

    def report(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
        metrics = tg.compare(got, ref)
        differing = int((got.float() != ref.float()).sum())
        return {
            **{k: (round(v, 9) if isinstance(v, float) else v) for k, v in metrics.items()},
            "elements": int(ref.numel()),
            "elements_differing": differing,
            "bit_exact": differing == 0,
        }

    # Which GEMM shape produced the pooled rows? TensorRT-LLM issues one fused
    # [2 * state_dim, dim] projection; the source issues two [state_dim, dim]
    # ones. Same maths, different cuBLAS accumulation, and the FP8 quantiser
    # downstream turns a last-bit difference into a whole level.
    fused_w = torch.cat([attn.compressor.wkv.weight, attn.compressor.wgate.weight], dim=0)
    cutoff = (rows * ratio) if ratio == 4 else x_in.shape[1] - x_in.shape[1] % ratio
    xf = x_in.float()
    state_dim = attn.compressor.wkv.weight.shape[0]
    fused = F.linear(xf, fused_w.float())[:, :cutoff]
    split = (
        F.linear(xf, attn.compressor.wkv.weight.float())[:, :cutoff],
        F.linear(xf, attn.compressor.wgate.weight.float())[:, :cutoff],
    )
    # The projection as `Compressor.forward` itself issues it: 2-D rows, and
    # the two halves as slices of the fused parameter rather than as separate
    # tensors. Compared against the source's own form so a residual pooling
    # difference can be attributed to the GEMM or to the reduction, not left
    # ambiguous between them.
    own_w = backend.compressor.wkv_gate.weight.float()
    own_x = hidden.float()
    own = (
        F.linear(own_x, own_w[:state_dim])[:cutoff],
        F.linear(own_x, own_w[state_dim:])[:cutoff],
    )

    def pooled_from(kv: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        return tg.compressor_pool(
            kv, score, attn.compressor.ape, ratio=ratio, head_dim=attn.head_dim
        )[0, :rows].to(kv_comp.dtype)

    return {
        "pooling": report(kv_comp[:rows], stages["pooled"][0, :rows]),
        "projection_kv": report(own[0], split[0][0]),
        "projection_gate": report(own[1], split[1][0]),
        "pooling_from_fused_projection": report(
            kv_comp[:rows], pooled_from(fused[..., :state_dim], fused[..., state_dim:])
        ),
        "pooling_from_split_projection": report(kv_comp[:rows], pooled_from(*split)),
        "pooling_from_own_projection": report(
            kv_comp[:rows], pooled_from(own[0].unsqueeze(0), own[1].unsqueeze(0))
        ),
        "postprocess_and_quant": report(hybrid, ref_rows[:rows]),
        "golden_end_to_end": report(golden[:rows], ref_rows[:rows]),
        "compressed_rows_compared": rows,
    }


def judge_dispatch(counts: dict[str, int], expected: int) -> dict[str, Any]:
    """`real_runtime` verdict for one replay run.

    The liveness clause reads ``append_rows_dropped``, which is what
    :func:`sm90.dispatch_counts` actually publishes. An earlier version asked
    for ``dropped_rows`` through ``.get(..., 0)``; that key has never existed,
    so the clause was vacuously true and a run that silently dropped
    compressed rows would still have been reported as live SM90 dispatch.
    Every key is indexed rather than defaulted for the same reason: a rename
    on the producing side must fail here, loudly, instead of going quiet.
    """
    dropped = counts["append_rows_dropped"]
    problems = []
    if counts["context"] != expected:
        problems.append(f"context dispatches {counts['context']} != {expected}")
    if counts["generation"] != expected:
        problems.append(f"generation dispatches {counts['generation']} != {expected}")
    if dropped != 0:
        problems.append(f"append_rows_dropped {dropped} != 0")
    return {
        "counts": counts,
        "expected_context": expected,
        "expected_generation": expected,
        "passed": not problems,
        "problems": problems,
    }


def run(
    src: Any,
    prompt: dict[str, Any],
    tol: dict[str, Any],
    ranks: Any,
    tg: Any,
    recorder_cls: Any,
    capture_fn: Any,
    judge: Any,
    tolerance: Any,
    ulp_report: Any,
    layer_ids: tuple[int, ...] = (0, 2, 3),
) -> dict[str, Any]:
    """Replay one layer of each compression schedule for prefill and decode.

    The registered gate names layers 0 / 2 / 3 --- one ratio-0, one ratio-4 and
    one ratio-128 layer --- and that stays the default. ``layer_ids`` exists so
    the same replay can be pointed at *deep* layers of the same kinds: the
    checkpoint repeats the schedule down 43 layers, and a per-layer contract
    proven only at the top of the stack says nothing about the bottom of it.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import module as v4_module
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import sm90

    cfg = src.args
    layer_ids = tuple(layer_ids)
    # The ratio-4 layer is the one that exercises every V4 operator at once ---
    # Compressor, Indexer and the dual-pool attention --- so it is the layer the
    # complete decoder-layer replay runs on.
    decoder_layer_id = next(
        (lid for lid in layer_ids if cfg.compress_ratios[lid] == 4), layer_ids[-1]
    )
    token_ids = prompt["token_ids"]
    seq_len = len(token_ids)
    local_heads = src.model.layers[0].attn.n_local_heads
    window = cfg.window_size
    eps = cfg.norm_eps

    ranks.log(f"[activation_replay] capturing official activations, {seq_len} tokens + 1 decode")
    phases = capture(src, token_ids, layer_ids, recorder_cls, capture_fn)

    class _Cfg:
        """The checkpoint dimensions the replay needs, from the official args."""

        v_head_dim = cfg.head_dim
        qk_rope_head_dim = cfg.rope_head_dim
        qk_nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        q_lora_rank = cfg.q_lora_rank
        kv_lora_rank = cfg.head_dim - cfg.rope_head_dim
        hidden_size = cfg.dim
        index_n_heads = cfg.index_n_heads
        index_head_dim = cfg.index_head_dim
        index_topk = cfg.index_topk
        window_size = cfg.window_size
        vocab_size = cfg.vocab_size
        compress_ratios = list(cfg.compress_ratios)
        rope_theta = cfg.rope_theta
        compress_rope_theta = cfg.compress_rope_theta
        rope_scaling = {
            "factor": cfg.rope_factor,
            "original_max_position_embeddings": cfg.original_seq_len,
            "beta_fast": cfg.beta_fast,
            "beta_slow": cfg.beta_slow,
        }

    sm90.reset_dispatch_counts()
    out: dict[str, Any] = {}
    with _host_default_device():
        replay = _Replay(
            _Cfg, layer_ids, max_seq_len=max(1024, seq_len + 128), local_heads=local_heads
        )
        try:
            request = _open_request(replay, seq_len)
            for phase in ("prefill", "decode"):
                info = phases[phase]
                if phase == "prefill":
                    metadata = _metadata(replay, seq_len, 0, num_contexts=1)
                else:
                    _advance_to_generation(replay, request, seq_len)
                    metadata = _metadata(replay, 1, seq_len, num_contexts=0)
                num_tokens = info["num_tokens"]
                for lid in layer_ids:
                    ratio = cfg.compress_ratios[lid]
                    attn = src.model.layers[lid].attn
                    mla = replay.mla[lid]
                    mla.mqa.attn_sink = torch.nn.Parameter(
                        attn.attn_sink.detach().clone(), requires_grad=False
                    )
                    if ratio > 1:
                        load_source_compressor(mla.mqa, attn)
                    if ratio == 4:
                        indexer_report = load_source_indexer(mla.mqa, attn, ranks.world)

                    q = _per_head_rms(
                        info["store"][f"l{lid}.wq_b"]["output"].reshape(num_tokens, -1),
                        local_heads,
                        replay.head_dim,
                        eps,
                    ).contiguous()
                    latent = (
                        info["store"][f"l{lid}.kv_norm"]["output"]
                        .reshape(num_tokens, replay.head_dim)
                        .contiguous()
                    )
                    call = info["calls"][lid]
                    offset = seq_len if phase == "prefill" else window
                    source_topk = _source_compressed_indices(
                        # Only the learned ratio-4 table is fixed-width; the
                        # ratio-128 rule emits exactly the valid slots.
                        call,
                        window,
                        offset,
                        cfg.index_topk if ratio == 4 else 0,
                    )
                    topk = source_topk
                    kv_len = seq_len if phase == "prefill" else seq_len + 1
                    rows = 0 if ratio <= 1 else kv_len // ratio
                    hidden = (
                        info["store"][f"l{lid}.compressor"]["inputs"][0].reshape(num_tokens, -1)
                        if ratio > 1
                        else None
                    )
                    if ratio > 1:
                        # The compressed rows are produced by TensorRT-LLM's own
                        # Compressor from the same hidden states the source's
                        # consumed, not copied out of the source's cache: the
                        # attention that follows then reads rows this
                        # implementation wrote, through the block table it built.
                        kv_comp, _ = mla.mqa.compressor(hidden, metadata)
                        got_rows = _compressed_pool_rows(replay, lid, rows)
                        ref_rows = attn.kv_cache[0, window : window + rows]
                        name = f"layer{lid}.{phase}.compressor"
                        metrics = tg.compare(got_rows, ref_rows)
                        limits = tolerance(tol, "compressor")
                        passed, problems = judge(metrics, limits)
                        out[name] = {
                            "module": "compressor",
                            "stage_bisection": _compressor_stages(
                                attn, mla.mqa, hidden, kv_comp, ref_rows, eps, tg, phase
                            ),
                            "metrics": {
                                k: (round(v, 9) if isinstance(v, float) else v)
                                for k, v in metrics.items()
                            },
                            "storage_resolution": ulp_report(got_rows, ref_rows),
                            "tolerance": limits,
                            "passed": passed,
                            "problems": problems,
                            "context": {
                                "layer": lid,
                                "ratio": ratio,
                                "phase": phase,
                                "prompt": prompt["id"],
                                "compressed_rows": rows,
                                "source_produced_this_step": (
                                    info["store"][f"l{lid}.compressor"]["output"] is not None
                                ),
                                "kv_len": kv_len,
                            },
                        }
                        ranks.log(
                            f"  replay {name:34s} cos={metrics['cosine']:.6f} "
                            f"rel_max_abs={metrics['rel_max_abs']:.3e} "
                            f"{'PASS' if passed else 'FAIL ' + str(problems)}"
                        )

                    if ratio == 4:
                        qr = (
                            info["store"][f"l{lid}.q_norm"]["output"]
                            .reshape(num_tokens, -1)
                            .contiguous()
                        )
                        positions = torch.arange(
                            info["start_pos"],
                            info["start_pos"] + num_tokens,
                            dtype=torch.int32,
                            device=qr.device,
                        )
                        got_topk = mla.mqa.indexer(qr, hidden.contiguous(), metadata, positions)
                        name = f"layer{lid}.{phase}.indexer_topk"
                        agreement = _topk_agreement(got_topk, source_topk)
                        out[name] = {
                            "module": "indexer_topk",
                            "metrics": agreement,
                            "tolerance": tolerance(tol, "indexer_topk"),
                            "passed": agreement["exact"],
                            "problems": []
                            if agreement["exact"]
                            else [
                                f"{agreement['rows_differing']} of {agreement['rows']} query rows "
                                f"selected a different compressed-slot set"
                            ],
                            "context": {
                                "layer": lid,
                                "ratio": ratio,
                                "phase": phase,
                                "prompt": prompt["id"],
                                "index_topk": cfg.index_topk,
                                "selectable_slots": rows,
                                **indexer_report,
                            },
                        }
                        ranks.log(
                            f"  replay {name:34s} rows={agreement['rows']} "
                            f"differing={agreement['rows_differing']} "
                            f"{'PASS' if agreement['exact'] else 'FAIL'}"
                        )
                        # From here the attention reads TensorRT-LLM's own
                        # selection, so nothing source-derived remains in the
                        # sparse path.
                        if agreement["exact"]:
                            topk = got_topk

                    output = torch.empty(
                        num_tokens, local_heads * replay.head_dim, dtype=q.dtype, device=q.device
                    )
                    # The *production* entry points, not `forward_sparse_attn_sm90`
                    # directly: these are the functions that choose between the
                    # Blackwell fused op and the SM90 branch, so calling them is
                    # what proves the architecture dispatch selected SM90 rather
                    # than the harness having selected it.
                    production_entry = (
                        v4_module.forward_context_sparse_attn
                        if phase == "prefill"
                        else v4_module.forward_generation_sparse_attn
                    )
                    with _SparseKernelArgs(sm90) as launch:
                        production_entry(
                            mla,
                            q,
                            None,
                            None,
                            metadata,
                            output,
                            latent_cache=latent,
                            topk_indices=topk,
                        )
                    got = output.reshape(num_tokens, local_heads, replay.head_dim)
                    ref = call["out"].reshape(num_tokens, local_heads, replay.head_dim)
                    provenance = _sparse_attention_provenance(launch.captured, call, sm90, tg)

                    name = f"layer{lid}.{phase}.sparse_attention"
                    metrics = tg.compare(got, ref)
                    limits = tolerance(tol, "sparse_attention_output")
                    storage = ulp_report(got, ref)
                    passed, problems = judge(metrics, limits, storage)
                    out[name] = {
                        "module": "sparse_attention_output",
                        "metrics": {
                            k: (round(v, 9) if isinstance(v, float) else v)
                            for k, v in metrics.items()
                        },
                        "storage_resolution": storage,
                        "tolerance": limits,
                        "passed": passed,
                        "problems": problems,
                        "context": {
                            "layer": lid,
                            "ratio": ratio,
                            "phase": phase,
                            "prompt": prompt["id"],
                            "num_tokens": num_tokens,
                            "start_pos": info["start_pos"],
                            "local_heads": local_heads,
                            "head_dim": replay.head_dim,
                            "window": window,
                            "index_topk": cfg.index_topk if ratio == 4 else None,
                            "compressed_slots": None if topk is None else int(topk.shape[-1]),
                            "compressed_rows_from_trtllm_compressor": rows,
                            **provenance,
                        },
                    }
                    ranks.log(
                        f"  replay {name:34s} cos={metrics['cosine']:.6f} "
                        f"rel_max_abs={metrics['rel_max_abs']:.3e} "
                        f"{'PASS' if passed else 'FAIL ' + str(problems)}"
                    )

        finally:
            replay.shutdown()

    # The complete decoder layer runs in its own reset replay, driven end to
    # end from TensorRT-LLM's own mHC output --- see `decoder_layer`.
    with _host_default_device():
        decoder_layer(
            src,
            _Cfg,
            prompt,
            phases,
            decoder_layer_id,
            local_heads,
            eps,
            out,
            tol,
            tg,
            ranks,
            judge,
            tolerance,
            ulp_report,
        )

    # Three layers through the module replay plus the decoder layer's own.
    dispatch = judge_dispatch(sm90.dispatch_counts(), len(layer_ids) + 1)
    return {"checks": out, "real_runtime": dispatch}
