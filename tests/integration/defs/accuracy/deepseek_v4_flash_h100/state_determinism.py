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
"""Does a request's result depend on which requests ran before it?

Iteration 38 measured the same prompt, the same weights, the same source
reference and the same hard configuration through the real LLM API twice and
got two different answers: ``chat_geography`` scored rel 0.637 when it was the
first prompt of the run and rel 0.828 when ``chat_arithmetic`` had run before
it. ``full_model.measure`` issues every prompt as its own one-item
``llm.generate``, so batching and the token budget cannot explain that. Some
state a request owns is outliving it.

This module answers *which* state, and it does so without the executor. Every
channel the executor shares between requests is present here and can be turned
off one at a time:

``cache``
    one :class:`DeepseekV4CacheManager` across the whole sequence, so pages a
    finished request owned are handed to the next one --- exactly what the
    executor does, and the only channel that carries *content*.
``metadata``
    one ``DeepseekV4TrtllmAttentionMetadata`` whose fields are reassigned per
    request and re-``prepare()``d, which is how ``model_engine`` drives it. Its
    device buffers are sized for the batch capacity and only the leading rows
    are rewritten, so a stale tail is a real possibility rather than a
    hypothetical one.
``zero_freed``
    not a channel: a probe. With the shared cache *and* every page a request
    owned memset at ``free_resources`` time, a difference that survives is not
    about page content, and a difference that disappears is exactly about it.

Running the same scripted sequence under each combination turns "something
leaks" into a table with one row per channel. The control --- a fresh cache
manager and fresh metadata per request --- is the same thing
``layer_localization`` measures, and it is here so that "the control is
reproducible" is a measured fact of *this* run rather than an assumption
carried over from another suite.

The comparison is TensorRT-LLM against itself. No source reference is needed
to prove that one implementation gave two answers to one question, and not
needing it keeps this suite to a single model load.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from typing import Any, Iterable, Sequence

import torch

#: The scripted history. Read it as: three identical requests to establish what
#: "no history change" looks like, then a different prompt, then the same
#: request again. ``chat_geography`` recurs at five distinct points because it
#: is the prompt whose executor-side disagreement started this; the two heavier
#: prompts recur so the sequence also covers a prefill that spans many pages
#: (``long_prefill_2304``) and one that lands exactly on a page boundary
#: (``cache_boundary_257``).
#:
#: Position matters and is the whole point: run 0 sees an untouched pool, runs
#: 1 and 2 see pages their own prompt just wrote, and runs 5, 7 and 10 see
#: pages a *different* prompt wrote. If runs 1 and 2 agree with each other but
#: 5, 7 and 10 do not agree with them, the difference is the content of the
#: recycled pages and not kernel nondeterminism.
DEFAULT_SEQUENCE: tuple[str, ...] = (
    "chat_geography",
    "chat_geography",
    "chat_geography",
    "chat_arithmetic",
    "chat_arithmetic",
    "chat_geography",
    "long_prefill_2304",
    "chat_geography",
    "cache_boundary_257",
    "cache_boundary_257",
    "chat_geography",
    "long_prefill_2304",
    "long_prefill_2304",
)

#: One mode per hypothesis, named for what it holds shared.
#:
#: ``control`` is the history-free baseline. ``metadata_only`` keeps the
#: executor's reused metadata object but gives every request a private cache.
#: ``cache_only`` does the opposite. ``executor_like`` shares both, which is
#: what ``PyExecutor`` does. ``executor_like_zero_freed`` is ``executor_like``
#: with every freed page memset, which is the discriminator: it isolates page
#: *content* from every other thing the shared cache manager carries (index
#: mapper slots, scratch descriptors, host block-offset tables).
MODES: tuple[dict[str, Any], ...] = (
    {"name": "control", "share_cache": False, "share_metadata": False, "zero_freed": False},
    {"name": "metadata_only", "share_cache": False, "share_metadata": True, "zero_freed": False},
    {"name": "cache_only", "share_cache": True, "share_metadata": False, "zero_freed": False},
    {"name": "executor_like", "share_cache": True, "share_metadata": True, "zero_freed": False},
    {
        "name": "executor_like_zero_freed",
        "share_cache": True,
        "share_metadata": True,
        "zero_freed": True,
    },
)

#: Layers whose page bookkeeping is recorded per request. One of each kind:
#: layer 0 is ratio 0 (sliding window only), layer 2 is ratio 4 (overlap
#: compressor plus indexer) and layer 3 is ratio 128 (compressor, no indexer).
#: Recording all 43 would multiply the artifact by fourteen and say the same
#: thing.
PROBE_LAYERS: tuple[int, ...] = (0, 2, 3)


def _sha(tensor: torch.Tensor) -> str:
    """A short content hash, so "identical" is checkable from the artifact."""
    host = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(host.numpy().tobytes()).hexdigest()[:16]


@contextlib.contextmanager
def _cuda_default_device():
    """The CUDA default device the production executor supplies implicitly.

    Through ``torch.set_default_device`` rather than ``with torch.device(...)``
    for the reason :func:`layer_localization._cuda_default_device` documents:
    only the former is visible to the host-default-device scope the metadata
    build needs.
    """
    previous = torch.utils._device.CURRENT_DEVICE
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(previous)


def cache_manager(live: Any, max_seq_len: int, max_batch_size: int, pool_tokens: int) -> Any:
    """A ``DeepseekV4CacheManager`` sized the way the executor sizes one.

    ``pool_tokens`` is stated rather than derived from a memory fraction: the
    point of this suite is that page *recycling* is observable, and a pool
    sized from 30% of an 80 GB device would recycle so rarely that a real leak
    could hide behind capacity.
    """
    import tensorrt_llm
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    config = live.model_config.pretrained_config
    tokens_per_block = 128
    return DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=pool_tokens,
            event_buffer_max_size=0,
        ),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=config.num_hidden_layers,
        num_kv_heads=1,
        head_dim=config.v_head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=live.mapping,
        dtype=tensorrt_llm.bindings.DataType.BF16,
        compressor_dtype=tensorrt_llm.bindings.DataType.FLOAT,
        vocab_size=config.vocab_size,
        max_num_tokens=max_seq_len,
        sparse_attn_config=live.model_config.sparse_attention_config,
        model_config=live.model_config,
    )


def _attn_types_for(ratio: int) -> list[Any]:
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType,
        compress_ratio_has_attention,
    )

    return [t for t in DeepseekV4AttentionType if compress_ratio_has_attention(ratio, t)]


def page_report(manager: Any, request_id: int, ratios: Sequence[int]) -> dict[str, Any]:
    """Which physical pages this request owns, per probe layer and cache role.

    Read off the manager rather than off the metadata: the metadata's block
    tables are a device copy taken at ``prepare()`` time, and the question here
    is what the *allocator* handed out, which is the thing that gets recycled.
    """
    report: dict[str, Any] = {}
    for layer in PROBE_LAYERS:
        if layer >= len(ratios):
            continue
        for attn_type in _attn_types_for(ratios[layer]):
            try:
                pages = manager.get_cache_indices(request_id, layer, attn_type)
            except KeyError:
                continue
            report[f"layer{layer}.{attn_type.name}"] = [int(p) for p in pages]
    return report


def _as_bytes(pool: torch.Tensor) -> torch.Tensor | None:
    """A ``uint8`` view of a cache pool, or ``None`` if it has no plain one.

    The pools are not all ordinary dtypes: the indexer's compressed cache is
    packed FP4 (``float4_e2m1fn_x2``), which has no arithmetic and cannot be
    indexed, filled or hashed as itself. Every operation this module performs
    on a pool is byte-wise anyway --- hash it, zero it --- so it goes through
    this view and nothing has to special-case a dtype.
    """
    try:
        return pool.view(torch.uint8)
    except RuntimeError:  # a pool whose layout admits no byte view
        return None


def owned_page_digest(manager: Any, pages: dict[str, Any]) -> dict[str, str]:
    """Content hash of the pages a request owns, per probe layer and role.

    Taken twice per request --- after the forward and after ``free_resources``
    --- so "the freed pages still hold this request's data" is recorded rather
    than argued.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType,
    )

    digests: dict[str, str] = {}
    for key, ids in pages.items():
        if not ids:
            continue
        layer_part, type_part = key.split(".", 1)
        layer = int(layer_part[len("layer") :])
        pool = _as_bytes(manager.get_buffers(layer, DeepseekV4AttentionType[type_part]))
        if pool is None:
            continue
        index = torch.as_tensor(
            [i for i in ids if 0 <= i < pool.shape[0]], dtype=torch.long, device=pool.device
        )
        if index.numel() == 0:
            continue
        digests[key] = _sha(pool.index_select(0, index))
    return digests


def zero_all_pages(manager: Any, ratios: Sequence[int], num_layers: int) -> int:
    """Memset every cache pool the model owns. Returns pages cleared.

    Every layer and every role its ratio declares, not just the probe layers:
    a leak at layer 17 would walk straight past a probe-only memset, and the
    whole value of this mode is that a surviving difference cannot be page
    content. Pools overlap across layers in ``PER_LAYER`` index mode, so the
    same bytes are cleared more than once; at the pool sizes this suite asks
    for that costs milliseconds and removes a correctness argument.
    """
    cleared = 0
    for layer in range(num_layers):
        for attn_type in _attn_types_for(ratios[layer]):
            pool = _as_bytes(manager.get_buffers(layer, attn_type))
            if pool is None:
                continue
            pool.zero_()
            cleared += int(pool.shape[0])
    return cleared


class SequenceRun:
    """One cache/metadata configuration, driven through a list of prompts.

    Holds the shared objects for its mode and builds the private ones per
    request, so the difference between two modes is exactly the sharing and
    nothing else --- same model, same weights, same order, same tokens.
    """

    def __init__(
        self,
        live: Any,
        mode: dict[str, Any],
        max_seq_len: int,
        max_batch_size: int,
        pool_tokens: int,
        decode_steps: int = 0,
    ) -> None:
        self.live = live
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.pool_tokens = pool_tokens
        self.decode_steps = decode_steps
        self.ratios = list(live.model_config.pretrained_config.compress_ratios)
        self.num_layers = int(live.model_config.pretrained_config.num_hidden_layers)
        self._shared_manager = None
        self._shared_metadata = None
        self._next_request_id = 0
        if mode["share_cache"]:
            with _cuda_default_device():
                self._shared_manager = cache_manager(
                    live, max_seq_len, max_batch_size, pool_tokens
                )

    # -- resources -------------------------------------------------------

    def _manager(self) -> Any:
        if self._shared_manager is not None:
            return self._shared_manager
        with _cuda_default_device():
            return cache_manager(self.live, self.max_seq_len, self.max_batch_size, self.pool_tokens)

    def _metadata(self, manager: Any) -> Any:
        """The metadata object this request will use.

        In the shared mode it is built once, on the first request, and its
        fields are reassigned afterwards --- ``model_engine`` does exactly
        that, and a metadata built fresh per iteration would hide any stale
        row in its device buffers.
        """
        from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
            DeepseekV4TrtllmAttentionMetadata,
        )
        from tensorrt_llm._torch.metadata import KVCacheParams

        if self._shared_metadata is not None:
            # ``metadata_only`` mode reuses one metadata across *different*
            # managers, so the reference has to follow the live one or the
            # attention would read block tables out of a shut-down pool. Every
            # manager this mode builds is constructed identically, so the
            # device buffers ``__post_init__`` sized from the first one still
            # describe the current one.
            self._shared_metadata.kv_cache_manager = manager
            return self._shared_metadata
        metadata = DeepseekV4TrtllmAttentionMetadata(
            seq_lens=torch.tensor([1], dtype=torch.int32, device="cpu"),
            num_contexts=1,
            max_num_requests=self.max_batch_size,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            kv_cache_manager=manager,
            request_ids=[0],
            prompt_lens=[1],
            max_num_tokens=self.max_seq_len,
            mapping=self.live.mapping,
            sparse_attention_config=self.live.model_config.sparse_attention_config,
        )
        if self.mode["share_metadata"]:
            self._shared_metadata = metadata
        return metadata

    # -- one request -----------------------------------------------------

    def run(self, prompt: dict[str, Any], index: int) -> dict[str, Any]:
        """Prefill one prompt and record everything that could carry over."""
        import weakref

        from activation_replay import _host_default_device
        from layer_localization import _extra_attrs, _resolve

        from tensorrt_llm._torch.metadata import KVCacheParams
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
        from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
        from tensorrt_llm._torch.utils import model_extra_attrs

        token_ids = list(prompt["token_ids"])
        prompt_len = len(token_ids)
        request_id = self._next_request_id
        self._next_request_id += 1
        started = time.time()

        manager = self._manager()
        free_before = int(manager.index_mapper.num_free_slots())
        with _cuda_default_device():
            request = LlmRequest(
                request_id=request_id,
                max_new_tokens=1,
                input_tokens=token_ids,
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            if not manager.prepare_context(request):
                raise RuntimeError(
                    f"{prompt['id']} (request {request_id}): prepare_context refused the "
                    f"request; the pool holds {self.pool_tokens} tokens"
                )
            if not manager.resize_context(request, request.context_chunk_size):
                raise RuntimeError(f"{prompt['id']} (request {request_id}): resize_context failed")
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = [request]
            manager.prepare_resources(batch)

            metadata = self._metadata(manager)
            with _host_default_device():
                metadata.seq_lens = torch.tensor(
                    [prompt_len], dtype=torch.int32, device="cpu"
                )
                metadata.num_contexts = 1
                metadata.request_ids = [request_id]
                metadata.prompt_lens = [prompt_len]
                metadata.kv_cache_params = KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[0]
                )
                _extra_attrs(self.live)["attention_metadata"] = weakref.ref(metadata)
                metadata.prepare()

            pages = page_report(manager, request_id, self.ratios)
            slot = _index_slot(manager, request_id)

            layers = self.live.model.model.layers[: self.num_layers]
            store: dict[int, torch.Tensor] = {}
            handles = [
                layer.register_forward_hook(
                    _layer_hook(store, layer_id, layer, _resolve)
                )
                for layer_id, layer in enumerate(layers)
            ]
            try:
                input_ids = torch.tensor(token_ids, dtype=torch.int32, device="cuda")
                position_ids = (
                    torch.arange(prompt_len, device="cuda").unsqueeze(0).to(torch.int32)
                )
                with torch.inference_mode(), model_extra_attrs(_extra_attrs(self.live)):
                    logits = self.live.model.forward(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attn_metadata=metadata,
                    )
                torch.cuda.synchronize()
                last = logits.reshape(-1, logits.shape[-1])[-1].float().clone()
            finally:
                for handle in handles:
                    handle.remove()

            decode = self._decode(manager, metadata, request, prompt_len, last)

            after_forward = owned_page_digest(manager, pages)
            manager.free_resources(request)
            after_free = owned_page_digest(manager, pages)
            cleared = 0
            if self.mode["zero_freed"]:
                cleared = zero_all_pages(manager, self.ratios, self.num_layers)
            if self._shared_manager is None:
                manager.shutdown()

        hidden = int(self.live.model_config.pretrained_config.hidden_size)
        return {
            "index": index,
            "prompt_id": prompt["id"],
            "request_id": request_id,
            "prompt_tokens": prompt_len,
            "decode": decode,
            "index_mapper_slot": slot,
            "index_mapper_free_slots_before": free_before,
            "index_mapper_free_slots_after": int(manager.index_mapper.num_free_slots())
            if self._shared_manager is not None
            else None,
            "request_still_mapped_after_free": request_id in manager.kv_cache_map
            if self._shared_manager is not None
            else False,
            "pages": pages,
            "page_digest_after_forward": after_forward,
            "page_digest_after_free": after_free,
            "freed_pages_survived": sorted(
                key
                for key, digest in after_free.items()
                if after_forward.get(key) == digest and not _is_zero_digest(digest)
            ),
            "pages_cleared": cleared,
            "logits_sha": _sha(last),
            "logits": last,
            "layers": {
                layer_id: value.reshape(-1, hidden)[-1:].clone()
                for layer_id, value in store.items()
            },
            "elapsed_s": round(time.time() - started, 2),
        }

    def _decode(
        self,
        manager: Any,
        metadata: Any,
        request: Any,
        prompt_len: int,
        prefill_logits: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Greedy decode steps on the request the prefill just built.

        The executor's own loop, reduced to what changes state: grow the KV
        cache by one, re-point the *same* metadata object at a
        generation-shaped batch, re-``prepare()``, forward one token. Without
        this the harness only ever exercised prefill --- which is why its
        five-mode table could exonerate the cache manager and the metadata for
        a prompt whose executor divergence starts at decode step 1, and why an
        intervening prompt's 32 decode iterations were outside everything it
        measured.

        Greedy and self-driven: the point is a reproducible trajectory through
        the decode path, not agreement with a reference, so the argmax of the
        previous step is the next input.
        """
        from activation_replay import _host_default_device

        from tensorrt_llm._torch.metadata import KVCacheParams
        from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
        from tensorrt_llm._torch.utils import model_extra_attrs

        steps: list[dict[str, Any]] = []
        if self.decode_steps <= 0:
            return steps

        request_id = request.py_request_id
        logits = prefill_logits
        for step in range(self.decode_steps):
            token = int(logits.argmax())
            request.add_new_token(token, 0)
            if not manager.try_allocate_generation(request):
                raise RuntimeError(
                    f"request {request_id}: try_allocate_generation failed at decode step "
                    f"{step}; the pool holds {self.pool_tokens} tokens"
                )
            batch = ScheduledRequests()
            batch.generation_requests = [request]
            manager.prepare_resources(batch)

            cached = prompt_len + step
            with _host_default_device():
                metadata.seq_lens = torch.tensor([1], dtype=torch.int32, device="cpu")
                metadata.num_contexts = 0
                metadata.request_ids = [request_id]
                metadata.prompt_lens = [prompt_len]
                metadata.kv_cache_params = KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[cached]
                )
                metadata.prepare()

            input_ids = torch.tensor([token], dtype=torch.int32, device="cuda")
            position_ids = torch.tensor([[cached]], dtype=torch.int32, device="cuda")
            with torch.inference_mode(), model_extra_attrs(_extra_attrs_of(self.live)):
                out = self.live.model.forward(
                    input_ids=input_ids, position_ids=position_ids, attn_metadata=metadata
                )
            torch.cuda.synchronize()
            logits = out.reshape(-1, out.shape[-1])[-1].float().clone()
            steps.append(
                {
                    "step": step,
                    "input_token": token,
                    "kv_len": cached + 1,
                    "logits_sha": _sha(logits),
                    "logits": logits,
                    "argmax": int(logits.argmax()),
                }
            )
        return steps

    def shutdown(self) -> None:
        if self._shared_manager is not None:
            self._shared_manager.shutdown()
            self._shared_manager = None
        self._shared_metadata = None


def _extra_attrs_of(live: Any) -> dict[str, Any]:
    """The model's own extra-attrs registry, checked rather than assumed.

    Imported through ``layer_localization`` so both suites resolve the custom
    ops through the same dict; see that module for why the caller's config copy
    is the wrong one.
    """
    from layer_localization import _extra_attrs

    return _extra_attrs(live)


def _layer_hook(store: dict[int, torch.Tensor], layer_id: int, layer: Any, resolve: Any) -> Any:
    def hook(_mod: Any, _inputs: tuple, output: Any) -> None:
        store[layer_id] = resolve(layer, output).detach().float().clone()

    return hook


def _index_slot(manager: Any, request_id: int) -> int | None:
    with contextlib.suppress(Exception):
        return int(manager.index_mapper.get_index(request_id))
    return None


#: The digest of an all-zero page, per distinct byte count, memoised so the
#: "did the freed page survive" test does not call the hash on a pool slice it
#: already knows the answer for.
_ZERO_DIGESTS: set[str] = set()


def _is_zero_digest(digest: str) -> bool:
    return digest in _ZERO_DIGESTS


def register_zero_digests(manager: Any, ratios: Sequence[int]) -> None:
    """Record what an all-zero page hashes to, per probe layer and role.

    Needed because ``zero_freed`` mode makes every freed page identical to
    every other freed page, and reporting that as "the request's data
    survived" would invert the reading of the one mode that exists to rule
    page content out.
    """
    for layer in PROBE_LAYERS:
        if layer >= len(ratios):
            continue
        for attn_type in _attn_types_for(ratios[layer]):
            pool = _as_bytes(manager.get_buffers(layer, attn_type))
            if pool is None:
                continue
            _ZERO_DIGESTS.add(
                _sha(torch.zeros((1, *pool.shape[1:]), dtype=pool.dtype, device=pool.device))
            )


# ---------------------------------------------------------------------------
# The comparison.
# ---------------------------------------------------------------------------


def compare_decode(steps: list[dict[str, Any]], anchor: list[dict[str, Any]]) -> dict[str, Any]:
    """Where two decode trajectories of the same prompt first part company.

    Prefill can be bit-identical while decode is not --- that is exactly what
    the executor reports for ``long_prefill_2304`` --- so the step index is the
    number that matters, not "the two differ". Reported for the whole
    trajectory, because a greedy self-driven decode forks permanently once its
    argmax moves and later steps are then answering different questions.
    """
    if not steps or not anchor:
        return {
            "decode_steps_compared": 0,
            "decode_first_differing_step": None,
            "decode_first_differing_token_step": None,
            "decode_tokens_identical": True,
        }
    compared = min(len(steps), len(anchor))
    first_logit = next(
        (s for s in range(compared) if steps[s]["logits_sha"] != anchor[s]["logits_sha"]), None
    )
    first_token = next(
        (s for s in range(compared) if steps[s]["argmax"] != anchor[s]["argmax"]), None
    )
    return {
        "decode_steps_compared": compared,
        "decode_first_differing_step": first_logit,
        "decode_first_differing_token_step": first_token,
        "decode_tokens_identical": first_token is None,
    }


def compare_sequence(runs: list[dict[str, Any]], tg: Any) -> dict[str, Any]:
    """Every later occurrence of a prompt, against that prompt's first one.

    The anchor is the first occurrence rather than the previous one on
    purpose: chained comparisons hide a drift that is constant per step, and
    "run 10 differs from run 0" is the statement the acceptance item makes.
    """
    anchors: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for run in runs:
        pid = run["prompt_id"]
        anchor = anchors.setdefault(pid, run)
        if anchor is run:
            continue
        metrics = tg.compare(run["logits"], anchor["logits"])
        first_layer = None
        layer_max = {}
        for layer_id in sorted(set(anchor["layers"]) & set(run["layers"])):
            delta = float(
                (run["layers"][layer_id] - anchor["layers"][layer_id]).abs().max()
            )
            layer_max[layer_id] = delta
            if first_layer is None and delta > 0.0:
                first_layer = layer_id
        decode = compare_decode(run.get("decode") or [], anchor.get("decode") or [])
        comparisons.append(
            {
                "prompt_id": pid,
                "anchor_index": anchor["index"],
                "index": run["index"],
                "anchor_request_id": anchor["request_id"],
                "request_id": run["request_id"],
                "logits_identical": run["logits_sha"] == anchor["logits_sha"],
                "logits_max_abs": round(float(metrics["max_abs"]), 8),
                "logits_rel_max_abs": round(float(metrics["rel_max_abs"]), 8),
                "logits_cosine": round(float(metrics["cosine"]), 8),
                "argmax_match": int(run["logits"].argmax()) == int(anchor["logits"].argmax()),
                "first_differing_layer": first_layer,
                "same_pages_as_anchor": run["pages"] == anchor["pages"],
                "layer_max_abs": {str(k): round(v, 8) for k, v in layer_max.items()},
                **decode,
            }
        )
    return {
        "anchors": {pid: run["index"] for pid, run in anchors.items()},
        "comparisons": comparisons,
        "decode_steps": max((len(r.get("decode") or []) for r in runs), default=0),
        "history_dependent": sorted(
            {
                c["prompt_id"]
                for c in comparisons
                if not c["logits_identical"] or c["decode_first_differing_step"] is not None
            }
        ),
    }


def teardown_report(runs: list[dict[str, Any]], mode: dict[str, Any]) -> dict[str, Any]:
    """What a finished request left behind, per the acceptance item's wording.

    Two different things are checked and reported separately, because only one
    of them is a defect on its own. A request that is still in ``kv_cache_map``
    or still holds an ``IndexMapper`` slot after ``free_resources`` is a leak.
    Pages that still hold its bytes are *expected* --- nothing memsets a freed
    page --- and matter only because the next owner can read them, which is
    what the ``zero_freed`` mode exists to test.
    """
    still_mapped = [r["index"] for r in runs if r["request_still_mapped_after_free"]]
    slots_leaked = [
        r["index"]
        for r in runs
        if r["index_mapper_free_slots_after"] is not None
        and r["index_mapper_free_slots_before"] is not None
        and r["index_mapper_free_slots_after"] < r["index_mapper_free_slots_before"]
    ]
    surviving = {r["index"]: r["freed_pages_survived"] for r in runs if r["freed_pages_survived"]}
    return {
        "mode": mode["name"],
        "requests_still_mapped_after_free": still_mapped,
        "index_mapper_slots_not_released": slots_leaked,
        "requests_with_surviving_page_content": sorted(surviving),
        "surviving_page_roles": {
            str(index): roles for index, roles in sorted(surviving.items())
        },
    }


def judge(per_mode: dict[str, Any]) -> list[str]:
    """The problems this suite reports, in the order they should be read."""
    problems: list[str] = []
    control = per_mode.get("control")
    if control and control["comparison"]["history_dependent"]:
        problems.append(
            "control (fresh cache and fresh metadata per request) is already not "
            f"reproducible for {control['comparison']['history_dependent']}; the "
            "measurement cannot attribute anything until that is explained"
        )
    for name, entry in per_mode.items():
        drifting = entry["comparison"]["history_dependent"]
        if name != "control" and drifting:
            problems.append(f"{name}: repeated prompts disagree with their first run {drifting}")
        leaked = entry["teardown"]["requests_still_mapped_after_free"]
        if leaked:
            problems.append(f"{name}: requests {leaked} are still mapped after free_resources")
        slots = entry["teardown"]["index_mapper_slots_not_released"]
        if slots:
            problems.append(f"{name}: requests {slots} did not release their IndexMapper slot")
    return problems


def reading(per_mode: dict[str, Any]) -> str:
    """One sentence naming the channel, derived from the mode table.

    Written as a function rather than left to the reader because the whole
    point of running five modes is that the *pattern* across them is the
    answer, and a reader scanning one mode at a time would miss it.
    """
    def drifts(name: str) -> bool:
        entry = per_mode.get(name)
        return bool(entry and entry["comparison"]["history_dependent"])

    if drifts("control"):
        return "not attributable: the control is already irreproducible"
    if not drifts("executor_like"):
        return "no history dependence reproduced in process under any mode"
    if drifts("cache_only") and not drifts("metadata_only"):
        channel = "the shared cache manager"
    elif drifts("metadata_only") and not drifts("cache_only"):
        channel = "the reused attention metadata"
    elif drifts("cache_only") and drifts("metadata_only"):
        channel = "both the shared cache manager and the reused attention metadata"
    else:
        channel = "the combination of the two, neither alone"
    if drifts("executor_like_zero_freed"):
        return f"{channel} carries it, and it survives memsetting every freed page"
    return f"{channel} carries it, through the content of recycled pages"


# ---------------------------------------------------------------------------
# The gating half: the same question through the real LLM API.
# ---------------------------------------------------------------------------

#: The two prompts the acceptance item names. ``cache_boundary_257`` lands one
#: token past a 128-token page boundary; ``long_prefill_2304`` spans eighteen
#: of them. Everything else in the sequences below is history for these two.
GATING_PROMPTS: tuple[str, ...] = ("cache_boundary_257", "long_prefill_2304")

#: Shared prefix of every engine's sequence, and a controlled experiment in its
#: own right.
#:
#: Its first job is that the gating pair never sees an untouched pool: an engine
#: whose first request is the one under test would prove determinism only for
#: the case that cannot go wrong.
#:
#: Its second job is to separate the two explanations for the executor's
#: request-position variance, which iteration 39 measured but could not
#: attribute. ``chat_geography`` runs three times *back to back* and then once
#: more *after a different prompt*, so the four executions differ in exactly one
#: property:
#:
#: * if #1 and #2 already disagree with #0, the executor is simply not
#:   reproducible after its first request, whatever ran in between;
#: * if #1 and #2 match #0 but #4 does not, the trigger is the intervening
#:   *shape* --- a recompile, a tuner cache entry, or a buffer sized for the
#:   other prompt --- and not the position.
#:
#: One engine build answers that; the alternative was two.
_PRIMER: tuple[str, ...] = (
    "chat_geography",
    "chat_geography",
    "chat_geography",
    "chat_arithmetic",
    "chat_geography",
)

#: One engine, three executions of each gating prompt. The pair repeats, so
#: execution *n* of ``cache_boundary_257`` is preceded by *n-1* long prefills
#: --- three different histories rather than three identical ones.
SAME_ENGINE_SEQUENCE: tuple[str, ...] = _PRIMER + GATING_PROMPTS * 3

#: A fresh engine's sequence is the shared prefix plus one execution of each,
#: so its first execution of a gating prompt has *exactly* the history the
#: same-engine anchor had. That is what makes the fresh-versus-same comparison
#: about engine identity rather than about request order.
FRESH_ENGINE_SEQUENCE: tuple[str, ...] = _PRIMER + GATING_PROMPTS


def kv_cache_teardown(llm: Any) -> dict[str, Any]:
    """What the runtime says it still holds once every request has finished.

    Read through ``LLM.get_stats`` because that is the only channel out of the
    spawned workers that carries structured numbers; the log carries text and
    ``collective_rpc`` is refused above ``model_world_size=1``. The last
    iteration that reported a KV-cache block count is the one that matters: if
    a finished request still owns pages, ``usedNumBlocks`` never returns to
    zero.
    """
    report: dict[str, Any] = {"iterations": 0, "last": None, "problems": []}
    try:
        stats = llm.get_stats(timeout=10)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal to the run
        report["problems"].append(f"get_stats unavailable: {exc}")
        return report
    report["iterations"] = len(stats)
    with_kv = [s for s in stats if isinstance(s, dict) and s.get("kvCacheStats")]
    if not with_kv:
        report["problems"].append("no iteration reported kvCacheStats")
        return report
    last = with_kv[-1]["kvCacheStats"]
    report["last"] = {
        k: last.get(k)
        for k in ("usedNumBlocks", "freeNumBlocks", "maxNumBlocks", "tokensPerBlock")
    }
    used = last.get("usedNumBlocks")
    if used:
        report["problems"].append(
            f"{used} KV blocks are still held after every request finished; a freed "
            "request still owns pages"
        )
    return report


def executor_pass(
    args: Any,
    full_model: Any,
    prompts: dict[str, dict[str, Any]],
    sequence: Sequence[str],
    log_path: str,
    tag: str,
) -> dict[str, Any]:
    """Construct one engine, drive it through ``sequence``, tear it down.

    Returns one record per request in order, each carrying the generated
    tokens and the full generation-logit tensor, plus the resolved contract,
    the teardown report and the worker-log scan. Construction runs with the
    caller's descriptors already redirected into ``log_path``: the spawned
    workers inherit them at MPI init, and that file is the only place their
    dispatch markers appear.
    """
    from tensorrt_llm import LLM

    # The registered eager construction, plus the one knob this suite needs
    # that `eager_full_model` does not: without `enable_iter_perf_stats` the
    # runtime publishes no iteration statistics, so `LLM.get_stats` comes back
    # empty and the "no block still held after teardown" half of the
    # acceptance item has nothing to read. It is added here rather than in
    # `llm_kwargs` so the contract `eager_full_model` builds and records stays
    # exactly what it was reviewed as; `_contract_problems` does not inspect
    # this field, so the shared contract check still applies unchanged.
    kwargs = full_model.llm_kwargs(args) | {"enable_iter_perf_stats": True}
    started = time.time()
    llm = LLM(**kwargs)
    construct_s = round(time.time() - started, 2)
    runs: list[dict[str, Any]] = []
    try:
        resolved = full_model._resolved_contract(llm)
        problems = full_model._contract_problems(resolved, args)
        for index, pid in enumerate(sequence):
            run = full_model._generate(llm, prompts[pid], args.parity_tokens, want_logits=True)
            rows = run.pop("_logits")
            runs.append(
                {
                    "engine": tag,
                    "index": index,
                    "prompt_id": pid,
                    "prompt_tokens": run["prompt_tokens"],
                    "token_ids": run["token_ids"],
                    "text": run["text"],
                    "finish_reason": run["finish_reason"],
                    "nonfinite_logprobs": run["nonfinite_logprobs"],
                    "logits_finite": bool(torch.isfinite(rows).all()),
                    "logits_sha": _sha(rows),
                    "logits": rows,
                    "elapsed_s": run["elapsed_s"],
                }
            )
        teardown = kv_cache_teardown(llm)
    finally:
        llm.shutdown()

    return {
        "engine": tag,
        "construct_s": construct_s,
        "worker_log": log_path,
        "construction": full_model._describe_kwargs(kwargs),
        "runtime_contract": {"resolved": resolved, "problems": problems, "passed": not problems},
        "teardown": teardown,
        "runs": runs,
    }


def worker_dispatch(full_model: Any, log_path: str) -> dict[str, Any]:
    """Scan the one log every engine's workers wrote to.

    One log for three engines, and not by choice: OpenMPI's daemon --- which
    forwards the spawned workers' output --- inherits the descriptors this
    process holds when MPI initialises, which is the first ``tensorrt_llm``
    import. Re-pointing stdout between engines therefore moves the *launcher's*
    output and leaves every worker still writing to the first file. Measured:
    giving each engine its own file left engines two and three with launcher
    lines only and no rank tags at all, which ``scan_worker_log`` correctly
    read as "0 of 8 ranks logged" --- a harness artefact reported as a runtime
    failure. Scanning the shared file once, after every engine has been shut
    down, states what is actually knowable: the eight ranks took the required
    path in this process.
    """
    with open(log_path, errors="replace") as handle:
        captured = handle.read()
    scanned = full_model.scan_worker_log(captured, world_size=8)
    scanned["attribution"] = (
        "one file for every engine in this process; MPI fixes the workers' "
        "descriptors at the first tensorrt_llm import, so per-engine worker "
        "output is not separable"
    )
    return scanned


#: Index of the primer execution that follows a *different* prompt. Everything
#: before it is the same prompt back to back, so the two groups differ in
#: exactly one property and the reading below can name it.
_PRIMER_AFTER_OTHER_SHAPE = _PRIMER.index("chat_arithmetic") + 1


def position_reading(diagnostics: list[dict[str, Any]]) -> str:
    """Which of the two request-position explanations the primer supports.

    A function rather than a note for the reader, for the same reason
    :func:`reading` is one: the answer is in the *pattern* across four
    executions of one prompt, and a reader scanning them one at a time would
    have to reconstruct the experiment to see it.
    """
    same_engine = [
        c
        for c in diagnostics
        if c["same_engine"] and c["prompt_id"] == _PRIMER[0] and c["anchor"].endswith("#0")
    ]
    if not same_engine:
        return "not measured: the primer produced no same-engine repeat to compare"

    def index_of(entry: dict[str, Any]) -> int:
        return int(entry["execution"].rsplit("#", 1)[1])

    back_to_back = [c for c in same_engine if index_of(c) < _PRIMER_AFTER_OTHER_SHAPE]
    after_other = [c for c in same_engine if index_of(c) >= _PRIMER_AFTER_OTHER_SHAPE]

    def reproduced(entries: list[dict[str, Any]]) -> bool:
        return all(e["tokens_identical"] and e["logits_identical"] for e in entries)

    # Does the *same* request position land on the same value in every engine?
    # A per-position drift that repeats to the bit across three independently
    # built engines is a counter, not a race, and the two send the next
    # iteration to different code.
    by_position: dict[int, set[float]] = {}
    for entry in diagnostics:
        if entry["prompt_id"] != _PRIMER[0] or not entry["anchor"].endswith("#0"):
            continue
        by_position.setdefault(index_of(entry), set()).add(entry["step0_max_abs"])
    same_across_engines = all(len(values) == 1 for values in by_position.values())

    if not back_to_back or not after_other:
        return "not measured: the primer did not cover both back-to-back and after-another-shape"
    if reproduced(back_to_back) and reproduced(after_other):
        return "the executor reproduces a repeated prompt regardless of what ran before it"
    if not reproduced(back_to_back):
        # A drift that repeats to the bit in every engine is a different
        # object from a race, and sends the next iteration to a different
        # place: something advances deterministically per request rather than
        # something landing in a different order.
        settled = "and it lands on the same value in every engine" if same_across_engines else (
            "and it differs between engines, so part of it is not reproducible either"
        )
        return (
            "the executor does not reproduce a repeated prompt even back to back, so the "
            f"trigger is not the intervening prompt --- {settled}"
        )
    return (
        "the executor reproduces a repeated prompt back to back but not after a different "
        "prompt, so the trigger is the intervening shape rather than the request position"
    )


def compare_executions(passes: list[dict[str, Any]], tg: Any) -> dict[str, Any]:
    """Every repeated execution against the first engine's first one.

    Token *prefix* rather than the whole sequence, because that is the wording
    the acceptance item uses and because it says something sharper: the step
    at which two greedy decodes first disagree is where the state that differs
    became visible, and reporting only "the texts differ" throws that away.

    The two gating prompts land in ``comparisons`` and are what the verdict
    rests on. The primer prompts land in ``diagnostics``, which the verdict
    ignores --- but they answer the question that opened this investigation:
    ``chat_geography`` runs twice inside the first engine and once in each of
    the other two, so "does the executor reproduce a short prompt within one
    engine, and across engines?" is measured here rather than inferred from
    two runs of a different suite.
    """
    anchors: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for entry in passes:
        for run in entry["runs"]:
            pid = run["prompt_id"]
            anchor = anchors.setdefault(pid, run)
            if anchor is run:
                continue
            tokens, ref_tokens = run["token_ids"], anchor["token_ids"]
            divergence = next(
                (
                    i
                    for i in range(min(len(tokens), len(ref_tokens)))
                    if tokens[i] != ref_tokens[i]
                ),
                None,
            )
            rows, ref_rows = run["logits"], anchor["logits"]
            steps = min(rows.shape[0], ref_rows.shape[0])
            step0 = tg.compare(rows[0], ref_rows[0])
            first_step = next(
                (s for s in range(steps) if not torch.equal(rows[s], ref_rows[s])), None
            )
            record = {
                "prompt_id": pid,
                "anchor": f"{anchor['engine']}#{anchor['index']}",
                "execution": f"{run['engine']}#{run['index']}",
                "same_engine": run["engine"] == anchor["engine"],
                "tokens_identical": tokens == ref_tokens,
                "first_token_divergence": divergence,
                "logits_identical": run["logits_sha"] == anchor["logits_sha"],
                "first_differing_logit_step": first_step,
                "step0_max_abs": round(float(step0["max_abs"]), 8),
                "step0_cosine": round(float(step0["cosine"]), 8),
            }
            (comparisons if pid in GATING_PROMPTS else diagnostics).append(record)
    same_engine = [c for c in comparisons if c["same_engine"]]
    fresh_engine = [c for c in comparisons if not c["same_engine"]]
    return {
        "anchors": {pid: f"{run['engine']}#{run['index']}" for pid, run in anchors.items()},
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "request_position_reading": position_reading(diagnostics),
        "non_gating_not_reproducible": sorted(
            {
                c["prompt_id"]
                for c in diagnostics
                if not (c["tokens_identical"] and c["logits_identical"])
            }
        ),
        "same_engine_executions_compared": len(same_engine),
        "fresh_engine_executions_compared": len(fresh_engine),
        "same_engine_identical": all(
            c["tokens_identical"] and c["logits_identical"] for c in same_engine
        ),
        "fresh_engine_identical": all(
            c["tokens_identical"] and c["logits_identical"] for c in fresh_engine
        ),
    }


def judge_executions(passes: list[dict[str, Any]], comparison: dict[str, Any]) -> list[str]:
    """The registered shape of this evidence, applied to what was measured."""
    problems: list[str] = []
    per_prompt: dict[str, int] = {}
    for entry in passes:
        for run in entry["runs"]:
            if run["prompt_id"] in GATING_PROMPTS:
                per_prompt[run["prompt_id"]] = per_prompt.get(run["prompt_id"], 0) + 1
            if not run["logits_finite"]:
                problems.append(f"{entry['engine']}#{run['index']} {run['prompt_id']}: nonfinite logits")
            if run["nonfinite_logprobs"]:
                problems.append(
                    f"{entry['engine']}#{run['index']} {run['prompt_id']}: "
                    f"{run['nonfinite_logprobs']} nonfinite logprobs"
                )
            if not run["text"]:
                problems.append(f"{entry['engine']}#{run['index']} {run['prompt_id']}: empty output")
    for pid in GATING_PROMPTS:
        if per_prompt.get(pid, 0) < 5:
            problems.append(
                f"{pid}: {per_prompt.get(pid, 0)} executions recorded; the acceptance item "
                "requires three same-engine plus two fresh-engine"
            )
    engines = {entry["engine"] for entry in passes}
    if len(engines) < 3:
        problems.append(f"{len(engines)} engines built; three are required (one same, two fresh)")
    # One scan for the whole process, so it is judged once rather than once per
    # engine -- see `worker_dispatch` for why per-engine attribution is not
    # available.
    scanned = next((e["worker_dispatch"] for e in passes if e.get("worker_dispatch")), None)
    if scanned is None:
        problems.append("no worker-dispatch scan was recorded")
    elif not scanned["passed"]:
        problems.append(f"worker dispatch {scanned['problems']}")
    for entry in passes:
        if not entry["runtime_contract"]["passed"]:
            problems.append(
                f"{entry['engine']}: runtime contract {entry['runtime_contract']['problems']}"
            )
        for problem in entry["teardown"]["problems"]:
            problems.append(f"{entry['engine']}: {problem}")
    for c in comparison["comparisons"]:
        if not c["tokens_identical"]:
            problems.append(
                f"{c['prompt_id']} {c['execution']} vs {c['anchor']}: tokens diverge at step "
                f"{c['first_token_divergence']}"
            )
        elif not c["logits_identical"]:
            problems.append(
                f"{c['prompt_id']} {c['execution']} vs {c['anchor']}: same tokens but logits "
                f"differ from step {c['first_differing_logit_step']} "
                f"(step0 max_abs {c['step0_max_abs']})"
            )
    return problems


def run_all(
    live: Any,
    prompts: dict[str, dict[str, Any]],
    sequence: Iterable[str],
    tg: Any,
    ranks: Any,
    max_seq_len: int,
    max_batch_size: int,
    pool_tokens: int,
    modes: Sequence[dict[str, Any]] = MODES,
    decode_steps: int = 0,
) -> dict[str, Any]:
    """Drive every mode through the same scripted sequence on one loaded model."""
    order = list(sequence)
    missing = sorted({pid for pid in order if pid not in prompts})
    if missing:
        raise RuntimeError(f"the sequence names unregistered prompts: {missing}")

    per_mode: dict[str, Any] = {}
    for mode in modes:
        ranks.log(
            f"[state_determinism] mode {mode['name']}: {len(order)} requests "
            f"x (prefill + {decode_steps} decode)"
        )
        runner = SequenceRun(
            live, mode, max_seq_len, max_batch_size, pool_tokens, decode_steps=decode_steps
        )
        try:
            if runner._shared_manager is not None:
                register_zero_digests(runner._shared_manager, runner.ratios)
            runs = [runner.run(prompts[pid], i) for i, pid in enumerate(order)]
        finally:
            runner.shutdown()
        comparison = compare_sequence(runs, tg)
        per_mode[mode["name"]] = {
            "mode": mode,
            "runs": [
                {
                    **{k: v for k, v in run.items() if k not in ("logits", "layers", "decode")},
                    "decode": [
                        {k: v for k, v in step.items() if k != "logits"}
                        for step in (run.get("decode") or [])
                    ],
                }
                for run in runs
            ],
            "comparison": comparison,
            "teardown": teardown_report(runs, mode),
        }
        ranks.log(
            f"  history_dependent={comparison['history_dependent']} "
            f"in {sum(r['elapsed_s'] for r in runs):.1f}s"
        )
        del runs
        torch.cuda.empty_cache()

    return {
        "sequence": order,
        "pool_tokens": pool_tokens,
        "decode_steps": decode_steps,
        "per_mode": per_mode,
        "reading": reading(per_mode),
        "problems": judge(per_mode),
    }
