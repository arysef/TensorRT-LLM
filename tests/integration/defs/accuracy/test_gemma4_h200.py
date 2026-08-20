# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint H200 (SM90) runtime, parity, and accuracy nodes for Gemma 4 26B-A4B.

Gemma 4 already ships in TensorRT-LLM with B200 coverage; this module closes the
*Hopper* gap.  Every node here runs the native BF16 ``google/gemma-4-26B-A4B-it``
checkpoint through the production LLM API (``backend=pytorch``, ``tp_size=1``,
FlashInfer attention, ``KVCacheManagerV2``) on one H200 and compares against the
native Transformers reference in :mod:`gemma4_h200_reference`.

Two invariants shape the whole file:

* **No skips.**  A pass-critical node that cannot run must *fail*.  These nodes
  exist to prove H200 behaviour, and a green skip proves nothing.
* **Both runtime matrices.**  Every node that executes model code runs with
  ``(cuda_graph=False, overlap_scheduler=False)`` and with
  ``(cuda_graph=True, overlap_scheduler=True)``, and the enabled case asserts
  *observed* CUDA-graph capture and replay rather than the config value.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import glob
import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig, SamplingParams

from .accuracy_core import MMLU, MMMU, LlmapiAccuracyTestHarness
from .gemma4_h200_reference import (
    CHAT_TEMPLATE_KWARGS,
    TEXT_PROMPTS,
    assert_cosine,
    checkpoint_eos_token_ids,
    compare_tensors,
    contiguous_image_runs,
    ensure_capture,
    ensure_native_completions,
    gemma4_26b_checkpoint,
    h200_environment_report,
    mmlu_five_shot_prompt,
    mmmu_canary_items,
    render_image_chat,
    require_single_h200,
    write_evidence,
)

MODEL_NAME = "google/gemma-4-26B-A4B-it"

# --------------------------------------------------------------------------
# Runtime matrix and gates
# --------------------------------------------------------------------------

# One boolean drives both switches: the acceptance matrix pairs them
# (``cuda_graph=false, overlap_scheduler=false`` vs
# ``cuda_graph=true, overlap_scheduler=true``), so splitting them would invent
# configurations the task never asked for.
RUNTIME_MATRIX = (
    pytest.param(False, id="cg0_ovl0"),
    pytest.param(True, id="cg1_ovl1"),
)

# Replay/parity gate: cosine >= 0.99 at every compared output plus exact
# discrete outcomes (expert indices, greedy tokens).  Stable BF16 last-bit
# jitter must not be chased, but a real contract break -- dropped mask, wrong
# pool, wrong RoPE pairing -- moves cosine far below this.
MIN_COSINE = 0.99

# How close the *reference's own* top-2 logits have to be before the reference
# is treated as not having resolved the ordering, in units of the bfloat16
# spacing at that logit's magnitude.
#
# This implements an explicit human amendment to acceptance criterion 6, taken
# in iteration 10 after the literal exact-token form was shown to be
# unsatisfiable by any implementation (see the validation report, section 12).
# The amended contract is: identical greedy tokens at every step, *except* at
# steps where the reference's own top-2 gap is within its bfloat16 storage
# resolution, where the runtime's token must be one of those two candidates --
# with the count and margins reported.
#
# The constant is 2, and it is deliberately conservative rather than fitted.
# bfloat16 carries 8 significand bits, so at a logit magnitude of ~24 one
# representable step is 0.125; two candidates within two steps of each other sit
# inside the grid the reference stores its own logits on.  The measured licence
# is far wider: running the identical reference with HF's other sanctioned
# attention kernel (`sdpa` instead of `eager`) moves its logits by up to 1.875
# at this magnitude, i.e. **15 ULP** -- so a 2-ULP window is 7.5x tighter than
# the reference's own demonstrated implementation spread.  Widening this
# constant to make a step pass would be exactly the gaming this gate exists to
# prevent; every exempted step is therefore reported with its margin in both
# absolute and ULP terms so the constant's actual reach is auditable.
REFERENCE_TIE_ULPS = 2.0


def bf16_ulp(value: float) -> float:
    """The spacing between adjacent bfloat16 values at ``value``'s magnitude.

    bfloat16 is IEEE-754 binary32 truncated to 8 significand bits (7 stored), so
    within the binade ``[2**e, 2**(e+1))`` consecutive values are ``2**(e-7)``
    apart.  Subnormals are irrelevant here -- these are logits of order 10.
    """
    magnitude = abs(float(value))
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return math.inf
    return 2.0 ** (math.floor(math.log2(magnitude)) - 7)


def reference_tie(
    expected_row: torch.Tensor, actual_token: int
) -> Optional[Dict[str, Any]]:
    """Describe a step the *reference* did not resolve, or ``None``.

    ``expected_row`` is the reference's logit row with the banned end-of-turn
    ids already masked, so "top-2" means the two candidates the reference was
    actually choosing between.  A step qualifies only when **both** hold:

    * the reference's own top-2 gap is at most ``REFERENCE_TIE_ULPS`` bfloat16
      steps at the magnitude it is storing them, and
    * the runtime picked one of those same two candidates.

    The second condition is what keeps this from being a waiver: the runtime
    does not get to emit an arbitrary token at a close step, it has to land on
    one of the two the reference itself was weighing.
    """
    top = torch.topk(expected_row, 2)
    first, second = int(top.indices[0]), int(top.indices[1])
    if actual_token not in (first, second):
        return None
    gap = float(top.values[0] - top.values[1])
    ulp = bf16_ulp(float(top.values[0]))
    if gap > REFERENCE_TIE_ULPS * ulp:
        return None
    return {
        "source_top2": [first, second],
        "source_top2_logits": [float(top.values[0]), float(top.values[1])],
        "gap": gap,
        "bf16_ulp": ulp,
        "gap_in_ulp": gap / ulp,
        "runtime_token": int(actual_token),
        "runtime_token_is_source_runner_up": int(actual_token) == second,
    }

# The MoE contract this bring-up validates on SM90: the unquantized BF16
# CUTLASS implementation, Gemma 4's exact GELU-tanh GeGLU activation, and the
# fused op it dispatches to.  Named explicitly so "MoE parity passed" can never
# stand in for "the production H200 MoE kernel ran".
MOE_EXPECTED_WRAPPER = "ConfigurableMoE"
MOE_EXPECTED_IMPL = "CutlassFusedMoE"
MOE_EXPECTED_ACTIVATION = "Geglu"
MOE_EXPECTED_OP = "torch.ops.trtllm.fused_moe"

# Fixed 32-sample MMLU canary: ``(subject, row index in <subject>_test.csv)``.
# Checked in so the native reference and TensorRT-LLM score the *same* items
# every run, and so a later iteration cannot quietly resample.
MMLU_CANARY_SAMPLES: Tuple[Tuple[str, int], ...] = (
    ("abstract_algebra", 0),
    ("anatomy", 3),
    ("astronomy", 5),
    ("business_ethics", 1),
    ("clinical_knowledge", 7),
    ("college_biology", 2),
    ("college_chemistry", 4),
    ("college_computer_science", 6),
    ("college_mathematics", 0),
    ("college_medicine", 8),
    ("college_physics", 1),
    ("computer_security", 3),
    ("conceptual_physics", 9),
    ("econometrics", 2),
    ("electrical_engineering", 5),
    ("elementary_mathematics", 11),
    ("formal_logic", 4),
    ("global_facts", 6),
    ("high_school_biology", 13),
    ("high_school_chemistry", 8),
    ("high_school_computer_science", 2),
    ("high_school_geography", 10),
    ("high_school_mathematics", 15),
    ("high_school_physics", 7),
    ("high_school_psychology", 21),
    ("human_aging", 12),
    ("international_law", 5),
    ("machine_learning", 3),
    ("marketing", 17),
    ("medical_genetics", 9),
    ("nutrition", 14),
    ("world_religions", 6),
)

# Fixed 16-sample MMMU canary: ``(config name, index into the validation split)``.
# Every MMMU config ships 30 validation rows, so these indices are always valid.
MMMU_CANARY_SAMPLES: Tuple[Tuple[str, int], ...] = (
    ("Accounting", 0),
    ("Agriculture", 1),
    ("Architecture_and_Engineering", 2),
    ("Art", 0),
    ("Basic_Medical_Science", 3),
    ("Biology", 1),
    ("Chemistry", 2),
    ("Clinical_Medicine", 0),
    ("Computer_Science", 4),
    ("Design", 1),
    ("Economics", 2),
    ("Electronics", 0),
    ("Finance", 3),
    ("Geography", 1),
    ("History", 2),
    ("Math", 0),
)

# Canary tolerances.  A canary is a catastrophic-regression gate, not a
# statistical benchmark: 32/16 samples cannot resolve small deltas, so the bar
# is "TensorRT-LLM tracks its own native reference on the same items".  One
# flipped MMLU item is 3.125 points; one flipped MMMU item is 6.25.
MMLU_CANARY_TOLERANCE = 9.5  # <= 3 flipped items out of 32
MMMU_CANARY_TOLERANCE = 12.6  # <= 2 flipped items out of 16

# Chunked-prefill canary geometry.  The chunk budget is *derived* from the
# measured image soft-token run (see ``image_chunk_budget``) so the scheduler is
# forced to snap a chunk boundary off the bidirectional block; a fixed constant
# either sits past the whole image prompt (proving nothing) or drifts if the
# processor's soft-token count changes.
LONG_CONTEXT_TOKENS = 2048
# Leading text for the chunked-prefill image canary.  Its only job is to push
# the image's soft-token block past a KV-page boundary so the scheduler's
# snap-down has somewhere to land; the content is irrelevant and fixed.
IMAGE_CANARY_PREFIX = (
    "Read the following note before answering. " + "This note is deliberately verbose. " * 24
)
# The canary asks a question the *reference* answers decisively.  This is a
# precondition of the test, not a preference: chunked and unchunked prefill are
# mathematically equivalent here (``get_context_mask`` for a chunk reproduces
# the full-sequence mask bitwise, and the multimodal embeddings are bitwise
# identical) but they are *numerically* different -- paged-prefill-with-prefix
# and ragged-prefill accumulate in different orders.  Measured against native
# Transformers on this checkpoint, both paths sit 0.2-2.3 logits away from HF
# across a 16-step horizon.  So a step the reference itself decides by less than
# that spread is a coin flip, and asserting exact token parity on it tests
# nothing about chunking.  The previous question ("Describe this landmark in one
# sentence.") had a 0.75-logit (6 bf16 ULP) step-4 margin: the unchunked path
# won that flip and the chunked path lost it, with no defect on either side.
# Measured alternatives are in ``repro/canary_determinacy.py`` /
# ``evidence-canary_determinacy.json``; this one is the only candidate that is
# both a full 16-step horizon and decisively resolved (worst step 3.0 logits).
IMAGE_CANARY_QUESTION = "Identify the landmark shown and state the city it stands in."
IMAGE_CANARY_DECODE_STEPS = 16
# Floor the canary enforces on the reference's own per-step top-2 margin, so the
# determinacy above is *checked* every run rather than assumed.  Set just above
# the 2.29-logit worst-case deviation measured between the runtime and HF over
# this horizon (``evidence-chunk_canary_hf_logits.json``); if a future prompt,
# checkpoint, or reference makes any compared step tighter than this, the canary
# fails loudly as an invalid canary instead of flaking on a coin flip.
IMAGE_CANARY_MIN_REFERENCE_MARGIN = 2.5
LONG_CONTEXT_DECODE_STEPS = 64
MIN_PREFILL_CHUNKS = 3

# Room for the longest request any node here issues (MMMU's 8192-token input
# plus its 512-token output).  Left unset, the runtime infers Gemma 4's
# 262144-token maximum and sizes the KV pools for it, which is both slower and
# unrepresentative of what these nodes actually exercise.
DEFAULT_MAX_SEQ_LEN = MMMU.MAX_INPUT_LEN + MMMU.MAX_OUTPUT_LEN


@pytest.fixture(autouse=True)
def single_process_worker(monkeypatch):
    """Run the TP1 executor in this process so the runtime can be inspected.

    Every node here proves something about what the runtime *actually did*:
    ``RuntimeProbe`` wraps ``FlashInferAttention.forward`` and
    ``CUDAGraphRunner.replay``, ``ReplayHarness`` hooks production modules, and
    the pool/manager assertions read live objects.  All of that requires the
    executor to live in the test process; the default TP1 path spawns an
    ``MpiPoolSession`` worker, where the patches and hooks would apply to the
    wrong process.  ``TLLM_WORKER_USE_SINGLE_PROCESS`` is the repository's own
    switch for exactly this (``llmapi/utils.py:enable_worker_single_process_for_tp1``,
    "helpful for return-logits performance and debugging"), and several
    ``tests/unittest/_torch`` suites already set it the same way.
    """
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")


# --------------------------------------------------------------------------
# Runtime probe: what actually dispatched
# --------------------------------------------------------------------------


@dataclasses.dataclass
class AttentionCall:
    """One observed FlashInfer attention invocation."""

    layer_idx: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    flashinfer_backend: str
    custom_mask: bool
    num_contexts: int
    num_generations: int
    num_ctx_tokens: int = 0
    # Page-index bookkeeping for this layer's pool, read from the host mirror
    # the metadata keeps in lockstep with the device buffer (no device sync, no
    # mutation of runtime state).
    kv_pool_id: Optional[int] = None
    pool_pages: Optional[int] = None
    min_page_index: Optional[int] = None
    max_page_index: Optional[int] = None
    live_page_entries: int = 0

    def key(self) -> Tuple[int, int, int]:
        return (self.num_heads, self.num_kv_heads, self.head_dim)

    def page_indices_in_bounds(self) -> bool:
        """Every live page index addresses a real page of this layer's pool."""
        if self.min_page_index is None or self.pool_pages is None:
            return True
        return 0 <= self.min_page_index and self.max_page_index < self.pool_pages


def _page_index_bounds(
    metadata, layer_idx: int
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], int]:
    """``(pool_id, pool_pages, min_index, max_index, live_entries)`` for a layer.

    Reads the host mirror ``prepare()`` retains, so the probe adds no device
    work and never mutates runtime state.  ``min_index`` is taken *after* the
    metadata's own ``BAD_PAGE_INDEX`` sanitization, i.e. it is the value
    FlashInfer's kernels will actually dereference; an out-of-range entry here
    is an illegal access waiting to happen.
    """
    pool_id = None
    mapping = getattr(metadata, "_vswa_layer_to_pool", None)
    if mapping is not None:
        pool_id = mapping.get(layer_idx)
    host = None
    if pool_id is not None:
        host = getattr(metadata, "_host_pool_indices", {}).get(pool_id)
    if host is None:
        host = getattr(metadata, "_host_paged_kv_indices", None)
    live = int(metadata.num_context_blocks) + int(metadata.num_generation_blocks)
    pages = None
    buffers = metadata.kv_cache_manager.get_buffers(layer_idx)
    if buffers is not None:
        pages = int(buffers.shape[0])
    if host is None or live <= 0:
        return pool_id, pages, None, None, 0
    entries = host[:live]
    # BAD_PAGE_INDEX (-1) placeholders mark blocks the request no longer owns;
    # the metadata clamps them before FlashInfer sees them, so compare against
    # the clamped value the kernel receives.
    lo = max(0, int(entries.min().item()))
    return pool_id, pages, lo, int(entries.max().item()), live


class RuntimeProbe:
    """Records what the runtime *actually* did, not what it was configured to do.

    Wraps the real call sites, so the evidence comes from execution:

    * ``CUDAGraphRunner.capture`` / ``.replay`` -- the CUDA-graph hard path.
    * ``FlashInferAttention.forward`` -- per-layer geometry, kernel family, and
      whether a custom mask actually reached the kernel.
    * ``FlashInferAttentionMetadata._plan_with_params`` -- the backend string of
      the wrapper objects FlashInfer built.
    * ``triton_prefill_with_custom_mask`` -- must stay at zero on SM90, where
      FA2 handles the custom mask directly.
    """

    def __init__(self) -> None:
        self.attention_calls: List[AttentionCall] = []
        self.planned_backends: Dict[Tuple[int, int, int], Dict[str, str]] = {}
        self.graph_captures = 0
        self.graph_replays = 0
        self.triton_prefill_calls = 0
        self.moe_op_calls = 0
        self.moe_op_name: Optional[str] = None
        self._undo: List[Callable[[], None]] = []

    def _patch(self, owner: Any, name: str, factory: Callable[[Any], Any]) -> None:
        original = getattr(owner, name)
        setattr(owner, name, factory(original))
        self._undo.append(lambda: setattr(owner, name, original))

    def __enter__(self) -> "RuntimeProbe":
        from tensorrt_llm._torch.attention_backend import flashinfer as fi_module
        from tensorrt_llm._torch.attention_backend import triton_prefill as triton_module
        from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cg_module

        probe = self

        def attention_forward(original):
            @functools.wraps(original)
            def wrapper(self, q, k, v, metadata, forward_args=None, **kwargs):
                mask_data = kwargs.get("attention_mask_data")
                if mask_data is None and forward_args is not None:
                    mask_data = getattr(forward_args, "attention_mask_data", None)
                pool_id, pages, lo, hi, live = _page_index_bounds(metadata, self.layer_idx)
                probe.attention_calls.append(
                    AttentionCall(
                        layer_idx=self.layer_idx,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        flashinfer_backend=getattr(self, "flashinfer_backend", "unknown"),
                        custom_mask=mask_data is not None,
                        num_contexts=int(metadata.num_contexts),
                        num_generations=int(metadata.num_generations),
                        num_ctx_tokens=int(getattr(metadata, "num_ctx_tokens", 0) or 0),
                        kv_pool_id=pool_id,
                        pool_pages=pages,
                        min_page_index=lo,
                        max_page_index=hi,
                        live_page_entries=live,
                    )
                )
                return original(self, q, k, v, metadata, forward_args=forward_args, **kwargs)

            return wrapper

        def plan_with_params(original):
            @functools.wraps(original)
            def wrapper(self, plan_params, flashinfer_backend="fa2"):
                result = original(self, plan_params, flashinfer_backend)
                wrappers = self._plan_params_to_wrappers.get(plan_params)
                if wrappers is not None:
                    entry = probe.planned_backends.setdefault(
                        (plan_params.num_heads, plan_params.num_kv_heads, plan_params.head_dim),
                        {},
                    )
                    entry["requested"] = flashinfer_backend
                    for role in ("prefill_wrapper", "decode_wrapper", "ragged_prefill_wrapper"):
                        obj = getattr(wrappers, role, None)
                        if obj is not None:
                            entry[role] = str(getattr(obj, "_backend", "unknown"))
                return result

            return wrapper

        def count_graph(attr: str):
            def factory(original):
                @functools.wraps(original)
                def wrapper(self, *args, **kwargs):
                    setattr(probe, attr, getattr(probe, attr) + 1)
                    return original(self, *args, **kwargs)

                return wrapper

            return factory

        def triton_prefill(original):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                probe.triton_prefill_calls += 1
                return original(*args, **kwargs)

            return wrapper

        self._patch(fi_module.FlashInferAttention, "forward", attention_forward)
        self._patch(fi_module.FlashInferAttentionMetadata, "_plan_with_params", plan_with_params)
        self._patch(cg_module.CUDAGraphRunner, "capture", count_graph("graph_captures"))
        self._patch(cg_module.CUDAGraphRunner, "replay", count_graph("graph_replays"))
        self._patch(triton_module, "triton_prefill_with_custom_mask", triton_prefill)
        self._patch_moe_op()
        return self

    def _patch_moe_op(self) -> None:
        """Count real invocations of the fused MoE op the experts dispatch to.

        The class name says which implementation was *selected*; only the op
        call proves the fused kernel actually executed rather than a Python
        fallback inside the same class.
        """
        probe = self
        namespace, _, op_name = MOE_EXPECTED_OP.rpartition(".")
        assert namespace == "torch.ops.trtllm", MOE_EXPECTED_OP
        ops_namespace = torch.ops.trtllm
        original = getattr(ops_namespace, op_name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            probe.moe_op_calls += 1
            probe.moe_op_name = MOE_EXPECTED_OP
            return original(*args, **kwargs)

        # ``_OpNamespace.__getattr__`` caches the resolved packet as an
        # instance attribute, so setting it here is what later lookups see and
        # deleting it restores the lazy resolution.
        setattr(ops_namespace, op_name, wrapper)
        self._undo.append(lambda: setattr(ops_namespace, op_name, original))

    def __exit__(self, *exc_info) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    # -- queries ----------------------------------------------------------
    def geometries(self) -> Dict[Tuple[int, int, int], int]:
        counts: Dict[Tuple[int, int, int], int] = {}
        for call in self.attention_calls:
            counts[call.key()] = counts.get(call.key(), 0) + 1
        return counts

    def backends_used(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for call in self.attention_calls:
            counts[call.flashinfer_backend] = counts.get(call.flashinfer_backend, 0) + 1
        return counts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attention_calls": len(self.attention_calls),
            "attention_backends": self.backends_used(),
            "attention_geometries": {str(k): v for k, v in self.geometries().items()},
            "planned_wrapper_backends": {str(k): v for k, v in self.planned_backends.items()},
            "custom_mask_calls": sum(1 for c in self.attention_calls if c.custom_mask),
            "decode_calls": sum(1 for c in self.attention_calls if c.num_generations > 0),
            "context_calls": sum(1 for c in self.attention_calls if c.num_contexts > 0),
            "graph_captures": self.graph_captures,
            "graph_replays": self.graph_replays,
            "triton_prefill_calls": self.triton_prefill_calls,
            "page_index_bounds": self.page_index_bounds(),
        }

    def page_index_bounds(self) -> Dict[str, Dict[str, int]]:
        """Observed page-index range per pool, with that pool's capacity."""
        summary: Dict[str, Dict[str, int]] = {}
        for call in self.attention_calls:
            if call.min_page_index is None:
                continue
            entry = summary.setdefault(
                f"pool{call.kv_pool_id}",
                {
                    "min_index": call.min_page_index,
                    "max_index": call.max_page_index,
                    "pool_pages": call.pool_pages,
                    "observations": 0,
                    "max_live_entries": 0,
                },
            )
            entry["min_index"] = min(entry["min_index"], call.min_page_index)
            entry["max_index"] = max(entry["max_index"], call.max_page_index)
            entry["max_live_entries"] = max(entry["max_live_entries"], call.live_page_entries)
            entry["observations"] += 1
        return summary

    def assert_page_indices_in_bounds(self, *, label: str) -> Dict[str, Dict[str, int]]:
        """No attention call may address a page outside its own pool."""
        offenders = [c for c in self.attention_calls if not c.page_indices_in_bounds()]
        if offenders:
            worst = offenders[0]
            raise AssertionError(
                f"{label}: layer {worst.layer_idx} (pool {worst.kv_pool_id}) handed FlashInfer "
                f"page indices in [{worst.min_page_index}, {worst.max_page_index}] but the pool "
                f"holds {worst.pool_pages} pages; {len(offenders)} of "
                f"{len(self.attention_calls)} calls are out of bounds"
            )
        bounds = self.page_index_bounds()
        if not bounds:
            raise AssertionError(
                f"{label}: no page-index observation was recorded, so pool bounds are unproven"
            )
        return bounds

    def pools_by_phase(self) -> Dict[str, List[int]]:
        """Which pools were touched during prefill and during cached decode."""
        prefill = {c.kv_pool_id for c in self.attention_calls if c.num_contexts > 0}
        decode = {c.kv_pool_id for c in self.attention_calls if c.num_generations > 0}
        return {
            "prefill": sorted(p for p in prefill if p is not None),
            "decode": sorted(p for p in decode if p is not None),
        }

    def assert_graph_hard_path(self, *, enabled: bool, label: str) -> None:
        """Enabled runs must show real capture *and* replay; baseline neither."""
        if enabled:
            if self.graph_captures <= 0 or self.graph_replays <= 0:
                raise AssertionError(
                    f"{label}: cuda_graph=True but observed captures="
                    f"{self.graph_captures} replays={self.graph_replays}; the run "
                    "silently used a non-graph path, so this is not "
                    "cuda_graph_hard_path evidence"
                )
        elif self.graph_captures or self.graph_replays:
            raise AssertionError(
                f"{label}: cuda_graph=False but observed captures="
                f"{self.graph_captures} replays={self.graph_replays}"
            )

    def assert_moe_op_ran(self, *, label: str) -> int:
        """The fused MoE op must have actually executed, not just been chosen."""
        if self.moe_op_calls <= 0:
            raise AssertionError(
                f"{label}: {MOE_EXPECTED_OP} was never invoked; the experts did not reach the "
                "fused CUTLASS kernel, so this is not MoE dispatch evidence"
            )
        return self.moe_op_calls

    def assert_sm90_dispatch(self, *, label: str) -> None:
        """No Blackwell-only kernel family and no silent Triton detour on SM90."""
        backends = self.backends_used()
        if not backends:
            raise AssertionError(f"{label}: no FlashInfer attention call was observed at all")
        unexpected = {name for name in backends if name != "fa2"}
        if unexpected:
            raise AssertionError(
                f"{label}: FlashInfer attention ran with {sorted(unexpected)}; SM90 must "
                "use the fa2 kernel family (trtllm-gen is Blackwell-only)"
            )
        for key, entry in self.planned_backends.items():
            for role, backend in entry.items():
                if role == "requested":
                    continue
                if backend not in ("fa2", "auto"):
                    raise AssertionError(
                        f"{label}: geometry {key} planned {role}={backend!r}; expected fa2/auto"
                    )
        if self.triton_prefill_calls:
            raise AssertionError(
                f"{label}: the Triton custom-mask prefill fallback ran "
                f"{self.triton_prefill_calls} times; on SM90 FA2 handles the custom mask "
                "directly, so this indicates an unexpected dispatch change"
            )


# --------------------------------------------------------------------------
# LLM construction and introspection
# --------------------------------------------------------------------------


def _kv_cache_config(free_fraction: float) -> KvCacheConfig:
    # Block and partial reuse stay off for the validated BF16 baseline (see the
    # plan's "Considered and rejected directions"); dtype "auto" keeps the KV
    # cache in the checkpoint's bfloat16 rather than quantizing it.
    return KvCacheConfig(
        enable_block_reuse=False,
        enable_partial_reuse=False,
        free_gpu_memory_fraction=free_fraction,
        dtype="auto",
    )


def _moe_config() -> MoeConfig:
    """MoE settings for the H200 BF16 baseline.

    ``disable_finalize_fusion`` is not a tuning choice, it is what makes this
    path reproducible at all. The CUTLASS FC2+finalize fusion accumulates each
    token's expert contributions in an order that is not fixed run to run, and
    ``MoeConfig`` documents the flag as recovering "deterministic numerical
    behavior with top-k > 2". Gemma 4 26B-A4B routes **top-k 8**, so this model
    sits squarely in the nondeterministic regime.

    Measured on this H200 with `repro/prefill_determinism_localize.py`: with the
    fusion on, two identical single-request context forwards through one engine
    diverge first at ``model.layers.0.moe.experts`` and then at 704 downstream
    modules, ending at a final-logit max-abs of ~1.5 -- larger than this path's
    entire disagreement with Transformers on text prompts. With the fusion off,
    the same comparison is **bitwise equal, 0 divergent modules**.

    Criterion 1 asks for deterministic generation and criterion 6 compares a
    fixed greedy token at every step; neither is well posed against a runtime
    that does not reproduce itself, so the deterministic mode is the correct
    configuration for the functional baseline. The cost is the unfused FC2 +
    finalize epilogue, which is a throughput trade this bring-up does not
    optimize.
    """
    return MoeConfig(disable_finalize_fusion=True)


@contextlib.contextmanager
def gemma4_llm(
    *,
    enabled: bool,
    max_batch_size: int = 8,
    max_num_tokens: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    enable_chunked_prefill: bool = False,
    free_gpu_memory_fraction: float = 0.55,
):
    """Build the production LLM for one runtime-matrix cell."""
    kwargs: Dict[str, Any] = dict(
        model=gemma4_26b_checkpoint(),
        tensor_parallel_size=1,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len if max_seq_len is not None else DEFAULT_MAX_SEQ_LEN,
        kv_cache_config=_kv_cache_config(free_gpu_memory_fraction),
        moe_config=_moe_config(),
        cuda_graph_config=CudaGraphConfig() if enabled else None,
        disable_overlap_scheduler=not enabled,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    if max_num_tokens is not None:
        kwargs["max_num_tokens"] = max_num_tokens

    llm = LLM(**kwargs)
    try:
        yield llm
    finally:
        llm.shutdown()
        import gc

        gc.collect()
        torch.cuda.empty_cache()


def engine_of(llm: LLM):
    """The in-process PyExecutor behind a TP1 LLM."""
    executor = getattr(llm, "_executor", None)
    engine = getattr(executor, "engine", None)
    if engine is None:
        raise AssertionError(
            "could not reach the in-process executor engine; these nodes need direct "
            "runtime introspection, which requires the single-process worker "
            "(tp_size=1, no external orchestrator)"
        )
    return engine


def model_of(llm: LLM):
    """The TensorRT-LLM ``Gemma4ForConditionalGeneration`` module."""
    return engine_of(llm).model_engine.model


def language_layers(llm: LLM) -> List[torch.nn.Module]:
    """The Gemma 4 text decoder layers of the running engine."""
    model = model_of(llm)
    llm_model = getattr(model, "llm", model)
    return list(llm_model.model.layers)


def vision_tower_layers(llm: LLM) -> List[torch.nn.Module]:
    """The production vision-tower encoder layers of the loaded model."""
    tower = getattr(model_of(llm), "vision_tower", None)
    assert tower is not None, "the loaded model has no vision tower"
    return list(tower.encoder.layers)


def multimodal_projector(llm: LLM) -> torch.nn.Module:
    """The module that turns tower outputs into language-model soft tokens."""
    projector = getattr(model_of(llm), "embed_vision", None)
    assert projector is not None, "the loaded model has no multimodal projector"
    return projector


def kv_cache_manager_of(llm: LLM):
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType

    return engine_of(llm).resource_manager.resource_managers[ResourceManagerType.KV_CACHE_MANAGER]


def assert_kv_cache_manager_v2(llm: LLM, *, label: str):
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    manager = kv_cache_manager_of(llm)
    if not isinstance(manager, KVCacheManagerV2):
        raise AssertionError(
            f"{label}: KV cache manager is {type(manager).__name__}, expected KVCacheManagerV2"
        )
    return manager


def pool_descriptors(llm: LLM) -> Dict[int, Dict[str, Any]]:
    """``layer_idx -> pool geometry``, read from the live V2 pools.

    The buffer is the paged pool itself in FlashInfer's NHD layout
    ``[num_pages, kv_factor, page_size, num_kv_heads, head_dim]``, so this is
    the runtime's own view rather than a restatement of the config.
    """
    manager = kv_cache_manager_of(llm)
    descriptors: Dict[int, Dict[str, Any]] = {}
    for layer_idx in range(len(language_layers(llm))):
        buffers = manager.get_buffers(layer_idx, kv_layout="NHD")
        if buffers is None:
            continue
        shape = tuple(buffers.shape)
        descriptors[layer_idx] = {
            "shape": shape,
            "num_pages": shape[0],
            "page_size": shape[2],
            "num_kv_heads": shape[3],
            "head_dim": shape[4],
            "dtype": str(buffers.dtype),
            "data_ptr": buffers.data_ptr(),
        }
    return descriptors


def package_provenance() -> Dict[str, Any]:
    """Which built artifacts this process actually imported.

    Kernel-level evidence is only worth as much as the binary behind it, so
    every dispatch claim records the resolved package paths and the native
    library's build time: a stale wheel or a leftover extension shows up here
    rather than as a mysterious numeric difference.
    """
    import tensorrt_llm
    import tensorrt_llm.bindings as bindings

    entry = {
        "tensorrt_llm_package": tensorrt_llm.__file__,
        "tensorrt_llm_version": tensorrt_llm.__version__,
        "bindings_extension": bindings.__file__,
    }
    for name, path in (("bindings_extension", bindings.__file__),):
        with contextlib.suppress(OSError, TypeError):
            entry[f"{name}_mtime"] = os.path.getmtime(path)
    native_lib = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", "libtensorrt_llm.so")
    if os.path.exists(native_lib):
        entry["native_library"] = native_lib
        entry["native_library_mtime"] = os.path.getmtime(native_lib)
    return entry


def assert_moe_dispatch(resolution: List[Dict[str, Any]], *, label: str) -> None:
    """The MoE that ran must be the SM90 CUTLASS GeGLU path, not a stand-in.

    "MoE parity passed" is not evidence on its own: TRTLLM-Gen is Blackwell
    only, VANILLA is a reference implementation that bypasses the fused kernel,
    and the Triton MoE targets a different activation contract.  Each of those
    can produce plausible numbers while proving nothing about the H200
    production path, so the implementation class, the activation, and the
    resolver's own verdict are all asserted.
    """
    for entry in resolution:
        implementation = entry.get("implementation")
        assert implementation == MOE_EXPECTED_IMPL, (
            f"{label}: experts ran through {implementation!r} (wrapper "
            f"{entry.get('wrapper')!r}), expected {MOE_EXPECTED_IMPL!r}; TRTLLM-Gen is "
            "Blackwell-only and VANILLA/Triton are not the H200 production path"
        )
        assert entry.get("wrapper") == MOE_EXPECTED_WRAPPER, (
            f"{label}: the MoE composition wrapper is {entry.get('wrapper')!r}, expected "
            f"{MOE_EXPECTED_WRAPPER!r} (the shared resolver path this bring-up validates)"
        )
        activation = entry.get("activation")
        assert activation == MOE_EXPECTED_ACTIVATION, (
            f"{label}: expert activation is {activation!r}, expected "
            f"{MOE_EXPECTED_ACTIVATION!r} (Gemma 4 uses exact GELU-tanh GeGLU)"
        )
        assert "resolution_report_error" not in entry, (
            f"{label}: the MoE resolver failed to produce a report: "
            f"{entry['resolution_report_error']}"
        )
        report = entry.get("resolution_report")
        assert report, f"{label}: no MoEResolutionReport was produced"
        winner = json.dumps(report)
        assert MOE_EXPECTED_IMPL in winner, (
            f"{label}: the resolution report does not name {MOE_EXPECTED_IMPL}: {winner[:400]}"
        )


def moe_resolution_reports(llm: LLM) -> List[Dict[str, Any]]:
    """Which MoE implementation actually runs, and why it won.

    The resolved class on the live module is the authoritative "what ran".  The
    resolver is then re-run with that module's own shapes to recover the full
    ``MoEResolutionReport`` -- winner, eligible alternatives, and rejection
    reasons -- which ``create_moe`` computes but does not retain on the module.
    """
    from tensorrt_llm._torch.modules.fused_moe.interface import ActivationType
    from tensorrt_llm._torch.modules.fused_moe.moe_resolution import resolve_moe_impl

    activation_names = {int(member): member.name for member in ActivationType}
    # The MoE module keeps only `mapping`/`quant_config`, not the whole
    # ModelConfig, so take it from the language model that built it.
    model = model_of(llm)
    model_config = getattr(model, "llm", model).model_config
    seen: Dict[str, Dict[str, Any]] = {}
    for layer in language_layers(llm):
        moe = getattr(layer, "moe", None)
        experts = getattr(moe, "experts", None) if moe is not None else None
        if experts is None:
            continue
        # ``ConfigurableMoE`` is a dispatch wrapper -- it holds the resolved
        # backend in ``.backend`` and does not compute anything itself, so the
        # wrapper's class name says nothing about which kernel ran.  Report both:
        # the wrapper for the composition path, and the backend that actually
        # executes the experts.
        compute = getattr(experts, "backend", None) or experts
        activation = getattr(compute, "activation_type", None)
        if activation is None:
            activation = getattr(experts, "activation_type", None)
        entry: Dict[str, Any] = {
            "wrapper": type(experts).__name__,
            # The class of the object that actually executes the experts.
            "implementation": type(compute).__name__,
            "activation_type": activation,
            "activation": activation_names.get(activation, "unknown"),
            "routing_method": type(
                getattr(compute, "routing_method", None) or getattr(experts, "routing_method", None)
            ).__name__,
            "num_experts": getattr(compute, "num_experts", None),
            "hidden_size": getattr(compute, "hidden_size", None),
            "intermediate_size": getattr(compute, "intermediate_size", None),
            # Read off the live backend, not off the config we passed in: this
            # is the setting that decides whether the runtime reproduces itself
            # at all (the CUTLASS FC2+finalize fusion accumulates a token's
            # expert contributions in an unfixed order, and `MoeConfig`
            # documents it as nondeterministic for top-k > 2 -- Gemma 4 routes
            # top-k 8). Evidence that does not name it cannot be checked.
            "use_fused_finalize": getattr(compute, "use_fused_finalize", None),
            "top_k": getattr(
                getattr(compute, "routing_method", None)
                or getattr(experts, "routing_method", None),
                "top_k",
                None,
            ),
        }
        try:
            report = resolve_moe_impl(
                model_config,
                dtype=experts.dtype,
                num_experts=experts.num_experts,
                hidden_size=experts.hidden_size,
                intermediate_size=experts.intermediate_size,
                bias=getattr(experts, "bias", False),
                activation_type=activation,
                routing=experts.routing_method,
                layer_idx=getattr(experts, "layer_idx", None),
            )
            entry["resolution_report"] = report.to_dict()
        except Exception as exc:  # noqa: BLE001 - reported, never silently dropped
            entry["resolution_report_error"] = f"{type(exc).__name__}: {exc}"
        seen[json.dumps(entry, sort_keys=True, default=str)] = entry
    return list(seen.values())


def runtime_report(llm: LLM, probe: RuntimeProbe, *, enabled: bool) -> Dict[str, Any]:
    """Everything the validation report must carry about one run."""
    engine = engine_of(llm)
    graph_runner = engine.model_engine.cuda_graph_runner
    return {
        "environment": h200_environment_report(),
        "model_class": type(model_of(llm)).__name__,
        "backend": str(llm.args.backend),
        "attn_backend": str(llm.args.attn_backend),
        "tensor_parallel_size": llm.args.tensor_parallel_size,
        "quant_algo": str(llm.args.quant_config.quant_algo),
        "kv_cache_quant_algo": str(llm.args.quant_config.kv_cache_quant_algo),
        "kv_cache_manager": type(kv_cache_manager_of(llm)).__name__,
        "kv_cache_block_reuse": llm.args.kv_cache_config.enable_block_reuse,
        "kv_cache_partial_reuse": llm.args.kv_cache_config.enable_partial_reuse,
        "enable_chunked_prefill": llm.args.enable_chunked_prefill,
        "cuda_graph": enabled,
        "overlap_scheduler": not llm.args.disable_overlap_scheduler,
        "cuda_graph_runner_enabled": bool(graph_runner.enabled),
        "cuda_graph_hard_path": enabled and probe.graph_captures > 0 and probe.graph_replays > 0,
        "moe_resolution": moe_resolution_reports(llm),
        "dispatch": probe.as_dict(),
    }


def assert_graph_hard_path(llm: LLM, probe: RuntimeProbe, *, enabled: bool, label: str) -> None:
    """Two independent CUDA-graph checks, both taken from the live runtime.

    The probe counts observed ``capture``/``replay`` calls; the runner's own
    ``graphs`` dict is the state those calls leave behind (the same signal
    ``test_llm_api_pytorch_whisper.py`` asserts on).  Requiring both means a
    configuration that reports ``cuda_graph=True`` while executing eagerly
    cannot pass.
    """
    probe.assert_graph_hard_path(enabled=enabled, label=label)
    runner = engine_of(llm).model_engine.cuda_graph_runner
    assert bool(runner.enabled) == enabled, (
        f"{label}: CUDAGraphRunner.enabled={runner.enabled} but cuda_graph={enabled}"
    )
    if enabled and not runner.graphs:
        raise AssertionError(
            f"{label}: CUDAGraphRunner holds no captured graphs after the run, so "
            "cuda_graph=True did not exercise the graph hard path"
        )


def greedy(max_tokens: int, *, logits: bool = False, suppress_eos: bool = False) -> SamplingParams:
    """Deterministic greedy decoding, optionally returning per-step logits.

    ``suppress_eos`` mirrors the reference's ``min_new_tokens=max_new_tokens``:
    Transformers implements that by banning every ``eos_token_id`` from
    selection for the whole generation, so a parity run that compares against
    such a reference has to ban the same ids or it will emit end-of-turn at the
    step the reference was forced past.  Measured on this checkpoint, the ban
    affects selection only -- the returned per-step logits stay unbanned (e.g.
    ``logit[106] = 23.65`` at a step where 106 was banned), so the logit
    comparison remains raw against raw.
    """
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        return_generation_logits=logits,
        bad_token_ids=list(checkpoint_eos_token_ids()) if suppress_eos else None,
    )


@functools.lru_cache(maxsize=1)
def gemma4_processor():
    """The checkpoint's own processor, used for chat/image rendering."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(gemma4_26b_checkpoint())


def render_text(text: str) -> List[int]:
    """Render a plain prompt through the checkpoint chat template into token ids."""
    tokenizer = gemma4_processor().tokenizer
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        add_generation_prompt=True,
        tokenize=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.flatten().tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(i) for i in ids]


def render_image_text(question: str, num_images: int = 1, *, text_first: bool = False) -> str:
    """Render a multimodal prompt exactly as the reference capture renders it.

    Delegates to the reference module so the native side and the TensorRT-LLM
    side cannot drift apart in chat rendering -- the single most common way a
    "model" regression turns out to be a prompt difference.
    """
    return render_image_chat(
        gemma4_processor(), question, num_images=num_images, text_first=text_first
    )


def token_prompt(input_ids: Sequence[int]) -> Dict[str, Any]:
    """Drive TensorRT-LLM with the reference's token ids, not a re-tokenization."""
    return {"prompt_token_ids": [int(i) for i in input_ids]}


def image_prompt(text: str, image_paths: Sequence[str]) -> Dict[str, Any]:
    from PIL import Image

    return {
        "prompt": text,
        "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in image_paths]},
    }


def generation_logits(output) -> torch.Tensor:
    """Per-step logits of one completion, as ``[steps, vocab]`` float32 on CPU."""
    logits = output.outputs[0].generation_logits
    assert logits is not None, (
        "the runtime returned no generation logits; "
        "SamplingParams(return_generation_logits=True) is required for logit parity"
    )
    tensor = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
    # ``.float().cpu()`` is a no-op when the runtime already hands back a CPU
    # float32 tensor, which would leave this aliasing a buffer the executor is
    # free to reuse on the next request.  Clone so a later ``llm.generate`` (the
    # teacher-forced localization pass, for one) cannot rewrite logits already
    # under comparison.
    return tensor.detach().float().cpu().reshape(-1, tensor.shape[-1]).clone()


# --------------------------------------------------------------------------
# Activation replay plumbing
# --------------------------------------------------------------------------


class GraphSafeDecodeCapture:
    """Record cached-decode attention activations so CUDA-graph replay keeps them.

    Python forward hooks are useless under CUDA graphs: replay executes the
    recorded kernels and never re-enters Python, so a hook installed after the
    engine exists simply never fires for a replayed decode.  This wraps
    ``Gemma4Attention.forward`` at class level *before* the engine is built, so
    the two ``copy_`` calls below are traced during graph capture and re-execute
    on every replay.  That is what lets the enabled matrix compare the same
    layer boundary as the baseline instead of falling back to a logits-only
    check.

    Everything inside the patched forward is fixed-shape and value-independent
    (two copies into pre-allocated buffers and one counter increment), so the
    recorded graph does not depend on the step it was captured with.  The
    counter is itself device-side and therefore also increments on replay, which
    is what makes "exactly one cached-decode forward wrote these buffers" an
    assertable fact rather than an assumption.
    """

    def __init__(self, layer_indices: Sequence[int], hidden_size: int, dtype=torch.bfloat16):
        self._layer_indices = sorted({int(i) for i in layer_indices})
        assert self._layer_indices, "GraphSafeDecodeCapture needs at least one layer"
        self._counter_layer = self._layer_indices[0]
        self._slots = {
            layer_idx: {
                "in": torch.zeros(hidden_size, device="cuda", dtype=dtype),
                "out": torch.zeros(hidden_size, device="cuda", dtype=dtype),
            }
            for layer_idx in self._layer_indices
        }
        self._steps = torch.zeros((), device="cuda", dtype=torch.int32)
        self._undo: List[Callable[[], None]] = []

    def __enter__(self) -> "GraphSafeDecodeCapture":
        from tensorrt_llm._torch.models import modeling_gemma4 as gemma4_module

        capture = self
        owner = gemma4_module.Gemma4Attention
        original = owner.forward

        @functools.wraps(original)
        def forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
            output = original(self, position_ids, hidden_states, attn_metadata, *args, **kwargs)
            slot = capture._slots.get(getattr(self, "layer_idx", None))
            # Decode-only forwards. Both operands are plain ints, so the branch
            # resolves while the graph is being traced and the copies below are
            # unconditionally part of the recorded graph.
            if (
                slot is not None
                and int(attn_metadata.num_contexts) == 0
                and int(attn_metadata.num_generations) > 0
            ):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                slot["in"].copy_(hidden_states.reshape(-1, hidden_states.shape[-1])[0])
                slot["out"].copy_(tensor.reshape(-1, tensor.shape[-1])[0])
                if self.layer_idx == capture._counter_layer:
                    capture._steps.add_(1)
            return output

        owner.forward = forward
        self._undo.append(lambda: setattr(owner, "forward", original))
        return self

    def __exit__(self, *exc_info) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    def reset(self) -> None:
        """Drop warmup/graph-capture writes so only the measured step counts."""
        self._steps.zero_()
        for slot in self._slots.values():
            slot["in"].zero_()
            slot["out"].zero_()

    def steps(self) -> int:
        return int(self._steps.item())

    def captured_layers(self) -> List[int]:
        return list(self._layer_indices)

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        slot = self._slots.get(int(layer_idx))
        if slot is None:
            return None
        return (
            slot["in"].detach().float().cpu().reshape(1, -1),
            slot["out"].detach().float().cpu().reshape(1, -1),
        )


class ReplayHarness:
    """Force source activations into production modules and read their outputs.

    A forward *pre*-hook overwrites the module's input with the tensor the
    native model saw at the same boundary; a forward hook records what the
    production module produced.  Every representative layer is forced, so each
    recorded output is a clean replay rather than a downstream consequence of
    the previous replacement.
    """

    def __init__(self) -> None:
        self.outputs: Dict[str, torch.Tensor] = {}
        # Every token count seen at a hooked module, so a miss reports what
        # actually arrived instead of just "nothing was captured".
        self.observed_rows: Dict[str, List[int]] = {}
        self._handles: List[Any] = []

    def _note(self, name: str, rows: int) -> None:
        self.observed_rows.setdefault(name, []).append(int(rows))

    def missed(self, name: str, expected_rows: int) -> str:
        return (
            f"no replay output for {name}: expected a {expected_rows}-token forward, "
            f"observed token counts {self.observed_rows.get(name, [])}"
        )

    @staticmethod
    def _replace(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return source.to(device=target.device, dtype=target.dtype)

    def force_attention(self, name: str, module: torch.nn.Module, hidden: torch.Tensor) -> None:
        flat = hidden.reshape(-1, hidden.shape[-1])
        rows = flat.shape[0]

        def pre_hook(_mod, args, kwargs):
            target = kwargs.get("hidden_states")
            if target is None:
                return None
            self._note(name, target.shape[0])
            # Only the context forward for this exact sequence is a replay
            # target; decode steps (one token per request) pass through so the
            # same engine still serves both phases.
            if target.shape[0] != rows:
                return None
            kwargs["hidden_states"] = self._replace(flat, target)
            return args, kwargs

        def hook(_mod, _args, _kwargs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if tensor.shape[0] == rows:
                self.outputs[name] = tensor.detach().float().cpu()

        self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def force_vision_layer(self, name: str, module: torch.nn.Module, hidden: torch.Tensor) -> None:
        """Replay a vision-tower encoder layer, whose input is positional.

        The tower's layers take ``hidden_states`` as their first positional
        argument (not a keyword), and the production tower runs flat across the
        batch, so the source capture is flattened to ``[patches, hidden]``
        before it is substituted.
        """
        flat = hidden.reshape(-1, hidden.shape[-1])
        rows = flat.shape[0]

        def pre_hook(_mod, args, kwargs):
            target = kwargs.get("hidden_states")
            positional = target is None
            if positional:
                if not args:
                    return None
                target = args[0]
            self._note(name, target.reshape(-1, target.shape[-1]).shape[0])
            if target.reshape(-1, target.shape[-1]).shape[0] != rows:
                return None
            replacement = self._replace(flat, target).reshape(target.shape)
            if positional:
                return (replacement,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = replacement
            return args, kwargs

        def hook(_mod, _args, _kwargs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if tensor.reshape(-1, tensor.shape[-1]).shape[0] == rows:
                self.outputs[name] = tensor.detach().float().cpu().reshape(rows, -1)

        self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def force_moe(
        self, name: str, module: torch.nn.Module, moe_in: torch.Tensor, router_in: torch.Tensor
    ) -> None:
        moe_flat = moe_in.reshape(-1, moe_in.shape[-1])
        router_flat = router_in.reshape(-1, router_in.shape[-1])
        rows = moe_flat.shape[0]

        def pre_hook(_mod, args, kwargs):
            if len(args) < 2:
                return None
            self._note(name, args[0].shape[0])
            if args[0].shape[0] != rows:
                return None
            new_args = (
                self._replace(moe_flat, args[0]),
                self._replace(router_flat, args[1]),
            ) + tuple(args[2:])
            return new_args, kwargs

        def hook(_mod, _args, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if tensor.shape[0] == rows:
                self.outputs[name] = tensor.detach().float().cpu()

        self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._handles.append(module.register_forward_hook(hook))

    def record(self, name: str, module: torch.nn.Module, *, expect_rows: int) -> None:
        def hook(_mod, _args, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self._note(name, tensor.shape[0])
            if tensor.shape[0] == expect_rows:
                self.outputs[name] = tensor.detach().float().cpu()

        self._handles.append(module.register_forward_hook(hook))

    def __enter__(self) -> "ReplayHarness":
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# --------------------------------------------------------------------------
# Reference-helper regressions (CPU)
# --------------------------------------------------------------------------


class TestGemma4H200ReferenceHelpers:
    """Guards on the helpers the replay/parity gates are measured with."""

    def test_metric_zero_norm_is_not_a_free_pass(self):
        """A collapsed-to-zero tensor must not score cosine 1.0.

        Regression for a real defect: cosine was defined as 1.0 whenever the
        product of the norms was zero, so an all-zero *actual* against a
        non-zero *expected* reported ``cosine=1.0`` and every ``assert_cosine``
        gate passed.  A dropped attention mask, an unwritten KV page, and an
        uninitialized CUDA-graph output buffer all surface as exactly that.
        """
        zeros = torch.zeros(8)
        ones = torch.ones(8)

        both_zero = compare_tensors(zeros, zeros.clone())
        assert both_zero.cosine == 1.0, "two all-zero tensors are identical"
        assert both_zero.max_abs == 0.0

        collapsed = compare_tensors(zeros, ones)
        assert collapsed.cosine == 0.0, (
            f"zero-vs-nonzero must score as a mismatch, got cosine={collapsed.cosine}"
        )
        assert collapsed.max_abs == 1.0
        assert compare_tensors(ones, zeros).cosine == 0.0

        with pytest.raises(AssertionError):
            assert_cosine(zeros, ones, label="collapsed-output", min_cosine=MIN_COSINE)

        # The ordinary path is unaffected.
        assert compare_tensors(ones, ones * 1.0001).cosine > 0.999999

    def test_contiguous_image_runs_are_measured_not_assumed(self):
        """Image-block bookkeeping follows the token types it is given."""
        token_types = torch.tensor([0, 1, 1, 1, 1, 0, 1, 1, 0])
        assert contiguous_image_runs(token_types) == [(1, 4), (6, 2)]
        assert contiguous_image_runs(torch.zeros(5, dtype=torch.long)) == []
        assert contiguous_image_runs(torch.ones(3, dtype=torch.long)) == [(0, 3)]

    def test_reference_tie_is_bounded_by_the_format_not_by_the_data(self):
        """The tie exemption must stay inside bfloat16's own grid.

        Criterion 6 was amended by explicit human decision in iteration 10 to
        accept, at steps the reference does not resolve at its own storage
        resolution, either of the reference's two candidates.  The danger of
        such an amendment is that the window quietly widens until it covers a
        real defect, so this pins both edges.
        """
        # bfloat16 keeps 8 significand bits: the grid at ~24 is 0.125 wide.
        assert bf16_ulp(24.0) == 0.125
        assert bf16_ulp(25.5) == 0.125
        assert bf16_ulp(1.0) == 2.0**-7
        assert bf16_ulp(0.0) == math.inf
        assert REFERENCE_TIE_ULPS == 2.0, "widening this window needs a human decision, not an edit"

        vocab = 4096
        first, second, other = 100, 200, 300

        def row(gap: float) -> torch.Tensor:
            values = torch.full((vocab,), -50.0)
            values[first] = 24.0
            values[second] = 24.0 - gap
            values[other] = 10.0
            return values

        one_ulp, two_ulp = 0.125, 0.250
        # Inside the window, and the runtime picked the reference's runner-up.
        tie = reference_tie(row(one_ulp), second)
        assert tie is not None and tie["gap_in_ulp"] == pytest.approx(1.0)
        assert tie["runtime_token_is_source_runner_up"] is True
        assert reference_tie(row(two_ulp), second)["gap_in_ulp"] == pytest.approx(2.0)
        # The runtime agreeing with the reference is also inside the window.
        assert reference_tie(row(one_ulp), first) is not None

        # One grid step beyond the window is a divergence, not a tie.
        assert reference_tie(row(two_ulp + 0.125), second) is None
        # A wide gap is a divergence however close the tokens look.
        assert reference_tie(row(4.0), second) is None
        # Landing on neither candidate is never exempt, however tight the tie.
        assert reference_tie(row(0.0), other) is None
        assert reference_tie(row(one_ulp), other) is None

        # The window is relative, so it must scale with magnitude rather than
        # being an absolute logit tolerance in disguise.
        small = torch.full((vocab,), -50.0)
        small[first], small[second] = 1.0, 1.0 - 0.125
        assert reference_tie(small, second) is None, (
            "0.125 is 16 ULP at magnitude 1.0 and must not be exempt just because "
            "it is exempt at magnitude 24"
        )

    def test_teacher_forced_outcome_requires_the_trtllm_token(self):
        """A third token must not be reported as reproducing the divergence.

        Regression for a real defect in this file: the predicate was
        ``forced_argmax != source_token``, so *any* argmax other than the
        source's counted as "reproduces from a single forward" -- including one
        that matched neither side.  On the measured ``image[1]`` step-31 case
        the teacher-forced pass returned 236764 against source 607 and
        TensorRT-LLM 531, and the report claimed a reproduction of a divergence
        that pass never produced.  The diagnostic is what a human reads to
        decide where to look next, so a false localization is worse than none.
        """
        source, trtllm = 607, 531

        assert teacher_forced_outcome(trtllm, source, trtllm) == "reproduces_trtllm_token"
        assert teacher_forced_outcome(source, source, trtllm) == "matches_source_token"
        assert teacher_forced_outcome(236764, source, trtllm) == "third_token"
        # No divergence to reproduce: agreeing with both sides is not evidence
        # of a model-math failure.
        assert teacher_forced_outcome(source, source, source) == "matches_source_token"

        vocab = 262144
        expected = torch.full((vocab,), -10.0)
        expected[source], expected[trtllm] = 2.0, 1.75
        actual = torch.full((vocab,), -10.0)
        actual[trtllm], actual[source] = 2.0, 1.75

        def report_for(forced_token: int) -> Dict[str, Any]:
            teacher = torch.full((1, vocab), -10.0)
            teacher[0, forced_token] = 5.0
            return step_divergence_report(0, trtllm, expected, actual, teacher)

        third = report_for(236764)
        assert third["teacher_forced_argmax"] == 236764
        assert third["teacher_forced_outcome"] == "third_token"
        assert third["reproduces_from_single_forward"] is False

        reproduced = report_for(trtllm)
        assert reproduced["teacher_forced_outcome"] == "reproduces_trtllm_token"
        assert reproduced["reproduces_from_single_forward"] is True

        matched = report_for(source)
        assert matched["teacher_forced_outcome"] == "matches_source_token"
        assert matched["reproduces_from_single_forward"] is False

        # The rest of the report is unchanged by the classification.
        assert third["source_token"] == source
        assert third["trtllm_token"] == trtllm
        assert third["source_top2"] == [source, trtllm]
        assert third["source_top2_margin"] == pytest.approx(0.25)
        # Without a teacher-forced pass the keys are absent rather than false.
        bare = step_divergence_report(0, trtllm, expected, actual, None)
        assert "teacher_forced_outcome" not in bare
        assert "reproduces_from_single_forward" not in bare

    def test_forced_decode_records_the_model_row_and_pins_the_argmax(self):
        """The recovery's forcing must not corrupt the evidence it records.

        ``ForcedDecode`` is what turns criterion 6's post-tie recovery into a
        real cached decode instead of a context prefill, so two things have to
        hold at once, and they pull against each other:

        * the recorded row is the model's own output, taken *before* the
          rewrite -- otherwise the recovery would compare the reference against
          a row this test file wrote; and
        * the rewrite reaches the sampler, i.e. it lands in the caller's tensor
          (the engine hands over a *view* of its logits buffer) and makes the
          intended token the strict argmax.

        The bump is deliberately not a ``-inf`` mask of every other candidate:
        that would destroy the row a later step might need and can hand a NaN to
        anything downstream that normalizes.
        """
        vocab, forced = 512, [7, 3, 11]
        processor = ForcedDecode(forced)
        rows = torch.zeros(len(forced) + 1, 1, 1, vocab)
        # A row whose incumbent argmax is never the forced token, plus one
        # bfloat16 row to prove the bump survives a coarse grid.
        for step in range(rows.shape[0]):
            rows[step, 0, 0].uniform_(-4.0, 4.0, generator=torch.Generator().manual_seed(step))
            rows[step, 0, 0, 100 + step] = 24.0
        bf16_row = rows[1].to(torch.bfloat16)

        recorded_before = [rows[step, 0, 0].clone() for step in range(rows.shape[0])]
        for step in range(rows.shape[0]):
            row = bf16_row if step == 1 else rows[step]
            processor(0, row, [[0] * (5 + step)], None, None)

        assert len(processor.rows) == rows.shape[0]
        assert processor.forced_steps == len(forced), "one rewrite per forced step, and no more"
        # Recorded rows are the model's, pre-rewrite.
        assert torch.equal(processor.rows[0], recorded_before[0])
        assert torch.equal(processor.rows[2], recorded_before[2])
        assert torch.equal(processor.rows[1], recorded_before[1].to(torch.bfloat16).float())
        # ... and the rewrite landed in the caller's tensor, not a copy.
        assert int(rows[0, 0, 0].argmax()) == forced[0]
        assert int(bf16_row[0, 0].argmax()) == forced[1]
        assert int(rows[2, 0, 0].argmax()) == forced[2]
        # Past the forced horizon the row is left exactly as the model produced it.
        assert torch.equal(rows[3, 0, 0], recorded_before[3])
        assert int(rows[3, 0, 0].argmax()) == 103
        # Only the forced entry moved.
        untouched = torch.ones(vocab, dtype=torch.bool)
        untouched[forced[0]] = False
        assert torch.equal(rows[0, 0, 0][untouched], recorded_before[0][untouched])
        assert processor.observed_token_counts == [5, 6, 7, 8]


# --------------------------------------------------------------------------
# Runtime nodes
# --------------------------------------------------------------------------


class TestGemma4H200Runtime(LlmapiAccuracyTestHarness):
    """Loader, dispatch, cache, and chunked-prefill behaviour on one H200."""

    MODEL_NAME = MODEL_NAME

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_native_bf16_load_and_dispatch(self, enabled: bool):
        env = require_single_h200()
        label = f"load_and_dispatch[cuda_graph={enabled}]"

        with RuntimeProbe() as probe, gemma4_llm(enabled=enabled) as llm:
            model = model_of(llm)
            assert type(model).__name__ == "Gemma4ForConditionalGeneration", (
                f"{label}: loaded {type(model).__name__}"
            )
            assert str(llm.args.backend) == "pytorch", f"{label}: backend={llm.args.backend}"
            assert llm.args.tensor_parallel_size == 1
            # The authoritative FLASHINFER evidence is that
            # ``FlashInferAttention.forward`` actually ran -- asserted by
            # ``probe.assert_sm90_dispatch`` below -- not the config string,
            # which ``apply_model_defaults_to_llm_args`` may or may not have
            # written back onto this object.
            assert llm.args.quant_config.quant_algo is None, (
                f"{label}: expected unquantized BF16 weights, got "
                f"{llm.args.quant_config.quant_algo}"
            )
            assert llm.args.quant_config.kv_cache_quant_algo is None, (
                f"{label}: expected a BF16 KV cache, got "
                f"{llm.args.quant_config.kv_cache_quant_algo}"
            )

            assert_kv_cache_manager_v2(llm, label=label)
            descriptors = pool_descriptors(llm)
            assert descriptors, f"{label}: KVCacheManagerV2 exposed no per-layer pools"
            for layer_idx, desc in descriptors.items():
                assert desc["dtype"] == "torch.bfloat16", (
                    f"{label}: layer {layer_idx} KV pool dtype {desc['dtype']}, expected bfloat16"
                )

            # A silently dropped or partially loaded tensor shows up here long
            # before it shows up as a bad benchmark score.
            state_dict = model.state_dict()
            nonfinite = [
                name
                for name, tensor in state_dict.items()
                if tensor.is_floating_point() and not torch.isfinite(tensor).all()
            ]
            assert not nonfinite, f"{label}: non-finite loaded tensors: {nonfinite[:8]}"

            accounting = state_dict_accounting(model)
            assert accounting["unexplained"] == [], (
                f"{label}: unexplained checkpoint tensors "
                f"({len(accounting['unexplained'])}): {accounting['unexplained'][:8]}"
            )
            # One destination per matched source tensor: if these ever diverge
            # the matcher has started crediting one loaded tensor to several
            # checkpoint tensors, which is exactly how a dropped weight hides.
            assert accounting["distinct_destinations"] == accounting["copied"], (
                f"{label}: {accounting['copied']} checkpoint tensors matched only "
                f"{accounting['distinct_destinations']} distinct loaded tensors"
            )
            assert accounting["loaded_expert_numel"] >= accounting["source_expert_numel"], (
                f"{label}: the fused MoE holds {accounting['loaded_expert_numel']} expert "
                f"elements but the checkpoint supplies {accounting['source_expert_numel']}; "
                "expert weights were dropped"
            )
            padding = assert_vision_padding_is_zero(model, label=label)

            outputs = llm.generate(
                [token_prompt(render_text(p)) for p in TEXT_PROMPTS[:2]],
                sampling_params=greedy(24),
            )
            texts = [o.outputs[0].text for o in outputs]
            for text in texts:
                assert text.strip(), f"{label}: empty generation"

            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)

        report["generations"] = texts
        report["state_dict_accounting"] = accounting
        report["vision_head_dim_padding"] = padding
        report["checkpoint"] = env["checkpoint"]
        write_evidence(f"load_and_dispatch-cg{int(enabled)}", report)

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_checkpoint_scale_attention_matrix(self, enabled: bool):
        """Sliding (16/8, d256, win 1024) and full (16/2, d512) really dispatch on H200."""
        require_single_h200()
        label = f"attention_matrix[cuda_graph={enabled}]"
        image_capture = ensure_capture("image_replay")
        image_case = image_capture["prompts"][0]

        with RuntimeProbe() as probe, gemma4_llm(enabled=enabled, max_batch_size=4) as llm:
            manager = assert_kv_cache_manager_v2(llm, label=label)
            config = model_of(llm).config
            text_config = getattr(config, "text_config", config)

            descriptors = pool_descriptors(llm)
            sliding_pools, full_pools = set(), set()
            sliding_ptrs, full_ptrs = set(), set()
            for layer_idx, desc in descriptors.items():
                is_sliding = text_config.layer_types[layer_idx] == "sliding_attention"
                geometry = (desc["num_kv_heads"], desc["head_dim"])
                (sliding_pools if is_sliding else full_pools).add(geometry)
                (sliding_ptrs if is_sliding else full_ptrs).add(desc["data_ptr"])
            assert sliding_pools == {(8, 256)}, f"{label}: sliding pools {sliding_pools}"
            assert full_pools == {(2, 512)}, f"{label}: full pools {full_pools}"
            assert not (sliding_ptrs & full_ptrs), (
                f"{label}: sliding and full layers alias the same KV pool memory"
            )

            free_before = int(manager.get_num_free_blocks())

            # Text request: prefill followed by multi-step cached decode.
            text_ids = reference_text_ids()[0]
            text_out = llm.generate(token_prompt(text_ids), sampling_params=greedy(16))
            assert text_out.outputs[0].text.strip(), f"{label}: text request produced no text"

            # Image request: exercises the source-equivalent bidirectional mask.
            image_out = llm.generate(
                image_prompt(render_image_text(image_case["prompt"]), [image_case["image"]]),
                sampling_params=greedy(16),
            )
            assert image_out.outputs[0].text.strip(), f"{label}: image request produced no text"

            geometries = probe.geometries()
            assert (16, 8, 256) in geometries, (
                f"{label}: no sliding-geometry attention call; observed {sorted(geometries)}"
            )
            assert (16, 2, 512) in geometries, (
                f"{label}: no full-geometry attention call; observed {sorted(geometries)}"
            )
            custom_mask_calls = [c for c in probe.attention_calls if c.custom_mask]
            assert custom_mask_calls, (
                f"{label}: the image request never reached attention with a custom mask, so "
                "the bidirectional image block was silently dropped"
            )
            assert any(c.num_generations > 0 for c in probe.attention_calls), (
                f"{label}: no cached-decode attention call was observed"
            )
            assert any(c.num_contexts > 0 for c in probe.attention_calls), (
                f"{label}: no prefill attention call was observed"
            )

            # Page-index bounds: every index FlashInfer dereferences must
            # address a real page of that layer's own pool.  This is the check
            # that turns "the pools are distinct" into "nothing reads across
            # them", and it is what an out-of-range or BAD_PAGE_INDEX entry
            # trips before it becomes an illegal memory access.
            page_bounds = probe.assert_page_indices_in_bounds(label=label)
            expected_pool_ids = set()
            for call in probe.attention_calls:
                if call.kv_pool_id is not None:
                    expected_pool_ids.add(call.kv_pool_id)
            assert len(expected_pool_ids) >= 2, (
                f"{label}: sliding and full layers reported {len(expected_pool_ids)} distinct "
                f"KV pool id(s) {sorted(expected_pool_ids)}; Gemma 4's two geometries must not "
                "share one page-index space"
            )
            sliding_pool_ids = {
                c.kv_pool_id for c in probe.attention_calls if c.key() == (16, 8, 256)
            }
            full_pool_ids = {c.kv_pool_id for c in probe.attention_calls if c.key() == (16, 2, 512)}
            assert sliding_pool_ids and full_pool_ids and not (sliding_pool_ids & full_pool_ids), (
                f"{label}: sliding layers used pools {sorted(sliding_pool_ids)} and full layers "
                f"{sorted(full_pool_ids)}; they must be disjoint"
            )
            phases = probe.pools_by_phase()
            assert set(phases["prefill"]) == expected_pool_ids, (
                f"{label}: prefill touched pools {phases['prefill']} but the model has "
                f"{sorted(expected_pool_ids)}"
            )
            assert set(phases["decode"]) == expected_pool_ids, (
                f"{label}: cached decode reused pools {phases['decode']} but the model has "
                f"{sorted(expected_pool_ids)}; a geometry lost its cache between prefill and decode"
            )

            # Request turnover must return every page: a leak here is what
            # later shows up as an out-of-pages failure mid-benchmark.  Release
            # is asynchronous with respect to the completed future, so poll
            # briefly rather than racing it.
            free_after = wait_for_free_blocks(manager, free_before)
            assert free_after >= free_before, (
                f"{label}: {free_before - free_after} KV blocks were still held 30 s after "
                "both requests completed"
            )

            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)
            report["pool_descriptors"] = {
                str(k): {kk: vv for kk, vv in v.items() if kk != "data_ptr"}
                for k, v in descriptors.items()
            }
            report["distinct_pools"] = len(sliding_ptrs | full_ptrs)
            report["custom_mask_layers"] = sorted({c.layer_idx for c in custom_mask_calls})
            report["free_blocks"] = {"before": free_before, "after": free_after}
            report["page_index_bounds"] = page_bounds
            report["pools_by_phase"] = phases
            report["sliding_pool_ids"] = sorted(sliding_pool_ids)
            report["full_pool_ids"] = sorted(full_pool_ids)

        write_evidence(f"attention_matrix-cg{int(enabled)}", report)

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_chunked_prefill_long_horizon(self, enabled: bool):
        """>=3 prefill chunks on a >=2048-token prompt, then >=64 cached decode steps."""
        require_single_h200()
        label = f"chunked_prefill[cuda_graph={enabled}]"

        image_capture = ensure_capture("image_replay")
        image_case = image_capture["prompts"][0]
        # The fixed replay prompt puts its image five tokens in, and the V2
        # scheduler snaps *down* to a KV-page boundary: 5 // 32 == 0, so the
        # snapped chunk would be empty and the request is deferred forever
        # (measured: the executor holds its memory at 0% GPU with no progress).
        # A text prefix moves the block past a page boundary, which is what
        # makes snap-down a legal move and the canary a test of the snap rather
        # than of that edge case.
        canary_question = IMAGE_CANARY_PREFIX + IMAGE_CANARY_QUESTION
        # Text before the image, so the block starts well past a KV page.
        canary_text = render_image_text(canary_question, text_first=True)
        run_start, run_end, canary_ids = measured_image_block(canary_text, image_case["image"])
        chunk_tokens = image_chunk_budget([(run_start, run_end - run_start)])
        # The point of the image half of this canary: a chunk boundary that
        # falls *inside* the bidirectional soft-token block, so the scheduler
        # has to snap it rather than never meeting the block at all.
        assert run_start < chunk_tokens < run_end, (
            f"{label}: chunk budget {chunk_tokens} does not fall inside the measured image "
            f"block [{run_start}, {run_end}); it cannot prove the block is kept intact"
        )
        # Token parity for the canary image request needs its own reference,
        # and -- see IMAGE_CANARY_MIN_REFERENCE_MARGIN -- the reference's own
        # per-step margins, so this node can prove it is asserting parity on
        # steps the reference actually decides.
        image_reference = ensure_native_completions(
            {
                "items": [
                    {
                        "id": "image_chunk_canary",
                        "text": canary_text,
                        "images": [image_case["image"]],
                    }
                ],
                "max_new_tokens": IMAGE_CANARY_DECODE_STEPS,
                "record_margins": True,
            },
            tag="imgchunk",
        )["completions"][0]
        # Validity precondition, checked before anything is run: exact greedy
        # token parity is only evidence about *chunking* at steps the reference
        # resolves by more than the runtime's numerical spread.  Chunked and
        # unchunked prefill are mathematically equivalent but numerically
        # distinct, so a step decided by less than that spread would flip
        # without any defect.  Fail as an invalid canary rather than flake.
        canary_margins = image_reference["step_top2_margin"][:IMAGE_CANARY_DECODE_STEPS]
        assert len(canary_margins) >= IMAGE_CANARY_DECODE_STEPS, (
            f"{label}: the reference produced only {len(canary_margins)} of "
            f"{IMAGE_CANARY_DECODE_STEPS} canary steps"
        )
        indecisive = [
            (step, margin)
            for step, margin in enumerate(canary_margins)
            if margin < IMAGE_CANARY_MIN_REFERENCE_MARGIN
        ]
        assert not indecisive, (
            f"{label}: the reference does not decide "
            f"{[(s, round(m, 4)) for s, m in indecisive]} by the required "
            f"{IMAGE_CANARY_MIN_REFERENCE_MARGIN} logits, so exact token parity there would be a "
            "coin flip between two numerically-equivalent prefill paths rather than evidence "
            "about chunk boundaries. Pick a canary question the reference resolves "
            "(repro/canary_determinacy.py measures candidates)."
        )

        long_ids = long_context_ids(LONG_CONTEXT_TOKENS)
        expected_chunks = -(-len(long_ids) // chunk_tokens)
        assert expected_chunks >= MIN_PREFILL_CHUNKS, (
            f"{label}: {len(long_ids)} tokens / {chunk_tokens} = {expected_chunks} chunks"
        )
        # Token parity for the long request needs its own native reference.
        long_reference = ensure_native_completions(
            {
                "items": [{"id": "long_context", "input_ids": long_ids, "images": []}],
                "max_new_tokens": LONG_CONTEXT_DECODE_STEPS,
            },
            tag=f"long{len(long_ids)}",
        )["completions"][0]

        with (
            RuntimeProbe() as probe,
            gemma4_llm(
                enabled=enabled,
                max_batch_size=4,
                max_num_tokens=chunk_tokens,
                enable_chunked_prefill=True,
            ) as llm,
        ):
            assert llm.args.enable_chunked_prefill
            assert_kv_cache_manager_v2(llm, label=label)

            manager = kv_cache_manager_of(llm)
            idle_blocks = int(manager.get_num_free_blocks())

            out = llm.generate(
                token_prompt(long_ids), sampling_params=greedy(LONG_CONTEXT_DECODE_STEPS)
            )
            actual_tokens = list(out.outputs[0].token_ids)
            assert len(actual_tokens) == LONG_CONTEXT_DECODE_STEPS, (
                f"{label}: expected {LONG_CONTEXT_DECODE_STEPS} decode steps, "
                f"got {len(actual_tokens)}"
            )
            assert_token_parity(
                actual_tokens, long_reference["tokens"], label=f"{label} long_context"
            )
            # Determinism, measured rather than assumed: the same request, run
            # again through the same engine, must produce the same tokens.  A
            # chunked prefill whose boundaries drift with the cache/budget state
            # would show up here as a differing sequence, and a canary that is
            # only sometimes right is not evidence.
            repeat = llm.generate(
                token_prompt(long_ids), sampling_params=greedy(LONG_CONTEXT_DECODE_STEPS)
            )
            assert list(repeat.outputs[0].token_ids) == actual_tokens, (
                f"{label}: the long-context request is not deterministic -- a repeat run of the "
                f"identical prompt produced {list(repeat.outputs[0].token_ids)} instead of "
                f"{actual_tokens}"
            )

            layer0_context_calls = [
                c for c in probe.attention_calls if c.num_contexts > 0 and c.layer_idx == 0
            ]
            assert len(layer0_context_calls) >= MIN_PREFILL_CHUNKS, (
                f"{label}: prefill used {len(layer0_context_calls)} chunks for "
                f"{len(long_ids)} tokens at max_num_tokens={chunk_tokens}; expected >= "
                f"{MIN_PREFILL_CHUNKS}"
            )
            # 2048 tokens through a >=3-chunk prefill and 64 decode steps is
            # also the cheapest place the sliding pool outgrows its 1024-token
            # window, so every page index must still be a live page.
            probe.assert_page_indices_in_bounds(label=f"{label} long_context")

            # The long request's pages are released asynchronously once its
            # response is returned.  Starting the image request while they are
            # still held changes the budget the scheduler sees on its first
            # chunk, which is one way the same canary produced different chunk
            # boundaries -- and different tokens -- between runs.  Wait for the
            # cache to come back to idle so the image request always starts from
            # the same state.
            free_now = wait_for_free_blocks(manager, idle_blocks)
            assert free_now >= idle_blocks, (
                f"{label}: the long request's pages were not released ({free_now} free, "
                f"{idle_blocks} at idle); the image request would not start deterministically"
            )

            image_probe_start = len(probe.attention_calls)
            image_out = llm.generate(
                image_prompt(canary_text, [image_case["image"]]),
                sampling_params=greedy(IMAGE_CANARY_DECODE_STEPS),
            )
            assert image_out.outputs[0].text.strip(), f"{label}: image request made no progress"
            image_tokens = list(image_out.outputs[0].token_ids)
            assert_token_parity(image_tokens, image_reference["tokens"], label=f"{label} image")
            image_calls = probe.attention_calls[image_probe_start:]
            assert [c for c in image_calls if c.custom_mask], (
                f"{label}: the image request lost its bidirectional custom mask"
            )
            # The block must be kept whole: with the budget landing inside the
            # run, the scheduler has to snap the chunk short of ``run_start``
            # or defer past ``run_end``.  A prefill chunk whose context token
            # count stops strictly between them means the bidirectional block
            # was cut in half.
            image_context_chunks = [
                c.num_ctx_tokens for c in image_calls if c.num_contexts > 0 and c.layer_idx == 0
            ]
            assert image_context_chunks, f"{label}: the image request never ran a prefill chunk"
            consumed = 0
            split_boundaries = []
            for chunk in image_context_chunks:
                consumed += chunk
                if run_start < consumed < run_end:
                    split_boundaries.append(consumed)
            assert not split_boundaries, (
                f"{label}: prefill chunk boundaries {split_boundaries} fall inside the "
                f"bidirectional image block [{run_start}, {run_end}); the block was split"
            )
            assert consumed >= run_end, (
                f"{label}: the image prefill consumed only {consumed} tokens, so it never "
                f"finished the image block ending at {run_end} -- no progress"
            )
            probe.assert_page_indices_in_bounds(label=f"{label} image")

            # Same determinism check for the multimodal half: the snap has to
            # land in the same place every time, not merely often enough.
            wait_for_free_blocks(manager, idle_blocks)
            image_repeat = llm.generate(
                image_prompt(canary_text, [image_case["image"]]),
                sampling_params=greedy(IMAGE_CANARY_DECODE_STEPS),
            )
            assert list(image_repeat.outputs[0].token_ids) == image_tokens, (
                f"{label}: the chunked image request is not deterministic -- a repeat run "
                f"produced {list(image_repeat.outputs[0].token_ids)} instead of {image_tokens}"
            )

            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)
            report.update(
                {
                    "chunk_tokens": chunk_tokens,
                    "prompt_tokens": len(long_ids),
                    "observed_prefill_chunks": len(layer0_context_calls),
                    "decode_steps": len(actual_tokens),
                    "measured_image_run": [run_start, run_end],
                    "image_prompt_tokens": len(canary_ids),
                    "image_prefill_chunks": image_context_chunks,
                    "configured_soft_tokens_per_image": image_capture[
                        "configured_soft_tokens_per_image"
                    ],
                    # What makes the image half's token-parity assertion mean
                    # something: the reference's own resolution at every step it
                    # is asserted over.
                    "image_canary": {
                        "question": IMAGE_CANARY_QUESTION,
                        "reference_text": image_reference["text"],
                        "reference_step_top2_margin": canary_margins,
                        "reference_min_margin": min(canary_margins),
                        "required_min_margin": IMAGE_CANARY_MIN_REFERENCE_MARGIN,
                        "tokens": image_tokens,
                        "source_tokens": list(image_reference["tokens"]),
                    },
                }
            )

        write_evidence(f"chunked_prefill-cg{int(enabled)}", report)

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_decode_is_bitwise_reproducible(self, enabled: bool):
        """The same request twice through one engine must give the same bits.

        Regression for a defect that silently cost this port more accuracy than
        any modelling bug found here. The CUTLASS MoE FC2+finalize fusion
        accumulates a token's expert contributions in an order that is not fixed
        between launches, and ``MoeConfig.disable_finalize_fusion`` documents it
        as nondeterministic for ``top_k > 2`` -- Gemma 4 26B-A4B routes top-k 8.
        With the fusion on, two identical single-request forwards through one
        engine first diverged at ``model.layers.0.moe.experts``, then at 704
        downstream modules, reaching a final-logit max-abs of ~1.5: larger than
        this path's entire disagreement with Transformers on text prompts, and
        enough to flip greedy tokens between repeats of the same run.

        Criterion 1 asks for deterministic generation and criterion 6 compares a
        fixed greedy token at every step; neither is well posed against a
        runtime that does not reproduce itself, so this is asserted directly
        rather than assumed. ``gemma4_llm`` supplies the deterministic MoE
        configuration (see ``_moe_config``); the assertion below is what keeps
        it from being dropped.

        Bitwise equality is deliberate. A tolerance here would pass exactly the
        state this test exists to reject.
        """
        require_single_h200()
        label = f"decode_reproducible[cuda_graph={enabled}]"
        capture = ensure_capture("text_replay")
        case = capture["prompts"][0]
        steps = 32
        params = greedy(steps + 1, logits=True, suppress_eos=True)

        with RuntimeProbe() as probe, gemma4_llm(enabled=enabled, max_batch_size=4) as llm:
            assert_kv_cache_manager_v2(llm, label=label)

            runs: List[Dict[str, Any]] = []
            for repeat in range(3):
                out = llm.generate(token_prompt(case["input_ids"]), sampling_params=params)
                runs.append(
                    {
                        "tokens": list(out.outputs[0].token_ids)[:steps],
                        "logits": generation_logits(out)[:steps],
                    }
                )

            # Every MoE layer must actually be running the deterministic
            # epilogue -- read off the live backend, so a silently changed
            # default fails here rather than as a flaky token much later.
            fused = [
                entry["use_fused_finalize"]
                for entry in moe_resolution_reports(llm)
                if entry.get("use_fused_finalize") is not None
            ]
            assert fused and not any(fused), (
                f"{label}: expected the deterministic (unfused) MoE finalize path, "
                f"got use_fused_finalize={fused}"
            )

            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)

            mismatches: List[str] = []
            for index, run in enumerate(runs[1:], start=1):
                if run["tokens"] != runs[0]["tokens"]:
                    first = next(
                        (
                            step
                            for step, (a, b) in enumerate(zip(runs[0]["tokens"], run["tokens"]))
                            if a != b
                        ),
                        None,
                    )
                    mismatches.append(f"repeat {index}: token differs first at step {first}")
                if not torch.equal(run["logits"], runs[0]["logits"]):
                    delta = (run["logits"] - runs[0]["logits"]).abs()
                    mismatches.append(
                        f"repeat {index}: logits not bitwise equal, max_abs {float(delta.max()):.6f}"
                    )

            report = runtime_report(llm, probe, enabled=enabled)
            report.update(
                {
                    "repeats": len(runs),
                    "compared_steps": steps,
                    "use_fused_finalize": fused,
                    "bitwise_reproducible": not mismatches,
                    "mismatches": mismatches,
                }
            )

        write_evidence(f"decode_reproducible-cg{int(enabled)}", report)
        assert not mismatches, f"{label}: " + "; ".join(mismatches)


# --------------------------------------------------------------------------
# Parity nodes
# --------------------------------------------------------------------------


class TestGemma4H200Parity(LlmapiAccuracyTestHarness):
    """Source-activation, logit, and generation parity against Transformers 5.5.4."""

    MODEL_NAME = MODEL_NAME

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_source_activation_replay(self, enabled: bool):
        require_single_h200()
        label = f"source_activation_replay[cuda_graph={enabled}]"
        text_capture = ensure_capture("text_replay")
        image_capture = ensure_capture("image_replay")

        case = text_capture["prompts"][0]
        decode_layer_indices = [r["layer_idx"] for r in case["decode"]["layers"].values()]
        decode_hidden = int(next(iter(case["decode"]["layers"].values()))["attn_in"].shape[-1])

        metrics: Dict[str, Any] = {}
        with (
            RuntimeProbe() as probe,
            # Armed before the engine exists so its copies land inside every
            # captured CUDA graph; see GraphSafeDecodeCapture.
            GraphSafeDecodeCapture(decode_layer_indices, decode_hidden) as decode_capture,
            gemma4_llm(enabled=enabled, max_batch_size=2) as llm,
        ):
            assert_kv_cache_manager_v2(llm, label=label)
            layers = language_layers(llm)

            # --- text: early/late sliding and early/late full layers ---------
            with ReplayHarness() as harness:
                for name, record in case["layers"].items():
                    harness.force_attention(
                        f"attn:{name}", layers[record["layer_idx"]].self_attn, record["attn_in"]
                    )
                llm.generate(token_prompt(case["input_ids"]), sampling_params=greedy(4))
                for name, record in case["layers"].items():
                    actual = harness.outputs.get(f"attn:{name}")
                    assert actual is not None, (
                        f"{label}: {harness.missed(f'attn:{name}', record['attn_in'].shape[-2])}"
                    )
                    metrics[f"text_attention[{name}]"] = assert_cosine(
                        actual,
                        record["attn_out"].reshape(actual.shape).float(),
                        label=f"{label} text_attention[{name}]",
                        min_cosine=MIN_COSINE,
                    ).as_dict()

            # --- text: cached decode over the paged KV cache ------------------
            # Prefill replay says nothing about whether the same layer is
            # source-equivalent once its K/V come back out of KVCacheManagerV2.
            # This is *observational*, not forced: `decode_capture` was armed
            # before the engine was built (see below), so its device-side copies
            # were recorded into every CUDA graph and re-execute on replay --
            # which is what makes the enabled matrix a real layer-level
            # comparison rather than a logits-only substitute.  `greedy(2)`
            # issues exactly one cached-decode forward, and the recorded step
            # counter is asserted to be exactly 1, so nothing can be compared
            # twice or overwritten by a later step.
            decode_record = case["decode"]
            decode_capture.reset()
            decode_out = llm.generate(
                token_prompt(case["input_ids"]),
                # Same end-of-turn suppression the reference generated under, so
                # the token fed into the decode step is the one the source fed.
                sampling_params=greedy(2, logits=True, suppress_eos=True),
            )
            decode_tokens = list(decode_out.outputs[0].token_ids)
            assert len(decode_tokens) == 2, (
                f"{label}: expected exactly 2 tokens (prefill + one cached decode), "
                f"got {len(decode_tokens)}"
            )
            observed_steps = decode_capture.steps()
            assert observed_steps == 1, (
                f"{label}: the cached-decode capture ran {observed_steps} times; the "
                "comparison is only meaningful when exactly one decode forward wrote it"
            )
            # The decode step must consume the same token the source consumed,
            # otherwise the two paths are not at the same position.
            assert decode_tokens[0] == decode_record["next_token"], (
                f"{label}: TensorRT-LLM decoded on token {decode_tokens[0]} but the source "
                f"decode capture used {decode_record['next_token']}"
            )
            decode_compared = 0
            for name, record in decode_record["layers"].items():
                captured = decode_capture.get(record["layer_idx"])
                assert captured is not None, (
                    f"{label}: no cached-decode activation was captured for layer "
                    f"{record['layer_idx']} ({name}); observed layers "
                    f"{sorted(decode_capture.captured_layers())}"
                )
                actual_in, actual_out = captured
                decode_compared += 1
                # Input first: if the cache/upstream state already differs, the
                # output metric alone would not say where it went wrong.
                metrics[f"decode_attention_in[{name}]"] = assert_cosine(
                    actual_in,
                    record["attn_in"].reshape(actual_in.shape).float(),
                    label=f"{label} decode_attention_in[{name}]",
                    min_cosine=MIN_COSINE,
                ).as_dict()
                metrics[f"decode_attention[{name}]"] = assert_cosine(
                    actual_out,
                    record["attn_out"].reshape(actual_out.shape).float(),
                    label=f"{label} decode_attention[{name}]",
                    min_cosine=MIN_COSINE,
                ).as_dict()
            assert decode_compared == len(decode_record["layers"]), (
                f"{label}: compared {decode_compared} of {len(decode_record['layers'])} "
                "cached-decode layers"
            )

            # Cached decode is additionally compared at the model boundary: step
            # 1 of a greedy generation is the first forward served entirely from
            # the paged cache, so its logits close the loop from layer to logit.
            decode_logits = generation_logits(decode_out)
            assert decode_logits.shape[0] >= 2, (
                f"{label}: the runtime returned {decode_logits.shape[0]} logit step(s); a "
                "cached-decode comparison needs at least 2"
            )
            metrics["cached_decode_logits"] = assert_cosine(
                decode_logits[1],
                case["step_logits"][1].float(),
                label=f"{label} cached_decode_logits",
                min_cosine=MIN_COSINE,
            ).as_dict()
            actual_decode_token = int(decode_logits[1].argmax().item())
            assert actual_decode_token == int(case["greedy_tokens"][1]), (
                f"{label}: cached-decode greedy token {actual_decode_token} != source "
                f"{int(case['greedy_tokens'][1])}"
            )

            # --- image: vision tower, projector, and vision-bearing prefill ---
            image_case = image_capture["prompts"][0]
            rendered = render_image_text(image_case["prompt"])
            vision_layers = vision_tower_layers(llm)
            with ReplayHarness() as harness:
                for name, record in image_case["layers"].items():
                    harness.force_attention(
                        f"img_attn:{name}", layers[record["layer_idx"]].self_attn, record["attn_in"]
                    )
                # Early/late vision-tower encoder layers replayed on the
                # production tower, plus the multimodal projector: without
                # these the image path is only ever observed through the
                # language layers that consume it.
                # The Gemma 4 processor pads every image up to a fixed patch
                # budget (2520 slots for the fixed reference image) and marks
                # the unused slots -1 in ``image_position_ids``; only 2394 of
                # them are real patches.  The native tower runs the padded
                # tensor and masks the padding inside attention, while the
                # production tower packs the real patches ragged.  Replay the
                # source's *real* patches, selected by the source's own
                # validity mask, so the comparison is over identical content —
                # matching padded slot counts would compare the processor's
                # padding budget, not the image.
                for name, record in image_case["vision_layers"].items():
                    harness.force_vision_layer(
                        f"vision:{name}",
                        vision_layers[record["layer_idx"]],
                        select_valid_patches(record["input"], record["patch_valid"]),
                    )
                projector_rows = (
                    image_case["projector_out"]
                    .reshape(-1, image_case["projector_out"].shape[-1])
                    .shape[0]
                )
                harness.record("projector", multimodal_projector(llm), expect_rows=projector_rows)
                llm.generate(
                    image_prompt(rendered, [image_case["image"]]), sampling_params=greedy(4)
                )
                for name, record in image_case["layers"].items():
                    actual = harness.outputs.get(f"img_attn:{name}")
                    assert actual is not None, (
                        f"{label}: the image prefill never reached attention layer {name} with "
                        "the reference token count, so the multimodal prompt did not render to "
                        f"the same sequence as the source -- "
                        f"{harness.missed(f'img_attn:{name}', record['attn_in'].shape[-2])}"
                    )
                    metrics[f"image_attention[{name}]"] = assert_cosine(
                        actual,
                        record["attn_out"].reshape(actual.shape).float(),
                        label=f"{label} image_attention[{name}]",
                        min_cosine=MIN_COSINE,
                    ).as_dict()
                for name, record in image_case["vision_layers"].items():
                    actual = harness.outputs.get(f"vision:{name}")
                    patch_rows = int(record["num_valid_patches"])
                    assert actual is not None, (
                        f"{label}: the vision tower never ran encoder layer {name} with the "
                        f"source's {patch_rows} real patches (of "
                        f"{int(record['num_padded_patches'])} padded slots) -- "
                        f"{harness.missed(f'vision:{name}', patch_rows)}"
                    )
                    metrics[f"vision_layer[{name}]"] = assert_cosine(
                        actual,
                        select_valid_patches(record["output"], record["patch_valid"]).reshape(
                            actual.shape
                        ),
                        label=f"{label} vision_layer[{name}]",
                        min_cosine=MIN_COSINE,
                    ).as_dict()
                    metrics[f"vision_patches[{name}]"] = {
                        "valid": patch_rows,
                        "padded_slots": int(record["num_padded_patches"]),
                        "production_rows": int(actual.shape[0]),
                    }
                projector_actual = harness.outputs.get("projector")
                assert projector_actual is not None, (
                    f"{label}: the multimodal projector never produced "
                    f"{projector_rows} soft tokens -- {harness.missed('projector', projector_rows)}"
                )
                metrics["multimodal_projector"] = assert_cosine(
                    projector_actual,
                    image_case["projector_out"].reshape(projector_actual.shape).float(),
                    label=f"{label} multimodal_projector",
                    min_cosine=MIN_COSINE,
                ).as_dict()

            assert any(c.custom_mask for c in probe.attention_calls), (
                f"{label}: the image replay ran without a custom mask"
            )
            assert any(c.num_generations > 0 for c in probe.attention_calls), (
                f"{label}: no cached-decode attention call was observed during replay"
            )
            probe.assert_page_indices_in_bounds(label=label)
            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)
            report["metrics"] = metrics
            report["text_layers"] = text_capture["layers"]
            report["vision_layers"] = image_capture["vision_layers"]
            report["image_runs"] = image_case["image_token_runs"]
            report["decode_layer_comparisons"] = decode_compared
            report["decode_cached_tokens"] = decode_record["cached_tokens"]
            report["decode_forwards_observed"] = observed_steps
            report["decode_capture"] = "graph-recorded device copies (survive CUDA-graph replay)"

        write_evidence(f"source_activation_replay-cg{int(enabled)}", report)

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_moe_source_activation_replay(self, enabled: bool):
        require_single_h200()
        label = f"moe_source_activation_replay[cuda_graph={enabled}]"
        capture = ensure_capture("text_replay")
        case = capture["prompts"][0]

        metrics: Dict[str, Any] = {}
        with RuntimeProbe() as probe, gemma4_llm(enabled=enabled, max_batch_size=2) as llm:
            layers = language_layers(llm)
            resolution = moe_resolution_reports(llm)
            assert resolution, f"{label}: no MoE implementation was resolved"
            assert_moe_dispatch(resolution, label=label)

            with ReplayHarness() as harness:
                for name, record in case["layers"].items():
                    moe = getattr(layers[record["layer_idx"]], "moe", None)
                    assert moe is not None, f"{label}: layer {record['layer_idx']} has no MoE block"
                    harness.force_moe(f"moe:{name}", moe, record["moe_in"], record["router_in"])
                    harness.record(
                        f"router:{name}", moe.router, expect_rows=record["router_in"].shape[0]
                    )
                    # The post-MoE norm is what the layer actually folds back
                    # into the residual stream; its input is the replayed
                    # expert output, so this stays a clean replay.
                    harness.record(
                        f"post_moe:{name}",
                        layers[record["layer_idx"]].post_feedforward_layernorm_2,
                        expect_rows=record["moe_in"].shape[0],
                    )
                llm.generate(token_prompt(case["input_ids"]), sampling_params=greedy(4))

                for name, record in case["layers"].items():
                    moe = layers[record["layer_idx"]].moe
                    actual = harness.outputs.get(f"moe:{name}")
                    assert actual is not None, (
                        f"{label}: {harness.missed(f'moe:{name}', record['moe_in'].shape[0])}"
                    )
                    metrics[f"expert_output[{name}]"] = assert_cosine(
                        actual,
                        record["expert_out"].reshape(actual.shape).float(),
                        label=f"{label} expert_output[{name}]",
                        min_cosine=MIN_COSINE,
                    ).as_dict()

                    post_moe = harness.outputs.get(f"post_moe:{name}")
                    assert post_moe is not None, (
                        f"{label}: {harness.missed(f'post_moe:{name}', record['moe_in'].shape[0])}"
                    )
                    metrics[f"post_moe_residual[{name}]"] = assert_cosine(
                        post_moe,
                        record["post_moe_out"].reshape(post_moe.shape).float(),
                        label=f"{label} post_moe_residual[{name}]",
                        min_cosine=MIN_COSINE,
                    ).as_dict()

                    logits = harness.outputs.get(f"router:{name}")
                    assert logits is not None, (
                        f"{label}: {harness.missed(f'router:{name}', record['router_in'].shape[0])}"
                    )
                    # Raw expert scores: TensorRT-LLM's Gemma4Router returns
                    # pre-softmax logits, and the source's router.proj output is
                    # the same boundary, so this compares before any softmax can
                    # hide a scale or ordering error.
                    metrics[f"router_logits[{name}]"] = assert_cosine(
                        logits,
                        record["router_logits"].reshape(logits.shape).float(),
                        label=f"{label} router_logits[{name}]",
                        min_cosine=MIN_COSINE,
                    ).as_dict()
                    metrics[f"router_probabilities[{name}]"] = assert_cosine(
                        torch.softmax(logits.float(), dim=-1),
                        record["router_probabilities"].reshape(logits.shape).float(),
                        label=f"{label} router_probabilities[{name}]",
                        min_cosine=MIN_COSINE,
                    ).as_dict()
                    indices, weights = moe.experts.routing_method.apply(
                        logits.to(device="cuda", dtype=torch.float32)
                    )
                    expected_index = record["top_k_index"].cpu()
                    actual_index = indices.cpu().to(expected_index.dtype)
                    if not torch.equal(actual_index, expected_index):
                        differing = int((actual_index != expected_index).sum())
                        raise AssertionError(
                            f"{label}: router[{name}] selected different experts "
                            f"({differing} of {expected_index.numel()} slots differ)"
                        )
                    metrics[f"routing_weights[{name}]"] = assert_cosine(
                        weights.float().cpu(),
                        record["top_k_weights"].float().reshape(weights.shape),
                        label=f"{label} routing_weights[{name}]",
                        min_cosine=MIN_COSINE,
                    ).as_dict()
                    metrics[f"expert_indices[{name}]"] = "exact match"

            moe_op_calls = probe.assert_moe_op_ran(label=label)
            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)
            report["metrics"] = metrics
            report["moe_resolution"] = resolution
            report["moe_op_calls"] = moe_op_calls
            report["moe_op_path"] = MOE_EXPECTED_OP
            report["package_provenance"] = package_provenance()

        write_evidence(f"moe_source_activation_replay-cg{int(enabled)}", report)

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_source_logit_replay(self, enabled: bool):
        """Final logits and greedy-argmax token: 5 fixed text + 2 fixed image prompts."""
        require_single_h200()
        label = f"source_logit_replay[cuda_graph={enabled}]"
        text_capture = ensure_capture("text_replay")
        image_capture = ensure_capture("image_replay")

        results: List[Dict[str, Any]] = []
        with RuntimeProbe() as probe, gemma4_llm(enabled=enabled, max_batch_size=4) as llm:
            assert_kv_cache_manager_v2(llm, label=label)

            # Two steps, not one: the overlap scheduler can drop the *last*
            # step's generation logits, so asking for a single token can return
            # an empty logit tensor.  Step 0 is the prefill's last-position
            # distribution either way.
            for idx, case in enumerate(text_capture["prompts"]):
                out = llm.generate(
                    token_prompt(case["input_ids"]), sampling_params=greedy(2, logits=True)
                )
                results.append(
                    compare_final_logits(
                        generation_logits(out)[0],
                        case["prefill_last_logits"],
                        label=f"{label} text[{idx}]",
                        extra={"prompt": case["prompt"], "prompt_tokens": len(case["input_ids"])},
                    )
                )

            for idx, case in enumerate(image_capture["prompts"]):
                out = llm.generate(
                    image_prompt(render_image_text(case["prompt"]), [case["image"]]),
                    sampling_params=greedy(2, logits=True),
                )
                results.append(
                    compare_final_logits(
                        generation_logits(out)[0],
                        case["prefill_last_logits"],
                        label=f"{label} image[{idx}]",
                        extra={
                            "image": case["image"],
                            "prompt": case["prompt"],
                            "image_runs": case["image_token_runs"],
                        },
                    )
                )

            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)
            report["cases"] = results

        write_evidence(f"source_logit_replay-cg{int(enabled)}", report)

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_generation_parity(self, enabled: bool):
        """>=32 greedy tokens per prompt must match the source at every step."""
        require_single_h200()
        label = f"generation_parity[cuda_graph={enabled}]"
        text_capture = ensure_capture("text_replay")
        image_capture = ensure_capture("image_replay")
        steps = text_capture["max_new_tokens"]
        assert steps >= 32, f"{label}: the reference generated only {steps} tokens"

        cases: List[Dict[str, Any]] = []
        with RuntimeProbe() as probe, gemma4_llm(enabled=enabled, max_batch_size=4) as llm:
            assert_kv_cache_manager_v2(llm, label=label)

            # One token past the compared horizon: the overlap scheduler can
            # drop the *last* step's generation logits, and asking for exactly
            # `steps` would then leave the final step's logit row missing.
            params = greedy(steps + 1, logits=True, suppress_eos=True)

            def forced_run(source_case: Dict[str, Any], *, image: bool) -> Dict[str, Any]:
                return forced_decode_run(
                    llm,
                    source_case,
                    steps=steps,
                    image=image,
                    probe=probe,
                    enabled=enabled,
                    label=label,
                )

            for idx, case in enumerate(text_capture["prompts"]):
                # Free-running decode on both sides: the run that exercises the
                # KV-cache / decode path token by token, and the only one whose
                # per-step logits are the same quantity as the reference's.
                out = llm.generate(token_prompt(case["input_ids"]), sampling_params=params)
                cases.append(
                    compare_generation(
                        out,
                        case,
                        label=f"{label} text[{idx}]",
                        extra={"prompt": case["prompt"]},
                        localize=lambda c=case: forced_run(c, image=False)["logits"],
                    )
                )

            for idx, case in enumerate(image_capture["prompts"]):
                out = llm.generate(
                    image_prompt(render_image_text(case["prompt"]), [case["image"]]),
                    sampling_params=params,
                )
                cases.append(
                    compare_generation(
                        out,
                        case,
                        label=f"{label} image[{idx}]",
                        extra={"image": case["image"], "prompt": case["prompt"]},
                        localize=lambda c=case: forced_run(c, image=True)["logits"],
                    )
                )

            # Any prompt that forked on an unresolved reference tie owes the
            # steps after the fork a matched-prefix comparison, and it has to be
            # on the *same* path the criterion names -- prefill plus cached
            # decode -- or the tie amendment would silently shrink both the
            # compared horizon and the exercised runtime surface.
            recoveries: List[Dict[str, Any]] = []
            for idx, case in enumerate(cases):
                if case.get("forked_at_tie") is None:
                    continue
                is_image = idx >= len(text_capture["prompts"])
                source_case = (
                    image_capture["prompts"][idx - len(text_capture["prompts"])]
                    if is_image
                    else text_capture["prompts"][idx]
                )
                recoveries.append(
                    forced_decode_recovery(
                        source_case,
                        forced_run(source_case, image=is_image),
                        label=case["label"],
                    )
                )

            assert any(c.num_generations > 0 for c in probe.attention_calls), (
                f"{label}: no cached-decode attention call was observed"
            )
            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)
            report["cases"] = cases
            report["forced_decode_recoveries"] = recoveries
            # The amended contract, restated in the artifact so a reader does
            # not have to reconstruct it from the code.
            report["token_gate"] = {
                "contract": (
                    "identical greedy tokens at every step, except where the reference's own "
                    "top-2 gap is within its bfloat16 storage resolution, where the runtime's "
                    "token must be one of those two candidates"
                ),
                "amended_by": "explicit human decision, iteration 10 (see validation-report.md section 12)",
                "reference_tie_ulps": REFERENCE_TIE_ULPS,
                "unresolved_reference_ties": [
                    {"prompt": c["label"].rsplit(" ", 1)[-1], **tie}
                    for c in cases
                    for tie in c["unresolved_reference_ties"]
                ],
            }
            # The reference forces exactly `steps` new tokens with
            # `min_new_tokens=max_new_tokens`; record that TensorRT-LLM ran the
            # matching ban so the configs can be compared, not assumed.
            report["eos_suppression"] = {
                "reference": "min_new_tokens == max_new_tokens",
                "trtllm_bad_token_ids": list(checkpoint_eos_token_ids()),
                "source": "checkpoint generation_config.json eos_token_id",
            }

        write_evidence(f"generation_parity-cg{int(enabled)}", report)

        # Discrete gate, asserted after the evidence is on disk: every one of
        # the >= 32 greedy tokens must equal the source's, for every prompt, in
        # this runtime matrix -- except at steps the reference itself did not
        # resolve, where the runtime must still have picked one of the
        # reference's own two candidates (see `reference_tie`).
        diverged = [c for c in cases if c["divergence"] is not None]
        assert not diverged, (
            f"{label}: {len(diverged)} of {len(cases)} prompts diverged from the source's "
            "greedy tokens: "
            + "; ".join(
                f"{c['label'].rsplit(' ', 1)[-1]} step {c['divergence']['step']} "
                f"({c['divergence']['trtllm_token']} != {c['divergence']['source_token']}, "
                f"source top-2 margin {c['divergence']['source_top2_margin']:.4f}, "
                f"step cosine {c['divergence']['step_metrics']['cosine']:.6f})"
                for c in diverged
            )
            + f". Full per-step evidence: generation_parity-cg{int(enabled)}.json"
        )

        # A prompt that forked on a tie only certified steps up to the fork, so
        # its matched-prefix pass has to cover the rest with no unexplained
        # mismatch.  Without this, exempting a tie would also quietly exempt
        # every step after it.
        broken = [r for r in recoveries if r["mismatches"]]
        assert not broken, (
            f"{label}: {len(broken)} prompt(s) forked on an unresolved reference tie and then "
            "disagreed with the source on a matched prefix, which the tie exemption does not "
            "cover: "
            + "; ".join(
                f"{r['label'].rsplit(' ', 1)[-1]} step {m['step']} "
                f"({m['trtllm_forced_argmax']} != {m['source_token']}, cosine {m['cosine']:.6f})"
                for r in broken
                for m in r["mismatches"]
            )
            + f". Full evidence: generation_parity-cg{int(enabled)}.json"
        )
        for recovery in recoveries:
            assert recovery["compared_steps"] >= 32, (
                f"{label}: {recovery['label']} recovered only "
                f"{recovery['compared_steps']} matched-prefix steps, need >= 32"
            )
            # The recovery only counts if it ran on the path the criterion names.
            # Restated at the node so the gate is readable without following the
            # helper: prefill, then one cached-decode forward per step, with
            # CUDA-graph replay in the enabled cell.
            runtime = recovery["runtime_evidence"]
            assert runtime["emitted_matches_source"], (
                f"{label}: {recovery['label']} did not walk the source's tokens"
            )
            assert runtime["prefill_forwards"] >= 1, (
                f"{label}: {recovery['label']} ran no context prefill"
            )
            assert runtime["decode_steps_accounted"] >= recovery["compared_steps"] - 1, (
                f"{label}: {recovery['label']} accounted for "
                f"{runtime['decode_steps_accounted']} decode steps, need "
                f"{recovery['compared_steps'] - 1}"
            )
            assert (runtime["cuda_graph_replays"] > 0) == enabled, (
                f"{label}: {recovery['label']} observed "
                f"{runtime['cuda_graph_replays']} CUDA-graph replays with cuda_graph={enabled}"
            )


# --------------------------------------------------------------------------
# Accuracy nodes
# --------------------------------------------------------------------------


class TestGemma4H200Accuracy(LlmapiAccuracyTestHarness):
    """LLM-API smoke, fixed-sample canaries, and the configured MMLU/MMMU gates."""

    MODEL_NAME = MODEL_NAME

    # MMLU is rendered through the checkpoint's chat template with thinking
    # off, which is the configuration under which the *source model* actually
    # reproduces the checked-in 71.296 reference (see MMLU_EVALUATOR_KWARGS).
    # The canary uses the identical rendering: matching the gate is the whole
    # point of a canary.
    MMLU_CANARY_MAX_TOKENS = 4
    # Match the gate's output budget.  Gemma 4 answers MMMU by working
    # through the problem and stating the option at the end, so a short
    # budget truncates mid-reasoning and
    # ``strip_thinking_and_extract_mmmu_answer`` then scores noise -- the
    # canary would stop predicting the gate it is meant to guard.
    MMMU_CANARY_MAX_TOKENS = MMMU.MAX_OUTPUT_LEN

    mmmu_sampling_params = SamplingParams(
        max_tokens=MMMU.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
        stop="<|endoftext|>",
    )
    EXTRA_EVALUATOR_KWARGS = {"chat_template_kwargs": CHAT_TEMPLATE_KWARGS}

    # MMLU's evaluator defaults to raw 5-shot completions, and MMMU's defaults
    # to `apply_chat_template=True`.  For this instruction-tuned checkpoint the
    # raw rendering is what the checked-in 71.296 reference disagrees with, not
    # the port: measured on 513 evaluator-selected samples, native Transformers
    # 5.5.4 scores **61.99** raw and **77.97** chat-templated with thinking off
    # (`repro/mmlu_native_vs_trt.py`, evidence `mmlu513-native*.json`).  The
    # reference is therefore only reproducible under the templated rendering,
    # which is also what the sibling B200 Gemma 4 test uses for MMMU.  The
    # config was settled on the *native reference* first and then applied
    # unchanged to TensorRT-LLM, so this aligns the harness with the reference
    # rather than tuning the harness around the runtime.
    MMLU_EVALUATOR_KWARGS = {
        "apply_chat_template": True,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_mmlu_canary(self, enabled: bool):
        require_single_h200()
        label = f"mmlu_canary[cuda_graph={enabled}]"
        samples = build_mmlu_canary_samples()
        reference = ensure_native_completions(
            {
                "items": [
                    {"id": s["id"], "input_ids": s["input_ids"], "images": []} for s in samples
                ],
                "max_new_tokens": self.MMLU_CANARY_MAX_TOKENS,
            },
            tag="mmlu32",
        )
        native_score = score_mmlu(reference["completions"], samples)

        with (
            RuntimeProbe() as probe,
            gemma4_llm(enabled=enabled, max_batch_size=8, enable_chunked_prefill=True) as llm,
        ):
            assert_kv_cache_manager_v2(llm, label=label)

            # Short LLM-API smoke first: it fails in seconds if the runtime is
            # mis-wired, instead of after the canary's full sweep.
            smoke = llm.generate(
                [token_prompt(render_text(TEXT_PROMPTS[0]))], sampling_params=greedy(16)
            )
            assert smoke[0].outputs[0].text.strip(), f"{label}: smoke produced no text"
            assert_graph_hard_path(llm, probe, enabled=enabled, label=f"{label} smoke")

            outputs = llm.generate(
                [token_prompt(s["input_ids"]) for s in samples],
                sampling_params=greedy(self.MMLU_CANARY_MAX_TOKENS),
            )
            trt_completions = [
                {"id": s["id"], "text": o.outputs[0].text}
                for s, o in zip(samples, outputs, strict=True)
            ]
            trt_score = score_mmlu(trt_completions, samples)

            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)

        report.update(
            {
                "benchmark": "mmlu_canary",
                "num_samples": len(samples),
                "sample_ids": [s["id"] for s in samples],
                "native_score": native_score,
                "trtllm_score": trt_score,
                "tolerance": MMLU_CANARY_TOLERANCE,
            }
        )
        write_evidence(f"mmlu_canary-cg{int(enabled)}", report)
        assert abs(trt_score - native_score) <= MMLU_CANARY_TOLERANCE, (
            f"{label}: TensorRT-LLM {trt_score:.2f} vs native {native_score:.2f} on the same "
            f"{len(samples)} samples (tolerance {MMLU_CANARY_TOLERANCE})"
        )

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_mmlu_bf16(self, enabled: bool):
        require_single_h200()
        label = f"mmlu_bf16[cuda_graph={enabled}]"
        with (
            RuntimeProbe() as probe,
            gemma4_llm(
                enabled=enabled,
                max_batch_size=32,
                enable_chunked_prefill=True,
                max_seq_len=MMLU.MAX_INPUT_LEN + MMLU.MAX_OUTPUT_LEN,
            ) as llm,
        ):
            assert_kv_cache_manager_v2(llm, label=label)
            assert llm.args.quant_config.quant_algo is None

            # Short same-config smoke before the long sweep: a mis-wired
            # runtime or a graph that never captures should cost seconds here
            # rather than a full MMLU pass.
            smoke = llm.generate(
                [token_prompt(render_text(TEXT_PROMPTS[0]))], sampling_params=greedy(16)
            )
            assert smoke[0].outputs[0].text.strip(), f"{label}: pre-benchmark smoke made no text"
            probe.assert_sm90_dispatch(label=f"{label} smoke")
            assert_graph_hard_path(llm, probe, enabled=enabled, label=f"{label} smoke")
            smoke_text = smoke[0].outputs[0].text

            score = MMLU(self.MODEL_NAME).evaluate(
                llm, extra_evaluator_kwargs=self.MMLU_EVALUATOR_KWARGS
            )
            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)
        report.update(
            {
                "benchmark": "mmlu",
                "score": score,
                "reference": 71.296,
                "pre_benchmark_smoke": smoke_text,
            }
        )
        write_evidence(f"mmlu_bf16-cg{int(enabled)}", report)

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_mmmu_canary(self, enabled: bool):
        require_single_h200()
        label = f"mmmu_canary[cuda_graph={enabled}]"
        samples = build_mmmu_canary_samples()
        reference = ensure_native_completions(
            {
                "items": [
                    {"id": s["id"], "text": s["prompt"], "images": s["images"]} for s in samples
                ],
                "max_new_tokens": self.MMMU_CANARY_MAX_TOKENS,
            },
            tag="mmmu16",
        )
        native_score = score_mmmu(reference["completions"], samples)

        with (
            RuntimeProbe() as probe,
            gemma4_llm(enabled=enabled, max_batch_size=4, enable_chunked_prefill=True) as llm,
        ):
            assert_kv_cache_manager_v2(llm, label=label)

            smoke_sample = samples[0]
            smoke = llm.generate(
                image_prompt(smoke_sample["prompt"], smoke_sample["images"]),
                sampling_params=greedy(32),
            )
            assert smoke.outputs[0].text.strip(), f"{label}: image smoke produced no text"
            assert any(c.custom_mask for c in probe.attention_calls), (
                f"{label}: the image smoke never used the bidirectional custom mask"
            )
            assert_graph_hard_path(llm, probe, enabled=enabled, label=f"{label} smoke")

            outputs = llm.generate(
                [image_prompt(s["prompt"], s["images"]) for s in samples],
                sampling_params=greedy(self.MMMU_CANARY_MAX_TOKENS),
            )
            trt_completions = [
                {"id": s["id"], "text": o.outputs[0].text}
                for s, o in zip(samples, outputs, strict=True)
            ]
            trt_score = score_mmmu(trt_completions, samples)

            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)

        report.update(
            {
                "benchmark": "mmmu_canary",
                "num_samples": len(samples),
                "sample_ids": [s["id"] for s in samples],
                "native_score": native_score,
                "trtllm_score": trt_score,
                "tolerance": MMMU_CANARY_TOLERANCE,
            }
        )
        write_evidence(f"mmmu_canary-cg{int(enabled)}", report)
        assert abs(trt_score - native_score) <= MMMU_CANARY_TOLERANCE, (
            f"{label}: TensorRT-LLM {trt_score:.2f} vs native {native_score:.2f} on the same "
            f"{len(samples)} samples (tolerance {MMMU_CANARY_TOLERANCE})"
        )

    @pytest.mark.parametrize("enabled", RUNTIME_MATRIX)
    def test_mmmu_bf16(self, enabled: bool):
        require_single_h200()
        label = f"mmmu_bf16[cuda_graph={enabled}]"
        with (
            RuntimeProbe() as probe,
            gemma4_llm(
                enabled=enabled,
                max_batch_size=16,
                enable_chunked_prefill=True,
                free_gpu_memory_fraction=0.5,
            ) as llm,
        ):
            assert_kv_cache_manager_v2(llm, label=label)
            assert llm.args.quant_config.quant_algo is None

            # Short same-config *image* smoke before the long sweep: it costs
            # seconds and fails immediately if the vision path, the custom
            # mask, or graph capture is not actually wired for this config.
            smoke_sample = build_mmmu_canary_samples()[0]
            smoke = llm.generate(
                image_prompt(smoke_sample["prompt"], smoke_sample["images"]),
                sampling_params=greedy(32),
            )
            assert smoke.outputs[0].text.strip(), f"{label}: pre-benchmark smoke made no text"
            assert any(c.custom_mask for c in probe.attention_calls), (
                f"{label}: the pre-benchmark image smoke never used the bidirectional mask"
            )
            probe.assert_sm90_dispatch(label=f"{label} smoke")
            assert_graph_hard_path(llm, probe, enabled=enabled, label=f"{label} smoke")
            smoke_text = smoke.outputs[0].text

            score = MMMU(self.MODEL_NAME).evaluate(
                llm,
                sampling_params=self.mmmu_sampling_params,
                extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS,
            )
            probe.assert_sm90_dispatch(label=label)
            assert_graph_hard_path(llm, probe, enabled=enabled, label=label)
            report = runtime_report(llm, probe, enabled=enabled)
        report.update(
            {
                "benchmark": "mmmu",
                "score": score,
                "reference": 56.667,
                "pre_benchmark_smoke": smoke_text,
            }
        )
        write_evidence(f"mmmu_bf16-cg{int(enabled)}", report)


# --------------------------------------------------------------------------
# Helpers used by the nodes above
# --------------------------------------------------------------------------


# Source key patterns the loader deliberately *transforms* rather than copying
# name-for-name.  Each entry is ``(regex, reason)``; a source tensor matching
# one is accounted as explained-by-transform instead of demanding an
# identically named destination.  Keeping the reasons enumerated here, rather
# than widening the name normalizer until nothing is ever unexplained, is what
# keeps this check able to fail: anything neither name-matched nor listed is a
# real gap.
TRANSFORMED_SOURCE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\.self_attn\.[qkv]_proj(\.linear)?\.weight$", "fused into the attention qkv_proj"),
    (r"\.mlp\.(gate|up)_proj(\.linear)?\.weight$", "fused into the MLP gate_up_proj"),
    (r"\.experts\.(gate_up_proj|down_proj)$", "packed into the fused-MoE expert parameters"),
    # The vision tower widens its head dimension from the checkpoint's 72 to a
    # kernel-friendly 80, so o_proj gains input channels that no source tensor
    # supplies.  Listing it here only *accounts* for the tensor;
    # `assert_vision_padding_is_zero` is what proves the added channels are
    # exactly zero rather than arbitrary.
    (
        r"vision_tower\..*\.self_attn\.o_proj(\.linear)?\.weight$",
        "zero-extended from vision head_dim 72 to the padded head_dim on o_proj's input",
    ),
)


def assert_vision_padding_is_zero(model, *, label: str) -> Dict[str, Any]:
    """The vision 72->80 extension must be exact zeros in every padded channel.

    A padded head dimension is only equivalent to the source when the added
    channels contribute nothing.  This reads the *loaded* weights and proves
    that, rather than trusting that a padding helper did the right thing.
    """
    from transformers import AutoConfig

    vision_tower = getattr(model, "vision_tower", None)
    assert vision_tower is not None, f"{label}: the model exposes no vision tower"
    # ``model.config`` is the *text* config on the conditional-generation
    # wrapper, so take the tower geometry from the checkpoint itself.
    vision_config = AutoConfig.from_pretrained(gemma4_26b_checkpoint()).vision_config
    num_heads = int(vision_config.num_attention_heads)
    head_dim = int(vision_config.head_dim)

    checked: List[str] = []
    padded_head_dim = None
    for name, tensor in vision_tower.state_dict().items():
        if name.endswith("self_attn.o_proj.weight"):
            # [out, num_heads * padded_head_dim]
            padded = tensor.shape[1] // num_heads
            tail = tensor.reshape(tensor.shape[0], num_heads, padded)[:, :, head_dim:]
        elif name.endswith("self_attn.qkv_proj.weight"):
            # [3 * num_heads * padded_head_dim, in]
            padded = tensor.shape[0] // (3 * num_heads)
            tail = tensor.reshape(3, num_heads, padded, tensor.shape[1])[:, :, head_dim:, :]
        else:
            continue
        padded_head_dim = padded
        if padded == head_dim:
            continue  # no extension on this build
        nonzero = int(tail.ne(0).sum())
        assert nonzero == 0, (
            f"{label}: {name} has {nonzero} non-zero values in the padded channels "
            f"[{head_dim}:{padded}); the vision head-dim extension is not a no-op, so "
            "tower attention no longer matches the source"
        )
        checked.append(name)

    return {
        "vision_head_dim": head_dim,
        "vision_padded_head_dim": padded_head_dim,
        "vision_num_heads": num_heads,
        "zero_padded_tensors_checked": len(checked),
    }


def state_dict_accounting(model) -> Dict[str, Any]:
    """Account for every checkpoint tensor: copied, transformed, or unexplained.

    Three outcomes, and only the third is a failure:

    * **copied** -- some loaded tensor carries the same normalized name.
    * **transformed** -- the loader fuses or repacks it, so no identically
      named destination exists.  Expert slabs are additionally accounted by
      element count, which a silent drop cannot pass.
    * **unexplained** -- neither, i.e. a source tensor nothing accounts for.

    Matching is **one-to-one**.  Signatures collide heavily on this checkpoint
    -- seven distinct layer-0 language norms all reduce to
    ``("language", 0, "weight", (2816,))`` -- so a set-membership test would let
    one surviving tensor vouch for every sibling the loader dropped.  Each
    loaded tensor is therefore consumed by at most one source tensor: the
    specific ``parent.leaf`` signature is matched first for every source key,
    and only the leftovers fall back to the leaf-only signature, so a dropped
    tensor leaves a source key with no partner rather than borrowing one.
    """
    import re
    from collections import defaultdict

    from safetensors import safe_open

    checkpoint = gemma4_26b_checkpoint()
    source_shapes: Dict[str, Tuple[int, ...]] = {}
    for shard in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                source_shapes[key] = tuple(handle.get_slice(key).get_shape())
    assert source_shapes, f"no safetensors shards found under {checkpoint}"

    loaded = dict(model.state_dict())
    by_specific: Dict[Signature, List[str]] = defaultdict(list)
    by_loose: Dict[Signature, List[str]] = defaultdict(list)
    for key, tensor in loaded.items():
        specific, loose = _tensor_signatures(key, tuple(tensor.shape))
        by_specific[specific].append(key)
        by_loose[loose].append(key)
    consumed: Dict[str, str] = {}

    def _claim(candidates: List[str], source_key: str) -> bool:
        for candidate in candidates:
            if candidate not in consumed:
                consumed[candidate] = source_key
                return True
        return False

    transforms = [(re.compile(pattern), reason) for pattern, reason in TRANSFORMED_SOURCE_PATTERNS]

    copied: List[str] = []
    unmatched: List[str] = []
    for key, shape in sorted(source_shapes.items()):
        specific, _ = _tensor_signatures(key, shape)
        if _claim(by_specific.get(specific, []), key):
            copied.append(key)
        else:
            unmatched.append(key)

    still_unmatched: List[str] = []
    for key in unmatched:
        _, loose = _tensor_signatures(key, source_shapes[key])
        if _claim(by_loose.get(loose, []), key):
            copied.append(key)
        else:
            still_unmatched.append(key)

    transformed: Dict[str, List[str]] = {}
    unexplained: List[str] = []
    for key in still_unmatched:
        for pattern, reason in transforms:
            if pattern.search(key):
                transformed.setdefault(reason, []).append(key)
                break
        else:
            unexplained.append(key)
    copied.sort()

    def _numel(shape: Tuple[int, ...]) -> int:
        total = 1
        for dim in shape:
            total *= int(dim)
        return total

    return {
        "source_tensors": len(source_shapes),
        "loaded_tensors": len(loaded),
        "copied": len(copied),
        "distinct_destinations": len(consumed),
        "transformed": {reason: len(keys) for reason, keys in transformed.items()},
        "transformed_examples": {reason: keys[:2] for reason, keys in transformed.items()},
        # Gemma 4 ties the LM head to the input embedding, so the head never
        # appears as an independent source tensor.
        "tied": sorted(k for k in loaded if k.endswith("lm_head.weight")),
        # The fused-MoE parameter layout is implementation-specific, so expert
        # slabs are accounted numerically rather than by name.
        "source_expert_numel": sum(
            _numel(shape) for k, shape in source_shapes.items() if ".experts." in k
        ),
        "loaded_expert_numel": sum(t.numel() for name, t in loaded.items() if ".experts." in name),
        "unexplained": unexplained,
    }


Signature = Tuple[str, Optional[int], str, Tuple[int, ...]]


def _tensor_signatures(key: str, shape: Tuple[int, ...]) -> Tuple[Signature, ...]:
    """Naming-agnostic identities for one tensor: ``(domain, layer, leaf, shape)``.

    The HF and TensorRT-LLM trees agree on *where* a tensor lives (vision vs
    language, which layer) and on its leaf name and shape, but not on the
    container modules in between -- TensorRT-LLM inserts ``moe.`` in front of
    the router and hangs ``per_expert_scale`` off the MoE block rather than the
    router, the checkpoint inserts ``.linear.`` inside vision projections, and
    the two disagree about the root prefix.  Matching on the parts they do
    agree on keeps this check from failing over pure naming, while domain,
    layer index, and shape keep it strict enough to still catch a drop.

    Two candidates are returned, most specific first: the last two path
    components, then the leaf alone.  Callers match on *any* candidate, so the
    looser form is reached only when the container names differ.
    """
    normalized = key.replace(".linear.weight", ".weight").replace(".linear.bias", ".bias")
    parts = normalized.split(".")
    domain = "vision" if any("vision" in part for part in parts) else "language"
    layer: Optional[int] = None
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            layer = int(parts[i + 1])
            break
    dims = tuple(int(d) for d in shape)
    return (
        (domain, layer, ".".join(parts[-2:]), dims),
        (domain, layer, parts[-1], dims),
    )


@functools.lru_cache(maxsize=1)
def reference_text_ids() -> Tuple[Tuple[int, ...], ...]:
    """Reference token ids for the fixed text prompts."""
    capture = ensure_capture("text_replay")
    return tuple(tuple(case["input_ids"]) for case in capture["prompts"])


@functools.lru_cache(maxsize=4)
def select_valid_patches(tensor: torch.Tensor, patch_valid: torch.Tensor) -> torch.Tensor:
    """Rows of ``tensor`` that hold real image patches, in source order.

    The source's per-patch validity mask comes straight from the processor's
    ``image_position_ids`` sentinels, so this selects content the model actually
    attended to and drops only slots the source itself masked out.
    """
    flat = tensor.reshape(-1, tensor.shape[-1])
    mask = patch_valid.reshape(-1).to(torch.bool)
    assert flat.shape[0] == mask.shape[0], (
        f"validity mask covers {mask.shape[0]} patch slots but the capture has {flat.shape[0]}"
    )
    return flat[mask].float()


def measured_image_block(text: str, image_path: str) -> Tuple[int, int, List[int]]:
    """``(block_start, block_end, input_ids)`` for a rendered image prompt.

    Runs the checkpoint's own processor -- no model, no GPU -- so the canary
    positions its chunk boundary from the real soft-token layout of the exact
    prompt it is about to send, rather than from a constant that silently stops
    meaning anything when the processor changes.
    """
    from PIL import Image

    processor = gemma4_processor()
    inputs = processor(
        text=[text], images=[Image.open(image_path).convert("RGB")], return_tensors="pt"
    )
    runs = contiguous_image_runs(inputs["mm_token_type_ids"][0].cpu())
    assert runs, "the rendered image prompt contains no image soft-token run"
    start, length = max(runs, key=lambda run: run[1])
    return start, start + length, [int(i) for i in inputs["input_ids"][0].tolist()]


def image_chunk_budget(image_token_runs: Sequence[Sequence[int]]) -> int:
    """Largest chunk budget that still forces the scheduler to snap the block.

    ``max_num_tokens`` sets the prefill chunk size, so an unsnapped first
    boundary would land at exactly that many tokens.  Two constraints fix the
    budget, and the runtime states both:

    * the bidirectional block has to fit in **one** chunk -- the V2 scheduler
      rejects a smaller budget outright ("the block must fit in a single chunk
      to preserve bidirectional attention; deferring would livelock"), so the
      budget must be at least the block length;
    * to make the scheduler actually *snap*, the unsnapped boundary must fall
      inside the block, so the budget must be below the block's end.

    ``end - 1`` is the largest budget that still lands inside the block, and it
    leaves the most room for the snapped-down chunk and the block-carrying chunk
    that follows it.  Derived from the measured run, so a change in the
    processor's soft-token count moves it rather than silently making the canary
    prove nothing.
    """
    start, length = max(image_token_runs, key=lambda run: run[1])
    end = start + length
    budget = end - 1
    assert start < budget, (
        f"the image block starts at {start} and ends at {end}; no chunk budget both lands "
        "inside it and leaves a non-empty chunk before it"
    )
    # The chunk that carries the block runs from the snapped-down page boundary
    # at or before `start` to `end`, so the budget has to hold that much.
    assert budget >= length, (
        f"chunk budget {budget} cannot hold the {length}-token bidirectional block, which the "
        "V2 scheduler requires to fit in a single chunk"
    )
    return budget


def long_context_ids(min_tokens: int) -> Tuple[int, ...]:
    """A deterministic prompt of at least ``min_tokens`` reference tokens."""
    base = reference_text_ids()
    ids: List[int] = []
    idx = 0
    while len(ids) < min_tokens:
        ids.extend(base[idx % len(base)])
        idx += 1
    return tuple(ids[:min_tokens])


def wait_for_free_blocks(manager, target: int, *, timeout_s: float = 30.0) -> int:
    """Poll ``get_num_free_blocks`` until it reaches ``target`` or times out."""
    import time

    deadline = time.monotonic() + timeout_s
    free = int(manager.get_num_free_blocks())
    while free < target and time.monotonic() < deadline:
        time.sleep(0.5)
        free = int(manager.get_num_free_blocks())
    return free


def assert_token_parity(actual: Sequence[int], expected: Sequence[int], *, label: str) -> int:
    """Fail at the *first* differing step, reporting both prefixes."""
    n = min(len(actual), len(expected))
    assert n > 0, f"{label}: nothing to compare"
    for step in range(n):
        if actual[step] != expected[step]:
            raise AssertionError(
                f"{label}: first divergence at step {step}: TensorRT-LLM {actual[step]} != "
                f"source {expected[step]}; TensorRT-LLM prefix {list(actual[: step + 1])} vs "
                f"source prefix {list(expected[: step + 1])}"
            )
    return n


def compare_final_logits(
    actual: torch.Tensor, expected: torch.Tensor, *, label: str, extra: Dict[str, Any]
) -> Dict[str, Any]:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    assert actual.shape == expected.shape, (
        f"{label}: logit shape {tuple(actual.shape)} vs reference {tuple(expected.shape)}"
    )
    metrics = assert_cosine(actual, expected, label=label, min_cosine=MIN_COSINE)
    actual_argmax, expected_argmax = int(actual.argmax()), int(expected.argmax())
    assert actual_argmax == expected_argmax, (
        f"{label}: greedy argmax {actual_argmax} != reference {expected_argmax} ({metrics})"
    )
    return {"label": label, "metrics": metrics.as_dict(), "argmax": actual_argmax, **extra}


class ForcedDecode:
    """Pin the sampler to a fixed token sequence without leaving the decode path.

    Teacher forcing on the *production* runtime.  Everything the free-running
    generation does still happens -- one context prefill, then one single-token
    cached-decode forward per step, through ``KVCacheManagerV2`` and (when the
    matrix cell enables it) CUDA-graph replay.  The only change is that after
    each forward this processor rewrites one entry of the logit row so the
    greedy sampler is obliged to emit the *source's* token, which keeps both
    paths on an identical prefix for the whole horizon.

    Two properties matter for the evidence to mean anything:

    * The row handed to a logits processor is a **view** into the engine's own
      logits tensor (``model_engine._apply_logits_processors``), so the recorded
      copy is the model's real output -- taken *before* the rewrite -- and the
      rewrite is what the sampler actually sees.
    * The forcing is a single-entry bump above the row's own maximum rather than
      a ``-inf`` mask, so no other candidate's value is disturbed and nothing
      downstream can produce a NaN from an all-masked row.

    Call ``i`` is the distribution after ``i`` generated tokens: the first call
    is the context request's last-position row, and every later call is one
    decode step.  That is the same alignment as the reference's ``step_logits``.
    """

    def __init__(self, forced: Sequence[int]) -> None:
        self.forced = [int(t) for t in forced]
        self.rows: List[torch.Tensor] = []
        self.forced_steps = 0
        self.observed_token_counts: List[int] = []

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        assert logits.dim() == 3 and logits.shape[0] == 1 and logits.shape[1] == 1, (
            f"forced decode expects one beam and one logit row per call, got "
            f"{tuple(logits.shape)}; this runtime batched the request differently and the "
            "recorded rows would not be this request's"
        )
        # Basic indexing (never a copy), so the write below reaches the sampler.
        row = logits[0, 0]
        self.rows.append(row.detach().float().cpu().clone())
        if token_ids:
            self.observed_token_counts.append(len(token_ids[0]))
        step = len(self.rows) - 1
        if step < len(self.forced):
            top = row.max()
            # Strictly above the row's own maximum, and by enough that bfloat16
            # rounding cannot collapse it back onto the incumbent.
            row[self.forced[step]] = top + max(1.0, abs(float(top)) * 0.02)
            self.forced_steps += 1


def forced_decode_run(
    llm: LLM,
    case: Dict[str, Any],
    *,
    steps: int,
    image: bool,
    probe: RuntimeProbe,
    enabled: bool,
    label: str,
) -> Dict[str, Any]:
    """Drive the production cached-decode path down the source's own tokens.

    Returns ``{"logits": [steps, vocab] float32 CPU, "evidence": {...}}``.  The
    prompt is the *same object* the free-running run used -- for an image case
    the real image and the real rendered chat text, not a re-rendered prompt
    with a detokenized continuation appended -- so the only difference between
    this run and the free-running one is which token the sampler is allowed to
    pick.

    The evidence records what actually executed: the number of prefill forwards,
    the number of single-token decode forwards observed eagerly, the number of
    CUDA-graph replays, and the emitted token ids.  Under CUDA graph the
    per-layer attention call is *not* re-entered on replay (the captured kernels
    run instead), so a decode step shows up as a replay rather than as an
    observed attention call; requiring ``observed + replays >= steps - 1``
    accounts for every step in either cell without pretending graph replays are
    eager calls.
    """
    forced = [int(t) for t in case["greedy_tokens"]][:steps]
    assert len(forced) == steps, (
        f"{label}: the reference supplies only {len(forced)} tokens, need {steps} to force"
    )
    processor = ForcedDecode(forced)
    params = SamplingParams(
        max_tokens=steps,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        bad_token_ids=list(checkpoint_eos_token_ids()),
        logits_processor=processor,
    )

    calls_before = len(probe.attention_calls)
    captures_before, replays_before = probe.graph_captures, probe.graph_replays
    if image:
        prompt: Dict[str, Any] = image_prompt(render_image_text(case["prompt"]), [case["image"]])
    else:
        prompt = token_prompt(case["input_ids"])
    out = llm.generate(prompt, sampling_params=params)

    new_calls = probe.attention_calls[calls_before:]
    replays = probe.graph_replays - replays_before
    captures = probe.graph_captures - captures_before
    # One forward touches every layer once, so count a single layer to turn
    # per-layer attention calls back into forwards.
    first_layer = min((c.layer_idx for c in new_calls), default=-1)
    decode_forwards = sum(
        1 for c in new_calls if c.layer_idx == first_layer and c.num_generations > 0
    )
    prefill_forwards = sum(
        1 for c in new_calls if c.layer_idx == first_layer and c.num_contexts > 0
    )
    emitted = [int(t) for t in out.outputs[0].token_ids][:steps]

    evidence = {
        "execution": (
            "production cached decode: one context prefill, then one single-token forward per "
            "step with the source's token forced by a logits processor (KVCacheManagerV2 reuse; "
            "CUDA-graph replay when the cell enables it)"
        ),
        "forced_tokens": steps,
        "emitted_tokens": len(out.outputs[0].token_ids),
        "emitted_matches_source": emitted == forced,
        "logit_rows_recorded": len(processor.rows),
        "prefill_forwards": prefill_forwards,
        "decode_forwards_observed": decode_forwards,
        "cuda_graph_replays": replays,
        "cuda_graph_captures": captures,
        "decode_steps_accounted": decode_forwards + replays,
        "request_token_counts": processor.observed_token_counts,
        "cuda_graph": enabled,
    }

    assert emitted == forced, (
        f"{label}: the forced decode did not walk the source's tokens -- emitted "
        f"{emitted[:8]}... against source {forced[:8]}...; the logits processor did not reach "
        "the sampler, so this run is not a matched-prefix decode"
    )
    assert len(processor.rows) >= steps, (
        f"{label}: forced decode recorded {len(processor.rows)} logit rows, need {steps}"
    )
    assert processor.forced_steps >= steps, (
        f"{label}: forced decode rewrote only {processor.forced_steps} of {steps} rows"
    )
    assert prefill_forwards >= 1, f"{label}: forced decode ran no context prefill"
    assert decode_forwards + replays >= steps - 1, (
        f"{label}: forced decode accounted for only {decode_forwards + replays} of the "
        f"{steps - 1} single-token decode steps ({decode_forwards} observed attention "
        f"forwards + {replays} CUDA-graph replays); the horizon was not walked on the "
        "cached-decode path"
    )
    if enabled:
        assert replays > 0, (
            f"{label}: cuda_graph=True but the forced decode triggered no CUDA-graph replay, "
            "so its decode steps are not cuda_graph_hard_path evidence"
        )
    else:
        assert replays == 0 and captures == 0, (
            f"{label}: cuda_graph=False but the forced decode observed captures={captures} "
            f"replays={replays}"
        )
        assert decode_forwards >= steps - 1, (
            f"{label}: cuda_graph=False forced decode observed {decode_forwards} decode "
            f"forwards, need {steps - 1}"
        )

    return {"logits": torch.stack(processor.rows[:steps]), "evidence": evidence}


def mask_banned(row: torch.Tensor, banned: Sequence[int]) -> torch.Tensor:
    """A logit row with the generation config's end-of-turn ids removed.

    Both paths generate with those ids banned (the reference via
    ``min_new_tokens``, TensorRT-LLM via ``bad_token_ids``), so "which token is
    greedy here" is a question about the *remaining* candidates.  Comparing raw
    argmaxes instead would compare a token neither path is allowed to emit --
    on this checkpoint the raw argmax really is end-of-turn at several steps.
    """
    masked = row.float().clone()
    for token in banned:
        if 0 <= int(token) < masked.shape[-1]:
            masked[int(token)] = float("-inf")
    return masked


def teacher_forced_outcome(forced_argmax: int, source_token: int, actual_token: int) -> str:
    """Name what a teacher-forced argmax actually showed, in three outcomes.

    Under the source's own prefix TensorRT-LLM can land in exactly one of three
    places, and they carry different meanings:

    ``reproduces_trtllm_token``
        The single forward emits the same token the free-running run did.  The
        divergence is reproducible from one forward on a matched prefix, so it
        is model math rather than decode/KV-reuse state.
    ``matches_source_token``
        The single forward agrees with the source.  The prefix-matched math is
        fine and the free-running divergence came from the decode path (cache
        reuse, mask, or accumulated drift over the preceding steps).
    ``third_token``
        Neither.  On the source's own prefix the runtime picked a candidate
        neither side chose free-running, so the free-running divergence is not
        reproduced as a same-prefix disagreement and the localization is
        inconclusive at this step -- read the per-step metrics instead.

    The previous predicate was ``forced_argmax != source_token``, which reports
    "reproduces" for the third case too; on this checkpoint that mislabelled a
    forced argmax of 236764 against source 607 / TensorRT-LLM 531 as a
    reproduction of a divergence it never produced.
    """
    if forced_argmax == actual_token and actual_token != source_token:
        return "reproduces_trtllm_token"
    if forced_argmax == source_token:
        return "matches_source_token"
    return "third_token"


def step_divergence_report(
    step: int,
    actual_token: int,
    expected_row: torch.Tensor,
    actual_row: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Everything needed to localize a first differing greedy token.

    Records both sides' top-2 among the allowed candidates, the two paths'
    logits for each other's choice, and -- when a teacher-forced pass is
    available -- the argmax TensorRT-LLM produces at the same step under the
    source's own prefix, classified by :func:`teacher_forced_outcome`.  Only an
    argmax that lands on the *TensorRT-LLM* token reproduces the divergence;
    landing on the source's token points at the decode/KV-reuse path, and
    landing on a third token says the teacher-forced pass localized nothing.
    """
    src_top = torch.topk(expected_row, 2)
    trt_top = torch.topk(actual_row, 2)
    source_token = int(src_top.indices[0])
    report: Dict[str, Any] = {
        "step": step,
        "trtllm_token": actual_token,
        "source_token": source_token,
        "source_top2": [source_token, int(src_top.indices[1])],
        "source_top2_logits": [float(src_top.values[0]), float(src_top.values[1])],
        "source_top2_margin": float(src_top.values[0] - src_top.values[1]),
        "trtllm_top2": [int(trt_top.indices[0]), int(trt_top.indices[1])],
        "trtllm_top2_logits": [float(trt_top.values[0]), float(trt_top.values[1])],
        "source_logits_at_candidates": [
            float(expected_row[source_token]),
            float(expected_row[actual_token]),
        ],
        "trtllm_logits_at_candidates": [
            float(actual_row[source_token]),
            float(actual_row[actual_token]),
        ],
    }
    if teacher_logits is not None and step < teacher_logits.shape[0]:
        forced_argmax = int(mask_banned(teacher_logits[step], checkpoint_eos_token_ids()).argmax())
        outcome = teacher_forced_outcome(forced_argmax, source_token, actual_token)
        report["teacher_forced_argmax"] = forced_argmax
        report["teacher_forced_outcome"] = outcome
        # True only when the single forward emits the token the free-running run
        # emitted; a third token means the diagnostic localized nothing.
        report["reproduces_from_single_forward"] = outcome == "reproduces_trtllm_token"
    return report


def compare_generation(
    output,
    case: Dict[str, Any],
    *,
    label: str,
    extra: Dict[str, Any],
    localize: Optional[Callable[[], torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Free-running per-step token and logit parity against the source.

    Both sides run the *same* thing: greedy decode, one token at a time, with
    the model's own KV cache.  ``generation_logits[i]`` and the reference's
    ``step_logits[i]`` are therefore the same quantity -- the distribution after
    ``i`` generated tokens -- computed on the same prefix for as long as the
    tokens agree, which is exactly what the acceptance contract requires them to
    do.  Comparing the free-running run against a *context* forward instead
    (teacher forcing) would compare prefill math with decode math and inflate
    the deviation, which is why that pass is now only a failure diagnostic.

    Every one of the >= 32 steps requires cosine >= MIN_COSINE and exact
    greedy-token equality, with exactly one exception, added in iteration 10
    under an explicit human amendment to criterion 6: a step at which the
    *reference's own* top-2 are within its bfloat16 storage resolution
    (:func:`reference_tie`) and the runtime picked one of those same two
    candidates is recorded as an unresolved-reference tie rather than a
    divergence.  A measured logit deviation still does not license an arbitrary
    token -- the runtime has to land on one of the two the reference itself was
    weighing, and every such step is reported with its margin in absolute and
    ULP terms.

    A real divergence is *recorded* rather than raised so the caller can run
    every prompt and persist the full evidence before failing the node -- the
    node still fails, but on a complete picture instead of the first casualty.
    Comparison stops at either kind of fork, because from the next step on the
    two runs are on different branches and their logits are no longer the same
    quantity; ``forked_at_tie`` tells the caller it must recover the remaining
    steps' coverage on a matched prefix instead.
    """
    actual_tokens = list(output.outputs[0].token_ids)
    actual_logits = generation_logits(output)
    expected_tokens = list(case["greedy_tokens"])
    expected_logits = case["step_logits"].float()
    banned = checkpoint_eos_token_ids()

    steps = min(len(expected_tokens), int(expected_logits.shape[0]))
    assert steps >= 32, f"{label}: the reference only supplies {steps} comparable steps"
    # The runtime is asked for one extra token precisely so the overlap
    # scheduler's documented last-step logit drop cannot silently shrink the
    # comparison; requiring the full count here is what makes that explicit.
    assert len(actual_tokens) >= steps, (
        f"{label}: runtime produced {len(actual_tokens)} tokens, need {steps}"
    )
    assert int(actual_logits.shape[0]) >= steps, (
        f"{label}: runtime returned {int(actual_logits.shape[0])} generation-logit rows for "
        f"{len(actual_tokens)} tokens, need {steps}"
    )

    step_metrics: List[Dict[str, Any]] = []
    divergence: Optional[Dict[str, Any]] = None
    unresolved_ties: List[Dict[str, Any]] = []
    for step in range(steps):
        metrics = assert_cosine(
            actual_logits[step],
            expected_logits[step],
            label=f"{label} step[{step}]",
            min_cosine=MIN_COSINE,
        ).as_dict()
        expected_row = mask_banned(expected_logits[step], banned)
        source_top2 = torch.topk(expected_row, 2).values
        # How much room the source itself leaves between its greedy choice and
        # the runner-up.  Not a gate -- reported so a token divergence can be
        # read against the margin it had to cross.
        metrics["source_top2_margin"] = float(source_top2[0] - source_top2[1])
        step_metrics.append(metrics)

        if actual_tokens[step] != expected_tokens[step]:
            tie = reference_tie(expected_row, actual_tokens[step])
            if tie is not None:
                # The reference did not resolve this step at its own storage
                # resolution and the runtime chose one of its two candidates.
                # Recorded, not failed -- and the branches have now forked, so
                # the caller re-establishes the remaining steps' coverage on a
                # matched prefix.
                tie.update(
                    {
                        "step": step,
                        "source_token": expected_tokens[step],
                        "trtllm_top2": [
                            int(i)
                            for i in torch.topk(mask_banned(actual_logits[step], banned), 2).indices
                        ],
                        "step_metrics": metrics,
                    }
                )
                tie["top2_sets_agree"] = sorted(tie["trtllm_top2"]) == sorted(tie["source_top2"])
                unresolved_ties.append(tie)
                break

            teacher: Optional[torch.Tensor] = None
            localization_error: Optional[str] = None
            if localize is not None:
                # Diagnostic only: a failure to produce it must not replace the
                # divergence -- the divergence is the finding.
                try:
                    teacher = localize()
                except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                    localization_error = f"{type(exc).__name__}: {exc}"
            divergence = step_divergence_report(
                step,
                actual_tokens[step],
                expected_row,
                mask_banned(actual_logits[step], banned),
                teacher,
            )
            divergence["step_metrics"] = metrics
            if localization_error is not None:
                divergence["teacher_forced_localization_error"] = localization_error
            break

    fork = divergence or (unresolved_ties[-1] if unresolved_ties else None)
    matched = steps if fork is None else int(fork["step"])
    return {
        "label": label,
        "compared_steps": steps,
        "compared_logit_steps": len(step_metrics),
        "logit_comparison": "free-running decode on both sides (no teacher forcing)",
        "free_running_matched_steps": matched,
        "divergence": divergence,
        "unresolved_reference_ties": unresolved_ties,
        # Set when the run forked on a tie rather than a divergence: the caller
        # owes the remaining steps a matched-prefix comparison.
        "forked_at_tie": None if divergence is not None or not unresolved_ties else matched,
        "tokens": actual_tokens[:steps],
        "source_tokens": expected_tokens[:steps],
        "step_metrics": step_metrics,
        **extra,
    }


def forced_decode_recovery(
    case: Dict[str, Any],
    run: Dict[str, Any],
    *,
    label: str,
) -> Dict[str, Any]:
    """Recover the steps a tie-fork cost, on the production cached-decode path.

    Once the free-running run picks the other side of an unresolved tie, the two
    decodes are on different branches and their later logits stop being the same
    quantity -- so the free-running pass can only certify steps up to the fork.
    Criterion 6 asks for >= 32 compared steps per prompt through prefill *and*
    cache-reuse/decode, and an amendment about *ties* must not quietly become an
    amendment about coverage.

    :func:`forced_decode_run` restores that coverage without leaving the path
    under test: one prefill, then one single-token cached-decode forward per
    step with the source's token forced, so every compared step sits on an
    identical prefix and still goes through ``KVCacheManagerV2`` reuse and,
    in the enabled cell, CUDA-graph replay.  Because both sides are now the same
    kind of computation -- prefill plus cached decode on the same prefix -- the
    per-step logits are directly comparable, so cosine is *gated* here at
    ``MIN_COSINE`` rather than merely reported, alongside greedy-argmax equality
    with the same tie exemption and no other.

    (An earlier revision recovered these steps with a single ``max_tokens=1``
    context request carrying ``prompt + source tokens`` and
    ``return_context_logits=True``.  That is a prefill over a re-rendered
    prompt: it never executes a decode step, never reuses a KV page, and never
    replays a graph, so it could not certify the very path criterion 6 names.)
    """
    forced_logits = run["logits"]
    expected_logits = case["step_logits"].float()
    expected_tokens = list(case["greedy_tokens"])
    banned = checkpoint_eos_token_ids()
    steps = min(len(expected_tokens), int(expected_logits.shape[0]), int(forced_logits.shape[0]))

    mismatches: List[Dict[str, Any]] = []
    ties: List[Dict[str, Any]] = []
    step_metrics: List[Dict[str, Any]] = []
    for step in range(steps):
        metrics = assert_cosine(
            forced_logits[step],
            expected_logits[step],
            label=f"{label} forced-decode step[{step}]",
            min_cosine=MIN_COSINE,
        ).as_dict()
        step_metrics.append(metrics)
        expected_row = mask_banned(expected_logits[step], banned)
        forced_row = mask_banned(forced_logits[step], banned)
        forced_argmax = int(forced_row.argmax())
        if forced_argmax == expected_tokens[step]:
            continue
        tie = reference_tie(expected_row, forced_argmax)
        record = {
            "step": step,
            "source_token": expected_tokens[step],
            "trtllm_forced_argmax": forced_argmax,
            "cosine": metrics["cosine"],
        }
        if tie is not None:
            record.update(tie)
            ties.append(record)
        else:
            mismatches.append(record)

    return {
        "label": label,
        "comparison": (
            "production cached decode forced onto the source's own tokens "
            "(matched prefix at every step, prefill + per-step KV-cache reuse)"
        ),
        "compared_steps": steps,
        "runtime_evidence": run["evidence"],
        "unresolved_reference_ties": ties,
        "mismatches": mismatches,
        "matched_steps": steps - len(ties) - len(mismatches),
        "step_metrics": step_metrics,
    }


def build_mmlu_canary_samples() -> List[Dict[str, Any]]:
    """The fixed 32 MMLU items, rendered exactly as the MMLU gate renders them."""
    import pandas as pd

    dataset_dir = MMLU.DATASET_DIR
    assert os.path.isdir(dataset_dir), (
        f"MMLU dataset not found at {dataset_dir}; the canary and the gate read the same tree"
    )
    # The 5-shot body comes from the reference module's independent
    # re-derivation (it must run without TensorRT-LLM, before the runtime
    # exists).  Cross-check it here against the evaluator the *gate* uses, so
    # "the canary prompt is a gate prompt" is asserted rather than assumed.
    helper = MMLU.EVALUATOR_CLS.__new__(MMLU.EVALUATOR_CLS)
    helper.num_fewshot = 5
    tokenizer = gemma4_processor().tokenizer

    samples: List[Dict[str, Any]] = []
    for subject, row in MMLU_CANARY_SAMPLES:
        prompt, answer = mmlu_five_shot_prompt(dataset_dir, subject, row)
        dev = pd.read_csv(f"{dataset_dir}/dev/{subject}_dev.csv", header=None)
        test = pd.read_csv(f"{dataset_dir}/test/{subject}_test.csv", header=None)
        gate_prompt = helper.gen_prompt(dev, subject, helper.num_fewshot) + helper.format_example(
            test, row, include_answer=False
        )
        assert prompt == gate_prompt, (
            f"MMLU canary prompt for {subject}:{row} differs from the gate evaluator's "
            "own rendering; the canary would not predict the gate"
        )
        # The gate runs the evaluator with `apply_chat_template=True`, which
        # wraps that body in one user turn; the canary has to wrap it the same
        # way or it stops predicting the gate.  `Evaluator.do_apply_chat_template`
        # builds exactly this message list.
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        samples.append(
            {
                "id": f"{subject}:{row}",
                "prompt": rendered,
                "five_shot_body": prompt,
                "input_ids": tokenizer(rendered)["input_ids"],
                "answer": answer,
                "images": [],
            }
        )
    return samples


def score_mmlu(completions: Sequence[Dict[str, Any]], samples: Sequence[Dict[str, Any]]) -> float:
    """The MMLU evaluator's own rule: the completion must start with the label."""
    by_id = {c["id"]: c["text"] for c in completions}
    correct = sum(1 for s in samples if by_id.get(s["id"], "").strip().startswith(s["answer"]))
    return 100.0 * correct / len(samples)


def build_mmmu_canary_samples() -> List[Dict[str, Any]]:
    """The fixed 16 MMMU validation items, rendered with the gate's chat template."""
    dataset_dir = MMMU.DATASET_DIR
    assert os.path.isdir(dataset_dir), f"MMMU dataset not found at {dataset_dir}"
    # Shared with the native-reference driver so both sides render the same
    # bytes; the reference has to build these without importing TensorRT-LLM.
    return mmmu_canary_items(dataset_dir, MMMU_CANARY_SAMPLES, gemma4_processor())


def score_mmmu(completions: Sequence[Dict[str, Any]], samples: Sequence[Dict[str, Any]]) -> float:
    """Score with the repository's own MMMU answer extraction, not a local rule."""
    from tensorrt_llm.evaluate.post_processing import strip_thinking_and_extract_mmmu_answer

    by_id = {c["id"]: c["text"] for c in completions}
    correct = 0
    for sample in samples:
        extracted = strip_thinking_and_extract_mmmu_answer(by_id.get(sample["id"], ""))
        if str(extracted).strip().upper().startswith(sample["answer"].upper()):
            correct += 1
    return 100.0 * correct / len(samples)
