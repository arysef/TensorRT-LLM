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
"""TP8/EP8 mixed-precision load and state-dict accounting for DeepSeek-V4-Flash.

Stage 3 / Goal 3.1. The claim under test is not "the load did not crash" --- a
loader that reads a quantized tensor with the wrong container, the wrong scale
granularity or the wrong dtype crashes on nothing at all and produces plausible
text. So this measures four separate things and fails on any of them:

*consumption*
    Every checkpoint tensor is read exactly once, or falls in a *named*
    structural class (this rank is not the expert-parallel owner; the MTP block
    is not instantiated because speculative decoding is off). Read counts come
    from the lazy ``safetensors`` slices themselves, so "never read" means the
    bytes never left the file rather than "the key was absent from a dict".

*layout*
    The bytes the fused-MoE weight method receives are compared to the bytes on
    disk. Not a tolerance --- byte equality, so a nibble swap, a re-encode or a
    dequantization at the loader boundary is a failure. Scale granularity is
    checked as an identity (``scale.numel() * 32 == out * logical_K``) rather
    than a shape guess, and the FP8/BF16 dense contract is checked per tensor
    family.

*residency*
    Routed-expert parameters must still be the packed byte container after the
    load. The expected size is computed from the config, so "no persistent BF16
    expansion" is an equality against a number this module derives, not a
    comparison against whatever the load happened to produce.

*memory*
    Peak host RSS and peak device allocation per rank, so an OOM-free load is a
    recorded number rather than the absence of a traceback.

Every rank reports its own fingerprint of the resolved MoE stack; the ranks must
agree, because a resolver that differs across ranks deadlocks at the first
routed collective rather than failing here.
"""

from __future__ import annotations

import collections
import os
import re
import resource
import time
from typing import Any

import torch

#: Two FP4 nibbles per byte.
MXFP4_PER_BYTE = 2
#: One UE8M0 scale per this many logical K values.
MXFP4_GROUP = 32
#: FP8 dense weights carry one UE8M0 scale per 128x128 block.
FP8_BLOCK = 128


# ---------------------------------------------------------------------------
# Reading the checkpoint without reading the checkpoint.
# ---------------------------------------------------------------------------


class CountingSlice:
    """A lazy ``safetensors`` slice that records how often it was read.

    The loader probes shape and dtype on every key and materializes only the
    ones it needs, so counting ``__getitem__`` is exactly the "was this tensor
    consumed" question. Wrapping rather than subclassing because ``PySafeSlice``
    is a built-in extension type.
    """

    __slots__ = ("_slice", "reads")

    def __init__(self, wrapped: Any):
        self._slice = wrapped
        self.reads = 0

    def get_shape(self):
        return self._slice.get_shape()

    def get_dtype(self):
        return self._slice.get_dtype()

    def __getitem__(self, item):
        self.reads += 1
        return self._slice[item]


class CountingWeights:
    """``ConsumableWeightsDict`` with per-key read counts.

    Wraps every value in :class:`CountingSlice` and otherwise forwards, so the
    loader sees the mapping it expects.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self._counters: dict[str, CountingSlice] = {}
        for key in list(inner.keys()):
            value = inner[key]
            if hasattr(value, "get_dtype"):
                counter = CountingSlice(value)
                self._counters[key] = counter
                inner[key] = counter

    @property
    def lazy_keys(self) -> int:
        return len(self._counters)

    def read_counts(self) -> dict[str, int]:
        return {key: counter.reads for key, counter in self._counters.items()}

    def __getitem__(self, key):
        return self._inner[key]

    def __setitem__(self, key, value):
        self._inner[key] = value

    def __contains__(self, key):
        return key in self._inner

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)

    def keys(self):
        return self._inner.keys()

    def values(self):
        return self._inner.values()

    def items(self):
        return self._inner.items()

    def get(self, key, default=None):
        return self._inner.get(key, default)


# ---------------------------------------------------------------------------
# Construction and load.
# ---------------------------------------------------------------------------


def build_model_config(checkpoint: str, ranks: Any, max_seq_len: int, max_num_tokens: int):
    """The TP8/EP8 `ModelConfig` this bring-up loads under."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=ranks.world,
        rank=ranks.rank,
        gpus_per_node=8,
        tp_size=ranks.world,
        pp_size=1,
        moe_ep_size=ranks.world,
        moe_tp_size=1,
    )
    return ModelConfig.from_pretrained(
        checkpoint,
        moe_backend="CUTLASS",
        attn_backend="TRTLLM",
        mapping=mapping,
        max_seq_len=max_seq_len,
        max_num_tokens=max_num_tokens,
        # This suite loads; it does not run a collective. NCCL is the one
        # strategy whose `AllReduce` needs no custom IPC workspace, which the
        # torch.distributed communicator cannot allocate.
        allreduce_strategy=AllReduceStrategy.NCCL,
    ), mapping


def construct(model_config: Any) -> tuple[Any, float, bool]:
    """Build the model exactly as ``ModelLoader`` does, meta-init and all."""
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

    started = time.time()
    meta_init = True
    try:
        with MetaInitMode():
            model = AutoModelForCausalLM.from_config(model_config)
        memo: dict[torch.Tensor, torch.Tensor] = {}

        def _init_meta_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.device != torch.device("meta"):
                return t
            if t not in memo:
                memo[t] = torch.empty_like(t, device="cuda")
            return memo[t]

        model._apply(_init_meta_tensor)
        memo.clear()
    except Exception:
        # `ModelLoader` takes the same fallback: `MetaInitMode` rejects the
        # allreduce workspace's `aten.set_.source_Storage`.
        meta_init = False
        model = AutoModelForCausalLM.from_config(model_config)
    model.to("cuda")
    return model, round(time.time() - started, 1), meta_init


def moe_fingerprint(model: Any, model_config: Any) -> dict[str, Any]:
    """What the routed-expert stack resolved to, as strings every rank can compare.

    A resolver that differs across ranks does not fail here --- it deadlocks at
    the first routed collective --- so the fingerprint has to cross an
    ``all_gather_object`` and be compared.
    """
    from tensorrt_llm._utils import get_sm_version

    layer = model.model.layers[3].mlp
    experts = layer.experts
    backend = getattr(experts, "backend", experts)
    quant_method = getattr(backend, "quant_method", None)
    quant_config = getattr(backend, "quant_config", None)
    return {
        "sm_version": get_sm_version(),
        "moe_module": type(experts).__name__,
        "moe_backend": type(backend).__name__,
        "weight_method": type(quant_method).__name__ if quant_method is not None else None,
        "quant_algo": str(getattr(quant_config, "quant_algo", None)),
        "group_size": getattr(quant_config, "group_size", None),
        "routed_scale_key": _routed_scale_key(model_config),
        "experts_per_rank": getattr(backend, "expert_size_per_partition", None),
        "moe_ep_size": model_config.mapping.moe_ep_size,
        "tp_size": model_config.mapping.tp_size,
        "attention_backend": model_config.attn_backend,
        "op_path": "torch.ops.trtllm.fused_moe (CutlassFusedMoE)",
    }


# ---------------------------------------------------------------------------
# Accounting.
# ---------------------------------------------------------------------------


def _expert_id(name: str) -> int | None:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "experts" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def raw_consumption_report(
    read_counts: dict[str, int], local_expert_ids: set | None
) -> dict[str, Any]:
    """Which checkpoint tensors the load actually read off disk.

    This is the streaming claim, measured at the file: a foreign expert with
    zero reads means those bytes never entered this process, which is what makes
    a 149 GB checkpoint loadable eight times on one host. Anything unread that
    is not a named structural class is a hole in the loader.
    """
    consumed, unread = {}, {}
    for name, reads in read_counts.items():
        (consumed if reads else unread)[name] = reads

    classes: dict[str, list[str]] = collections.defaultdict(list)
    for name in unread:
        expert = _expert_id(name)
        if expert is not None and local_expert_ids is not None and expert not in local_expert_ids:
            classes["foreign_expert_parallel_shard"].append(name)
        else:
            classes["unexplained"].append(name)

    read_twice = sorted(name for name, reads in consumed.items() if reads > 1)
    return {
        "checkpoint_tensors": len(read_counts),
        "read": len(consumed),
        "read_exactly_once": sum(1 for r in consumed.values() if r == 1),
        "read_more_than_once": len(read_twice),
        "read_more_than_once_examples": read_twice[:10],
        "never_read": {name: len(keys) for name, keys in classes.items() if name != "unexplained"},
        "unexplained": sorted(classes["unexplained"])[:20],
        "unexplained_count": len(classes["unexplained"]),
        "passed": not classes["unexplained"] and not read_twice,
    }


#: Model keys the V4 loader consumes through a path that is not a parameter of
#: the same name. ``load_flat_hc_weights`` looks up ``<stem>_{fn,base,scale}``
#: and writes ``<stem>.{fn,base,scale}``.
_FLAT_HC_SUFFIXES = tuple(
    f"hc_{site}_{attr}" for site in ("attn", "ffn", "head") for attr in ("fn", "base", "scale")
)

#: Keys the loader deliberately routes to a parameter of a *different* name.
#: These are real destinations, not holes, but matching them by name would fail:
#:
#: * ``self_attn.attn_sink`` becomes ``self_attn.mqa.attn_sink`` --- the sink is
#:   a property of the attention backend, and the loader creates it there;
#: * ``self_attn.o_a_proj.weight_scale_inv`` becomes ``self_attn.o_a_proj_scale``,
#:   because ``o_a_proj`` is a direct MLA parameter rather than a child Linear.
_RENAMED_DESTINATIONS = {
    "self_attn.attn_sink": "self_attn.mqa.attn_sink",
    "self_attn.o_a_proj.weight_scale_inv": "self_attn.o_a_proj_scale",
}

#: ``model.layers.<n>.mlp.experts.<id>.<proj>.<suffix>``.
_ROUTED_EXPERT_KEY = re.compile(r"\.mlp\.experts\.\d+\.")

#: Projections the *model* represents as one wider tensor than the checkpoint
#: does. Both are real many-to-one destinations, so the narrower name never
#: exists as a parameter:
#:
#: * the checkpoint's ``wq_a`` and ``wkv`` are one fused A projection, which is
#:   why ``_copy_deepseek_v4_fused_a_weight_scale`` has to rebuild the block
#:   scale for the fused height;
#: * the shared expert's ``gate_proj`` and ``up_proj`` are one ``gate_up_proj``.
_FUSED_DESTINATIONS = {
    "self_attn.q_a_proj": "self_attn.kv_a_proj_with_mqa",
    "mlp.shared_experts.gate_proj": "mlp.shared_experts.gate_up_proj",
    "mlp.shared_experts.up_proj": "mlp.shared_experts.gate_up_proj",
}

#: The checkpoint's UE8M0 block scale is remapped to ``weight_scale_inv`` --- the
#: DeepSeek-V3 spelling the shared ``Linear`` loader expects --- and that loader
#: writes it into the module's ``weight_scale`` parameter. Same tensor, same
#: meaning (a direct multiplier), different parameter name.
_FP8_SCALE_ALIAS = (".weight_scale_inv", ".weight_scale")


def destination_report(
    remapped_keys: list[str], model: Any, num_hidden_layers: int
) -> dict[str, Any]:
    """Every remapped tensor either lands in a parameter or is named structural.

    "Consumed once" is only meaningful against a destination, so this is the
    other half of :func:`raw_consumption_report`: the model keys the remap emits
    are matched to the parameters that exist after the load. The one structural
    class is the MTP block --- the checkpoint carries it, speculative decoding is
    out of scope for this task, so those layers are never instantiated.
    """
    # `remove_duplicate=False`: `post_load_weights` aliases every layer's
    # `input_layernorm` as the previous layer's `next_layer_layernorm`, and the
    # de-duplicating default reports such a Parameter under only its first name.
    # Layers 1..N-1 would then look like holes while being perfectly loaded.
    parameters = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    mtp_prefix = f"model.layers.{num_hidden_layers}."
    matched, classes = [], collections.defaultdict(list)
    for key in remapped_keys:
        renamed = _renamed_destination(key)
        fused = _fused_destination(key)
        if key in parameters:
            matched.append(key)
        elif key.startswith(mtp_prefix):
            classes["mtp_block_not_instantiated"].append(key)
        elif key.rsplit(".", 1)[-1] in _FLAT_HC_SUFFIXES:
            classes["flat_mhc_weights_via_load_flat_hc_weights"].append(key)
        elif _ROUTED_EXPERT_KEY.search(key):
            # One tensor per expert per projection on the checkpoint side, four
            # tensors per layer on the model side: the fused-MoE method packs
            # every local expert's w1/w3 into `backend.w3_w1_weight` and w2 into
            # `backend.w2_weight`. There is no per-expert parameter to match, so
            # the arrival of these is proven by size instead --- see
            # `residency_report`, which requires the packed tensors to weigh
            # exactly what this rank's experts weigh, and `routed_expert_layout`,
            # which compares the bytes handed over against the file.
            classes["packed_into_fused_moe_weights"].append(key)
        elif renamed is not None and (
            renamed in parameters or _resolve_attribute(model, renamed) is not None
        ):
            matched.append(renamed)
        elif fused is not None and _scale_aliases(fused) & parameters:
            classes["fused_into_a_wider_projection"].append(key)
        elif _scale_aliases(key) & parameters:
            classes["fp8_block_scale_written_to_weight_scale"].append(key)
        else:
            classes["unexplained"].append(key)
    return {
        "remapped_keys": len(remapped_keys),
        "matched_parameters": len(matched),
        "structural": {n: len(v) for n, v in classes.items() if n != "unexplained"},
        "unexplained": sorted(classes["unexplained"])[:20],
        "unexplained_count": len(classes["unexplained"]),
        "passed": not classes["unexplained"],
    }


def _resolve_attribute(model: Any, dotted: str) -> Any:
    """Walk ``dotted`` by attribute, returning ``None`` if any hop is missing.

    Needed because ``MLA`` holds its attention backend (``mqa``) as a plain
    attribute rather than a registered submodule, so nothing the backend owns
    --- the FP32 attention sink among them --- appears in
    ``named_parameters()``. A destination census that only looked at parameter
    names would call the sink a hole on every layer.
    """
    node = model
    for part in dotted.split("."):
        if part == "layers" or part.isdigit():
            node = node[int(part)] if part.isdigit() else getattr(node, part, None)
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    return node


def _renamed_destination(key: str) -> str | None:
    """The parameter a renamed destination actually writes, if this is one."""
    for suffix, replacement in _RENAMED_DESTINATIONS.items():
        if key.endswith("." + suffix):
            return key[: -len(suffix)] + replacement
    return None


def _fused_destination(key: str) -> str | None:
    """The wider parameter this key is fused into, if it is fused at all."""
    for narrow, wide in _FUSED_DESTINATIONS.items():
        marker = f".{narrow}."
        if marker in key:
            return key.replace(marker, f".{wide}.", 1)
    return None


def _scale_aliases(key: str) -> set:
    """``key`` plus the ``weight_scale`` spelling of it, if it is a block scale."""
    inv, plain = _FP8_SCALE_ALIAS
    return {key, key[: -len(inv)] + plain} if key.endswith(inv) else {key}


def conservation_report(surviving_keys: list[str], remapped_keys: list[str]) -> dict[str, Any]:
    """Two checkpoint tensors landing in one model slot is a silent overwrite.

    The remap returns a dict, which is exactly the structure that would hide a
    collision, so the count is reconciled instead: every surviving checkpoint key
    produces one model key, except the documented many-to-one fusions (a
    Compressor's ``wkv`` and ``wgate`` become one ``wkv_gate``) and the
    documented drop (``mtp.0.head.weight``, superseded by the shared head). Any
    other shortfall is a collision.
    """
    fused = collections.Counter()
    dropped = 0
    for key in surviving_keys:
        if key == "mtp.0.head.weight":
            dropped += 1
        elif ".compressor." in key and (
            key.endswith(".wkv.weight") or key.endswith(".wgate.weight")
        ):
            fused[key.rsplit(".", 2)[0]] += 1
    fusions = sum(1 for halves in fused.values() if halves == 2)
    expected = len(surviving_keys) - dropped - fusions
    return {
        "surviving_checkpoint_keys": len(surviving_keys),
        "documented_drops": dropped,
        "documented_fusions": fusions,
        "expected_model_keys": expected,
        "actual_model_keys": len(remapped_keys),
        "collisions": expected - len(remapped_keys),
        "passed": expected == len(remapped_keys),
    }


# ---------------------------------------------------------------------------
# Layout and residency.
# ---------------------------------------------------------------------------


def routed_expert_layout(
    checkpoint: str,
    checkpoint_tensors: dict[str, Any],
    remapped: dict[str, torch.Tensor],
    scale_key: str,
    config: Any,
    samples: list[tuple[int, int, str]],
) -> dict[str, Any]:
    """Byte-level proof that the loader boundary re-encoded nothing.

    ``remapped`` holds exactly what the fused-MoE weight method is handed. The
    comparison against the file is byte equality after the ``I8 -> U8`` container
    view --- the only reinterpretation the boundary is allowed to make --- so a
    nibble swap, a re-encode or a dequantization here is a failure rather than a
    tolerance. What the Cutlass method then does to those bytes (shuffle and
    interleave for its mixed GEMM) is that method's own contract, and the Goal
    3.2 MoE activation replay is what proves it.
    """
    import safetensors

    hidden = config.hidden_size
    inter = config.moe_intermediate_size
    checks: dict[str, Any] = {}
    problems: list[str] = []
    handles: dict[str, Any] = {}
    for layer, expert, proj in samples:
        raw_key = f"layers.{layer}.ffn.experts.{expert}.{proj}"
        info = checkpoint_tensors.get(f"{raw_key}.weight")
        if info is None:
            problems.append(f"{raw_key}.weight absent from the checkpoint")
            continue
        path = os.path.join(checkpoint, info.file)
        if path not in handles:
            handles[path] = safetensors.safe_open(path, framework="pt", device="cpu")
        handle = handles[path]

        out_features, logical_k = (hidden, inter) if proj == "w2" else (inter, hidden)
        on_disk = handle.get_tensor(f"{raw_key}.weight")
        on_disk_scale = handle.get_tensor(f"{raw_key}.scale")
        model_key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
        loaded = remapped.get(f"{model_key}.weight")
        loaded_scale = remapped.get(f"{model_key}.{scale_key}")

        entry: dict[str, Any] = {
            "checkpoint_key": raw_key,
            "model_key": model_key,
            "model_scale_key": f"{model_key}.{scale_key}",
            "out_features": out_features,
            "logical_k": logical_k,
            "checkpoint_dtype": str(on_disk.dtype),
            "checkpoint_shape": list(on_disk.shape),
            "checkpoint_scale_dtype": str(on_disk_scale.dtype),
            "checkpoint_scale_shape": list(on_disk_scale.shape),
        }
        if loaded is None or loaded_scale is None:
            problems.append(f"{model_key}: weight or {scale_key} missing after remap")
            entry["passed"] = False
            checks[f"L{layer}.E{expert}.{proj}"] = entry
            continue

        entry.update(
            {
                "loaded_dtype": str(loaded.dtype),
                "loaded_shape": list(loaded.shape),
                "loaded_scale_dtype": str(loaded_scale.dtype),
                "loaded_scale_shape": list(loaded_scale.shape),
                "bytes_identical": bool(
                    torch.equal(loaded.view(torch.uint8), on_disk.view(torch.uint8))
                ),
                "scale_bytes_identical": bool(
                    torch.equal(loaded_scale.view(torch.uint8), on_disk_scale.view(torch.uint8))
                ),
                "packed_two_nibbles_per_byte": loaded.numel() * MXFP4_PER_BYTE
                == out_features * logical_k,
                "one_ue8m0_scale_per_32_k": loaded_scale.numel() * MXFP4_GROUP
                == out_features * logical_k,
                "container_is_uint8": loaded.dtype == torch.uint8
                and loaded_scale.dtype == torch.uint8,
            }
        )
        entry["passed"] = all(
            entry[key]
            for key in (
                "bytes_identical",
                "scale_bytes_identical",
                "packed_two_nibbles_per_byte",
                "one_ue8m0_scale_per_32_k",
                "container_is_uint8",
            )
        )
        if not entry["passed"]:
            problems.append(f"{model_key}: {entry}")
        checks[f"L{layer}.E{expert}.{proj}"] = entry

    return {"samples": checks, "problems": problems, "passed": bool(checks) and not problems}


def expected_routed_bytes(config: Any, num_layers: int, experts_per_rank: int) -> int:
    """Packed size of one rank's routed experts, derived from the config alone.

    Independent of what the load produced, so comparing against it is a test of
    "still packed" rather than a restatement of the measurement.
    """
    hidden = config.hidden_size
    inter = config.moe_intermediate_size
    per_expert = 0
    for out_features, logical_k in ((inter, hidden), (inter, hidden), (hidden, inter)):
        per_expert += out_features * logical_k // MXFP4_PER_BYTE
        per_expert += out_features * logical_k // MXFP4_GROUP
    return per_expert * experts_per_rank * num_layers


def residency_report(model: Any, config: Any, num_layers: int) -> dict[str, Any]:
    """Parameter dtypes and byte totals after the load.

    ``routed_expert_bytes`` against :func:`expected_routed_bytes` is the "no
    persistent BF16 expansion" check: a dequantized routed expert is four times
    the packed size, and nothing else about a successful load would say so.
    """
    by_dtype: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tensors": 0, "bytes": 0}
    )
    routed_bytes = 0
    routed_dtypes: collections.Counter = collections.Counter()
    experts_per_rank = None
    for name, param in model.named_parameters():
        entry = by_dtype[str(param.dtype)]
        entry["tensors"] += 1
        entry["bytes"] += param.numel() * param.element_size()
        if ".experts." in name:
            routed_bytes += param.numel() * param.element_size()
            routed_dtypes[str(param.dtype)] += 1
            if name.endswith("w2_weight"):
                experts_per_rank = param.shape[0]

    expected = (
        expected_routed_bytes(config, num_layers, experts_per_rank) if experts_per_rank else None
    )
    fp8 = by_dtype.get("torch.float8_e4m3fn", {}).get("tensors", 0)
    bf16 = by_dtype.get("torch.bfloat16", {}).get("tensors", 0)
    unpacked = sorted(d for d in routed_dtypes if d not in ("torch.uint8", "torch.int8"))
    problems = []
    if unpacked:
        problems.append(f"routed-expert parameters left in {unpacked}, not the packed container")
    if expected is not None and routed_bytes != expected:
        problems.append(
            f"routed-expert parameters are {routed_bytes} bytes, packed layout is {expected}"
        )
    if not fp8:
        problems.append("no FP8-E4M3 parameter survived the load; the dense contract is gone")
    if not bf16:
        problems.append("no BF16 parameter survived the load")
    return {
        "by_dtype": {k: dict(v) for k, v in sorted(by_dtype.items())},
        "parameter_bytes": sum(v["bytes"] for v in by_dtype.values()),
        "routed_expert_bytes": routed_bytes,
        "routed_expert_bytes_expected_packed": expected,
        "routed_expert_dtypes": dict(routed_dtypes),
        "experts_per_rank": experts_per_rank,
        "fp8_e4m3_tensors": fp8,
        "bfloat16_tensors": bf16,
        "problems": problems,
        "passed": not problems,
    }


def dense_contract_report(model: Any, layer: int = 0) -> dict[str, Any]:
    """The attention/dense FP8 + BF16 contract, tensor family by tensor family.

    Names are the model's, not the checkpoint's, and the two differ in ways that
    matter: the checkpoint's ``wq_a`` and ``wkv`` are one fused A projection in
    the model, the output projection is ``o_a_proj`` / ``o_b_proj`` rather than a
    single ``o_proj``, and the FP32 attention sink lives on the attention backend,
    which ``MLA`` holds outside the module tree.

    ``o_a_proj`` is the one family the loader deliberately widens: the checkpoint
    stores it FP8 with a UE8M0 block scale and the SM90 grouped O-LoRA path
    consumes BF16, so the loader dequantizes and keeps the scale alongside in
    ``o_a_proj_scale``. That is a declared conversion with the scale retained,
    which is what separates it from a silent reinterpretation.
    """
    params = dict(model.named_parameters())
    prefix = f"model.layers.{layer}"
    attn = f"{prefix}.self_attn"
    families = {
        # wq_a + wkv, fused into one A projection by the model.
        "fused_a_proj": (f"{attn}.kv_a_proj_with_mqa.weight", "F8_E4M3", torch.float8_e4m3fn),
        "wq_b": (f"{attn}.q_b_proj.weight", "F8_E4M3", torch.float8_e4m3fn),
        "wo_b": (f"{attn}.o_b_proj.weight", "F8_E4M3", torch.float8_e4m3fn),
        "wo_a_dequantized": (f"{attn}.o_a_proj", "F8_E4M3", torch.bfloat16),
        "wo_a_scale_retained": (f"{attn}.o_a_proj_scale", "F8_E8M0", torch.float32),
        "q_norm": (f"{attn}.q_a_layernorm.weight", "BF16", torch.bfloat16),
        "kv_norm": (f"{attn}.kv_a_layernorm.weight", "BF16", torch.bfloat16),
        "attn_norm": (f"{prefix}.input_layernorm.weight", "BF16", torch.bfloat16),
        "ffn_norm": (f"{prefix}.post_attention_layernorm.weight", "BF16", torch.bfloat16),
        "mhc_mix": (f"{prefix}.hc_attn.fn", "F32", torch.float32),
        "shared_expert": (
            f"{prefix}.mlp.shared_experts.down_proj.weight",
            "F8_E4M3",
            torch.float8_e4m3fn,
        ),
        "router": (f"{prefix}.mlp.gate.weight", "BF16", torch.bfloat16),
        "hash_table": (f"{prefix}.mlp.gate.tid2eid", "I64", torch.int32),
    }
    checks: dict[str, Any] = {}
    problems: list[str] = []
    for family, (model_key, ckpt_dtype, want) in families.items():
        param = params.get(model_key)
        if param is None:
            # The sink is the case this exists for: it is owned by the attention
            # backend, which is not in the module tree.
            param = _resolve_attribute(model, model_key)
        if param is None:
            problems.append(f"{family}: {model_key} is absent from the loaded model")
            continue
        ok = param.dtype == want
        checks[family] = {
            "model_key": model_key,
            "checkpoint_dtype": ckpt_dtype,
            "loaded_dtype": str(param.dtype),
            "expected_dtype": str(want),
            "passed": ok,
        }
        if not ok:
            problems.append(f"{family}: loaded as {param.dtype}, expected {want}")

    sink = _resolve_attribute(model, f"{attn}.mqa.attn_sink")
    checks["attn_sink"] = {
        "model_key": f"{attn}.mqa.attn_sink",
        "checkpoint_dtype": "F32",
        "loaded_dtype": str(sink.dtype) if sink is not None else None,
        "expected_dtype": str(torch.float32),
        "note": "owned by the attention backend, which MLA holds outside the module tree",
        "passed": sink is not None and sink.dtype == torch.float32,
    }
    if not checks["attn_sink"]["passed"]:
        problems.append("attn_sink is missing or not FP32 after the load")

    weight = params.get(f"{attn}.kv_a_proj_with_mqa.weight")
    scale = params.get(f"{attn}.kv_a_proj_with_mqa.weight_scale")
    if weight is not None and scale is not None:
        blocks = (-(-weight.shape[0] // FP8_BLOCK), -(-weight.shape[1] // FP8_BLOCK))
        ok = tuple(scale.shape[-2:]) == blocks
        checks["fp8_block_scale_granularity"] = {
            "model_key": f"{attn}.kv_a_proj_with_mqa.weight_scale",
            "weight_shape": list(weight.shape),
            "scale_shape": list(scale.shape),
            "expected_blocks": list(blocks),
            "passed": ok,
        }
        if not ok:
            problems.append(
                f"fused A projection scale {tuple(scale.shape)} is not one per "
                f"{FP8_BLOCK}x{FP8_BLOCK} block"
            )
    else:
        problems.append("the fused A projection has no FP8 block scale after the load")
    return {"families": checks, "problems": problems, "passed": not problems}


# ---------------------------------------------------------------------------
# Memory.
# ---------------------------------------------------------------------------


def memory_report() -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_host_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
        "gpu_free_gb": round(free / 2**30, 2),
        "gpu_total_gb": round(total / 2**30, 2),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run(args: Any, ranks: Any, inventory_module: Any) -> tuple[dict[str, Any], "Loaded"]:
    """Load the checkpoint at TP8/EP8 and report every account.

    Two passes over the surviving keys, deliberately: the first is instrumentation
    (it produces the model-key list, the collision reconciliation and the byte
    sample the fused-MoE method will see), the second is the *production* load
    whose read counts are the consumption evidence. The counters are zeroed
    between them, and the instrumentation dict is dropped before the real load,
    so neither the counts nor the peak memory measure this module's own work.

    Returns the report *and* the live model. Goal 3.2's MoE activation replay
    runs against the very model this accounting just proved, so handing it back
    is what makes "the weights the replay used are the weights this suite
    audited" a fact rather than a second load nobody checked.
    """
    from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
    from tensorrt_llm._torch.models.modeling_deepseekv4 import (
        _deepseek_v4_local_expert_ids,
        _remap_deepseek_v4_checkpoint_keys,
    )

    inventory = inventory_module.load_inventory(args.checkpoint)
    inventory_module.raise_for_problems(inventory_module.verify(inventory))
    ranks.log(f"  checkpoint inventory verified: {len(inventory.tensors)} tensors")

    model_config, mapping = build_model_config(
        args.checkpoint, ranks, args.max_seq_len, args.max_num_tokens
    )
    model, construct_s, meta_init = construct(model_config)
    ranks.log(f"  constructed {construct_s:.1f}s (meta_init={meta_init})")

    streamed = HfWeightLoader._streams_rank_local_weights(args.checkpoint)
    loader = HfWeightLoader()
    raw = loader.load_weights(args.checkpoint, mapping)
    counting = CountingWeights(raw)
    lazy = counting.lazy_keys == len(counting)

    num_layers = model_config.pretrained_config.num_hidden_layers
    local_expert_ids = _deepseek_v4_local_expert_ids(model_config)
    routed_quant_config = _routed_quant_config(model_config)
    remap_kwargs = dict(
        num_hidden_layers=num_layers,
        kv_lora_rank=model_config.pretrained_config.kv_lora_rank,
        local_expert_ids=local_expert_ids,
        routed_quant_config=routed_quant_config,
    )

    # -- pass 1: instrumentation ------------------------------------------
    surviving = [
        key
        for key in counting.keys()
        if local_expert_ids is None
        or (_expert_id(key) is None or _expert_id(key) in local_expert_ids)
    ]
    remapped = _remap_deepseek_v4_checkpoint_keys(dict(counting.items()), **remap_kwargs)
    scale_key = _routed_scale_key(model_config)
    sample_expert = _sample_expert(local_expert_ids)
    # One ratio-4 layer and one ratio-128 layer, and all three projections, so
    # the sample spans both compression classes and both K geometries.
    layout_samples = [(2, sample_expert, "w1"), (2, sample_expert, "w3"), (3, sample_expert, "w2")]
    layout = routed_expert_layout(
        args.checkpoint,
        inventory.tensors,
        remapped,
        scale_key,
        model_config.pretrained_config,
        samples=layout_samples,
    )
    conservation = conservation_report(surviving, list(remapped))
    remapped_keys = list(remapped)
    del remapped
    for counter in counting._counters.values():
        counter.reads = 0
    torch.cuda.reset_peak_memory_stats()

    # -- pass 2: the production load --------------------------------------
    started = time.time()
    model.load_weights(counting)
    model.post_load_weights()
    load_s = round(time.time() - started, 1)
    ranks.log(f"  loaded {load_s:.1f}s")

    fingerprint = moe_fingerprint(model, model_config)
    # After the load, not before: `attn_sink` and the derived projections are
    # created *by* the load, so a destination census taken earlier would call
    # them holes.
    destinations = destination_report(remapped_keys, model, num_layers)
    checks = {
        "raw_tensor_consumption": raw_consumption_report(counting.read_counts(), local_expert_ids),
        "model_key_destinations": destinations,
        "no_duplicate_slots": conservation,
        "routed_expert_layout": layout,
        "routed_expert_residency": residency_report(
            model, model_config.pretrained_config, num_layers
        ),
        "dense_fp8_bf16_contract": dense_contract_report(model),
    }
    report = {
        "hard_config": {
            "tensor_parallel_size": mapping.tp_size,
            "moe_expert_parallel_size": mapping.moe_ep_size,
            "backend": "pytorch",
            "attention_backend": "TRTLLM",
            "max_seq_len": args.max_seq_len,
            "max_num_tokens": args.max_num_tokens,
            "tokens_per_block": 128,
            "cuda_graph": False,
            "overlap_scheduler": False,
            "streamed_rank_local_weights": streamed,
            "lazy_checkpoint_values": lazy,
            "local_expert_ids": sorted(local_expert_ids) if local_expert_ids else None,
            "layout_sample": [
                f"layer{layer}.expert{expert}.{proj}" for layer, expert, proj in layout_samples
            ],
        },
        "moe_fingerprint": fingerprint,
        "timings": {"construct_s": construct_s, "load_s": load_s, "meta_init": meta_init},
        "memory": memory_report(),
        "checks": checks,
        "local_passed": all(c["passed"] for c in checks.values()),
    }
    return report, Loaded(
        model=model,
        model_config=model_config,
        mapping=mapping,
        local_expert_ids=local_expert_ids,
        routed_scale_key=scale_key,
    )


class Loaded:
    """The live TP8/EP8 model plus the handles Goal 3.2's replay needs.

    Deliberately not part of the JSON report: these are CUDA objects, and
    letting them leak into the artifact would either explode it or silently
    stringify a 20 GiB model.
    """

    __slots__ = ("model", "model_config", "mapping", "local_expert_ids", "routed_scale_key")

    def __init__(
        self,
        model: Any,
        model_config: Any,
        mapping: Any,
        local_expert_ids: set | None,
        routed_scale_key: str,
    ):
        self.model = model
        self.model_config = model_config
        self.mapping = mapping
        self.local_expert_ids = local_expert_ids
        self.routed_scale_key = routed_scale_key


def _routed_quant_config(model_config: Any) -> Any:
    from tensorrt_llm._torch.models.modeling_deepseekv4 import _deepseek_v4_routed_quant_config

    return _deepseek_v4_routed_quant_config(model_config)


def _routed_scale_key(model_config: Any) -> str:
    from tensorrt_llm._torch.models.modeling_deepseekv4 import _deepseek_v4_routed_moe_scale_name

    return _deepseek_v4_routed_moe_scale_name(True, _routed_quant_config(model_config))


def _sample_expert(local_expert_ids: set | None) -> int:
    return min(local_expert_ids) if local_expert_ids else 0
