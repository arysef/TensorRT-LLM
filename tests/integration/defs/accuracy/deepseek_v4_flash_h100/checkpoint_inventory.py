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
"""Structural inventory gate for the DeepSeek-V4-Flash checkpoint.

This reads only the safetensors *headers* (name, dtype, shape) plus
``config.json``. It never materialises a weight, so it checks the 159 GB
checkpoint in seconds and needs no GPU.

The gate has two independent halves, and that split is the whole point:

1. :func:`check_config` compares ``config.json`` against
   :data:`TARGET_CONFIG` / :data:`TARGET_COMPRESS_RATIOS` --- *pinned
   literals* transcribed from the plan's variant-inventory table. Nothing is
   read back out of the file to decide what the file should contain, so
   editing a target-critical field (RoPE theta, sliding window, Indexer
   top-k, scoring function, SwiGLU limit, ...) fails here instead of silently
   moving the goalposts. The comparison is *exact in both directions*: a key
   the contract does not pin is a problem too, because an unpinned key that
   the model actually reads (``rope_scaling.mscale`` amplitude scaling, an
   extra ``quantization_config`` field) changes semantics just as much as a
   wrong value. Non-semantic provenance keys are allow-listed by name in
   :data:`NON_SEMANTIC_CONFIG_KEYS` and echoed into the summary rather than
   ignored.
2. :func:`check_tensors` builds the *complete* expected tensor map from those
   same pinned literals and diffs it against every header in the checkpoint.
   Every layer, every one of the 256 routed experts per layer, and every
   weight/scale companion is covered --- not a per-layer sample --- so a
   single wrong dtype, wrong shape, missing tensor, or unexpected extra
   tensor is reported.

The expected shapes are *derived from the quantisation rules* rather than
hand-copied, so the map also encodes the contract itself: FP8 E4M3 weights
carry one UE8M0 scale per 128x128 block, and packed MXFP4 routed weights sit
in an ``I8`` byte container with one UE8M0 scale per 32 *logical* K values.

``tests/unittest/_torch/modeling/test_deepseekv4_checkpoint_inventory.py``
holds the negative coverage: it builds a synthetic checkpoint that satisfies
the contract, then mutates one field at a time and asserts each mutation is
caught.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any

# Byte width per safetensors dtype string, for on-disk accounting.
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "U8": 1,
    "I32": 4,
    "I64": 8,
}

# FP8 dense/attention weights carry one UE8M0 scale per 128x128 weight block.
FP8_BLOCK = 128
# Packed MXFP4 routed-expert weights carry one UE8M0 scale per 32 logical K.
MXFP4_GROUP = 32
# Two 4-bit nibbles share one I8 container byte.
MXFP4_PER_BYTE = 2

CHECKPOINT_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

# ---------------------------------------------------------------------------
# The pinned target contract.
#
# These are literals, not values read back from the checkpoint. They are the
# single supported variant of DeepSeek-V4-Flash for this bring-up (plan.md,
# "Source and Variant Inventory"). A checkpoint or config that disagrees with
# any of them is out of scope and must fail loudly rather than be silently
# accepted with different semantics.
# ---------------------------------------------------------------------------
TARGET_CONFIG: dict[str, Any] = {
    # -- identity ----------------------------------------------------------
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "torch_dtype": "bfloat16",
    # -- decoder -----------------------------------------------------------
    "num_hidden_layers": 43,
    "hidden_size": 4096,
    "vocab_size": 129280,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "use_cache": True,
    # -- attention geometry ------------------------------------------------
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "attention_bias": False,
    "attention_dropout": 0.0,
    # -- sparse schedule ---------------------------------------------------
    "sliding_window": 128,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
    # -- positional encoding ----------------------------------------------
    "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "compress_rope_theta": 160000,
    # The official `precompute_freqs_cis` (inference/model.py) applies YaRN
    # *frequency interpolation only* -- there is no amplitude term anywhere in
    # the source formula. So this mapping is exact: an extra `mscale` /
    # `mscale_all_dim` (which V3 does honour) would mean the checkpoint expects
    # attention-magnitude scaling that V4 never applies.
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    # -- MoE ---------------------------------------------------------------
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "swiglu_limit": 10.0,
    "num_hash_layers": 3,
    "expert_dtype": "fp4",
    # -- mHC ---------------------------------------------------------------
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    # -- speculative decoding (structural only for this task) --------------
    "num_nextn_predict_layers": 1,
    # -- quantisation ------------------------------------------------------
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
}

# Layers 0-1 are SWA-only (ratio 0); even layers 2-42 run ratio-4 CSA with a
# learned Indexer; odd layers 3-41 run ratio-128 HCA. The trailing entry is
# the ratio-0 MTP layer, which is structural/unreachable for this task.
TARGET_COMPRESS_RATIOS: list[int] = [0, 0] + [4, 128] * 20 + [4] + [0]

# `compress_ratios` is a pinned list rather than a scalar, so it is diffed
# element-wise by check_config instead of living in TARGET_CONFIG.
COMPRESS_RATIOS_KEY = "compress_ratios"

# Top-level keys that are deliberately *not* pinned, with the reason each one
# is safe to leave unpinned. Everything else at top level must be in
# TARGET_CONFIG: an unrecognised key may well be one the model reads, and
# accepting it silently is how a different variant gets mistaken for this one.
# These are echoed into the summary (`config_metadata`) so an evidence reader
# still sees what was allowed through.
NON_SEMANTIC_CONFIG_KEYS: dict[str, str] = {
    # Training-time weight-init spread; never read at inference.
    "initializer_range": "training-only initialisation parameter",
    # Provenance stamp of the exporting library. The checkpoint revision, not
    # this string, is what pins the weights.
    "transformers_version": "exporter provenance, not consumed by the model",
}

# Which routed-expert projection is K-major over which dimension. w1/w3 read
# the 4096-wide hidden state; w2 reads the 2048-wide SwiGLU intermediate.
_ROUTED_PROJECTIONS = ("w1", "w2", "w3")


class InventoryError(AssertionError):
    """Raised when the checkpoint does not match the expected V4 contract."""


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: str

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self) -> int:
        return self.numel * _DTYPE_BYTES.get(self.dtype, 0)


@dataclass
class Inventory:
    checkpoint: str
    config: dict[str, Any]
    tensors: dict[str, TensorInfo]
    # Problems found while reading the checkpoint off disk (index vs shard
    # disagreements). :func:`verify` seeds ``problems`` from these.
    load_problems: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _read_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def _cross_check_index(
    tensors: dict[str, TensorInfo],
    index_map: dict[str, str],
    shard_files: list[str],
    problems: list[str],
) -> None:
    """Require the index and the shard headers to describe the same weights."""
    for shard in sorted(set(index_map.values()) - set(shard_files)):
        problems.append(f"shard {shard}: listed in the index but not present on disk")
    for name in sorted(set(tensors) - set(index_map)):
        problems.append(
            f"{name}: present in {tensors[name].file} but absent from the index "
            "-- a loader that walks the shards would still read it"
        )
    for name in sorted(set(index_map) - set(tensors)):
        problems.append(
            f"{name}: listed in the index (shard {index_map[name]}) but absent "
            "from every shard header"
        )
    for name, shard in sorted(index_map.items()):
        got = tensors.get(name)
        if got is not None and got.file != shard:
            problems.append(f"{name}: index says shard {shard}, header found it in {got.file}")


def load_inventory(checkpoint: str) -> Inventory:
    """Read config.json plus every safetensors header under ``checkpoint``.

    Tensors come from the *shard headers*, not from
    ``model.safetensors.index.json``: the index is a convenience map, while
    the shards are what a loader actually opens. The two are then cross-
    checked, so a tensor that exists on disk but is missing from the index
    (which the index-driven walk would silently skip, yet a shard-walking
    loader would still read) is reported rather than ignored.
    """
    with open(os.path.join(checkpoint, "config.json")) as f:
        config = json.load(f)
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    with open(index_path) as f:
        index_map: dict[str, str] = json.load(f)["weight_map"]

    problems: list[str] = []
    shard_files = sorted(f for f in os.listdir(checkpoint) if f.endswith(".safetensors"))
    if not shard_files:
        problems.append(f"{checkpoint}: no *.safetensors shard found")

    tensors: dict[str, TensorInfo] = {}
    for shard in shard_files:
        for name, entry in _read_header(os.path.join(checkpoint, shard)).items():
            if name == "__metadata__":
                continue
            if name in tensors:
                problems.append(
                    f"{name}: duplicated across shards {tensors[name].file} and {shard}"
                )
                continue
            tensors[name] = TensorInfo(
                name=name, dtype=entry["dtype"], shape=tuple(entry["shape"]), file=shard
            )

    _cross_check_index(tensors, index_map, shard_files, problems)
    return Inventory(checkpoint=checkpoint, config=config, tensors=tensors, load_problems=problems)


def _normalise(name: str) -> str:
    """Collapse layer/expert indices so tensor families group together."""
    name = re.sub(r"(^|\.)layers\.\d+\.", r"\1layers.N.", name)
    name = re.sub(r"(^|\.)mtp\.\d+\.", r"\1mtp.M.", name)
    name = re.sub(r"(^|\.)experts\.\d+\.", r"\1experts.E.", name)
    return name


# ---------------------------------------------------------------------------
# Expected tensor map, derived from the pinned contract.
# ---------------------------------------------------------------------------

Spec = dict[str, tuple[str, tuple[int, ...]]]


def _add(spec: Spec, name: str, dtype: str, shape: tuple[int, ...]) -> None:
    spec[name] = (dtype, tuple(shape))


def _add_fp8(spec: Spec, name: str, out_features: int, in_features: int) -> None:
    """FP8 E4M3 weight plus its one-UE8M0-scale-per-128x128-block companion."""
    _add(spec, f"{name}.weight", "F8_E4M3", (out_features, in_features))
    _add(
        spec,
        f"{name}.scale",
        "F8_E8M0",
        (
            -(-out_features // FP8_BLOCK),
            -(-in_features // FP8_BLOCK),
        ),
    )


def _add_mxfp4(spec: Spec, name: str, out_features: int, logical_in: int) -> None:
    """Packed MXFP4 weight (I8 container) plus one UE8M0 scale per 32 K."""
    _add(spec, f"{name}.weight", "I8", (out_features, logical_in // MXFP4_PER_BYTE))
    _add(spec, f"{name}.scale", "F8_E8M0", (out_features, logical_in // MXFP4_GROUP))


def _add_block(spec: Spec, prefix: str, ratio: int, *, is_mtp: bool) -> None:
    """Emit every tensor of one decoder block (a real layer or the MTP head).

    ``prefix`` is ``layers.<i>`` or ``mtp.<j>``; ``ratio`` is that block's
    compression ratio, which decides whether a Compressor and an Indexer are
    present.
    """
    cfg = TARGET_CONFIG
    dim = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    q_rank = cfg["q_lora_rank"]
    o_groups = cfg["o_groups"]
    o_rank = cfg["o_lora_rank"]
    idx_heads = cfg["index_n_heads"]
    idx_dim = cfg["index_head_dim"]
    n_experts = cfg["n_routed_experts"]
    moe_inter = cfg["moe_intermediate_size"]
    hc_mult = cfg["hc_mult"]
    mix_hc = (2 + hc_mult) * hc_mult

    # -- norms -------------------------------------------------------------
    _add(spec, f"{prefix}.attn_norm.weight", "BF16", (dim,))
    _add(spec, f"{prefix}.ffn_norm.weight", "BF16", (dim,))

    # -- attention ---------------------------------------------------------
    attn = f"{prefix}.attn"
    _add(spec, f"{attn}.q_norm.weight", "BF16", (q_rank,))
    _add(spec, f"{attn}.kv_norm.weight", "BF16", (head_dim,))
    _add(spec, f"{attn}.attn_sink", "F32", (n_heads,))
    _add_fp8(spec, f"{attn}.wq_a", q_rank, dim)
    _add_fp8(spec, f"{attn}.wq_b", n_heads * head_dim, q_rank)
    _add_fp8(spec, f"{attn}.wkv", head_dim, dim)
    _add_fp8(spec, f"{attn}.wo_a", o_groups * o_rank, n_heads * head_dim // o_groups)
    _add_fp8(spec, f"{attn}.wo_b", dim, o_groups * o_rank)

    # -- Compressor / Indexer, keyed off the compression ratio -------------
    if ratio != 0:
        # Ratio 4 pools overlapping groups, so its Compressor is twice as
        # wide; ratio 128 pools disjoint groups at the plain latent width.
        width = (2 if ratio == 4 else 1) * head_dim
        comp = f"{attn}.compressor"
        _add(spec, f"{comp}.ape", "F32", (ratio, width))
        _add(spec, f"{comp}.norm.weight", "BF16", (head_dim,))
        _add(spec, f"{comp}.wgate.weight", "BF16", (width, dim))
        _add(spec, f"{comp}.wkv.weight", "BF16", (width, dim))

        if ratio == 4:
            # Only ratio-4 layers learn a selection Indexer; ratio-128 layers
            # attend to every valid compressed slot.
            idx = f"{attn}.indexer"
            _add_fp8(spec, f"{idx}.wq_b", idx_heads * idx_dim, q_rank)
            _add(spec, f"{idx}.weights_proj.weight", "BF16", (idx_heads, dim))
            _add(spec, f"{idx}.compressor.ape", "F32", (ratio, 2 * idx_dim))
            _add(spec, f"{idx}.compressor.norm.weight", "BF16", (idx_dim,))
            _add(spec, f"{idx}.compressor.wgate.weight", "BF16", (2 * idx_dim, dim))
            _add(spec, f"{idx}.compressor.wkv.weight", "BF16", (2 * idx_dim, dim))

    # -- mHC ---------------------------------------------------------------
    for kind in ("attn", "ffn"):
        _add(spec, f"{prefix}.hc_{kind}_fn", "F32", (mix_hc, hc_mult * dim))
        _add(spec, f"{prefix}.hc_{kind}_base", "F32", (mix_hc,))
        _add(spec, f"{prefix}.hc_{kind}_scale", "F32", (3,))

    # -- routing -----------------------------------------------------------
    gate = f"{prefix}.ffn.gate"
    _add(spec, f"{gate}.weight", "BF16", (n_experts, dim))
    layer_index = int(prefix.rsplit(".", 1)[1])
    hash_routed = (not is_mtp) and layer_index < cfg["num_hash_layers"]
    if hash_routed:
        # Layers 0-2 route by a checkpoint token-id table, so they carry no
        # correction bias.
        _add(spec, f"{gate}.tid2eid", "I64", (cfg["vocab_size"], cfg["num_experts_per_tok"]))
    else:
        _add(spec, f"{gate}.bias", "F32", (n_experts,))

    # -- experts -----------------------------------------------------------
    for proj in _ROUTED_PROJECTIONS:
        # w2 contracts the SwiGLU intermediate; w1/w3 contract the hidden dim.
        out_f, in_f = (dim, moe_inter) if proj == "w2" else (moe_inter, dim)
        _add_fp8(spec, f"{prefix}.ffn.shared_experts.{proj}", out_f, in_f)
        for expert in range(n_experts):
            _add_mxfp4(spec, f"{prefix}.ffn.experts.{expert}.{proj}", out_f, in_f)

    # -- MTP-only glue -----------------------------------------------------
    if is_mtp:
        _add(spec, f"{prefix}.enorm.weight", "BF16", (dim,))
        _add(spec, f"{prefix}.hnorm.weight", "BF16", (dim,))
        _add(spec, f"{prefix}.norm.weight", "BF16", (dim,))
        _add_fp8(spec, f"{prefix}.e_proj", dim, dim)
        _add_fp8(spec, f"{prefix}.h_proj", dim, dim)
        _add(spec, f"{prefix}.hc_head_fn", "F32", (hc_mult, hc_mult * dim))
        _add(spec, f"{prefix}.hc_head_base", "F32", (hc_mult,))
        _add(spec, f"{prefix}.hc_head_scale", "F32", (1,))


def expected_tensors() -> Spec:
    """Every tensor the target checkpoint must contain, and nothing else.

    Built purely from :data:`TARGET_CONFIG` / :data:`TARGET_COMPRESS_RATIOS`,
    never from the checkpoint under test.
    """
    cfg = TARGET_CONFIG
    dim = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    hc_mult = cfg["hc_mult"]
    n_layers = cfg["num_hidden_layers"]

    spec: Spec = {}
    _add(spec, "embed.weight", "BF16", (vocab, dim))
    _add(spec, "head.weight", "BF16", (vocab, dim))
    _add(spec, "norm.weight", "BF16", (dim,))
    _add(spec, "hc_head_fn", "F32", (hc_mult, hc_mult * dim))
    _add(spec, "hc_head_base", "F32", (hc_mult,))
    _add(spec, "hc_head_scale", "F32", (1,))

    for layer in range(n_layers):
        _add_block(spec, f"layers.{layer}", TARGET_COMPRESS_RATIOS[layer], is_mtp=False)
    for mtp in range(cfg["num_nextn_predict_layers"]):
        _add_block(spec, f"mtp.{mtp}", TARGET_COMPRESS_RATIOS[n_layers + mtp], is_mtp=True)
    return spec


# ---------------------------------------------------------------------------
# Checks.
# ---------------------------------------------------------------------------


def _diff_value(path: str, want: Any, got: Any, problems: list[str]) -> None:
    if isinstance(want, dict):
        if not isinstance(got, dict):
            problems.append(f"config {path}: expected a mapping, got {got!r}")
            return
        for key, sub in want.items():
            if key not in got:
                problems.append(f"config {path}.{key}: missing, expected {sub!r}")
            else:
                _diff_value(f"{path}.{key}", sub, got[key], problems)
        # A pinned nested mapping is exact. Checking only the expected keys
        # would accept an *active* extra field -- `rope_scaling.mscale` would
        # turn on amplitude scaling the source never applies, and an extra
        # `quantization_config` entry would change how weights are read.
        for key in sorted(set(got) - set(want)):
            problems.append(
                f"config {path}.{key}: unexpected key ({got[key]!r}) "
                "-- not part of the pinned target contract"
            )
        return
    if isinstance(want, bool) or isinstance(got, bool):
        equal = want is got
    elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
        equal = float(want) == float(got)
    else:
        equal = want == got
    if not equal:
        problems.append(f"config {path}: expected {want!r}, got {got!r}")


def check_config(config: dict[str, Any], problems: list[str]) -> None:
    """Compare ``config.json`` against the pinned target contract.

    Exact in both directions: every pinned key must be present and equal, and
    every key the checkpoint carries must be either pinned, the separately
    diffed ``compress_ratios``, or explicitly allow-listed as non-semantic.
    """
    for key, want in TARGET_CONFIG.items():
        if key not in config:
            problems.append(f"config {key}: missing, expected {want!r}")
            continue
        _diff_value(key, want, config[key], problems)

    known = set(TARGET_CONFIG) | set(NON_SEMANTIC_CONFIG_KEYS) | {COMPRESS_RATIOS_KEY}
    for key in sorted(set(config) - known):
        problems.append(
            f"config {key}: unexpected key ({config[key]!r}) -- not pinned by the "
            "target contract and not allow-listed as non-semantic"
        )

    ratios = list(config.get(COMPRESS_RATIOS_KEY, []))
    if ratios != TARGET_COMPRESS_RATIOS:
        if len(ratios) != len(TARGET_COMPRESS_RATIOS):
            problems.append(
                f"config compress_ratios: expected {len(TARGET_COMPRESS_RATIOS)} "
                f"entries (num_hidden_layers + num_nextn_predict_layers), got "
                f"{len(ratios)}"
            )
        for i, (want, got) in enumerate(zip(TARGET_COMPRESS_RATIOS, ratios)):
            if want != got:
                problems.append(f"config compress_ratios[{i}]: expected ratio {want}, got {got}")


def check_tensors(tensors: dict[str, TensorInfo], problems: list[str]) -> None:
    """Diff every safetensors header against the expected tensor map."""
    spec = expected_tensors()

    for name, (dtype, shape) in sorted(spec.items()):
        got = tensors.get(name)
        if got is None:
            problems.append(f"{name}: missing, expected {dtype} {shape}")
            continue
        if got.dtype != dtype:
            problems.append(f"{name}: expected dtype {dtype}, got {got.dtype}")
        if got.shape != shape:
            problems.append(f"{name}: expected shape {shape}, got {got.shape}")

    for name in sorted(set(tensors) - set(spec)):
        got = tensors[name]
        problems.append(
            f"{name}: unexpected tensor ({got.dtype} {got.shape}) "
            "-- not part of the target contract"
        )


def verify(inv: Inventory) -> Inventory:
    """Check the checkpoint against the DeepSeek-V4-Flash structural contract.

    Populates ``inv.problems``; call :func:`raise_for_problems` to turn a
    non-empty problem list into an exception. Idempotent -- re-verifying an
    Inventory recomputes the list rather than appending to it.
    """
    inv.problems = list(inv.load_problems)
    check_config(inv.config, inv.problems)
    check_tensors(inv.tensors, inv.problems)
    return inv


def raise_for_problems(inv: Inventory) -> None:
    if inv.problems:
        joined = "\n  - ".join(inv.problems[:40])
        more = "" if len(inv.problems) <= 40 else f"\n  ... and {len(inv.problems) - 40} more"
        raise InventoryError(
            f"{len(inv.problems)} checkpoint contract violation(s):\n  - {joined}{more}"
        )


def summarise(inv: Inventory) -> dict[str, Any]:
    """Group tensors into families and account for reachability."""
    groups: dict[tuple[str, str, tuple[int, ...]], int] = collections.Counter()
    for t in inv.tensors.values():
        groups[(_normalise(t.name), t.dtype, t.shape)] += 1

    families = [
        {
            "pattern": pat,
            "dtype": dtype,
            "shape": list(shape),
            "count": count,
        }
        for (pat, dtype, shape), count in sorted(groups.items())
    ]

    dtype_totals = collections.Counter(t.dtype for t in inv.tensors.values())
    total_bytes = sum(t.nbytes for t in inv.tensors.values())

    # The MTP block exists in the checkpoint but speculative decoding is out
    # of scope for this task, so it is structural-but-unreachable rather than
    # missing. Record it explicitly so loader accounting can subtract it.
    mtp = {n: t for n, t in inv.tensors.items() if n.startswith("mtp.")}
    ratios = TARGET_COMPRESS_RATIOS[: TARGET_CONFIG["num_hidden_layers"]]

    return {
        "checkpoint": inv.checkpoint,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "architecture": TARGET_CONFIG["architectures"][0],
        "num_hidden_layers": TARGET_CONFIG["num_hidden_layers"],
        "num_tensors": len(inv.tensors),
        "num_expected_tensors": len(expected_tensors()),
        "total_tensor_bytes": total_bytes,
        "dtype_totals": dict(dtype_totals.most_common()),
        "compress_schedule": {
            "ratio_0_layers": [i for i, r in enumerate(ratios) if r == 0],
            "ratio_4_layers": [i for i, r in enumerate(ratios) if r == 4],
            "ratio_128_layers": [i for i, r in enumerate(ratios) if r == 128],
        },
        "structural_unreachable": {
            "reason": "MTP / speculative decoding is out of scope for this task",
            "tensor_count": len(mtp),
            "tensor_bytes": sum(t.nbytes for t in mtp.values()),
            "prefix": "mtp.",
        },
        # Top-level keys the contract deliberately does not pin. Recorded with
        # their observed values so "allow-listed" stays auditable rather than
        # meaning "invisible".
        "config_metadata": {
            key: {"value": inv.config.get(key), "reason": reason}
            for key, reason in sorted(NON_SEMANTIC_CONFIG_KEYS.items())
        },
        "quantization_contract": {
            "dense_attention": {
                "weight_dtype": "F8_E4M3",
                "scale_dtype": "F8_E8M0",
                "scale_block": [FP8_BLOCK, FP8_BLOCK],
            },
            "routed_experts": {
                "weight_dtype": "I8 (packed MXFP4, two nibbles per byte)",
                "scale_dtype": "F8_E8M0",
                "scale_group_logical_k": MXFP4_GROUP,
            },
        },
        "families": families,
        "problems": inv.problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--output", default=None, help="Write the summary JSON here.")
    parser.add_argument(
        "--strict", action="store_true", help="Exit non-zero if any contract check failed."
    )
    args = parser.parse_args(argv)

    inv = verify(load_inventory(args.checkpoint))
    summary = summarise(inv)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(f"checkpoint : {summary['checkpoint']}")
    print(f"architecture: {summary['architecture']}")
    print(
        f"tensors    : {summary['num_tensors']} checked against "
        f"{summary['num_expected_tensors']} expected"
    )
    print(f"bytes      : {summary['total_tensor_bytes'] / 1e9:.2f} GB")
    print(f"dtypes     : {summary['dtype_totals']}")
    sched = summary["compress_schedule"]
    print(f"ratio 0    : {sched['ratio_0_layers']}")
    print(f"ratio 4    : {len(sched['ratio_4_layers'])} layers {sched['ratio_4_layers'][:4]}...")
    print(
        f"ratio 128  : {len(sched['ratio_128_layers'])} layers {sched['ratio_128_layers'][:4]}..."
    )
    print(
        f"structural : {summary['structural_unreachable']['tensor_count']} "
        f"MTP tensors "
        f"({summary['structural_unreachable']['tensor_bytes'] / 1e9:.2f} GB)"
    )

    if inv.problems:
        print(f"\nPROBLEMS ({len(inv.problems)}):")
        for p in inv.problems[:40]:
            print(f"  - {p}")
        if len(inv.problems) > 40:
            print(f"  ... and {len(inv.problems) - 40} more")
        if args.strict:
            return 1
    else:
        print("\nAll structural contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
