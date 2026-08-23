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
"""Negative coverage for the DeepSeek-V4-Flash checkpoint inventory gate.

Running the gate against the real checkpoint only proves the checkpoint is
*accepted*. It says nothing about whether the gate would notice if something
were wrong --- a verifier that returns ``problems == []`` unconditionally
passes that test too.

So these tests work the other way round. They synthesise a checkpoint that
satisfies the pinned contract exactly, confirm it is accepted, then mutate one
field at a time and require every single mutation to be rejected. No GPU, no
checkpoint on disk, runs in well under a second.
"""

import copy
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

_INVENTORY_PATH = (
    Path(__file__).resolve().parents[3]
    / "integration"
    / "defs"
    / "accuracy"
    / "deepseek_v4_flash_h100"
    / "checkpoint_inventory.py"
)


def _load_inventory_module():
    name = "deepseek_v4_flash_h100_checkpoint_inventory"
    spec = importlib.util.spec_from_file_location(name, _INVENTORY_PATH)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__],
    # so the module has to be registered before it is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ci = _load_inventory_module()


def _synthetic_config() -> dict:
    """A config.json that satisfies the pinned target contract exactly."""
    config = copy.deepcopy(ci.TARGET_CONFIG)
    config["compress_ratios"] = list(ci.TARGET_COMPRESS_RATIOS)
    return config


def _synthetic_config_with_metadata() -> dict:
    """The conforming config plus the allow-listed non-semantic keys.

    This is the shape of the real ``config.json``: pinned fields, plus
    ``initializer_range`` / ``transformers_version``, which the contract
    tolerates by name.
    """
    config = _synthetic_config()
    config["initializer_range"] = 0.02
    config["transformers_version"] = "4.57.1"
    return config


def _synthetic_tensors() -> dict:
    """Safetensors headers that satisfy the pinned target contract exactly."""
    return {
        name: ci.TensorInfo(name=name, dtype=dtype, shape=shape, file="synthetic.safetensors")
        for name, (dtype, shape) in ci.expected_tensors().items()
    }


def _problems(config: dict, tensors: dict) -> list[str]:
    inv = ci.Inventory(checkpoint="synthetic", config=config, tensors=tensors)
    return ci.verify(inv).problems


def _set_path(config: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        config = config[key]
    config[keys[-1]] = value


def test_synthetic_contract_is_accepted():
    assert _problems(_synthetic_config(), _synthetic_tensors()) == []


def test_every_routed_expert_of_every_layer_is_covered():
    """Guard against regressing to a per-layer expert-0 sample."""
    spec = ci.expected_tensors()
    cfg = ci.TARGET_CONFIG
    blocks = cfg["num_hidden_layers"] + cfg["num_nextn_predict_layers"]
    n_experts = cfg["n_routed_experts"]

    weights = [n for n in spec if ".ffn.experts." in n and n.endswith(".weight")]
    scales = [n for n in spec if ".ffn.experts." in n and n.endswith(".scale")]
    expected = blocks * n_experts * 3
    assert len(weights) == expected
    assert len(scales) == expected

    # ... and the last expert of the last layer really is one of them.
    last = f"layers.{cfg['num_hidden_layers'] - 1}.ffn.experts.{n_experts - 1}"
    assert f"{last}.w3.weight" in spec
    assert f"{last}.w3.scale" in spec


# ---------------------------------------------------------------------------
# config.json mutations
# ---------------------------------------------------------------------------

# Each entry is a target-critical value: change it and the model silently
# computes something other than DeepSeek-V4-Flash.
_CONFIG_MUTATIONS = [
    ("head_dim", 576),
    ("qk_rope_head_dim", 63),
    ("q_lora_rank", 512),
    ("num_attention_heads", 128),
    ("num_key_value_heads", 8),
    ("o_groups", 4),
    ("o_lora_rank", 512),
    ("sliding_window", 127),
    ("index_topk", 511),
    ("index_n_heads", 32),
    ("index_head_dim", 64),
    ("rope_theta", 10001),
    ("compress_rope_theta", 160001),
    ("rope_scaling.factor", 8),
    ("rope_scaling.original_max_position_embeddings", 32768),
    ("rope_scaling.type", "linear"),
    ("rope_scaling.beta_fast", 16),
    ("scoring_func", "softmax"),
    ("topk_method", "greedy"),
    ("norm_topk_prob", False),
    ("swiglu_limit", 7.0),
    ("routed_scaling_factor", 2.5),
    ("n_routed_experts", 128),
    ("num_experts_per_tok", 8),
    ("moe_intermediate_size", 1024),
    ("n_shared_experts", 2),
    ("num_hash_layers", 2),
    ("hc_mult", 2),
    ("hc_sinkhorn_iters", 10),
    ("hc_eps", 1e-05),
    ("rms_norm_eps", 1e-05),
    ("expert_dtype", "fp8"),
    ("num_hidden_layers", 42),
    ("num_nextn_predict_layers", 0),
    ("hidden_size", 8192),
    ("vocab_size", 129279),
    ("hidden_act", "gelu"),
    ("model_type", "deepseek_v3"),
    ("torch_dtype", "float16"),
    ("tie_word_embeddings", True),
    ("attention_dropout", 0.1),
    ("use_cache", False),
    ("quantization_config.weight_block_size", [64, 64]),
    ("quantization_config.scale_fmt", "float"),
    ("quantization_config.fmt", "e5m2"),
    ("quantization_config.quant_method", "mxfp4"),
    ("quantization_config.activation_scheme", "static"),
]


@pytest.mark.parametrize("dotted,value", _CONFIG_MUTATIONS, ids=[m[0] for m in _CONFIG_MUTATIONS])
def test_config_mutation_is_rejected(dotted, value):
    config = _synthetic_config()
    _set_path(config, dotted, value)
    problems = _problems(config, _synthetic_tensors())
    assert any(dotted.split(".")[-1] in p for p in problems), (
        f"mutating {dotted} -> {value!r} was not reported: {problems[:5]}"
    )


def test_missing_config_key_is_rejected():
    config = _synthetic_config()
    del config["scoring_func"]
    assert any("scoring_func" in p for p in _problems(config, _synthetic_tensors()))


# ---------------------------------------------------------------------------
# Unexpected config keys.
#
# A value mutation is the easy half. The half that used to false-pass is an
# *extra* key: the contract checked every key it expected and never looked at
# what else was there, so an active field the target never applies --- most
# importantly `rope_scaling.mscale`, the amplitude scaling the plan pins as
# absent --- sailed through with problems == [].
# ---------------------------------------------------------------------------

_UNEXPECTED_CONFIG_KEYS = [
    # YaRN amplitude scaling. The source's precompute_freqs_cis interpolates
    # frequencies only, so either of these means different attention magnitudes.
    ("rope_scaling.mscale", 2.0),
    ("rope_scaling.mscale_all_dim", 1.0),
    ("rope_scaling.attn_factor", 0.7),
    # An extra quantisation directive changes how weights are read even though
    # every pinned quantisation field still matches.
    ("quantization_config.group_size", 32),
    ("quantization_config.modules_to_not_convert", ["lm_head"]),
    # Unpinned top-level fields that would change semantics if honoured.
    ("kv_lora_rank", 512),
    ("engram_enabled", True),
    ("mtp_enabled", True),
    ("first_k_dense_replace", 3),
]


@pytest.mark.parametrize(
    "dotted,value", _UNEXPECTED_CONFIG_KEYS, ids=[m[0] for m in _UNEXPECTED_CONFIG_KEYS]
)
def test_unexpected_config_key_is_rejected(dotted, value):
    config = _synthetic_config()
    _set_path(config, dotted, value)
    problems = _problems(config, _synthetic_tensors())
    assert any(dotted.split(".")[-1] in p for p in problems), (
        f"adding {dotted} = {value!r} was not reported: {problems[:5]}"
    )


def test_amplitude_scaling_extra_is_rejected_on_the_real_shaped_config():
    """The exact probe that used to return ``[]`` -- now on the real config shape."""
    config = _synthetic_config_with_metadata()
    config["rope_scaling"]["mscale"] = 2.0
    problems = _problems(config, _synthetic_tensors())
    assert [p for p in problems if "rope_scaling.mscale" in p], problems[:5]


def test_allow_listed_metadata_is_tolerated_and_reported():
    """Non-semantic keys pass by name, and stay visible in the summary."""
    config = _synthetic_config_with_metadata()
    assert _problems(config, _synthetic_tensors()) == []

    inv = ci.Inventory(checkpoint="synthetic", config=config, tensors=_synthetic_tensors())
    metadata = ci.summarise(ci.verify(inv))["config_metadata"]
    assert metadata["transformers_version"]["value"] == "4.57.1"
    assert metadata["initializer_range"]["value"] == 0.02
    # Allow-listing is by name only; it must not become a wildcard.
    assert set(metadata) == set(ci.NON_SEMANTIC_CONFIG_KEYS)


def test_allow_list_does_not_cover_nested_mappings():
    """`transformers_version` is tolerated at top level, not inside a pinned map."""
    config = _synthetic_config()
    config["rope_scaling"]["transformers_version"] = "4.57.1"
    assert any(
        "rope_scaling.transformers_version" in p for p in _problems(config, _synthetic_tensors())
    )


def test_non_mapping_where_a_mapping_is_pinned_is_rejected():
    config = _synthetic_config()
    config["rope_scaling"] = "yarn"
    assert any("rope_scaling" in p for p in _problems(config, _synthetic_tensors()))


@pytest.mark.parametrize("index,value", [(0, 4), (2, 128), (3, 4), (42, 128), (43, 4)])
def test_compress_ratio_mutation_is_rejected(index, value):
    config = _synthetic_config()
    config["compress_ratios"][index] = value
    assert any(f"compress_ratios[{index}]" in p for p in _problems(config, _synthetic_tensors()))


def test_truncated_compress_ratios_is_rejected():
    config = _synthetic_config()
    config["compress_ratios"] = config["compress_ratios"][:-1]
    assert any("compress_ratios" in p for p in _problems(config, _synthetic_tensors()))


# ---------------------------------------------------------------------------
# load_inventory: index vs shard headers.
#
# The tensor map is read from the shard headers, because that is what a loader
# actually opens. `model.safetensors.index.json` is then cross-checked against
# it --- an index-driven walk on its own would silently skip a tensor that
# exists on disk but is not indexed.
#
# These are the only tests that exercise load_inventory itself, so they build a
# real (tiny) checkpoint on disk rather than a synthetic in-memory one.
# ---------------------------------------------------------------------------

_MINI = {
    "shard-a.safetensors": {"embed.weight": ("BF16", (4, 2)), "norm.weight": ("BF16", (2,))},
    "shard-b.safetensors": {"head.weight": ("BF16", (4, 2))},
}


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Write a valid, zero-filled safetensors file with the given headers."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * ci._DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        blobs.append(b"\0" * nbytes)
        offset += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for chunk in blobs:
            f.write(chunk)


def _write_mini_checkpoint(root: Path, shards: dict, index: dict | None = None) -> Path:
    """Materialise a tiny checkpoint; ``index`` defaults to matching the shards."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_synthetic_config()))
    for shard, tensors in shards.items():
        _write_safetensors(root / shard, tensors)
    if index is None:
        index = {name: shard for shard, tensors in shards.items() for name in tensors}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return root


def test_load_inventory_reads_shard_headers(tmp_path):
    root = _write_mini_checkpoint(tmp_path / "clean", copy.deepcopy(_MINI))
    inv = ci.load_inventory(str(root))
    assert inv.load_problems == []
    assert set(inv.tensors) == {"embed.weight", "norm.weight", "head.weight"}
    assert inv.tensors["embed.weight"].dtype == "BF16"
    assert inv.tensors["embed.weight"].shape == (4, 2)
    assert inv.tensors["head.weight"].file == "shard-b.safetensors"


def test_tensor_on_disk_but_absent_from_the_index_is_reported(tmp_path):
    """The hole an index-only walk leaves: a shard-walking loader still reads it."""
    shards = copy.deepcopy(_MINI)
    index = {name: shard for shard, tensors in shards.items() for name in tensors}
    shards["shard-b.safetensors"]["stowaway.weight"] = ("BF16", (2, 2))
    root = _write_mini_checkpoint(tmp_path / "stowaway", shards, index=index)

    inv = ci.load_inventory(str(root))
    assert "stowaway.weight" in inv.tensors
    assert any("stowaway.weight" in p and "absent from the index" in p for p in inv.load_problems)


def test_indexed_tensor_missing_from_every_shard_is_reported(tmp_path):
    shards = copy.deepcopy(_MINI)
    index = {name: shard for shard, tensors in shards.items() for name in tensors}
    index["ghost.weight"] = "shard-a.safetensors"
    root = _write_mini_checkpoint(tmp_path / "ghost", shards, index=index)

    problems = ci.load_inventory(str(root)).load_problems
    assert any("ghost.weight" in p and "absent" in p for p in problems)


def test_index_pointing_at_the_wrong_shard_is_reported(tmp_path):
    shards = copy.deepcopy(_MINI)
    index = {name: shard for shard, tensors in shards.items() for name in tensors}
    index["head.weight"] = "shard-a.safetensors"
    root = _write_mini_checkpoint(tmp_path / "misfiled", shards, index=index)

    problems = ci.load_inventory(str(root)).load_problems
    assert any("head.weight" in p and "shard-a" in p for p in problems)


def test_shard_listed_in_the_index_but_missing_on_disk_is_reported(tmp_path):
    shards = copy.deepcopy(_MINI)
    index = {name: shard for shard, tensors in shards.items() for name in tensors}
    index["extra.weight"] = "shard-c.safetensors"
    root = _write_mini_checkpoint(tmp_path / "missing_shard", shards, index=index)

    problems = ci.load_inventory(str(root)).load_problems
    assert any("shard-c.safetensors" in p and "not present on disk" in p for p in problems)


def test_tensor_duplicated_across_shards_is_reported(tmp_path):
    shards = copy.deepcopy(_MINI)
    shards["shard-b.safetensors"]["norm.weight"] = ("BF16", (2,))
    root = _write_mini_checkpoint(tmp_path / "duplicate", shards)

    problems = ci.load_inventory(str(root)).load_problems
    assert any("norm.weight" in p and "duplicated across shards" in p for p in problems)


def test_verify_surfaces_load_problems(tmp_path):
    """A disk-level problem must reach ``problems``, not stop at ``load_problems``."""
    shards = copy.deepcopy(_MINI)
    index = {name: shard for shard, tensors in shards.items() for name in tensors}
    shards["shard-b.safetensors"]["stowaway.weight"] = ("BF16", (2, 2))
    root = _write_mini_checkpoint(tmp_path / "surfaced", shards, index=index)

    inv = ci.verify(ci.load_inventory(str(root)))
    assert any("stowaway.weight" in p and "absent from the index" in p for p in inv.problems)
    # ... and verify stays idempotent rather than accumulating duplicates.
    assert ci.verify(inv).problems == inv.problems


# ---------------------------------------------------------------------------
# safetensors header mutations
# ---------------------------------------------------------------------------


def _mutate_dtype(name, dtype):
    def apply(tensors):
        old = tensors[name]
        tensors[name] = ci.TensorInfo(name=old.name, dtype=dtype, shape=old.shape, file=old.file)

    return apply


def _mutate_shape(name, shape):
    def apply(tensors):
        old = tensors[name]
        tensors[name] = ci.TensorInfo(name=old.name, dtype=old.dtype, shape=shape, file=old.file)

    return apply


def _drop(name):
    def apply(tensors):
        del tensors[name]

    return apply


def _add_extra(name, dtype, shape):
    def apply(tensors):
        tensors[name] = ci.TensorInfo(name=name, dtype=dtype, shape=shape, file="extra.safetensors")

    return apply


# The first entry is the exact probe that used to false-pass: a single routed
# expert (not expert 0) silently dequantised out of its MXFP4 container.
_TENSOR_MUTATIONS = [
    ("expert_1_w2_dequantised", _mutate_dtype("layers.0.ffn.experts.1.w2.weight", "BF16")),
    ("last_expert_of_last_layer", _mutate_dtype("layers.42.ffn.experts.255.w3.weight", "BF16")),
    ("mid_expert_scale_dtype", _mutate_dtype("layers.21.ffn.experts.130.w1.scale", "BF16")),
    ("expert_scale_group_size", _mutate_shape("layers.7.ffn.experts.9.w1.scale", (2048, 64))),
    ("expert_nibble_packing", _mutate_shape("layers.7.ffn.experts.9.w1.weight", (2048, 4096))),
    ("expert_dropped", _drop("layers.13.ffn.experts.200.w2.weight")),
    ("shared_expert_dtype", _mutate_dtype("layers.5.ffn.shared_experts.w1.weight", "BF16")),
    ("attention_weight_dtype", _mutate_dtype("layers.9.attn.wq_b.weight", "BF16")),
    ("fp8_block_scale_shape", _mutate_shape("layers.9.attn.wq_b.scale", (256, 16))),
    ("attention_geometry", _mutate_shape("layers.0.attn.wkv.weight", (576, 4096))),
    ("attn_sink_dropped", _drop("layers.3.attn.attn_sink")),
    ("mhc_precision", _mutate_dtype("layers.4.hc_attn_fn", "BF16")),
    ("compressor_width", _mutate_shape("layers.2.attn.compressor.ape", (4, 512))),
    ("compressor_on_swa_layer", _add_extra("layers.0.attn.compressor.ape", "F32", (4, 1024))),
    (
        "indexer_on_ratio_128_layer",
        _add_extra("layers.3.attn.indexer.wq_b.weight", "F8_E4M3", (8192, 1024)),
    ),
    ("indexer_dropped_on_ratio_4", _drop("layers.2.attn.indexer.wq_b.weight")),
    ("hash_table_dropped", _drop("layers.1.ffn.gate.tid2eid")),
    ("hash_table_dtype", _mutate_dtype("layers.0.ffn.gate.tid2eid", "I32")),
    ("bias_on_hash_layer", _add_extra("layers.0.ffn.gate.bias", "F32", (256,))),
    ("bias_dropped_on_score_layer", _drop("layers.3.ffn.gate.bias")),
    ("unknown_tensor_family", _add_extra("layers.0.attn.rotary_emb.inv_freq", "F32", (32,))),
    ("lm_head_dropped", _drop("head.weight")),
]


@pytest.mark.parametrize(
    "mutate", [m[1] for m in _TENSOR_MUTATIONS], ids=[m[0] for m in _TENSOR_MUTATIONS]
)
def test_tensor_mutation_is_rejected(mutate):
    tensors = _synthetic_tensors()
    mutate(tensors)
    assert _problems(_synthetic_config(), tensors) != []


def test_raise_for_problems_is_the_programmatic_gate():
    """The importable entry point must fail loudly, not just collect strings."""
    clean = ci.Inventory(
        checkpoint="synthetic", config=_synthetic_config(), tensors=_synthetic_tensors()
    )
    ci.raise_for_problems(ci.verify(clean))  # no-op on a conforming checkpoint

    tensors = _synthetic_tensors()
    _mutate_dtype("layers.0.ffn.experts.1.w2.weight", "BF16")(tensors)
    dirty = ci.Inventory(checkpoint="synthetic", config=_synthetic_config(), tensors=tensors)
    with pytest.raises(ci.InventoryError, match="experts.1.w2.weight"):
        ci.raise_for_problems(ci.verify(dirty))
