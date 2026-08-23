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
"""CPU coverage for the Goal 3.1 load-accounting rules.

The eight-rank ``load_and_moe`` suite reports "every checkpoint tensor is
consumed once or named structural", "no two tensors share a model slot" and "the
routed experts are still packed". Each of those is a rule, and a rule that only
ever sees a passing checkpoint is indistinguishable from a rule that always
passes. These drive them with hand-built inputs so each failure mode is pinned,
on CPU, with no GPU and no checkpoint.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SUPPORT = (
    Path(__file__).resolve().parents[3]
    / "integration"
    / "defs"
    / "accuracy"
    / "deepseek_v4_flash_h100"
)


def _load():
    name = "deepseek_v4_flash_h100_load_accounting"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SUPPORT / "load_accounting.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


la = _load()


# ---------------------------------------------------------------------------
# Read counting.
# ---------------------------------------------------------------------------


class _Slice:
    def __init__(self, tensor):
        self._tensor = tensor

    def get_shape(self):
        return list(self._tensor.shape)

    def get_dtype(self):
        return "I8"

    def __getitem__(self, item):
        return self._tensor[item]


def test_counting_weights_wraps_lazy_values_and_forwards_everything_else():
    inner = {"a": _Slice(torch.zeros(2, 2, dtype=torch.int8)), "b": torch.ones(3)}
    counting = la.CountingWeights(inner)

    assert counting.lazy_keys == 1
    assert len(counting) == 2
    assert "a" in counting and "b" in counting
    assert counting.read_counts() == {"a": 0}

    counting["a"][:]
    counting["a"][:]
    assert counting.read_counts() == {"a": 2}
    # An eager tensor is left alone; reading it is not a checkpoint read.
    assert torch.equal(counting["b"], torch.ones(3))


# ---------------------------------------------------------------------------
# Consumption.
# ---------------------------------------------------------------------------


def test_an_unread_tensor_that_is_nobody_elses_expert_is_a_hole():
    report = la.raw_consumption_report(
        {"layers.0.attn.wkv.weight": 0, "layers.0.attn.wkv.scale": 1},
        local_expert_ids={0},
    )
    assert not report["passed"]
    assert report["unexplained"] == ["layers.0.attn.wkv.weight"]


def test_a_foreign_experts_bytes_are_allowed_to_stay_on_disk():
    report = la.raw_consumption_report(
        {
            "layers.0.ffn.experts.0.w1.weight": 1,
            "layers.0.ffn.experts.5.w1.weight": 0,
        },
        local_expert_ids={0},
    )
    assert report["passed"]
    assert report["never_read"] == {"foreign_expert_parallel_shard": 1}
    assert report["read_exactly_once"] == 1


def test_reading_the_same_tensor_twice_fails():
    """Two reads is either double work on a 149 GB checkpoint or two
    destinations for one tensor. Neither is "consumed once"."""
    report = la.raw_consumption_report({"layers.0.attn.wkv.weight": 2}, local_expert_ids=None)
    assert not report["passed"]
    assert report["read_more_than_once"] == 1


def test_without_expert_parallelism_an_unread_expert_is_still_a_hole():
    report = la.raw_consumption_report(
        {"layers.0.ffn.experts.5.w1.weight": 0}, local_expert_ids=None
    )
    assert not report["passed"]
    assert report["unexplained_count"] == 1


# ---------------------------------------------------------------------------
# Destinations.
# ---------------------------------------------------------------------------


class _Model(torch.nn.Module):
    def __init__(self, names):
        super().__init__()
        for name in names:
            parts = name.split(".")
            parent = self
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    setattr(parent, part, torch.nn.Module())
                parent = getattr(parent, part)
            setattr(parent, parts[-1], torch.nn.Parameter(torch.zeros(1)))


def test_a_remapped_key_with_no_destination_is_a_hole():
    model = _Model(["model.layers.0.self_attn.q_a_proj.weight"])
    report = la.destination_report(
        ["model.layers.0.self_attn.q_a_proj.weight", "model.layers.0.self_attn.mystery.weight"],
        model,
        num_hidden_layers=1,
    )
    assert not report["passed"]
    assert report["unexplained"] == ["model.layers.0.self_attn.mystery.weight"]


def test_the_mtp_block_and_the_flat_mhc_keys_are_named_structural():
    """Both are real: the MTP layer is in the checkpoint but not instantiated
    with speculative decoding off, and `load_flat_hc_weights` writes
    `hc_attn.fn` from a key spelled `hc_attn_fn`."""
    model = _Model(["model.layers.0.self_attn.q_a_proj.weight"])
    report = la.destination_report(
        [
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.1.self_attn.q_a_proj.weight",  # MTP block, layers == 1
            "model.layers.0.hc_attn_fn",
            "model.hc_head_scale",
        ],
        model,
        num_hidden_layers=1,
    )
    assert report["passed"]
    assert report["structural"] == {
        "mtp_block_not_instantiated": 1,
        "flat_mhc_weights_via_load_flat_hc_weights": 2,
    }
    assert report["matched_parameters"] == 1


# ---------------------------------------------------------------------------
# Collisions.
# ---------------------------------------------------------------------------


def test_the_documented_compressor_fusion_is_not_counted_as_a_collision():
    surviving = [
        "layers.2.attn.compressor.wkv.weight",
        "layers.2.attn.compressor.wgate.weight",
        "layers.2.attn.kv_norm.weight",
    ]
    remapped = [
        "model.layers.2.self_attn.compressor.wkv_gate.weight",
        "model.layers.2.self_attn.kv_a_layernorm.weight",
    ]
    report = la.conservation_report(surviving, remapped)
    assert report["passed"]
    assert report["documented_fusions"] == 1
    assert report["collisions"] == 0


def test_two_tensors_landing_in_one_slot_is_reported():
    """The remap returns a dict, so a collision silently drops one tensor; the
    reconciliation is what makes it visible."""
    surviving = ["layers.0.attn.wq_a.weight", "layers.0.attn.wq_b.weight"]
    report = la.conservation_report(surviving, ["model.layers.0.self_attn.q_a_proj.weight"])
    assert not report["passed"]
    assert report["collisions"] == 1


def test_the_documented_mtp_head_drop_is_not_counted_as_a_collision():
    report = la.conservation_report(["mtp.0.head.weight"], [])
    assert report["passed"]
    assert report["documented_drops"] == 1


# ---------------------------------------------------------------------------
# Residency.
# ---------------------------------------------------------------------------


class _Config:
    hidden_size = 4096
    moe_intermediate_size = 2048


def test_the_packed_expert_size_is_derived_from_the_config():
    """Checkpoint dimensions: 32 experts/rank x 43 layers is 17544 MiB, which is
    what a packed load must weigh."""
    total = la.expected_routed_bytes(_Config(), num_layers=43, experts_per_rank=32)
    assert total == 17544 * 2**20


def _params_model(named: dict[str, torch.Tensor]) -> torch.nn.Module:
    """A module whose ``named_parameters()`` are exactly ``named``.

    Names matter here: ``residency_report`` finds routed experts by the
    ``.experts.`` path segment the real model has
    (``model.layers.N.mlp.experts.backend.w2_weight``), so a flat name would
    exercise a matcher the production model never presents.
    """
    root = torch.nn.Module()
    for name, tensor in named.items():
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, torch.nn.Module())
            parent = getattr(parent, part)
        setattr(parent, parts[-1], torch.nn.Parameter(tensor, requires_grad=False))
    return root


_EXPERTS = "model.layers.0.mlp.experts.backend"


def _packed_expert_params(experts: int = 2) -> dict[str, torch.Tensor]:
    hidden, inter = _Config.hidden_size, _Config.moe_intermediate_size
    return {
        f"{_EXPERTS}.w3_w1_weight": torch.zeros(experts, 2 * inter, hidden // 2, dtype=torch.uint8),
        f"{_EXPERTS}.w2_weight": torch.zeros(experts, hidden, inter // 2, dtype=torch.uint8),
        f"{_EXPERTS}.fc31_weight_scale": torch.zeros(
            experts, 2 * inter, hidden // 32, dtype=torch.uint8
        ),
        f"{_EXPERTS}.fc2_weight_scale": torch.zeros(
            experts, hidden, inter // 32, dtype=torch.uint8
        ),
        "model.layers.0.self_attn.q_a_proj.weight": torch.zeros(4, dtype=torch.float8_e4m3fn),
        "model.layers.0.input_layernorm.weight": torch.zeros(4, dtype=torch.bfloat16),
    }


def test_a_bf16_expansion_of_the_routed_experts_fails_residency():
    hidden, inter = _Config.hidden_size, _Config.moe_intermediate_size
    params = _packed_expert_params()
    # Dequantized: full width, bfloat16.
    params[f"{_EXPERTS}.w2_weight"] = torch.zeros(2, hidden, inter, dtype=torch.bfloat16)

    report = la.residency_report(_params_model(params), _Config(), num_layers=1)

    assert not report["passed"]
    assert any("not the packed container" in p for p in report["problems"])


def test_a_packed_load_of_the_right_size_passes_residency():
    report = la.residency_report(_params_model(_packed_expert_params()), _Config(), num_layers=1)

    assert report["passed"], report["problems"]
    assert report["experts_per_rank"] == 2
    assert report["routed_expert_bytes"] == report["routed_expert_bytes_expected_packed"]


def test_a_short_count_of_packed_experts_fails_residency():
    """Same container, wrong amount: a layer whose experts never arrived weighs
    less than the packed layout, and only the derived size notices."""
    params = _packed_expert_params()
    del params[f"{_EXPERTS}.fc2_weight_scale"]

    report = la.residency_report(_params_model(params), _Config(), num_layers=1)

    assert not report["passed"]
    assert any("packed layout is" in p for p in report["problems"])


def test_losing_the_fp8_dense_contract_fails_residency():
    """An all-BF16 model loads fine and is wrong; the dtype census is what
    notices."""
    params = _packed_expert_params()
    params["model.layers.0.self_attn.q_a_proj.weight"] = torch.zeros(4, dtype=torch.bfloat16)

    report = la.residency_report(_params_model(params), _Config(), num_layers=1)

    assert not report["passed"]
    assert any("FP8-E4M3" in p for p in report["problems"])


@pytest.mark.parametrize("name,expected", [("a.experts.7.w1.weight", 7), ("a.attn.wkv", None)])
def test_expert_ids_are_read_out_of_the_key(name, expected):
    assert la._expert_id(name) == expected


def test_a_destination_the_loader_renames_is_matched_not_flagged():
    """`self_attn.attn_sink` is written to `self_attn.mqa.attn_sink` and
    `o_a_proj.weight_scale_inv` to `o_a_proj_scale`. Both are real destinations;
    matching them by name alone would report two holes per layer."""
    model = _Model(
        [
            "model.layers.0.self_attn.mqa.attn_sink",
            "model.layers.0.self_attn.o_a_proj_scale",
        ]
    )
    report = la.destination_report(
        [
            "model.layers.0.self_attn.attn_sink",
            "model.layers.0.self_attn.o_a_proj.weight_scale_inv",
        ],
        model,
        num_hidden_layers=1,
    )
    assert report["passed"], report["unexplained"]
    assert report["matched_parameters"] == 2


def test_a_renamed_destination_that_does_not_exist_is_still_a_hole():
    """The rename is a lookup, not a licence: if the parameter it names is
    absent the tensor still went nowhere."""
    report = la.destination_report(
        ["model.layers.0.self_attn.attn_sink"], _Model([]), num_hidden_layers=1
    )
    assert not report["passed"]
    assert report["unexplained"] == ["model.layers.0.self_attn.attn_sink"]


def test_routed_expert_keys_are_named_as_packed_not_flagged_as_holes():
    """The fused-MoE method packs 32 experts x w1/w3 into one `w3_w1_weight`, so
    no per-expert parameter exists to match. Their arrival is proven by
    `residency_report`'s derived byte total, not by name."""
    model = _Model(["model.layers.0.mlp.experts.backend.w3_w1_weight"])
    report = la.destination_report(
        [
            "model.layers.0.mlp.experts.0.w1.weight",
            "model.layers.0.mlp.experts.0.w1.weight_scale_inv",
            "model.layers.0.mlp.experts.31.w2.weight",
        ],
        model,
        num_hidden_layers=1,
    )
    assert report["passed"]
    assert report["structural"] == {"packed_into_fused_moe_weights": 3}


def test_a_destination_outside_the_module_tree_is_still_resolved():
    """`MLA` holds the attention backend as a plain attribute, so nothing it
    owns is in `named_parameters()`; the sink would otherwise read as a hole on
    every layer."""

    class _Backend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_sink = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

    class _Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, "mqa", _Backend())

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].self_attn = _Attn()

    root = _Root()
    assert not any(n.endswith("attn_sink") for n, _ in root.named_parameters())
    assert la._resolve_attribute(root, "model.layers.0.self_attn.mqa.attn_sink") is not None

    report = la.destination_report(
        ["model.layers.0.self_attn.attn_sink"], root, num_hidden_layers=1
    )
    assert report["passed"], report["unexplained"]


def test_a_fused_projection_is_named_not_flagged():
    """The checkpoint's `wq_a` and `wkv` are one `kv_a_proj_with_mqa` in the
    model, and the shared expert's `gate_proj`/`up_proj` are one `gate_up_proj`.
    Neither narrow name exists as a parameter."""
    model = _Model(
        [
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.mlp.shared_experts.gate_up_proj.weight",
        ]
    )
    report = la.destination_report(
        [
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
            "model.layers.0.mlp.shared_experts.up_proj.weight",
        ],
        model,
        num_hidden_layers=1,
    )
    assert report["passed"], report["unexplained"]
    assert report["structural"] == {"fused_into_a_wider_projection": 3}


def test_a_fusion_target_that_does_not_exist_is_still_a_hole():
    report = la.destination_report(
        ["model.layers.0.self_attn.q_a_proj.weight"], _Model([]), num_hidden_layers=1
    )
    assert not report["passed"]


def test_the_block_scale_rename_is_matched():
    """The remap emits the DeepSeek-V3 `weight_scale_inv` spelling the shared
    `Linear` loader expects; that loader writes `weight_scale`."""
    model = _Model(["model.layers.0.self_attn.q_b_proj.weight_scale"])
    report = la.destination_report(
        ["model.layers.0.self_attn.q_b_proj.weight_scale_inv"], model, num_hidden_layers=1
    )
    assert report["passed"]
    assert report["structural"] == {"fp8_block_scale_written_to_weight_scale": 1}


def test_an_aliased_parameter_is_not_reported_as_a_hole():
    """`post_load_weights` aliases layer N+1's `input_layernorm` as layer N's
    `next_layer_layernorm`, and `named_parameters()` de-duplicates by default,
    so the second name disappears. Every layer but the first would read as a
    hole."""
    norm = torch.nn.Module()
    norm.weight = torch.nn.Parameter(torch.zeros(2))

    class _Aliased(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
            self.model.layers[0].next_layer_layernorm = norm
            self.model.layers[1].input_layernorm = norm

    aliased = _Aliased()
    deduplicated = {n for n, _ in aliased.named_parameters()}
    assert "model.layers.1.input_layernorm.weight" not in deduplicated

    report = la.destination_report(
        ["model.layers.1.input_layernorm.weight"], aliased, num_hidden_layers=2
    )
    assert report["passed"], report["unexplained"]
