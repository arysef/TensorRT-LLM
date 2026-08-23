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
"""Focused coverage for the Stage 3 MoE replay and its rank-verdict plumbing.

The eight-rank ``load_and_moe`` suite is the evidence; these tests pin the
parts of it whose failure mode is a *silently wrong measurement* rather than a
crash --- a golden that clamps where the source does not, a routing comparison
that hides a slot-order difference, an expert-parallel gate that accepts a
partition with a hole, an aggregation that lets one half's verdict overwrite
the other's, and an auditor that skips exact rules.
"""

import inspect
import os
import sys
import types
import unittest.mock

import pytest
import torch
import torch.nn.functional as F

EVIDENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "integration",
    "defs",
    "accuracy",
)
SUPPORT_PKG = os.path.join(EVIDENCE_DIR, "deepseek_v4_flash_h100")
for path in (EVIDENCE_DIR, SUPPORT_PKG):
    if path not in sys.path:
        sys.path.insert(0, path)

import deepseek_v4_flash_h100_evidence as ev  # noqa: E402
import moe_replay as mr  # noqa: E402
import torch_goldens as tg  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# ---------------------------------------------------------------------------
# Config field spellings.
# ---------------------------------------------------------------------------


def test_moe_config_reads_the_real_v4_config():
    """Every field the replay reads must exist on ``DeepseekV4Config``.

    ``PretrainedConfig.__getattribute__`` raises on an unknown name rather than
    returning a default, so one stale spelling is an eight-rank run that dies
    *after* a 20 GiB checkpoint load. This is that failure for the price of a
    CPU test.
    """
    from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config

    shape = mr.moe_config(DeepseekV4Config())
    assert shape["n_routed_experts"] == 256
    assert shape["n_hash_layers"] == 3
    assert shape["routed_scaling_factor"] == 1.5
    assert shape["swiglu_limit"] == 10.0
    assert shape["num_experts_per_tok"] == 6


def test_moe_config_names_every_missing_field():
    class Stub:
        n_routed_experts = 256

    with pytest.raises(AttributeError, match="n_hash_layers"):
        mr.moe_config(Stub())


# ---------------------------------------------------------------------------
# The expert golden.
# ---------------------------------------------------------------------------


def _weights(dim: int, inter: int, device: str) -> dict:
    torch.manual_seed(7)
    return {
        "w1": torch.randn(inter, dim, device=device) * 0.05,
        "w3": torch.randn(inter, dim, device=device) * 0.05,
        "w2": torch.randn(dim, inter, device=device) * 0.05,
    }


@CUDA
def test_golden_reports_fc1_before_the_clamp():
    """``w1``/``w3`` outputs are pre-clamp on the source side.

    Returning clamped values under the ``fc1_*`` keys would compare a clamped
    golden against an unclamped reference, which passes an implementation that
    never clamps and fails one that does.
    """
    device = "cuda"
    w = _weights(256, 512, device)
    x = (torch.randn(8, 256, device=device) * 12.0).to(torch.bfloat16)
    out = mr._golden_expert(x, w, tg, swiglu_limit=10.0, routing_weight=None, quantize_act=False)
    assert float(out["fc1_gate"].max()) > 10.0, "test input never reaches the clamp"
    assert float(out["fc1_up"].abs().max()) > 10.0
    unclamped = mr._golden_expert(
        x, w, tg, swiglu_limit=0.0, routing_weight=None, quantize_act=False
    )
    assert torch.equal(out["fc1_gate"], unclamped["fc1_gate"])
    assert torch.equal(out["fc1_up"], unclamped["fc1_up"])
    # The clamp still has to reach the value that feeds FC2.
    assert not torch.equal(out["swiglu"], unclamped["swiglu"])
    assert float(out["swiglu"].float().abs().max()) < float(unclamped["swiglu"].float().abs().max())


@CUDA
def test_golden_matches_the_source_expert_expression():
    """The golden is the source's ``Expert.forward``, term for term."""
    device = "cuda"
    w = _weights(256, 512, device)
    x = (torch.randn(4, 256, device=device) * 2.0).to(torch.bfloat16)
    routing = torch.rand(4, 1, device=device)
    got = mr._golden_expert(x, w, tg, swiglu_limit=10.0, routing_weight=routing, quantize_act=False)

    gate = F.linear(x, w["w1"].to(torch.bfloat16)).float().clamp(max=10.0)
    up = F.linear(x, w["w3"].to(torch.bfloat16)).float().clamp(min=-10.0, max=10.0)
    expect = F.linear(
        (routing * (F.silu(gate) * up)).to(torch.bfloat16), w["w2"].to(torch.bfloat16)
    )
    assert torch.equal(got["fc2"], expect)


@CUDA
def test_activation_quantisation_is_the_only_difference_between_the_two_modes():
    """``quantize_act`` reproduces the source's W4A8 activation round trip."""
    device = "cuda"
    w = _weights(256, 512, device)
    x = (torch.randn(4, 256, device=device)).to(torch.bfloat16)
    quantised = mr._golden_expert(
        x, w, tg, swiglu_limit=10.0, routing_weight=None, quantize_act=True
    )
    exact = mr._golden_expert(x, w, tg, swiglu_limit=10.0, routing_weight=None, quantize_act=False)
    assert not torch.equal(quantised["fc1_gate"], exact["fc1_gate"])
    replay = F.linear(tg.fp8_quant_dequant(x, 128).to(torch.bfloat16), w["w1"].to(torch.bfloat16))
    assert torch.equal(quantised["fc1_gate"], replay.float())


@CUDA
def test_mxfp4_dequant_round_trips_through_the_packed_container():
    """The golden decodes the byte container the loader keeps packed."""
    torch.manual_seed(3)
    packed = torch.randint(0, 256, (16, 64), dtype=torch.uint8, device="cuda")
    scale = torch.full((16, 4), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    out = tg.dequant_mxfp4(packed, scale, group=mr.MXFP4_GROUP)
    assert out.shape == (16, 128)
    assert out.numel() == packed.numel() * mr.MXFP4_PER_BYTE
    # A UE8M0 exponent of 127 is 2**0, so the decoded values are the raw E2M1
    # levels; every one must be representable.
    assert torch.isfinite(out).all()
    assert set(out.abs().unique().tolist()) <= set(tg.FP4_LEVELS)


# ---------------------------------------------------------------------------
# Routing comparison.
# ---------------------------------------------------------------------------


def test_routing_weights_are_compared_in_expert_space():
    """Slot order must not change the comparison; a value must."""
    indices = torch.tensor([[3, 1, 7]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.3, 0.2]])
    shuffled_idx = torch.tensor([[7, 3, 1]], dtype=torch.int32)
    shuffled_w = torch.tensor([[0.2, 0.5, 0.3]])
    assert torch.equal(
        mr._align_by_expert(indices, weights, 8), mr._align_by_expert(shuffled_idx, shuffled_w, 8)
    )
    moved = mr._align_by_expert(torch.tensor([[3, 1, 6]], dtype=torch.int32), weights, 8)
    assert not torch.equal(mr._align_by_expert(indices, weights, 8), moved)


def test_align_by_expert_leaves_unselected_experts_at_zero():
    dense = mr._align_by_expert(
        torch.tensor([[2, 5]], dtype=torch.int32), torch.tensor([[1.5, 2.5]]), 8
    )
    assert dense.shape == (1, 8)
    assert float(dense[0, 2]) == 1.5 and float(dense[0, 5]) == 2.5
    assert float(dense.sum()) == 4.0


# ---------------------------------------------------------------------------
# Expert-parallel coverage gate.
# ---------------------------------------------------------------------------


def _by_rank(shards):
    return {r: {"local_expert_ids": ids} for r, ids in enumerate(shards)}


def test_exact_partition_of_the_expert_range_passes():
    shards = [list(range(32 * r, 32 * (r + 1))) for r in range(8)]
    assert ev._expert_shard_problems(_by_rank(shards)) == []
    coverage = ev._expert_shard_coverage(_by_rank(shards))
    assert coverage["distinct_experts"] == 256
    assert coverage["disjoint"] and coverage["contiguous_from_zero"]


def test_distinct_shards_that_overlap_are_rejected():
    """The old gate accepted this: eight different tuples, one expert twice."""
    shards = [list(range(32 * r, 32 * (r + 1))) for r in range(8)]
    shards[7] = shards[7][:-1] + [0]
    assert len({tuple(s) for s in shards}) == len(shards), "shards are still distinct"
    problems = ev._expert_shard_problems(_by_rank(shards))
    assert problems and "overlap" in problems[0]


def test_a_hole_in_the_expert_range_is_rejected():
    shards = [list(range(32 * r, 32 * (r + 1))) for r in range(8)]
    shards[3] = [e + 1000 for e in shards[3]]
    problems = ev._expert_shard_problems(_by_rank(shards))
    assert any("0..255 exactly" in p for p in problems)


def test_unequal_shard_sizes_are_rejected():
    shards = [list(range(32 * r, 32 * (r + 1))) for r in range(8)]
    shards[0] = shards[0] + shards[1][:1]
    shards[1] = shards[1][1:]
    problems = ev._expert_shard_problems(_by_rank(shards))
    assert any("different expert counts" in p for p in problems)


# ---------------------------------------------------------------------------
# Merged rank verdicts.
# ---------------------------------------------------------------------------


def _agg(passed, ranks_failed, failures):
    return {
        "passed": passed,
        "ranks_passed": [r for r in range(8) if r not in ranks_failed],
        "ranks_failed": list(ranks_failed),
        "per_rank_failures": failures,
        "worst_rank_metrics": {},
    }


def test_a_failing_load_rank_survives_a_passing_moe_replay():
    """The bug this guards: ``dict.update`` erasing the first half's failures."""
    load = _agg(False, [5], {5: {"routed_expert_layout": {"problems": ["nibbles"]}}})
    moe = _agg(True, [], {})
    merged = ev._merge_rank_verdicts(load, moe)
    assert merged["passed"] is False
    assert merged["ranks_failed"] == [5]
    assert "load.routed_expert_layout" in merged["per_rank_failures"]["5"]


def test_a_failing_moe_rank_survives_a_passing_load():
    load = _agg(True, [], {})
    moe = _agg(False, [2], {"2": {"layer3.routed_output": {"problems": ["cosine"]}}})
    merged = ev._merge_rank_verdicts(load, moe)
    assert merged["passed"] is False
    assert merged["ranks_failed"] == [2]
    assert "moe.layer3.routed_output" in merged["per_rank_failures"]["2"]


def test_both_halves_must_pass_and_ranks_passed_is_the_intersection():
    load = _agg(True, [], {})
    moe = _agg(True, [], {})
    merged = ev._merge_rank_verdicts(load, moe)
    assert merged["passed"] is True
    assert merged["ranks_passed"] == list(range(8))
    assert merged["load_accounting_passed"] and merged["moe_replay_passed"]


def test_failures_on_different_ranks_are_unioned():
    load = _agg(False, [1], {1: {"residency": {"problems": ["bf16"]}}})
    moe = _agg(False, [6], {"6": {"layer0.expert_ids": {"problems": ["set"]}}})
    merged = ev._merge_rank_verdicts(load, moe)
    assert merged["ranks_failed"] == [1, 6]
    assert merged["ranks_passed"] == [0, 2, 3, 4, 5, 7]


# ---------------------------------------------------------------------------
# Auditing exact rules.
# ---------------------------------------------------------------------------


TOLERANCES = {
    "modules": {
        "moe_expert_ids": {"rule": "exact"},
        "moe_expert_output": {"cosine_min": 0.999, "rel_max_abs_max": 0.04},
    }
}


def test_a_failed_exact_rule_cannot_audit_clean():
    """Exact rules carry no ``cosine``, so the tolerance pass would skip them."""
    artifact = {
        "passed": True,
        "status": "passed",
        "error": None,
        "module_goldens": {
            "layer0.expert_ids": {
                "module": "moe_expert_ids",
                "rule": "exact",
                "passed": False,
                "problems": ["4 of 257 tokens selected a different expert set"],
            }
        },
    }
    report = ev.audit_artifact(artifact, TOLERANCES)
    assert not report["clean"]
    assert report["strict_failures"][0]["check"] == "layer0.expert_ids"
    assert any("strict failures" in note for note in report["verdict_disagreements"])


def test_a_passing_exact_rule_audits_clean():
    artifact = {
        "passed": True,
        "status": "passed",
        "error": None,
        "module_goldens": {
            "layer0.expert_ids": {
                "module": "moe_expert_ids",
                "rule": "exact",
                "passed": True,
                "problems": [],
            }
        },
    }
    assert ev.audit_artifact(artifact, TOLERANCES)["clean"]


def test_an_exact_rule_with_scalar_evidence_still_skips_the_float_pass():
    """Structural scalars must not be mistaken for tolerance metrics.

    The clamp rule records counts and peaks so a single-rank failure is
    diagnosable from the artifact; none of them is a registered limit, and the
    entry must still be judged by its rule rather than by ``moe_expert_output``.
    """
    artifact = {
        "passed": True,
        "status": "passed",
        "error": None,
        "module_goldens": {
            "layer3.swiglu_clamp_is_live": {
                "module": "moe_expert_output",
                "rule": "exact",
                "metrics": {"changed_fraction": 0.99, "abs_max_with_limit": 164.0},
                "passed": True,
                "problems": [],
            }
        },
    }
    assert ev.audit_artifact(artifact, TOLERANCES)["clean"]


def test_float_checks_are_still_re_judged_from_the_manifest():
    artifact = {
        "passed": True,
        "status": "passed",
        "error": None,
        "module_goldens": {
            "layer3.routed_output": {
                "module": "moe_expert_output",
                "passed": True,
                "metrics": {"cosine": 0.9999, "rel_max_abs": 0.9, "finite": True},
            }
        },
    }
    report = ev.audit_artifact(artifact, TOLERANCES)
    assert not report["clean"]
    assert report["strict_failures"][0]["module"] == "moe_expert_output"


# ---------------------------------------------------------------------------
# Fused-op recording.
# ---------------------------------------------------------------------------


def _detached_recorder(op, block_scale=None, dense=None):
    """A recorder wired to fakes, so the namespace tests touch no real op.

    The recorder patches three surfaces --- the ``torch.ops.trtllm`` namespace,
    the block-scale backend's module globals, and the dense parity GEMM's ---
    and all three have to be put back, so all three are represented here rather
    than only the historical one.
    """

    class _Namespace:
        pass

    namespace = _Namespace()
    namespace.fused_moe = op
    if block_scale is None:
        block_scale = types.SimpleNamespace(
            moe_w4a8_gemm=lambda *a, **kw: None,
            quantize_blockwise_ue8m0=lambda *a, **kw: None,
            swiglu_and_quantize=lambda *a, **kw: None,
        )
    if dense is None:
        dense = types.SimpleNamespace(fp8_blockwise_gemm=lambda *a, **kw: None)
    recorder = mr.FusedMoERecorder.__new__(mr.FusedMoERecorder)
    recorder.calls = []
    recorder.block_scale_calls = []
    recorder.act_quant_calls = []
    recorder.dense_gemm_calls = []
    recorder._ns = namespace
    recorder._orig = op
    recorder._bs = block_scale
    recorder._orig_gemm = block_scale.moe_w4a8_gemm
    recorder._orig_quant = block_scale.quantize_blockwise_ue8m0
    recorder._orig_swiglu = block_scale.swiglu_and_quantize
    recorder._dense = dense
    recorder._orig_dense_gemm = dense.fp8_blockwise_gemm
    return recorder, namespace, block_scale, dense


def test_fused_moe_recorder_restores_the_namespace():
    """The recorder must not leave a wrapper behind on the op namespace."""
    calls = []

    def op(x, sel, scales, w3_w1, **kw):
        calls.append(kw)
        return "result"

    recorder, namespace, block_scale, dense = _detached_recorder(op)
    originals = (
        block_scale.moe_w4a8_gemm,
        block_scale.quantize_blockwise_ue8m0,
        block_scale.swiglu_and_quantize,
        dense.fp8_blockwise_gemm,
    )

    x = torch.zeros(3, 8, dtype=torch.bfloat16)
    sel = torch.zeros(3, 6, dtype=torch.int32)
    w = torch.zeros(4, 16, 4, dtype=torch.uint8)
    with recorder:
        assert namespace.fused_moe is not op
        out = namespace.fused_moe(
            x,
            sel,
            None,
            w,
            use_w4_group_scaling=True,
            swiglu_limit=torch.full((4,), 10.0),
            ep_size=8,
            ep_rank=1,
        )
    assert out == "result"
    assert namespace.fused_moe is op
    assert (
        block_scale.moe_w4a8_gemm,
        block_scale.quantize_blockwise_ue8m0,
        block_scale.swiglu_and_quantize,
        dense.fp8_blockwise_gemm,
    ) == originals
    assert len(recorder.calls) == 1
    seen = recorder.calls[0]
    assert seen["activation_dtype"] == "torch.bfloat16"
    assert seen["weight_dtype"] == "torch.uint8"
    assert seen["experts_per_token"] == 6
    assert seen["use_w4_group_scaling"] is True
    assert seen["swiglu_limit"] == 10.0
    assert seen["ep_size"] == 8 and seen["ep_rank"] == 1


def test_recorder_restores_the_namespace_after_an_exception():
    def op(*a, **kw):
        raise RuntimeError("kernel blew up")

    recorder, namespace, block_scale, dense = _detached_recorder(op)
    original_gemm = block_scale.moe_w4a8_gemm
    original_dense = dense.fp8_blockwise_gemm

    with pytest.raises(RuntimeError, match="kernel blew up"):
        with recorder:
            namespace.fused_moe(
                torch.zeros(1, 2, dtype=torch.bfloat16),
                torch.zeros(1, 6, dtype=torch.int32),
                None,
                torch.zeros(1, 2, 1, dtype=torch.uint8),
            )
    assert namespace.fused_moe is op
    assert block_scale.moe_w4a8_gemm is original_gemm
    assert dense.fp8_blockwise_gemm is original_dense


# ---------------------------------------------------------------------------
# Routed accumulation semantics.
# ---------------------------------------------------------------------------


def test_source_routed_uses_the_sources_own_scatter_semantics():
    """``y[idx] += v`` assigns rather than accumulates on repeated rows.

    That is what ``MoE.forward`` does, so the reconstruction has to do it too;
    a reconstruction that accumulated would disagree with the source on any
    token that ever selected one expert twice.
    """
    indices = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.int32)
    store = {
        "l0.e0": {"output": torch.ones(1, 4)},
        "l0.e1": {"output": torch.full((2, 4), 2.0)},
    }
    y, fired = mr._source_routed(store, 0, indices, (2, 4), [0, 1, 2, 3])
    assert fired == [0, 1]
    # Token 0 picked expert 0 twice: one assignment, not two.
    assert torch.equal(y[0], torch.ones(4) + 2.0)
    assert torch.equal(y[1], torch.full((4,), 2.0))


def test_source_routed_skips_experts_that_never_fired():
    indices = torch.tensor([[4, 5]], dtype=torch.int32)
    y, fired = mr._source_routed({}, 0, indices, (1, 4), [0, 1])
    assert fired == []
    assert float(y.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Can the repo's existing SM90 Triton W4A8 path serve DeepSeek-V4?
# ---------------------------------------------------------------------------

import moe_w4a8_feasibility as feas  # noqa: E402


def _routed_quant_config():
    """The routed experts' own contract, as `ModelConfig` builds it for SM90.

    Packed MXFP4 in an I8 container with one UE8M0 scale per 32 logical K
    values -- deliberately *not* the model-global algorithm.
    """
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    return QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4, group_size=32)


def _v4_resolver_inputs():
    """A real DeepSeek-V4 ModelConfig and its real routing method, TP8/EP8.

    The model-global config is the checkpoint's *dense* contract,
    ``FP8_BLOCK_SCALES`` with 128x128 scales, because that is what
    ``ModelConfig`` resolves for DeepSeek-V4 and what a probe that forgets the
    routed override would silently answer about. Every test here that cares
    about the routed experts must pass ``_routed_quant_config()`` explicitly,
    which is what production does.
    """
    from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe.routing import DeepSeekV4MoeRoutingMethod
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    cfg = DeepseekV4Config()
    model_config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(world_size=8, tp_size=8, moe_ep_size=8, rank=0),
        moe_backend="CUTLASS",
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128),
    )
    routing = DeepSeekV4MoeRoutingMethod(
        top_k=cfg.num_experts_per_tok,
        n_group=getattr(cfg, "n_group", 1) or 1,
        topk_group=getattr(cfg, "topk_group", 1) or 1,
        routed_scaling_factor=cfg.routed_scaling_factor,
        callable_e_score_correction_bias=lambda: None,
        callable_tid2eid=lambda: None,
        is_hashed=True,
    )
    shape = {
        "n_routed_experts": cfg.n_routed_experts,
        "n_hash_layers": cfg.n_hash_layers,
        "hidden_size": cfg.hidden_size,
        "moe_inter_dim": cfg.moe_intermediate_size,
    }
    return model_config, routing, shape


def _probe(**kwargs):
    model_config, routing, shape = _v4_resolver_inputs()
    return feas.resolver_probe(
        model_config,
        routing,
        shape,
        experts_quant_config=kwargs.pop("experts_quant_config", _routed_quant_config()),
        **kwargs,
    )


def test_the_resolver_does_not_dispatch_triton_for_deepseek_v4():
    """The rejection is the resolver's own, recorded rather than paraphrased."""
    probe = _probe()
    assert probe["as_configured"]["triton_eligible"] is False
    assert probe["as_configured"]["winner"] == "CutlassFusedMoE"
    rejected = probe["as_configured"]["triton_rejected"]
    assert [r["reason"] for r in rejected] == ["activation_unsupported"]
    assert "swiglu_gptoss_style=True" in rejected[0]["detail"]


def test_the_gate_ladder_names_every_claim_dispatch_would_need():
    """Two contracts, not one: the SwiGLU package and the routing family.

    The ladder must keep finding the *next* gate, because "fix this one thing"
    is exactly the wrong summary of a two-contract obstruction.
    """
    probe = _probe()
    assert probe["claims_required_to_dispatch"] == ["routing", "swiglu_gptoss_style"]
    reasons = [
        step["result"]["triton_rejected"][0]["reason"]
        for step in probe["gate_ladder"]
        if step["result"]["triton_rejected"]
    ]
    assert reasons == ["activation_unsupported", "routing_unsupported"]
    dispatching = [s for s in probe["gate_ladder"] if s["result"]["triton_eligible"]]
    assert dispatching and dispatching[0]["result"]["winner"] == "TritonFusedMoE"


# ---------------------------------------------------------------------------
# The mixed-precision leak: whose quantization contract is being asked about?
# ---------------------------------------------------------------------------


def test_the_probe_asks_about_the_routed_experts_not_the_dense_layers():
    """DeepSeek-V4 is mixed precision, so the two configs must not be confused.

    The regression this pins is a real one: asking ``build_moe_problem``
    without the routed override answered with the model-global
    ``FP8_BLOCK_SCALES`` dense contract and reported a *quant* rejection that
    the routed experts never hit, inflating the blocking-claim list from two to
    three. Every recorded problem must carry the routed algorithm.
    """
    probe = _probe()
    provenance = probe["quant_provenance"]
    assert provenance["routed_expert_quant_algo"] == "W4A16_MXFP4"
    assert provenance["routed_expert_group_size"] == 32
    assert provenance["model_global_quant_algo"] == "FP8_BLOCK_SCALES"

    for ladder in ("as_loaded", "at_source_precision"):
        for step in probe[ladder]["gate_ladder"]:
            assert step["result"]["quant"] != "FP8_BLOCK_SCALES", (ladder, step["step"])
            for rejection in step["result"]["triton_rejected"]:
                assert "FP8_BLOCK_SCALES" not in rejection["detail"], (ladder, step["step"])


def test_the_dense_config_cannot_silently_stand_in_for_the_experts():
    """A probe that loses the override must fail loudly, not answer anyway.

    This replays the iteration-30 defect exactly --- a ``build_moe_problem``
    call whose override does not reach the problem --- because without the
    guard the failure is invisible: the resolver happily answers a well-formed
    question about the dense layers and the artifact attributes that answer to
    the experts.
    """
    from tensorrt_llm._torch.modules.fused_moe import moe_resolution

    real = moe_resolution.build_moe_problem

    def drops_the_override(model_config, *, override_quant_config=None, **kwargs):
        return real(model_config, **kwargs)

    model_config, routing, shape = _v4_resolver_inputs()
    with unittest.mock.patch.object(moe_resolution, "build_moe_problem", drops_the_override):
        with pytest.raises(AssertionError, match="routed-expert contract"):
            feas.resolver_probe(
                model_config,
                routing,
                shape,
                experts_quant_config=_routed_quant_config(),
            )


def test_the_routed_contract_is_read_off_the_running_module():
    """The live backend's config is authoritative; the override is provenance."""
    routed = _routed_quant_config()
    moe = types.SimpleNamespace(
        experts=types.SimpleNamespace(
            backend=types.SimpleNamespace(quant_config=routed),
            _override_quant_config=routed,
        )
    )
    assert feas.live_experts_quant_config(moe) is routed


def test_a_backend_running_a_different_contract_than_its_override_is_rejected():
    """If those two disagree the question has no single answer, so refuse it."""
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    moe = types.SimpleNamespace(
        experts=types.SimpleNamespace(
            backend=types.SimpleNamespace(quant_config=_routed_quant_config()),
            _override_quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        )
    )
    with pytest.raises(AssertionError, match="resolver question is ambiguous"):
        feas.live_experts_quant_config(moe)


def test_the_w4a8_question_selects_w4a8_from_the_start():
    """Asking about W4A8 means running the whole ladder at W4A8.

    Declaring it part-way up a ladder that began at another contract cannot
    distinguish "the quant was the gate" from "the quant changed at the same
    step something else did".
    """
    probe = _probe()
    at_source = probe["at_source_precision"]
    assert [step["result"]["quant"] for step in at_source["gate_ladder"]] == ["W4A8_MXFP4_FP8"] * 3
    # Same two semantic gates: the activation precision was never what blocked
    # dispatch, which is why switching to it does not unblock anything either.
    assert at_source["claims_required_to_dispatch"] == ["routing", "swiglu_gptoss_style"]
    assert at_source["as_configured"]["triton_rejected"][0]["reason"] == "activation_unsupported"


def test_no_ladder_claims_the_quantization_algorithm():
    """`quant` is chosen at each ladder's base, never claimed as a step."""
    assert "quant" not in {field for _, field, _ in feas._LADDER_CLAIMS}
    probe = _probe()
    for ladder in ("as_loaded", "at_source_precision"):
        assert "quant" not in (probe[ladder]["claims_required_to_dispatch"] or [])
        for step in probe[ladder]["gate_ladder"]:
            assert "quant" not in step["problem_claims"]


@CUDA
def test_sm90_cannot_reach_w4a8_by_choosing_the_triton_backend():
    """Recorded because it is the obvious next suggestion, and it is wrong.

    ``get_mxfp4_quant_algo`` gates W4A8 on ``sm >= 100``; below that every
    backend name resolves the routed experts to ``W4A16_MXFP4``, so reaching
    the source's activation precision is not a backend-selection question.
    """
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() != 90:
        pytest.skip("describes the SM90 selection gate")
    selection = _probe()["sm90_mxfp4_algo_selection"]
    assert selection["sm_version"] == 90
    assert selection["by_moe_backend"] == {
        "TRITON": "W4A16_MXFP4",
        "CUTLASS": "W4A16_MXFP4",
    }


def test_the_dispatch_reading_keeps_both_halves_of_the_answer():
    """Neither "relax the gates" nor "switch backend" is the whole finding."""
    probe = _probe()
    arms = {
        "source_blockwise128_ue8m0": {"fc2": {"rel_max_abs": 0.0}},
        "triton_per_tensor_fp8_layer_scope": {"fc2": {"rel_max_abs": 0.30}},
        "triton_per_tensor_fp8_expert_scope": {"fc2": {"rel_max_abs": 0.26}},
        "bf16_w4a16": {"fc2": {"rel_max_abs": 0.23}},
    }
    reading = feas._dispatch_reading(probe, {"reading": feas._reading(arms)})
    assert reading["quant_is_not_among_them"] is True
    assert reading["blocking_claims_at_the_loaded_quant_contract"] == [
        "routing",
        "swiglu_gptoss_style",
    ]
    assert reading["would_dispatching_close_the_gap"] is False
    assert "W4A16_MXFP4" in reading["verdict"]
    assert "would not move FC2 closer" in reading["verdict"]


@pytest.mark.parametrize(
    "per_tensor_fc2, expected_improves",
    [(0.10, True), (0.60, False)],
)
def test_the_reading_states_whether_reuse_would_help(per_tensor_fc2, expected_improves):
    """The verdict is derived from the measured arms, never asserted."""
    arms = {
        "source_blockwise128_ue8m0": {"fc2": {"rel_max_abs": 0.01}},
        "triton_per_tensor_fp8_layer_scope": {"fc2": {"rel_max_abs": per_tensor_fc2}},
        "triton_per_tensor_fp8_expert_scope": {"fc2": {"rel_max_abs": per_tensor_fc2}},
        "bf16_w4a16": {"fc2": {"rel_max_abs": 0.30}},
    }
    reading = feas._reading(arms)
    assert reading["triton_w4a8_improves_on_todays_w4a16"] is expected_improves
    assert ("would move FC2 closer" in reading["verdict"]) is expected_improves


def test_the_optimistic_arm_is_the_one_that_decides():
    """Expert-scope is a tighter scale than the kernel's, so it cannot lose.

    If the layer-scope arm were used alone a reader could object that the
    comparison was rigged pessimistically; the verdict therefore takes the
    better of the two.
    """
    arms = {
        "source_blockwise128_ue8m0": {"fc2": {"rel_max_abs": 0.01}},
        "triton_per_tensor_fp8_layer_scope": {"fc2": {"rel_max_abs": 0.90}},
        "triton_per_tensor_fp8_expert_scope": {"fc2": {"rel_max_abs": 0.10}},
        "bf16_w4a16": {"fc2": {"rel_max_abs": 0.30}},
    }
    assert feas._reading(arms)["triton_w4a8_improves_on_todays_w4a16"] is True


@CUDA
def test_the_two_w4a8_activation_contracts_are_not_interchangeable():
    """Blockwise-128 UE8M0 and per-tensor FP8 are different numbers.

    Not a directional claim: E4M3 carries its own exponent, so unlike an
    integer format its accuracy is largely scale-invariant and the finer
    scope does *not* automatically win --- measured, it can lose, because the
    source's UE8M0 scale is rounded up to a power of two while the per-tensor
    op keeps an exact amax/448. What matters for feasibility is that swapping
    one for the other changes the result, so reuse is a semantic change and
    not a drop-in.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    x[0] *= 64.0  # one loud row, as a real activation has
    block = tg.fp8_quant_dequant(x, feas.SOURCE_ACT_BLOCK).to(x.dtype)
    flat = feas._per_tensor(x)
    assert not torch.equal(block, flat)
    assert float((block.float() - flat.float()).abs().max()) > 0.0


@CUDA
def test_the_blockwise_arm_is_the_sources_own_quantiser():
    """The arm labelled `source_blockwise128_ue8m0` must be exactly that."""
    torch.manual_seed(2)
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(
        feas._blockwise(tg)(x), tg.fp8_quant_dequant(x, feas.SOURCE_ACT_BLOCK).to(x.dtype)
    )


@CUDA
def test_the_layer_scale_is_the_one_the_kernel_would_impose():
    """The kernel's scale comes from the whole tensor, not one expert's rows.

    Asserts the scope, not which scope wins: `_with_scale_of` must reproduce
    quantising with the layer-wide scale, and must differ from quantising the
    rows alone whenever the layer contains something louder.
    """
    torch.manual_seed(1)
    whole = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    whole[0] *= 64.0
    rows = whole[8:16]
    assert not torch.equal(feas._with_scale_of(rows, whole), feas._per_tensor(rows))
    # Same tensor, same scale: the two agree when the scope is not wider.
    assert torch.equal(feas._with_scale_of(rows, rows), feas._per_tensor(rows))


# ---------------------------------------------------------------------------
# The SM90 block-scale W4A8 routed-expert backend.
#
# The gap Stage 3 measured is an activation-precision gap: the checkpoint's own
# `linear` quantises routed-expert activations to FP8 per token and per 128 K
# values with a power-of-two scale before both GEMMs, and the pre-Blackwell
# packed-MXFP4 Cutlass path leaves them in BF16. These tests hold weights,
# clamp, routing weight and GEMM dtype fixed and vary only that quantiser, so a
# pass here means the backend implements the source's contract rather than
# merely being close to it.
# ---------------------------------------------------------------------------

from tensorrt_llm._torch.modules.fused_moe import mxfp4_blockscale_kernels as bsk  # noqa: E402
from tensorrt_llm._torch.modules.fused_moe.fused_moe_mxfp4_blockscale import (  # noqa: E402
    BLOCK_M,
    BlockScaleMXFP4FusedMoE,
)
from tensorrt_llm._torch.modules.fused_moe.impl_contract import (  # noqa: E402
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
)
from tensorrt_llm.quantization.mode import QuantAlgo  # noqa: E402

#: Real DeepSeek-V4 routed-expert dimensions. The expert count is reduced --- a
#: rank owns 32 of them and the arithmetic under test is per expert --- but
#: hidden size, intermediate size and the clamp are the checkpoint's.
V4_HIDDEN, V4_INTER, V4_LIMIT = 4096, 2048, 10.0


def _block_scale_problem(**overrides):
    problem = MoEProblem(
        quant=QuantAlgo.W4A8_MXFP4_FP8.value,
        dtype_act=torch.bfloat16,
        hidden_size=V4_HIDDEN,
        intermediate_size=V4_INTER,
        num_experts=256,
        top_k=6,
        swiglu_gptoss_style=False,
        bias=False,
        routing="DeepSeekV4",
        act_scale_fmt="ue8m0",
    )
    return type(problem)(**{**problem.__dict__, **overrides})


def _deployment(**overrides):
    fields = dict(
        ep_size=8,
        tp_size=1,
        use_dp=False,
        num_slots=256,
        env=MoEEnvironment(sm=90),
        parallel_size=8,
    )
    fields.update(overrides)
    return MoEDeployment(**fields)


def test_the_block_scale_backend_serves_the_checkpoints_declared_contract():
    assert BlockScaleMXFP4FusedMoE.can_implement(_block_scale_problem(), _deployment()).eligible


@pytest.mark.parametrize(
    "overrides, reason",
    [
        # The distinguishing case: `W4A8_MXFP4_FP8` is also the algorithm of
        # the per-tensor FP8 path, and the two quantise the same activation to
        # different codes. Without the declared format this backend must not
        # claim the layer.
        ({"act_scale_fmt": None}, MoERejectReason.QUANT_UNSUPPORTED),
        ({"act_scale_fmt": "e8m0"}, MoERejectReason.QUANT_UNSUPPORTED),
        ({"quant": QuantAlgo.W4A16_MXFP4.value}, MoERejectReason.QUANT_UNSUPPORTED),
        ({"quant": QuantAlgo.W4A8_MXFP4_MXFP8.value}, MoERejectReason.QUANT_UNSUPPORTED),
        ({"quant": None}, MoERejectReason.QUANT_UNSUPPORTED),
        ({"dtype_act": torch.float16}, MoERejectReason.DTYPE_UNSUPPORTED),
        ({"swiglu_gptoss_style": True}, MoERejectReason.ACTIVATION_UNSUPPORTED),
        ({"hidden_size": 4000}, MoERejectReason.SHAPE_UNALIGNED),
        ({"intermediate_size": 2000}, MoERejectReason.SHAPE_UNALIGNED),
    ],
)
def test_a_layer_outside_the_contract_is_rejected_with_its_own_reason(overrides, reason):
    verdict = BlockScaleMXFP4FusedMoE.can_implement(_block_scale_problem(**overrides), _deployment())
    assert not verdict.eligible
    assert verdict.reject_reason is reason


def test_eplb_and_indivisible_shards_are_rejected():
    eplb = BlockScaleMXFP4FusedMoE.can_implement(
        _block_scale_problem(), _deployment(eplb_enabled=True)
    )
    assert eplb.reject_reason is MoERejectReason.EPLB_UNSUPPORTED
    ragged = BlockScaleMXFP4FusedMoE.can_implement(
        _block_scale_problem(num_experts=255), _deployment()
    )
    assert ragged.reject_reason is MoERejectReason.SLOTS_NOT_DIVISIBLE_BY_EP


# ---------------------------------------------------------------------------
# Which contract the checkpoint's routed experts are declared to have.
# ---------------------------------------------------------------------------


def _declared_routed_quant_config(sm: int, scale_fmt="ue8m0", moe_backend="CUTLASS"):
    from tensorrt_llm._torch import model_config as mc

    pretrained = types.SimpleNamespace(
        num_hidden_layers=2,
        quantization_config=(
            {} if scale_fmt is None else {"quant_method": "fp8", "scale_fmt": scale_fmt}
        ),
    )
    with unittest.mock.patch.object(
        mc.ModelConfig, "_detect_deepseek_v4_routed_moe_layout", staticmethod(lambda _d: "mxfp4")
    ), unittest.mock.patch.object(mc, "get_sm_version", lambda: sm):
        layers = mc.ModelConfig._set_deepseek_v4_routed_moe_quant_config(
            pretrained, "/nonexistent", moe_backend, None
        )
    return layers["model.layers.0.mlp.experts"]


def test_sm90_routed_experts_declare_the_checkpoints_w4a8_contract():
    """The layer says what the checkpoint is, not what one kernel can do.

    `get_mxfp4_quant_algo` answers W4A16 before Blackwell because that is the
    only thing the Cutlass packed-MXFP4 kernel implements. That is a statement
    about a kernel; the layer's quantization config is a statement about the
    checkpoint, and this one declares block-scaled FP8 activations.
    """
    config = _declared_routed_quant_config(sm=90)
    assert config.quant_algo is QuantAlgo.W4A8_MXFP4_FP8
    assert config.scale_fmt == "ue8m0"
    assert config.group_size == 32


def test_a_checkpoint_that_declares_no_scale_format_keeps_the_w4a16_path():
    """Nothing declared, nothing assumed: the historical answer stands."""
    config = _declared_routed_quant_config(sm=90, scale_fmt=None)
    assert config.quant_algo is QuantAlgo.W4A16_MXFP4
    assert config.scale_fmt is None


@pytest.mark.parametrize("sm", [100, 103])
def test_blackwell_routed_expert_selection_is_unchanged(sm):
    """The protected path keeps the algorithm its own selector returns."""
    from tensorrt_llm._torch.model_config import ModelConfig

    for backend in ("CUTLASS", "TRTLLM", "TRITON"):
        config = _declared_routed_quant_config(sm=sm, moe_backend=backend)
        with unittest.mock.patch(
            "tensorrt_llm._torch.model_config.get_sm_version", lambda: sm
        ):
            expected = ModelConfig.get_mxfp4_quant_algo(backend)
        assert config.quant_algo is expected
        assert config.scale_fmt is None


def test_an_unrecognised_scale_format_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match="not a supported"):
        _declared_routed_quant_config(sm=90, scale_fmt="UE8M0")


# ---------------------------------------------------------------------------
# The kernels, against the independent golden.
# ---------------------------------------------------------------------------


@CUDA
def test_the_activation_quantiser_is_the_sources_own():
    """Same codes and same scales as `act_quant(x, 128, "ue8m0")`, not close.

    The scale is a power of two derived by exponent arithmetic; a float `log2`
    of an exact power of two can round up and shift every code in the block, so
    equality is the only assertion that catches it.
    """
    torch.manual_seed(3)
    x = (torch.randn(70, 512, device="cuda") * 3).to(torch.bfloat16)
    q, scale = bsk.quantize_blockwise_ue8m0(x)
    blocks = x.float().unflatten(-1, (-1, 128))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=bsk.AMAX_FLOOR)
    ref_scale = tg._round_scale(amax, 1.0 / bsk.FP8_MAX)
    ref_q = (blocks / ref_scale).clamp(-bsk.FP8_MAX, bsk.FP8_MAX).to(torch.float8_e4m3fn)

    assert scale.shape == (70, 512 // bsk.ACT_BLOCK_SIZE)
    assert torch.equal(scale, ref_scale.squeeze(-1))
    assert torch.equal(q.view(torch.uint8), ref_q.flatten(-2).view(torch.uint8))
    # Every scale is a power of two, which is what "ue8m0" means.
    assert torch.equal(scale, torch.exp2(torch.log2(scale).round()))
    # And the round trip is the harness's own quantiser.
    assert torch.equal(
        (q.float() * scale.repeat_interleave(128, dim=-1)).to(torch.bfloat16),
        tg.fp8_quant_dequant(x, 128),
    )


@CUDA
def test_an_all_zero_block_quantises_to_zero_rather_than_dividing_by_it():
    q, scale = bsk.quantize_blockwise_ue8m0(torch.zeros(2, 256, device="cuda", dtype=torch.bfloat16))
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert q.float().abs().sum() == 0


@CUDA
def test_the_gemm_reads_the_checkpoints_nibble_order_and_group_32_scales():
    """Decode agreement with the independent golden, and a control that moves.

    Swapping the two nibbles of every byte is the single most likely way to get
    a packed-MXFP4 reader wrong while still producing plausible numbers, so the
    swapped container is asserted to give a different answer.
    """
    torch.manual_seed(4)
    experts, n, k, rows = 2, 128, 256, BLOCK_M
    packed = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda")
    scale_bits = torch.randint(115, 135, (experts, n, k // 32), dtype=torch.uint8, device="cuda")
    a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    a_q, a_scale = bsk.quantize_blockwise_ue8m0(a)
    a_row = torch.arange(rows, dtype=torch.int32, device="cuda")
    expert_ids = torch.ones(1, dtype=torch.int32, device="cuda")

    got = bsk.moe_w4a8_gemm(a_q, a_scale, a_row, packed, scale_bits, expert_ids, experts, BLOCK_M)
    weight = tg.dequant_mxfp4(packed[1], scale_bits[1].view(torch.float8_e8m0fnu), group=32)
    dequantised = (a_q.float() * a_scale.repeat_interleave(128, dim=-1)).to(torch.bfloat16)
    ref = F.linear(dequantised, weight.to(torch.bfloat16))
    assert torch.equal(got, ref)

    swapped = ((packed << 4) | (packed >> 4)).to(torch.uint8)
    other = bsk.moe_w4a8_gemm(
        a_q, a_scale, a_row, swapped, scale_bits, expert_ids, experts, BLOCK_M
    )
    assert not torch.equal(got, other)

    # One UE8M0 exponent per 32 K values: bumping group 3 must change the
    # result, and it must change it by exactly a doubling of that group.
    louder = scale_bits.clone()
    louder[1, :, 3] += 1
    bumped = bsk.moe_w4a8_gemm(
        a_q, a_scale, a_row, packed, louder, expert_ids, experts, BLOCK_M
    )
    louder_weight = tg.dequant_mxfp4(
        packed[1], louder[1].view(torch.float8_e8m0fnu), group=32
    )
    group = slice(3 * 32, 4 * 32)
    assert torch.equal(louder_weight[:, group], 2.0 * weight[:, group])
    assert torch.equal(
        louder_weight[:, : 3 * 32], weight[:, : 3 * 32]
    ), "bumping one exponent must not move another group"
    assert not torch.equal(bumped, got)
    # Tracks the bumped reference. Not bit-equality: the two sum the same exact
    # per-group products in different orders, and at these magnitudes one FP32
    # ordering difference is enough to land on the other side of a BF16 tie.
    assert _relative(bumped, F.linear(dequantised, louder_weight.to(torch.bfloat16))) < 0.05


@CUDA
def test_padding_rows_and_foreign_expert_blocks_store_zeros():
    """Unfilled slots must be written, not left as whatever was in memory."""
    torch.manual_seed(5)
    experts, n, k = 2, 128, 256
    packed = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda")
    scale_bits = torch.randint(120, 130, (experts, n, k // 32), dtype=torch.uint8, device="cuda")
    a_q, a_scale = bsk.quantize_blockwise_ue8m0(
        torch.randn(BLOCK_M, k, device="cuda", dtype=torch.bfloat16)
    )
    a_row = torch.full((2 * BLOCK_M,), -1, dtype=torch.int32, device="cuda")
    a_row[:BLOCK_M] = torch.arange(BLOCK_M, dtype=torch.int32, device="cuda")

    # Second block is all padding; first block belongs to the sentinel expert.
    out = bsk.moe_w4a8_gemm(
        a_q,
        a_scale,
        a_row,
        packed,
        scale_bits,
        torch.tensor([experts, 0], dtype=torch.int32, device="cuda"),
        experts,
        BLOCK_M,
    )
    assert out.float().abs().sum() == 0


# ---------------------------------------------------------------------------
# The backend, at real expert dimensions.
# ---------------------------------------------------------------------------


def _build_block_scale_moe(num_experts=4, top_k=2):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe.routing import DefaultMoeRoutingMethod
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    quant = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_FP8, group_size=32, scale_fmt="ue8m0")
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, tp_size=1, moe_ep_size=1, rank=0),
        quant_config=quant,
        max_num_tokens=8192,
    )
    moe = BlockScaleMXFP4FusedMoE(
        routing_method=DefaultMoeRoutingMethod(top_k=top_k),
        num_experts=num_experts,
        hidden_size=V4_HIDDEN,
        intermediate_size=V4_INTER,
        dtype=torch.bfloat16,
        model_config=model_config,
        swiglu_limit_scalar=V4_LIMIT,
    )
    moe.create_weights()
    moe = moe.cuda()
    generator = torch.Generator(device="cuda").manual_seed(11)
    for weight in (moe.w3_w1_weight, moe.w2_weight):
        weight.data.copy_(
            torch.randint(
                0, 256, weight.shape, dtype=torch.uint8, device="cuda", generator=generator
            )
        )
    for scale in (moe.fc31_weight_scale, moe.fc2_weight_scale):
        scale.data.copy_(
            torch.randint(
                120, 128, scale.shape, dtype=torch.uint8, device="cuda", generator=generator
            )
        )
    return moe


def _source_arm(x, weights, route, *, quantise, swap_halves=False):
    """One expert, the source's own expression, with a stated act precision."""

    def project(v, w):
        src = tg.fp8_quant_dequant(v, 128) if quantise else v
        return F.linear(src.to(torch.bfloat16), w.to(torch.bfloat16))

    up_weight, gate_weight = weights["w3"], weights["w1"]
    if swap_halves:
        up_weight, gate_weight = gate_weight, up_weight
    up = project(x, up_weight).float().clamp(min=-V4_LIMIT, max=V4_LIMIT)
    gate = project(x, gate_weight).float().clamp(max=V4_LIMIT)
    return project((F.silu(gate) * up * route).to(x.dtype), weights["w2"])


def _expert_weights_of(moe, expert_id):
    fused = tg.dequant_mxfp4(
        moe.w3_w1_weight[expert_id],
        moe.fc31_weight_scale[expert_id].view(torch.float8_e8m0fnu),
        group=32,
    )
    return {
        "w3": fused[:V4_INTER],
        "w1": fused[V4_INTER:],
        "w2": tg.dequant_mxfp4(
            moe.w2_weight[expert_id],
            moe.fc2_weight_scale[expert_id].view(torch.float8_e8m0fnu),
            group=32,
        ),
    }


def _routed_reference(moe, x, experts, scales, **arm):
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert_id in range(moe.expert_size_per_partition):
        idx, top = torch.where(experts == expert_id + moe.slot_start)
        if idx.numel() == 0:
            continue
        out[idx] += _source_arm(
            x[idx], _expert_weights_of(moe, expert_id), scales[idx, top, None], **arm
        ).float()
    return out


def _relative(got, ref):
    """`max_abs / ref_rms`, the same reading the replay's tolerance uses."""
    return (
        (got.float() - ref.float()).abs().max() / ref.float().pow(2).mean().sqrt()
    ).item()


@CUDA
def test_the_backend_reproduces_the_source_arm_and_the_controls_stay_distinguishable():
    """The discriminating experiment, at the checkpoint's expert dimensions.

    Weights, clamp, routing weight and GEMM dtype are identical across arms;
    only the activation quantiser differs. The block-scale backend has to land
    on the source arm, and the two contracts it is *not* --- W4A16, and the
    per-tensor FP8 the existing Triton path would apply --- have to stay
    measurably further away, otherwise the test would pass for a backend that
    had not implemented anything in particular.
    """
    moe = _build_block_scale_moe()
    torch.manual_seed(13)
    tokens = 64
    x = (torch.randn(tokens, V4_HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    # Distinct experts per token: the source's `y[idx] +=` assigns rather than
    # accumulates on a repeated index, so a duplicate would put the golden and
    # any summing implementation into disagreement for reasons of its own.
    experts = torch.rand(tokens, moe.num_experts, device="cuda").topk(2, dim=-1).indices.int()
    scales = torch.rand(tokens, 2, device="cuda") + 0.5

    got = moe.run_moe(
        MoERunContext(
            token_selected_experts=experts,
            token_final_scales=scales,
            x=x,
            x_sf=None,
            output_dtype=torch.bfloat16,
        )
    )
    source = _routed_reference(moe, x, experts, scales, quantise=True)
    w4a16 = _routed_reference(moe, x, experts, scales, quantise=False)
    swapped = _routed_reference(moe, x, experts, scales, quantise=True, swap_halves=True)

    measured = _relative(got, source)
    assert measured < 0.04, f"block-scale backend is {measured} from the source arm"
    assert torch.nn.functional.cosine_similarity(
        got.float().flatten(), source.flatten(), dim=0
    ) > 0.9999

    # Controls. Both are correct implementations of a *different* contract, so
    # they must be visibly further from the source than the backend is.
    for name, arm in (("w4a16", w4a16), ("swapped_swiglu_halves", swapped)):
        distance = _relative(arm, source)
        assert distance > 4 * max(measured, 1e-6), (
            f"{name} is {distance} from the source arm and the backend is "
            f"{measured}; the experiment does not discriminate"
        )


@CUDA
def test_the_backend_honours_expert_parallel_shard_ownership():
    """Only this rank's slots contribute, and they contribute the same values."""
    moe = _build_block_scale_moe()
    torch.manual_seed(17)
    tokens = 32
    x = (torch.randn(tokens, V4_HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    experts = torch.rand(tokens, moe.num_experts, device="cuda").topk(2, dim=-1).indices.int()
    scales = torch.rand(tokens, 2, device="cuda") + 0.5

    def run(selected):
        return moe.run_moe(
            MoERunContext(
                token_selected_experts=selected,
                token_final_scales=scales,
                x=x,
                x_sf=None,
                output_dtype=torch.bfloat16,
            )
        )

    owned = run(experts)
    local_count = moe.expert_size_per_partition
    moe.slot_start = local_count
    try:
        assert torch.equal(run(experts + local_count), owned)
        assert run(experts).float().abs().sum() == 0
    finally:
        moe.slot_start = 0


@CUDA
def test_externally_routed_experts_are_consumed_rather_than_re_derived():
    """A hash-routed layer has no logits to route on, so the ids must be used.

    Feeding two different expert assignments for the same tokens and router
    state has to produce two different answers; a backend that re-derived
    top-k inside its kernel would return the same one twice.
    """
    moe = _build_block_scale_moe()
    torch.manual_seed(19)
    tokens = 16
    x = (torch.randn(tokens, V4_HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    scales = torch.ones(tokens, 2, device="cuda")
    first = torch.stack(
        [
            torch.zeros(tokens, dtype=torch.int32, device="cuda"),
            torch.ones(tokens, dtype=torch.int32, device="cuda"),
        ],
        dim=-1,
    )
    second = first + 2

    def run(selected):
        return moe.run_moe(
            MoERunContext(
                token_selected_experts=selected,
                token_final_scales=scales,
                x=x,
                x_sf=None,
                output_dtype=torch.bfloat16,
            )
        )

    assert not torch.equal(run(first), run(second))
    assert _relative(run(first), _routed_reference(moe, x, first, scales, quantise=True)) < 0.04


@CUDA
def test_two_identical_requests_produce_identical_tokens():
    """Greedy decoding has to be reproducible, so the combine must be too.

    A scatter-add would sum a token's expert rows in whatever order the atomics
    land; this asserts the fixed-order reduction the backend uses instead.
    """
    moe = _build_block_scale_moe()
    torch.manual_seed(23)
    tokens = 48
    x = (torch.randn(tokens, V4_HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    experts = torch.rand(tokens, moe.num_experts, device="cuda").topk(2, dim=-1).indices.int()
    scales = torch.rand(tokens, 2, device="cuda") + 0.5
    ctx = MoERunContext(
        token_selected_experts=experts,
        token_final_scales=scales,
        x=x,
        x_sf=None,
        output_dtype=torch.bfloat16,
    )
    first = moe.run_moe(ctx)
    for _ in range(4):
        assert torch.equal(moe.run_moe(ctx), first)


@CUDA
def test_the_routed_weights_stay_packed_after_construction():
    """No persistent BF16 expansion: the parameters are the checkpoint bytes."""
    moe = _build_block_scale_moe()
    assert moe.w3_w1_weight.dtype is torch.uint8
    assert moe.w2_weight.dtype is torch.uint8
    assert moe.fc31_weight_scale.dtype is torch.uint8
    assert moe.fc2_weight_scale.dtype is torch.uint8
    assert tuple(moe.w3_w1_weight.shape) == (
        moe.expert_size_per_partition,
        2 * V4_INTER,
        V4_HIDDEN // 2,
    )
    assert tuple(moe.fc31_weight_scale.shape) == (
        moe.expert_size_per_partition,
        2 * V4_INTER,
        V4_HIDDEN // 32,
    )
    assert tuple(moe.w2_weight.shape) == (
        moe.expert_size_per_partition,
        V4_HIDDEN,
        V4_INTER // 2,
    )
    assert tuple(moe.fc2_weight_scale.shape) == (
        moe.expert_size_per_partition,
        V4_HIDDEN,
        V4_INTER // 32,
    )


# ---------------------------------------------------------------------------
# `real_runtime` for whichever routed-expert path the resolver selected.
#
# The failure this guards against is a backend swap inheriting the previous
# backend's clean dispatch report: every gate below is checked by making it
# fail, because a gate that has only ever been seen passing has not been shown
# to be a gate at all.
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self, name, quant_algo="W4A8_MXFP4_FP8", scale_fmt="ue8m0"):
        self.__class__ = type(name, (_FakeBackend,), {})
        self.quant_method = types.SimpleNamespace()
        self.quant_config = types.SimpleNamespace(quant_algo=quant_algo, scale_fmt=scale_fmt)
        self.ep_size, self.ep_rank = 8, 3
        self.expert_size_per_partition = 32


def _block_scale_gemm_call(**overrides):
    call = {
        "rows": 1024,
        "k": 4096,
        "n": 4096,
        "act_dtype": "torch.float8_e4m3fn",
        "act_scale_dtype": "torch.float32",
        "act_scale_block": 128,
        "weight_dtype": "torch.uint8",
        "weight_scale_dtype": "torch.uint8",
        "weight_group_size": 32,
        "experts": 32,
        "act_scales_are_powers_of_two": True,
    }
    call.update(overrides)
    return call


def _block_scale_recorder(gemm_overrides=(), stages=("fc1_input", "fc2_input"), fused_calls=0):
    recorder = mr.FusedMoERecorder.__new__(mr.FusedMoERecorder)
    recorder.calls = [
        {
            "tokens": 8,
            "hidden_size": 4096,
            "activation_dtype": "torch.bfloat16",
            "experts_per_token": 6,
            "weight_dtype": "torch.uint8",
            "w3_w1_shape": [32, 4096, 2048],
            "use_w4_group_scaling": True,
            "swiglu_limit": 10.0,
            "ep_size": 8,
            "ep_rank": 3,
        }
    ] * fused_calls
    recorder.block_scale_calls = [_block_scale_gemm_call(**o) for o in (gemm_overrides or ({},))]
    recorder.act_quant_calls = [
        {"stage": stage, "block": 128, "swiglu_limit": 10.0} for stage in stages
    ]
    return recorder


def test_the_block_scale_dispatch_report_names_the_op_and_the_granularity():
    dispatch = mr._dispatch_evidence(
        _block_scale_recorder(), _FakeBackend("BlockScaleMXFP4FusedMoE")
    )
    assert dispatch["passed"], dispatch["problems"]
    assert dispatch["backend"] == "BlockScaleMXFP4FusedMoE"
    assert "moe_w4a8_gemm" in dispatch["op_path"] and "Triton" in dispatch["op_path"]
    assert dispatch["quant_algo"] == "W4A8_MXFP4_FP8"
    assert dispatch["scale_fmt"] == "ue8m0"
    assert dispatch["weight_group_sizes"] == [32]
    assert dispatch["activation_scale_blocks"] == [128]
    assert dispatch["quantised_stages"] == ["fc1_input", "fc2_input"]
    assert dispatch["ep"] == [8, 3]


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"gemm_overrides": []}, "did not run"),
        ({"gemm_overrides": [{"weight_dtype": "torch.bfloat16"}]}, "not packed"),
        ({"gemm_overrides": [{"weight_scale_dtype": "torch.bfloat16"}]}, "UE8M0 exponent bytes"),
        ({"gemm_overrides": [{"weight_group_size": 128}]}, "per 32 K values"),
        ({"gemm_overrides": [{"act_dtype": "torch.bfloat16"}]}, "not FP8 E4M3"),
        ({"gemm_overrides": [{"act_scale_block": 32}]}, "per 128 K values"),
        (
            {"gemm_overrides": [{"act_scales_are_powers_of_two": False}]},
            "not the UE8M0 recipe",
        ),
        ({"stages": ("fc1_input",)}, "quantises the input of both GEMMs"),
        ({"fused_calls": 2}, "must not be served by two kernels"),
    ],
)
def test_a_block_scale_dispatch_that_is_not_the_source_contract_fails(kwargs, fragment):
    if kwargs.get("gemm_overrides") == []:
        recorder = _block_scale_recorder()
        recorder.block_scale_calls = []
    else:
        recorder = _block_scale_recorder(**kwargs)
    dispatch = mr._dispatch_evidence(recorder, _FakeBackend("BlockScaleMXFP4FusedMoE"))
    assert not dispatch["passed"]
    assert any(fragment in problem for problem in dispatch["problems"]), dispatch["problems"]


def test_an_unregistered_backend_cannot_inherit_a_clean_dispatch_report():
    """A future backend has to declare its own contract before it can pass."""
    recorder = _block_scale_recorder()
    dispatch = mr._dispatch_evidence(recorder, _FakeBackend("SomeNewFusedMoE"))
    assert not dispatch["passed"]
    assert any("no dispatch contract is registered" in p for p in dispatch["problems"])


def test_the_cutlass_dispatch_report_is_unchanged():
    """The protected W4A16 path keeps the report it had before the split."""
    recorder = _block_scale_recorder(fused_calls=1)
    recorder.block_scale_calls = []
    recorder.act_quant_calls = []
    dispatch = mr._dispatch_evidence(recorder, _FakeBackend("CutlassFusedMoE", "W4A16_MXFP4", None))
    assert dispatch["passed"], dispatch["problems"]
    assert dispatch["op_path"] == "torch.ops.trtllm.fused_moe"
    assert dispatch["use_w4_group_scaling"] == [True]
    assert dispatch["weight_dtypes"] == ["torch.uint8"]

    recorder.calls = []
    empty = mr._dispatch_evidence(recorder, _FakeBackend("CutlassFusedMoE", "W4A16_MXFP4", None))
    assert not empty["passed"]
    assert any("the kernel did not run" in p for p in empty["problems"])


# ---------------------------------------------------------------------------
# The clamp-liveness control must perturb the clamp the backend actually reads.
#
# `BlockScaleMXFP4FusedMoE` consumes `swiglu_limit_scalar`; the Cutlass path
# consumes the per-slot `swiglu_limit` tensor. A control that toggles only one
# of them reports "the limit does not reach the kernel" about itself whenever
# the other one is live, which is exactly what happened on the first real run
# of the new backend.
# ---------------------------------------------------------------------------


class _ClampSpyBackend:
    """A stand-in fused-MoE backend that reads one named clamp attribute."""

    def __init__(self, reads, swiglu_limit=None, swiglu_limit_scalar=None):
        self.reads = reads
        self.swiglu_limit = swiglu_limit
        self.swiglu_limit_scalar = swiglu_limit_scalar

    def limit(self):
        value = getattr(self, self.reads) if self.reads else None
        if value is None:
            return None
        return float(value.flatten()[0]) if torch.is_tensor(value) else float(value)


class _ClampSpyMoE:
    """Applies the backend's live clamp so the control's two runs can differ."""

    def __init__(self, backend):
        self.experts = backend

    def gate(self, x):
        return torch.zeros(x.shape[0], 4, device=x.device)

    def __call__(self, *a, **kw):  # pragma: no cover - not used by the control
        raise AssertionError("the clamp control must not call the MoE wrapper")


def _clamp_spy_moe(backend):
    moe = _ClampSpyMoE(backend)

    def experts(x, router_logits, **kw):
        limit = backend.limit()
        value = x.float()
        return (value if limit is None else value.clamp(-limit, limit)).to(kw["output_dtype"])

    moe.experts = experts
    moe.experts.backend = backend
    return moe


class _CollectingRecorder:
    def __init__(self):
        self.exacts = {}

    def exact(self, name, module, passed, detail):
        self.exacts[name] = (passed, detail)


def _run_clamp_control(backend):
    recorder = _CollectingRecorder()
    # Every element must clear the limit once the control drives the input by
    # 16x, otherwise the "more than half the elements changed" rule fails for
    # a reason that is about the fixture rather than about the clamp.
    hidden = (torch.rand(64, 8, device="cuda") + 1.0).to(torch.bfloat16)
    evidence = mr._clamp_evidence(
        recorder,
        "layerX",
        {},
        _clamp_spy_moe(backend),
        backend,
        hidden,
        torch.zeros(64, dtype=torch.int32, device="cuda"),
    )
    return recorder.exacts["layerX.swiglu_clamp_is_live"], evidence


@CUDA
@pytest.mark.parametrize(
    "backend",
    [
        _ClampSpyBackend("swiglu_limit_scalar", swiglu_limit_scalar=10.0),
        _ClampSpyBackend(
            "swiglu_limit", swiglu_limit=torch.full((4,), 10.0, dtype=torch.float32)
        ),
    ],
    ids=["scalar_backed", "tensor_backed"],
)
def test_the_clamp_control_perturbs_whichever_owner_the_backend_reads(backend):
    (passed, detail), evidence = _run_clamp_control(backend)
    assert passed, detail.get("problem")
    assert evidence["configured_limit"] == 10.0
    assert evidence["clamp_owners"] == [backend.reads]
    assert evidence["changed_fraction"] > 0.5
    # Restored, not left disabled for whatever runs next.
    assert backend.limit() == 10.0


@CUDA
def test_a_backend_that_ignores_the_clamp_still_fails_the_control():
    """The control must not become 'always passes' once it toggles both owners."""
    backend = _ClampSpyBackend(None, swiglu_limit_scalar=10.0)
    (passed, detail), evidence = _run_clamp_control(backend)
    assert not passed
    assert evidence["elements_changed"] == 0
    assert "does not reach the kernel" in detail["problem"]


def test_a_backend_with_no_clamp_at_all_cannot_be_probed():
    with pytest.raises(AssertionError, match="has nothing to perturb"):
        mr._live_clamp_owners(_ClampSpyBackend(None))


def test_clamp_owners_that_disagree_are_refused_rather_than_averaged():
    backend = _ClampSpyBackend(
        "swiglu_limit_scalar",
        swiglu_limit=torch.full((4,), 7.0, dtype=torch.float32),
        swiglu_limit_scalar=10.0,
    )
    owners = mr._live_clamp_owners(backend)
    assert sorted(owners) == ["swiglu_limit", "swiglu_limit_scalar"]
    with pytest.raises(AssertionError, match="clamp owners disagree"):
        mr._clamp_value(owners)


def test_the_real_block_scale_backend_exposes_a_scalar_clamp_the_control_can_see():
    """Ties the control to the class it failed on, not to a stand-in."""
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_mxfp4_blockscale import (
        BlockScaleMXFP4FusedMoE,
    )

    assert "swiglu_limit_scalar" in mr._CLAMP_OWNERS
    # `run_moe` reads the scalar, so the scalar is what the control must toggle.
    source = inspect.getsource(BlockScaleMXFP4FusedMoE.run_moe)
    assert "self.swiglu_limit_scalar" in source


# ---------------------------------------------------------------------------
# The source's accumulation boundary: FP32 routed accumulator, replicated
# shared expert, one final cast.
#
# `MoE.forward` in the reference keeps `y` in FP32 across every local expert
# and across the expert-parallel reduction, adds a *replicated* shared expert
# to it, and casts once. Each of the three pieces is pinned here, including the
# negative case, because each one silently reverts to the old behaviour if its
# capability marker stops being read.
# ---------------------------------------------------------------------------


def _forward_impl_output_dtype(backend, requested):
    """What `ConfigurableMoE.forward_impl` hands the scheduler."""
    from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE

    seen = {}

    class _Scheduler:
        def forward(self, x, router_logits, **kwargs):
            seen["output_dtype"] = kwargs["output_dtype"]
            return x

    stub = types.SimpleNamespace(
        backend=backend,
        scheduler=_Scheduler(),
        enable_dwdp=False,
        repeat_idx=0,
        repeat_count=1,
    )
    ConfigurableMoE.forward_impl(
        stub,
        torch.zeros(2, 4, dtype=torch.bfloat16),
        torch.zeros(2, 4),
        output_dtype=requested,
    )
    return seen["output_dtype"]


def test_a_backend_that_advertises_an_fp32_accumulator_is_given_the_dtype_it_is_asked_for():
    backend = types.SimpleNamespace(returns_fp32_accumulator=True)
    assert _forward_impl_output_dtype(backend, torch.float32) is torch.float32
    # Unset still follows the input, so nothing changes for callers that say nothing.
    assert _forward_impl_output_dtype(backend, None) is torch.bfloat16


@pytest.mark.parametrize("backend", [types.SimpleNamespace(), None])
def test_every_other_backend_keeps_following_the_input_dtype(backend):
    """The protection: models that already pass an output dtype are untouched.

    ``modeling_laguna`` asks for float32 today and is given bfloat16; that
    behaviour is load-bearing for a path this bring-up does not test, so the
    exemption is keyed to the marker rather than to the mere presence of an
    argument.
    """
    assert _forward_impl_output_dtype(backend, torch.float32) is torch.bfloat16


def test_the_block_scale_backend_is_the_one_that_advertises_it():
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_mxfp4_blockscale import (
        BlockScaleMXFP4FusedMoE,
    )

    assert BlockScaleMXFP4FusedMoE.returns_fp32_accumulator is True
    assert getattr(CutlassFusedMoE, "returns_fp32_accumulator", False) is False


@pytest.mark.parametrize(
    "backend, expected",
    [
        (types.SimpleNamespace(returns_fp32_accumulator=True), torch.float32),
        (types.SimpleNamespace(returns_fp32_accumulator=False), torch.bfloat16),
        (types.SimpleNamespace(), torch.bfloat16),
    ],
)
def test_the_routed_accumulator_dtype_follows_the_resolved_backend(backend, expected):
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE

    stub = types.SimpleNamespace(
        experts=types.SimpleNamespace(backend=backend), dtype=torch.bfloat16
    )
    assert DeepseekV4MoE.routed_accumulator_dtype.fget(stub) is expected


def test_the_routed_accumulator_dtype_is_resolved_late_not_at_construction():
    """`ConfigurableMoE` only binds `.backend` in `create_weights`.

    Reading the capability during `__init__` answers for the wrapper, which
    never advertises it, and the whole boundary silently reverts to BF16 --- the
    exact defect that made the first attempt at this change a no-op.
    """
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE

    wrapper = types.SimpleNamespace()  # no `.backend` yet
    stub = types.SimpleNamespace(experts=wrapper, dtype=torch.bfloat16)
    assert DeepseekV4MoE.routed_accumulator_dtype.fget(stub) is torch.bfloat16
    wrapper.backend = types.SimpleNamespace(returns_fp32_accumulator=True)
    assert DeepseekV4MoE.routed_accumulator_dtype.fget(stub) is torch.float32


@pytest.mark.parametrize(
    "sm, expected_tp, expected_scale",
    [(90, 1, 0.125), (100, 8, None), (103, 8, None)],
)
def test_the_shared_expert_is_replicated_before_blackwell_only(sm, expected_tp, expected_scale):
    """Replicated matches the reference's own unsharded dense `Expert`.

    Blackwell keeps the sharded layout it ships: this bring-up measures parity
    on Hopper and must not move another architecture's memory footprint.
    """
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl

    stub = types.SimpleNamespace(use_dp=False, mapping=types.SimpleNamespace(tp_size=8))
    with unittest.mock.patch.object(mdl, "get_sm_version", lambda: sm):
        tp_size, scale = mdl.DeepseekV4MoE._compute_shared_expert_tp_size(stub, 2048, 128)
    assert tp_size == expected_tp
    assert scale == expected_scale


def test_attention_dp_still_wins_over_the_replication_choice():
    """DP already replicates the shared expert; its own branch must stay first."""
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl

    stub = types.SimpleNamespace(use_dp=True, mapping=types.SimpleNamespace(tp_size=8))
    with unittest.mock.patch.object(mdl, "get_sm_version", lambda: 90):
        assert mdl.DeepseekV4MoE._compute_shared_expert_tp_size(stub, 2048, 128) == (1, None)


def test_the_replicated_output_scale_is_a_power_of_two():
    """So that summing `tp_size` copies of `S/tp_size` reproduces `S` exactly.

    A non-dyadic scale would round each copy and reintroduce the very rounding
    replication removes.
    """
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl

    for tp_size in (2, 4, 8, 16):
        stub = types.SimpleNamespace(
            use_dp=False, mapping=types.SimpleNamespace(tp_size=tp_size)
        )
        with unittest.mock.patch.object(mdl, "get_sm_version", lambda: 90):
            _, scale = mdl.DeepseekV4MoE._compute_shared_expert_tp_size(stub, 2048, 128)
        assert scale == 1.0 / tp_size
        value = torch.tensor([1.0], dtype=torch.bfloat16) * scale
        assert torch.equal(value.float() * tp_size, torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# The shared expert's dense FP8 block-scale GEMM.
#
# The shared expert is compared against the checkpoint's own kernel on BF16
# tensors, so "within one storage step everywhere" -- where the shipped SM90
# kernel sits -- is not close enough for a limit measured against the tensor's
# RMS. `fp8_blockwise_gemm` reproduces the reference's structure instead: K in
# 128-wide blocks, each block's FP32 dot rescaled before it joins a second
# accumulator, one BF16 rounding at the end.
# ---------------------------------------------------------------------------

from tensorrt_llm._torch.modules.fp8_blockwise_parity_gemm import (  # noqa: E402
    FP8BlockScalesParityLinearMethod,
    fp8_blockwise_gemm,
)


def _blockwise_fp8_weight(n, k, generator, device):
    """An FP8 weight with one power-of-two scale per 128x128 block.

    The per-block magnitudes are deliberately spread over several octaves. A
    plain Gaussian gives every block nearly the same absolute maximum, which
    rounds to the *same* power of two everywhere, and a scale that is constant
    cannot catch a kernel that indexes it wrongly.
    """
    dense = torch.randn(n, k, generator=generator, device=device, dtype=torch.float32)
    view = dense.view(n // 128, 128, k // 128, 128)
    row_block = torch.arange(n // 128, device=device)[:, None]
    col_block = torch.arange(k // 128, device=device)[None, :]
    view = view * torch.exp2(((row_block + 2 * col_block) % 7).float())[:, None, :, None]
    amax = view.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    packed = (view / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return packed.view(n, k).contiguous(), scale.contiguous()


def _blockwise_golden(a_q, a_scale, b_q, b_scale):
    """The reference kernel's structure, written out in plain torch.

    Independent of the Triton kernel under test and of TensorRT-LLM: walk K in
    128-wide blocks, take each block's FP32 dot product, multiply it by
    ``act_scale * weight_scale``, and add it into a separate FP32 accumulator.
    Rounded to BF16 once, because that is what the reference GEMM returns.
    """
    m, k = a_q.shape
    n = b_q.shape[0]
    acc = torch.zeros(m, n, dtype=torch.float32, device=a_q.device)
    for block in range(k // 128):
        cols = slice(block * 128, (block + 1) * 128)
        part = a_q[:, cols].float() @ b_q[:, cols].float().T
        combined = a_scale[:, block, None] * b_scale[None, :, block].repeat_interleave(128, dim=1)
        acc += part * combined
    return acc.to(torch.bfloat16)


@CUDA
@pytest.mark.parametrize("tokens", [1, 33, 512], ids=["decode", "ragged", "prefill"])
def test_the_dense_gemm_reproduces_the_block_scaled_accumulation(tokens):
    """At the shared expert's own shapes, against an independent golden."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20260822)
    hidden = torch.randn(tokens, 2048, generator=generator, device=device, dtype=torch.bfloat16)
    weight, weight_scale = _blockwise_fp8_weight(4096, 2048, generator, device)

    act, act_scale = bsk.quantize_blockwise_ue8m0(hidden)
    got = fp8_blockwise_gemm(act, act_scale, weight, weight_scale)
    golden = _blockwise_golden(act, act_scale, weight, weight_scale)

    rms = golden.float().pow(2).mean().sqrt()
    disagreeing = int((got.float() != golden.float()).sum())
    # Two FP32 accumulation orders over the same 128-value blocks: a handful of
    # elements may land on either side of a BF16 rounding boundary, nothing
    # more. A kernel that applied the scales anywhere else disagrees on a
    # percent of its elements, not on a ten-thousandth of them.
    assert disagreeing / golden.numel() < 1e-4, disagreeing
    assert (got.float() - golden.float()).abs().max() / rms < 0.04


@CUDA
def test_the_dense_gemm_beats_the_shipped_kernel_it_replaces():
    """The discriminating experiment: same inputs, same golden, both kernels.

    The shipped SM90 block-scale GEMM is a correct implementation -- this is not
    a bug report about it -- but it accumulates differently, and against a
    structure-faithful golden that difference is visible on percent of the
    elements. If this test ever stops separating the two, the parity GEMM has
    stopped being the reason the shared expert passes its gate.
    """
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(99)
    hidden = torch.randn(512, 2048, generator=generator, device=device, dtype=torch.bfloat16)
    weight, weight_scale = _blockwise_fp8_weight(4096, 2048, generator, device)

    act, act_scale = bsk.quantize_blockwise_ue8m0(hidden)
    golden = _blockwise_golden(act, act_scale, weight, weight_scale).float()
    ours = fp8_blockwise_gemm(act, act_scale, weight, weight_scale).float()
    shipped = torch.ops.trtllm.fp8_block_scaling_gemm(
        act, weight, act_scale, weight_scale
    ).float()

    ours_off = int((ours != golden).sum()) / golden.numel()
    shipped_off = int((shipped != golden).sum()) / golden.numel()
    assert ours_off < 1e-4, ours_off
    assert shipped_off > 1e-2, shipped_off
    assert shipped_off > 100 * max(ours_off, 1e-9)


@CUDA
def test_the_dense_gemm_reads_the_weight_scale_per_128_rows():
    """A weight-scale row mapping that is off by one block must be visible.

    The reference indexes ``scales_b`` by ``output_row // 128``; using a single
    row for the whole weight, or rolling the mapping, is the class of mistake
    that leaves a kernel plausible and wrong.
    """
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(5)
    hidden = torch.randn(64, 2048, generator=generator, device=device, dtype=torch.bfloat16)
    weight, weight_scale = _blockwise_fp8_weight(512, 2048, generator, device)
    act, act_scale = bsk.quantize_blockwise_ue8m0(hidden)

    correct = fp8_blockwise_gemm(act, act_scale, weight, weight_scale).float()
    rolled = fp8_blockwise_gemm(
        act, act_scale, weight, weight_scale.roll(1, dims=0).contiguous()
    ).float()
    assert not torch.equal(correct, rolled)
    golden = _blockwise_golden(act, act_scale, weight, weight_scale).float()
    assert (correct - golden).abs().max() < (rolled - golden).abs().max()


@CUDA
def test_the_dense_gemm_rejects_scales_that_do_not_describe_its_operands():
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(6)
    hidden = torch.randn(8, 2048, generator=generator, device=device, dtype=torch.bfloat16)
    weight, weight_scale = _blockwise_fp8_weight(256, 2048, generator, device)
    act, act_scale = bsk.quantize_blockwise_ue8m0(hidden)
    with pytest.raises(AssertionError, match="weight scale"):
        fp8_blockwise_gemm(act, act_scale, weight, weight_scale[:, :-1].contiguous())
    with pytest.raises(AssertionError, match="activation scale"):
        fp8_blockwise_gemm(act, act_scale[:, :-1].contiguous(), weight, weight_scale)


# ---------------------------------------------------------------------------
# Who gets the parity GEMM, and who keeps the kernel they ship with.
# ---------------------------------------------------------------------------


def _fp8_block_scale_linear(scale_fmt):
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    quant_config.scale_fmt = scale_fmt
    return Linear(256, 256, bias=False, dtype=torch.bfloat16, quant_config=quant_config)


def test_an_ordinary_fp8_block_scale_linear_keeps_the_shipped_kernel():
    """The control: nothing changes for a layer that did not ask."""
    from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod

    linear = _fp8_block_scale_linear("ue8m0")
    assert type(linear.quant_method) is FP8BlockScalesLinearMethod
    assert not FP8BlockScalesParityLinearMethod.is_enabled(linear)


def test_the_parity_gemm_is_reached_only_by_an_explicit_opt_in():
    linear = _fp8_block_scale_linear("ue8m0")
    linear.use_blockwise_parity_gemm = True
    assert FP8BlockScalesParityLinearMethod.is_enabled(linear)
    assert type(linear.get_quant_method(linear.quant_config)) is FP8BlockScalesParityLinearMethod


def test_a_checkpoint_that_did_not_declare_ue8m0_is_refused_the_parity_gemm():
    """Its activations were quantized against FP32 scales, not powers of two.

    Handing it this kernel would quantize the same tensor a different way and
    call the result parity.
    """
    from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod

    linear = _fp8_block_scale_linear(None)
    linear.use_blockwise_parity_gemm = True
    assert not FP8BlockScalesParityLinearMethod.is_enabled(linear)
    assert type(linear.get_quant_method(linear.quant_config)) is FP8BlockScalesLinearMethod


def _shared_expert_stub():
    return types.SimpleNamespace(
        gate_up_proj=_fp8_block_scale_linear("ue8m0"),
        down_proj=_fp8_block_scale_linear("ue8m0"),
    )


@pytest.mark.parametrize(
    "sm, expected",
    [(90, "FP8BlockScalesParityLinearMethod"), (100, "FP8BlockScalesLinearMethod"),
     (103, "FP8BlockScalesLinearMethod")],
)
def test_the_shared_expert_opts_into_the_parity_gemm_before_blackwell_only(sm, expected):
    """Blackwell's dense path is not measured here, so it keeps its kernel."""
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl

    shared = _shared_expert_stub()
    with unittest.mock.patch.object(mdl, "get_sm_version", lambda: sm):
        mdl._shared_expert_parity_gemm(shared)
    for projection in (shared.gate_up_proj, shared.down_proj):
        assert type(projection.quant_method).__name__ == expected
        assert getattr(projection, "use_blockwise_parity_gemm", False) is (sm < 100)


def test_the_opt_in_reaches_an_already_created_weight():
    """`Linear.__init__` resolves the method when it creates weights.

    Setting the flag afterwards without re-resolving would leave the shipped
    kernel bound and the opt-in silently one construction too late -- the same
    shape of defect as reading a backend capability before the backend exists.
    """
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl

    shared = _shared_expert_stub()
    from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod

    assert type(shared.gate_up_proj.quant_method) is FP8BlockScalesLinearMethod
    with unittest.mock.patch.object(mdl, "get_sm_version", lambda: 90):
        mdl._shared_expert_parity_gemm(shared)
    assert type(shared.gate_up_proj.quant_method) is FP8BlockScalesParityLinearMethod


# ---------------------------------------------------------------------------
# The attention block, which iteration 38's sub-boundary localization named.
#
# Layer 0 is ratio-0 (SWA only, no Compressor, no Indexer). Splitting it at
# every observable boundary put the entry residual, the mHC attention pre-map
# and the attention norm at rel_max_abs 0.0 / cosine 1.0 -- bit-exact -- and
# the attention output first over the threshold at 0.072. Enabling the parity
# GEMM on its projections halved that to 0.036 and halved every downstream
# boundary with it, so the projections are a measured owner rather than a
# guessed one.
# ---------------------------------------------------------------------------


class _AttentionStub(torch.nn.Module):
    """An attention subtree: two FP8 block-scale projections and two that are not.

    The BF16 projection and the FP32-scale one are the controls. A selector
    that swept every ``Linear`` unconditionally would hand them a kernel that
    quantizes their weights a different way and call the result parity.
    """

    def __init__(self):
        super().__init__()
        self.kv_a_proj_with_mqa = _fp8_block_scale_linear("ue8m0")
        self.q_b_proj = _fp8_block_scale_linear("ue8m0")
        self.o_a_proj = _fp8_block_scale_linear(None)
        self.bf16_proj = torch.nn.Linear(8, 8)


def _switch_attention(sm):
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl

    attention = _AttentionStub()
    with unittest.mock.patch.object(mdl, "get_sm_version", lambda: sm):
        switched = mdl._attention_parity_gemm(attention)
    return attention, switched


@pytest.mark.parametrize("sm, expected", [(90, ["kv_a_proj_with_mqa", "q_b_proj"]), (100, [])])
def test_the_attention_projections_opt_in_before_blackwell_only(sm, expected):
    attention, switched = _switch_attention(sm)
    assert sorted(switched) == expected
    for name in ("kv_a_proj_with_mqa", "q_b_proj"):
        projection = getattr(attention, name)
        assert type(projection.quant_method).__name__ == (
            "FP8BlockScalesParityLinearMethod" if sm < 100 else "FP8BlockScalesLinearMethod"
        )


def test_a_projection_the_parity_kernel_cannot_serve_keeps_its_own():
    """And is left with no stray opt-in for a later create_weights to act on."""
    from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod

    attention, switched = _switch_attention(90)
    assert "o_a_proj" not in switched
    assert type(attention.o_a_proj.quant_method) is FP8BlockScalesLinearMethod
    assert getattr(attention.o_a_proj, "use_blockwise_parity_gemm", False) is False
    assert getattr(attention.bf16_proj, "use_blockwise_parity_gemm", False) is False


def test_the_selector_finds_projections_by_resolution_not_by_name():
    """V4's O-LoRA lives under the sparse-attention hooks, not beside o_proj.

    The eight-rank run reports ``o_b_proj`` among the switched projections; a
    hard-coded list written from ``mla.py`` alone would have missed it, and the
    measurement would have been of a partial fix.
    """
    from tensorrt_llm._torch.models import modeling_deepseekv4 as mdl

    attention = _AttentionStub()
    attention.sparse_hooks = torch.nn.Module()
    attention.sparse_hooks.o_b_proj = _fp8_block_scale_linear("ue8m0")
    with unittest.mock.patch.object(mdl, "get_sm_version", lambda: 90):
        switched = mdl._attention_parity_gemm(attention)
    assert "sparse_hooks.o_b_proj" in switched


# ---------------------------------------------------------------------------
# The reference's collective order.
#
#   dist.all_reduce(y)            # routed accumulator only
#   y += self.shared_experts(x)   # one whole replicated shared expert
#
# TensorRT-LLM's default scales the shared expert by 1/tp_size and reduces the
# sum. Algebraically equal; a ring reduction forms 3S/8 on the way, which
# rounds, so the two orders disagree by a rounding the parity gate can see.
# ---------------------------------------------------------------------------


def _moe_order_stub(*, replicated, tp_size=8, allreduce=None):
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE

    stub = types.SimpleNamespace(
        shared_expert_is_replicated=replicated,
        shared_output_scale=(1.0 / tp_size) if replicated else None,
        allreduce=allreduce,
        mapping=types.SimpleNamespace(tp_size=tp_size),
        use_dp=False,
        dtype=torch.bfloat16,
        top_k=6,
    )
    stub.adds_shared_after_the_reduction = (
        DeepseekV4MoE.adds_shared_after_the_reduction.__get__(stub)
    )
    return stub


@pytest.mark.parametrize(
    "replicated, has_allreduce, enable, expected",
    [
        (True, True, True, True),
        (True, True, False, False),  # the reduction is deferred to a fused epilogue
        (True, False, True, False),  # tp_size 1 or attention DP: nothing to reduce
        (False, True, True, False),  # sharded shared expert has no whole output to add
    ],
)
def test_when_the_reference_collective_order_can_be_used(
    replicated, has_allreduce, enable, expected
):
    from tensorrt_llm._torch.distributed import AllReduceParams

    stub = _moe_order_stub(
        replicated=replicated, allreduce=(lambda *a, **k: None) if has_allreduce else None
    )
    assert stub.adds_shared_after_the_reduction(AllReduceParams(enable_allreduce=enable)) is expected
    if has_allreduce:
        # No params at all is the replay path, and it reduces here.
        assert stub.adds_shared_after_the_reduction(None) is (replicated and has_allreduce)


class _RecordingAllReduce:
    def __init__(self):
        self.seen = []

    def __call__(self, tensor, all_reduce_params=None):
        self.seen.append(tensor.clone())
        return tensor * 8.0  # eight ranks holding the same partial


def _run_forward(stub, routed, shared):
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE

    stub.compute_routed_output = lambda *a, **k: routed
    stub.compute_shared_output = DeepseekV4MoE.compute_shared_output.__get__(stub)
    stub.shared_experts = lambda x: shared.clone()
    stub.event_dict = {k: torch.cuda.Event() for k in ("main", "shared")}
    stub.aux_stream = None
    stub.experts = types.SimpleNamespace(backend=types.SimpleNamespace())
    with unittest.mock.patch.object(
        __import__(
            "tensorrt_llm._torch.models.modeling_deepseekv4", fromlist=["EventType"]
        ),
        "EventType",
        types.SimpleNamespace(Main="main", MoeShared="shared"),
    ):
        return DeepseekV4MoE.forward(stub, torch.zeros_like(shared))


@CUDA
def test_the_replicated_shared_expert_is_added_after_the_reduction():
    """The reduction sees the routed accumulator alone, unscaled shared after."""
    reduce = _RecordingAllReduce()
    stub = _moe_order_stub(replicated=True, allreduce=reduce)
    routed = torch.full((4, 8), 2.0, device="cuda", dtype=torch.float32)
    shared = torch.full((4, 8), 1.0, device="cuda", dtype=torch.bfloat16)

    out = _run_forward(stub, routed, shared)

    assert len(reduce.seen) == 1
    assert torch.equal(reduce.seen[0], routed)  # routed only, not routed + shared/8
    assert out.dtype is torch.bfloat16
    assert torch.equal(out.float(), torch.full((4, 8), 2.0 * 8 + 1.0, device="cuda"))


@CUDA
def test_a_sharded_shared_expert_still_travels_through_the_reduction():
    """The protection: every other topology keeps the order it has today."""
    reduce = _RecordingAllReduce()
    stub = _moe_order_stub(replicated=False, allreduce=reduce)
    routed = torch.full((4, 8), 2.0, device="cuda", dtype=torch.float32)
    shared = torch.full((4, 8), 1.0, device="cuda", dtype=torch.bfloat16)

    out = _run_forward(stub, routed, shared)

    assert len(reduce.seen) == 1
    assert torch.equal(reduce.seen[0], routed + 1.0)
    assert torch.equal(out.float(), torch.full((4, 8), (2.0 + 1.0) * 8, device="cuda"))


@CUDA
def test_a_deferred_reduction_still_gets_the_shared_expert_share():
    """`POST_MOE_FUSION` reduces downstream, so the shared expert is scaled."""
    from tensorrt_llm._torch.distributed import AllReduceParams

    reduce = _RecordingAllReduce()
    stub = _moe_order_stub(replicated=True, allreduce=reduce)
    routed = torch.full((4, 8), 2.0, device="cuda", dtype=torch.float32)
    shared = torch.full((4, 8), 1.0, device="cuda", dtype=torch.bfloat16)

    stub.compute_routed_output = lambda *a, **k: routed
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE

    stub.compute_shared_output = DeepseekV4MoE.compute_shared_output.__get__(stub)
    stub.shared_experts = lambda x: shared.clone()
    stub.event_dict = {k: torch.cuda.Event() for k in ("main", "shared")}
    stub.aux_stream = None
    with unittest.mock.patch.object(
        __import__(
            "tensorrt_llm._torch.models.modeling_deepseekv4", fromlist=["EventType"]
        ),
        "EventType",
        types.SimpleNamespace(Main="main", MoeShared="shared"),
    ):
        out = DeepseekV4MoE.forward(
            stub,
            torch.zeros_like(shared),
            final_all_reduce_params=AllReduceParams(enable_allreduce=False),
        )
    # The wrapper still calls the AllReduce object; the object itself honours
    # `enable_allreduce`. What matters here is that the shared expert carried
    # its 1/tp_size share into whatever reduces downstream.
    assert torch.equal(reduce.seen[0], routed + 0.125)
    assert out.dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# The shared expert as its own `real_runtime` claim.
# ---------------------------------------------------------------------------


def _shared_dispatch_moe(method_name, calls):
    projection = types.SimpleNamespace(
        quant_method=type(method_name, (), {})(),
        quant_config=types.SimpleNamespace(scale_fmt="ue8m0"),
        weight=torch.empty(0, dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(0, dtype=torch.float32),
    )
    moe = types.SimpleNamespace(
        shared_experts=types.SimpleNamespace(gate_up_proj=projection, down_proj=projection),
        shared_expert_is_replicated=True,
        shared_output_scale=0.125,
        adds_shared_after_the_reduction=lambda params: True,
    )
    recorder = types.SimpleNamespace(dense_gemm_calls=calls)
    return mr._shared_dispatch_evidence(recorder, moe)


def _dense_call(**overrides):
    call = {
        "rows": 512,
        "k": 2048,
        "n": 4096,
        "act_dtype": "torch.float8_e4m3fn",
        "weight_dtype": "torch.float8_e4m3fn",
        "act_scale_block": 128,
        "weight_scale_block": 128,
        "act_scales_are_powers_of_two": True,
    }
    call.update(overrides)
    return call


def test_the_shared_expert_dispatch_report_names_the_dense_kernel():
    evidence = _shared_dispatch_moe("FP8BlockScalesParityLinearMethod", [_dense_call()])
    assert evidence["passed"], evidence["problems"]
    assert evidence["op_path"] == "fp8_blockwise_parity_gemm.fp8_blockwise_gemm (OpenAI Triton)"
    assert evidence["replicated"] is True
    assert evidence["adds_shared_after_the_reduction"] is True


@pytest.mark.parametrize(
    "method, calls, fragment",
    [
        ("FP8BlockScalesLinearMethod", [_dense_call()], "not the parity GEMM method"),
        ("FP8BlockScalesParityLinearMethod", [], "the dense kernel did not run"),
        (
            "FP8BlockScalesParityLinearMethod",
            [_dense_call(act_dtype="torch.bfloat16")],
            "not FP8 E4M3",
        ),
        (
            "FP8BlockScalesParityLinearMethod",
            [_dense_call(act_scale_block=2048)],
            "one scale per 128 K values",
        ),
        (
            "FP8BlockScalesParityLinearMethod",
            [_dense_call(weight_scale_block=32)],
            "one scale per 128x128 block",
        ),
        (
            "FP8BlockScalesParityLinearMethod",
            [_dense_call(act_scales_are_powers_of_two=False)],
            "not the UE8M0 recipe",
        ),
    ],
)
def test_a_shared_expert_dispatch_that_is_not_the_source_contract_fails(method, calls, fragment):
    evidence = _shared_dispatch_moe(method, calls)
    assert not evidence["passed"]
    assert any(fragment in problem for problem in evidence["problems"]), evidence["problems"]
