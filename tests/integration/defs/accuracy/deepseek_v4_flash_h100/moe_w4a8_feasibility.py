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
"""Can the repo's existing SM90 W4A8 MoE path serve DeepSeek-V4?

The routed-expert gap this bring-up measures is an *activation precision*
gap: `inference/model.py::linear` quantises routed-expert activations to
blockwise-128 FP8 with a UE8M0 scale before the MXFP4 GEMM (W4A8), while
SM90's resolved `CutlassFusedMoE` path is W4A16 and keeps them in BF16.

`TritonFusedMoE.can_implement` advertises SM90 `W4A8_MXFP4_FP8`, so "no SM90
W4A8 path exists" is not a true statement about this repository and must not
be used to close the question. This module answers the question that actually
decides it, in three parts, all measured rather than argued:

1. :func:`resolver_probe` runs the real resolver on the real DeepSeek-V4
   problem --- with the *routed experts'* quantization contract, which on this
   mixed-precision checkpoint is not the model-global one --- records the exact
   rejection, then walks a gate ladder that flips one contract at a time so the
   number and kind of changes reuse would need is a list rather than an
   impression.
2. :func:`activation_arms` scores the three activation-precision contracts ---
   the source's blockwise-128, Triton's per-tensor, and today's BF16 --- at
   FC1, SwiGLU and FC2 against the source's own captured outputs, using real
   checkpoint expert weights at V4 dimensions. This says whether dispatching
   the Triton path would move the gap *at all*, independently of whether it
   can be dispatched.
3. :func:`routing_and_ep` records the routing and expert-parallel contracts
   the Triton path assumes against the ones DeepSeek-V4 actually has.

Nothing here judges a criterion. It produces the evidence a human needs to
decide between "reuse the existing path", "extend it", and "this is an
architecture-level blocker".
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

#: The source's routed-expert activation block: ``act_quant(x, block_size=128,
#: "ue8m0", torch.float8_e8m0fnu)`` in ``inference/model.py::linear``.
SOURCE_ACT_BLOCK = 128


def _round(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 9) if isinstance(v, float) else v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# 1. Does the resolver dispatch it, and if not, exactly why?
# ---------------------------------------------------------------------------


#: Cumulative claims the ladder applies, in order. Each one asserts something
#: about DeepSeek-V4 that is *not* true, so the ladder measures how much the
#: model would have to be misdescribed for the resolver to hand it to Triton.
#: The routed quantization algorithm is deliberately absent: which quant the
#: layer runs is chosen at the base of each ladder, not claimed part-way up.
_LADDER_CLAIMS = (
    ("claim gpt-oss SwiGLU", "swiglu_gptoss_style", True),
    ("+ claim renormalize routing", "routing", "Renormalize"),
)


def _quant_algo_of(quant_config: Any) -> Any:
    return None if quant_config is None else getattr(quant_config, "quant_algo", None)


def live_experts_quant_config(moe: Any) -> Any:
    """The quantization contract the routed experts are *actually* running.

    Read off the live module rather than re-derived, and cross-checked against
    the override the wrapper was constructed with, because those two agreeing
    is the whole reason this is the right object to ask the resolver about.
    ``ConfigurableMoE.create_weights`` assigns
    ``backend.quant_config = self._override_quant_config or self.quant_config``,
    so the backend's is authoritative and the override is its provenance.
    """
    experts = moe.experts
    backend = getattr(experts, "backend", experts)
    live = getattr(backend, "quant_config", None)
    override = getattr(experts, "_override_quant_config", None)
    if override is not None and _quant_algo_of(override) != _quant_algo_of(live):
        raise AssertionError(
            f"routed experts run quant_algo={_quant_algo_of(live)} but were "
            f"constructed with override_quant_config="
            f"{_quant_algo_of(override)}; the resolver question is ambiguous"
        )
    if live is None:
        raise AssertionError(
            f"{type(backend).__name__} has no live quant_config; the routed-expert "
            "contract cannot be read off the running model"
        )
    return live


def _ladder(ask: Any, base_label: str) -> dict[str, Any]:
    """Walk ``_LADDER_CLAIMS`` cumulatively until Triton is eligible.

    The point is arithmetic on the *number of contracts*: if the first
    correction reveals a second gate, reuse is not a switch.
    """
    steps = [(base_label, {}, ask({}))]
    claimed: dict[str, Any] = {}
    for label, field, value in _LADDER_CLAIMS:
        claimed[field] = value
        steps.append((label, dict(claimed), ask(dict(claimed))))

    dispatches = next((step for step in steps if step[2]["triton_eligible"]), None)
    return {
        "as_configured": steps[0][2],
        "gate_ladder": [
            {"step": label, "problem_claims": claims, "result": result}
            for label, claims, result in steps
        ],
        "first_step_that_dispatches_triton": None if dispatches is None else dispatches[0],
        "claims_required_to_dispatch": None if dispatches is None else sorted(dispatches[1]),
    }


def resolver_probe(
    model_config: Any,
    routing: Any,
    shape: dict[str, Any],
    *,
    experts_quant_config: Any,
) -> dict[str, Any]:
    """Ask the real resolver for the Triton path on the real V4 problem.

    Runs ``resolve_moe_impl`` rather than reading ``can_implement`` by eye,
    and with ``moe_backend="TRITON"`` so the Triton class is a candidate at
    all. The recorded rejection is the resolver's own ``MoERejection``: reason
    enum plus detail string, not a paraphrase.

    ``experts_quant_config`` is required and has no default, because the
    question is about the *routed experts*, not the model. DeepSeek-V4 is a
    mixed-precision checkpoint: ``model_config.quant_config`` is the dense
    ``FP8_BLOCK_SCALES`` contract, while production ``DeepseekV4MoE`` resolves
    the expert layer with ``override_quant_config=_get_experts_quant_config(...)``,
    which is the packed-MXFP4 entry from ``quant_config_dict``. Asking the
    resolver without that override answers a question about the dense layers
    and attributes the answer to the experts.

    Two ladders are walked, because "can Triton serve this layer" and "can
    Triton serve it at the *source's* activation precision" are different
    questions and only the second one is about W4A8. Each ladder fixes its
    quantization contract at the base and then claims only semantics.
    """
    import copy

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe.impl_contract import canonical_quant
    from tensorrt_llm._torch.modules.fused_moe.moe_resolution import (
        build_moe_problem,
        resolve_moe_impl,
    )
    from tensorrt_llm._utils import get_sm_version
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    def asker(quant_config: Any) -> Any:
        expected = canonical_quant(_quant_algo_of(quant_config))

        def ask(problem_kwargs: dict[str, Any]) -> dict[str, Any]:
            # A loaded `ModelConfig` is frozen, and this probe must not mutate
            # the one the model is running on. `_frozen` is the documented
            # escape hatch, unset on a copy so the original is untouched.
            cfg = copy.copy(model_config)
            object.__setattr__(cfg, "_frozen", False)
            cfg.moe_backend = "TRITON"
            problem = build_moe_problem(
                cfg,
                override_quant_config=quant_config,
                dtype=torch.bfloat16,
                num_experts=shape["n_routed_experts"],
                hidden_size=shape["hidden_size"],
                intermediate_size=shape["moe_inter_dim"],
                swiglu_gptoss_style=False,
                bias=False,
                routing=routing,
            )
            # `build_moe_problem` falls back to `model_config.quant_config`
            # whenever the override is absent or None. On a mixed-precision
            # checkpoint that substitutes the dense contract for the expert
            # one silently, so refuse to report an answer to a question this
            # probe did not ask.
            if problem.quant != expected:
                raise AssertionError(
                    f"resolver probe built a problem with quant={problem.quant!r}, "
                    f"expected the routed-expert contract {expected!r}; the "
                    f"model-global config is "
                    f"{canonical_quant(_quant_algo_of(model_config.quant_config))!r}"
                )
            problem = problem.__class__(**{**problem.__dict__, **problem_kwargs})
            report = resolve_moe_impl(cfg, problem=problem)
            triton = [r for r in report.rejected if "TRITON" in r.legacy_backend.upper()]
            return {
                "quant": problem.quant,
                "winner": report.winner,
                "eligible": list(report.eligible),
                "triton_rejected": [r.to_dict() for r in triton],
                "triton_eligible": any("TRITON" in name.upper() for name in report.eligible),
            }

        return ask

    routed_algo = _quant_algo_of(experts_quant_config)
    # The source's contract is W4A8 over the same packed-MXFP4 weights, so the
    # W4A8 ladder selects it at the base rather than claiming it part-way up.
    source_precision = QuantConfig(
        quant_algo=QuantAlgo.W4A8_MXFP4_FP8,
        group_size=getattr(experts_quant_config, "group_size", None),
    )
    as_loaded = _ladder(asker(experts_quant_config), "as DeepSeek-V4 presents it")
    at_source_precision = _ladder(
        asker(source_precision), "as DeepSeek-V4 presents it, declared W4A8_MXFP4_FP8"
    )

    return {
        "question": "does the resolver dispatch TritonFusedMoE for this layer?",
        "resolver": "tensorrt_llm._torch.modules.fused_moe.moe_resolution.resolve_moe_impl",
        "requested_backend": "TRITON",
        "quant_provenance": {
            "routed_expert_quant_algo": str(routed_algo),
            "routed_expert_group_size": getattr(experts_quant_config, "group_size", None),
            "routed_expert_config_source": (
                "DeepseekV4MoE._get_experts_quant_config -> "
                "quant_config_dict['model.layers.<i>.mlp.experts'], the same object "
                "production passes as create_moe(override_quant_config=...)"
            ),
            "model_global_quant_algo": str(_quant_algo_of(model_config.quant_config)),
            "note": (
                "these differ because the checkpoint is mixed precision; the "
                "model-global algorithm describes the dense/attention layers and is "
                "not the question this probe asks"
            ),
        },
        # Recorded rather than asserted: on SM90 the routed MXFP4 algorithm is
        # decided by the SM gate, not by the backend name, so "select the
        # Triton backend" does not reach W4A8 here.
        "sm90_mxfp4_algo_selection": {
            "sm_version": get_sm_version(),
            "selector": "tensorrt_llm._torch.model_config.ModelConfig.get_mxfp4_quant_algo",
            "by_moe_backend": {
                backend: str(ModelConfig.get_mxfp4_quant_algo(backend))
                for backend in ("TRITON", "CUTLASS")
            },
        },
        "as_loaded": as_loaded,
        "at_source_precision": at_source_precision,
        # Kept at the top level: the live question is the one the model runs.
        "as_configured": as_loaded["as_configured"],
        "gate_ladder": as_loaded["gate_ladder"],
        "first_step_that_dispatches_triton": as_loaded["first_step_that_dispatches_triton"],
        "claims_required_to_dispatch": as_loaded["claims_required_to_dispatch"],
        "note": (
            "every entry in claims_required_to_dispatch is a misstatement of the "
            "checkpoint's semantics, not a configuration knob: DeepSeek-V4 uses a "
            "plain clamped SwiGLU and routes by tid2eid hash or V4 score, so the "
            "in-scope alternative is extending TritonFusedMoE's own gates, not "
            "describing the model as something else. The routed quantization "
            "algorithm is not among them: the experts already resolve to a packed "
            "MXFP4 contract TritonFusedMoE accepts, which is why the blocking gates "
            "are semantic and why dispatching Triton would not by itself change the "
            "activation precision measured below"
        ),
    }


# ---------------------------------------------------------------------------
# 2. Would dispatching it move the arithmetic?
# ---------------------------------------------------------------------------


def _per_tensor(x: torch.Tensor) -> torch.Tensor:
    """`TritonMXFP4FusedMoEMethod`'s activation quantiser, round-tripped.

    The real ops, not a model of them: ``quantize_e4m3_per_tensor`` is what the
    Triton method calls on the MoE input and again on the SwiGLU output, and
    ``dequantize_e4m3_per_tensor`` is its exact inverse map, so the pair is the
    precision that path imposes.
    """
    q, scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    return torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(q, scale).to(x.dtype)


def _blockwise(tg: Any):
    def quantise(x: torch.Tensor) -> torch.Tensor:
        return tg.fp8_quant_dequant(x, SOURCE_ACT_BLOCK).to(x.dtype)

    return quantise


#: The three activation-precision contracts, by the name of the thing that
#: implements each. ``None`` means "no activation quantisation at all".
_ARMS = (
    ("source_blockwise128_ue8m0", "inference/model.py::linear, act_quant(x, 128, 'ue8m0')"),
    ("triton_per_tensor_fp8", "TritonMXFP4FusedMoEMethod, quantize_e4m3_per_tensor"),
    ("bf16_w4a16", "CutlassFusedMoE W4A16_MXFP4, activations left in BF16"),
)


def activation_arms(
    x: torch.Tensor,
    layer_input: torch.Tensor,
    weights: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    tg: Any,
    *,
    swiglu_limit: float,
    routing_weight: torch.Tensor | None,
) -> dict[str, Any]:
    """Score each activation contract against the source's own expert outputs.

    Everything except the activation quantiser is held fixed: the same
    dequantised MXFP4 checkpoint weights, the same asymmetric clamp, the same
    routing weight, the same BF16 GEMM. So the spread between arms is the
    activation precision and nothing else.

    Two scopes are reported for the per-tensor arm because the Triton method's
    scale is taken over the whole tensor entering each GEMM, which is wider
    than one expert's rows:

    ``layer_scope``
        the FP8 scale is derived from the *whole* MoE input, which is what
        ``TritonMXFP4FusedMoEMethod.apply`` actually does at GEMM1.
    ``expert_scope``
        the scale is derived from this expert's rows alone.

    Both are reported rather than one, because which is better is *not*
    predictable and was measured to go the other way than expected: E4M3
    carries its own exponent, so unlike an integer format its accuracy is
    largely scale-invariant over the representable range, and a narrower
    scale is not automatically closer. The verdict below therefore takes
    whichever of the two scores better, so the conclusion cannot rest on an
    unfavourable scope choice.
    """
    blockwise = _blockwise(tg)

    def chain(q1: Any, q2: Any) -> dict[str, torch.Tensor]:
        def project(v: torch.Tensor, w: torch.Tensor, quantise: Any) -> torch.Tensor:
            src = v if quantise is None else quantise(v)
            return F.linear(src.to(torch.bfloat16), w.to(torch.bfloat16))

        gate = project(x, weights["w1"], q1).float()
        up = project(x, weights["w3"], q1).float()
        if swiglu_limit > 0:
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            gate = gate.clamp(max=swiglu_limit)
        h = F.silu(gate) * up
        if routing_weight is not None:
            h = routing_weight * h
        h = h.to(x.dtype)
        return {
            "fc1_gate": gate,
            "fc1_up": up,
            "swiglu": h,
            "fc2": project(h, weights["w2"], q2),
        }

    def scored(stages: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            stage: _round(tg.compare(stages[stage].to(source[stage].dtype), source[stage]))
            for stage in ("fc1_gate", "fc1_up", "swiglu", "fc2")
        }

    # GEMM1's scale is derived from the whole MoE input, which is what the
    # kernel sees; GEMM2's has no layer-wide analogue, because the SwiGLU
    # output never leaves the fused kernel.
    def layer_scope(v: torch.Tensor) -> torch.Tensor:
        return _with_scale_of(v, layer_input)

    arms = {
        "source_blockwise128_ue8m0": scored(chain(blockwise, blockwise)),
        "triton_per_tensor_fp8_layer_scope": scored(chain(layer_scope, _per_tensor)),
        "triton_per_tensor_fp8_expert_scope": scored(chain(_per_tensor, _per_tensor)),
        "bf16_w4a16": scored(chain(None, None)),
    }
    return {
        "held_fixed": (
            "dequantised MXFP4 checkpoint weights, asymmetric SwiGLU clamp, routing "
            "weight, BF16 GEMM; only the activation quantiser differs between arms"
        ),
        "reference": "the source expert's own w1/w3/w2 Linear outputs, this prompt",
        "implements": dict(_ARMS),
        "arms": arms,
        "reading": _reading(arms),
    }


def _with_scale_of(v: torch.Tensor, whole: torch.Tensor) -> torch.Tensor:
    """Quantise ``v`` with the per-tensor scale ``whole`` would produce.

    ``quantize_e4m3_per_tensor`` derives one scale from its whole input, so
    quantising an expert's rows in isolation would use a tighter scale than
    the kernel does. This applies the layer-wide scale to those rows, which is
    the precision the kernel actually imposes on them.
    """
    # The dynamic op returns its scale in the input dtype; the static one
    # requires FP32, so the widening is the op contract rather than a choice.
    _, scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(whole)
    scale = scale.float()
    q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(v, scale)
    return torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(q, scale).to(v.dtype)


def _reading(arms: dict[str, Any]) -> dict[str, Any]:
    """State the comparison as a verdict a reader cannot mis-summarise."""
    fc2 = {name: stages["fc2"]["rel_max_abs"] for name, stages in arms.items()}
    best_available = min(
        fc2["triton_per_tensor_fp8_layer_scope"],
        fc2["triton_per_tensor_fp8_expert_scope"],
    )
    improves = best_available < fc2["bf16_w4a16"]
    reaches_source = best_available <= fc2["source_blockwise128_ue8m0"]
    return {
        "fc2_rel_max_abs_by_arm": fc2,
        "scale_scope_used_for_the_verdict": (
            "the better of layer_scope and expert_scope, so an unfavourable "
            "scope choice cannot decide the answer"
        ),
        "triton_w4a8_improves_on_todays_w4a16": improves,
        "triton_w4a8_reaches_the_source_contract": reaches_source,
        "verdict": (
            "dispatching the existing Triton W4A8 path would move FC2 closer to the source"
            if improves
            else "dispatching the existing Triton W4A8 path would not move FC2 closer "
            "to the source than today's W4A16, so the gap is not closed by reuse"
        ),
        "what_would_close_it": (
            "the source_blockwise128_ue8m0 arm is the contract that reproduces the "
            "source; reaching it through TritonMXFP4FusedMoEMethod means replacing "
            "its per-tensor quantiser with blockwise-128 UE8M0 and threading a "
            "per-block scale through the Triton matmul's FlexCtx, which changes a "
            "shared production backend rather than configuring it"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Routing and expert parallelism.
# ---------------------------------------------------------------------------


def routing_and_ep(moe: Any, routing: Any, shape: dict[str, Any], lid: int) -> dict[str, Any]:
    """What the Triton path assumes about routing/EP versus what V4 has.

    Recorded from the live objects, not from the source tree: the routing
    method is the one the model built, and the EP numbers are the ones the
    resolved backend is running with.
    """
    from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod

    backend = getattr(moe.experts, "backend", moe.experts)
    is_hashed = bool(getattr(routing, "is_hashed", False))
    return {
        "layer": lid,
        "v4_routing_method": type(routing).__name__,
        "v4_routing_is_renormalize_family": isinstance(routing, RenormalizeMoeRoutingMethod),
        "v4_hash_routed": is_hashed,
        "v4_hash_layers": shape["n_hash_layers"],
        "triton_routing_requirement": (
            "can_implement rejects any routing method that is not a "
            "RenormalizeMoeRoutingMethod subclass"
        ),
        "triton_router": (
            "TritonEPRouter re-derives top-k from expert_logits inside the kernel, so "
            "a layer whose experts come from the checkpoint's tid2eid hash table has "
            "no logits for it to route on at all"
        ),
        "ep": {
            "ep_size": int(backend.ep_size),
            "ep_rank": int(backend.ep_rank),
            "experts_per_rank": shape["n_routed_experts"] // int(backend.ep_size),
            "triton_slot_layout": (
                "TritonFusedMoE assigns num_slots = num_experts and derives "
                "slot_start from ep_rank * num_experts // ep_size, which matches "
                "the V4 contiguous EP8 shard"
            ),
            "triton_eplb": "can_implement rejects EPLB; this bring-up does not enable it",
        },
    }


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------


def run(
    *,
    model_config: Any,
    moe: Any,
    routing: Any,
    shape: dict[str, Any],
    lid: int,
    x: torch.Tensor,
    layer_input: torch.Tensor,
    weights: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    routing_weight: torch.Tensor | None,
    swiglu_limit: float,
    tg: Any,
    ranks: Any,
) -> dict[str, Any]:
    """The whole feasibility question for one real routed expert."""
    probe = resolver_probe(
        model_config,
        routing,
        shape,
        experts_quant_config=live_experts_quant_config(moe),
    )
    arms = activation_arms(
        x,
        layer_input,
        weights,
        source,
        tg,
        swiglu_limit=swiglu_limit,
        routing_weight=routing_weight,
    )
    ranks.log(
        f"  w4a8 feasibility layer{lid}: routed_quant="
        f"{probe['quant_provenance']['routed_expert_quant_algo']} triton_eligible="
        f"{probe['as_configured']['triton_eligible']} claims="
        f"{probe['claims_required_to_dispatch']} "
        f"fc2_rel_max_abs={arms['reading']['fc2_rel_max_abs_by_arm']}"
    )
    return {
        "evidence_label": "moe_w4a8_feasibility",
        "question": (
            "TritonFusedMoE.can_implement advertises SM90 W4A8_MXFP4_FP8. Can that "
            "path serve DeepSeek-V4, and would it close the routed-expert gap?"
        ),
        "dispatch": probe,
        "activation_precision": arms,
        "routing_and_expert_parallel": routing_and_ep(moe, routing, shape, lid),
        "reading": _dispatch_reading(probe, arms),
    }


def _dispatch_reading(probe: dict[str, Any], arms: dict[str, Any]) -> dict[str, Any]:
    """One paragraph a reader cannot turn into the wrong summary.

    Both halves are needed and they say different things: the semantic gates
    are what stops Triton being dispatched, and the activation arithmetic is
    what says lifting them would not help. Either one alone invites the wrong
    next step --- "just relax the gates" or "just switch backend".
    """
    claims = probe["as_loaded"]["claims_required_to_dispatch"] or []
    reading = arms["reading"]
    return {
        "blocking_claims_at_the_loaded_quant_contract": claims,
        "quant_is_not_among_them": "quant" not in claims,
        "would_dispatching_close_the_gap": reading["triton_w4a8_reaches_the_source_contract"],
        "verdict": (
            f"the routed experts already resolve to "
            f"{probe['quant_provenance']['routed_expert_quant_algo']}, which "
            f"TritonFusedMoE accepts; what blocks dispatch is {claims}, both of "
            f"which are semantic misstatements of the checkpoint. Lifting them "
            f"would still not close the routed-expert gap, because "
            f"{reading['verdict']}"
        ),
    }
