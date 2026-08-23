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
"""Real-checkpoint MoE ``source_activation_replay`` on the SM90 Cutlass path.

Stage 3 / Goal 3.2. The official model runs one real prompt; forward hooks
capture the hidden state entering two MoE layers --- one hash-routed
(``layer_idx < num_hash_layers``, experts come from the checkpoint's ``tid2eid``
table) and one score-routed --- together with what the source's own router,
routed experts and shared expert did with it. Those captured activations are
then driven through the *loaded production* ``DeepseekV4MoE``: the same module
Goal 3.1 audited, resolving to ``CutlassFusedMoE`` /
``WFP4A16FusedMoEMethod`` / ``W4A16_MXFP4`` on EP8, calling
``torch.ops.trtllm.fused_moe``.

Four separable claims, because "the outputs are close" conflates them:

*routing*
    Router scores, the exact selected expert set, and the normalised
    ``route_scale``-d weights. The expert set is judged exactly --- a routing
    difference is a different subnetwork, not a tolerance.

*expert arithmetic*
    FC1 (w1 and w3), the asymmetric SwiGLU clamp, and FC2, each against the
    source's own ``Linear`` outputs. The reference for these stages is an
    independent dequantise-and-matmul golden evaluated at the *source's*
    activation precision, so a nibble-order, scale-group, W3/W1-order or clamp
    error shows up here undiluted by the activation-precision difference
    described below.

*kernel*
    The fused op's routed output and the module's combined routed+shared
    output, against the source's. This is the production path end to end.

*clamp*
    A structural check rather than a tolerance: running the same fused op with
    ``swiglu_limit`` removed must change the result, and the limited run must
    be the one that tracks the clamped golden.

One precision asymmetry is inherent and is measured rather than hidden. The
source's routed GEMMs are W4A8: ``inference/model.py::linear`` quantises the
activation to blockwise FP8 (``act_quant``) before ``fp4_gemm``. SM90's only
packed-MXFP4 Cutlass path is W4A16, which keeps the activation in BF16 --- more
precision, not less, but a different number. ``activation_precision_gap``
quantifies it by scoring both implementations against a common exact-arithmetic
golden, so a disagreement at the expert output can be attributed instead of
guessed at.
"""

from __future__ import annotations

import collections
from typing import Any

import moe_w4a8_feasibility
import torch
import torch.nn.functional as F

#: Two FP4 nibbles per byte, one UE8M0 scale per 32 logical K values.
MXFP4_PER_BYTE = 2
MXFP4_GROUP = 32
#: The source quantises routed-expert activations in blocks of this many K.
SOURCE_ACT_BLOCK = 128


# ---------------------------------------------------------------------------
# Recording the production fused-MoE op.
# ---------------------------------------------------------------------------


class FusedMoERecorder:
    """Describe the routed-expert kernels that actually ran.

    ``real_runtime`` for this Goal means a kernel executed at checkpoint
    dimensions with the checkpoint's own tensors --- not that a module resolved
    to a class name. Two production paths can serve packed MXFP4 routed
    experts, and they are recorded separately because they are different
    claims:

    ``fused_moe``
        the Cutlass W4A16 op, reached as an attribute of ``torch.ops.trtllm``,
        which caches overload packets as instance attributes, so replacing the
        attribute intercepts every call site without touching the library.
    ``block_scale``
        the SM90 W4A8 path, whose GEMM and activation quantizer are module
        globals of ``fused_moe_mxfp4_blockscale``. They are intercepted *there*
        rather than at their defining module, because that is the call site the
        backend actually reaches; patching the definition would leave the
        already-imported name bound to the original.

    Recording both and letting the verdict name whichever ran is what keeps
    this honest across a backend change: a run in which neither fired reports
    no dispatch at all rather than inheriting the other path's clean report.

    The *shared* expert is a third, independent claim: it is a dense FP8
    block-scale path, not a routed one, and on SM90 it runs the parity GEMM.
    Its calls are recorded separately so "the routed kernel ran" can never
    stand in for "the dense one did".
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.block_scale_calls: list[dict[str, Any]] = []
        self.act_quant_calls: list[dict[str, Any]] = []
        self.dense_gemm_calls: list[dict[str, Any]] = []
        self._ns = torch.ops.trtllm
        self._orig = self._ns.fused_moe
        from tensorrt_llm._torch.modules import fp8_blockwise_parity_gemm as dense
        from tensorrt_llm._torch.modules.fused_moe import fused_moe_mxfp4_blockscale as bs

        self._bs = bs
        self._orig_gemm = bs.moe_w4a8_gemm
        self._orig_quant = bs.quantize_blockwise_ue8m0
        self._orig_swiglu = bs.swiglu_and_quantize
        self._dense = dense
        self._orig_dense_gemm = dense.fp8_blockwise_gemm

    def __enter__(self) -> FusedMoERecorder:
        orig = self._orig

        def wrapper(x, token_selected_experts, token_final_scales, w3_w1, *a, **kw):
            self.calls.append(
                {
                    "tokens": int(x.shape[0]),
                    "hidden_size": int(x.shape[-1]),
                    "activation_dtype": str(x.dtype),
                    "experts_per_token": int(token_selected_experts.shape[-1]),
                    "weight_dtype": str(w3_w1.dtype),
                    "w3_w1_shape": list(w3_w1.shape),
                    "use_w4_group_scaling": bool(kw.get("use_w4_group_scaling")),
                    "swiglu_limit": (
                        None
                        if kw.get("swiglu_limit") is None
                        else float(kw["swiglu_limit"].flatten()[0])
                    ),
                    "ep_size": kw.get("ep_size"),
                    "ep_rank": kw.get("ep_rank"),
                }
            )
            return orig(x, token_selected_experts, token_final_scales, w3_w1, *a, **kw)

        orig_gemm = self._orig_gemm

        def gemm_wrapper(a_q, a_scale, a_row, b_packed, b_scale, *args, **kw):
            k = b_packed.shape[-1] * 2
            self.block_scale_calls.append(
                {
                    "rows": int(a_row.shape[0]),
                    "k": k,
                    "n": int(b_packed.shape[1]),
                    "act_dtype": str(a_q.dtype),
                    "act_scale_dtype": str(a_scale.dtype),
                    # One activation scale per this many K values.
                    "act_scale_block": k // int(a_scale.shape[-1]),
                    "weight_dtype": str(b_packed.dtype),
                    "weight_scale_dtype": str(b_scale.dtype),
                    # One UE8M0 exponent per this many K values.
                    "weight_group_size": k // int(b_scale.shape[-1]),
                    "experts": int(b_packed.shape[0]),
                    # Powers of two are what "ue8m0" means; a scale that is not
                    # one would be a different quantizer wearing the same name.
                    "act_scales_are_powers_of_two": bool(
                        torch.equal(a_scale, torch.exp2(torch.log2(a_scale).round()))
                    ),
                }
            )
            return orig_gemm(a_q, a_scale, a_row, b_packed, b_scale, *args, **kw)

        orig_quant, orig_swiglu = self._orig_quant, self._orig_swiglu

        def quant_wrapper(x, block_size=128):
            self.act_quant_calls.append({"stage": "fc1_input", "block": int(block_size)})
            return orig_quant(x, block_size)

        def swiglu_wrapper(fc1_out, routing_weight, inter_size, limit, block_size=128, **kw):
            self.act_quant_calls.append(
                {"stage": "fc2_input", "block": int(block_size), "swiglu_limit": limit}
            )
            return orig_swiglu(fc1_out, routing_weight, inter_size, limit, block_size, **kw)

        orig_dense_gemm = self._orig_dense_gemm

        def dense_gemm_wrapper(a, a_scale, b, b_scale, *args, **kw):
            k = int(a.shape[-1])
            self.dense_gemm_calls.append(
                {
                    "rows": int(a.shape[0]),
                    "k": k,
                    "n": int(b.shape[0]),
                    "act_dtype": str(a.dtype),
                    "weight_dtype": str(b.dtype),
                    # One activation scale per this many K values, and one
                    # weight scale per this many rows and K values.
                    "act_scale_block": k // int(a_scale.shape[-1]),
                    "weight_scale_block": k // int(b_scale.shape[-1]),
                    "act_scales_are_powers_of_two": bool(
                        torch.equal(a_scale, torch.exp2(torch.log2(a_scale).round()))
                    ),
                }
            )
            return orig_dense_gemm(a, a_scale, b, b_scale, *args, **kw)

        self._ns.fused_moe = wrapper
        self._bs.moe_w4a8_gemm = gemm_wrapper
        self._bs.quantize_blockwise_ue8m0 = quant_wrapper
        self._bs.swiglu_and_quantize = swiglu_wrapper
        self._dense.fp8_blockwise_gemm = dense_gemm_wrapper
        return self

    def __exit__(self, *exc: Any) -> None:
        self._ns.fused_moe = self._orig
        self._bs.moe_w4a8_gemm = self._orig_gemm
        self._bs.quantize_blockwise_ue8m0 = self._orig_quant
        self._bs.swiglu_and_quantize = self._orig_swiglu
        self._dense.fp8_blockwise_gemm = self._orig_dense_gemm


# ---------------------------------------------------------------------------
# Capturing the source.
# ---------------------------------------------------------------------------


def capture(src: Any, token_ids: list[int], layer_ids: tuple[int, ...], capture_fn: Any) -> dict:
    """One prefill of the real prompt, recording every MoE site on ``layer_ids``.

    Hooks sit on the ``MoE`` block, its ``Gate``, its shared ``Expert`` and, for
    every routed expert this rank owns, on the expert and on its ``w1`` / ``w3``
    / ``w2`` ``Linear`` submodules. The ``Linear`` hooks are what make the FC1 /
    SwiGLU / FC2 stages separable: ``Expert.forward`` computes the clamped
    SwiGLU inline, so ``w2``'s *input* is the post-clamp, post-routing-weight
    tensor and needs no reconstruction.
    """
    store: dict[str, Any] = {}
    handles = []
    for lid in layer_ids:
        ffn = src.model.layers[lid].ffn
        handles.append(capture_fn(ffn, store, f"l{lid}.ffn"))
        handles.append(capture_fn(ffn.gate, store, f"l{lid}.gate"))
        handles.append(capture_fn(ffn.shared_experts, store, f"l{lid}.shared"))
        # The shared expert is the same `Expert` class as a routed one, so its
        # stages separate the same way: `w2`'s input is the post-clamp SwiGLU
        # output at full width, which is what a TP-sharded implementation has to
        # reproduce one slice at a time.
        for proj in ("w1", "w3", "w2"):
            handles.append(
                capture_fn(getattr(ffn.shared_experts, proj), store, f"l{lid}.shared.{proj}")
            )
        for eid in range(ffn.experts_start_idx, ffn.experts_end_idx):
            expert = ffn.experts[eid]
            if expert is None:
                continue
            handles.append(capture_fn(expert, store, f"l{lid}.e{eid}"))
            for proj in ("w1", "w3", "w2"):
                handles.append(capture_fn(getattr(expert, proj), store, f"l{lid}.e{eid}.{proj}"))

    src.reset_cache()
    toks = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        src.model.forward(toks, 0)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    return store


def _source_router_scores(gate: Any, x: torch.Tensor) -> torch.Tensor:
    """The source ``Gate``'s ``original_scores``, from the source's own code.

    ``Gate.forward`` keeps the pre-bias sqrt-softplus scores in a local, so they
    cannot be hooked. These are the two lines that produce them, executed with
    the source module's ``linear`` and the source module's parameter --- not a
    reimplementation. :func:`_routing_checks` proves the reconstruction is the
    real one by re-deriving the Gate's returned weights and indices from it and
    requiring bitwise equality.
    """
    import sys

    linear = sys.modules["model"].linear
    return F.softplus(linear(x.float(), gate.weight.float())).sqrt()


# ---------------------------------------------------------------------------
# Independent goldens on the checkpoint's packed weights.
# ---------------------------------------------------------------------------


#: The config fields this replay reads, spelled the way ``DeepseekV4MoE`` and
#: ``DeepseekV4Gate`` read them. ``PretrainedConfig`` raises on an unknown
#: attribute rather than returning a default, so a stale spelling here is an
#: eight-rank run that dies after the checkpoint is already loaded --- which is
#: why the names live in one place and are covered by a CPU test.
MOE_CONFIG_FIELDS = (
    "n_routed_experts",
    "n_hash_layers",
    "routed_scaling_factor",
    "swiglu_limit",
    "num_experts_per_tok",
)


def moe_config(cfg: Any) -> dict[str, Any]:
    """Read the MoE shape from the model config, failing loudly on a rename."""
    missing = [name for name in MOE_CONFIG_FIELDS if not hasattr(cfg, name)]
    if missing:
        raise AttributeError(
            f"{type(cfg).__name__} has no {missing}; the MoE replay reads the same fields "
            "DeepseekV4MoE does and cannot substitute a default for any of them"
        )
    return {
        "n_routed_experts": int(cfg.n_routed_experts),
        "n_hash_layers": int(cfg.n_hash_layers),
        "routed_scaling_factor": float(cfg.routed_scaling_factor),
        "swiglu_limit": float(cfg.swiglu_limit),
        "num_experts_per_tok": int(cfg.num_experts_per_tok),
    }


def _round(metrics: dict[str, Any]) -> dict[str, Any]:
    """Round floats for the artifact, leaving booleans such as ``finite`` alone."""
    return {k: (round(v, 9) if isinstance(v, float) else v) for k, v in metrics.items()}


def _packed(param: torch.Tensor) -> torch.Tensor:
    """The raw byte container behind an FP4 / E8M0 parameter."""
    return param.detach().view(torch.uint8)


def _expert_weights(expert: Any, tg: Any) -> dict[str, torch.Tensor]:
    """Dequantise one source expert's packed MXFP4 weights to FP32.

    Independent arithmetic --- ``tg.dequant_mxfp4`` decodes the nibbles and
    applies the group-32 UE8M0 exponents itself --- over the *checkpoint's*
    bytes. Goal 3.1 already proved byte equality between those bytes and the
    ones handed to ``WFP4A16FusedMoEMethod``, so a golden built here is a golden
    built on the weights the fused kernel holds.
    """
    return {
        proj: tg.dequant_mxfp4(
            _packed(getattr(expert, proj).weight),
            getattr(expert, proj).scale,
            group=MXFP4_GROUP,
        )
        for proj in ("w1", "w2", "w3")
    }


def _golden_expert(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    tg: Any,
    *,
    swiglu_limit: float,
    routing_weight: torch.Tensor | None,
    quantize_act: bool,
    quantize_fc2_act: bool | None = None,
) -> dict[str, torch.Tensor]:
    """FC1 -> clamped SwiGLU -> FC2, stage by stage, with a stated act precision.

    ``quantize_act=True`` reproduces the source's W4A8 arithmetic (blockwise FP8
    activations); ``False`` is the W4A16 contract SM90's Cutlass path actually
    implements. Both are needed: the first isolates weight/layout/clamp
    correctness from precision, the second says what the production path is
    supposed to compute.

    ``quantize_fc2_act`` splits the two GEMMs apart. Quantising the MoE *input*
    is something a caller can do; quantising the SwiGLU intermediate is not,
    because it never leaves the fused kernel. Setting this to ``False`` while
    ``quantize_act`` is ``True`` therefore measures the exact ceiling of
    "pre-quantise the activation at the module boundary" --- how close that
    approach could get even if it were adopted.
    """
    if quantize_fc2_act is None:
        quantize_fc2_act = quantize_act

    def project(v: torch.Tensor, weight: torch.Tensor, quantize: bool) -> torch.Tensor:
        src = tg.fp8_quant_dequant(v, SOURCE_ACT_BLOCK) if quantize else v
        return F.linear(src.to(torch.bfloat16), weight.to(torch.bfloat16))

    # Kept separately from the clamped values: the source's ``w1``/``w3``
    # ``Linear`` outputs are pre-clamp, and comparing a clamped golden against
    # them would silently pass an implementation that never clamps.
    fc1_gate = project(x, w["w1"], quantize_act).float()
    fc1_up = project(x, w["w3"], quantize_act).float()
    gate, up = fc1_gate, fc1_up
    if swiglu_limit > 0:
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    h = F.silu(gate) * up
    if routing_weight is not None:
        h = routing_weight * h
    h = h.to(x.dtype)
    return {
        "fc1_gate": fc1_gate,
        "fc1_up": fc1_up,
        "swiglu": h,
        "fc2": project(h, w["w2"], quantize_fc2_act),
    }


# ---------------------------------------------------------------------------
# Checks.
# ---------------------------------------------------------------------------


class _Recorder:
    """Accumulate judged checks in the shape ``_aggregate_ranks`` consumes."""

    def __init__(self, tol: dict, tg: Any, judge: Any, tolerance: Any, ulp_report: Any, ranks: Any):
        self.out: dict[str, Any] = {}
        self.tol, self.tg = tol, tg
        self.judge, self.tolerance, self.ulp_report = judge, tolerance, ulp_report
        self.ranks = ranks

    def record(
        self,
        name: str,
        module: str,
        got: torch.Tensor,
        ref: torch.Tensor,
        context: dict[str, Any],
        grid: torch.dtype | None = None,
    ) -> dict[str, float]:
        """Judge one check against its pre-registered tolerance.

        ``grid`` names the dtype the storage-step diagnostic should be read in.
        The MoE quantities are BF16-valued even where the comparison itself is
        carried out in FP32, and measuring their grid distance in FP32 steps
        would report a number about the harness rather than the tensors.
        """
        metrics = self.tg.compare(got, ref)
        limits = self.tolerance(self.tol, module)
        storage = self.ulp_report(
            got if grid is None else got.to(grid), ref if grid is None else ref.to(grid)
        )
        passed, problems = self.judge(metrics, limits, storage)
        self.out[name] = {
            "module": module,
            "metrics": {
                k: (round(v, 9) if isinstance(v, float) else v) for k, v in metrics.items()
            },
            "storage_resolution": storage,
            "tolerance": limits,
            "passed": passed,
            "problems": problems,
            "context": context,
        }
        self.ranks.log(
            f"  moe {name:38s} cos={metrics['cosine']:.6f} "
            f"rel_max_abs={metrics['rel_max_abs']:.3e} "
            f"{'PASS' if passed else 'FAIL ' + str(problems)}"
        )
        return metrics

    def exact(self, name: str, module: str, passed: bool, detail: dict[str, Any]) -> None:
        """An exact rule --- selected experts, structural clamp evidence.

        The rule is the verdict; there is no tolerance to apply, and the
        auditor skips ``rule="exact"`` entries in its float re-judging pass.
        The scalar parts of ``detail`` are still copied into ``metrics``
        because that is what crosses the rank gather: without it, a rule that
        fails on rank 7 alone reaches the artifact as a bare sentence and
        diagnosing it costs another eight-rank run.
        """
        self.out[name] = {
            "module": module,
            "rule": "exact",
            "tolerance": self.tolerance(self.tol, module)
            if module in self.tol["modules"]
            else None,
            "metrics": {
                k: float(v)
                for k, v in detail.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "layer"
            },
            "passed": bool(passed),
            "problems": [] if passed else [detail.get("problem", "exact rule violated")],
            "context": detail,
        }
        self.ranks.log(f"  moe {name:38s} exact {'PASS' if passed else 'FAIL ' + str(detail)}")


def _align_by_expert(
    indices: torch.Tensor, weights: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Scatter ``(indices, weights)`` into a dense per-expert row.

    Top-k order is an implementation detail --- two correct routers can list the
    same six experts in a different order --- so weights are compared in expert
    space rather than slot space. A dense row also makes a *missing* expert show
    up as a zero rather than silently shifting every later comparison.
    """
    dense = torch.zeros(indices.shape[0], num_experts, dtype=torch.float32, device=weights.device)
    dense.scatter_(1, indices.long(), weights.float())
    return dense


def _routing_checks(
    rec: _Recorder,
    stem: str,
    ctx: dict[str, Any],
    gate: Any,
    hidden: torch.Tensor,
    src_gate_out: tuple[torch.Tensor, torch.Tensor],
    trtllm_logits: torch.Tensor,
    trtllm_indices: torch.Tensor,
    trtllm_weights: torch.Tensor,
    num_experts: int,
    route_scale: float,
) -> dict[str, Any]:
    """Router scores, exact expert sets, and normalised routing weights."""
    src_weights, src_indices = src_gate_out
    src_scores = _source_router_scores(gate, hidden)

    # Prove the reconstruction *is* the Gate's own `original_scores` by
    # re-deriving what the Gate returned from it. Anything short of bitwise
    # equality would mean the score tensor being judged is not the one the
    # source selected and weighted with.
    replay_weights = src_scores.gather(1, src_indices.long())
    replay_weights = replay_weights / replay_weights.sum(dim=-1, keepdim=True) * route_scale
    reconstruct_exact = bool(torch.equal(replay_weights, src_weights))

    trtllm_scores = F.softplus(trtllm_logits.float()).sqrt()
    rec.record(
        f"{stem}.router_scores",
        "moe_router_logits",
        trtllm_scores,
        src_scores,
        {
            **ctx,
            "got": "softplus(dsv3_router_gemm_op(hidden, gate.weight)).sqrt()",
            "ref": "source Gate arithmetic (model.linear + sqrt-softplus) on the same hidden state",
            "source_scores_reconstruct_bitwise_exact": reconstruct_exact,
        },
    )

    src_sets = [set(row.tolist()) for row in src_indices]
    got_sets = [set(row.tolist()) for row in trtllm_indices.long()]
    mismatched = [t for t, (a, b) in enumerate(zip(got_sets, src_sets)) if a != b]
    positional = int((trtllm_indices.long() == src_indices.long()).all(dim=-1).sum())
    rec.exact(
        f"{stem}.expert_ids",
        "moe_expert_ids",
        not mismatched,
        {
            **ctx,
            "tokens": len(src_sets),
            "tokens_with_identical_expert_set": len(src_sets) - len(mismatched),
            "tokens_with_identical_slot_order": positional,
            "first_mismatched_tokens": mismatched[:8],
            "source_first_token": sorted(src_sets[0]) if src_sets else [],
            "trtllm_first_token": sorted(got_sets[0]) if got_sets else [],
            "problem": (
                None
                if not mismatched
                else f"{len(mismatched)} of {len(src_sets)} tokens selected a different expert set"
            ),
        },
    )

    rec.record(
        f"{stem}.routing_weights",
        "moe_routing_weights",
        _align_by_expert(trtllm_indices, trtllm_weights, num_experts),
        _align_by_expert(src_indices, src_weights, num_experts),
        {
            **ctx,
            "compared_in": "expert space (dense scatter), so top-k slot order cannot mask a value",
            "route_scale": route_scale,
        },
    )
    return {
        "source_scores_reconstruct_bitwise_exact": reconstruct_exact,
        "tokens_with_identical_expert_set": len(src_sets) - len(mismatched),
        "tokens": len(src_sets),
    }


def _expert_stage_checks(
    rec: _Recorder,
    stem: str,
    ctx: dict[str, Any],
    store: dict[str, Any],
    lid: int,
    eid: int,
    expert: Any,
    tg: Any,
    swiglu_limit: float,
) -> dict[str, Any]:
    """FC1 / clamped SwiGLU / FC2 for one real routed expert.

    The golden runs at the *source's* activation precision so that this stage
    isolates the packed-weight contract --- nibble order, group-32 UE8M0 scale,
    W1 vs W3 identity, the asymmetric clamp --- from the W4A16-vs-W4A8
    activation difference, which is measured separately.
    """
    w1_cap = store[f"l{lid}.e{eid}.w1"]
    w3_cap = store[f"l{lid}.e{eid}.w3"]
    w2_cap = store[f"l{lid}.e{eid}.w2"]
    expert_cap = store[f"l{lid}.e{eid}"]
    x = w1_cap["inputs"][0]
    routing_weight = expert_cap["inputs"][1] if len(expert_cap["inputs"]) > 1 else None

    weights = _expert_weights(expert, tg)
    golden = _golden_expert(
        x,
        weights,
        tg,
        swiglu_limit=swiglu_limit,
        routing_weight=routing_weight,
        quantize_act=True,
    )
    shared_ctx = {
        **ctx,
        "expert": eid,
        "tokens_routed_here": int(x.shape[0]),
        "got": "independent dequantise(MXFP4 nibbles, group-32 UE8M0) + matmul, "
        "source activation precision (blockwise FP8)",
        "ref": "the source expert's own Linear output",
        "swiglu_limit": swiglu_limit,
    }
    for name, ref in (
        ("expert_fc1_gate", w1_cap["output"]),
        ("expert_fc1_up", w3_cap["output"]),
        ("expert_swiglu_clamped", w2_cap["inputs"][0]),
        ("expert_fc2", w2_cap["output"]),
    ):
        key = {"expert_fc1_gate": "fc1_gate", "expert_fc1_up": "fc1_up"}.get(
            name, {"expert_swiglu_clamped": "swiglu"}.get(name, "fc2")
        )
        rec.record(
            f"{stem}.{name}", "moe_expert_output", golden[key].to(ref.dtype), ref, shared_ctx
        )

    clamped = int(
        (w1_cap["output"].float() > swiglu_limit).sum()
        + (w3_cap["output"].float().abs() > swiglu_limit).sum()
    )
    return {
        "expert": eid,
        "tokens_routed_here": int(x.shape[0]),
        "source_elements_at_or_beyond_clamp": clamped,
        "fc1_gate_max": float(w1_cap["output"].float().max()),
        "fc1_up_absmax": float(w3_cap["output"].float().abs().max()),
    }


def _precision_gap(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    reference_fc2: torch.Tensor,
    routing_weight: torch.Tensor | None,
    tg: Any,
    swiglu_limit: float,
) -> dict[str, Any]:
    """How far each implementation sits from a common exact-arithmetic golden.

    The source's routed GEMMs quantise their activations to FP8; SM90's W4A16
    Cutlass path does not. Scoring both against the same FP32-weight,
    BF16-activation evaluation says which of the two is nearer the arithmetic
    both are approximating --- so a difference at the expert output can be
    attributed to the reference's own quantisation instead of being read as an
    implementation error.
    """
    exact = _golden_expert(
        x,
        weights,
        tg,
        swiglu_limit=swiglu_limit,
        routing_weight=routing_weight,
        quantize_act=False,
    )["fc2"]
    return {
        "exact_reference": "dequantised MXFP4 weights, BF16 activations, no activation quantisation",
        "source_fp8_activation_vs_exact": _round(tg.compare(reference_fc2, exact)),
        "why": (
            "inference/model.py::linear act_quants routed-expert activations to blockwise FP8 "
            "before fp4_gemm (W4A8); SM90's only packed-MXFP4 Cutlass path is W4A16 and keeps "
            "them in BF16. The two therefore differ by the source's activation quantisation "
            "even when both are correct."
        ),
    }


# ---------------------------------------------------------------------------
# Routed-output reconstruction.
# ---------------------------------------------------------------------------


def _fired(src_indices: torch.Tensor, eid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``MoE.forward``'s own ``idx, top = torch.where(indices == i)``."""
    return torch.where(src_indices == eid)


def _source_routed(store: dict, lid: int, src_indices: torch.Tensor, shape, local_ids) -> Any:
    """Rebuild the source's pre-shared, pre-all-reduce routed accumulator.

    ``MoE.forward`` keeps it in a local, so it is rebuilt from the per-expert
    outputs the hooks captured, scattered with the source's own indexing
    statement --- including its ``y[idx] +=`` semantics, which assign rather
    than accumulate if a token ever selected the same expert twice.
    """
    y = torch.zeros(shape, dtype=torch.float32, device=src_indices.device)
    fired = []
    for eid in local_ids:
        key = f"l{lid}.e{eid}"
        if key not in store:
            continue
        idx, _ = _fired(src_indices, eid)
        y[idx] += store[key]["output"].float()
        fired.append(eid)
    return y, fired


def _golden_routed(
    hidden: torch.Tensor,
    ffn: Any,
    src_indices: torch.Tensor,
    src_weights: torch.Tensor,
    local_ids,
    tg: Any,
    swiglu_limit: float,
    *,
    quantize_act: bool,
    quantize_fc2_act: bool | None = None,
) -> torch.Tensor:
    """The routed accumulator an independent implementation would produce.

    Weights are dequantised one expert at a time rather than cached: the full
    local shard expanded to FP32 is over 3 GB, and nothing here needs two
    experts at once.
    """
    y = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    for eid in local_ids:
        expert = ffn.experts[eid]
        if expert is None:
            continue
        idx, top = _fired(src_indices, eid)
        if idx.numel() == 0:
            continue
        weights = _expert_weights(expert, tg)
        out = _golden_expert(
            hidden[idx],
            weights,
            tg,
            swiglu_limit=swiglu_limit,
            routing_weight=src_weights[idx, top, None].float(),
            quantize_act=quantize_act,
            quantize_fc2_act=quantize_fc2_act,
        )["fc2"]
        y[idx] += out.float()
        del weights, out
    return y


def _boundary_prequantisation_experiment(
    rec: _Recorder,
    moe: Any,
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    ids: torch.Tensor,
    ffn: Any,
    src_indices: torch.Tensor,
    src_weights: torch.Tensor,
    src_routed: torch.Tensor,
    local_ids,
    tg: Any,
    swiglu_limit: float,
    ranks: Any,
) -> dict[str, Any]:
    """Can the routed gate be met by quantising the MoE input at the boundary?

    The source's routed GEMMs are W4A8 and SM90's only packed-MXFP4 Cutlass
    path is W4A16, so the two disagree by the source's own activation
    quantisation. The one thing a caller *can* do about that without a new
    kernel is quantise the activation before handing it over, which makes FC1
    the source's arithmetic exactly. FC2's operand is the SwiGLU intermediate,
    which never leaves the fused kernel, so it stays BF16 either way.

    This runs that approach for real -- the same production fused op, the same
    routing, a blockwise-FP8 round-tripped input -- and also computes the
    golden that quantises FC1 only, which is the exact ceiling of the approach.
    Recorded rather than adopted: whether it is worth doing depends on whether
    it reaches the registered limit, and that is what this measures.
    """
    prequantised = tg.fp8_quant_dequant(hidden, SOURCE_ACT_BLOCK).to(hidden.dtype)
    # The router keeps the unquantised hidden state: the source's ``Gate`` runs
    # on ``x.float()`` with no ``act_quant``, so quantising its input too would
    # change routing and confound the experiment.
    routed = _all_reduce(
        moe.experts(
            prequantised,
            router_logits,
            input_ids=ids,
            do_finalize=True,
            output_dtype=hidden.dtype,
            all_rank_num_tokens=None,
            use_dp_padding=False,
        ).float(),
        ranks,
    )
    ceiling = _all_reduce(
        _golden_routed(
            hidden,
            ffn,
            src_indices,
            src_weights,
            local_ids,
            tg,
            swiglu_limit,
            quantize_act=True,
            quantize_fc2_act=False,
        ),
        ranks,
    )
    limits = rec.tolerance(rec.tol, "moe_expert_output")
    measured = tg.compare(routed, src_routed)
    return {
        "approach": "blockwise-FP8 round trip of the MoE input, then the production "
        "W4A16 fused op; router input unchanged",
        "measured_vs_source": _round(measured),
        "fc1_only_golden_vs_source": _round(tg.compare(ceiling, src_routed)),
        "registered_limits": limits,
        "meets_registered_limits": bool(
            measured["rel_max_abs"] <= limits.get("rel_max_abs_max", float("inf"))
            and measured["cosine"] >= limits.get("cosine_min", -1.0)
        ),
        "reading": (
            "fc1_only_golden_vs_source is the ceiling: it is what an exact "
            "implementation of this approach would score, with only the "
            "unreachable FC2 activation left unquantised. If the ceiling itself "
            "misses the registered limit, the approach cannot close the gate."
        ),
    }


def _all_reduce(t: torch.Tensor, ranks: Any) -> torch.Tensor:
    """Sum a rank-local partial across the expert/tensor-parallel world."""
    import torch.distributed as dist

    out = t.clone()
    if ranks.world > 1 and dist.is_initialized():
        dist.all_reduce(out)
    return out


def _thaw(obj: Any) -> Any:
    """Copy captured tensors out of inference mode.

    ``capture`` runs the source under ``torch.inference_mode``, so everything
    the hooks cloned is an inference tensor. Cloning again *outside* that mode
    yields ordinary tensors, which is what the TensorRT-LLM modules and the
    goldens downstream expect.
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_thaw(o) for o in obj)
    return obj


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run(
    args: Any,
    ranks: Any,
    loaded: Any,
    src: Any,
    prompt: dict[str, Any],
    tol: dict[str, Any],
    tg: Any,
    judge: Any,
    tolerance: Any,
    ulp_report: Any,
    capture_fn: Any,
) -> dict[str, Any]:
    """Replay one hash-routed and one score-routed MoE layer on the SM90 path."""
    model = loaded.model
    shape = moe_config(loaded.model_config.pretrained_config)
    num_experts = shape["n_routed_experts"]
    route_scale = shape["routed_scaling_factor"]
    swiglu_limit = shape["swiglu_limit"]
    hash_layers = shape["n_hash_layers"]
    token_ids = list(prompt["token_ids"])

    # One hash-routed layer and one score-routed layer: the two routing variants
    # in the checkpoint's inventory. They are different subnetwork selectors,
    # not two instances of one.
    layers = {"hash": 0, "score": hash_layers}
    ranks.log(f"[moe] source prefill of {len(token_ids)} tokens, capturing layers {layers}")
    store = _thaw(capture(src, token_ids, tuple(layers.values()), capture_fn))

    rec = _Recorder(tol, tg, judge, tolerance, ulp_report, ranks)
    per_layer: dict[str, Any] = {}
    recorder = FusedMoERecorder()
    with recorder, torch.no_grad():
        for kind, lid in layers.items():
            per_layer[f"layer{lid}"] = _replay_layer(
                rec,
                ranks,
                model,
                src,
                store,
                lid,
                kind,
                tg,
                num_experts=num_experts,
                hash_layers=hash_layers,
                route_scale=route_scale,
                swiglu_limit=swiglu_limit,
                prompt_id=prompt["id"],
            )

    backend = _resolved_backend(model, layers["score"])
    dispatch = _dispatch_evidence(recorder, backend)
    dispatch["shared_expert"] = _shared_dispatch_evidence(
        recorder, model.model.layers[layers["score"]].mlp
    )
    dispatch["problems"] = dispatch["problems"] + [
        f"shared expert: {p}" for p in dispatch["shared_expert"]["problems"]
    ]
    dispatch["passed"] = not dispatch["problems"]
    ranks.log(f"  moe real_runtime {dispatch}")

    return {
        "module_goldens": rec.out,
        "moe_real_runtime": dispatch,
        "moe_layers": per_layer,
        "local_passed": bool(rec.out)
        and all(c["passed"] for c in rec.out.values())
        and dispatch["passed"],
    }


def _shared_dispatch_evidence(recorder: FusedMoERecorder, moe: Any) -> dict[str, Any]:
    """`real_runtime` for the dense shared expert, as a claim of its own.

    The shared expert is a separate owner from the routed experts: separate
    weights, a separate quantization contract (dense FP8 block-scale, not
    packed MXFP4) and, on SM90, a separate kernel. Reporting it here means a
    reader cannot mistake "the routed W4A8 kernel ran" for "the dense one did",
    and a silent revert to the shipped GEMM --- which is within a BF16 step of
    the reference but not within the registered limit --- fails loudly instead
    of showing up as a metric someone has to interpret.
    """
    gate_up, down = moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj
    calls = recorder.dense_gemm_calls
    evidence: dict[str, Any] = {
        "quant_method": type(getattr(gate_up, "quant_method", None)).__name__,
        "scale_fmt": getattr(getattr(gate_up, "quant_config", None), "scale_fmt", None),
        "weight_dtypes": sorted({str(gate_up.weight.dtype), str(down.weight.dtype)}),
        "weight_scale_dtypes": sorted(
            {str(gate_up.weight_scale.dtype), str(down.weight_scale.dtype)}
        ),
        "replicated": bool(moe.shared_expert_is_replicated),
        "shared_output_scale": moe.shared_output_scale,
        "adds_shared_after_the_reduction": bool(moe.adds_shared_after_the_reduction(None)),
        "op_path": "fp8_blockwise_parity_gemm.fp8_blockwise_gemm (OpenAI Triton)",
        "gemm_calls": len(calls),
        "gemm_k": sorted({c["k"] for c in calls}),
        "gemm_n": sorted({c["n"] for c in calls}),
        "activation_dtypes": sorted({c["act_dtype"] for c in calls}),
        "activation_scale_blocks": sorted({c["act_scale_block"] for c in calls}),
        "weight_scale_blocks": sorted({c["weight_scale_block"] for c in calls}),
        "activation_scales_are_powers_of_two": sorted(
            {c["act_scales_are_powers_of_two"] for c in calls}
        ),
    }
    problems: list[str] = []
    if evidence["quant_method"] != "FP8BlockScalesParityLinearMethod":
        problems.append(
            f"the shared expert resolved {evidence['quant_method']}, not the parity GEMM method"
        )
    if not calls:
        problems.append("no fp8_blockwise_gemm call was observed; the dense kernel did not run")
    else:
        if evidence["activation_dtypes"] != ["torch.float8_e4m3fn"]:
            problems.append(
                f"dense activations reached the kernel as {evidence['activation_dtypes']}, "
                "not FP8 E4M3"
            )
        if evidence["activation_scale_blocks"] != [128]:
            problems.append(
                f"dense activation scale granularity is {evidence['activation_scale_blocks']}, "
                "not one scale per 128 K values"
            )
        if evidence["weight_scale_blocks"] != [128]:
            problems.append(
                f"dense weight scale granularity is {evidence['weight_scale_blocks']}, "
                "not one scale per 128x128 block"
            )
        if evidence["activation_scales_are_powers_of_two"] != [True]:
            problems.append(
                "dense activation scales are not powers of two; this is not the UE8M0 recipe"
            )
    if evidence["weight_dtypes"] != ["torch.float8_e4m3fn"]:
        problems.append(f"shared weights are {evidence['weight_dtypes']}, not FP8 E4M3")
    evidence["problems"] = problems
    evidence["passed"] = not problems
    return evidence


def _routed_without_the_output_cast(backend: Any, hidden: Any, indices: Any, weights: Any) -> Any:
    """The same kernel, with the routed accumulator left in FP32.

    A diagnostic, not a gate. The source keeps its routed accumulator in FP32
    from the first expert until after the shared expert is added; a TensorRT-LLM
    MoE returns the model's activation dtype, so the production path rounds that
    accumulator to BF16 one step earlier. Running the identical kernel with
    ``output_dtype=torch.float32`` separates "the arithmetic still differs" from
    "the interface rounds once more than the source does", which are different
    problems with different fixes.

    Returns ``None`` for a backend whose ``run_moe`` this harness does not know
    how to call directly, so the diagnostic is absent rather than invented.
    """
    if type(backend).__name__ != "BlockScaleMXFP4FusedMoE":
        return None
    from tensorrt_llm._torch.modules.fused_moe.impl_contract import MoERunContext

    return backend.run_moe(
        MoERunContext(
            token_selected_experts=indices,
            token_final_scales=weights,
            x=hidden,
            x_sf=None,
            output_dtype=torch.float32,
        )
    )


def _resolved_backend(model: Any, lid: int) -> Any:
    """The routed-expert implementation object this rank is running."""
    experts = model.model.layers[lid].mlp.experts
    return getattr(experts, "backend", experts)


#: What each production routed-expert path has to be able to say about itself.
#: Keyed by the implementation class the resolver picked, so a backend change
#: cannot inherit the previous one's clean dispatch report: an unknown class
#: has no entry and fails.
_DISPATCH_CONTRACTS = {
    "CutlassFusedMoE": {
        "op_path": "torch.ops.trtllm.fused_moe",
        "quantization": "W4A16_MXFP4, BF16 activations",
    },
    "BlockScaleMXFP4FusedMoE": {
        "op_path": (
            "tensorrt_llm._torch.modules.fused_moe.mxfp4_blockscale_kernels."
            "moe_w4a8_gemm (OpenAI Triton)"
        ),
        "quantization": "W4A8_MXFP4_FP8, blockwise-128 UE8M0 FP8 activations",
    },
}


def _dispatch_evidence(recorder: FusedMoERecorder, backend: Any) -> dict[str, Any]:
    """`real_runtime` for whichever routed-expert path the resolver selected.

    The verdict is keyed off the live backend class rather than off whichever
    recorder happened to see traffic: a run where the selected backend's kernel
    never fired has to fail, and it would silently pass if the report were
    assembled from "the calls we did see".
    """
    name = type(backend).__name__
    contract = _DISPATCH_CONTRACTS.get(name)
    dispatch: dict[str, Any] = {
        "backend": name,
        "weight_method": type(getattr(backend, "quant_method", None)).__name__,
        "quant_algo": str(getattr(getattr(backend, "quant_config", None), "quant_algo", None)),
        "scale_fmt": getattr(getattr(backend, "quant_config", None), "scale_fmt", None),
        "op_path": None if contract is None else contract["op_path"],
        "quantization": None if contract is None else contract["quantization"],
        "ep": [int(getattr(backend, "ep_size", -1)), int(getattr(backend, "ep_rank", -1))],
        "experts_per_rank": int(getattr(backend, "expert_size_per_partition", -1)),
        "fused_moe_calls": len(recorder.calls),
        "block_scale_gemm_calls": len(recorder.block_scale_calls),
    }
    problems: list[str] = []
    if contract is None:
        problems.append(f"no dispatch contract is registered for backend {name}")

    if name == "CutlassFusedMoE":
        ops = recorder.calls
        dispatch.update(
            {
                "activation_dtypes": sorted({c["activation_dtype"] for c in ops}),
                "weight_dtypes": sorted({c["weight_dtype"] for c in ops}),
                "use_w4_group_scaling": sorted({c["use_w4_group_scaling"] for c in ops}),
                "swiglu_limits_seen": sorted(
                    {c["swiglu_limit"] for c in ops}, key=lambda v: (v is None, v)
                ),
                "tokens": sorted({c["tokens"] for c in ops}),
                "hidden_size": sorted({c["hidden_size"] for c in ops}),
                "experts_per_token": sorted({c["experts_per_token"] for c in ops}),
                "w3_w1_shapes": sorted({tuple(c["w3_w1_shape"]) for c in ops}),
            }
        )
        if not ops:
            problems.append(
                "no torch.ops.trtllm.fused_moe call was observed; the kernel did not run"
            )
        if dispatch["activation_dtypes"] != ["torch.bfloat16"]:
            problems.append(f"unexpected activation dtype {dispatch['activation_dtypes']}")
        if dispatch["weight_dtypes"] != ["torch.uint8"]:
            problems.append(
                f"routed weights reached the op as {dispatch['weight_dtypes']}, not packed"
            )
        if dispatch["use_w4_group_scaling"] != [True]:
            problems.append("the op ran without W4 group scaling; this is not the MXFP4 path")

    elif name == "BlockScaleMXFP4FusedMoE":
        gemms = recorder.block_scale_calls
        quants = recorder.act_quant_calls
        dispatch.update(
            {
                "weight_dtypes": sorted({c["weight_dtype"] for c in gemms}),
                "weight_scale_dtypes": sorted({c["weight_scale_dtype"] for c in gemms}),
                "weight_group_sizes": sorted({c["weight_group_size"] for c in gemms}),
                "activation_dtypes": sorted({c["act_dtype"] for c in gemms}),
                "activation_scale_blocks": sorted({c["act_scale_block"] for c in gemms}),
                "activation_scales_are_powers_of_two": sorted(
                    {c["act_scales_are_powers_of_two"] for c in gemms}
                ),
                "gemm_k": sorted({c["k"] for c in gemms}),
                "gemm_n": sorted({c["n"] for c in gemms}),
                "experts_in_weight": sorted({c["experts"] for c in gemms}),
                "quantised_stages": sorted({c["stage"] for c in quants}),
                "swiglu_limits_seen": sorted(
                    {c.get("swiglu_limit") for c in quants if c["stage"] == "fc2_input"},
                    key=lambda v: (v is None, v),
                ),
                "swiglu_limits_note": (
                    "a None here is the clamp-liveness control, which deliberately "
                    "runs one pass with the limit removed to prove it reaches the "
                    "kernel; the configured value is recorded in swiglu_clamp"
                ),
            }
        )
        if not gemms:
            problems.append(
                "no moe_w4a8_gemm call was observed; the block-scale kernel did not run"
            )
        if dispatch["weight_dtypes"] != ["torch.uint8"]:
            problems.append(
                f"routed weights reached the kernel as {dispatch['weight_dtypes']}, not packed"
            )
        if dispatch["weight_scale_dtypes"] != ["torch.uint8"]:
            problems.append(
                f"weight scales reached the kernel as {dispatch['weight_scale_dtypes']}, "
                "not the checkpoint's UE8M0 exponent bytes"
            )
        if dispatch["weight_group_sizes"] != [32]:
            problems.append(
                f"weight scale granularity is {dispatch['weight_group_sizes']}, not one "
                "UE8M0 exponent per 32 K values"
            )
        if dispatch["activation_dtypes"] != ["torch.float8_e4m3fn"]:
            problems.append(
                f"activations reached the kernel as {dispatch['activation_dtypes']}, not FP8 E4M3"
            )
        if dispatch["activation_scale_blocks"] != [128]:
            problems.append(
                f"activation scale granularity is {dispatch['activation_scale_blocks']}, "
                "not one scale per 128 K values"
            )
        if dispatch["activation_scales_are_powers_of_two"] != [True]:
            problems.append("activation scales are not powers of two; this is not the UE8M0 recipe")
        # Both GEMM inputs must be quantized: quantising only the layer input
        # is the approach the plan rejected, and it would show up here as a
        # missing fc2 stage rather than as a metric a reader has to interpret.
        if dispatch["quantised_stages"] != ["fc1_input", "fc2_input"]:
            problems.append(
                f"only {dispatch['quantised_stages']} were block-quantised; the source "
                "quantises the input of both GEMMs"
            )
        if recorder.calls:
            problems.append(
                f"{len(recorder.calls)} torch.ops.trtllm.fused_moe calls ran alongside the "
                "block-scale path; the routed experts must not be served by two kernels"
            )

    dispatch["problems"] = problems
    dispatch["passed"] = not problems
    return dispatch


def _replay_layer(
    rec: _Recorder,
    ranks: Any,
    model: Any,
    src: Any,
    store: dict[str, Any],
    lid: int,
    kind: str,
    tg: Any,
    *,
    num_experts: int,
    hash_layers: int,
    route_scale: float,
    swiglu_limit: float,
    prompt_id: str,
) -> dict[str, Any]:
    """Everything one layer contributes: routing, stages, kernel, clamp."""
    ffn = src.model.layers[lid].ffn
    moe = model.model.layers[lid].mlp
    backend = getattr(moe.experts, "backend", moe.experts)

    ffn_cap = store[f"l{lid}.ffn"]
    hidden = ffn_cap["inputs"][0].reshape(-1, ffn.dim).contiguous()
    src_combined = ffn_cap["output"].reshape(-1, ffn.dim)
    src_weights, src_indices = store[f"l{lid}.gate"]["output"]
    src_shared = store[f"l{lid}.shared"]["output"].reshape(-1, ffn.dim)
    ids = torch.tensor(
        store[f"l{lid}.ffn"]["inputs"][1].flatten().tolist(),
        dtype=torch.int32,
        device=hidden.device,
    )
    local_ids = list(range(ffn.experts_start_idx, ffn.experts_end_idx))
    ctx = {
        "layer": lid,
        "routing": f"{kind}-routed",
        "prompt": prompt_id,
        "num_tokens": int(hidden.shape[0]),
        "moe_backend": type(backend).__name__,
        "weight_method": type(getattr(backend, "quant_method", None)).__name__,
        "op_path": _DISPATCH_CONTRACTS.get(type(backend).__name__, {}).get("op_path"),
        "routed_quantization": _DISPATCH_CONTRACTS.get(type(backend).__name__, {}).get(
            "quantization"
        ),
        "activation": "clamped SwiGLU (silu(clamp(w1x, max=L)) * clamp(w3x, +-L))",
        "expert_parallel": f"ep_size={backend.ep_size} ep_rank={backend.ep_rank}",
        "local_expert_ids": [local_ids[0], local_ids[-1]],
    }

    # -- routing ----------------------------------------------------------
    router_logits = moe.gate(hidden)
    got_indices, got_weights = moe.gate.routing_method.apply(router_logits, ids)
    routing = _routing_checks(
        rec,
        f"layer{lid}",
        ctx,
        ffn.gate,
        hidden,
        (src_weights, src_indices),
        router_logits,
        got_indices,
        got_weights,
        num_experts,
        route_scale,
    )

    # -- expert stages on one real expert ---------------------------------
    counts = collections.Counter()
    for eid in local_ids:
        idx, _ = _fired(src_indices, eid)
        counts[eid] = int(idx.numel())
    sampled = max(counts, key=lambda e: counts[e])
    stages: dict[str, Any] = {"no_expert_fired_on_this_rank": counts[sampled] == 0}
    w4a8_feasibility: dict[str, Any] | None = None
    if counts[sampled]:
        stages.update(
            _expert_stage_checks(
                rec, f"layer{lid}", ctx, store, lid, sampled, ffn.experts[sampled], tg, swiglu_limit
            )
        )
        w1_cap = store[f"l{lid}.e{sampled}.w1"]
        w2_cap = store[f"l{lid}.e{sampled}.w2"]
        w3_cap = store[f"l{lid}.e{sampled}.w3"]
        expert_cap = store[f"l{lid}.e{sampled}"]
        routing_weight = expert_cap["inputs"][1] if len(expert_cap["inputs"]) > 1 else None
        weights = _expert_weights(ffn.experts[sampled], tg)
        stages["activation_precision_gap"] = _precision_gap(
            w1_cap["inputs"][0],
            weights,
            w2_cap["output"],
            routing_weight,
            tg,
            swiglu_limit,
        )
        # The gap above says the source quantises its activations and the
        # production W4A16 path does not. Whether the repo's *other* SM90
        # packed-MXFP4 path could close that is a separate question, and it is
        # answered here rather than argued about.
        w4a8_feasibility = moe_w4a8_feasibility.run(
            model_config=model.model_config,
            moe=moe,
            routing=moe.gate.routing_method,
            shape={
                "n_routed_experts": num_experts,
                "n_hash_layers": hash_layers,
                "hidden_size": int(hidden.shape[-1]),
                "moe_inter_dim": int(weights["w1"].shape[0]),
            },
            lid=lid,
            x=w1_cap["inputs"][0],
            layer_input=hidden,
            weights=weights,
            source={
                "fc1_gate": w1_cap["output"],
                "fc1_up": w3_cap["output"],
                "swiglu": w2_cap["inputs"][0],
                "fc2": w2_cap["output"],
            },
            routing_weight=routing_weight,
            swiglu_limit=swiglu_limit,
            tg=tg,
            ranks=ranks,
        )
        del weights

    # -- the production kernel --------------------------------------------
    routed_local = moe.experts(
        hidden,
        router_logits,
        input_ids=ids,
        do_finalize=True,
        # Read off the model rather than assumed: `DeepseekV4MoE` asks the
        # routed experts for the dtype the reference accumulates `y` in, and a
        # replay that asked for a different one would measure a tensor
        # production never computes.
        output_dtype=moe.routed_accumulator_dtype,
        all_rank_num_tokens=None,
        use_dp_padding=False,
    )
    routed = _all_reduce(routed_local.float(), ranks)
    src_routed_local, fired = _source_routed(store, lid, src_indices, hidden.shape, local_ids)
    src_routed = _all_reduce(src_routed_local, ranks)
    rec.record(
        f"layer{lid}.routed_output",
        "moe_expert_output",
        routed,
        src_routed,
        {
            **ctx,
            "got": "torch.ops.trtllm.fused_moe output, all-reduced over the expert-parallel world",
            "ref": "the source MoE's own routed accumulator, all-reduced the same way",
            "local_experts_that_fired": len(fired),
        },
        grid=torch.bfloat16,
    )

    golden_w4a16 = _all_reduce(
        _golden_routed(
            hidden, ffn, src_indices, src_weights, local_ids, tg, swiglu_limit, quantize_act=False
        ),
        ranks,
    )
    golden_w4a8 = _all_reduce(
        _golden_routed(
            hidden, ffn, src_indices, src_weights, local_ids, tg, swiglu_limit, quantize_act=True
        ),
        ranks,
    )
    boundary = _boundary_prequantisation_experiment(
        rec,
        moe,
        hidden,
        router_logits,
        ids,
        ffn,
        src_indices,
        src_weights,
        src_routed,
        local_ids,
        tg,
        swiglu_limit,
        ranks,
    )
    fp32_routed = _routed_without_the_output_cast(backend, hidden, got_indices, got_weights)
    attribution = {
        # Where the residual routed-output difference lives. If the FP32 entry
        # is materially closer than the production one, the remainder is the
        # MoE interface's BF16 output dtype rather than the kernel arithmetic.
        "trtllm_fp32_accumulator_vs_source": (
            None if fp32_routed is None else _round(tg.compare(_all_reduce(fp32_routed, ranks), src_routed))
        ),
        "trtllm_vs_w4a16_golden": tg.compare(routed, golden_w4a16),
        "source_vs_w4a16_golden": tg.compare(src_routed, golden_w4a16),
        # The reference-ladder anchor. At *matched* activation precision the
        # independent golden and the checkpoint's own TileLang fp4_gemm are the
        # same computation, so this number is how much of the routed-output
        # disagreement is attributable to anything other than activation
        # precision. If it is ~0 the packed-weight contract is exact and the
        # rest of the gap is W4A16-vs-W4A8, by elimination.
        "source_vs_w4a8_golden": tg.compare(src_routed, golden_w4a8),
        "reading": (
            "all three are scored against the same independent dequantise-and-matmul golden. "
            "source_vs_w4a8_golden isolates everything that is *not* activation precision; "
            "the remaining distance between trtllm_vs_w4a16_golden and source_vs_w4a16_golden "
            "is the source's own FP8 activation quantisation, which SM90's only packed-MXFP4 "
            "Cutlass path does not perform."
        ),
    }

    # -- shared expert and the single combine -----------------------------
    shared_stages = _shared_expert_stages(
        rec, f"layer{lid}", ctx, store, lid, moe, ffn, hidden, tg, ranks, swiglu_limit
    )
    # Mirrors whichever collective order production is about to use. When the
    # shared expert is replicated, `forward` adds one whole shared output
    # *after* reducing the routed accumulator, exactly as the reference does,
    # so reducing `shared/tp_size` here would measure a tensor production no
    # longer computes --- and would charge the shared expert for a rounding
    # (summing tp_size copies of S/tp_size) that the new order removes.
    after_reduction = moe.adds_shared_after_the_reduction(None)
    shared_local = moe.compute_shared_output(hidden, apply_output_scale=not after_reduction).float()
    shared = shared_local if after_reduction else _all_reduce(shared_local, ranks)
    rec.record(
        f"layer{lid}.shared_expert",
        "moe_expert_output",
        shared,
        src_shared,
        {
            **ctx,
            "got": "DeepseekV4MoE.compute_shared_output (FP8 block-scale GatedMLP),\n"
            "combined the way production combines it",
            "ref": "the source's own shared Expert",
            "collective_order": (
                "replicated shared expert added after the routed all-reduce, as the reference does"
                if after_reduction
                else "shared expert scaled by 1/tp_size and reduced with the routed output"
            ),
        },
        grid=torch.bfloat16,
    )
    combined = moe(hidden, input_ids=ids)
    rec.record(
        f"layer{lid}.combined_output",
        "moe_combined_output",
        combined,
        src_combined,
        {
            **ctx,
            "got": "DeepseekV4MoE.forward: routed + shared combined once, then TP all-reduce",
            "ref": "the source MoE block's own output",
        },
    )

    clamp = _clamp_evidence(rec, f"layer{lid}", ctx, moe, backend, hidden, ids)

    return {
        "routing": routing,
        "expert_stages": stages,
        "routed_output_attribution": {
            k: (_round(v) if isinstance(v, dict) else v) for k, v in attribution.items()
        },
        "swiglu_clamp": clamp,
        "shared_expert_stages": shared_stages,
        "boundary_prequantisation_experiment": boundary,
        "w4a8_backend_feasibility": w4a8_feasibility,
        "local_experts_that_fired": len(fired),
        "sampled_expert": sampled,
        "tokens_on_sampled_expert": counts[sampled],
    }


def _source_linear_on_a_slice(source_linear: Any, x: torch.Tensor, columns: slice) -> torch.Tensor:
    """Run the *source's own* FP8 kernel over one row-parallel input slice.

    The point is to emulate tensor parallelism without reimplementing anything:
    ``inference/model.py::linear`` dispatches on the weight dtype and reads
    ``weight.scale``, so handing it a column slice of the weight plus the
    matching slice of its 128x128 block scales runs the checkpoint's own
    ``act_quant`` + ``fp8_gemm`` on exactly the partial a TP rank owns. The
    slice boundary is 128-aligned, so the activation blocks and the weight
    scale blocks are the same blocks the unsharded call uses --- the only thing
    that changes is that the K sum is split, and each part is rounded to BF16
    before the parts are added.
    """
    import sys

    weight = source_linear.weight
    scale = source_linear.scale
    block = weight.shape[1] // scale.shape[1]
    assert columns.start % block == 0 and (columns.stop - columns.start) % block == 0, (
        f"slice {columns} does not land on this weight's {block}-wide scale blocks"
    )
    part = weight[:, columns].contiguous()
    part.scale = scale[:, columns.start // block : columns.stop // block].contiguous()
    return sys.modules["model"].linear(x[:, columns].contiguous(), part)


def _shared_expert_stages(
    rec: _Recorder,
    stem: str,
    ctx: dict[str, Any],
    store: dict[str, Any],
    lid: int,
    moe: Any,
    ffn: Any,
    hidden: torch.Tensor,
    tg: Any,
    ranks: Any,
    swiglu_limit: float,
) -> dict[str, Any]:
    """FC1, clamp/SwiGLU, FC2 partial, TP reduction and output scale, separately.

    The shared expert is a *dense* FP8 block-scale path, not a routed one, so a
    routed-expert fix cannot be assumed to move it. Each stage is compared
    against the source's own captured tensors, restricted to the columns this
    rank owns, so a disagreement lands on one owner instead of on the sum.

    The last entry is the one that decides the diagnosis. ``MoE.__init__``
    builds the shared expert as a plain replicated ``Expert``: one FP32
    accumulation across the full intermediate width, rounded to BF16 once.
    TensorRT-LLM shards it, which rounds every rank's partial to BF16 and then
    adds them. ``source_kernel_sharded_the_same_way`` runs the *source's own*
    kernel under that same split, so if it lands at the same distance from the
    unsharded source output as TensorRT-LLM does, the gap is the sharding and
    not the implementation.
    """
    shared = moe.shared_experts
    gate_up, down = shared.gate_up_proj, shared.down_proj

    src_gate = store[f"l{lid}.shared.w1"]["output"]
    src_up = store[f"l{lid}.shared.w3"]["output"]
    src_swiglu = store[f"l{lid}.shared.w2"]["inputs"][0]
    src_out = store[f"l{lid}.shared"]["output"].reshape(-1, hidden.shape[-1])

    full_inter = int(src_gate.shape[-1])
    local_inter = int(down.weight.shape[1])
    tp_size = full_inter // local_inter
    start = (full_inter // tp_size) * (ranks.rank % tp_size) if tp_size > 1 else 0
    columns = slice(start, start + local_inter)
    shared_ctx = {
        **ctx,
        "module": "shared expert (dense FP8 block-scale GatedMLP)",
        "shared_tp_size": tp_size,
        "intermediate_total": full_inter,
        "intermediate_local": local_inter,
        "columns": [columns.start, columns.stop],
        "shared_output_scale": moe.shared_output_scale,
    }

    fused = gate_up(hidden)
    got_gate, got_up = fused[:, :local_inter], fused[:, local_inter:]
    rec.record(
        f"{stem}.shared_fc1_gate",
        "moe_expert_output",
        got_gate,
        src_gate[:, columns],
        {**shared_ctx, "stage": "gate_up_proj gate half vs source w1 output on the owned columns"},
        grid=torch.bfloat16,
    )
    rec.record(
        f"{stem}.shared_fc1_up",
        "moe_expert_output",
        got_up,
        src_up[:, columns],
        {**shared_ctx, "stage": "gate_up_proj up half vs source w3 output on the owned columns"},
        grid=torch.bfloat16,
    )

    got_swiglu = shared._apply_activation(fused)
    rec.record(
        f"{stem}.shared_swiglu_clamped",
        "moe_expert_output",
        got_swiglu,
        src_swiglu[:, columns],
        {**shared_ctx, "stage": f"clamped SwiGLU (limit {swiglu_limit}) on the owned columns"},
        grid=torch.bfloat16,
    )

    # The FC2 partial, judged against the source's own kernel on the same
    # columns. This separates "our row-parallel GEMM is wrong" from "splitting
    # the sum costs a rounding".
    src_partial = _source_linear_on_a_slice(ffn.shared_experts.w2, src_swiglu, columns)
    rec.record(
        f"{stem}.shared_fc2_partial",
        "moe_expert_output",
        down(got_swiglu),
        src_partial,
        {**shared_ctx, "stage": "down_proj partial vs the source's own fp8_gemm on the same slice"},
        grid=torch.bfloat16,
    )

    # Same GEMM, same columns, but driven by the *source's* SwiGLU output
    # rather than by TensorRT-LLM's. FC2 quantizes its input per 128 values
    # with a power-of-two scale, so a single BF16 step of difference at a
    # block's maximum flips that block's scale and requantizes all 128 codes.
    # Feeding both sides the identical tensor separates "the row-parallel GEMM
    # disagrees" from "its input did, and the quantizer amplified it".
    rec.record(
        f"{stem}.shared_fc2_partial_on_source_input",
        "moe_expert_output",
        down(src_swiglu[:, columns].contiguous()),
        src_partial,
        {**shared_ctx, "stage": "down_proj and the source's fp8_gemm on the *same* input slice"},
        grid=torch.bfloat16,
    )

    # What splitting the shared expert costs, measured with the source's own
    # kernel and independent of whatever topology this build actually uses:
    # always the full parallel-group split, so the number stays comparable
    # across a topology change instead of collapsing to a tautology.
    split = int(ranks.world)
    width = full_inter // split
    owned = slice(width * int(ranks.rank), width * (int(ranks.rank) + 1))
    emulated = _all_reduce(
        _source_linear_on_a_slice(ffn.shared_experts.w2, src_swiglu, owned).float(), ranks
    )
    diagnosis = {
        "what_it_runs": (
            f"the source's own act_quant + fp8_gemm, once per rank over a {width}-wide "
            f"slice of the {full_inter}-wide intermediate, summed in FP32"
        ),
        "split_ways": split,
        "source_kernel_split_that_way": _round(tg.compare(emulated, src_out)),
        "source_kernel_unsplit": _round(tg.compare(src_out.float(), src_out.float())),
        "reading": (
            "the source's shared Expert is replicated and accumulates the whole "
            "intermediate width in FP32 before its single BF16 rounding. Where "
            "source_kernel_split_that_way misses the registered limit, the distance "
            "is the tensor-parallel split itself rather than the implementation, and "
            "the source-faithful answer is not to split it."
        ),
    }
    ranks.log(
        f"  shared tp_size={tp_size} cols={columns.start}:{columns.stop} "
        f"source-kernel-split-{split} rel_max_abs="
        f"{diagnosis['source_kernel_split_that_way']['rel_max_abs']}"
    )
    return {
        "columns": [columns.start, columns.stop],
        "shared_tp_size": tp_size,
        "shared_output_scale": moe.shared_output_scale,
        "tp_diagnosis": diagnosis,
    }


#: Every attribute a fused-MoE backend may read its SwiGLU clamp from. The
#: Cutlass path takes a per-slot tensor (``swiglu_limit``); the SM90 block-scale
#: path takes the uniform scalar (``swiglu_limit_scalar``). Both are listed
#: because the control has to perturb whichever one the *running* backend
#: consumes --- toggling the other leaves the kernel unchanged and the control
#: then reports "the clamp is not wired in" about itself.
_CLAMP_OWNERS = ("swiglu_limit", "swiglu_limit_scalar")


def _live_clamp_owners(backend: Any) -> dict[str, Any]:
    """The clamp attributes this backend actually has set, and their values.

    Raises rather than returning nothing: a backend that has no clamp at all
    cannot be probed for one, and silently probing zero attributes would make
    the control pass or fail for a reason unrelated to the kernel.
    """
    owners = {
        name: getattr(backend, name)
        for name in _CLAMP_OWNERS
        if getattr(backend, name, None) is not None
    }
    if not owners:
        raise AssertionError(
            f"{type(backend).__name__} has no SwiGLU clamp set on any of "
            f"{list(_CLAMP_OWNERS)}; the clamp-liveness control has nothing to perturb"
        )
    return owners


def _clamp_value(owners: dict[str, Any]) -> float:
    """The single limit those owners agree on, as a float."""
    values = set()
    for value in owners.values():
        values.add(float(value.flatten()[0]) if torch.is_tensor(value) else float(value))
    if len(values) != 1:
        raise AssertionError(
            f"the backend's clamp owners disagree: {sorted(values)}; the control "
            "cannot say which limit the kernel ran with"
        )
    return values.pop()


def _clamp_evidence(
    rec: _Recorder,
    stem: str,
    ctx: dict[str, Any],
    moe: Any,
    backend: Any,
    hidden: torch.Tensor,
    ids: torch.Tensor,
) -> dict[str, Any]:
    """Structural proof that ``swiglu_limit`` reaches the kernel.

    A tolerance cannot show this: a clamp that never fires and a clamp that is
    not wired in produce identical numbers, and on this prompt the real
    activations never reach the limit. So the same fused op runs twice on an
    input scaled far past it --- once as configured, once with the limit
    removed --- and the rule is that essentially every output element must
    change.

    The rule is deliberately *not* "the limited run has the smaller peak". That
    looks like a consequence of clamping and is not one: ``w2`` is signed, so
    reducing every SwiGLU input can still raise an individual output element
    through cancellation, and requiring it produced a false failure on rank 7
    layer 3 while the same rank's outputs did change everywhere. What clamping
    is actually guaranteed to do is change the result, and the changed fraction
    is what distinguishes "the limit is live" from a one-element fluke. The
    peaks are kept as diagnostics.
    """

    def run_experts(x: torch.Tensor) -> torch.Tensor:
        return moe.experts(
            x,
            moe.gate(x),
            input_ids=ids,
            do_finalize=True,
            output_dtype=x.dtype,
            all_rank_num_tokens=None,
            use_dp_padding=False,
        ).float()

    # Large enough that the FC1 projections of a 4096-wide BF16 hidden state
    # clear a limit of 10 by a wide margin, small enough to stay finite.
    driven = (hidden.float() * 16.0).to(hidden.dtype)
    saved = _live_clamp_owners(backend)
    limited = run_experts(driven)
    try:
        for name in saved:
            setattr(backend, name, None)
        unlimited = run_experts(driven)
    finally:
        for name, value in saved.items():
            setattr(backend, name, value)

    routed = (limited != 0) | (unlimited != 0)
    changed = int(((limited != unlimited) & routed).sum())
    routed_elements = int(routed.sum())
    changed_fraction = changed / routed_elements if routed_elements else 0.0
    limited_max = float(limited.abs().max())
    unlimited_max = float(unlimited.abs().max())
    evidence = {
        "input_scale": 16.0,
        "clamp_owners": sorted(saved),
        "configured_limit": _clamp_value(saved),
        "outputs_differ": changed > 0,
        "routed_elements": routed_elements,
        "elements_changed": changed,
        "changed_fraction": round(changed_fraction, 6),
        "abs_max_with_limit": round(limited_max, 6),
        "abs_max_without_limit": round(unlimited_max, 6),
    }
    passed = changed_fraction > 0.5
    rec.exact(
        f"{stem}.swiglu_clamp_is_live",
        "moe_expert_output",
        passed,
        {
            **ctx,
            **evidence,
            "rule": "removing swiglu_limit must change more than half of the fused op's "
            "routed output elements on activations driven past the limit",
            "problem": (
                None
                if passed
                else f"removing swiglu_limit changed only {changed}/{routed_elements} "
                "routed output elements; the limit does not reach the kernel"
            ),
        },
    )
    return evidence
