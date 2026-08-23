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
"""Packed-MXFP4 routed experts with block-scaled FP8 activations.

A checkpoint that stores routed experts as packed MXFP4 and declares
``scale_fmt="ue8m0"`` was produced and evaluated with W4A8 arithmetic: its own
reference implementation quantizes every routed-expert activation to FP8 E4M3
per token and per 128 K values, with a power-of-two scale, before both the
FC1 and the FC2 GEMM. The pre-Blackwell packed-MXFP4 kernel is W4A16 and
leaves those activations in BF16, which is a different function of the same
weights --- correct as an approximation, but not the one the checkpoint was
tuned against.

This backend runs that W4A8 contract in OpenAI Triton. It reuses the
``ConfigurableMoE`` weight lifecycle, routing, dispatch/combine and
expert-parallel communication unchanged; only the two GEMMs and the
activation between them are its own. Selection goes through the existing
quantization vocabulary (``W4A8_MXFP4_FP8`` plus the checkpoint's declared
``scale_fmt``), so nothing about this class is model-specific.
"""

from typing import Optional, Tuple, Union

import torch

from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.models.modeling_utils import QuantAlgo

from .fused_moe_cutlass import CutlassFusedMoE
from .impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
)
from .interface import _reject
from .mxfp4_blockscale_kernels import (
    ACT_BLOCK_SIZE,
    combine_expert_rows,
    moe_w4a8_gemm,
    quantize_blockwise_ue8m0,
    swiglu_and_quantize,
)
from .quantization import MXFP4BlockScaleFusedMoEMethod

#: Rows per scheduling block. ``moe_align_block_size`` pads every expert's row
#: count up to this, so the padding cost is ``local_experts * BLOCK_M`` rows
#: per GEMM. 32 keeps that bounded for single-token decode while still giving
#: the MMA a full tile.
BLOCK_M = 32

#: The activation scale format this backend implements. A checkpoint that
#: declares anything else was quantized against a different recipe.
SUPPORTED_ACT_SCALE_FMT = "ue8m0"


class BlockScaleMXFP4FusedMoE(CutlassFusedMoE):
    """W4A8 packed-MXFP4 experts with per-token, per-128-K FP8 activations.

    Subclasses :class:`CutlassFusedMoE` for its weight lifecycle, expert
    parallel bookkeeping and communication, exactly as
    :class:`~.fused_moe_marlin.MarlinFusedMoE` does; ``run_moe`` and the
    activation quantization are the only execution overrides.
    """

    # Restated rather than inherited: CutlassFusedMoE claims fused
    # routed-expert LoRA on the strength of its own kernel, which this
    # backend does not run.
    capabilities = MoEStaticCapability(supports_moe_lora=False)

    #: This backend can hand back its routed accumulator without rounding it.
    #:
    #: The reference MoE keeps ``y`` in FP32 from the first expert until after
    #: the shared expert is added, and only then casts once. A caller that asks
    #: for FP32 here gets that accumulator; a caller that asks for the model
    #: dtype gets it rounded, which is one rounding earlier than the reference.
    #: Advertised as a plain class attribute so a caller reads it with
    #: ``getattr(backend, "returns_fp32_accumulator", False)`` and every backend
    #: that has not thought about it keeps today's behaviour.
    returns_fp32_accumulator = True

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        if p.quant_algo is not QuantAlgo.W4A8_MXFP4_FP8:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"BlockScaleMXFP4FusedMoE serves packed MXFP4 weights with FP8 "
                f"activations (got quant_algo={p.quant_algo})",
            )
        # The distinguishing gate. ``W4A8_MXFP4_FP8`` is also the algorithm of
        # the per-tensor FP8 Triton path, and the two quantize the same
        # activation to different codes, so the scale format is what says
        # which of them the checkpoint asked for.
        if p.act_scale_fmt != SUPPORTED_ACT_SCALE_FMT:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"BlockScaleMXFP4FusedMoE implements block-scaled "
                f"{SUPPORTED_ACT_SCALE_FMT} FP8 activations; this layer declares "
                f"act_scale_fmt={p.act_scale_fmt!r}",
            )
        if p.dtype_act is not torch.bfloat16:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"BlockScaleMXFP4FusedMoE quantizes bfloat16 activations, got {p.dtype_act}",
            )
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "BlockScaleMXFP4FusedMoE implements a plain clamped SwiGLU, not "
                "the gpt-oss bias/alpha/beta package",
            )
        for name, value in (("hidden_size", p.hidden_size), ("intermediate_size", p.intermediate_size)):
            if value is not None and value % ACT_BLOCK_SIZE:
                return _reject(
                    MoERejectReason.SHAPE_UNALIGNED,
                    f"BlockScaleMXFP4FusedMoE needs {name} to be a multiple of the "
                    f"{ACT_BLOCK_SIZE}-wide activation block, got {value}",
                )
        if p.num_experts is not None and d.ep_size and p.num_experts % d.ep_size:
            return _reject(
                MoERejectReason.SLOTS_NOT_DIVISIBLE_BY_EP,
                f"{p.num_experts} experts do not divide over ep_size {d.ep_size}",
            )
        # Sorted-token dispatch has no EPLB slot layout. Answered from ``d``
        # because selection happens before the object exists.
        if d.eplb_enabled:
            return _reject(
                MoERejectReason.EPLB_UNSUPPORTED,
                "BlockScaleMXFP4FusedMoE has no EPLB slot layout",
            )
        return MoEEligibility.ok()

    def _get_quant_method(self):
        assert self.quant_config is not None and (
            self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8()
        ), f"BlockScaleMXFP4FusedMoE only serves W4A8_MXFP4_FP8, got {self.quant_config}"
        return MXFP4BlockScaleFusedMoEMethod()

    def _supports_load_balancer(self) -> bool:
        return False

    def supports_moe_output_in_alltoall_workspace(self):
        # The combine allocates and returns its own output, so a
        # workspace-backed buffer would be read by combine() and written by
        # nobody.
        return False

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Leave the input high precision; the block scales are per expert row.

        The FP8 scale this backend needs is derived per token and per 128 K
        values *inside* ``run_moe``. Quantizing at the module boundary instead
        would only cover FC1 --- the FC2 input never exists outside the
        expert --- and would send FP8 rather than BF16 across the
        expert-parallel exchange.
        """
        return x, None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _routing_plan(
        self, token_selected_experts: torch.Tensor, num_tokens: int, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Group the token-expert pairs this rank owns into fixed-size blocks.

        Returns ``(sorted_ids, expert_ids, pair_valid, pair_pos)``.
        Non-local pairs are given the sentinel expert id ``local_experts``, so
        they keep their slot in the permutation --- which the combine indexes
        by pair --- while the GEMM skips their block and stores zeros.
        """
        device = token_selected_experts.device
        local_experts = self.expert_size_per_partition
        num_pairs = num_tokens * top_k

        local = token_selected_experts.to(torch.int32) - int(self.slot_start)
        local = torch.where(
            (local >= 0) & (local < local_experts),
            local,
            torch.full_like(local, local_experts),
        ).contiguous()

        # +1 for the sentinel expert, which moe_align_block_size gives its own
        # padded region rather than mixing it into a real expert's rows. The
        # bound is the pair count plus one short block per expert, rounded up
        # so the GEMM grid divides evenly.
        num_align_experts = local_experts + 1
        padded_rows = (
            (num_pairs + BLOCK_M - 1) // BLOCK_M + num_align_experts
        ) * BLOCK_M
        sorted_ids = torch.full((padded_rows,), num_pairs, dtype=torch.int32, device=device)
        expert_ids = torch.full(
            (padded_rows // BLOCK_M,), local_experts, dtype=torch.int32, device=device
        )
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
        torch.ops.trtllm.moe_align_block_size(
            local, num_align_experts, BLOCK_M, sorted_ids, expert_ids, num_tokens_post_pad
        )

        pair_valid = sorted_ids < num_pairs
        # Every pair appears exactly once, so scattering positions by pair id
        # inverts the permutation. Padding slots all carry ``num_pairs``, which
        # the extra trailing entry absorbs.
        pair_pos = torch.zeros(num_pairs + 1, dtype=torch.int32, device=device)
        pair_pos.scatter_(
            0,
            sorted_ids.to(torch.int64),
            torch.arange(padded_rows, dtype=torch.int32, device=device),
        )
        return sorted_ids, expert_ids, pair_valid, pair_pos[:num_pairs]

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        del workspace  # This backend allocates its own intermediates.
        x = ctx.x
        assert x.dtype == torch.bfloat16, (
            f"BlockScaleMXFP4FusedMoE quantizes bfloat16 activations, got {x.dtype}"
        )
        assert self.hidden_size == self.unpadded_hidden_size, (
            "BlockScaleMXFP4FusedMoE does not pad the hidden size; a padded "
            "layer would need the activation padded to match"
        )

        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        if token_selected_experts is None:
            assert ctx.router_logits is not None, (
                "BlockScaleMXFP4FusedMoE.run_moe needs token_selected_experts or router_logits"
            )
            token_selected_experts, token_final_scales = self.routing_method.apply(ctx.router_logits)

        num_tokens = x.shape[0]
        top_k = token_selected_experts.shape[1]
        num_pairs = num_tokens * top_k
        output_dtype = ctx.output_dtype or x.dtype
        if num_tokens == 0:
            return torch.zeros(
                (0, self.unpadded_hidden_size), dtype=output_dtype, device=x.device
            )

        sorted_ids, expert_ids, pair_valid, pair_pos = self._routing_plan(
            token_selected_experts, num_tokens, top_k
        )
        padded_rows = sorted_ids.shape[0]
        invalid = torch.full_like(sorted_ids, -1)

        # FC1 reads one activation row per *token*: a token routed to several
        # local experts quantizes once, exactly as the reference does, because
        # the block maximum is a property of the row and not of the routing.
        x_q, x_scale = quantize_blockwise_ue8m0(x, ACT_BLOCK_SIZE)
        fc1_rows = torch.where(
            pair_valid, torch.div(sorted_ids, top_k, rounding_mode="floor"), invalid
        )
        fc1 = moe_w4a8_gemm(
            x_q,
            x_scale,
            fc1_rows,
            self.w3_w1_weight,
            self.fc31_weight_scale,
            expert_ids,
            self.expert_size_per_partition,
            BLOCK_M,
        )

        if token_final_scales is None:
            routing_weight = torch.ones(num_pairs + 1, dtype=torch.float32, device=x.device)
        else:
            routing_weight = torch.cat(
                [
                    token_final_scales.reshape(-1).to(torch.float32),
                    torch.zeros(1, dtype=torch.float32, device=x.device),
                ]
            )
        # The trailing zero absorbs every padding slot, whose sorted id is
        # exactly ``num_pairs``, so no branch is needed here.
        routing_sorted = routing_weight[sorted_ids.to(torch.int64)]

        fc2_in, fc2_scale = swiglu_and_quantize(
            fc1,
            routing_sorted,
            self.intermediate_size_per_partition,
            self.swiglu_limit_scalar,
            ACT_BLOCK_SIZE,
        )
        del fc1

        fc2_rows = torch.where(
            pair_valid,
            torch.arange(padded_rows, dtype=torch.int32, device=x.device),
            invalid,
        )
        fc2 = moe_w4a8_gemm(
            fc2_in,
            fc2_scale,
            fc2_rows,
            self.w2_weight,
            self.fc2_weight_scale,
            expert_ids,
            self.expert_size_per_partition,
            BLOCK_M,
        )

        combined = combine_expert_rows(
            fc2, pair_pos, num_tokens, top_k, self.unpadded_hidden_size
        )
        return combined.to(output_dtype)
