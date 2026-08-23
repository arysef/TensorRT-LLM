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
"""An FP8 block-scale GEMM that reproduces a producer's own kernel structure.

A dense FP8 block-scale ``Linear`` normally wants the fastest kernel that is
within tolerance. This one exists for the opposite reason: some checkpoints
ship the exact kernel they were evaluated with, and a bring-up has to show it
reproduces that kernel's *output*, not merely its mathematics. The two goals
differ by about one BF16 storage step, which is invisible in a benchmark and
fatal to an element-wise parity gate whose reference tensor is BF16: the gate
measures the largest element-wise difference against the tensor's RMS, and one
step at a large element is several percent of a typical RMS.

So this kernel is written to the reference's shape rather than to the
hardware's preference:

* K is walked in blocks of 128 -- one activation-scale block, one weight-scale
  block -- and each block's FP32 dot product is multiplied by
  ``act_scale * weight_scale`` before it joins the output accumulator. That
  second accumulator is the structure the reference calls "2x accumulation
  precision"; a kernel that instead accumulates first and rescales afterwards
  is a different computation, not a rounder one.
* The tile is 32 rows by 128 columns, so the weight scale is one scalar per
  tile (a 128-column tile is exactly one weight-scale group) and the
  accumulation order over K matches block for block.
* The output is rounded to BF16 once, at the end, because the reference GEMM
  returns the default dtype and everything downstream consumes that rounded
  value.

Nothing here is model-specific. The entry point takes quantized activations,
their per-128 scales, an FP8 weight and its per-128x128 scales.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .linear import FP8BlockScalesLinearMethod, Linear

#: Activation block: one FP8 scale per this many K values.
ACT_BLOCK_SIZE = 128
#: Weight block: one scale per this many rows and this many K values.
WEIGHT_BLOCK_SIZE = 128
#: Rows per tile, matching the reference kernel's ``block_M``.
BLOCK_M = 32


@triton.jit
def _fp8_blockwise_gemm_kernel(
    A,
    AS,
    B,
    BS,
    C,
    stride_am,
    stride_as,
    stride_bn,
    stride_bs,
    stride_cm,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``C[m, n] = sum_k A[m, k] * B[n, k]`` with per-block FP8 scaling.

    One program owns a ``BLOCK_M x BLOCK_N`` output tile. ``BLOCK_N`` is one
    weight-scale group, so the weight scale for the whole tile is a single
    scalar per K block and the per-row combined scale is exactly the
    reference's ``scales_a[row, k] * Scale_B``.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_mask = offs_m < M
    col_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs = k * BLOCK_K + offs_k
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        b = tl.load(
            B + offs_n[:, None] * stride_bn + offs[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        block = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
        act_scale = tl.load(AS + offs_m * stride_as + k, mask=row_mask, other=0.0)
        weight_scale = tl.load(BS + pid_n * stride_bs + k)
        # The reference forms one combined per-row scale and applies it to the
        # whole tile, then adds into a separate FP32 accumulator.
        acc += block * (act_scale * weight_scale)[:, None]

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def fp8_blockwise_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """``a @ b.T`` for FP8 ``a[M, K]``, FP8 ``b[N, K]`` with block scales.

    ``a_scale`` is one value per ``ACT_BLOCK_SIZE`` K values per row;
    ``b_scale`` is one value per ``WEIGHT_BLOCK_SIZE`` rows per
    ``WEIGHT_BLOCK_SIZE`` K values. The result is BF16, which is the dtype the
    reference GEMM returns and therefore the dtype everything downstream of it
    consumes.
    """
    assert a.dim() == 2 and b.dim() == 2, "expected 2-D operands"
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn, (
        f"expected FP8 E4M3 operands, got {a.dtype} and {b.dtype}"
    )
    assert a.stride(-1) == 1 and b.stride(-1) == 1, "K must be the contiguous dimension"
    m, k = a.shape
    n, k_b = b.shape
    assert k == k_b, f"K mismatch: activation {k} vs weight {k_b}"
    assert k % ACT_BLOCK_SIZE == 0, f"K {k} is not a multiple of {ACT_BLOCK_SIZE}"
    assert n % WEIGHT_BLOCK_SIZE == 0, f"N {n} is not a multiple of {WEIGHT_BLOCK_SIZE}"
    assert a_scale.shape == (m, k // ACT_BLOCK_SIZE), (
        f"activation scale {tuple(a_scale.shape)} is not one value per "
        f"{ACT_BLOCK_SIZE} of {k} K values for {m} rows"
    )
    assert b_scale.shape == (n // WEIGHT_BLOCK_SIZE, k // WEIGHT_BLOCK_SIZE), (
        f"weight scale {tuple(b_scale.shape)} is not one value per "
        f"{WEIGHT_BLOCK_SIZE}x{WEIGHT_BLOCK_SIZE} block of a {n}x{k} weight"
    )
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    if m == 0:
        return out
    a_scale = a_scale.to(torch.float32).contiguous()
    b_scale = b_scale.to(torch.float32).contiguous()
    _fp8_blockwise_gemm_kernel[(n // WEIGHT_BLOCK_SIZE, triton.cdiv(m, BLOCK_M))](
        a,
        a_scale,
        b,
        b_scale,
        out,
        a.stride(0),
        a_scale.stride(0),
        b.stride(0),
        b_scale.stride(0),
        out.stride(0),
        m,
        n,
        k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=WEIGHT_BLOCK_SIZE,
        BLOCK_K=ACT_BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )
    return out


class FP8BlockScalesParityLinearMethod(FP8BlockScalesLinearMethod):
    """The same FP8 block-scale weights, run through :func:`fp8_blockwise_gemm`.

    Everything about the weights --- their shapes, their scales, how they are
    created, sharded and loaded --- is inherited unchanged; only the GEMM
    differs. A module opts in by setting ``use_blockwise_parity_gemm``, and
    :meth:`Linear.get_quant_method` honours the opt-in only for a checkpoint
    that declares ``scale_fmt="ue8m0"``, because the activation quantizer here
    reproduces exactly that recipe: a checkpoint quantized against FP32 scales
    would be given power-of-two ones, which is a different quantization of the
    same tensor rather than a more accurate one.

    Opting in trades throughput for element-wise agreement with the
    checkpoint's own kernel, so it belongs to a module whose parity is being
    measured against that kernel, not to a dense layer chosen for speed.
    """

    @staticmethod
    def is_enabled(module: Linear) -> bool:
        """Whether ``module`` asked for the parity GEMM and may have it."""
        if not getattr(module, "use_blockwise_parity_gemm", False):
            return False
        quant_config = getattr(module, "quant_config", None)
        return getattr(quant_config, "scale_fmt", None) == "ue8m0"

    def apply(
        self, module: Linear, input: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Local import: `fused_moe` imports `Linear`, so importing its kernels
        # at module scope would close a cycle through this file.
        from .fused_moe.mxfp4_blockscale_kernels import quantize_blockwise_ue8m0

        original_shape = input.shape
        if input.dim() > 2:
            input = input.reshape(-1, input.shape[-1])
        if input.dtype == torch.float8_e4m3fn:
            input = input.to(torch.bfloat16) * module.input_scale
        assert input.dtype == torch.bfloat16

        act, act_scale = quantize_blockwise_ue8m0(input.contiguous(), ACT_BLOCK_SIZE)
        output = fp8_blockwise_gemm(act, act_scale, module.weight, module.weight_scale)

        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], output.shape[-1])
        if bias is not None:
            output = output + bias
        return output
