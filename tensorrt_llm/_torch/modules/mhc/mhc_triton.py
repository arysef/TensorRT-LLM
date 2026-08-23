# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-faithful mHC post-mapping for pre-Blackwell parts.

``mhc_post_mapping`` (``cpp/tensorrt_llm/kernels/mhcKernels``) computes

    out[j][h] = post[j] * x[h] + sum_k comb[k][j] * residual[k][h]

with an FMA chain seeded by ``post * x``. That is one rounding *fewer* per term
than the reference, which is a plain Torch expression --- ``post[..., None] *
x[..., None, :] + torch.sum(comb[..., None] * residual[..., None, :], dim=2)``
--- and therefore rounds every product to FP32 before summing them, in index
order, and adds the ``post * x`` term last.

The two agree on all but ~1.6e-05 of elements at BF16 storage resolution, and
where they differ the FMA chain is the more accurate of the two. What it is not
is the same number, and a parity gate that compares a BF16 tensor against the
checkpoint's own output sees those elements as a full storage step apart. This
module supplies the reference's association exactly, so on Hopper the
hyper-connection post-mapping reproduces the checkpoint bit for bit instead of
landing within a storage step of it.

``enable_fp_fusion=False`` is what makes that reachable from Triton: without it
the backend contracts ``acc + comb * residual`` back into an FMA and the kernel
reproduces the CUDA path rather than the reference. Measured on 4 M random
elements: 405,569 of 1,048,576 FP32 results differ with fusion enabled and 0
with it disabled.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _source_post_mapping_kernel(
    residual_ptr,
    x_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden,
    MULT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One token's ``hidden`` slice, all ``MULT`` output streams.

    The accumulation is written the way the reference brackets it: each
    ``comb[k] * residual[k]`` is a rounded FP32 product, they are summed in
    ascending ``k``, and ``post * x`` joins last. Compiled with
    ``enable_fp_fusion=False`` so that bracketing survives codegen.
    """
    token = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden
    x = tl.load(x_ptr + token * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    for j in tl.static_range(MULT):
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(MULT):
            comb = tl.load(comb_ptr + token * MULT * MULT + k * MULT + j)
            residual = tl.load(
                residual_ptr + token * MULT * hidden + k * hidden + offs, mask=mask, other=0.0
            ).to(tl.float32)
            term = comb * residual
            acc = term if k == 0 else acc + term
        post = tl.load(post_ptr + token * MULT + j)
        out = post * x + acc
        tl.store(
            out_ptr + token * MULT * hidden + j * hidden + offs, out.to(tl.bfloat16), mask=mask
        )


def source_post_mapping(
    residual: torch.Tensor,
    x: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    mult: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``Block.hc_post`` from the checkpoint's ``inference/model.py``, bit for bit.

    Args:
        residual: ``[M, mult, hidden]`` bf16.
        x:        ``[M, hidden]`` bf16 --- the block output being mapped back in.
        post_mix: ``[M, mult]`` fp32.
        comb_mix: ``[M, mult, mult]`` fp32.
        mult:     hyper-connection multiplier.
        out:      optional ``[M, mult, hidden]`` bf16 destination.

    Returns:
        ``[M, mult, hidden]`` bf16.
    """
    assert residual.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    m, got_mult, hidden = residual.shape
    assert got_mult == mult, f"residual carries {got_mult} streams, expected {mult}"
    residual = residual.contiguous()
    x = x.contiguous()
    post_mix = post_mix.reshape(m, mult).float().contiguous()
    comb_mix = comb_mix.reshape(m, mult, mult).float().contiguous()
    if out is None:
        out = torch.empty_like(residual)
    block = 1024 if hidden >= 1024 else triton.next_power_of_2(hidden)
    _source_post_mapping_kernel[(m, triton.cdiv(hidden, block))](
        residual,
        x,
        post_mix,
        comb_mix,
        out,
        hidden,
        MULT=mult,
        BLOCK=block,
        # Not a tuning knob: FMA contraction is exactly the difference between
        # this kernel and the CUDA one it exists to replace.
        enable_fp_fusion=False,
    )
    return out
