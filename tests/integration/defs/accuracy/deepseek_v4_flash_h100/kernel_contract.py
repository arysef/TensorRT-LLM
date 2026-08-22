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
"""Proof that the sparse-attention golden reproduces the source kernel's GEMMs.

The golden in :mod:`torch_goldens` claims a specific arithmetic contract: the
source's ``T.gemm`` calls are BF16 x BF16 with an FP32 accumulator, so the
reference must keep the operands in BF16 and ask Torch for the FP32
accumulator rather than widening to FP32 first. That claim decides whether the
reference is trustworthy at all, and until it is measured it is only an
assertion.

This module measures it on real SM90 silicon, at the bring-up geometry, with
no checkpoint: it drives the checkpoint's own ``inference/kernel.sparse_attn``
on synthetic BF16 activations and compares, element for element, against the
golden and against the FP32-widened variant it replaced. Two isolated tilelang
kernels reproduce the score and value ``T.gemm`` on their own, so a
disagreement can be attributed to the matmul or to the softmax bookkeeping
instead of being averaged into one number.

It runs in about half a minute and needs one GPU, which makes it the cheapest
rung of the ladder to re-run when the golden's arithmetic is touched.
"""

# NOTE: no ``from __future__ import annotations`` here. The tilelang prim_funcs
# below carry their shapes and dtypes *in* their annotations, and tilelang reads
# them at definition time --- turning them into strings would break the kernels
# and makes static checkers read the dtype literals as forward references. The
# checkpoint's own ``inference/kernel.py`` is written the same way.
import sys
from typing import Any

import torch

#: The checkpoint's own inference code, which owns the kernel under study.
SOURCE_INFERENCE = "/models/DeepSeek-V4-Flash/inference"

#: Shared-memory and accumulator dtypes, named as ``inference/kernel.py``
#: names them. Module constants rather than inline literals because the
#: tilelang prim_funcs below carry them in annotation position.
BF16 = "bfloat16"
FP32 = "float32"

#: Bring-up geometry: 512-wide latent head, 8 Q heads per TP8 rank (the kernel
#: pads to 16 itself), a 257-token prefill crossing the 128-token SWA boundary.
HEAD_DIM = 512
LOCAL_HEADS = 8
SEQ_LEN = 257
PADDED_HEADS = 16


def _load_source_kernel() -> Any:
    if SOURCE_INFERENCE not in sys.path:
        sys.path.insert(0, SOURCE_INFERENCE)
    import kernel as source_kernel

    return source_kernel


def _isolated_gemm_kernels(source_kernel: Any) -> tuple[Any, Any]:
    """The kernel's two ``T.gemm`` calls, each on its own.

    Same shared-memory dtypes, same accumulator dtype, same warp policy and
    same transposition as ``sparse_attn_kernel``, with everything else removed.
    """
    import tilelang
    import tilelang.language as T

    @tilelang.jit(pass_configs=source_kernel.pass_configs)
    def score_gemm(h: int, d: int, block: int):
        b, m = T.symbolic("b"), T.symbolic("m")

        @T.prim_func
        def score_gemm_(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, block, d), BF16],
            o: T.Tensor[(b, m, h, block), FP32],
        ):
            with T.Kernel(m, b, threads=256) as (bx, by):
                q_shared = T.alloc_shared((h, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                acc_s = T.alloc_fragment((h, block), FP32)
                T.copy(q[by, bx, :, :], q_shared)
                T.copy(kv[by, :, :], kv_shared)
                T.clear(acc_s)
                T.gemm(
                    q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                )
                T.copy(acc_s, o[by, bx, :, :])

        return score_gemm_

    @tilelang.jit(pass_configs=source_kernel.pass_configs)
    def value_gemm(h: int, d: int, block: int):
        b, m = T.symbolic("b"), T.symbolic("m")

        @T.prim_func
        def value_gemm_(
            p: T.Tensor[(b, m, h, block), BF16],
            kv: T.Tensor[(b, block, d), BF16],
            o: T.Tensor[(b, m, h, d), FP32],
        ):
            with T.Kernel(m, b, threads=256) as (bx, by):
                p_shared = T.alloc_shared((h, block), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                acc_o = T.alloc_fragment((h, d), FP32)
                T.copy(p[by, bx, :, :], p_shared)
                T.copy(kv[by, :, :], kv_shared)
                T.clear(acc_o)
                T.gemm(p_shared, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_o, o[by, bx, :, :])

        return value_gemm_

    return score_gemm, value_gemm


def disagreement(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Element-level disagreement, counted rather than averaged.

    A tolerance cannot separate these implementations --- they compute the same
    mathematical quantity and agree to FP32 round-off either way. What
    distinguishes them is *how many* elements land on a different value, so
    that is what is reported.
    """
    ref, got = reference.float(), candidate.float()
    diff = (got - ref).abs()
    rms = max(float(ref.square().mean().sqrt()), 1e-30)
    return {
        "elements_differing": int((got != ref).sum()),
        "elements": int(ref.numel()),
        "fraction_differing": round(float((got != ref).float().mean()), 9),
        "max_abs": round(float(diff.max()), 9),
        "rel_max_abs": round(float(diff.max()) / rms, 9),
        "bit_exact": bool(torch.equal(got, ref)),
    }


def _widened_sparse_attention(tg: Any, q, kv, attn_sink, topk_idxs, softmax_scale) -> torch.Tensor:
    """The golden as it was written before the operands were kept in BF16.

    Carried here rather than deleted so the improvement is a measurement in
    the artifact instead of a claim in a commit message.
    """
    original = tg._accumulating_matmul

    def widened(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a.float(), b.float())

    tg._accumulating_matmul = widened
    try:
        return tg.sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)
    finally:
        tg._accumulating_matmul = original


def _attribute_residual(kernel_out: torch.Tensor, golden_out: torch.Tensor) -> dict[str, Any]:
    """Say *where* the handful of remaining disagreements come from.

    With both GEMMs bit-exact, the residual has to be softmax bookkeeping, and
    the two candidates need different fixes. A different denominator
    (``T.reduce_sum`` order) rescales an entire (query, head) row by one
    factor, so its flips would cluster --- one affected row would show many of
    its 512 elements move. The value GEMM's live-accumulator ordering (the
    kernel accumulates the tile *into* the rescaled ``acc_o``, which no Torch
    ``bmm`` can express) perturbs elements independently, so its flips scatter
    at most one or two per row.

    Reported rather than gated: it is a diagnosis, and the pass/fail claims are
    the bit-exact GEMMs above.
    """
    differing = golden_out.float() != kernel_out.float()
    per_row = differing.flatten(0, -2).sum(dim=-1)
    hit = per_row > 0
    return {
        "elements_differing": int(differing.sum()),
        "rows": int(per_row.numel()),
        "row_width": int(differing.shape[-1]),
        "rows_with_a_flip": int(hit.sum()),
        "max_flips_in_one_row": int(per_row.max()),
        "attribution": (
            "value-GEMM live-accumulator ordering: flips scatter across distinct rows"
            if int(per_row.max()) <= 2
            else "denominator ordering: flips cluster within rows"
        ),
    }


def run(tg: Any, device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    """Measure the golden's GEMM contract against the real source kernel."""
    source_kernel = _load_source_kernel()
    score_gemm, value_gemm = _isolated_gemm_kernels(source_kernel)
    torch.manual_seed(seed)

    block = tg.SPARSE_ATTN_BLOCK
    d, h = HEAD_DIM, PADDED_HEADS
    checks: dict[str, Any] = {}

    # -- the score GEMM on its own: (h, d) x (block, d)^T -> FP32 -------------
    q = (torch.randn(1, block, h, d, device=device) * 0.5).bfloat16()
    kv = (torch.randn(1, block, d, device=device) * 0.5).bfloat16()
    kernel_scores = torch.empty(1, block, h, block, device=device, dtype=torch.float32)
    score_gemm(h, d, block)(q, kv, kernel_scores)
    rows = q.shape[0] * q.shape[1]
    checks["score_gemm.bf16_operands"] = disagreement(
        kernel_scores,
        tg._accumulating_matmul(
            q.reshape(rows, h, d), kv.expand(rows, block, d).transpose(1, 2)
        ).reshape(kernel_scores.shape),
    )
    checks["score_gemm.widened_to_fp32"] = disagreement(
        kernel_scores, torch.einsum("bmhd,bnd->bmhn", q.float(), kv.float())
    )

    # -- the value GEMM on its own: (h, block) x (block, d) -> FP32 ----------
    probs = (torch.randn(1, block, h, block, device=device) * 0.5).bfloat16()
    kernel_values = torch.empty(1, block, h, d, device=device, dtype=torch.float32)
    value_gemm(h, d, block)(probs, kv, kernel_values)
    checks["value_gemm.bf16_operands"] = disagreement(
        kernel_values,
        tg._accumulating_matmul(probs.reshape(rows, h, block), kv.expand(rows, block, d)).reshape(
            kernel_values.shape
        ),
    )
    checks["value_gemm.widened_to_fp32"] = disagreement(
        kernel_values, torch.einsum("bmhn,bnd->bmhd", probs.float(), kv.float())
    )

    # -- the whole kernel, at both sparse schedules -------------------------
    torch.manual_seed(seed + 1)
    q = (torch.randn(1, SEQ_LEN, LOCAL_HEADS, d, device=device) * 0.5).bfloat16()
    kv = (torch.randn(1, SEQ_LEN + block, d, device=device) * 0.5).bfloat16()
    attn_sink = torch.randn(LOCAL_HEADS, device=device, dtype=torch.float32)
    scale = float(d) ** -0.5
    position = torch.arange(SEQ_LEN, device=device).unsqueeze(1)

    residual: dict[str, Any] = {}
    for selected, label in ((128, "window128"), (512, "selected512")):
        offset = torch.arange(selected, device=device).unsqueeze(0)
        slot = position - offset
        topk = torch.where(slot >= 0, slot, torch.full_like(slot, -1)).int().unsqueeze(0)
        topk = topk.contiguous()
        kernel_out = source_kernel.sparse_attn(q, kv, attn_sink, topk, scale)
        golden_out = tg.sparse_attention(q, kv, attn_sink, topk, scale)
        checks[f"sparse_attention.{label}.bf16_operands"] = disagreement(kernel_out, golden_out)
        checks[f"sparse_attention.{label}.widened_to_fp32"] = disagreement(
            kernel_out, _widened_sparse_attention(tg, q, kv, attn_sink, topk, scale)
        )
        residual[label] = _attribute_residual(kernel_out, golden_out)

    problems: list[str] = []
    # Both GEMMs must be bit-exact. This is the whole claim: nothing about the
    # reference's arithmetic is approximate at the matmul level.
    for name in ("score_gemm.bf16_operands", "value_gemm.bf16_operands"):
        if not checks[name]["bit_exact"]:
            problems.append(f"{name} is not bit-exact against T.gemm: {checks[name]}")
    # And widening must be demonstrably worse, or keeping BF16 buys nothing and
    # the isolated results above were a coincidence of the shapes chosen.
    for name in ("score_gemm", "value_gemm"):
        if checks[f"{name}.widened_to_fp32"]["bit_exact"]:
            problems.append(f"{name}: widening to FP32 also matched, so this proves nothing")
    for label in ("window128", "selected512"):
        kept = checks[f"sparse_attention.{label}.bf16_operands"]["elements_differing"]
        widened = checks[f"sparse_attention.{label}.widened_to_fp32"]["elements_differing"]
        if kept >= widened:
            problems.append(
                f"sparse_attention.{label}: keeping BF16 operands did not reduce disagreement "
                f"({kept} vs {widened} elements)"
            )

    return {
        "evidence_label": "kernel_contract",
        "reference_tier": "real_source",
        "validation_tier": "unit",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "geometry": {
            "head_dim": d,
            "local_heads": LOCAL_HEADS,
            "padded_heads": PADDED_HEADS,
            "seq_len": SEQ_LEN,
            "block": block,
        },
        "contract": (
            "source T.gemm is BF16 x BF16 with an FP32 accumulator; the golden keeps "
            "the operands in their storage dtype and requests an FP32 accumulator"
        ),
        "checks": checks,
        "residual_attribution": residual,
        "residual_note": (
            "The kernel accumulates each tile's value GEMM into the already-rescaled FP32 "
            "acc_o, so the running partial sum joins the wgmma accumulator chain. torch.bmm "
            "always starts from zero and the tile is added afterwards; torch.baddbmm rejects "
            "an FP32 accumulator with BF16 operands, and emulating the k=16 wgmma steps "
            "measured further from the kernel, not closer. torch.exp and Tensor.sum are the "
            "closest available exp and reduction orders --- an exp2-based fast exp and "
            "pairwise/sequential reductions were each measured worse. This is the floor for "
            "plain Torch."
        ),
        "problems": problems,
        "passed": not problems,
    }
