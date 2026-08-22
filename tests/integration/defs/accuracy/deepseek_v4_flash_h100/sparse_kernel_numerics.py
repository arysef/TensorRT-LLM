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
"""Where, exactly, the SM90 sparse-MLA kernel disagrees with the reference.

A whole-kernel comparison reports a number; it cannot say which arithmetic
produced it. This walks the online-softmax recurrence stage by stage against the
independent pure-Torch golden --- scores, attention weights before and after the
``acc_s_cast`` rounding, the denominator, the FP32 output accumulator, and the
stored BF16 result --- and then sweeps the spellings each stage could plausibly
have used. Every case is a fixed seed and one GPU, so the whole thing is a few
seconds and reproduces exactly.

Two stages are *gating* here, because a disagreement in them is a defect rather
than a rounding order:

``scores``
    Both sides run BF16 operands into an FP32 accumulator over the same
    contraction, so they must agree bit for bit.
``attention_weights``
    The exponential has more than one legal implementation and only one correct
    one. Triton lowers ``tl.exp`` to the hardware ``ex2.approx`` sequence, which
    carries ~15 FP32 ulp and disagrees with ``torch.exp`` on 86% of the
    exponents this softmax produces; ``libdevice.exp`` is correctly rounded and
    bit-identical to it. That is the difference between a faithful attention
    weight and an approximate one, and nothing downstream can recover it, so it
    is asserted rather than measured.

The remaining stages are reported, not gated. ``sum_exp`` and the output
accumulator differ by the reduction and contraction order the two backends
choose, which is not a property either implementation can dictate to the other,
and the sweeps below record what does and does not move them.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

HEAD_DIM = 512
HEADS = 8
BLOCK_N = 64
BLOCK_H = 16
POOL_ROWS = 4096
LIVE_SLOTS = 192
SEEDS = tuple(range(1000, 1012))


@triton.jit
def _instrumented_kernel(
    q_ptr,
    pool_ptr,
    idx_ptr,
    sink_ptr,
    scores_ptr,
    probs_ptr,
    sum_exp_ptr,
    acc_ptr,
    q_stride_t,
    q_stride_h,
    idx_stride_t,
    num_heads,
    num_selected,
    softmax_scale,
    EXP_MODE: tl.constexpr,
    ACC_MODE: tl.constexpr,
    SUM_MODE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """The shipped recurrence with every stage stored and three axes switchable.

    Deliberately a copy of the production kernel rather than an import of it:
    the point is to vary the arithmetic, and a production kernel that grew
    ``constexpr`` switches so a test could sweep them would be carrying test
    scaffolding into the shipped path. ``EXP_MODE=0`` reproduces production.
    """
    token = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    head_mask = offs_h < num_heads
    q = tl.load(
        q_ptr + token * q_stride_t + offs_h[:, None] * q_stride_h + offs_d[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_H], dtype=tl.float32)
    run_max = tl.full([BLOCK_H], float("-inf"), tl.float32)

    for start in range(0, num_selected, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        idx = tl.load(idx_ptr + token * idx_stride_t + offs_n, mask=offs_n < num_selected, other=-1)
        valid = idx >= 0
        rows = tl.where(valid, idx, 0)[:, None].to(tl.int64) * HEAD_DIM + offs_d[None, :]
        kv = tl.load(pool_ptr + rows, mask=valid[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(valid[None, :], scores, float("-inf"))
        prev_max = run_max
        run_max = tl.maximum(run_max, tl.max(scores, 1))

        if EXP_MODE == 0:
            rescale = tl.where(run_max == float("-inf"), 0.0, libdevice.exp(prev_max - run_max))
            probs = tl.where(valid[None, :], libdevice.exp(scores - run_max[:, None]), 0.0)
        elif EXP_MODE == 1:
            rescale = tl.where(run_max == float("-inf"), 0.0, tl.exp(prev_max - run_max))
            probs = tl.where(valid[None, :], tl.exp(scores - run_max[:, None]), 0.0)
        else:
            log2e = 1.4426950408889634
            rescale = tl.where(run_max == float("-inf"), 0.0, tl.exp2((prev_max - run_max) * log2e))
            probs = tl.where(valid[None, :], tl.exp2((scores - run_max[:, None]) * log2e), 0.0)

        if SUM_MODE == 0:
            tile_sum = tl.sum(probs, 1)
        elif SUM_MODE == 1:
            lo, hi = tl.split(tl.permute(tl.reshape(probs, [BLOCK_H, 2, BLOCK_N // 2]), 0, 2, 1))
            tile_sum = tl.sum(lo, 1) + tl.sum(hi, 1)
        else:
            tile_sum = tl.sum(probs.to(kv.dtype).to(tl.float32), 1)
        sum_exp = sum_exp * rescale + tile_sum

        if ACC_MODE == 0:
            acc = acc * rescale[:, None]
            acc = tl.dot(probs.to(kv.dtype), kv, acc)
        else:
            tile = tl.dot(probs.to(kv.dtype), kv)
            acc = acc * rescale[:, None] + tile

        store_mask = head_mask[:, None] & (offs_n < num_selected)[None, :]
        tl.store(
            scores_ptr + offs_h[:, None] * num_selected + offs_n[None, :], scores, mask=store_mask
        )
        tl.store(
            probs_ptr + offs_h[:, None] * num_selected + offs_n[None, :], probs, mask=store_mask
        )

    sink = tl.load(sink_ptr + offs_h, mask=head_mask, other=float("-inf"))
    if EXP_MODE == 1:
        sum_exp = sum_exp + tl.exp(sink - run_max)
    else:
        sum_exp = sum_exp + libdevice.exp(sink - run_max)
    tl.store(sum_exp_ptr + offs_h, sum_exp, mask=head_mask)
    tl.store(acc_ptr + offs_h[:, None] * HEAD_DIM + offs_d[None, :], acc, mask=head_mask[:, None])


def _kernel_stages(
    q: torch.Tensor,
    pool: torch.Tensor,
    idx: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    *,
    exp_mode: int = 0,
    acc_mode: int = 0,
    sum_mode: int = 0,
) -> dict[str, torch.Tensor]:
    n = idx.shape[1]
    scores = torch.zeros(BLOCK_H, n, device=q.device, dtype=torch.float32)
    probs = torch.zeros(BLOCK_H, n, device=q.device, dtype=torch.float32)
    sum_exp = torch.zeros(BLOCK_H, device=q.device, dtype=torch.float32)
    acc = torch.zeros(BLOCK_H, HEAD_DIM, device=q.device, dtype=torch.float32)
    _instrumented_kernel[(1,)](
        q,
        pool,
        idx,
        sink,
        scores,
        probs,
        sum_exp,
        acc,
        q.stride(0),
        q.stride(1),
        idx.stride(0),
        HEADS,
        n,
        scale,
        EXP_MODE=exp_mode,
        ACC_MODE=acc_mode,
        SUM_MODE=sum_mode,
        HEAD_DIM=HEAD_DIM,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
    )
    return {
        "scores": scores[:HEADS],
        "probs": probs[:HEADS],
        "sum_exp": sum_exp[:HEADS],
        "acc": acc[:HEADS],
    }


def _golden_stages(
    tg: Any,
    q: torch.Tensor,
    pool: torch.Tensor,
    idx: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> dict[str, torch.Tensor]:
    """The golden's recurrence, opened up to the same intermediates.

    ``torch_goldens.sparse_attention`` returns only the final tensor, so the
    same loop is written out here. It is the golden's arithmetic verbatim ---
    including ``_accumulating_matmul``, so the BF16-operand/FP32-accumulator
    contract is the golden's own and not re-derived.
    """
    n = idx.shape[1]
    acc = torch.zeros(1, HEADS, HEAD_DIM, device=q.device, dtype=torch.float32)
    sum_exp = torch.zeros(1, HEADS, device=q.device, dtype=torch.float32)
    scores_max = torch.full((1, HEADS), float("-inf"), device=q.device, dtype=torch.float32)
    all_scores, all_probs = [], []
    for start in range(0, n, BLOCK_N):
        sub = idx[:, start : start + BLOCK_N]
        valid = sub >= 0
        sel = pool[sub.clamp(min=0).long()]
        sel = torch.where(valid.unsqueeze(-1), sel, sel.new_zeros(()))
        acc_s = tg._accumulating_matmul(q, sel.transpose(1, 2)) * scale
        acc_s = torch.where(valid.unsqueeze(1), acc_s, float("-inf"))
        prev_max, scores_max = scores_max, torch.maximum(scores_max, acc_s.amax(-1))
        rescale = torch.nan_to_num(torch.exp(prev_max - scores_max), nan=0.0)
        probs = torch.nan_to_num(torch.exp(acc_s - scores_max.unsqueeze(-1)), nan=0.0)
        sum_exp = sum_exp * rescale + probs.sum(-1)
        acc = acc * rescale.unsqueeze(-1) + tg._accumulating_matmul(probs.to(q.dtype), sel)
        all_scores.append(acc_s)
        all_probs.append(probs)
    sum_exp = sum_exp + torch.exp(sink.view(1, HEADS) - scores_max)
    return {
        "scores": torch.cat(all_scores, -1)[0],
        "probs": torch.cat(all_probs, -1)[0],
        "sum_exp": sum_exp[0],
        "acc": acc[0],
    }


def _case(seed: int) -> tuple[torch.Tensor, ...]:
    """One decode-shaped case: the shape the eight-rank replay's decode step has."""
    torch.manual_seed(seed)
    pool = (torch.randn(POOL_ROWS, HEAD_DIM, device="cuda") * 0.5).bfloat16()
    q = (torch.randn(1, HEADS, HEAD_DIM, device="cuda") * 0.5).bfloat16()
    sink = torch.randn(HEADS, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, POOL_ROWS - 96, (1, LIVE_SLOTS), device="cuda", dtype=torch.int32)
    return q, pool, idx, sink


def _differ(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a != b).sum())


def exp_implementations() -> dict[str, Any]:
    """How far each available exponential is from the correctly-rounded one.

    Run over the range an online softmax actually produces --- ``score - max``
    is always <= 0 --- rather than a symmetric sweep, so the number is the one
    that applies to this kernel.
    """

    @triton.jit
    def _exp(x_ptr, out_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if MODE == 0:
            y = libdevice.exp(x)
        elif MODE == 1:
            y = tl.exp(x)
        else:
            y = tl.exp2(x * 1.4426950408889634)
        tl.store(out_ptr + offs, y, mask=mask)

    torch.manual_seed(0)
    x = -torch.rand(1 << 22, device="cuda", dtype=torch.float32) * 30.0
    ref = torch.exp(x)
    names = {0: "libdevice.exp", 1: "tl.exp", 2: "tl.exp2(x*log2e)"}
    out: dict[str, Any] = {"samples": int(x.numel()), "range": "[-30, 0]"}
    for mode, name in names.items():
        got = torch.empty_like(x)
        _exp[(triton.cdiv(x.numel(), 1024),)](x, got, x.numel(), MODE=mode, BLOCK=1024)
        differing = _differ(got, ref)
        rel = (got - ref).abs() / ref.abs().clamp_min(1e-30) / torch.finfo(torch.float32).eps
        out[name] = {
            "differing_vs_torch_exp": differing,
            "fraction": round(differing / x.numel(), 6),
            "worst_fp32_ulp": round(float(rel.max()), 2),
        }
    return out


def stage_agreement(tg: Any) -> dict[str, Any]:
    """Kernel vs golden, per stage, summed over the fixed seeds."""
    totals = {k: 0 for k in ("scores", "probs_fp32", "probs_bf16", "sum_exp", "acc")}
    counts = {k: 0 for k in totals}
    for seed in SEEDS:
        q, pool, idx, sink = _case(seed)
        scale = float(HEAD_DIM) ** -0.5
        k = _kernel_stages(q, pool, idx, sink, scale)
        g = _golden_stages(tg, q, pool, idx, sink, scale)
        totals["scores"] += _differ(k["scores"], g["scores"])
        totals["probs_fp32"] += _differ(k["probs"], g["probs"])
        totals["probs_bf16"] += _differ(k["probs"].bfloat16(), g["probs"].bfloat16())
        totals["sum_exp"] += _differ(k["sum_exp"], g["sum_exp"])
        totals["acc"] += _differ(k["acc"], g["acc"])
        counts["scores"] += k["scores"].numel()
        counts["probs_fp32"] += k["probs"].numel()
        counts["probs_bf16"] += k["probs"].numel()
        counts["sum_exp"] += k["sum_exp"].numel()
        counts["acc"] += k["acc"].numel()
    return {
        "seeds": list(SEEDS),
        "stages": {name: {"differing": totals[name], "elements": counts[name]} for name in totals},
    }


def variant_sweeps(tg: Any) -> dict[str, Any]:
    """Does any spelling of the two non-bit-exact stages close the gap?

    ``sum_exp`` and ``acc`` are the only stages left once the scores and the
    attention weights agree bit for bit, so these are the only two axes with
    anything to sweep.
    """
    sweeps: dict[str, Any] = {}
    for axis, modes, labels in (
        ("value_gemm_association", (0, 1), ("accumulate_into", "fresh_tile_then_add")),
        ("denominator_reduction", (0, 1, 2), ("tl_sum", "split_halves", "bf16_cast_numerators")),
    ):
        rows = {}
        for mode, label in zip(modes, labels):
            acc_d = sum_d = 0
            for seed in SEEDS:
                q, pool, idx, sink = _case(seed)
                scale = float(HEAD_DIM) ** -0.5
                kw = {"acc_mode": mode} if axis == "value_gemm_association" else {"sum_mode": mode}
                k = _kernel_stages(q, pool, idx, sink, scale, **kw)
                g = _golden_stages(tg, q, pool, idx, sink, scale)
                acc_d += _differ(k["acc"], g["acc"])
                sum_d += _differ(k["sum_exp"], g["sum_exp"])
            rows[label] = {"acc_differing": acc_d, "sum_exp_differing": sum_d}
        sweeps[axis] = rows
    return sweeps


def index_table_padding(sm90: Any) -> dict[str, Any]:
    """Widening the table with padding slots must not move a single bit.

    Ratio-4 hands the kernel a table padded to the Indexer top-k where the
    source's is only as wide as its live slots, so this is what says the extra
    all-padding tiles are neutral rather than a hidden arithmetic difference.
    """
    rows = {}
    for tokens, live, label in (
        (1, 64, "decode_ratio4"),
        (1, 4, "decode_ratio128"),
        (257, 64, "prefill_ratio4"),
    ):
        torch.manual_seed(23)
        swa = (torch.randn(POOL_ROWS, HEAD_DIM, device="cuda") * 0.5).bfloat16()
        cmp_ = (torch.randn(2048, HEAD_DIM, device="cuda") * 0.5).bfloat16()
        q = (torch.randn(tokens, HEADS, HEAD_DIM, device="cuda") * 0.5).bfloat16()
        sink = torch.randn(HEADS, device="cuda", dtype=torch.float32)
        scale = float(HEAD_DIM) ** -0.5
        window = 128
        swa_idx = torch.arange(window, device="cuda").unsqueeze(0).repeat(tokens, 1).int()
        live_idx = torch.arange(live, device="cuda").unsqueeze(0).repeat(tokens, 1).int()
        pad = torch.full((tokens, 512 - live), -1, device="cuda", dtype=torch.int32)
        narrow = torch.cat([swa_idx, live_idx], 1).contiguous()
        wide = torch.cat([swa_idx, live_idx, pad], 1).contiguous()
        a = sm90.sparse_mla_dual_pool(q, swa, cmp_, narrow, window, scale, attn_sink=sink)
        b = sm90.sparse_mla_dual_pool(q, swa, cmp_, wide, window, scale, attn_sink=sink)
        rows[label] = {
            "narrow_slots": int(narrow.shape[1]),
            "wide_slots": int(wide.shape[1]),
            "differing": _differ(a, b),
            "elements": int(a.numel()),
        }
    return rows


def output_agreement(tg: Any, sm90: Any) -> dict[str, Any]:
    """The shipped kernel's stored BF16 output against the golden's.

    This is the quantity the registered gate judges, measured on the decode
    shape, so the storage-step mean can be read next to the stage numbers that
    produce it.
    """
    per_case, over = [], 0
    for seed in SEEDS:
        q, pool, idx, sink = _case(seed)
        scale = float(HEAD_DIM) ** -0.5
        got = sm90.sparse_mla_dual_pool(q, pool, None, idx, LIVE_SLOTS, scale, attn_sink=sink)
        gold = tg.sparse_attention(
            q.unsqueeze(0), pool.unsqueeze(0), sink, idx.unsqueeze(0), scale
        ).squeeze(0)
        d = (got.float() - gold.float()).abs()
        rms = max(float(gold.float().square().mean().sqrt()), 1e-30)
        eps = torch.finfo(torch.bfloat16).eps
        step = (torch.maximum(got.float().abs(), gold.float().abs()) * eps).clamp_min(rms * eps)
        mean_steps = float((d / step).mean())
        over += mean_steps > 1e-4
        per_case.append(
            {
                "seed": seed,
                "differing": _differ(got, gold),
                "elements": int(got.numel()),
                "mean_abs_in_dtype_steps": round(mean_steps, 8),
            }
        )
    return {
        "registered_mean_step_limit": 1e-4,
        "cases_over_limit": over,
        "cases": len(per_case),
        "per_case": per_case,
    }


def run(tg: Any, sm90: Any) -> dict[str, Any]:
    """Every sweep, plus the verdict on the two stages that are gating."""
    result: dict[str, Any] = {
        "evidence_label": "sparse_kernel_numerics",
        "reference_tier": "minimal_golden",
        "validation_tier": "unit",
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "tokens": 1,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "live_slots": LIVE_SLOTS,
            "block_n": BLOCK_N,
            "block_h": BLOCK_H,
        },
        "exp_implementations": exp_implementations(),
        "stage_agreement": stage_agreement(tg),
        "variant_sweeps": variant_sweeps(tg),
        "index_table_padding": index_table_padding(sm90),
        "output_agreement": output_agreement(tg, sm90),
    }

    problems: list[str] = []
    stages = result["stage_agreement"]["stages"]
    for gating in ("scores", "probs_fp32", "probs_bf16"):
        if stages[gating]["differing"]:
            problems.append(
                f"{gating} disagrees with the golden on {stages[gating]['differing']} of "
                f"{stages[gating]['elements']} values; that stage has one correct answer"
            )
    if result["exp_implementations"]["libdevice.exp"]["differing_vs_torch_exp"]:
        problems.append("libdevice.exp is no longer bit-identical to torch.exp on this range")
    for label, row in result["index_table_padding"].items():
        if row["differing"]:
            problems.append(f"index-table padding is not neutral for {label}: {row['differing']}")
    result["problems"] = problems
    result["passed"] = not problems
    return result
