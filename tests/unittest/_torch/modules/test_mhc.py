# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests for Multi-Head Hyper-Connection (mHC) module
from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.modules.mhc.hyper_connection import HCHead, mHC

BENCH_WARMUP = 50
BENCH_ITERS = 200

timing_stats = defaultdict(dict)


def _mhc_fused_hc_mma_available() -> bool:
    try:
        from tensorrt_llm._torch.modules.mhc.mhc_cuda import _fused_hc_mma_supported

        return _fused_hc_mma_supported()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Vanilla (PyTorch) reference implementations for correctness testing
# ---------------------------------------------------------------------------


def _sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def vanilla_pre_mapping(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    mult: int,
    norm_eps: float,
    eps: float,
    sinkhorn_eps: float,
    post_mult_value: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference pre_mapping implementation in pure PyTorch."""
    assert mult == x.shape[-2]
    residual_flat = x.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + norm_eps).rsqrt()
    scale_expanded = torch.cat(
        [
            scale[0].expand(mult),
            scale[1].expand(mult),
            scale[2].expand(mult * mult),
        ],
    )
    mixes = mixes * scale_expanded + base
    pre_mix = mixes[:, :mult].sigmoid().unsqueeze(-1) + eps
    post_mix = (mixes[:, mult : 2 * mult].sigmoid() * post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * mult :].view(-1, mult, mult)
    res_mix = _sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_iters, eps=sinkhorn_eps)
    layer_input = (x * pre_mix).sum(-2).bfloat16()
    return post_mix, res_mix, layer_input


def vanilla_post_mapping(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Reference post_mapping implementation in pure PyTorch."""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def vanilla_hc_head(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    eps: float,
) -> torch.Tensor:
    """Reference HCHead forward implementation in pure PyTorch."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(-2, -1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


# ---------------------------------------------------------------------------
# Profiling helpers (from bench_dg_vs_fma_nsys.py)
# ---------------------------------------------------------------------------


def profile_fn(fn, warmup=BENCH_WARMUP, iters=BENCH_ITERS):
    """Return dict of {kernel_name: avg_us} for all CUDA kernels."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    result = {}
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0:
            result[evt.key] = evt.self_device_time_total / evt.count
    return result


def profile_fn_total(fn, warmup: int = BENCH_WARMUP, iters: int = BENCH_ITERS) -> float:
    """Return average per-iter kernel time (us) via torch.profiler.

    Sums `self_device_time_total` (microseconds) across every CUDA event
    recorded between start() and stop() and divides by `iters`. This
    captures the true per-iter kernel time and excludes host-side gaps
    between launches (e.g. between post_mapping and pre_mapping in the
    unfused path), regardless of how kernel counts differ across paths.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = sum(evt.self_device_time_total for evt in prof.key_averages())
    return total_us / iters


def sum_kernel_times(timings, filters):
    """Sum times for kernel names matching any filter substring."""
    total = 0.0
    for name, us in timings.items():
        if any(f in name for f in filters):
            total += us
    return total


def sum_all_kernel_times(timings):
    """Sum all GPU kernel times."""
    return sum(timings.values())


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------


def generate_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> dict[str, torch.Tensor | float]:
    """Generate test data for big fuse operator."""
    torch.random.manual_seed(42)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    device = "cuda"

    residual = (
        (torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1

    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def generate_realistic_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> dict[str, torch.Tensor | float]:
    """Generate real-scale mHC data to catch RMS denominator regressions."""
    torch.random.manual_seed(123)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    device = "cuda"

    residual = torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device).bfloat16()
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device) * 0.05
    ).flatten(1, 2)
    hc_scale = torch.tensor([0.10, 0.10, 0.30], dtype=torch.float, device=device)
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 2.0

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def generate_post_data(
    n: int,
    hidden_size: int,
    hc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)

    x = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual = torch.randn((n, hc_mult, hidden_size), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device)

    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


def generate_head_data(
    m: int,
    hidden_size: int,
    hc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)

    x = torch.randn((m, hc_mult, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    hc_fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32, device=device)
    hc_base = torch.randn((hc_mult,), dtype=torch.float32, device=device)
    hc_scale = torch.randn((1,), dtype=torch.float32, device=device)

    return {
        "x": x,
        "hc_fn": hc_fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
    }


# ---------------------------------------------------------------------------
# Correctness + profiling tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 32, 64, 128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_mapping(n: int, hidden_size: int, hc_mult: int):
    test_data = generate_pre_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    test_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=test_data["sinkhorn_repeat"],
        dtype=None,
        eps=test_data["hc_pre_eps"],
        norm_eps=test_data["rms_eps"],
        post_mult_value=test_data["hc_post_mult_value"],
    ).cuda()
    test_module.fn.copy_(test_data["fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    residual = test_data["residual"]

    t = profile_fn(lambda: test_module.pre_mapping(residual))
    total_us = sum_all_kernel_times(t)
    timing_stats[("pre_mapping", n, hidden_size)]["cuda"] = total_us

    post_mix_cuda, comb_mix_cuda, layer_input_cuda = test_module.pre_mapping(residual)
    post_mix_ref, comb_mix_ref, layer_input_ref = vanilla_pre_mapping(
        residual,
        test_data["fn"],
        test_data["hc_scale"],
        test_data["hc_base"],
        hc_mult,
        test_data["rms_eps"],
        test_data["hc_pre_eps"],
        test_data["hc_sinkhorn_eps"],
        test_data["hc_post_mult_value"],
        test_data["sinkhorn_repeat"],
    )
    torch.testing.assert_close(post_mix_ref, post_mix_cuda, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb_mix_ref, comb_mix_cuda, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(layer_input_ref, layer_input_cuda, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("n", [64])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_mapping_pro_hidden_size(n: int, hidden_size: int, hc_mult: int):
    test_data = generate_pre_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    test_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=test_data["sinkhorn_repeat"],
        dtype=None,
        eps=test_data["hc_pre_eps"],
        norm_eps=test_data["rms_eps"],
        post_mult_value=test_data["hc_post_mult_value"],
    ).cuda()
    test_module.fn.copy_(test_data["fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    residual = test_data["residual"]

    post_mix_cuda, comb_mix_cuda, layer_input_cuda = test_module.pre_mapping(residual)
    post_mix_ref, comb_mix_ref, layer_input_ref = vanilla_pre_mapping(
        residual,
        test_data["fn"],
        test_data["hc_scale"],
        test_data["hc_base"],
        hc_mult,
        test_data["rms_eps"],
        test_data["hc_pre_eps"],
        test_data["hc_sinkhorn_eps"],
        test_data["hc_post_mult_value"],
        test_data["sinkhorn_repeat"],
    )
    torch.testing.assert_close(post_mix_ref, post_mix_cuda, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb_mix_ref, comb_mix_cuda, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(layer_input_ref, layer_input_cuda, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("n", [64, 128, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_post_mapping(n: int, hidden_size: int, hc_mult: int):
    test_data = generate_post_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    test_module = mHC(mult=hc_mult, hidden_size=hidden_size, sinkhorn_iters=10)

    t = profile_fn(lambda: test_module.post_mapping(**test_data))
    total_us = sum_all_kernel_times(t)
    timing_stats[("post_mapping", n, hidden_size)]["cuda"] = total_us

    output_cuda = test_module.post_mapping(**test_data)
    output_ref = vanilla_post_mapping(**test_data)
    torch.testing.assert_close(output_ref, output_cuda, rtol=1e-2, atol=0.1)


@pytest.mark.parametrize("n", [1, 32, 128, 512, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_fused_hc(n: int, hidden_size: int, hc_mult: int):
    """Correctness test for mHC.fused_hc.

    fused_hc(x_prev, residual_prev, post_mix_prev, comb_mix_prev) must be
    numerically equivalent to:
        residual_cur = post_mapping(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
        post_mix_cur, comb_mix_cur, layer_input_cur = pre_mapping(residual_cur)

    Uses two distinct mHC modules so that the 'prev' and 'cur' blocks have
    different weights — mirroring the real decoder layer boundary.
    """
    # Generate parameters for the 'current' mHC (consumed by pre_mapping part).
    pre_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    # Generate the incoming (residual_prev, x_prev, post_mix_prev, comb_mix_prev)
    # that the 'previous' block would have emitted.
    torch.random.manual_seed(7)
    device = "cuda"
    x_prev = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual_prev = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size
    ).bfloat16()
    post_mix_prev = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    # --- fused_hc path ---
    (
        residual_cur_f,
        post_mix_cur_f,
        comb_mix_cur_f,
        layer_input_cur_f,
    ) = cur_module.fused_hc(x_prev, residual_prev, post_mix_prev, comb_mix_prev)

    # --- two-step reference (post_mapping then pre_mapping via the same module) ---
    residual_cur_ref = cur_module.post_mapping(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
    post_mix_cur_ref, comb_mix_cur_ref, layer_input_cur_ref = cur_module.pre_mapping(
        residual_cur_ref
    )

    # Timing: fused_hc vs separate (post_mapping + pre_mapping).
    # Both paths sum every CUDA event's self_device_time_total via
    # torch.profiler and divide by the iteration count, so host-side gaps
    # between post_mapping and pre_mapping in the unfused path are excluded.
    def _unfused():
        residual_cur = cur_module.post_mapping(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
        cur_module.pre_mapping(residual_cur)

    fused_us = profile_fn_total(
        lambda: cur_module.fused_hc(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
    )
    unfused_us = profile_fn_total(_unfused)

    timing_stats[("fused_hc", n, hidden_size)]["cuda"] = fused_us
    timing_stats[("fused_hc", n, hidden_size)]["cuda_unfused"] = unfused_us
    speedup = (unfused_us / fused_us) if fused_us > 0 else 0.0
    print(
        f"[fused_hc benchmark] n={n} hidden={hidden_size}  "
        f"fused={fused_us:7.2f}us  unfused={unfused_us:7.2f}us  "
        f"speedup={speedup:.2f}x"
    )

    # fused_hc is a Python-level chain of the same kernels that pre_mapping and
    # post_mapping use (mhc_post_mapping then the bigfuse pre_mapping pipeline).
    # Tolerance matches the baseline post_mapping test (residuals are bf16).
    torch.testing.assert_close(residual_cur_ref, residual_cur_f, rtol=1e-2, atol=0.1)
    torch.testing.assert_close(post_mix_cur_ref, post_mix_cur_f, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(comb_mix_cur_ref, comb_mix_cur_f, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(layer_input_cur_ref, layer_input_cur_f, rtol=1e-3, atol=5e-3)


# Explicit backend coverage. The autotuner picks one tactic per M-bucket at
# warmup; to actually exercise every backend across CI we force each tactic.
# Tactic format mirrors MhcFusedHcRunner: (backend, tile_n, num_k_splits,
# bigfuse_bs, tile_m).
#
# FMA tactics intentionally sweep both ks>1 (cross-CTA atomicAdd into y_acc /
# r_acc) and tile_m>1 (Path F only; multi-token per CTA, which reshapes how
# the atomic accumulation buckets tokens). Keeping ks=1,tm=1 only would leave
# the cross-CTA atomic path uncovered.
_BACKEND_TACTICS_BY_M = {
    64: [
        ("fused_half_mma", 0, 8, 256, 1),
        ("fused_all_mma", 0, 1, 0, 1),
        ("fused_half_fma", 2, 2, 256, 1),  # FMA cross-CTA atomic (ks=2)
        ("fused_half_fma", 2, 4, 256, 1),  # FMA deeper cross-CTA atomic (ks=4)
        ("fused_all_fma", 2, 1, 0, 1),
        ("fused_all_fma", 2, 2, 0, 1),  # Path F ks=2 atomic
        ("fused_all_fma", 2, 1, 0, 2),  # Path F tile_m=2 (multi-token CTA)
    ],
    256: [
        ("fused_half_mma", 0, 4, 256, 1),
        ("fused_all_mma", 0, 1, 0, 1),
        ("fused_half_fma", 4, 1, 256, 1),
        ("fused_half_fma", 2, 2, 256, 1),  # ks=2 atomic
        ("fused_all_fma", 4, 1, 0, 1),
        ("fused_all_fma", 2, 2, 0, 1),  # Path F ks=2 atomic (tn=2 required for ks>1)
        ("fused_all_fma", 4, 1, 0, 2),  # Path F tile_m=2
    ],
    # fused_half_fma is intentionally omitted at M=2048: the runner guards the
    # FMA 2-kernel path to M <= 512 (it stops scaling past that M).
    2048: [
        ("fused_half_mma", 0, 2, 128, 1),
        ("fused_all_mma", 0, 1, 0, 1),
        ("fused_all_fma", 4, 1, 0, 1),
        ("fused_all_fma", 2, 2, 0, 1),  # Path F ks=2 atomic (tn=2 required for ks>1)
        ("fused_all_fma", 4, 1, 0, 4),  # Path F tile_m=4
    ],
}


def test_mhc_fused_hc_mma_tactic_filter_hidden_sizes():
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import (
        _FUSED_HC_HALF_MMA_KS,
        _fused_hc_mma_ks_supported,
    )

    supported_by_hidden_size = {
        hidden_size: {
            ks for ks in _FUSED_HC_HALF_MMA_KS if _fused_hc_mma_ks_supported(hidden_size, ks)
        }
        for hidden_size in (4096, 7168, 8192)
    }

    # After P0 (Path D KS=112 enable + scalar-vec tail in Phase 4), the
    # support trait reduces to `hidden % bf16_vec == 0` and `h_tiles % ks == 0`,
    # so any KS in the table that divides HIDDEN/BLOCK_K is supported.
    # h_tiles(4096) = 64 → KS divisors of 64; h_tiles(7168) = 112 → divisors
    # of 112; hidden=8192 is not in the supported-hidden allowlist.
    assert supported_by_hidden_size[4096] == {1, 2, 4, 8, 16, 32, 64}
    assert supported_by_hidden_size[7168] == {1, 2, 4, 7, 8, 14, 16, 28, 56, 112}
    assert supported_by_hidden_size[8192] == set()


@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_fused_hc_fused_norm(n: int, hidden_size: int, hc_mult: int):
    """mHC.fused_hc with ``norm_weight`` must return layer_input already
    RMSNorm-normalized, matching ``rmsnorm(fused_hc(...)[3], norm_weight,
    norm_eps)`` within bf16 tolerance. The residual / post_mix / comb_mix
    outputs are independent of the norm fold-in and must match bit-identically
    across the two paths.
    """
    pre_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    torch.random.manual_seed(23)
    device = "cuda"
    x_prev = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual_prev = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size
    ).bfloat16()
    post_mix_prev = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    # Unfused: fused_hc + separate RMSNorm.
    residual_u, post_mix_u, comb_mix_u, layer_input_u = cur_module.fused_hc(
        x_prev, residual_prev, post_mix_prev, comb_mix_prev
    )
    norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device) * 0.1 + 1.0
    norm_eps = 1e-6
    li_fp32 = layer_input_u.to(torch.float32)
    inv_rms = torch.rsqrt(li_fp32.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
    layer_input_ref = (li_fp32 * inv_rms).to(torch.bfloat16) * norm_weight

    # Fused: norm folded into layer_input epilogue.
    residual_f, post_mix_f, comb_mix_f, layer_input_f = cur_module.fused_hc(
        x_prev,
        residual_prev,
        post_mix_prev,
        comb_mix_prev,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    # Non-layer_input outputs are unaffected by the norm fold-in *semantically*;
    # bit-equality is not guaranteed because the kFuseNorm branch alters Phase 4
    # scheduling enough to perturb some Phase 3 reduction orderings by sub-ULP.
    # Match the unfused path within fp32 tolerance.
    torch.testing.assert_close(residual_u, residual_f, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(post_mix_u, post_mix_f, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(comb_mix_u, comb_mix_f, rtol=1e-3, atol=5e-3)
    # layer_input_f is bf16(rmsnorm(layer_input_u)); two-stage cast plus weight
    # multiply has the same precision profile as a bf16 RMSNorm op.
    torch.testing.assert_close(layer_input_ref, layer_input_f, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("n", list(_BACKEND_TACTICS_BY_M.keys()))
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
@skip_pre_blackwell
def test_mhc_fused_hc_backends(n: int, hidden_size: int, hc_mult: int):
    """Every wired fused_hc backend sees bit-identical input and is checked
    against one shared torch reference and one golden backend output.

    Paths D (fused_all_mma) and F (fused_all_fma) are single-kernel
    all-in-one variants; fused_half_mma (Path B) and fused_half_fma (Path E)
    are the 2-kernel baselines. Each is forced by calling
    MhcFusedHcRunner.forward directly with an explicit tactic, bypassing the
    autotuner.

    Path C (bigfuse tcgen05) is not covered: its kernel emits (D_next,
    sqr_sum_next, layer_input) as the layer-to-layer state carrier and does
    not produce post_mix_cur / comb_mix_cur, so it cannot be dropped in
    behind the current mhc_fused_hc API without a kernel-side modification
    that adds post_mix_out / comb_mix_out stores.
    """
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import MhcFusedHcRunner

    pre_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    torch.random.manual_seed(13)
    device = "cuda"
    # Canonical input tensors — generated once, then deep-cloned per consumer
    # so the torch ref and each backend each get an independent byte-identical
    # copy. Protects the test from any hypothetical in-place mutation inside
    # a kernel launcher or a contiguous() call.
    x_prev_ref = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual_prev_ref = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size
    ).bfloat16()
    post_mix_prev_ref = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev_ref = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    # Torch ground-truth — computed from clones so the ref path cannot
    # perturb the canonical input tensors either.
    residual_cur_ref = cur_module.post_mapping(
        x_prev_ref.clone(),
        residual_prev_ref.clone(),
        post_mix_prev_ref.clone(),
        comb_mix_prev_ref.clone(),
    )
    post_mix_ref, comb_mix_ref, layer_input_ref = cur_module.pre_mapping(residual_cur_ref.clone())

    runner = MhcFusedHcRunner(
        n=hc_mult,
        hidden_size=hidden_size,
        rms_eps=pre_data["rms_eps"],
        hc_pre_eps=pre_data["hc_pre_eps"],
        hc_sinkhorn_eps=pre_data["hc_sinkhorn_eps"],
        hc_post_mult_value=pre_data["hc_post_mult_value"],
        sinkhorn_repeat=pre_data["sinkhorn_repeat"],
    )

    def make_runner_inputs():
        return [
            x_prev_ref.clone(),
            residual_prev_ref.reshape(n, hc_mult, hidden_size).clone().contiguous(),
            post_mix_prev_ref.reshape(n, hc_mult).clone().contiguous(),
            comb_mix_prev_ref.reshape(n, hc_mult, hc_mult).clone().contiguous(),
            cur_module.fn.detach().clone().contiguous(),
            cur_module.scale.detach().clone(),
            cur_module.base.detach().clone(),
        ]

    mma_available = _mhc_fused_hc_mma_available()
    tactics = [
        tactic
        for tactic in _BACKEND_TACTICS_BY_M[n]
        if mma_available or not tactic[0].endswith("_mma")
    ]

    tactic_outputs = {}
    for tactic in tactics:
        residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = runner(
            inputs=make_runner_inputs(), tactic=tactic
        )
        tactic_outputs[tactic] = (
            residual_cur,
            post_mix_cur.view(n, hc_mult, 1),
            comb_mix_cur.view(n, hc_mult, hc_mult),
            layer_input_cur,
        )

    # Tolerances: bf16 has a 7-bit mantissa so 1 ulp ~ 7.8e-3. For outputs
    # near unit scale with fp32-accumulated reductions, rtol=1e-2 atol=1e-2
    # is the expected bf16 parity — tighter than test_mhc_post_mapping's
    # atol=0.1 which compared against a pure-bf16 vanilla reference.
    bf16_tol = dict(rtol=1e-2, atol=1e-2)
    fp32_tol = dict(rtol=1e-3, atol=5e-3)

    # (1) Every backend must match the torch reference.
    for tactic, (
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
    ) in tactic_outputs.items():
        torch.testing.assert_close(
            residual_cur_ref,
            residual_cur,
            **bf16_tol,
            msg=f"[vs torch-ref] tactic={tactic} n={n} hidden={hidden_size} residual mismatch",
        )
        torch.testing.assert_close(
            post_mix_ref,
            post_mix_cur,
            **fp32_tol,
            msg=f"[vs torch-ref] tactic={tactic} n={n} hidden={hidden_size} post_mix mismatch",
        )
        torch.testing.assert_close(
            comb_mix_ref,
            comb_mix_cur,
            **fp32_tol,
            msg=f"[vs torch-ref] tactic={tactic} n={n} hidden={hidden_size} comb_mix mismatch",
        )
        torch.testing.assert_close(
            layer_input_ref,
            layer_input_cur,
            **bf16_tol,
            msg=f"[vs torch-ref] tactic={tactic} n={n} hidden={hidden_size} layer_input mismatch",
        )

    # (2) All backends must agree with one golden backend at the same tolerance
    # as vs the torch ref. Different backends vary only in tile shape and
    # reduction order, so cross-backend divergence would indicate a kernel
    # correctness bug rather than expected rounding drift.
    gold = next(
        (tactic for tactic in tactic_outputs if tactic[0] == "fused_half_mma"),
        next(iter(tactic_outputs)),
    )
    gr, gpm, gcm, gli = tactic_outputs[gold]
    for tactic, (
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
    ) in tactic_outputs.items():
        if tactic == gold:
            continue
        torch.testing.assert_close(
            gr,
            residual_cur,
            **bf16_tol,
            msg=f"[vs {gold}] tactic={tactic} n={n} hidden={hidden_size} residual mismatch",
        )
        torch.testing.assert_close(
            gpm,
            post_mix_cur,
            **fp32_tol,
            msg=f"[vs {gold}] tactic={tactic} n={n} hidden={hidden_size} post_mix mismatch",
        )
        torch.testing.assert_close(
            gcm,
            comb_mix_cur,
            **fp32_tol,
            msg=f"[vs {gold}] tactic={tactic} n={n} hidden={hidden_size} comb_mix mismatch",
        )
        torch.testing.assert_close(
            gli,
            layer_input_cur,
            **bf16_tol,
            msg=f"[vs {gold}] tactic={tactic} n={n} hidden={hidden_size} layer_input mismatch",
        )


@pytest.mark.parametrize(
    "tactic",
    [
        pytest.param(("fused_half_mma", 0, 1, 256, 1), marks=skip_pre_blackwell),
        ("fused_half_fma", 2, 1, 256, 1),
        pytest.param(("fused_all_mma", 0, 1, 0, 1), marks=skip_pre_blackwell),
        ("fused_all_fma", 2, 1, 0, 1),
    ],
)
@pytest.mark.parametrize("hidden_size", [4096, 7168])
def test_mhc_fused_hc_realistic_scale_regression(tactic, hidden_size: int):
    """Real-scale mHC data catches fused_hc RMS normalization regressions."""
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import MhcFusedHcRunner

    if tactic[0].endswith("_mma") and not _mhc_fused_hc_mma_available():
        pytest.skip("mHC fused-HC MMA kernels require SM100 and BUILD_DEEP_GEMM=ON")

    n = 16
    hc_mult = 4
    pre_data = generate_realistic_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    torch.random.manual_seed(17)
    device = "cuda"
    x_prev = torch.randn((n, hidden_size), dtype=torch.float, device=device).bfloat16()
    residual_prev = torch.randn(
        (n, hc_mult, hidden_size), dtype=torch.float, device=device
    ).bfloat16()
    post_mix_prev = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    residual_cur_ref = cur_module.post_mapping(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
    post_mix_ref, comb_mix_ref, layer_input_ref = cur_module.pre_mapping(residual_cur_ref)

    runner = MhcFusedHcRunner(
        n=hc_mult,
        hidden_size=hidden_size,
        rms_eps=pre_data["rms_eps"],
        hc_pre_eps=pre_data["hc_pre_eps"],
        hc_sinkhorn_eps=pre_data["hc_sinkhorn_eps"],
        hc_post_mult_value=pre_data["hc_post_mult_value"],
        sinkhorn_repeat=pre_data["sinkhorn_repeat"],
    )

    residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = runner(
        inputs=[
            x_prev.contiguous(),
            residual_prev.contiguous(),
            post_mix_prev.view(n, hc_mult).contiguous(),
            comb_mix_prev.contiguous(),
            cur_module.fn.detach().contiguous(),
            cur_module.scale.detach().contiguous(),
            cur_module.base.detach().contiguous(),
        ],
        tactic=tactic,
    )

    torch.testing.assert_close(residual_cur_ref, residual_cur, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(post_mix_ref, post_mix_cur.view(n, hc_mult, 1), rtol=3e-3, atol=5e-3)
    torch.testing.assert_close(
        comb_mix_ref, comb_mix_cur.view(n, hc_mult, hc_mult), rtol=3e-3, atol=5e-3
    )
    torch.testing.assert_close(layer_input_ref, layer_input_cur, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
@skip_pre_blackwell
def test_mhc_fused_hc_cuda_graph(n: int, hidden_size: int, hc_mult: int):
    """CUDA-graph capture/replay of mHC.fused_hc.

    The decoder uses fused_hc at every non-first layer boundary; the whole
    decoder is expected to be traced into a single CUDA graph. This test
    verifies that (a) fused_hc can be captured without host syncs, and
    (b) replay produces bit-exact results to eager.

    To keep the bit-exact assertion structurally valid, we drive the runner
    with an explicit ``num_k_splits=1`` tactic (Path B, fused_half_mma).
    That disables split-K atomic accumulation entirely, so none of the four
    outputs depend on the non-deterministic FP ordering that pickKSplits(M)
    would otherwise introduce (it picks ks=16 at M=128 and ks=4 at M=2048
    for the autotuner fallback — atomics active, not deterministic across
    replays).
    """
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import MhcFusedHcRunner

    if not _mhc_fused_hc_mma_available():
        pytest.skip("mHC fused-HC MMA kernels require SM100 and BUILD_DEEP_GEMM=ON")

    pre_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    torch.random.manual_seed(11)
    device = "cuda"
    x_prev = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual_prev = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size
    ).bfloat16()
    post_mix_prev = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    runner = MhcFusedHcRunner(
        n=hc_mult,
        hidden_size=hidden_size,
        rms_eps=pre_data["rms_eps"],
        hc_pre_eps=pre_data["hc_pre_eps"],
        hc_sinkhorn_eps=pre_data["hc_sinkhorn_eps"],
        hc_post_mult_value=pre_data["hc_post_mult_value"],
        sinkhorn_repeat=pre_data["sinkhorn_repeat"],
    )
    # Pin tactic to Path B with num_k_splits=1 → no atomic accumulation on
    # any output. Tactic tuple matches MhcFusedHcRunner.get_tactics().
    tactic = ("fused_half_mma", 0, 1, 128, 1)
    assert tactic[2] == 1, "bit-exact assertion requires num_k_splits=1"

    def _inputs():
        return [
            x_prev,
            residual_prev.reshape(n, hc_mult, hidden_size).contiguous(),
            post_mix_prev.reshape(n, hc_mult).contiguous(),
            comb_mix_prev.reshape(n, hc_mult, hc_mult).contiguous(),
            cur_module.fn,
            cur_module.scale,
            cur_module.base,
        ]

    # Clone the eager result into an immutable golden before warmup and graph
    # capture exercise additional allocations.
    eager_raw = runner(inputs=_inputs(), tactic=tactic)
    eager_out = tuple(t.clone() for t in eager_raw)

    # Warm up on a side stream — required for CUDA graph capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            runner(inputs=_inputs(), tactic=tactic)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_out = runner(inputs=_inputs(), tactic=tactic)

    # Replay — outputs should update in place.
    g.replay()
    torch.cuda.synchronize()

    # With ks=1 the kernel has no atomic accumulation anywhere, so replay must
    # be bit-exact against eager on all four outputs.
    for ge, ee, name in zip(
        graph_out, eager_out, ["residual", "post_mix", "comb_mix", "layer_input"]
    ):
        torch.testing.assert_close(
            ge, ee, rtol=0, atol=0, msg=f"fused_hc CUDA-graph mismatch in {name}"
        )

    # Mutate inputs in-place and replay; result should follow — proves the graph
    # is parameterised by input storage, not cached constants.
    x_prev.mul_(1.001)
    residual_prev.mul_(1.001)
    post_mix_prev.mul_(1.001)
    comb_mix_prev.mul_(1.001)
    eager_raw2 = runner(inputs=_inputs(), tactic=tactic)
    eager_out2 = tuple(t.clone() for t in eager_raw2)
    g.replay()
    torch.cuda.synchronize()
    for ge, ee, name in zip(
        graph_out, eager_out2, ["residual", "post_mix", "comb_mix", "layer_input"]
    ):
        torch.testing.assert_close(
            ge, ee, rtol=0, atol=0, msg=f"fused_hc CUDA-graph replay mismatch in {name}"
        )


def _make_fused_hc_runner_case(n: int, hidden_size: int, hc_mult: int, seed: int):
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import MhcFusedHcRunner

    pre_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    torch.random.manual_seed(seed)
    device = "cuda"
    x_prev = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual_prev = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size
    ).bfloat16()
    post_mix_prev = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    runner = MhcFusedHcRunner(
        n=hc_mult,
        hidden_size=hidden_size,
        rms_eps=pre_data["rms_eps"],
        hc_pre_eps=pre_data["hc_pre_eps"],
        hc_sinkhorn_eps=pre_data["hc_sinkhorn_eps"],
        hc_post_mult_value=pre_data["hc_post_mult_value"],
        sinkhorn_repeat=pre_data["sinkhorn_repeat"],
    )

    inputs = [
        x_prev,
        residual_prev.reshape(n, hc_mult, hidden_size).contiguous(),
        post_mix_prev.reshape(n, hc_mult).contiguous(),
        comb_mix_prev.reshape(n, hc_mult, hc_mult).contiguous(),
        cur_module.fn,
        cur_module.scale,
        cur_module.base,
    ]
    return runner, inputs


@pytest.mark.parametrize(
    "backend,num_k_splits,tile_m,expected_shapes",
    [
        ("fused_half_mma", 56, 1, ((129, 24), (129,), (1,))),
        ("fused_half_fma", 4, 1, ((4, 129, 24), (4, 129), (1,))),
        ("fused_all_mma", 56, 1, ((129, 24), (129,), (3,))),
        ("fused_all_fma", 2, 4, ((129, 24), (129,), (33,))),
    ],
)
def test_mhc_fused_hc_allocates_minimal_scratch(
    backend: str,
    num_k_splits: int,
    tile_m: int,
    expected_shapes: tuple[tuple[int, ...], ...],
) -> None:
    from tensorrt_llm._torch.modules.mhc import mhc_cuda

    alloc_scratch = mhc_cuda._alloc_fused_hc_scratch

    scratch = alloc_scratch(
        backend=backend,
        B=129,
        n=4,
        num_k_splits=num_k_splits,
        tile_m=tile_m,
        device=torch.device("cpu"),
    )

    assert tuple(tuple(tensor.shape) for tensor in scratch) == expected_shapes


def test_mhc_fused_hc_preserves_chained_inputs():
    """A fused-HC call must not overwrite state returned by the previous call."""
    runner, inputs = _make_fused_hc_runner_case(n=6, hidden_size=7168, hc_mult=4, seed=61)
    tactic = ("fused_half_fma", 2, 4, 512, 1)

    first_outputs = runner(inputs=inputs, tactic=tactic)
    chained_inputs = [
        inputs[0],
        first_outputs[0],
        first_outputs[1],
        first_outputs[2],
        *inputs[4:],
    ]
    saved_inputs = tuple(tensor.clone() for tensor in chained_inputs[1:4])

    second_outputs = runner(inputs=chained_inputs, tactic=tactic)

    for actual, expected, name in zip(
        chained_inputs[1:4],
        saved_inputs,
        ("residual", "post_mix", "comb_mix"),
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            msg=f"fused_hc mutated its {name} input",
        )

    reference_inputs = [inputs[0].clone(), *saved_inputs, *inputs[4:]]
    reference_outputs = runner(inputs=reference_inputs, tactic=tactic)
    for actual, expected, name, tolerance in zip(
        second_outputs,
        reference_outputs,
        ("residual", "post_mix", "comb_mix", "layer_input"),
        (1e-2, 5e-3, 5e-3, 1e-2),
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
            msg=f"chained fused_hc produced an incorrect {name}",
        )


def test_mhc_fused_hc_three_call_cuda_graph_replay():
    """An odd-length fused-HC chain must remain stable across graph replays."""
    runner, inputs = _make_fused_hc_runner_case(n=6, hidden_size=4096, hc_mult=4, seed=73)
    tactic = ("fused_half_fma", 2, 1, 512, 1)

    def run_chain():
        first = runner(inputs=inputs, tactic=tactic)
        second = runner(
            inputs=[inputs[0], first[0], first[1], first[2], *inputs[4:]],
            tactic=tactic,
        )
        return runner(
            inputs=[inputs[0], second[0], second[1], second[2], *inputs[4:]],
            tactic=tactic,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run_chain()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = run_chain()

    for scale in (1.0003, 1.0005, 1.0007):
        for tensor in inputs[:4]:
            tensor.mul_(scale)
        expected = tuple(tensor.clone() for tensor in run_chain())
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference, name in zip(
            graph_outputs,
            expected,
            ("residual", "post_mix", "comb_mix", "layer_input"),
        ):
            torch.testing.assert_close(
                actual,
                reference,
                rtol=0,
                atol=0,
                msg=f"three-call CUDA graph replay mismatch in {name}",
            )


def test_mhc_fused_hc_concurrent_streams() -> None:
    """Concurrent fused-HC calls on separate streams must remain independent."""
    runner, first_inputs = _make_fused_hc_runner_case(n=6, hidden_size=4096, hc_mult=4, seed=79)
    second_inputs = [tensor.clone() for tensor in first_inputs]
    for tensor in second_inputs[:4]:
        tensor.mul_(1.01)
    tactic = ("fused_half_fma", 2, 1, 512, 1)

    current_stream = torch.cuda.current_stream()
    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    first_stream.wait_stream(current_stream)
    second_stream.wait_stream(current_stream)

    with torch.cuda.stream(first_stream):
        first_outputs = runner(inputs=first_inputs, tactic=tactic)
    with torch.cuda.stream(second_stream):
        second_outputs = runner(inputs=second_inputs, tactic=tactic)

    current_stream.wait_stream(first_stream)
    current_stream.wait_stream(second_stream)
    torch.cuda.synchronize()

    first_reference = tuple(tensor.clone() for tensor in runner(inputs=first_inputs, tactic=tactic))
    second_reference = tuple(
        tensor.clone() for tensor in runner(inputs=second_inputs, tactic=tactic)
    )
    for actual_outputs, reference_outputs in (
        (first_outputs, first_reference),
        (second_outputs, second_reference),
    ):
        for actual, reference in zip(actual_outputs, reference_outputs):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def _assert_graph_replay_matches_eager(runner, inputs, tactic):
    runner(inputs=inputs, tactic=tactic)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = runner(inputs=inputs, tactic=tactic)
    graph.replay()
    torch.cuda.synchronize()

    for tensor in inputs[:4]:
        tensor.mul_(1.0007)

    eager_out = tuple(t.clone() for t in runner(inputs=inputs, tactic=tactic))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    for actual, expected, name in zip(
        graph_out, eager_out, ("residual", "post_mix", "comb_mix", "layer_input")
    ):
        # splitK atomics are not bit-exact, but stale graph pointers or bad
        # workspace reuse would diverge far beyond this bf16-scale tolerance.
        torch.testing.assert_close(
            actual,
            expected,
            rtol=3e-2,
            atol=5e-2,
            msg=f"fused_hc CUDA-graph replay mismatch in {name}; tactic={tactic}",
        )


@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize(
    "tactic",
    [
        # Both tactics are tcgen05 TF32 MMA, which `fused_tf32_pmap_gemm.cuh`
        # hard-asserts to sm_100a. Requesting one on Hopper aborts the process
        # and poisons the CUDA context for every later test in the session, so
        # they carry the same marker the MMA params of
        # `test_mhc_fused_hc_realistic_scale_regression` already use.
        pytest.param(("fused_half_mma", 0, 64, 512, 1), marks=skip_pre_blackwell),
        pytest.param(("fused_all_mma", 0, 64, 0, 1), marks=skip_pre_blackwell),
    ],
)
def test_mhc_fused_hc_cuda_graph_high_splitk_tactics(n: int, tactic):
    """Reduced autotune maps decode buckets to high-splitK MMA tactics.

    Unlike the bit-exact ks=1 graph test above, this covers the actual
    M=64/128 PR autotune path where splitK atomics and CUDA graph replay are
    both active.
    """
    runner, inputs = _make_fused_hc_runner_case(n=n, hidden_size=4096, hc_mult=4, seed=41 + n)
    _assert_graph_replay_matches_eager(runner, inputs, tactic)


@skip_pre_blackwell
def test_mhc_fused_hc_cuda_graph_decode_buckets_then_prefill():
    """Capture reduced-autotune decode buckets, then run the large prefill path.

    The CI failure happened after CUDA graph warmup for decode buckets and on
    the subsequent M=8192 warmup. This test keeps that ordering local to
    fused_hc and covers both PR-selected M=8192 MMA tactics.

    Every tactic it names is tcgen05 TF32 MMA, which `fused_tf32_pmap_gemm.cuh`
    hard-asserts to sm_100a; on Hopper the abort also poisons the CUDA context
    for the rest of the session, so the whole test is Blackwell-only.
    """
    hidden_size = 4096
    hc_mult = 4

    for n in (128, 64):
        for tactic in (
            ("fused_half_mma", 0, 64, 512, 1),
            ("fused_all_mma", 0, 64, 0, 1),
        ):
            runner, inputs = _make_fused_hc_runner_case(
                n=n, hidden_size=hidden_size, hc_mult=hc_mult, seed=100 + n
            )
            _assert_graph_replay_matches_eager(runner, inputs, tactic)

    runner, inputs = _make_fused_hc_runner_case(
        n=8192, hidden_size=hidden_size, hc_mult=hc_mult, seed=8192
    )
    for tactic in (
        ("fused_half_mma", 0, 1, 128, 1),
        ("fused_half_mma", 0, 1, 256, 1),
        ("fused_all_mma", 0, 1, 0, 1),
    ):
        outputs = runner(inputs=inputs, tactic=tactic)
        torch.cuda.synchronize()
        for output in outputs:
            assert torch.isfinite(output.float()).all(), (
                f"non-finite fused_hc output; tactic={tactic}"
            )


@pytest.mark.parametrize("m", [64, 128, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head(m: int, hidden_size: int, hc_mult: int):
    test_data = generate_head_data(
        m=m,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    test_module = HCHead(mult=hc_mult, hidden_size=hidden_size).cuda()
    test_module.fn.copy_(test_data["hc_fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    t = profile_fn(lambda: test_module(test_data["x"]))
    total_us = sum_all_kernel_times(t)
    timing_stats[("hc_head", m, hidden_size)]["cuda"] = total_us

    output_cuda = test_module(test_data["x"])
    output_ref = vanilla_hc_head(
        test_data["x"],
        test_data["hc_fn"],
        test_data["hc_scale"],
        test_data["hc_base"],
        norm_eps=1e-6,
        eps=1e-6,
    )
    torch.testing.assert_close(output_ref, output_cuda, rtol=1e-2, atol=0.1)


# ---------------------------------------------------------------------------
# Low-level pre_mapping pipeline benchmark: DG / DG-s16 / FMA
# ---------------------------------------------------------------------------

HC_MULT = 4
HIDDEN_SIZE = 4096
_N = HC_MULT * (HC_MULT + 1 + 1)  # 24
_K = HC_MULT * HIDDEN_SIZE  # 16384
_NUM_SPLITS = 16
_SINKHORN_REPEAT = 20


def _try_import_backends():
    """Return (tf32_hc_prenorm_gemm|None, mhc_gemm_rms_fma_cuda|None,
    mhc_big_fuse_cuda|None)."""
    tf32_hc_prenorm_gemm = None
    try:
        from deep_gemm import tf32_hc_prenorm_gemm
    except ImportError:
        try:
            from tensorrt_llm.deep_gemm import tf32_hc_prenorm_gemm
        except ImportError:
            pass

    mhc_gemm_rms_fma_cuda = mhc_big_fuse_cuda = None
    try:
        from tensorrt_llm._torch.modules.mhc.mhc_cuda import (
            mhc_big_fuse_cuda,
            mhc_gemm_rms_fma_cuda,
        )
    except Exception:
        pass

    return (
        tf32_hc_prenorm_gemm,
        mhc_gemm_rms_fma_cuda,
        mhc_big_fuse_cuda,
    )


def run_bench_pre_mapping(M: int) -> dict:
    """Low-level kernel benchmark for one M: profiles GEMM + BigFuse per backend.
    Returns dict like {"DG": (gemm_us, fuse_us), "FMA": (...), ...}.
    """
    device = "cuda"
    (
        tf32_hc_prenorm_gemm,
        mhc_gemm_rms_fma_cuda,
        mhc_big_fuse_cuda,
    ) = _try_import_backends()

    w_nk = torch.randn(_N, _K, dtype=torch.float32, device=device) * 0.01
    hc_scale = torch.randn(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(_N, dtype=torch.float32, device=device)
    x = (torch.randn(M, _K, dtype=torch.float32, device=device) * 0.01).bfloat16()
    residual = (
        torch.randn(M, HC_MULT, HIDDEN_SIZE, dtype=torch.float32, device=device) / HIDDEN_SIZE
    ).bfloat16()

    times = {}

    if tf32_hc_prenorm_gemm is not None and mhc_big_fuse_cuda is not None:
        y = torch.empty(M, _N, dtype=torch.float32, device=device)
        r = torch.empty(M, dtype=torch.float32, device=device)
        pm = torch.empty(M, HC_MULT, dtype=torch.float32, device=device)
        cm = torch.empty(M, HC_MULT * HC_MULT, dtype=torch.float32, device=device)
        li = torch.empty(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        def dg_fn():
            tf32_hc_prenorm_gemm(x, w_nk, y, r)
            mhc_big_fuse_cuda(
                y,
                r,
                residual,
                hc_scale,
                hc_base,
                pm,
                cm,
                li,
                M,
                _K,
                HIDDEN_SIZE,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                _SINKHORN_REPEAT,
                num_splits=1,
            )

        t = profile_fn(dg_fn)
        times["DG"] = (sum_kernel_times(t, ["hc_prenorm_gemm"]), sum_kernel_times(t, ["BigFuse"]))

        y_s = torch.empty(_NUM_SPLITS, M, _N, dtype=torch.float32, device=device)
        r_s = torch.empty(_NUM_SPLITS, M, dtype=torch.float32, device=device)
        pm_s = torch.empty(M, HC_MULT, dtype=torch.float32, device=device)
        cm_s = torch.empty(M, HC_MULT * HC_MULT, dtype=torch.float32, device=device)
        li_s = torch.empty(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        def dg_s16_fn():
            tf32_hc_prenorm_gemm(x, w_nk, y_s, r_s, num_splits=_NUM_SPLITS)
            mhc_big_fuse_cuda(
                y_s,
                r_s,
                residual,
                hc_scale,
                hc_base,
                pm_s,
                cm_s,
                li_s,
                M,
                _K,
                HIDDEN_SIZE,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                _SINKHORN_REPEAT,
                num_splits=_NUM_SPLITS,
            )

        t = profile_fn(dg_s16_fn)
        times["DG-s16"] = (
            sum_kernel_times(t, ["hc_prenorm_gemm"]),
            sum_kernel_times(t, ["BigFuse"]),
        )

    if mhc_gemm_rms_fma_cuda is not None and mhc_big_fuse_cuda is not None:
        pm_f = torch.empty(M, HC_MULT, dtype=torch.float32, device=device)
        cm_f = torch.empty(M, HC_MULT * HC_MULT, dtype=torch.float32, device=device)
        li_f = torch.empty(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        def fma_fn():
            y_f, r_f = mhc_gemm_rms_fma_cuda(x, None, M, _N, _K, w_t=w_nk)
            mhc_big_fuse_cuda(
                y_f,
                r_f,
                residual,
                hc_scale,
                hc_base,
                pm_f,
                cm_f,
                li_f,
                M,
                _K,
                HIDDEN_SIZE,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                _SINKHORN_REPEAT,
                num_splits=1,
            )

        t = profile_fn(fma_fn)
        times["FMA"] = (sum_kernel_times(t, ["GemmSqrsumFma"]), sum_kernel_times(t, ["BigFuse"]))

    return times


def _print_bench_timing_table(bench_entries: dict):
    """Print the pre_mapping pipeline (GEMM + BigFuse) benchmark table."""
    if not bench_entries:
        return
    all_cols = []
    for v in bench_entries.values():
        for c in v:
            if c not in all_cols:
                all_cols.append(c)
    print("\nPRE_MAPPING PIPELINE (GEMM + BigFuse)")
    header = f"  {'M':>6s}"
    for c in all_cols:
        header += f"  {c:>16s}"
    header += f"  {'best':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in sorted(bench_entries):
        _, M, _ = key
        times = bench_entries[key]
        totals = {c: times[c][0] + times[c][1] for c in times}
        best = min(totals, key=totals.get) if totals else "N/A"
        row = f"  {M:6d}"
        for c in all_cols:
            if c in times:
                g, f = times[c]
                row += f"  {g + f:8.1f}({g:4.1f}+{f:4.1f})"
            else:
                row += f"  {'N/A':>16s}"
        row += f"  {best:>8s}"
        print(row)


# ---------------------------------------------------------------------------
# Session-scoped fixture: print timing table at end (pytest only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def print_timing_stats():
    """Print collected GPU profiler timings at end of session."""
    yield

    if not timing_stats:
        return

    print("\n" + "=" * 90)
    print("GPU Kernel Timing (torch.profiler, microseconds)")
    print("=" * 90)

    # --- Per-backend correctness/perf tests (pre_mapping, post_mapping, hc_head) ---
    for test_type in ("pre_mapping", "post_mapping", "fused_hc", "hc_head"):
        entries = {
            k: v for k, v in timing_stats.items() if isinstance(k, tuple) and k[0] == test_type
        }
        if not entries:
            continue

        dim_label = "m" if test_type == "hc_head" else "n"
        print(f"\n{test_type.upper()}")

        all_backends = sorted({b for d in entries.values() for b in d})
        header = f"  {dim_label:>6s}  hidden"
        for b in all_backends:
            header += f"  {b:>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for key in sorted(entries):
            _, dim_val, hidden = key
            row = f"  {dim_val:6d}  {hidden:6d}"
            for b in all_backends:
                us = entries[key].get(b)
                row += f"  {us:10.1f}" if us is not None else f"  {'N/A':>10s}"
            print(row)

    # --- Low-level pipeline bench table (only populated when run via main()) ---
    bench_entries = {
        k: v for k, v in timing_stats.items() if isinstance(k, tuple) and k[0] == "bench_pre"
    }
    _print_bench_timing_table(bench_entries)

    print("\n" + "=" * 90)


def main():
    """Run pre_mapping pipeline benchmark (GEMM + BigFuse) for various M.
    Invoked when running: python test_mhc.py
    """
    torch.manual_seed(42)
    bench_M = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    bench_stats = {}
    for M in bench_M:
        bench_stats[("bench_pre", M, HIDDEN_SIZE)] = run_bench_pre_mapping(M)

    print("\n" + "=" * 90)
    print("GPU Kernel Timing (torch.profiler, microseconds) — benchmark only")
    print("=" * 90)
    _print_bench_timing_table(bench_stats)
    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# DeepSeek-V4 on Hopper (SM90): the FMA ladder is the only fused path, and it
# has to agree with the checkpoint's own mHC rather than only with itself.
#
# The tcgen05 TF32 paths ("fused_*_mma") need SM100, so on Hopper the module
# must fall back to the FP32 FMA kernels. These tests carry `sm90` in their
# names so the DeepSeek-V4 Hopper gate selects them.
# ---------------------------------------------------------------------------


def _dsv4_mhc_goldens():
    import importlib.util
    import sys
    from pathlib import Path

    name = "deepseek_v4_flash_h100_torch_goldens"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[3]
        / "integration"
        / "defs"
        / "accuracy"
        / "deepseek_v4_flash_h100"
        / "torch_goldens.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dsv4_mhc_tolerance():
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "integration"
        / "defs"
        / "accuracy"
        / "deepseek_v4_flash_h100"
        / "manifests"
        / "tolerances.json"
    )
    return json.loads(path.read_text())["modules"]["mhc"]


# Checkpoint mHC contract: multiplier 4, 20 Sinkhorn iterations, eps 1e-6,
# and the source's `post = 2 * sigmoid(...)` expansion gain.
_DSV4_HC_MULT = 4
_DSV4_SINKHORN_ITERS = 20
_DSV4_EPS = 1e-6
_DSV4_POST_MULT = 2.0


def _dsv4_mhc_module(hidden_size: int, data):
    module = mHC(
        mult=_DSV4_HC_MULT,
        hidden_size=hidden_size,
        sinkhorn_iters=_DSV4_SINKHORN_ITERS,
        dtype=None,
        eps=_DSV4_EPS,
        norm_eps=_DSV4_EPS,
        sinkhorn_eps=_DSV4_EPS,
        post_mult_value=_DSV4_POST_MULT,
    ).cuda()
    module.fn.copy_(data["fn"])
    module.scale.copy_(data["hc_scale"])
    module.base.copy_(data["hc_base"])
    return module


@pytest.mark.skipif(
    torch.cuda.is_available() and _mhc_fused_hc_mma_available(),
    reason="MMA tactics are available; this is the pre-Blackwell expectation",
)
def test_mhc_sm90_offers_only_fma_tactics_and_falls_back_to_fma():
    """On Hopper the tcgen05 TF32 mHC paths must be absent, not merely unused.

    Selecting an MMA tactic on SM90 fails at capture or launch time rather
    than degrading, so the autotuner's *candidate list* is what has to be
    clean --- checking only the chosen tactic would pass even if a bad one
    were still reachable.
    """
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import (
        _FUSED_HC_FALLBACK_TACTIC_FMA,
        _fused_hc_mma_supported,
        _get_fused_hc_fallback_tactic,
    )
    from tensorrt_llm._utils import get_sm_version

    assert get_sm_version() < 100
    assert not _fused_hc_mma_supported()
    assert _get_fused_hc_fallback_tactic(4096) == _FUSED_HC_FALLBACK_TACTIC_FMA
    assert _get_fused_hc_fallback_tactic(None) == _FUSED_HC_FALLBACK_TACTIC_FMA
    backend, *_ = _FUSED_HC_FALLBACK_TACTIC_FMA
    assert backend.endswith("_fma")


def test_mhc_dsv4_sm100_pre_mapping_still_offers_the_tf32_deepgemm_tactics(monkeypatch):
    """The SM90 guard withholds the TF32 ladder; it must not delete it.

    Mocked rather than skipped. This machine is Hopper, so a guard that dropped
    the DeepGEMM tactics on *every* architecture would look identical here to
    one that only withholds them below SM100 --- and the Blackwell mHC path is
    protected behaviour with no GPU here to measure it on. Both halves are
    driven through the real ``get_valid_tactics`` with only the compute
    capability faked.
    """
    from tensorrt_llm._torch.modules.mhc import mhc_cuda

    # The predicate itself, as a function of compute capability.
    try:
        for sm, expected in ((90, False), (100, True), (103, True)):
            mhc_cuda._pre_mapping_tf32_allowed.cache_clear()
            monkeypatch.setattr(mhc_cuda, "get_sm_version", lambda sm=sm: sm)
            assert mhc_cuda._pre_mapping_tf32_allowed() is expected, (
                f"sm{sm} should {'offer' if expected else 'withhold'} the TF32 tactics"
            )
    finally:
        monkeypatch.undo()
        mhc_cuda._pre_mapping_tf32_allowed.cache_clear()

    # And the candidate list it gates, with DeepGEMM forced present so the
    # result reflects the guard rather than this container's packages.
    runner = mhc_cuda.MhcPreMappingRunner(
        n=_DSV4_HC_MULT,
        hidden_size=4096,
        rms_eps=_DSV4_EPS,
        hc_pre_eps=_DSV4_EPS,
        hc_sinkhorn_eps=_DSV4_EPS,
        hc_post_mult_value=_DSV4_POST_MULT,
        sinkhorn_repeat=_DSV4_SINKHORN_ITERS,
    )
    mix_hc = (2 + _DSV4_HC_MULT) * _DSV4_HC_MULT
    inputs = [None, torch.empty((mix_hc, 1), device="meta")]
    monkeypatch.setattr(mhc_cuda, "_get_dg_fn", lambda: object())

    monkeypatch.setattr(mhc_cuda, "_pre_mapping_tf32_allowed", lambda: False)
    hopper = runner.get_valid_tactics(inputs, None)
    monkeypatch.setattr(mhc_cuda, "_pre_mapping_tf32_allowed", lambda: True)
    blackwell = runner.get_valid_tactics(inputs, None)

    dg_hopper = [t for t in hopper if str(t[0]).startswith("dg_")]
    dg_blackwell = [t for t in blackwell if str(t[0]).startswith("dg_")]
    assert not dg_hopper, f"Hopper still offers {dg_hopper}"
    assert {t[0] for t in dg_blackwell} == {"dg_splitk", "dg_nosplit"}, (
        f"Blackwell lost the DeepGEMM ladder: {dg_blackwell}"
    )
    assert [t for t in blackwell if t[0] == "fma"] == [t for t in hopper if t[0] == "fma"], (
        "the guard changed the FMA ladder as well; it must only withhold the TF32 backends"
    )


def test_mhc_dsv4_source_faithful_post_mapping_selects_sm90_only(monkeypatch):
    """The source-faithful post-mapping is scoped to the architecture it was measured on.

    ``get_sm_version()`` returns ``-1`` with no visible CUDA device, so a
    "below Blackwell" predicate would be *true* on a CPU/meta build and route
    it onto a Triton kernel that cannot launch there. It would also change the
    numerics of every Ada/Ampere/Turing deployment, none of which this bring-up
    has run. Both are silent, so they are pinned here rather than argued: the
    predicate is exercised at ``-1``, ``90``, ``100`` and ``103`` with only the
    compute capability faked, and a raising ``get_sm_version`` (no CUDA context
    at all) must also keep the shipped kernel.
    """
    from tensorrt_llm import _utils
    from tensorrt_llm._torch.modules.mhc import hyper_connection

    def _select(sm_source):
        hyper_connection._source_faithful_post_mapping.cache_clear()
        monkeypatch.setattr(_utils, "get_sm_version", sm_source)
        return hyper_connection._source_faithful_post_mapping()

    def _raises():
        raise RuntimeError("no CUDA context")

    try:
        for sm, expected in ((-1, False), (90, True), (100, False), (103, False)):
            assert _select(lambda sm=sm: sm) is expected, (
                f"sm{sm} should {'take' if expected else 'skip'} the source-faithful post-mapping"
            )
        assert _select(_raises) is False, "a raising get_sm_version must keep the CUDA kernel"
    finally:
        monkeypatch.undo()
        hyper_connection._source_faithful_post_mapping.cache_clear()

    # And on this machine, which is the architecture the path exists for.
    assert _utils.get_sm_version() == 90
    assert hyper_connection._source_faithful_post_mapping()


@pytest.mark.parametrize("n", [1, 257])
def test_mhc_sm90_post_mapping_is_bit_exact_with_the_deepseek_v4_source_expression(n: int):
    """``post_mapping`` must reproduce ``Block.hc_post``, not merely approximate it.

    The reference is a plain Torch expression, so its FP32 association is
    reachable: each ``comb[k] * residual[k]`` is a rounded product, they are
    summed in index order, and ``post * x`` is added last. The fused CUDA kernel
    seeds an FMA chain with ``post * x`` instead. Both are correct arithmetic
    and the FMA chain is the more accurate one, but they are not the same
    number, and against the checkpoint's own BF16 output the difference reads as
    a full storage step on the elements that straddle a rounding boundary.

    The second assertion is what stops this from silently reverting: the
    shipped CUDA kernel must be measurably *not* bit-exact on the same inputs,
    so deleting the Hopper path fails here instead of passing on a tolerance.
    """
    from tensorrt_llm._torch.modules.mhc.hyper_connection import _source_faithful_post_mapping
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_post_mapping_cuda
    from tensorrt_llm._utils import get_sm_version

    assert get_sm_version() < 100
    assert _source_faithful_post_mapping(), "the Hopper post-mapping path is not selected"

    hidden = 4096
    torch.random.manual_seed(29 + n)
    residual = torch.randn((n, _DSV4_HC_MULT, hidden), dtype=torch.bfloat16, device="cuda")
    x = torch.randn((n, hidden), dtype=torch.bfloat16, device="cuda")
    post = torch.rand((n, _DSV4_HC_MULT), dtype=torch.float32, device="cuda") * _DSV4_POST_MULT
    comb = torch.rand((n, _DSV4_HC_MULT, _DSV4_HC_MULT), dtype=torch.float32, device="cuda")

    # `Block.hc_post` from the checkpoint's inference/model.py, verbatim.
    reference = (
        post.unsqueeze(-1) * x.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
    ).to(x.dtype)

    module = mHC(
        mult=_DSV4_HC_MULT,
        hidden_size=hidden,
        sinkhorn_iters=_DSV4_SINKHORN_ITERS,
        dtype=None,
        eps=_DSV4_EPS,
        norm_eps=_DSV4_EPS,
        sinkhorn_eps=_DSV4_EPS,
        post_mult_value=_DSV4_POST_MULT,
    ).cuda()

    def differing(got):
        return int(
            (got.contiguous().view(torch.int16) != reference.contiguous().view(torch.int16)).sum()
        )

    got = module.post_mapping(x, residual, post, comb)
    assert differing(got) == 0, (
        f"post_mapping differs from the source expression on {differing(got)} of "
        f"{reference.numel()} BF16 values"
    )
    fused = mhc_post_mapping_cuda(residual, x, post, comb, _DSV4_HC_MULT)
    if n > 1:
        assert differing(fused) > 0, (
            "the shipped FMA kernel is already bit-exact here, so this test cannot "
            "tell the Hopper path from its absence"
        )


def test_mhc_sm90_fused_hc_boundary_matches_the_source_expression_and_norm():
    """The layer boundary must not keep an association the standalone call dropped.

    ``fused_hc`` is the *default* boundary in ``modeling_deepseekv4`` --- the
    standalone ``post_mapping`` only runs on the unfused, engram and DSpark
    paths. Fixing one and not the other would leave the served model with the
    FMA association while a layer replay, which necessarily calls the standalone
    form, proved the source-faithful one. Measured before this was wired: 206 of
    4,210,688 BF16 values apart.

    The folded next-layer RMSNorm is checked in the same shot, because resolving
    the boundary in two steps means this path applies that norm itself.
    """
    from tensorrt_llm._torch.modules.mhc.hyper_connection import _source_faithful_post_mapping
    from tensorrt_llm._utils import get_sm_version

    assert get_sm_version() < 100 and _source_faithful_post_mapping()

    n, hidden = 257, 4096
    torch.random.manual_seed(41)
    residual = torch.randn((n, _DSV4_HC_MULT, hidden), dtype=torch.bfloat16, device="cuda")
    x = torch.randn((n, hidden), dtype=torch.bfloat16, device="cuda")
    post = torch.rand((n, _DSV4_HC_MULT), dtype=torch.float32, device="cuda") * _DSV4_POST_MULT
    comb = torch.rand((n, _DSV4_HC_MULT, _DSV4_HC_MULT), dtype=torch.float32, device="cuda")
    weight = torch.randn((hidden,), dtype=torch.bfloat16, device="cuda")

    module = mHC(
        mult=_DSV4_HC_MULT,
        hidden_size=hidden,
        sinkhorn_iters=_DSV4_SINKHORN_ITERS,
        dtype=None,
        eps=_DSV4_EPS,
        norm_eps=_DSV4_EPS,
        sinkhorn_eps=_DSV4_EPS,
        post_mult_value=_DSV4_POST_MULT,
    ).cuda()
    module.fn.normal_(0, 1e-4)
    module.base.normal_(0, 0.1)
    module.scale.normal_(0, 0.1)

    reference = (
        post.unsqueeze(-1) * x.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
    ).to(x.dtype)
    residual_cur, _, _, layer_input = module.fused_hc(x, residual, post, comb)
    apart = int(
        (
            residual_cur.contiguous().view(torch.int16) != reference.contiguous().view(torch.int16)
        ).sum()
    )
    assert apart == 0, f"fused_hc residual differs from the source expression on {apart} values"

    # `RMSNorm.forward` from the checkpoint: FP32 throughout, one rounding.
    _, _, _, normed = module.fused_hc(x, residual, post, comb, weight, _DSV4_EPS)
    f = layer_input.float()
    expected = (
        weight.float() * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + _DSV4_EPS))
    ).to(layer_input.dtype)
    apart = int(
        (normed.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)).sum()
    )
    assert apart == 0, f"the folded norm differs from the source RMSNorm on {apart} values"


@pytest.mark.parametrize("n", [1, 257])
def test_mhc_sm90_every_offered_pre_mapping_tactic_matches_the_source_golden(n: int):
    """The autotuner picks on latency, so every *candidate* has to be accurate.

    ``pre_mapping``'s tuner may offer a DeepGEMM backend alongside the FP32 FMA
    ladder, and that backend runs the mix GEMM in TF32 --- 10 mantissa bits
    against 23. The reference upcasts on purpose (``hc_pre`` computes its mixes
    on ``x.flatten(2).float()``) and Sinkhorn turns a mix error straight into a
    residual weight error, so on DeepSeek-V4-Flash layer 2 with 257 real tokens
    the TF32 tactics scored ``rel_max_abs`` 1.71e-01 on ``layer_input`` where
    every FMA tactic scored 1.07e-02.

    Two assertions, because either alone is weak: the candidate list must carry
    no TF32 backend below SM100, *and* every tactic it does carry must meet the
    registered ``mhc`` tolerance against the independent golden. The first
    without the second would pass a list of accurate-looking but wrong kernels;
    the second without the first would pass whichever tactic happened to be
    selected.
    """
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import MhcPreMappingRunner
    from tensorrt_llm._utils import get_sm_version

    assert get_sm_version() < 100
    tg = _dsv4_mhc_goldens()
    limits = _dsv4_mhc_tolerance()
    hidden_size = 4096
    data = generate_pre_data(
        n=n,
        hc_mult=_DSV4_HC_MULT,
        hidden_size=hidden_size,
        hc_post_mult_value=_DSV4_POST_MULT,
        sinkhorn_repeat=_DSV4_SINKHORN_ITERS,
    )
    residual = data["residual"].contiguous()
    runner = MhcPreMappingRunner(
        n=_DSV4_HC_MULT,
        hidden_size=hidden_size,
        rms_eps=_DSV4_EPS,
        hc_pre_eps=_DSV4_EPS,
        hc_sinkhorn_eps=_DSV4_EPS,
        hc_post_mult_value=_DSV4_POST_MULT,
        sinkhorn_repeat=_DSV4_SINKHORN_ITERS,
    )
    inputs = [
        residual.view(n, _DSV4_HC_MULT * hidden_size),
        data["fn"].contiguous(),
        residual,
        data["hc_scale"].contiguous(),
        data["hc_base"].contiguous(),
    ]
    tactics = runner.get_valid_tactics(inputs, None)
    assert tactics, "the pre_mapping tuner offered no tactic at all"
    tf32 = [t for t in tactics if str(t[0]).startswith("dg_")]
    assert not tf32, f"SM90 pre_mapping still offers the TF32 DeepGEMM tactics {tf32}"

    ref_input, ref_post, ref_comb = tg.hc_pre(
        residual.unsqueeze(0),
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        hc_mult=_DSV4_HC_MULT,
        iters=_DSV4_SINKHORN_ITERS,
        norm_eps=_DSV4_EPS,
        hc_eps=_DSV4_EPS,
    )
    for tactic in tactics:
        post_mix, comb_mix, layer_input = runner(inputs=inputs, tactic=tactic)
        for label, got, ref in (
            ("layer_input", layer_input, ref_input.squeeze(0)),
            ("post_mix", post_mix.reshape(n, _DSV4_HC_MULT), ref_post.squeeze(0)),
            ("comb_mix", comb_mix.reshape(n, _DSV4_HC_MULT, _DSV4_HC_MULT), ref_comb.squeeze(0)),
        ):
            m = tg.compare(got, ref)
            assert m["finite"], f"tactic {tactic} {label}: non-finite"
            assert m["cosine"] >= limits["cosine_min"], (
                f"tactic {tactic} {label}: cosine {m['cosine']:.6f} < {limits['cosine_min']}"
            )
            assert m["rel_max_abs"] <= limits["rel_max_abs_max"], (
                f"tactic {tactic} {label}: rel_max_abs {m['rel_max_abs']:.4f} > "
                f"{limits['rel_max_abs_max']}"
            )


@pytest.mark.parametrize("n", [1, 32, 512])
def test_mhc_sm90_pre_and_post_mapping_match_the_deepseek_v4_source_golden(n: int):
    """The SM90 FMA kernels vs the checkpoint's own mHC, not vs each other.

    `test_mhc_fused_hc` already pins fused against unfused; both are
    TensorRT-LLM, so they would agree on a shared mistake. The reference here
    is the independent DeepSeek-V4 ladder golden, gated by the pre-registered
    `mhc` tolerance.
    """
    tg = _dsv4_mhc_goldens()
    limits = _dsv4_mhc_tolerance()
    hidden_size = 4096
    data = generate_pre_data(
        n=n,
        hc_mult=_DSV4_HC_MULT,
        hidden_size=hidden_size,
        hc_post_mult_value=_DSV4_POST_MULT,
        sinkhorn_repeat=_DSV4_SINKHORN_ITERS,
    )
    module = _dsv4_mhc_module(hidden_size, data)
    residual = data["residual"]

    post_mix, comb_mix, layer_input = module.pre_mapping(residual)
    ref_input, ref_post, ref_comb = tg.hc_pre(
        residual.unsqueeze(0),
        data["fn"],
        data["hc_scale"],
        data["hc_base"],
        hc_mult=_DSV4_HC_MULT,
        iters=_DSV4_SINKHORN_ITERS,
        norm_eps=_DSV4_EPS,
        hc_eps=_DSV4_EPS,
    )
    for label, got, ref in (
        ("layer_input", layer_input, ref_input.squeeze(0)),
        ("post_mix", post_mix.reshape(n, _DSV4_HC_MULT), ref_post.squeeze(0)),
        ("comb_mix", comb_mix.reshape(n, _DSV4_HC_MULT, _DSV4_HC_MULT), ref_comb.squeeze(0)),
    ):
        m = tg.compare(got, ref)
        assert m["finite"], f"pre_mapping {label}: non-finite"
        assert m["cosine"] >= limits["cosine_min"], (
            f"pre_mapping {label}: cosine {m['cosine']:.6f} < {limits['cosine_min']}"
        )
        assert m["rel_max_abs"] <= limits["rel_max_abs_max"], (
            f"pre_mapping {label}: rel_max_abs {m['rel_max_abs']:.4f} > {limits['rel_max_abs_max']}"
        )

    torch.random.manual_seed(13)
    block_out = torch.randn((n, hidden_size), dtype=torch.bfloat16, device="cuda") / hidden_size
    got_residual = module.post_mapping(block_out, residual, post_mix, comb_mix)
    ref_residual = tg.hc_post(
        block_out.unsqueeze(0), residual.unsqueeze(0), ref_post, ref_comb
    ).squeeze(0)
    m = tg.compare(got_residual, ref_residual)
    assert m["finite"] and m["cosine"] >= limits["cosine_min"], (
        f"post_mapping: cosine {m['cosine']:.6f}"
    )
    assert m["rel_max_abs"] <= limits["rel_max_abs_max"], (
        f"post_mapping: rel_max_abs {m['rel_max_abs']:.4f} > {limits['rel_max_abs_max']}"
    )


def test_mhc_sm90_head_reduction_matches_the_deepseek_v4_source_golden():
    """`HCHead` is a plain sigmoid gate with no Sinkhorn --- easy to conflate."""
    tg = _dsv4_mhc_goldens()
    limits = _dsv4_mhc_tolerance()
    n, hidden_size = 64, 4096
    torch.random.manual_seed(17)
    x = (
        torch.randn((n, _DSV4_HC_MULT, hidden_size), dtype=torch.bfloat16, device="cuda")
        / hidden_size
    )
    fn = (
        torch.randn((_DSV4_HC_MULT, _DSV4_HC_MULT * hidden_size), dtype=torch.float, device="cuda")
        * 1e-4
    )
    scale = torch.randn((1,), dtype=torch.float, device="cuda") * 0.1
    base = torch.randn((_DSV4_HC_MULT,), dtype=torch.float, device="cuda") * 0.1

    head = HCHead(
        mult=_DSV4_HC_MULT, hidden_size=hidden_size, eps=_DSV4_EPS, norm_eps=_DSV4_EPS
    ).cuda()
    head.fn.copy_(fn)
    head.scale.copy_(scale)
    head.base.copy_(base)

    got = head(x)
    ref = tg.hc_head(x.unsqueeze(0), fn, scale, base, norm_eps=_DSV4_EPS, hc_eps=_DSV4_EPS).squeeze(
        0
    )
    m = tg.compare(got, ref)
    assert m["finite"] and m["cosine"] >= limits["cosine_min"], f"hc_head: cosine {m['cosine']:.6f}"
    assert m["rel_max_abs"] <= limits["rel_max_abs_max"], (
        f"hc_head: rel_max_abs {m['rel_max_abs']:.4f} > {limits['rel_max_abs_max']}"
    )


if __name__ == "__main__":
    main()
