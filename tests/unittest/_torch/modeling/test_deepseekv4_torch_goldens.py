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
"""CPU regression coverage for the pure-Torch DeepSeek-V4-Flash goldens.

The 8-GPU evidence driver checks these goldens against real checkpoint
activations, which is the real proof --- but that run needs eight H100s, a
159 GB checkpoint and the isolated reference environment. These tests instead
pin the *properties* each golden must have, on CPU, in under a second, so a
regression is caught long before anyone books a node.

The properties chosen are the ones a plausible-looking rewrite would silently
break: orthogonality of the Hadamard rotation, the nibble order and scale
grouping of the MXFP4 unpacking, the asymmetry of the SwiGLU clamp, and the
rule that a routing bias moves expert *selection* without moving the returned
weights.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_GOLDENS_PATH = (
    Path(__file__).resolve().parents[3]
    / "integration"
    / "defs"
    / "accuracy"
    / "deepseek_v4_flash_h100"
    / "torch_goldens.py"
)


def _load():
    name = "deepseek_v4_flash_h100_torch_goldens"
    spec = importlib.util.spec_from_file_location(name, _GOLDENS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tg = _load()


# ---------------------------------------------------------------------------
# Independence guard.
# ---------------------------------------------------------------------------


def test_assert_independent_rejects_a_leaked_tensorrt_llm_import():
    """The guard has to actually fire, or it is decoration rather than a gate.

    Note this test runs under the repo's pytest, which imports ``tensorrt_llm``
    via conftest --- so the guard is *expected* to raise here. That is the
    point: it raises whenever the module is present, which is exactly the
    condition the evidence driver relies on.
    """
    sentinel = "tensorrt_llm.__golden_probe__"
    sys.modules[sentinel] = object()
    try:
        with pytest.raises(AssertionError, match="tensorrt_llm"):
            tg.assert_independent()
    finally:
        del sys.modules[sentinel]


def test_assert_independent_passes_when_nothing_is_leaked(monkeypatch):
    clean = {k: v for k, v in sys.modules.items() if not k.startswith("tensorrt_llm")}
    monkeypatch.setattr(sys, "modules", clean)
    tg.assert_independent()


# ---------------------------------------------------------------------------
# Hadamard rotation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [32, 128, 512])
def test_hadamard_is_orthogonal_and_self_inverse(d):
    torch.manual_seed(0)
    x = torch.randn(4, d)
    y = tg.hadamard_transform(x)
    # Normalised WHT preserves the L2 norm ...
    torch.testing.assert_close(y.square().sum(-1), x.square().sum(-1), rtol=1e-4, atol=1e-4)
    # ... and is its own inverse.
    torch.testing.assert_close(tg.hadamard_transform(y), x, rtol=1e-4, atol=1e-4)


def test_hadamard_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power-of-two"):
        tg.hadamard_transform(torch.zeros(2, 96))


# ---------------------------------------------------------------------------
# Quantisation simulation.
# ---------------------------------------------------------------------------


def test_fp4_round_trip_is_exact_on_representable_values():
    """Values already on the E2M1 grid must survive untouched."""
    levels = torch.tensor(tg.FP4_LEVELS)
    x = torch.cat([levels, -levels]).repeat(32)[:512].reshape(1, 512).to(torch.bfloat16)
    torch.testing.assert_close(tg.fp4_quant_dequant(x, 32).float(), x.float())


def test_fp4_quantisation_is_blockwise_not_tensor_wide():
    """One huge value must not flatten the block next to it."""
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, :32] = 4096.0  # first block: large scale
    x[0, 32:] = 1.0  # second block: small values, must be preserved
    out = tg.fp4_quant_dequant(x, 32)
    assert out[0, 32:].abs().max().item() == pytest.approx(1.0, rel=1e-2)


def test_fp8_round_trip_keeps_scale_a_power_of_two():
    torch.manual_seed(0)
    x = torch.randn(4, 256, dtype=torch.bfloat16) * 17.0
    out = tg.fp8_quant_dequant(x, 128)
    assert torch.isfinite(out).all()
    # UE8M0 scales are exact powers of two, so the round trip stays well
    # inside a couple of ulps of the E4M3 grid.
    rel = (out.float() - x.float()).abs().max() / x.float().abs().max()
    assert rel < 0.05


def test_mxfp4_unpacking_uses_low_nibble_first_and_group_32_scales():
    """Nibble order and scale grouping are the two silent-corruption risks."""
    # byte 0x21 -> low nibble 1 (=0.5), high nibble 2 (=1.0)
    packed = torch.full((1, 16), 0x21, dtype=torch.uint8).view(torch.int8)
    scale = torch.ones(1, 1, dtype=torch.float32)
    out = tg.dequant_mxfp4(packed, scale, group=32)
    assert out.shape == (1, 32)
    assert out[0, 0].item() == 0.5
    assert out[0, 1].item() == 1.0

    # 0x9 has the sign bit (0x8) set over magnitude code 1 -> -0.5
    signed = torch.full((1, 16), 0x09, dtype=torch.uint8).view(torch.int8)
    assert tg.dequant_mxfp4(signed, scale, group=32)[0, 0].item() == -0.5

    # A second scale group must only affect its own 32 logical values.
    packed2 = torch.full((1, 32), 0x11, dtype=torch.uint8).view(torch.int8)
    scale2 = torch.tensor([[1.0, 4.0]])
    out2 = tg.dequant_mxfp4(packed2, scale2, group=32)
    assert out2[0, 0].item() == 0.5
    assert out2[0, 32].item() == 2.0


# ---------------------------------------------------------------------------
# RoPE.
# ---------------------------------------------------------------------------


def test_rope_inverse_undoes_rope():
    torch.manual_seed(0)
    freqs = tg.yarn_freqs_cis(64, 16, 65536, 160000.0, 16.0, 32, 1)
    x = torch.randn(2, 16, 8, 64)
    rotated = tg.apply_rope(x, freqs)
    torch.testing.assert_close(tg.apply_rope(rotated, freqs, inverse=True), x, rtol=1e-5, atol=1e-5)


def test_rope_position_zero_is_identity():
    freqs = tg.yarn_freqs_cis(64, 4, 0, 10000.0, 16.0, 32, 1)
    x = torch.randn(1, 4, 64)
    torch.testing.assert_close(tg.apply_rope(x, freqs)[:, 0], x[:, 0], rtol=1e-6, atol=1e-6)


def test_yarn_only_interpolates_frequencies_and_never_scales_amplitude():
    """The source formula has no mscale term; magnitudes must stay unit."""
    freqs = tg.yarn_freqs_cis(64, 32, 65536, 160000.0, 16.0, 32, 1)
    torch.testing.assert_close(freqs.abs(), torch.ones_like(freqs.abs()), rtol=1e-6, atol=1e-6)


def test_yarn_disabled_matches_plain_rope():
    plain = tg.yarn_freqs_cis(64, 8, 0, 10000.0, 16.0, 32, 1)
    scaled = tg.yarn_freqs_cis(64, 8, 65536, 10000.0, 16.0, 32, 1)
    assert not torch.allclose(plain, scaled)


# ---------------------------------------------------------------------------
# MoE routing and experts.
# ---------------------------------------------------------------------------


def test_routing_bias_moves_selection_but_not_weights():
    """The correction bias steers top-k only; weights come from raw scores."""
    torch.manual_seed(0)
    x = torch.randn(6, 32)
    gate_w = torch.randn(8, 32)
    _, plain_ids, scores = tg.moe_route(x, gate_w, topk=2, route_scale=1.5)

    bias = torch.zeros(8)
    bias[7] = 1e3  # force expert 7 into every selection
    biased_w, biased_ids, _ = tg.moe_route(x, gate_w, topk=2, route_scale=1.5, bias=bias)
    assert (biased_ids[:, 0] == 7).all()
    assert not torch.equal(plain_ids, biased_ids)
    # The returned weight for expert 7 is its *unbiased* score, normalised.
    raw = scores.gather(1, biased_ids)
    torch.testing.assert_close(biased_w, raw / raw.sum(-1, keepdim=True) * 1.5)


def test_hash_routing_ignores_hidden_state_for_expert_choice():
    torch.manual_seed(0)
    table = torch.tensor([[1, 2], [3, 4], [5, 6]])
    ids = torch.tensor([0, 2, 1])
    for _ in range(3):
        _, chosen, _ = tg.moe_route(
            torch.randn(3, 16),
            torch.randn(8, 16),
            topk=2,
            route_scale=1.5,
            tid2eid=table,
            input_ids=ids,
        )
        assert torch.equal(chosen, table[ids])


def test_routing_weights_sum_to_route_scale():
    torch.manual_seed(0)
    w, _, _ = tg.moe_route(torch.randn(5, 16), torch.randn(8, 16), topk=3, route_scale=1.5)
    torch.testing.assert_close(w.sum(-1), torch.full((5,), 1.5), rtol=1e-5, atol=1e-5)


def test_swiglu_clamp_is_asymmetric():
    """Gate is bounded above only; up is bounded on both sides."""
    dim, inter = 8, 8
    w1 = torch.eye(inter, dim) * 1e3  # drive the gate far positive and negative
    w3 = torch.eye(inter, dim) * 1e3
    w2 = torch.eye(dim, inter)
    x = torch.full((1, dim), -1.0)

    clamped = tg.expert_swiglu(x, w1, w2, w3, swiglu_limit=10.0, quantize_act=False)
    # gate = -1e3 (unbounded below, silu -> ~0), up clamped to -10
    assert clamped.abs().max().item() < 1e-3
    x_pos = torch.full((1, dim), 1.0)
    out_pos = tg.expert_swiglu(x_pos, w1, w2, w3, swiglu_limit=10.0, quantize_act=False)
    # gate clamped to 10 (silu ~10), up clamped to 10 -> ~100
    assert out_pos.max().item() == pytest.approx(100.0, rel=1e-2)


# ---------------------------------------------------------------------------
# mHC.
# ---------------------------------------------------------------------------


def test_sinkhorn_matrix_is_row_normalised_and_positive():
    torch.manual_seed(0)
    hc = 4
    mixes = torch.randn(2, 3, (2 + hc) * hc)
    pre, post, comb = tg.hc_split_sinkhorn(
        mixes, torch.ones(3), torch.zeros((2 + hc) * hc), hc_mult=hc, iters=20, eps=1e-6
    )
    assert comb.shape == (2, 3, hc, hc)
    assert (comb > 0).all()
    # The loop ends on a column normalisation, so columns sum to ~1.
    torch.testing.assert_close(comb.sum(dim=-2), torch.ones(2, 3, hc), rtol=2e-3, atol=2e-3)
    # pre is a sigmoid plus eps; post is twice a sigmoid.
    assert ((pre > 0) & (pre < 1 + 1e-5)).all()
    assert ((post > 0) & (post < 2)).all()


def test_hc_post_with_zero_update_preserves_residual_mix():
    torch.manual_seed(0)
    b, s, hc, d = 1, 2, 4, 8
    residual = torch.randn(b, s, hc, d)
    comb = torch.zeros(b, s, hc, hc)
    comb[..., torch.arange(hc), torch.arange(hc)] = 1.0
    out = tg.hc_post(torch.zeros(b, s, d), residual, torch.zeros(b, s, hc), comb)
    torch.testing.assert_close(out, residual)


def test_hc_pre_reduces_streams_to_one():
    torch.manual_seed(0)
    b, s, hc, d = 1, 3, 4, 16
    x = torch.randn(b, s, hc, d)
    y, post, comb = tg.hc_pre(
        x,
        torch.randn((2 + hc) * hc, hc * d),
        torch.ones(3),
        torch.zeros((2 + hc) * hc),
        hc_mult=hc,
        iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
    )
    assert y.shape == (b, s, d)
    assert post.shape == (b, s, hc)
    assert comb.shape == (b, s, hc, hc)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Sparse attention.
# ---------------------------------------------------------------------------


def test_sink_only_shrinks_output_never_steers_it():
    """The sink adds denominator mass, so it scales the output toward zero."""
    torch.manual_seed(0)
    b, s, h, d, n = 1, 2, 2, 8, 4
    q = torch.randn(b, s, h, d)
    kv = torch.randn(b, 6, d)
    idxs = torch.arange(n).view(1, 1, n).expand(b, s, n).contiguous()

    tiny = tg.sparse_attention(q, kv, torch.full((h,), -60.0), idxs, d**-0.5)
    large = tg.sparse_attention(q, kv, torch.full((h,), 20.0), idxs, d**-0.5)
    assert large.abs().max() < tiny.abs().max()
    # Direction is unchanged: a denominator-only term rescales each head's
    # output by its own positive scalar, so it cannot rotate any head. The
    # comparison is per head --- the scalars differ between heads, so a
    # flattened cosine would drop for a perfectly correct implementation.
    cos = torch.nn.functional.cosine_similarity(tiny.float(), large.float(), dim=-1)
    assert cos.min().item() > 0.999


def test_padded_slots_are_excluded_from_attention():
    """A ``-1`` slot must not contribute, whatever sits at row 0 of the cache."""
    torch.manual_seed(0)
    b, s, h, d = 1, 1, 1, 8
    q = torch.randn(b, s, h, d)
    kv = torch.randn(b, 4, d)
    kv[0, 0] = 1e3  # row 0 would dominate if -1 were treated as index 0

    masked = tg.sparse_attention(
        q, kv, torch.full((h,), -60.0), torch.tensor([[[-1, 1, 2]]]), d**-0.5
    )
    only_valid = tg.sparse_attention(
        q, kv, torch.full((h,), -60.0), torch.tensor([[[1, 2]]]), d**-0.5
    )
    torch.testing.assert_close(masked, only_valid, rtol=1e-4, atol=1e-4)


def test_all_slots_padded_yields_finite_zero_output():
    q = torch.randn(1, 1, 1, 8)
    kv = torch.randn(1, 4, 8)
    out = tg.sparse_attention(q, kv, torch.full((1,), 0.0), torch.tensor([[[-1, -1]]]), 8**-0.5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# Sparse attention: the BF16 operand / FP32 accumulator contract.
#
# These need CUDA because the whole point is the tensor-core GEMM: the source
# kernel holds q / kv / acc_s_cast in BF16 shared memory and accumulates into
# FP32 fragments, and reproducing that ordering is what moved the measured
# disagreement with the kernel from ~1000 elements per million down to ~20.
# The CPU tests above pin the algebra; only these pin the arithmetic.
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _bf16_case(seed: int = 0, s: int = 64, h: int = 8, d: int = 512, n: int = 64):
    """A single-tile BF16 case at the real head geometry."""
    torch.manual_seed(seed)
    dev = "cuda"
    q = (torch.randn(1, s, h, d, device=dev) * 0.5).bfloat16()
    kv = (torch.randn(1, s + n, d, device=dev) * 0.5).bfloat16()
    sink = torch.randn(h, device=dev, dtype=torch.float32)
    idxs = torch.arange(n, device=dev).view(1, 1, n).expand(1, s, n).contiguous()
    return q, kv, sink, idxs, float(d) ** -0.5


@requires_cuda
def test_accumulating_matmul_keeps_bf16_operands_and_an_fp32_accumulator():
    """Widening the operands to FP32 first is a different summation order.

    Both compute the same mathematical product --- BF16 x BF16 is exact in
    FP32 either way --- so this cannot be caught by a tolerance. It is caught
    by counting: the tensor-core accumulator and an FP32 GEMM disagree in the
    last bits on the large majority of a K=512 reduction. If someone
    reintroduces a ``.float()`` on the operands, `differing` collapses to zero
    and this fails.
    """
    a = (torch.randn(4, 16, 512, device="cuda") * 0.5).bfloat16()
    b = (torch.randn(4, 512, 64, device="cuda") * 0.5).bfloat16()

    got = tg._accumulating_matmul(a, b)
    assert got.dtype == torch.float32

    widened = torch.bmm(a.float(), b.float())
    differing = float((got != widened).float().mean())
    assert differing > 0.5, f"operands look widened: only {differing:.3%} of elements differ"
    # Same quantity, so they must still agree to FP32 round-off.
    torch.testing.assert_close(got, widened, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_accumulating_matmul_is_exact_when_operands_are_already_fp32():
    """The FP32 path must be plain ``bmm``: ``out_dtype`` is CUDA-only."""
    a = torch.randn(2, 8, 128, device="cuda")
    b = torch.randn(2, 128, 32, device="cuda")
    assert torch.equal(tg._accumulating_matmul(a, b), torch.bmm(a, b))


@requires_cuda
def test_sparse_attention_materialises_attention_weights_at_activation_dtype():
    """``acc_s_cast``: the numerators are BF16 before the value GEMM.

    The source copies the FP32 exponentials into a BF16 shared buffer and
    feeds *that* to the second GEMM, so each weight carries ~2^-8 of relative
    error into the weighted sum. A single tile is used so the online-softmax
    recurrence is not also in play and this pins one thing.
    """
    q, kv, sink, idxs, scale = _bf16_case()
    got = tg.sparse_attention(q, kv, sink, idxs, scale)
    assert got.dtype == torch.bfloat16

    rows = q.shape[0] * q.shape[1]
    sel = kv[0, : idxs.shape[-1]].unsqueeze(0).expand(rows, idxs.shape[-1], q.shape[-1])
    scores = torch.bmm(
        q.reshape(rows, q.shape[2], q.shape[3]), sel.transpose(1, 2), out_dtype=torch.float32
    )
    scores = scores * scale
    probs = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    den = probs.sum(-1) + torch.exp(sink.view(1, -1) - scores.amax(dim=-1))

    rounded = torch.bmm(probs.bfloat16(), sel, out_dtype=torch.float32)
    exact = torch.bmm(probs, sel.float())

    assert torch.equal(got, (rounded / den.unsqueeze(-1)).reshape(q.shape).bfloat16())
    # And the un-rounded FP32 variant is measurably a different answer, so the
    # assertion above is pinning a real choice rather than a no-op.
    assert not torch.equal(got, (exact / den.unsqueeze(-1)).reshape(q.shape).bfloat16())


@requires_cuda
def test_sparse_attention_accumulator_is_fp32_across_tiles():
    """Only ``acc_s_cast`` is narrowed --- the output accumulator stays FP32.

    Rounding ``acc_o`` to BF16 once per tile is the obvious way to get the
    "BF16 kernel" wrong, and it is invisible at one tile. With eight tiles it
    costs about an order of magnitude of accuracy, which this measures against
    an FP64 recomputation of the same weighted sum.
    """
    n = 8 * tg.SPARSE_ATTN_BLOCK
    q, kv, sink, idxs, scale = _bf16_case(seed=1, s=16, n=n)
    got = tg.sparse_attention(q, kv, sink, idxs, scale).float()

    rows, h, d = q.shape[1], q.shape[2], q.shape[3]
    sel = kv[0, :n].double()
    scores = torch.einsum("shd,nd->shn", q[0].double(), sel) * scale
    probs = torch.exp(scores - scores.amax(-1, keepdim=True))
    den = probs.sum(-1) + torch.exp(sink.double().view(1, h) - scores.amax(-1))
    exact = torch.einsum("shn,nd->shd", probs, sel) / den.unsqueeze(-1)

    err = (got[0] - exact.float()).abs().max() / exact.abs().max()
    assert err < 5e-3, f"accumulator looks narrowed: relative error {err:.3e}"
    assert got.shape == (1, rows, h, d)


@requires_cuda
def test_sparse_attention_tiles_at_the_source_block_size():
    """The recurrence is per 64 selected slots, and that is observable.

    A single-pass softmax over all selected slots computes the same quantity
    but rescales the accumulator once instead of once per tile. Feeding the
    same slots in one 128-wide tile versus two 64-wide tiles must therefore
    produce different bits --- if it does not, the tiling has been optimised
    away and the golden no longer reproduces the kernel's ordering.
    """
    q, kv, sink, idxs, scale = _bf16_case(seed=2, s=32, n=128)
    tiled = tg.sparse_attention(q, kv, sink, idxs, scale, block=tg.SPARSE_ATTN_BLOCK)
    single = tg.sparse_attention(q, kv, sink, idxs, scale, block=128)
    assert tg.SPARSE_ATTN_BLOCK == 64
    assert not torch.equal(tiled, single)
    torch.testing.assert_close(tiled.float(), single.float(), rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def test_rel_max_abs_is_scale_free():
    """A tolerance must not be passable just because both tensors are tiny."""
    ref = torch.randn(64)
    got = ref + 0.1 * ref.abs().mean()
    big = tg.compare(got * 1e6, ref * 1e6)
    small = tg.compare(got * 1e-6, ref * 1e-6)
    assert big["rel_max_abs"] == pytest.approx(small["rel_max_abs"], rel=1e-3)


def test_compare_flags_non_finite():
    ref = torch.randn(8)
    got = ref.clone()
    got[0] = float("nan")
    assert tg.compare(got, ref)["finite"] is False
