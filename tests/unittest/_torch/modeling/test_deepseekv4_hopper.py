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
"""Hopper (SM90) semantics for the DeepSeek-V4 projection, norm and output path.

Every operator here is a real TensorRT-LLM binding executed on an H100 and
compared against the independent source ladder goldens in
`tests/integration/defs/accuracy/deepseek_v4_flash_h100/torch_goldens.py`,
using the tolerances pre-registered in `manifests/tolerances.json`. The
tolerances are read from that file rather than restated, so this file cannot
silently loosen one.

Covered: per-head Q normalisation, the full-width latent KV norm, the
per-ratio RoPE tables, inverse RoPE ahead of the output projection, the
grouped O-LoRA BMM, and the TP8 head/group geometry those all assume.
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import RopeParams
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.sm90_quant import q_norm_source_dtype
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import RotaryScalingType

_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "integration"
    / "defs"
    / "accuracy"
    / "deepseek_v4_flash_h100"
)


def _load_goldens():
    name = "deepseek_v4_flash_h100_torch_goldens"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _EVIDENCE / "torch_goldens.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tg = _load_goldens()
TOL = json.loads((_EVIDENCE / "manifests" / "tolerances.json").read_text())["modules"]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# Checkpoint geometry at TP8: 64 Q heads -> 8 local, 8 O groups -> 1 local,
# head dim 512 = 448 non-RoPE + 64 RoPE, O-LoRA rank 1024.
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
NUM_HEADS = 64
NUM_GROUPS = 8
TP_SIZE = 8
LOCAL_HEADS = NUM_HEADS // TP_SIZE
LOCAL_GROUPS = NUM_GROUPS // TP_SIZE
O_LORA_RANK = 1024
MAX_SEQ_LEN = 1024
EPS = 1e-6


def _assert_within(metrics, module, label):
    limits = TOL[module]
    assert metrics["finite"], f"{label}: non-finite output"
    assert metrics["cosine"] >= limits["cosine_min"], (
        f"{label}: cosine {metrics['cosine']:.6f} < {limits['cosine_min']} ({module})"
    )
    assert metrics["rel_max_abs"] <= limits["rel_max_abs_max"], (
        f"{label}: rel_max_abs {metrics['rel_max_abs']:.4f} > "
        f"{limits['rel_max_abs_max']} ({module})"
    )


def _rotary(compress_ratio: int) -> RotaryEmbedding:
    rope = RopeParams(
        dim=ROPE_DIM,
        max_positions=MAX_SEQ_LEN,
        original_max_positions=65536,
        max_seq_len=MAX_SEQ_LEN,
        beta_fast=32,
        beta_slow=1,
    )
    if compress_ratio > 1:
        rope.theta = 160000.0
        rope.scale_type = RotaryScalingType.yarn
        rope.scale = 16.0
        rope.mscale = 0.0
        rope.mscale_all_dim = 0.0
    else:
        rope.theta = 10000.0
        rope.scale_type = RotaryScalingType.none
        rope.scale = 1.0
    return RotaryEmbedding(rope, head_dim=ROPE_DIM, is_neox=False)


def _golden_freqs(compress_ratio: int) -> torch.Tensor:
    if compress_ratio > 1:
        return tg.yarn_freqs_cis(ROPE_DIM, MAX_SEQ_LEN, 65536, 160000.0, 16.0, 32, 1).cuda()
    return tg.yarn_freqs_cis(ROPE_DIM, MAX_SEQ_LEN, 0, 10000.0, 1.0, 32, 1).cuda()


def test_sm90_is_the_device_under_test():
    """Everything below is a Hopper claim; a Blackwell run would prove nothing."""
    assert get_sm_version() < 100, (
        f"expected a pre-Blackwell device, got SM{get_sm_version()} "
        f"({torch.cuda.get_device_name(0)})"
    )


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_sm90_rope_tables_match_the_source_per_ratio_contract(compress_ratio):
    """Ratio-0 uses theta 10000 without YaRN; compressed layers 160000 + YaRN 16.

    Getting this wrong is invisible at position 0 and grows with position, so
    the whole table is compared against an independent YaRN implementation
    rather than a few sampled angles.
    """
    table = _rotary(compress_ratio).rotary_cos_sin.cpu()
    freqs = _golden_freqs(compress_ratio).cpu()
    assert table.shape == (MAX_SEQ_LEN, 2, ROPE_DIM // 2)
    cos_err = float((table[:, 0, :] - freqs.real).abs().max())
    sin_err = float((table[:, 1, :] - freqs.imag).abs().max())
    assert cos_err < 1e-6 and sin_err < 1e-6, (
        f"ratio {compress_ratio}: RoPE table differs from the source YaRN "
        f"frequencies (max |dcos|={cos_err:.2e}, max |dsin|={sin_err:.2e})"
    )

    # The two schedules must not collapse onto the same table.
    if compress_ratio > 1:
        other = _rotary(1).rotary_cos_sin.cpu()
        assert not torch.allclose(table, other), (
            "compressed and SWA-only layers produced identical RoPE tables"
        )


def test_sm90_q_norm_matches_the_source_per_head_rms_scaling():
    """The SM90 per-head Q norm vs the source's own formula, at the registered gate.

    The source does `q *= rsqrt(q.square().mean(-1) + eps)` entirely in BF16,
    so the square, the mean, the epsilon add, the reciprocal square root and
    the multiply all round. `deepseek_v4_q_norm` evaluates the same expression
    in FP32 and rounds once, which lands *above* the registered
    `q_projection_and_norm` `rel_max_abs` --- roughly two BF16 steps at the
    peak element. `q_norm_source_dtype` is what SM90 dispatches instead, and it
    is held to the unmodified manifest gate here.

    Both forms are compared, so the test states the size of the native
    kernel's divergence rather than merely avoiding it, and it would fail if
    SM90 silently went back to the native op.
    """
    torch.manual_seed(5)
    q = (torch.randn(37, LOCAL_HEADS * HEAD_DIM, device="cuda") * 0.5).bfloat16()
    q3 = q.view(-1, LOCAL_HEADS, HEAD_DIM)
    bf16_ref = tg.per_head_rms_scale(q3, EPS)

    got = q_norm_source_dtype(q, LOCAL_HEADS, HEAD_DIM, EPS)
    assert got.shape == q.shape and got.dtype == q.dtype
    assert torch.equal(got.view_as(q3), bf16_ref), (
        "the SM90 per-head Q norm is not bit-identical to the source formula"
    )
    _assert_within(tg.compare(got.view_as(q3), bf16_ref), "q_projection_and_norm", "sm90 q_norm")

    # The native kernel is the FP32 form -- only true if the axis, the epsilon
    # placement and the absence of a learned gain all match -- and that one
    # difference is what puts it outside the registered tolerance.
    native = torch.ops.trtllm.deepseek_v4_q_norm(q, LOCAL_HEADS, HEAD_DIM, EPS)
    fp32_scale = torch.rsqrt(q3.float().square().mean(-1, keepdim=True) + EPS)
    assert torch.equal(native.view_as(q3), (q3.float() * fp32_scale).bfloat16()), (
        "the native kernel is not the FP32 form of the source's per-head RMS scaling"
    )
    native_metrics = tg.compare(native.view_as(q3), bf16_ref)
    assert native_metrics["rel_max_abs"] > TOL["q_projection_and_norm"]["rel_max_abs_max"], (
        f"the native FP32 kernel now measures rel_max_abs "
        f"{native_metrics['rel_max_abs']:.4f} against the source form, inside the "
        "registered gate; the SM90 override exists because it was outside it, so "
        "either the kernel or the manifest changed and this needs re-deciding"
    )
    # The whole residual is the source rounding its scale factor to BF16.
    bf16_scale = torch.rsqrt(q3.square().mean(-1, keepdim=True).float() + EPS).bfloat16().float()
    scale_step = float(((bf16_scale - fp32_scale).abs() / fp32_scale).max())
    assert scale_step < 2.0**-7, (
        f"the per-head scale differs by {scale_step:.3e}, more than one BF16 step; "
        "that is not explained by the source's in-dtype rsqrt"
    )


def test_sm90_dispatches_the_source_dtype_q_norm_and_blackwell_keeps_the_kernel():
    """The branch itself: SM90 takes the source-dtype path, SM100 the native op.

    The numerics above are worthless if the model never calls them, and the
    branch lives inside a closure in `module.py`, so it is checked here through
    the same `get_sm_version` symbol that function reads.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import module as v4_module

    assert get_sm_version() < 100, "this Hopper expectation only holds pre-Blackwell"
    body = inspect.getsource(v4_module.forward_sparse_attn)
    start = body.index("def _q_b_layernorm(")
    branch = body[start : body.index("def _q_b_layernorm_fused_fp8", start)]
    assert "get_sm_version() < 100" in branch and "q_norm_source_dtype(" in branch, (
        "the SM90 Q-norm branch is no longer in the V4 forward's q_b_layernorm"
    )
    assert "torch.ops.trtllm.deepseek_v4_q_norm(" in branch, (
        "the Blackwell branch no longer calls the native deepseek_v4_q_norm kernel"
    )


def test_sm90_latent_kv_norm_matches_the_source_full_width_rms_norm():
    """V4 normalises the whole 512-wide latent with a learned gain.

    Using a V3-style 448-wide norm, or dropping the gain, both still produce a
    normalised latent; only the comparison against the source catches it.
    """
    norm = RMSNorm(hidden_size=HEAD_DIM, eps=EPS, dtype=torch.bfloat16).cuda()
    torch.manual_seed(7)
    with torch.no_grad():
        norm.weight.copy_((torch.randn(HEAD_DIM, device="cuda") * 0.1 + 1.0).bfloat16())
    x = (torch.randn(37, HEAD_DIM, device="cuda") * 0.5).bfloat16()

    _assert_within(
        tg.compare(norm(x), tg.rms_norm(x, norm.weight, EPS)), "kv_latent_and_norm", "kv norm"
    )
    # A 448-wide norm of the same row is a different tensor: the check has bite.
    narrow = tg.rms_norm(x[..., :NOPE_DIM], norm.weight[:NOPE_DIM], EPS)
    assert not torch.allclose(norm(x)[..., :NOPE_DIM].float(), narrow.float(), atol=1e-2), (
        "the full-width and 448-wide norms agree, so this test cannot detect the "
        "V3 latent width being used by mistake"
    )


@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_sm90_inverse_rope_matches_the_source_and_undoes_the_forward_rotation(compress_ratio):
    """Inverse RoPE runs on the attention output before the grouped O-LoRA.

    Both directions come off the same table, so this checks the conjugate is
    really applied (not a second forward rotation) and that the round trip
    returns the original row to BF16 precision.
    """
    rotary = _rotary(compress_ratio)
    torch.manual_seed(31)
    y = (torch.randn(37, LOCAL_HEADS, HEAD_DIM, device="cuda") * 0.5).bfloat16()
    positions = torch.arange(37, dtype=torch.int32, device="cuda")

    inverted = y.clone()
    torch.ops.trtllm.mla_rope_inplace(
        inverted, positions, rotary.rotary_cos_sin, LOCAL_HEADS, NOPE_DIM, ROPE_DIM, True, False
    )
    assert torch.equal(inverted[..., :NOPE_DIM], y[..., :NOPE_DIM]), (
        "inverse RoPE touched the non-RoPE 448 dims"
    )

    freqs = _golden_freqs(compress_ratio)[:37]
    ref = y.clone().float()
    ref[..., NOPE_DIM:] = tg.apply_rope(
        y[..., NOPE_DIM:].float().unsqueeze(0), freqs, inverse=True
    ).squeeze(0)
    _assert_within(
        tg.compare(inverted, ref.bfloat16()), "inverse_rope", f"ratio {compress_ratio} inverse"
    )

    # Inverse then forward returns the original; a repeated forward does not.
    # The registered `rope` tolerance covers a single rotation against the
    # source, so the round trip is held to its own bound instead: two BF16
    # rotations of the same row, i.e. a couple of storage steps at the peak.
    restored = inverted.clone()
    torch.ops.trtllm.mla_rope_inplace(
        restored, positions, rotary.rotary_cos_sin, LOCAL_HEADS, NOPE_DIM, ROPE_DIM, False, False
    )
    round_trip = tg.compare(restored, y)
    peak = y.float().abs().max()
    two_steps = float(2 * 2.0 ** (torch.floor(torch.log2(peak)) - 7))
    assert round_trip["cosine"] >= TOL["rope"]["cosine_min"], (
        f"ratio {compress_ratio} round trip: cosine {round_trip['cosine']:.6f}"
    )
    assert round_trip["max_abs"] <= two_steps, (
        f"ratio {compress_ratio} round trip: max_abs {round_trip['max_abs']:.5f} exceeds two "
        f"BF16 steps at the peak ({two_steps:.5f}); the inverse is not the conjugate"
    )
    doubled = y.clone()
    for _ in range(2):
        torch.ops.trtllm.mla_rope_inplace(
            doubled,
            positions,
            rotary.rotary_cos_sin,
            LOCAL_HEADS,
            NOPE_DIM,
            ROPE_DIM,
            False,
            False,
        )
    assert not torch.allclose(doubled.float(), y.float(), atol=1e-2), (
        "rotating twice forward matched the identity, so the round trip proves nothing"
    )


def test_sm90_grouped_o_lora_bmm_matches_an_independent_projection():
    """The O-LoRA BMM is grouped: each O group sees only its own heads.

    Flattening all heads into one projection, or transposing the group axis,
    both yield an output of exactly the right shape, so the grouping is
    checked against an explicit per-group einsum.
    """
    heads_per_group = LOCAL_HEADS // LOCAL_GROUPS
    num_tokens = 24
    torch.manual_seed(41)
    attn = (torch.randn(num_tokens, LOCAL_HEADS, HEAD_DIM, device="cuda") * 0.3).bfloat16()
    o_a = (
        torch.randn(LOCAL_GROUPS, O_LORA_RANK, heads_per_group * HEAD_DIM, device="cuda")
        * (heads_per_group * HEAD_DIM) ** -0.5
    ).bfloat16()

    grouped = attn.view(num_tokens, LOCAL_GROUPS, heads_per_group * HEAD_DIM)
    out = torch.empty(num_tokens, LOCAL_GROUPS, O_LORA_RANK, device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.bmm_out(grouped.transpose(0, 1), o_a.transpose(1, 2), out.transpose(0, 1))

    ref = torch.einsum("tgd,grd->tgr", grouped.float(), o_a.float()).bfloat16()
    _assert_within(tg.compare(out, ref), "o_lora_output", "grouped O-LoRA")

    # Reversing the group order must change the result whenever there is more
    # than one group; with a single group per rank the check is vacuous, so it
    # is asserted only where it can bite.
    if LOCAL_GROUPS > 1:
        flipped = torch.einsum("tgd,grd->tgr", grouped.float(), o_a.flip(0).float()).bfloat16()
        assert not torch.equal(out, flipped), "the group axis is not being respected"


def test_sm90_tp8_head_and_group_geometry_is_the_checkpoint_contract():
    """64 Q heads shard to 8 per rank and 8 O groups to 1, with a 512 latent.

    These are the shapes every kernel above assumes; padding heads or faking a
    group count to satisfy a kernel is explicitly out of bounds for this
    bring-up, so the arithmetic is asserted rather than left implicit.
    """
    assert NUM_HEADS % TP_SIZE == 0 and LOCAL_HEADS == 8
    assert NUM_GROUPS % TP_SIZE == 0 and LOCAL_GROUPS == 1
    assert NUM_HEADS % NUM_GROUPS == 0
    assert LOCAL_HEADS % LOCAL_GROUPS == 0
    # The latent stays 512-wide per rank: it is one KV head, never sharded.
    assert NOPE_DIM + ROPE_DIM == HEAD_DIM
    assert HEAD_DIM % 128 == 0, "the FP8 block-scale path needs a 128-multiple latent"
