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
"""``QuantConfig.scale_fmt`` and the Hopper FP8 block-scale activation quantizer.

A checkpoint whose ``quantization_config`` declares ``scale_fmt: "ue8m0"`` was
produced against *power-of-two* block scales, and its own reference
implementation quantizes activations the same way. Quantizing with an FP32
scale instead is a different quantization of the same tensor -- not a more
accurate one -- and the two disagree by roughly one FP8 step, which is large
enough to dominate a numerical-parity comparison against that reference.

These tests pin three things:

* the op reproduces the reference quantizer *bitwise* under ``use_ue8m0=True``
  and provably does not under the default, so the flag is the whole difference;
* ``Linear`` reads the format from its own ``quant_config``, so a mixed-precision
  model cannot quantize one layer against another layer's declared recipe;
* a checkpoint that declares nothing keeps the previous behaviour exactly --
  the protected case, since this path is shared with every other FP8
  block-scale model;
* the accepted set is closed, matching the reference implementation's
  ``Literal[None, "ue8m0"]``, so a mis-cased or unknown declaration fails
  instead of quietly selecting the default quantizer.
"""

from typing import get_args

import pytest
import torch
from pydantic import ValidationError

from tensorrt_llm._torch.model_config import _validate_scale_fmt
from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod, Linear
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig, ScaleFmt
from tensorrt_llm.quantization.mode import QuantAlgo

FP8_MAX = 448.0
BLOCK = 128

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="needs a GPU")
skip_not_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 90,
    reason="the fp8_block_scaling_gemm branch under test is the SM90 one")


def reference_act_quant(x: torch.Tensor, block: int = BLOCK):
    """``inference/kernel.py::act_quant(x, 128, "ue8m0", float8_e8m0fnu)``.

    Written out rather than imported so this test, which guards a path shared
    with every FP8 block-scale model, does not depend on one model's evidence
    tree. ``fast_round_scale`` is ``2 ** ceil(log2(amax / 448))`` computed with
    IEEE-754 bit tricks; exponent arithmetic reproduces it exactly for finite
    positive inputs.
    """
    m, n = x.shape
    blocks = x.float().reshape(m, n // block, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(m, n), scale.reshape(m, n // block)


def dequantized_activation(x: torch.Tensor, use_ue8m0: bool) -> torch.Tensor:
    """What the op's FP8 codes and scales decode back to, in FP32."""
    m, n = x.shape
    m_padded = (m + 3) // 4 * 4
    blocks = n // BLOCK
    q, sf = torch.ops.trtllm.fp8_quantize_1x128(x, use_ue8m0)
    # The scale buffer is a flat, 128-byte-aligned, column-major
    # [n // 128, m_padded] region.
    scale = sf[:blocks * m_padded].reshape(blocks, m_padded)[:, :m].t()
    return (q[:m].float().reshape(m, blocks, BLOCK) *
            scale.contiguous()[:, :, None]).reshape(m, n)


# ---------------------------------------------------------------------------
# The op.
# ---------------------------------------------------------------------------


@skip_no_cuda
@pytest.mark.parametrize("shape", [(257, 4096), (32, 2048), (8, 256)])
def test_ue8m0_activation_quantization_is_bitwise_the_reference(shape):
    """``use_ue8m0=True`` reproduces the reference quantizer exactly."""
    torch.manual_seed(0)
    m, n = shape
    x = (torch.randn(m, n, device="cuda") * 3).to(torch.bfloat16)

    q_ref, s_ref = reference_act_quant(x)
    reference = (q_ref.float().reshape(m, n // BLOCK, BLOCK) *
                 s_ref[:, :, None]).reshape(m, n)

    assert torch.equal(dequantized_activation(x, True), reference)


@skip_no_cuda
@pytest.mark.parametrize("shape", [(257, 4096), (32, 2048)])
def test_the_default_scale_format_is_a_different_quantization(shape):
    """The default is not the reference, so the flag is load-bearing.

    Without this the bitwise test above could pass for a build in which the
    flag does nothing at all.
    """
    torch.manual_seed(0)
    m, n = shape
    x = (torch.randn(m, n, device="cuda") * 3).to(torch.bfloat16)

    q_ref, s_ref = reference_act_quant(x)
    reference = (q_ref.float().reshape(m, n // BLOCK, BLOCK) *
                 s_ref[:, :, None]).reshape(m, n)
    default = dequantized_activation(x, False)

    assert not torch.equal(default, reference)
    # The disagreement is on the order of the FP8 step, not of rounding noise.
    assert float((default - reference).abs().max()) > 0.1


@skip_no_cuda
def test_ue8m0_scales_are_powers_of_two_and_fp32_scales_are_not():
    torch.manual_seed(1)
    x = (torch.randn(64, 1024, device="cuda") * 3).to(torch.bfloat16)
    for use_ue8m0, expect_pow2 in ((True, True), (False, False)):
        _, sf = torch.ops.trtllm.fp8_quantize_1x128(x, use_ue8m0)
        scales = sf[:8 * 64].float()
        scales = scales[scales > 0]
        is_pow2 = torch.equal(scales, torch.exp2(torch.log2(scales).round()))
        assert is_pow2 is expect_pow2


# ---------------------------------------------------------------------------
# The module contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale_fmt, expected", [("ue8m0", True),
                                                 (None, False)])
def test_linear_reads_the_declared_format_from_its_own_quant_config(
        scale_fmt, expected):
    module = Linear(
        in_features=256,
        out_features=128,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                                 scale_fmt=scale_fmt),
        skip_create_weights_in_init=True,
    )
    assert FP8BlockScalesLinearMethod._use_ue8m0_activation_scales(
        module) is expected


def test_a_linear_without_a_quant_config_keeps_the_default():
    """The protected case: nothing declared, nothing changes."""

    class Bare:
        pass

    assert FP8BlockScalesLinearMethod._use_ue8m0_activation_scales(
        Bare()) is False
    module = Linear(in_features=256,
                    out_features=128,
                    bias=False,
                    dtype=torch.bfloat16)
    assert FP8BlockScalesLinearMethod._use_ue8m0_activation_scales(
        module) is False


def _fp8_block_scale_linear(scale_fmt, weight_bf16, device="cuda"):
    """A real FP8 block-scale ``Linear`` holding ``weight_bf16``'s quantization."""
    out_features, in_features = weight_bf16.shape
    module = Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                                 scale_fmt=scale_fmt),
    ).to(device)

    blocks = weight_bf16.float().reshape(out_features // BLOCK, BLOCK,
                                         in_features // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    scale = amax / FP8_MAX
    module.weight.data.copy_(
        (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn).reshape(out_features, in_features))
    module.weight_scale.data.copy_(scale.reshape(out_features // BLOCK,
                                                 in_features // BLOCK))
    dequantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn).float().reshape(
            out_features // BLOCK, BLOCK, in_features // BLOCK,
            BLOCK) * scale
    return module, dequantized.reshape(out_features, in_features)


@skip_not_hopper
def test_declared_ue8m0_moves_the_hopper_linear_towards_the_reference():
    """The end-to-end module claim, on the SM90 branch that was changed.

    The reference is the quantization the checkpoint's own implementation
    performs: activations through ``act_quant`` with power-of-two scales,
    against the same dequantized weights. The declared-format module must be
    strictly closer to it than the default, and the two must differ at all --
    a module that ignored the declaration would be bit-identical to the
    default.
    """
    torch.manual_seed(2)
    x = (torch.randn(256, 1024, device="cuda") * 2).to(torch.bfloat16)
    weight = (torch.randn(512, 1024, device="cuda") * 0.05).to(torch.bfloat16)

    declared, dequantized_weight = _fp8_block_scale_linear("ue8m0", weight)
    default, _ = _fp8_block_scale_linear(None, weight)

    q_ref, s_ref = reference_act_quant(x)
    act_ref = (q_ref.float().reshape(256, 1024 // BLOCK, BLOCK) *
               s_ref[:, :, None]).reshape(256, 1024)
    reference = (act_ref.to(torch.bfloat16)
                 @ dequantized_weight.to(torch.bfloat16).t()).float()

    with torch.inference_mode():
        got_declared = declared(x).float()
        got_default = default(x).float()

    assert not torch.equal(got_declared, got_default), (
        "the declared scale format did not reach the kernel")
    err_declared = float((got_declared - reference).abs().mean())
    err_default = float((got_default - reference).abs().mean())
    assert err_declared < err_default, (
        f"declared ue8m0 is not closer to the reference: "
        f"{err_declared:.6g} vs {err_default:.6g}")


# ---------------------------------------------------------------------------
# Config plumbing, including the protected no-declaration case.
# ---------------------------------------------------------------------------


def _load(hf_quant_config):
    from tensorrt_llm._torch.model_config import ModelConfig

    return ModelConfig.load_hf_quant_config(hf_quant_config,
                                            moe_backend="CUTLASS")


def test_declared_scale_fmt_is_carried_into_the_quant_config():
    quant_config, _ = _load({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    })
    assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert quant_config.group_size == 128
    assert quant_config.scale_fmt == "ue8m0"


def test_a_checkpoint_that_declares_nothing_keeps_the_previous_behaviour():
    """DeepSeek-V3 / R1 style: no ``scale_fmt`` key at all."""
    quant_config, _ = _load({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
    })
    assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert quant_config.scale_fmt is None


def test_the_default_quant_config_declares_nothing():
    assert QuantConfig().scale_fmt is None
    assert QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES).scale_fmt is None


# ---------------------------------------------------------------------------
# The closed contract. An unrecognised spelling has to be rejected: kept as
# data it still misses every ``== "ue8m0"`` comparison, so the model would
# quantize against a recipe the checkpoint never declared and say nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("declared", ["UE8M0", "Ue8m0", "fp32", "typo", ""])
def test_an_unrecognised_scale_fmt_is_rejected_by_the_public_config(declared):
    with pytest.raises(ValidationError):
        QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                    scale_fmt=declared)


@pytest.mark.parametrize("declared", ["UE8M0", "Ue8m0", "fp32", "typo"])
def test_an_unrecognised_scale_fmt_is_rejected_when_loading_a_checkpoint(
        declared):
    """The mis-cased case is the realistic one and must not load silently."""
    with pytest.raises(ValueError, match="scale_fmt"):
        _load({
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": declared,
            "weight_block_size": [128, 128],
        })


def test_the_supported_set_matches_the_reference_implementation():
    """``inference/model.py`` declares ``Literal[None, "ue8m0"]``."""
    assert get_args(ScaleFmt) == ("ue8m0", )
    assert _validate_scale_fmt("ue8m0") == "ue8m0"
    assert _validate_scale_fmt(None) is None
