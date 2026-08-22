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
"""SM90 sparse-MLA kernel tests, against the independent pure-Torch golden.

The reference here is `torch_goldens.sparse_attention` from the Stage-1
reference ladder, not a second implementation written alongside this one. That
golden is anchored: its two GEMMs are bit-exact against the checkpoint's own
tilelang `sparse_attn_kernel` (`kernel_contract` evidence suite), and it passes
the pre-registered tolerances against real checkpoint activations on all eight
ranks. Comparing against it is therefore a comparison against the source, one
rung removed, rather than two implementations agreeing on the same mistake.

Every test here needs CUDA: the kernel *is* the artifact, so a CPU-only or
skipped run proves nothing about it.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import sm90

_EVIDENCE = Path(__file__).resolve().parents[5] / "integration" / "defs" / "accuracy"
_GOLDENS_PATH = _EVIDENCE / "deepseek_v4_flash_h100" / "torch_goldens.py"
_DRIVER_PATH = _EVIDENCE / "deepseek_v4_flash_h100_evidence.py"


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tg = _load_module("deepseek_v4_flash_h100_torch_goldens", _GOLDENS_PATH)
# The evidence driver owns the registered judgment and the BF16 storage-step
# report, so this file gates on the same code the eight-rank suites do rather
# than on a copy of their numbers.
ev = _load_module("deepseek_v4_flash_h100_evidence", _DRIVER_PATH)
# The stage-by-stage localisation and its sweeps live next to the goldens so the
# `sparse_kernel_numerics` evidence suite and these tests run the same code.
numerics = _load_module(
    "deepseek_v4_flash_h100_sparse_kernel_numerics",
    _EVIDENCE / "deepseek_v4_flash_h100" / "sparse_kernel_numerics.py",
)
TOL = json.loads(
    (_EVIDENCE / "deepseek_v4_flash_h100" / "manifests" / "tolerances.json").read_text()
)["modules"]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# Bring-up geometry: 512-wide latent row (448 non-RoPE + 64 RoPE), 8 Q heads
# per TP8 rank, 128-token sliding window.
HEAD_DIM = 512
LOCAL_HEADS = 8
WINDOW = 128


def _pools(num_swa_tokens: int, num_cmp_tokens: int, device: str, seed: int):
    torch.manual_seed(seed)
    swa = (torch.randn(num_swa_tokens, HEAD_DIM, device=device) * 0.5).bfloat16()
    cmp_ = (torch.randn(num_cmp_tokens, HEAD_DIM, device=device) * 0.5).bfloat16()
    return swa, cmp_


def _dual_pool_indices(
    positions: torch.Tensor, num_swa_indices: int, num_cmp_indices: int, ratio: int
):
    """The index layout `deepseek_v4_local_to_global_indices` emits.

    SWA slots first, then compressed slots, `-1` for padding. Written from the
    documented layout rather than by calling that kernel, so a bug there cannot
    make both sides agree.
    """
    device = positions.device
    swa_offsets = torch.arange(num_swa_indices, device=device)
    start = (positions.unsqueeze(1) - num_swa_indices + 1).clamp(min=0)
    swa = start + swa_offsets
    swa = torch.where(swa > positions.unsqueeze(1), -1, swa)

    if num_cmp_indices == 0:
        return swa.int().contiguous()

    col = torch.arange(num_cmp_indices, device=device)
    num_valid = (positions + 1) // ratio
    cmp_ = torch.where(col.unsqueeze(0) < num_valid.unsqueeze(1), col.unsqueeze(0), -1)
    return torch.cat([swa, cmp_], dim=1).int().contiguous()


def _golden(q, swa_pool, cmp_pool, indices, num_swa_indices, sink, scale):
    """Drive the ladder golden with the same dual-pool selection.

    The golden takes a single `kv` plane, so the two pools are concatenated and
    the compressed slots are shifted past the SWA rows. That is a re-indexing of
    the same rows, not a change of arithmetic.
    """
    if cmp_pool is None:
        kv = swa_pool
        shifted = indices
    else:
        kv = torch.cat([swa_pool, cmp_pool], dim=0)
        slot = torch.arange(indices.shape[1], device=indices.device)
        shifted = torch.where(
            (indices >= 0) & (slot.unsqueeze(0) >= num_swa_indices),
            indices + swa_pool.shape[0],
            indices,
        )
    return tg.sparse_attention(
        q.unsqueeze(0), kv.unsqueeze(0), sink, shifted.unsqueeze(0), scale
    ).squeeze(0)


def _assert_matches_golden(got, ref, label):
    """Gate on the registered `sparse_attention_output` entry, read from the manifest.

    The limits are not spelled out here on purpose. This kernel and the eight-
    rank evidence suites judge the same tensor, so a number copied into this
    file would be a second, silently diverging copy of the gate --- which is
    exactly what happened before the entry was re-registered on BF16 storage
    resolution, leaving a stale `rel_max_abs <= 0.03` here that no longer
    matched what the artifacts enforced.
    """
    got_f, ref_f = got.float(), ref.float()
    metrics = {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                ref_f.flatten().double(), got_f.flatten().double(), dim=0
            )
        ),
        "rel_max_abs": float((got_f - ref_f).abs().max()) / float(ref_f.square().mean().sqrt()),
        "finite": bool(torch.isfinite(got_f).all()),
    }
    storage = ev._ulp_report(got, ref)
    passed, problems = ev._judge(metrics, TOL["sparse_attention_output"], storage)
    assert passed, (
        f"{label}: {problems} (cosine {metrics['cosine']:.6f}, "
        f"rel_max_abs {metrics['rel_max_abs']:.3e}, "
        f"steps {storage['abs_max_element_steps']}, "
        f"beyond_one_step {storage['elements_beyond_one_step']})"
    )
    return metrics["rel_max_abs"], metrics["cosine"]


def test_the_softmax_exponential_is_correctly_rounded_not_the_hardware_approximation():
    """`tl.exp` is an approximation, and this kernel must not be using it.

    Triton lowers `tl.exp` to the hardware `ex2.approx` sequence. On the range
    an online softmax produces --- `score - running_max`, always <= 0 --- it
    carries up to ~15 FP32 ulp and disagrees with `torch.exp` on 86% of values,
    where `libdevice.exp` is bit-identical to it. That is not a free speedup:
    it made the eight-rank decode replay disagree with the checkpoint's own
    kernel on a handful of elements per rank, and swapping it made every decode
    step bit-exact. The approximation is easy to reintroduce by writing the
    obvious `tl.exp`, so the difference between the two is asserted here rather
    than left as a comment.
    """
    report = numerics.exp_implementations()
    assert report["libdevice.exp"]["differing_vs_torch_exp"] == 0
    # Self-verifying: if `tl.exp` ever became correctly rounded this test would
    # stop discriminating, and it should say so rather than silently pass.
    assert report["tl.exp"]["differing_vs_torch_exp"] > 0, (
        "tl.exp now matches torch.exp, so this test no longer proves anything"
    )
    assert report["tl.exp"]["worst_fp32_ulp"] > 1.0


def test_the_scores_and_attention_weights_are_bit_exact_with_the_reference():
    """The two stages that have one correct answer, not a rounding order.

    Scores are BF16 operands into an FP32 accumulator over the same contraction
    on both sides, and the attention weights are a transcendental with one
    correctly-rounded value; both must agree exactly. The denominator and the
    output accumulator are *not* asserted, because those differ by the reduction
    and contraction order each backend picks, which neither implementation can
    dictate to the other.
    """
    stages = numerics.stage_agreement(tg)["stages"]
    for name in ("scores", "probs_fp32", "probs_bf16"):
        assert stages[name]["differing"] == 0, (
            f"{name} disagrees with the golden on {stages[name]['differing']} of "
            f"{stages[name]['elements']} values"
        )


@pytest.mark.parametrize(
    "seq_len,ratio,num_cmp_indices,label,seed",
    [
        (257, 1, 0, "ratio0_swa_only", 101),
        (257, 4, 64, "ratio4_indexer", 103),
        (257, 128, 4, "ratio128_hca", 107),
    ],
)
def test_matches_the_source_anchored_golden_for_each_sparse_schedule(
    seq_len, ratio, num_cmp_indices, label, seed
):
    """Prefill across all three schedules the checkpoint actually uses.

    257 tokens crosses the 128-token SWA boundary twice, so the window's
    wrap-around and the ratio-4/128 compression boundaries are all exercised
    rather than only the first block. Seeds are fixed literals, not
    `hash(label)`: PYTHONHASHSEED is randomized per process, so the reported
    metrics would not be reproducible between runs.
    """
    device = "cuda"
    swa_pool, cmp_pool = _pools(seq_len, max(num_cmp_indices, 1), device, seed=seed)
    torch.manual_seed(7)
    q = (torch.randn(seq_len, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    sink = torch.randn(LOCAL_HEADS, device=device, dtype=torch.float32)
    scale = float(HEAD_DIM) ** -0.5

    positions = torch.arange(seq_len, device=device)
    indices = _dual_pool_indices(positions, WINDOW, num_cmp_indices, max(ratio, 1))
    pool = cmp_pool if num_cmp_indices else None

    got = sm90.sparse_mla_dual_pool(q, swa_pool, pool, indices, WINDOW, scale, attn_sink=sink)
    ref = _golden(q, swa_pool, pool, indices, WINDOW, sink, scale)

    assert got.shape == (seq_len, LOCAL_HEADS, HEAD_DIM)
    assert got.dtype == torch.bfloat16
    _assert_matches_golden(got, ref, label)


def test_decode_step_reads_cached_rows_at_a_block_boundary():
    """One-token decode at 127/128/129, where the window wraps a KV block."""
    device = "cuda"
    kv_len = 300
    swa_pool, cmp_pool = _pools(kv_len, 8, device, seed=11)
    torch.manual_seed(3)
    positions = torch.tensor([127, 128, 129, 255, 256], device=device)
    q = (torch.randn(positions.numel(), LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    sink = torch.randn(LOCAL_HEADS, device=device, dtype=torch.float32)
    scale = float(HEAD_DIM) ** -0.5

    indices = _dual_pool_indices(positions, WINDOW, 4, 128)
    got = sm90.sparse_mla_dual_pool(q, swa_pool, cmp_pool, indices, WINDOW, scale, attn_sink=sink)
    ref = _golden(q, swa_pool, cmp_pool, indices, WINDOW, sink, scale)
    _assert_matches_golden(got, ref, "decode_boundary")


def test_a_padded_slot_never_contributes_whatever_sits_at_pool_row_zero():
    """`-1` must mask, not index row 0 --- the classic gather bug."""
    device = "cuda"
    swa_pool, cmp_pool = _pools(64, 8, device, seed=5)
    swa_pool[0] = 1e3  # would dominate every softmax if -1 were read as 0
    cmp_pool[0] = 1e3
    torch.manual_seed(1)
    q = (torch.randn(4, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    sink = torch.full((LOCAL_HEADS,), -60.0, device=device, dtype=torch.float32)
    scale = float(HEAD_DIM) ** -0.5

    # Rows 1..3 selected, everything else padded, in both pool regions.
    indices = torch.full((4, WINDOW + 4), -1, dtype=torch.int32, device=device)
    indices[:, :3] = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    indices[:, WINDOW] = 1

    got = sm90.sparse_mla_dual_pool(q, swa_pool, cmp_pool, indices, WINDOW, scale, attn_sink=sink)
    ref = _golden(q, swa_pool, cmp_pool, indices, WINDOW, sink, scale)
    _assert_matches_golden(got, ref, "padded_slots")
    assert got.abs().max() < 100.0, "row 0 leaked into the output"


def test_the_sink_only_shrinks_the_output_and_is_actually_wired_in():
    """A denominator-only term rescales each head; it cannot rotate one."""
    device = "cuda"
    swa_pool, _ = _pools(200, 1, device, seed=13)
    torch.manual_seed(2)
    q = (torch.randn(16, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    scale = float(HEAD_DIM) ** -0.5
    positions = torch.arange(16, device=device) + 150
    indices = _dual_pool_indices(positions, WINDOW, 0, 1)

    tiny = sm90.sparse_mla_dual_pool(
        q, swa_pool, None, indices, WINDOW, scale, torch.full((LOCAL_HEADS,), -60.0, device=device)
    )
    large = sm90.sparse_mla_dual_pool(
        q, swa_pool, None, indices, WINDOW, scale, torch.full((LOCAL_HEADS,), 20.0, device=device)
    )
    assert large.abs().max() < tiny.abs().max(), "sink is not reaching the denominator"
    cos = torch.nn.functional.cosine_similarity(tiny.float(), large.float(), dim=-1)
    assert float(cos.min()) > 0.999, "sink steered the output instead of scaling it"


def test_perturbing_one_selected_index_changes_the_output():
    """Guards against the table being ignored and a dense window read instead.

    A sparse path that quietly attends to the whole window produces perfectly
    plausible text, so this is checked rather than assumed.
    """
    device = "cuda"
    swa_pool, cmp_pool = _pools(300, 16, device, seed=17)
    torch.manual_seed(4)
    q = (torch.randn(8, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    sink = torch.zeros(LOCAL_HEADS, device=device, dtype=torch.float32)
    scale = float(HEAD_DIM) ** -0.5
    positions = torch.arange(8, device=device) + 280
    indices = _dual_pool_indices(positions, WINDOW, 2, 128)

    base = sm90.sparse_mla_dual_pool(q, swa_pool, cmp_pool, indices, WINDOW, scale, attn_sink=sink)
    moved = indices.clone()
    moved[:, WINDOW] = 15  # a different compressed row
    other = sm90.sparse_mla_dual_pool(q, swa_pool, cmp_pool, moved, WINDOW, scale, attn_sink=sink)
    assert not torch.equal(base, other), "selected indices are not reaching the gather"


def test_the_two_pools_are_addressed_separately():
    """Compressed slots must read the compressed pool, not the SWA pool.

    Both regions carry small token indices, so a kernel that used one base
    pointer for the whole row would still run and still look reasonable. Here
    the pools hold deliberately different values, so mixing them up shows.
    """
    device = "cuda"
    swa_pool, cmp_pool = _pools(160, 8, device, seed=19)
    cmp_pool = cmp_pool * 4.0  # distinguishable from any SWA row
    torch.manual_seed(6)
    q = (torch.randn(4, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    sink = torch.zeros(LOCAL_HEADS, device=device, dtype=torch.float32)
    scale = float(HEAD_DIM) ** -0.5
    positions = torch.full((4,), 155, device=device)
    indices = _dual_pool_indices(positions, WINDOW, 1, 128)

    got = sm90.sparse_mla_dual_pool(q, swa_pool, cmp_pool, indices, WINDOW, scale, attn_sink=sink)
    ref = _golden(q, swa_pool, cmp_pool, indices, WINDOW, sink, scale)
    _assert_matches_golden(got, ref, "dual_pool")

    # And the same run against a kernel fed the SWA pool twice must differ.
    wrong = sm90.sparse_mla_dual_pool(q, swa_pool, swa_pool, indices, WINDOW, scale, attn_sink=sink)
    assert not torch.equal(got, wrong), "compressed slots did not use the compressed pool"


@pytest.mark.parametrize(
    "tokens,live_compressed,label",
    [(1, 64, "decode_ratio4"), (1, 4, "decode_ratio128"), (257, 64, "prefill_ratio4")],
)
def test_padding_the_index_table_wider_does_not_change_a_single_bit(tokens, live_compressed, label):
    """The table width the runtime happens to allocate must not be observable.

    TensorRT-LLM pads the compressed region to the configured Indexer top-k, so
    a ratio-4 layer hands the kernel a 640-wide table where the source's own is
    192. That is seven extra all-padding tiles through the online softmax, and
    if any of them were not exactly neutral the replay would read it as an
    arithmetic disagreement with the source and there would be no way to tell
    the two apart. Bitwise equality is the right assertion here, not a
    tolerance: neutral means neutral.
    """
    device = "cuda"
    swa_pool, cmp_pool = _pools(4096, 2048, device, seed=23)
    torch.manual_seed(11)
    q = (torch.randn(tokens, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    sink = torch.randn(LOCAL_HEADS, device=device, dtype=torch.float32)
    scale = float(HEAD_DIM) ** -0.5

    swa = torch.arange(WINDOW, device=device).unsqueeze(0).repeat(tokens, 1).int()
    live = torch.arange(live_compressed, device=device).unsqueeze(0).repeat(tokens, 1).int()
    pad = torch.full((tokens, 512 - live_compressed), -1, device=device, dtype=torch.int32)

    narrow = torch.cat([swa, live], dim=1).contiguous()
    wide = torch.cat([swa, live, pad], dim=1).contiguous()

    a = sm90.sparse_mla_dual_pool(q, swa_pool, cmp_pool, narrow, WINDOW, scale, attn_sink=sink)
    b = sm90.sparse_mla_dual_pool(q, swa_pool, cmp_pool, wide, WINDOW, scale, attn_sink=sink)
    assert torch.equal(a, b), (
        f"{label}: widening the table from {narrow.shape[1]} to {wide.shape[1]} slots changed "
        f"{int((a != b).sum())} of {a.numel()} values; the padding tiles are not neutral"
    )


def test_dispatch_counter_proves_this_path_ran():
    """Evidence hook: a silent fallback is otherwise indistinguishable."""
    device = "cuda"
    swa_pool, _ = _pools(64, 1, device, seed=23)
    q = (torch.randn(2, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    indices = _dual_pool_indices(torch.arange(2, device=device), WINDOW, 0, 1)

    sm90.reset_dispatch_counts()
    assert sm90.dispatch_counts()["sparse_mla_dual_pool"] == 0
    sm90.sparse_mla_dual_pool(q, swa_pool, None, indices, WINDOW, float(HEAD_DIM) ** -0.5)
    sm90.sparse_mla_dual_pool(q, swa_pool, None, indices, WINDOW, float(HEAD_DIM) ** -0.5)
    assert sm90.dispatch_counts()["sparse_mla_dual_pool"] == 2
    # A snapshot, not a live handle -- a caller cannot silently reset it.
    snapshot = sm90.dispatch_counts()
    snapshot["sparse_mla_dual_pool"] = 0
    assert sm90.dispatch_counts()["sparse_mla_dual_pool"] == 2


@pytest.mark.parametrize("num_cmp_slots", [1, 4, 512])
def test_a_missing_compressed_pool_is_rejected_rather_than_read_from_swa(num_cmp_slots):
    """A compressed region without its pool must fail, not silently misroute.

    Compressed slot values are small token indices, so reading them from the
    SWA pool returns a perfectly ordinary attention output -- measured at
    max_abs 2.7 against the correct dual-pool call for a single extra slot.
    Presence of the region is therefore decided by the index table's width
    alone, never by whether the caller happened to pass a pool.
    """
    device = "cuda"
    swa_pool, _ = _pools(160, 1, device, seed=29)
    q = (torch.randn(4, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    indices = torch.zeros((4, WINDOW + num_cmp_slots), dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="compress_pool is None"):
        sm90.sparse_mla_dual_pool(q, swa_pool, None, indices, WINDOW, float(HEAD_DIM) ** -0.5)

    # The SWA-only table of the same kernel still runs: it is the compressed
    # region that is required to come with a pool, not the pool that is
    # required always.
    swa_only = indices[:, :WINDOW].contiguous()
    sm90.sparse_mla_dual_pool(q, swa_pool, None, swa_only, WINDOW, float(HEAD_DIM) ** -0.5)


def test_a_quantized_pool_is_rejected_rather_than_misread():
    """The kernel reads latent rows directly; a silent reinterpretation of an
    FP8 pool as BF16 would produce garbage that still looks like attention."""
    device = "cuda"
    q = (torch.randn(2, LOCAL_HEADS, HEAD_DIM, device=device) * 0.5).bfloat16()
    fp8_pool = torch.zeros(64, HEAD_DIM, device=device, dtype=torch.float8_e4m3fn)
    indices = _dual_pool_indices(torch.arange(2, device=device), WINDOW, 0, 1)
    with pytest.raises(AssertionError, match="dtype"):
        sm90.sparse_mla_dual_pool(q, fp8_pool, None, indices, WINDOW, float(HEAD_DIM) ** -0.5)


# ---------------------------------------------------------------------------
# The rotary table the production model builds.
#
# `rope.py` replaces the shared NumPy table with the checkpoint's own recipe.
# That replacement has to reproduce the *layout* the native kernels read as
# exactly as it reproduces the values, and at checkpoint scale it also has to
# not cost one table per layer: `max_position_embeddings` is 1,048,576, so an
# interleaved float32 table is 512 MiB and 43 private copies do not fit.
# ---------------------------------------------------------------------------


def _production_rope(duplicate_data: bool, max_positions: int = 4096):
    from tensorrt_llm._torch.attention_backend.interface import RopeParams

    return RopeParams(
        dim=64,
        theta=10000,
        max_positions=max_positions,
        original_max_positions=65536,
        max_seq_len=max_positions,
        duplicate_data=duplicate_data,
    )


def test_the_interleaved_table_uses_the_position_stride_the_native_kernel_reads():
    """`mlaKernels.cu` addresses the table as `float2*` and strides one position
    by ROPE_DIM float2 entries, so a position occupies `2 * dim` floats: the
    `dim // 2` (cos, sin) pairs written twice. `RopeParams.from_config` turns
    that duplication on for every model carrying `qk_rope_head_dim`, DeepSeek-V4
    included. Emitting one copy would halve the stride and rotate every token
    past the first by the wrong angle -- a silent accuracy failure, not a crash.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import rope as rope_mod

    dim, positions = 64, 4096
    single = rope_mod.deepseek_v4_rope_const_params(
        _production_rope(duplicate_data=False, max_positions=positions), interleave=True
    )[1]
    duplicated = rope_mod.deepseek_v4_rope_const_params(
        _production_rope(duplicate_data=True, max_positions=positions), interleave=True
    )[1]

    assert single.numel() == positions * dim
    assert duplicated.numel() == positions * 2 * dim

    per_pos = duplicated.reshape(positions, dim, 2)
    torch.testing.assert_close(per_pos[:, : dim // 2], per_pos[:, dim // 2 :], rtol=0, atol=0)
    torch.testing.assert_close(
        per_pos[:, : dim // 2].reshape(positions, -1),
        single.reshape(positions, -1),
        rtol=0,
        atol=0,
    )


def test_the_non_interleaved_table_never_duplicates():
    """`create_rope_const_params` slices the duplicate away for this layout
    (`[:, :dim // 2, :]`), so `RotaryEmbedding` must be handed the short one
    whatever `duplicate_data` says."""
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import rope as rope_mod

    positions = 4096
    tables = [
        rope_mod.deepseek_v4_rope_const_params(
            _production_rope(duplicate_data=dup, max_positions=positions), interleave=False
        )[1]
        for dup in (False, True)
    ]
    assert tables[0].numel() == tables[1].numel() == positions * 64
    torch.testing.assert_close(tables[0], tables[1], rtol=0, atol=0)


def test_equal_rope_parameters_share_one_table_inside_a_model_context():
    """43 layers resolve to two distinct RopeParams values, so they must resolve
    to two tables. A weak cache is not sufficient: the installer hands each
    module a *view*, and a view keeps the storage alive without keeping the
    cached Python object alive, so weak entries died between layers and every
    layer rebuilt its own copy.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import rope as rope_mod
    from tensorrt_llm._torch.utils import model_extra_attrs

    class _Holder(torch.nn.Module):
        pass

    rope = _production_rope(duplicate_data=True)
    with model_extra_attrs({}) as _:
        first, second = _Holder(), _Holder()
        # An existing table forces the reshape-to-existing-shape path, which is
        # what defeated the weak cache.
        size = 4096 * 2 * 64
        first.rotary_cos_sin = torch.empty(1, size, device="cuda")
        second.rotary_cos_sin = torch.empty(1, size, device="cuda")
        rope_mod.install_deepseek_v4_rope_table(first, rope, interleave=True)
        rope_mod.install_deepseek_v4_rope_table(second, _production_rope(True), interleave=True)
        assert first.rotary_cos_sin.data_ptr() == second.rotary_cos_sin.data_ptr()


def test_a_table_whose_length_disagrees_with_the_shared_builder_is_rejected():
    """The two sides must agree on the position stride; a mismatch is a layout
    bug in this module, not something to reshape around."""
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import rope as rope_mod

    class _Holder(torch.nn.Module):
        pass

    holder = _Holder()
    holder.rotary_cos_sin = torch.empty(1, 4096 * 64 * 2 + 8, device="cuda")
    with pytest.raises(RuntimeError, match="layouts have diverged"):
        rope_mod.install_deepseek_v4_rope_table(
            holder, _production_rope(duplicate_data=True), interleave=True
        )
