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
"""Rules behind the two-interpreter DeepSeek-V4-Flash layer localization.

The localization itself needs eight H100s and ten minutes. Two classes of rule
decide whether what it produces means anything, and both are ordinary functions
that this pins at CPU speed:

* **which interpreter each half runs in.** The official model needs the
  reference venv; production modules must not run there, because that venv's
  ``tvm_ffi`` makes flashinfer's CuTe RMSNorm raise --- the first module of the
  first decoder layer. Registering the whole suite in ``NEEDS_REFERENCE_ENV``
  is what killed iteration 36, and nothing about that failure was cheap to see:
  it cost a full checkpoint load and an official-source load to reach.
* **whether the persisted source capture may judge the run.** Once the two
  halves are separate processes the reference arrives as a file, and a file
  that describes a different checkpoint, a different prompt rendering, a
  different world size or different tensors than it holds would still produce a
  per-layer curve --- one that reads exactly like a real divergence.
"""

import copy
import importlib.util
import json
import os
import sys

import pytest
import torch

_ACCURACY = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ),
    "tests",
    "integration",
    "defs",
    "accuracy",
)
sys.path.insert(0, os.path.join(_ACCURACY, "deepseek_v4_flash_h100"))

import layer_localization  # noqa: E402
import torch_goldens as tg  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dsv4_evidence_localization", os.path.join(_ACCURACY, "deepseek_v4_flash_h100_evidence.py")
)
evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence)


# ---------------------------------------------------------------------------
# Which interpreter each half runs in.
# ---------------------------------------------------------------------------


def test_both_halves_are_registered_suites():
    assert "layer_source_capture" in evidence.SUITES
    assert "layer_localization" in evidence.SUITES


def test_the_source_half_runs_in_the_reference_interpreter():
    """Only the interpreter that can import the official model captures it."""
    assert "layer_source_capture" in evidence.NEEDS_REFERENCE_ENV
    assert evidence._wants_reference_env(
        ["driver.py", "--suite", "layer_source_capture", "--output", "x.json"]
    )


def test_the_production_half_runs_in_the_pinned_interpreter():
    """The half that executes production modules must not run in the reference venv.

    This is the iteration-36 failure as a unit test: the reference venv's
    ``tvm_ffi`` raises inside flashinfer's CuTe RMSNorm, which a DeepSeek-V4
    decoder layer reaches before it reaches anything this suite measures.
    """
    assert "layer_localization" not in evidence.NEEDS_REFERENCE_ENV
    assert not evidence._wants_reference_env(
        ["driver.py", "--suite", "layer_localization", "--output", "x.json"]
    )


def test_both_halves_state_their_communicator():
    """Neither half may inherit the MPI/torch-dist choice from whoever spawned it."""
    assert "layer_localization" in evidence.NEEDS_TORCH_DISTRIBUTED
    assert "layer_source_capture" in evidence.NEEDS_TORCH_DISTRIBUTED


def test_the_production_half_shares_the_reference_venv_exclusion_with_eager_full_model():
    """The two suites that run production code are excluded for the same reason."""
    assert "eager_full_model" not in evidence.NEEDS_REFERENCE_ENV
    assert "layer_localization" not in evidence.NEEDS_REFERENCE_ENV


# ---------------------------------------------------------------------------
# The bridge between the two processes.
# ---------------------------------------------------------------------------


def _capture(tokens=3, layers=4, hidden=8, hc=2, seed=0, deep=(0,)):
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    def sub():
        stacked = {"layer_entry", "mid_residual"}
        return {
            name: rand(tokens, hc, hidden) if name in stacked else rand(tokens, hidden)
            for name in layer_localization.SUBLAYER_ORDER
        }

    return {
        "embed": rand(tokens, hidden),
        "layers": {lid: rand(tokens, hc, hidden) for lid in range(layers)},
        "sublayers": {lid: sub() for lid in deep},
        "hc_head": rand(1, hidden),
        "final_norm": rand(1, hidden),
        "logits": rand(17),
    }


def _write(tmp_path, ranks=(0, 1), prompts=("chat_geography",), **overrides):
    """A capture artifact plus its sidecars, as the source half would write them."""
    artifact_path = str(tmp_path / "cap.json")
    per_rank = {}
    for rank in ranks:
        captured = {pid: _capture(seed=rank * 10 + i) for i, pid in enumerate(prompts)}
        entry = layer_localization.save_capture(
            artifact_path, rank, captured, meta={"rank": rank, "world_size": len(ranks)}
        )
        per_rank[str(rank)] = entry
    artifact = {
        "checkpoint_revision": "rev0",
        "world_size": len(ranks),
        "manifest_provenance": {"sha256": {"prompts.json": "phash"}},
        "prompts": [{"id": pid} for pid in prompts],
        "compress_ratios": [0, 0, 4, 128],
        "deep_layers": [0],
        "identical_boundary_shapes": True,
        "reference_env": {"interpreter": "/refenv/bin/python3"},
        "per_rank": per_rank,
        "passed": True,
    }
    artifact.update(overrides)
    with open(artifact_path, "w") as handle:
        json.dump(artifact, handle)
    return artifact_path, artifact


def _usable(artifact, **overrides):
    kwargs = {
        "checkpoint_revision": "rev0",
        "prompts_sha256": "phash",
        "prompt_ids": ["chat_geography"],
        "world_size": 2,
    }
    kwargs.update(overrides)
    return layer_localization.capture_usable(artifact, **kwargs)


def test_each_rank_gets_its_own_sidecar(tmp_path):
    """Both stacks are model-parallel; a rank-crossed comparison is a wrong one."""
    _, artifact = _write(tmp_path, ranks=(0, 1, 2))
    paths = {entry["path"] for entry in artifact["per_rank"].values()}
    assert len(paths) == 3
    assert layer_localization.sidecar_path("/a/b.json", 5).endswith(".activations.rank5.pt")


def test_a_capture_round_trips(tmp_path):
    _, artifact = _write(tmp_path)
    loaded = layer_localization.load_capture(artifact, 0)
    assert sorted(loaded) == ["chat_geography"]
    entry = loaded["chat_geography"]
    assert sorted(entry["layers"]) == [0, 1, 2, 3]
    assert entry["embed"].shape == (3, 8)
    assert entry["hc_head"].shape == (1, 8)
    assert entry["final_norm"].shape == (1, 8)
    assert entry["logits"].shape == (17,)
    assert entry["layers"][2].shape == (3, 2, 8)


def test_the_epilogue_boundary_is_persisted():
    """``hc_head`` closes the gap between the last layer and the logits.

    Without it the curve stops at the final residual stack and everything
    between there and the head is one unsplit step --- which is precisely
    where iteration 37's curve put the divergence.
    """
    names = [name for name, _ in layer_localization._boundaries(_capture(layers=2, deep=()))]
    assert names == ["embed", "layer0", "layer1", "hc_head", "final_norm", "logits"]


def test_the_sub_boundaries_are_persisted_in_execution_order():
    """The split layer's chain crosses the process boundary with everything else."""
    names = [name for name, _ in layer_localization._boundaries(_capture(layers=2, deep=(1,)))]
    assert names[:3] == ["embed", "layer0", "layer1"]
    assert names[3:-3] == [f"sub1.{b}" for b in layer_localization.SUBLAYER_ORDER]
    assert names[-3:] == ["hc_head", "final_norm", "logits"]


def test_a_split_layer_round_trips(tmp_path):
    _, artifact = _write(tmp_path)
    entry = layer_localization.load_capture(artifact, 0)["chat_geography"]
    assert sorted(entry["sublayers"]) == [0]
    chain = entry["sublayers"][0]
    assert sorted(chain) == sorted(layer_localization.SUBLAYER_ORDER)
    # The two mHC residual stacks keep their hc dimension; the rest are 2-D.
    assert chain["layer_entry"].shape == (3, 2, 8)
    assert chain["mid_residual"].shape == (3, 2, 8)
    assert chain["attn_out"].shape == (3, 8)


def test_a_capture_that_split_a_different_layer_may_not_judge(tmp_path):
    """Same boundary names, different layer, is the silent-mismatch case."""
    _, artifact = _write(tmp_path)
    assert _usable(artifact, deep_layers=[0]) == []
    problems = _usable(artifact, deep_layers=[0, 5])
    assert any("did not split layers [5]" in p for p in problems)


def test_a_replaced_sidecar_is_refused(tmp_path):
    """The file hash catches a sidecar swapped wholesale."""
    _, artifact = _write(tmp_path)
    path = artifact["per_rank"]["0"]["path"]
    blob = torch.load(path, map_location="cpu", weights_only=True)
    blob["tensors"]["chat_geography/layer1"] += 1.0
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="not the file that capture produced"):
        layer_localization.load_capture(artifact, 0)


def test_a_missing_sidecar_is_refused(tmp_path):
    _, artifact = _write(tmp_path)
    os.remove(artifact["per_rank"]["0"]["path"])
    with pytest.raises(RuntimeError, match="recorded but missing"):
        layer_localization.load_capture(artifact, 0)


def test_a_rank_the_capture_never_covered_is_refused(tmp_path):
    _, artifact = _write(tmp_path, ranks=(0, 1))
    with pytest.raises(RuntimeError, match="has no rank 4"):
        layer_localization.load_capture(artifact, 4)


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("shape", [99, 8], "shape"),
        ("dtype", "torch.float16", "dtype"),
        ("sha256", "0" * 64, "do not hash"),
    ],
)
def test_a_tensor_that_is_not_what_the_capture_described_is_refused(
    tmp_path, field, value, expected
):
    """Beyond the file hash: one tensor reshaped, recast or edited in place."""
    _, artifact = _write(tmp_path)
    artifact = copy.deepcopy(artifact)
    artifact["per_rank"]["0"]["tensors"]["chat_geography/layer1"][field] = value
    with pytest.raises(RuntimeError, match=expected):
        layer_localization.load_capture(artifact, 0)


def test_a_dropped_or_added_boundary_is_refused(tmp_path):
    _, artifact = _write(tmp_path)
    dropped = copy.deepcopy(artifact)
    dropped["per_rank"]["0"]["tensors"]["chat_geography/layer99"] = {
        "shape": [1],
        "dtype": "torch.float32",
        "sha256": "0" * 64,
        "finite": True,
    }
    with pytest.raises(RuntimeError, match="absent from the sidecar"):
        layer_localization.load_capture(dropped, 0)

    added = copy.deepcopy(artifact)
    added["per_rank"]["0"]["tensors"].pop("chat_geography/layer1")
    with pytest.raises(RuntimeError, match="never declared"):
        layer_localization.load_capture(added, 0)


def test_provenance_problems_names_every_kind_of_disagreement():
    good = torch.ones(2, 3)
    declared = {"a": layer_localization._describe(good), "b": layer_localization._describe(good)}
    assert layer_localization.provenance_problems({"a": good, "b": good}, declared) == []
    problems = layer_localization.provenance_problems({"a": good, "c": good}, declared)
    assert any("b:" in p for p in problems)
    assert any("c:" in p for p in problems)


# ---------------------------------------------------------------------------
# Whether the persisted capture may judge the run.
# ---------------------------------------------------------------------------


def test_a_matching_capture_may_judge(tmp_path):
    _, artifact = _write(tmp_path)
    assert _usable(artifact) == []


def test_a_capture_of_another_checkpoint_may_not_judge(tmp_path):
    _, artifact = _write(tmp_path, checkpoint_revision="other")
    assert any("checkpoint" in p for p in _usable(artifact))


def test_a_capture_of_another_prompt_rendering_may_not_judge(tmp_path):
    """The manifest hash covers the rendered text, not only the prompt ids."""
    _, artifact = _write(tmp_path)
    artifact["manifest_provenance"]["sha256"]["prompts.json"] = "different"
    assert any("prompts manifest" in p for p in _usable(artifact))


def test_a_capture_from_another_world_size_may_not_judge(tmp_path):
    _, artifact = _write(tmp_path, ranks=(0, 1))
    assert any("ranks" in p for p in _usable(artifact, world_size=8))


def test_a_capture_missing_a_prompt_may_not_judge(tmp_path):
    _, artifact = _write(tmp_path)
    assert any("missing prompts" in p for p in _usable(artifact, prompt_ids=["cache_boundary_257"]))


def test_a_capture_missing_a_rank_may_not_judge(tmp_path):
    _, artifact = _write(tmp_path, ranks=(0, 1))
    artifact["world_size"] = 3
    assert any("no sidecar for ranks" in p for p in _usable(artifact, world_size=3))


def test_a_capture_that_failed_its_own_checks_may_not_judge(tmp_path):
    _, artifact = _write(tmp_path, passed=False)
    assert any("did not pass" in p for p in _usable(artifact))


def test_a_missing_capture_artifact_is_named_not_swallowed(tmp_path):
    with pytest.raises(RuntimeError, match="layer_source_capture"):
        layer_localization.read_artifact(str(tmp_path / "absent.json"))


# ---------------------------------------------------------------------------
# The reading the curve supports.
# ---------------------------------------------------------------------------


def _curve(deltas, tokens=1, sub=None):
    """A source/production pair whose per-layer rel_max_abs follows ``deltas``.

    ``compare`` divides by the reference RMS, so a reference of all-ones makes
    the recorded ``rel_max_abs`` equal the perturbation injected at that layer.
    ``sub`` injects the same way into layer 0's sub-boundary chain.
    """
    stacked = {"layer_entry", "mid_residual"}
    source = {
        "embed": torch.ones(tokens, 4),
        "layers": {lid: torch.ones(tokens, 2, 4) for lid in range(len(deltas))},
        "sublayers": {
            0: {
                name: torch.ones(tokens, 2, 4) if name in stacked else torch.ones(tokens, 4)
                for name in layer_localization.SUBLAYER_ORDER
            }
        },
        "hc_head": torch.ones(1, 4),
        "final_norm": torch.ones(1, 4),
        "logits": torch.ones(5),
    }
    trtllm = copy.deepcopy(source)
    trtllm["sublayers"][0]["path"] = "fused_hc"
    for lid, delta in enumerate(deltas):
        trtllm["layers"][lid][-1, 0, 0] += delta
    for name, delta in (sub or {}).items():
        trtllm["sublayers"][0][name][-1, ..., 0] += delta
    return source, trtllm


def test_a_step_curve_names_one_layer():
    """One wrong kernel: a single jump far above the median."""
    source, trtllm = _curve([0.001, 0.002, 0.003, 0.9, 0.901, 0.902])
    result = layer_localization.compare_stacks(source, trtllm, [0, 0, 4, 128, 4, 128], tg)
    assert result["largest_single_layer_jump"]["layer"] == 3
    assert result["jump_ratio_largest_over_median"] > 100
    assert result["first_layer_over_rel_max"] == 3


def test_a_ramp_curve_names_no_layer():
    """Accumulated rounding: a steady climb and a jump ratio near one."""
    source, trtllm = _curve([0.01 * (i + 1) for i in range(8)])
    result = layer_localization.compare_stacks(source, trtllm, [0] * 8, tg)
    assert result["jump_ratio_largest_over_median"] == pytest.approx(1.0, abs=0.05)


def test_a_boundary_shape_disagreement_is_not_compared():
    source, trtllm = _curve([0.0, 0.0])
    trtllm["layers"][1] = torch.ones(1, 2, 5)
    with pytest.raises(AssertionError, match="not the same boundary"):
        layer_localization.compare_stacks(source, trtllm, [0, 0], tg)


def test_the_source_may_live_on_another_device_than_the_production_half():
    """The source half ran in another process, so its tensors arrive on the host."""
    source, trtllm = _curve([0.0, 0.5])
    result = layer_localization.compare_stacks(source, trtllm, [0, 0], tg)
    assert result["per_layer"][1]["rel_max_abs"] == pytest.approx(0.5, rel=1e-5)


def test_a_last_position_error_is_not_diluted_by_the_positions_that_agree():
    """The epilogue is last-row only, so the layer curve must report that slice too.

    Eight positions, one of them wrong: the all-position cosine barely moves
    while the last-row cosine records the real damage. Reading the first
    against ``final_norm``'s cosine would compare two different measurements.
    """
    # One element of sixty-four perturbed by a full unit: over all eight
    # positions that is a cosine of ~0.993, over the last row alone ~0.959.
    source, trtllm = _curve([0.0, 1.0], tokens=8)
    result = layer_localization.compare_stacks(source, trtllm, [0, 0], tg)
    layer1 = result["per_layer"][1]
    assert layer1["cosine"] > layer1["last_row_cosine"]
    assert layer1["last_row_max_abs"] == pytest.approx(1.0, rel=1e-5)
    assert result["first_layer_under_cosine_last_row"] == 1
    # All eight positions together stay above the reading threshold, which is
    # exactly the dilution this metric exists to defeat.
    assert result["first_layer_under_cosine"] is None


def test_the_epilogue_boundaries_are_measured():
    source, trtllm = _curve([0.0])
    trtllm["hc_head"] = trtllm["hc_head"] + 0.25
    result = layer_localization.compare_stacks(source, trtllm, [0], tg)
    assert result["boundaries"]["hc_head"]["rel_max_abs"] == pytest.approx(0.25, rel=1e-5)
    assert result["boundaries"]["final_norm"]["rel_max_abs"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Inside the layer: which part of it owns the divergence.
# ---------------------------------------------------------------------------


def _first_over(sub):
    source, trtllm = _curve([0.0], sub=sub)
    result = layer_localization.compare_stacks(source, trtllm, [0], tg)
    return result["per_sublayer"]["0"]


def test_the_chain_is_reported_in_execution_order():
    detail = _first_over({})
    assert [e["boundary"] for e in detail["chain"]] == list(layer_localization.SUBLAYER_ORDER)
    assert detail["first_boundary_over_rel_max"] is None
    assert detail["mid_boundary_path"] == "fused_hc"


@pytest.mark.parametrize(
    "injected, owner",
    [
        ({"attn_out": 0.5}, "attn_out"),
        ({"attn_norm_out": 0.5, "attn_out": 0.5}, "attn_norm_out"),
        ({"moe_out": 0.5}, "moe_out"),
        ({"mid_residual": 0.5, "moe_in": 0.5}, "mid_residual"),
        ({"layer_entry": 0.5}, "layer_entry"),
    ],
)
def test_the_first_boundary_over_the_threshold_names_the_owner(injected, owner):
    """A divergence introduced at one step must be attributed to that step.

    This is the whole point of the split: with only the layer boundary,
    ``attn_out`` and ``moe_out`` are the same observation.
    """
    assert _first_over(injected)["first_boundary_over_rel_max"] == owner


def test_a_clean_chain_before_a_dirty_one_is_the_evidence():
    detail = _first_over({"attn_out": 0.5, "mid_residual": 0.5, "moe_in": 0.5, "moe_out": 0.5})
    chain = {e["boundary"]: e for e in detail["chain"]}
    assert chain["attn_norm_out"]["rel_max_abs"] == pytest.approx(0.0, abs=1e-9)
    assert chain["attn_out"]["rel_max_abs"] == pytest.approx(0.5, rel=1e-5)
    assert detail["first_boundary_over_rel_max"] == "attn_out"


def test_an_unsplit_layer_reports_no_chain():
    source, trtllm = _curve([0.0])
    source.pop("sublayers")
    trtllm.pop("sublayers")
    assert layer_localization.compare_stacks(source, trtllm, [0], tg)["per_sublayer"] == {}


# ---------------------------------------------------------------------------
# The two stacks must agree on which layer compresses how much.
# ---------------------------------------------------------------------------


def test_matching_compression_is_silent():
    assert layer_localization.ratio_problems([0, 0, 4, 128, 4], [0, 0, 4, 128, 4, 128], 5) == []


def test_a_compression_disagreement_names_the_layers():
    problems = layer_localization.ratio_problems([0, 0, 4, 4], [0, 0, 4, 128], 4)
    assert problems and "layers [3]" in problems[0]


def test_a_capture_without_ratios_is_a_problem():
    assert layer_localization.ratio_problems([0, 0], None, 2)
