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
"""CPU coverage for the DeepSeek-V4-Flash native-generate cross-check.

The cross-check is the only thing standing between the evidence ladder and an
unverified assumption: the checkpoint's official generation loop is
hand-written, so its ``position_ids``, window bookkeeping, KV threading and
greedy tie-breaking are re-implementations that could silently disagree with
canonical ``generate()``.

A comparison that tolerates a short fixture, a missing prompt, a matching
prefix, or a fixture produced by a sampling run would report success for
exactly those bugs. These tests drive the comparison with hand-built fixtures
so each of those failure modes is pinned, on CPU, with no GPU and no
checkpoint.
"""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_DRIVER_PATH = (
    Path(__file__).resolve().parents[3]
    / "integration"
    / "defs"
    / "accuracy"
    / "deepseek_v4_flash_h100_evidence.py"
)
_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "integration"
    / "defs"
    / "accuracy"
    / "deepseek_v4_flash_h100"
    / "manifests"
    / "prompts.json"
)
_MANIFEST_DIR = _MANIFEST_PATH.parent
_TOLERANCES_PATH = _MANIFEST_DIR / "tolerances.json"


def _load():
    name = "deepseek_v4_flash_h100_evidence"
    spec = importlib.util.spec_from_file_location(name, _DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load()

REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
MANIFEST_SHA = "0" * 64

# One prompt per category the cross-check requires, plus a spare, so a test can
# drop any single one and still be arguing about the rule under test.
_SPECS = [
    ("chat_a", "plain_chat"),
    ("chat_b", "plain_chat"),
    ("reason_a", "reasoning"),
    ("code_a", "code"),
    ("boundary_a", "cache_boundary"),
]


def _manifest():
    return {
        "checkpoint_revision": REVISION,
        "prompts": [
            {
                "id": pid,
                "category": category,
                "rendered_sha256": f"sha-{pid}",
                "token_ids": [1, 2, 3],
                "num_tokens": 3,
                "thinking_mode": "chat",
            }
            for pid, category in _SPECS
        ],
    }


def _tokens(pid: str) -> list[int]:
    return [100 + i + len(pid) for i in range(8)]


def _margins(pid: str) -> list[float]:
    """Per-step top1-top2 gaps. Diagnostics only: they never gate anything."""
    return [1.0 + i for i in range(len(_tokens(pid)))]


def _fixture():
    return {
        "checkpoint_revision": REVISION,
        "decoding": {"do_sample": False, "num_beams": 1, "temperature": 0, "top_k": 1},
        "provenance": {
            "generator": "AutoModelForCausalLM.generate(do_sample=False)",
            "transformers_version": "5.15.1",
            "conversion_code_sha256": "c" * 64,
            "prompts_manifest_sha256": MANIFEST_SHA,
        },
        "required_prompt_ids": [pid for pid, _ in _SPECS],
        "prompts": {
            pid: {
                "tokens": _tokens(pid),
                "rendered_sha256": f"sha-{pid}",
                "top1_top2_margin": _margins(pid),
            }
            for pid, _ in _SPECS
        },
    }


def _generation():
    return {pid: {"tokens": _tokens(pid)} for pid, _ in _SPECS}


def _compare(fixture=None, generation=None, manifest=None, sha=MANIFEST_SHA):
    return ev.compare_native_golden(
        fixture if fixture is not None else _fixture(),
        generation if generation is not None else _generation(),
        manifest if manifest is not None else _manifest(),
        sha,
    )


# ---------------------------------------------------------------------------
# Positive control.
# ---------------------------------------------------------------------------


def test_matching_fixture_and_loop_pass():
    result = _compare()
    assert result["passed"], result["problems"]
    assert result["status"] == "compared"
    assert result["prompts_compared"] == len(_SPECS)
    assert all(p["match"] for p in result["per_prompt"].values())


# ---------------------------------------------------------------------------
# Token-equality rules. The first of these is the exact probe that exposed the
# original prefix-only comparison.
# ---------------------------------------------------------------------------


def test_official_loop_stopping_early_is_not_a_match():
    gen = _generation()
    gen["code_a"]["tokens"] = _tokens("code_a")[:1]
    result = _compare(generation=gen)
    assert not result["passed"]
    assert "length mismatch" in result["per_prompt"]["code_a"]["problem"]


def test_official_loop_running_long_is_not_a_match():
    gen = _generation()
    gen["code_a"]["tokens"] = _tokens("code_a") + [7]
    result = _compare(generation=gen)
    assert not result["passed"]
    assert "length mismatch" in result["per_prompt"]["code_a"]["problem"]


def test_token_divergence_is_reported_with_its_index():
    gen = _generation()
    gen["reason_a"]["tokens"] = list(_tokens("reason_a"))
    gen["reason_a"]["tokens"][3] = -1
    result = _compare(generation=gen)
    assert not result["passed"]
    assert result["per_prompt"]["reason_a"]["first_divergence"] == 3


def test_empty_fixture_generation_fails():
    fixture = _fixture()
    fixture["prompts"]["chat_a"]["tokens"] = []
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert "empty" in result["per_prompt"]["chat_a"]["problem"]


def test_prompt_missing_from_the_fixture_fails():
    fixture = _fixture()
    del fixture["prompts"]["boundary_a"]
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert result["per_prompt"]["boundary_a"]["problem"] == "prompt missing from fixture"


def test_prompt_missing_from_this_run_fails():
    gen = _generation()
    del gen["boundary_a"]
    result = _compare(generation=gen)
    assert not result["passed"]
    assert "not generated" in result["per_prompt"]["boundary_a"]["problem"]


def test_fixture_prompt_outside_its_required_set_fails():
    fixture = _fixture()
    fixture["prompts"]["smuggled"] = {"tokens": [1], "rendered_sha256": "x"}
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert any("neither requires nor declares" in p for p in result["problems"])


# ---------------------------------------------------------------------------
# Coverage rules.
# ---------------------------------------------------------------------------


def test_fewer_than_five_prompts_fails():
    fixture = _fixture()
    fixture["required_prompt_ids"] = ["chat_a", "reason_a", "code_a", "boundary_a"]
    fixture["prompts"].pop("chat_b")
    gen = _generation()
    gen.pop("chat_b")
    result = _compare(fixture=fixture, generation=gen)
    assert not result["passed"]
    assert any("need >= 5" in p for p in result["problems"])


# ---------------------------------------------------------------------------
# Explicitly declared non-gating prompts. A prompt the fixture does not gate on
# is still compared and still reported with its true result -- the one thing
# that must never happen is a divergent prompt being labelled a match.
# ---------------------------------------------------------------------------


def test_a_non_gating_prompt_that_diverges_is_reported_as_a_mismatch():
    fixture = _fixture()
    fixture["required_prompt_ids"] = [p for p, _ in _SPECS if p != "chat_a"]
    fixture["non_gating_prompt_ids"] = ["chat_a"]
    fixture["non_gating_reason"] = "recorded reason"
    gen = _generation()
    gen["chat_a"]["tokens"] = [999]
    result = _compare(fixture=fixture, generation=gen)
    detail = result["per_prompt"]["chat_a"]
    assert detail["match"] is False
    assert detail["gating"] is False
    assert "chat_a" in result["mismatched_prompt_ids"]
    assert "chat_a" not in result["matched_prompt_ids"]
    # Four required prompts is below the floor, so the gate still fails loudly.
    assert not result["passed"]
    assert any("need >= 5" in p for p in result["problems"])


def test_a_non_gating_mismatch_does_not_fail_a_gate_that_is_otherwise_met():
    manifest = _manifest()
    manifest["prompts"].append(
        {
            "id": "chat_c",
            "category": "plain_chat",
            "rendered_sha256": "sha-chat_c",
            "token_ids": [1],
            "num_tokens": 1,
            "thinking_mode": "chat",
        }
    )
    fixture = _fixture()
    fixture["prompts"]["chat_c"] = {
        "tokens": _tokens("chat_c"),
        "rendered_sha256": "sha-chat_c",
        "top1_top2_margin": _margins("chat_c"),
    }
    fixture["required_prompt_ids"] = [p for p, _ in _SPECS if p != "chat_a"] + ["chat_c"]
    fixture["non_gating_prompt_ids"] = ["chat_a"]
    fixture["non_gating_reason"] = "two reference implementations disagree on a near-tie"
    gen = _generation()
    gen["chat_c"] = {"tokens": _tokens("chat_c")}
    gen["chat_a"]["tokens"] = [999]
    result = _compare(fixture=fixture, generation=gen, manifest=manifest)
    assert result["passed"], result["problems"]
    assert result["per_prompt"]["chat_a"]["match"] is False
    assert result["mismatched_prompt_ids"] == ["chat_a"]
    assert len(result["matched_prompt_ids"]) == 5


def test_declaring_a_prompt_non_gating_without_a_reason_fails():
    fixture = _fixture()
    fixture["required_prompt_ids"] = [p for p, _ in _SPECS if p != "chat_a"]
    fixture["non_gating_prompt_ids"] = ["chat_a"]
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert any("without recording why" in p for p in result["problems"])


def test_a_prompt_cannot_be_both_required_and_non_gating():
    fixture = _fixture()
    fixture["non_gating_prompt_ids"] = ["chat_a"]
    fixture["non_gating_reason"] = "recorded reason"
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert any("both required and non-gating" in p for p in result["problems"])


def test_a_gating_prompt_that_diverges_still_fails_the_gate():
    gen = _generation()
    gen["boundary_a"]["tokens"] = [999]
    result = _compare(generation=gen)
    assert not result["passed"]
    assert result["per_prompt"]["boundary_a"]["match"] is False
    assert any(p.startswith("boundary_a:") for p in result["problems"])


def test_divergence_evidence_records_both_sides():
    fixture = _fixture()
    fixture["prompts"]["code_a"]["top2_candidates"] = [[7, 8]] * len(_tokens("code_a"))
    gen = _generation()
    gen["code_a"]["tokens"] = list(_tokens("code_a"))
    gen["code_a"]["tokens"][2] = -1
    gen["code_a"]["top1_top2_margin"] = [0.5] * len(_tokens("code_a"))
    gen["code_a"]["top2_candidates"] = [[9, 10]] * len(_tokens("code_a"))
    result = _compare(fixture=fixture, generation=gen)
    evidence = result["per_prompt"]["code_a"]["divergence_evidence"]
    assert evidence["step"] == 2
    assert evidence["fixture_token"] == _tokens("code_a")[2]
    assert evidence["official_token"] == -1
    assert evidence["fixture_candidates"] == [7, 8]
    assert evidence["official_candidates"] == [9, 10]
    assert evidence["official_margin"] == 0.5


@pytest.mark.parametrize(
    "drop,category",
    [
        ("boundary_a", "cache_boundary"),
        ("code_a", "code"),
        ("reason_a", "reasoning"),
    ],
)
def test_missing_required_category_fails_even_with_five_prompts(drop, category):
    manifest = _manifest()
    fixture = _fixture()
    gen = _generation()
    # Replace the dropped prompt with another plain-chat one, keeping the count
    # at five so only the *category* rule can be doing the work.
    manifest["prompts"] = [p for p in manifest["prompts"] if p["id"] != drop]
    manifest["prompts"].append(
        {
            "id": "chat_c",
            "category": "plain_chat",
            "rendered_sha256": "sha-chat_c",
            "token_ids": [1],
            "num_tokens": 1,
            "thinking_mode": "chat",
        }
    )
    fixture["prompts"].pop(drop)
    fixture["prompts"]["chat_c"] = {
        "tokens": _tokens("chat_c"),
        "rendered_sha256": "sha-chat_c",
        "top1_top2_margin": _margins("chat_c"),
    }
    fixture["required_prompt_ids"] = sorted(fixture["prompts"])
    gen.pop(drop)
    gen["chat_c"] = {"tokens": _tokens("chat_c")}
    result = _compare(fixture=fixture, generation=gen, manifest=manifest)
    assert not result["passed"]
    assert any(f"no {category!r} prompt" in p for p in result["problems"])


# ---------------------------------------------------------------------------
# Provenance rules.
# ---------------------------------------------------------------------------


def test_wrong_checkpoint_revision_fails():
    fixture = _fixture()
    fixture["checkpoint_revision"] = "deadbeef"
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert any("checkpoint_revision" in p for p in result["problems"])


def test_fixture_from_a_different_prompt_manifest_fails():
    result = _compare(sha="1" * 64)
    assert not result["passed"]
    assert any("different prompts manifest" in p for p in result["problems"])


def test_fixture_prompt_text_must_match_the_pre_registered_rendering():
    fixture = _fixture()
    fixture["prompts"]["chat_a"]["rendered_sha256"] = "tampered"
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert "differs from the pre-registered rendering" in result["per_prompt"]["chat_a"]["problem"]


@pytest.mark.parametrize(
    "decoding",
    [
        {"do_sample": True, "num_beams": 1},
        {"do_sample": False, "num_beams": 4},
        {},
    ],
)
def test_non_greedy_decoding_fails(decoding):
    fixture = _fixture()
    fixture["decoding"] = decoding
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert any("not deterministic greedy" in p for p in result["problems"])


def test_fixture_not_produced_by_native_generate_fails():
    fixture = _fixture()
    fixture["provenance"]["generator"] = "official hand-written loop"
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert any("not a native greedy generate" in p for p in result["problems"])


@pytest.mark.parametrize("key", ["transformers_version", "conversion_code_sha256"])
def test_missing_provenance_field_fails(key):
    fixture = _fixture()
    fixture["provenance"].pop(key)
    result = _compare(fixture=fixture)
    assert not result["passed"]


def test_a_fixture_with_no_provenance_at_all_fails():
    fixture = _fixture()
    fixture["provenance"] = None
    result = _compare(fixture=fixture)
    assert not result["passed"]
    assert len(result["problems"]) >= 3


# ---------------------------------------------------------------------------
# Registered tolerances are applied literally. task.yaml forbids loosening a
# tolerance or waiving a failure, so there must be no path -- dtype resolution,
# per-module override or otherwise -- by which a metric above its registered
# limit is reported as a pass.
# ---------------------------------------------------------------------------


def _limits(cosine_min=0.999, rel_max=0.03):
    return {"cosine_min": cosine_min, "rel_max_abs_max": rel_max}


def test_rel_max_abs_above_the_registered_limit_always_fails():
    metrics = {"cosine": 0.9999999, "rel_max_abs": 0.0746, "finite": True}
    passed, problems = ev._judge(metrics, _limits())
    assert not passed
    assert "rel_max_abs" in problems[0]


def test_judge_takes_no_argument_that_could_waive_a_failure():
    """The third parameter supplies measurements; it cannot remove a verdict.

    ``storage_resolution`` exists so the re-registered BF16 sparse-attention
    and sink entries can be judged on their grid metrics. The property that
    matters is that it is strictly additive: passing it, omitting it, or
    passing an empty one must never turn a metric above its registered limit
    into a pass.
    """
    import inspect

    assert list(inspect.signature(ev._judge).parameters) == [
        "metrics",
        "limits",
        "storage_resolution",
    ]
    metrics = {"cosine": 0.9999999, "rel_max_abs": 0.0746, "finite": True}
    clean = {"abs_max_element_steps": 0.0, "elements_beyond_one_step": 0}
    for storage in (None, {}, clean):
        assert not ev._judge(metrics, _limits(), storage)[0]


def _storage_limits():
    """The re-registered BF16 storage-resolution entry, as it ships."""
    return {
        "cosine_min": 0.999,
        "abs_max_element_steps_max": 1.0,
        "elements_beyond_one_step_max": 0,
        "mean_abs_in_dtype_steps_max": 1e-4,
    }


def _storage(steps=0.85, beyond=0, mean=2.6e-05):
    return {
        "abs_max_element_steps": steps,
        "elements_beyond_one_step": beyond,
        "mean_abs_in_dtype_steps": mean,
    }


def test_a_registered_storage_limit_with_no_measurement_fails_rather_than_skips():
    """Forgetting to thread the measurement through must not read as a pass.

    This is the failure mode the re-registration is most exposed to: the
    limits live in the manifest but the numbers are computed elsewhere, so a
    missing hand-off would silently leave the entry ungated.
    """
    metrics = {"cosine": 1.0, "rel_max_abs": 0.04, "finite": True}
    passed, problems = ev._judge(metrics, _storage_limits())
    assert not passed
    assert len(problems) == 3
    assert all("was not measured" in p for p in problems)


def test_one_storage_step_at_the_deciding_element_passes_the_re_registered_entry():
    """Rank 4's measured case: one adjacent BF16 value, cosine 1.0."""
    metrics = {"cosine": 1.0, "rel_max_abs": 0.040346961, "finite": True}
    assert ev._judge(metrics, _storage_limits(), _storage())[0]


@pytest.mark.parametrize(
    "storage,expected",
    [
        (_storage(steps=1.7), "abs_max_element_steps"),
        (_storage(beyond=1), "elements_beyond_one_step"),
        (_storage(mean=2e-4), "mean_abs_in_dtype_steps"),
    ],
)
def test_each_storage_bound_rejects_on_its_own(storage, expected):
    metrics = {"cosine": 1.0, "rel_max_abs": 0.004, "finite": True}
    passed, problems = ev._judge(metrics, _storage_limits(), storage)
    assert not passed
    assert expected in problems[0]


def test_the_re_registered_entry_still_holds_the_original_cosine_floor():
    metrics = {"cosine": 0.99, "rel_max_abs": 0.004, "finite": True}
    passed, problems = ev._judge(metrics, _storage_limits(), _storage())
    assert not passed
    assert "cosine" in problems[0]


def test_rel_max_abs_exactly_at_the_registered_limit_passes():
    metrics = {"cosine": 1.0, "rel_max_abs": 0.03, "finite": True}
    assert ev._judge(metrics, _limits())[0]


def test_cosine_below_the_registered_floor_fails():
    metrics = {"cosine": 0.5, "rel_max_abs": 0.0, "finite": True}
    passed, problems = ev._judge(metrics, _limits())
    assert not passed
    assert "cosine" in problems[0]


def test_non_finite_values_fail():
    metrics = {"cosine": 1.0, "rel_max_abs": 0.0, "finite": False}
    passed, problems = ev._judge(metrics, _limits())
    assert not passed
    assert "non-finite" in problems[0]


def test_ulp_report_is_diagnostics_and_reports_adjacent_bfloat16_values():
    import torch

    # 4.78125 and 4.8125 are consecutive representable BF16 values (the step in
    # [4, 8) is 2^2 * 2^-7 = 0.03125).
    ref = torch.tensor([[4.78125, 0.5, -2.0]], dtype=torch.bfloat16)
    got = torch.tensor([[4.8125, 0.5, -2.0]], dtype=torch.bfloat16)
    report = ev._ulp_report(got, ref)
    assert 0.5 <= report["abs_max_element_steps"] <= 1.0
    assert report["elements_beyond_one_step"] == 0
    assert report["worst_absolute_element"]["ref_value"] == pytest.approx(4.78125)
    assert report["storage_dtype"] == "torch.bfloat16"


def test_ulp_report_flags_a_genuinely_large_difference():
    import torch

    ref = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.bfloat16)
    got = torch.tensor([[1.0, 2.0, 8.0]], dtype=torch.bfloat16)
    report = ev._ulp_report(got, ref)
    assert report["abs_max_element_steps"] == pytest.approx(64.0)
    assert report["elements_beyond_one_step"] == 1
    assert report["worst_absolute_element"]["index"] == [0, 2]


def test_a_cancellation_near_zero_is_measured_against_the_tensors_own_resolution():
    """The floor is what makes the step count a storage claim, not a relative one.

    An attention output is dominated by cancellation, so a few elements per
    million land four orders of magnitude below the tensor's scale. Their
    difference is absolutely negligible and relatively enormous, and the
    source's own kernel produces such elements against the independent golden
    just as TensorRT-LLM does. Measured against the tensor's own storage
    resolution they are inside one step; measured against their own magnitude
    they read tens of steps, which is reported but not gated.
    """
    import torch

    ref = torch.tensor([[4.0, -4.0, 4.0, -4.0, 4.2e-05]], dtype=torch.bfloat16)
    got = torch.tensor([[4.0, -4.0, 4.0, -4.0, 6.0e-05]], dtype=torch.bfloat16)
    report = ev._ulp_report(got, ref)

    # rms is ~3.58, so one storage step of the tensor is ~0.028 and the 1.8e-05
    # disagreement is a thousandth of it.
    assert report["one_storage_step_at_rms"] == pytest.approx(3.578 * 0.0078125, rel=0.01)
    assert report["elements_beyond_one_step"] == 0
    assert report["abs_max_element_steps"] < 0.01
    # ... and the unfloored view, kept so the choice is auditable, sees exactly
    # the ratio that made the pre-registration unreachable.
    assert report["unfloored_relative"]["elements_beyond_one_step"] == 1
    assert report["unfloored_relative"]["max_steps"] > 10.0
    assert report["unfloored_relative"]["worst_element"]["index"] == [0, 4]


def test_the_floor_does_not_touch_an_element_above_the_tensor_rms():
    """The deciding element of the real failure must keep its measured value.

    Rank 4 layer 3 disagrees by one BF16 value at 2.35 against an RMS of 0.387;
    the element is six times the RMS, so the local grid step governs and the
    floor is inert. If the floor moved this number the re-registration would be
    measuring something other than what was reported.
    """
    import torch

    ref = torch.full((64, 64), 0.387, dtype=torch.bfloat16)
    ref[3, 7] = 2.34375
    got = ref.clone()
    got[3, 7] = 2.359375
    report = ev._ulp_report(got, ref)

    # 0.015625 / (2.359375 * 2**-7) = 0.8477, the number the failing artifact
    # recorded, and it clears the registered bound of one step.
    assert report["abs_max_element_steps"] == pytest.approx(0.8477, abs=1e-3)
    assert report["worst_absolute_element"]["dtype_steps_apart"] == pytest.approx(0.8477, abs=1e-3)
    assert report["elements_beyond_one_step"] == 0


# ---------------------------------------------------------------------------
# Artifact status truthfulness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "passed,error,expected",
    [
        (True, None, "passed"),
        (False, None, "failed"),
        (None, None, "failed"),
        (True, "boom", "error"),
        (False, "boom", "error"),
    ],
)
def test_artifact_status_follows_the_measurement(passed, error, expected):
    assert ev.artifact_status(passed, error) == expected


# ---------------------------------------------------------------------------
# The real, checked-in manifests have to satisfy the same rules.
# ---------------------------------------------------------------------------


def test_real_prompt_manifest_covers_every_required_category():
    manifest = json.loads(_MANIFEST_PATH.read_text())
    categories = {p["category"] for p in manifest["prompts"]}
    for required in ev.REQUIRED_NATIVE_CATEGORIES:
        assert required in categories, f"prompts.json has no {required!r} prompt"
    assert len(manifest["prompts"]) >= ev.MIN_NATIVE_PROMPTS
    assert manifest["checkpoint_revision"] == REVISION


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"deepseek-v4-flash" * 4096
    path.write_bytes(payload)
    assert ev._sha256_file(str(path)) == hashlib.sha256(payload).hexdigest()


def test_comparison_does_not_mutate_its_inputs():
    fixture, gen, manifest = _fixture(), _generation(), _manifest()
    before = (deepcopy(fixture), deepcopy(gen), deepcopy(manifest))
    ev.compare_native_golden(fixture, gen, manifest, MANIFEST_SHA)
    assert (fixture, gen, manifest) == before


# ---------------------------------------------------------------------------
# Strict artifact audit. The driver judges each golden as it measures it, but
# a reader should not have to take that on trust -- the audit re-derives every
# verdict from the manifest, so a written artifact cannot claim a pass the
# registered numbers do not support.
# ---------------------------------------------------------------------------

_TOLERANCES = {
    "modules": {
        "sparse_attention_output": {"cosine_min": 0.999, "rel_max_abs_max": 0.03},
        "rope": {"cosine_min": 0.9999, "rel_max_abs_max": 0.01},
    }
}


def _artifact(**overrides):
    artifact = {
        "suite": "reference_ladder",
        "passed": True,
        "status": "passed",
        "error": None,
        "module_goldens": {
            "layer0.sparse_attention": {
                "module": "sparse_attention_output",
                "metrics": {"cosine": 1.0, "rel_max_abs": 0.018, "finite": True},
                "passed": True,
            },
            "layer0.rope_freqs": {
                "module": "rope",
                "metrics": {"cosine": 1.0, "rel_max_abs": 0.0, "finite": True},
                "passed": True,
            },
        },
        "native_generate_golden": {"mismatched_prompt_ids": [], "per_prompt": {}},
    }
    artifact.update(overrides)
    return artifact


def test_audit_of_a_clean_artifact_is_clean():
    report = ev.audit_artifact(_artifact(), _TOLERANCES)
    assert report["clean"], report
    assert report["strict_failures"] == []


def test_audit_catches_a_metric_above_its_registered_limit():
    artifact = _artifact()
    artifact["module_goldens"]["layer0.sparse_attention"]["metrics"]["rel_max_abs"] = 0.0746
    report = ev.audit_artifact(artifact, _TOLERANCES)
    assert not report["clean"]
    assert report["strict_failures"][0]["check"] == "layer0.sparse_attention"
    # The artifact still claimed it passed, so the audit calls that out too.
    assert any("passed=True" in d for d in report["verdict_disagreements"])
    assert any("claims passed=true" in d for d in report["verdict_disagreements"])


def test_audit_reports_per_rank_failures_the_rank_zero_log_would_hide():
    artifact = _artifact(
        per_rank_failures={
            "4": {
                "layer0.sparse_attention": {
                    "problems": ["rel_max_abs 5.029e-02 > 0.03"],
                    "metrics": {"rel_max_abs": 0.05029},
                }
            }
        },
        ranks_failed=[4],
    )
    report = ev.audit_artifact(artifact, _TOLERANCES)
    assert not report["clean"]
    assert report["ranks_failed"] == [4]
    assert report["strict_failures"][0]["check"] == "rank4.layer0.sparse_attention"


def test_audit_treats_a_gating_prompt_mismatch_as_a_strict_failure():
    artifact = _artifact(passed=False, status="failed")
    artifact["native_generate_golden"] = {
        "mismatched_prompt_ids": ["boundary_a"],
        "per_prompt": {"boundary_a": {"gating": True, "problem": "token sequences differ"}},
    }
    report = ev.audit_artifact(artifact, _TOLERANCES)
    assert not report["clean"]
    assert report["strict_failures"][0]["check"] == "native_generate_golden.boundary_a"


def test_audit_lists_a_declared_non_gating_divergence_without_failing_on_it():
    artifact = _artifact()
    artifact["native_generate_golden"] = {
        "mismatched_prompt_ids": ["chat_a"],
        "per_prompt": {"chat_a": {"gating": False, "problem": "length mismatch"}},
    }
    report = ev.audit_artifact(artifact, _TOLERANCES)
    assert report["clean"], report
    assert report["non_gating_divergences"] == ["chat_a"]


def test_audit_catches_a_status_that_does_not_follow_the_measurement():
    report = ev.audit_artifact(_artifact(passed=False), _TOLERANCES)
    assert not report["clean"]
    assert any("does not follow passed" in d for d in report["verdict_disagreements"])


# ---------------------------------------------------------------------------
# Cross-rank worst case. The module goldens run against rank-local weights, so
# rank 0's numbers are one eighth of the evidence; "no rank failed" is a much
# weaker claim than "the worst rank still had this much headroom".
# ---------------------------------------------------------------------------


def _by_rank(*per_rank):
    return {
        rank: {
            "rank": rank,
            "passed": True,
            "failed_checks": {},
            "metrics": {"layer0.sparse_attention": m},
            "steps": {"layer0.sparse_attention": s},
        }
        for rank, (m, s) in enumerate(per_rank)
    }


_GOLDEN_SHAPE = {
    "layer0.sparse_attention": {
        "module": "sparse_attention_output",
        "tolerance": _TOLERANCES["modules"]["sparse_attention_output"],
    }
}


def test_worst_rank_metrics_take_the_max_of_an_error_and_the_min_of_an_agreement():
    worst = ev.worst_rank_metrics(
        _by_rank(
            ({"cosine": 1.0, "rel_max_abs": 0.004}, 0.54),
            ({"cosine": 0.9995, "rel_max_abs": 0.026}, 0.71),
            ({"cosine": 0.99999, "rel_max_abs": 0.011}, 0.60),
        ),
        _GOLDEN_SHAPE,
    )["layer0.sparse_attention"]

    assert worst["rel_max_abs"] == {
        "value": 0.026,
        "rank": 1,
        "limit": 0.03,
        "headroom_x": pytest.approx(1.1538, rel=1e-3),
    }
    # Agreement metrics measure headroom against their distance from 1.0:
    # (1 - 0.999) / (1 - 0.9995) = 2x.
    assert worst["cosine"]["value"] == 0.9995
    assert worst["cosine"]["rank"] == 1
    assert worst["cosine"]["headroom_x"] == pytest.approx(2.0, rel=1e-3)
    assert worst["abs_max_element_steps"] == {"value": 0.71, "rank": 1}


def test_a_perfect_metric_reports_no_headroom_ratio_rather_than_an_infinity():
    """Zero error is unbounded headroom, not zero headroom.

    Recording ``0.0`` there would sort every exactly-correct check above the
    ones that are genuinely close to their limit, burying the only rows worth
    reading. ``Infinity`` is not valid JSON, so the field is simply omitted.
    """
    worst = ev.worst_rank_metrics(
        _by_rank(({"cosine": 1.0, "rel_max_abs": 0.0}, 0.0)), _GOLDEN_SHAPE
    )["layer0.sparse_attention"]
    assert worst["rel_max_abs"] == {"value": 0.0, "rank": 0, "limit": 0.03}
    assert "headroom_x" not in worst["rel_max_abs"]
    assert "headroom_x" not in worst["cosine"]


def test_audit_fails_when_only_a_non_zero_rank_exceeds_the_limit():
    """The exact false pass this aggregation exists to prevent."""
    artifact = _artifact(
        worst_rank_metrics={
            "layer0.sparse_attention": {
                "cosine": {"value": 1.0, "rank": 0},
                "rel_max_abs": {"value": 0.0503, "rank": 4, "limit": 0.03, "headroom_x": 0.596},
            }
        }
    )
    # Rank 0's own entry is comfortably inside the limit and unchanged.
    assert artifact["module_goldens"]["layer0.sparse_attention"]["metrics"]["rel_max_abs"] == 0.018

    report = ev.audit_artifact(artifact, _TOLERANCES)
    assert not report["clean"]
    failure = next(f for f in report["strict_failures"] if f["check"].startswith("worst_rank."))
    assert failure["check"] == "worst_rank.layer0.sparse_attention"
    assert any("rank" in str(p) for p in failure["problems"])
    assert any("claims passed=true" in d for d in report["verdict_disagreements"])


def test_audit_reports_the_tightest_margin_first():
    artifact = _artifact(
        worst_rank_metrics={
            "layer0.sparse_attention": {
                "cosine": {"value": 1.0, "rank": 0},
                "rel_max_abs": {"value": 0.024, "rank": 5, "limit": 0.03, "headroom_x": 1.25},
            },
            "layer0.rope_freqs": {
                "cosine": {"value": 1.0, "rank": 0},
                "rel_max_abs": {"value": 0.0, "rank": 0, "limit": 0.01, "headroom_x": 9999.0},
            },
        }
    )
    report = ev.audit_artifact(artifact, _TOLERANCES)
    assert report["clean"], report
    check, metric, detail = report["tightest_margins"][0]
    assert (check, metric, detail["rank"]) == ("layer0.sparse_attention", "rel_max_abs", 5)


# ---------------------------------------------------------------------------
# The shipped re-registration. The point of a pre-registered manifest is that
# the change to it is auditable, so these pin both halves of the claim: the two
# BF16 output entries carry exactly the new conjunction, and nothing else in
# the manifest moved.
# ---------------------------------------------------------------------------

# The superseded manifest, hash 7641c45b..., limit for limit. Anything that
# drifts from this table is a tolerance change beyond the re-registration.
_SUPERSEDED_LIMITS = {
    "q_projection_and_norm": {"cosine_min": 0.9995, "rel_max_abs_max": 0.02},
    "kv_latent_and_norm": {"cosine_min": 0.9995, "rel_max_abs_max": 0.02},
    "rope": {"cosine_min": 0.9999, "rel_max_abs_max": 0.01},
    "inverse_rope": {"cosine_min": 0.9999, "rel_max_abs_max": 0.01},
    "compressor": {"cosine_min": 0.999, "rel_max_abs_max": 0.03},
    "compressor_state": {"cosine_min": 0.999, "rel_max_abs_max": 0.03},
    "indexer_scores": {"cosine_min": 0.995, "rel_max_abs_max": 0.05},
    "o_lora_output": {"cosine_min": 0.999, "rel_max_abs_max": 0.03},
    "moe_router_logits": {"cosine_min": 0.999, "rel_max_abs_max": 0.02},
    "moe_routing_weights": {"cosine_min": 0.999, "rel_max_abs_max": 0.02},
    "moe_expert_output": {"cosine_min": 0.999, "rel_max_abs_max": 0.04},
    "moe_combined_output": {"cosine_min": 0.999, "rel_max_abs_max": 0.04},
    "mhc": {"cosine_min": 0.999, "rel_max_abs_max": 0.03},
    "decoder_layer": {"cosine_min": 0.99, "rel_max_abs_max": 0.05},
    "final_logits": {"cosine_min": 0.99, "rel_max_abs_max": 0.05},
}
_RE_REGISTERED = ("sparse_attention_output", "attention_sink")


def _real_tolerances():
    return json.loads(_TOLERANCES_PATH.read_text())


def test_the_two_bf16_output_entries_carry_exactly_the_registered_conjunction():
    modules = _real_tolerances()["modules"]
    for name in _RE_REGISTERED:
        entry = modules[name]
        assert entry["cosine_min"] == 0.999
        assert entry["abs_max_element_steps_max"] == 1.0
        assert entry["elements_beyond_one_step_max"] == 0
        assert entry["mean_abs_in_dtype_steps_max"] == 1e-4
        # The metric the re-registration supersedes must be gone, not kept
        # alongside: leaving it would re-impose the unreachable rule.
        assert "rel_max_abs_max" not in entry


def test_no_other_module_limit_moved_in_the_re_registration():
    modules = _real_tolerances()["modules"]
    for name, expected in _SUPERSEDED_LIMITS.items():
        actual = {k: v for k, v in modules[name].items() if k.endswith(("_min", "_max"))}
        assert actual == expected, f"{name} limits changed outside the re-registration"


def test_the_exact_rules_are_untouched():
    modules = _real_tolerances()["modules"]
    for name in ("indexer_topk", "moe_expert_ids"):
        assert modules[name]["rule"] == "exact"
        assert not any(k.endswith(("_min", "_max")) for k in modules[name])
    assert _real_tolerances()["modules"]["final_logits"]["argmax_rule"] == "exact"


def test_the_storage_rule_is_scoped_to_the_two_bf16_outputs():
    modules = _real_tolerances()["modules"]
    carrying = {n for n, e in modules.items() if "abs_max_element_steps_max" in e}
    assert carrying == set(_RE_REGISTERED)


def test_the_re_registration_names_the_manifest_it_supersedes():
    """A re-registration has to say what it replaced, or it is just an edit."""
    import hashlib

    manifest = _real_tolerances()
    reg = manifest["re_registration"]
    assert manifest["schema_version"] == 2
    assert reg["modules"] == list(_RE_REGISTERED)
    assert len(reg["supersedes_sha256"]) == 64
    assert reg["supersedes_sha256"] != hashlib.sha256(_TOLERANCES_PATH.read_bytes()).hexdigest()
    assert reg["registered_at"].endswith("Z")


def test_the_checksum_file_registers_the_manifest_that_is_on_disk():
    """`_manifest_provenance` refuses to run otherwise; prove it agrees here too."""
    import hashlib

    registered = {
        name: digest
        for digest, name in (
            line.split() for line in (_MANIFEST_DIR / "MANIFEST.sha256").read_text().splitlines()
        )
    }
    on_disk = sorted(p.name for p in _MANIFEST_DIR.glob("*.json"))
    assert sorted(registered) == on_disk, (
        "every manifest in the directory must be registered; an unregistered one "
        "could be edited without a hash moving"
    )
    for name in on_disk:
        actual = hashlib.sha256((_MANIFEST_DIR / name).read_bytes()).hexdigest()
        assert registered[name] == actual, f"{name} drifted from MANIFEST.sha256"


def test_the_native_generate_fixture_is_registered():
    """It decides which prompts gate, so it belongs under the same registration.

    ``native_generate_golden.json`` carries ``non_gating_prompt_ids``. Left
    unregistered, a prompt could be moved out of the gating set -- with a
    written reason, but with no hash change -- and the reference ladder would
    still report the same manifest provenance.
    """
    assert "native_generate_golden.json" in ev.MANIFEST_FILES
    fixture = json.loads((_MANIFEST_DIR / "native_generate_golden.json").read_text())
    assert fixture["prompts"], "an empty fixture would gate nothing"


def test_the_regression_baseline_is_registered():
    """It decides which regression failures count as this container's fault.

    ``regression_baseline.json`` names the failures Stage 3 attributes to the
    dev container's missing ``CAP_SYS_PTRACE``. Left unregistered, a genuine
    regression could be appended to that list -- with a written reason, but
    with no hash change -- and the regression report would still call the run
    clean.
    """
    assert "regression_baseline.json" in ev.MANIFEST_FILES
    baseline = json.loads((_MANIFEST_DIR / "regression_baseline.json").read_text())
    assert baseline["environment_failure_signatures"], "no signature gates nothing"
    for signature in baseline["environment_failure_signatures"]:
        assert signature["all_of"], f"{signature['id']} would match every failure"


def test_the_superseded_manifest_is_retained_and_differs_only_where_declared():
    """ "No other limit moved" should be a diff, not a claim.

    The bytes of the file Stage 2 replaced were not retained, so this checks the
    reconstruction that *is* retained: its limits must equal the superseded
    table pinned above, and diffing it against the active manifest must produce
    exactly the two entries the re-registration named.
    """
    superseded = json.loads((_MANIFEST_DIR / "tolerances.superseded.json").read_text())
    active = _real_tolerances()

    assert superseded["schema_version"] == 1
    assert superseded["status"] == "superseded"
    assert superseded["original_file_sha256"] == active["re_registration"]["supersedes_sha256"]
    # It is a reconstruction and has to say so: silently presenting it as the
    # original bytes would be a stronger claim than the evidence supports.
    assert superseded["retention"]["not_the_original_bytes"]
    assert superseded["retention"]["extracted_from"], "no provenance for the recovered limits"

    def limits(entry):
        return {k: v for k, v in entry.items() if k.endswith(("_min", "_max"))}

    for name, expected in _SUPERSEDED_LIMITS.items():
        assert limits(superseded["modules"][name]) == expected, name

    moved = {
        name
        for name, entry in active["modules"].items()
        if limits(entry) != limits(superseded["modules"][name])
    }
    assert moved == set(_RE_REGISTERED)


def test_the_recovered_superseded_limits_come_from_pre_registration_evidence():
    """The two re-registered entries are the ones that must not be re-authored.

    Both are recovered from artifacts written before the registration, which is
    what makes the diff above evidence rather than a restatement of the story
    the active manifest tells about itself.
    """
    superseded = json.loads((_MANIFEST_DIR / "tolerances.superseded.json").read_text())
    sources = superseded["retention"]["module_sources"]
    for name in _RE_REGISTERED:
        assert sources[name] == "pre_registration_artifact", name
        entry = superseded["modules"][name]
        assert entry["rel_max_abs_max"] == 0.03
        assert "abs_max_element_steps_max" not in entry
    assert set(sources) == set(superseded["modules"])


def test_provenance_refuses_a_manifest_registered_after_the_run_started():
    """The ordering claim is the whole value of a pre-registered limit."""
    registered_at = _real_tolerances()["re_registration"]["registered_at"]
    with pytest.raises(RuntimeError, match="not a pre-registered limit"):
        ev._manifest_provenance("2020-01-01T00:00:00Z")
    block = ev._manifest_provenance("2099-01-01T00:00:00Z")
    assert block["re_registration"]["registered_at"] == registered_at
    assert block["tolerances_sha256_old"] != block["tolerances_sha256_new"]
    assert len(block["tolerances_sha256_new"]) == 64


def test_the_worst_rank_storage_metrics_are_folded_in_and_audited():
    """A storage limit checked only on rank 0 would not be a gate.

    The re-registered entries are judged on numbers the ranks disagree about,
    so the worst rank has to reach the audit. Here rank 2 is the only one with
    an element beyond a storage step, and that alone must fail the artifact.
    """
    limits = {
        "cosine_min": 0.999,
        "abs_max_element_steps_max": 1.0,
        "elements_beyond_one_step_max": 0,
        "mean_abs_in_dtype_steps_max": 1e-4,
    }
    by_rank = {
        rank: {
            "rank": rank,
            "passed": True,
            "failed_checks": {},
            "metrics": {"layer3.sparse_attention": {"cosine": 1.0, "rel_max_abs": 0.04}},
            "steps": {"layer3.sparse_attention": steps},
            "grid_agreement": {
                "layer3.sparse_attention": {
                    "elements": 1052672,
                    "elements_beyond_one_step": beyond,
                    "mean_abs_in_dtype_steps": 2.6e-05,
                }
            },
        }
        for rank, (steps, beyond) in enumerate([(0.53, 0), (0.85, 0), (0.92, 1)])
    }
    shape = {"layer3.sparse_attention": {"module": "sparse_attention_output", "tolerance": limits}}
    worst = ev.worst_rank_metrics(by_rank, shape)["layer3.sparse_attention"]

    assert worst["abs_max_element_steps"] == {
        "value": 0.92,
        "rank": 2,
        "limit": 1.0,
        "headroom_x": pytest.approx(1.087, rel=1e-2),
    }
    # A limit of zero has no budget, so any violation is zero headroom rather
    # than a division by the limit.
    assert worst["elements_beyond_one_step"]["rank"] == 2
    assert worst["elements_beyond_one_step"]["headroom_x"] == 0.0

    report = ev.audit_artifact(
        _artifact(
            module_goldens={
                "layer3.sparse_attention": {
                    "module": "sparse_attention_output",
                    "metrics": {"cosine": 1.0, "rel_max_abs": 0.04, "finite": True},
                    "storage_resolution": {
                        "abs_max_element_steps": 0.53,
                        "elements_beyond_one_step": 0,
                        "mean_abs_in_dtype_steps": 2.6e-05,
                    },
                    "passed": True,
                }
            },
            worst_rank_metrics={"layer3.sparse_attention": worst},
        ),
        {"modules": {"sparse_attention_output": limits}},
    )
    assert not report["clean"]
    failure = next(f for f in report["strict_failures"] if f["check"].startswith("worst_rank."))
    assert "elements_beyond_one_step" in failure["problems"][0]


def test_worst_rank_metrics_ignores_checks_with_no_registered_float_limit():
    """An ``exact`` rule has no ``_max``/``_min`` entry, so no headroom exists."""
    by_rank = {
        0: {
            "rank": 0,
            "passed": True,
            "failed_checks": {},
            "metrics": {"layer2.indexer_topk": {"exact_set": 1.0}},
            "steps": {"layer2.indexer_topk": None},
        }
    }
    shape = {"layer2.indexer_topk": {"module": "indexer_topk", "tolerance": {"rule": "exact"}}}
    entry = ev.worst_rank_metrics(by_rank, shape)["layer2.indexer_topk"]
    assert entry["exact_set"] == {"value": 1.0, "rank": 0}
    assert "abs_max_element_steps" not in entry
