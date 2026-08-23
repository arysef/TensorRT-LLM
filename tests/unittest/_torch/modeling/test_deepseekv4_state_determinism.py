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
"""Rules behind the DeepSeek-V4-Flash eager-state determinism evidence.

Both halves of that evidence cost eight H100s and tens of minutes --- three
``LLM`` constructions for the gating half, a full TP8 checkpoint load for the
diagnostic one. What decides whether either says anything is a handful of
ordinary functions, and this pins them at CPU speed:

* **the reading of the five-mode table.** Running a control, a
  metadata-sharing mode, a cache-sharing mode, both, and both-with-freed-pages-
  memset is only worth the wall clock if the *pattern* is turned into a
  statement. A reading that named the wrong channel would send the next
  iteration to the wrong file.
* **what counts as a repeat.** The anchor is a prompt's first occurrence, not
  the previous one; chained comparisons hide a drift that is constant per step.
* **what the gating half must contain before it may pass.** Three same-engine
  and two fresh-engine executions of each named prompt, three distinct
  engines, and a teardown that reports no held block. A suite that passed with
  four executions, or with two engines, would be reporting a weaker fact under
  the acceptance item's name.
* **which interpreter and which communicator each suite runs under.** The
  diagnostic drives production modules at TP8 under ``torchrun``, so it needs
  the pinned interpreter and the torch.distributed communicator; the gating
  half spawns its own MPI world and must not be in either set.
"""

import importlib.util
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

import state_determinism as sd  # noqa: E402
import torch_goldens as tg  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dsv4_evidence_state", os.path.join(_ACCURACY, "deepseek_v4_flash_h100_evidence.py")
)
evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence)


# ---------------------------------------------------------------------------
# Registration: which process each half runs in.
# ---------------------------------------------------------------------------


def test_both_halves_are_registered_suites():
    assert "state_lifecycle" in evidence.SUITES
    assert "eager_state_determinism" in evidence.SUITES


def test_the_diagnostic_runs_in_the_pinned_interpreter_under_torch_distributed():
    # It executes production modules, so the reference venv --- whose tvm_ffi
    # makes flashinfer's CuTe RMSNorm raise --- must not be in the path; and it
    # is launched by torchrun as one process per rank, which is the
    # communicator TLLM_DISABLE_MPI selects.
    assert "state_lifecycle" not in evidence.NEEDS_REFERENCE_ENV
    assert "state_lifecycle" in evidence.NEEDS_TORCH_DISTRIBUTED


def test_the_gating_half_spawns_its_own_world():
    # It builds three LLMs, each of which spawns eight MPI workers; forcing the
    # torch.distributed communicator on the launcher would leave it unable to.
    assert "eager_state_determinism" not in evidence.NEEDS_REFERENCE_ENV
    assert "eager_state_determinism" not in evidence.NEEDS_TORCH_DISTRIBUTED


# ---------------------------------------------------------------------------
# The scripted sequences.
# ---------------------------------------------------------------------------


def test_the_diagnostic_sequence_repeats_a_prompt_across_different_histories():
    seq = sd.DEFAULT_SEQUENCE
    positions = [i for i, pid in enumerate(seq) if pid == "chat_geography"]
    assert len(positions) >= 4, "the prompt under test must recur"
    # At least one recurrence immediately after itself (so nondeterminism is
    # separable) and at least one after a different prompt (so history is).
    assert any(seq[i - 1] == "chat_geography" for i in positions[1:])
    assert any(seq[i - 1] != "chat_geography" for i in positions[1:])


def test_every_mode_differs_from_the_control_in_exactly_one_way():
    modes = {m["name"]: m for m in sd.MODES}
    assert modes["control"] == {
        "name": "control",
        "share_cache": False,
        "share_metadata": False,
        "zero_freed": False,
    }
    assert modes["metadata_only"]["share_metadata"] and not modes["metadata_only"]["share_cache"]
    assert modes["cache_only"]["share_cache"] and not modes["cache_only"]["share_metadata"]
    assert modes["executor_like"]["share_cache"] and modes["executor_like"]["share_metadata"]
    # The probe is executor_like plus the memset, nothing else.
    probe = dict(modes["executor_like_zero_freed"])
    assert probe.pop("zero_freed") is True
    assert probe | {"zero_freed": False} == modes["executor_like"] | {"name": probe["name"]}


def test_the_primer_separates_a_back_to_back_repeat_from_one_after_another_shape():
    """The experiment the primer *is*: three executions of one prompt with
    nothing in between, then one more after a different prompt. Without both
    groups the reading cannot distinguish "irreproducible after the first
    request" from "perturbed by the intervening shape"."""
    primer = sd._PRIMER
    first = primer[0]
    cut = sd._PRIMER_AFTER_OTHER_SHAPE
    assert primer[:cut].count(first) >= 2, "no back-to-back repeat to compare"
    assert primer[cut - 1] != first, "the execution at the cut must follow a different prompt"
    assert first in primer[cut:], "no execution after the other shape"


def _diag(index, same_engine=True, identical=True, prompt=None, step0=0.0, engine="same_engine"):
    return {
        "prompt_id": prompt or sd._PRIMER[0],
        "anchor": "same_engine#0",
        "execution": f"{engine}#{index}",
        "same_engine": same_engine,
        "tokens_identical": identical,
        "logits_identical": identical,
        "step0_max_abs": step0,
    }


def test_the_reading_names_the_intervening_shape_when_only_the_later_repeat_moves():
    cut = sd._PRIMER_AFTER_OTHER_SHAPE
    diagnostics = [_diag(1), _diag(2), _diag(cut, identical=False)]
    assert "intervening shape" in sd.position_reading(diagnostics)


def test_the_reading_exonerates_the_intervening_prompt_when_the_repeat_already_moves():
    cut = sd._PRIMER_AFTER_OTHER_SHAPE
    diagnostics = [_diag(1, identical=False), _diag(2), _diag(cut, identical=False)]
    assert "not the intervening prompt" in sd.position_reading(diagnostics)


def test_a_per_position_drift_that_repeats_in_every_engine_is_called_settled():
    # A counter and a race look the same in one engine and send the next
    # iteration to different code, so the reading has to separate them.
    cut = sd._PRIMER_AFTER_OTHER_SHAPE
    diagnostics = [
        _diag(1, identical=False, step0=4.58),
        _diag(1, same_engine=False, identical=False, step0=4.58, engine="fresh_engine_a"),
        _diag(2),
        _diag(cut, identical=False, step0=4.0),
    ]
    assert "same value in every engine" in sd.position_reading(diagnostics)


def test_a_per_position_drift_that_differs_between_engines_is_not_called_settled():
    cut = sd._PRIMER_AFTER_OTHER_SHAPE
    diagnostics = [
        _diag(1, identical=False, step0=4.58),
        _diag(1, same_engine=False, identical=False, step0=3.11, engine="fresh_engine_a"),
        _diag(2),
        _diag(cut, identical=False, step0=4.0),
    ]
    assert "not reproducible either" in sd.position_reading(diagnostics)


def test_the_reading_reports_a_fully_reproducible_executor_as_such():
    cut = sd._PRIMER_AFTER_OTHER_SHAPE
    assert "regardless of what ran before it" in sd.position_reading(
        [_diag(1), _diag(2), _diag(cut)]
    )


def test_the_reading_refuses_to_conclude_from_half_the_experiment():
    cut = sd._PRIMER_AFTER_OTHER_SHAPE
    assert "not measured" in sd.position_reading([_diag(1), _diag(2)])
    assert "not measured" in sd.position_reading([_diag(cut)])
    assert "not measured" in sd.position_reading([])


def test_the_gating_sequences_share_a_prefix_so_the_first_execution_is_comparable():
    same, fresh = sd.SAME_ENGINE_SEQUENCE, sd.FRESH_ENGINE_SEQUENCE
    prefix = len(fresh) - len(sd.GATING_PROMPTS)
    assert same[: prefix + len(sd.GATING_PROMPTS)] == fresh
    # Neither gating prompt may be an engine's first request: an engine whose
    # first request is the one under test proves determinism only for the case
    # that cannot go wrong.
    assert same[0] not in sd.GATING_PROMPTS
    assert fresh[0] not in sd.GATING_PROMPTS


def test_the_same_engine_sequence_runs_each_gating_prompt_three_times():
    for pid in sd.GATING_PROMPTS:
        assert sd.SAME_ENGINE_SEQUENCE.count(pid) == 3
        assert sd.FRESH_ENGINE_SEQUENCE.count(pid) == 1


# ---------------------------------------------------------------------------
# The comparison.
# ---------------------------------------------------------------------------


def _run(index, pid, logits, layers=None):
    return {
        "index": index,
        "prompt_id": pid,
        "request_id": index,
        "logits": torch.tensor(logits, dtype=torch.float32),
        "logits_sha": sd._sha(torch.tensor(logits, dtype=torch.float32)),
        "layers": {k: torch.tensor([v], dtype=torch.float32) for k, v in (layers or {}).items()},
        "pages": {},
        "request_still_mapped_after_free": False,
        "index_mapper_free_slots_before": 4,
        "index_mapper_free_slots_after": 4,
        "freed_pages_survived": [],
    }


def test_a_reproducible_sequence_reports_nothing():
    runs = [_run(i, "a", [1.0, 2.0, 3.0]) for i in range(3)]
    result = sd.compare_sequence(runs, tg)
    assert result["history_dependent"] == []
    assert all(c["logits_identical"] for c in result["comparisons"])


def test_every_repeat_is_compared_against_the_first_not_the_previous():
    # Drift of one unit per run. Against the previous run each step is 1.0;
    # against the first, run 2 is 2.0. Chained comparison would understate it.
    runs = [_run(i, "a", [1.0 + i, 2.0, 3.0]) for i in range(3)]
    result = sd.compare_sequence(runs, tg)
    assert [c["anchor_index"] for c in result["comparisons"]] == [0, 0]
    assert result["comparisons"][-1]["logits_max_abs"] == pytest.approx(2.0)
    assert result["history_dependent"] == ["a"]


def test_prompts_are_anchored_independently():
    runs = [
        _run(0, "a", [1.0, 0.0]),
        _run(1, "b", [5.0, 0.0]),
        _run(2, "a", [1.0, 0.0]),
        _run(3, "b", [9.0, 0.0]),
    ]
    result = sd.compare_sequence(runs, tg)
    assert result["anchors"] == {"a": 0, "b": 1}
    assert result["history_dependent"] == ["b"]


def _decode(shas, argmaxes=None):
    argmaxes = argmaxes if argmaxes is not None else list(range(len(shas)))
    return [
        {"step": i, "logits_sha": s, "argmax": a} for i, (s, a) in enumerate(zip(shas, argmaxes))
    ]


def test_a_decode_trajectory_that_matches_reports_no_differing_step():
    steps = _decode(["a", "b", "c"])
    result = sd.compare_decode(steps, _decode(["a", "b", "c"]))
    assert result["decode_first_differing_step"] is None
    assert result["decode_tokens_identical"]
    assert result["decode_steps_compared"] == 3


def test_the_decode_comparison_names_the_step_where_the_logits_first_move():
    # The distinction that matters for `long_prefill_2304`: prefill is
    # bit-identical and the logits part company at a decode step, which is a
    # different statement from "the two runs differ".
    result = sd.compare_decode(_decode(["a", "X", "c"]), _decode(["a", "b", "c"]))
    assert result["decode_first_differing_step"] == 1


def test_a_logit_difference_that_has_not_yet_changed_a_token_is_reported_separately():
    # A greedy decode forks permanently once its argmax moves, so the two
    # indices answer different questions and are recorded as two fields.
    result = sd.compare_decode(
        _decode(["a", "X", "Y"], [7, 7, 9]), _decode(["a", "b", "c"], [7, 7, 8])
    )
    assert result["decode_first_differing_step"] == 1
    assert result["decode_first_differing_token_step"] == 2
    assert not result["decode_tokens_identical"]


def test_no_decode_steps_is_not_a_difference():
    result = sd.compare_decode([], [])
    assert result["decode_steps_compared"] == 0
    assert result["decode_first_differing_step"] is None
    assert result["decode_tokens_identical"]


def test_a_decode_only_difference_still_makes_the_prompt_history_dependent():
    # Prefill bit-identical, decode not: the sequence comparison must not call
    # that reproducible just because the prefill logits hash the same.
    anchor = _run(0, "a", [1.0, 0.0])
    later = _run(1, "a", [1.0, 0.0])
    anchor["decode"] = _decode(["a", "b"])
    later["decode"] = _decode(["a", "X"])
    result = sd.compare_sequence([anchor, later], tg)
    assert result["comparisons"][0]["logits_identical"]
    assert result["comparisons"][0]["decode_first_differing_step"] == 1
    assert result["history_dependent"] == ["a"]


def test_the_first_differing_layer_is_the_first_one_that_actually_differs():
    anchor = _run(0, "a", [1.0, 0.0], layers={0: 1.0, 1: 1.0, 2: 1.0})
    later = _run(1, "a", [1.0, 0.0], layers={0: 1.0, 1: 1.5, 2: 9.0})
    result = sd.compare_sequence([anchor, later], tg)
    assert result["comparisons"][0]["first_differing_layer"] == 1
    assert result["comparisons"][0]["layer_max_abs"]["2"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# The reading of the mode table.
# ---------------------------------------------------------------------------


def _table(**drifting):
    return {
        name: {"comparison": {"history_dependent": ["chat_geography"] if drifting.get(name) else []}}
        for name in (
            "control",
            "metadata_only",
            "cache_only",
            "executor_like",
            "executor_like_zero_freed",
        )
    }


def test_an_irreproducible_control_refuses_to_attribute_anything():
    reading = sd.reading(_table(control=True, executor_like=True, cache_only=True))
    assert "not attributable" in reading


def test_a_cache_only_drift_names_the_cache_manager():
    reading = sd.reading(_table(executor_like=True, cache_only=True))
    assert "shared cache manager" in reading
    assert "metadata" not in reading


def test_a_metadata_only_drift_names_the_metadata():
    reading = sd.reading(_table(executor_like=True, metadata_only=True))
    assert "reused attention metadata" in reading
    assert "cache manager" not in reading


def test_a_drift_that_only_appears_with_both_says_so():
    reading = sd.reading(_table(executor_like=True))
    assert "neither alone" in reading


def test_the_memset_probe_decides_whether_page_content_is_the_carrier():
    content = sd.reading(_table(executor_like=True, cache_only=True))
    assert "recycled pages" in content
    survives = sd.reading(
        _table(executor_like=True, cache_only=True, executor_like_zero_freed=True)
    )
    assert "survives memsetting" in survives


def test_no_drift_at_all_is_reported_as_such():
    assert "no history dependence" in sd.reading(_table())


# ---------------------------------------------------------------------------
# What the diagnostic reports as a problem.
# ---------------------------------------------------------------------------


def test_a_request_still_mapped_after_free_is_a_problem_even_when_results_agree():
    runs = [_run(i, "a", [1.0, 2.0]) for i in range(2)]
    runs[0]["request_still_mapped_after_free"] = True
    per_mode = {
        "executor_like": {
            "comparison": sd.compare_sequence(runs, tg),
            "teardown": sd.teardown_report(runs, {"name": "executor_like"}),
        }
    }
    problems = sd.judge(per_mode)
    assert any("still mapped after free_resources" in p for p in problems)


def test_an_unreleased_index_mapper_slot_is_a_problem():
    runs = [_run(i, "a", [1.0, 2.0]) for i in range(2)]
    runs[1]["index_mapper_free_slots_after"] = 3
    per_mode = {
        "executor_like": {
            "comparison": sd.compare_sequence(runs, tg),
            "teardown": sd.teardown_report(runs, {"name": "executor_like"}),
        }
    }
    assert any("IndexMapper slot" in p for p in sd.judge(per_mode))


def test_surviving_page_content_is_reported_but_is_not_by_itself_a_problem():
    # Nothing memsets a freed page, so its bytes surviving is expected. It
    # matters only because the next owner can read them, which is what the
    # zero_freed mode tests -- calling it a defect here would make every run
    # fail for a reason that is not one.
    runs = [_run(i, "a", [1.0, 2.0]) for i in range(2)]
    runs[0]["freed_pages_survived"] = ["layer0.SWA"]
    teardown = sd.teardown_report(runs, {"name": "executor_like"})
    assert teardown["requests_with_surviving_page_content"] == [0]
    per_mode = {"executor_like": {"comparison": sd.compare_sequence(runs, tg), "teardown": teardown}}
    assert sd.judge(per_mode) == []


# ---------------------------------------------------------------------------
# The gating half's own rules.
# ---------------------------------------------------------------------------


def _exec_run(engine, index, pid, tokens, logits):
    rows = torch.tensor(logits, dtype=torch.float32)
    return {
        "engine": engine,
        "index": index,
        "prompt_id": pid,
        "prompt_tokens": 8,
        "token_ids": list(tokens),
        "text": "x",
        "finish_reason": "length",
        "nonfinite_logprobs": 0,
        "logits_finite": True,
        "logits_sha": sd._sha(rows),
        "logits": rows,
        "elapsed_s": 0.1,
    }


def _pass(engine, runs):
    return {
        "engine": engine,
        "construct_s": 1.0,
        "worker_log": "/dev/null",
        "runtime_contract": {"resolved": {}, "problems": [], "passed": True},
        # One scan shared by every engine, as the suite attaches it.
        "worker_dispatch": {"problems": [], "passed": True},
        "teardown": {"iterations": 3, "last": {"usedNumBlocks": 0}, "problems": []},
        "runs": runs,
    }


def _full_plan(mutate=None):
    passes = []
    for engine, repeats in (("same_engine", 3), ("fresh_engine_a", 1), ("fresh_engine_b", 1)):
        runs = []
        for r in range(repeats):
            for pid in sd.GATING_PROMPTS:
                runs.append(
                    _exec_run(engine, len(runs), pid, [1, 2, 3], [[1.0, 0.0], [0.0, 1.0]])
                )
        passes.append(_pass(engine, runs))
    if mutate is not None:
        mutate(passes)
    return passes


def test_the_primer_prompts_are_recorded_as_diagnostics_and_do_not_gate():
    # They answer the question that opened this investigation --- does the
    # executor reproduce a short prompt within one engine and across engines?
    # --- but the acceptance item names two prompts, and quietly widening the
    # gate would be changing the criterion rather than meeting it.
    passes = _full_plan()
    passes[0]["runs"].insert(0, _exec_run("same_engine", 99, "chat_geography", [7], [[1.0, 0.0]]))
    passes[1]["runs"].insert(0, _exec_run("fresh_engine_a", 99, "chat_geography", [8], [[0.0, 1.0]]))
    comparison = sd.compare_executions(passes, tg)
    assert [c["prompt_id"] for c in comparison["diagnostics"]] == ["chat_geography"]
    assert comparison["non_gating_not_reproducible"] == ["chat_geography"]
    assert sd.judge_executions(passes, comparison) == []


def test_a_fully_reproducible_plan_passes():
    passes = _full_plan()
    comparison = sd.compare_executions(passes, tg)
    assert comparison["same_engine_executions_compared"] == 4
    assert comparison["fresh_engine_executions_compared"] == 4
    assert comparison["same_engine_identical"] and comparison["fresh_engine_identical"]
    assert sd.judge_executions(passes, comparison) == []


def test_a_token_divergence_names_the_step():
    def mutate(passes):
        passes[0]["runs"][2]["token_ids"] = [1, 9, 3]

    passes = _full_plan(mutate)
    comparison = sd.compare_executions(passes, tg)
    problems = sd.judge_executions(passes, comparison)
    assert any("tokens diverge at step 1" in p for p in problems)


def test_identical_tokens_with_different_logits_is_still_a_failure():
    # The sharper of the two signals: greedy decoding can absorb a logit
    # difference that a later prompt or a longer generation would not.
    def mutate(passes):
        rows = torch.tensor([[1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
        passes[1]["runs"][0]["logits"] = rows
        passes[1]["runs"][0]["logits_sha"] = sd._sha(rows)

    passes = _full_plan(mutate)
    comparison = sd.compare_executions(passes, tg)
    problems = sd.judge_executions(passes, comparison)
    assert any("same tokens but logits differ from step 1" in p for p in problems)


def test_too_few_executions_cannot_pass_under_this_criterion_name():
    def mutate(passes):
        passes[2]["runs"] = [r for r in passes[2]["runs"] if r["prompt_id"] != "long_prefill_2304"]

    passes = _full_plan(mutate)
    problems = sd.judge_executions(passes, sd.compare_executions(passes, tg))
    assert any("long_prefill_2304: 4 executions recorded" in p for p in problems)


def test_two_engines_cannot_stand_in_for_three():
    def mutate(passes):
        passes.pop()

    passes = _full_plan(mutate)
    problems = sd.judge_executions(passes, sd.compare_executions(passes, tg))
    assert any("engines built" in p for p in problems)


def test_the_shared_worker_scan_is_judged_once_not_once_per_engine():
    """MPI fixes the workers' descriptors at the first `tensorrt_llm` import,
    so all three engines' worker output lands in one file and there is one
    scan. Judging it per engine reported the same failure three times; judging
    a *missing* scan is still a failure, because absence is how a false pass
    would get in."""
    passes = _full_plan()
    scan = {"problems": ["executor_built: 0 of 8 ranks logged it"], "passed": False}
    for entry in passes:
        entry["worker_dispatch"] = scan
    problems = sd.judge_executions(passes, sd.compare_executions(passes, tg))
    assert sum("worker dispatch" in p for p in problems) == 1

    for entry in _full_plan():
        entry.pop("worker_dispatch", None)
    missing = [{k: v for k, v in e.items() if k != "worker_dispatch"} for e in _full_plan()]
    assert any(
        "no worker-dispatch scan" in p
        for p in sd.judge_executions(missing, sd.compare_executions(missing, tg))
    )


def test_a_held_block_after_teardown_fails_the_suite():
    def mutate(passes):
        passes[0]["teardown"]["problems"] = ["7 KV blocks are still held after every request"]

    passes = _full_plan(mutate)
    problems = sd.judge_executions(passes, sd.compare_executions(passes, tg))
    assert any("still held" in p for p in problems)


def test_a_nonfinite_logit_fails_even_when_the_tokens_agree():
    def mutate(passes):
        passes[0]["runs"][0]["logits_finite"] = False

    passes = _full_plan(mutate)
    problems = sd.judge_executions(passes, sd.compare_executions(passes, tg))
    assert any("nonfinite logits" in p for p in problems)


def test_kv_cache_teardown_reads_the_last_iteration_that_reported_blocks():
    class _LLM:
        def get_stats(self, timeout):
            return [
                {"iter": 0, "kvCacheStats": {"usedNumBlocks": 5, "freeNumBlocks": 95}},
                {"iter": 1},
                {"iter": 2, "kvCacheStats": {"usedNumBlocks": 0, "freeNumBlocks": 100}},
            ]

    report = sd.kv_cache_teardown(_LLM())
    assert report["last"]["usedNumBlocks"] == 0
    assert report["problems"] == []


def test_kv_cache_teardown_reports_blocks_that_were_never_returned():
    class _LLM:
        def get_stats(self, timeout):
            return [{"iter": 9, "kvCacheStats": {"usedNumBlocks": 7, "freeNumBlocks": 93}}]

    assert "still held" in sd.kv_cache_teardown(_LLM())["problems"][0]


def test_kv_cache_teardown_records_an_unavailable_stats_channel_rather_than_raising():
    class _LLM:
        def get_stats(self, timeout):
            raise RuntimeError("no stats queue")

    report = sd.kv_cache_teardown(_LLM())
    assert report["problems"] and "no stats queue" in report["problems"][0]


# ---------------------------------------------------------------------------
# The strict auditor, on this artifact shape.
# ---------------------------------------------------------------------------


def _artifact(passes, passed):
    comparison = sd.compare_executions(passes, tg)
    return {
        "suite": "eager_state_determinism",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "engines": [
            {k: v for k, v in entry.items() if k != "runs"}
            | {"runs": [{k: v for k, v in run.items() if k != "logits"} for run in entry["runs"]]}
            for entry in passes
        ],
        "comparison": comparison,
    }


def test_the_auditor_re_derives_a_clean_run_as_clean():
    artifact = _artifact(_full_plan(), passed=True)
    report = evidence.audit_artifact(artifact, {"modules": {}})
    assert report["strict_failures"] == []
    assert report["verdict_disagreements"] == []


def test_the_auditor_refuses_a_run_that_claims_to_pass_with_a_divergence():
    # Without a rule for this artifact shape the auditor would find no module
    # goldens and no `checks`, and would call a failed run clean.
    def mutate(passes):
        passes[1]["runs"][0]["token_ids"] = [1, 9, 3]

    artifact = _artifact(_full_plan(mutate), passed=True)
    report = evidence.audit_artifact(artifact, {"modules": {}})
    assert report["strict_failures"]
    assert any("re-derive" in note for note in report["verdict_disagreements"])
    assert not report["clean"]


def test_the_auditor_refuses_an_artifact_with_no_record_to_re_derive():
    artifact = {"suite": "eager_state_determinism", "status": "passed", "passed": True}
    report = evidence.audit_artifact(artifact, {"modules": {}})
    assert any(f["check"].endswith("missing_record") for f in report["strict_failures"])
