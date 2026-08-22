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
"""Rules behind the DeepSeek-V4-Flash TP8/EP8 LLM API construction evidence.

The construction itself needs eight H100s and several minutes. The *rules* it
is judged by --- which environment the spawned workers must not inherit, which
resolved arguments count as the required contract, and which worker log lines
mean the runtime degraded --- are ordinary functions, and this pins them at CPU
speed so a rule change cannot ride along unnoticed inside an eight-GPU run.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
        ),
        "tests",
        "integration",
        "defs",
        "accuracy",
        "deepseek_v4_flash_h100",
    ),
)

import full_model  # noqa: E402


def _args(**overrides):
    args = SimpleNamespace(
        checkpoint="/models/DeepSeek-V4-Flash",
        max_seq_len=4096,
        max_num_tokens=2048,
        max_batch_size=4,
        max_new_tokens=8,
        kv_fraction=0.3,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _resolved(**overrides):
    """A resolved contract that satisfies every requirement."""
    resolved = {
        "backend": "pytorch",
        "tensor_parallel_size": 8,
        "moe_expert_parallel_size": 8,
        "world_size": 8,
        "custom_tokenizer": "deepseek_v4",
        "tokenizer_class": "DeepseekV4Tokenizer",
        "attn_backend": "TRTLLM",
        "moe_backend": "CUTLASS",
        "max_seq_len": 4096,
        "tokens_per_block": 128,
        "free_gpu_memory_fraction": 0.3,
        "use_kv_cache_manager_v2": True,
        "cuda_graph_config": None,
        "disable_overlap_scheduler": True,
        "enable_chunked_prefill": False,
        "speculative_config": None,
    }
    resolved.update(overrides)
    return resolved


# ---------------------------------------------------------------------------
# Detaching from the launcher.
# ---------------------------------------------------------------------------


def test_detach_removes_every_launcher_variable_and_reports_it():
    env = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "8",
        "MASTER_ADDR": "host",
        "MASTER_PORT": "35915",
        "TORCHELASTIC_RUN_ID": "abc",
        "PATH": "/usr/bin",
        "LLM_MODELS_ROOT": "/models",
    }
    removed = full_model.detach_from_launcher(env)

    assert removed["RANK"] == "0" and removed["WORLD_SIZE"] == "8"
    assert not {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
    } & set(env)
    # Everything the worker legitimately needs survives.
    assert env == {"PATH": "/usr/bin", "LLM_MODELS_ROOT": "/models"}


def test_detach_is_a_no_op_outside_a_launcher():
    env = {"PATH": "/usr/bin"}
    assert full_model.detach_from_launcher(env) == {}
    assert env == {"PATH": "/usr/bin"}


def test_rank_variables_are_all_covered():
    """The exact set torchrun exports per rank; missing one is a silent hang."""
    assert {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    } <= set(full_model.LAUNCHER_ENV_VARS)


# ---------------------------------------------------------------------------
# The required runtime contract.
# ---------------------------------------------------------------------------


def test_the_required_contract_has_no_problems():
    assert full_model._contract_problems(_resolved(), _args()) == []


@pytest.mark.parametrize(
    "field, wrong",
    [
        ("backend", "tensorrt"),
        ("tensor_parallel_size", 4),
        ("moe_expert_parallel_size", 1),
        ("world_size", 4),
        ("custom_tokenizer", None),
        ("tokenizer_class", "PreTrainedTokenizerFast"),
        ("attn_backend", "VANILLA"),
        ("moe_backend", "TRTLLM"),
        ("tokens_per_block", 32),
        ("use_kv_cache_manager_v2", False),
        ("disable_overlap_scheduler", False),
        ("enable_chunked_prefill", True),
    ],
)
def test_each_required_field_is_actually_required(field, wrong):
    problems = full_model._contract_problems(_resolved(**{field: wrong}), _args())
    assert len(problems) == 1 and problems[0].startswith(f"{field}=")


def test_an_enabled_cuda_graph_is_a_contract_violation_for_the_eager_baseline():
    problems = full_model._contract_problems(
        _resolved(cuda_graph_config={"batch_sizes": [1]}), _args()
    )
    assert problems and problems[0].startswith("cuda_graph_config=")


@pytest.mark.parametrize("fraction", [None, 0.0, 1.0, 1.5, "0.3"])
def test_the_kv_fraction_must_be_an_explicit_fraction(fraction):
    problems = full_model._contract_problems(_resolved(free_gpu_memory_fraction=fraction), _args())
    assert any("free_gpu_memory_fraction" in p for p in problems)


def test_every_disagreement_is_reported_in_one_pass():
    """An eight-rank construction must not have to be repeated per problem."""
    problems = full_model._contract_problems(
        _resolved(attn_backend="VANILLA", tokens_per_block=32, use_kv_cache_manager_v2=False),
        _args(),
    )
    assert len(problems) == 3


# ---------------------------------------------------------------------------
# Reading the workers' log.
# ---------------------------------------------------------------------------


def _healthy_log(ranks=range(8), drop=()):
    """The lines a healthy eight-worker construction writes, one set per rank.

    Copied from the markers a real passing run emitted, so a rule change that
    stops matching the runtime's actual output fails here first.
    """
    template = [
        "ATTENTION RUNTIME FEATURES:  AttentionRuntimeFeatures(chunked_prefill=False)",
        "DeepseekV4CacheManager role-to-pool/lifecycle mapping:",
        "KVCacheManagerV2 pools: 12",
        "deepseek_role=COMPRESS, compress_ratio=128, role=deepseek_v4_compress",
        "Detected DeepSeek-V4 routed MoE %s checkpoint layout; using %s for routed "
        "experts. MXFP4 W4A16_MXFP4",
    ]
    lines = []
    for rank in ranks:
        for text in template:
            if any(token in text for token in drop):
                continue
            lines.append(f"[08/22/2026-11:31:42] [TRT-LLM] [I] [_torch][RANK {rank}] {text}")
    return "\n".join(lines)


def test_a_healthy_log_passes_and_names_every_rank():
    scanned = full_model.scan_worker_log(_healthy_log(), world_size=8)
    assert scanned["passed"] and scanned["problems"] == []
    assert scanned["ranks_logged"] == list(range(8))
    for marker in scanned["required_markers"].values():
        assert marker["ranks"] == list(range(8))


def test_a_missing_rank_fails():
    scanned = full_model.scan_worker_log(_healthy_log(range(7)), world_size=8)
    assert not scanned["passed"]
    assert any("7 of 8 worker ranks logged at all" in p for p in scanned["problems"])


@pytest.mark.parametrize(
    "token, marker",
    [
        ("ATTENTION RUNTIME", "executor_built"),
        ("role-to-pool", "deepseek_v4_cache_manager"),
        ("KVCacheManagerV2", "kv_cache_manager_v2"),
        ("compress_ratio=128", "compressed_sparse_pools"),
        ("W4A16_MXFP4", "routed_moe_w4a16_mxfp4"),
    ],
)
def test_each_required_marker_is_actually_required(token, marker):
    scanned = full_model.scan_worker_log(_healthy_log(drop=(token,)), world_size=8)
    assert not scanned["passed"]
    assert any(p.startswith(f"{marker}: 0 of 8") for p in scanned["problems"])


def test_a_marker_missing_on_one_rank_fails():
    """The dangerous shape: seven ranks on the right path and one that is not."""
    log = _healthy_log(range(7)) + "\n" + _healthy_log([7], drop=("W4A16_MXFP4",))
    scanned = full_model.scan_worker_log(log, world_size=8)
    assert not scanned["passed"]
    assert any(p.startswith("routed_moe_w4a16_mxfp4: 7 of 8") for p in scanned["problems"])
    assert scanned["ranks_logged"] == list(range(8))


@pytest.mark.parametrize(
    "line, expected",
    [
        ("[TRT-LLM] [E] [executor][RANK 4] Executor worker initialization error", "worker_error"),
        ("Traceback (most recent call last):", "bare_traceback"),
        ("[RANK 1] falling back to VANILLA attention", "attention_backend_fallback"),
        ("[RANK 2] falling back to KVCacheManager", "kv_cache_manager_downgrade"),
        ("[RANK 5] torch.OutOfMemoryError: CUDA out of memory", "cuda_oom"),
        ("[RANK 6] MPI_ABORT was invoked", "mpi_abort"),
    ],
)
def test_each_degradation_marker_fails_the_scan(line, expected):
    scanned = full_model.scan_worker_log(_healthy_log() + "\n" + line, world_size=8)
    assert not scanned["passed"]
    assert expected in scanned["degradation_markers"]
    assert any(p.startswith(expected) for p in scanned["problems"])


@pytest.mark.parametrize(
    "line",
    [
        "[TRT-LLM] [W] [_torch] TileIR requires compute capability 10.0 or higher, but the "
        "current device has 9.0. TileIR kernels will not be available",
        "[TRT-LLM] [W] [_torch][RANK 3] [AutoTuner] trtllm::fused_moe::gemm1 using the "
        "fallback tactic, due to cache miss on input shapes",
        "* NVLS (NVLink SHARP) DISABLED for NCCL -- falling back to NVLink P2P *",
        "[TRT-LLM] [W] [_torch][RANK 0] `torch.isnan` or `torch.isinf` is not implemented "
        "for current kv cache dtype, related checks are skipped",
    ],
)
def test_benign_hopper_chatter_does_not_fail_the_scan(line):
    """Every one of these appeared in a run that constructed and generated.

    A Blackwell-only kernel family declining to load, an autotuner cache miss
    and an NCCL transport choice are not the model taking a different path, and
    treating them as failures would make this evidence unusable.
    """
    scanned = full_model.scan_worker_log(_healthy_log() + "\n" + line, world_size=8)
    assert scanned["passed"], scanned["problems"]


def test_the_indexer_dtype_substitution_is_recorded_not_failed():
    """SM90 has no FP4 indexer; the FP8 route is the one Stage 1 validated."""
    line = (
        "[TRT-LLM] [W] [llmapi][RANK 0] DeepSeek-V4 defaults indexer_k_dtype to 'fp4', "
        "but the current device is SM90; falling back to 'fp8'."
    )
    scanned = full_model.scan_worker_log(_healthy_log() + "\n" + line, world_size=8)
    assert scanned["passed"], scanned["problems"]
    assert "fp8" in scanned["notable"]["indexer_dtype_on_sm90"]


def test_the_scan_reports_the_offending_line_not_only_the_rule():
    scanned = full_model.scan_worker_log(
        _healthy_log() + "\n[RANK 2] falling back to VANILLA attention", world_size=8
    )
    assert "VANILLA" in scanned["degradation_markers"]["attention_backend_fallback"]


# ---------------------------------------------------------------------------
# Logprob flattening: the NaN/Inf gate reads whatever shape the sampler returns.
# ---------------------------------------------------------------------------


class _Logprob:
    def __init__(self, value):
        self.logprob = value


def test_logprobs_flatten_from_the_mapping_shape():
    steps = [{5: _Logprob(-0.5)}, {7: _Logprob(-1.25)}]
    assert full_model._flatten_logprobs(steps) == [-0.5, -1.25]


def test_logprobs_flatten_from_the_flat_shape():
    assert full_model._flatten_logprobs([-0.5, -1.25]) == [-0.5, -1.25]


def test_no_logprobs_flattens_to_nothing():
    assert full_model._flatten_logprobs(None) == []


def test_nonfinite_logprobs_survive_flattening_so_they_can_be_counted():
    values = full_model._flatten_logprobs(
        [{1: _Logprob(float("nan"))}, {2: _Logprob(float("-inf"))}]
    )
    assert len(values) == 2 and values[0] != values[0]


# ---------------------------------------------------------------------------
# The construction kwargs are the criterion, spelled once.
# ---------------------------------------------------------------------------


def test_llm_kwargs_spell_the_criterion():
    kwargs = full_model.llm_kwargs(_args())
    assert kwargs["backend"] == "pytorch"
    assert kwargs["tensor_parallel_size"] == 8
    assert kwargs["moe_expert_parallel_size"] == 8
    assert kwargs["custom_tokenizer"] == "deepseek_v4"
    assert kwargs["max_seq_len"] == 4096
    assert kwargs["attn_backend"] == "TRTLLM"
    assert kwargs["moe_config"].backend == "CUTLASS"
    assert kwargs["kv_cache_config"].tokens_per_block == 128
    assert kwargs["kv_cache_config"].free_gpu_memory_fraction == 0.3
    assert kwargs["kv_cache_config"].use_kv_cache_manager_v2 is True
    assert kwargs["cuda_graph_config"] is None
    assert kwargs["disable_overlap_scheduler"] is True


def test_the_described_kwargs_are_json_safe():
    import json

    described = full_model._describe_kwargs(full_model.llm_kwargs(_args()))
    json.dumps(described)
    assert described["moe_config"]["backend"] == "CUTLASS"
    assert described["kv_cache_config"]["tokens_per_block"] == 128


def test_llm_kwargs_ask_for_the_logits_the_parity_gate_needs():
    """Generation logits only come back when the runtime was built to gather them."""
    assert full_model.llm_kwargs(_args())["gather_generation_logits"] is True


# ---------------------------------------------------------------------------
# Judging the source comparison. Both halves are pure functions over recorded
# measurements, so every failure mode is reachable without eight GPUs.
# ---------------------------------------------------------------------------

GATING = [
    "cache_boundary_257",
    "chat_geography",
    "code_python_function",
    "long_prefill_2304",
    "reasoning_word_problem",
]
NON_GATING = ["chat_arithmetic"]
LIMITS = {"cosine_min": 0.99, "rel_max_abs_max": 0.05, "argmax_rule": "exact"}
REPLAY_GATE = {"min_prompts": 5}
PARITY_GATE = {"min_prompts": 5, "min_new_tokens": 32}

CATEGORIES = {
    "cache_boundary_257": "cache_boundary",
    "chat_geography": "plain_chat",
    "code_python_function": "code",
    "long_prefill_2304": "long_prefill",
    "reasoning_word_problem": "reasoning",
    "chat_arithmetic": "plain_chat",
}


def _step(index=0, cosine=0.9995, rel_max_abs=0.01, argmax_match=True, finite=True):
    return {
        "step": index,
        "cosine": cosine,
        "max_abs": 0.5,
        "mean_abs": 0.05,
        "rel_max_abs": rel_max_abs,
        "argmax_match": argmax_match,
        "finite": finite,
        "source_token": 671,
        "trtllm_token": 671 if argmax_match else 42,
    }


def _entry(pid, steps=32, tokens=32, divergence=None, eos_at=None, **overrides):
    """A measured prompt. ``steps`` is a count, or the step list to use as is."""
    source = list(range(100, 100 + tokens))
    got = list(source)
    if divergence is not None:
        got[divergence] = 999999
    entry = {
        "category": CATEGORIES[pid],
        "thinking_mode": "chat",
        "prompt_tokens": 16,
        "source_tokens": source,
        "trtllm_tokens": got,
        "repeat_tokens": list(got),
        "first_divergence": divergence,
        "source_eos_at": eos_at,
        "text": "an answer",
        "repeat_text_identical": True,
        "finish_reason": "length",
        "nonfinite_logprobs": 0,
        "logprob_steps": tokens,
        "trtllm_logits_finite": True,
        "trtllm_logit_rows": [tokens, 129280],
        "reference_logit_rows": [tokens, 129280],
        "steps": steps if isinstance(steps, list) else [_step(i) for i in range(steps)],
        "elapsed_s": [1.0, 1.0],
    }
    entry.update(overrides)
    return entry


def _measured(**overrides):
    measured = {pid: _entry(pid) for pid in GATING + NON_GATING}
    measured.update(overrides)
    return measured


def test_a_clean_measurement_passes_both_gates():
    measured = _measured()
    replay = full_model.judge_logit_replay(measured, LIMITS, REPLAY_GATE, GATING, NON_GATING)
    parity = full_model.judge_generation_parity(measured, PARITY_GATE, GATING, NON_GATING)
    assert replay["passed"] and replay["problems"] == []
    assert parity["passed"] and parity["problems"] == []
    assert replay["prompts_passing"] == sorted(GATING)
    assert set(parity["categories_passing"]) >= {"plain_chat", "reasoning"}


@pytest.mark.parametrize(
    "step_kwargs, expected",
    [
        ({"argmax_match": False}, "greedy argmax"),
        ({"cosine": 0.5}, "cosine"),
        ({"rel_max_abs": 0.9}, "rel_max_abs"),
        ({"finite": False}, "non-finite"),
    ],
)
def test_each_registered_logit_rule_can_fail(step_kwargs, expected):
    measured = _measured(chat_geography=_entry("chat_geography", steps=[_step(0, **step_kwargs)]))
    replay = full_model.judge_logit_replay(measured, LIMITS, REPLAY_GATE, GATING, NON_GATING)
    assert not replay["passed"]
    assert any(expected in p for p in replay["problems"])
    assert "chat_geography" not in replay["prompts_passing"]


def test_a_non_gating_prompt_is_reported_but_cannot_fail_the_logit_gate():
    """The registered fixture's exclusion, honoured exactly: recorded, not waived."""
    measured = _measured(
        chat_arithmetic=_entry("chat_arithmetic", steps=[_step(0, argmax_match=False, cosine=0.2)])
    )
    replay = full_model.judge_logit_replay(measured, LIMITS, REPLAY_GATE, GATING, NON_GATING)
    assert replay["passed"]
    detail = replay["per_prompt"]["chat_arithmetic"]
    assert detail["gating"] is False and detail["passed"] is False
    assert "greedy argmax" in detail["problem"]


def test_a_missing_prompt_fails_rather_than_being_skipped():
    measured = _measured()
    del measured["chat_geography"]
    replay = full_model.judge_logit_replay(measured, LIMITS, REPLAY_GATE, GATING, NON_GATING)
    parity = full_model.judge_generation_parity(measured, PARITY_GATE, GATING, NON_GATING)
    assert not replay["passed"] and not parity["passed"]
    assert any("was not measured" in p for p in replay["problems"])


def test_too_few_gating_prompts_fails_even_when_each_one_passes():
    """Four perfect prompts are not the five the registered gate asks for."""
    gating = GATING[:4]
    measured = {pid: _entry(pid) for pid in gating}
    replay = full_model.judge_logit_replay(measured, LIMITS, REPLAY_GATE, gating, [])
    assert not replay["passed"]
    assert any("requires 5" in p for p in replay["problems"])


def test_a_token_divergence_fails_and_says_where_it_is():
    measured = _measured(cache_boundary_257=_entry("cache_boundary_257", divergence=27, eos_at=25))
    parity = full_model.judge_generation_parity(measured, PARITY_GATE, GATING, NON_GATING)
    assert not parity["passed"]
    detail = parity["per_prompt"]["cache_boundary_257"]
    assert detail["divergence_after_eos"] is True
    assert "2 steps after the source's EOS at 25" in detail["problem"]


def test_a_divergence_inside_the_answer_is_named_differently():
    measured = _measured(chat_geography=_entry("chat_geography", divergence=5, eos_at=None))
    parity = full_model.judge_generation_parity(measured, PARITY_GATE, GATING, NON_GATING)
    detail = parity["per_prompt"]["chat_geography"]
    assert detail["divergence_after_eos"] is False
    assert "inside the answer" in detail["problem"]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"repeat_tokens": [1, 2, 3]}, "different tokens"),
        ({"text": "   "}, "generated no text"),
        ({"nonfinite_logprobs": 3}, "non-finite logprobs"),
        ({"trtllm_logits_finite": False}, "non-finite generation logits"),
    ],
)
def test_each_registered_parity_rule_can_fail(overrides, expected):
    measured = _measured(chat_geography=_entry("chat_geography", **overrides))
    parity = full_model.judge_generation_parity(measured, PARITY_GATE, GATING, NON_GATING)
    assert not parity["passed"]
    assert any(expected in p for p in parity["problems"])


def test_a_short_generation_cannot_satisfy_the_thirty_two_step_rule():
    measured = _measured(chat_geography=_entry("chat_geography", steps=12, tokens=12))
    parity = full_model.judge_generation_parity(measured, PARITY_GATE, GATING, NON_GATING)
    assert not parity["passed"]
    assert any("the registered gate requires 32" in p for p in parity["problems"])


def test_a_pass_built_only_from_one_mode_is_not_the_thing_that_was_asked_for():
    """The task names plain chat *and* reasoning; five of one is not five."""
    gating = ["chat_geography", "a", "b", "c", "d"]
    measured = {pid: _entry("chat_geography") for pid in gating}
    parity = full_model.judge_generation_parity(measured, PARITY_GATE, gating, [])
    assert not parity["passed"]
    assert any("'reasoning'" in p for p in parity["problems"])


# ---------------------------------------------------------------------------
# The reference must describe the measurement it is about to judge.
# ---------------------------------------------------------------------------

import source_reference  # noqa: E402


def _reference(**overrides):
    artifact = {
        "checkpoint_revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
        "manifest_provenance": {"sha256": {"prompts.json": "abc"}},
        "parity_tokens": 32,
        "passed": True,
        "prompts": {pid: {"tokens": [1]} for pid in GATING + NON_GATING},
    }
    artifact.update(overrides)
    return artifact


def _usable(artifact, **overrides):
    kwargs = {
        "checkpoint_revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
        "prompts_sha256": "abc",
        "parity_tokens": 32,
        "prompt_ids": GATING + NON_GATING,
    }
    kwargs.update(overrides)
    return source_reference.usable(artifact, **kwargs)


def test_a_matching_reference_is_usable():
    assert _usable(_reference()) == []


@pytest.mark.parametrize(
    "artifact_overrides, call_overrides, expected",
    [
        ({"checkpoint_revision": "other"}, {}, "checkpoint"),
        (
            {"manifest_provenance": {"sha256": {"prompts.json": "different"}}},
            {},
            "different prompts manifest",
        ),
        ({"parity_tokens": 16}, {}, "steps per prompt"),
        ({"passed": False}, {}, "did not pass its own checks"),
        ({}, {"prompt_ids": GATING + ["unseen"]}, "missing prompts"),
    ],
)
def test_a_reference_for_a_different_measurement_is_refused(
    artifact_overrides, call_overrides, expected
):
    """Each of these would otherwise compare against the wrong thing silently."""
    problems = _usable(_reference(**artifact_overrides), **call_overrides)
    assert any(expected in p for p in problems), problems


def test_the_sidecar_path_follows_the_artifact():
    assert source_reference.logits_path("/cache/x/source_reference.json").endswith(
        "/cache/x/source_reference.logits.npz"
    )


def test_a_reference_whose_sidecar_was_replaced_is_refused(tmp_path):
    """A half-written or swapped .npz must not be silently compared against."""
    import json

    import numpy as np

    artifact_path = tmp_path / "source_reference.json"
    payload = source_reference.write(
        str(artifact_path), _reference(), {"chat_geography": np.zeros((2, 4), dtype=np.float32)}
    )
    artifact_path.write_text(json.dumps(payload))
    loaded, logits = source_reference.load(str(artifact_path))
    assert logits["chat_geography"].shape == (2, 4)

    np.savez(
        payload["logits_sidecar"]["path"],
        **{"chat_geography__logits": np.ones((2, 4), dtype=np.float32)},
    )
    with pytest.raises(RuntimeError, match="not the file that capture produced"):
        source_reference.load(str(artifact_path))


# ---------------------------------------------------------------------------
# The strict auditor must re-derive the two Goal-3.4 verdicts, not trust them.
# ---------------------------------------------------------------------------

import importlib.util  # noqa: E402

_EVIDENCE = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    ),
    "tests",
    "integration",
    "defs",
    "accuracy",
    "deepseek_v4_flash_h100_evidence.py",
)
_spec = importlib.util.spec_from_file_location("dsv4_evidence", _EVIDENCE)
evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence)


def _audited_artifact(**overrides):
    """An artifact whose checks genuinely pass, as the suite would write them.

    Carries all five `eager_full_model` checks, not only the two parity ones:
    the audit now requires the suite's whole defined set, so a fixture missing
    the runtime three would not be an honest artifact.
    """
    measured = _measured()
    checks = {
        "runtime_contract": {"passed": True, "problems": []},
        "custom_tokenizer": {"passed": True, "problems": []},
        "worker_dispatch": {"passed": True, "problems": []},
        "source_logit_replay": full_model.judge_logit_replay(
            measured, LIMITS, REPLAY_GATE, GATING, NON_GATING
        ),
        "generation_parity": full_model.judge_generation_parity(
            measured, PARITY_GATE, GATING, NON_GATING
        ),
    }
    assert all(c["passed"] for c in checks.values())
    artifact = {
        "suite": "eager_full_model",
        "passed": True,
        "status": "passed",
        "checkpoint_revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
        "checks": checks,
        "source_reference": {
            "reference_provenance": {
                "checkpoint_revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
                "native_generate_golden_passed": True,
                "logits_sidecar": {
                    "path": "/cache/dsv4_flash_h100/source_reference.logits.npz",
                    "sha256": "0" * 64,
                    "prompts": sorted(GATING + NON_GATING),
                },
            }
        },
    }
    artifact.update(overrides)
    return artifact


def _audit(artifact):
    return evidence.audit_artifact(artifact, {"modules": {}})


def test_an_honest_parity_artifact_audits_clean():
    report = _audit(_audited_artifact())
    assert report["clean"], (report["strict_failures"], report["verdict_disagreements"])


@pytest.mark.parametrize(
    "deleted",
    [
        ["source_logit_replay"],
        ["generation_parity"],
        ["source_logit_replay", "generation_parity"],
        ["runtime_contract"],
        ["custom_tokenizer"],
        ["worker_dispatch"],
        ["runtime_contract", "custom_tokenizer", "worker_dispatch"],
    ],
)
def test_a_deleted_check_cannot_audit_clean(deleted):
    """Re-deriving the checks that are present says nothing about the ones removed.

    This is the Reviewer's probe: drop a check, claim `passed=true`, and every
    remaining rule still agrees, because the rules that would have disagreed
    went with the check.
    """
    artifact = _audited_artifact()
    for name in deleted:
        artifact["checks"].pop(name, None)
    artifact["passed"] = True
    artifact["status"] = "passed"
    report = _audit(artifact)
    assert not report["clean"], f"deleting {deleted} still audited clean"
    problems = [p for f in report["strict_failures"] for p in f["problems"]]
    for name in deleted:
        assert any(repr(name) in p for p in problems), (problems, name)


def test_an_empty_check_set_cannot_audit_clean():
    artifact = _audited_artifact(suite="kernel_contract", checks={}, source_reference={})
    report = _audit(artifact)
    assert not report["clean"]
    assert any("empty check set" in p for f in report["strict_failures"] for p in f["problems"])


def test_a_suite_with_an_argument_driven_check_set_is_not_over_constrained():
    """`activation_replay_eager` names its checks after `--replay-layers`.

    Requiring a fixed set there would fail every legitimate diagnostic run,
    so those suites are covered by the non-empty rule alone.
    """
    artifact = _audited_artifact(
        suite="activation_replay_eager",
        checks={"layer40.prefill.compressor": {"module": "compressor", "passed": True}},
        source_reference={},
    )
    assert _audit(artifact)["clean"]


@pytest.mark.parametrize(
    "check, pid, field, value",
    [
        ("source_logit_replay", "chat_geography", "cosine", 0.5),
        ("source_logit_replay", "chat_geography", "rel_max_abs", 0.9),
        ("source_logit_replay", "chat_geography", "argmax_match", False),
        ("source_logit_replay", "chat_geography", "finite", False),
        ("generation_parity", "chat_geography", "first_divergence", 7),
        ("generation_parity", "chat_geography", "repeat_identical", False),
        ("generation_parity", "chat_geography", "nonempty_text", False),
        ("generation_parity", "chat_geography", "nonfinite_logprobs", 3),
        ("generation_parity", "chat_geography", "logits_finite", False),
        ("generation_parity", "chat_geography", "trtllm_tokens", 8),
    ],
)
def test_a_forged_pass_over_failing_numbers_is_caught(check, pid, field, value):
    """The exact probe that found this gap: flip the numbers, keep passed=true."""
    artifact = _audited_artifact()
    detail = artifact["checks"][check]["per_prompt"][pid]
    detail[field] = value
    assert detail["passed"] is True, "the forged entry must still claim it passed"
    report = _audit(artifact)
    assert not report["clean"]
    assert any(f"{check}.{pid}" in f["check"] for f in report["strict_failures"])
    assert any(f"{check}.{pid}" in note for note in report["verdict_disagreements"])


def test_a_forged_check_level_pass_is_caught():
    """Every prompt fails, but the check and the artifact both claim success."""
    artifact = _audited_artifact()
    for pid, detail in artifact["checks"]["generation_parity"]["per_prompt"].items():
        detail["first_divergence"] = 3
        detail["source_token_at_divergence"] = 1
        detail["trtllm_token_at_divergence"] = 2
        detail["passed"] = True
    report = _audit(artifact)
    assert not report["clean"]
    assert any(f["check"] == "generation_parity" for f in report["strict_failures"])


def test_a_check_that_drops_its_registered_thresholds_cannot_audit_clean():
    artifact = _audited_artifact()
    artifact["checks"]["source_logit_replay"].pop("limits")
    report = _audit(artifact)
    assert not report["clean"]
    assert any("no registered 'limits'" in n for n in report["verdict_disagreements"])


def test_a_prompt_recorded_without_the_facts_cannot_audit_clean():
    """A verdict nobody can recompute is not evidence, however it was reached."""
    artifact = _audited_artifact()
    artifact["checks"]["source_logit_replay"]["per_prompt"]["chat_geography"].pop("cosine")
    report = _audit(artifact)
    assert not report["clean"]
    assert any("no recorded cosine" in str(f["problems"]) for f in report["strict_failures"])


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda p: p["logits_sidecar"].pop("sha256"), "without a hashed source-reference"),
        (lambda p: p.update(native_generate_golden_passed=False), "not anchored to the native"),
        (lambda p: p.update(checkpoint_revision="deadbeef"), "captured at checkpoint revision"),
        (lambda p: p["logits_sidecar"].update(prompts=["chat_geography"]), "does not contain"),
    ],
)
def test_reference_provenance_is_required_for_a_parity_verdict(mutate, expected):
    artifact = _audited_artifact()
    mutate(artifact["source_reference"]["reference_provenance"])
    report = _audit(artifact)
    assert not report["clean"]
    assert any(expected in n for n in report["verdict_disagreements"]), report


def test_an_artifact_without_parity_checks_needs_no_reference():
    """The provenance rule must not fire on the suites that have no parity."""
    artifact = _audited_artifact(checks={}, source_reference={})
    assert evidence._audit_reference_provenance(artifact) == []


def test_a_tampered_sidecar_on_disk_is_caught(tmp_path):
    import numpy as np

    path = tmp_path / "source_reference.logits.npz"
    np.savez(path, chat_geography__logits=np.zeros((2, 4), dtype=np.float32))
    good = source_reference.sha256_file(str(path))
    artifact = _audited_artifact()
    sidecar = artifact["source_reference"]["reference_provenance"]["logits_sidecar"]
    sidecar.update(path=str(path), sha256=good)
    assert evidence._rehash_sidecars(artifact) == []

    np.savez(path, chat_geography__logits=np.ones((2, 4), dtype=np.float32))
    notes = evidence._rehash_sidecars(artifact)
    assert notes and "were not judged against this file" in notes[0]


def test_a_sidecar_recorded_but_missing_is_caught(tmp_path):
    artifact = _audited_artifact()
    sidecar = artifact["source_reference"]["reference_provenance"]["logits_sidecar"]
    sidecar.update(path=str(tmp_path / "gone.npz"), sha256="1" * 64)
    notes = evidence._rehash_sidecars(artifact)
    assert notes and "missing on disk" in notes[0]


# ---------------------------------------------------------------------------
# The replay's liveness verdict reads the counter the runtime actually exports.
# ---------------------------------------------------------------------------

import activation_replay  # noqa: E402


def _counts(**overrides):
    counts = {"context": 4, "generation": 4, "append_rows_dropped": 0}
    counts.update(overrides)
    return counts


def test_a_live_dispatch_passes():
    assert activation_replay.judge_dispatch(_counts(), 4)["passed"]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"append_rows_dropped": 1}, "append_rows_dropped 1 != 0"),
        ({"context": 3}, "context dispatches 3 != 4"),
        ({"generation": 0}, "generation dispatches 0 != 4"),
    ],
)
def test_each_liveness_clause_can_fail(overrides, expected):
    """A dropped compressed row must fail `real_runtime`, not be defaulted away."""
    verdict = activation_replay.judge_dispatch(_counts(**overrides), 4)
    assert not verdict["passed"]
    assert expected in verdict["problems"]


def test_a_renamed_counter_fails_loudly_instead_of_passing_vacuously():
    """`.get('dropped_rows', 0)` is how this clause was silently always true."""
    counts = _counts()
    counts["dropped_rows"] = counts.pop("append_rows_dropped")
    with pytest.raises(KeyError, match="append_rows_dropped"):
        activation_replay.judge_dispatch(counts, 4)


# ---------------------------------------------------------------------------
# The protected-regression classification.
# ---------------------------------------------------------------------------

import regression_report as rr  # noqa: E402

_PTRACE = (
    "E       OSError: [Errno 1] Operation not permitted\n"
    "E       pidfd_getfd(pidfd, targetfd, 0) failed; try adding --cap-add=SYS_PTRACE\n"
)

_BASELINE = {
    "registered_command": "pytest -q <registered files>",
    "required_h100_command": "pytest -q <registered files> -k '...' --junitxml=...",
    "what_gates": "the signature gates; the id list only diffs",
    "environment_failure_signatures": [
        {
            "id": "container_lacks_cap_sys_ptrace",
            "all_of": ["pidfd_getfd", "Operation not permitted", "SYS_PTRACE"],
            "explanation": "the dev container is created without CAP_SYS_PTRACE",
        }
    ],
    "expected_skip_signatures": [
        {
            "id": "upstream_blackwell_only_guard",
            "all_of": ["not supported in pre-Blackwell architecture"],
            "explanation": "skip_pre_blackwell in an unmodified upstream helper",
        }
    ],
    "required_h100_rule": {
        "node_id_any_of": ["deepseek_v4", "dsv4", "sm90"],
        "blackwell_any_of": ["sm100", "blackwell"],
        "excluded_with_evidence": [
            {
                "node_id": "tests.dsv4::test_sm90_too_big",
                "reason": "needs more HBM than this node has",
                "evidence": "fails identically on the pristine tree",
            }
        ],
    },
    "expected_environment_failures": ["tests/moe.py::test_pool"],
    "protected_blackwell_controls": ["tests.dsv4::test_sm100_dispatch_unchanged"],
}

_DEVICES = {"names": ["NVIDIA H100 80GB HBM3"], "compute_capability": ["9.0"]}

#: Every report built here has the registered Blackwell control present and
#: passing, so a test that is not about that gate is not accidentally failing on
#: it. Cases that *are* about it override this.
_BLACKWELL_OK = "tests.dsv4::test_sm100_dispatch_unchanged"


def _log(failures=(), counts=None, exit_code=1):
    failures = list(failures)
    if counts is None:
        counts = {"passed": 3, "failed": len(failures), "skipped": 0}
    return {
        "path": "registered.log",
        "counts": counts,
        "exit_code": exit_code,
        "failures": failures,
        "failed_ids": sorted(f["node_id"] for f in failures),
        "error_ids": [],
    }


def _failure(node_id, message="OSError", text=_PTRACE, outcome="failed", source="banner"):
    return {
        "node_id": node_id,
        "outcome": outcome,
        "message": message,
        "text": text,
        "traceback_source": source,
    }


def _junit(node_id, outcome="passed", message="", text=""):
    return {
        "node_id": node_id,
        "file": "",
        "outcome": outcome,
        "message": message,
        "text": text,
    }


def _report(*, failures=(), required=None, counts=None, baseline=None, blackwell=True):
    """Build a report. `blackwell=False` omits the registered control on purpose.

    The control is injected by default so a test about something else does not
    trip the Blackwell gate; the tests that *are* about that gate opt out.
    """
    if required is None:
        required = [_junit("tests.sparse.deepseek_v4::test_sm90_kernel")]
    if blackwell and not any(r["node_id"] == _BLACKWELL_OK for r in required):
        required = [*required, _junit(_BLACKWELL_OK)]
    return rr.build_report(
        baseline=dict(_BASELINE if baseline is None else baseline),
        registered_log=_log(failures, counts),
        required_records=required,
        required_command="pytest -k ...",
        device_report=_DEVICES,
    )


def test_the_known_container_fault_classifies_as_environment():
    """The signature is matched against the failure's own traceback, clause by clause."""
    report = _report(failures=[_failure("tests/moe.py::test_pool")])
    assert report["passed"], report["problems"]
    assert report["failures"]["environment"] == 1
    assert report["failures"]["genuine"] == []
    assert report["failures"]["by_signature"]["container_lacks_cap_sys_ptrace"] == [
        "tests/moe.py::test_pool"
    ]


def test_a_baselined_test_that_fails_differently_is_genuine():
    """The baseline narrows what counts as expected; it cannot forgive a new bug.

    This is the difference between adjudicating and waiving. `tests/moe.py::test_pool`
    is in the registered baseline, so an id-only rule would call it expected. Its
    traceback is now an assertion error, so it is reported genuine and fails.
    """
    report = _report(
        failures=[_failure("tests/moe.py::test_pool", "AssertionError", "E   assert 3 == 4\n")]
    )
    assert not report["passed"]
    assert report["failures"]["genuine"] == ["tests/moe.py::test_pool"]
    assert report["failures"]["new_against_baseline"] == []
    assert "match no registered environment signature" in report["problems"][0]
    assert "assert 3 == 4" in report["failures"]["detail"][0]["excerpt"]


def test_a_partial_signature_match_is_not_a_match():
    """Every clause must be present, so a lookalike failure is not absorbed."""
    text = "E   OSError: [Errno 1] Operation not permitted\n"  # no pidfd_getfd, no cap hint
    report = _report(failures=[_failure("tests/moe.py::test_pool", "OSError", text)])
    assert report["failures"]["genuine"] == ["tests/moe.py::test_pool"]


def test_a_failure_whose_traceback_is_missing_is_not_classified_silently():
    """No evidence is not the same as no problem."""
    report = _report(failures=[_failure("tests/moe.py::test_pool", "", "", source="unresolved")])
    assert not report["passed"]
    assert any("no traceback of their own" in p for p in report["problems"])


def test_a_new_failure_is_reported_even_when_it_looks_environmental():
    """A *new* environment failure is still a change, so it is surfaced."""
    report = _report(failures=[_failure("tests/other.py::test_new")])
    assert not report["passed"]
    assert report["failures"]["new_against_baseline"] == ["tests/other.py::test_new"]
    assert report["failures"]["environment"] == 1


def test_a_baselined_failure_that_now_passes_is_reported_resolved():
    """Reported rather than ignored: the environment changed under us."""
    report = _report(failures=[])
    assert report["failures"]["resolved_against_baseline"] == ["tests/moe.py::test_pool"]
    assert report["passed"], report["problems"]


def test_an_error_outcome_is_classified_like_a_failure():
    """A setup error hides a regression just as well as a failure does."""
    report = _report(
        failures=[_failure("tests/moe.py::test_pool", outcome="error")],
        counts={"passed": 3, "failed": 0, "error": 1, "skipped": 0},
    )
    assert report["failures"]["total"] == 1
    assert report["failures"]["environment"] == 1
    assert report["passed"], report["problems"]


def test_failures_the_log_summarised_but_did_not_yield_are_a_problem():
    """A parser that silently recovers 3 of 74 tracebacks would look clean."""
    report = _report(
        failures=[_failure("tests/moe.py::test_pool")],
        counts={"passed": 3, "failed": 74, "skipped": 0},
    )
    assert not report["passed"]
    assert any("were recovered from it" in p for p in report["problems"])


def test_a_skip_with_an_unregistered_reason_fails_the_report():
    """A required SM90 case that stops running is coverage silently disappearing."""
    report = _report(
        required=[
            _junit("tests.sparse.deepseek_v4::test_sm90_kernel", "skipped", "needs a GPU", "")
        ]
    )
    assert not report["passed"]
    assert report["required_h100_cases"]["skipped"] == [
        {
            "node_id": "tests.sparse.deepseek_v4::test_sm90_kernel",
            "reason": "needs a GPU",
            "signature": None,
        }
    ]
    assert any("required H100 case(s) skipped" in p for p in report["problems"])


def test_a_skip_whose_reason_is_registered_is_recorded_not_flagged():
    """An upstream Blackwell-only guard is not coverage that disappeared."""
    report = _report(
        required=[
            _junit(
                "tests.dsv4::test_sm90_blackwell_kernel",
                "skipped",
                "This test is not supported in pre-Blackwell architecture",
                "",
            )
        ]
    )
    assert report["passed"], report["problems"]
    assert report["required_h100_cases"]["skipped"] == []
    assert report["required_h100_cases"]["skipped_with_registered_reason"] == {
        "upstream_blackwell_only_guard": ["tests.dsv4::test_sm90_blackwell_kernel"]
    }


def test_a_failing_required_h100_case_fails_even_with_a_known_signature():
    """The container fault does not get to take an H100 gate down with it."""
    report = _report(failures=[_failure("tests/dsv4.py::test_sm90_thing")])
    assert not report["passed"]
    assert report["required_h100_cases"]["failed"] == ["tests/dsv4.py::test_sm90_thing"]


def test_an_exclusion_is_one_named_test_and_not_a_pattern():
    """A pattern would quietly grow to cover whatever is failing today.

    `test_sm90_too_big` is excluded because a named entry carries the evidence
    that this node cannot run it. A sibling whose name merely looks similar is
    still required, which is what stops the exclusion list from spreading.
    """
    rule = _BASELINE["required_h100_rule"]
    assert rr.required_here("tests.dsv4::test_sm90_too_big", rule) is False
    assert rr.required_here("tests.dsv4::test_sm90_too_big_variant", rule) is True
    assert rr.required_here("tests.dsv4::test_sm90_dispatch", rule) is True
    assert rr.required_here("tests.unrelated::test_other", rule) is False


def test_an_empty_required_pass_cannot_pass():
    """Selecting nothing is the cheapest way to have no skips."""
    report = _report(required=[], blackwell=False)
    assert not report["passed"]
    assert any("matched no cases" in p for p in report["problems"])


def test_a_run_with_no_passing_tests_cannot_pass():
    report = _report(counts={"passed": 0, "failed": 0, "skipped": 10})
    assert not report["passed"]
    assert any("no passing tests" in p for p in report["problems"])


def test_blackwell_runtime_is_reported_not_measured_with_its_reason():
    """Static SM100 dispatch coverage and Blackwell runtime are separate claims."""
    required = [
        _junit("tests.dsv4::test_sm100_dispatch_unchanged"),
        _junit("tests.dsv4::test_blackwell_branch"),
        _junit("tests.dsv4::test_sm90_thing"),
    ]
    report = _report(required=required)
    block = report["protected_blackwell_dispatch"]
    assert report["passed"], report["problems"]
    assert block["blackwell_runtime"] == "Not measured"
    assert "H100" in block["blackwell_runtime_reason"]
    assert block["static_dispatch_tests"]["registered"] == 1
    assert block["static_dispatch_tests"]["passed"] == 1
    assert block["static_dispatch_tests"]["failed"] == []
    # Discovered but unregistered is context, not a finding: a *new* control is
    # not this criterion's risk. Coverage disappearing is.
    assert block["static_dispatch_tests"]["discovered_not_registered"] == [
        "tests.dsv4::test_blackwell_branch"
    ]


# ---------------------------------------------------------------------------
# The registered protected-Blackwell control inventory.
# ---------------------------------------------------------------------------


def test_a_deleted_blackwell_control_fails_the_report():
    """The exact fail-open this replaces: an empty set counted as clean.

    Counting whatever the run contained could only say "the ones that ran,
    ran". A registered inventory turns a control that stopped being collected
    into a named missing entry.
    """
    report = _report(
        required=[_junit("tests.sparse.deepseek_v4::test_sm90_kernel")], blackwell=False
    )
    assert not report["passed"]
    static = report["protected_blackwell_dispatch"]["static_dispatch_tests"]
    assert static["missing"] == [_BLACKWELL_OK]
    assert static["count"] == 0
    assert any("Blackwell control(s) missing" in p for p in report["problems"])


def test_a_skipped_blackwell_control_fails_the_report():
    """A control that runs but skips proves nothing about the branch."""
    report = _report(required=[_junit(_BLACKWELL_OK, "skipped", "needs SM100 hardware", "")])
    assert not report["passed"]
    static = report["protected_blackwell_dispatch"]["static_dispatch_tests"]
    assert static["skipped"] == [_BLACKWELL_OK]
    assert static["passed"] == 0
    assert any("Blackwell control(s) skipped" in p for p in report["problems"])


def test_a_failing_blackwell_control_fails_the_report():
    """A changed Blackwell branch is exactly what this criterion protects."""
    report = _report(required=[_junit(_BLACKWELL_OK, "failed", "AssertionError", "")])
    assert not report["passed"]
    static = report["protected_blackwell_dispatch"]["static_dispatch_tests"]
    assert static["failed"] == [_BLACKWELL_OK]
    assert any("Blackwell control(s) failed" in p for p in report["problems"])


def test_a_run_that_selected_no_blackwell_control_at_all_fails_the_report():
    """Selecting nothing is the cheapest way to have no failing controls."""
    report = _report(
        required=[_junit("tests.sparse.deepseek_v4::test_sm90_kernel")], blackwell=False
    )
    assert not report["passed"]
    assert report["protected_blackwell_dispatch"]["static_dispatch_tests"]["count"] == 0


def test_an_empty_registered_control_inventory_cannot_pass():
    """Emptying the registry must not be a way to satisfy the registry."""
    report = _report(baseline=dict(_BASELINE, protected_blackwell_controls=[]))
    assert not report["passed"]
    assert any("no protected Blackwell control is registered" in p for p in report["problems"])


def test_the_checked_in_baseline_registers_the_blackwell_controls():
    """The real inventory must be non-empty and free of duplicates."""
    controls = rr.load_baseline()["protected_blackwell_controls"]
    assert controls, "an empty inventory would gate nothing"
    assert controls == sorted(set(controls))
    assert all("::" in node_id for node_id in controls)


def test_the_pytest_log_parser_reads_counts_the_exit_code_and_the_tracebacks(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "....ssss....\n"
        "=================================== FAILURES ===================================\n"
        "_________________________________ test_pool __________________________________\n"
        "    with MPIPoolExecutor() as pool:\n"
        "E       OSError: [Errno 1] Operation not permitted\n"
        "E       pidfd_getfd failed; try adding --cap-add=SYS_PTRACE\n"
        "==================================== ERRORS ====================================\n"
        "________________________ ERROR at setup of test_x _________________________\n"
        "E       OSError: boom\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/unittest/_torch/modules/moe/test_moe_module.py::test_pool - OSError\n"
        "ERROR tests/unittest/_torch/modules/moe/test_moe_backend.py::test_x - OSError\n"
        "407 passed, 1356 skipped, 74 failed, 3 warnings in 512.34s\n"
        "EXIT=1\n"
    )
    parsed = rr.parse_pytest_log(str(log))
    assert parsed["counts"] == {"passed": 407, "skipped": 1356, "failed": 74}
    assert parsed["exit_code"] == 1
    assert parsed["failed_ids"] == [
        "tests/unittest/_torch/modules/moe/test_moe_module.py::test_pool"
    ]
    assert parsed["error_ids"] == ["tests/unittest/_torch/modules/moe/test_moe_backend.py::test_x"]

    by_id = {f["node_id"]: f for f in parsed["failures"]}
    pool = by_id["tests/unittest/_torch/modules/moe/test_moe_module.py::test_pool"]
    assert "pidfd_getfd" in pool["text"] and "SYS_PTRACE" in pool["text"]
    assert pool["message"] == "OSError"
    # The `ERROR at setup of` heading names the same test as the summary line.
    setup = by_id["tests/unittest/_torch/modules/moe/test_moe_backend.py::test_x"]
    assert setup["outcome"] == "error" and "boom" in setup["text"]


def test_the_traceback_of_one_test_does_not_leak_into_another(tmp_path):
    """Blocks are delimited, so a neighbour's ptrace text cannot absolve a bug."""
    log = tmp_path / "run.log"
    log.write_text(
        "=================================== FAILURES ===================================\n"
        "_________________________________ test_env ___________________________________\n"
        "E       OSError: Operation not permitted pidfd_getfd --cap-add=SYS_PTRACE\n"
        "_________________________________ test_real __________________________________\n"
        "E       AssertionError: assert 1 == 2\n"
        "=========================== short test summary info ============================\n"
        "FAILED a.py::test_env - OSError\n"
        "FAILED a.py::test_real - AssertionError\n"
        "1 passed, 2 failed in 1.00s\n"
    )
    parsed = rr.parse_pytest_log(str(log))
    report = rr.build_report(
        baseline=dict(
            _BASELINE, expected_environment_failures=["a.py::test_env", "a.py::test_real"]
        ),
        registered_log=parsed,
        required_records=[_junit("tests.dsv4::test_sm90_thing")],
        required_command="pytest -k ...",
        device_report=_DEVICES,
    )
    assert report["failures"]["genuine"] == ["a.py::test_real"]
    assert report["failures"]["by_signature"]["container_lacks_cap_sys_ptrace"] == [
        "a.py::test_env"
    ]


def test_two_files_with_the_same_test_name_keep_their_own_tracebacks(tmp_path):
    """The fail-open this replaces, end to end from the log.

    pytest heads each traceback with the *leaf* name, which is not unique
    across files. Keying blocks by that name merged `a.py::test_same` and
    `b.py::test_same` into one entry holding both texts, so the assertion
    failure inherited its namesake's `pidfd_getfd` line, matched the container
    signature, and the report came back clean with `genuine == []`.
    """
    log = tmp_path / "run.log"
    log.write_text(
        "=================================== FAILURES ===================================\n"
        "___________________________________ test_same __________________________________\n"
        "a.py:10: in test_same\n"
        "    with MPIPoolExecutor() as pool:\n"
        "E       OSError: [Errno 1] Operation not permitted\n"
        "E       pidfd_getfd failed; try adding --cap-add=SYS_PTRACE\n"
        "___________________________________ test_same __________________________________\n"
        "b.py:20: in test_same\n"
        "E       AssertionError: assert kv_cache_len == 128\n"
        "=========================== short test summary info ============================\n"
        "FAILED a.py::test_same - OSError\n"
        "FAILED b.py::test_same - AssertionError\n"
        "1 passed, 2 failed in 1.00s\n"
    )
    parsed = rr.parse_pytest_log(str(log))
    by_id = {f["node_id"]: f for f in parsed["failures"]}

    # Each record carries only its own traceback.
    assert "pidfd_getfd" in by_id["a.py::test_same"]["text"]
    assert "AssertionError" not in by_id["a.py::test_same"]["text"]
    assert "AssertionError" in by_id["b.py::test_same"]["text"]
    assert "pidfd_getfd" not in by_id["b.py::test_same"]["text"]

    report = rr.build_report(
        baseline=dict(
            _BASELINE,
            expected_environment_failures=["a.py::test_same", "b.py::test_same"],
        ),
        registered_log=parsed,
        required_records=[_junit("tests.dsv4::test_sm90_thing"), _junit(_BLACKWELL_OK)],
        required_command="pytest -k ...",
        device_report=_DEVICES,
    )
    # The assertion stays genuine and the report fails on it.
    assert report["failures"]["genuine"] == ["b.py::test_same"]
    assert report["failures"]["by_signature"]["container_lacks_cap_sys_ptrace"] == [
        "a.py::test_same"
    ]
    assert not report["passed"]
    assert any("match no registered environment signature" in p for p in report["problems"])


def test_same_named_tests_are_disambiguated_by_the_file_in_their_traceback(tmp_path):
    """Order is not relied on: the file pytest prints in every frame decides.

    Here the summary lists `b.py` first while the blocks are emitted `a.py`
    first, so a positional match would hand each the other's text.
    """
    log = tmp_path / "run.log"
    log.write_text(
        "=================================== FAILURES ===================================\n"
        "___________________________________ test_same __________________________________\n"
        "a.py:10: in test_same\n"
        "E       OSError: Operation not permitted pidfd_getfd --cap-add=SYS_PTRACE\n"
        "___________________________________ test_same __________________________________\n"
        "b.py:20: in test_same\n"
        "E       AssertionError: assert 1 == 2\n"
        "=========================== short test summary info ============================\n"
        "FAILED b.py::test_same - AssertionError\n"
        "FAILED a.py::test_same - OSError\n"
        "1 passed, 2 failed in 1.00s\n"
    )
    by_id = {f["node_id"]: f for f in rr.parse_pytest_log(str(log))["failures"]}
    assert "AssertionError" in by_id["b.py::test_same"]["text"]
    assert "pidfd_getfd" in by_id["a.py::test_same"]["text"]
    # `b.py` was the ambiguous one and the file settled it; `a.py` was then the
    # only block left, so the banner alone was enough.
    assert by_id["b.py::test_same"]["traceback_source"] == "banner+file"
    assert by_id["a.py::test_same"]["traceback_source"] == "banner"


def test_an_ambiguous_traceback_is_refused_rather_than_guessed(tmp_path):
    """When the file cannot separate them, hand over nothing and say so.

    Two same-named tests whose tracebacks both name the same file cannot be
    told apart. Handing either one over is how a real bug inherits an
    environment signature, so the report reports no evidence and fails.
    """
    log = tmp_path / "run.log"
    log.write_text(
        "=================================== FAILURES ===================================\n"
        "___________________________________ test_same __________________________________\n"
        "E       OSError: Operation not permitted pidfd_getfd --cap-add=SYS_PTRACE\n"
        "___________________________________ test_same __________________________________\n"
        "E       AssertionError: assert 1 == 2\n"
        "=========================== short test summary info ============================\n"
        "FAILED a.py::test_same - OSError\n"
        "FAILED a.py::test_same - AssertionError\n"
        "1 passed, 2 failed in 1.00s\n"
    )
    parsed = rr.parse_pytest_log(str(log))
    # The first claims its block by order; the second finds none left that the
    # file can single out, so it is refused rather than given the survivor.
    sources = [f["traceback_source"] for f in parsed["failures"]]
    assert "unresolved" in sources

    report = rr.build_report(
        baseline=dict(_BASELINE, expected_environment_failures=["a.py::test_same"]),
        registered_log=parsed,
        required_records=[_junit("tests.dsv4::test_sm90_thing"), _junit(_BLACKWELL_OK)],
        required_command="pytest -k ...",
        device_report=_DEVICES,
    )
    assert not report["passed"]
    assert any("no traceback of their own" in p for p in report["problems"])


def test_a_later_report_section_does_not_leak_into_the_last_traceback(tmp_path):
    """`--durations=0` is in this repo's addopts and prints right after FAILURES.

    Closing a traceback section only on a named banner would append the whole
    durations report to the last failure, which is harmless until the day a
    genuine failure sits last and inherits a neighbour's text.
    """
    log = tmp_path / "run.log"
    log.write_text(
        "=================================== FAILURES ===================================\n"
        "_________________________________ test_pool __________________________________\n"
        "E       OSError: Operation not permitted pidfd_getfd --cap-add=SYS_PTRACE\n"
        "============================== slowest durations ===============================\n"
        "0.31s call     something::test_other\n"
        "=========================== short test summary info ============================\n"
        "FAILED a.py::test_pool - OSError\n"
        "1 passed, 1 failed in 5.00s\n"
    )
    failure = rr.parse_pytest_log(str(log))["failures"][0]
    assert "pidfd_getfd" in failure["text"]
    assert "slowest durations" not in failure["text"]
    assert "0.31s" not in failure["text"]


def test_the_log_parser_ignores_prose_that_looks_like_a_summary(tmp_path):
    """`in <n>s` anchors the real summary against a traceback that quotes counts."""
    log = tmp_path / "run.log"
    log.write_text("assert '3 passed' in output\n2 passed, 1 failed in 1.20s\n")
    assert rr.parse_pytest_log(str(log))["counts"] == {"passed": 2, "failed": 1}


def test_the_junit_parser_keeps_the_skip_reason(tmp_path):
    xml = tmp_path / "report.xml"
    xml.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest">'
        '<testcase classname="tests.a" name="test_ok" file="tests/a.py"/>'
        '<testcase classname="tests.a" name="test_skip">'
        '<skipped message="needs a GPU">why</skipped></testcase>'
        "</testsuite></testsuites>"
    )
    records = {r["node_id"]: r for r in rr.parse_junit(str(xml))}
    assert records["tests.a::test_ok"]["outcome"] == "passed"
    assert records["tests.a::test_ok"]["file"] == "tests/a.py"
    assert records["tests.a::test_skip"]["message"] == "needs a GPU"


def test_the_checked_in_baseline_is_well_formed():
    """The registered baseline has to be usable by the code that reads it."""
    baseline = rr.load_baseline()
    assert baseline["_sha256"]
    for signature in baseline["environment_failure_signatures"]:
        assert signature["id"] and signature["all_of"] and signature["explanation"]
    for signature in baseline["expected_skip_signatures"]:
        assert signature["id"] and signature["all_of"] and signature["explanation"]
    rule = baseline["required_h100_rule"]
    assert rule["node_id_any_of"] and rule["blackwell_any_of"]
    for excluded in rule["excluded_with_evidence"]:
        # An exclusion without evidence is a waiver wearing a schema.
        assert excluded["node_id"] and excluded["reason"] and excluded["evidence"]
    assert baseline["expected_environment_failures"] == sorted(
        set(baseline["expected_environment_failures"])
    )
    assert baseline["registered_command"] and baseline["required_h100_command"]
