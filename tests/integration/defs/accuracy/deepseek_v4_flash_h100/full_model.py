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
"""TP8/EP8 construction of DeepSeek-V4-Flash through the LLM API on SM90.

Every earlier suite builds the model itself: it constructs a ``ModelConfig``,
runs the weight loader, and drives modules directly. That is the right shape
for measuring one contract at a time, and it is *not* what the task's first
completion criterion asks for --- which is that the unmodified checkpoint loads
through the public ``LLM`` entry point with ``backend=pytorch``,
``tensor_parallel_size=8``, ``moe_expert_parallel_size=8`` and
``custom_tokenizer=deepseek_v4``, without a pre-Blackwell rejection, an
SM100-only kernel selection, a rank hang, or an OOM.

Two structural facts shape this module.

**The LLM API owns its own process world.** ``LLM(tensor_parallel_size=8)``
spawns eight MPI workers; the acceptance command launches this driver under
``torchrun --standalone --nproc-per-node=8``. Eight torchrun ranks each
spawning eight workers would be sixty-four processes on eight GPUs, so the
construction runs on the launcher's rank 0 alone, after the other ranks have
released their CUDA contexts and exited. :func:`detach_from_launcher` is what
makes that safe: ``MPI_Comm_spawn`` hands the workers a *snapshot* of this
process's environment taken when MPI initialised, so without it all eight
workers would inherit ``RANK=0``/``WORLD_SIZE=8``/``MASTER_PORT`` from torchrun
and any ``init_process_group()`` among them would deadlock with eight processes
claiming rank 0. Measured, not assumed: with the variables removed before the
first ``tensorrt_llm`` import the spawned workers see none of them.

**Worker-side truth arrives as log output.** The MPI proxy refuses
``collective_rpc`` for ``model_world_size > 1``, so the resolved backend,
resolved cache manager and per-rank memory numbers cannot be fetched back as
objects. They are printed, though, and spawned workers inherit the launcher's
stdout and stderr --- so redirecting those to a file *before* MPI initialises
captures all eight workers' logs, and :func:`scan_worker_log` turns them into
verdicts. Degradation shows up there as a fallback line, a Blackwell-only
kernel selection, or a rank that never reported at all.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Iterable

#: Variables ``torchrun`` exports to identify a rank. They describe *this*
#: process's place in the launcher's world and are meaningless --- actively
#: harmful --- inside a worker the LLM API spawns for its own world.
LAUNCHER_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_USE_AGENT_STORE",
)

#: Log fragments that mean a required path was *not* taken. Matched
#: case-insensitively against the captured worker log.
#:
#: Deliberately narrow, because a healthy SM90 run says plenty that *looks*
#: like degradation and is not: "TileIR requires compute capability 10.0 or
#: higher" is a Blackwell-only kernel family correctly declining to load, and
#: the AutoTuner's "using the fallback tactic" is a cache miss, not a backend
#: change. Both appeared in the first passing run. A real pre-Blackwell
#: rejection or an SM100-only kernel selection cannot be silent: it either
#: raises, which fails construction outright, or it is logged at error level,
#: which ``worker_error`` catches. Zero error-level lines appear in a healthy
#: run, which is what makes that a usable rule.
DEGRADATION_PATTERNS = (
    ("worker_error", r"\[TRT-LLM\] \[E\]"),
    ("bare_traceback", r"^Traceback \(most recent call last\)"),
    ("attention_backend_fallback", r"fall(?:ing|s|en)? back to (?:vanilla|dense|flashinfer)"),
    ("kv_cache_manager_downgrade", r"falling back to kvcachemanager\b|kv cache manager v1"),
    ("moe_backend_fallback", r"fall(?:ing|s|en)? back to \w*moe"),
    ("cuda_oom", r"cuda out of memory|out of memory error"),
    ("mpi_abort", r"mpi_abort"),
    ("nan_or_inf", r"\bnan\b detected|inf detected"),
)

#: Lines worth carrying into the artifact without failing on them. The indexer
#: dtype is the one that matters: DeepSeek-V4 defaults it to FP4, SM90 has no
#: FP4 indexer, and the FP8 route is the one Stage 1 validated against source
#: parity. Recording it keeps that substitution visible in the runtime evidence
#: rather than only in a design document.
NOTABLE_PATTERNS = (
    ("indexer_dtype_on_sm90", r"indexer_k_dtype[^\n]*sm90"),
    ("attention_runtime_features", r"ATTENTION RUNTIME FEATURES"),
)

#: Log fragments every worker rank must emit, keyed by what they prove. Each is
#: a line TensorRT-LLM already writes from inside a worker while building the
#: executor, and each names a component this bring-up could otherwise lose
#: silently: the executor reaching attention setup at all, the DeepSeek-V4
#: cache manager rather than a generic one, its V2 base, the compressed sparse
#: pools with the checkpoint's ratios, and the packed-MXFP4 routed-expert
#: layout Goal 3.2 audited. Required *per rank*, because a degradation that
#: hits one rank is the one that produces plausible output and wrong tokens.
REQUIRED_PATTERNS = (
    ("executor_built", r"ATTENTION RUNTIME FEATURES"),
    ("deepseek_v4_cache_manager", r"DeepseekV4CacheManager role-to-pool"),
    ("kv_cache_manager_v2", r"\bKVCacheManagerV2\b"),
    ("compressed_sparse_pools", r"deepseek_role=COMPRESS, compress_ratio=128"),
    ("routed_moe_w4a16_mxfp4", r"routed MoE.*W4A16_MXFP4"),
)

#: ``[RANK n]`` prefix TensorRT-LLM's logger writes in every worker line.
RANK_TAG = re.compile(r"\[RANK (\d+)\]")


def detach_from_launcher(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Remove ``torchrun``'s rank variables and return what was removed.

    Must run *before* the first ``tensorrt_llm`` import: that import
    initialises MPI, and OpenMPI snapshots the environment its spawned
    children receive at that moment. Variables removed afterwards are still
    handed to the workers.
    """
    env = os.environ if environ is None else environ
    return {name: env.pop(name) for name in LAUNCHER_ENV_VARS if name in env}


def scan_worker_log(text: str, world_size: int) -> dict[str, Any]:
    """Turn captured worker output into a dispatch verdict.

    Reports which ranks logged, which ranks emitted each required marker, and
    which degradation markers appeared with the first offending line, so a
    failure names its evidence rather than only its rule.
    """
    lines = text.splitlines()

    def ranks_of(matched: list[str]) -> list[int]:
        found = {int(m.group(1)) for line in matched for m in [RANK_TAG.search(line)] if m}
        return sorted(found)

    def matching(pattern: str) -> list[str]:
        rx = re.compile(pattern, re.IGNORECASE)
        return [line for line in lines if rx.search(line)]

    ranks_seen = ranks_of(lines)
    required: dict[str, Any] = {}
    problems: list[str] = []
    for name, pattern in REQUIRED_PATTERNS:
        matched = matching(pattern)
        ranks = ranks_of(matched)
        required[name] = {
            "ranks": ranks,
            "first_line": matched[0].strip()[:400] if matched else None,
        }
        if len(ranks) != world_size:
            problems.append(
                f"{name}: {len(ranks)} of {world_size} ranks logged it (saw {ranks}); "
                "a rank missing this line did not take the path it proves"
            )

    degraded = {}
    for name, pattern in DEGRADATION_PATTERNS:
        matched = matching(pattern)
        if matched:
            degraded[name] = matched[0].strip()[:400]
            problems.append(f"{name}: {degraded[name]}")

    if len(ranks_seen) != world_size:
        problems.append(
            f"{len(ranks_seen)} of {world_size} worker ranks logged at all (saw {ranks_seen}); "
            "a rank that never reports is a rank that never built its executor"
        )

    notable = {}
    for name, pattern in NOTABLE_PATTERNS:
        matched = matching(pattern)
        if matched:
            notable[name] = matched[0].strip()[:400]

    return {
        "lines": len(lines),
        "ranks_logged": ranks_seen,
        "required_markers": required,
        "degradation_markers": degraded,
        "notable": notable,
        "problems": problems,
        "passed": not problems,
    }


def llm_kwargs(args: Any) -> dict[str, Any]:
    """The exact construction contract this criterion names, as kwargs.

    Spelled once, here, so the artifact records the same object the ``LLM``
    was built from rather than a hand-copied description of it.
    """
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    return {
        "model": args.checkpoint,
        "backend": "pytorch",
        "tensor_parallel_size": 8,
        "moe_expert_parallel_size": 8,
        "custom_tokenizer": "deepseek_v4",
        "max_seq_len": args.max_seq_len,
        "max_num_tokens": args.max_num_tokens,
        "max_batch_size": args.max_batch_size,
        "attn_backend": "TRTLLM",
        # The parity gate compares logits, not only tokens, and the runtime
        # only returns them when it was built to gather them.
        "gather_generation_logits": True,
        # The SM90 packed-MXFP4 route Goal 3.2 audited. Pinned rather than
        # left at "AUTO": a resolver that picked another backend would still
        # produce plausible text, and this suite is the one place that can
        # still see the difference cheaply.
        "moe_config": MoeConfig(backend="CUTLASS"),
        "kv_cache_config": KvCacheConfig(
            free_gpu_memory_fraction=args.kv_fraction,
            tokens_per_block=128,
            # DeepseekV4CacheManager *is* a KVCacheManagerV2 subclass. Asking
            # for it explicitly turns "V2 was resolved" from an inference
            # about the model defaults into a requirement the runtime has to
            # satisfy or refuse.
            use_kv_cache_manager_v2=True,
            enable_block_reuse=False,
        ),
        # The eager baseline: no graph capture, no overlap. Stage 4 is what
        # turns these on, against this run's tokens.
        "cuda_graph_config": None,
        "disable_overlap_scheduler": True,
        "enable_attention_dp": False,
        "enable_chunked_prefill": False,
    }


def _describe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe echo of the construction kwargs."""
    described: dict[str, Any] = {}
    for key, value in kwargs.items():
        dump = getattr(value, "model_dump", None)
        if dump is not None:
            described[key] = {k: v for k, v in dump().items() if v is not None}
        else:
            described[key] = value
    return described


def _resolved_contract(llm: Any) -> dict[str, Any]:
    """What the constructed ``LLM`` resolved its arguments to."""
    a = llm.args
    kv = a.kv_cache_config
    return {
        "backend": getattr(a, "backend", None),
        "tensor_parallel_size": a.tensor_parallel_size,
        "moe_expert_parallel_size": a.moe_expert_parallel_size,
        "world_size": a.parallel_config.world_size,
        "custom_tokenizer": a.custom_tokenizer,
        "tokenizer_class": type(getattr(llm, "tokenizer", None)).__name__,
        "attn_backend": a.attn_backend,
        "moe_backend": a.moe_config.backend,
        "max_seq_len": a.max_seq_len,
        "max_num_tokens": a.max_num_tokens,
        "max_batch_size": a.max_batch_size,
        "tokens_per_block": kv.tokens_per_block,
        "free_gpu_memory_fraction": kv.free_gpu_memory_fraction,
        "use_kv_cache_manager_v2": kv.use_kv_cache_manager_v2,
        "enable_block_reuse": kv.enable_block_reuse,
        "cuda_graph_config": a.cuda_graph_config,
        "disable_overlap_scheduler": a.disable_overlap_scheduler,
        "enable_attention_dp": a.enable_attention_dp,
        "enable_chunked_prefill": a.enable_chunked_prefill,
        "speculative_config": a.speculative_config,
    }


def _contract_problems(resolved: dict[str, Any], args: Any) -> list[str]:
    """Every way the resolved contract can differ from the required one.

    Collected rather than asserted one at a time: a construction that took
    eight ranks and several minutes should report all of its disagreements in
    one pass.
    """
    expected = {
        "backend": "pytorch",
        "tensor_parallel_size": 8,
        "moe_expert_parallel_size": 8,
        "world_size": 8,
        "custom_tokenizer": "deepseek_v4",
        "tokenizer_class": "DeepseekV4Tokenizer",
        "attn_backend": "TRTLLM",
        "moe_backend": "CUTLASS",
        "max_seq_len": args.max_seq_len,
        "tokens_per_block": 128,
        "use_kv_cache_manager_v2": True,
        "cuda_graph_config": None,
        "disable_overlap_scheduler": True,
        "enable_chunked_prefill": False,
        "speculative_config": None,
    }
    problems = [
        f"{key}={resolved.get(key)!r}, required {want!r}"
        for key, want in expected.items()
        if resolved.get(key) != want
    ]
    fraction = resolved.get("free_gpu_memory_fraction")
    if not isinstance(fraction, float) or not 0.0 < fraction < 1.0:
        problems.append(
            f"free_gpu_memory_fraction={fraction!r}: bring-up requires an explicit fraction"
        )
    return problems


def _tokenizer_check(llm: Any, prompts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Does the runtime's tokenizer reproduce the registered prompt encodings?

    The prompt manifest was registered with the checkpoint's own encoder. If
    the custom tokenizer the runtime loaded disagrees with it, every later
    token-level parity claim would compare two different prompts.
    """
    tokenizer = getattr(llm, "tokenizer", None)
    checked: dict[str, Any] = {"tokenizer_class": type(tokenizer).__name__, "prompts": {}}
    problems: list[str] = []
    for prompt in prompts:
        encoded = tokenizer.encode(prompt["rendered"], add_special_tokens=False)
        matches = list(encoded) == list(prompt["token_ids"])
        checked["prompts"][prompt["id"]] = {
            "registered_tokens": len(prompt["token_ids"]),
            "encoded_tokens": len(encoded),
            "identical": matches,
        }
        if not matches:
            problems.append(
                f"{prompt['id']}: the runtime tokenizer encodes the registered rendering to "
                f"{len(encoded)} tokens, the manifest registered {len(prompt['token_ids'])}"
            )
    checked["problems"] = problems
    checked["passed"] = not problems
    return checked


def _generate(
    llm: Any, prompt: dict[str, Any], max_new_tokens: int, want_logits: bool = False
) -> dict[str, Any]:
    """One deterministic greedy request, described by what came back.

    ``ignore_eos`` mirrors the reference capture: the parity gate compares a
    fixed number of steps, and several registered prompts answer in fewer than
    that. Both sides therefore keep decoding with EOS in context rather than
    one stopping and the other continuing, which would compare two different
    experiments.
    """
    from tensorrt_llm import SamplingParams

    # No seed: the decode is greedy, so a seed would only be a knob that could
    # be credited with a determinism the two repetitions are supposed to prove.
    #
    # No ``min_tokens`` either, deliberately. It looks like the natural way to
    # ask for a fixed number of steps and is not: ``min_length`` *bans* the EOS
    # token until the minimum is reached, so the two prompts whose answer ends
    # inside the window emitted a different token exactly where the source
    # emitted EOS. ``ignore_eos`` alone is the right knob --- it clears the stop
    # condition without touching the distribution --- and ``max_tokens`` then
    # fixes the step count.
    params = SamplingParams(
        max_tokens=max_new_tokens,
        ignore_eos=True,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        logprobs=1,
        return_generation_logits=want_logits,
        add_special_tokens=False,
        detokenize=True,
    )
    started = time.time()
    output = llm.generate([list(prompt["token_ids"])], sampling_params=params, use_tqdm=False)[0]
    completion = output.outputs[0]
    logprobs = _flatten_logprobs(completion.logprobs)
    run = {
        "prompt_id": prompt["id"],
        "prompt_tokens": len(prompt["token_ids"]),
        "token_ids": list(completion.token_ids),
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "logprobs": logprobs,
        "nonfinite_logprobs": sum(1 for v in logprobs if not math.isfinite(v)),
        "elapsed_s": round(time.time() - started, 2),
    }
    if want_logits:
        rows = completion.generation_logits
        if rows is None:
            raise RuntimeError(
                f"{prompt['id']}: generation logits were requested but none came back; "
                "source_logit_replay has nothing to compare"
            )
        run["_logits"] = rows.detach().float().cpu()
    return run


def _flatten_logprobs(logprobs: Any) -> list[float]:
    """Per-step log-probabilities as plain floats, whatever shape they arrive in.

    The sampler returns one mapping of ``token_id -> Logprob`` per step; older
    shapes hand back bare floats. Flattened here so the finiteness check --- the
    NaN/Inf gate the task's second completion criterion names --- does not
    depend on which of those it got.
    """
    flat: list[float] = []
    for step in logprobs or []:
        values = step.values() if isinstance(step, dict) else [step]
        for value in values:
            flat.append(float(getattr(value, "logprob", value)))
    return flat


def _decompose(got: Any, ref: Any, compare: Any) -> dict[str, Any]:
    """Say *what* a logit difference is made of, not only how large it is.

    A logit vector is only defined up to an additive constant --- softmax,
    argmax and every sampling rule are invariant to it --- so a raw
    element-wise metric can be dominated by a shift that cannot change a single
    decision. These fields separate the two: the mean offset and the spread
    around it, the same metrics recomputed on mean-centred vectors, the same
    again on log-probabilities (the decision-relevant form), and whether the
    two rank their top candidates identically.

    Diagnostics only. Nothing here relaxes a registered limit; they exist so a
    failure can be attributed instead of merely reported.
    """
    import torch

    g, r = got.float().flatten(), ref.float().flatten()
    diff = g - r
    centred = compare(g - g.mean(), r - r.mean())
    logprob = compare(torch.log_softmax(g, dim=0), torch.log_softmax(r, dim=0))
    top_got = torch.topk(g, 8)
    top_ref = torch.topk(r, 8)
    got_margin = float(top_got.values[0] - top_got.values[1])
    ref_margin = float(top_ref.values[0] - top_ref.values[1])
    return {
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "ref_mean": float(r.mean()),
        "centred_rel_max_abs": float(centred["rel_max_abs"]),
        "centred_cosine": float(centred["cosine"]),
        "logprob_max_abs": float(logprob["max_abs"]),
        "logprob_mean_abs": float(logprob["mean_abs"]),
        "logprob_cosine": float(logprob["cosine"]),
        "top8_ids_match": [int(i) for i in top_got.indices] == [int(i) for i in top_ref.indices],
        "top1_top2_margin": round(got_margin, 6),
        "source_top1_top2_margin": round(ref_margin, 6),
    }


def measure(
    llm: Any,
    prompts: list[dict[str, Any]],
    reference: dict[str, Any],
    ref_logits: dict[str, Any],
    parity_tokens: int,
    compare: Any,
) -> dict[str, Any]:
    """Generate twice per prompt and measure both halves against the source.

    One pass over the prompts produces everything both gates need: the first
    request carries the logits (step 0 is ``source_logit_replay``, every step
    is a ``generation_parity`` diagnostic) and the second exists only to prove
    the first is reproducible. Nondeterminism at TP8 is how a race in a
    collective or the cache shows up, and a single run cannot see it.

    ``compare`` is the metric function every other gate in this bring-up is
    judged by, passed in rather than reimplemented so "cosine" and
    "rel_max_abs" mean here exactly what they mean there.
    """
    import torch

    measured: dict[str, Any] = {}
    kept: dict[str, Any] = {}
    for prompt in prompts:
        pid = prompt["id"]
        first = _generate(llm, prompt, parity_tokens, want_logits=True)
        second = _generate(llm, prompt, parity_tokens)
        rows = first.pop("_logits")
        # Kept so a disagreement can be dissected afterwards from the artifact
        # alone. Re-running to inspect a difference costs eight GPUs and ten
        # minutes; the sidecar costs 16 MB per prompt.
        kept[pid] = rows.numpy()
        ref_run = (reference.get("prompts") or {}).get(pid) or {}
        ref_rows = torch.as_tensor(ref_logits[pid]) if pid in ref_logits else None

        source_tokens = list(ref_run.get("tokens") or [])
        got_tokens = list(first["token_ids"])
        divergence = next(
            (
                i
                for i in range(min(len(source_tokens), len(got_tokens)))
                if source_tokens[i] != got_tokens[i]
            ),
            None,
        )
        steps: list[dict[str, Any]] = []
        if ref_rows is not None:
            # Only steps that share a prefix are comparable: after the first
            # differing token the two models are answering different questions,
            # so metrics past it would describe nothing.
            comparable = len(got_tokens) if divergence is None else divergence + 1
            for step in range(min(comparable, rows.shape[0], ref_rows.shape[0])):
                metrics = compare(rows[step], ref_rows[step])
                metrics["step"] = step
                metrics["source_token"] = source_tokens[step] if step < len(source_tokens) else None
                metrics["trtllm_token"] = got_tokens[step] if step < len(got_tokens) else None
                metrics["argmax_match"] = int(rows[step].argmax()) == int(ref_rows[step].argmax())
                metrics.update(_decompose(rows[step], ref_rows[step], compare))
                steps.append(
                    {k: (round(v, 8) if isinstance(v, float) else v) for k, v in metrics.items()}
                )

        measured[pid] = {
            "category": prompt["category"],
            "thinking_mode": prompt["thinking_mode"],
            "prompt_tokens": len(prompt["token_ids"]),
            "source_tokens": source_tokens,
            "trtllm_tokens": got_tokens,
            "repeat_tokens": list(second["token_ids"]),
            "first_divergence": divergence,
            "source_eos_at": ref_run.get("eos_at"),
            "text": first["text"],
            "repeat_text_identical": first["text"] == second["text"],
            "finish_reason": first["finish_reason"],
            "nonfinite_logprobs": first["nonfinite_logprobs"] + second["nonfinite_logprobs"],
            "logprob_steps": len(first["logprobs"]),
            "trtllm_logits_finite": bool(torch.isfinite(rows).all()),
            "trtllm_logit_rows": list(rows.shape),
            "reference_logit_rows": None if ref_rows is None else list(ref_rows.shape),
            "steps": steps,
            "elapsed_s": [first["elapsed_s"], second["elapsed_s"]],
        }
    return {"measured": measured, "logits": kept}


#: Fields a recorded ``per_prompt`` entry must carry for its verdict to be
#: re-derivable. A prompt that was never measured has none of them, and that
#: is itself the failure -- never a silent skip.
_LOGIT_FIELDS = ("finite", "argmax_match", "cosine", "rel_max_abs")
_PARITY_FIELDS = (
    "first_divergence",
    "trtllm_tokens",
    "source_tokens",
    "repeat_identical",
    "nonempty_text",
    "nonfinite_logprobs",
    "logits_finite",
)


def logit_prompt_failures(detail: dict[str, Any], limits: dict[str, Any]) -> list[str]:
    """Registered step-0 logit rules, applied to one recorded prompt entry.

    Split out of :func:`judge_logit_replay` so the strict auditor re-derives
    the verdict through *this* function rather than a second copy of the
    rules. A copy is how a recorded ``passed=true`` starts auditing clean
    while the numbers underneath it say otherwise.
    """
    missing = [f for f in _LOGIT_FIELDS if f not in detail]
    if missing:
        return [detail.get("problem") or f"no recorded {', '.join(missing)}"]
    failures = []
    if not detail["finite"]:
        failures.append("non-finite logits")
    if not detail["argmax_match"]:
        failures.append(
            f"greedy argmax {detail.get('trtllm_token')} != source {detail.get('source_token')}"
        )
    if detail["cosine"] < limits["cosine_min"]:
        failures.append(f"cosine {detail['cosine']:.6f} < {limits['cosine_min']}")
    if detail["rel_max_abs"] > limits["rel_max_abs_max"]:
        failures.append(f"rel_max_abs {detail['rel_max_abs']:.6f} > {limits['rel_max_abs_max']}")
    return failures


def parity_prompt_failures(detail: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    """Registered per-step token rules, applied to one recorded prompt entry.

    Same contract as :func:`logit_prompt_failures`: one implementation of the
    rules, used both when measuring and when auditing what was measured.
    """
    # Key presence, not truthiness: `first_divergence=None` means "never
    # diverged" and `nonfinite_logprobs=0` means "none", and both are results
    # rather than absences. Only a field the record never carried is missing.
    missing = [f for f in _PARITY_FIELDS if f not in detail]
    if missing:
        return [detail.get("problem") or f"no recorded {', '.join(missing)}"]
    min_new = int(gate.get("min_new_tokens", 32))
    failures = []
    if detail["first_divergence"] is not None:
        where = detail["first_divergence"]
        eos_at = detail.get("source_eos_at")
        position = (
            f"{where - eos_at} steps after the source's EOS at {eos_at}"
            if eos_at is not None and where > eos_at
            else "inside the answer"
        )
        failures.append(
            f"tokens diverge at step {where} ({position}): source "
            f"{detail.get('source_token_at_divergence')}, got "
            f"{detail.get('trtllm_token_at_divergence')}"
        )
    compared = min(detail["trtllm_tokens"], detail["source_tokens"])
    if compared < min_new:
        failures.append(f"compared {compared} steps, the registered gate requires {min_new}")
    if not detail["repeat_identical"]:
        failures.append("two identical greedy requests produced different tokens")
    if not detail["nonempty_text"]:
        failures.append("generated no text")
    if detail["nonfinite_logprobs"]:
        failures.append(f"{detail['nonfinite_logprobs']} non-finite logprobs")
    if not detail["logits_finite"]:
        failures.append("non-finite generation logits")
    return failures


def judge_logit_replay(
    measured: dict[str, Any],
    limits: dict[str, Any],
    gate: dict[str, Any],
    gating_ids: list[str],
    non_gating_ids: list[str],
) -> dict[str, Any]:
    """Judge step 0 of every prompt against the source's next-token logits.

    Pure, so the failure modes -- a wrong argmax, a low cosine, a missing
    prompt, a non-finite row, too few gating prompts -- are unit-testable on
    CPU rather than only reachable through an eight-GPU run.

    Non-gating prompts are measured and reported with their true result and
    cannot make the gate pass or fail. The registered fixture declares which
    those are and why; nothing here may add to that list.
    """
    per_prompt: dict[str, Any] = {}
    problems: list[str] = []
    matched: list[str] = []
    for pid in sorted(set(gating_ids) | set(non_gating_ids)):
        gating = pid in gating_ids
        entry = measured.get(pid)
        detail: dict[str, Any] = {"gating": gating, "passed": False}
        if entry is None:
            detail["problem"] = "prompt was not measured"
        elif not entry["steps"]:
            detail["problem"] = "no comparable logit step was recorded"
        else:
            step0 = entry["steps"][0]
            detail.update(
                {
                    "cosine": step0["cosine"],
                    "max_abs": step0["max_abs"],
                    "mean_abs": step0["mean_abs"],
                    "rel_max_abs": step0["rel_max_abs"],
                    "argmax_match": step0["argmax_match"],
                    "source_token": step0["source_token"],
                    "trtllm_token": step0["trtllm_token"],
                    "finite": step0["finite"] and entry["trtllm_logits_finite"],
                }
            )
            failures = logit_prompt_failures(detail, limits)
            detail["passed"] = not failures
            if failures:
                detail["problem"] = "; ".join(failures)
            elif gating:
                matched.append(pid)
        if gating and not detail["passed"]:
            problems.append(f"{pid}: {detail.get('problem')}")
        per_prompt[pid] = detail

    min_prompts = int(gate.get("min_prompts", 5))
    if len(matched) < min_prompts:
        problems.append(
            f"{len(matched)} gating prompts met the source logit gate, the registered "
            f"gate requires {min_prompts}"
        )
    return {
        "evidence_label": "source_logit_replay",
        "limits": limits,
        "gate": gate,
        "gating_prompt_ids": sorted(gating_ids),
        "non_gating_prompt_ids": sorted(non_gating_ids),
        "prompts_passing": sorted(matched),
        "per_prompt": per_prompt,
        "problems": problems,
        "passed": not problems,
    }


def judge_generation_parity(
    measured: dict[str, Any],
    gate: dict[str, Any],
    gating_ids: list[str],
    non_gating_ids: list[str],
) -> dict[str, Any]:
    """Judge the whole generated sequence, step by step, against the source.

    Also pure. The rules are the registered ones: exact per-step tokens for at
    least ``min_new_tokens`` steps, two identical TensorRT-LLM repetitions,
    non-empty output, finite logits -- plus the task's requirement that a
    plain-chat and a reasoning prompt both actually produced text.
    """
    min_new = int(gate.get("min_new_tokens", 32))
    min_prompts = int(gate.get("min_prompts", 5))
    per_prompt: dict[str, Any] = {}
    problems: list[str] = []
    matched: list[str] = []
    categories: set[str] = set()
    for pid in sorted(set(gating_ids) | set(non_gating_ids)):
        gating = pid in gating_ids
        entry = measured.get(pid)
        detail: dict[str, Any] = {"gating": gating, "passed": False}
        if entry is None:
            detail["problem"] = "prompt was not measured"
        else:
            worst = min((s["cosine"] for s in entry["steps"]), default=None)
            detail.update(
                {
                    "category": entry["category"],
                    "thinking_mode": entry["thinking_mode"],
                    "steps_compared": len(entry["steps"]),
                    "trtllm_tokens": len(entry["trtllm_tokens"]),
                    "source_tokens": len(entry["source_tokens"]),
                    "first_divergence": entry["first_divergence"],
                    "source_eos_at": entry["source_eos_at"],
                    "repeat_identical": entry["trtllm_tokens"] == entry["repeat_tokens"],
                    "nonempty_text": bool(entry["text"].strip()),
                    "worst_step_cosine": worst,
                    "nonfinite_logprobs": entry["nonfinite_logprobs"],
                    "logits_finite": entry["trtllm_logits_finite"],
                }
            )
            where = entry["first_divergence"]
            if where is not None:
                eos_at = entry["source_eos_at"]
                # Whether the divergence is inside the answer or past its end is
                # the first thing anyone asks, and the artifact should not make
                # them derive it. It changes nothing about the verdict.
                detail["divergence_after_eos"] = eos_at is not None and where > eos_at
                detail["source_token_at_divergence"] = entry["source_tokens"][where]
                detail["trtllm_token_at_divergence"] = entry["trtllm_tokens"][where]
            failures = parity_prompt_failures(detail, gate)
            detail["passed"] = not failures
            if failures:
                detail["problem"] = "; ".join(failures)
            elif gating:
                matched.append(pid)
                categories.add(entry["category"])
        if gating and not detail["passed"]:
            problems.append(f"{pid}: {detail.get('problem')}")
        per_prompt[pid] = detail

    if len(matched) < min_prompts:
        problems.append(
            f"{len(matched)} gating prompts matched the source token for token, the "
            f"registered gate requires {min_prompts}"
        )
    # The task asks for representative plain chat *and* reasoning, so a pass
    # built entirely out of one mode would not be the thing that was asked for.
    for required in ("plain_chat", "reasoning"):
        if required not in categories:
            problems.append(f"no {required!r} prompt passed with non-empty output")
    return {
        "evidence_label": "generation_parity",
        "gate": gate,
        "min_new_tokens": min_new,
        "gating_prompt_ids": sorted(gating_ids),
        "non_gating_prompt_ids": sorted(non_gating_ids),
        "prompts_passing": sorted(matched),
        "categories_passing": sorted(categories),
        "per_prompt": per_prompt,
        "problems": problems,
        "passed": not problems,
    }


def memory_report() -> dict[str, Any]:
    """Peak host RSS of this process and the node's GPU occupancy.

    The workers are separate processes, so this process's CUDA statistics say
    nothing about them; ``nvidia-smi`` is what sees their allocations.
    """
    import resource
    import subprocess

    report: dict[str, Any] = {
        "host_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
    }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
        used = []
        for line in out.splitlines():
            index, mem_used, mem_total = (int(v) for v in line.split(","))
            used.append(
                {
                    "index": index,
                    "used_gb": round(mem_used / 1024, 2),
                    "total_gb": round(mem_total / 1024, 2),
                }
            )
        report["gpus"] = used
        report["peak_gpu_used_gb"] = max((g["used_gb"] for g in used), default=0.0)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        report["gpus"] = f"unavailable: {exc}"
    return report


def free_port() -> int:
    """A port nothing is listening on, for the nested launcher's rendezvous."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def capture_reference(args: Any, driver: str, output: str) -> dict[str, Any]:
    """Run the official-source capture in its own eight-rank job, and wait.

    Out of process by necessity, not preference: the official model needs the
    reference interpreter, that interpreter breaks the runtime under test, and
    both want the same eight GPUs. Running it here --- after the launcher's
    other ranks have exited and before this rank builds anything --- keeps the
    acceptance command a single command while keeping the two models apart.
    """
    import subprocess

    command = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={args.reference_world_size}",
        f"--rdzv-endpoint=localhost:{free_port()}",
        driver,
        "--checkpoint",
        args.checkpoint,
        "--suite",
        "source_reference",
        "--output",
        output,
        "--parity-tokens",
        str(args.parity_tokens),
        "--max-seq-len",
        str(args.max_seq_len),
    ]
    if args.prompt_ids:
        command += ["--prompt-ids", *args.prompt_ids]
    started = time.time()
    completed = subprocess.run(command, check=False)
    elapsed = round(time.time() - started, 2)
    if completed.returncode != 0:
        raise RuntimeError(
            f"official-source reference capture failed with exit code "
            f"{completed.returncode}: {' '.join(command)}"
        )
    return {"command": " ".join(command), "elapsed_s": elapsed, "artifact": output}


def run(
    args: Any,
    ranks: Any,
    prompts: list[dict[str, Any]],
    log_path: str,
    reference: dict[str, Any],
    ref_logits: dict[str, Any],
    limits: dict[str, Any],
    gates: dict[str, Any],
    gating_ids: list[str],
    non_gating_ids: list[str],
) -> dict[str, Any]:
    """Construct the model through the LLM API and measure it against the source.

    Runs on the launcher's rank 0 only, with stdout and stderr already
    redirected into ``log_path`` so the spawned workers' output is captured.
    Returns the artifact fragment; raising is reserved for a construction that
    fails outright, which the driver records as an error.
    """
    import source_reference
    import torch_goldens as tg

    from tensorrt_llm import LLM

    kwargs = llm_kwargs(args)
    checks: dict[str, Any] = {}
    result: dict[str, Any] = {
        "construction": _describe_kwargs(kwargs),
        "worker_log": log_path,
        "checks": checks,
    }

    started = time.time()
    llm = LLM(**kwargs)
    construct_s = round(time.time() - started, 2)
    try:
        resolved = _resolved_contract(llm)
        problems = _contract_problems(resolved, args)
        checks["runtime_contract"] = {
            "resolved": resolved,
            "construct_s": construct_s,
            "problems": problems,
            "passed": not problems,
        }
        checks["custom_tokenizer"] = _tokenizer_check(llm, prompts)
        run_out = measure(llm, prompts, reference, ref_logits, args.parity_tokens, tg.compare)
        measured = run_out["measured"]
        result["measured"] = measured
        result["logits_sidecar"] = source_reference.write(args.output, {}, run_out["logits"])[
            "logits_sidecar"
        ]
        checks["source_logit_replay"] = judge_logit_replay(
            measured, limits, gates["source_logit_replay"], gating_ids, non_gating_ids
        )
        checks["generation_parity"] = judge_generation_parity(
            measured, gates["generation_parity"], gating_ids, non_gating_ids
        )
        result["memory"] = memory_report()
    finally:
        llm.shutdown()

    # Read only after shutdown: the workers write to this file until they exit,
    # and a scan taken while they are still running would judge a partial log.
    with open(log_path, errors="replace") as handle:
        captured = handle.read()
    checks["worker_dispatch"] = scan_worker_log(captured, world_size=8)

    result["memory_after_shutdown"] = memory_report()
    result["passed"] = all(check["passed"] for check in checks.values())
    return result


def write_json(path: str, payload: dict[str, Any]) -> None:
    """Small helper so probes can dump a fragment without the driver."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
