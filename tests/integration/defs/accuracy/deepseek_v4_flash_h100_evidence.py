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
"""8-GPU evidence driver for the DeepSeek-V4-Flash Hopper (SM90) bring-up.

Run one suite per invocation under ``torchrun --standalone --nproc-per-node=8``
and it writes a single JSON artifact carrying the evidence label, the exact
configuration, the measured metrics and the exit status.

Implemented suites:

``reference_ladder``
    Stage 1 / Goal 1.2. Builds the trusted reference tier that every later
    parity claim is judged against: it runs the checkpoint's *official*
    ``inference/model.py`` on all eight GPUs, drives the official hand-written
    greedy loop over the pre-registered prompts, and checks independent
    pure-Torch module goldens against the activations that real run produces.

The official source model needs ``tilelang`` and ``fast_hadamard_transform``,
which deliberately do **not** live in the container's pinned environment ---
installing them there would risk the runtime the bring-up itself depends on.
They live in an isolated reference venv instead, and this script re-executes
itself under that interpreter when it needs them (see :func:`_reference_env`).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from typing import Any

REPO_EVIDENCE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORT_PKG = os.path.join(REPO_EVIDENCE_DIR, "deepseek_v4_flash_h100")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(REPO_EVIDENCE_DIR))))
# The repo's own venv, which is where the linked `tensorrt_llm` lives. Derived
# from this file's location rather than hard-coded so it follows the checkout.
TRTLLM_ENV_PYTHON = os.path.join(REPO_ROOT, ".venv-3.12", "bin", "python3")

# Isolated interpreter carrying the official source model's dependencies.
REFERENCE_ENV_PYTHON = "/cache/dsv4_flash_h100/refenv/bin/python3"
# Official MP-sharded checkpoint produced by the checkpoint's own convert.py.
OFFICIAL_MP_DIR = "/cache/dsv4_flash_h100/official_mp8"
TILELANG_CACHE = "/cache/dsv4_flash_h100/tilelang_cache"


#: Result of :func:`_preload_nvrtc`, recorded in every reference-tier artifact.
_NVRTC_PRELOAD: dict[str, Any] = {"attempted": False}


def _preload_nvrtc() -> dict[str, Any]:
    """Load the real ``libnvrtc`` globally before ``tilelang`` loads its stub.

    ``tilelang`` installs ``libnvrtc_stub.so`` into the global symbol namespace
    and forwards to whatever NVRTC it can find; if it finds none it calls
    ``abort()``. TensorRT-LLM's DeepGEMM compiler is an NVRTC client too --- the
    FP8 block-scale Hopper GEMM behind every dense and shared-expert ``Linear``
    JIT-compiles through ``deep_gemm::jit::Compiler::build`` --- so in the
    reference interpreter its first call lands in tilelang's stub and takes the
    whole rank down with SIGABRT, several minutes into an eight-rank run.

    Loading the real library ``RTLD_GLOBAL`` first makes the symbols the stub
    looks for actually present. Verified both ways: without this the shared
    expert aborts at ``nvrtcCreateProgram``; with it the same GEMM runs.
    """
    import ctypes

    for name in ("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"):
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return {"attempted": True, "loaded": name}
    return {
        "attempted": True,
        "loaded": None,
        "why_it_matters": (
            "tilelang's libnvrtc stub aborts the process when TensorRT-LLM's DeepGEMM "
            "JIT compiler calls nvrtcCreateProgram and no real NVRTC is global"
        ),
    }


def _reference_env() -> None:
    """Re-exec under the isolated reference interpreter when needed.

    Keeps the acceptance command spellable as a plain ``torchrun ...`` while
    still honouring the reference-test policy: the container's pinned
    ``transformers`` is never touched, and the official model's heavyweight
    kernel dependencies stay in a throwaway environment.
    """
    global _NVRTC_PRELOAD
    _NVRTC_PRELOAD = _preload_nvrtc()
    try:
        import tilelang  # noqa: F401

        return
    except ImportError:
        pass
    if not os.path.exists(REFERENCE_ENV_PYTHON):
        raise RuntimeError(
            f"the official source model needs tilelang, and the reference venv "
            f"{REFERENCE_ENV_PYTHON} does not exist; create it by running "
            f"build_reference_env.sh from the bring-up harness directory"
        )
    if os.environ.get("_DSV4_REEXEC") == "1":
        raise RuntimeError(f"{REFERENCE_ENV_PYTHON} still cannot import tilelang")
    os.environ["_DSV4_REEXEC"] = "1"
    os.environ.setdefault("TILELANG_CACHE_DIR", TILELANG_CACHE)
    os.execv(REFERENCE_ENV_PYTHON, [REFERENCE_ENV_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])


def _trtllm_env() -> None:
    """Re-exec under the repo venv when the launcher's interpreter cannot import it.

    The acceptance commands are spelled ``trtllm_dev.sh torchrun ... <this
    file>``, and ``/usr/local/bin/torchrun``'s shebang is the *system*
    ``/usr/bin/python3``, which has no ``tensorrt_llm`` at all. Suites that
    exercise TensorRT-LLM therefore have to reach the venv the container
    activates for a plain ``python3``, exactly as :func:`_reference_env` reaches
    the isolated source environment.

    The probe is a module *lookup*, deliberately not an import: importing
    ``tensorrt_llm`` initialises MPI, and OpenMPI captures the descriptors its
    spawned workers write to at that moment. Importing here would fix them to
    the launcher's console before ``eager_full_model`` can redirect them into
    its log file --- measured, by losing an entire eight-worker log that way.
    """
    import importlib.util

    if importlib.util.find_spec("tensorrt_llm") is not None:
        return
    if os.environ.get("_DSV4_TRTLLM_REEXEC") == "1":
        raise RuntimeError(f"{TRTLLM_ENV_PYTHON} still cannot import tensorrt_llm")
    if not os.path.exists(TRTLLM_ENV_PYTHON):
        raise RuntimeError(
            f"this interpreter ({sys.executable}) cannot import tensorrt_llm and "
            f"{TRTLLM_ENV_PYTHON} does not exist"
        )
    os.environ["_DSV4_TRTLLM_REEXEC"] = "1"
    os.execv(TRTLLM_ENV_PYTHON, [TRTLLM_ENV_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])


# ---------------------------------------------------------------------------
# Distributed helpers.
# ---------------------------------------------------------------------------


class Ranks:
    """Torchrun rank bookkeeping plus rank-0-only printing."""

    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def log(self, *a: Any) -> None:
        if self.is_main:
            print(*a, flush=True)


def _init_distributed(ranks: Ranks) -> None:
    import torch
    import torch.distributed as dist

    if ranks.world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(ranks.local)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    # Honour the launcher's thread budget instead of overriding it. `torchrun`
    # sets OMP_NUM_THREADS=1 per process, so raising torch's intra-op count to 8
    # afterwards oversubscribes the host 64-fold across the eight ranks *and*
    # grows the OpenMP team after libgomp has already initialised. Under that
    # combination this suite segfaulted intermittently inside
    # `create_sinusoidal_positions_yarn` -- a CPU `torch.cos` in shared code,
    # reached from `TrtllmAttention.__init__` -- on one arbitrary rank per run.
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    _evaluate_float32_as_float32()


def _evaluate_float32_as_float32() -> None:
    """Make ``.float()`` in the reference mean float32, not TF32.

    ``inference/model.py`` upcasts deliberately --- ``hc_pre`` runs the mix GEMM
    on ``x.flatten(2).float()``, ``Gate`` on ``x.float()``, ``Expert`` "in
    float32 for stability". Every one of those is an FP32 ``F.linear``, and on
    stock PyTorch (``allow_tf32`` defaults to False) that is what it computes.
    This container's PyTorch build ships the opposite default
    (``fp32_precision='tf32'``), so the reference silently evaluated those
    GEMMs on TF32 tensor cores --- 10 mantissa bits, two fewer than BF16 has
    after the implicit one.

    Measured on layer 2 of the real checkpoint, 257 tokens, against a float64
    evaluation of the same expression: the source's ``hc_pre`` mixes scored
    ``rel_max_abs`` 9.57e-05 under the container default and 1.07e-06 with it
    off, while TensorRT-LLM's FP32 FMA kernel scored 7.14e-07 either way. The
    oracle was 134x further from its own exact value than the implementation it
    was judging, and the resulting disagreement --- 1.71e-01 on ``layer_input``
    --- was the reference's error, not TensorRT-LLM's.

    Applied process-wide rather than around the source alone: a parity harness
    that measured the two sides at different precisions would be measuring the
    harness. Recorded in every artifact's environment block.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _float32_precision_report() -> dict[str, Any]:
    """The FP32 matmul precision every measurement in this artifact ran at."""
    import torch

    return {
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "why": (
            "the checkpoint's explicit .float() upcasts (hc_pre mixes, Gate, Expert) are "
            "FP32 F.linear calls; this container's PyTorch defaults them to TF32, which "
            "degrades the reference below the implementation it judges"
        ),
    }


def _reference_env_report() -> dict[str, Any]:
    """Record exactly which reference-tier dependencies produced this artifact.

    The reference ladder is only as reproducible as the environment it runs in.
    ``fast_hadamard_transform`` in particular has no usable PyPI artifact and has
    to be built from source, so its Git commit --- not just its version --- is
    part of the evidence.
    """
    import subprocess

    report: dict[str, Any] = {"interpreter": sys.executable}
    for module in ("tilelang", "transformers", "fast_hadamard_transform", "safetensors"):
        try:
            report[module] = __import__(module).__version__
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            report[module] = f"unavailable: {exc}"
    repo = "/cache/dsv4_flash_h100/fast-hadamard-transform"
    try:
        report["fast_hadamard_transform_commit"] = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        report["fast_hadamard_transform_commit"] = f"unavailable: {exc}"
    report["float32_precision"] = _float32_precision_report()
    report["nvrtc_preload"] = _NVRTC_PRELOAD
    return report


def _linked_package_report() -> dict[str, Any]:
    """Prove the ``tensorrt_llm`` under test is the linked source checkout.

    The reference interpreter reaches TensorRT-LLM through a ``.pth`` rather
    than an install, so "which package did this measure" is a real question. A
    suite that measured a site-packages wheel while claiming to measure the
    working tree would be evidence about the wrong code.
    """
    import tensorrt_llm

    path = os.path.abspath(tensorrt_llm.__file__)
    linked = os.path.join(REPO_ROOT, "tensorrt_llm") + os.sep
    if not path.startswith(linked):
        raise RuntimeError(
            f"tensorrt_llm resolves to {path}, not the linked checkout under {linked}; "
            "this run would measure a different package than the one being changed"
        )
    return {"tensorrt_llm": path, "repo_root": REPO_ROOT, "version": tensorrt_llm.__version__}


def _device_report() -> dict[str, Any]:
    import torch

    props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    return {
        "device_count": torch.cuda.device_count(),
        "names": sorted({p.name for p in props}),
        "compute_capability": sorted({f"{p.major}.{p.minor}" for p in props}),
        "total_memory_gb": round(props[0].total_memory / 1e9, 2) if props else 0.0,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


# ---------------------------------------------------------------------------
# Official source model.
# ---------------------------------------------------------------------------


class OfficialSource:
    """The checkpoint's own ``inference/model.py``, loaded MP-sharded on 8 GPUs.

    This is the top of the reference ladder: the semantics the TensorRT-LLM
    implementation must reproduce. Nothing here is re-derived --- the module
    imports and runs the checkpoint's shipped code verbatim.
    """

    def __init__(
        self, checkpoint: str, ranks: Ranks, *, max_seq_len: int = 4096, max_batch: int = 1
    ):
        import torch
        from safetensors.torch import load_model

        self.ranks = ranks
        self.checkpoint = checkpoint
        for sub in ("inference", "encoding"):
            path = os.path.join(checkpoint, sub)
            if path not in sys.path:
                sys.path.insert(0, path)

        from model import ModelArgs, Transformer  # official source, not a re-implementation

        with open(os.path.join(checkpoint, "inference", "config.json")) as f:
            cfg = json.load(f)
        self.args = ModelArgs(**cfg, max_batch_size=max_batch, max_seq_len=max_seq_len)

        t0 = time.time()
        with torch.device("cuda"):
            self.model = Transformer(self.args)
        self.construct_s = time.time() - t0

        shard = os.path.join(OFFICIAL_MP_DIR, f"model{ranks.rank}-mp{ranks.world}.safetensors")
        if not os.path.exists(shard):
            raise RuntimeError(
                f"missing official shard {shard}; produce it with the checkpoint's own "
                f"inference/convert.py at --model-parallel {ranks.world}"
            )
        t0 = time.time()
        missing, unexpected = load_model(self.model, shard, strict=False)
        torch.cuda.synchronize()
        self.load_s = time.time() - t0
        self.missing = sorted(missing)
        self.unexpected = sorted(unexpected)
        if self.missing or self.unexpected:
            raise RuntimeError(
                f"rank {ranks.rank} state-dict mismatch: {len(self.missing)} missing "
                f"{self.missing[:4]}, {len(self.unexpected)} unexpected {self.unexpected[:4]}"
            )
        self.alloc_gb = torch.cuda.memory_allocated() / 1e9
        # The official generate.py switches the default device after loading;
        # the model's index-table helpers (get_window_topk_idxs and friends)
        # allocate with bare torch.arange and land on CPU otherwise.
        torch.set_default_device("cuda")

    def reset_cache(self) -> None:
        """Clear every KV / compressor buffer so prompts cannot leak into each other."""
        import torch

        for layer in self.model.layers:
            attn = layer.attn
            attn.kv_cache.zero_()
            if attn.compress_ratio:
                attn.compressor.kv_state.zero_()
                attn.compressor.score_state.fill_(float("-inf"))
                if attn.indexer is not None:
                    attn.indexer.kv_cache.zero_()
                    attn.indexer.compressor.kv_state.zero_()
                    attn.indexer.compressor.score_state.fill_(float("-inf"))
        torch.cuda.synchronize()

    def greedy(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        *,
        stop_at_eos: bool = True,
        capture_logits: bool = False,
    ) -> dict[str, Any]:
        """The official hand-written prefill + greedy decode loop.

        This mirrors ``inference/generate.py`` with ``temperature == 0``: one
        prefill over the whole prompt, then one token at a time, taking the
        argmax and stopping at EOS. It is exactly the loop the golden-generate
        cross-check exists to anchor.

        ``stop_at_eos=False`` keeps decoding past the EOS token, which the
        parity gate needs: it asks for at least 32 compared steps, and several
        registered prompts finish sooner than that. Nothing else changes ---
        same forward, same selection rule --- so the tokens before EOS are still
        the sequence the fixture anchors, and the comparison against it is what
        proves that.

        ``capture_logits=True`` additionally returns every step's full logit
        row on the CPU, which is the reference side of ``source_logit_replay``.
        Off by default because it is ~0.5 MB per step and the golden-generate
        cross-check does not need it.
        """
        import torch

        self.reset_cache()
        eos = self.args.vocab_size and 1  # checkpoint eos_token_id
        toks = torch.tensor([token_ids], dtype=torch.long, device="cuda")

        t0 = time.time()
        logits = self.model.forward(toks, 0)
        torch.cuda.synchronize()
        prefill_s = time.time() - t0

        def select(logits: Any) -> int:
            """Token selection, byte for byte what ``inference/generate.py`` does.

            Nothing else may pick the token. ``topk(2).indices[0]`` looks
            equivalent and is not: on CUDA the two order equal values
            differently, so substituting it silently replaces the source's
            decoding rule with a lookalike.
            """
            return int(logits.argmax(dim=-1)[0])

        def diagnose(logits: Any) -> tuple[list[int], float]:
            """Top-2 ids and their gap. Diagnostics only --- never selection."""
            top = logits[0].float().topk(2)
            return [int(i) for i in top.indices], float(top.values[0] - top.values[1])

        first = logits.float()
        out: list[int] = []
        nxt = select(first)
        cand, margin = diagnose(first)
        step_max = [float(first.max())]
        margins, candidates = [round(margin, 6)], [cand]
        captured = [first[0].cpu()] if capture_logits else []
        eos_at: int | None = None
        pos = len(token_ids)
        t0 = time.time()
        while True:
            out.append(nxt)
            if nxt == eos and eos_at is None:
                eos_at = len(out) - 1
            if (nxt == eos and stop_at_eos) or len(out) >= max_new_tokens:
                break
            cur = torch.tensor([[nxt]], dtype=torch.long, device="cuda")
            lg = self.model.forward(cur, pos).float()
            step_max.append(float(lg.max()))
            if capture_logits:
                captured.append(lg[0].cpu())
            nxt = select(lg)
            cand, margin = diagnose(lg)
            margins.append(round(margin, 6))
            candidates.append(cand)
            pos += 1
        torch.cuda.synchronize()

        run: dict[str, Any] = {
            "tokens": out,
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(time.time() - t0, 3),
            "prefill_logits_finite": bool(torch.isfinite(first).all()),
            "prefill_logit_max": float(first.max()),
            "prefill_logit_min": float(first.min()),
            "step_logit_max": [round(v, 4) for v in step_max[:8]],
            "top1_top2_margin": margins[: len(out)],
            "top2_candidates": candidates[: len(out)],
            "min_top1_top2_margin": round(min(margins[: len(out)]), 6) if out else None,
            "stopped_on_eos": bool(out and out[-1] == eos and stop_at_eos),
            "eos_at": eos_at,
        }
        if capture_logits:
            # One row per emitted token: row i is the distribution the token at
            # index i was selected from, so a comparison is always against the
            # step that produced the token it is judged with.
            run["logits"] = torch.stack(captured[: len(out)])
        return run


# ---------------------------------------------------------------------------
# Manifests.
# ---------------------------------------------------------------------------


MANIFEST_DIR = os.path.join(SUPPORT_PKG, "manifests")
#: Every registered manifest, all covered by ``MANIFEST.sha256``.
#:
#: ``native_generate_golden.json`` joined the set for Stage 3: it declares which
#: prompts gate the reference ladder and which are recorded as non-gating, so
#: leaving it unregistered would have let a prompt be reclassified without any
#: hash moving. ``tolerances.superseded.json`` joined so the limits Stage 2
#: replaced sit beside the ones that replaced them and cannot themselves drift.
#: ``regression_baseline.json`` joined for Stage 3 Goal 3.5: it names the
#: failures attributed to this container's missing ``CAP_SYS_PTRACE``, and an
#: unregistered edit to that list would silently absorb a real regression.
MANIFEST_FILES = (
    "prompts.json",
    "tolerances.json",
    "tolerances.superseded.json",
    "native_generate_golden.json",
    "regression_baseline.json",
)


def _load_manifest(name: str) -> dict[str, Any]:
    with open(os.path.join(MANIFEST_DIR, name)) as f:
        return json.load(f)


def _manifest_provenance(started_at: str) -> dict[str, Any]:
    """Verify the registered manifests, then report what is registered and when.

    A pre-registered tolerance is only worth anything if the registration
    provably preceded the measurement, so this runs before a suite loads the
    checkpoint and refuses to continue otherwise. Three things are checked:

    * each manifest still hashes to what ``MANIFEST.sha256`` registered, so a
      limit cannot be edited to fit a number that has already been measured;
    * a manifest that supersedes an earlier one names the hash it replaced, so
      the change is a documented re-registration rather than a silent edit;
    * the recorded registration timestamp is strictly before this run started.

    The returned block carries both hashes into the artifact, which is what
    lets a reader confirm the ordering from the evidence alone.
    """
    registered: dict[str, str] = {}
    with open(os.path.join(MANIFEST_DIR, "MANIFEST.sha256")) as f:
        for line in f:
            if line.strip():
                digest, name = line.split()
                registered[name] = digest

    hashes: dict[str, str] = {}
    for name in MANIFEST_FILES:
        actual = _sha256_file(os.path.join(MANIFEST_DIR, name))
        if registered.get(name) != actual:
            raise RuntimeError(
                f"{name} hashes to {actual} but MANIFEST.sha256 registered "
                f"{registered.get(name)}; a manifest that changed after registration "
                "cannot gate a measurement"
            )
        hashes[name] = actual

    # A JSON file sitting in the manifest directory but outside the checksum set
    # would be a manifest nobody registered, which is the hole the native-generate
    # fixture used to sit in: it declared which prompts gate and which do not, and
    # could have been reclassified without any hash moving.
    on_disk = {n for n in os.listdir(MANIFEST_DIR) if n.endswith(".json")}
    unregistered = sorted(on_disk - set(MANIFEST_FILES))
    if unregistered:
        raise RuntimeError(
            f"{unregistered} live in {MANIFEST_DIR} but are not in the registered set "
            f"{list(MANIFEST_FILES)}; every manifest a gate reads must be checksummed"
        )

    tol = _load_manifest("tolerances.json")
    block: dict[str, Any] = {
        "manifest_dir": MANIFEST_DIR,
        "sha256": hashes,
        "tolerances_schema_version": tol.get("schema_version"),
        "run_started_at": started_at,
    }
    reg = tol.get("re_registration")
    if reg:
        registered_at = str(reg.get("registered_at", ""))
        if not registered_at or registered_at >= started_at:
            raise RuntimeError(
                f"tolerances.json records registered_at={registered_at!r}, which is not "
                f"strictly before this run's start {started_at!r}; a limit registered "
                "after the measurement is not a pre-registered limit"
            )
        block["re_registration"] = {
            "registered_at": registered_at,
            "supersedes_sha256": reg.get("supersedes_sha256"),
            "modules": reg.get("modules"),
            "reason": reg.get("reason"),
        }
        block["tolerances_sha256_old"] = reg.get("supersedes_sha256")
        block["tolerances_sha256_new"] = hashes["tolerances.json"]
    return block


def _tolerance(tol: dict[str, Any], module: str) -> dict[str, Any]:
    entry = tol["modules"].get(module)
    if entry is None:
        raise KeyError(f"no pre-registered tolerance for module {module!r}")
    return entry


def _judge(
    metrics: dict[str, float],
    limits: dict[str, Any],
    storage_resolution: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Check measured metrics against the *pre-registered* tolerance entry.

    The registered numbers are applied literally. Every ``<metric>_max`` key in
    the entry is enforced against the measurement of the same name, so a limit
    cannot be registered and then quietly not applied, and ``cosine_min`` is
    enforced as a floor. There is no per-module override and no way for a
    measurement above a registered limit to be reported as a pass:
    ``task.yaml`` forbids loosening tolerances or waiving failures.

    ``storage_resolution`` supplies the BF16-grid measurements
    (``abs_max_element_steps``, ``elements_beyond_one_step``,
    ``mean_abs_in_dtype_steps``) that the re-registered sparse-attention and
    sink entries are judged on. It can only *add* failures: a registered limit
    with no matching measurement is a failure rather than a skipped check, so
    forgetting to thread it through cannot turn into a silent pass.
    """
    problems = []
    if not metrics.get("finite", True):
        problems.append("non-finite values")
    if "cosine_min" in limits and metrics["cosine"] < limits["cosine_min"]:
        problems.append(f"cosine {metrics['cosine']:.6f} < {limits['cosine_min']}")
    measured = {**metrics, **(storage_resolution or {})}
    for key, limit in limits.items():
        if not key.endswith("_max") or not isinstance(limit, (int, float)):
            continue
        metric = key[: -len("_max")]
        if metric not in measured:
            problems.append(f"{metric} has a registered limit of {limit} but was not measured")
        elif measured[metric] > limit:
            problems.append(f"{metric} {measured[metric]:.6g} > {limit}")
    return not problems, problems


# ---------------------------------------------------------------------------
# Suite: reference_ladder.
# ---------------------------------------------------------------------------


def suite_reference_ladder(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    import torch

    sys.path.insert(0, SUPPORT_PKG)
    import torch_goldens as tg

    prompts_manifest = _load_manifest("prompts.json")
    tol_manifest = _load_manifest("tolerances.json")
    prompts = {p["id"]: p for p in prompts_manifest["prompts"]}
    # Before the checkpoint is even opened: the limits this suite is judged
    # against must already be registered, and must hash to what was registered.
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")

    ranks.log(f"[reference_ladder] loading official source on {ranks.world} ranks")
    src = OfficialSource(args.checkpoint, ranks, max_seq_len=args.max_seq_len)
    ranks.log(
        f"  constructed {src.construct_s:.1f}s, loaded {src.load_s:.1f}s, "
        f"{src.alloc_gb:.2f} GB/rank, state-dict exact"
    )

    result: dict[str, Any] = {
        "evidence_label": "reference_ladder",
        "reference_tier": "real_source",
        "validation_tier": "integration",
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "official_mp_dir": OFFICIAL_MP_DIR,
        "world_size": ranks.world,
        "devices": _device_report(),
        "reference_env": _reference_env_report(),
        "manifest_provenance": provenance,
        "decoding": prompts_manifest["decoding"],
        "hard_config": {
            "max_seq_len": args.max_seq_len,
            "max_batch_size": 1,
            "cuda_graph": False,
            "overlap_scheduler": False,
            "expert_dtype": src.args.expert_dtype,
            "dtype": src.args.dtype,
            "scale_dtype": src.args.scale_dtype,
        },
        "state_dict": {
            "missing": len(src.missing),
            "unexpected": len(src.unexpected),
            "alloc_gb_rank0": round(src.alloc_gb, 3),
            "construct_s": round(src.construct_s, 2),
            "load_s": round(src.load_s, 2),
        },
    }

    # -- tier 1: the official hand-written greedy loop --------------------
    gen: dict[str, Any] = {}
    for pid in args.prompt_ids or list(prompts):
        spec = prompts[pid]
        run = src.greedy(spec["token_ids"], args.max_new_tokens)
        # Repeat over the *complete* requested generation. A prefix-only repeat
        # would miss exactly the nondeterminism that shows up deep in a decode,
        # where cache reuse and compression state have had time to drift.
        repeat = src.greedy(spec["token_ids"], args.max_new_tokens)
        run["deterministic_repeat"] = run["tokens"] == repeat["tokens"]
        run["deterministic_repeat_tokens"] = len(repeat["tokens"])
        run["category"] = spec["category"]
        run["thinking_mode"] = spec["thinking_mode"]
        run["prompt_tokens"] = spec["num_tokens"]
        gen[pid] = run
        ranks.log(
            f"  {pid:24s} {spec['num_tokens']:5d} tok -> {len(run['tokens']):3d} generated, "
            f"finite={run['prefill_logits_finite']} repeat_ok={run['deterministic_repeat']} "
            f"prefill={run['prefill_s']}s"
        )
    result["official_generation"] = gen

    # -- tier 2: independent pure-Torch module goldens --------------------
    tg.assert_independent()
    # Captured activations are inference tensors, and several golden checks
    # re-run official submodules whose parameters still require grad, so the
    # whole comparison has to stay inside inference mode.
    with torch.inference_mode():
        result["module_goldens"] = _module_goldens(
            src, prompts, tol_manifest, ranks, tg, layer_ids=args.golden_layers
        )
    result["golden_layers"] = list(args.golden_layers)

    # -- tier 0: the canonical native-generate fixture --------------------
    result["native_generate_golden"] = _check_native_golden(gen, ranks)

    checks = [g["passed"] for g in result["module_goldens"].values()]
    local_passed = (
        bool(checks)
        and all(checks)
        and all(
            r["prefill_logits_finite"] and r["deterministic_repeat"] and r["tokens"]
            for r in gen.values()
        )
        and result["native_generate_golden"]["passed"]
    )
    result.update(_aggregate_ranks(local_passed, result["module_goldens"], ranks))
    tightest = sorted(
        (
            (detail["headroom_x"], check, metric, detail)
            for check, entry in result["worst_rank_metrics"].items()
            for metric, detail in entry.items()
            if isinstance(detail, dict) and isinstance(detail.get("headroom_x"), (int, float))
        )
    )[:5]
    for headroom, check, metric, detail in tightest:
        ranks.log(
            f"  tightest margin {check}.{metric} = {detail['value']:.6g} vs "
            f"{detail['limit']:.6g} on rank {detail['rank']} ({headroom:.2f}x headroom)"
        )
    return result


def _aggregate_ranks(
    local_passed: bool, module_goldens: dict[str, Any], ranks: Ranks
) -> dict[str, Any]:
    """Fold every rank's verdict into the one artifact rank 0 writes.

    The module goldens are computed against *rank-local* weights and heads, so
    they can fail on one rank and pass on another. Logging is rank-0-only, so
    without this the artifact would report ``passed=true`` from rank 0 while
    other ranks exited non-zero --- which is precisely the shape of false pass
    this evidence is supposed to make impossible. The suite now passes only if
    every rank passed, and every rank's failing checks are named in the report.

    Each rank also contributes its float metrics for *every* check, not just
    its failing ones, so the artifact can state the worst number any rank
    produced. "No rank failed" and "the worst rank cleared the limit by 20%"
    are very different claims, and only the second is checkable --- rank 0's
    own numbers say nothing about how much headroom rank 5 had.
    """
    import torch.distributed as dist

    summary = {
        "rank": ranks.rank,
        "passed": bool(local_passed),
        "failed_checks": {
            name: {
                "problems": g.get("problems"),
                "metrics": g.get("metrics"),
                # Carried so a failing rank can be diagnosed from the artifact
                # instead of needing another eight-GPU run to reproduce it.
                "storage_resolution": g.get("storage_resolution"),
                "provenance": _provenance_digest(g.get("context")),
            }
            for name, g in module_goldens.items()
            if not g["passed"]
        },
        # Provenance for *every* rank, not only failing ones. Rank 0's context
        # is the one that lands in `checks`, so without this the artifact could
        # only say where rank 0's disagreement came from --- and rank 0 is not
        # the rank that fails. Kept to a digest because it crosses an
        # all_gather_object.
        "provenance": {
            name: digest
            for name, g in module_goldens.items()
            if (digest := _provenance_digest(g.get("context"))) is not None
        },
        "metrics": {
            name: {k: v for k, v in (g.get("metrics") or {}).items() if isinstance(v, float)}
            for name, g in module_goldens.items()
        },
        "steps": {
            name: (g.get("storage_resolution") or {}).get("abs_max_element_steps")
            for name, g in module_goldens.items()
        },
        # How much of each check's disagreement is BF16 grid resolution rather
        # than arithmetic. Recorded for passing checks too: a check that passes
        # at rel_max_abs 4e-03 and one that fails at 5e-02 can be the same
        # single-grid-step difference landing on elements of different
        # magnitude, and only this tells them apart.
        "grid_agreement": {
            name: {
                k: (g.get("storage_resolution") or {}).get(k)
                for k in (
                    "elements",
                    "elements_differing",
                    "total_dtype_steps",
                    "elements_beyond_one_step",
                    "mean_abs_in_dtype_steps",
                )
            }
            for name, g in module_goldens.items()
            if g.get("storage_resolution")
        },
    }
    if ranks.world > 1 and dist.is_initialized():
        gathered: list[Any] = [None] * ranks.world
        dist.all_gather_object(gathered, summary)
    else:
        gathered = [summary]

    by_rank = {int(s["rank"]): s for s in gathered if s}
    failing = sorted(r for r, s in by_rank.items() if not s["passed"])
    if failing:
        ranks.log(f"  RANK FAILURES on {failing} (rank-0 logging alone would have hidden these)")
        for rank in failing:
            for name, detail in by_rank[rank]["failed_checks"].items():
                res = detail.get("storage_resolution") or {}
                ranks.log(
                    f"    rank {rank} {name}: {detail['problems']} "
                    f"steps={res.get('abs_max_element_steps')} "
                    f"beyond_one_step={res.get('elements_beyond_one_step')} "
                    f"worst={res.get('worst_absolute_element')}"
                )
    return {
        "passed": not failing and all(s["passed"] for s in by_rank.values()),
        "ranks_passed": sorted(r for r, s in by_rank.items() if s["passed"]),
        "ranks_failed": failing,
        "per_rank_failures": {str(r): by_rank[r]["failed_checks"] for r in failing},
        "worst_rank_metrics": worst_rank_metrics(by_rank, module_goldens),
        # The BF16-grid measurements the re-registered sparse-attention and
        # sink entries are judged on, for every rank rather than only the worst
        # one. `worst_rank_metrics` states whether the gate held; this states
        # how each rank got there, which is what a reader needs to see that a
        # single rank is not carrying the verdict.
        "storage_resolution_by_rank": {
            str(rank): {
                name: {
                    **{
                        k: v
                        for k, v in ((state.get("grid_agreement") or {}).get(name) or {}).items()
                        if v is not None
                    },
                    "abs_max_element_steps": (state.get("steps") or {}).get(name),
                }
                for name in (state.get("grid_agreement") or {})
            }
            for rank, state in sorted(by_rank.items())
            if state.get("grid_agreement")
        },
        "provenance_by_rank": {
            str(r): s["provenance"] for r, s in sorted(by_rank.items()) if s.get("provenance")
        },
    }


def _provenance_digest(context: Any) -> dict[str, Any] | None:
    """Compact isolation evidence for a check, small enough to gather.

    A check that compares two whole implementations cannot, on its own, say
    whether a disagreement is the implementation's arithmetic or the numbers it
    was handed. The sparse-attention checks record both --- whether the two
    kernels received bit-identical inputs, and what the shipped kernel scores
    when driven by the reference's own arguments --- and this reduces that to
    the few fields worth carrying across every rank.
    """
    if not isinstance(context, dict):
        return None
    identity = context.get("input_identity")
    isolated = context.get("kernel_on_source_inputs")
    if identity is None and isolated is None:
        return None
    digest: dict[str, Any] = {}
    if isinstance(identity, dict):
        digest["inputs_bit_exact"] = {
            key: bool(entry.get("bit_exact"))
            for key, entry in identity.items()
            if isinstance(entry, dict) and "bit_exact" in entry
        }
        digest["inputs_differing_values"] = {
            key: int(entry.get("differing_values") or 0)
            for key, entry in identity.items()
            if isinstance(entry, dict) and "differing_values" in entry
        }
        for key in ("softmax_scale_equal", "compared_slot_width", "live_slots_beyond_shared_width"):
            if key in identity:
                digest[key] = identity[key]
    if isinstance(isolated, dict):
        digest["kernel_on_source_inputs"] = {
            k: isolated[k] for k in ("rel_max_abs", "cosine", "max_abs") if k in isolated
        }
    golden = context.get("independent_golden_on_source_inputs")
    if isinstance(golden, dict):
        digest["independent_golden_on_source_inputs"] = {
            k: golden[k] for k in ("rel_max_abs", "cosine", "max_abs") if k in golden
        }
    for key in ("trtllm_table_width", "source_table_width"):
        if key in context:
            digest[key] = context[key]
    return digest


def worst_rank_metrics(
    by_rank: dict[int, Any], module_goldens: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Per check, the least favourable value any rank produced, and where.

    Error metrics are worst at their maximum and agreement metrics (``cosine``,
    anything named ``exact``/``jaccard``/``match``) at their minimum, so the
    direction is chosen per metric rather than globally. The result is what a
    reviewer needs to judge margin: a check that clears its limit on rank 0 but
    only barely on rank 5 is one unlucky rounding away from failing.

    ``headroom_x`` is how many times the measured *distance from perfect* could
    grow before reaching the registered limit --- ``limit / value`` for an
    error metric, whose perfect value is 0, and ``(1 - limit) / (1 - value)``
    for an agreement metric, whose perfect value is 1. Both read >1 when
    passing and <1 when failing, so the two kinds sort together. A check that
    measured exactly perfect has unbounded headroom and simply omits the field
    rather than recording an infinity that no strict JSON reader accepts.
    """
    worst: dict[str, dict[str, Any]] = {}
    for name in module_goldens:
        limits = module_goldens[name].get("tolerance") or {}
        # Float metrics and BF16-grid measurements are folded into one table
        # rather than reported side by side, because the re-registered
        # sparse-attention and sink entries are *judged* on the grid numbers. A
        # registered limit that only rank 0 is ever checked against is not a
        # gate, and this is what lets the audit re-judge the worst rank.
        per_rank: dict[int, dict[str, float]] = {}
        for rank, state in by_rank.items():
            values = {m: float(v) for m, v in (state.get("metrics", {}).get(name) or {}).items()}
            step = (state.get("steps") or {}).get(name)
            if isinstance(step, (int, float)):
                values["abs_max_element_steps"] = float(step)
            for metric, value in ((state.get("grid_agreement") or {}).get(name) or {}).items():
                if metric != "elements" and isinstance(value, (int, float)):
                    values[metric] = float(value)
            per_rank[rank] = values

        entry: dict[str, Any] = {}
        for metric in {m for values in per_rank.values() for m in values}:
            seen = [
                (values[metric], rank)
                for rank, values in sorted(per_rank.items())
                if metric in values
            ]
            higher_is_worse = not any(
                token in metric for token in ("cosine", "exact", "jaccard", "match")
            )
            value, rank = max(seen) if higher_is_worse else min(seen)
            entry[metric] = {"value": round(value, 9), "rank": rank}
            limit = limits.get(f"{metric}_max") if higher_is_worse else limits.get(f"{metric}_min")
            if isinstance(limit, (int, float)):
                entry[metric]["limit"] = limit
                gap, budget = (value, limit) if higher_is_worse else (1.0 - value, 1.0 - limit)
                if gap > 0:
                    # A limit of exactly zero -- `elements_beyond_one_step` --
                    # has no budget to spend, so any gap at all is zero
                    # headroom rather than a division by the limit.
                    entry[metric]["headroom_x"] = round(budget / gap, 4) if budget else 0.0
        worst[name] = entry
    return worst


def _capture(module: Any, store: dict[str, Any], key: str) -> Any:
    """Forward hook recording a module's inputs and output.

    Tensors are cloned on the way out, and that is load-bearing rather than
    defensive: the source mutates several of these buffers *in place* after
    the module returns. ``Attention.forward`` RoPEs and then FP8-quantises the
    tensor ``kv_norm`` produced, writing through the very object the hook saw,
    so capturing by reference would compare a golden against a value three
    transformations further down the pipeline.
    """
    import torch

    def snap(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone()
        if isinstance(obj, (tuple, list)):
            return type(obj)(snap(o) for o in obj)
        return obj

    def hook(_mod: Any, inputs: tuple, output: Any) -> None:
        store[key] = {"inputs": snap(inputs), "output": snap(output)}

    return module.register_forward_hook(hook)


class _SparseAttnRecorder:
    """Record the real arguments and result of the source's own ``sparse_attn``.

    The sparse kernel is a free function the official ``model`` module imports
    from ``kernel``, not a submodule, so a forward hook cannot see it. This
    wrapper replaces the name for the duration of one forward, delegates to the
    original tilelang kernel unchanged, and clones what went in and came out.

    Cloning the result matters: ``Attention.forward`` applies the inverse RoPE
    to the returned tensor *in place*, so a stored reference would show
    post-inverse-RoPE values.

    Calls arrive in layer order within a single forward pass, so the call index
    is the layer index; :meth:`record` asserts that invariant held.
    """

    def __init__(self, layer_ids: tuple[int, ...], n_layers: int):
        self.layer_ids = set(layer_ids)
        self.n_layers = n_layers
        self.calls: dict[int, dict[str, Any]] = {}
        self.n_calls = 0
        self.kernel: str | None = None
        self._model = sys.modules["model"]
        self._orig = self._model.sparse_attn

    def __enter__(self) -> _SparseAttnRecorder:
        orig = self._orig
        self.kernel = f"{getattr(orig, '__module__', '?')}.{getattr(orig, '__name__', '?')}"

        def wrapper(q, kv, attn_sink, topk_idxs, softmax_scale):
            index = self.n_calls
            self.n_calls += 1
            out = orig(q, kv, attn_sink, topk_idxs, softmax_scale)
            if index in self.layer_ids:
                self.calls[index] = {
                    "q": q.detach().clone(),
                    "kv": kv.detach().clone(),
                    "attn_sink": attn_sink.detach().clone(),
                    "topk_idxs": topk_idxs.detach().clone(),
                    "softmax_scale": float(softmax_scale),
                    "out": out.detach().clone(),
                }
            return out

        self._model.sparse_attn = wrapper
        return self

    def __exit__(self, *exc: Any) -> None:
        self._model.sparse_attn = self._orig

    def record(self, layer: int) -> dict[str, Any]:
        if self.n_calls != self.n_layers:
            raise RuntimeError(
                f"expected one sparse_attn call per layer ({self.n_layers}), saw {self.n_calls}; "
                "call index can no longer be treated as the layer index"
            )
        return self.calls[layer]


def _ulp_report(got: Any, ref: Any) -> dict[str, Any]:
    """How far apart two tensors are in units of their storage dtype's step.

    ``rel_max_abs`` divides the worst absolute difference by the tensor-wide
    RMS, which is the right scale-free metric for a homogeneous tensor. For a
    heavy-tailed one it reports the tensor's kurtosis as much as its error: a
    single element sitting far above the RMS carries a rounding error that is
    tiny *for that element* and large relative to the RMS.

    This measures the question that actually matters for a BF16 result --- are
    these the same number at the precision it is stored in? --- by expressing
    each difference in steps of the local BF16 grid, and records where the
    worst one is so a systematic error cannot hide behind an average.

    The step is the grid spacing *at the element*, floored at the grid spacing
    of the tensor's own RMS. That floor is what makes the metric a statement
    about storage resolution rather than about relative precision, and it is
    load-bearing rather than a softening: an attention output is dominated by
    cancellation, so a handful of elements per million land near zero, where an
    unfloored ratio divides a negligible absolute difference by a vanishing
    denominator and reads tens of steps. Measured on the real checkpoint, the
    *reference ladder* --- the checkpoint's own TileLang kernel against the
    independent pure-Torch golden, with no TensorRT-LLM code in the comparison
    --- has one such element on layer 3 at 18.2 unfloored steps whose two
    values both print as ``0.0``. A rule stated on the unfloored ratio is
    therefore unreachable by any implementation, including the source.

    Note the direction the floor moves the bound for the elements it touches:
    below the RMS it permits an absolute difference of one storage step *of the
    tensor*, which is ``eps`` (7.8e-3) of the RMS --- four times tighter than
    the 0.03 ``rel_max_abs`` limit it replaces. Above the RMS the floor is
    inert and the local grid step governs, which is the case the
    re-registration exists to fix. The unfloored ratios are still reported
    under ``unfloored_relative`` so nothing is hidden by the choice.
    """
    import torch

    d = (got.float() - ref.float()).abs()
    scale = torch.maximum(got.float().abs(), ref.float().abs())
    eps = torch.finfo(got.dtype).eps
    rms = max(float(ref.float().square().mean().sqrt()), 1e-30)
    local = scale * eps
    floor = rms * eps
    step = local.clamp_min(floor)
    ulps = d / step
    # The same ratio without the floor: pure relative precision, kept as a
    # diagnostic so a reader can see exactly which elements the floor covers.
    relative = d / local.clamp_min(torch.finfo(got.dtype).smallest_normal * eps)

    # What `rel_max_abs` would read if the single largest element landed one
    # storage step away and nothing else moved. This is *not* a floor: an
    # implementation that agrees with the reference bit for bit scores 0, and
    # one that agrees on the peak element while differing elsewhere scores less
    # than this. It is a scale reading --- how much of the registered budget one
    # rounding flip at the tail consumes --- and it is a diagnostic, never a
    # reason to treat a limit as unreachable. When it is large, the way to find
    # out whether the limit is reachable is to measure a known-exact candidate
    # against the same reference, not to reason from this number.
    peak = float(ref.float().abs().max())
    mantissa_bits = 8 if got.dtype == torch.bfloat16 else torch.finfo(got.dtype).bits
    spacing = float(2 ** (math.floor(math.log2(peak)) - (mantissa_bits - 1))) if peak > 0 else 0.0

    def describe(flat: int) -> dict[str, Any]:
        index, rest = [], flat
        for size in reversed(ulps.shape):
            index.append(rest % size)
            rest //= size
        return {
            "index": list(reversed(index)),
            "ref_value": round(float(ref.float().flatten()[flat]), 6),
            "got_value": round(float(got.float().flatten()[flat]), 6),
            "rms_multiple": round(abs(float(ref.float().flatten()[flat])) / rms, 3),
            "dtype_steps_apart": round(float(ulps.flatten()[flat]), 4),
            "unfloored_relative_steps_apart": round(float(relative.flatten()[flat]), 4),
        }

    return {
        # The number the judge uses: how far apart, in steps of the storage
        # dtype's grid, the single element that *drives* `rel_max_abs` is. That
        # element is by definition the largest absolute difference, and
        # `rel_max_abs` is that difference over the tensor RMS.
        "abs_max_element_steps": round(float(ulps.flatten()[int(d.argmax())]), 4),
        "max_abs_in_dtype_steps": round(float(ulps.max()), 4),
        "mean_abs_in_dtype_steps": round(float(ulps.mean()), 6),
        "elements_beyond_one_step": int((ulps > 1.0).sum()),
        "elements": int(ulps.numel()),
        # How many elements disagree at all, and by how much in total. Without
        # these a reader can only divide `mean_abs` by `elements` and guess:
        # `mean_abs_in_dtype_steps` is `total_dtype_steps / elements`, so these
        # two make the gating number re-derivable from the artifact instead of
        # inferable. A mean that implies one differing element while three
        # differ is exactly the kind of claim this closes.
        "elements_differing": int((got != ref).sum()),
        "total_dtype_steps": round(float(ulps.sum()), 6),
        "worst_absolute_element": describe(int(d.argmax())),
        "worst_relative_element": describe(int(ulps.argmax())),
        "ref_rms": round(rms, 6),
        "ref_peak_abs": round(peak, 6),
        "ref_peak_over_rms": round(peak / rms, 3),
        "one_storage_step_at_peak": spacing,
        "one_storage_step_at_rms": float(f"{floor:.6g}"),
        "rel_max_abs_if_peak_element_moves_one_step": round(spacing / rms, 6),
        "storage_dtype": str(got.dtype),
        # The unfloored view, recorded next to the gated one so the floor is
        # auditable rather than invisible. These are the numbers a pure
        # relative-precision reading produces; on an attention output they are
        # dominated by cancellation near-zeros and the source fails them too.
        "unfloored_relative": {
            "max_steps": round(float(relative.max()), 4),
            "mean_steps": round(float(relative.mean()), 6),
            "elements_beyond_one_step": int((relative > 1.0).sum()),
            "worst_element": describe(int(relative.argmax())),
        },
    }


def _sparse_attention_goldens(
    src: OfficialSource,
    layer: Any,
    lid: int,
    call: dict[str, Any],
    kv_norm_out: Any,
    q_pre_norm: Any,
    attn_out: Any,
    freqs: Any,
    tol: dict[str, Any],
    ranks: Ranks,
    tg: Any,
    ctx: dict[str, Any],
    record: Any,
    out: dict[str, Any],
) -> None:
    """Replay the real sparse-attention call through the independent golden.

    Covers four things the frequency-table comparison cannot: RoPE actually
    applied to real Q and latent-K at real positions (including the 127/128/129
    SWA block boundary), the dual-pool gather against the source's own selected
    indices, the FP32 denominator-only attention sink, and the inverse-RoPE plus
    grouped O-LoRA output path.
    """
    import torch
    import torch.distributed as dist

    attn = layer.attn
    rd = attn.rope_head_dim
    eps = src.args.norm_eps
    q_got, kv_got = call["q"], call["kv"]
    seqlen = q_got.shape[1]
    boundary = [p for p in (0, 127, 128, 129, seqlen - 1) if p < seqlen]

    # -- applied RoPE on real Q: per-head RMS scaling, then rotation of the
    #    trailing 64 dims. Compared against the Q the kernel actually consumed.
    q_ref = tg.per_head_rms_scale(
        q_pre_norm.unflatten(-1, (attn.n_local_heads, attn.head_dim)), eps
    )
    q_ref = torch.cat([q_ref[..., :-rd], tg.apply_rope(q_ref[..., -rd:], freqs[:seqlen])], dim=-1)
    record(
        f"layer{lid}.q_rope_applied",
        q_got,
        q_ref,
        "rope",
        {**ctx, "part": "query", "boundary_positions": boundary, "rope_dim": rd},
    )

    # -- applied RoPE on the real latent K/V row, including the FP8 QAT round
    #    trip the source runs on the non-RoPE dims only (block 64, not 128).
    kv_ref = torch.cat(
        [
            tg.fp8_quant_dequant(kv_norm_out[..., :-rd], 64),
            tg.apply_rope(kv_norm_out[..., -rd:], freqs[:seqlen]),
        ],
        dim=-1,
    )
    record(
        f"layer{lid}.kv_rope_applied",
        kv_got[:, :seqlen],
        kv_ref,
        "rope",
        {
            **ctx,
            "part": "latent_kv",
            "boundary_positions": boundary,
            "fp8_quant_block": 64,
            "rope_dim": rd,
        },
    )

    # -- the sparse gather itself, driven by the source's own selected indices.
    topk = call["topk_idxs"]
    o_ref = tg.sparse_attention(q_got, kv_got, call["attn_sink"], topk, call["softmax_scale"])
    valid = topk >= 0
    record(
        f"layer{lid}.sparse_attention",
        call["out"],
        o_ref,
        "sparse_attention_output",
        {
            **ctx,
            "kernel": ctx.get("kernel"),
            "kv_rows": int(kv_got.shape[1]),
            "window_rows": int(seqlen),
            "compressed_rows": int(kv_got.shape[1] - seqlen),
            "selected_per_query": int(topk.shape[-1]),
            "valid_selected_slots": int(valid.sum()),
            "padded_slots": int((~valid).sum()),
            "softmax_scale": call["softmax_scale"],
        },
    )

    # -- the sink. It contributes denominator mass only, so it can shrink the
    #    output but never steer it. Recomputing with the sink removed proves it
    #    is wired in at all, and that the direction of its effect is right.
    sink = call["attn_sink"]
    # A finite sentinel rather than -inf: exp(-1e30 - peak) underflows to a
    # clean zero, while -inf would make `peak` itself -inf for a query whose
    # slots are all padded and turn the denominator into a NaN.
    no_sink = torch.full_like(sink, -1e30)
    o_no_sink = tg.sparse_attention(q_got, kv_got, no_sink, topk, call["softmax_scale"])
    m_sink = tg.compare(call["out"], o_ref)
    m_no_sink = tg.compare(call["out"], o_no_sink)
    effect = tg.compare(o_ref, o_no_sink)
    sink_ulp = _ulp_report(call["out"], o_ref)
    norm_with = o_ref.float().square().sum(dim=-1).sqrt()
    norm_without = o_no_sink.float().square().sum(dim=-1).sqrt()
    # Both goldens are BF16, so allow one BF16 ulp of slack before calling an
    # increase real; an actual sign/placement error moves the norm far more.
    growth = float((norm_with - norm_without).max())
    shrink_only = bool((norm_with <= norm_without * 1.01 + 1e-3).all())
    limits = _tolerance(tol, "attention_sink")
    ok, problems = _judge(m_sink, limits, sink_ulp)
    if not shrink_only:
        problems.append("sink increased the output magnitude; it must only shrink it")
    if effect["rel_max_abs"] <= 0.0:
        problems.append("removing the sink changed nothing; it is not wired into the softmax")
    if m_sink["rel_max_abs"] > m_no_sink["rel_max_abs"]:
        problems.append("source output is closer to the sink-free golden than to the sink golden")
    passed = ok and not problems
    ranks.log(
        f"  golden layer{lid}.attention_sink        cos={m_sink['cosine']:.6f} "
        f"sink_effect={effect['rel_max_abs']:.3e} {'PASS' if passed else 'FAIL ' + str(problems)}"
    )
    ctx_out = {
        **ctx,
        "sink_values_min": float(sink.min()),
        "sink_values_max": float(sink.max()),
        "sink_dtype": str(sink.dtype),
        "rule": "denominator-only: shrinks output magnitude, never steers direction",
    }
    out[f"layer{lid}.attention_sink"] = {
        "module": "attention_sink",
        "metrics": {
            "cosine": round(m_sink["cosine"], 9),
            "max_abs": round(m_sink["max_abs"], 9),
            "mean_abs": round(m_sink["mean_abs"], 9),
            "rel_max_abs": round(m_sink["rel_max_abs"], 9),
            "finite": m_sink["finite"],
            "sink_effect_rel_max_abs": round(effect["rel_max_abs"], 9),
            "rel_max_abs_without_sink": round(m_no_sink["rel_max_abs"], 9),
            "shrink_only": shrink_only,
            "max_norm_increase_from_sink": round(growth, 9),
        },
        "storage_resolution": sink_ulp,
        "tolerance": limits,
        "passed": passed,
        "problems": problems,
        "context": ctx_out,
    }

    # -- inverse RoPE + grouped O-LoRA + row-parallel reduction, end to end.
    o = call["out"]
    o_rot = torch.cat([o[..., :-rd], tg.apply_rope(o[..., -rd:], freqs[:seqlen], inverse=True)], -1)
    grouped = o_rot.view(o.shape[0], seqlen, attn.n_local_groups, -1)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    y = torch.einsum("bsgd,grd->bsgr", grouped, wo_a).flatten(2)
    wo_b = tg.dequant_fp8_blockwise(attn.wo_b.weight, attn.wo_b.scale)
    # The source rounds the row-parallel partial to BF16 *before* the FP32
    # all-reduce, so the golden has to as well.
    partial = (tg.fp8_quant_dequant(y, 128).float() @ wo_b.t()).to(y.dtype).float()
    if ranks.world > 1:
        dist.all_reduce(partial)
    record(
        f"layer{lid}.o_lora_output",
        attn_out,
        partial.to(o.dtype),
        "o_lora_output",
        {
            **ctx,
            "covers": "inverse RoPE, grouped wo_a einsum, FP8 wo_b, TP all-reduce",
            "local_groups": attn.n_local_groups,
            "o_lora_rank": attn.o_lora_rank,
        },
    )


def _module_goldens(
    src: OfficialSource,
    prompts: dict[str, Any],
    tol: dict[str, Any],
    ranks: Ranks,
    tg: Any,
    layer_ids: Sequence[int] = (0, 2, 3),
) -> dict[str, Any]:
    """Replay real activations through independent pure-Torch implementations.

    The registered layers cover every active attention variant: layer 0 is
    SWA-only (ratio 0), layer 2 is ratio-4 CSA and carries the learned
    Indexer, layer 3 is ratio-128 HCA. ``layer_ids`` is a diagnostic override
    only -- it lets the same independent ladder be pointed at a *deep* layer
    to separate "TensorRT-LLM drifts from the source" from "any Torch
    implementation drifts from the source's TileLang kernels at this depth".
    """
    import torch

    spec = prompts["cache_boundary_257"]
    token_ids = spec["token_ids"]
    toks = torch.tensor([token_ids], dtype=torch.long, device="cuda")

    store: dict[str, Any] = {}
    handles = []
    layer_ids = list(layer_ids)
    for lid in layer_ids:
        layer = src.model.layers[lid]
        handles.append(_capture(layer.attn.q_norm, store, f"l{lid}.q_norm"))
        handles.append(_capture(layer.attn.kv_norm, store, f"l{lid}.kv_norm"))
        handles.append(_capture(layer.attn.wq_b, store, f"l{lid}.wq_b"))
        handles.append(_capture(layer.attn, store, f"l{lid}.attn"))
        handles.append(_capture(layer.ffn.gate, store, f"l{lid}.gate"))
        if layer.attn.compress_ratio:
            handles.append(_capture(layer.attn.compressor, store, f"l{lid}.compressor"))
            if layer.attn.indexer is not None:
                handles.append(_capture(layer.attn.indexer, store, f"l{lid}.indexer"))
    handles.append(_capture(src.model.layers[0], store, "block0"))

    src.reset_cache()
    recorder = _SparseAttnRecorder(tuple(layer_ids), len(src.model.layers))
    with recorder, torch.inference_mode():
        src.model.forward(toks, 0)
    torch.cuda.synchronize()
    for h in handles:
        h.remove()

    out: dict[str, Any] = {}
    eps = src.args.norm_eps

    def record(name: str, got: torch.Tensor, ref: torch.Tensor, module: str, ctx: dict) -> None:
        metrics = tg.compare(got, ref)
        ulp = _ulp_report(got, ref)
        limits = _tolerance(tol, module)
        passed, problems = _judge(metrics, limits, ulp)
        out[name] = {
            "module": module,
            "metrics": {
                k: (round(v, 9) if isinstance(v, float) else v) for k, v in metrics.items()
            },
            "storage_resolution": ulp,
            "tolerance": limits,
            "passed": passed,
            "problems": problems,
            "context": ctx,
        }
        ranks.log(
            f"  golden {name:32s} cos={metrics['cosine']:.6f} "
            f"rel_max_abs={metrics['rel_max_abs']:.3e} "
            f"steps={ulp['abs_max_element_steps']:.2f} "
            f"{'PASS' if passed else 'FAIL ' + str(problems)}"
        )

    for lid in layer_ids:
        layer = src.model.layers[lid]
        attn = layer.attn
        ctx = {
            "layer": lid,
            "ratio": attn.compress_ratio,
            "prompt": "cache_boundary_257",
            "seqlen": len(token_ids),
        }

        # Q-LoRA norm and latent-KV norm.
        cap = store[f"l{lid}.q_norm"]
        record(
            f"layer{lid}.q_norm",
            cap["output"],
            tg.rms_norm(cap["inputs"][0], attn.q_norm.weight, eps),
            "q_projection_and_norm",
            ctx,
        )
        cap = store[f"l{lid}.kv_norm"]
        record(
            f"layer{lid}.kv_norm",
            cap["output"],
            tg.rms_norm(cap["inputs"][0], attn.kv_norm.weight, eps),
            "kv_latent_and_norm",
            ctx,
        )

        # RoPE: rebuild the layer's own frequency table from the pinned recipe.
        if attn.compress_ratio:
            orig_len, theta = src.args.original_seq_len, src.args.compress_rope_theta
        else:
            orig_len, theta = 0, src.args.rope_theta
        freqs = tg.yarn_freqs_cis(
            attn.rope_head_dim,
            src.args.max_seq_len,
            orig_len,
            theta,
            src.args.rope_factor,
            src.args.beta_fast,
            src.args.beta_slow,
        ).to(attn.freqs_cis.device)
        record(
            f"layer{lid}.rope_freqs",
            torch.view_as_real(attn.freqs_cis),
            torch.view_as_real(freqs),
            "rope",
            {**ctx, "theta": theta, "yarn": orig_len > 0},
        )

        # Compressor gated pooling (ratio-4 and ratio-128 layers only).
        if attn.compress_ratio:
            cap = store[f"l{lid}.compressor"]
            x_in = cap["inputs"][0]
            ref = tg.compressor_prefill(
                x_in,
                attn.compressor.wkv.weight,
                attn.compressor.wgate.weight,
                attn.compressor.ape,
                attn.compressor.norm.weight,
                freqs[: len(token_ids)],
                ratio=attn.compress_ratio,
                head_dim=attn.head_dim,
                rope_dim=attn.rope_head_dim,
                eps=eps,
                rotate=False,
            )
            record(
                f"layer{lid}.compressor",
                cap["output"],
                ref,
                "compressor",
                {
                    **ctx,
                    "overlap": attn.compress_ratio == 4,
                    "compressed_rows": int(cap["output"].shape[1]),
                },
            )

        # Indexer: recompute the whole selection path independently and require
        # the *selected slot set* to match exactly. Top-k is a discrete
        # decision, so a close score is not a pass.
        if f"l{lid}.indexer" in store:
            out.update(_indexer_golden(src, layer, store[f"l{lid}.indexer"], tol, ranks, tg, ctx))

        # Sparse attention, sink, applied RoPE on real Q/K, and the output path.
        _sparse_attention_goldens(
            src,
            layer,
            lid,
            recorder.record(lid),
            store[f"l{lid}.kv_norm"]["output"],
            store[f"l{lid}.wq_b"]["output"],
            store[f"l{lid}.attn"]["output"],
            freqs,
            tol,
            ranks,
            tg,
            {**ctx, "kernel": recorder.kernel},
            record,
            out,
        )

    # mHC: recompute the block-0 pre-mix from its real input.
    blk = src.model.layers[0]
    x_in = store["block0"]["inputs"][0]
    ref_pre, ref_post, ref_comb = tg.hc_pre(
        x_in,
        blk.hc_attn_fn,
        blk.hc_attn_scale,
        blk.hc_attn_base,
        hc_mult=blk.hc_mult,
        iters=blk.hc_sinkhorn_iters,
        norm_eps=blk.norm_eps,
        hc_eps=blk.hc_eps,
    )
    got_pre, got_post, got_comb = blk.hc_pre(
        x_in, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base
    )
    record("block0.hc_pre", got_pre, ref_pre, "mhc", {"layer": 0, "part": "pre"})
    record("block0.hc_post_weights", got_post, ref_post, "mhc", {"layer": 0, "part": "post"})
    record(
        "block0.hc_sinkhorn",
        got_comb,
        ref_comb,
        "mhc",
        {"layer": 0, "part": "comb", "iters": blk.hc_sinkhorn_iters},
    )

    # MoE routing: hash-routed layer 0 and score-routed layer 3. Both are in
    # the registered layer set; a diagnostic `--golden-layers` that drops one
    # simply does not measure it, rather than dying on a missing capture.
    for lid in (lid for lid in (0, 3) if f"l{lid}.gate" in store):
        layer = src.model.layers[lid]
        cap = store[f"l{lid}.gate"]
        x_g, ids_g = cap["inputs"][0], cap["inputs"][1]
        got_w, got_i = cap["output"]
        gate = layer.ffn.gate
        ref_w, ref_i, _ = tg.moe_route(
            x_g,
            gate.weight,
            topk=gate.topk,
            route_scale=gate.route_scale,
            bias=gate.bias,
            tid2eid=getattr(gate, "tid2eid", None),
            input_ids=ids_g if gate.hash else None,
        )
        agree = float((got_i.long() == ref_i.long()).float().mean())
        out[f"layer{lid}.moe_expert_ids"] = {
            "module": "routing_ids",
            "metrics": {"exact_agreement": agree},
            "tolerance": {"rule": "exact"},
            "passed": agree == 1.0,
            "problems": [] if agree == 1.0 else [f"expert-id agreement {agree:.6f} != 1.0"],
            "context": {"layer": lid, "hash_routed": bool(gate.hash), "topk": gate.topk},
        }
        ranks.log(
            f"  golden layer{lid}.moe_expert_ids        exact={agree:.6f} "
            f"{'PASS' if agree == 1.0 else 'FAIL'}"
        )
        record(
            f"layer{lid}.moe_route_weights",
            got_w,
            ref_w,
            "moe_routing_weights",
            {"layer": lid, "hash_routed": bool(gate.hash)},
        )

    # Routed expert: clamped SwiGLU on real MXFP4 weights, real routed tokens.
    lid = 3
    layer = src.model.layers[lid]
    local0 = layer.ffn.experts_start_idx
    expert = layer.ffn.experts[local0]
    cap = store[f"l{lid}.gate"]
    x_g = cap["inputs"][0]
    sample = x_g[:64]
    got_e = expert(sample)
    ref_e = tg.expert_swiglu(
        sample,
        tg.dequant_mxfp4(expert.w1.weight, expert.w1.scale).to(sample.dtype),
        tg.dequant_mxfp4(expert.w2.weight, expert.w2.scale).to(sample.dtype),
        tg.dequant_mxfp4(expert.w3.weight, expert.w3.scale).to(sample.dtype),
        swiglu_limit=expert.swiglu_limit,
    )
    record(
        f"layer{lid}.routed_expert",
        got_e,
        ref_e,
        "moe_expert_output",
        {
            "layer": lid,
            "expert": local0,
            "tokens": int(sample.shape[0]),
            "weight_dtype": "packed MXFP4 (I8 container)",
            "swiglu_limit": expert.swiglu_limit,
        },
    )

    return out


def _indexer_golden(
    src: OfficialSource,
    layer: Any,
    cap: dict[str, Any],
    tol: dict[str, Any],
    ranks: Ranks,
    tg: Any,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Independently reproduce the ratio-4 Indexer selection.

    Every step is recomputed from rank-local weights and the real captured
    inputs: Q projection, RoPE, Hadamard rotation, FP4 round trip, the
    compressed-KV build, the weighted 64-head ReLU reduction (summed across
    tensor-parallel ranks, which is part of the contract), the causal mask,
    and finally top-k. The selected set must be exactly equal.
    """
    import torch
    import torch.distributed as dist

    idx = layer.attn.indexer
    x_in, qr_in, start_pos, offset = cap["inputs"][:4]
    got_topk = cap["output"]
    bsz, seqlen, _ = x_in.shape
    ratio, rd = idx.compress_ratio, idx.rope_head_dim

    freqs = tg.yarn_freqs_cis(
        rd,
        src.args.max_seq_len,
        src.args.original_seq_len,
        src.args.compress_rope_theta,
        src.args.rope_factor,
        src.args.beta_fast,
        src.args.beta_slow,
    ).to(x_in.device)[start_pos : start_pos + seqlen]

    q = idx.wq_b(qr_in).unflatten(-1, (idx.n_local_heads, idx.head_dim))
    q = torch.cat([q[..., :-rd], tg.apply_rope(q[..., -rd:], freqs)], dim=-1)
    q = tg.fp4_quant_dequant(tg.hadamard_transform(q), 32)

    comp = tg.compressor_prefill(
        x_in,
        idx.compressor.wkv.weight,
        idx.compressor.wgate.weight,
        idx.compressor.ape,
        idx.compressor.norm.weight,
        freqs,
        ratio=ratio,
        head_dim=idx.head_dim,
        rope_dim=rd,
        eps=src.args.norm_eps,
        rotate=True,
        hadamard=tg.hadamard_transform,
    )

    weights = idx.weights_proj(x_in) * (idx.softmax_scale * idx.n_heads**-0.5)
    score = torch.einsum("bshd,btd->bsht", q.float(), comp.float())
    score = (score.relu() * weights.float().unsqueeze(-1)).sum(dim=2)
    if ranks.world > 1:
        dist.all_reduce(score)

    slots = torch.arange(score.shape[-1], device=score.device)
    limit = (torch.arange(1, seqlen + 1, device=score.device) // ratio).unsqueeze(1)
    score = score + torch.where(slots.unsqueeze(0) >= limit, float("-inf"), 0.0)
    ref_idx = score.topk(min(idx.index_topk, comp.shape[1]), dim=-1)[1]
    ref_topk = torch.where(ref_idx >= limit, -1, ref_idx + offset)

    # Compare as sets per query: top-k ordering among equal scores is not part
    # of the contract, but which slots are visible is.
    def as_sets(t: torch.Tensor) -> list[set]:
        return [set(row[row >= 0].tolist()) for row in t.reshape(-1, t.shape[-1]).cpu()]

    got_sets, ref_sets = as_sets(got_topk), as_sets(ref_topk)
    equal = sum(a == b for a, b in zip(got_sets, ref_sets))
    agreement = equal / max(len(got_sets), 1)
    # Early queries have no complete compressed group yet, so both sides are
    # legitimately empty. Scoring those as Jaccard 0 would understate agreement.
    jaccard = sum(
        1.0 if not (a | b) else len(a & b) / len(a | b) for a, b in zip(got_sets, ref_sets)
    ) / max(len(got_sets), 1)
    empty = sum(1 for a, b in zip(got_sets, ref_sets) if not (a | b))

    limits = _tolerance(tol, "indexer_topk")
    passed = agreement == 1.0
    ranks.log(
        f"  golden layer{ctx['layer']}.indexer_topk        exact_set={agreement:.6f} "
        f"jaccard={jaccard:.6f} {'PASS' if passed else 'FAIL'}"
    )
    return {
        f"layer{ctx['layer']}.indexer_topk": {
            "module": "indexer_topk",
            "metrics": {
                "exact_set_agreement": round(agreement, 9),
                "mean_jaccard": round(jaccard, 9),
                "queries_with_no_valid_slot": empty,
                "queries": len(got_sets),
                "selected_per_query": int(got_topk.shape[-1]),
                "compressed_slots": int(comp.shape[1]),
            },
            "tolerance": limits,
            "passed": passed,
            "problems": [] if passed else [f"exact set agreement {agreement:.6f} != 1.0"],
            "context": {**ctx, "index_topk": idx.index_topk, "offset": int(offset)},
        }
    }


# The fixture has to span the prompt shapes the task calls out, not just five
# copies of the easiest one: a loop bug in window/compression bookkeeping only
# shows up once a prompt crosses a 128-token SWA block.
MIN_NATIVE_PROMPTS = 5
REQUIRED_NATIVE_CATEGORIES = ("plain_chat", "reasoning", "code", "cache_boundary")


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _at(seq: Any, index: int) -> Any:
    """Element ``index`` of an optional recorded per-step list, or ``None``."""
    items = list(seq or [])
    return items[index] if 0 <= index < len(items) else None


def compare_native_golden(
    golden: dict[str, Any],
    gen: dict[str, Any],
    prompts_manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    """Judge the official hand-written loop against the canonical fixture.

    Pure: no torch, no filesystem, no ranks --- so the negative cases (a
    truncated sequence, a missing prompt, a fixture from the wrong revision or
    from a sampling run) are unit-testable on CPU.

    Every rule here is a *hard* equality: same length, same tokens, all the way
    through. The cross-check exists because the official loop re-implements
    generation, and a comparison that tolerates a short fixture, a missing
    prompt, a matching prefix or an "undecidable" step would report success for
    exactly the bugs it is meant to catch. A prompt whose sequences differ is
    reported as a mismatch, always --- there is no path by which a divergent
    prompt is labelled a match.
    """
    problems: list[str] = []
    prompts = {p["id"]: p for p in prompts_manifest["prompts"]}

    # -- provenance: the fixture must describe *this* checkpoint, *these*
    #    prompts and a deterministic native generate run.
    prov = golden.get("provenance") or {}
    if golden.get("checkpoint_revision") != prompts_manifest["checkpoint_revision"]:
        problems.append(
            f"fixture checkpoint_revision {golden.get('checkpoint_revision')!r} != "
            f"manifest {prompts_manifest['checkpoint_revision']!r}"
        )
    if prov.get("prompts_manifest_sha256") != manifest_sha256:
        problems.append("fixture was produced against a different prompts manifest")
    generator = str(prov.get("generator", ""))
    if "generate(do_sample=False)" not in generator:
        problems.append(f"fixture generator {generator!r} is not a native greedy generate")
    if not prov.get("transformers_version"):
        problems.append("fixture does not record the transformers version that produced it")
    if not prov.get("conversion_code_sha256"):
        problems.append("fixture does not record the checkpoint-conversion code it used")
    decoding = golden.get("decoding") or {}
    if decoding.get("do_sample") is not False or decoding.get("num_beams") != 1:
        problems.append(f"fixture decoding is not deterministic greedy: {decoding}")

    fixture_prompts = golden.get("prompts") or {}
    required = sorted(golden.get("required_prompt_ids") or fixture_prompts)
    # Prompts the fixture records but explicitly does not gate on. They are
    # still compared and reported with their true result --- a divergent prompt
    # is never called a match --- but they cannot make the gate pass or fail.
    # Declaring one costs a written reason in the fixture, so the exclusion is
    # visible in the evidence instead of implied by an absence.
    non_gating = sorted(set(golden.get("non_gating_prompt_ids") or []))
    if non_gating and not golden.get("non_gating_reason"):
        problems.append("fixture declares non-gating prompts without recording why")
    overlap = sorted(set(required) & set(non_gating))
    if overlap:
        problems.append(f"prompts are both required and non-gating: {overlap}")
    extra = sorted(set(fixture_prompts) - set(required) - set(non_gating))
    if extra:
        problems.append(f"fixture carries prompts it neither requires nor declares: {extra}")

    per_prompt: dict[str, Any] = {}
    matched_ids: list[str] = []
    for pid in sorted(set(required) | set(non_gating)):
        gating = pid in required
        entry = fixture_prompts.get(pid)
        run = gen.get(pid)
        want = list(entry.get("tokens") or []) if entry else []
        got = list(run.get("tokens") or []) if run else []
        detail: dict[str, Any] = {
            "gating": gating,
            "fixture_tokens": len(want),
            "official_tokens": len(got),
            "match": False,
            "first_divergence": None,
        }
        if entry is None:
            detail["problem"] = "prompt missing from fixture"
        elif run is None:
            detail["problem"] = "prompt was not generated by the official loop this run"
        elif pid not in prompts:
            detail["problem"] = "prompt id is not in the pre-registered prompt manifest"
        elif entry.get("rendered_sha256") != prompts[pid].get("rendered_sha256"):
            detail["problem"] = "fixture prompt text differs from the pre-registered rendering"
        elif not want:
            detail["problem"] = "fixture recorded an empty generation"
        elif len(want) != len(got):
            detail["problem"] = f"length mismatch: fixture {len(want)} vs official {len(got)}"
            detail["first_divergence"] = next(
                (i for i in range(min(len(want), len(got))) if want[i] != got[i]),
                min(len(want), len(got)),
            )
        elif want != got:
            detail["problem"] = "token sequences differ"
            detail["first_divergence"] = next(i for i in range(len(want)) if want[i] != got[i])
        else:
            detail["match"] = True
            if gating:
                matched_ids.append(pid)
        if not detail["match"]:
            # Both sides' view of the first differing step, so a reader can see
            # whether the two references were close or genuinely disagreed.
            step = detail["first_divergence"]
            if step is not None:
                detail["divergence_evidence"] = {
                    "step": step,
                    "fixture_token": want[step] if step < len(want) else None,
                    "fixture_margin": _at(entry.get("top1_top2_margin"), step),
                    "fixture_candidates": _at(entry.get("top2_candidates"), step),
                    "official_token": got[step] if step < len(got) else None,
                    "official_margin": _at((run or {}).get("top1_top2_margin"), step),
                    "official_candidates": _at((run or {}).get("top2_candidates"), step),
                }
            if gating:
                problems.append(f"{pid}: {detail['problem']}")
        per_prompt[pid] = detail

    if len(matched_ids) < MIN_NATIVE_PROMPTS:
        problems.append(
            f"only {len(matched_ids)} required prompts reproduced token-for-token, "
            f"need >= {MIN_NATIVE_PROMPTS}"
        )
    matched_categories = {prompts[pid]["category"] for pid in matched_ids if pid in prompts}
    for category in REQUIRED_NATIVE_CATEGORIES:
        if category not in matched_categories:
            problems.append(f"no {category!r} prompt reproduced token-for-token")

    return {
        "passed": not problems,
        "status": "compared",
        "required_prompt_ids": required,
        "non_gating_prompt_ids": non_gating,
        "non_gating_reason": golden.get("non_gating_reason"),
        "prompts_compared": len(per_prompt),
        "matched_prompt_ids": matched_ids,
        "mismatched_prompt_ids": sorted(p for p, d in per_prompt.items() if not d["match"]),
        "fixture_provenance": prov,
        "fixture_decoding": decoding,
        "per_prompt": per_prompt,
        "problems": problems,
    }


def _check_native_golden(gen: dict[str, Any], ranks: Ranks) -> dict[str, Any]:
    """Load the checked-in canonical fixture and compare the official loop to it.

    The fixture is produced once, in an isolated environment, by the source
    model's own ``AutoModelForCausalLM.generate(do_sample=False)`` --- see
    ``deepseek_v4_flash_h100/hf_native_golden.py``. If it is absent the check
    reports ``missing`` and fails the suite rather than quietly passing: an
    unanchored hand-written loop is exactly what the cross-check exists to
    catch.
    """
    path = os.path.join(SUPPORT_PKG, "manifests", "native_generate_golden.json")
    manifest_path = os.path.join(SUPPORT_PKG, "manifests", "prompts.json")
    if not os.path.exists(path):
        ranks.log("  native-generate golden: NOT PRESENT -- reference ladder incomplete")
        return {
            "passed": False,
            "status": "missing",
            "fixture": path,
            "problems": [
                "canonical AutoModelForCausalLM.generate fixture has not been produced yet"
            ],
        }
    with open(path) as f:
        golden = json.load(f)
    result = compare_native_golden(
        golden, gen, _load_manifest("prompts.json"), _sha256_file(manifest_path)
    )
    result["fixture"] = path
    result["fixture_sha256"] = _sha256_file(path)
    verdict = "PASS" if result["passed"] else "FAIL " + str(result["problems"][:3])
    ranks.log(
        f"  native-generate golden: {len(result['matched_prompt_ids'])}/"
        f"{len(result['required_prompt_ids'])} required prompts token-identical {verdict}"
    )
    for pid in result["mismatched_prompt_ids"]:
        detail = result["per_prompt"][pid]
        ranks.log(
            f"    MISMATCH {pid}: {detail['problem']} "
            f"(gating={detail['gating']}, first divergence at step {detail['first_divergence']})"
        )
    return result


#: The two Goal-3.4 checks, with the judge that owns each one's rules and the
#: key under which the artifact records that judge's registered thresholds.
#: Keyed by check name so a renamed check fails loudly here rather than
#: silently dropping out of the audit.
_PARITY_CHECKS = (
    ("source_logit_replay", "limits", "logit_prompt_failures"),
    ("generation_parity", "gate", "parity_prompt_failures"),
)

#: Checks a suite is *defined* to produce, by suite name. Re-deriving the
#: verdicts that are present says nothing about the ones that are not, so an
#: artifact that simply omits a check would otherwise audit clean: delete
#: ``source_logit_replay`` and ``generation_parity``, set ``passed=true``, and
#: every remaining rule agrees. Suites whose check set depends on their
#: arguments (``activation_replay_eager`` follows ``--replay-layers``,
#: ``kernel_contract`` follows its case list) are not listed; they are covered
#: by the "no checks at all" rule instead.
_REQUIRED_CHECKS = {
    "eager_full_model": (
        "runtime_contract",
        "custom_tokenizer",
        "worker_dispatch",
        "source_logit_replay",
        "generation_parity",
    ),
    "load_and_moe": (
        "raw_tensor_consumption",
        "model_key_destinations",
        "no_duplicate_slots",
        "routed_expert_layout",
        "routed_expert_residency",
        "dense_fp8_bf16_contract",
    ),
}


def _audit_required_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """A suite may not audit clean while a check it must produce is missing.

    Absence is the one failure mode a re-derivation cannot see, and it is the
    cheapest to exploit: every rule that runs still agrees, because the rules
    that would have disagreed were removed with the check they belonged to.
    """
    if "checks" not in artifact:
        return []
    checks = artifact.get("checks") or {}
    suite = artifact.get("suite")
    missing = [name for name in _REQUIRED_CHECKS.get(suite, ()) if name not in checks]
    if missing:
        return [
            {
                "check": f"{suite}.missing_checks",
                "module": suite,
                "problems": [f"suite {suite!r} recorded no {name!r} check" for name in missing],
            }
        ]
    if not checks:
        return [
            {
                "check": f"{suite}.no_checks",
                "module": suite,
                "problems": [f"suite {suite!r} recorded an empty check set"],
            }
        ]
    return []


def _audit_state_determinism(
    artifact: dict[str, Any], disagreements: list[str]
) -> list[dict[str, Any]]:
    """Re-derive ``eager_state_determinism``'s verdict from what it recorded.

    Every fact that rule consumes --- which engine ran which prompt, whether
    the tokens and logits matched the anchor, whether the runtime still held a
    block at teardown --- is in the artifact, so the verdict is recomputed by
    calling the suite's own rule function on the recorded record rather than
    by a second copy of the rule. Without this the auditor would find no
    module goldens and no ``checks`` in this artifact and call it clean, which
    is precisely the false pass it exists to prevent.
    """
    if artifact.get("suite") != "eager_state_determinism":
        return []
    sys.path.insert(0, SUPPORT_PKG)
    import state_determinism

    engines = artifact.get("engines")
    comparison = artifact.get("comparison")
    if not engines or not comparison:
        return [
            {
                "check": "eager_state_determinism.missing_record",
                "module": "eager_state_determinism",
                "problems": ["the artifact records no engines or no comparison to re-derive"],
            }
        ]
    problems = state_determinism.judge_executions(engines, comparison)
    if bool(artifact.get("passed")) != (not problems):
        disagreements.append(
            f"eager_state_determinism: artifact says passed={artifact.get('passed')} but the "
            f"recorded executions re-derive to {not problems} ({problems})"
        )
    if not problems:
        return []
    return [
        {
            "check": "eager_state_determinism",
            "module": "eager_state_determinism",
            "metrics": None,
            "problems": problems,
        }
    ]


def _audit_parity_checks(
    artifact: dict[str, Any], disagreements: list[str]
) -> list[dict[str, Any]]:
    """Re-derive ``source_logit_replay`` and ``generation_parity`` from record.

    The driver's judges run on tensors and token streams that never reach the
    artifact, so unlike the module goldens these two verdicts cannot be
    recomputed from their inputs. What *is* recorded is every fact the
    registered rule consumes, per prompt --- so the verdicts are re-derived by
    calling the judges' own rule functions on the recorded facts. Anything the
    artifact asserts and this disagrees with is a disagreement; any gating
    prompt that fails is a strict failure, whatever the artifact claims.

    Also re-checks the two aggregate rules, which a per-prompt pass cannot
    imply: enough gating prompts passed, and both required categories are
    represented.
    """
    sys.path.insert(0, SUPPORT_PKG)
    import full_model

    failures: list[dict[str, Any]] = []
    for name, threshold_key, rule_name in _PARITY_CHECKS:
        check = (artifact.get("checks") or {}).get(name)
        if not check:
            continue
        rule = getattr(full_model, rule_name)
        thresholds = check.get(threshold_key)
        if thresholds is None:
            disagreements.append(f"{name}: no registered {threshold_key!r} recorded to re-derive")
            continue
        gating_ids = set(check.get("gating_prompt_ids") or [])
        recorded_pass, passing, categories = [], [], set()
        for pid, detail in sorted((check.get("per_prompt") or {}).items()):
            problems = rule(detail, thresholds)
            expected = not problems
            if bool(detail.get("passed")) != expected:
                disagreements.append(
                    f"{name}.{pid}: artifact says passed={detail.get('passed')} but the "
                    f"registered rules say {expected} ({problems})"
                )
            if pid in gating_ids:
                recorded_pass.append(expected)
                if expected:
                    passing.append(pid)
                    if detail.get("category"):
                        categories.add(detail["category"])
                else:
                    failures.append(
                        {"check": f"{name}.{pid}", "module": name, "problems": problems}
                    )

        aggregate = []
        min_prompts = int(
            (thresholds if threshold_key == "gate" else check["gate"]).get("min_prompts", 5)
        )
        if len(passing) < min_prompts:
            aggregate.append(
                f"{len(passing)} gating prompts pass, the registered gate requires {min_prompts}"
            )
        if name == "generation_parity":
            aggregate += [
                f"no {required!r} prompt passed with non-empty output"
                for required in ("plain_chat", "reasoning")
                if required not in categories
            ]
        if aggregate:
            failures.append({"check": name, "module": name, "problems": aggregate})
        expected_check = not aggregate and all(recorded_pass)
        if bool(check.get("passed")) != expected_check:
            disagreements.append(
                f"{name}: artifact says passed={check.get('passed')} but re-deriving every "
                f"recorded prompt and the aggregate rules gives {expected_check}"
            )
    return failures


def _audit_reference_provenance(artifact: dict[str, Any]) -> list[str]:
    """The reference the parity checks were judged against must be identified.

    A parity verdict is only as good as the reference behind it. The sidecar
    holding the source's logits is a separate 99 MB file, so the artifact has
    to name it, hash it, and cover every prompt it judged --- otherwise a
    verdict could have been computed against a different capture, or against
    prompts the sidecar never held, and nothing in the JSON would show it.
    """
    checks = artifact.get("checks") or {}
    if not any(name in checks for name, _, _ in _PARITY_CHECKS):
        return []
    problems = []
    provenance = (artifact.get("source_reference") or {}).get("reference_provenance") or {}
    sidecar = provenance.get("logits_sidecar") or {}
    if not sidecar.get("sha256"):
        problems.append("parity was judged without a hashed source-reference logit sidecar")
    if not provenance.get("native_generate_golden_passed"):
        problems.append(
            "the source reference is not anchored to the native-generate fixture "
            f"(native_generate_golden_passed={provenance.get('native_generate_golden_passed')!r})"
        )
    revision = provenance.get("checkpoint_revision")
    if revision and revision != artifact.get("checkpoint_revision"):
        problems.append(
            f"reference captured at checkpoint revision {revision!r}, artifact measured "
            f"{artifact.get('checkpoint_revision')!r}"
        )
    covered = set(sidecar.get("prompts") or [])
    for name, _, _ in _PARITY_CHECKS:
        judged = set((checks.get(name) or {}).get("per_prompt") or {})
        if judged - covered:
            problems.append(
                f"{name} judged {sorted(judged - covered)} which the reference sidecar "
                "does not contain"
            )
    return problems


def audit_artifact(artifact: dict[str, Any], tolerances: dict[str, Any]) -> dict[str, Any]:
    """Re-derive every verdict in a written artifact from the manifest.

    The driver already judges each golden as it measures it, but a reader has
    no reason to take that on trust: the whole point of a pre-registered
    tolerance manifest is that anyone can recompute the verdicts from the
    recorded numbers. This walks the artifact, applies ``tolerances.json``
    literally to every metric it finds, and reports any entry whose recorded
    ``passed`` disagrees with what the registered limits say --- plus any rank
    that failed, any non-gating prompt that diverged, and any status field that
    does not match the measurement.

    Pure and unit-testable: it reads dicts, not files or GPUs.
    """
    strict_failures: list[dict[str, Any]] = []
    disagreements: list[str] = []

    for name, entry in (artifact.get("module_goldens") or {}).items():
        module = entry.get("module")
        limits = (tolerances.get("modules") or {}).get(module)
        metrics = entry.get("metrics") or {}
        # An exact rule carries no float metrics, so the tolerance pass below
        # would skip it entirely --- and a failed exact rule skipped by the
        # auditor is exactly the false pass this function exists to prevent.
        if entry.get("rule") == "exact":
            if not entry.get("passed"):
                strict_failures.append(
                    {
                        "check": name,
                        "module": module,
                        "metrics": None,
                        "problems": entry.get("problems") or ["exact rule violated"],
                    }
                )
            continue
        if limits is None or "cosine" not in metrics:
            continue
        expected, problems = _judge(metrics, limits, entry.get("storage_resolution"))
        if not expected:
            strict_failures.append(
                {"check": name, "module": module, "metrics": metrics, "problems": problems}
            )
        if bool(entry.get("passed")) != expected:
            disagreements.append(
                f"{name}: artifact says passed={entry.get('passed')} but the registered "
                f"limits for {module!r} say {expected} ({problems})"
            )

    for rank, checks in (artifact.get("per_rank_failures") or {}).items():
        for name, detail in checks.items():
            strict_failures.append(
                {
                    "check": f"rank{rank}.{name}",
                    "module": None,
                    "metrics": detail.get("metrics"),
                    "problems": detail.get("problems"),
                }
            )

    # Rank 0's own metrics are only one eighth of the evidence. Re-judge the
    # least favourable value any rank produced, so a check that clears the
    # limit on rank 0 and misses it on rank 5 cannot audit clean.
    for name, entry in (artifact.get("worst_rank_metrics") or {}).items():
        module = ((artifact.get("module_goldens") or {}).get(name) or {}).get("module")
        limits = (tolerances.get("modules") or {}).get(module)
        metrics = {m: v["value"] for m, v in entry.items() if isinstance(v, dict) and "value" in v}
        if limits is None or "cosine" not in metrics:
            continue
        expected, problems = _judge(metrics, limits)
        if not expected:
            worst_ranks = {m: entry[m].get("rank") for m in metrics if m in entry}
            strict_failures.append(
                {
                    "check": f"worst_rank.{name}",
                    "module": module,
                    "metrics": metrics,
                    "problems": problems + [f"worst rank per metric: {worst_ranks}"],
                }
            )

    strict_failures.extend(_audit_required_checks(artifact))
    strict_failures.extend(_audit_parity_checks(artifact, disagreements))
    strict_failures.extend(_audit_state_determinism(artifact, disagreements))
    disagreements.extend(_audit_reference_provenance(artifact))

    native = artifact.get("native_generate_golden") or {}
    for pid in native.get("mismatched_prompt_ids") or []:
        detail = (native.get("per_prompt") or {}).get(pid) or {}
        if detail.get("gating"):
            strict_failures.append(
                {"check": f"native_generate_golden.{pid}", "problems": [detail.get("problem")]}
            )

    expected_status = artifact_status(artifact.get("passed"), artifact.get("error"))
    if artifact.get("status") != expected_status:
        disagreements.append(
            f"status {artifact.get('status')!r} does not follow passed="
            f"{artifact.get('passed')!r} (expected {expected_status!r})"
        )
    if strict_failures and artifact.get("passed"):
        disagreements.append(
            f"artifact claims passed=true with {len(strict_failures)} strict failures"
        )

    # Surfaced so "it passed" can be read as a margin rather than a boolean:
    # the checks closest to their registered limit are the ones a reviewer
    # should look at, and they are usually not on rank 0.
    margins = [
        (check, metric, detail)
        for check, entry in (artifact.get("worst_rank_metrics") or {}).items()
        for metric, detail in entry.items()
        if isinstance(detail, dict) and isinstance(detail.get("headroom_x"), (int, float))
    ]
    margins.sort(key=lambda m: m[2]["headroom_x"])

    return {
        "strict_failures": strict_failures,
        "verdict_disagreements": disagreements,
        "tightest_margins": margins[:5],
        "ranks_failed": artifact.get("ranks_failed") or [],
        "non_gating_divergences": [
            pid
            for pid in native.get("mismatched_prompt_ids") or []
            if not ((native.get("per_prompt") or {}).get(pid) or {}).get("gating")
        ],
        "clean": not strict_failures and not disagreements,
    }


def _rehash_sidecars(artifact: dict[str, Any]) -> list[str]:
    """Re-hash every logit sidecar the artifact names, on disk, right now.

    Both the artifact's own sidecar and the source reference's are checked:
    a parity verdict rests on the two of them agreeing prompt by prompt, and
    a sidecar that has been replaced, truncated or half-written since the run
    would otherwise be indistinguishable from the one that was measured.
    """
    sys.path.insert(0, SUPPORT_PKG)
    import source_reference

    named = [("artifact", artifact.get("logits_sidecar"))]
    provenance = (artifact.get("source_reference") or {}).get("reference_provenance") or {}
    named.append(("source_reference", provenance.get("logits_sidecar")))

    notes = []
    for which, sidecar in named:
        if not sidecar or not sidecar.get("sha256"):
            continue
        path = sidecar.get("path")
        if not path or not os.path.exists(path):
            notes.append(f"{which} logit sidecar {path!r} is recorded but missing on disk")
            continue
        actual = source_reference.sha256_file(path)
        if actual != sidecar["sha256"]:
            notes.append(
                f"{which} logit sidecar {path} now hashes to {actual}, but the run "
                f"recorded {sidecar['sha256']}; the verdicts were not judged against this file"
            )
    return notes


def _run_audit(paths: list[str]) -> int:
    tolerances = _load_manifest("tolerances.json")
    worst = 0
    for path in paths:
        with open(path) as f:
            artifact = json.load(f)
        report = audit_artifact(artifact, tolerances)
        # `audit_artifact` is pure by contract, so the one provenance question
        # that needs the filesystem is answered here: does the sidecar on disk
        # still hash to what the capture recorded? Everything above judges the
        # numbers; this judges that they came from the file they claim.
        for note in _rehash_sidecars(artifact):
            report["verdict_disagreements"].append(note)
            report["clean"] = False
        print(f"\n== strict audit of {path} ==", flush=True)
        print(f"  suite            {artifact.get('suite')}", flush=True)
        print(
            f"  artifact says    passed={artifact.get('passed')} status={artifact.get('status')}",
            flush=True,
        )
        print(f"  ranks failed     {report['ranks_failed']}", flush=True)
        print(f"  strict failures  {len(report['strict_failures'])}", flush=True)
        for check, metric, detail in report["tightest_margins"]:
            print(
                f"  tightest margin  {check}.{metric} = {detail['value']:.6g} "
                f"vs {detail['limit']:.6g} on rank {detail['rank']} "
                f"({detail['headroom_x']:.2f}x headroom)",
                flush=True,
            )
        for failure in report["strict_failures"]:
            print(f"    {failure['check']}: {failure['problems']}", flush=True)
        for note in report["verdict_disagreements"]:
            print(f"    DISAGREEMENT {note}", flush=True)
        if report["non_gating_divergences"]:
            print(
                f"  declared non-gating divergences (reported, not gating): "
                f"{report['non_gating_divergences']}",
                flush=True,
            )
        print(f"  AUDIT {'CLEAN' if report['clean'] else 'FAILED'}", flush=True)
        worst = max(worst, 0 if report["clean"] else 1)
    return worst


def _run_regression_report(registered_log: str, junit_xml: str, *, output: str) -> int:
    """Classify the registered regression run and write the verdict.

    Needs no GPU beyond reading the device list, and deliberately does not
    re-run pytest: the inputs are the registered command's own log and the
    focused required-H100 pass, so the adjudication is reproducible from
    artifacts that already exist.
    """
    sys.path.insert(0, SUPPORT_PKG)
    import build_manifests
    import regression_report as rr

    # The expected-failure list is only meaningful if it cannot be edited to
    # absorb a real regression, so the report refuses to run against an
    # unregistered manifest exactly as the measurement suites do.
    build_manifests.verify_checksums()
    baseline = rr.load_baseline()
    report = rr.build_report(
        baseline=baseline,
        registered_log=rr.parse_pytest_log(registered_log),
        required_records=rr.parse_junit(junit_xml),
        required_command=baseline["required_h100_command"],
        device_report=_device_report(),
    )
    report.update(
        {
            "suite": "regression_report",
            "status": artifact_status(report["passed"], None),
            "error": None,
            "environment": {
                "linked_package": _linked_package_report(),
                "devices": _device_report(),
            },
            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    print(f"\n== protected regression classification -> {output} ==", flush=True)
    print(f"  registered exit  {report['registered_run']['exit_code']}", flush=True)
    print(f"  registered counts {report['registered_run']['counts']}", flush=True)
    failures = report["failures"]
    print(
        f"  failures         {failures['total']} "
        f"({failures['environment']} environment, {len(failures['genuine'])} genuine)",
        flush=True,
    )
    for signature, ids in failures["by_signature"].items():
        print(f"    signature {signature}: {len(ids)}", flush=True)
    print(f"  new vs baseline  {failures['new_against_baseline']}", flush=True)
    print(f"  resolved         {failures['resolved_against_baseline']}", flush=True)
    required = report["required_h100_cases"]
    print(
        f"  required H100    {required['passed']}/{required['count']} passed, "
        f"failed={required['failed']}, skipped={required['skipped']}",
        flush=True,
    )
    blackwell = report["protected_blackwell_dispatch"]
    static = blackwell["static_dispatch_tests"]
    print(
        f"  blackwell static {static['passed']}/{static['registered']} registered "
        f"controls passed, missing={static['missing']}, skipped={static['skipped']}, "
        f"failed={static['failed']}; runtime {blackwell['blackwell_runtime']}",
        flush=True,
    )
    for problem in report["problems"]:
        print(f"    PROBLEM {problem}", flush=True)
    print(f"  REGRESSION {'CLEAN' if report['passed'] else 'FAILED'}", flush=True)
    return 0 if report["passed"] else 1


def suite_kernel_contract(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Prove the golden's two GEMMs reproduce the source kernel's ``T.gemm``.

    The cheapest rung of the ladder: one GPU, no checkpoint, about half a
    minute. It exists because the sparse-attention golden's trustworthiness
    rests on an arithmetic claim --- BF16 operands with an FP32 accumulator ---
    that is invisible to any tolerance, since widening the operands computes
    the same quantity and still agrees to FP32 round-off. Only counting
    differing elements separates them, so that is what this measures.
    """
    sys.path.insert(0, SUPPORT_PKG)
    import kernel_contract
    import torch_goldens as tg

    tg.assert_independent()
    result = kernel_contract.run(tg)
    for name, check in result["checks"].items():
        ranks.log(
            f"  {name:44s} differing={check['elements_differing']:>7d}/{check['elements']} "
            f"bit_exact={check['bit_exact']}"
        )
    for label, attribution in result["residual_attribution"].items():
        ranks.log(
            f"  residual {label:14s} {attribution['elements_differing']} flips over "
            f"{attribution['rows_with_a_flip']} of {attribution['rows']} rows "
            f"({attribution['row_width']} wide), at most "
            f"{attribution['max_flips_in_one_row']} per row -> {attribution['attribution']}"
        )
    for problem in result["problems"]:
        ranks.log(f"  PROBLEM {problem}")
    return result


def suite_activation_replay_eager(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Stage 1 / Goal 1.5. Real checkpoint activations through the SM90 backend.

    Loads the official model MP-sharded on all eight ranks exactly as
    ``reference_ladder`` does, captures the activations its own sparse-attention
    kernel consumes, and replays them through ``forward_sparse_attn_sm90`` with
    the real ``DeepseekV4CacheManager``. The reference is the official kernel's
    own output on the same rank, so this measures TensorRT-LLM's attention
    semantics with the checkpoint's real weights and real activation
    distribution rather than a Gaussian stand-in.

    ``tg`` is used only for its metric definitions here --- the reference is the
    source itself, not a pure-Torch golden --- so ``assert_independent`` does
    not apply and TensorRT-LLM is imported into this process.
    """
    import torch

    sys.path.insert(0, SUPPORT_PKG)
    import activation_replay
    import torch_goldens as tg

    prompts_manifest = _load_manifest("prompts.json")
    tol_manifest = _load_manifest("tolerances.json")
    prompts = {p["id"]: p for p in prompts_manifest["prompts"]}
    prompt = prompts[args.replay_prompt]
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")

    ranks.log(f"[activation_replay_eager] loading official source on {ranks.world} ranks")
    src = OfficialSource(args.checkpoint, ranks, max_seq_len=args.max_seq_len)
    ranks.log(
        f"  constructed {src.construct_s:.1f}s, loaded {src.load_s:.1f}s, "
        f"{src.alloc_gb:.2f} GB/rank, state-dict exact"
    )

    result: dict[str, Any] = {
        "evidence_label": "source_activation_replay",
        "reference_tier": "real_source",
        "validation_tier": "real_runtime",
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "official_mp_dir": OFFICIAL_MP_DIR,
        "world_size": ranks.world,
        "devices": _device_report(),
        "reference_env": _reference_env_report(),
        "manifest_provenance": provenance,
        "prompt": {k: prompt[k] for k in ("id", "category", "num_tokens", "thinking_mode")},
        "hard_config": {
            "attention_backend": "TRTLLM",
            "attention_entry_points": [
                "deepseek_v4.module.forward_context_sparse_attn",
                "deepseek_v4.module.forward_generation_sparse_attn",
            ],
            "kv_cache_manager": "DeepseekV4CacheManager(KVCacheManagerV2)",
            "cuda_graph": False,
            "overlap_scheduler": False,
            "max_seq_len": args.max_seq_len,
            "tokens_per_block": 128,
            # Stated exactly as constructed, because the two are not the same
            # claim. The *checkpoint* is sharded eight ways -- each rank loads
            # the official `model<rank>-mp8` shard and replays its own 8 of the
            # 64 query heads, which is the TP8 tensor geometry -- but the
            # TensorRT-LLM `Mapping` objects are rank-local, because DeepSeek-V4
            # sparse attention is head-local and issues no collective. A claim
            # of `tensor_parallel_size=8` here would assert TP8 *communication*
            # that this suite does not exercise; the LLM API suites in Stage 2
            # are what carry that.
            "replay_layers": list(args.replay_layers),
            "checkpoint_sharding": f"official model-parallel {ranks.world}",
            "query_heads_per_rank": 64 // ranks.world,
            "trtllm_mapping": "Mapping(world_size=1, tp_size=1, rank=0) per process",
            # One collective, and it is a setup step rather than a runtime one.
            # The source shards the Indexer's `wq_b`/`weights_proj` column-wise
            # and finishes `Indexer.forward` with `dist.all_reduce`;
            # TensorRT-LLM replicates both over all 64 index heads and needs no
            # runtime collective. The eight rank-local shards are therefore
            # gathered once at load. Sparse attention itself stays head-local.
            "collectives_exercised": True,
            "collectives": [
                "torch.distributed.all_gather of the official Indexer wq_b/weights_proj shards"
            ],
            "world_size": ranks.world,
            "phases": ["prefill", "cached_decode"],
        },
    }
    replayed = activation_replay.run(
        src,
        prompt,
        tol_manifest,
        ranks,
        tg,
        _SparseAttnRecorder,
        _capture,
        _judge,
        _tolerance,
        _ulp_report,
        layer_ids=tuple(args.replay_layers),
    )
    result.update(replayed)
    local_passed = (
        bool(replayed["checks"])
        and all(c["passed"] for c in replayed["checks"].values())
        and replayed["real_runtime"]["passed"]
    )
    ranks.log(f"  SM90 dispatch counts: {replayed['real_runtime']['counts']}")
    result.update(_aggregate_ranks(local_passed, replayed["checks"], ranks))
    del src
    torch.cuda.empty_cache()
    return result


def suite_sparse_kernel_numerics(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Stage-by-stage localisation of the SM90 kernel against the golden.

    One GPU, no checkpoint, a few seconds. It exists because a whole-kernel
    metric cannot say *which* arithmetic produced it: this one showed that the
    scores and the attention weights are bit-exact while the denominator and the
    output accumulator differ by reduction and contraction order, which is what
    turned "the kernel disagrees somewhere" into a localised, sweepable claim.
    """
    sys.path.insert(0, SUPPORT_PKG)
    import sparse_kernel_numerics
    import torch_goldens as tg

    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import sm90

    result = sparse_kernel_numerics.run(tg, sm90)
    stages = result["stage_agreement"]["stages"]
    for name, row in stages.items():
        ranks.log(f"  stage {name:12s} differing {row['differing']:>8d}/{row['elements']}")
    for name, row in result["exp_implementations"].items():
        if isinstance(row, dict):
            ranks.log(
                f"  exp {name:18s} differing vs torch.exp "
                f"{row['differing_vs_torch_exp']:>9d} ({100 * row['fraction']:.3f}%), "
                f"worst {row['worst_fp32_ulp']} fp32 ulp"
            )
    for axis, rows in result["variant_sweeps"].items():
        for label, row in rows.items():
            ranks.log(
                f"  sweep {axis}.{label:22s} acc {row['acc_differing']:>7d} "
                f"sum_exp {row['sum_exp_differing']:>4d}"
            )
    agree = result["output_agreement"]
    ranks.log(
        f"  output vs golden: {agree['cases_over_limit']}/{agree['cases']} cases exceed the "
        f"registered {agree['registered_mean_step_limit']} mean-step limit"
    )
    for problem in result["problems"]:
        ranks.log(f"  PROBLEM {problem}")
    return result


def suite_source_reference(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Stage 3 / Goal 3.4, reference half. The official source's tokens and logits.

    Runs the checkpoint's own model on all eight ranks and records, per
    registered prompt, ``--parity-tokens`` greedy steps and the full logit row
    behind each one. Nothing about TensorRT-LLM is measured here --- this suite
    exists so that the suite which *does* measure it can compare against a
    reference produced in a different process, under the interpreter the
    official model needs and the production runtime cannot use.

    The loop keeps decoding past EOS so every prompt reaches the same number of
    compared steps. That is only safe because the tokens it produced before EOS
    are checked against the checked-in native-``generate()`` fixture in this
    same run: the fixture is the anchor, and a loop that drifted from it would
    fail here rather than silently become the reference.
    """
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SUPPORT_PKG)
    import source_reference

    prompts_manifest = _load_manifest("prompts.json")
    prompts = {p["id"]: p for p in prompts_manifest["prompts"]}
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")
    wanted = args.prompt_ids or list(prompts)
    selected = [prompts[pid] for pid in wanted]

    ranks.log(f"[source_reference] loading official source on {ranks.world} ranks")
    src = OfficialSource(args.checkpoint, ranks, max_seq_len=args.max_seq_len)
    ranks.log(
        f"  constructed {src.construct_s:.1f}s, loaded {src.load_s:.1f}s, "
        f"{src.alloc_gb:.2f} GB/rank, state-dict exact"
    )

    captured = source_reference.capture(src, selected, args.parity_tokens, ranks)
    runs = captured["prompts"]

    # An independent reference point for the logit tolerance: how far the same
    # activations move when only the head's dtype changes. Recorded next to the
    # reference it belongs to, so a later logit comparison can attribute its
    # own difference instead of only reporting it.
    import torch_goldens as tg

    head_probe = source_reference.head_precision_probe(src, selected, ranks, tg.compare)

    # The official model is model-parallel: every rank runs every forward and
    # ends with the same all-gathered logits, so their tokens must be identical.
    # Ranks that disagree would make the reference depend on which rank wrote it.
    digest = hashlib.sha256(
        json.dumps({pid: run["tokens"] for pid, run in runs.items()}, sort_keys=True).encode()
    ).hexdigest()
    digests = [None] * ranks.world
    if ranks.world > 1 and dist.is_initialized():
        dist.all_gather_object(digests, digest)
    else:
        digests = [digest]
    rank_agreement = len(set(digests)) == 1

    # The anchor. The fixture stops at EOS and this capture does not, so the
    # comparison is over the fixture's own length --- truncation stated here
    # rather than hidden inside the comparator, which keeps its hard equality.
    with open(os.path.join(MANIFEST_DIR, "native_generate_golden.json")) as f:
        golden = json.load(f)
    truncated = {
        pid: {
            "tokens": run["tokens"][: len((golden["prompts"].get(pid) or {}).get("tokens") or [])]
        }
        for pid, run in runs.items()
    }
    anchor = _check_native_golden(truncated, ranks)

    result: dict[str, Any] = {
        "evidence_label": "source_reference_capture",
        "reference_tier": "real_source",
        "validation_tier": "integration",
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "official_mp_dir": OFFICIAL_MP_DIR,
        "world_size": ranks.world,
        "parity_tokens": args.parity_tokens,
        "devices": _device_report(),
        "reference_env": _reference_env_report(),
        "manifest_provenance": provenance,
        "decoding": prompts_manifest["decoding"],
        "hard_config": {
            "loop": "OfficialSource.greedy(stop_at_eos=False)",
            "max_seq_len": args.max_seq_len,
            "max_batch_size": 1,
            "expert_dtype": src.args.expert_dtype,
            "dtype": src.args.dtype,
            "scale_dtype": src.args.scale_dtype,
        },
        "prompts": runs,
        "rank_token_digests": digests,
        "rank_agreement": rank_agreement,
        "head_precision_probe": head_probe,
        "native_generate_golden": anchor,
    }
    problems = []
    if not rank_agreement:
        problems.append(f"ranks disagree on the reference tokens: {digests}")
    if not anchor["passed"]:
        problems.append(f"native-generate anchor failed: {anchor['problems'][:3]}")
    for pid, run in runs.items():
        if not run["logits_finite"] or not run["prefill_logits_finite"]:
            problems.append(f"{pid}: non-finite reference logits")
    result["problems"] = problems
    result["passed"] = not problems

    if ranks.is_main:
        source_reference.write(args.output, result, captured["logits"])
        ranks.log(
            f"  wrote {result['logits_sidecar']['bytes'] / 1e6:.1f} MB of logits to "
            f"{result['logits_sidecar']['path']}"
        )
    del src
    torch.cuda.empty_cache()
    return result


def suite_eager_full_model(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Stage 3 / Goals 3.3 and 3.4. The eager full model, judged against the source.

    The task's first completion criterion is about the *entry point*: the
    unmodified checkpoint loading through ``LLM`` with ``backend=pytorch``,
    TP8/EP8 and ``custom_tokenizer=deepseek_v4``, with no pre-Blackwell
    rejection, no SM100-only kernel selection, no rank hang and no OOM. The two
    that follow are about what that entry point *computes*: the same greedy
    next token as the checkpoint's own implementation, and the same tokens step
    after step for a whole generation.

    Three process facts shape the suite. ``LLM(tensor_parallel_size=8)`` spawns
    its own eight-process MPI world, so only the launcher's rank 0 constructs
    it and the other ranks release their CUDA contexts and exit first. The
    official source cannot share this process --- it needs the reference
    interpreter, which breaks this runtime --- so its tokens and logits are
    captured by a nested eight-rank job *before* anything is built here, and
    read back from disk. And the construction runs with stdout and stderr
    redirected into a log file, because that file is the only channel carrying
    worker-side truth back: the MPI proxy refuses ``collective_rpc`` above
    ``model_world_size=1``.
    """
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SUPPORT_PKG)
    import full_model
    import source_reference

    prompts_manifest = _load_manifest("prompts.json")
    tol_manifest = _load_manifest("tolerances.json")
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")
    wanted = tuple(args.prompt_ids) if args.prompt_ids else None
    selected = [p for p in prompts_manifest["prompts"] if wanted is None or p["id"] in wanted]
    missing = set(wanted or ()) - {p["id"] for p in selected}
    if missing:
        raise RuntimeError(f"unregistered prompt ids: {sorted(missing)}")

    # Which prompts carry the exact rules, and which are recorded but do not
    # gate, is the registered fixture's decision -- taken once, with a written
    # reason, and not re-taken here. `chat_arithmetic` is the non-gating one:
    # the official head computes the LM projection in FP32 and every BF16 head
    # ranks its top three candidates differently at step 0. It is still run,
    # still compared, and still reported with its true result.
    with open(os.path.join(MANIFEST_DIR, "native_generate_golden.json")) as f:
        golden = json.load(f)
    gating_ids = sorted(golden["required_prompt_ids"])
    non_gating_ids = sorted(golden.get("non_gating_prompt_ids") or [])

    # Every rank reports its device before the world is dissolved; after this
    # point only rank 0 exists as far as this suite is concerned.
    devices = _device_report()
    if ranks.world > 1 and dist.is_initialized():
        dist.barrier()

    # Ordering is load-bearing. ``MPI_Comm_spawn`` hands each worker the
    # environment this process had when MPI initialised, which happens on the
    # first ``tensorrt_llm`` import, so the launcher detach has to precede that
    # import --- and this suite has not imported it yet on any rank.
    stripped = full_model.detach_from_launcher()
    if ranks.world > 1 and dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()

    if not ranks.is_main:
        # Releasing the process is the point: its CUDA context is memory the
        # spawned worker on this GPU is about to need.
        return {
            "evidence_label": "llm_api_construction",
            "role": "standby",
            "rank": ranks.rank,
            "passed": True,
        }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    log_path = os.path.join(out_dir, "eager_full_model.log")
    os.makedirs(out_dir, exist_ok=True)

    # The reference, first and elsewhere. It has to exist before the runtime is
    # built --- both want all eight GPUs, and the interpreter that can run the
    # official model cannot run this one --- and it has to describe *this*
    # measurement, which `source_reference.usable` is what checks.
    reference_path = args.reuse_reference or os.path.join(out_dir, "source_reference.json")
    capture: dict[str, Any] = {"reused": bool(args.reuse_reference), "artifact": reference_path}
    if not args.reuse_reference:
        ranks.log(f"[eager_full_model] capturing the official-source reference -> {reference_path}")
        capture.update(
            full_model.capture_reference(args, os.path.abspath(__file__), reference_path)
        )
        ranks.log(f"  reference captured in {capture['elapsed_s']}s")
    reference, ref_logits = source_reference.load(reference_path)
    stale = source_reference.usable(
        reference,
        checkpoint_revision=prompts_manifest["checkpoint_revision"],
        prompts_sha256=provenance["sha256"]["prompts.json"],
        parity_tokens=args.parity_tokens,
        prompt_ids=[p["id"] for p in selected],
    )
    if stale:
        raise RuntimeError(f"the official-source reference cannot judge this run: {stale}")
    capture["reference_provenance"] = {
        "checkpoint_revision": reference["checkpoint_revision"],
        "parity_tokens": reference["parity_tokens"],
        "world_size": reference["world_size"],
        "loop": reference["hard_config"]["loop"],
        "native_generate_golden_passed": reference["native_generate_golden"]["passed"],
        "rank_agreement": reference["rank_agreement"],
        "logits_sidecar": reference["logits_sidecar"],
        "reference_env": reference["reference_env"],
        "elapsed_s": reference.get("elapsed_s"),
    }

    result: dict[str, Any] = {
        "evidence_label": ["llm_api_construction", "source_logit_replay", "generation_parity"],
        "reference_tier": "real_source",
        "validation_tier": "real_runtime",
        "source_reference": capture,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "world_size": ranks.world,
        "devices": devices,
        # The pinned container environment, which is the runtime this suite is
        # about. The reference venv's extra kernel dependencies are absent here
        # on purpose --- see the note on NEEDS_REFERENCE_ENV.
        "runtime_env": {
            "interpreter": sys.executable,
            "float32_precision": _float32_precision_report(),
        },
        "manifest_provenance": provenance,
        "launcher_env_detached": stripped,
        "hard_config": {
            "entry_point": "tensorrt_llm.LLM",
            "backend": "pytorch",
            "tensor_parallel_size": 8,
            "moe_expert_parallel_size": 8,
            "custom_tokenizer": "deepseek_v4",
            "attention_backend": "TRTLLM",
            "kv_cache_manager": "DeepseekV4CacheManager(KVCacheManagerV2)",
            "cuda_graph": False,
            "overlap_scheduler": False,
            "max_seq_len": args.max_seq_len,
            "tokens_per_block": 128,
            "free_gpu_memory_fraction": args.kv_fraction,
            "max_num_tokens": args.max_num_tokens,
            "decoding": "deterministic greedy (temperature 0, top_k 1, no sampling, ignore_eos)",
            "parity_tokens": args.parity_tokens,
            "worker_launch": "MPI_Comm_spawn from launcher rank 0",
            "reference": "official inference/model.py, captured out of process",
        },
        "gating_prompt_ids": gating_ids,
        "non_gating_prompt_ids": non_gating_ids,
        "non_gating_reason": golden.get("non_gating_reason"),
        "prompts": [
            {k: p[k] for k in ("id", "category", "num_tokens", "thinking_mode")} for p in selected
        ],
    }
    print(f"[eager_full_model] constructing at TP8/EP8; worker log -> {log_path}", flush=True)
    # The redirection has to precede the first ``tensorrt_llm`` import as well:
    # MPI initialises there, and OpenMPI's daemon --- which forwards the spawned
    # workers' output --- inherits the descriptors this process holds at that
    # moment. Redirecting afterwards would leave the worker log going to the
    # launcher's console and this file nearly empty.
    saved = (os.dup(1), os.dup(2))
    with open(log_path, "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            # Also before the import, for the same snapshot reason: the workers
            # read their log level from the environment they were handed.
            os.environ["TLLM_LOG_LEVEL"] = "info"
            result["linked_package"] = _linked_package_report()
            built = full_model.run(
                args,
                ranks,
                selected,
                log_path,
                reference,
                ref_logits,
                _tolerance(tol_manifest, "final_logits"),
                tol_manifest["gates"],
                gating_ids,
                non_gating_ids,
            )
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
    result.update(built)

    for name, check in result["checks"].items():
        ranks.log(f"  {name}: passed={check['passed']}")
        for problem in check.get("problems", []) or []:
            ranks.log(f"    PROBLEM {problem}")
    contract = result["checks"]["runtime_contract"]
    ranks.log(f"  constructed in {contract['construct_s']}s")
    ranks.log(f"  memory {result.get('memory')}")
    for pid, detail in result["checks"]["source_logit_replay"]["per_prompt"].items():
        ranks.log(
            f"  logit replay {pid:24s} gating={detail['gating']} "
            f"argmax_match={detail.get('argmax_match')} cosine={detail.get('cosine')} "
            f"rel_max_abs={detail.get('rel_max_abs')}"
        )
    for pid, detail in result["checks"]["generation_parity"]["per_prompt"].items():
        ranks.log(
            f"  parity       {pid:24s} gating={detail['gating']} "
            f"steps={detail.get('steps_compared')} divergence={detail.get('first_divergence')} "
            f"repeat_ok={detail.get('repeat_identical')} worst_cos={detail.get('worst_step_cosine')}"
        )
    for pid, entry in result["measured"].items():
        ranks.log(f"  {pid}: {entry['text']!r}")
    return result


def suite_load_and_moe(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Stage 3 / Goals 3.1 and 3.2. TP8/EP8 load accounting, then MoE replay.

    Loads the *unmodified* checkpoint through the production
    ``DeepseekV4WeightLoader`` on all eight ranks and then accounts for what the
    load did: which checkpoint tensors it read, where each remapped tensor
    landed, whether the routed-expert bytes handed to the fused-MoE method are
    the bytes on disk, whether the packed container and the group-of-32 UE8M0
    scales survived, whether the dense FP8/BF16 contract survived, and what it
    cost in host and device memory.

    The *same loaded model* then runs Goal 3.2's MoE ``source_activation_replay``:
    real hidden states captured from the official model at one hash-routed and
    one score-routed layer, driven through ``DeepseekV4MoE`` on the SM90
    Cutlass W4A16-MXFP4 path. Sharing the model is the point --- the weights the
    replay measures are the weights this suite just audited, not a second load
    nobody checked.

    Every rank reports its resolved MoE fingerprint and they must agree: a
    resolver that differs across ranks does not fail here, it deadlocks at the
    first routed collective.
    """
    import torch

    sys.path.insert(0, SUPPORT_PKG)
    import checkpoint_inventory
    import load_accounting
    import moe_replay
    import torch_goldens as tg

    prompts_manifest = _load_manifest("prompts.json")
    tol_manifest = _load_manifest("tolerances.json")
    prompts = {p["id"]: p for p in prompts_manifest["prompts"]}
    prompt = prompts[args.replay_prompt]
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")

    ranks.log(f"[load_and_moe] loading the checkpoint on {ranks.world} ranks at TP8/EP8")
    loaded, live = load_accounting.run(args, ranks, checkpoint_inventory)

    result: dict[str, Any] = {
        "evidence_label": ["mixed_precision_load_and_state_accounting", "source_activation_replay"],
        "reference_tier": "real_source",
        "validation_tier": "real_runtime",
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "official_mp_dir": OFFICIAL_MP_DIR,
        "world_size": ranks.world,
        "devices": _device_report(),
        "reference_env": _reference_env_report(),
        "linked_package": _linked_package_report(),
        "manifest_provenance": provenance,
        "prompt": {k: prompt[k] for k in ("id", "category", "num_tokens", "thinking_mode")},
    }
    result.update({k: v for k, v in loaded.items() if k != "local_passed"})

    for name, check in loaded["checks"].items():
        ranks.log(f"  {name}: passed={check['passed']}")
        for problem in check.get("problems", []) or []:
            ranks.log(f"    PROBLEM {problem}")
    consumption = loaded["checks"]["raw_tensor_consumption"]
    ranks.log(
        f"  read {consumption['read']}/{consumption['checkpoint_tensors']} checkpoint tensors, "
        f"never read {consumption['never_read']}"
    )
    ranks.log(f"  memory {loaded['memory']}")
    load_agg = _aggregate_load_ranks(loaded, ranks)

    # Every dense and shared-expert Linear on this path JIT-compiles through
    # DeepGEMM's NVRTC client, which tilelang's stub aborts unless the real
    # library is global. Checked before the source load rather than after,
    # because the failure mode is SIGABRT minutes later, not an exception.
    if not _NVRTC_PRELOAD.get("loaded"):
        raise RuntimeError(
            f"no real libnvrtc was preloaded ({_NVRTC_PRELOAD}); TensorRT-LLM's FP8 "
            "block-scale GEMM would abort inside tilelang's NVRTC stub"
        )

    ranks.log(f"[load_and_moe] loading official source on {ranks.world} ranks for the MoE replay")
    src = OfficialSource(args.checkpoint, ranks, max_seq_len=args.max_seq_len)
    ranks.log(
        f"  constructed {src.construct_s:.1f}s, loaded {src.load_s:.1f}s, "
        f"{src.alloc_gb:.2f} GB/rank, state-dict exact"
    )
    replayed = moe_replay.run(
        args,
        ranks,
        live,
        src,
        prompt,
        tol_manifest,
        tg,
        _judge,
        _tolerance,
        _ulp_report,
        _capture,
    )
    del src
    torch.cuda.empty_cache()

    moe_agg = _aggregate_ranks(replayed["local_passed"], replayed["module_goldens"], ranks)
    result.update({k: v for k, v in replayed.items() if k != "local_passed"})
    result["memory_after_moe_replay"] = load_accounting.memory_report()
    result.update(_merge_rank_verdicts(load_agg, moe_agg))
    torch.cuda.empty_cache()
    return result


def _selected_prompts(
    args: argparse.Namespace, manifest: dict[str, Any], default: Sequence[str]
) -> list[dict[str, Any]]:
    """The registered prompt records the caller asked for, in their order."""
    registered = {p["id"]: p for p in manifest["prompts"]}
    wanted = list(args.prompt_ids or default)
    unknown = [pid for pid in wanted if pid not in registered]
    if unknown:
        raise RuntimeError(f"unregistered prompt ids: {unknown}")
    return [registered[pid] for pid in wanted]


def _deep_layers(args: argparse.Namespace, module: Any) -> list[int]:
    """Layers the sub-boundary split drives, as both halves must agree on them.

    Both phases read it from the same flag, and the capture records what it
    used --- a production replay that split layer 0 while the capture split
    layer 2 would compare two different boundaries under one name.
    """
    chosen = args.localize_layers if args.localize_layers is not None else module.DEFAULT_DEEP_LAYERS
    return sorted({int(lid) for lid in chosen})


def _host_barrier(ranks: Ranks) -> None:
    """Wait for every rank without spinning a kernel on any GPU.

    NCCL's barrier busy-waits with a live kernel per rank. Here the ranks are
    waiting for rank 0 to finish an eight-rank *child* job that needs those
    same GPUs, so seven spinning kernels would be competing with the work they
    are waiting for --- and the wait outlasts the collective watchdog besides.
    A gloo group blocks on the host instead.
    """
    import datetime

    import torch.distributed as dist

    if ranks.world <= 1 or not dist.is_initialized():
        return
    try:
        group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=2))
    except (RuntimeError, ValueError):  # no gloo in this build; spin rather than hang
        dist.barrier()
        return
    try:
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group(group)


def suite_layer_source_capture(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Diagnostic, phase 1 of 2: the official source's per-layer boundaries.

    Runs in the isolated reference interpreter, which is the only one that can
    import the official model, and touches no production module --- so the
    reference venv's ``tvm_ffi``, which makes flashinfer's CuTe RMSNorm raise,
    is never in the path of anything under test. What it saw is written to a
    per-rank sidecar with the provenance :func:`suite_layer_localization` needs
    to refuse it: checkpoint revision, prompt-manifest hash, world size, and
    per-tensor shape, dtype and SHA-256.
    """
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SUPPORT_PKG)
    import layer_localization

    prompts_manifest = _load_manifest("prompts.json")
    prompts = _selected_prompts(args, prompts_manifest, layer_localization.DEFAULT_PROMPTS)
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")

    deep_layers = _deep_layers(args, layer_localization)
    ranks.log(f"[layer_source_capture] loading official source on {ranks.world} ranks")
    src = OfficialSource(args.checkpoint, ranks, max_seq_len=args.max_seq_len)
    ranks.log(f"  constructed {src.construct_s:.1f}s, loaded {src.load_s:.1f}s")
    ranks.log(f"  splitting layers {deep_layers} at every observable sub-boundary")
    captured = layer_localization.capture_all(ranks, src, prompts, _capture, deep_layers)
    ratios = list(src.args.compress_ratios)
    num_layers = len(src.model.layers)
    del src
    torch.cuda.empty_cache()

    sidecar = layer_localization.save_capture(
        args.output,
        ranks.rank,
        captured,
        meta={
            "checkpoint_revision": prompts_manifest["checkpoint_revision"],
            "prompts_sha256": provenance["sha256"]["prompts.json"],
            "prompt_ids": [p["id"] for p in prompts],
            "deep_layers": list(deep_layers),
            "rank": ranks.rank,
            "world_size": ranks.world,
            "interpreter": sys.executable,
        },
    )
    print(
        f"  rank {ranks.rank} wrote {sidecar['bytes'] / 1e6:.1f} MB -> {sidecar['path']}",
        flush=True,
    )

    gathered: list[Any] = [sidecar]
    if ranks.world > 1 and dist.is_initialized():
        gathered = [None] * ranks.world
        dist.all_gather_object(gathered, sidecar)
    per_rank = {str(entry["rank"]): entry for entry in gathered}

    problems = []
    for rank in range(ranks.world):
        entry = per_rank.get(str(rank))
        if entry is None:
            problems.append(f"rank {rank} produced no sidecar")
        elif not entry["all_finite"]:
            problems.append(f"rank {rank} captured a non-finite boundary")
    # The residual stack is replicated across a model-parallel world, so the
    # ranks must at least agree on its *shape*; a rank that disagrees is
    # sharding a tensor the comparison assumes whole.
    shapes = {
        json.dumps(
            {name: entry["tensors"][name]["shape"] for name in sorted(entry["tensors"])},
            sort_keys=True,
        )
        for entry in per_rank.values()
    }
    if len(shapes) > 1:
        problems.append(f"ranks captured {len(shapes)} different boundary shapes")

    return {
        "evidence_label": ["source_activation_replay"],
        "reference_tier": "real_source",
        "validation_tier": "real_runtime",
        "phase": "source_capture",
        "diagnostic_only": True,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "world_size": ranks.world,
        "devices": _device_report(),
        "reference_env": _reference_env_report(),
        "linked_package": _linked_package_report(),
        "manifest_provenance": provenance,
        "prompts": [{k: p[k] for k in ("id", "category", "num_tokens")} for p in prompts],
        "compress_ratios": ratios,
        "num_layers": num_layers,
        "deep_layers": list(deep_layers),
        "sublayer_order": list(layer_localization.SUBLAYER_ORDER),
        "per_rank": per_rank,
        "identical_boundary_shapes": len(shapes) == 1,
        "memory": {
            "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        },
        "problems": problems,
        "passed": not problems,
    }


def suite_layer_localization(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Diagnostic, phase 2 of 2: where does the production stack leave the source?

    The MoE replay passes its registered limits by a wide margin and the LM
    head's own dtype cost is smaller still, yet ``eager_full_model``'s logits
    miss theirs by an order of magnitude. Those three facts cannot all be about
    one layer, so this suite measures the quantity that connects them: the mHC
    residual stack after *every* decoder layer, on both stacks, for the same
    tokens.

    Runs in the *pinned* interpreter, because it executes production modules
    and the reference venv is not the runtime anybody ships. The source half is
    therefore captured out of process first --- a nested eight-rank
    ``layer_source_capture`` job launched from rank 0 while every rank is still
    empty, so the official model has the GPUs to itself --- and read back per
    rank once this rank's production model is loaded.

    It gates nothing: the frozen manifest registers no intermediate-boundary
    tolerance and inventing one here would be registering a tolerance after
    measuring. ``passed`` reports only that the source capture could judge this
    run and that both stacks produced finite numbers on every rank; the reading
    is in the per-layer curve.
    """
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SUPPORT_PKG)
    import checkpoint_inventory
    import layer_localization
    import load_accounting
    import torch_goldens as tg

    prompts_manifest = _load_manifest("prompts.json")
    prompts = _selected_prompts(args, prompts_manifest, layer_localization.DEFAULT_PROMPTS)
    wanted = [p["id"] for p in prompts]
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.output))[0]
    capture_path = args.reuse_source_capture or os.path.join(out_dir, f"{stem}.source_capture.json")
    capture: dict[str, Any] = {"reused": bool(args.reuse_source_capture), "artifact": capture_path}
    deep_layers = _deep_layers(args, layer_localization)
    if not args.reuse_source_capture:
        ranks.log(f"[layer_localization] capturing the source boundaries -> {capture_path}")
        failure = None
        if ranks.is_main:
            try:
                capture.update(
                    layer_localization.capture_job(
                        args, os.path.abspath(__file__), capture_path, wanted, deep_layers
                    )
                )
            except RuntimeError as exc:
                # Reach the barrier either way: the other seven ranks are
                # waiting on it, and hanging them is worse than failing here.
                failure = str(exc)
        _host_barrier(ranks)
        if failure:
            raise RuntimeError(failure)
        ranks.log(f"  source captured in {capture.get('elapsed_s')}s")

    artifact = layer_localization.read_artifact(capture_path)
    stale = layer_localization.capture_usable(
        artifact,
        checkpoint_revision=prompts_manifest["checkpoint_revision"],
        prompts_sha256=provenance["sha256"]["prompts.json"],
        prompt_ids=wanted,
        world_size=ranks.world,
        deep_layers=deep_layers,
    )
    if stale:
        raise RuntimeError(f"the source capture cannot judge this run: {stale}")
    source = layer_localization.load_capture(artifact, ranks.rank)
    mine = artifact["per_rank"][str(ranks.rank)]
    capture["provenance"] = {
        "checkpoint_revision": artifact["checkpoint_revision"],
        "world_size": artifact["world_size"],
        "prompts": [p["id"] for p in artifact["prompts"]],
        "identical_boundary_shapes": artifact["identical_boundary_shapes"],
        "reference_env": artifact["reference_env"],
        "elapsed_s": artifact.get("elapsed_s"),
        "this_rank_sidecar": {k: v for k, v in mine.items() if k != "tensors"},
        "this_rank_tensor_count": len(mine["tensors"]),
    }

    ranks.log(f"[layer_localization] loading the checkpoint on {ranks.world} ranks at TP8/EP8")
    loaded, live = load_accounting.run(args, ranks, checkpoint_inventory)
    if not loaded["local_passed"]:
        ranks.log("  WARNING load accounting reported problems; the curve is still measured")

    ratios = list(live.model_config.pretrained_config.compress_ratios)
    num_layers = live.model_config.pretrained_config.num_hidden_layers
    mismatched = layer_localization.ratio_problems(ratios, artifact.get("compress_ratios"), num_layers)
    if mismatched:
        raise RuntimeError(f"the two stacks disagree about compression: {mismatched}")

    ranks.log("[layer_localization] running the production prefills")
    per_prompt = layer_localization.compare_all(
        live, prompts, source, ratios, tg, ranks, deep_layers
    )

    finite = all(
        entry["finite"]
        for result in per_prompt.values()
        for entry in result["per_layer"] + [result["boundaries"]["logits"]]
    )
    local_passed = bool(per_prompt) and finite
    # Every rank runs the same forwards, so a rank that produced a nonfinite
    # boundary has to fail the suite rather than be invisible behind rank 0.
    if ranks.world > 1 and dist.is_initialized():
        verdicts: list[Any] = [None] * ranks.world
        dist.all_gather_object(verdicts, local_passed)
    else:
        verdicts = [local_passed]
    result: dict[str, Any] = {
        "evidence_label": ["source_activation_replay"],
        "reference_tier": "real_source",
        "validation_tier": "real_runtime",
        "phase": "production_replay",
        "diagnostic_only": True,
        "why_not_a_gate": (
            "the frozen manifest registers no intermediate decoder-layer tolerance; "
            "registering one after measuring is exactly what the human froze"
        ),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "world_size": ranks.world,
        "devices": _device_report(),
        # The pinned container environment, which is the runtime this suite is
        # about; the source half's environment is under ``source_capture``.
        "runtime_env": {
            "interpreter": sys.executable,
            "float32_precision": _float32_precision_report(),
        },
        "source_capture": capture,
        "linked_package": _linked_package_report(),
        "manifest_provenance": provenance,
        "prompts": [{k: p[k] for k in ("id", "category", "num_tokens")} for p in prompts],
        "compress_ratios": ratios[:num_layers],
        "deep_layers": deep_layers,
        "sublayer_order": list(layer_localization.SUBLAYER_ORDER),
        "attention_gemm": layer_localization.attention_gemm_report(live, deep_layers),
        "per_prompt": per_prompt,
        "memory": load_accounting.memory_report(),
        "ranks_failed": [i for i, ok in enumerate(verdicts) if not ok],
        "passed": all(verdicts),
    }
    torch.cuda.empty_cache()
    return result


def suite_eager_state_determinism(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Stage 4 / Goal 4.1. Does the same request give the same answer, twice over?

    The acceptance item asks for two separable things and this suite measures
    both. *Same engine*: three executions of ``cache_boundary_257`` and three
    of ``long_prefill_2304`` on one ``LLM``, each with a different history
    behind it, must produce identical tokens and identical logits. *Fresh
    engine*: two more ``LLM`` instances, each replaying the same prefix, must
    agree with the first engine's execution of the same prompt. And after every
    request has finished, the runtime must report no KV block still held.

    Three constructions, in one process, in sequence --- the second cannot
    start until the first has released its eight GPUs. Each gets its own log
    file, because that file is where its spawned workers' dispatch markers go
    and a shared one could not attribute them.

    The comparison is TensorRT-LLM against TensorRT-LLM, so no official-source
    reference is captured here. That is deliberate: this criterion is about
    reproducibility, and a run that disagrees with itself has already failed
    without any reference to appeal to.
    """
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SUPPORT_PKG)
    import full_model
    import state_determinism
    import torch_goldens as tg

    prompts_manifest = _load_manifest("prompts.json")
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")
    registered = {p["id"]: p for p in prompts_manifest["prompts"]}

    devices = _device_report()
    if ranks.world > 1 and dist.is_initialized():
        dist.barrier()

    # Same ordering constraint as ``eager_full_model``: the launcher detach has
    # to precede the first ``tensorrt_llm`` import, because MPI snapshots the
    # environment its spawned workers receive at that moment.
    stripped = full_model.detach_from_launcher()
    if ranks.world > 1 and dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()

    if not ranks.is_main:
        return {
            "evidence_label": "eager_state_determinism",
            "role": "standby",
            "rank": ranks.rank,
            "passed": True,
        }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.output))[0]
    os.makedirs(out_dir, exist_ok=True)

    plan = [
        ("same_engine", state_determinism.SAME_ENGINE_SEQUENCE),
        ("fresh_engine_a", state_determinism.FRESH_ENGINE_SEQUENCE),
        ("fresh_engine_b", state_determinism.FRESH_ENGINE_SEQUENCE),
    ]
    passes: list[dict[str, Any]] = []
    # One log for all three engines, and one redirection around all three. MPI
    # fixes the descriptors its spawned workers inherit at the first
    # ``tensorrt_llm`` import, so re-pointing between engines moves only this
    # process's own output and leaves every worker writing to the first file
    # --- which the first run of this suite duly reported as "0 of 8 ranks
    # logged" for engines two and three. See ``worker_dispatch``.
    log_path = os.path.join(out_dir, f"{stem}.workers.log")
    print(f"[eager_state_determinism] worker log -> {log_path}", flush=True)
    saved = (os.dup(1), os.dup(2))
    with open(log_path, "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            os.environ["TLLM_LOG_LEVEL"] = "info"
            for tag, sequence in plan:
                passes.append(
                    state_determinism.executor_pass(
                        args, full_model, registered, sequence, log_path, tag
                    )
                )
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
    for entry in passes:
        ranks.log(
            f"  {entry['engine']}: constructed in {entry['construct_s']}s, "
            f"{len(entry['runs'])} requests, teardown {entry['teardown'].get('last')}"
        )

    # Read only after every engine has shut down: the workers write to this
    # file until they exit, and a scan taken mid-run would judge a partial log.
    dispatch = state_determinism.worker_dispatch(full_model, log_path)
    for entry in passes:
        entry["worker_dispatch"] = dispatch
    comparison = state_determinism.compare_executions(passes, tg)
    problems = state_determinism.judge_executions(passes, comparison)
    for c in comparison["comparisons"] + comparison["diagnostics"]:
        gate = "gate " if c["prompt_id"] in state_determinism.GATING_PROMPTS else "diag "
        ranks.log(
            f"  {gate}{c['prompt_id']:20s} {c['execution']:16s} vs {c['anchor']:16s} "
            f"tokens_identical={c['tokens_identical']} logits_identical={c['logits_identical']} "
            f"step0_max_abs={c['step0_max_abs']}"
        )
    if comparison["non_gating_not_reproducible"]:
        ranks.log(
            f"  NOTE the non-gating primer prompts did not reproduce: "
            f"{comparison['non_gating_not_reproducible']} (recorded, not gating)"
        )
    for problem in problems:
        ranks.log(f"  PROBLEM {problem}")

    return {
        "evidence_label": ["eager_state_determinism"],
        "reference_tier": "real_source",
        "validation_tier": "real_runtime",
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "world_size": ranks.world,
        "devices": devices,
        "runtime_env": {
            "interpreter": sys.executable,
            "float32_precision": _float32_precision_report(),
        },
        "linked_package": _linked_package_report(),
        "manifest_provenance": provenance,
        "launcher_env_detached": stripped,
        "hard_config": {
            "entry_point": "tensorrt_llm.LLM",
            "backend": "pytorch",
            "tensor_parallel_size": 8,
            "moe_expert_parallel_size": 8,
            "custom_tokenizer": "deepseek_v4",
            "attention_backend": "TRTLLM",
            "kv_cache_manager": "DeepseekV4CacheManager(KVCacheManagerV2)",
            "cuda_graph": False,
            "overlap_scheduler": False,
            "enable_block_reuse": False,
            "enable_chunked_prefill": False,
            "max_seq_len": args.max_seq_len,
            "tokens_per_block": 128,
            "free_gpu_memory_fraction": args.kv_fraction,
            "max_num_tokens": args.max_num_tokens,
            "decoding": "deterministic greedy (temperature 0, top_k 1, no sampling, ignore_eos)",
            "parity_tokens": args.parity_tokens,
        },
        "gating_prompt_ids": list(state_determinism.GATING_PROMPTS),
        "worker_dispatch": dispatch,
        "worker_log": log_path,
        "engines": [
            {k: v for k, v in entry.items() if k != "runs"}
            | {
                "runs": [
                    {k: v for k, v in run.items() if k != "logits"} for run in entry["runs"]
                ]
            }
            for entry in passes
        ],
        "comparison": comparison,
        "memory": full_model.memory_report(),
        "problems": problems,
        "passed": not problems,
    }


def suite_state_lifecycle(args: argparse.Namespace, ranks: Ranks) -> dict[str, Any]:
    """Stage 4 / Goal 4.1, diagnostic: which shared channel carries request history?

    ``eager_full_model`` measured the same prompt twice --- same weights, same
    reference, same hard config, every request a one-item ``llm.generate`` ---
    and got rel 0.637 when it ran first and rel 0.828 when another prompt had
    run before it. That is state a finished request left behind, and the
    executor is the worst place to look for it: its workers are eight spawned
    processes whose only channel back is a log file.

    So the same question is asked in process. This suite loads the checkpoint
    at TP8/EP8 once and drives a scripted prompt sequence through five
    configurations that differ *only* in what they share between requests: a
    control that shares nothing, one that shares the attention metadata, one
    that shares the cache manager, one that shares both (which is what
    ``PyExecutor`` does), and one that shares both but memsets every page a
    request owned when it is freed. The pattern across those five names the
    channel; ``state_determinism.reading`` states it in one sentence.

    Diagnostic, not a gate: the registered manifest has no tolerance for
    "TensorRT-LLM against itself", and the acceptance item's own eager-state
    evidence is the LLM API suite. ``passed`` here means the control was
    reproducible and no request outlived ``free_resources`` in the cache
    manager's own bookkeeping --- both of which are prerequisites for the
    reading to mean anything.
    """
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SUPPORT_PKG)
    import checkpoint_inventory
    import load_accounting
    import state_determinism
    import torch_goldens as tg

    prompts_manifest = _load_manifest("prompts.json")
    provenance = _manifest_provenance(args.started_at)
    ranks.log(f"  manifests verified {provenance['sha256']}")
    registered = {p["id"]: p for p in prompts_manifest["prompts"]}
    sequence = list(args.state_sequence or state_determinism.DEFAULT_SEQUENCE)
    modes = [m for m in state_determinism.MODES if not args.state_modes or m["name"] in args.state_modes]
    if not modes:
        raise RuntimeError(
            f"--state-modes selected nothing; known modes are "
            f"{[m['name'] for m in state_determinism.MODES]}"
        )

    ranks.log(f"[state_lifecycle] loading the checkpoint on {ranks.world} ranks at TP8/EP8")
    loaded, live = load_accounting.run(args, ranks, checkpoint_inventory)
    if not loaded["local_passed"]:
        ranks.log("  WARNING load accounting reported problems; the sequence is still measured")

    measured = state_determinism.run_all(
        live,
        registered,
        sequence,
        tg,
        ranks,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        pool_tokens=args.state_pool_tokens,
        modes=modes,
    )

    local_passed = not measured["problems"]
    if ranks.world > 1 and dist.is_initialized():
        verdicts: list[Any] = [None] * ranks.world
        dist.all_gather_object(verdicts, local_passed)
    else:
        verdicts = [local_passed]

    ranks.log(f"[state_lifecycle] reading: {measured['reading']}")
    for problem in measured["problems"]:
        ranks.log(f"  PROBLEM {problem}")

    result: dict[str, Any] = {
        "evidence_label": ["eager_state_determinism"],
        "reference_tier": "static",
        "validation_tier": "real_runtime",
        "phase": "in_process_lifecycle",
        "diagnostic_only": True,
        "why_not_a_gate": (
            "this compares TensorRT-LLM against itself; the acceptance item's eager-state "
            "evidence is the LLM API suite, and no registered tolerance describes a "
            "self-comparison"
        ),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "world_size": ranks.world,
        "devices": _device_report(),
        "runtime_env": {
            "interpreter": sys.executable,
            "float32_precision": _float32_precision_report(),
        },
        "linked_package": _linked_package_report(),
        "manifest_provenance": provenance,
        "hard_config": {
            "entry_point": "DeepseekV4CacheManager + DeepseekV4TrtllmAttentionMetadata, in process",
            "tensor_parallel_size": ranks.world,
            "attention_backend": "TRTLLM",
            "kv_cache_manager": "DeepseekV4CacheManager(KVCacheManagerV2)",
            "cuda_graph": False,
            "overlap_scheduler": False,
            "enable_block_reuse": False,
            "max_seq_len": args.max_seq_len,
            "tokens_per_block": 128,
            "max_batch_size": args.max_batch_size,
            "pool_tokens": args.state_pool_tokens,
        },
        "compress_ratios": list(live.model_config.pretrained_config.compress_ratios),
        "probe_layers": list(state_determinism.PROBE_LAYERS),
        "memory": load_accounting.memory_report(),
        "ranks_failed": [i for i, ok in enumerate(verdicts) if not ok],
        "passed": all(verdicts),
    }
    result.update(measured)
    torch.cuda.empty_cache()
    return result


def _merge_rank_verdicts(load_agg: dict[str, Any], moe_agg: dict[str, Any]) -> dict[str, Any]:
    """Fold the load accounting's rank verdicts and the MoE replay's into one.

    Both halves gather per-rank verdicts, and both use the same field names, so
    a plain ``dict.update`` would let the second silently overwrite the first
    --- a rank that failed the load would vanish behind a rank that passed the
    replay. Failures are namespaced and unioned instead, and the suite passes
    only if both halves passed on every rank.
    """
    failures: dict[str, dict[str, Any]] = {}
    for prefix, agg in (("load", load_agg), ("moe", moe_agg)):
        for rank, checks in (agg.get("per_rank_failures") or {}).items():
            failures.setdefault(str(rank), {}).update(
                {f"{prefix}.{name}": detail for name, detail in checks.items()}
            )
    merged = {k: v for k, v in load_agg.items() if k not in ("passed", "per_rank_failures")}
    merged.update(
        {
            k: v
            for k, v in moe_agg.items()
            if k not in ("passed", "per_rank_failures", "ranks_passed", "ranks_failed")
        }
    )
    ranks_failed = sorted(
        {int(r) for r in failures} | set(load_agg["ranks_failed"]) | set(moe_agg["ranks_failed"])
    )
    merged.update(
        {
            "per_rank_failures": failures,
            "ranks_failed": ranks_failed,
            "ranks_passed": sorted(set(load_agg["ranks_passed"]) & set(moe_agg["ranks_passed"])),
            "load_accounting_passed": bool(load_agg["passed"]),
            "moe_replay_passed": bool(moe_agg["passed"]),
            "passed": bool(load_agg["passed"] and moe_agg["passed"]),
        }
    )
    return merged


def _aggregate_load_ranks(loaded: dict[str, Any], ranks: Ranks) -> dict[str, Any]:
    """Fold every rank's verdict and fingerprint into rank 0's artifact.

    Two things have to cross ranks. A per-rank failure, because rank 0 is not
    the rank that fails and logging is rank-0-only. And the resolved MoE
    fingerprint, because ranks that disagree hang rather than fail, so the
    disagreement has to be caught here while it is still cheap.
    """
    import torch.distributed as dist

    summary = {
        "rank": ranks.rank,
        "passed": bool(loaded["local_passed"]),
        "failed_checks": {
            name: check.get("problems") or check
            for name, check in loaded["checks"].items()
            if not check["passed"]
        },
        "fingerprint": loaded["moe_fingerprint"],
        "memory": loaded["memory"],
        "local_expert_ids": loaded["hard_config"]["local_expert_ids"],
    }
    gathered = [summary]
    if ranks.world > 1 and dist.is_initialized():
        gathered = [None] * ranks.world
        dist.all_gather_object(gathered, summary)

    by_rank = {int(entry["rank"]): entry for entry in gathered}
    failed = sorted(r for r, entry in by_rank.items() if not entry["passed"])
    fingerprints = {json.dumps(entry["fingerprint"], sort_keys=True) for entry in by_rank.values()}
    problems = []
    if len(fingerprints) > 1:
        problems.append(f"ranks resolved {len(fingerprints)} different MoE stacks")
    problems.extend(_expert_shard_problems(by_rank))
    return {
        "ranks_passed": sorted(r for r in by_rank if by_rank[r]["passed"]),
        "ranks_failed": failed,
        "per_rank_failures": {r: by_rank[r]["failed_checks"] for r in failed},
        "identical_rank_fingerprints": len(fingerprints) == 1,
        "rank_fingerprint_problems": problems,
        "memory_by_rank": {r: by_rank[r]["memory"] for r in sorted(by_rank)},
        "peak_host_rss_gb_worst_rank": max(
            entry["memory"]["peak_host_rss_gb"] for entry in by_rank.values()
        ),
        "peak_gpu_allocated_gb_worst_rank": max(
            entry["memory"]["peak_gpu_allocated_gb"] for entry in by_rank.values()
        ),
        "expert_parallel_shards": {r: by_rank[r]["local_expert_ids"] for r in sorted(by_rank)},
        "expert_parallel_coverage": _expert_shard_coverage(by_rank),
        "passed": not failed and not problems and len(by_rank) == ranks.world,
    }


def _expert_shard_coverage(by_rank: dict[int, Any]) -> dict[str, Any]:
    """What the gathered expert-parallel shards actually cover."""
    shards = {rank: list(entry["local_expert_ids"] or ()) for rank, entry in by_rank.items()}
    flat = [eid for shard in shards.values() for eid in shard]
    union = sorted(set(flat))
    return {
        "shards": len(shards),
        "assignments": len(flat),
        "distinct_experts": len(union),
        "disjoint": len(flat) == len(union),
        "min_expert": union[0] if union else None,
        "max_expert": union[-1] if union else None,
        "contiguous_from_zero": union == list(range(len(union))),
        "experts_per_rank": sorted({len(s) for s in shards.values()}),
    }


def _expert_shard_problems(by_rank: dict[int, Any]) -> list[str]:
    """Fail unless the shards partition ``0..N-1`` exactly.

    Distinct shard *tuples* is a much weaker statement than a partition: eight
    ranks can each hold a different set and still overlap, still skip an expert,
    or still land outside the range --- and any of those routes a token to a
    slot nobody owns while every rank reports success. So the union is required
    to be exactly ``range(sum of shard sizes)``, with no expert claimed twice.
    """
    coverage = _expert_shard_coverage(by_rank)
    problems = []
    if not coverage["disjoint"]:
        problems.append(
            f"expert-parallel shards overlap: {coverage['assignments']} assignments cover "
            f"only {coverage['distinct_experts']} distinct experts"
        )
    if not coverage["contiguous_from_zero"]:
        problems.append(
            f"expert-parallel shards do not cover 0..{coverage['assignments'] - 1} exactly: "
            f"{coverage['distinct_experts']} distinct experts spanning "
            f"{coverage['min_expert']}..{coverage['max_expert']}"
        )
    if len(coverage["experts_per_rank"]) > 1:
        problems.append(f"ranks own different expert counts {coverage['experts_per_rank']}")
    return problems


SUITES = {
    "activation_replay_eager": suite_activation_replay_eager,
    "eager_full_model": suite_eager_full_model,
    "eager_state_determinism": suite_eager_state_determinism,
    "kernel_contract": suite_kernel_contract,
    "layer_localization": suite_layer_localization,
    "layer_source_capture": suite_layer_source_capture,
    "load_and_moe": suite_load_and_moe,
    "reference_ladder": suite_reference_ladder,
    "source_reference": suite_source_reference,
    "sparse_kernel_numerics": suite_sparse_kernel_numerics,
    "state_lifecycle": suite_state_lifecycle,
}


def artifact_status(passed: Any, error: str | None) -> str:
    """Map a suite outcome onto the artifact's ``status`` field.

    Kept separate so the rule is unit-testable: writing ``status="passed"``
    next to ``passed=false`` was a false pass in the evidence itself, and a
    reader scanning statuses would have believed it.
    """
    if error:
        return "error"
    return "passed" if passed else "failed"


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--suite", choices=sorted(SUITES))
    parser.add_argument("--output")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    # Conservative by task contract: bring-up sizes the token budget down until
    # the functional and numerical gates pass. 2560 is the smallest round budget
    # that admits the longest registered prompt (2304 tokens) in one piece,
    # which Stage 3 needs because chunked prefill is a Stage 4 item and a prompt
    # that does not fit is simply rejected.
    parser.add_argument("--max-num-tokens", type=int, default=2560)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=4)
    # Explicit by task contract: bring-up states the KV fraction rather than
    # inheriting the 90% default, so a construction failure is never "the cache
    # took whatever was left".
    parser.add_argument("--kv-fraction", type=float, default=0.3)
    parser.add_argument(
        "--parity-tokens",
        type=int,
        default=32,
        help="generated tokens compared step by step against the official source; the "
        "registered generation_parity gate requires at least 32",
    )
    parser.add_argument(
        "--reuse-reference",
        default=None,
        metavar="ARTIFACT",
        help="skip the official-source capture and judge against this existing "
        "source_reference artifact; it is still checked against this run's checkpoint "
        "revision, prompt manifest and token budget before it may judge anything",
    )
    parser.add_argument(
        "--localize-layers",
        nargs="+",
        type=int,
        default=None,
        help="layers the localization splits at every observable sub-boundary "
        "(entry, attention pre-map, attention norm, attention, mid residual, FFN "
        "pre-map, MoE in, MoE out); defaults to layer 0, the one the 43-layer curve "
        "indicts. Both halves read this flag, and the capture records what it used",
    )
    parser.add_argument(
        "--reuse-source-capture",
        default=None,
        metavar="ARTIFACT",
        help="skip layer_localization's nested source capture and read the per-layer "
        "boundaries from this existing layer_source_capture artifact; it is still "
        "checked against this run's checkpoint revision, prompt manifest and world "
        "size, and every sidecar is re-hashed, before it may judge anything",
    )
    parser.add_argument(
        "--replay-layers",
        nargs="+",
        type=int,
        default=[0, 2, 3],
        help="layers the activation replay drives; the registered gate is 0 2 3 (one "
        "ratio-0, one ratio-4 and one ratio-128 layer) and other values are diagnostics",
    )
    parser.add_argument(
        "--golden-layers",
        nargs="+",
        type=int,
        default=[0, 2, 3],
        help="layers the independent pure-Torch reference ladder drives; the registered "
        "gate is 0 2 3 and other values are diagnostics that isolate whether a deviation "
        "is TensorRT-LLM's or is shared by every Torch reimplementation at that depth",
    )
    parser.add_argument(
        "--reference-world-size",
        type=int,
        default=8,
        help="ranks the nested official-source capture runs on; the checkpoint's own "
        "model-parallel shards fix this at 8",
    )
    parser.add_argument(
        "--state-sequence",
        nargs="+",
        default=None,
        metavar="PROMPT_ID",
        help="the scripted request history state_lifecycle drives, in order; repeats "
        "are the point, and the default interleaves five occurrences of one prompt "
        "with three others so a difference can be attributed to what ran before it",
    )
    parser.add_argument(
        "--state-modes",
        nargs="+",
        default=None,
        metavar="MODE",
        help="restrict state_lifecycle to these sharing configurations (control, "
        "metadata_only, cache_only, executor_like, executor_like_zero_freed); the "
        "default runs all five, which is what makes the reading attributable",
    )
    parser.add_argument(
        "--state-pool-tokens",
        type=int,
        default=8192,
        help="KV pool size state_lifecycle allocates, in tokens. Stated rather than "
        "derived from a memory fraction: page recycling is the thing under test, and a "
        "pool sized from 30%% of an 80 GB device would recycle too rarely to observe",
    )
    parser.add_argument("--prompt-ids", nargs="*", default=None)
    parser.add_argument(
        "--replay-prompt",
        default="cache_boundary_257",
        help="prompt id the activation replay drives; the default crosses the 128-token "
        "SWA/KV block boundary and a ratio-128 compression boundary",
    )
    parser.add_argument(
        "--audit",
        nargs="+",
        default=None,
        metavar="ARTIFACT",
        help="re-derive every verdict in a written artifact from tolerances.json and "
        "exit non-zero on any strict failure; needs no GPU",
    )
    parser.add_argument(
        "--regression-report",
        nargs=2,
        default=None,
        metavar=("REGISTERED_LOG", "REQUIRED_JUNIT_XML"),
        help="adjudicate the registered protected-regression run: classify every "
        "failure in its log against the registered environment signatures, diff "
        "against the registered baseline, and check that no required H100 case in "
        "the focused --junitxml pass skipped or failed",
    )
    args = parser.parse_args(argv)
    if args.audit:
        return _run_audit(args.audit)
    if args.regression_report:
        if not args.output:
            parser.error("--regression-report needs --output")
        return _run_regression_report(*args.regression_report, output=args.output)
    if not args.suite or not args.output:
        parser.error("--suite and --output are required unless --audit is given")

    ranks = Ranks()
    _init_distributed(ranks)

    started = time.time()
    # Stamped so the artifact can be checked against the manifest's own
    # `registered_at` without trusting file mtimes, which a copy destroys.
    args.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    error = None
    try:
        result = SUITES[args.suite](args, ranks)
    except Exception as exc:  # recorded, then reported via the exit code
        import traceback

        result = {"evidence_label": args.suite, "passed": False}
        error = traceback.format_exc()
        ranks.log(f"[{args.suite}] FAILED: {exc}")
    # The artifact's status follows the measurement, not the fact that the
    # driver reached the end without raising.
    status = artifact_status(result.get("passed"), error)

    result.update(
        {
            "suite": args.suite,
            "status": status,
            "error": error,
            "started_at": args.started_at,
            "elapsed_s": round(time.time() - started, 2),
            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
            "interpreter": sys.executable,
        }
    )

    if ranks.is_main:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True, default=str)
        print(f"\n[{args.suite}] passed={result.get('passed')} -> {args.output}", flush=True)
        if error:
            print(error, flush=True)

    with contextlib.suppress(Exception):
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    return 0 if result.get("passed") else 1


#: Suites that load the checkpoint's official model and therefore need the
#: isolated reference interpreter. The rest run against TensorRT-LLM and the
#: pure-Torch goldens alone, and re-execing them into that venv would only risk
#: importing a different TensorRT-LLM than the one under test. The reference
#: venv is built ``--system-site-packages`` with a ``.pth`` that puts the linked
#: ``/code/tensorrt_llm`` checkout first, so it is a strict superset of the
#: normal interpreter --- :func:`_linked_package_report` asserts that per run
#: rather than trusting the layout.
#:
#: ``eager_full_model`` and ``layer_localization`` are deliberately *not* in
#: this set. They run production modules end to end, and the production runtime
#: is the container's pinned environment rather than this one: measured on this
#: machine, the reference venv's ``tvm_ffi`` makes nvidia-cutlass-dsl's TVM-FFI
#: provider raise ``make_kwargs_wrapper() got an unexpected keyword argument
#: 'map_dataclass_to_tuple'`` inside flashinfer's CuTe RMSNorm. That killed all
#: eight ``eager_full_model`` executor workers during warmup, and it killed
#: ``layer_localization`` at the *first* module of its first decoder layer when
#: iteration 36 registered the whole suite here. The identical call succeeds
#: under the pinned interpreter. A suite that measured production code there
#: would be measuring a runtime nobody ships --- so each of those two suites
#: keeps its source half in a separate process (``source_reference`` and
#: ``layer_source_capture`` respectively) and reads the result back from disk.
NEEDS_REFERENCE_ENV = frozenset(
    {
        "activation_replay_eager",
        "layer_source_capture",
        "load_and_moe",
        "reference_ladder",
        "source_reference",
    }
)

#: Suites that build a real multi-rank TensorRT-LLM model, or that are launched
#: alongside one. Under ``torchrun`` the MPI world is one process per rank, so
#: ``MPIDist`` refuses a ``Mapping`` with ``world_size=8``; ``TLLM_DISABLE_MPI=1``
#: selects the torch.distributed communicator, which is the one that matches how
#: these ranks were launched. Set before ``tensorrt_llm`` is imported, since the
#: choice is made on import of the distributed helpers.
#:
#: ``layer_source_capture`` is here even though it builds no TensorRT-LLM model:
#: it is spawned by ``layer_localization`` and would otherwise inherit that
#: choice from its parent's environment instead of stating it, so running it
#: directly and running it nested would not be the same measurement.
NEEDS_TORCH_DISTRIBUTED = frozenset(
    {"layer_localization", "layer_source_capture", "load_and_moe", "state_lifecycle"}
)


def _suite_of(argv: list[str]) -> str | None:
    for flag, value in zip(argv, argv[1:]):
        if flag == "--suite":
            return value
    return None


def _wants_reference_env(argv: list[str]) -> bool:
    if "--audit" in argv:  # reads JSON only
        return False
    suite = _suite_of(argv)
    return True if suite is None else suite in NEEDS_REFERENCE_ENV


if __name__ == "__main__":
    _suite = _suite_of(sys.argv)
    if _suite in NEEDS_TORCH_DISTRIBUTED:
        os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    if _wants_reference_env(sys.argv):
        _reference_env()
    elif "--audit" not in sys.argv:
        _trtllm_env()
    raise SystemExit(main())
