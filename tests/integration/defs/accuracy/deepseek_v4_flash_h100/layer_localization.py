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
"""Where does the full stack first leave the source? Per decoder-layer boundary.

The MoE layers reproduce the checkpoint's own kernels to a small fraction of
their registered limit, and the LM head's own dtype cost is smaller still, yet
the end-to-end logits miss the registered limit by an order of magnitude. That
is a statement about the *composition* of 43 layers, and no single-layer replay
can answer it: a layer replay hands each layer the source's own input, so an
error that is small per layer and compounding across layers is invisible.

This module runs the same tokens through both complete stacks and compares the
one tensor that crosses every layer boundary --- the mHC residual stack
``h`` of shape ``[tokens, hc_mult, hidden]`` --- after every layer. Teacher
forcing is trivially satisfied for a prefill: both stacks see the identical
token ids, so position ``i`` of both is conditioned on exactly the same prefix
and the comparison is well posed at every layer, which is not true of two
free-running decodes.

**Two interpreters, two processes, one artifact between them.** The official
model needs ``tilelang``, which only exists in the isolated reference venv, and
that venv is not the runtime under test: its ``tvm_ffi`` makes nvidia-cutlass-dsl
raise ``make_kwargs_wrapper() got an unexpected keyword argument
'map_dataclass_to_tuple'`` inside flashinfer's CuTe RMSNorm --- which is the
*first* production module a DeepSeek-V4 decoder layer runs, so a localization
suite that executes production code there dies before its first boundary. That
is exactly what iteration 36 did. So the source half runs in the reference
interpreter and persists what it saw, and the production half runs in the
pinned interpreter and reads it back. Neither half substitutes a non-production
module to make the other one's environment work.

The bridge is a per-rank sidecar plus the provenance needed to refuse it:
checkpoint revision, prompt-manifest hash, world size, and per-tensor shape,
dtype and SHA-256. Rank ``r``'s production replay is judged against rank ``r``'s
own capture, because both stacks are model-parallel and rank-crossed
comparisons would attribute one rank's arithmetic to another's.

Reading the result:

* a curve that grows smoothly from layer 0 says no layer is wrong and the
  end-to-end miss is accumulated rounding --- the fix is a precision decision,
  not a kernel;
* a step at one layer names that layer's owner;
* a step that repeats at one compression ratio names the sparse path for that
  ratio rather than the layer.

Nothing here gates. The registered manifest does not describe intermediate
decoder-layer boundaries, and inventing a limit for them would be registering a
tolerance after the fact. Every number is reported as a diagnostic, and the
``first_over`` fields quote the diagnostic threshold they used.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Sequence
from typing import Any, Iterator

import torch
import torch.utils._device

#: Diagnostic reading thresholds. Not gates --- see the module docstring. The
#: rel-max value is the one the *logit* gate uses, so "the layer curve crosses
#: it at layer k" is directly comparable with the failing end-to-end number.
READING_REL_MAX = 0.05
READING_COSINE = 0.99

#: Prompts both halves drive when the caller names none. ``chat_geography`` is
#: the cheapest gating prompt (16 tokens) and has the worst logit cosine;
#: ``cache_boundary_257`` is the one whose generation parity still fails.
DEFAULT_PROMPTS = ("chat_geography", "cache_boundary_257")

#: Layers the sub-boundary decomposition drives when the caller names none.
#: Layer 0 is the one the 43-layer curve indicts: ratio-0 (SWA only, no
#: Compressor and no Indexer), a bit-exact input, a MoE already measured at
#: 0.004-0.025 of its 0.04 limit, and still 0.144 last-row rel from the source.
DEFAULT_DEEP_LAYERS = (0,)

#: One decoder layer, split at every boundary both stacks can be observed at,
#: in execution order. The names are shared, so the comparison is a lookup
#: rather than a mapping table maintained in two places.
#:
#: ==================  ===============================  ==========================
#: name                source (``Block.forward``)        production
#: ==================  ===============================  ==========================
#: ``layer_entry``     ``hc_pre`` call 0, argument       ``hc_attn.pre_mapping`` arg
#: ``attn_premap``     ``hc_pre`` call 0, result         ``hc_attn.pre_mapping`` [2]
#: ``attn_norm_out``   ``attn_norm`` output              ``input_layernorm`` output
#: ``attn_out``        ``attn`` output                   ``self_attn`` output
#: ``mid_residual``    ``hc_post`` call 0, result        ``hc_ffn.fused_hc`` [0]
#: ``ffn_premap``      ``hc_pre`` call 1, result         ``hc_ffn.fused_hc`` [3]
#: ``moe_in``          ``ffn_norm`` output               ``post_attention_layernorm``
#: ``moe_out``         ``ffn`` output                    ``mlp`` output
#: ==================  ===============================  ==========================
#:
#: Every one is the *real* value the real call produced --- module hooks where
#: a module exists, and a delegating wrapper where the step is a method
#: (``hc_pre``/``hc_post`` on the source's ``Block``, ``pre_mapping``/
#: ``fused_hc`` on production's ``mHC``). Nothing here recomputes a boundary,
#: because a harness copy of a step is exactly the thing that cannot be trusted
#: to disagree with it.
SUBLAYER_ORDER = (
    "layer_entry",
    "attn_premap",
    "attn_norm_out",
    "attn_out",
    "mid_residual",
    "ffn_premap",
    "moe_in",
    "moe_out",
)


def _round(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 9) if isinstance(v, float) else v) for k, v in metrics.items()}


class _MethodTap:
    """Record what a bound method really returned, without changing it.

    ``hc_pre``/``hc_post`` on the source's ``Block`` and ``pre_mapping``/
    ``fused_hc``/``post_mapping`` on production's ``mHC`` are methods, not
    submodules, so no forward hook can see them. This shadows the attribute
    with a wrapper that delegates to the original, keeps the call's arguments
    and result, and restores the attribute on exit. Same pattern
    ``activation_replay._SparseAttnRecorder`` already uses for the source's
    free-function ``sparse_attn``.
    """

    def __init__(self, owner: Any, name: str) -> None:
        self.owner, self.name = owner, name
        self.original = getattr(owner, name)
        self.had_own = name in owner.__dict__
        self.calls: list[tuple[tuple, dict, Any]] = []

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = self.original(*args, **kwargs)
            self.calls.append((args, kwargs, result))
            return result

        setattr(owner, name, wrapper)

    def remove(self) -> None:
        if self.had_own:
            setattr(self.owner, self.name, self.original)
        else:
            delattr(self.owner, self.name)

    def result(self, call: int, index: int | None = None) -> Any:
        _, _, result = self.calls[call]
        return result if index is None else result[index]

    def arg(self, call: int, index: int) -> Any:
        args, _, _ = self.calls[call]
        return args[index]


# ---------------------------------------------------------------------------
# The source side. Runs in the reference interpreter.
# ---------------------------------------------------------------------------


def capture_source(
    src: Any, token_ids: list[int], capture_fn: Any, deep_layers: Sequence[int] = ()
) -> dict[str, Any]:
    """One source prefill, keeping every block's output residual stack.

    ``Transformer.forward`` threads ``h`` of shape ``[b, s, hc, d]`` through
    every ``Block``, so a forward hook on the block *is* the layer boundary ---
    no reconstruction, and nothing downstream of the hook can change what it
    saw because the capture helper clones.

    ``deep_layers`` additionally splits those layers at every
    :data:`SUBLAYER_ORDER` boundary. One layer's split costs eight small
    tensors, so it is on by default for layer 0 rather than being a separate
    run: the layer curve says *which* layer, and only this says which part.

    Everything lands on the host in FP32: it has to be written to disk for the
    other interpreter anyway, and holding 43 boundaries on the GPU would eat
    the headroom the same rank's production prefill needs later.
    """
    store: dict[str, Any] = {}
    handles = [capture_fn(src.model.embed, store, "embed")]
    for lid, layer in enumerate(src.model.layers):
        handles.append(capture_fn(layer, store, f"l{lid}"))
    handles.append(capture_fn(src.model.norm, store, "norm"))

    taps: dict[int, dict[str, _MethodTap]] = {}
    for lid in deep_layers:
        block = src.model.layers[lid]
        for name in ("attn_norm", "attn", "ffn_norm", "ffn"):
            handles.append(capture_fn(getattr(block, name), store, f"s{lid}.{name}"))
        taps[lid] = {
            "hc_pre": _MethodTap(block, "hc_pre"),
            "hc_post": _MethodTap(block, "hc_post"),
        }

    src.reset_cache()
    tokens = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    try:
        with torch.inference_mode():
            logits = src.model.forward(tokens, 0)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        for per_layer in taps.values():
            for tap in per_layer.values():
                tap.remove()

    hidden = src.args.dim

    def host(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to("cpu", torch.float32).contiguous().clone()

    layers = {
        lid: host(store[f"l{lid}"]["output"].reshape(-1, src.args.hc_mult, hidden))
        for lid in range(len(src.model.layers))
    }

    # ``Block.forward`` calls ``hc_pre`` twice and ``hc_post`` twice --- once
    # around attention, once around the FFN --- so the call index *is* the
    # boundary, and nothing has to be inferred from shapes.
    sublayers = {}
    for lid, per_layer in taps.items():
        pre, post = per_layer["hc_pre"], per_layer["hc_post"]
        flat = lambda t: t.reshape(-1, hidden)  # noqa: E731 - one shape, used eight times
        sublayers[lid] = {
            "layer_entry": host(pre.arg(0, 0).reshape(-1, src.args.hc_mult, hidden)),
            "attn_premap": host(flat(pre.result(0, 0))),
            "attn_norm_out": host(flat(store[f"s{lid}.attn_norm"]["output"])),
            "attn_out": host(flat(store[f"s{lid}.attn"]["output"])),
            "mid_residual": host(post.result(0).reshape(-1, src.args.hc_mult, hidden)),
            "ffn_premap": host(flat(pre.result(1, 0))),
            "moe_in": host(flat(store[f"s{lid}.ffn_norm"]["output"])),
            "moe_out": host(flat(store[f"s{lid}.ffn"]["output"])),
        }
    # The production epilogue selects the last token *before* hc_head and the
    # final norm (``DeepseekV4LogitsProcessor``), so its norm only ever sees one
    # row. The source norms every position and selects inside ``get_logits``.
    # Comparing the last row on both sides compares the same computation.
    #
    # ``hc_head`` is a *method* on ``ParallelHead``, not a submodule, so no hook
    # can see its output directly --- but ``Transformer.forward`` passes that
    # output straight into ``norm``, so the norm hook's own input is it. That
    # closes the last gap in the chain: without it, the 43-layer curve stops at
    # the final residual stack and everything between there and the logits is
    # one unsplit step.
    return {
        "embed": host(store["embed"]["output"].reshape(-1, hidden)),
        "layers": layers,
        "sublayers": sublayers,
        "hc_head": host(store["norm"]["inputs"][0].reshape(-1, hidden)[-1:]),
        "final_norm": host(store["norm"]["output"].reshape(-1, hidden)[-1:]),
        "logits": host(logits.reshape(-1, logits.shape[-1])[-1]),
    }


def capture_all(
    ranks: Any,
    src: Any,
    prompts: list[dict[str, Any]],
    capture_fn: Any,
    deep_layers: Sequence[int] = (),
) -> dict:
    """Every prompt's source boundaries, on this rank."""
    captured: dict[str, Any] = {}
    for prompt in prompts:
        started = time.time()
        captured[prompt["id"]] = capture_source(src, prompt["token_ids"], capture_fn, deep_layers)
        ranks.log(
            f"  source prefill {prompt['id']:24s} {len(prompt['token_ids']):5d} tokens "
            f"{time.time() - started:.1f}s"
        )
    return captured


# ---------------------------------------------------------------------------
# The bridge: persist a rank's capture, and refuse one that cannot judge a run.
# ---------------------------------------------------------------------------


def sidecar_path(artifact_path: str, rank: int) -> str:
    """Where rank ``rank``'s activation sidecar for ``artifact_path`` lives."""
    return f"{os.path.splitext(os.path.abspath(artifact_path))[0]}.activations.rank{rank}.pt"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash a tensor's actual bytes, whatever its dtype.

    Via a ``uint8`` view rather than ``numpy``: the boundaries are FP32 today,
    but a capture that changed dtype and silently kept passing its checksum
    would be the exact failure this provenance exists to catch.
    """
    flat = tensor.detach().cpu().contiguous().flatten()
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def _describe(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": _sha256_tensor(tensor),
        "finite": bool(torch.isfinite(tensor).all()),
    }


def _boundaries(capture: dict[str, Any]) -> Iterator[tuple[str, torch.Tensor]]:
    """Every tensor in one prompt's capture, under a stable flat name."""
    yield "embed", capture["embed"]
    for lid in sorted(capture["layers"]):
        yield f"layer{lid}", capture["layers"][lid]
    for lid in sorted(capture.get("sublayers") or {}):
        for name in SUBLAYER_ORDER:
            yield f"sub{lid}.{name}", capture["sublayers"][lid][name]
    yield "hc_head", capture["hc_head"]
    yield "final_norm", capture["final_norm"]
    yield "logits", capture["logits"]


def _regroup(flat: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Invert :func:`_boundaries` for one prompt."""
    layers = {
        int(name[len("layer") :]): tensor
        for name, tensor in flat.items()
        if name.startswith("layer")
    }
    sublayers: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in flat.items():
        if not name.startswith("sub"):
            continue
        lid, boundary = name[len("sub") :].split(".", 1)
        sublayers.setdefault(int(lid), {})[boundary] = tensor
    return {
        "embed": flat["embed"],
        "layers": layers,
        "sublayers": sublayers,
        "hc_head": flat["hc_head"],
        "final_norm": flat["final_norm"],
        "logits": flat["logits"],
    }


def save_capture(
    artifact_path: str, rank: int, captured: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any]:
    """Write this rank's sidecar and return the provenance to embed in the JSON."""
    path = sidecar_path(artifact_path, rank)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat: dict[str, torch.Tensor] = {}
    described: dict[str, Any] = {}
    for pid, capture in captured.items():
        for key, tensor in _boundaries(capture):
            name = f"{pid}/{key}"
            flat[name] = tensor
            described[name] = _describe(tensor)
    torch.save({"meta": meta, "tensors": flat}, path)
    return {
        "rank": rank,
        "path": path,
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "prompts": sorted(captured),
        "tensors": described,
        "all_finite": all(entry["finite"] for entry in described.values()),
    }


def provenance_problems(
    loaded: dict[str, torch.Tensor], declared: dict[str, Any]
) -> list[str]:
    """Why the tensors on disk are not the tensors the capture described.

    A sidecar that was replaced wholesale is caught by the file hash; this
    catches the rest --- a tensor dropped, added, reshaped, recast, or edited
    in place --- so a curve can never be measured against contents nobody
    registered.
    """
    problems: list[str] = []
    for name in sorted(set(declared) - set(loaded)):
        problems.append(f"{name}: declared by the capture but absent from the sidecar")
    for name in sorted(set(loaded) - set(declared)):
        problems.append(f"{name}: present in the sidecar but never declared")
    for name in sorted(set(loaded) & set(declared)):
        entry, tensor = declared[name], loaded[name]
        if list(tensor.shape) != list(entry["shape"]):
            problems.append(f"{name}: shape {list(tensor.shape)}, capture recorded {entry['shape']}")
        elif str(tensor.dtype) != entry["dtype"]:
            problems.append(f"{name}: dtype {tensor.dtype}, capture recorded {entry['dtype']}")
        elif _sha256_tensor(tensor) != entry["sha256"]:
            problems.append(f"{name}: contents do not hash to what the capture recorded")
    return problems


def load_capture(artifact: dict[str, Any], rank: int) -> dict[str, Any]:
    """Read this rank's persisted source boundaries, or refuse to.

    Every failure here would otherwise become a per-layer curve measured
    against the wrong tensors, which reads exactly like a real divergence.
    """
    per_rank = artifact.get("per_rank") or {}
    entry = per_rank.get(str(rank))
    if entry is None:
        raise RuntimeError(
            f"the source capture has no rank {rank}; it covers {sorted(per_rank)} and "
            "a rank-crossed comparison would attribute another rank's arithmetic to this one"
        )
    path = entry["path"]
    if not os.path.exists(path):
        raise RuntimeError(f"rank {rank}'s activation sidecar {path} is recorded but missing")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise RuntimeError(
            f"activation sidecar {path} hashes to {actual}, but the capture recorded "
            f"{entry['sha256']}; it is not the file that capture produced"
        )
    blob = torch.load(path, map_location="cpu", weights_only=True)
    tensors = blob["tensors"]
    problems = provenance_problems(tensors, entry["tensors"])
    if problems:
        raise RuntimeError(f"rank {rank}'s activation sidecar disagrees with its provenance: {problems}")

    by_prompt: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        pid, key = name.split("/", 1)
        by_prompt.setdefault(pid, {})[key] = tensor
    return {pid: _regroup(flat) for pid, flat in by_prompt.items()}


def capture_usable(
    artifact: dict[str, Any],
    *,
    checkpoint_revision: str,
    prompts_sha256: str,
    prompt_ids: list[str],
    world_size: int,
    deep_layers: Sequence[int] = (),
) -> list[str]:
    """Why this capture may not judge the run about to happen, if it may not.

    Same contract as ``source_reference.usable``: a capture is a reference only
    for the measurement it was taken for. A different checkpoint, a different
    prompt rendering, a different world size or a capture that failed its own
    checks would each produce a comparison against the wrong thing.
    """
    problems: list[str] = []
    if artifact.get("checkpoint_revision") != checkpoint_revision:
        problems.append(
            f"capture is for checkpoint {artifact.get('checkpoint_revision')!r}, "
            f"this run is {checkpoint_revision!r}"
        )
    recorded = (artifact.get("manifest_provenance") or {}).get("sha256", {})
    if recorded.get("prompts.json") != prompts_sha256:
        problems.append("capture was taken against a different prompts manifest")
    if artifact.get("world_size") != world_size:
        problems.append(
            f"capture ran on {artifact.get('world_size')} ranks, this run has {world_size}; "
            "the source model is model-parallel, so the shards do not correspond"
        )
    if not artifact.get("passed"):
        problems.append("capture artifact did not pass its own checks")
    captured = {p["id"] for p in artifact.get("prompts") or []}
    missing = sorted(set(prompt_ids) - captured)
    if missing:
        problems.append(f"capture is missing prompts {missing}")
    per_rank = artifact.get("per_rank") or {}
    absent = [r for r in range(world_size) if str(r) not in per_rank]
    if absent:
        problems.append(f"capture has no sidecar for ranks {absent}")
    unsplit = sorted(set(deep_layers) - set(artifact.get("deep_layers") or []))
    if unsplit:
        problems.append(
            f"capture did not split layers {unsplit} at their sub-boundaries; it split "
            f"{artifact.get('deep_layers')}"
        )
    return problems


def capture_job(
    args: Any, driver: str, output: str, prompt_ids: list[str], deep_layers: Sequence[int] = ()
) -> dict[str, Any]:
    """Run the source capture in its own eight-rank job under the reference venv.

    Out of process by necessity, not preference --- see the module docstring.
    Launched from the production job's rank 0 *before* anything production is
    built, so the source model has the GPUs to itself, and with the launcher's
    rank variables removed from the child's environment only: the parent is
    still an active ``torchrun`` rank and must keep its own.
    """
    import full_model

    command = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={args.reference_world_size}",
        f"--rdzv-endpoint=localhost:{full_model.free_port()}",
        driver,
        "--checkpoint",
        args.checkpoint,
        "--suite",
        "layer_source_capture",
        "--output",
        output,
        "--max-seq-len",
        str(args.max_seq_len),
        "--prompt-ids",
        *prompt_ids,
    ]
    if deep_layers:
        # Stated rather than inherited: the two halves have to split the same
        # layers, or the production replay would compare boundaries the capture
        # never took under names that look identical.
        command += ["--localize-layers", *(str(lid) for lid in deep_layers)]
    env = {k: v for k, v in os.environ.items() if k not in full_model.LAUNCHER_ENV_VARS}
    started = time.time()
    completed = subprocess.run(command, check=False, env=env)
    elapsed = round(time.time() - started, 2)
    if completed.returncode != 0:
        raise RuntimeError(
            f"source layer capture failed with exit code {completed.returncode}: "
            f"{' '.join(command)}"
        )
    return {"command": " ".join(command), "elapsed_s": elapsed, "artifact": output}


def read_artifact(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"no source capture at {path}; run the layer_source_capture suite first")
    with open(path) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# The TensorRT-LLM side. Runs in the pinned interpreter.
# ---------------------------------------------------------------------------


def _extra_attrs(live: Any) -> dict[str, Any]:
    """The extra-attrs registry the model's own modules registered into.

    ``DeepseekV4ForCausalLM.__init__`` deep-copies its ``ModelConfig`` whenever
    the checkpoint carries a ``quant_config_dict`` --- which this mixed FP8/FP4
    checkpoint does --- so every ``MLA`` and ``DeepseekV4MoE`` writes its
    ``mla_layers`` / ``moe_layers`` weak reference into *that* copy, and the
    custom ops (``create_mla_outputs`` and friends) resolve through whichever
    dict is installed as the thread-local model extra attrs. Handing them the
    config the caller built is therefore an ``Attention layer is not
    registered`` assertion three frames inside a custom op.

    Measured on this checkpoint at TP8: the caller's dict holds
    ``['deepseek_v4_rope_tables', 'rope_const_params']`` and the model's holds
    those plus 43 ``mla_layers`` and ``moe_layers``.

    Checked rather than assumed, because the failure it prevents is late,
    per-layer and unreadable.
    """
    attrs = live.model.model_config.extra_attrs
    missing = [key for key in ("mla_layers", "moe_layers") if key not in attrs]
    if missing:
        raise RuntimeError(
            f"the model's extra attrs are missing {missing}; the custom ops resolve "
            f"their owning module through this registry and would assert inside the "
            f"first decoder layer (present: {sorted(attrs)})"
        )
    return attrs


@contextlib.contextmanager
def _cuda_default_device():
    """Install the CUDA default device the production executor supplies implicitly.

    Through ``torch.set_default_device`` rather than ``with torch.device(...)``,
    and the difference is not stylistic. The context-manager form pushes a
    ``DeviceContext`` straight onto the torch-function mode stack without
    touching ``_GLOBAL_DEVICE_CONTEXT`` or ``torch.utils._device.CURRENT_DEVICE``
    --- so :func:`activation_replay._host_default_device`, which removes the
    override through exactly those, cannot see it and the metadata build would
    still allocate on the GPU. Measured: inside ``with torch.device("cuda")``, a
    ``set_default_device(None)`` leaves ``torch.empty(2).device`` at ``cuda:0``;
    through ``set_default_device("cuda")`` the same call returns ``cpu``.
    """
    previous = torch.utils._device.CURRENT_DEVICE
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(previous)


class _Prefill:
    """One prefill of the production model, with the resources it needs.

    Built the way ``PyExecutor`` builds them --- a real
    ``DeepseekV4CacheManager``, a real ``DeepseekV4TrtllmAttentionMetadata``,
    a real ``LlmRequest`` registered in the cache --- because the point is to
    measure the production path, and a hand-rolled attention shim would be
    measuring something else.
    """

    def __init__(self, live: Any, prompt_len: int, max_seq_len: int) -> None:
        import tensorrt_llm
        from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import DeepseekV4CacheManager
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        config = live.model_config.pretrained_config
        self.live = live
        self.prompt_len = prompt_len
        tokens_per_block = 128
        blocks = max(4, (max_seq_len + tokens_per_block - 1) // tokens_per_block)
        self.max_seq_len = blocks * tokens_per_block

        self.cache_manager = DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False,
                max_tokens=blocks * tokens_per_block,
                event_buffer_max_size=0,
            ),
            kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=config.num_hidden_layers,
            num_kv_heads=1,
            head_dim=config.v_head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=self.max_seq_len,
            max_batch_size=1,
            mapping=live.mapping,
            dtype=tensorrt_llm.bindings.DataType.BF16,
            compressor_dtype=tensorrt_llm.bindings.DataType.FLOAT,
            vocab_size=config.vocab_size,
            max_num_tokens=self.max_seq_len,
            sparse_attn_config=live.model_config.sparse_attention_config,
            model_config=live.model_config,
        )
        self.request = LlmRequest(
            request_id=0,
            max_new_tokens=1,
            input_tokens=list(range(prompt_len)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert self.cache_manager.prepare_context(self.request), "prepare_context failed"
        assert self.cache_manager.resize_context(
            self.request, self.request.context_chunk_size
        ), "resize_context failed"

    def build_metadata(self) -> Any:
        """Construct and prepare the attention metadata.

        Split from :meth:`logits` because the two halves want opposite default
        devices. The metadata pins host bookkeeping and its index-conversion
        pass builds host-side tables, so it needs the default device off; the
        forward issues TP collectives, and NCCL refuses a CPU tensor, so it
        needs the default device back where the process left it. Running both
        under one scope fails one way or the other --- which is exactly what
        the first two attempts at this suite did.
        """
        import weakref

        from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
            DeepseekV4TrtllmAttentionMetadata,
        )
        from tensorrt_llm._torch.metadata import KVCacheParams
        from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

        prompt_len = self.prompt_len
        metadata = DeepseekV4TrtllmAttentionMetadata(
            # Explicitly host-side: the metadata setter pins this buffer, so a
            # bare ``torch.tensor`` here under a CUDA default device is a CUDA
            # tensor that cannot be pinned.
            seq_lens=torch.tensor([prompt_len], dtype=torch.int32, device="cpu"),
            num_contexts=1,
            max_num_requests=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            kv_cache_manager=self.cache_manager,
            request_ids=[0],
            prompt_lens=[prompt_len],
            max_num_tokens=self.max_seq_len,
            mapping=self.live.mapping,
            sparse_attention_config=self.live.model_config.sparse_attention_config,
        )
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [self.request]
        self.cache_manager.prepare_resources(batch)
        # Keep the metadata reachable from the model's extra attrs the way the
        # executor does; the reference is weak, so the caller holds the object.
        # Into the model's own registry --- see :func:`_extra_attrs`.
        _extra_attrs(self.live)["attention_metadata"] = weakref.ref(metadata)
        metadata.prepare()
        return metadata

    def logits(self, metadata: Any, token_ids: list[int]) -> torch.Tensor:
        """Run the prefill and return the final position's logits."""
        from tensorrt_llm._torch.utils import model_extra_attrs

        input_ids = torch.tensor(token_ids, dtype=torch.int32, device="cuda")
        position_ids = (
            torch.arange(self.prompt_len, device="cuda").unsqueeze(0).to(torch.int32)
        )
        with torch.inference_mode(), model_extra_attrs(_extra_attrs(self.live)):
            logits = self.live.model.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=metadata
            )
        torch.cuda.synchronize()
        return logits.reshape(-1, logits.shape[-1])[-1].float().clone()

    def shutdown(self) -> None:
        self.cache_manager.free_resources(self.request)
        self.cache_manager.shutdown()


def _resolve(layer: Any, state: Any) -> torch.Tensor:
    """The residual stack a layer hands the next one, in the source's shape.

    A layer that deferred its ``hc_ffn.post_mapping`` into the next layer's
    ``fused_hc`` returns the *pre*-post-mapping residual plus the three tensors
    the fold needs. Resolving it here calls the layer's own ``post_mapping`` ---
    the same call the unfused branch of ``forward`` makes --- rather than a
    harness copy of it, so the comparison cannot drift from production by being
    reimplemented.
    """
    if not state.is_deferred:
        return state.residual
    return layer.hc_ffn.post_mapping(
        x=state.x_prev,
        residual=state.residual,
        post_layer_mix=state.post_mix,
        comb_res_mix=state.comb_mix,
    )


class _SublayerTaps:
    """Every :data:`SUBLAYER_ORDER` boundary of one production decoder layer.

    The mid-layer boundary is the awkward one. Production folds
    ``hc_attn.post_mapping`` and ``hc_ffn.pre_mapping`` into a single
    ``hc_ffn.fused_hc`` call, so the mid residual and the FFN pre-map come back
    as elements of one tuple; with ``enable_fused_hc=False`` they are two calls
    instead. Both are tapped and ``path`` records which one actually ran, so
    the artifact says how the numbers it reports were produced rather than
    assuming a configuration.
    """

    def __init__(self, layer: Any) -> None:
        self.layer = layer
        self.store: dict[str, Any] = {}
        self.taps = {
            "pre_mapping": _MethodTap(layer.hc_attn, "pre_mapping"),
            "fused_hc": _MethodTap(layer.hc_ffn, "fused_hc"),
            "post_mapping": _MethodTap(layer.hc_attn, "post_mapping"),
            "ffn_pre_mapping": _MethodTap(layer.hc_ffn, "pre_mapping"),
        }
        # Taken from the *consumers*, not from the norms that feed them.
        # ``forward_MoE`` only calls ``post_attention_layernorm`` as a module
        # when ``PRE_MOE_FUSION`` is off; with it on --- which is this
        # configuration --- the norm is folded into the all-reduce's fusion op
        # and the module never fires. Measured, by a run that died on a missing
        # ``moe_in``. Whatever normalized the tensor, what attention and the MoE
        # were actually handed is the boundary worth comparing.
        self.handles = [
            self._hook("attn", layer.self_attn),
            self._hook("moe", layer.mlp),
            self._hook("input_layernorm", layer.input_layernorm),
            self._hook("post_attention_layernorm", layer.post_attention_layernorm),
        ]

    def _hook(self, key: str, module: Any) -> Any:
        def snap(tensor: Any) -> Any:
            tensor = tensor[0] if isinstance(tensor, tuple) else tensor
            return tensor.detach().float().clone() if isinstance(tensor, torch.Tensor) else None

        def hook(_mod: Any, args: tuple, kwargs: dict, output: Any) -> None:
            entry = {"out": snap(output)}
            if args:
                entry["in"] = snap(args[0])
            elif "hidden_states" in kwargs:
                entry["in"] = snap(kwargs["hidden_states"])
            self.store[key] = entry

        return module.register_forward_hook(hook, with_kwargs=True)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        for tap in self.taps.values():
            tap.remove()

    def collect(self, hidden: int, hc_mult: int) -> dict[str, Any]:
        pre, fused = self.taps["pre_mapping"], self.taps["fused_hc"]
        post, ffn_pre = self.taps["post_mapping"], self.taps["ffn_pre_mapping"]
        fused_ran = bool(fused.calls)

        def flat(tensor: Any, mult: int | None = None) -> torch.Tensor:
            tensor = tensor.detach().float()
            return (
                tensor.reshape(-1, hidden) if mult is None else tensor.reshape(-1, mult, hidden)
            ).clone()

        return {
            "layer_entry": flat(pre.arg(0, 0), hc_mult),
            "attn_premap": flat(pre.result(0, 2)),
            "attn_norm_out": flat(self.store["attn"]["in"]),
            "attn_out": flat(self.store["attn"]["out"]),
            "mid_residual": flat(
                fused.result(0, 0) if fused_ran else post.result(0), hc_mult
            ),
            "ffn_premap": flat(fused.result(0, 3) if fused_ran else ffn_pre.result(0, 2)),
            "moe_in": flat(self.store["moe"]["in"]),
            "moe_out": flat(self.store["moe"]["out"]),
            "path": "fused_hc" if fused_ran else "post_mapping+pre_mapping",
            "moe_norm": (
                "post_attention_layernorm module"
                if "post_attention_layernorm" in self.store
                else "folded into the all-reduce fusion op (PRE_MOE_FUSION)"
            ),
            "attn_norm": (
                "input_layernorm module"
                if "input_layernorm" in self.store
                else "folded into fused_hc's layer_input epilogue"
            ),
        }


def capture_trtllm(
    live: Any, token_ids: list[int], max_seq_len: int, deep_layers: Sequence[int] = ()
) -> dict[str, Any]:
    """One production prefill, keeping every decoder layer's residual stack."""
    store: dict[str, Any] = {}
    layers = live.model.model.layers[: live.model_config.pretrained_config.num_hidden_layers]

    def hook_for(lid: int, layer: Any) -> Any:
        def hook(_mod: Any, _inputs: tuple, output: Any) -> None:
            store[lid] = _resolve(layer, output).detach().float().clone()

        return layer.register_forward_hook(hook)

    handles = [hook_for(lid, layer) for lid, layer in enumerate(layers)]
    embed_store: dict[str, Any] = {}

    def embed_hook(_mod: Any, _inputs: tuple, output: Any) -> None:
        embed_store["out"] = output.detach().float().clone()

    handles.append(live.model.model.embed_tokens.register_forward_hook(embed_hook))

    norm_store: dict[str, Any] = {}

    def norm_hook(_mod: Any, inputs: tuple, output: Any) -> None:
        # The input is ``hc_head``'s output on both sides --- see
        # :func:`capture_source`.
        norm_store["in"] = inputs[0].detach().float().clone()
        norm_store["out"] = output.detach().float().clone()

    handles.append(live.model.model.norm.register_forward_hook(norm_hook))

    sub_taps = {lid: _SublayerTaps(layers[lid]) for lid in deep_layers}

    from activation_replay import _host_default_device

    prefill = None
    try:
        # Default-device scoping is per *step*, not per phase, and each step's
        # requirement was established by a run that failed the other way:
        #
        #   cache manager   CUDA -- `DeepseekV4CacheManager.__init__` all-reduces
        #                   to agree pool sizes, and `TorchDist.allreduce` wraps
        #                   a plain Python int with a bare `torch.tensor`, which
        #                   NCCL rejects when that lands on the host;
        #   metadata build  host -- the metadata setter pins `seq_lens`, and its
        #                   index-conversion pass builds host-side tables;
        #   forward         CUDA -- device-side tables and TP collectives.
        #
        # Until iteration 36 the CUDA half was inherited by accident, from the
        # `torch.set_default_device("cuda")` `OfficialSource` leaves behind. The
        # source no longer shares this process, so the scope is stated.
        with _cuda_default_device():
            prefill = _Prefill(live, len(token_ids), max_seq_len)
            with _host_default_device():
                metadata = prefill.build_metadata()
            logits = prefill.logits(metadata, token_ids)
        hidden = live.model_config.pretrained_config.hidden_size
        hc_mult = live.model_config.pretrained_config.hc_mult
        sublayers = {lid: tap.collect(hidden, hc_mult) for lid, tap in sub_taps.items()}
    finally:
        for handle in handles:
            handle.remove()
        for tap in sub_taps.values():
            tap.remove()
        if prefill is not None:
            prefill.shutdown()

    def last_row(tensor: Any) -> Any:
        return None if tensor is None else tensor.reshape(-1, hidden)[-1:]

    return {
        "embed": embed_store["out"],
        "layers": store,
        "sublayers": sublayers,
        "hc_head": last_row(norm_store.get("in")),
        "final_norm": last_row(norm_store.get("out")),
        "logits": logits,
    }


# ---------------------------------------------------------------------------
# The comparison.
# ---------------------------------------------------------------------------


def compare_stacks(
    source: dict[str, Any], trtllm: dict[str, Any], ratios: list[int], tg: Any
) -> dict[str, Any]:
    """Per-layer metrics plus the reading the curve supports."""

    def _measure(got: torch.Tensor, ref: torch.Tensor, what: str) -> dict[str, Any]:
        # A silent shape disagreement would be compared as two flat vectors of
        # different lengths and raise something unreadable three frames down.
        assert got.shape == ref.shape, (
            f"{what}: TensorRT-LLM produced {tuple(got.shape)} where the source "
            f"produced {tuple(ref.shape)}; these are not the same boundary"
        )
        # The source half ran in another process, so its tensors arrive on the
        # host; move each one to the device it is being compared on, and only
        # for as long as that comparison takes.
        ref = ref.to(got.device)
        metrics = _round(tg.compare(got, ref))
        if got.dim() >= 2 and got.shape[0] > 1:
            # The production epilogue keeps only the last position, so hc_head,
            # final_norm and logits are last-row comparisons by construction. A
            # per-layer cosine averaged over every position cannot be read
            # against them: an error concentrated at the last row is diluted by
            # the fifteen rows that agree. Report the same slice per layer so
            # the curve and the epilogue are the same measurement.
            last = _round(tg.compare(got[-1:], ref[-1:]))
            metrics["last_row_rel_max_abs"] = last["rel_max_abs"]
            metrics["last_row_cosine"] = last["cosine"]
            metrics["last_row_max_abs"] = last["max_abs"]
        return metrics

    per_layer = []
    first_rel = None
    first_cos = None
    first_last_row_cos = None
    for lid in sorted(source["layers"]):
        got = trtllm["layers"].get(lid)
        if got is None:
            continue
        metrics = _measure(got, source["layers"][lid], f"layer {lid}")
        metrics["layer"] = lid
        metrics["compress_ratio"] = ratios[lid] if lid < len(ratios) else None
        per_layer.append(metrics)
        if first_rel is None and metrics["rel_max_abs"] > READING_REL_MAX:
            first_rel = lid
        if first_cos is None and metrics["cosine"] < READING_COSINE:
            first_cos = lid
        if first_last_row_cos is None and metrics.get("last_row_cosine", 1.0) < READING_COSINE:
            first_last_row_cos = lid

    # Inside the layers the curve indicts. Reported in execution order, so the
    # first boundary that leaves the source names the owner directly: an entry
    # that already differs means the previous layer, a clean ``attn_norm_out``
    # with a dirty ``attn_out`` means MLA, a clean ``moe_in`` with a dirty
    # ``moe_out`` means the MoE, and a jump across ``mid_residual`` or either
    # pre-map means mHC.
    per_sublayer: dict[str, Any] = {}
    for lid in sorted(source.get("sublayers") or {}):
        got_layer = (trtllm.get("sublayers") or {}).get(lid)
        if got_layer is None:
            continue
        ref_layer = source["sublayers"][lid]
        chain = []
        first_over = None
        for name in SUBLAYER_ORDER:
            metrics = _measure(got_layer[name], ref_layer[name], f"layer {lid} {name}")
            metrics["boundary"] = name
            chain.append(metrics)
            worst = max(metrics["rel_max_abs"], metrics.get("last_row_rel_max_abs", 0.0))
            if first_over is None and worst > READING_REL_MAX:
                first_over = name
        per_sublayer[str(lid)] = {
            "chain": chain,
            "mid_boundary_path": got_layer.get("path"),
            "attn_norm_path": got_layer.get("attn_norm"),
            "moe_norm_path": got_layer.get("moe_norm"),
            "first_boundary_over_rel_max": first_over,
            "reading": (
                "boundaries are in execution order; the first one over the threshold is "
                "the owner, and the one before it is the last agreement"
            ),
        }

    boundaries = {
        "embedding": _measure(trtllm["embed"], source["embed"], "embedding"),
        "hc_head": (
            None
            if trtllm.get("hc_head") is None
            else _measure(trtllm["hc_head"], source["hc_head"], "hc_head")
        ),
        "final_norm": (
            None
            if trtllm.get("final_norm") is None
            else _measure(trtllm["final_norm"], source["final_norm"], "final_norm")
        ),
        "logits": _measure(trtllm["logits"], source["logits"], "logits"),
    }
    boundaries["logits"]["argmax_match"] = bool(
        int(trtllm["logits"].argmax()) == int(source["logits"].argmax())
    )

    # Is the curve a step or a ramp? Compare the largest single-layer jump in
    # rel_max_abs against the median jump: a kernel that is wrong at one layer
    # produces one jump far above the rest, accumulation does not.
    jumps = [
        {"layer": b["layer"], "delta": round(b["rel_max_abs"] - a["rel_max_abs"], 9)}
        for a, b in zip(per_layer, per_layer[1:])
    ]
    ranked = sorted(jumps, key=lambda j: j["delta"], reverse=True)
    positive = sorted(j["delta"] for j in jumps if j["delta"] > 0)
    median_jump = positive[len(positive) // 2] if positive else 0.0
    largest = ranked[0] if ranked else {"layer": None, "delta": 0.0}

    by_ratio: dict[str, list[float]] = {}
    for entry in per_layer:
        by_ratio.setdefault(str(entry["compress_ratio"]), []).append(entry["rel_max_abs"])

    return {
        "per_layer": per_layer,
        "per_sublayer": per_sublayer,
        "boundaries": boundaries,
        "reading_thresholds": {"rel_max_abs": READING_REL_MAX, "cosine": READING_COSINE},
        "first_layer_over_rel_max": first_rel,
        "first_layer_under_cosine": first_cos,
        "first_layer_under_cosine_last_row": first_last_row_cos,
        "largest_single_layer_jump": largest,
        "median_positive_jump": round(median_jump, 9),
        "jump_ratio_largest_over_median": (
            round(largest["delta"] / median_jump, 3) if median_jump > 0 else None
        ),
        "mean_rel_max_abs_by_compress_ratio": {
            ratio: round(sum(values) / len(values), 9) for ratio, values in by_ratio.items()
        },
        "top_jumps": ranked[:5],
        "reading": (
            "a single wrong kernel shows up as one jump far above the median; a curve "
            "that climbs steadily from layer 0 with a jump ratio near 1 is accumulated "
            "rounding, which is a precision decision rather than a layer to fix"
        ),
    }


def attention_gemm_report(live: Any, layer_ids: Sequence[int]) -> dict[str, Any]:
    """Which GEMM each attention projection actually resolved to.

    The sub-boundary chain names *attention* as the owner; this names the
    kernels inside it, from the live modules rather than from the code that
    was supposed to configure them. A run that reported a parity result while
    every projection still resolved to the shipped method would be reporting
    the wrong experiment.
    """
    from tensorrt_llm._torch.modules.linear import Linear

    layers = live.model.model.layers
    report: dict[str, Any] = {}
    for lid in layer_ids:
        attention = layers[lid].self_attn
        by_method: dict[str, list[str]] = {}
        for name, module in attention.named_modules():
            if not isinstance(module, Linear):
                continue
            method = type(getattr(module, "quant_method", None)).__name__
            by_method.setdefault(method, []).append(name)
        report[str(lid)] = {
            method: sorted(names) for method, names in sorted(by_method.items())
        }
    return report


def ratio_problems(live_ratios: list[int], captured: list[int] | None, num_layers: int) -> list[str]:
    """The two stacks must agree on which layer compresses how much.

    Cheap, and it fails loudly rather than producing a curve whose per-ratio
    grouping quietly means something different on each side.
    """
    if captured is None:
        return ["the source capture recorded no compress ratios"]
    got, ref = list(live_ratios)[:num_layers], list(captured)[:num_layers]
    if got != ref:
        differing = [i for i, (a, b) in enumerate(zip(got, ref)) if a != b]
        return [f"compress ratios differ from the source at layers {differing[:8]}"]
    return []


def compare_all(
    live: Any,
    prompts: list[dict[str, Any]],
    source_captures: dict[str, Any],
    ratios: list[int],
    tg: Any,
    ranks: Any,
    deep_layers: Sequence[int] = (),
) -> dict[str, Any]:
    """The TensorRT-LLM half, judged against the persisted source capture."""
    out: dict[str, Any] = {}
    for prompt in prompts:
        started = time.time()
        got = capture_trtllm(
            live,
            prompt["token_ids"],
            max_seq_len=len(prompt["token_ids"]) + 64,
            deep_layers=deep_layers,
        )
        result = compare_stacks(source_captures[prompt["id"]], got, ratios, tg)
        result["elapsed_s"] = round(time.time() - started, 2)
        out[prompt["id"]] = result
        curve = result["per_layer"]
        ranks.log(
            f"  trtllm prefill {prompt['id']:24s} {result['elapsed_s']}s "
            f"layer0 rel={curve[0]['rel_max_abs']:.4g} "
            f"layer{curve[-1]['layer']} rel={curve[-1]['rel_max_abs']:.4g} "
            f"logits rel={result['boundaries']['logits']['rel_max_abs']:.4g} "
            f"cos={result['boundaries']['logits']['cosine']:.6f}"
        )
        ranks.log(
            f"    first layer over rel {READING_REL_MAX}: {result['first_layer_over_rel_max']}; "
            f"largest jump {result['largest_single_layer_jump']} vs median "
            f"{result['median_positive_jump']} "
            f"(ratio {result['jump_ratio_largest_over_median']})"
        )
        ranks.log(f"    mean rel by ratio {result['mean_rel_max_abs_by_compress_ratio']}")
        ranks.log(
            f"    last row: layer0 cos={curve[0].get('last_row_cosine')} "
            f"layer{curve[-1]['layer']} cos={curve[-1].get('last_row_cosine')}; "
            f"first under {READING_COSINE}: {result['first_layer_under_cosine_last_row']}"
        )
        hc = result["boundaries"].get("hc_head")
        if hc:
            ranks.log(
                f"    hc_head rel={hc['rel_max_abs']:.4g} cos={hc['cosine']:.6f} "
                f"-> final_norm cos={result['boundaries']['final_norm']['cosine']:.6f}"
            )
        for lid, detail in sorted(result["per_sublayer"].items()):
            ranks.log(f"    layer {lid} sub-boundaries (mid via {detail['mid_boundary_path']}):")
            for entry in detail["chain"]:
                ranks.log(
                    f"      {entry['boundary']:14s} rel={entry['rel_max_abs']:11.6g} "
                    f"cos={entry['cosine']:.8f}  last_row rel="
                    f"{entry.get('last_row_rel_max_abs', float('nan')):11.6g} "
                    f"cos={entry.get('last_row_cosine', float('nan')):.8f}"
                )
            ranks.log(f"      first over rel {READING_REL_MAX}: {detail['first_boundary_over_rel_max']}")
        del got
        torch.cuda.empty_cache()
    return out
