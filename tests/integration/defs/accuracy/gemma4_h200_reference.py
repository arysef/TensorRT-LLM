# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Transformers reference captures and pure-Torch goldens for Gemma 4 on H200.

The Gemma 4 26B-A4B BF16 checkpoint is ~50 GiB, so the native Transformers
reference and the TensorRT-LLM path never co-reside on the GPU.  Everything
here runs the reference *first*, moves its artifacts to CPU/disk, and releases
the GPU before the TensorRT-LLM side is constructed.  Captures are memoized on
disk so a rerun of a sibling test does not pay for the reference again.

The golden functions in this module are deliberately an *independent*
re-derivation of the source semantics from ``transformers`` Gemma 4: they never
import or call TensorRT-LLM modeling code.  A golden is only trustworthy after
:func:`assert_golden_matches_source` has aligned it against the hooked native
model on real-checkpoint activations, which is what the attention/MoE replay
tests do before they compare TensorRT-LLM.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# --------------------------------------------------------------------------
# Fixed inputs
# --------------------------------------------------------------------------

# Five fixed text prompts (acceptance requires >= 5) and two fixed image
# prompts (requires >= 2).  They are deliberately short so that
# generation-parity stays a few-minute signal.
TEXT_PROMPTS: Tuple[str, ...] = (
    "Explain in two sentences why the sky appears blue.",
    "List the first five prime numbers and their sum.",
    "Write a haiku about a lighthouse in a winter storm.",
    "What is the difference between a stack and a queue?",
    "Translate 'the library opens at nine' into French.",
)

# Repository-local images, so the reference does not depend on network fetches.
_TEST_INPUT_FILES = Path(__file__).resolve().parents[2] / "test_input_files"

IMAGE_PROMPTS: Tuple[Tuple[str, str], ...] = (
    (str(_TEST_INPUT_FILES / "merlion.png"), "Describe this landmark in one sentence."),
    (
        str(_TEST_INPUT_FILES / "pexels-maxim-shklyaev-1511525-2914194.jpg"),
        "What is the main subject of this photograph?",
    ),
)

# Deterministic decoding on both sides.
GREEDY_MAX_NEW_TOKENS = 32

# Gemma 4 sets ``Gemma4TextAttention.scaling = 1.0`` and passes it explicitly to
# the attention interface, so scores are *not* divided by ``sqrt(head_dim)``.
GEMMA4_ATTENTION_SCALING = 1.0

# Chat rendering must match the configured accuracy runs (see
# ``TestGemma4_26B_A4B`` in test_llm_api_pytorch_multimodal.py).
CHAT_TEMPLATE_KWARGS: Dict[str, Any] = {"enable_thinking": False}


def gemma4_26b_checkpoint() -> str:
    """Absolute path of the native BF16 26B-A4B checkpoint."""
    models_root = os.environ.get("LLM_MODELS_ROOT")
    if not models_root:
        raise RuntimeError("LLM_MODELS_ROOT is not set; cannot locate the Gemma 4 checkpoint.")
    path = Path(models_root) / "gemma" / "gemma-4-26B-A4B-it"
    if not (path / "config.json").is_file():
        raise RuntimeError(f"Gemma 4 checkpoint not found at {path}.")
    return str(path)


def hf_reference_root() -> Path:
    """Root of the Transformers source tree the task supplies as the reference.

    ``task.yaml`` names an exact Transformers commit as an evidence source, and
    the reference-test policy rules out an environment-installed
    ``transformers`` as pass evidence.  ``trtllm_dev.sh`` mounts that tree at
    ``/references`` and puts its ``src`` ahead of site-packages on
    ``PYTHONPATH``; this locates it so the provenance can be asserted rather
    than assumed.
    """
    root = os.environ.get("HF_REFERENCE_ROOT")
    if not root:
        raise RuntimeError(
            "HF_REFERENCE_ROOT is not set. Run through "
            "workspace/gemma4-h200/trtllm_dev.sh, which mounts the supplied "
            "Transformers source and points PYTHONPATH at it; an "
            "environment-installed transformers is not valid reference evidence."
        )
    path = Path(root) / "transformers"
    if not (path / "src" / "transformers" / "__init__.py").is_file():
        raise RuntimeError(f"supplied Transformers source not found under {path}")
    return path


# Commit the task pins as the reference (``transformers_reference_commit``).
EXPECTED_TRANSFORMERS_COMMIT = "75d3bdcd4b3ba70cee21287218d9764f33da41f0"
EXPECTED_TRANSFORMERS_VERSION = "5.5.4"


@functools.lru_cache(maxsize=1)
def transformers_provenance() -> Dict[str, Any]:
    """Assert ``transformers`` resolved to the supplied tree; report its identity.

    Raises rather than warns: a capture taken from a different Transformers is
    not the independent reference this task requires, and silently accepting one
    would make every downstream cosine meaningless.
    """
    import subprocess

    import transformers

    root = hf_reference_root()
    resolved = Path(transformers.__file__).resolve()
    expected_prefix = (root / "src" / "transformers").resolve()
    if not str(resolved).startswith(str(expected_prefix)):
        raise RuntimeError(
            f"transformers resolved to {resolved}, which is outside the supplied "
            f"reference tree {expected_prefix}. PYTHONPATH must place "
            f"{root / 'src'} ahead of site-packages (trtllm_dev.sh does this)."
        )
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"supplied Transformers reports version {transformers.__version__}, "
            f"expected {EXPECTED_TRANSFORMERS_VERSION}"
        )

    commit = "unknown"
    with contextlib.suppress(Exception):
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    if commit != EXPECTED_TRANSFORMERS_COMMIT:
        raise RuntimeError(
            f"supplied Transformers is at commit {commit}, but task.yaml pins "
            f"{EXPECTED_TRANSFORMERS_COMMIT}"
        )
    return {
        "transformers_module": str(resolved),
        "transformers_version": transformers.__version__,
        "transformers_commit": commit,
        "transformers_source_root": str(root),
    }


@functools.lru_cache(maxsize=1)
def checkpoint_eos_token_ids() -> Tuple[int, ...]:
    """Every ``eos_token_id`` the checkpoint's generation config declares.

    The replay/parity references generate a fixed number of new tokens with
    ``min_new_tokens=max_new_tokens``, which Transformers implements by banning
    *all* of these ids from selection for the whole generation.  A TensorRT-LLM
    run that does not ban the same set is not config-matched: it will emit the
    end-of-turn token at the step the reference was forced past, and the
    resulting "divergence" says nothing about the model.  Read from the
    checkpoint rather than hard-coded so the two paths cannot drift apart.
    """
    config_path = Path(gemma4_26b_checkpoint()) / "generation_config.json"
    with open(config_path) as handle:
        eos = json.load(handle).get("eos_token_id")
    if eos is None:
        raise AssertionError(f"{config_path} declares no eos_token_id")
    if isinstance(eos, int):
        eos = [eos]
    return tuple(int(token) for token in eos)


def reference_cache_dir() -> Path:
    """Writable directory holding memoized reference captures.

    Defaults under the workspace rather than a persistent ``HOME`` so captures
    travel with the run's evidence and cannot outlive the source they were
    taken from.
    """
    root = os.environ.get("GEMMA4_H200_REFERENCE_CACHE")
    if root:
        path = Path(root)
    else:
        path = Path("/workspace/run/reference-cache")
        if not path.parent.is_dir():
            path = Path(__file__).resolve().parents[4] / "gemma4-h200-reference-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TensorMetrics:
    """Comparison metrics reported by every replay/parity assertion."""

    max_abs: float
    mean_abs: float
    cosine: float
    shape: Tuple[int, ...]

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def __str__(self) -> str:  # pragma: no cover - human-readable report only
        return (
            f"shape={tuple(self.shape)} max_abs={self.max_abs:.6g} "
            f"mean_abs={self.mean_abs:.6g} cosine={self.cosine:.8f}"
        )


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> TensorMetrics:
    """max-abs / mean-abs / cosine between two same-shaped tensors."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch: actual {tuple(actual.shape)} vs expected {tuple(expected.shape)}"
        )
    a = actual.detach().to(torch.float32).flatten().cpu()
    b = expected.detach().to(torch.float32).flatten().cpu()
    if not torch.isfinite(a).all():
        raise AssertionError("actual tensor contains non-finite values")
    if not torch.isfinite(b).all():
        raise AssertionError("expected tensor contains non-finite values")
    diff = (a - b).abs()
    # Reduce in float64.  These tensors reach millions of elements (one vision
    # layer is 2394x1152), and a float32 dot/norm accumulation over that many
    # near-identical terms rounds enough to report cosines slightly *above* 1 --
    # not a value a cosine can take, which makes the metric look untrustworthy
    # exactly where it matters most.
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    a_norm, b_norm = float(a64.norm()), float(b64.norm())
    if a_norm > 0.0 and b_norm > 0.0:
        cosine = float(torch.dot(a64, b64) / (a_norm * b_norm))
    elif a_norm == 0.0 and b_norm == 0.0:
        # Two all-zero tensors really are identical.
        cosine = 1.0
    else:
        # Exactly one side is all-zero: cosine is undefined, but the tensors are
        # *not* equal.  Reporting 1.0 here would let a collapsed-to-zero output
        # (dropped mask, unwritten KV page, uninitialized graph buffer) pass
        # every cosine gate, so score it as a total mismatch instead.
        cosine = 0.0
    return TensorMetrics(
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        cosine=cosine,
        shape=tuple(actual.shape),
    )


def assert_cosine(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    min_cosine: float = 0.99,
    report: Optional[List[str]] = None,
) -> TensorMetrics:
    """Assert cosine >= ``min_cosine`` and record the metrics line."""
    metrics = compare_tensors(actual, expected)
    line = f"{label}: {metrics}"
    if report is not None:
        report.append(line)
    print(f"[gemma4-h200] {line}", flush=True)
    if metrics.cosine < min_cosine:
        raise AssertionError(f"{label}: cosine {metrics.cosine:.8f} < {min_cosine} ({metrics})")
    return metrics


# --------------------------------------------------------------------------
# Native Transformers reference
# --------------------------------------------------------------------------


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextlib.contextmanager
def native_gemma4(*, device: str = "cuda"):
    """Load the native Transformers Gemma 4 model, then fully release it.

    Yields ``(model, processor, tokenizer)``.  On exit the model is deleted and
    the CUDA allocator is emptied so the TensorRT-LLM engine can claim the GPU.
    """
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    checkpoint = gemma4_26b_checkpoint()
    processor = AutoProcessor.from_pretrained(checkpoint)
    tokenizer = processor.tokenizer
    model = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    try:
        yield model, processor, tokenizer
    finally:
        del model
        del processor
        _free_gpu()


def render_text_prompt(tokenizer, prompt: str) -> List[int]:
    """Render one text prompt through the checkpoint chat template."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    # ``apply_chat_template`` returns a plain list, a tensor, or a
    # ``BatchEncoding`` depending on the template; ``BatchEncoding`` is a
    # ``UserDict``, so test for the mapping protocol rather than ``dict``.
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.flatten().tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(i) for i in ids]


def render_image_prompt(processor, image_path: str, prompt: str) -> Dict[str, Any]:
    """Render one image prompt into processor tensors (token ids + pixel values)."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **CHAT_TEMPLATE_KWARGS,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return dict(inputs)


def text_layer_module(model, layer_idx: int):
    """The ``layer_idx``-th text decoder layer of the conditional-generation model."""
    return model.model.language_model.layers[layer_idx]


def vision_layer_module(model, layer_idx: int):
    """The ``layer_idx``-th vision-tower encoder layer."""
    return model.model.vision_tower.encoder.layers[layer_idx]


def representative_text_layers(config) -> Dict[str, int]:
    """Early/late sliding and early/late full layer indices for this checkpoint.

    Derived from the checkpoint's own ``layer_types`` rather than assumed, so a
    different Gemma 4 variant reports its own inventory instead of inheriting
    a family-wide guess.
    """
    layer_types = list(config.text_config.layer_types)
    sliding = [i for i, t in enumerate(layer_types) if t == "sliding_attention"]
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    if not sliding or not full:
        raise AssertionError(f"checkpoint has no sliding/full mix: {sorted(set(layer_types))}")
    return {
        "sliding_early": sliding[0],
        "sliding_late": sliding[-1],
        "full_early": full[0],
        "full_late": full[-1],
    }


def _layer_rope_spec(config, layer_idx: int) -> Dict[str, Any]:
    """RoPE geometry of one text layer, read from the checkpoint config."""
    text_config = config.text_config
    is_sliding = text_config.layer_types[layer_idx] == "sliding_attention"
    rope_parameters = text_config.rope_parameters or {}
    if is_sliding:
        params = rope_parameters.get("sliding_attention", {})
        return {
            "is_sliding": True,
            "head_dim": text_config.head_dim,
            "theta": params.get("rope_theta", 10_000.0),
            "partial_rotary_factor": 1.0,
            "sliding_window": text_config.sliding_window,
            "num_kv_heads": text_config.num_key_value_heads,
        }
    params = rope_parameters.get("full_attention", {})
    use_k_eq_v = getattr(text_config, "attention_k_eq_v", False)
    return {
        "is_sliding": False,
        "head_dim": text_config.global_head_dim or text_config.head_dim,
        "theta": params.get("rope_theta", 1_000_000.0),
        "partial_rotary_factor": params.get("partial_rotary_factor", 0.25),
        "sliding_window": None,
        "num_kv_heads": (
            text_config.num_global_key_value_heads
            if use_k_eq_v
            else text_config.num_key_value_heads
        ),
    }


@torch.inference_mode()
def golden_text_attention_layer(
    attn_module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    *,
    spec: Dict[str, Any],
    allow_mask: Optional[torch.Tensor] = None,
    norm_eps: float,
) -> Dict[str, torch.Tensor]:
    """Independent re-derivation of one Gemma 4 text attention layer.

    Reads only *weights* from ``attn_module``; every operation (norms, RoPE
    pairing, K=V ordering, masking, softmax, ``o_proj``) is recomputed here so
    a bug in the TensorRT-LLM path cannot be mirrored by the golden.
    """
    head_dim = spec["head_dim"]
    hidden = hidden_states.squeeze(0)  # [T, hidden]
    seq_len = hidden.shape[0]

    q = torch.nn.functional.linear(hidden, attn_module.q_proj.weight).view(seq_len, -1, head_dim)
    k_raw = torch.nn.functional.linear(hidden, attn_module.k_proj.weight).view(
        seq_len, -1, head_dim
    )
    if attn_module.v_proj is None:
        # K=V layers: V is the *raw* k_proj output, normalized by the weightless
        # v_norm; k_norm and RoPE apply to K only.
        v_raw = k_raw
    else:
        v_raw = torch.nn.functional.linear(hidden, attn_module.v_proj.weight).view(
            seq_len, -1, head_dim
        )

    q = golden_rms_norm(q, attn_module.q_norm.weight, norm_eps)
    k = golden_rms_norm(k_raw, attn_module.k_norm.weight, norm_eps)
    v = golden_rms_norm(v_raw, None, norm_eps)

    cos, sin = golden_rope_cos_sin(
        positions,
        head_dim,
        spec["theta"],
        partial_rotary_factor=spec["partial_rotary_factor"],
    )
    cos = cos.to(hidden.device)
    sin = sin.to(hidden.device)
    q = golden_apply_rope(q, cos, sin)
    k = golden_apply_rope(k, cos, sin)

    if allow_mask is None:
        allow_mask = golden_causal_mask(positions, positions, sliding_window=spec["sliding_window"])
    allow_mask = allow_mask.to(hidden.device)

    # Gemma 4 attention uses ``scaling = 1.0``: ``Gemma4TextAttention`` sets
    # ``self.scaling = 1.0`` and passes it explicitly, so the usual
    # ``head_dim ** -0.5`` factor is *not* applied.  (TensorRT-LLM expresses the
    # same thing as ``q_scaling = 1 / sqrt(head_dim)``, which cancels its own
    # ``1 / (sqrt(head_dim) * q_scaling)``.)
    context = golden_attention(q, k, v, allow_mask, scaling=GEMMA4_ATTENTION_SCALING)
    out = torch.nn.functional.linear(context.reshape(seq_len, -1), attn_module.o_proj.weight)
    return {"q": q, "k": k, "v": v, "context": context, "output": out.unsqueeze(0)}


@torch.inference_mode()
def golden_moe_block(
    router_module,
    experts_module,
    hidden_states: torch.Tensor,
    *,
    expert_input_norm_weight: torch.Tensor,
    norm_eps: float,
    top_k: int,
) -> Dict[str, torch.Tensor]:
    """Independent re-derivation of the Gemma 4 router-plus-experts path.

    The router and the experts do **not** see the same tensor: the source
    routes on the pre-MLP residual and then feeds the experts that residual
    after ``pre_feedforward_layernorm_2``.  Collapsing the two inputs silently
    changes expert outputs while leaving router logits correct, so the norm
    weight is a required argument rather than an optional refinement.
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    probabilities, weights, index = golden_router(
        flat,
        norm_eps=norm_eps,
        scale=router_module.scale,
        proj_weight=router_module.proj.weight,
        per_expert_scale=router_module.per_expert_scale,
        top_k=top_k,
    )
    expert_input = golden_rms_norm(flat, expert_input_norm_weight, norm_eps)
    expert_out = golden_experts(
        expert_input,
        index,
        weights,
        gate_up_proj=experts_module.gate_up_proj,
        down_proj=experts_module.down_proj,
    )
    return {
        "router_probabilities": probabilities,
        "top_k_weights": weights,
        "top_k_index": index,
        "expert_output": expert_out,
    }


@torch.inference_mode()
def capture_text_reference(
    *,
    max_new_tokens: int = GREEDY_MAX_NEW_TOKENS,
    layer_names: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Native-Transformers greedy reference for :data:`TEXT_PROMPTS`.

    Uses the source model's own ``generate()`` (``do_sample=False``) rather than
    a hand-written decode loop, so the reference tokens are canonical HF greedy
    output and need no separate golden fixture to anchor them.

    Returns, per prompt: rendered token ids, greedy token ids, raw per-step
    logits, the prefill last-position logits, and hidden states entering and
    leaving the representative decoder layers.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(gemma4_26b_checkpoint())
    layers = layer_names or representative_text_layers(config)

    prompts: List[Dict[str, Any]] = []
    with native_gemma4() as (model, _processor, tokenizer):
        device = next(model.parameters()).device
        for prompt in TEXT_PROMPTS:
            input_ids = render_text_prompt(tokenizer, prompt)
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)

            with ActivationRecorder() as recorder:
                for name, layer_idx in layers.items():
                    recorder.watch(f"{name}@{layer_idx}", text_layer_module(model, layer_idx))
                prefill = model(input_ids=ids, use_cache=False)
            prefill_logits = prefill.logits[0, -1, :].float().cpu()

            generated = model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_logits=True,
                return_dict_in_generate=True,
            )
            new_tokens = generated.sequences[0, ids.shape[1] :].cpu().tolist()
            step_logits = torch.stack([s[0].float().cpu() for s in generated.logits])

            prompts.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "greedy_tokens": new_tokens,
                    "greedy_text": tokenizer.decode(new_tokens),
                    "prefill_last_logits": prefill_logits,
                    "step_logits": step_logits,
                    "activations": recorder.records,
                }
            )
    return {"layers": layers, "prompts": prompts, "max_new_tokens": max_new_tokens}


@torch.inference_mode()
def capture_image_reference(
    *,
    max_new_tokens: int = GREEDY_MAX_NEW_TOKENS,
    vision_layers: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Native-Transformers greedy reference for :data:`IMAGE_PROMPTS`.

    Also records the measured contiguous image soft-token run per prompt.  The
    checkpoint's ``vision_soft_tokens_per_image`` (280) is the configured
    upper bound; the processor emits a smaller image-dependent run, and the
    chunk-alignment contract is about *that* measured run.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(gemma4_26b_checkpoint())
    text_layers = representative_text_layers(config)
    tower_depth = config.vision_config.num_hidden_layers
    vision_layers = vision_layers or {"vision_early": 0, "vision_late": tower_depth - 1}

    prompts: List[Dict[str, Any]] = []
    with native_gemma4() as (model, processor, tokenizer):
        device = next(model.parameters()).device
        for image_path, question in IMAGE_PROMPTS:
            inputs = render_image_prompt(processor, image_path, question)
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            input_ids = inputs["input_ids"][0].cpu().tolist()
            token_type_ids = inputs["mm_token_type_ids"][0].cpu()

            with ActivationRecorder() as recorder:
                for name, layer_idx in text_layers.items():
                    recorder.watch(f"{name}@{layer_idx}", text_layer_module(model, layer_idx))
                for name, layer_idx in vision_layers.items():
                    recorder.watch(f"{name}@{layer_idx}", vision_layer_module(model, layer_idx))
                # ``embed_vision`` is the multimodal projector that maps pooled
                # tower features into the language embedding space.
                recorder.watch("multimodal_projector", model.model.embed_vision)
                recorder.watch("vision_pooler", model.model.vision_tower.pooler)
                prefill = model(**inputs, use_cache=False)
            prefill_logits = prefill.logits[0, -1, :].float().cpu()

            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_logits=True,
                return_dict_in_generate=True,
            )
            new_tokens = generated.sequences[0, len(input_ids) :].cpu().tolist()
            step_logits = torch.stack([s[0].float().cpu() for s in generated.logits])

            prompts.append(
                {
                    "image": image_path,
                    "prompt": question,
                    "input_ids": input_ids,
                    "mm_token_type_ids": token_type_ids,
                    "image_token_runs": contiguous_image_runs(token_type_ids),
                    "greedy_tokens": new_tokens,
                    "greedy_text": tokenizer.decode(new_tokens),
                    "prefill_last_logits": prefill_logits,
                    "step_logits": step_logits,
                    "activations": recorder.records,
                }
            )
    return {
        "text_layers": text_layers,
        "vision_layers": vision_layers,
        "configured_soft_tokens_per_image": config.vision_soft_tokens_per_image,
        "prompts": prompts,
        "max_new_tokens": max_new_tokens,
    }


@torch.inference_mode()
def align_goldens_to_source(*, prompt_index: int = 0) -> Dict[str, Any]:
    """Anchor every pure-Torch golden against the hooked native model.

    This is step 2 of the reference ladder: a golden that only agrees with
    itself proves nothing, so each one is compared here against the real
    checkpoint's own module outputs before any TensorRT-LLM comparison uses it.
    Returns the metric lines plus the layer inventory that was exercised.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(gemma4_26b_checkpoint())
    text_config = config.text_config
    layers = representative_text_layers(config)
    norm_eps = text_config.rms_norm_eps
    report: List[str] = []
    results: Dict[str, Any] = {"layers": layers, "metrics": {}}

    with native_gemma4() as (model, _processor, tokenizer):
        device = next(model.parameters()).device
        input_ids = render_text_prompt(tokenizer, TEXT_PROMPTS[prompt_index])
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        positions = torch.arange(len(input_ids), device=device)

        recorder = ActivationRecorder()
        modules: Dict[str, Any] = {}
        for name, layer_idx in layers.items():
            layer = text_layer_module(model, layer_idx)
            modules[name] = layer
            recorder.watch(f"attn:{name}", layer.self_attn)
            recorder.watch(f"router:{name}", layer.router)
            recorder.watch(f"experts:{name}", layer.experts)
        with recorder:
            model(input_ids=ids, use_cache=False)

        for name, layer_idx in layers.items():
            layer = modules[name]
            spec = _layer_rope_spec(config, layer_idx)

            attn_record = recorder.records[f"attn:{name}"]
            attn_kwargs = attn_record["kwargs"]
            attn_hidden = attn_kwargs["hidden_states"].to(device)
            source_attn_out = attn_record["output"][0].to(device)
            golden_attn = golden_text_attention_layer(
                layer.self_attn,
                attn_hidden,
                positions,
                spec=spec,
                norm_eps=norm_eps,
            )
            results["metrics"][f"attention[{name}@{layer_idx}]"] = assert_golden_matches_source(
                golden_attn["output"],
                source_attn_out,
                label=f"attention[{name}@{layer_idx}] "
                f"head_dim={spec['head_dim']} kv_heads={spec['num_kv_heads']} "
                f"window={spec['sliding_window']} theta={spec['theta']} "
                f"partial={spec['partial_rotary_factor']}",
                report=report,
            ).as_dict()

            router_record = recorder.records[f"router:{name}"]
            router_hidden = router_record["args"][0].to(device)
            _src_probs, src_weights, src_index = (t.to(device) for t in router_record["output"])
            experts_record = recorder.records[f"experts:{name}"]
            src_expert_out = experts_record["output"].to(device)

            golden_moe = golden_moe_block(
                layer.router,
                layer.experts,
                router_hidden,
                expert_input_norm_weight=layer.pre_feedforward_layernorm_2.weight,
                norm_eps=norm_eps,
                top_k=text_config.top_k_experts,
            )
            if not torch.equal(golden_moe["top_k_index"], src_index):
                mismatched = int((golden_moe["top_k_index"] != src_index).sum())
                raise AssertionError(
                    f"router[{name}@{layer_idx}]: golden selected different experts "
                    f"({mismatched} of {src_index.numel()} slots differ)"
                )
            report.append(
                f"router[{name}@{layer_idx}]: top-{text_config.top_k_experts} expert indices identical"
            )
            results["metrics"][f"routing_weights[{name}@{layer_idx}]"] = (
                assert_golden_matches_source(
                    golden_moe["top_k_weights"],
                    src_weights,
                    label=f"routing_weights[{name}@{layer_idx}]",
                    report=report,
                ).as_dict()
            )
            results["metrics"][f"expert_output[{name}@{layer_idx}]"] = assert_golden_matches_source(
                golden_moe["expert_output"],
                src_expert_out.reshape(golden_moe["expert_output"].shape),
                label=f"expert_output[{name}@{layer_idx}]",
                report=report,
            ).as_dict()

    results["report"] = report
    return results


def source_allow_mask(raw_mask: Any, seq_len: int) -> torch.Tensor:
    """Normalize a source attention mask to a boolean ``[Tq, Tkv]`` allow-mask.

    Transformers hands attention either an additive float mask (0 / -inf) or a
    boolean mask, with leading batch/head dims.
    """
    if raw_mask is None:
        raise AssertionError("source produced no attention mask; cannot compare mask semantics")
    mask = raw_mask
    while mask.dim() > 2:
        mask = mask[0]
    if mask.shape != (seq_len, seq_len):
        raise AssertionError(
            f"unexpected source mask shape {tuple(raw_mask.shape)} for seq_len {seq_len}"
        )
    if mask.dtype == torch.bool:
        return mask
    return mask > torch.finfo(mask.dtype).min / 2


@torch.inference_mode()
def align_image_goldens_to_source(*, prompt_index: int = 0) -> Dict[str, Any]:
    """Anchor the multimodal mask golden against the source's own masks.

    Gemma 4 makes each contiguous image soft-token run bidirectional **only on
    sliding-attention layers**; full-attention layers stay causal.  This
    compares the golden allow-mask against the mask the source actually hands
    to a sliding layer and to a full layer for a real image prompt, and
    verifies the bidirectional block is discriminating (i.e. it really differs
    from plain causal).
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(gemma4_26b_checkpoint())
    layers = representative_text_layers(config)
    report: List[str] = []
    results: Dict[str, Any] = {"layers": layers, "checks": report}

    image_path, question = IMAGE_PROMPTS[prompt_index]
    with native_gemma4() as (model, processor, tokenizer):
        del tokenizer
        device = next(model.parameters()).device
        inputs = render_image_prompt(processor, image_path, question)
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        token_type_ids = inputs["mm_token_type_ids"][0].cpu()
        runs = contiguous_image_runs(token_type_ids)

        recorder = ActivationRecorder()
        for name, layer_idx in layers.items():
            recorder.watch_pre(f"attn:{name}", text_layer_module(model, layer_idx).self_attn)
        with recorder:
            model(**inputs, use_cache=False)

        positions = torch.arange(seq_len)
        causal_only = golden_causal_mask(
            positions, positions, sliding_window=config.text_config.sliding_window
        )
        bidirectional = golden_bidirectional_image_mask(
            token_type_ids, positions, sliding_window=config.text_config.sliding_window
        )
        extra = int((bidirectional & ~causal_only).sum())
        if extra == 0:
            raise AssertionError(
                "bidirectional golden is not discriminating: it allows nothing beyond causal, "
                "so it cannot detect a dropped custom mask"
            )
        report.append(
            f"image runs={runs} seq_len={seq_len} "
            f"bidirectional-only positions={extra} (golden differs from causal)"
        )

        for name, layer_idx in layers.items():
            is_sliding = config.text_config.layer_types[layer_idx] == "sliding_attention"
            kwargs = recorder.records[f"attn:{name}"]["kwargs"]
            observed = source_allow_mask(kwargs.get("attention_mask"), seq_len)
            expected = bidirectional if is_sliding else golden_causal_mask(positions, positions)
            mismatched = int((observed != expected).sum())
            kind = "sliding/bidirectional" if is_sliding else "full/causal"
            if mismatched:
                raise AssertionError(
                    f"mask[{name}@{layer_idx}] ({kind}): {mismatched} of {seq_len * seq_len} "
                    "positions differ from the golden"
                )
            report.append(
                f"mask[{name}@{layer_idx}] ({kind}): identical to golden over {seq_len}x{seq_len}"
            )
            print(f"[gemma4-h200] {report[-1]}", flush=True)

    results["image_runs"] = runs
    results["seq_len"] = seq_len
    return results


def contiguous_image_runs(
    token_type_ids: torch.Tensor, image_token_type: int = 1
) -> List[Tuple[int, int]]:
    """``(start, length)`` of each contiguous image soft-token run."""
    values = token_type_ids.flatten().tolist()
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, value in enumerate(values):
        if value == image_token_type and start is None:
            start = idx
        elif value != image_token_type and start is not None:
            runs.append((start, idx - start))
            start = None
    if start is not None:
        runs.append((start, len(values) - start))
    return runs


# --------------------------------------------------------------------------
# Activation capture
# --------------------------------------------------------------------------


class ActivationRecorder:
    """Forward-hook recorder keeping every capture on CPU.

    Captures are stored under caller-chosen names so the replay tests can ask
    for exactly the module boundaries they compare (attention input/output,
    router logits, expert output, ...) without holding GPU memory.
    """

    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self._handles: List[Any] = []

    def _store(self, name: str, key: str, value: Any) -> None:
        slot = self.records.setdefault(name, {})
        slot[key] = _to_cpu(value)

    def watch(self, name: str, module: torch.nn.Module) -> None:
        """Record ``module``'s positional inputs, kwargs, and output."""

        def hook(_mod, args, kwargs, output):
            self._store(name, "args", args)
            self._store(name, "kwargs", kwargs)
            self._store(name, "output", output)

        self._handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def watch_pre(self, name: str, module: torch.nn.Module) -> None:
        """Record only ``module``'s inputs (cheaper for large submodules)."""

        def hook(_mod, args, kwargs):
            self._store(name, "args", args)
            self._store(name, "kwargs", kwargs)

        self._handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivationRecorder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove()


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_to_cpu(v) for v in value]
        return type(value)(converted) if not isinstance(value, tuple) else tuple(converted)
    return value


# --------------------------------------------------------------------------
# On-disk memoization
# --------------------------------------------------------------------------


def capture_key(*parts: Any) -> str:
    """Stable cache key for a capture request."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def cached_capture(name: str, key: str, producer: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Return a memoized capture, producing (and persisting) it on a miss."""
    path = reference_cache_dir() / f"{name}-{key}.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)
    payload = producer()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return payload


# --------------------------------------------------------------------------
# Pure-Torch goldens (independent re-derivation of the source semantics)
# --------------------------------------------------------------------------


def golden_rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """Gemma 4 RMSNorm: float32 normalize, then scale by plain ``w``.

    Note the convention change from Gemma 1-3, which scaled by ``1 + w``.
    Gemma 4's ``Gemma4RMSNorm.forward`` multiplies by ``self.weight`` directly,
    and a weightless norm (``with_scale=False``) applies no scale at all.
    """
    dtype = x.dtype
    xf = x.float()
    normed = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        normed = normed * weight.float()
    return normed.to(dtype)


def golden_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF ``rotate_half``: split at ``head_dim // 2`` and pair ``i`` with ``i + d/2``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def golden_rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
    partial_rotary_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape ``[T, head_dim]`` for default and proportional Gemma 4 RoPE.

    ``partial_rotary_factor < 1`` (the full-attention layers) keeps only the
    first ``partial_rotary_factor * head_dim / 2`` frequency pairs live; the
    remainder is zero-frequency, i.e. ``cos = 1`` / ``sin = 0``, while the
    ``rotate_half`` pairing still splits at ``head_dim // 2``.
    """
    half = head_dim // 2
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    exponents = torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / (theta ** (exponents / head_dim))
    if rope_angles < half:
        inv_freq = torch.cat(
            [
                inv_freq,
                torch.zeros(half - rope_angles, dtype=torch.float32, device=positions.device),
            ]
        )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def golden_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``[T, H, D]`` given ``[T, D]`` cos/sin."""
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return x * cos + golden_rotate_half(x) * sin


def golden_causal_mask(
    q_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    *,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Boolean allow-mask ``[Tq, Tkv]`` for causal / sliding-window attention.

    Matches ``sliding_window_mask_function`` in the source: a key is visible
    when it is causal *and* strictly inside the ``sliding_window`` span.
    """
    q = q_positions.view(-1, 1)
    k = kv_positions.view(1, -1)
    allow = k <= q
    if sliding_window is not None:
        allow = allow & (k > q - sliding_window)
    return allow


def golden_bidirectional_image_mask(
    token_type_ids: torch.Tensor,
    q_positions: torch.Tensor,
    *,
    sliding_window: Optional[int],
    image_token_type: int = 1,
) -> torch.Tensor:
    """Allow-mask where each contiguous image run attends bidirectionally.

    Text stays causal.  Two image tokens see each other only when they belong
    to the *same* contiguous run, which is what keeps separate images from
    leaking into one another.
    """
    token_type_ids = token_type_ids.flatten()
    seq_len = token_type_ids.numel()
    allow = golden_causal_mask(q_positions, q_positions, sliding_window=sliding_window)

    is_image = token_type_ids == image_token_type
    # Label each contiguous image run; non-image positions get label -1.
    run_id = torch.cumsum((~is_image).to(torch.int64), dim=0)
    run_id = torch.where(is_image, run_id, torch.full_like(run_id, -1))

    same_run = (
        (run_id.view(-1, 1) == run_id.view(1, -1)) & is_image.view(-1, 1) & is_image.view(1, -1)
    )
    allow = allow | same_run
    if sliding_window is not None:
        # Bidirectionality never escapes the window.
        within = (q_positions.view(-1, 1) - q_positions.view(1, -1)).abs() < sliding_window
        allow = allow & within
    assert allow.shape == (seq_len, seq_len)
    return allow


def golden_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow_mask: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    """Reference SDPA over ``[T, H, D]`` q and ``[T, Hkv, D]`` k/v with an allow-mask."""
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    repeat = num_heads // num_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)

    qf = q.transpose(0, 1).float()  # [H, Tq, D]
    kf = k.transpose(0, 1).float()  # [H, Tkv, D]
    vf = v.transpose(0, 1).float()

    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scaling
    bias = torch.zeros_like(allow_mask, dtype=torch.float32)
    bias.masked_fill_(~allow_mask, float("-inf"))
    scores = scores + bias.unsqueeze(0)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, vf)  # [H, Tq, D]
    return out.transpose(0, 1).to(q.dtype)


def golden_router(
    hidden_states: torch.Tensor,
    *,
    norm_eps: float,
    scale: torch.Tensor,
    proj_weight: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gemma 4 router: weightless RMSNorm, learned scale, ``h^-0.5``, FP32 softmax, top-k.

    Returns ``(router_probabilities, top_k_weights, top_k_index)`` where the
    weights are renormalized and then multiplied by ``per_expert_scale``.
    """
    hidden_size = hidden_states.shape[-1]
    normed = golden_rms_norm(hidden_states, None, norm_eps)
    normed = normed * scale.to(normed.dtype) * (hidden_size**-0.5)
    logits = torch.nn.functional.linear(normed, proj_weight.to(normed.dtype))
    probabilities = torch.softmax(logits.float(), dim=-1)
    top_k_weights, top_k_index = torch.topk(probabilities, k=top_k, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights * per_expert_scale.float()[top_k_index]
    return probabilities, top_k_weights, top_k_index


def golden_experts(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Gemma 4 routed experts with exact GELU-tanh GeGLU.

    ``gate_up_proj`` is ``[E, 2 * inter, hidden]`` and ``down_proj`` is
    ``[E, hidden, inter]``, matching the source parameter layout.
    """
    out = torch.zeros_like(hidden_states)
    num_experts = gate_up_proj.shape[0]
    for expert_idx in range(num_experts):
        hit = (top_k_index == expert_idx).nonzero()
        if hit.numel() == 0:
            continue
        token_idx, slot = hit[:, 0], hit[:, 1]
        current = hidden_states[token_idx]
        fused = torch.nn.functional.linear(current, gate_up_proj[expert_idx].to(current.dtype))
        gate, up = fused.chunk(2, dim=-1)
        activated = torch.nn.functional.gelu(gate, approximate="tanh") * up
        projected = torch.nn.functional.linear(activated, down_proj[expert_idx].to(activated.dtype))
        # The routing weights stay in fp32 through the scale, exactly as the
        # source does: rounding them to bf16 first costs ~1e-2 of cosine.
        projected = projected * top_k_weights[token_idx, slot, None]
        out.index_add_(0, token_idx, projected.to(out.dtype))
    return out


def golden_logit_softcap(logits: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
    """Gemma final-logit soft cap: ``cap * tanh(logits / cap)``."""
    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


def assert_golden_matches_source(
    golden: torch.Tensor,
    source: torch.Tensor,
    *,
    label: str,
    min_cosine: float = 0.999,
    report: Optional[List[str]] = None,
) -> TensorMetrics:
    """Align a golden against the hooked native model before it is used as a reference.

    A golden that has not been anchored to a real-checkpoint source capture is
    not evidence: it only proves self-consistency.  The threshold is tighter
    than the TensorRT-LLM gate because both sides here are plain Torch.
    """
    return assert_cosine(
        golden, source, label=f"golden-vs-source[{label}]", min_cosine=min_cosine, report=report
    )


# --------------------------------------------------------------------------
# Environment / dispatch reporting
# --------------------------------------------------------------------------


def h200_environment_report() -> Dict[str, Any]:
    """Hardware / package provenance recorded by every pass-critical node."""
    import subprocess

    import transformers

    props = torch.cuda.get_device_properties(0)
    driver = "unknown"
    with contextlib.suppress(Exception):
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()

    flashinfer_version = "not installed"
    with contextlib.suppress(Exception):
        import flashinfer

        flashinfer_version = flashinfer.__version__

    commit, dirty = "unknown", "unknown"
    repo_root = Path(__file__).resolve().parents[4]
    with contextlib.suppress(Exception):
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
            ).strip()
        )

    return {
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_memory_gib": round(props.total_memory / (1024**3), 1),
        "gpu_count": torch.cuda.device_count(),
        "driver_version": driver,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "flashinfer_version": flashinfer_version,
        "trtllm_commit": commit,
        "trtllm_dirty": dirty,
        "checkpoint": gemma4_26b_checkpoint(),
        "reference_cache_dir": str(reference_cache_dir()),
        **transformers_provenance(),
    }


def require_single_h200() -> Dict[str, Any]:
    """Fail (never skip) unless exactly one SM90 H200 is visible.

    Pass-critical nodes must not report a skip as success, so this raises
    instead of calling ``pytest.skip``.
    """
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is not available; H200 nodes must execute on GPU.")
    env = h200_environment_report()
    if env["compute_capability"] != "9.0":
        raise AssertionError(
            f"expected SM90 (H200), got compute capability {env['compute_capability']}"
        )
    if "H200" not in env["gpu_name"]:
        raise AssertionError(f"expected an H200, got {env['gpu_name']!r}")
    print(f"[gemma4-h200] environment: {json.dumps(env, indent=2)}", flush=True)
    return env


def write_evidence(name: str, payload: Dict[str, Any]) -> Path:
    """Persist a node's evidence as JSON next to the reference cache."""
    path = reference_cache_dir() / f"evidence-{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[gemma4-h200] evidence written to {path}", flush=True)
    return path


# --------------------------------------------------------------------------
# Replay captures (produced in a subprocess, memoized on disk)
# --------------------------------------------------------------------------

# Bump when the *content* of a capture changes so stale files are not reused.
CAPTURE_VERSION = 5


def _watch_replay_boundaries(recorder: "ActivationRecorder", layer, prefix: str) -> None:
    """Hook the exact module boundaries the TensorRT-LLM replay compares.

    Chosen to line up one-for-one with the TensorRT-LLM module tree:

    ``self_attn``   input/output -> ``Gemma4Attention``
    ``router``      input/output -> ``Gemma4MoE.router`` (pre-MLP residual)
    ``router.proj`` output       -> ``Gemma4Router.forward``'s raw expert scores
    ``experts``     input/output -> ``Gemma4MoE.experts`` (post ``pre_feedforward_layernorm_2``)
    ``post_ffn_2``  output       -> the post-MoE contribution folded back into
                                    the residual stream

    The router and the experts deliberately see *different* tensors, so both
    are captured rather than one being derived from the other.  The router
    module's own output is ``(probabilities, top_k_weights, top_k_index)``,
    i.e. the *post-softmax* distribution, while TensorRT-LLM's
    ``Gemma4Router.forward`` returns the pre-softmax scores -- so ``proj`` is
    hooked separately to give that boundary an exact source counterpart
    instead of comparing across a softmax.
    """
    recorder.watch(f"{prefix}:attn", layer.self_attn)
    recorder.watch(f"{prefix}:router", layer.router)
    recorder.watch(f"{prefix}:router_proj", layer.router.proj)
    recorder.watch(f"{prefix}:experts", layer.experts)
    recorder.watch(f"{prefix}:post_ffn_2", layer.post_feedforward_layernorm_2)


def _replay_layer_record(
    recorder: "ActivationRecorder", prefix: str, layer_idx: int
) -> Dict[str, Any]:
    """Flatten one layer's hooked boundaries into the replay record."""
    attn = recorder.records[f"{prefix}:attn"]
    router = recorder.records[f"{prefix}:router"]
    experts = recorder.records[f"{prefix}:experts"]
    probabilities, top_k_weights, top_k_index = router["output"]
    return {
        "layer_idx": layer_idx,
        # Attention module boundary (input is already post-input_layernorm).
        "attn_in": attn["kwargs"]["hidden_states"],
        "attn_out": attn["output"][0],
        # MoE boundaries.
        "router_in": router["args"][0],
        "router_logits": recorder.records[f"{prefix}:router_proj"]["output"],
        "router_probabilities": probabilities,
        "moe_in": experts["args"][0],
        "expert_out": experts["output"],
        # Post-MoE norm output: what the layer actually folds back into the
        # residual stream, so a drift between expert output and residual
        # contribution cannot hide.
        "post_moe_out": recorder.records[f"{prefix}:post_ffn_2"]["output"],
        "top_k_index": top_k_index,
        "top_k_weights": top_k_weights,
    }


@torch.inference_mode()
def _capture_decode_boundaries(model, ids: torch.Tensor, layers: Dict[str, int]) -> Dict[str, Any]:
    """Attention boundaries at the first *cached decode* step.

    Prefill replay proves the context path; it says nothing about whether the
    same layer produces source-equivalent output when its keys and values come
    back out of a paged cache.  This runs the native model's own prefill with
    ``use_cache=True``, appends the greedy next token, and hooks the same
    layers for that one-token forward -- the boundary a cached-decode replay
    compares against.
    """
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    prefill = model(input_ids=ids, use_cache=True, past_key_values=cache)
    next_token = prefill.logits[0, -1, :].argmax(dim=-1).reshape(1, 1)

    recorder = ActivationRecorder()
    for name, layer_idx in layers.items():
        recorder.watch(f"{name}:attn", text_layer_module(model, layer_idx).self_attn)
    with recorder:
        model(input_ids=next_token, use_cache=True, past_key_values=cache)

    return {
        "next_token": int(next_token.item()),
        "cached_tokens": int(ids.shape[1]),
        "layers": {
            f"{name}@{layer_idx}": {
                "layer_idx": layer_idx,
                "attn_in": recorder.records[f"{name}:attn"]["kwargs"]["hidden_states"],
                "attn_out": recorder.records[f"{name}:attn"]["output"][0],
            }
            for name, layer_idx in layers.items()
        },
    }


@torch.inference_mode()
def capture_text_replay_reference(*, max_new_tokens: int = GREEDY_MAX_NEW_TOKENS) -> Dict[str, Any]:
    """Everything the text replay/parity nodes compare, for every fixed prompt.

    One native-model load produces: rendered token ids, prefill last-position
    logits, greedy tokens, per-step logits, and the attention/router/expert
    boundary tensors of the representative sliding and full layers.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(gemma4_26b_checkpoint())
    layers = representative_text_layers(config)

    prompts: List[Dict[str, Any]] = []
    with native_gemma4() as (model, _processor, tokenizer):
        device = next(model.parameters()).device
        for prompt in TEXT_PROMPTS:
            input_ids = render_text_prompt(tokenizer, prompt)
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)

            recorder = ActivationRecorder()
            for name, layer_idx in layers.items():
                _watch_replay_boundaries(recorder, text_layer_module(model, layer_idx), name)
            with recorder:
                prefill = model(input_ids=ids, use_cache=False)
            layer_records = {
                f"{name}@{layer_idx}": _replay_layer_record(recorder, name, layer_idx)
                for name, layer_idx in layers.items()
            }
            decode_record = _capture_decode_boundaries(model, ids, layers)

            generated = model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_logits=True,
                return_dict_in_generate=True,
            )
            prompts.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "prefill_last_logits": prefill.logits[0, -1, :].float().cpu(),
                    "greedy_tokens": generated.sequences[0, ids.shape[1] :].cpu().tolist(),
                    "greedy_text": tokenizer.decode(
                        generated.sequences[0, ids.shape[1] :].cpu().tolist()
                    ),
                    "step_logits": torch.stack([s[0].float().cpu() for s in generated.logits]),
                    "layers": layer_records,
                    "decode": decode_record,
                }
            )
    return {
        "kind": "text_replay",
        "version": CAPTURE_VERSION,
        "layers": layers,
        "max_new_tokens": max_new_tokens,
        "prompts": prompts,
    }


@torch.inference_mode()
def capture_image_replay_reference(
    *, max_new_tokens: int = GREEDY_MAX_NEW_TOKENS
) -> Dict[str, Any]:
    """The image counterpart of :func:`capture_text_replay_reference`.

    Adds the vision-tower boundaries, the multimodal projector, the source's
    own attention masks on a sliding and a full layer, and the *measured*
    contiguous image soft-token runs.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(gemma4_26b_checkpoint())
    text_layers = representative_text_layers(config)
    tower_depth = config.vision_config.num_hidden_layers
    vision_layers = {"vision_early": 0, "vision_late": tower_depth - 1}

    prompts: List[Dict[str, Any]] = []
    with native_gemma4() as (model, processor, tokenizer):
        device = next(model.parameters()).device
        for image_path, question in IMAGE_PROMPTS:
            inputs = render_image_prompt(processor, image_path, question)
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            input_ids = inputs["input_ids"][0].cpu().tolist()
            token_type_ids = inputs["mm_token_type_ids"][0].cpu()

            recorder = ActivationRecorder()
            for name, layer_idx in text_layers.items():
                _watch_replay_boundaries(recorder, text_layer_module(model, layer_idx), name)
            for name, layer_idx in vision_layers.items():
                recorder.watch(name, vision_layer_module(model, layer_idx))
            recorder.watch("multimodal_projector", model.model.embed_vision)
            with recorder:
                prefill = model(**inputs, use_cache=False)

            layer_records = {
                f"{name}@{layer_idx}": _replay_layer_record(recorder, name, layer_idx)
                for name, layer_idx in text_layers.items()
            }
            seq_len = len(input_ids)
            masks = {
                f"{name}@{layer_idx}": source_allow_mask(
                    recorder.records[f"{name}:attn"]["kwargs"].get("attention_mask"), seq_len
                )
                for name, layer_idx in text_layers.items()
            }
            # The processor pads every image up to a fixed patch budget and
            # marks the unused slots with -1 in ``image_position_ids``.  The
            # native tower runs the padded tensor and masks the padding inside
            # attention; a ragged implementation packs only the real patches.
            # Record the source's own validity mask so a replay can compare the
            # same *real* patches instead of comparing a padded slot count.
            position_ids = inputs.get("image_position_ids")
            if position_ids is None:
                raise AssertionError(
                    "the Gemma 4 processor returned no image_position_ids, so the "
                    "real patch count cannot be derived from the source"
                )
            patch_valid = (position_ids != -1).all(dim=-1)[0].cpu()
            num_valid_patches = int(patch_valid.sum())
            num_padded_patches = int(patch_valid.numel())
            vision_records = {
                f"{name}@{layer_idx}": {
                    "layer_idx": layer_idx,
                    "input": recorder.records[name]["args"][0],
                    "output": recorder.records[name]["output"],
                    # Boolean over the padded patch axis; True = real patch.
                    "patch_valid": patch_valid,
                    "num_valid_patches": num_valid_patches,
                    "num_padded_patches": num_padded_patches,
                }
                for name, layer_idx in vision_layers.items()
            }

            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_logits=True,
                return_dict_in_generate=True,
            )
            prompts.append(
                {
                    "image": image_path,
                    "prompt": question,
                    "input_ids": input_ids,
                    "mm_token_type_ids": token_type_ids,
                    "image_token_runs": contiguous_image_runs(token_type_ids),
                    "prefill_last_logits": prefill.logits[0, -1, :].float().cpu(),
                    "greedy_tokens": generated.sequences[0, seq_len:].cpu().tolist(),
                    "greedy_text": tokenizer.decode(
                        generated.sequences[0, seq_len:].cpu().tolist()
                    ),
                    "step_logits": torch.stack([s[0].float().cpu() for s in generated.logits]),
                    "layers": layer_records,
                    "masks": masks,
                    "vision_layers": vision_records,
                    "patch_valid": patch_valid,
                    "num_valid_patches": num_valid_patches,
                    "num_padded_patches": num_padded_patches,
                    "projector_out": recorder.records["multimodal_projector"]["output"],
                }
            )
    return {
        "kind": "image_replay",
        "version": CAPTURE_VERSION,
        "text_layers": text_layers,
        "vision_layers": vision_layers,
        "configured_soft_tokens_per_image": config.vision_soft_tokens_per_image,
        "max_new_tokens": max_new_tokens,
        "prompts": prompts,
    }


@torch.inference_mode()
def native_greedy_completions(request: Dict[str, Any]) -> Dict[str, Any]:
    """Greedy-decode caller-supplied prompts with the native model.

    ``request`` carries fully rendered prompts so the accuracy canaries can
    drive *byte-identical* input through the native reference and through
    TensorRT-LLM.  Each item is either ``{"input_ids": [...]}`` (preferred for
    text-only work, since identical ids remove every tokenizer question) or
    ``{"text": str, "images": [path, ...]}`` for multimodal items, where the
    processor has to run.  ``text`` is used verbatim -- this function never
    re-applies a chat template, because the caller has already rendered one
    and re-rendering is exactly the drift the canary exists to rule out.

    ``request["record_margins"]`` additionally records, per step, the
    *reference's own* top-2 logit margin.  A canary that asserts exact greedy
    token parity is only meaningful at steps the reference resolves by more
    than the runtime's numerical spread; without the margin a caller cannot
    tell a real divergence from a coin flip, and silently assumes a determinacy
    it never checked.  Off by default, so every existing capture key -- and its
    memoized payload -- is unchanged.
    """
    from PIL import Image

    items: List[Dict[str, Any]] = request["items"]
    max_new_tokens = int(request.get("max_new_tokens", 32))
    record_margins = bool(request.get("record_margins", False))

    completions: List[Dict[str, Any]] = []
    with native_gemma4() as (model, processor, tokenizer):
        device = next(model.parameters()).device
        for item in items:
            image_paths = item.get("images") or []
            if image_paths:
                images = [Image.open(p).convert("RGB") for p in image_paths]
                inputs = processor(text=[item["text"]], images=images, return_tensors="pt")
            elif item.get("input_ids") is not None:
                inputs = {"input_ids": torch.tensor([item["input_ids"]], dtype=torch.long)}
            else:
                inputs = tokenizer([item["text"]], return_tensors="pt", add_special_tokens=False)
            inputs = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dict(inputs).items()
            }
            prompt_len = inputs["input_ids"].shape[1]
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=record_margins,
                output_logits=record_margins,
            )
            sequences = generated.sequences if record_margins else generated
            new_tokens = sequences[0, prompt_len:].cpu().tolist()
            completion = {
                "id": item.get("id"),
                "prompt_len": prompt_len,
                "tokens": new_tokens,
                "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
            }
            if record_margins:
                margins, top2 = [], []
                for row in generated.logits:
                    values, indices = torch.topk(row[0].float(), 2)
                    margins.append(float(values[0] - values[1]))
                    top2.append([int(i) for i in indices])
                completion["step_top2_margin"] = margins
                completion["step_top2"] = top2
            completions.append(completion)
    return {
        "kind": "greedy_completions",
        "version": CAPTURE_VERSION,
        "max_new_tokens": max_new_tokens,
        "completions": completions,
    }


# --------------------------------------------------------------------------
# Accuracy-canary inputs (pure functions; no TensorRT-LLM import)
# --------------------------------------------------------------------------

MMLU_CHOICES = ("A", "B", "C", "D")


def mmlu_five_shot_prompt(dataset_dir: str, subject: str, row: int) -> Tuple[str, str]:
    """Independent re-derivation of the repository MMLU prompt, plus its label.

    Mirrors ``tensorrt_llm.evaluate.mmlu.MMLU.gen_prompt`` /``format_example``
    with ``num_fewshot=5``.  It is re-derived here rather than imported so this
    module stays free of TensorRT-LLM imports (the native reference has to run
    before the runtime exists); the H200 test asserts the two agree character
    for character, so a drift in either is caught rather than assumed away.
    """
    import pandas as pd

    def format_example(frame, idx: int, include_answer: bool) -> str:
        prompt = frame.iloc[idx, 0]
        num_choices = frame.shape[1] - 2
        for j in range(num_choices):
            prompt += "\n{}. {}".format(MMLU_CHOICES[j], frame.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(frame.iloc[idx, num_choices + 1])
        return prompt

    dev = pd.read_csv(f"{dataset_dir}/dev/{subject}_dev.csv", header=None)
    test = pd.read_csv(f"{dataset_dir}/test/{subject}_test.csv", header=None)
    if row >= test.shape[0]:
        raise AssertionError(f"{subject} has {test.shape[0]} rows; index {row} is out of range")

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        subject.replace("_", " ")
    )
    for i in range(5):
        prompt += format_example(dev, i, include_answer=True)
    prompt += format_example(test, row, include_answer=False)
    return prompt, str(test.iloc[row, test.shape[1] - 1])


def mmmu_question_and_images(
    doc: Dict[str, Any], image_dir: Path, stem: str
) -> Tuple[str, List[str]]:
    """MMMU's multiple-choice rendering plus the image files an item references."""
    import ast
    import re

    options = doc.get("options")
    if isinstance(options, str):
        options = ast.literal_eval(options)
    lines = [str(doc["question"])]
    if options:
        lines.extend(
            f"{letter}. {option}" for letter, option in zip("ABCDEFGHIJ", options, strict=False)
        )
        lines.append("Answer with the option's letter from the given choices directly.")
    else:
        lines.append("Answer the question using a single word or phrase.")
    # The dataset marks image slots inline as ``<image N>``; the chat template
    # emits the real image placeholders, so the textual markers are removed.
    text = re.sub(r"<image\s+\d+>", "", "\n".join(lines)).strip()

    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: List[str] = []
    for slot in range(1, 8):
        image = doc.get(f"image_{slot}")
        if image is None:
            continue
        path = image_dir / f"{stem}_img{slot}.png"
        if not path.is_file():
            image.convert("RGB").save(path)
        image_paths.append(str(path))
    if not image_paths:
        raise AssertionError(f"MMMU item {stem} has no image")
    return text, image_paths


def render_image_chat(
    processor, question: str, num_images: int = 1, *, text_first: bool = False
) -> str:
    """Render a multimodal prompt the way every capture and canary renders it.

    ``text_first`` puts the question ahead of the image, which moves the image's
    soft-token block later in the sequence.  The chunked-prefill canary needs
    that: with the image first, its block starts a handful of tokens in, and the
    V2 scheduler's snap-down (which rounds to a KV-page boundary) has nowhere to
    land.
    """
    image_parts = [{"type": "image"}] * num_images
    text_part = [{"type": "text", "text": question}]
    content = text_part + image_parts if text_first else image_parts + text_part
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
        **CHAT_TEMPLATE_KWARGS,
    )


def mmmu_canary_items(
    dataset_dir: str,
    samples: Tuple[Tuple[str, int], ...],
    processor,
) -> List[Dict[str, Any]]:
    """Build the fixed MMMU canary items: rendered prompt, image files, answer.

    Shared by the H200 canary node and by the native-reference driver so both
    sides feed the model the same bytes.
    """
    from datasets import load_dataset

    image_dir = reference_cache_dir() / "mmmu_canary_images"
    items: List[Dict[str, Any]] = []
    for config_name, index in samples:
        split = load_dataset(dataset_dir, config_name, split="validation")
        if index >= len(split):
            raise AssertionError(f"{config_name} validation has {len(split)} rows")
        doc = split[index]
        question, images = mmmu_question_and_images(doc, image_dir, f"{config_name}_{index}")
        items.append(
            {
                "id": f"{config_name}:{index}",
                "prompt": render_image_chat(processor, question, num_images=len(images)),
                "images": images,
                "answer": str(doc["answer"]).strip(),
            }
        )
    return items


CAPTURE_PRODUCERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "text_replay": capture_text_replay_reference,
    "image_replay": capture_image_replay_reference,
}

# Producers that consume a caller-supplied request payload (``--in``).
REQUEST_PRODUCERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "greedy_completions": native_greedy_completions,
}


def ensure_native_completions(request: Dict[str, Any], *, tag: str) -> Dict[str, Any]:
    """Memoized :func:`native_greedy_completions`, produced in a subprocess."""
    import subprocess
    import sys

    provenance = transformers_provenance()
    key = capture_key(
        "greedy_completions",
        CAPTURE_VERSION,
        tag,
        gemma4_26b_checkpoint(),
        provenance["transformers_commit"],
        request,
    )
    out = reference_cache_dir() / f"greedy_completions-{tag}-{key}.pt"
    if not out.is_file():
        payload_path = reference_cache_dir() / f"request-{tag}-{key}.pt"
        torch.save(request, payload_path)
        print(
            f"[gemma4-h200] producing {tag} native completions "
            f"({len(request['items'])} items) in a subprocess",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--capture",
                "greedy_completions",
                "--in",
                str(payload_path),
                "--out",
                str(out),
            ],
            check=True,
        )
    payload = torch.load(out, map_location="cpu", weights_only=False)
    recorded = payload.get("transformers_provenance")
    if recorded != provenance:
        raise AssertionError(
            f"completions {out} were produced against {recorded}, but this "
            f"process resolved {provenance}"
        )
    return payload


def ensure_capture(kind: str) -> Dict[str, Any]:
    """Return a memoized reference capture, producing it in a *subprocess*.

    The native 26B model needs ~50 GiB of device memory.  Producing the
    capture out-of-process is what guarantees that memory is back with the
    driver before the TensorRT-LLM engine is built in the test process --
    ``del model`` plus ``empty_cache()`` returns it to the caching allocator,
    but not necessarily to the device.
    """
    import subprocess
    import sys

    if kind not in CAPTURE_PRODUCERS:
        raise ValueError(f"unknown capture {kind!r}; known: {sorted(CAPTURE_PRODUCERS)}")
    provenance = transformers_provenance()
    # The reference commit is part of the cache key, so a capture taken against
    # a different Transformers source can never be silently reused, and the
    # recorded provenance is re-checked on load.
    key = capture_key(
        kind,
        CAPTURE_VERSION,
        gemma4_26b_checkpoint(),
        provenance["transformers_commit"],
        TEXT_PROMPTS,
        IMAGE_PROMPTS,
        GREEDY_MAX_NEW_TOKENS,
        CHAT_TEMPLATE_KWARGS,
    )
    path = reference_cache_dir() / f"{kind}-{key}.pt"
    if not path.is_file():
        print(f"[gemma4-h200] producing {kind} reference capture in a subprocess", flush=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--capture", kind, "--out", str(path)],
            check=True,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != CAPTURE_VERSION:
        raise AssertionError(
            f"stale capture {path} (version {payload.get('version')} != {CAPTURE_VERSION})"
        )
    recorded = payload.get("transformers_provenance")
    if recorded != provenance:
        raise AssertionError(
            f"capture {path} was produced against {recorded}, but this process "
            f"resolved {provenance}"
        )
    return payload


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        required=True,
        choices=sorted(set(CAPTURE_PRODUCERS) | set(REQUEST_PRODUCERS)),
    )
    parser.add_argument("--in", dest="request", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Assert provenance in the producing process too: the capture subprocess is
    # where the reference model is actually loaded, so this is the point where
    # "which Transformers produced these tensors" is decided.
    provenance = transformers_provenance()
    if args.capture in REQUEST_PRODUCERS:
        if not args.request:
            parser.error(f"--capture {args.capture} requires --in")
        request = torch.load(args.request, map_location="cpu", weights_only=False)
        payload = REQUEST_PRODUCERS[args.capture](request)
    else:
        payload = CAPTURE_PRODUCERS[args.capture]()
    payload["transformers_provenance"] = provenance
    out = Path(args.out)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"[gemma4-h200] wrote {args.capture} capture to {out}", flush=True)


if __name__ == "__main__":
    _main()


__all__ = [
    "ActivationRecorder",
    "CAPTURE_PRODUCERS",
    "CAPTURE_VERSION",
    "CHAT_TEMPLATE_KWARGS",
    "GREEDY_MAX_NEW_TOKENS",
    "IMAGE_PROMPTS",
    "TEXT_PROMPTS",
    "TensorMetrics",
    "assert_cosine",
    "assert_golden_matches_source",
    "cached_capture",
    "capture_image_replay_reference",
    "capture_key",
    "capture_text_replay_reference",
    "compare_tensors",
    "contiguous_image_runs",
    "ensure_capture",
    "ensure_native_completions",
    "gemma4_26b_checkpoint",
    "golden_apply_rope",
    "golden_attention",
    "golden_bidirectional_image_mask",
    "golden_causal_mask",
    "golden_experts",
    "golden_logit_softcap",
    "golden_rms_norm",
    "golden_rope_cos_sin",
    "golden_rotate_half",
    "golden_router",
    "h200_environment_report",
    "native_gemma4",
    "mmlu_five_shot_prompt",
    "mmmu_canary_items",
    "mmmu_question_and_images",
    "native_greedy_completions",
    "REQUEST_PRODUCERS",
    "reference_cache_dir",
    "render_image_chat",
    "render_image_prompt",
    "render_text_prompt",
    "representative_text_layers",
    "require_single_h200",
    "source_allow_mask",
    "text_layer_module",
    "vision_layer_module",
    "write_evidence",
]
