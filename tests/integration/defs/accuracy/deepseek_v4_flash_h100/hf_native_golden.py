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
"""Produce the canonical native-generate fixture for DeepSeek-V4-Flash.

Why this file exists
--------------------
The checkpoint's official ``inference/generate.py`` is a *hand-written* prefill
plus greedy-decode loop. Its position bookkeeping, window/compression state, KV
threading and argmax tie-breaking are all re-implementations, so an evidence
ladder rooted on that loop is rooted on an unverified assumption. The
golden-generate cross-check anchors it: generate the greedy token ids once with
the *source-native* ``AutoModelForCausalLM.generate(do_sample=False)`` in an
isolated environment, commit them as a fixture, and assert the hand-written
loop reproduces them token-for-token.

``transformers`` 5.15.1 ships a complete ``DeepseekV4ForCausalLM`` *and* a
``deepseek_v4`` entry in its weight-conversion registry, so it consumes this
checkpoint's official naming (``embed.weight``,
``layers.N.attn.wq_a.weight``, ``attn_sink``, per-expert ``w1``/``w3``) with no
help from us --- and its FP8 quantizer understands the checkpoint's own
``quantization_config``: FP8 E4M3 dense weights with 128x128 UE8M0 block
scales, and routed experts packed MXFP4 in an I8 container.

That matters for more than convenience. Nothing here converts, dequantizes or
renames anything, so the fixture is produced from the *unmodified* checkpoint
bytes at the checkpoint's own numerical precision, which is also the precision
the official loop runs at. A token disagreement therefore points at generation
semantics rather than at a precision gap this file introduced.

Two things are deliberately *not* done, per the reference-test policy: the
container's pinned ``transformers`` (5.5.4) is never touched --- this runs in
a throwaway venv built by ``build_reference_env.sh`` --- and no model code is
patched, shimmed or monkeypatched.

Usage (inside the golden venv)::

    python3 hf_native_golden.py --checkpoint /models/DeepSeek-V4-Flash --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# The one configuration this fixture is valid for. Anything else --- sampling,
# beams, a different horizon --- is not the canonical greedy reference.
GENERATE_KWARGS: dict[str, Any] = {"do_sample": False, "num_beams": 1, "use_cache": True}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _quantized_module_report(model: torch.nn.Module) -> dict[str, Any]:
    """Record which classes the loader actually built.

    The point of running the raw checkpoint is that it stays FP8/MXFP4. If the
    quantizer had silently dequantized to BF16 --- which it does on an
    unsupported GPU --- the fixture would be a different numerical experiment,
    so the artifact records the module classes rather than assuming them.
    """
    counts: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    for module in model.modules():
        name = type(module).__name__
        if name.startswith(("FP8", "Mxfp4", "FP4")):
            counts[name] = counts.get(name, 0) + 1
    for _, param in model.named_parameters():
        key = str(param.dtype)
        dtypes[key] = dtypes.get(key, 0) + 1
    return {"quantized_module_classes": counts, "parameter_dtypes": dtypes}


def generate_fixture(args: argparse.Namespace) -> dict[str, Any]:
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    with open(args.prompts) as f:
        prompts_manifest = json.load(f)
    prompts = {p["id"]: p for p in prompts_manifest["prompts"]}
    wanted = args.prompt_ids or list(prompts)

    config = AutoConfig.from_pretrained(args.checkpoint)
    n_gpus = torch.cuda.device_count()

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map=args.device_map,
        max_memory={i: args.max_memory_per_gpu for i in range(n_gpus)},
    )
    model.eval()
    load_s = time.time() - t0
    modules = _quantized_module_report(model)
    print(f"  loaded in {load_s:.1f}s: {modules['quantized_module_classes']}", flush=True)

    first_device = next(model.parameters()).device
    results: dict[str, Any] = {}
    for pid in wanted:
        spec = prompts[pid]
        ids = torch.tensor([spec["token_ids"]], dtype=torch.long, device=first_device)
        t1 = time.time()
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                eos_token_id=config.eos_token_id,
                pad_token_id=config.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                **GENERATE_KWARGS,
            )
        generated = out.sequences[0, ids.shape[1] :].tolist()
        # Per-step top1-top2 margin. Recorded because it is what makes a future
        # disagreement diagnosable: a divergence at a step whose margin is ~0 is
        # arithmetic noise between two implementations, while a divergence at a
        # confident step is a real semantic difference.
        top2 = [s[0].float().topk(2) for s in out.scores][: len(generated)]
        margins = [round(float(t.values[0] - t.values[1]), 6) for t in top2]
        candidates = [[int(i) for i in t.indices] for t in top2]
        results[pid] = {
            "tokens": generated,
            "num_prompt_tokens": len(spec["token_ids"]),
            "num_generated": len(generated),
            "category": spec["category"],
            "thinking_mode": spec["thinking_mode"],
            "rendered_sha256": spec["rendered_sha256"],
            "top1_top2_margin": margins,
            "top2_candidates": candidates,
            "min_top1_top2_margin": round(min(margins), 6) if margins else None,
            "elapsed_s": round(time.time() - t1, 2),
        }
        print(
            f"  {pid:24s} {len(spec['token_ids']):5d} tok -> {len(generated):3d} generated "
            f"in {results[pid]['elapsed_s']}s, min margin "
            f"{results[pid]['min_top1_top2_margin']}",
            flush=True,
        )

    fixture = {
        "schema_version": 1,
        "evidence_label": "native_generate_golden",
        "reference_tier": "real_source",
        "validation_tier": "integration",
        "checkpoint_revision": prompts_manifest["checkpoint_revision"],
        "checkpoint": args.checkpoint,
        "decoding": {
            "rule": "deterministic greedy",
            "sampling": False,
            "temperature": 0,
            "top_k": 1,
            "max_new_tokens": args.max_new_tokens,
            "eos_token_id": config.eos_token_id,
            **GENERATE_KWARGS,
        },
        "provenance": {
            "generator": "AutoModelForCausalLM.generate(do_sample=False)",
            "transformers_version": transformers.__version__,
            "transformers_path": os.path.dirname(transformers.__file__),
            "torch_version": torch.__version__,
            "attn_implementation": model.config._attn_implementation,
            "devices": f"{n_gpus} x {torch.cuda.get_device_name(0)}",
            "device_map": args.device_map,
            "checkpoint_bytes": "unmodified; loaded through transformers' own "
            "deepseek_v4 weight-conversion mapping and FP8/MXFP4 quantizer",
            "conversion_code": "none -- no remap, dequantization or shim applied",
            "conversion_code_sha256": _sha256(os.path.abspath(__file__)),
            "generator_code": "tests/integration/defs/accuracy/deepseek_v4_flash_h100/"
            "hf_native_golden.py",
            "load_seconds": round(load_s, 1),
            "prompts_manifest_sha256": _sha256(args.prompts),
            "isolated_env": sys.prefix,
            "generated_utc": args.stamp,
            **modules,
        },
        "prompts": results,
        "required_prompt_ids": sorted(set(results) - set(args.non_gating_prompt_ids or [])),
        "non_gating_prompt_ids": sorted(set(args.non_gating_prompt_ids or []) & set(results)),
        "non_gating_reason": args.non_gating_reason,
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out} ({len(results)} prompts)", flush=True)
    return fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--prompts", default=os.path.join(HERE, "manifests", "prompts.json"))
    parser.add_argument(
        "--out", default=os.path.join(HERE, "manifests", "native_generate_golden.json")
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-ids", nargs="*", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-per-gpu", default="70GiB")
    parser.add_argument(
        "--non-gating-prompt-ids",
        nargs="*",
        default=None,
        help="prompts to record but not gate on; still compared and reported truthfully",
    )
    parser.add_argument(
        "--non-gating-reason",
        default=None,
        help="why those prompts do not gate; the comparator rejects a fixture that omits it",
    )
    parser.add_argument("--stamp", default=None, help="UTC timestamp recorded in the fixture")
    args = parser.parse_args(argv)
    if args.stamp is None:
        args.stamp = subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=True
        ).stdout.strip()
    generate_fixture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
