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
"""The official source's greedy tokens and logits, captured for later parity.

Goal 3.4 compares TensorRT-LLM against the checkpoint's own
``inference/model.py``. Both sides cannot live in one process: the official
model needs ``tilelang``, which only exists in the isolated reference
interpreter, and that interpreter breaks the very runtime under test (its
``tvm_ffi`` makes flashinfer's CuTe RMSNorm raise, which killed all eight
executor workers when the LLM API suite was first run there). Both sides also
want the same eight GPUs.

So the reference is produced by this module in its own process, written to
disk, and read back by the suite that drives TensorRT-LLM. The artifact is a
JSON file plus an ``.npz`` sidecar holding the logit rows, which are far too
large for JSON: 32 steps x 129280 vocabulary entries x 4 bytes is ~16 MB per
prompt.

Two properties make the capture trustworthy rather than merely convenient:

* the loop is :meth:`OfficialSource.greedy`, the same one the checked-in
  native-``generate()`` fixture anchors --- not a second implementation written
  for this purpose. It is asked to continue past EOS so that every prompt
  reaches the 32 compared steps the parity gate requires, and the tokens it
  produced *before* EOS are compared against the fixture in the same run;
* the artifact records the prompt manifest hash, the checkpoint revision and
  the token budget it was captured at, so the consumer can refuse a reference
  that describes a different measurement instead of silently comparing against
  it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

#: Sidecar suffix for the logit arrays that accompany a reference artifact.
LOGITS_SUFFIX = ".logits.npz"


def logits_path(artifact_path: str) -> str:
    """Where the sidecar for ``artifact_path`` lives."""
    return os.path.splitext(os.path.abspath(artifact_path))[0] + LOGITS_SUFFIX


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(
    src: Any,
    prompts: list[dict[str, Any]],
    parity_tokens: int,
    ranks: Any,
) -> dict[str, Any]:
    """Run the official loop over ``prompts`` and keep tokens plus logits.

    Returns the per-prompt record and, on rank 0 only, the stacked logit rows
    keyed by prompt id. Ranks other than 0 still run: the model is
    model-parallel, so every rank has to take part in the same forwards, and
    their agreement on the resulting tokens is checked by the caller.
    """
    import torch

    captured: dict[str, Any] = {}
    logits: dict[str, Any] = {}
    for spec in prompts:
        started = time.time()
        run = src.greedy(
            spec["token_ids"],
            parity_tokens,
            stop_at_eos=False,
            capture_logits=True,
        )
        rows = run.pop("logits")
        if len(run["tokens"]) != parity_tokens:
            raise RuntimeError(
                f"{spec['id']}: asked for {parity_tokens} tokens past EOS and got "
                f"{len(run['tokens'])}; the parity gate compares a fixed number of steps"
            )
        run["logits_finite"] = bool(torch.isfinite(rows).all())
        run["logits_shape"] = list(rows.shape)
        run["category"] = spec["category"]
        run["thinking_mode"] = spec["thinking_mode"]
        run["prompt_tokens"] = spec["num_tokens"]
        run["rendered_sha256"] = spec["rendered_sha256"]
        run["elapsed_s"] = round(time.time() - started, 2)
        captured[spec["id"]] = run
        if ranks.is_main:
            logits[spec["id"]] = rows.float().numpy()
        ranks.log(
            f"  {spec['id']:24s} {spec['num_tokens']:5d} tok -> {len(run['tokens'])} steps, "
            f"eos_at={run['eos_at']}, finite={run['logits_finite']}, {run['elapsed_s']}s"
        )
    return {"prompts": captured, "logits": logits}


def head_precision_probe(
    src: Any, prompts: list[dict[str, Any]], ranks: Any, compare: Any
) -> dict[str, Any]:
    """What the LM head's GEMM precision alone costs, nothing else compared.

    The source contract, stated precisely: ``ParallelHead`` keeps ``lm_head``
    in FP32 -- "stored in bf16 [in the checkpoint], while the parameter here is
    stored in fp32 for easier computation of logits" -- while the final
    ``RMSNorm.forward`` ends in ``.to(dtype)`` and so hands back **BF16**, the
    dtype ``hc_head`` returned. The widening happens one line later, inside
    ``ParallelHead.get_logits``: ``F.linear(x[:, -1].float(), self.weight)``.
    So the head is an FP32 GEMM over an FP32-widened BF16 activation, not over
    an activation that was ever carried in FP32. A production runtime that
    instead runs the GEMM itself in BF16 computes the same quantity at a
    different precision.

    That is what this measures, on the source's own activations: same hidden
    state, same weights, same all-gather, only the GEMM precision changed. The
    activation is already BF16 on both sides, so ``x.bfloat16()`` below is a
    no-op cast that documents the arm rather than degrading it. No
    TensorRT-LLM is involved, so whatever it reports is a property of the
    arithmetic rather than of this bring-up --- which is what makes it usable
    as a reference point when judging a logit tolerance.
    """
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    head = src.model.head
    seen: dict[str, Any] = {}
    original = head.get_logits

    def spy(x: Any) -> Any:
        seen["x"] = x.detach()
        return original(x)

    def gathered(rows: Any) -> Any:
        if ranks.world > 1 and dist.is_initialized():
            parts = [torch.empty_like(rows) for _ in range(ranks.world)]
            dist.all_gather(parts, rows)
            return torch.cat(parts, dim=-1)
        return rows

    head.get_logits = spy
    probed: dict[str, Any] = {}
    try:
        # The captured activations are inference tensors and the head weight is
        # still a Parameter that requires grad, so the recomputation has to stay
        # inside inference mode --- exactly as the module goldens do.
        with torch.inference_mode():
            for spec in prompts:
                src.reset_cache()
                toks = torch.tensor([spec["token_ids"]], dtype=torch.long, device="cuda")
                src.model.forward(toks, 0)
                x = seen["x"][:, -1]
                weight = head.weight
                fp32 = gathered(F.linear(x.float(), weight))
                bf16 = gathered(F.linear(x.bfloat16(), weight.bfloat16()).float())
                metrics = compare(bf16, fp32)
                metrics["argmax_match"] = int(bf16.argmax()) == int(fp32.argmax())
                metrics["head_input_dtype"] = str(x.dtype)
                metrics["head_weight_dtype"] = str(weight.dtype)
                probed[spec["id"]] = {
                    k: (round(v, 8) if isinstance(v, float) else v) for k, v in metrics.items()
                }
                ranks.log(
                    f"  head dtype {spec['id']:24s} rel_max_abs={probed[spec['id']]['rel_max_abs']:.4f} "
                    f"cosine={probed[spec['id']]['cosine']:.6f} "
                    f"argmax_match={probed[spec['id']]['argmax_match']}"
                )
    finally:
        head.get_logits = original
    return {
        "description": (
            "official FP32 head/norm versus the same activation and weights in BF16; "
            "no TensorRT-LLM in the comparison"
        ),
        "per_prompt": probed,
    }


def write(artifact_path: str, payload: dict[str, Any], logits: dict[str, Any]) -> dict[str, Any]:
    """Write the ``.npz`` sidecar and return the provenance to embed in the JSON."""
    import numpy as np

    path = logits_path(artifact_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{f"{pid}__logits": rows for pid, rows in logits.items()})
    payload["logits_sidecar"] = {
        "path": path,
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "prompts": sorted(logits),
    }
    return payload


def load(artifact_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a reference artifact and its logit sidecar.

    The sidecar is verified against the hash the capture recorded, because a
    truncated or half-written ``.npz`` would otherwise be compared against and
    reported as a parity result.
    """
    import numpy as np

    with open(artifact_path) as handle:
        artifact = json.load(handle)
    sidecar = artifact.get("logits_sidecar") or {}
    path = sidecar.get("path") or logits_path(artifact_path)
    if not os.path.exists(path):
        raise RuntimeError(f"reference artifact {artifact_path} has no logit sidecar at {path}")
    actual = sha256_file(path)
    if sidecar.get("sha256") and actual != sidecar["sha256"]:
        raise RuntimeError(
            f"logit sidecar {path} hashes to {actual}, but the reference recorded "
            f"{sidecar['sha256']}; it is not the file that capture produced"
        )
    with np.load(path) as data:
        logits = {key[: -len("__logits")]: data[key] for key in data.files}
    return artifact, logits


def usable(
    artifact: dict[str, Any],
    *,
    checkpoint_revision: str,
    prompts_sha256: str,
    parity_tokens: int,
    prompt_ids: list[str],
) -> list[str]:
    """Why this reference may not judge the run about to happen, if it may not.

    A reference is only a reference for the measurement it was taken for. Each
    of these has produced, or would produce, a comparison against the wrong
    thing: a different checkpoint, a different prompt set or rendering, or
    fewer steps than the gate compares.
    """
    problems: list[str] = []
    if artifact.get("checkpoint_revision") != checkpoint_revision:
        problems.append(
            f"reference is for checkpoint {artifact.get('checkpoint_revision')!r}, "
            f"this run is {checkpoint_revision!r}"
        )
    recorded = (artifact.get("manifest_provenance") or {}).get("sha256", {})
    if recorded.get("prompts.json") != prompts_sha256:
        problems.append("reference was captured against a different prompts manifest")
    if artifact.get("parity_tokens", 0) < parity_tokens:
        problems.append(
            f"reference has {artifact.get('parity_tokens')} steps per prompt, "
            f"the gate compares {parity_tokens}"
        )
    if not artifact.get("passed"):
        problems.append("reference artifact did not pass its own checks")
    missing = sorted(set(prompt_ids) - set(artifact.get("prompts") or {}))
    if missing:
        problems.append(f"reference is missing prompts {missing}")
    return problems
