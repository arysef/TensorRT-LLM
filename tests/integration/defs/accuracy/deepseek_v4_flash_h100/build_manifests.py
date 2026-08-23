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
"""Build (and re-verify) the pre-registered DeepSeek-V4-Flash prompt manifest.

The manifest pins the *exact token ids* of every fixed prompt used by the
bring-up gates, so that "the same five prompts" means the same integers on
every rerun, on both the source side and the TensorRT-LLM side.

Rendering deliberately uses the checkpoint's own ``encoding/encoding_dsv4.py``
and a tokenizer loaded straight from ``tokenizer.json``. It does **not** import
any TensorRT-LLM helper, so the manifest cannot inherit a bug from the code
under test. ``tests/unittest/llmapi/test_deepseek_v4_tokenizer.py`` separately
covers that TensorRT-LLM's ``DeepseekV4Tokenizer`` reproduces these ids.

Two prompts are length-targeted:

* ``cache_boundary_257`` -- exactly 257 prefill tokens, which crosses the
  128-token SWA ring twice, completes ratio-4 compression groups, and lands
  just past a 128-token KV block boundary;
* ``long_prefill_2304`` -- 2304 tokens, comfortably past the >=2048 chunked
  prefill requirement while staying inside ``max_seq_len=4096`` with room for
  decode.

Both are padded to their exact target with single-token filler words, chosen
and verified at build time, so the target is hit precisely rather than
approximately.

Usage::

    python3 build_manifests.py --checkpoint /models/DeepSeek-V4-Flash
    python3 build_manifests.py --checkpoint /models/DeepSeek-V4-Flash --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_DIR = os.path.join(_HERE, "manifests")
PROMPTS_PATH = os.path.join(MANIFEST_DIR, "prompts.json")
TOLERANCES_PATH = os.path.join(MANIFEST_DIR, "tolerances.json")
CHECKSUM_PATH = os.path.join(MANIFEST_DIR, "MANIFEST.sha256")

# Every manifest a gate reads, in registration order. The superseded tolerance
# file and the native-generate fixture are registered alongside the two active
# manifests: the first so "no other limit moved" is a one-command diff rather
# than a claim, the second so a prompt cannot move between the gating and
# non-gating sets without a hash moving.
REGISTERED_PATHS = (
    PROMPTS_PATH,
    TOLERANCES_PATH,
    os.path.join(MANIFEST_DIR, "tolerances.superseded.json"),
    os.path.join(MANIFEST_DIR, "native_generate_golden.json"),
    # Stage 3 Goal 3.5. It declares which regression failures count as this
    # container's missing CAP_SYS_PTRACE rather than as a DeepSeek-V4 defect,
    # so an unregistered edit could absorb a real regression into the expected
    # set. Hashed for the same reason the prompt classification is.
    os.path.join(MANIFEST_DIR, "regression_baseline.json"),
)

CHECKPOINT_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"

# Deterministic filler sentence used to grow the length-targeted prompts.
_FILLER_SENTENCE = (
    "The cache boundary test repeats this sentence so the prefill length is "
    "deterministic and reproducible across runs. "
)

# Candidate single-token filler words, verified at build time.
_FILLER_WORDS = [" and", " the", " of", " to", " in", " a"]


def _load_encoder(checkpoint: str):
    encoding_dir = os.path.join(checkpoint, "encoding")
    if encoding_dir not in sys.path:
        sys.path.insert(0, encoding_dir)
    from encoding_dsv4 import encode_messages  # noqa: PLC0415

    return encode_messages


def _load_tokenizer(checkpoint: str):
    # Load straight from tokenizer.json: the checkpoint's config.json declares
    # model_type "deepseek_v4", which the repo's pinned transformers does not
    # register, so AutoTokenizer's config probe would fail. The tokenizer file
    # itself is self-contained.
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    return PreTrainedTokenizerFast(tokenizer_file=os.path.join(checkpoint, "tokenizer.json"))


def _single_token_filler(tok) -> str:
    for word in _FILLER_WORDS:
        if len(tok.encode(word, add_special_tokens=False)) == 1:
            return word
    raise RuntimeError("No single-token filler word found; cannot hit an exact token target.")


def _pad_content_to_exact(
    tok, encode_messages, prefix: str, suffix: str, thinking_mode: str, target: int
) -> str:
    """Grow ``prefix`` until the rendered prompt is exactly ``target`` tokens.

    ``suffix`` is appended after the filler so the prompt still ends in a real
    question rather than trailing filler.
    """

    def n_tokens(content: str) -> int:
        rendered = encode_messages(
            [{"role": "user", "content": content}], thinking_mode=thinking_mode
        )
        return len(tok.encode(rendered))

    body = prefix
    if n_tokens(body + suffix) > target:
        raise RuntimeError(f"Base prompt already exceeds target {target} tokens.")

    # Coarse: whole sentences.
    while n_tokens(body + _FILLER_SENTENCE + suffix) <= target:
        body += _FILLER_SENTENCE

    # Fine: single-token filler words.
    filler = _single_token_filler(tok)
    while n_tokens(body + filler + suffix) <= target:
        body += filler

    content = body + suffix
    got = n_tokens(content)
    if got != target:
        raise RuntimeError(f"Could not hit exact token target: wanted {target}, got {got}.")
    return content


def _prompt_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "chat_arithmetic",
            "category": "plain_chat",
            "thinking_mode": "chat",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        {
            "id": "chat_geography",
            "category": "plain_chat",
            "thinking_mode": "chat",
            "messages": [
                {
                    "role": "user",
                    "content": "Name the capital of France and say one sentence about it.",
                }
            ],
        },
        {
            "id": "reasoning_word_problem",
            "category": "reasoning",
            "thinking_mode": "thinking",
            "messages": [
                {
                    "role": "user",
                    "content": "A train leaves at 09:15 and arrives at 13:40. It stops twice "
                    "for 12 minutes each. How long is the train actually moving? "
                    "Show your reasoning step by step.",
                }
            ],
        },
        {
            "id": "code_python_function",
            "category": "code",
            "thinking_mode": "chat",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function that returns the n-th Fibonacci "
                    "number iteratively. Include a short docstring.",
                }
            ],
        },
        {
            "id": "cache_boundary_257",
            "category": "cache_boundary",
            "thinking_mode": "chat",
            "target_tokens": 257,
            "prefix": "Read the following passage carefully. ",
            "suffix": "Now answer in one sentence: what does the passage repeat?",
        },
        {
            "id": "long_prefill_2304",
            "category": "long_prefill",
            "thinking_mode": "chat",
            "target_tokens": 2304,
            "prefix": "Read the following long passage carefully. ",
            "suffix": "Now summarise the passage in exactly one sentence.",
        },
    ]


def build(checkpoint: str) -> dict[str, Any]:
    encode_messages = _load_encoder(checkpoint)
    tok = _load_tokenizer(checkpoint)

    prompts: list[dict[str, Any]] = []
    for spec in _prompt_specs():
        spec = dict(spec)
        target = spec.pop("target_tokens", None)
        if target is not None:
            content = _pad_content_to_exact(
                tok,
                encode_messages,
                prefix=spec.pop("prefix"),
                suffix=spec.pop("suffix"),
                thinking_mode=spec["thinking_mode"],
                target=target,
            )
            spec["messages"] = [{"role": "user", "content": content}]

        rendered = encode_messages(spec["messages"], thinking_mode=spec["thinking_mode"])
        ids = tok.encode(rendered)
        if target is not None and len(ids) != target:
            raise RuntimeError(f"{spec['id']}: expected {target} tokens, got {len(ids)}")

        prompts.append(
            {
                "id": spec["id"],
                "category": spec["category"],
                "thinking_mode": spec["thinking_mode"],
                "messages": spec["messages"],
                "rendered": rendered,
                "num_tokens": len(ids),
                "token_ids": ids,
                "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            }
        )

    encoding_src = os.path.join(checkpoint, "encoding", "encoding_dsv4.py")
    tokenizer_src = os.path.join(checkpoint, "tokenizer.json")
    return {
        "schema_version": 1,
        "status": "pre-registered",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "decoding": {
            "temperature": 0,
            "top_k": 1,
            "sampling": False,
            "rule": "deterministic greedy",
        },
        "tokenizer": {
            "source": "tokenizer.json",
            "sha256": _sha256_file(tokenizer_src),
            "vocab_size": len(tok),
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": tok.eos_token_id,
        },
        "encoding": {
            "source": "encoding/encoding_dsv4.py",
            "sha256": _sha256_file(encoding_src),
        },
        "prompts": prompts,
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _serialise(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def write_checksums() -> str:
    lines = []
    for path in REGISTERED_PATHS:
        lines.append(f"{_sha256_file(path)}  {os.path.basename(path)}")
    body = "\n".join(lines) + "\n"
    with open(CHECKSUM_PATH, "w") as f:
        f.write(body)
    return body


def verify_checksums() -> None:
    """Raise if any registered manifest drifted from its registered hash."""
    with open(CHECKSUM_PATH) as f:
        registered = {name: digest for digest, name in (line.split() for line in f if line.strip())}
    for path in REGISTERED_PATHS:
        name = os.path.basename(path)
        actual = _sha256_file(path)
        if name not in registered:
            raise RuntimeError(f"{name} is not registered in MANIFEST.sha256")
        if registered[name] != actual:
            raise RuntimeError(
                f"{name} changed after registration: registered "
                f"{registered[name]}, actual {actual}. Re-registering a "
                "manifest mid-bring-up requires an explicit progress.yaml "
                "entry explaining why."
            )


def load_prompts() -> dict[str, Any]:
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def load_tolerances() -> dict[str, Any]:
    with open(TOLERANCES_PATH) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/models/DeepSeek-V4-Flash")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild in memory and fail if it differs from the checked-in "
        "manifest, instead of overwriting it.",
    )
    args = parser.parse_args(argv)

    manifest = build(args.checkpoint)
    payload = _serialise(manifest)

    if args.check:
        with open(PROMPTS_PATH, "rb") as f:
            existing = f.read()
        if existing != payload:
            print("prompts.json does not match a fresh rebuild.")
            return 1
        verify_checksums()
        print("Manifests match their registered content and hashes.")
    else:
        os.makedirs(MANIFEST_DIR, exist_ok=True)
        with open(PROMPTS_PATH, "wb") as f:
            f.write(payload)
        print(write_checksums())

    for p in manifest["prompts"]:
        print(
            f"  {p['id']:24s} {p['category']:16s} "
            f"{p['thinking_mode']:9s} {p['num_tokens']:5d} tokens"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
