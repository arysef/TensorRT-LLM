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

import sys

from click.testing import CliRunner

from trtllm_cli import bench, eval as eval_cli, serve
from trtllm_cli.main import cli as root_cli


def _new_tensorrt_llm_modules(baseline: set[str]) -> set[str]:
    return {
        name
        for name in set(sys.modules) - baseline
        if name == "tensorrt_llm" or name.startswith("tensorrt_llm.")
    }


def test_root_help_lists_available_commands_without_importing_runtime():
    runner = CliRunner()
    baseline = set(sys.modules)

    result = runner.invoke(root_cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "serve" in result.output
    assert "bench" in result.output
    assert "eval" in result.output
    assert "build" in result.output
    assert "prune" in result.output
    assert "refit" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_root_serve_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(root_cli, ["serve", "--help"])

    assert result.exit_code == 0, result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "disaggregated" in result.output
    assert "mm_embedding_serve" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_bench_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(bench, "_delegate_to_legacy_bench",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(bench.cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "throughput" in result.output
    assert "prepare-dataset" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_eval_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(eval_cli, "_delegate_to_legacy_eval",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(eval_cli.cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "mmlu" in result.output
    assert "longbench_v2" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_root_serve_default_command_delegates(monkeypatch):
    runner = CliRunner()
    calls = []
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: calls.append("serve"))

    result = runner.invoke(root_cli,
                           ["serve", "dummy-model", "--port", "8000"])

    assert result.exit_code == 0, result.output
    assert calls == ["serve"]


def test_root_bench_command_delegates(monkeypatch):
    runner = CliRunner()
    calls = []
    monkeypatch.setattr(bench, "_delegate_to_legacy_bench",
                        lambda: calls.append("bench"))

    result = runner.invoke(root_cli,
                           ["bench", "--model", "dummy-model", "throughput"])

    assert result.exit_code == 0, result.output
    assert calls == ["bench"]
