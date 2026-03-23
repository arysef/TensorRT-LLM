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

import importlib
import sys

from click.testing import CliRunner
import pytest

import trtllm_cli.main as main_cli
from trtllm_cli.main import cli as root_cli


def _purge_modules(*prefixes: str) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.")
               for prefix in prefixes):
            sys.modules.pop(name, None)


def test_import_tensorrt_llm_is_lazy(capsys):
    _purge_modules("tensorrt_llm", "torch", "triton_kernels")

    module = importlib.import_module("tensorrt_llm")

    captured = capsys.readouterr()
    assert module.__version__
    assert captured.out == ""
    assert "torch" not in sys.modules
    assert "tensorrt_llm._common" not in sys.modules
    assert not any(name == "triton_kernels"
                   or name.startswith("triton_kernels.")
                   for name in sys.modules)


def test_root_help_lists_commands_without_importing_tensorrt_llm():
    runner = CliRunner()
    _purge_modules("tensorrt_llm", "torch")

    result = runner.invoke(root_cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "serve" in result.output
    assert "bench" in result.output
    assert "eval" in result.output
    assert "build" in result.output
    assert "prune" in result.output
    assert "refit" in result.output
    assert "tensorrt_llm" not in sys.modules
    assert "torch" not in sys.modules


def test_root_serve_help_is_lazy():
    runner = CliRunner()
    _purge_modules("tensorrt_llm", "torch", "tensorrt", "triton_kernels")

    result = runner.invoke(root_cli, ["serve", "--help"])

    assert result.exit_code == 0, result.output
    assert "disaggregated" in result.output
    assert "mm_embedding_serve" in result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "torch" not in sys.modules
    assert "tensorrt_llm._common" not in sys.modules


def test_root_serve_default_command_help_is_lazy():
    runner = CliRunner()
    _purge_modules("tensorrt_llm", "torch", "tensorrt", "triton_kernels")

    result = runner.invoke(root_cli, ["serve", "dummy-model", "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: trtllm serve [OPTIONS] MODEL" in result.output
    assert "--backend [pytorch|tensorrt|_autodeploy]" in result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "torch" not in sys.modules
    assert "tensorrt_llm._common" not in sys.modules


@pytest.mark.parametrize(
    ("module_path", "argv", "prog_name", "expected_snippet"),
    [
        ("tensorrt_llm.commands.serve", ["dummy-model", "--help"],
         "trtllm-serve", "Usage: trtllm-serve [OPTIONS] MODEL"),
        ("tensorrt_llm.commands.bench", ["--help"], "trtllm-bench",
         "throughput"),
        ("tensorrt_llm.commands.eval", ["--help"], "trtllm-eval", "mmlu"),
    ],
)
def test_standalone_command_help_is_lazy(module_path, argv, prog_name,
                                         expected_snippet):
    runner = CliRunner()
    _purge_modules("tensorrt_llm", "torch", "tensorrt", "triton_kernels")

    command_mod = importlib.import_module(module_path)
    result = runner.invoke(command_mod.main, argv, prog_name=prog_name)

    assert result.exit_code == 0, result.output
    assert expected_snippet in result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "torch" not in sys.modules
    assert "tensorrt_llm._common" not in sys.modules


@pytest.mark.parametrize(
    ("args", "expected_snippets"),
    [
        (["bench", "--help"], ("throughput", "prepare-dataset")),
        (["eval", "--help"], ("mmlu", "longbench_v2")),
    ],
)
def test_root_command_help_is_lazy(args, expected_snippets):
    runner = CliRunner()
    _purge_modules("tensorrt_llm", "torch", "tensorrt", "triton_kernels")

    result = runner.invoke(root_cli, args)

    assert result.exit_code == 0, result.output
    for snippet in expected_snippets:
        assert snippet in result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "torch" not in sys.modules
    assert "tensorrt_llm._common" not in sys.modules


@pytest.mark.parametrize("command_name", ["serve", "bench", "eval"])
def test_root_click_proxies_delegate_with_expected_prog_name(monkeypatch,
                                                             command_name):
    runner = CliRunner()
    calls = []
    monkeypatch.setattr(main_cli, "_run_click_command",
                        lambda module_path, *, prog_name, args: calls.append(
                            (module_path, prog_name, args)))

    result = runner.invoke(root_cli, [command_name, "arg1", "--flag"])

    assert result.exit_code == 0, result.output
    assert calls == [(f"tensorrt_llm.commands.{command_name}",
                      f"trtllm {command_name}", ("arg1", "--flag"))]


@pytest.mark.parametrize("command_name", ["build", "prune", "refit"])
def test_root_argparse_proxies_delegate_with_expected_prog_name(
        monkeypatch, command_name):
    runner = CliRunner()
    calls = []
    monkeypatch.setattr(main_cli, "_run_argparse_command",
                        lambda module_path, *, prog_name, args: calls.append(
                            (module_path, prog_name, args)))

    result = runner.invoke(root_cli, [command_name, "--help"])

    assert result.exit_code == 0, result.output
    assert calls == [(f"tensorrt_llm.commands.{command_name}",
                      f"trtllm {command_name}", ("--help", ))]
