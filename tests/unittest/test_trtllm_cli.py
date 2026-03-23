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

import ast
import importlib
import sys
from pathlib import Path

from click.testing import CliRunner
import pytest

import trtllm_cli.main as main_cli
from trtllm_cli._serve_metadata import (LOG_LEVELS, REASONING_PARSER_CHOICES,
                                        TOOL_PARSER_CHOICES)
from trtllm_cli.main import cli as root_cli


def _purge_modules(*prefixes: str) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.")
               for prefix in prefixes):
            sys.modules.pop(name, None)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_source(relative_path: str) -> ast.Module:
    return ast.parse((_repo_root() / relative_path).read_text(encoding="utf-8"))


def _extract_assignment_node(relative_path: str, name: str):
    tree = _parse_source(relative_path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                           ast.Name):
            if node.target.id == name:
                return node.value
    raise AssertionError(f"Could not find assignment for {name} in {relative_path}")


def _extract_class_attribute_node(relative_path: str, class_name: str,
                                  name: str):
    tree = _parse_source(relative_path)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for class_node in node.body:
            if isinstance(class_node, ast.Assign):
                for target in class_node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return class_node.value
            if isinstance(class_node, ast.AnnAssign) and isinstance(
                    class_node.target, ast.Name):
                if class_node.target.id == name:
                    return class_node.value
    raise AssertionError(
        f"Could not find class attribute {class_name}.{name} in {relative_path}"
    )


def _dict_string_keys(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, ast.Dict):
        raise AssertionError(f"Expected ast.Dict, got {type(node).__name__}")
    keys: list[str] = []
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("Expected all dict keys to be constant strings")
        keys.append(key.value)
    return tuple(keys)


def _extract_reasoning_parser_keys(relative_path: str) -> tuple[str, ...]:
    tree = _parse_source(relative_path)
    keys: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Name):
                continue
            if decorator.func.id != "register_reasoning_parser":
                continue
            for arg in decorator.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    keys.append(arg.value)
    return tuple(keys)


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
    assert "[TensorRT-LLM]" not in result.output
    assert "disaggregated" in result.output
    assert "mm_embedding_serve" in result.output
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


def test_standalone_serve_default_command_help_is_lazy():
    runner = CliRunner()
    _purge_modules("tensorrt_llm", "torch", "tensorrt", "triton_kernels")

    serve_mod = importlib.import_module("tensorrt_llm.commands.serve")
    result = runner.invoke(serve_mod.main,
                           ["dummy-model", "--help"],
                           prog_name="trtllm-serve")

    assert result.exit_code == 0, result.output
    assert "Usage: trtllm-serve [OPTIONS] MODEL" in result.output
    assert "--backend [pytorch|tensorrt|_autodeploy]" in result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "torch" not in sys.modules
    assert "tensorrt_llm._common" not in sys.modules


def test_root_bench_help_is_lazy():
    runner = CliRunner()
    _purge_modules("tensorrt_llm", "torch", "tensorrt", "triton_kernels")

    result = runner.invoke(root_cli, ["bench", "--help"])

    assert result.exit_code == 0, result.output
    assert "throughput" in result.output
    assert "prepare-dataset" in result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "torch" not in sys.modules


def test_root_eval_help_is_lazy():
    runner = CliRunner()
    _purge_modules("tensorrt_llm", "torch", "tensorrt", "triton_kernels")

    result = runner.invoke(root_cli, ["eval", "--help"])

    assert result.exit_code == 0, result.output
    assert "mmlu" in result.output
    assert "longbench_v2" in result.output
    assert "[TensorRT-LLM]" not in result.output
    assert "torch" not in sys.modules


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


def test_serve_metadata_reasoning_parser_choices_match_runtime_source():
    assert REASONING_PARSER_CHOICES == _extract_reasoning_parser_keys(
        "tensorrt_llm/llmapi/reasoning_parser.py")


def test_serve_metadata_tool_parser_choices_match_runtime_source():
    tool_parsers = _extract_class_attribute_node(
        "tensorrt_llm/serve/tool_parser/tool_parser_factory.py",
        "ToolParserFactory", "parsers")

    assert TOOL_PARSER_CHOICES == _dict_string_keys(tool_parsers)


def test_serve_metadata_log_levels_match_runtime_source():
    severity_map = _extract_assignment_node("tensorrt_llm/logger.py",
                                            "severity_map")

    assert LOG_LEVELS == _dict_string_keys(severity_map)
