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
import sys
from pathlib import Path

from click.testing import CliRunner
import pytest

import trtllm_cli.main as main_cli
from trtllm_cli import bench, eval as eval_cli, serve
from trtllm_cli._dispatch import invocation_state
from trtllm_cli._serve_metadata import (LOG_LEVELS, REASONING_PARSER_CHOICES,
                                        TOOL_PARSER_CHOICES)
from trtllm_cli.main import cli as root_cli


def _new_tensorrt_llm_modules(baseline: set[str]) -> set[str]:
    return {
        name
        for name in set(sys.modules) - baseline
        if name == "tensorrt_llm" or name.startswith("tensorrt_llm.")
    }


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
        f"Could not find class attribute {class_name}.{name} in {relative_path}")


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


def test_root_serve_default_command_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["serve", "dummy-model", "--help"],
                          prog_name="trtllm",
                          root_mode=True):
        result = runner.invoke(root_cli, ["serve", "dummy-model", "--help"])

    assert result.exit_code == 0, result.output
    assert "Running an OpenAI API compatible server" in result.output
    assert "--backend [pytorch|tensorrt|_autodeploy]" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_serve_default_command_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["dummy-model", "--help"],
                          prog_name="trtllm-serve",
                          root_mode=False):
        result = runner.invoke(serve.cli,
                               ["dummy-model", "--help"],
                               prog_name="trtllm-serve")

    assert result.exit_code == 0, result.output
    assert "Running an OpenAI API compatible server" in result.output
    assert "--backend [pytorch|tensorrt|_autodeploy]" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_root_explicit_serve_subcommand_help_collapses_usage(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["serve", "serve", "--help"],
                          prog_name="trtllm",
                          root_mode=True):
        result = runner.invoke(root_cli,
                               ["serve", "serve", "--help"],
                               prog_name="trtllm")

    assert result.exit_code == 0, result.output
    assert "Usage: trtllm serve [OPTIONS] MODEL" in result.output
    assert "Usage: trtllm serve serve" not in result.output
    assert "Additional arguments are forwarded" not in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_explicit_serve_subcommand_help_collapses_usage(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["serve", "--help"],
                          prog_name="trtllm-serve",
                          root_mode=False):
        result = runner.invoke(serve.cli,
                               ["serve", "--help"],
                               prog_name="trtllm-serve")

    assert result.exit_code == 0, result.output
    assert "Usage: trtllm-serve [OPTIONS] MODEL" in result.output
    assert "Usage: trtllm-serve serve" not in result.output
    assert "Additional arguments are forwarded" not in result.output
    assert "--reasoning_parser [deepseek-r1|qwen3|nano-v3]" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_serve_help_skips_option_validation(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state([
            "serve",
            "--backend",
            "not-a-backend",
            "--custom_module_dirs",
            "/does/not/exist",
            "--help",
    ],
                          prog_name="trtllm-serve",
                          root_mode=False):
        result = runner.invoke(
            serve.cli,
            [
                "serve",
                "--backend",
                "not-a-backend",
                "--custom_module_dirs",
                "/does/not/exist",
                "--help",
            ],
            prog_name="trtllm-serve",
        )

    assert result.exit_code == 0, result.output
    assert "--backend [pytorch|tensorrt|_autodeploy]" in result.output
    assert "--custom_module_dirs PATH" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_mm_embedding_serve_help_is_rich(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["mm_embedding_serve", "--help"],
                          prog_name="trtllm-serve",
                          root_mode=False):
        result = runner.invoke(serve.cli,
                               ["mm_embedding_serve", "--help"],
                               prog_name="trtllm-serve")

    assert result.exit_code == 0, result.output
    assert "Running an OpenAI API compatible server" in result.output
    assert "--tensor_parallel_size" in result.output
    assert "Additional arguments are forwarded" not in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_disaggregated_help_is_rich(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["disaggregated", "--help"],
                          prog_name="trtllm-serve",
                          root_mode=False):
        result = runner.invoke(serve.cli,
                               ["disaggregated", "--help"],
                               prog_name="trtllm-serve")

    assert result.exit_code == 0, result.output
    assert "Running server in disaggregated mode" in result.output
    assert "--server_start_timeout INTEGER" in result.output
    assert "Additional arguments are forwarded" not in result.output
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


def test_root_bench_subcommand_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(bench, "_delegate_to_legacy_bench",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["bench", "throughput", "--help"],
                          prog_name="trtllm",
                          root_mode=True):
        result = runner.invoke(root_cli, ["bench", "throughput", "--help"])

    assert result.exit_code == 0, result.output
    assert "Run throughput benchmarking." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_bench_subcommand_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(bench, "_delegate_to_legacy_bench",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(bench.cli,
                           ["throughput", "--help"],
                           prog_name="trtllm-bench")

    assert result.exit_code == 0, result.output
    assert "Run throughput benchmarking." in result.output
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


def test_root_eval_subcommand_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(eval_cli, "_delegate_to_legacy_eval",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["eval", "mmlu", "--help"],
                          prog_name="trtllm",
                          root_mode=True):
        result = runner.invoke(root_cli, ["eval", "mmlu", "--help"])

    assert result.exit_code == 0, result.output
    assert "Evaluate on MMLU." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_eval_subcommand_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(eval_cli, "_delegate_to_legacy_eval",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(eval_cli.cli,
                           ["mmlu", "--help"],
                           prog_name="trtllm-eval")

    assert result.exit_code == 0, result.output
    assert "Evaluate on MMLU." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


@pytest.mark.parametrize("command_name", ["build", "prune", "refit"])
@pytest.mark.parametrize("help_flag", ["--help", "-h"])
def test_root_argparse_help_delegates_without_importing_runtime(
        monkeypatch, command_name, help_flag):
    runner = CliRunner()
    baseline = set(sys.modules)
    calls = []
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: calls.append((args, kwargs)))

    result = runner.invoke(root_cli, [command_name, help_flag])

    assert result.exit_code == 0, result.output
    assert calls == [(
        (f"tensorrt_llm.commands.{command_name}", "main"),
        {
            "top_level_command": command_name,
            "standalone_prog_name": f"trtllm-{command_name}",
        },
    )]
    assert not _new_tensorrt_llm_modules(baseline)


@pytest.mark.parametrize(
    ("command_name", "args"),
    [
        ("build", ["--checkpoint_dir", "dummy-checkpoint"]),
        ("prune", ["--checkpoint_dir", "dummy-checkpoint"]),
        ("refit", [
            "--engine_dir",
            "dummy-engine",
            "--checkpoint_dir",
            "dummy-checkpoint",
            "--output_dir",
            "dummy-output",
        ]),
    ],
)
def test_root_argparse_command_delegates_raw_args_without_importing_runtime(
        monkeypatch, command_name, args):
    runner = CliRunner()
    baseline = set(sys.modules)
    calls = []
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *delegate_args, **kwargs: calls.append(
                            (delegate_args, kwargs)))

    result = runner.invoke(root_cli, [command_name, *args])

    assert result.exit_code == 0, result.output
    assert calls == [(
        (f"tensorrt_llm.commands.{command_name}", "main"),
        {
            "top_level_command": command_name,
            "standalone_prog_name": f"trtllm-{command_name}",
        },
    )]
    assert not _new_tensorrt_llm_modules(baseline)


def test_eval_invalid_config_fails_before_delegate(monkeypatch, tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("foo: [\n", encoding="utf-8")
    monkeypatch.setattr(eval_cli, "_delegate_to_legacy_eval",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(eval_cli.cli, [
        "--model",
        "dummy-model",
        "--config",
        str(bad_config),
        "mmlu",
    ],
                           prog_name="trtllm-eval")

    assert result.exit_code == 2, result.output
    assert "Invalid value for --config" in result.output
    assert "Invalid YAML" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_eval_help_skips_invalid_config_validation(monkeypatch, tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("foo: [\n", encoding="utf-8")
    monkeypatch.setattr(eval_cli, "_delegate_to_legacy_eval",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state([
            "--model",
            "dummy-model",
            "--config",
            str(bad_config),
            "mmlu",
            "--help",
    ],
                          prog_name="trtllm-eval",
                          root_mode=False):
        result = runner.invoke(eval_cli.cli, [
            "--model",
            "dummy-model",
            "--config",
            str(bad_config),
            "mmlu",
            "--help",
        ],
                               prog_name="trtllm-eval")

    assert result.exit_code == 0, result.output
    assert "Evaluate on MMLU." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_serve_invalid_config_fails_before_delegate(monkeypatch, tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("foo: [\n", encoding="utf-8")
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(serve.cli, [
        "dummy-model",
        "--config",
        str(bad_config),
    ],
                           prog_name="trtllm-serve")

    assert result.exit_code == 2, result.output
    assert "Usage: trtllm-serve [OPTIONS] MODEL" in result.output
    assert "Usage: trtllm-serve serve" not in result.output
    assert "Invalid value for --config" in result.output
    assert "Invalid YAML" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_serve_invalid_media_io_kwargs_fails_before_delegate(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(serve.cli, [
        "dummy-model",
        "--media_io_kwargs",
        "{invalid-json",
    ],
                           prog_name="trtllm-serve")

    assert result.exit_code == 2, result.output
    assert "Usage: trtllm-serve [OPTIONS] MODEL" in result.output
    assert "Usage: trtllm-serve serve" not in result.output
    assert "Invalid value for --media_io_kwargs" in result.output
    assert "Invalid JSON" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_disaggregated_invalid_config_fails_before_delegate(monkeypatch,
                                                            tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("foo: [\n", encoding="utf-8")
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(serve.cli, [
        "disaggregated",
        "--config",
        str(bad_config),
    ],
                           prog_name="trtllm-serve")

    assert result.exit_code == 2, result.output
    assert "Invalid value for --config" in result.output
    assert "Invalid YAML" in result.output
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


def test_root_serve_implicit_default_validation_usage_is_collapsed(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(serve, "_delegate_to_legacy_serve",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(root_cli, [
        "serve",
        "dummy-model",
        "--media_io_kwargs",
        "{invalid-json",
    ],
                           prog_name="trtllm")

    assert result.exit_code == 2, result.output
    assert "Usage: trtllm serve [OPTIONS] MODEL" in result.output
    assert "Usage: trtllm serve serve" not in result.output
    assert "Invalid value for --media_io_kwargs" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_root_bench_command_delegates(monkeypatch):
    runner = CliRunner()
    calls = []
    monkeypatch.setattr(bench, "_delegate_to_legacy_bench",
                        lambda: calls.append("bench"))

    result = runner.invoke(root_cli,
                           ["bench", "--model", "dummy-model", "throughput"])

    assert result.exit_code == 0, result.output
    assert calls == ["bench"]


def test_bench_model_requirement_is_restored_after_help(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(bench, "_delegate_to_legacy_bench", lambda: None)

    help_result = runner.invoke(bench.cli,
                                ["throughput", "--help"],
                                prog_name="trtllm-bench")
    missing_model_result = runner.invoke(bench.cli, ["throughput"])

    assert help_result.exit_code == 0, help_result.output
    assert "Run throughput benchmarking." in help_result.output
    assert missing_model_result.exit_code == 2
    assert "Missing option '--model' / '-m'." in missing_model_result.output


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
