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

import trtllm_cli.main as main_cli
from trtllm_cli import bench, eval as eval_cli, serve
from trtllm_cli._dispatch import invocation_state
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


def test_root_build_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    with invocation_state(["build", "--help"],
                          prog_name="trtllm",
                          root_mode=True):
        result = runner.invoke(root_cli, ["build", "--help"])

    assert result.exit_code == 0, result.output
    assert "Build TensorRT-LLM engines." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_build_help_is_lightweight(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.build_command,
                           ["--help"],
                           prog_name="trtllm-build")

    assert result.exit_code == 0, result.output
    assert "Build TensorRT-LLM engines." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_build_requires_checkpoint_dir_or_model_config_before_delegate(
        monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.build_command, [], prog_name="trtllm-build")

    assert result.exit_code == 2, result.output
    assert "Either --checkpoint_dir or --model_config is required." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_build_invalid_model_config_fails_before_delegate(monkeypatch,
                                                          tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    bad_config = tmp_path / "model-config.json"
    bad_config.write_text("{invalid-json", encoding="utf-8")
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.build_command,
                           ["--model_config", str(bad_config)],
                           prog_name="trtllm-build")

    assert result.exit_code == 2, result.output
    assert "Invalid value for --model_config" in result.output
    assert "Invalid JSON" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_build_help_skips_invalid_model_config_validation(monkeypatch,
                                                          tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    bad_config = tmp_path / "model-config.json"
    bad_config.write_text("{invalid-json", encoding="utf-8")
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.build_command,
                           ["--model_config", str(bad_config), "--help"],
                           prog_name="trtllm-build")

    assert result.exit_code == 0, result.output
    assert "Build TensorRT-LLM engines." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_prune_missing_checkpoint_dir_fails_before_delegate(monkeypatch):
    runner = CliRunner()
    baseline = set(sys.modules)
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.prune_command, [], prog_name="trtllm-prune")

    assert result.exit_code == 2, result.output
    assert "Invalid value for --checkpoint_dir: Option is required." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_prune_invalid_checkpoint_config_fails_before_delegate(monkeypatch,
                                                               tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{invalid-json",
                                                 encoding="utf-8")
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.prune_command,
                           ["--checkpoint_dir", str(checkpoint_dir)],
                           prog_name="trtllm-prune")

    assert result.exit_code == 2, result.output
    assert "Invalid value for --checkpoint_dir" in result.output
    assert "Invalid JSON" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_prune_checkpoint_dir_must_be_directory(monkeypatch, tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    config_file = tmp_path / "config.json"
    config_file.write_text('{"architecture": "LlamaForCausalLM"}',
                           encoding="utf-8")
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.prune_command,
                           ["--checkpoint_dir", str(config_file)],
                           prog_name="trtllm-prune")

    assert result.exit_code == 2, result.output
    assert "Invalid value for --checkpoint_dir" in result.output
    assert "Directory" in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_refit_missing_output_dir_fails_before_delegate(monkeypatch, tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    engine_dir = tmp_path / "engine"
    checkpoint_dir = tmp_path / "checkpoint"
    engine_dir.mkdir()
    checkpoint_dir.mkdir()
    (engine_dir / "config.json").write_text(
        '{"pretrained_config": {"architecture": "LlamaForCausalLM"}}',
        encoding="utf-8")
    (checkpoint_dir / "config.json").write_text(
        '{"architecture": "LlamaForCausalLM"}', encoding="utf-8")
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.refit_command, [
        "--engine_dir",
        str(engine_dir),
        "--checkpoint_dir",
        str(checkpoint_dir),
    ],
                           prog_name="trtllm-refit")

    assert result.exit_code == 2, result.output
    assert "Invalid value for --output_dir: Option is required." in result.output
    assert not _new_tensorrt_llm_modules(baseline)


def test_refit_architecture_mismatch_fails_before_delegate(monkeypatch,
                                                           tmp_path):
    runner = CliRunner()
    baseline = set(sys.modules)
    engine_dir = tmp_path / "engine"
    checkpoint_dir = tmp_path / "checkpoint"
    engine_dir.mkdir()
    checkpoint_dir.mkdir()
    (engine_dir / "config.json").write_text(
        '{"pretrained_config": {"architecture": "LlamaForCausalLM"}}',
        encoding="utf-8")
    (checkpoint_dir / "config.json").write_text(
        '{"architecture": "MistralForCausalLM"}', encoding="utf-8")
    monkeypatch.setattr(main_cli, "delegate_argparse_command",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("delegate should not be called")))

    result = runner.invoke(main_cli.refit_command, [
        "--engine_dir",
        str(engine_dir),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--output_dir",
        str(tmp_path / "refit-out"),
    ],
                           prog_name="trtllm-refit")

    assert result.exit_code == 2, result.output
    assert "Engine architecture does not match checkpoint architecture." in result.output
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
