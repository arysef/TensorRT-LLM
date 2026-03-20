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

from __future__ import annotations

from pathlib import Path

import click

from trtllm_cli import bench, eval as eval_cli, serve
from trtllm_cli._dispatch import (delegate_argparse_command, invoke_entry)
from trtllm_cli._help import (echo_lightweight_help,
                              should_show_lightweight_help)
from trtllm_cli._validation import (_extract_option_values,
                                    validate_existing_directory,
                                    validate_existing_file_options,
                                    validate_json_file,
                                    validate_json_file_options)

PROXY_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _proxy_argparse_command(name: str, module_path: str,
                            standalone_prog_name: str,
                            short_help: str,
                            validator=None):

    @click.command(
        name=name,
        short_help=short_help,
        help=f"{short_help} Additional arguments are forwarded to "
        f"the legacy {standalone_prog_name} command at runtime.",
        add_help_option=False,
        context_settings=PROXY_CONTEXT_SETTINGS,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def command(ctx, args) -> None:
        if should_show_lightweight_help(ctx, args):
            echo_lightweight_help(ctx)
            return
        if validator is not None:
            validator(args)
        delegate_argparse_command(
            module_path,
            "main",
            top_level_command=name,
            standalone_prog_name=standalone_prog_name,
        )

    return command


def _single_option_value(args: tuple[str, ...], *option_names: str) -> str | None:
    values = _extract_option_values(args, option_names)
    return values[-1] if values else None


def _require_option(args: tuple[str, ...], *option_names: str) -> str:
    value = _single_option_value(args, *option_names)
    if value is None:
        option_name = next(
            option_name for option_name in option_names
            if option_name.startswith("--"))
        raise click.BadParameter("Option is required.", param_hint=option_name)
    return value


def _validate_checkpoint_config_path(path_value: str, *, param_hint: str) -> None:
    path = Path(path_value)
    if path.suffix == ".json":
        validate_json_file(path_value, param_hint=param_hint)
        return

    validate_existing_directory(path_value, param_hint=param_hint)
    config_path = path / "config.json"
    validate_json_file(str(config_path), param_hint=param_hint)


def _validate_checkpoint_directory(path_value: str, *, param_hint: str) -> None:
    validate_existing_directory(path_value, param_hint=param_hint)
    validate_json_file(str(Path(path_value) / "config.json"),
                       param_hint=param_hint)


def _validate_build_args(args: tuple[str, ...]) -> None:
    checkpoint_dir = _single_option_value(args, "--checkpoint_dir")
    model_config = _single_option_value(args, "--model_config")
    if checkpoint_dir is None and model_config is None:
        raise click.BadParameter("Either --checkpoint_dir or --model_config is required.",
                                 param_hint="--checkpoint_dir")

    if checkpoint_dir is not None:
        _validate_checkpoint_config_path(checkpoint_dir,
                                         param_hint="--checkpoint_dir")
    if model_config is not None:
        validate_json_file(model_config, param_hint="--model_config")

    validate_json_file_options(args, "--build_config")
    validate_existing_file_options(args, "--model_cls_file")

    model_cls_file = _single_option_value(args, "--model_cls_file")
    model_cls_name = _single_option_value(args, "--model_cls_name")
    if model_cls_file is not None and model_cls_name is None:
        raise click.BadParameter(
            "Option is required when --model_cls_file is set.",
            param_hint="--model_cls_name")


def _validate_prune_args(args: tuple[str, ...]) -> None:
    checkpoint_dir = _require_option(args, "--checkpoint_dir")
    _validate_checkpoint_directory(checkpoint_dir, param_hint="--checkpoint_dir")


def _refit_architecture(engine_config: dict, checkpoint_config: dict) -> tuple[str, str]:
    if not isinstance(engine_config, dict):
        raise click.BadParameter("Engine config must be a JSON object.",
                                 param_hint="--engine_dir")
    if not isinstance(checkpoint_config, dict):
        raise click.BadParameter("Checkpoint config must be a JSON object.",
                                 param_hint="--checkpoint_dir")
    engine_arch = engine_config.get("pretrained_config", {}).get("architecture")
    checkpoint_arch = checkpoint_config.get("architecture")
    if not engine_arch:
        raise click.BadParameter(
            "Engine config is missing pretrained_config.architecture.",
            param_hint="--engine_dir")
    if not checkpoint_arch:
        raise click.BadParameter("Checkpoint config is missing architecture.",
                                 param_hint="--checkpoint_dir")
    return engine_arch, checkpoint_arch


def _validate_refit_args(args: tuple[str, ...]) -> None:
    engine_dir = _require_option(args, "--engine_dir")
    checkpoint_dir = _require_option(args, "--checkpoint_dir")
    _require_option(args, "--output_dir")

    validate_existing_directory(engine_dir, param_hint="--engine_dir")
    validate_existing_directory(checkpoint_dir, param_hint="--checkpoint_dir")

    engine_config = validate_json_file(str(Path(engine_dir) / "config.json"),
                                       param_hint="--engine_dir")
    checkpoint_config = validate_json_file(
        str(Path(checkpoint_dir) / "config.json"),
        param_hint="--checkpoint_dir")
    engine_arch, checkpoint_arch = _refit_architecture(engine_config,
                                                       checkpoint_config)
    if engine_arch != checkpoint_arch:
        raise click.BadParameter(
            "Engine architecture does not match checkpoint architecture.",
            param_hint="--checkpoint_dir")


@click.group()
def cli() -> None:
    """TensorRT-LLM command line interface."""


cli.add_command(serve.cli, name="serve")
cli.add_command(bench.cli, name="bench")
cli.add_command(eval_cli.cli, name="eval")
build_command = _proxy_argparse_command("build",
                                        "tensorrt_llm.commands.build",
                                        "trtllm-build",
                                        "Build TensorRT-LLM engines.",
                                        validator=_validate_build_args)
prune_command = _proxy_argparse_command("prune",
                                        "tensorrt_llm.commands.prune",
                                        "trtllm-prune",
                                        "Prune model checkpoints.",
                                        validator=_validate_prune_args)
refit_command = _proxy_argparse_command("refit",
                                        "tensorrt_llm.commands.refit",
                                        "trtllm-refit",
                                        "Refit TensorRT-LLM engines.",
                                        validator=_validate_refit_args)

cli.add_command(build_command)
cli.add_command(prune_command)
cli.add_command(refit_command)


def main():
    return invoke_entry(cli, prog_name="trtllm", root_mode=True)


def build_entry():
    return invoke_entry(build_command, prog_name="trtllm-build")


def prune_entry():
    return invoke_entry(prune_command, prog_name="trtllm-prune")


def refit_entry():
    return invoke_entry(refit_command, prog_name="trtllm-refit")


def serve_entry():
    return invoke_entry(serve.cli, prog_name="trtllm-serve")


def bench_entry():
    return invoke_entry(bench.cli, prog_name="trtllm-bench")


def eval_entry():
    return invoke_entry(eval_cli.cli, prog_name="trtllm-eval")
