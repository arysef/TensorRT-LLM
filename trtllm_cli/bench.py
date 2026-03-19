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

from trtllm_cli._dispatch import delegate_click_command

LOG_LEVELS = (
    "internal_error",
    "error",
    "warning",
    "info",
    "verbose",
    "debug",
    "trace",
)
PROXY_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _delegate_to_legacy_bench() -> None:
    delegate_click_command(
        "tensorrt_llm.commands.bench",
        "main",
        top_level_command="bench",
        standalone_prog_name="trtllm-bench",
    )


class NotRequiredForHelp(click.Option):
    """A click.Option that is not enforced as required when --help is in args."""

    def handle_parse_result(self, ctx, opts, args):
        help_flags = ctx.help_option_names or ["--help"]
        if any(arg in help_flags for arg in args):
            self.required = False
        return super().handle_parse_result(ctx, opts, args)


def _proxy_subcommand(name: str, short_help: str):

    @click.command(
        name=name,
        short_help=short_help,
        add_help_option=False,
        context_settings=PROXY_CONTEXT_SETTINGS,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def command(args) -> None:
        del args
        _delegate_to_legacy_bench()

    return command


@click.group(context_settings={"show_default": True})
@click.option(
    "--model",
    "-m",
    required=True,
    type=str,
    cls=NotRequiredForHelp,
    help="The Huggingface name of the model to benchmark.",
)
@click.option(
    "--model_path",
    required=False,
    default=None,
    type=click.Path(writable=False, readable=True, path_type=Path),
    help=
    "Path to a Huggingface checkpoint directory for loading model components.",
)
@click.option(
    "--workspace",
    "-w",
    required=False,
    type=click.Path(writable=True, readable=True, path_type=Path),
    default="/tmp",  # nosec B108
    help="The directory to store benchmarking intermediate files.",
)
@click.option(
    "--log_level",
    type=click.Choice(LOG_LEVELS),
    default="info",
    help="The logging level.",
)
@click.option(
    "--revision",
    type=str,
    default=None,
    help="The revision to use for the HuggingFace model "
    "(branch name, tag name, or commit id).",
)
def cli(model: str, model_path: Path, workspace: Path, log_level: str,
        revision: str | None) -> None:
    """Benchmark TensorRT-LLM models."""
    del model, model_path, workspace, log_level, revision


cli.add_command(_proxy_subcommand("build", "Build a benchmark engine."))
cli.add_command(
    _proxy_subcommand("throughput", "Run throughput benchmarking."))
cli.add_command(_proxy_subcommand("latency", "Run latency benchmarking."))
cli.add_command(
    _proxy_subcommand("prepare-dataset", "Prepare benchmark datasets."))
cli.add_command(
    _proxy_subcommand("visual-gen", "Run visual generation benchmarking."))
