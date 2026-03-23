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

import click

from trtllm_cli import bench, eval as eval_cli, serve
from trtllm_cli._dispatch import (delegate_argparse_command, invoke_entry)

PROXY_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _route_argparse_command(name: str, module_path: str,
                            standalone_prog_name: str, short_help: str):

    @click.command(
        name=name,
        short_help=short_help,
        help=f"{short_help} Additional arguments are forwarded to "
        f"the legacy {standalone_prog_name} command at runtime.",
        add_help_option=False,
        context_settings=PROXY_CONTEXT_SETTINGS,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def command(args) -> None:
        delegate_argparse_command(
            module_path,
            "main",
            top_level_command=name,
            standalone_prog_name=standalone_prog_name,
        )

    return command


@click.group()
def cli() -> None:
    """TensorRT-LLM command line interface."""


cli.add_command(serve.cli, name="serve")
cli.add_command(bench.cli, name="bench")
cli.add_command(eval_cli.cli, name="eval")
build_command = _route_argparse_command("build", "tensorrt_llm.commands.build",
                                        "trtllm-build",
                                        "Build TensorRT-LLM engines.")
prune_command = _route_argparse_command("prune", "tensorrt_llm.commands.prune",
                                        "trtllm-prune",
                                        "Prune model checkpoints.")
refit_command = _route_argparse_command("refit", "tensorrt_llm.commands.refit",
                                        "trtllm-refit",
                                        "Refit TensorRT-LLM engines.")

cli.add_command(build_command)
cli.add_command(prune_command)
cli.add_command(refit_command)


def main():
    return invoke_entry(cli, prog_name="trtllm", root_mode=True)


def serve_entry():
    return invoke_entry(serve.cli, prog_name="trtllm-serve")


def bench_entry():
    return invoke_entry(bench.cli, prog_name="trtllm-bench")


def eval_entry():
    return invoke_entry(eval_cli.cli, prog_name="trtllm-eval")
