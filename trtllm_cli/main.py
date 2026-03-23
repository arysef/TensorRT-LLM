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

import importlib
import sys
from contextlib import contextmanager

import click

PROXY_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


@contextmanager
def _patched_argv(argv0: str, argv: list[str]):
    previous_argv = sys.argv
    sys.argv = [argv0, *argv]
    try:
        yield
    finally:
        sys.argv = previous_argv


def _run_click_command(module_path: str, *, prog_name: str,
                       args: tuple[str, ...]) -> None:
    command = getattr(importlib.import_module(module_path), "main")
    command.main(args=list(args), prog_name=prog_name)


def _run_argparse_command(module_path: str, *, prog_name: str,
                          args: tuple[str, ...]) -> None:
    command = getattr(importlib.import_module(module_path), "main")
    with _patched_argv(prog_name, list(args)):
        command()


def _click_proxy(name: str, module_path: str, short_help: str):

    @click.command(
        name=name,
        short_help=short_help,
        help=f"{short_help} Additional arguments are forwarded to the "
        f"legacy trtllm-{name} command.",
        add_help_option=False,
        context_settings=PROXY_CONTEXT_SETTINGS,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def command(args) -> None:
        _run_click_command(module_path,
                           prog_name=f"trtllm {name}",
                           args=args)

    return command


def _argparse_proxy(name: str, module_path: str, short_help: str):

    @click.command(
        name=name,
        short_help=short_help,
        help=f"{short_help} Additional arguments are forwarded to the "
        f"legacy trtllm-{name} command.",
        add_help_option=False,
        context_settings=PROXY_CONTEXT_SETTINGS,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def command(args) -> None:
        _run_argparse_command(module_path,
                              prog_name=f"trtllm {name}",
                              args=args)

    return command


@click.group()
def cli() -> None:
    """TensorRT-LLM command line interface."""


cli.add_command(
    _click_proxy("serve", "tensorrt_llm.commands.serve",
                 "Serve TensorRT-LLM models."))
cli.add_command(
    _click_proxy("bench", "tensorrt_llm.commands.bench",
                 "Benchmark TensorRT-LLM models."))
cli.add_command(
    _click_proxy("eval", "tensorrt_llm.commands.eval",
                 "Evaluate models with TensorRT-LLM."))
cli.add_command(
    _argparse_proxy("build", "tensorrt_llm.commands.build",
                    "Build TensorRT-LLM engines."))
cli.add_command(
    _argparse_proxy("prune", "tensorrt_llm.commands.prune",
                    "Prune model checkpoints."))
cli.add_command(
    _argparse_proxy("refit", "tensorrt_llm.commands.refit",
                    "Refit TensorRT-LLM engines."))


def main():
    return cli.main(args=sys.argv[1:], prog_name="trtllm")
