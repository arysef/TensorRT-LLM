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

from trtllm_cli._dispatch import delegate_click_command

PROXY_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _delegate_to_legacy_serve() -> None:
    delegate_click_command(
        "tensorrt_llm.commands.serve",
        "main",
        top_level_command="serve",
        standalone_prog_name="trtllm-serve",
    )


class DefaultGroup(click.Group):
    """Click group that preserves trtllm-serve's default serve subcommand."""

    def resolve_command(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            return "serve", self.commands["serve"], args
        return super().resolve_command(ctx, args)


@click.command(
    name="serve",
    short_help="Serve a model with the OpenAI-compatible API.",
    add_help_option=False,
    context_settings=PROXY_CONTEXT_SETTINGS,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def serve_proxy(args) -> None:
    del args
    _delegate_to_legacy_serve()


@click.command(
    name="disaggregated",
    short_help="Launch a disaggregated serving stack.",
    add_help_option=False,
    context_settings=PROXY_CONTEXT_SETTINGS,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def disaggregated_proxy(args) -> None:
    del args
    _delegate_to_legacy_serve()


@click.command(
    name="disaggregated_mpi_worker",
    short_help="Start a disaggregated MPI worker.",
    add_help_option=False,
    context_settings=PROXY_CONTEXT_SETTINGS,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def disaggregated_mpi_worker_proxy(args) -> None:
    del args
    _delegate_to_legacy_serve()


@click.command(
    name="mm_embedding_serve",
    short_help="Serve multimodal embedding models.",
    add_help_option=False,
    context_settings=PROXY_CONTEXT_SETTINGS,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def mm_embedding_serve_proxy(args) -> None:
    del args
    _delegate_to_legacy_serve()


@click.group(cls=DefaultGroup)
def cli() -> None:
    """TensorRT-LLM serving commands."""


cli.add_command(serve_proxy)
cli.add_command(disaggregated_proxy)
cli.add_command(disaggregated_mpi_worker_proxy)
cli.add_command(mm_embedding_serve_proxy)
