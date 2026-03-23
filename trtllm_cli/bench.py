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

from trtllm_cli._bench_metadata import (BENCH_GROUP_CONTEXT_SETTINGS,
                                        BENCH_SUBCOMMANDS, add_bench_options)
from trtllm_cli._dispatch import delegate_click_command
from trtllm_cli._help import (NotRequiredForHelp, echo_lightweight_help,
                              should_show_lightweight_help)
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

def _proxy_subcommand(name: str, short_help: str):

    @click.command(
        name=name,
        short_help=short_help,
        help=f"{short_help} Additional arguments are forwarded to the "
        "legacy trtllm-bench command at runtime.",
        add_help_option=False,
        context_settings=PROXY_CONTEXT_SETTINGS,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def command(ctx, args) -> None:
        if should_show_lightweight_help(ctx, args):
            echo_lightweight_help(ctx)
            return
        _delegate_to_legacy_bench()

    return command


@click.group(context_settings=BENCH_GROUP_CONTEXT_SETTINGS)
@add_bench_options(option_cls=NotRequiredForHelp)
def cli(model: str, model_path, workspace, log_level: str,
        revision: str | None) -> None:
    """Benchmark TensorRT-LLM models."""
    del model, model_path, workspace, log_level, revision


for name, short_help in BENCH_SUBCOMMANDS:
    cli.add_command(_proxy_subcommand(name, short_help))
