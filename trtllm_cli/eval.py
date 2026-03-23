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

from trtllm_cli._dispatch import current_invocation_argv, delegate_click_command
from trtllm_cli._eval_metadata import EVAL_SUBCOMMANDS, add_eval_options
from trtllm_cli._help import (NotRequiredForHelp, echo_lightweight_help,
                              has_help_flag,
                              should_show_lightweight_help)
from trtllm_cli._validation import validate_yaml_file
PROXY_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _delegate_to_legacy_eval() -> None:
    delegate_click_command(
        "tensorrt_llm.commands.eval",
        "main",
        top_level_command="eval",
        standalone_prog_name="trtllm-eval",
    )


def _proxy_subcommand(name: str, short_help: str):

    @click.command(
        name=name,
        short_help=short_help,
        help=f"{short_help} Additional arguments are forwarded to the "
        "legacy trtllm-eval command at runtime.",
        add_help_option=False,
        context_settings=PROXY_CONTEXT_SETTINGS,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def command(ctx, args) -> None:
        if should_show_lightweight_help(ctx, args):
            echo_lightweight_help(ctx)
            return
        _delegate_to_legacy_eval()

    return command


@click.group()
@add_eval_options(option_cls=NotRequiredForHelp)
@click.pass_context
def cli(ctx, model: str, tokenizer: str | None, custom_tokenizer: str | None,
        backend: str, log_level: str, max_beam_width: int,
        max_batch_size: int, max_num_tokens: int, max_seq_len: int | None,
        tp_size: int, pp_size: int, ep_size: int | None,
        gpus_per_node: int | None,
        kv_cache_free_gpu_memory_fraction: float, trust_remote_code: bool,
        revision: str | None, extra_llm_api_options: str | None,
        disable_kv_cache_reuse: bool) -> None:
    """Evaluate models with TensorRT-LLM."""
    if not has_help_flag(current_invocation_argv(), ctx.help_option_names):
        validate_yaml_file(extra_llm_api_options, param_hint="--config")
    del model
    del tokenizer
    del custom_tokenizer
    del ctx
    del backend
    del log_level
    del max_beam_width
    del max_batch_size
    del max_num_tokens
    del max_seq_len
    del tp_size
    del pp_size
    del ep_size
    del gpus_per_node
    del kv_cache_free_gpu_memory_fraction
    del trust_remote_code
    del revision
    del extra_llm_api_options
    del disable_kv_cache_reuse


for name, short_help in EVAL_SUBCOMMANDS:
    cli.add_command(_proxy_subcommand(name, short_help))
