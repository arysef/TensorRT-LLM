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
from trtllm_cli._help import (NotRequiredForHelp, echo_lightweight_help,
                              has_help_flag,
                              should_show_lightweight_help)
from trtllm_cli._validation import validate_yaml_file

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
@click.option(
    "--model",
    required=True,
    type=str,
    cls=NotRequiredForHelp,
    help="model name | HF checkpoint path | TensorRT engine path",
)
@click.option(
    "--tokenizer",
    type=str,
    default=None,
    help="Path | Name of the tokenizer."
    "Specify this value only if using TensorRT engine as model.",
)
@click.option(
    "--custom_tokenizer",
    type=str,
    default=None,
    help=
    "Custom tokenizer type: alias (e.g., 'deepseek_v32') or Python import path "
    "(e.g., 'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer'). [Experimental]",
)
@click.option(
    "--backend",
    type=click.Choice(["pytorch", "tensorrt"]),
    default="pytorch",
    help="The backend to use for evaluation. Default is pytorch backend.",
)
@click.option(
    "--log_level",
    type=click.Choice(LOG_LEVELS),
    default="info",
    help="The logging level.",
)
@click.option("--max_beam_width",
              type=int,
              default=1,
              help="Maximum number of beams for beam search decoding.")
@click.option("--max_batch_size",
              type=int,
              default=2048,
              help="Maximum number of requests that the engine can schedule.")
@click.option(
    "--max_num_tokens",
    type=int,
    default=8192,
    help=
    "Maximum number of batched input tokens after padding is removed in each batch.",
)
@click.option(
    "--max_seq_len",
    type=int,
    default=None,
    help="Maximum total length of one request, including prompt and outputs. "
    "If unspecified, the value is deduced from the model config.",
)
@click.option("--tp_size", type=int, default=1, help="Tensor parallelism size.")
@click.option("--pp_size", type=int, default=1, help="Pipeline parallelism size.")
@click.option("--ep_size", type=int, default=None, help="expert parallelism size")
@click.option(
    "--gpus_per_node",
    type=int,
    default=None,
    help="Number of GPUs per node. Default to None, and it will be "
    "detected automatically.",
)
@click.option(
    "--kv_cache_free_gpu_memory_fraction",
    type=float,
    default=0.9,
    help="Free GPU memory fraction reserved for KV Cache, "
    "after allocating model weights and buffers.",
)
@click.option("--trust_remote_code",
              is_flag=True,
              default=False,
              help="Flag for HF transformers.")
@click.option(
    "--revision",
    type=str,
    default=None,
    help="The revision to use for the HuggingFace model "
    "(branch name, tag name, or commit id).",
)
@click.option(
    "--config",
    "--extra_llm_api_options",
    "extra_llm_api_options",
    type=str,
    default=None,
    help="Path to a YAML file that overwrites the parameters. "
    "Can be specified as either --config or --extra_llm_api_options.",
)
@click.option("--disable_kv_cache_reuse",
              is_flag=True,
              default=False,
              help="Flag for disabling KV cache reuse.")
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


cli.add_command(_proxy_subcommand("cnn_dailymail", "Evaluate on CNN/DailyMail."))
cli.add_command(_proxy_subcommand("mmlu", "Evaluate on MMLU."))
cli.add_command(_proxy_subcommand("gsm8k", "Evaluate on GSM8K."))
cli.add_command(_proxy_subcommand("gpqa_diamond", "Evaluate on GPQA Diamond."))
cli.add_command(_proxy_subcommand("gpqa_main", "Evaluate on GPQA Main."))
cli.add_command(
    _proxy_subcommand("gpqa_extended", "Evaluate on GPQA Extended."))
cli.add_command(
    _proxy_subcommand("json_mode_eval", "Evaluate JSON mode accuracy."))
cli.add_command(_proxy_subcommand("mmmu", "Evaluate on MMMU."))
cli.add_command(_proxy_subcommand("longbench_v1", "Evaluate on LongBench v1."))
cli.add_command(_proxy_subcommand("longbench_v2", "Evaluate on LongBench v2."))
