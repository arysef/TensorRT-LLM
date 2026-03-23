# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import click

import tensorrt_llm.profiler as profiler
from trtllm_cli._eval_metadata import EVAL_SUBCOMMANDS, add_eval_options
from trtllm_cli._help import NotRequiredForHelp

from .. import LLM as PyTorchLLM
from .._tensorrt_engine import LLM
from ..evaluate import (GSM8K, MMLU, MMMU, CnnDailymail, GPQADiamond,
                        GPQAExtended, GPQAMain, JsonModeEval, LongBenchV1,
                        LongBenchV2)
from ..llmapi import BuildConfig, KvCacheConfig
from ..llmapi.llm_utils import update_llm_args_with_extra_options
from ..logger import logger

EVAL_COMMANDS_BY_NAME = {
    CnnDailymail.command.name: CnnDailymail.command,
    MMLU.command.name: MMLU.command,
    GSM8K.command.name: GSM8K.command,
    GPQADiamond.command.name: GPQADiamond.command,
    GPQAMain.command.name: GPQAMain.command,
    GPQAExtended.command.name: GPQAExtended.command,
    JsonModeEval.command.name: JsonModeEval.command,
    MMMU.command.name: MMMU.command,
    LongBenchV1.command.name: LongBenchV1.command,
    LongBenchV2.command.name: LongBenchV2.command,
}


@click.group()
@add_eval_options(option_cls=NotRequiredForHelp)
@click.pass_context
def main(ctx, model: str, tokenizer: Optional[str],
         custom_tokenizer: Optional[str], log_level: str, backend: str,
         max_beam_width: int, max_batch_size: int, max_num_tokens: int,
         max_seq_len: Optional[int], tp_size: int, pp_size: int,
         ep_size: Optional[int],
         gpus_per_node: Optional[int], kv_cache_free_gpu_memory_fraction: float,
         trust_remote_code: bool, revision: Optional[str],
         extra_llm_api_options: Optional[str], disable_kv_cache_reuse: bool):
    logger.set_level(log_level)

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        enable_block_reuse=not disable_kv_cache_reuse)

    llm_args = {
        "model": model,
        "tokenizer": tokenizer,
        "custom_tokenizer": custom_tokenizer,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "moe_expert_parallel_size": ep_size,
        "gpus_per_node": gpus_per_node,
        "trust_remote_code": trust_remote_code,
        "revision": revision,
        "kv_cache_config": kv_cache_config,
    }

    if backend == 'pytorch':
        llm_cls = PyTorchLLM
        llm_args.update(max_batch_size=max_batch_size,
                        max_num_tokens=max_num_tokens,
                        max_beam_width=max_beam_width,
                        max_seq_len=max_seq_len)
    elif backend == 'tensorrt':
        llm_cls = LLM
        build_config = BuildConfig(max_batch_size=max_batch_size,
                                   max_num_tokens=max_num_tokens,
                                   max_beam_width=max_beam_width,
                                   max_seq_len=max_seq_len)
        llm_args.update(build_config=build_config)
    else:
        raise click.BadParameter(
            f"{backend} is not a known backend, check help for available options.",
            param_hint="backend")

    if extra_llm_api_options is not None:
        llm_args = update_llm_args_with_extra_options(llm_args,
                                                      extra_llm_api_options)

    profiler.start("trtllm init")
    llm = llm_cls(**llm_args)
    profiler.stop("trtllm init")
    elapsed_time = profiler.elapsed_time_in_sec("trtllm init")
    logger.info(f"TRTLLM initialization time: {elapsed_time:.3f} seconds.")
    profiler.reset("trtllm init")

    # Pass llm to subcommands
    ctx.obj = llm


for name, short_help in EVAL_SUBCOMMANDS:
    command = EVAL_COMMANDS_BY_NAME[name]
    command.short_help = short_help
    main.add_command(command, name=name)

if __name__ == "__main__":
    main()
