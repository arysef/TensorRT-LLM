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
from __future__ import annotations

import importlib
from typing import Optional

import click

LOG_LEVELS = (
    "internal_error",
    "error",
    "warning",
    "info",
    "verbose",
    "debug",
    "trace",
)
EVAL_BACKEND_CHOICES = ("pytorch", "tensorrt")
MAX_BEAM_WIDTH_DEFAULT = 1
MAX_BATCH_SIZE_DEFAULT = 2048
MAX_NUM_TOKENS_DEFAULT = 8192
MAX_SEQ_LEN_DEFAULT = None
KV_CACHE_FREE_GPU_MEMORY_FRACTION_DEFAULT = 0.9
EVAL_SUBCOMMANDS = (
    ("cnn_dailymail", "Evaluate on CNN/DailyMail."),
    ("mmlu", "Evaluate on MMLU."),
    ("gsm8k", "Evaluate on GSM8K."),
    ("gpqa_diamond", "Evaluate on GPQA Diamond."),
    ("gpqa_main", "Evaluate on GPQA Main."),
    ("gpqa_extended", "Evaluate on GPQA Extended."),
    ("json_mode_eval", "Evaluate JSON mode accuracy."),
    ("mmmu", "Evaluate on MMMU."),
    ("longbench_v1", "Evaluate on LongBench v1."),
    ("longbench_v2", "Evaluate on LongBench v2."),
)

SUBCOMMAND_SPECS = {
    "cnn_dailymail": ("tensorrt_llm.evaluate.cnn_dailymail", "CnnDailymail"),
    "mmlu": ("tensorrt_llm.evaluate.mmlu", "MMLU"),
    "gsm8k": ("tensorrt_llm.evaluate.lm_eval", "GSM8K"),
    "gpqa_diamond": ("tensorrt_llm.evaluate.lm_eval", "GPQADiamond"),
    "gpqa_main": ("tensorrt_llm.evaluate.lm_eval", "GPQAMain"),
    "gpqa_extended": ("tensorrt_llm.evaluate.lm_eval", "GPQAExtended"),
    "json_mode_eval": ("tensorrt_llm.evaluate.json_mode_eval",
                        "JsonModeEval"),
    "mmmu": ("tensorrt_llm.evaluate.lm_eval", "MMMU"),
    "longbench_v1": ("tensorrt_llm.evaluate.lm_eval", "LongBenchV1"),
    "longbench_v2": ("tensorrt_llm.evaluate.longbench_v2", "LongBenchV2"),
}

SHORT_HELP_BY_NAME = dict(EVAL_SUBCOMMANDS)


class LazyEvalGroup(click.Group):

    def list_commands(self, ctx):
        return [name for name, _ in EVAL_SUBCOMMANDS]

    def format_commands(self, ctx, formatter):
        with formatter.section("Commands"):
            formatter.write_dl(list(EVAL_SUBCOMMANDS))

    def get_command(self, ctx, name):
        spec = SUBCOMMAND_SPECS.get(name)
        if spec is None:
            return None

        module = importlib.import_module(spec[0])
        command = getattr(getattr(module, spec[1]), "command")
        command.short_help = SHORT_HELP_BY_NAME[name]
        return command


@click.group(cls=LazyEvalGroup)
@click.option(
    "--model",
    required=True,
    type=str,
    help="model name | HF checkpoint path | TensorRT engine path",
)
@click.option("--tokenizer",
              type=str,
              default=None,
              help="Path | Name of the tokenizer."
              "Specify this value only if using TensorRT engine as model.")
@click.option(
    "--custom_tokenizer",
    type=str,
    default=None,
    help=
    "Custom tokenizer type: alias (e.g., 'deepseek_v32') or Python import path "
    "(e.g., 'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer'). [Experimental]"
)
@click.option(
    "--backend",
    type=click.Choice(EVAL_BACKEND_CHOICES),
    default="pytorch",
    help="The backend to use for evaluation. Default is pytorch backend.")
@click.option('--log_level',
              type=click.Choice(LOG_LEVELS),
              default='info',
              help="The logging level.")
@click.option("--max_beam_width",
              type=int,
              default=MAX_BEAM_WIDTH_DEFAULT,
              help="Maximum number of beams for beam search decoding.")
@click.option("--max_batch_size",
              type=int,
              default=MAX_BATCH_SIZE_DEFAULT,
              help="Maximum number of requests that the engine can schedule.")
@click.option(
    "--max_num_tokens",
    type=int,
    default=MAX_NUM_TOKENS_DEFAULT,
    help=
    "Maximum number of batched input tokens after padding is removed in each batch."
)
@click.option(
    "--max_seq_len",
    type=int,
    default=MAX_SEQ_LEN_DEFAULT,
    help="Maximum total length of one request, including prompt and outputs. "
    "If unspecified, the value is deduced from the model config.")
@click.option("--tp_size", type=int, default=1, help='Tensor parallelism size.')
@click.option("--pp_size",
              type=int,
              default=1,
              help='Pipeline parallelism size.')
@click.option("--ep_size",
              type=int,
              default=None,
              help="expert parallelism size")
@click.option("--gpus_per_node",
              type=int,
              default=None,
              help="Number of GPUs per node. Default to None, and it will be "
              "detected automatically.")
@click.option("--kv_cache_free_gpu_memory_fraction",
              type=float,
              default=KV_CACHE_FREE_GPU_MEMORY_FRACTION_DEFAULT,
              help="Free GPU memory fraction reserved for KV Cache, "
              "after allocating model weights and buffers.")
@click.option("--trust_remote_code",
              is_flag=True,
              default=False,
              help="Flag for HF transformers.")
@click.option("--revision",
              type=str,
              default=None,
              help="The revision to use for the HuggingFace model "
              "(branch name, tag name, or commit id).")
@click.option("--config",
              "--extra_llm_api_options",
              "extra_llm_api_options",
              type=str,
              default=None,
              help="Path to a YAML file that overwrites the parameters. "
              "Can be specified as either --config or --extra_llm_api_options.")
@click.option("--disable_kv_cache_reuse",
              is_flag=True,
              default=False,
              help="Flag for disabling KV cache reuse.")
@click.pass_context
def main(ctx, model: str, tokenizer: Optional[str],
         custom_tokenizer: Optional[str], log_level: str, backend: str,
         max_beam_width: int, max_batch_size: int, max_num_tokens: int,
         max_seq_len: Optional[int], tp_size: int, pp_size: int,
         ep_size: Optional[int],
         gpus_per_node: Optional[int], kv_cache_free_gpu_memory_fraction: float,
         trust_remote_code: bool, revision: Optional[str],
         extra_llm_api_options: Optional[str], disable_kv_cache_reuse: bool):
    import tensorrt_llm.profiler as profiler

    from tensorrt_llm import LLM as PyTorchLLM
    from tensorrt_llm._tensorrt_engine import LLM
    from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig
    from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
    from tensorrt_llm.logger import logger

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

    if backend == "pytorch":
        llm_cls = PyTorchLLM
        llm_args.update(max_batch_size=max_batch_size,
                        max_num_tokens=max_num_tokens,
                        max_beam_width=max_beam_width,
                        max_seq_len=max_seq_len)
    elif backend == "tensorrt":
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

    ctx.obj = llm


if __name__ == "__main__":
    main()
