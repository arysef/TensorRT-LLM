from __future__ import annotations

from typing import Callable

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
EVAL_HELP = "Evaluate models with TensorRT-LLM."
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


def _option(*param_decls,
            option_cls: type[click.Option] | None = None,
            **kwargs) -> Callable:
    if option_cls is not None:
        kwargs.setdefault("cls", option_cls)
    return click.option(*param_decls, **kwargs)


def add_eval_options(
    *, option_cls: type[click.Option] | None = None
) -> Callable:

    def decorator(command: Callable) -> Callable:
        command = _option("--disable_kv_cache_reuse",
                          is_flag=True,
                          default=False,
                          help="Flag for disabling KV cache reuse.",
                          option_cls=option_cls)(command)
        command = _option(
            "--config",
            "--extra_llm_api_options",
            "extra_llm_api_options",
            type=str,
            default=None,
            help="Path to a YAML file that overwrites the parameters. "
            "Can be specified as either --config or --extra_llm_api_options.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--revision",
            type=str,
            default=None,
            help="The revision to use for the HuggingFace model "
            "(branch name, tag name, or commit id).",
            option_cls=option_cls,
        )(command)
        command = _option("--trust_remote_code",
                          is_flag=True,
                          default=False,
                          help="Flag for HF transformers.",
                          option_cls=option_cls)(command)
        command = _option(
            "--kv_cache_free_gpu_memory_fraction",
            type=float,
            default=KV_CACHE_FREE_GPU_MEMORY_FRACTION_DEFAULT,
            help="Free GPU memory fraction reserved for KV Cache, "
            "after allocating model weights and buffers.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--gpus_per_node",
            type=int,
            default=None,
            help="Number of GPUs per node. Default to None, and it will be "
            "detected automatically.",
            option_cls=option_cls,
        )(command)
        command = _option("--ep_size",
                          type=int,
                          default=None,
                          help="expert parallelism size",
                          option_cls=option_cls)(command)
        command = _option("--pp_size",
                          type=int,
                          default=1,
                          help="Pipeline parallelism size.",
                          option_cls=option_cls)(command)
        command = _option("--tp_size",
                          type=int,
                          default=1,
                          help="Tensor parallelism size.",
                          option_cls=option_cls)(command)
        command = _option(
            "--max_seq_len",
            type=int,
            default=MAX_SEQ_LEN_DEFAULT,
            help="Maximum total length of one request, including prompt and outputs. "
            "If unspecified, the value is deduced from the model config.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--max_num_tokens",
            type=int,
            default=MAX_NUM_TOKENS_DEFAULT,
            help="Maximum number of batched input tokens after padding is removed in each batch.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--max_batch_size",
            type=int,
            default=MAX_BATCH_SIZE_DEFAULT,
            help="Maximum number of requests that the engine can schedule.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--max_beam_width",
            type=int,
            default=MAX_BEAM_WIDTH_DEFAULT,
            help="Maximum number of beams for beam search decoding.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--log_level",
            type=click.Choice(LOG_LEVELS),
            default="info",
            help="The logging level.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--backend",
            type=click.Choice(EVAL_BACKEND_CHOICES),
            default="pytorch",
            help="The backend to use for evaluation. Default is pytorch backend.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--custom_tokenizer",
            type=str,
            default=None,
            help="Custom tokenizer type: alias (e.g., 'deepseek_v32') or Python import path "
            "(e.g., 'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer'). [Experimental]",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--tokenizer",
            type=str,
            default=None,
            help="Path | Name of the tokenizer."
            "Specify this value only if using TensorRT engine as model.",
            option_cls=option_cls,
        )(command)
        command = _option("--model",
                          required=True,
                          type=str,
                          help="model name | HF checkpoint path | TensorRT engine path",
                          option_cls=option_cls)(command)
        return command

    return decorator
