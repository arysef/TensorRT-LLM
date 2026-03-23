from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
SERVE_BACKEND_CHOICES = ("pytorch", "tensorrt", "_autodeploy")
SERVE_BACKEND_ALIASES = {"trt": "tensorrt"}
KV_CACHE_DTYPE_CHOICES = ("auto", "fp8", "nvfp4")
REASONING_PARSER_CHOICES = ("deepseek-r1", "qwen3", "nano-v3")
TOOL_PARSER_CHOICES = (
    "qwen3",
    "qwen3_coder",
    "kimi_k2",
    "deepseek_v3",
    "deepseek_v31",
    "deepseek_v32",
    "glm4",
)
HOST_DEFAULT = "localhost"
PORT_DEFAULT = 8000
MAX_BEAM_WIDTH_DEFAULT = 1
MAX_BATCH_SIZE_DEFAULT = 2048
MAX_NUM_TOKENS_DEFAULT = 8192
MAX_NUM_TOKENS_MM_EMBEDDING_DEFAULT = 16384
FREE_GPU_MEMORY_FRACTION_DEFAULT = 0.9
SERVE_SHORT_HELP = "Serve a model with the OpenAI-compatible API."
SERVE_HELP = """Running an OpenAI API compatible server

MODEL: model name | HF checkpoint path | TensorRT engine path
"""
MM_EMBEDDING_SERVE_SHORT_HELP = "Serve multimodal embedding models."
MM_EMBEDDING_SERVE_HELP = """Running an OpenAI API compatible server

MODEL: model name | HF checkpoint path | TensorRT engine path
"""
DISAGGREGATED_SHORT_HELP = "Launch a disaggregated serving stack."
DISAGGREGATED_HELP = "Running server in disaggregated mode"
DISAGGREGATED_MPI_WORKER_SHORT_HELP = "Start a disaggregated MPI worker."
DISAGGREGATED_MPI_WORKER_HELP = "Launching disaggregated MPI worker"


def help_info_with_stability_tag(
        help_str: str, tag: str) -> str:
    return f":tag:`{tag}` {help_str}"


class ChoiceWithAlias(click.Choice):

    def __init__(self,
                 choices: Sequence[str],
                 aliases: Mapping[str, str],
                 case_sensitive: bool = True) -> None:
        super().__init__(choices, case_sensitive)
        self.aliases = aliases

    def to_info_dict(self) -> dict[str, Any]:
        info_dict = super().to_info_dict()
        info_dict["aliases"] = self.aliases
        return info_dict

    def convert(self, value: Any, param: click.Parameter | None,
                ctx: click.Context | None) -> Any:
        if value in self.aliases:
            value = self.aliases[value]
        return super().convert(value, param, ctx)


def _option(*param_decls, **kwargs) -> Callable:
    return click.option(*param_decls, **kwargs)


def add_serve_options(*, model_required: bool) -> Callable:

    def decorator(command: Callable) -> Callable:
        command = _option(
            "--extra_visual_gen_options",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Path to a YAML file with extra VISUAL_GEN model options.",
                "prototype"),
        )(command)
        command = _option(
            "--served_model_name",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "The model name used in the API. If not specified, the model path is "
                "used as the model name. This is useful when the model path is long or "
                "when you want to expose a custom name to clients.",
                "prototype"),
        )(command)
        command = _option(
            "--grpc",
            is_flag=True,
            default=False,
            help="Run gRPC server instead of OpenAI HTTP server. "
            "gRPC server accepts pre-tokenized requests and returns raw token IDs.",
        )(command)
        command = _option(
            "--chat_template",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Specify a custom chat template. "
                "Can be a file path or one-liner template string",
                "prototype"),
        )(command)
        command = _option(
            "--video_pruning_rate",
            type=float,
            default=None,
            help=help_info_with_stability_tag(
                "Pruning rate for video frames in multimodal models. "
                "Applied by Efficient Video Sampling (EVS). "
                "None disables EVS, values in [0, 1) enable pruning.",
                "prototype"),
        )(command)
        command = _option(
            "--media_io_kwargs",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Keyword arguments for media I/O.", "prototype"),
        )(command)
        command = _option(
            "--enable_attention_dp",
            is_flag=True,
            default=False,
            help=help_info_with_stability_tag(
                "Enable attention data parallel.", "beta"),
        )(command)
        command = _option(
            "--enable_chunked_prefill",
            is_flag=True,
            default=False,
            help=help_info_with_stability_tag(
                "Enable chunked prefill", "prototype"),
        )(command)
        command = _option(
            "--disagg_cluster_uri",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "URI of the disaggregated cluster.", "prototype"),
        )(command)
        command = _option(
            "--otlp_traces_endpoint",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Target URL to which OpenTelemetry traces will be sent.",
                "prototype"),
        )(command)
        command = _option(
            "--fail_fast_on_attention_window_too_large",
            is_flag=True,
            default=True,
            help=help_info_with_stability_tag(
                "[Deprecated] Exit with runtime error when attention window is too large "
                "to fit even a single sequence in the KV cache. Now defaults to True. "
                "This flag only affects the TRT backend and will be removed in a future release.",
                "deprecated"),
        )(command)
        command = _option(
            "--server_role",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Server role for disaggregated serving. "
                "CONTEXT=prefill (prompt processing), GENERATION=decode (token generation), "
                "MM_ENCODER=multimodal encoder, VISUAL_GEN=visual generation. "
                "Required when using service registry.",
                "prototype"),
        )(command)
        command = _option(
            "--metadata_server_config_file",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Path to metadata server config file", "prototype"),
        )(command)
        command = _option(
            "--tool_parser",
            type=click.Choice(TOOL_PARSER_CHOICES),
            default=None,
            help=help_info_with_stability_tag(
                "Specify the parser for tool models.", "prototype"),
        )(command)
        command = _option(
            "--reasoning_parser",
            type=click.Choice(REASONING_PARSER_CHOICES),
            default=None,
            help=help_info_with_stability_tag(
                "Specify the parser for reasoning models.", "prototype"),
        )(command)
        command = _option(
            "--config",
            "--extra_llm_api_options",
            "extra_llm_api_options",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Path to a YAML file that overwrites the parameters specified by trtllm-serve. "
                "Can be specified as either --config or --extra_llm_api_options.",
                "prototype"),
        )(command)
        command = _option(
            "--hf_revision",
            "--revision",
            "revision",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "The revision to use for the HuggingFace model "
                "(branch name, tag name, or commit id). "
                "Prefer --hf_revision over --revision.",
                "beta"),
        )(command)
        command = _option(
            "--trust_remote_code",
            is_flag=True,
            default=False,
            help=help_info_with_stability_tag(
                "Flag for HF transformers.", "beta"),
        )(command)
        command = _option(
            "--num_postprocess_workers",
            type=int,
            default=0,
            help=help_info_with_stability_tag(
                "Number of workers to postprocess raw responses "
                "to comply with OpenAI protocol.",
                "prototype"),
        )(command)
        command = _option(
            "--kv_cache_dtype",
            type=click.Choice(KV_CACHE_DTYPE_CHOICES),
            default="auto",
            help=help_info_with_stability_tag(
                "KV cache quantization dtype for PyTorch backend. "
                "'auto' uses checkpoint/model metadata; explicit values force override.",
                "prototype"),
        )(command)
        command = _option(
            "--free_gpu_memory_fraction",
            "--kv_cache_free_gpu_memory_fraction",
            type=float,
            default=FREE_GPU_MEMORY_FRACTION_DEFAULT,
            help=help_info_with_stability_tag(
                "Free GPU memory fraction reserved for KV Cache, "
                "after allocating model weights and buffers.",
                "beta"),
        )(command)
        command = _option(
            "--gpus_per_node",
            type=int,
            default=None,
            help=help_info_with_stability_tag(
                "Number of GPUs per node. Default to None, and it will be detected automatically.",
                "beta"),
        )(command)
        command = _option(
            "--moe_cluster_parallel_size",
            "--cluster_size",
            type=int,
            default=None,
            help=help_info_with_stability_tag(
                "[Deprecated] Expert cluster parallelism size. "
                "This option is no longer supported and will be removed in a future release.",
                "deprecated"),
        )(command)
        command = _option(
            "--moe_expert_parallel_size",
            "--ep_size",
            type=int,
            default=None,
            help=help_info_with_stability_tag(
                "expert parallelism size", "beta"),
        )(command)
        command = _option(
            "--context_parallel_size",
            "--cp_size",
            type=int,
            default=1,
            help=help_info_with_stability_tag(
                "Context parallelism size.", "beta"),
        )(command)
        command = _option(
            "--pipeline_parallel_size",
            "--pp_size",
            type=int,
            default=1,
            help=help_info_with_stability_tag(
                "Pipeline parallelism size.", "beta"),
        )(command)
        command = _option(
            "--tensor_parallel_size",
            "--tp_size",
            type=int,
            default=1,
            help=help_info_with_stability_tag(
                "Tensor parallelism size.", "beta"),
        )(command)
        command = _option(
            "--max_seq_len",
            type=int,
            default=None,
            help=help_info_with_stability_tag(
                "Maximum total length of one request, including prompt and outputs. "
                "If unspecified, the value is deduced from the model config.",
                "beta"),
        )(command)
        command = _option(
            "--max_num_tokens",
            type=int,
            default=MAX_NUM_TOKENS_DEFAULT,
            help=help_info_with_stability_tag(
                "Maximum number of batched input tokens after padding is removed in each batch.",
                "beta"),
        )(command)
        command = _option(
            "--max_batch_size",
            type=int,
            default=MAX_BATCH_SIZE_DEFAULT,
            help=help_info_with_stability_tag(
                "Maximum number of requests that the engine can schedule.",
                "beta"),
        )(command)
        command = _option(
            "--max_beam_width",
            type=int,
            default=MAX_BEAM_WIDTH_DEFAULT,
            help=help_info_with_stability_tag(
                "Maximum number of beams for beam search decoding.", "beta"),
        )(command)
        command = _option(
            "--log_level",
            type=click.Choice(LOG_LEVELS),
            default="info",
            help=help_info_with_stability_tag(
                "The logging level.", "beta"),
        )(command)
        command = _option(
            "--custom_module_dirs",
            type=click.Path(exists=True,
                            readable=True,
                            path_type=Path,
                            resolve_path=True),
            default=None,
            multiple=True,
            help=help_info_with_stability_tag(
                "Paths to custom module directories to import.", "prototype"),
        )(command)
        command = _option(
            "--backend",
            type=ChoiceWithAlias(SERVE_BACKEND_CHOICES,
                                 SERVE_BACKEND_ALIASES),
            default="pytorch",
            help=help_info_with_stability_tag(
                "The backend to use to serve the model. Default is pytorch backend.",
                "beta"),
        )(command)
        command = _option(
            "--port",
            type=int,
            default=PORT_DEFAULT,
            help=help_info_with_stability_tag(
                "Port of the server.", "beta"),
        )(command)
        command = _option(
            "--host",
            type=str,
            default=HOST_DEFAULT,
            help=help_info_with_stability_tag(
                "Hostname of the server.", "beta"),
        )(command)
        command = _option(
            "--custom_tokenizer",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Custom tokenizer type: alias (e.g., 'deepseek_v32') or Python import path "
                "(e.g., 'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer').",
                "prototype"),
        )(command)
        command = _option(
            "--tokenizer",
            type=str,
            default=None,
            help=help_info_with_stability_tag(
                "Path or name of the tokenizer. When using the PyTorch backend, "
                "this replaces the default HuggingFace tokenizer.",
                "beta"),
        )(command)
        command = click.argument("model", required=model_required, type=str)(command)
        return command

    return decorator


def add_mm_embedding_serve_options(*, model_required: bool) -> Callable:

    def decorator(command: Callable) -> Callable:
        command = _option(
            "--metadata_server_config_file",
            type=str,
            default=None,
            help="Path to metadata server config file",
        )(command)
        command = _option(
            "--tensor_parallel_size",
            "--tp_size",
            type=int,
            default=1,
            help="Tensor parallelism size.",
        )(command)
        command = _option(
            "--free_gpu_memory_fraction",
            type=float,
            default=FREE_GPU_MEMORY_FRACTION_DEFAULT,
            help="Free GPU memory fraction reserved for KV Cache, "
            "after allocating model weights and buffers.",
        )(command)
        command = _option(
            "--hf_revision",
            "--revision",
            "revision",
            type=str,
            default=None,
            help="The revision to use for the HuggingFace model "
            "(branch name, tag name, or commit id).",
        )(command)
        command = _option(
            "--config",
            "--extra_encoder_options",
            "extra_encoder_options",
            type=str,
            default=None,
            help="Path to a YAML file that overwrites the parameters specified by trtllm-serve. "
            "Prefer --config over --extra_encoder_options.",
        )(command)
        command = _option(
            "--trust_remote_code",
            is_flag=True,
            default=False,
            help="Flag for HF transformers.",
        )(command)
        command = _option(
            "--gpus_per_node",
            type=int,
            default=None,
            help="Number of GPUs per node. Default to None, and it will be "
            "detected automatically.",
        )(command)
        command = _option(
            "--max_num_tokens",
            type=int,
            default=MAX_NUM_TOKENS_MM_EMBEDDING_DEFAULT,
            help="Maximum number of batched input tokens after padding is removed in each batch.",
        )(command)
        command = _option(
            "--max_batch_size",
            type=int,
            default=MAX_BATCH_SIZE_DEFAULT,
            help="Maximum number of requests that the engine can schedule.",
        )(command)
        command = _option(
            "--log_level",
            type=click.Choice(LOG_LEVELS),
            default="info",
            help="The logging level.",
        )(command)
        command = _option(
            "--port",
            type=int,
            default=PORT_DEFAULT,
            help="Port of the server.",
        )(command)
        command = _option(
            "--host",
            type=str,
            default=HOST_DEFAULT,
            help="Hostname of the server.",
        )(command)
        command = click.argument("model", required=model_required, type=str)(command)
        return command

    return decorator


def add_disaggregated_options() -> Callable:

    def decorator(command: Callable) -> Callable:
        command = _option(
            "--metrics-log-interval",
            type=int,
            default=0,
            help="[Deprecated] The interval of logging metrics in seconds. "
            "This option is not connected to any functionality and will be removed in a future release.",
        )(command)
        command = _option(
            "-l",
            "--log_level",
            type=click.Choice(LOG_LEVELS),
            default="info",
            help="The logging level.",
        )(command)
        command = _option(
            "-r",
            "--request_timeout",
            type=int,
            default=180,
            help="Request timeout",
        )(command)
        command = _option(
            "-t",
            "--server_start_timeout",
            type=int,
            default=180,
            help="Server start timeout",
        )(command)
        command = _option(
            "-m",
            "--metadata_server_config_file",
            type=str,
            default=None,
            help="Path to metadata server config file",
        )(command)
        command = _option(
            "-c",
            "--config",
            "--config_file",
            "config_file",
            type=str,
            default=None,
            help="Path to the disaggregated serving configuration YAML file.",
        )(command)
        return command

    return decorator


def add_disaggregated_mpi_worker_options() -> Callable:

    def decorator(command: Callable) -> Callable:
        command = _option(
            "--log_level",
            type=click.Choice(LOG_LEVELS),
            default="info",
            help="The logging level.",
        )(command)
        command = _option(
            "-c",
            "--config",
            "--config_file",
            "config_file",
            type=str,
            default=None,
            help="Path to the disaggregated serving configuration YAML file.",
        )(command)
        return command

    return decorator
