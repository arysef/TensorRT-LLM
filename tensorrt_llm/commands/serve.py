from __future__ import annotations

import asyncio
import gc
import json
import os
import signal
import socket
import subprocess  # nosec B404
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import click
import yaml

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

from trtllm_cli._serve_metadata import (
    add_disaggregated_mpi_worker_options,
    add_disaggregated_options,
    add_mm_embedding_serve_options,
    add_serve_options,
)

# Global variable to store the Popen object of the child process
_child_p_global: Optional[subprocess.Popen] = None


@cache
def _runtime() -> SimpleNamespace:
    import torch as torch_module
    from torch.cuda import device_count as torch_device_count

    from tensorrt_llm import LLM as PyTorchLLM, MultimodalEncoder
    from tensorrt_llm._tensorrt_engine import LLM
    from tensorrt_llm._torch.visual_gen.config import VisualGenArgs
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.commands.utils import get_is_diffusion_model
    from tensorrt_llm.executor.utils import LlmLauncherEnvs
    from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
    from tensorrt_llm.llmapi import (BuildConfig, CapacitySchedulerPolicy,
                                     DynamicBatchConfig, KvCacheConfig,
                                     SchedulerConfig, VisualGen)
    from tensorrt_llm.llmapi.disagg_utils import (
        DisaggClusterConfig, MetadataServerConfig, ServerRole,
        extract_disagg_cluster_config, parse_disagg_config_file,
        parse_metadata_server_config_file)
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs, TrtLlmArgs
    from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict
    from tensorrt_llm.llmapi.mpi_session import find_free_ipc_addr
    from tensorrt_llm.logger import logger
    from tensorrt_llm.mapping import CpType
    from tensorrt_llm.serve import OpenAIDisaggServer, OpenAIServer
    from tensorrt_llm.tools.importlib_utils import import_custom_module_from_dir

    return SimpleNamespace(
        torch=torch_module,
        device_count=torch_device_count,
        PyTorchLLM=PyTorchLLM,
        MultimodalEncoder=MultimodalEncoder,
        LLM=LLM,
        VisualGenArgs=VisualGenArgs,
        mpi_rank=mpi_rank,
        get_is_diffusion_model=get_is_diffusion_model,
        LlmLauncherEnvs=LlmLauncherEnvs,
        MultimodalServerConfig=MultimodalServerConfig,
        BuildConfig=BuildConfig,
        CapacitySchedulerPolicy=CapacitySchedulerPolicy,
        DynamicBatchConfig=DynamicBatchConfig,
        KvCacheConfig=KvCacheConfig,
        SchedulerConfig=SchedulerConfig,
        VisualGen=VisualGen,
        DisaggClusterConfig=DisaggClusterConfig,
        MetadataServerConfig=MetadataServerConfig,
        ServerRole=ServerRole,
        extract_disagg_cluster_config=extract_disagg_cluster_config,
        parse_disagg_config_file=parse_disagg_config_file,
        parse_metadata_server_config_file=parse_metadata_server_config_file,
        TorchLlmArgs=TorchLlmArgs,
        TrtLlmArgs=TrtLlmArgs,
        update_llm_args_with_extra_dict=update_llm_args_with_extra_dict,
        find_free_ipc_addr=find_free_ipc_addr,
        logger=logger,
        CpType=CpType,
        OpenAIDisaggServer=OpenAIDisaggServer,
        OpenAIServer=OpenAIServer,
        import_custom_module_from_dir=import_custom_module_from_dir,
    )


class _AttrProxy:

    def __init__(self, name: str):
        self._name = name

    def _target(self):
        return getattr(_runtime(), self._name)

    def __call__(self, *args, **kwargs):
        return self._target()(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._target(), attr)

    def __getitem__(self, key):
        return self._target()[key]

    def __iter__(self):
        return iter(self._target())


torch = _AttrProxy("torch")
device_count = _AttrProxy("device_count")
PyTorchLLM = _AttrProxy("PyTorchLLM")
MultimodalEncoder = _AttrProxy("MultimodalEncoder")
LLM = _AttrProxy("LLM")
VisualGenArgs = _AttrProxy("VisualGenArgs")
mpi_rank = _AttrProxy("mpi_rank")
get_is_diffusion_model = _AttrProxy("get_is_diffusion_model")
LlmLauncherEnvs = _AttrProxy("LlmLauncherEnvs")
MultimodalServerConfig = _AttrProxy("MultimodalServerConfig")
BuildConfig = _AttrProxy("BuildConfig")
CapacitySchedulerPolicy = _AttrProxy("CapacitySchedulerPolicy")
DynamicBatchConfig = _AttrProxy("DynamicBatchConfig")
KvCacheConfig = _AttrProxy("KvCacheConfig")
SchedulerConfig = _AttrProxy("SchedulerConfig")
VisualGen = _AttrProxy("VisualGen")
DisaggClusterConfig = _AttrProxy("DisaggClusterConfig")
MetadataServerConfig = _AttrProxy("MetadataServerConfig")
ServerRole = _AttrProxy("ServerRole")
extract_disagg_cluster_config = _AttrProxy("extract_disagg_cluster_config")
parse_disagg_config_file = _AttrProxy("parse_disagg_config_file")
parse_metadata_server_config_file = _AttrProxy(
    "parse_metadata_server_config_file")
TorchLlmArgs = _AttrProxy("TorchLlmArgs")
TrtLlmArgs = _AttrProxy("TrtLlmArgs")
update_llm_args_with_extra_dict = _AttrProxy("update_llm_args_with_extra_dict")
find_free_ipc_addr = _AttrProxy("find_free_ipc_addr")
logger = _AttrProxy("logger")
CpType = _AttrProxy("CpType")
OpenAIDisaggServer = _AttrProxy("OpenAIDisaggServer")
OpenAIServer = _AttrProxy("OpenAIServer")
import_custom_module_from_dir = _AttrProxy("import_custom_module_from_dir")


def _signal_handler_cleanup_child(signum, frame):
    """Signal handler to clean up the child process."""
    global _child_p_global
    if _child_p_global and _child_p_global.poll() is None:
        logger.info(
            f"Parent process (PID {os.getpid()}) received signal {signal.Signals(signum).name}. Terminating child process (PID {_child_p_global.pid})."
        )
        _child_p_global.terminate()
        try:
            _child_p_global.wait(
                timeout=10)  # Allow 10 seconds for graceful termination
        except subprocess.TimeoutExpired:
            logger.info(
                f"Child process (PID {_child_p_global.pid}) did not terminate gracefully after signal. Killing."
            )
            _child_p_global.kill()
            try:
                _child_p_global.wait(timeout=10)  # Allow 10 seconds for kill
            except subprocess.TimeoutExpired:
                logger.info(
                    f"Child process (PID {_child_p_global.pid}) failed to die even after kill command from signal handler."
                )

        if _child_p_global.poll() is not None:
            logger.info(
                f"Child process (PID {_child_p_global.pid}) confirmed terminated due to signal {signal.Signals(signum).name}."
            )
        else:
            logger.info(
                f"Child process (PID {_child_p_global.pid}) is still running after cleanup attempt for signal {signal.Signals(signum).name}."
            )

    # Standard exit code for signal termination
    sys.exit(128 + signum)


def is_non_default_or_required(param_name, value, backend):
    """
    Check if a parameter should be explicitly included in llm_args.

    Returns True if parameter is either:
    1. Always required (core params that must be present), OR
    2. Different from its default value in the backend's LlmArgs class
    """
    always_include = {
        "model", "backend", "tokenizer", "custom_tokenizer",
        "postprocess_tokenizer_dir"
    }

    if param_name in always_include:
        return True

    if value is None:
        return False

    if backend == "tensorrt":
        llm_args_class = TrtLlmArgs
    elif backend == "_autodeploy":
        from tensorrt_llm._torch.auto_deploy.llm_args import \
            LlmArgs as AutoDeployLlmArgs
        llm_args_class = AutoDeployLlmArgs
    else:
        llm_args_class = TorchLlmArgs

    field_info = llm_args_class.model_fields.get(param_name)
    if not field_info:
        return False

    default = field_info.default
    if callable(default):
        default = default()

    return value != default


def get_llm_args(
        model: str,
        tokenizer: Optional[str] = None,
        custom_tokenizer: Optional[str] = None,
        backend: str = "pytorch",
        max_beam_width: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        max_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        context_parallel_size: int = 1,
        cp_config: Optional[dict] = None,
        moe_expert_parallel_size: Optional[int] = None,
        gpus_per_node: Optional[int] = None,
        free_gpu_memory_fraction: float = 0.9,
        kv_cache_dtype: str = "auto",
        num_postprocess_workers: int = 0,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        reasoning_parser: Optional[str] = None,
        fail_fast_on_attention_window_too_large: bool = True,
        otlp_traces_endpoint: Optional[str] = None,
        enable_chunked_prefill: bool = False,
        enable_attention_dp: bool = False,
        video_pruning_rate: Optional[float] = None,
        **llm_args_extra_dict: Any):
    build_defaults = BuildConfig.model_fields
    if max_beam_width is None:
        max_beam_width = build_defaults["max_beam_width"].default
    if max_batch_size is None:
        max_batch_size = build_defaults["max_batch_size"].default
    if max_num_tokens is None:
        max_num_tokens = build_defaults["max_num_tokens"].default
    if max_seq_len is None:
        max_seq_len = build_defaults["max_seq_len"].default

    if gpus_per_node is None:
        gpus_per_node = device_count()
        if gpus_per_node == 0:
            raise ValueError("No GPU devices found on the node")

    # TODO: This manual cp_type conversion can be removed once cp_config
    # is refactored to a typed Pydantic model with enum coercion
    if cp_config is not None and "cp_type" in cp_config:
        cp_config = cp_config.copy()
        try:
            cp_config["cp_type"] = CpType[cp_config["cp_type"].upper()]
        except KeyError:
            raise ValueError(f"Invalid cp_type: {cp_config['cp_type']}. " \
                             f"Must be one of: {', '.join([t.name for t in CpType])}")

    kv_cache_default_fraction = KvCacheConfig.model_fields[
        'free_gpu_memory_fraction'].default

    cli_maybe_overrides = {
        "model":
        model,
        "backend":
        backend,
        "tokenizer":
        tokenizer,
        "custom_tokenizer":
        custom_tokenizer,
        "postprocess_tokenizer_dir":
        tokenizer or model,
        "kv_cache_config":
        KvCacheConfig(free_gpu_memory_fraction=free_gpu_memory_fraction,
                      dtype=kv_cache_dtype) if free_gpu_memory_fraction
        != kv_cache_default_fraction or kv_cache_dtype != "auto" else None,
        "cp_config":
        cp_config,
        "build_config":
        BuildConfig(max_batch_size=max_batch_size,
                    max_num_tokens=max_num_tokens,
                    max_beam_width=max_beam_width,
                    max_seq_len=max_seq_len) if backend == "tensorrt" else None,
        "scheduler_config":
        SchedulerConfig(capacity_scheduler_policy=CapacitySchedulerPolicy.
                        GUARANTEED_NO_EVICT,
                        dynamic_batch_config=DynamicBatchConfig(
                            enable_batch_size_tuning=True,
                            enable_max_num_tokens_tuning=False,
                            dynamic_batch_moving_average_window=128))
        if backend == "tensorrt" else None,
        "max_batch_size":
        max_batch_size,
        "max_beam_width":
        max_beam_width,
        "tensor_parallel_size":
        tensor_parallel_size,
        "pipeline_parallel_size":
        pipeline_parallel_size,
        "context_parallel_size":
        context_parallel_size,
        "moe_expert_parallel_size":
        moe_expert_parallel_size,
        "gpus_per_node":
        gpus_per_node,
        "trust_remote_code":
        trust_remote_code,
        "max_num_tokens":
        max_num_tokens,
        "max_seq_len":
        max_seq_len,
        "num_postprocess_workers":
        num_postprocess_workers,
        "enable_chunked_prefill":
        enable_chunked_prefill,
        "enable_attention_dp":
        enable_attention_dp,
        "revision":
        revision,
        "reasoning_parser":
        reasoning_parser,
        "otlp_traces_endpoint":
        otlp_traces_endpoint,
        "fail_fast_on_attention_window_too_large":
        fail_fast_on_attention_window_too_large,
        "video_pruning_rate":
        video_pruning_rate,
    }

    llm_args = {
        param: value
        for param, value in cli_maybe_overrides.items()
        if is_non_default_or_required(param, value, backend)
    }

    return llm_args, llm_args_extra_dict


def launch_server(
        host: str,
        port: int,
        llm_args: dict,
        tool_parser: Optional[str] = None,
        chat_template: Optional[str] = None,
        metadata_server_cfg: Optional[MetadataServerConfig] = None,
        server_role: Optional[ServerRole] = None,
        disagg_cluster_config: Optional[DisaggClusterConfig] = None,
        multimodal_server_config: Optional[MultimodalServerConfig] = None,
        served_model_name: Optional[str] = None):

    backend = llm_args["backend"]
    model = served_model_name or llm_args["model"]
    addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC,
                                   socket.SOCK_STREAM)
    address_family = socket.AF_INET6 if all(
        [info[0] == socket.AF_INET6 for info in addr_info]) else socket.AF_INET
    with socket.socket(address_family, socket.SOCK_STREAM) as s:
        # If disagg cluster config is provided and port is not specified, try to find a free port, otherwise try to bind to the specified port
        assert port > 0 or disagg_cluster_config is not None, "Port must be specified if disagg cluster config is not provided"
        try:
            s.bind((host, port))
            if port == 0:
                port = s.getsockname()[1]
        except OSError as e:
            raise RuntimeError(f"Failed to bind socket to {host}:{port}: {e}")

        if backend == 'pytorch':
            llm_args.pop("build_config", None)
            llm = PyTorchLLM(**llm_args)
        elif backend == '_autodeploy':
            from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM

            # AutoDeploy does not support build_config
            llm_args.pop("build_config", None)
            llm = AutoDeployLLM(**llm_args)
        elif backend == 'tensorrt' or backend == 'trt':
            llm_args.pop("backend")
            llm = LLM(**llm_args)
        else:
            raise click.BadParameter(
                f"{backend} is not a known backend, check help for available options.",
                param_hint="backend")

        server = OpenAIServer(generator=llm,
                              model=model,
                              tool_parser=tool_parser,
                              server_role=server_role,
                              metadata_server_cfg=metadata_server_cfg,
                              disagg_cluster_config=disagg_cluster_config,
                              multimodal_server_config=multimodal_server_config,
                              chat_template=chat_template)

        # Optionally disable GC (default: not disabled)
        if os.getenv("TRTLLM_SERVER_DISABLE_GC", "0") == "1":
            gc.disable()

        asyncio.run(server(host, port, sockets=[s]))


def launch_grpc_server(host: str,
                       port: int,
                       llm_args: dict,
                       served_model_name: Optional[str] = None):
    """
    Launch a gRPC server for TensorRT-LLM.

    This provides a high-performance gRPC interface designed for external routers
    (e.g., sgl-router) using pre-tokenized input and raw token ID output.

    Args:
        host: Host to bind to
        port: Port to bind to
        llm_args: Arguments for LLM initialization (from get_llm_args)
        served_model_name: Custom model name for API responses (defaults to model path)
    """
    import grpc

    try:
        from grpc_reflection.v1alpha import reflection
        REFLECTION_AVAILABLE = True
    except ImportError:
        REFLECTION_AVAILABLE = False

    from tensorrt_llm.grpc import trtllm_service_pb2, trtllm_service_pb2_grpc
    from tensorrt_llm.grpc.grpc_request_manager import GrpcRequestManager
    from tensorrt_llm.grpc.grpc_servicer import TrtllmServiceServicer

    async def serve_grpc_async():
        logger.info("Initializing TensorRT-LLM gRPC server...")

        backend = llm_args.get("backend")
        model_path = served_model_name or llm_args.get("model", "")

        if backend == "pytorch":
            llm_args.pop("build_config", None)
            llm = PyTorchLLM(**llm_args)
        elif backend == "_autodeploy":
            from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
            llm_args.pop("build_config", None)
            llm = AutoDeployLLM(**llm_args)
        elif backend == "tensorrt" or backend == "trt":
            llm_args.pop("backend")
            llm = LLM(**llm_args)
        else:
            raise click.BadParameter(
                f"{backend} is not a known backend, check help for available options.",
                param_hint="backend")

        logger.info("Model loaded successfully")

        # Create request manager
        request_manager = GrpcRequestManager(llm)

        # Create servicer
        servicer = TrtllmServiceServicer(request_manager, model_path=model_path)

        # Create gRPC server
        server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", -1),  # Unlimited
                ("grpc.max_receive_message_length", -1),  # Unlimited
                ("grpc.keepalive_time_ms", 30000),  # 30s keepalive
                ("grpc.keepalive_timeout_ms", 10000),  # 10s timeout
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
            ], )

        # Add servicer to server
        trtllm_service_pb2_grpc.add_TrtllmServiceServicer_to_server(
            servicer, server)

        # Enable reflection for grpcurl and other tools
        if REFLECTION_AVAILABLE:
            service_names = (
                trtllm_service_pb2.DESCRIPTOR.services_by_name["TrtllmService"].
                full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(service_names, server)
            logger.info("gRPC reflection enabled")

        # Bind to address
        address = f"{host}:{port}"
        server.add_insecure_port(address)

        # Start server
        await server.start()
        logger.info(f"TensorRT-LLM gRPC server started on {address}")
        logger.info("Server is ready to accept requests")

        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def signal_handler():
            logger.info("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        # Serve until shutdown signal
        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            logger.info("Shutting down TensorRT-LLM gRPC server...")

            # Stop gRPC server
            await server.stop(grace=5.0)
            logger.info("gRPC server stopped")

            # Shutdown LLM
            if hasattr(llm, "shutdown"):
                llm.shutdown()
            logger.info("LLM engine stopped")

            logger.info("Shutdown complete")

    asyncio.run(serve_grpc_async())


def launch_mm_encoder_server(
    host: str,
    port: int,
    encoder_args: dict,
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
):
    model = encoder_args["model"]
    encoder_args.pop("build_config", None)
    mm_encoder = MultimodalEncoder(**encoder_args)

    server = OpenAIServer(generator=mm_encoder,
                          model=model,
                          server_role=ServerRole.MM_ENCODER,
                          metadata_server_cfg=metadata_server_cfg,
                          tool_parser=None)
    asyncio.run(server(host, port))


def launch_visual_gen_server(
    host: str,
    port: int,
    model: str,
    diffusion_args: Optional[VisualGenArgs] = None,
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
):
    """Launch a VISUAL_GEN model server for image/video generation.

    Args:
        host: Server hostname.
        port: Server port.
        model: Model path or HuggingFace Hub model ID.
        diffusion_args: Optional validated VisualGenArgs for model configuration.
        metadata_server_cfg: Optional metadata server configuration.
    """
    logger.info(f"Initializing VisualGen ({model})")

    visual_gen_model = VisualGen(model_path=model,
                                 diffusion_args=diffusion_args)

    n_workers = visual_gen_model.diffusion_args.parallel.n_workers
    logger.info(f"World size: {n_workers}")
    logger.info(
        f"CFG size: {visual_gen_model.diffusion_args.parallel.dit_cfg_size}")
    logger.info(
        f"Ulysses size: {visual_gen_model.diffusion_args.parallel.dit_ulysses_size}"
    )

    server = OpenAIServer(generator=visual_gen_model,
                          model=model,
                          server_role=ServerRole.VISUAL_GEN,
                          metadata_server_cfg=metadata_server_cfg,
                          tool_parser=None)
    asyncio.run(server(host, port))


@click.command("serve")
@add_serve_options(model_required=True)
def serve(
        model: str, tokenizer: Optional[str], custom_tokenizer: Optional[str],
        host: str, port: int, log_level: str, backend: str, max_beam_width: int,
        max_batch_size: int, max_num_tokens: int, max_seq_len: int,
        tensor_parallel_size: int, pipeline_parallel_size: int,
        context_parallel_size: int, moe_expert_parallel_size: Optional[int],
        moe_cluster_parallel_size: Optional[int], gpus_per_node: Optional[int],
        free_gpu_memory_fraction: float, kv_cache_dtype: str,
        num_postprocess_workers: int, trust_remote_code: bool,
        revision: Optional[str], extra_llm_api_options: Optional[str],
        reasoning_parser: Optional[str], tool_parser: Optional[str],
        metadata_server_config_file: Optional[str], server_role: Optional[str],
        fail_fast_on_attention_window_too_large: bool,
        otlp_traces_endpoint: Optional[str], enable_chunked_prefill: bool,
        enable_attention_dp: bool, disagg_cluster_uri: Optional[str],
        media_io_kwargs: Optional[str], video_pruning_rate: Optional[float],
        custom_module_dirs: list[Path], chat_template: Optional[str],
        grpc: bool, served_model_name: Optional[str],
        extra_visual_gen_options: Optional[str]):
    """Running an OpenAI API compatible server

    MODEL: model name | HF checkpoint path | TensorRT engine path
    """
    logger.set_level(log_level)

    if moe_cluster_parallel_size is not None:
        logger.warning(
            "--moe_cluster_parallel_size / --cluster_size is deprecated and "
            "no longer supported. This option will be removed in a future release."
        )

    if "--fail_fast_on_attention_window_too_large" in sys.argv:
        logger.warning(
            "--fail_fast_on_attention_window_too_large is deprecated. "
            "It now defaults to True and will be removed in a future release. "
            "This flag only affects the TRT backend.")

    if "--revision" in sys.argv:
        logger.warning("--revision is deprecated, use --hf_revision instead.")

    for custom_module_dir in custom_module_dirs:
        try:
            import_custom_module_from_dir(custom_module_dir)
        except Exception as e:
            logger.error(
                f"Failed to import custom module from {custom_module_dir}: {e}")
            raise e

    def _serve_llm():
        nonlocal server_role
        llm_args, _ = get_llm_args(
            model=model,
            tokenizer=tokenizer,
            custom_tokenizer=custom_tokenizer,
            backend=backend,
            max_beam_width=max_beam_width,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            moe_expert_parallel_size=moe_expert_parallel_size,
            moe_cluster_parallel_size=moe_cluster_parallel_size,
            gpus_per_node=gpus_per_node,
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            kv_cache_dtype=kv_cache_dtype,
            num_postprocess_workers=num_postprocess_workers,
            trust_remote_code=trust_remote_code,
            revision=revision,
            reasoning_parser=reasoning_parser,
            fail_fast_on_attention_window_too_large=
            fail_fast_on_attention_window_too_large,
            otlp_traces_endpoint=otlp_traces_endpoint,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_attention_dp=enable_attention_dp,
            video_pruning_rate=video_pruning_rate)

        llm_args_extra_dict = {}
        if extra_llm_api_options is not None:
            with open(extra_llm_api_options, 'r') as f:
                llm_args_extra_dict = yaml.safe_load(f)
        llm_args = update_llm_args_with_extra_dict(llm_args,
                                                   llm_args_extra_dict)

        metadata_server_cfg = parse_metadata_server_config_file(
            metadata_server_config_file)

        # Specify disagg_cluster_config in config file or through command line "--disagg_cluster_uri",
        # but disagg_cluster_uri takes precedence over cluster uri in config file
        disagg_cluster_config = llm_args.pop("disagg_cluster", None)
        if disagg_cluster_config:
            disagg_cluster_config = extract_disagg_cluster_config(
                disagg_cluster_config, disagg_cluster_uri)
        elif disagg_cluster_uri:
            disagg_cluster_config = DisaggClusterConfig(
                cluster_uri=disagg_cluster_uri)

        if metadata_server_cfg is not None or disagg_cluster_config is not None:
            assert (
                server_role is not None
            ), "server_role is required when metadata_server_cfg or disagg_cluster_config is provided"
            try:
                server_role = ServerRole[server_role.upper()]
            except ValueError:
                raise ValueError(f"Invalid server role: {server_role}. " \
                                f"Must be one of: {', '.join([role.name for role in ServerRole])}")
        # Parse media_io_kwargs from JSON string to dict if provided
        parsed_media_io_kwargs = None
        if media_io_kwargs is not None:
            try:
                parsed_media_io_kwargs = json.loads(media_io_kwargs)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for media_io_kwargs: {e}")

        multimodal_server_config = MultimodalServerConfig(
            media_io_kwargs=parsed_media_io_kwargs)

        if grpc:
            # gRPC mode: launch gRPC server instead of OpenAI HTTP server
            # Check for unsupported arguments that are silently ignored in gRPC mode
            unsupported_args = {
                "tool_parser": tool_parser,
                "chat_template": chat_template,
                "metadata_server_config_file": metadata_server_config_file,
                "server_role": server_role,
                "disagg_cluster_config": disagg_cluster_config,
            }
            for name, value in unsupported_args.items():
                if value is not None:
                    raise ValueError(
                        f"Argument '{name}' is not supported when running in gRPC mode. "
                        f"The gRPC server is designed for use with external routers that handle "
                        f"these features (e.g., tool parsing, chat templates).")
            launch_grpc_server(host,
                               port,
                               llm_args,
                               served_model_name=served_model_name)
        else:
            # Default: launch OpenAI HTTP server
            launch_server(host,
                          port,
                          llm_args,
                          tool_parser,
                          chat_template,
                          metadata_server_cfg,
                          server_role,
                          disagg_cluster_config,
                          multimodal_server_config,
                          served_model_name=served_model_name)

    def _serve_visual_gen():
        extra_args = {}
        if extra_visual_gen_options is not None:
            with open(extra_visual_gen_options, 'r') as f:
                extra_args = yaml.safe_load(f) or {}

        diffusion_args = VisualGenArgs(**extra_args) if extra_args else None

        metadata_server_cfg = parse_metadata_server_config_file(
            metadata_server_config_file)

        launch_visual_gen_server(host, port, model, diffusion_args,
                                 metadata_server_cfg)

    if get_is_diffusion_model(model):
        _serve_visual_gen()
    else:
        _serve_llm()


@click.command("mm_embedding_serve")
@add_mm_embedding_serve_options(model_required=True)
def serve_encoder(model: str, host: str, port: int, log_level: str,
                  max_batch_size: int, max_num_tokens: int,
                  gpus_per_node: Optional[int], trust_remote_code: bool,
                  extra_encoder_options: Optional[str], revision: Optional[str],
                  free_gpu_memory_fraction: float, tensor_parallel_size: int,
                  metadata_server_config_file: Optional[str]):
    """Running an OpenAI API compatible server

    MODEL: model name | HF checkpoint path | TensorRT engine path
    """
    logger.set_level(log_level)

    if "--extra_encoder_options" in sys.argv:
        logger.warning(
            "--extra_encoder_options is deprecated, use --config instead.")

    llm_args, _ = get_llm_args(
        model=model,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        gpus_per_node=gpus_per_node,
        trust_remote_code=trust_remote_code,
        revision=revision,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        tensor_parallel_size=tensor_parallel_size)

    encoder_args_extra_dict = {}
    if extra_encoder_options is not None:
        with open(extra_encoder_options, 'r') as f:
            encoder_args_extra_dict = yaml.safe_load(f)
    encoder_args = update_llm_args_with_extra_dict(llm_args,
                                                   encoder_args_extra_dict)

    metadata_server_cfg = parse_metadata_server_config_file(
        metadata_server_config_file)

    launch_mm_encoder_server(host, port, encoder_args, metadata_server_cfg)


@click.command("disaggregated")
@add_disaggregated_options()
def disaggregated(
    config_file: Optional[str],
    metadata_server_config_file: Optional[str],
    server_start_timeout: int,
    request_timeout: int,
    log_level: str,
    metrics_log_interval: int,
):
    """Running server in disaggregated mode"""

    logger.set_level(log_level)

    if metrics_log_interval != 0:
        logger.warning(
            "--metrics-log-interval is deprecated and not connected to any "
            "functionality. This option will be removed in a future release.")

    if "--config_file" in sys.argv:
        logger.warning("--config_file is deprecated, use --config instead.")

    disagg_cfg = parse_disagg_config_file(config_file)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((disagg_cfg.hostname, disagg_cfg.port))
        except OSError as e:
            raise RuntimeError(
                f"Failed to bind socket to {disagg_cfg.hostname}:{disagg_cfg.port}: {e}"
            )

        metadata_server_cfg = parse_metadata_server_config_file(
            metadata_server_config_file)

        server = OpenAIDisaggServer(
            config=disagg_cfg,
            req_timeout_secs=request_timeout,
            server_start_timeout_secs=server_start_timeout,
            metadata_server_cfg=metadata_server_cfg,
            metrics_interval_secs=metrics_log_interval)

        # Disable GC by default
        #   When concurrency is high, the number of Python objects increases, so
        #   GC runs frequently and takes a long time to process. In this case,
        #   requests are not immediately forwarded to CTX workers and GEN workers,
        #   causing them to run with small batch sizes. Disabling GC can mitigate
        #   this problem.
        #   By testing this feature, we didn't observe significant RSS or VMS
        #   increment, and observed that `count0` (obtained by `gc.get_count()`)
        #   increases by fewer than 1,000 after every 200,000 requests, while the
        #   maximum value of `count0` exceeded 3,000,000 during the test.
        if os.getenv("TRTLLM_DISAGG_SERVER_DISABLE_GC", "1") == "1":
            gc.disable()

        asyncio.run(server(disagg_cfg.hostname, disagg_cfg.port, sockets=[s]))


def set_cuda_device():
    if (os.getenv("OMPI_COMM_WORLD_RANK")):
        env_global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif (os.getenv("SLURM_PROCID")):
        env_global_rank = int(os.environ["SLURM_PROCID"])
    else:
        raise RuntimeError("Could not determine rank from environment")
    device_id = env_global_rank % device_count()
    print(
        f"env_global_rank: {env_global_rank}, set device_id: {device_id} before importing mpi4py"
    )
    torch.cuda.set_device(device_id)


@click.command("disaggregated_mpi_worker")
@add_disaggregated_mpi_worker_options()
def disaggregated_mpi_worker(config_file: Optional[str], log_level: str):
    """Launching disaggregated MPI worker"""

    from tensorrt_llm._utils import mpi_rank
    if os.environ.get(DisaggLauncherEnvs.
                      TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT) != "1":
        set_cuda_device()
    # Importing mpi4py after setting CUDA device. This is needed to work around an issue with mpi4py and CUDA
    from mpi4py.futures import MPICommExecutor

    from tensorrt_llm._utils import global_mpi_rank, mpi_rank, set_mpi_comm
    from tensorrt_llm.llmapi.disagg_utils import split_world_comm

    disagg_cfg = parse_disagg_config_file(config_file)

    # Run a server with the underlying LLM invokes a RemoteMPISessionClient
    if os.environ.get(DisaggLauncherEnvs.
                      TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT) == "1":
        instance_idx = os.environ.get(
            DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX)
        server_cfg = disagg_cfg.server_configs[int(instance_idx)]

        llm_args, llm_args_extra_dict = get_llm_args(**server_cfg.other_args)
        llm_args = update_llm_args_with_extra_dict(llm_args,
                                                   llm_args_extra_dict)

        # Ignore the non-LLM args
        llm_args.pop("router", None)
        _launch_disaggregated_server(config_file, llm_args)
        return

    is_leader, instance_idx, sub_comm = split_world_comm(
        disagg_cfg.server_configs)

    logger.set_level(log_level)
    set_mpi_comm(sub_comm)
    logger.info(
        f"mpi_session is provided for LLM instance. Global MPI rank: {global_mpi_rank()}, sub-comm MPI rank: {mpi_rank()}"
    )

    # Leader ranks will start the trtllm-server using its own server config
    # and start a RemoteMPISessionServer to accept MPI tasks
    if is_leader:
        os.environ[DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX] = str(
            instance_idx)
        server_cfg = disagg_cfg.server_configs[instance_idx]

        llm_args, llm_args_extra_dict = get_llm_args(**server_cfg.other_args)
        llm_args = update_llm_args_with_extra_dict(llm_args,
                                                   llm_args_extra_dict)

        _launch_disaggregated_leader(sub_comm, instance_idx, config_file,
                                     log_level)

    else:
        # Common workers
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(
                    f"rank{global_mpi_rank()} should not have executor")


class DisaggLauncherEnvs(StrEnum):
    TLLM_DISAGG_INSTANCE_IDX = "TLLM_DISAGG_INSTANCE_IDX"
    TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT = "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT"


def _launch_disaggregated_server(disagg_config_file: str, llm_args: dict):
    # Launching the server
    instance_idx = os.environ.get(DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX)
    assert instance_idx is not None, f"{DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX} should be set by the launcher"
    disagg_config = parse_disagg_config_file(disagg_config_file)
    server_cfg = disagg_config.server_configs[int(instance_idx)]

    logger.info(
        f"rank {mpi_rank()} for index {instance_idx} launch the disagg server")

    launch_server(host=server_cfg.hostname,
                  port=server_cfg.port,
                  llm_args=llm_args)


def _launch_disaggregated_leader(sub_comm, instance_idx: int, config_file: str,
                                 log_level: str):
    global _child_p_global  # Declare usage of global variable
    # Assuming logger and mpi_rank are available from module imports or passed in
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.llmapi.mgmn_leader_node import \
        launch_server_main as launch_remote_mpi_session_server
    from tensorrt_llm.llmapi.mpi_session import split_mpi_env

    # This mimics the behavior of trtllm-llmapi-launch
    # TODO: Make the port allocation atomic
    free_ipc_addr = find_free_ipc_addr()
    os.environ[LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS] = "1"
    os.environ[
        LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR.value] = free_ipc_addr
    os.environ[DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT.
               value] = "1"
    os.environ[DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX] = str(instance_idx)

    logger.debug(
        f"proxy controller address: {os.environ[LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR]}"
    )

    # The MPI-related environment variables will invoke duplicate MPI_Init in
    # the forked process, so we need to remove them before launching the server
    # process.
    non_mpi_env, mpi_env = split_mpi_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS in non_mpi_env
    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR in non_mpi_env
    assert DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX in non_mpi_env
    assert DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT in non_mpi_env

    # Two steps:
    # 1. Run the LLM-API Proxy in a separate process for streaming performance.
    #      The Proxy will create a RemoteMpiSessionClient as mpi_session in LLM
    #      class.
    command = [
        "python3", sys.argv[0], "disaggregated_mpi_worker", "-c", config_file,
        "--log_level", log_level
    ]
    logger.info(
        f"rank {mpi_rank()} step1: preparing to launch command: {command}")

    # Store original signal handlers
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    # Register new signal handlers
    signal.signal(signal.SIGTERM, _signal_handler_cleanup_child)
    signal.signal(signal.SIGINT, _signal_handler_cleanup_child)

    try:
        _child_p_global = subprocess.Popen(
            command,
            env=non_mpi_env,
            stdout=sys.stdout,  # Redirect to parent's stdout
            stderr=sys.stderr,  # Redirect to parent's stderr
            start_new_session=True)

        logger.info(
            f"Parent process (PID {os.getpid()}) launched child process (PID {_child_p_global.pid})."
        )

        logger.info(f"rank {mpi_rank()} step2: start the mpi session server")
        # 2. Run the RemoteMpiSessionServer to accept MPI tasks
        assert sub_comm is not None
        assert sub_comm.Get_rank() == 0
        # This is a blocking call
        launch_remote_mpi_session_server(sub_comm)

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        signal.signal(signal.SIGINT, original_sigint_handler)

        if _child_p_global:  # If Popen was successful and object exists
            logger.info(
                f"Parent process (PID {os.getpid()}) in finally block. Cleaning up child process (PID: {_child_p_global.pid})."
            )
            # Check if child is still running
            if _child_p_global.poll() is None:
                _child_p_global.terminate()
                try:
                    _child_p_global.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Child process {_child_p_global.pid} timed out on terminate (30s), killing."
                    )
                    _child_p_global.kill()
                    try:
                        _child_p_global.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        logger.error(
                            f"Child process {_child_p_global.pid} failed to be killed even after 30s."
                        )
            assert _child_p_global.poll(
            ) is not None, f"the subprocess should be terminated"

    # Check if the process was launched and assert it's terminated
    if _child_p_global and hasattr(_child_p_global,
                                   'pid') and _child_p_global.pid is not None:
        final_status = _child_p_global.poll()
        assert final_status is not None, \
            f"The subprocess (PID {_child_p_global.pid}) should be terminated, but its status is {final_status}"
        logger.info(
            f"Subprocess (PID {_child_p_global.pid}) final status: {final_status}"
        )
    elif _child_p_global is None:
        # This implies Popen might have failed or was not reached.
        # If Popen failed, an exception would likely have occurred earlier.
        logger.info(
            "Child process was not assigned to _child_p_global, skipping final termination assertion."
        )


class DefaultGroup(click.Group):
    """Click group that preserves trtllm-serve's default serve subcommand."""

    def _resolve_default_command(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            return "serve", self.commands["serve"], args
        return None

    def resolve_command(self, ctx, args):
        default_command = self._resolve_default_command(ctx, args)
        if default_command is not None:
            return default_command
        return super().resolve_command(ctx, args)

    def invoke(self, ctx):
        def _process_result(value):
            if self._result_callback is not None:
                value = ctx.invoke(self._result_callback, value, **ctx.params)
            return value

        protected_args = getattr(ctx, "_protected_args", ctx.protected_args)
        if not protected_args:
            return super().invoke(ctx)

        args = [*protected_args, *ctx.args]
        ctx.args = []
        if hasattr(ctx, "_protected_args"):
            ctx._protected_args = []
        else:
            ctx.protected_args = []

        with ctx:
            default_command = self._resolve_default_command(ctx, args)
            if default_command is None:
                cmd_name, cmd, args = super().resolve_command(ctx, args)
                ctx.invoked_subcommand = cmd_name
                click.Command.invoke(self, ctx)
                sub_ctx = cmd.make_context(cmd_name, args, parent=ctx)
            else:
                cmd_name, cmd, args = default_command
                ctx.invoked_subcommand = cmd_name
                click.Command.invoke(self, ctx)
                sub_ctx = cmd.make_context(ctx.info_name,
                                           args,
                                           parent=ctx.parent)

            with sub_ctx:
                return _process_result(sub_ctx.command.invoke(sub_ctx))


main = DefaultGroup(
    commands={
        "serve": serve,
        "disaggregated": disaggregated,
        "disaggregated_mpi_worker": disaggregated_mpi_worker,
        "mm_embedding_serve": serve_encoder
    })

if __name__ == "__main__":
    main()
