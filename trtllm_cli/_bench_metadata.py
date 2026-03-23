from __future__ import annotations

from pathlib import Path
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
BENCH_GROUP_CONTEXT_SETTINGS = {"show_default": True}
BENCH_HELP = "Benchmark TensorRT-LLM models."
BENCH_SUBCOMMANDS = (
    ("build", "Build a benchmark engine."),
    ("throughput", "Run throughput benchmarking."),
    ("latency", "Run latency benchmarking."),
    ("prepare-dataset", "Prepare benchmark datasets."),
    ("visual-gen", "Run visual generation benchmarking."),
)


def _option(*param_decls,
            option_cls: type[click.Option] | None = None,
            **kwargs) -> Callable:
    if option_cls is not None:
        kwargs.setdefault("cls", option_cls)
    return click.option(*param_decls, **kwargs)


def add_bench_options(
    *, option_cls: type[click.Option] | None = None
) -> Callable:

    def decorator(command: Callable) -> Callable:
        command = _option(
            "--revision",
            type=str,
            default=None,
            help="The revision to use for the HuggingFace model "
            "(branch name, tag name, or commit id).",
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
            "--workspace",
            "-w",
            required=False,
            type=click.Path(writable=True, readable=True, path_type=Path),
            default="/tmp",  # nosec B108
            help="The directory to store benchmarking intermediate files.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--model_path",
            required=False,
            default=None,
            type=click.Path(writable=False, readable=True, path_type=Path),
            help="Path to a Huggingface checkpoint directory for loading model components.",
            option_cls=option_cls,
        )(command)
        command = _option(
            "--model",
            "-m",
            required=True,
            type=str,
            help="The Huggingface name of the model to benchmark.",
            option_cls=option_cls,
        )(command)
        return command

    return decorator
