from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import click

from trtllm_cli._bench_metadata import (BENCH_GROUP_CONTEXT_SETTINGS,
                                        BENCH_SUBCOMMANDS,
                                        add_bench_options)
from trtllm_cli._help import NotRequiredForHelp

SUBCOMMAND_SPECS = {
    "build": ("tensorrt_llm.bench.build.build", "build_command"),
    "throughput": ("tensorrt_llm.bench.benchmark.throughput",
                    "throughput_command"),
    "latency": ("tensorrt_llm.bench.benchmark.low_latency",
                 "latency_command"),
    "prepare-dataset": ("tensorrt_llm.bench.dataset.prepare_dataset",
                         "prepare_dataset"),
    "visual-gen": ("tensorrt_llm.bench.benchmark.visual_gen",
                    "visual_gen_command"),
}

SHORT_HELP_BY_NAME = dict(BENCH_SUBCOMMANDS)


class LazyBenchGroup(click.Group):

    def list_commands(self, ctx):
        return [name for name, _ in BENCH_SUBCOMMANDS]

    def format_commands(self, ctx, formatter):
        with formatter.section("Commands"):
            formatter.write_dl(list(BENCH_SUBCOMMANDS))

    def get_command(self, ctx, name):
        spec = SUBCOMMAND_SPECS.get(name)
        if spec is None:
            return None

        module = importlib.import_module(spec[0])
        command = getattr(module, spec[1])
        command.short_help = SHORT_HELP_BY_NAME[name]
        return command


@click.group(name="trtllm-bench",
             cls=LazyBenchGroup,
             context_settings=BENCH_GROUP_CONTEXT_SETTINGS)
@add_bench_options(option_cls=NotRequiredForHelp)
@click.pass_context
def main(
    ctx,
    model: str,
    model_path: Path,
    workspace: Path,
    log_level: str,
    revision: Optional[str],
) -> None:
    from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
    from tensorrt_llm.logger import logger

    logger.set_level(log_level)
    if model is None:
        return

    ctx.obj = BenchmarkEnvironment(model=model,
                                   checkpoint_path=model_path,
                                   workspace=workspace,
                                   revision=revision)
    ctx.obj.workspace.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
