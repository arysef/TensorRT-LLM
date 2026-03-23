from pathlib import Path
from typing import Optional

import click

from trtllm_cli._bench_metadata import (BENCH_GROUP_CONTEXT_SETTINGS,
                                        BENCH_SUBCOMMANDS,
                                        add_bench_options)
from trtllm_cli._help import NotRequiredForHelp
from tensorrt_llm.bench.benchmark.low_latency import latency_command
from tensorrt_llm.bench.benchmark.throughput import throughput_command
from tensorrt_llm.bench.benchmark.visual_gen import visual_gen_command
from tensorrt_llm.bench.build.build import build_command
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.dataset.prepare_dataset import prepare_dataset
from tensorrt_llm.logger import logger

SUBCOMMANDS_BY_NAME = {
    "build": build_command,
    "throughput": throughput_command,
    "latency": latency_command,
    "prepare-dataset": prepare_dataset,
    "visual-gen": visual_gen_command,
}


@click.group(name="trtllm-bench", context_settings=BENCH_GROUP_CONTEXT_SETTINGS)
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
    logger.set_level(log_level)
    if model is None:
        return

    ctx.obj = BenchmarkEnvironment(model=model,
                                   checkpoint_path=model_path,
                                   workspace=workspace,
                                   revision=revision)

    # Create the workspace where we plan to store intermediate files.
    ctx.obj.workspace.mkdir(parents=True, exist_ok=True)


for name, short_help in BENCH_SUBCOMMANDS:
    command = SUBCOMMANDS_BY_NAME[name]
    command.short_help = short_help
    main.add_command(command, name=name)

if __name__ == "__main__":
    main()
