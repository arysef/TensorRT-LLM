import importlib
from pathlib import Path
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
BENCH_GROUP_CONTEXT_SETTINGS = {"show_default": True}
BENCH_SUBCOMMANDS = (
    ("build", "Build a benchmark engine."),
    ("throughput", "Run throughput benchmarking."),
    ("latency", "Run latency benchmarking."),
    ("prepare-dataset", "Prepare benchmark datasets."),
    ("visual-gen", "Run visual generation benchmarking."),
)

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
@click.option(
    "--model",
    "-m",
    required=True,
    type=str,
    help="The Huggingface name of the model to benchmark.",
)
@click.option(
    "--model_path",
    required=False,
    default=None,
    type=click.Path(writable=False, readable=True, path_type=Path),
    help=
    "Path to a Huggingface checkpoint directory for loading model components.",
)
@click.option(
    "--workspace",
    "-w",
    required=False,
    type=click.Path(writable=True, readable=True, path_type=Path),
    default="/tmp",  # nosec B108
    help="The directory to store benchmarking intermediate files.",
)
@click.option('--log_level',
              type=click.Choice(LOG_LEVELS),
              default='info',
              help="The logging level.")
@click.option("--revision",
              type=str,
              default=None,
              help="The revision to use for the HuggingFace model "
              "(branch name, tag name, or commit id).")
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

    # Create the workspace where we plan to store intermediate files.
    ctx.obj.workspace.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
