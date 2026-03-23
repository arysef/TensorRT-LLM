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

from trtllm_cli._dispatch import delegate_click_command
from trtllm_cli._help import (HelpPassthroughArgument,
                              HelpPassthroughOption, echo_lightweight_help,
                              should_show_lightweight_help)
from trtllm_cli._serve_metadata import (
    DISAGGREGATED_HELP,
    DISAGGREGATED_MPI_WORKER_HELP,
    DISAGGREGATED_MPI_WORKER_SHORT_HELP,
    DISAGGREGATED_SHORT_HELP,
    MM_EMBEDDING_SERVE_HELP,
    MM_EMBEDDING_SERVE_SHORT_HELP,
    SERVE_HELP,
    SERVE_SHORT_HELP,
    add_disaggregated_mpi_worker_options,
    add_disaggregated_options,
    add_mm_embedding_serve_options,
    add_serve_options,
)
from trtllm_cli._validation import validate_json_string, validate_yaml_file

PROXY_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _delegate_to_legacy_serve() -> None:
    delegate_click_command(
        "tensorrt_llm.commands.serve",
        "main",
        top_level_command="serve",
        standalone_prog_name="trtllm-serve",
    )


def _validate_serve_options(extra_llm_api_options: str | None,
                            metadata_server_config_file: str | None,
                            extra_visual_gen_options: str | None,
                            media_io_kwargs: str | None) -> None:
    validate_yaml_file(extra_llm_api_options, param_hint="--config")
    validate_yaml_file(metadata_server_config_file,
                       param_hint="--metadata_server_config_file")
    validate_yaml_file(extra_visual_gen_options,
                       param_hint="--extra_visual_gen_options")
    validate_json_string(media_io_kwargs, param_hint="--media_io_kwargs")


def _validate_mm_embedding_serve_options(extra_encoder_options: str | None,
                                         metadata_server_config_file:
                                         str | None) -> None:
    validate_yaml_file(extra_encoder_options, param_hint="--config")
    validate_yaml_file(metadata_server_config_file,
                       param_hint="--metadata_server_config_file")


def _validate_disaggregated_options(config_file: str | None,
                                    metadata_server_config_file:
                                    str | None) -> None:
    validate_yaml_file(config_file, param_hint="--config")
    validate_yaml_file(metadata_server_config_file,
                       param_hint="--metadata_server_config_file")


def _validate_disaggregated_mpi_worker_options(config_file: str | None) -> None:
    validate_yaml_file(config_file, param_hint="--config")


class DefaultGroup(click.Group):
    """Click group that preserves trtllm-serve's default serve subcommand."""

    def _resolve_default_command(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.commands:
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
                sub_ctx.meta["implicit_default_command"] = True

            with sub_ctx:
                return _process_result(sub_ctx.command.invoke(sub_ctx))


@click.command(
    name="serve",
    short_help=SERVE_SHORT_HELP,
    help=SERVE_HELP,
    add_help_option=False,
    context_settings=PROXY_CONTEXT_SETTINGS,
)
@add_serve_options(model_required=True,
                   option_cls=HelpPassthroughOption,
                   argument_cls=HelpPassthroughArgument)
@click.pass_context
def serve_proxy(ctx, model: str | None, **_kwargs) -> None:
    if should_show_lightweight_help(ctx, ctx.args):
        if ctx.meta.get("implicit_default_command"):
            echo_lightweight_help(ctx)
            return
        if ctx.parent:
            if ctx.parent.parent:
                echo_lightweight_help(ctx,
                                      parent=ctx.parent.parent,
                                      info_name=ctx.parent.info_name)
            else:
                echo_lightweight_help(ctx, info_name=ctx.parent.info_name)
            return
        echo_lightweight_help(ctx)
        return
    if model is None:
        raise click.UsageError("Missing argument 'MODEL'.")
    _validate_serve_options(_kwargs.get("extra_llm_api_options"),
                            _kwargs.get("metadata_server_config_file"),
                            _kwargs.get("extra_visual_gen_options"),
                            _kwargs.get("media_io_kwargs"))
    _delegate_to_legacy_serve()


@click.command(
    name="disaggregated",
    short_help=DISAGGREGATED_SHORT_HELP,
    help=DISAGGREGATED_HELP,
    add_help_option=False,
    context_settings=PROXY_CONTEXT_SETTINGS,
)
@add_disaggregated_options(option_cls=HelpPassthroughOption)
@click.pass_context
def disaggregated_proxy(ctx, **_kwargs) -> None:
    if should_show_lightweight_help(ctx, ctx.args):
        echo_lightweight_help(ctx)
        return
    _validate_disaggregated_options(_kwargs.get("config_file"),
                                    _kwargs.get(
                                        "metadata_server_config_file"))
    _delegate_to_legacy_serve()


@click.command(
    name="disaggregated_mpi_worker",
    short_help=DISAGGREGATED_MPI_WORKER_SHORT_HELP,
    help=DISAGGREGATED_MPI_WORKER_HELP,
    add_help_option=False,
    context_settings=PROXY_CONTEXT_SETTINGS,
)
@add_disaggregated_mpi_worker_options(option_cls=HelpPassthroughOption)
@click.pass_context
def disaggregated_mpi_worker_proxy(ctx, **_kwargs) -> None:
    if should_show_lightweight_help(ctx, ctx.args):
        echo_lightweight_help(ctx)
        return
    _validate_disaggregated_mpi_worker_options(_kwargs.get("config_file"))
    _delegate_to_legacy_serve()


@click.command(
    name="mm_embedding_serve",
    short_help=MM_EMBEDDING_SERVE_SHORT_HELP,
    help=MM_EMBEDDING_SERVE_HELP,
    add_help_option=False,
    context_settings=PROXY_CONTEXT_SETTINGS,
)
@add_mm_embedding_serve_options(model_required=True,
                                option_cls=HelpPassthroughOption,
                                argument_cls=HelpPassthroughArgument)
@click.pass_context
def mm_embedding_serve_proxy(ctx, model: str | None, **_kwargs) -> None:
    if should_show_lightweight_help(ctx, ctx.args):
        echo_lightweight_help(ctx)
        return
    if model is None:
        raise click.UsageError("Missing argument 'MODEL'.")
    _validate_mm_embedding_serve_options(_kwargs.get("extra_encoder_options"),
                                         _kwargs.get(
                                             "metadata_server_config_file"))
    _delegate_to_legacy_serve()


@click.group(cls=DefaultGroup)
def cli() -> None:
    """TensorRT-LLM serving commands."""


cli.add_command(serve_proxy)
cli.add_command(disaggregated_proxy)
cli.add_command(disaggregated_mpi_worker_proxy)
cli.add_command(mm_embedding_serve_proxy)
