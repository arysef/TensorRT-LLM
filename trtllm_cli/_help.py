from __future__ import annotations

from typing import Sequence

import click

from trtllm_cli._dispatch import current_invocation_argv


def has_help_flag(args: Sequence[str],
                  help_option_names: Sequence[str] | None = None) -> bool:
    help_flags = tuple(help_option_names or ("--help",))
    return any(arg in help_flags for arg in args)


def should_show_lightweight_help(
        ctx: click.Context, args: Sequence[str]) -> bool:
    return (has_help_flag(args, ctx.help_option_names)
            or has_help_flag(current_invocation_argv(), ctx.help_option_names))


def echo_lightweight_help(ctx: click.Context,
                          *,
                          parent: click.Context | None = None,
                          info_name: str | None = None) -> None:
    help_ctx = ctx
    if parent is not None or info_name is not None:
        help_ctx = click.Context(ctx.command,
                                 info_name=info_name or ctx.info_name,
                                 parent=parent)
    click.echo(help_ctx.get_help())


class HelpPassthroughOption(click.Option):
    """A click.Option that defers required/type validation when help is requested."""

    def _help_requested(self, ctx: click.Context, args: Sequence[str]) -> bool:
        return (has_help_flag(args, ctx.help_option_names)
                or has_help_flag(current_invocation_argv(),
                                 ctx.help_option_names))

    def handle_parse_result(self, ctx, opts, args):
        required = self.required
        if self._help_requested(ctx, args):
            self.required = False
        try:
            return super().handle_parse_result(ctx, opts, args)
        finally:
            self.required = required

    def type_cast_value(self, ctx, value):
        if has_help_flag(current_invocation_argv(), ctx.help_option_names):
            return value
        return super().type_cast_value(ctx, value)


class NotRequiredForHelp(HelpPassthroughOption):
    """Backward-compatible alias for help-aware option parsing."""


class HelpPassthroughArgument(click.Argument):
    """A click.Argument that defers required validation when help is requested."""

    def process_value(self, ctx, value):
        if has_help_flag(current_invocation_argv(), ctx.help_option_names):
            return None if value is None else self.type_cast_value(ctx, value)
        return super().process_value(ctx, value)
