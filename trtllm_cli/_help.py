from __future__ import annotations

import click


class HelpPassthroughOption(click.Option):
    """Defer required/type validation when Click is rendering help."""

    def handle_parse_result(self, ctx, opts, args):
        required = self.required
        if any(arg in (ctx.help_option_names or ("--help", )) for arg in args):
            self.required = False
        try:
            return super().handle_parse_result(ctx, opts, args)
        finally:
            self.required = required

    def type_cast_value(self, ctx, value):
        if ctx.resilient_parsing:
            return value
        return super().type_cast_value(ctx, value)


class NotRequiredForHelp(HelpPassthroughOption):
    """Backward-compatible alias for help-aware option parsing."""


class HelpPassthroughArgument(click.Argument):
    """Defer required argument validation when Click is rendering help."""

    def process_value(self, ctx, value):
        if ctx.resilient_parsing:
            return None if value is None else self.type_cast_value(ctx, value)
        return super().process_value(ctx, value)
