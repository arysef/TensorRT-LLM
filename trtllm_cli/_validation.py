from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import click
import yaml


def _primary_option_name(option_names: Sequence[str]) -> str:
    for option_name in option_names:
        if option_name.startswith("--"):
            return option_name
    return option_names[0]


def _extract_option_values(args: Sequence[str],
                           option_names: Sequence[str]) -> list[str]:
    values: list[str] = []
    index = 0
    option_names = tuple(option_names)

    while index < len(args):
        arg = args[index]
        if arg == "--":
            break

        for option_name in option_names:
            if arg == option_name:
                if index + 1 >= len(args):
                    raise click.BadParameter("Option requires a value.",
                                             param_hint=option_name)
                values.append(args[index + 1])
                index += 2
                break

            if option_name.startswith("--") and arg.startswith(f"{option_name}="):
                values.append(arg.split("=", 1)[1])
                index += 1
                break
        else:
            index += 1

    return values


def validate_yaml_file(value: str | None, *, param_hint: str) -> None:
    if value is None:
        return

    path = Path(value)
    if not path.is_file():
        raise click.BadParameter(f"File '{value}' does not exist.",
                                 param_hint=param_hint)

    try:
        with path.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise click.BadParameter(f"Invalid YAML in '{value}': {exc}",
                                 param_hint=param_hint) from exc
    except OSError as exc:
        raise click.BadParameter(f"Failed to read '{value}': {exc}",
                                 param_hint=param_hint) from exc


def validate_yaml_file_options(args: Sequence[str], *option_names: str) -> None:
    param_hint = _primary_option_name(option_names)
    for value in _extract_option_values(args, option_names):
        validate_yaml_file(value, param_hint=param_hint)


def validate_json_file(value: str | None, *, param_hint: str) -> Any:
    if value is None:
        return None

    path = Path(value)
    if not path.is_file():
        raise click.BadParameter(f"File '{value}' does not exist.",
                                 param_hint=param_hint)

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON in '{value}': {exc}",
                                 param_hint=param_hint) from exc
    except OSError as exc:
        raise click.BadParameter(f"Failed to read '{value}': {exc}",
                                 param_hint=param_hint) from exc


def validate_json_file_options(args: Sequence[str], *option_names: str) -> None:
    param_hint = _primary_option_name(option_names)
    for value in _extract_option_values(args, option_names):
        validate_json_file(value, param_hint=param_hint)


def validate_existing_file(value: str | None, *, param_hint: str) -> None:
    if value is None:
        return

    path = Path(value)
    if not path.is_file():
        raise click.BadParameter(f"File '{value}' does not exist.",
                                 param_hint=param_hint)


def validate_existing_directory(value: str | None, *, param_hint: str) -> None:
    if value is None:
        return

    path = Path(value)
    if not path.is_dir():
        raise click.BadParameter(f"Directory '{value}' does not exist.",
                                 param_hint=param_hint)


def validate_existing_file_options(args: Sequence[str], *option_names: str) -> None:
    param_hint = _primary_option_name(option_names)
    for value in _extract_option_values(args, option_names):
        validate_existing_file(value, param_hint=param_hint)


def validate_existing_directory_options(args: Sequence[str],
                                        *option_names: str) -> None:
    param_hint = _primary_option_name(option_names)
    for value in _extract_option_values(args, option_names):
        validate_existing_directory(value, param_hint=param_hint)


def validate_json_string(value: str | None, *, param_hint: str) -> None:
    if value is None:
        return

    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON: {exc}",
                                 param_hint=param_hint) from exc


def validate_json_string_options(args: Sequence[str],
                                 *option_names: str) -> None:
    param_hint = _primary_option_name(option_names)
    for value in _extract_option_values(args, option_names):
        validate_json_string(value, param_hint=param_hint)
