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

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class InvocationState:
    argv: tuple[str, ...]
    prog_name: str
    root_mode: bool


_current_invocation: Optional[InvocationState] = None


@contextmanager
def invocation_state(argv: list[str], *, prog_name: str,
                     root_mode: bool) -> Iterator[None]:
    global _current_invocation
    previous_state = _current_invocation
    _current_invocation = InvocationState(tuple(argv),
                                          prog_name=prog_name,
                                          root_mode=root_mode)
    try:
        yield
    finally:
        _current_invocation = previous_state


def invoke_entry(command, *, prog_name: str, root_mode: bool = False):
    argv = list(sys.argv[1:])
    with invocation_state(argv, prog_name=prog_name, root_mode=root_mode):
        return command.main(args=argv, prog_name=prog_name)


def current_invocation_argv() -> tuple[str, ...]:
    if _current_invocation is not None:
        return _current_invocation.argv
    return tuple(sys.argv[1:])


def current_delegated_argv(top_level_command: str) -> tuple[str, ...]:
    return tuple(_get_delegated_argv(top_level_command))


def _get_delegated_argv(top_level_command: str) -> list[str]:
    if _current_invocation is None:
        return list(sys.argv[1:])

    argv = list(_current_invocation.argv)
    if _current_invocation.root_mode:
        if not argv or argv[0] != top_level_command:
            raise RuntimeError(
                f"Expected '{top_level_command}' to be the first CLI token, got {argv!r}."
            )
        return argv[1:]
    return argv


def _get_prog_name(top_level_command: str, standalone_prog_name: str) -> str:
    if _current_invocation is None:
        return standalone_prog_name

    if _current_invocation.root_mode:
        return f"{_current_invocation.prog_name} {top_level_command}"
    return standalone_prog_name


def delegate_click_command(module_path: str,
                           attribute_name: str,
                           *,
                           top_level_command: str,
                           standalone_prog_name: str):
    command = getattr(importlib.import_module(module_path), attribute_name)
    return command.main(
        args=_get_delegated_argv(top_level_command),
        prog_name=_get_prog_name(top_level_command, standalone_prog_name),
    )


@contextmanager
def patched_argv(argv0: str, argv: list[str]) -> Iterator[None]:
    previous_argv = sys.argv
    sys.argv = [argv0, *argv]
    try:
        yield
    finally:
        sys.argv = previous_argv


def delegate_argparse_command(module_path: str,
                              attribute_name: str,
                              *,
                              top_level_command: str,
                              standalone_prog_name: str):
    argv = _get_delegated_argv(top_level_command)
    prog_name = _get_prog_name(top_level_command, standalone_prog_name)
    command = getattr(importlib.import_module(module_path), attribute_name)
    with patched_argv(prog_name, argv):
        return command()
