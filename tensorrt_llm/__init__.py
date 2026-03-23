# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib
import os
from typing import Any

# Disable UCC to WAR allgather issue before NGC PyTorch 25.12 upgrade.
os.environ["OMPI_MCA_coll_ucc_enable"] = "0"

_runtime_environment_prepared = False
_runtime_initialized = False


def _add_trt_llm_dll_directory():
    import platform
    on_windows = platform.system() == "Windows"
    if on_windows:
        import sysconfig
        from pathlib import Path
        os.add_dll_directory(
            Path(sysconfig.get_paths()['purelib']) / "tensorrt_llm" / "libs")


def _preload_python_lib():
    """
    Preload Python library.

    On Linux, the python executable links to libpython statically,
    so the dynamic library `libpython3.x.so` is not loaded.
    When using virtual environment on top of non-system Python installation,
    our libraries installed under `$VENV_PREFIX/lib/python3.x/site-packages/`
    have difficulties loading `$PREFIX/lib/libpython3.x.so.1.0` on their own,
    since venv does not symlink `libpython3.x.so` into `$VENV_PREFIX/lib/`,
    and the relative path from `$VENV_PREFIX` to `$PREFIX` is arbitrary.

    We preload the libraries here since the Python executable under `$PREFIX/bin`
    can easily find the library.
    """
    import platform
    on_linux = platform.system() == "Linux"
    if on_linux:
        import sys
        from ctypes import cdll
        v_major, v_minor, *_ = sys.version_info
        pythonlib = f'libpython{v_major}.{v_minor}.so'
        _ = cdll.LoadLibrary(pythonlib + '.1.0')
        _ = cdll.LoadLibrary(pythonlib)


import sys
from pathlib import Path


def _setup_vendored_triton_kernels():
    """Ensure our vendored triton_kernels takes precedence over any existing installation.

    Some environments bundle triton_kernels, which can conflict with our vendored version. This function:
    1. Clears any pre-loaded triton_kernels from sys.modules
    2. Temporarily adds our package root to sys.path
    3. Imports triton_kernels (caching our version in sys.modules)
    4. Removes the package root from sys.path
    """

    # Clear any pre-loaded triton_kernels from cache
    for mod in list(sys.modules.keys()):
        if mod == "triton_kernels" or mod.startswith("triton_kernels."):
            del sys.modules[mod]

    # Temporarily add our package root to sys.path
    root = Path(__file__).parent.parent

    vendored = root / "triton_kernels"
    if not vendored.exists():
        raise RuntimeError(
            f"Vendored triton_kernels module not found at {vendored}")

    should_add_to_path = str(root) not in sys.path
    if should_add_to_path:
        sys.path.insert(0, str(root))

    import triton_kernels  # noqa: F401

    if should_add_to_path:
        sys.path.remove(str(root))


def _prepare_runtime_environment() -> None:
    global _runtime_environment_prepared
    if _runtime_environment_prepared:
        return

    # Preserve the old runtime setup sequence, but defer it until a real export is used.
    _add_trt_llm_dll_directory()
    _preload_python_lib()
    _setup_vendored_triton_kernels()
    _runtime_environment_prepared = True


def _initialize_runtime() -> None:
    global _runtime_initialized
    if _runtime_initialized:
        return

    from ._common import _init

    _init()
    _runtime_initialized = True


from .version import __version__

__all__ = [
    'AutoConfig',
    'AutoModelForCausalLM',
    'logger',
    'str_dtype_to_trt',
    'torch_dtype_to_trt',
    'str_dtype_to_torch',
    'default_gpus_per_node',
    'local_mpi_rank',
    'local_mpi_size',
    'mpi_barrier',
    'mpi_comm',
    'mpi_rank',
    'set_mpi_comm',
    'mpi_world_size',
    'constant',
    'default_net',
    'default_trtnet',
    'precision',
    'net_guard',
    'torch_models',
    'Network',
    'Mapping',
    'MnnvlMemory',
    'MnnvlMoe',
    'MoEAlltoallInfo',
    'PluginBase',
    'Builder',
    'BuilderConfig',
    'build',
    'BuildConfig',
    'Tensor',
    'Parameter',
    'runtime',
    'Module',
    'functional',
    'models',
    'quantization',
    'tools',
    'LLM',
    'AsyncLLM',
    'MultimodalEncoder',
    'LlmArgs',
    'TorchLlmArgs',
    'TrtLlmArgs',
    'SamplingParams',
    'VisualGenArgs',
    'DisaggregatedParams',
    'KvCacheConfig',
    'math_utils',
    'VisualGen',
    'VisualGenParams',
    '__version__',
]

_LAZY_EXPORTS = {
    "AutoConfig": (".models.automodel", "AutoConfig"),
    "AutoModelForCausalLM": (".models.automodel", "AutoModelForCausalLM"),
    "logger": (".logger", "logger"),
    "str_dtype_to_trt": ("._utils", "str_dtype_to_trt"),
    "torch_dtype_to_trt": ("._utils", "torch_dtype_to_trt"),
    "str_dtype_to_torch": ("._utils", "str_dtype_to_torch"),
    "default_gpus_per_node": ("._utils", "default_gpus_per_node"),
    "local_mpi_rank": ("._utils", "local_mpi_rank"),
    "local_mpi_size": ("._utils", "local_mpi_size"),
    "mpi_barrier": ("._utils", "mpi_barrier"),
    "mpi_comm": ("._utils", "mpi_comm"),
    "mpi_rank": ("._utils", "mpi_rank"),
    "set_mpi_comm": ("._utils", "set_mpi_comm"),
    "mpi_world_size": ("._utils", "mpi_world_size"),
    "constant": (".functional", "constant"),
    "default_net": ("._common", "default_net"),
    "default_trtnet": ("._common", "default_trtnet"),
    "precision": ("._common", "precision"),
    "net_guard": (".network", "net_guard"),
    "torch_models": ("._torch.models", None),
    "Network": (".network", "Network"),
    "Mapping": (".mapping", "Mapping"),
    "MnnvlMemory": ("._mnnvl_utils", "MnnvlMemory"),
    "MnnvlMoe": ("._mnnvl_utils", "MnnvlMoe"),
    "MoEAlltoallInfo": ("._mnnvl_utils", "MoEAlltoallInfo"),
    "PluginBase": (".python_plugin", "PluginBase"),
    "Builder": (".builder", "Builder"),
    "BuilderConfig": (".builder", "BuilderConfig"),
    "build": (".builder", "build"),
    "BuildConfig": (".builder", "BuildConfig"),
    "Tensor": (".functional", "Tensor"),
    "Parameter": (".parameter", "Parameter"),
    "runtime": (".runtime", None),
    "Module": (".module", "Module"),
    "functional": (".functional", None),
    "models": (".models", None),
    "quantization": (".quantization", None),
    "tools": (".tools", None),
    "LLM": (".llmapi", "LLM"),
    "AsyncLLM": (".llmapi", "AsyncLLM"),
    "MultimodalEncoder": (".llmapi", "MultimodalEncoder"),
    "LlmArgs": (".llmapi.llm_args", "LlmArgs"),
    "TorchLlmArgs": (".llmapi.llm_args", "TorchLlmArgs"),
    "TrtLlmArgs": (".llmapi.llm_args", "TrtLlmArgs"),
    "SamplingParams": (".sampling_params", "SamplingParams"),
    "VisualGenArgs": ("._torch.visual_gen.config", "VisualGenArgs"),
    "DisaggregatedParams": (".disaggregated_params", "DisaggregatedParams"),
    "KvCacheConfig": (".llmapi", "KvCacheConfig"),
    "math_utils": (".math_utils", None),
    "VisualGen": (".llmapi", "VisualGen"),
    "VisualGenParams": (".llmapi", "VisualGenParams"),
}

_LIGHT_EXPORTS = {"__version__"}


def _load_export(name: str) -> Any:
    module_name, attr_name = _LAZY_EXPORTS[name]

    if name not in _LIGHT_EXPORTS:
        _prepare_runtime_environment()

    module = importlib.import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)

    if name not in _LIGHT_EXPORTS:
        _initialize_runtime()

    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return __version__
    if name in _LAZY_EXPORTS:
        return _load_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
