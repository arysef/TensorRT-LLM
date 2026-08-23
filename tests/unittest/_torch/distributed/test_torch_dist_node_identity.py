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
"""``TorchDist`` node identity with and without Ray.

``TorchDist`` is the non-MPI communicator. It was reachable only under Ray,
because ``_get_cluster_info`` took the node address and the GPU index from Ray
and raised otherwise -- which meant a plain ``torchrun`` job could not build the
intra-node process group at all, and so could not form a TP world without MPI.

The pair is used for one thing: grouping ranks by node. These tests pin that the
Ray answer is still preferred when Ray is running, that the fallback is used
(rather than an exception) when Ray is absent or idle, and that the fallback
values are the ones that identify this process's node and device.
"""

import socket
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.distributed import communicator

pytestmark = pytest.mark.cpu_only


class _RayStub:
    """Stands in for the real ``ray`` module."""

    class util:  # noqa: N801 - mirrors ray.util
        @staticmethod
        def get_node_ip_address():
            return "10.0.0.7"

    def __init__(self, initialized: bool, gpu_ids=(3,)):
        self._initialized = initialized
        self._gpu_ids = list(gpu_ids)

    def is_initialized(self):
        return self._initialized

    def get_gpu_ids(self):
        return self._gpu_ids


class _AbsentRay:
    """``executor.ray.stub``: every attribute raises, including the probe."""

    def __getattr__(self, name):
        raise RuntimeError(f'Ray not installed, so "ray.{name}" is unavailable.')


def test_ray_owns_the_identity_when_ray_is_running():
    with patch.object(communicator, "ray", _RayStub(initialized=True, gpu_ids=(3,))):
        node, gpu = communicator.TorchDist._node_identity()
    assert node == "10.0.0.7"
    assert gpu == 3


def test_an_idle_ray_falls_back_rather_than_raising():
    """Ray importable but not initialized is the state a ``torchrun`` job is in
    when Ray happens to be installed. The old code raised "Ray is not
    initialized" here."""
    with patch.object(communicator, "ray", _RayStub(initialized=False)):
        node, gpu = communicator.TorchDist._node_identity()
    assert node == socket.gethostname()
    assert gpu == torch.cuda.current_device() if torch.cuda.is_available() else gpu == 0


def test_an_absent_ray_falls_back_rather_than_raising():
    """The stub raises from ``__getattr__``, so even asking ``is_initialized``
    is an error; the probe has to survive that."""
    with patch.object(communicator, "ray", _AbsentRay()):
        node, gpu = communicator.TorchDist._node_identity()
    assert node == socket.gethostname()
    assert isinstance(gpu, int)


def test_ray_reporting_more_than_one_gpu_is_still_rejected():
    """One rank owns one device; the original assertion is load-bearing and the
    fallback must not have quietly removed it."""
    with patch.object(communicator, "ray", _RayStub(initialized=True, gpu_ids=(0, 1))):
        with pytest.raises(AssertionError):
            communicator.TorchDist._node_identity()


def test_the_fallback_key_is_stable_across_calls():
    """It groups ranks by node, so two calls in one process must agree."""
    with patch.object(communicator, "ray", _AbsentRay()):
        assert communicator.TorchDist._node_identity() == communicator.TorchDist._node_identity()
