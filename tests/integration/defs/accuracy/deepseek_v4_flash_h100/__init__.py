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
"""Evidence support package for DeepSeek-V4-Flash Hopper (SM90) bring-up.

The modules here own the *pre-registered* side of the bring-up gate and the
independent reference tier:

``checkpoint_inventory``  exact structural contract of the raw checkpoint.
``build_manifests``       fixed prompt manifest and numerical tolerances.
``torch_goldens``         pure-Torch goldens for every V4 module.
``hf_native_golden``      remaps the checkpoint into HF naming/BF16 and runs
                          ``AutoModelForCausalLM.generate(do_sample=False)`` to
                          produce ``manifests/native_generate_golden.json``.

They are deliberately free of any TensorRT-LLM modelling import so that they
can be consulted from the independent reference ladder without sharing
implementation helpers with the code under test.
"""
