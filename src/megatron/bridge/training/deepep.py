# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
from megatron.core.transformer import TransformerConfig


@dataclass
class DeepEPConfig:
    """
    A configuration class to enable DeepEP optimizations in Megatron-Bridge training if the hardware is supported.
    Per official documentation https://github.com/deepseek-ai/DeepEP,
    DeepEP is supported for Ampere (SM80) and Hopper (SM90) GPUs.

    This enables the following MoE optimizations on the model configuration:
    - Sets moe_token_dispatcher_type to "flex"
    - Enables moe_enable_deepep
    - Disables moe_shared_expert_overlap

    Enabling this configuratoin class on the ConfigContainer will automatically apply DeepEP optimizations
    when running on supported hardware.
    """

    def setup(self, model_config: TransformerConfig) -> None:
        apply_deepep(model_config)


def apply_deepep(model_config: TransformerConfig) -> None:
    """Apply DeepEP optimizations to the model config."""
    if torch.cuda.get_device_properties(0).major not in (8, 9):
        return

    model_config.moe_token_dispatcher_type = "flex"
    model_config.moe_enable_deepep = True
    model_config.moe_shared_expert_overlap = False
