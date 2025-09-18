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

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils import fusions


logger = logging.getLogger(__name__)


def squared_relu(x):
    """Squared ReLU activation function."""
    return torch.pow(torch.nn.functional.relu(x), 2)


@dataclass
class NemotronModelProvider(GPTModelProvider):
    """Configuration class for Nemotron models."""

    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    activation_func: Callable = squared_relu
    position_embedding_type: str = "rope"
    share_embeddings_and_output_weights: bool = False
    add_bias_linear: bool = False

    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    rotary_percent: float = 0.5
    masked_softmax_fusion: bool = field(default_factory=fusions.can_enable_masked_softmax_fusion)
    persist_layer_norm: bool = True
    bias_dropout_add_fusion: bool = False
    layernorm_zero_centered_gamma: bool = True
    cross_entropy_loss_fusion: bool = True
    apply_rope_fusion: bool = field(default_factory=fusions.can_enable_apply_rope_fusion)
