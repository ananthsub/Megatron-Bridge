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

"""
Unit tests for Nemotron4 recipe configuration builders.

Patterned after DeepSeek recipe tests: import all exported helpers from
`megatron.bridge.recipes.nemotron`, monkeypatch the model providers to lightweight
fakes, and assert a valid ConfigContainer is built with small overrides.
"""

import importlib
from typing import Callable

import pytest


_nemotron4_module = importlib.import_module("megatron.bridge.recipes.nemotron")
_NEMOTRON4_RECIPE_FUNCS = [
    getattr(_nemotron4_module, name)
    for name in getattr(_nemotron4_module, "__all__", [])
    if callable(getattr(_nemotron4_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    # Minimal overrides for fast unit testing without external I/O
    return {
        "name": f"unit_{name}",
        "dir": ".",
        "mock": True,
        "train_iters": 10,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "seq_length": 64,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
    }


class _FakeModelProvider:
    """Minimal provider to accept attribute assignments used in recipes."""

    def __init__(self):
        # Provide defaults for attributes that recipes might read or set
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 4096
        self.num_layers = 32
        self.hidden_size = 3072
        self.num_attention_heads = 24

    def finalize(self):
        return None


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1
    assert cfg.dataset.sequence_length >= 1


@pytest.mark.parametrize("recipe_func", _NEMOTRON4_RECIPE_FUNCS)
def test_each_nemotron4_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    # Monkeypatch model providers in the nemotron4 module
    module_name = "megatron.bridge.recipes.nemotron.nemotron4"
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "Nemotron4ModelProvider15B", _FakeModelProvider)
    monkeypatch.setattr(mod, "Nemotron4ModelProvider340B", _FakeModelProvider)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Ensure tokenizer is always NullTokenizer
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.tokenizer.vocab_size is not None

    # Parallelism and shaping
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 100
    assert cfg.train.manual_gc_eval == 100
