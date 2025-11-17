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

import importlib
from typing import Callable

import pytest


_nemotronh_module = importlib.import_module("megatron.bridge.recipes.nemotronh")
_NEMOTRONH_RECIPE_FUNCS = [
    getattr(_nemotronh_module, name)
    for name in getattr(_nemotronh_module, "__all__", [])
    if callable(getattr(_nemotronh_module, name, None))
]


# NemotronH finetune-specific tests
_NEMOTRONH_FINETUNE_FUNCS = [
    getattr(_nemotronh_module, name)
    for name in [
        "nemotronh_4b_finetune_config",
        "nemotronh_8b_finetune_config",
        "nemotronh_47b_finetune_config",
        "nemotronh_56b_finetune_config",
    ]
    if callable(getattr(_nemotronh_module, name, None))
]


# Nemotron Nano V2 finetune-specific tests
_NEMOTRON_NANO_V2_FINETUNE_FUNCS = [
    getattr(_nemotronh_module, name)
    for name in [
        "nemotron_nano_9b_v2_finetune_config",
        "nemotron_nano_12b_v2_finetune_config",
    ]
    if callable(getattr(_nemotronh_module, name, None))
]


def _safe_overrides_for(name: str) -> dict:
    # Detect if this is a finetune recipe
    is_finetune = "finetune" in name.lower()
    lname = name.lower()

    overrides = {
        "name": f"unit_{name}",
        "dir": ".",
        "train_iters": 10,
        "micro_batch_size": 1,
        "seq_length": 64,
        "min_lr": 1e-5,
        "lr_warmup_iters": 2,
        "global_batch_size": 2,
    }

    if is_finetune:
        # Finetuning-specific overrides
        overrides.update(
            {
                "finetune_lr": 1e-4,
            }
        )
    else:
        # Pretrain-specific overrides
        overrides.update(
            {
                "mock": True,
                "lr": 1e-4,
                "use_null_tokenizer": True,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            }
        )

        # Large models may need additional flags
        if "47b" in lname or "56b" in lname:
            overrides.update(
                {
                    "virtual_pipeline_model_parallel_size": None,
                    "sequence_parallel": True,
                }
            )

    return overrides


class _FakeModelCfg:
    def __init__(self):
        self.cross_entropy_fusion_impl = "te"

    def finalize(self):
        return None


class _FakeBridge:
    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()


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

    # Check sequence length (different attribute names for different dataset types)
    if hasattr(cfg.dataset, "sequence_length"):
        assert cfg.dataset.sequence_length >= 1  # GPTDatasetConfig
    elif hasattr(cfg.dataset, "seq_length"):
        assert cfg.dataset.seq_length >= 1  # FinetuningDatasetConfig / HFDatasetConfig
    else:
        # Some other dataset type
        assert cfg.dataset is not None


@pytest.mark.parametrize("recipe_func", _NEMOTRONH_RECIPE_FUNCS)
def test_each_nemotronh_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)

    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches recipe type
    is_finetune = "finetune" in recipe_func.__name__.lower()
    if is_finetune:
        # Finetuning recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes honor use_null_tokenizer override
        if overrides.get("use_null_tokenizer"):
            assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1


@pytest.mark.parametrize("recipe_func", _NEMOTRONH_FINETUNE_FUNCS)
def test_nemotronh_finetune_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each NemotronH finetune recipe builds a valid config (Full SFT only)."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)
    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Finetuning always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    # NemotronH only supports full SFT (no PEFT)
    assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _NEMOTRON_NANO_V2_FINETUNE_FUNCS)
def test_nemotron_nano_v2_finetune_config_builds(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    """Test that each Nemotron Nano V2 finetune recipe builds a valid config (Full SFT only)."""
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    overrides = _safe_overrides_for(recipe_func.__name__)
    cfg = recipe_func(**overrides)

    _assert_basic_config(cfg)

    # Finetuning always uses HF tokenizer
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model is not None

    # Check parallelism
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1

    # Nemotron Nano V2 only supports full SFT (no PEFT)
    assert cfg.peft is None
