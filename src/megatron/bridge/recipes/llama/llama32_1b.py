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

import os
from pathlib import Path
from typing import List, Optional, Union

import torch
from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.llama import Llama32ModelProvider1B
from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.checkpointing import checkpoint_exists
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    HFDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def model_config(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
) -> Llama32ModelProvider1B:
    """
    Configure the Llama3.2 1B model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.

    Returns:
        Llama32ModelProvider1B: Configuration for the Llama3.2 1B model.
    """
    return Llama32ModelProvider1B(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
    )


def pretrain_config(
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Llama3.2 1B model.

    Args:
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (Optional[List[str]]): List of paths to dataset files. If None, mock data will be used.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism to be passed to model_config.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for the model.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration for the model.

    Returns:
        ConfigContainer: Configuration for pre-training.

    Note:
        Sequence length is hardcoded to 8192 for Llama3.2 1B pretraining.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    model_cfg = model_config(
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=lr,
        min_lr=min_lr,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=2000,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            # Dataloader config parameters
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
        )

    return cfg


def finetune_config(
    dir: Optional[str] = None,
    name: str = "default",
    hf_model_id: Optional[str] = "meta-llama/Llama-3.2-1B",
    pretrained_checkpoint: Optional[str] = None,
    hf_conversion_kwargs: Optional[dict] = None,
    # Training hyperparameters
    train_iters: int = 1000,
    packed_sequence: bool = False,
    # PEFT settings
    peft_scheme: Optional[str] = "lora",  # "lora", "dora", or None for full fine-tuning
    # Learning rate (auto-adjusted based on PEFT scheme)
    lr: Optional[float] = None,  # Will be set based on peft_scheme
    min_lr: float = 0,
    lr_warmup_iters: int = 50,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
) -> ConfigContainer:
    """
    Create a fine-tuning configuration for Llama3.2 1B model on SQuAD dataset.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The configuration uses the SQuAD dataset for question-answering fine-tuning.

    Supports automatic HuggingFace model conversion, PEFT (LoRA/DoRA) with optimized
    defaults, and packed sequence training for improved efficiency.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        hf_model_id (Optional[str]): HuggingFace model ID (e.g., 'meta-llama/Llama-3.2-1B')
                    for automatic conversion. Mutually exclusive with pretrained_checkpoint.
        pretrained_checkpoint (Optional[str]): Path to existing Megatron checkpoint directory.
                              Mutually exclusive with hf_model_id.
        hf_conversion_kwargs (Optional[dict]): Additional arguments for HF model loading
                             (torch_dtype, device_map, trust_remote_code, etc.)
        train_iters (int): Total number of training iterations.
        packed_sequence (bool): Whether to use packed sequences for better efficiency.
                       Automatically configures sequence length (4096 vs 2048) and batch size (8 vs 128).
        peft_scheme (Optional[str]): PEFT scheme to use. Options: "lora", "dora", or None for full fine-tuning.
                    Defaults to "lora". Uses fixed dim=8 and alpha=16 for both LoRA and DoRA.
        lr (Optional[float]): Learning rate. If None, automatically set to 5e-6 for full fine-tuning
            or 1e-4 for PEFT schemes.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.

    Returns:
        ConfigContainer: Configuration for fine-tuning on SQuAD dataset.

    Examples:
        Basic LoRA fine-tuning:
        >>> cfg = finetune_config(name="my_lora_finetune")

        Full fine-tuning:
        >>> cfg = finetune_config(name="full_finetune", peft_scheme=None)

        DoRA with packed sequences:
        >>> cfg = finetune_config(
        ...     name="dora_packed",
        ...     peft_scheme="dora",
        ...     packed_sequence=True
        ... )
    """
    # Handle checkpoint resolution
    if hf_model_id and pretrained_checkpoint:
        raise ValueError("Specify either hf_model_id or pretrained_checkpoint, not both")

    if not hf_model_id and not pretrained_checkpoint:
        # Default to Llama 3.2 1B if nothing specified
        hf_model_id = "meta-llama/Llama-3.2-1B"

    # Resolve HF model to Megatron checkpoint path if needed
    if hf_model_id:
        assert AutoBridge.can_handle(hf_model_id), f"Model {hf_model_id} is not supported by AutoBridge"
        pretrained_checkpoint = _resolve_hf_model(hf_model_id, **(hf_conversion_kwargs or {}))

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    seq_length = 4096 if packed_sequence else 2048
    global_batch_size = 8 if packed_sequence else 128
    micro_batch_size = 1

    # PEFT configuration
    peft_config = None
    use_distributed_optimizer = True

    # Set learning rate based on PEFT scheme (match NeMo defaults)
    if lr is None:
        if peft_scheme is None or peft_scheme.lower() == "none":
            lr = 5e-6  # Full fine-tuning learning rate
        else:
            lr = 1e-4  # PEFT learning rate

    # Configure PEFT if specified
    if peft_scheme and peft_scheme.lower() != "none":
        if peft_scheme.lower() == "lora":
            peft_config = LoRA(dim=8, alpha=16)
        elif peft_scheme.lower() == "dora":
            peft_config = DoRA(dim=8, alpha=16)
        else:
            raise ValueError(f"Unrecognized PEFT scheme: {peft_scheme}. Supported schemes: 'lora', 'dora', or None")

        # PEFT uses different optimizer settings
        use_distributed_optimizer = False

    # Model configuration
    model_cfg = model_config()

    # PEFT-specific model settings (match NeMo behavior)
    if peft_config is not None:
        # Some settings currently do not function correctly with PEFT
        model_cfg.cross_entropy_loss_fusion = False

    # Optimizer and scheduler configuration
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=30,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            use_distributed_optimizer=use_distributed_optimizer,
        ),
        dataset=HFDatasetConfig(
            dataset_name="squad",
            process_example_fn=process_squad_example,
            seq_length=seq_length,
            seed=1234,
            dataloader_type="cyclic" if packed_sequence else "single",
            num_workers=1,
            do_validation=False,
            do_test=False,
            val_proportion=None,
            dataset_kwargs={"pad_to_max_length": True} if packed_sequence else {},
            packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=seq_length) if packed_sequence else None,
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Llama-3.2-1B",
        ),
        checkpoint=CheckpointConfig(
            save_interval=50,
            save=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=True,
            pretrained_checkpoint=pretrained_checkpoint,
        ),
        rng=RNGConfig(seed=1234),
        mixed_precision=precision_config,
        peft=peft_config,
    )

    return cfg


def _resolve_hf_model(hf_model_id: str, **kwargs) -> str:
    """
    Resolve HF model ID to Megatron checkpoint path, converting if needed.

    Args:
        hf_model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-3.2-1B')
        **kwargs: Additional arguments for HF model loading

    Returns:
        Path to Megatron checkpoint directory
    """
    # Define cache location
    cache_base = Path(os.environ.get("NEMO_HOME", os.path.expanduser("~/.cache/nemo")))
    cache_dir = cache_base / "hf_models" / hf_model_id.replace("/", "--")

    # Check if already converted
    if _is_valid_checkpoint(cache_dir):
        return str(cache_dir)

    _convert_model(hf_model_id, cache_dir, kwargs)

    return str(cache_dir)


def _is_valid_checkpoint(path: Path) -> bool:
    """Check if path contains a valid Megatron checkpoint."""
    return checkpoint_exists(str(path))


def _convert_model(hf_model_id: str, cache_dir: Path, kwargs: dict) -> None:
    """Convert HF model to Megatron format."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    AutoBridge.import_ckpt(hf_model_id=hf_model_id, megatron_path=str(cache_dir), **kwargs)
