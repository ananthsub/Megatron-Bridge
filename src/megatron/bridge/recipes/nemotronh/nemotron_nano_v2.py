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

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.models.nemotronh import (
    NemotronNano9Bv2Provider,
    NemotronNano12Bv2Provider,
)
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed, get_mixed_precision_config


class NemotronNanoV2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Nemotron Nano v2 recipe helper functions."""

    # Core identifiers
    model_provider: NemotronNano9Bv2Provider | NemotronNano12Bv2Provider
    tokenizer_model: str | None
    dir: str | None
    name: str
    # Dataset configuration
    data_paths: list[str] | None
    data_args_path: str | None
    train_data_path: list[str] | None
    valid_data_path: list[str] | None
    test_data_path: list[str] | None
    per_split_data_args_path: str | None
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    sequence_parallel: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None
    # CommOverlap setting
    enable_default_comm_overlap: bool


def nemotron_nano_9b_v2_pretrain_config(**user_kwargs: Unpack[NemotronNanoV2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Nemotron Nano 9B v2.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=2, PP=1, SP=True.

    See `_nemotron_nano_v2_common` for the full list of parameters.
    """
    recommended_kwargs: NemotronNanoV2CommonKwargs = {
        "model_provider": NemotronNano9Bv2Provider,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": True,
        "precision_config": "bf16_mixed",
        "enable_default_comm_overlap": True,
    }
    combined_kwargs: NemotronNanoV2CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron_nano_v2_common(tokenizer_model="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", **combined_kwargs)


def nemotron_nano_12b_v2_pretrain_config(**user_kwargs: Unpack[NemotronNanoV2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Nemotron Nano 12B v2.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=4, PP=1, SP=True.

    Note: Uses FP8 precision by default. Communication overlap is disabled by default.

    See `_nemotron_nano_v2_common` for the full list of parameters.
    """
    recommended_kwargs: NemotronNanoV2CommonKwargs = {
        "model_provider": NemotronNano12Bv2Provider,
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": True,
        "precision_config": "nanov2_bf16_with_fp8_current_scaling_mixed",
        "enable_default_comm_overlap": False,
    }
    combined_kwargs: NemotronNanoV2CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron_nano_v2_common(tokenizer_model="nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base", **combined_kwargs)


def _nemotron_nano_v2_common(
    model_provider: type[NemotronNano9Bv2Provider] | type[NemotronNano12Bv2Provider],
    tokenizer_model: str | None = None,
    dir: str | None = None,
    name: str = "default",
    # Dataset configuration
    data_paths: list[str] | None = None,
    data_args_path: str | None = None,
    train_data_path: list[str] | None = None,
    valid_data_path: list[str] | None = None,
    test_data_path: list[str] | None = None,
    per_split_data_args_path: str | None = None,
    mock: bool = False,
    # Model configuration
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = True,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 768,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int | None = None,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    # CommOverlap setting
    enable_default_comm_overlap: bool = True,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Nemotron Nano v2 models.

    Args:
        model_provider: The model provider class for the specific Nemotron Nano v2 variant.
        tokenizer_model: HuggingFace tokenizer model name (only used when use_null_tokenizer=False).
        dir: Base directory for saving logs and checkpoints.
        name: Name of the pre-training run.
        data_paths: List of paths to dataset files. If None, mock data will be used.
        data_args_path: Path to file containing data arguments.
        train_data_path: List of training data paths.
        valid_data_path: List of validation data paths.
        test_data_path: List of test data paths.
        per_split_data_args_path: Path to JSON file with per-split data configuration.
        mock: Whether to use mock data. If True, ignores data_paths.
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        pipeline_dtype: Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size: Size of virtual pipeline parallelism.
        context_parallel_size: Degree of context parallelism to be passed to model_config.
        sequence_parallel: Whether to use sequence parallelism.
        train_iters: Total number of training iterations.
        global_batch_size: Global batch size for training.
        micro_batch_size: Micro batch size for training.
        seq_length: Sequence length for training data.
        lr: Learning rate.
        min_lr: Minimum learning rate for cosine decay.
        lr_warmup_iters: Number of warmup iterations for the learning rate.
        lr_decay_iters: Number of iterations for learning rate decay.
        use_null_tokenizer: Whether to use NullTokenizer instead of HuggingFaceTokenizer.
        precision_config: Precision configuration for the model.
        comm_overlap_config: Communication overlap configuration for the model.
        enable_default_comm_overlap: Whether to enable default comm overlap config if none is provided.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=10,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,
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
            num_workers=8,
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer" if use_null_tokenizer else "HuggingFaceTokenizer",
            tokenizer_model=tokenizer_model if not use_null_tokenizer else None,
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE if use_null_tokenizer else None,
        ),
        checkpoint=CheckpointConfig(
            save_interval=10,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if cfg.comm_overlap is None and enable_default_comm_overlap:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_bootstrap_backend="nccl",
            tp_comm_overlap=True,
        )

    return cfg


class NemotronNanoV2FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Nemotron Nano v2 finetuning recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: str | None
    name: str

    # Finetuning-specific
    pretrained_checkpoint: str | None
    packed_sequence: bool
    vocab_file: str | None

    # Training hyperparameters
    train_iters: int
    global_batch_size: int | None
    micro_batch_size: int
    seq_length: int | None
    eval_interval: int
    save_interval: int

    # Optimizer
    finetune_lr: float | None
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None

    # Precision
    precision_config: MixedPrecisionConfig | str | None


def nemotron_nano_9b_v2_finetune_config(**user_kwargs: Unpack[NemotronNanoV2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Nemotron Nano 9B v2.

    Default configuration: 8 nodes, 64 GPUs, Full SFT only
    - Full SFT: TP=2, PP=1, SP=True, LR=1e-4
    """
    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _nemotron_nano_v2_finetune_common(hf_path="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", **user_kwargs)

    # Model-specific parallelism settings
    config.model.tensor_model_parallel_size = 2
    config.model.pipeline_model_parallel_size = 1
    config.model.sequence_parallel = True

    return config


def nemotron_nano_12b_v2_finetune_config(**user_kwargs: Unpack[NemotronNanoV2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Nemotron Nano 12B v2.

    Default configuration: 8 nodes, 64 GPUs, Full SFT only
    - Full SFT: TP=4, PP=1, SP=True, LR=1e-4
    - Uses FP8 precision by default
    """
    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Use FP8 precision by default for 12B
    if "precision_config" not in user_kwargs:
        user_kwargs["precision_config"] = "nanov2_bf16_with_fp8_current_scaling_mixed"

    # Build base config
    config = _nemotron_nano_v2_finetune_common(hf_path="nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base", **user_kwargs)

    # Model-specific parallelism settings
    config.model.tensor_model_parallel_size = 4
    config.model.pipeline_model_parallel_size = 1
    config.model.sequence_parallel = True

    return config


def _nemotron_nano_v2_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Finetuning-specific
    pretrained_checkpoint: str | None = None,
    vocab_file: str | None = None,
    # Training hyperparameters
    train_iters: int = 300,
    global_batch_size: int | None = None,
    micro_batch_size: int = 1,
    seq_length: int | None = None,
    eval_interval: int = 50,
    save_interval: int = 100,
    # Optimizer
    finetune_lr: float | None = None,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = None,
) -> ConfigContainer:
    """
    Create a finetuning configuration for Nemotron Nano v2 models using AutoBridge.

    Args:
        hf_path (str): HuggingFace model path (e.g., "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base").
        dir (str | None): Base directory for saving logs and checkpoints.
        name (str): Name of the finetuning run.
        pretrained_checkpoint (str | None): Path to pretrained checkpoint to load.
        vocab_file (str | None): Path to Tiktoken vocab file. If provided, uses TiktokenTokenizer;
            otherwise uses HuggingFaceTokenizer with hf_path.
        train_iters (int): Total number of training iterations.
        global_batch_size (int | None): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int | None): Sequence length for training data.
        eval_interval (int): Evaluation interval.
        save_interval (int): Checkpoint save interval.
        finetune_lr (float | None): Learning rate for finetuning.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (int | None): Number of iterations over which to decay the LR.
        wandb_project (str | None): Weights & Biases project name.
        wandb_entity (str | None): Weights & Biases entity name.
        wandb_exp_name (str | None): Weights & Biases experiment name.
        precision_config (MixedPrecisionConfig | str | None): Precision configuration for the model.

    Returns:
        ConfigContainer: Configuration for finetuning.
    """
    # Default sequence length for finetuning (Nemotron Nano v2 uses 8K context)
    if seq_length is None:
        seq_length = 8192

    # Default global batch size
    if global_batch_size is None:
        global_batch_size = 768

    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Create model config using AutoBridge (like NemotronH)
    bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    # Precision configuration
    if precision_config is None:
        precision_config = bf16_mixed()
    elif isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    # Sequence length
    model_cfg.seq_length = seq_length

    # Optimizer and scheduler
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=finetune_lr if finetune_lr is not None else 1e-4,
        min_lr=min_lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
    )

    # Dataset configuration (SQuAD by default)
    dataset_config = default_squad_config(seq_length=seq_length, packed_sequence=False)

    # W&B logger configuration
    logger_config = LoggerConfig(
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=10,
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
            overlap_param_gather=False,
            use_distributed_optimizer=True,
        ),
        dataset=dataset_config,
        logger=logger_config,
        tokenizer=TokenizerConfig(
            tokenizer_type="TiktokenTokenizer" if vocab_file else "HuggingFaceTokenizer",
            tokenizer_model=vocab_file if vocab_file else hf_path,
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
        ),
        rng=RNGConfig(seed=5678),  # Different seed for finetuning
        mixed_precision=precision_config,
    )

    # Enable default comm overlap for Nemotron Nano v2
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )

    return cfg
