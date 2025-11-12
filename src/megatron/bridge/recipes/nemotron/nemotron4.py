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

from megatron.bridge.models.nemotron.nemotron_provider import (
    Nemotron4ModelProvider15B,
    Nemotron4ModelProvider340B,
    NemotronModelProvider,
)
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
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
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, get_mixed_precision_config


class Nemotron4CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Nemotron4 recipe helpers."""

    # Core identifiers
    model_provider: NemotronModelProvider
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
    eval_interval: int
    save_interval: int

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None

    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


def nemotron4_15b_pretrain_config(**user_kwargs: Unpack[Nemotron4CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Nemotron4 15B.

    Default configuration: 1 node, 8 GPUs
    - TP=4, PP=1, CP=1, SP=True
    - GBS=32, MBS=2, SeqLen=4096
    - LR=4.5e-5

    This configuration includes performance optimizations:
    - Tensor parallelism communication overlap
    - Manual garbage collection
    - Optimized mixed precision settings

    See `_nemotron4_common` for the full list of parameters.
    """
    recommended_kwargs: Nemotron4CommonKwargs = {
        "model_provider": Nemotron4ModelProvider15B(),
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": True,
        "global_batch_size": 32,
        "micro_batch_size": 2,
        "seq_length": 4096,
        "lr": 4.5e-5,
        "min_lr": 4.5e-5,
        "lr_warmup_iters": 500,
        "train_iters": 300000,
        "comm_overlap_config": CommOverlapConfig(tp_comm_overlap=True),
    }
    combined_kwargs: Nemotron4CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron4_common(**combined_kwargs)


def nemotron4_15b_16k_pretrain_config(**user_kwargs: Unpack[Nemotron4CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Nemotron4 15B with 16K sequence length.

    Default configuration: 1 node, 8 GPUs
    - TP=2, PP=2, CP=2, SP=True
    - GBS=32, MBS=2, SeqLen=16384
    - LR=4.5e-5

    See `_nemotron4_common` for the full list of parameters.
    """
    recommended_kwargs: Nemotron4CommonKwargs = {
        "model_provider": Nemotron4ModelProvider15B(),
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "global_batch_size": 32,
        "micro_batch_size": 2,
        "seq_length": 16384,
        "lr": 4.5e-5,
        "min_lr": 4.5e-5,
        "lr_warmup_iters": 500,
        "train_iters": 300000,
    }
    combined_kwargs: Nemotron4CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron4_common(**combined_kwargs)


def nemotron4_15b_64k_pretrain_config(**user_kwargs: Unpack[Nemotron4CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Nemotron4 15B with 64K sequence length.

    Default configuration: 4 nodes, 8 GPUs per node (32 GPUs total)
    - TP=4, PP=2, CP=4, SP=True
    - GBS=32, MBS=2, SeqLen=65536
    - LR=4.5e-5

    See `_nemotron4_common` for the full list of parameters.
    """
    recommended_kwargs: Nemotron4CommonKwargs = {
        "model_provider": Nemotron4ModelProvider15B(),
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 4,
        "sequence_parallel": True,
        "global_batch_size": 32,
        "micro_batch_size": 2,
        "seq_length": 65536,
        "lr": 4.5e-5,
        "min_lr": 4.5e-5,
        "lr_warmup_iters": 500,
        "train_iters": 300000,
    }
    combined_kwargs: Nemotron4CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron4_common(**combined_kwargs)


def nemotron4_340b_pretrain_config(**user_kwargs: Unpack[Nemotron4CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Nemotron4 340B.

    Default configuration: 768 nodes, 8 GPUs per node (6144 GPUs total)
    - TP=8, PP=12, VPP=8, CP=2, SP=True
    - GBS=2304, MBS=1, SeqLen=4096
    - LR=1.0e-4, min_LR=1.0e-5

    This configuration includes performance optimizations:
    - Tensor parallelism communication overlap with embedding weight gradient deferral
    - Manual garbage collection
    - Optimized mixed precision settings

    See `_nemotron4_common` for the full list of parameters.
    """
    recommended_kwargs: Nemotron4CommonKwargs = {
        "model_provider": Nemotron4ModelProvider340B(),
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 12,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": 8,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "global_batch_size": 2304,
        "micro_batch_size": 1,
        "seq_length": 4096,
        "lr": 1.0e-4,
        "min_lr": 1.0e-5,
        "lr_warmup_iters": 500,
        "train_iters": 100000,
        "comm_overlap_config": CommOverlapConfig(
            tp_comm_overlap=True,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=22,
        ),
    }
    combined_kwargs: Nemotron4CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron4_common(**combined_kwargs)


def _nemotron4_common(
    model_provider: NemotronModelProvider,
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
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Training hyperparameters
    train_iters: int = 300000,
    global_batch_size: int = 32,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 4.5e-5,
    min_lr: float = 4.5e-5,
    lr_warmup_iters: int = 500,
    lr_decay_iters: int | None = None,
    eval_interval: int = 2000,
    save_interval: int = 500,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Nemotron4 models.

    Args:
        model_provider: Instance of NemotronModelProvider (15B, 340B, etc.).
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
        context_parallel_size: Degree of context parallelism.
        sequence_parallel: Whether to use sequence parallelism.
        train_iters: Total number of training iterations.
        global_batch_size: Global batch size for training.
        micro_batch_size: Micro batch size for training.
        seq_length: Sequence length for training data.
        lr: Learning rate.
        min_lr: Minimum learning rate for cosine decay.
        lr_warmup_iters: Number of warmup iterations for the learning rate.
        lr_decay_iters: Number of iterations over which to decay the LR.
        eval_interval: Run validation every N steps.
        save_interval: Save checkpoint every N steps.
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity name.
        wandb_exp_name: Weights & Biases experiment name.
        precision_config: Precision configuration for the model.
        comm_overlap_config: Communication overlap configuration.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "megatron_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    # Use the provider directly and configure parallelism
    model_cfg = model_provider
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=lr,
        min_lr=min_lr,
    )

    # Performance optimizations: disable precision-aware optimizer for models with comm overlap
    if comm_overlap_config is not None and comm_overlap_config.tp_comm_overlap:
        opt_config.use_precision_aware_optimizer = False

    # Setup DDP config with performance optimizations
    ddp_config = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        average_in_collective=True,
        use_distributed_optimizer=True,
    )

    # Apply mixed precision config and performance optimizations
    mixed_precision_cfg = get_mixed_precision_config(precision_config)
    if comm_overlap_config is not None and comm_overlap_config.tp_comm_overlap:
        # Performance optimization: disable FP32 grad reduction
        mixed_precision_cfg.grad_reduce_in_fp32 = False
        ddp_config.grad_reduce_in_fp32 = False

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            # Performance optimization: manual garbage collection
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=ddp_config,
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
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_exp_name=wandb_exp_name,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=mixed_precision_cfg,
    )

    return cfg
