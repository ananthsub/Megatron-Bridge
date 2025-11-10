#!/usr/bin/env python3
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
Test script for Gemma3 1B finetuning with both LoRA and full SFT.

Usage:
    # Test with LoRA (default)
    torchrun --nproc_per_node=2 test_gemma3_finetune.py --mode lora

    # Test with full SFT
    torchrun --nproc_per_node=2 test_gemma3_finetune.py --mode sft
"""

import argparse
import os
import sys

from megatron.bridge.recipes.gemma import gemma3_1b_finetune_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def parse_args():
    parser = argparse.ArgumentParser(description="Test Gemma3 1B finetuning")
    parser.add_argument(
        "--mode",
        type=str,
        default="lora",
        choices=["lora", "sft"],
        help="Finetuning mode: 'lora' for LoRA finetuning, 'sft' for full supervised finetuning",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="/tmp/hf_exports/gemma3_1b",
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of training steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: /tmp/gemma3_finetune_{mode})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine PEFT setting based on mode
    if args.mode == "lora":
        peft_setting = "lora"
        run_name = "gemma3_1b_lora_test"
    else:  # sft
        peft_setting = None
        run_name = "gemma3_1b_sft_test"

    # Set output directory
    if args.output_dir is None:
        output_dir = f"/tmp/gemma3_finetune_{args.mode}"
    else:
        output_dir = args.output_dir

    print("=" * 80)
    print("Testing Gemma3 1B Finetuning")
    print(f"Mode: {args.mode.upper()}")
    print(f"PEFT: {peft_setting}")
    print(f"Steps: {args.num_steps}")
    print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Verify checkpoint exists
    if not os.path.exists(args.pretrained_checkpoint):
        print(f"ERROR: Pretrained checkpoint not found at {args.pretrained_checkpoint}")
        print("Please ensure you have converted and saved the checkpoint.")
        sys.exit(1)

    # Configure environment for Transformer Engine attention backends
    # Enable FusedAttention as FlashAttention 2 doesn't support head_dim=256 on sm89
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_FLASH_ATTN"] = "0"  # Disable FlashAttention 2 due to head_dim=256 limitation

    # Configure W&B logging (hardcoded as requested)
    wandb_project = "gemma3_1b_finetune"
    wandb_entity = "joc"  # Set to your W&B username/team if needed
    wandb_exp_name = f"{run_name}_steps{args.num_steps}"

    os.environ["WANDB_API_KEY"] = "ce93efc376a2de5d75438b27faf967fc27b93e62"

    # Create finetuning configuration
    config = gemma3_1b_finetune_config(
        name=run_name,
        dir=output_dir,
        pretrained_checkpoint=args.pretrained_checkpoint,
        peft=peft_setting,
        train_iters=args.num_steps,
        eval_interval=50,  # Evaluate every 50 steps
        save_interval=100,  # Save checkpoint every 100 steps
        # W&B configuration
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
        # Use unpacked sequences for testing
        packed_sequence=False,
        seq_length=2048,  # Shorter sequence for faster testing
        global_batch_size=16,  # Smaller batch for 2 GPUs
        micro_batch_size=1,
        # Parallelism settings for 2 GPUs
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    config.model.vocab_size = 262_145

    print("\nStarting finetuning...")
    print("Configuration:")
    print(f"  - Seq length: {config.model.seq_length}")
    print(f"  - Global batch size: {config.train.global_batch_size}")
    print(f"  - Micro batch size: {config.train.micro_batch_size}")
    print(f"  - TP: {config.model.tensor_model_parallel_size}")
    print(f"  - PP: {config.model.pipeline_model_parallel_size}")
    if config.peft:
        print(f"  - PEFT dim: {config.peft.dim}")
        print(f"  - PEFT alpha: {config.peft.alpha}")
    print()

    # Run finetuning
    finetune(config, forward_step)

    print("\n" + "=" * 80)
    print("Finetuning completed successfully!")
    print(f"Checkpoints saved to: {os.path.join(output_dir, run_name, 'checkpoints')}")
    print(f"TensorBoard logs: {os.path.join(output_dir, run_name, 'tb_logs')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
