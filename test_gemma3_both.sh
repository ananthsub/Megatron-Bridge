#!/bin/bash
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

# Test script to run both LoRA and full SFT finetuning tests

set -e

# Configure Transformer Engine attention backends
# Enable FusedAttention as FlashAttention 2 doesn't support head_dim=256 on sm89
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0

CHECKPOINT_PATH="/tmp/hf_exports/gemma3_1b"
NUM_STEPS=200

echo "=================================="
echo "Gemma3 1B Finetuning Test Suite"
echo "=================================="
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please ensure you have converted and saved the checkpoint."
    exit 1
fi

# Test 1: LoRA finetuning
echo "Test 1: LoRA Finetuning"
echo "=================================="
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 test_gemma3_finetune.py \
    --mode lora \
    --pretrained_checkpoint "$CHECKPOINT_PATH" \
    --num_steps "$NUM_STEPS"

echo ""
echo "=================================="
echo "LoRA test completed!"
echo "=================================="
echo ""
sleep 2

# Test 2: Full SFT
echo "Test 2: Full SFT"
echo "=================================="
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 test_gemma3_finetune.py \
    --mode sft \
    --pretrained_checkpoint "$CHECKPOINT_PATH" \
    --num_steps "$NUM_STEPS"

echo ""
echo "=================================="
echo "Full SFT test completed!"
echo "=================================="
echo ""

echo "All tests completed successfully!"
echo ""
echo "Results:"
echo "  LoRA checkpoint: /tmp/gemma3_finetune_lora/gemma3_1b_lora_test/checkpoints"
echo "  SFT checkpoint:  /tmp/gemma3_finetune_sft/gemma3_1b_sft_test/checkpoints"
echo ""
echo "View logs with:"
echo "  tensorboard --logdir /tmp/gemma3_finetune_lora/gemma3_1b_lora_test/tb_logs"
echo "  tensorboard --logdir /tmp/gemma3_finetune_sft/gemma3_1b_sft_test/tb_logs"

