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

# Wrapper script to run a single Gemma3 finetuning test with proper environment setup

set -e

# Configure Transformer Engine attention backends
# Enable FusedAttention as FlashAttention 2 doesn't support head_dim=256 on sm89
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0

# Get the directory where this script is located (Megatron-Bridge root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add Megatron-Bridge src to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Default values
MODE="${1:-lora}"
CHECKPOINT_PATH="${2:-/tmp/hf_exports/gemma3_1b}"
NUM_STEPS="${3:-200}"

echo "=================================="
echo "Gemma3 1B Finetuning Test"
echo "=================================="
echo ""
echo "Mode: ${MODE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Steps: ${NUM_STEPS}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "Script dir: ${SCRIPT_DIR}"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please ensure you have converted and saved the checkpoint."
    exit 1
fi

# Run the test
uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 test_gemma3_finetune.py \
    --mode "${MODE}" \
    --pretrained_checkpoint "${CHECKPOINT_PATH}" \
    --num_steps "${NUM_STEPS}"

echo ""
echo "=================================="
echo "Test completed!"
echo "=================================="

