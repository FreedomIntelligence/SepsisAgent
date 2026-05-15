#!/bin/bash
# One-click launcher for the SepsisAgent inference demo
# Usage:
#   bash run_inference.sh /path/to/SepsisAgent-4B [num_gpus]

set -e

MODEL_PATH=${1:-"FreedomIntelligence/SepsisAgent-4B"}
NUM_GPUS=${2:-1}
MODEL_NAME="SepsisAgent-4B"

cd "$(dirname "$0")"

echo "======================================================================"
echo "SepsisAgent Inference Demo"
echo "======================================================================"
echo "Model path:  $MODEL_PATH"
echo "Num GPUs:    $NUM_GPUS"
echo "======================================================================"

python inference.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --num_gpus "$NUM_GPUS" \
    --base_port 8000 \
    --test
