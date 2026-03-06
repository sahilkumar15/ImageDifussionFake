#!/usr/bin/env bash
set -euo pipefail

# Go to project root (ImageDifussionFake/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

# --------------------------------------------------
# GPU selection (can override from command line)
# Example:
#   ./scripts/train.sh 0,1
#   ./scripts/train.sh 4,5,6,7
# --------------------------------------------------

GPUS="${1:-4,5,6,7}"   # default GPUs if none provided

export CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=8

# Count number of GPUs automatically
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

echo "Using GPUs: $GPUS"
echo "Num processes: $NUM_GPUS"
echo "Project root: $PROJECT_ROOT"

torchrun --nproc_per_node=$NUM_GPUS train.py -c configs/train.yaml


# bash scripts/train.sh
# bash scripts/train.sh 0,1,7,5