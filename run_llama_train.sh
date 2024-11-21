#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Auto-detect number of available CUDA devices
get_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l
    else
        echo "0"
    fi
}

# use envs as local overrides for convenience
# NGPU will now auto-detect if not manually specified
# e.g.
# LOG_RANK=0,1 ./run_llama_train.sh
NGPU=${NGPU:-$(get_gpu_count)}
if [ "$NGPU" -eq "0" ]; then
    echo "No CUDA devices detected. Please check your GPU installation."
    exit 1
fi

LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

echo "Running training with $NGPU GPU(s)"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides