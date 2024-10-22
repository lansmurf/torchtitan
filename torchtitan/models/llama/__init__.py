# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.llama.model import ModelArgs, Transformer

__all__ = ["Transformer"]

llama2_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16),
    "271M": ModelArgs(dim=1024, n_layers=16, n_heads=8),
    "1B": ModelArgs(dim=2048, n_layers=18, n_heads=16),
    "7B": ModelArgs(dim=4096, n_layers=32, n_heads=32),
    "13B": ModelArgs(dim=5120, n_layers=40, n_heads=40),
    "26B": ModelArgs(dim=5120, n_layers=80, n_heads=40),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
    ),
}

llama3_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "124M": ModelArgs(
        dim=768,  # Reduced from 2048
        n_layers=12,  # Reduced from 16
        n_heads=12,  # Reduced from 32
        n_kv_heads=4,  # Reduced from 8
        multiple_of=128,  # Reduced from 256
        ffn_dim_multiplier=4,  # Keeping the same ratio
        norm_eps=1e-5,  # Same as 1B
        max_seq_len=131072,  # Keeping the same as 1B
        vocab_size=128256,  # Keeping the same as 1B
        rope_theta=500000.0,  # Same as 1B
        rope_scaling={
            "type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
        attention_bias=False,
        mlp_bias=False,
        head_dim=64,  # Keeping the same as 1B
    ),
    "1B": ModelArgs(
        dim=2048,  # hidden_size
        n_layers=16,  # num_hidden_layers
        n_heads=32,  # num_attention_heads
        n_kv_heads=8,  # num_key_value_heads
        multiple_of=256,  # Assuming this based on intermediate_size / hidden_size ratio
        ffn_dim_multiplier=4,  # intermediate_size / hidden_size
        norm_eps=1e-5,  # rms_norm_eps
        max_seq_len=131072,  # max_position_embeddings
        vocab_size=128256,
        rope_theta=500000.0,
        rope_scaling={
            "type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
        attention_bias=False,
        mlp_bias=False,
        head_dim=64,
    ),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": ModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}
