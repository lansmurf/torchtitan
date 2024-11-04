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
    "debugmodel": ModelArgs(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, multiple_of=256, ffn_dim_multiplier=1.3, 
        max_seq_len=2048, rope_theta=500000, vocab_size=128256),
    "124M": ModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        multiple_of=1024,
        ffn_dim_multiplier=1.3,  # We'll explicitly set intermediate_size instead
        norm_eps=1e-5,
        rope_theta=500000.0,
        max_seq_len=2048,
        vocab_size=128256,
        ),
        "1B": ModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        multiple_of=256,
        ffn_dim_multiplier=1.0,  # We'll explicitly set intermediate_size
        norm_eps=1e-5,
        rope_theta=500000.0,
        vocab_size=128256,
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
