# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.norms import build_norm
from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    norm_type: str = "rmsnorm"

    # Add new param for FFN scaling
    ffn_scaling_fn: Optional[str] = None  # e.g. "s_curve", "linear", etc
    
    # Mixture of Depths parameters
    use_mixture_of_depths: bool = False  # Toggle for MoD
    capacity_ratio: float = 1.0  # Default capacity ratio (process all tokens)
    route_every_n_layers: int = 1  # Route every n layers (1 means route every layer)

    # Fuse head
    use_fused_head: bool = True  # New parameter to control head fusion


def get_ffn_scaling_fn(name: Optional[str]):
    if name is None:
        # Default function needs all args to match signature
        return lambda layer_id, n_layers, base_dim, model_dim, multiple_of: base_dim

    def s_curve_multiple_constrained(layer_id, n_layers, base_dim, model_dim, multiple_of):
        # Function from before, unchanged
        dims = []
        for l in range(n_layers):
            x = (l - n_layers*0.7) / (n_layers/5)
            scale = 0.5 * (1 + np.tanh(x))
            if l < n_layers//3:
                dim = base_dim * (0.7 + scale * 1.3)
            else:
                dim = base_dim * (1 + scale * 1.5)
            dims.append(dim)
        
        target_params = n_layers * (model_dim * base_dim * 2)
        current_params = sum((model_dim * dim + dim * model_dim) for dim in dims)
        scale_factor = target_params / current_params
        
        scaled_dims = [multiple_of * round(dim * scale_factor / multiple_of) for dim in dims]
        return scaled_dims[layer_id]

    def s_curve_simple(layer_id, n_layers, base_dim, model_dim, multiple_of):
        # Simple version needs all args to match signature even if unused
        x = (layer_id - n_layers*0.7) / (n_layers/5)
        scale = 0.5 * (1 + np.tanh(x))
        dim = base_dim * (1 + scale * (1.5 if layer_id >= n_layers//3 else 1.3))
        return multiple_of * round(dim / multiple_of)

    def linear_scale(layer_id, n_layers, base_dim, model_dim, multiple_of):
        # Linear version needs all args to match signature even if unused
        dim = base_dim * (0.7 + 1.3 * layer_id / n_layers)
        return multiple_of * round(dim / multiple_of)

    def stepped_scale(layer_id, n_layers, base_dim, model_dim, multiple_of):
        # Define the tiers - threshold: scale factor
        tiers = {
            0: 0.7,    # First third - reduced size
            n_layers // 3: 1.0,  # Second third - baseline size
            2 * n_layers // 3: 1.4,  # Final third - enlarged
            int(0.85 * n_layers): 1.8  # Last few layers - much larger
        }
        
        # Find appropriate tier
        scale = tiers[0]  # Default to first tier
        for threshold, tier_scale in tiers.items():
            if layer_id >= threshold:
                scale = tier_scale
                
        dim = base_dim * scale
        
        # Adjust for param count to match standard FFN
        total_params = n_layers * (model_dim * base_dim * 2)  # target params
        correction = 1.0  # Could add correction factor if needed
        
        return multiple_of * round(dim * correction / multiple_of)

    scaling_fns = {
        "s_curve": s_curve_simple,
        "linear": linear_scale,
        "s_curve_constrained": s_curve_multiple_constrained,
        "stepped": stepped_scale,
    }
    
    assert name in scaling_fns, f"Unknown scaling function: {name}. Available: {list(scaling_fns.keys())}"
    return scaling_fns[name]

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    #print(f"\nprecompute_freqs_cis input dims: dim={dim}, end={end}")
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    #print(f"freqs shape: {freqs.shape}")
    
    t = torch.arange(end, device=freqs.device)
    #print(f"t shape: {t.shape}")
    
    freqs = torch.outer(t, freqs).float()
    #print(f"freqs after outer shape: {freqs.shape}")
    
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    #print(f"freqs_cis final shape: {freqs_cis.shape}")
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    #print(f"\nreshape_for_broadcast input shapes:")
    #print(f"freqs_cis: {freqs_cis.shape}")
    #print(f"x: {x.shape}")
    
    ndim = x.ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    #print(f"freqs_cis after slice: {freqs_cis.shape}")
    #print(f"expecting shape: ({seqlen}, {x.shape[-1]})")
    
    if freqs_cis.shape != (seqlen, x.shape[-1]):
        raise ValueError(f"Shape mismatch: freqs_cis {freqs_cis.shape} vs expected {(seqlen, x.shape[-1])}")

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    result = freqs_cis.view(*shape)
    #print(f"result shape: {result.shape}")
    return result


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #print(f"\napply_rotary_emb input shapes:")
    #print(f"xq: {xq.shape}")
    #print(f"xk: {xk.shape}")
    #print(f"freqs_cis: {freqs_cis.shape}")
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    #print(f"xq_ after complex: {xq_.shape}")

    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    #print(f"xk_ after complex: {xk_.shape}")
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    #print(f"freqs_cis after reshape: {freqs_cis.shape}")
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    #print(f"outputs: xq={xq_out.shape}, xk={xk_out.shape}")
    return xq_out.type_as(xq), xk_out.type_as(xk)

def _precompute_freqs_cis(self): 
    """
    The head dimension is halved for differential attention, so we need to adjust
    the dimension for freqs_cis computation accordingly.
    """
    return precompute_freqs_cis(
        self.head_dim * 2,  # Use full head dimension for freq computation
        self.model_args.max_seq_len,
        self.model_args.rope_theta,
    )

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats the kv heads n_rep times."""
    #print(f"\nIn repeat_kv:")
    # Input expected to be [bs, slen, n_kv_heads, head_dim]
    bs, n_kv_heads, slen, head_dim = x.shape  # Note: input should be transposed already
    #print(f"input shape: bs={bs}, n_kv_heads={n_kv_heads}, slen={slen}, head_dim={head_dim}")
    #print(f"n_rep: {n_rep}")
    
    if n_rep == 1:
        #print("n_rep=1, returning input unchanged")
        return x
    
    # Shape progression:
    # 1. unsqueeze: [bs, n_kv_heads, slen, 1, head_dim]
    # 2. expand: [bs, n_kv_heads, slen, n_rep, head_dim]
    # 3. reshape: [bs, n_kv_heads * n_rep, slen, head_dim]
    
    expanded = torch.unsqueeze(x, dim=3)
    #print(f"after unsqueeze: {expanded.shape}")
    
    expanded = expanded.expand(bs, n_kv_heads, slen, n_rep, head_dim)
    #print(f"after expand: {expanded.shape}")
    
    result = expanded.reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    #print(f"after reshape: {result.shape}")
    
    return result

def lambda_init_fn(depth):
    """Initialize lambda based on depth, with a minimum value to avoid division by zero"""
    return 0.8 - 0.6 * math.exp(-0.3 * max(1, depth))

class DifferentialAttention(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else model_args.n_heads
        self.head_dim = model_args.dim // model_args.n_heads // 2  # Halved head dimension
        self.scale = self.head_dim ** -0.5
        
        # Linear projections using native GQA dimensioning
        self.wq = nn.Linear(model_args.dim, model_args.dim, bias=False)
        self.wk = nn.Linear(
            model_args.dim, 
            (model_args.dim // model_args.n_heads) * self.n_kv_heads, 
            bias=False
        )
        self.wv = nn.Linear(
            model_args.dim, 
            (model_args.dim // model_args.n_heads) * self.n_kv_heads, 
            bias=False
        )
        self.wo = nn.Linear(model_args.dim, model_args.dim, bias=False)

        # Lambda parameters
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init_fn(model_args.n_layers)

        # Sublayer norm
        self.subln = build_norm(model_args.norm_type, 2 * self.head_dim, eps=model_args.norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[BlockMask], freqs_cis: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seqlen, embed_dim = x.shape
        
        assert mask is not None, "No mask has been passed!"
        
        # Linear projections
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Initial reshape
        q = q.view(bsz, seqlen, 2 * self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, 2 * self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, 2, self.head_dim)

        # Apply rotary embeddings if provided
        if freqs_cis is not None:
            freqs_cis = freqs_cis[:, :freqs_cis.size(1)//2]
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # Split queries and keys for dual attention streams
        q = q.reshape(bsz, seqlen, self.n_heads, 2, self.head_dim)
        k = k.reshape(bsz, seqlen, self.n_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # Prepare for attention (only transpose, no repeat_kv needed)
        q1 = q1.transpose(1, 2)
        q2 = q2.transpose(1, 2)
        k1 = k1.transpose(1, 2)
        k2 = k2.transpose(1, 2)
        v1 = v1.transpose(1, 2)
        v2 = v2.transpose(1, 2)

        # Compute attentions with enable_gqa=True
        attn11 = flex_attention(q1, k1, v1, scale=self.scale, block_mask=mask, enable_gqa=True)
        attn12 = flex_attention(q1, k1, v2, scale=self.scale, block_mask=mask, enable_gqa=True)
        attn1 = torch.cat([attn11, attn12], dim=-1)
        
        attn21 = flex_attention(q2, k2, v1, scale=self.scale, block_mask=mask, enable_gqa=True)
        attn22 = flex_attention(q2, k2, v2, scale=self.scale, block_mask=mask, enable_gqa=True)
        attn2 = torch.cat([attn21, attn22], dim=-1) 

        # Apply differential attention
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        attn = attn1 - lambda_full * attn2
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        
        # Reshape to final dimensions
        attn = attn.transpose(1, 2)
        attn = attn.reshape(bsz, seqlen, self.n_heads * 2 * self.head_dim)
        
        return self.wo(attn)

    def init_weights(self, init_std: float):
        # Initialize projections
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)


# revise: https://x.com/_xjdr/status/1860754674506465754
# softmax check on fp32

class FlexAttention(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[BlockMask], freqs_cis: Optional[torch.Tensor]
    ):

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = flex_attention(xq, xk, xv, block_mask=mask, scale=self.scale, enable_gqa=True)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class Attention(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[BlockMask], freqs_cis: Optional[torch.Tensor]
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to allow for head splitting
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Rearrange for attention computation
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True, enable_gqa=True)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self, 
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        layer_id: Optional[int] = None,
        n_layers: Optional[int] = None, 
        scaling_fn: Optional[str] = None,
    ):
        super().__init__()
        
        # Apply base hidden dim calculation
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            
        # Apply scaling function if layer info provided
        if layer_id is not None and n_layers is not None and scaling_fn is not None:
            scale_fn = get_ffn_scaling_fn(scaling_fn)
            hidden_dim = scale_fn(layer_id, n_layers, hidden_dim, dim, multiple_of)
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)).square() * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        
        # Only use fused FFN for last layer when fused head is enabled
        is_last_layer = layer_id == model_args.n_layers - 1
        if is_last_layer and model_args.use_fused_head:
            self.feed_forward = FusedFeedForward(
                dim=model_args.dim,
                hidden_dim=4 * model_args.dim,
                multiple_of=model_args.multiple_of,
                ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            )
            self.hidden_norm = build_norm(
                model_args.norm_type, 
                dim=self.feed_forward.hidden_dim, 
                eps=model_args.norm_eps
            )
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=4 * model_args.dim,
                multiple_of=model_args.multiple_of,
                ffn_dim_multiplier=model_args.ffn_dim_multiplier,
                layer_id=layer_id,
                n_layers=model_args.n_layers,
            )
            self.hidden_norm = None

        self.layer_id = layer_id
        self.num_layers = model_args.n_layers
        self.is_last_layer = is_last_layer
        self.use_fused_head = model_args.use_fused_head

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(self, x: torch.Tensor, mask: Optional[BlockMask], freqs_cis: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), mask, freqs_cis)
        
        # Only use fused path for last layer when fused head is enabled
        if self.is_last_layer and self.use_fused_head:
            # Before FFN
            h_normed = self.ffn_norm(h)
            print("Pre-FFN norm stats:", 
                h_normed.mean().item(), h_normed.std().item())
            
            # After FFN 
            h_ffn = self.feed_forward(h_normed)
            print("Post-FFN stats:", 
                h_ffn.mean().item(), h_ffn.std().item(),
                "\nFFN w1 norm:", self.feed_forward.w1.weight.norm().item())
            
            # After hidden norm
            h_norm = self.hidden_norm(h_ffn)
            print("Post-hidden norm stats:",
                h_norm.mean().item(), h_norm.std().item())
            
            # Sample of actual values to verify they're changing
            print("Sample outputs:", h_norm[0,0,:5].tolist())
            
            return h_norm
        else:
            # Regular path for all other layers
            return h + self.feed_forward(self.ffn_norm(h))
        
    def init_weights(self):
            for norm in (self.attention_norm, self.ffn_norm):
                norm.reset_parameters()
            self.attention.init_weights(self.weight_init_std)
            self.feed_forward.init_weights(self.weight_init_std)

class FusedFeedForward(nn.Module):
    def __init__(
        self, 
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.hidden_dim = hidden_dim  # Store for access by Transformer
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        # No down projection needed as it's fused with head

    def forward(self, x):
        # Only up project and activate
        return F.relu(self.w1(x)).square() * self.w3(x)
        
    def init_weights(self, init_std: float):
        for linear in (self.w1, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.use_fused_head = model_args.use_fused_head

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        # Output head configuration
        if self.use_fused_head:
            # Project from hidden_dim of last layer's FFN
            last_layer = self.layers[str(self.n_layers - 1)]
            assert isinstance(last_layer.feed_forward, FusedFeedForward)
            hidden_dim = last_layer.feed_forward.hidden_dim
            self.output = nn.Linear(hidden_dim, model_args.vocab_size, bias=False)
            self.norm = None  # No final norm needed for fused head
        else:
            # Regular output head
            self.norm = build_norm(
                model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
            )
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, mask: Optional[BlockMask]):
        h = self.tok_embeddings(tokens)
        
        for layer in self.layers.values():
            h = layer(h, mask, self.freqs_cis)
        
        if not self.use_fused_head:
            h = self.norm(h)
            
        return self.output(h)

    def init_weights(self, buffer_device: Optional[torch.device] = None):
        buffer_device = buffer_device or next(self.parameters()).device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
            
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        if self.output is not None:
            nn.init.zeros_(self.output.weight)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[BlockMask]):
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, mask, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
