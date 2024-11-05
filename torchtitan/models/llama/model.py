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
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, keys, values, is_causal=True, enable_gqa=True)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

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
        out = h + self.feed_forward(self.ffn_norm(h))
        return out 

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
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
