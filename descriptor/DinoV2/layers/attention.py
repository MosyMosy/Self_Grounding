# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, g_info: Tensor = None) -> Tensor:
        g_info_layer = g_info[0]
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, g_info[1:]


class MemEffAttention(Attention):
    def forward(self, x: Tensor, g_info: Tensor = None, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)
        g_info_layer = g_info[0]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        g_info_layer = g_info_layer.reshape(
            g_info_layer.shape[0], self.num_heads, C // self.num_heads
        )
        q_g = g_info_layer[0]
        k_g = g_info_layer[1]
        v_g = g_info_layer[2]
        t_g = g_info_layer[3]

        k_g = k_g.unsqueeze(0).unsqueeze(0)  # .repeat(B, N, 1, 1)
        q_g = q_g.unsqueeze(0).unsqueeze(0)

        k_min = k.flatten(2, 3).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        k_max = k.flatten(2, 3).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        k_g_min = k_g.flatten(2, 3).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        k_g_max = k_g.flatten(2, 3).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        k_g_scaled = (k_g - k_g_min) / (k_g_max - k_g_min)
        k_g_scaled = k_g_scaled * (k_max - k_min) + k_min

        q_min = q.flatten(2, 3).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        q_max = q.flatten(2, 3).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        q_g_min = q_g.flatten(2, 3).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        q_g_max = q_g.flatten(2, 3).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        q_g_scaled = (q_g - q_g_min) / (q_g_max - q_g_min)
        q_g_scaled = q_g_scaled * (q_max - q_min) + q_min

        k_fused = (k + k_g)
        q_fused = (q + q_g)

        if len(g_info) > 0:
            x = memory_efficient_attention(q_fused, k_fused, v)
        else:
            x = memory_efficient_attention(q, k, v)

        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, g_info[1:]
