"""
Implement model blocks for deepseek

DeepSeek似乎很多地方都要做混合精度，需要注意不能直接全都bf16或者fp32进去
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.model as model


# 仿照DeepSeek V3的代码，设置好一个ModelArgs，直接传Args
# reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
@dataclass
class DSArgs:
    # Basic Setup
    device: str = "cuda:0"
    dtype: str = torch.bfloat16

    # Basic Transformer
    max_batch_len: int = 8
    max_seq_len: int = 1024
    vocab_size: int = 32000
    d_model: int = 768
    d_ff: int = 2048

    # MoE
    n_experts: int = 16
    n_shared_experts: int = 1  # DeepSeekMoE里shared expert
    topk_experts: int = 4  # 选4个激活
    d_moe_ff: int = 512
    route_scale: float = 1.0


# MoE的部分是在DeepSeekMoE代码的基础上按照V4的改动做的


class Gate(nn.Module):
    """
    Compared with V3, DeepSeekV4 applies Sqrt(Softplus) instead of sigmoid

    我的个人理解：

    在大模型中，gating function有个bias term
    我能理解用这个来调节gate的weight

    但是在我的复现实验中，我的计算资源很有限
    我觉得我暂且不要考虑大量expert的负载均衡
    也许这个可以只是简单加一个auxiliary load balancing loss来规避collapse
    而不是维护bias weight
    """

    def __init__(self, args: DSArgs):
        super().__init__()
        self.topk_experts = args.topk_experts
        self.route_scale = args.route_scale

        self.route = model.Linear(
            args.d_model, args.n_experts, device=args.device, dtype=torch.float32
        )

    def forward(self, x: torch.Tensor):
        scores = self.route(x)
        scores = torch.sqrt(F.softplus(scores))
        weights, indices = torch.topk(scores, self.topk_experts, dim=-1)  # (T, num_exp)
        weights *= self.route_scale
        return weights.type_as(x), indices


class HashGate(nn.Module):
    def __init__(self, args: DSArgs):
        super().__init__()
        self.n_experts = args.n_experts
        self.route_scale = args.route_scale

    def forward(self, token_ids: torch.Tensor):
        """
        前面几层直接用HashGate来负载均衡
        """
        indices = (token_ids % self.n_experts).view(-1, 1)
        weights = torch.full(
            (indices.size(0), 1),
            self.route_scale,
            dtype=torch.float32,
            device=indices.device,
        )
        return weights, indices


class Expert(nn.Module):
    def __init__(self, args: DSArgs):
        super().__init__()
        self.ffn = model.SwiGLU(
            args.d_model, args.d_moe_ff, device=args.device, dtype=args.dtype
        )

    def forward(self, x):
        return self.ffn(x)


class DSMoE(nn.Module):
    """
    我的个人理解：

    我觉得浅层的 HashGate 可以加上shared expert
    hash 是为了防止 routing 网络在训练初期就坍缩到少数几个expert

    shared experts 则保证无论 routing 怎么分配，所有 token 都能学到一层公共变换

    我觉得这个不违背设计初衷
    """

    def __init__(self, args: DSArgs, use_hash_routing: bool = False):
        super().__init__()
        self.d_model = args.d_model
        self.use_hash_routing = use_hash_routing
        self.n_experts = args.n_experts

        self.gate = HashGate(args) if use_hash_routing else Gate(args)
        self.experts = nn.ModuleList([Expert(args) for i in range(args.n_experts)])
        self.shared_expert = model.SwiGLU(
            args.d_model,
            args.d_moe_ff * args.n_shared_experts,
            device=args.device,
            dtype=args.dtype,
        )

    def forward(
        self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 似乎tokenwise的处理（或者说，每个token独立的操作）
        # 最好一开始就flatten到tokenwise上
        # 最后再变回去原来的shape
        y = torch.zeros_like(x)
        shape = x.size()
        x = x.view(-1, self.d_model)

        # weights, indices: shape (T, num_experts)
        if self.use_hash_routing:
            assert token_ids is not None, "token_ids required for hash routing"
            token_ids = token_ids.view(-1)
            weights, indices = self.gate(token_ids)
        else:
            weights, indices = self.gate(x)

        # counts: (num_experts) 的列表
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts).tolist()

        for i in range(self.n_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        z = self.shared_experts(x)
        return (y + z).view(shape)
