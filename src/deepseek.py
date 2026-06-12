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

    # mHC
    expansion_rate: int = 4
    use_checkpoint: bool = False


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

        z = self.shared_expert(x)
        return (y + z).view(shape)


## mHC实现
class ManifoldHyperConnections(nn.Module):
    """Manifold Hyper Connections (mHC) wrapping a transformer block.

    Multi-stream representation: streams are stored in the batch dimension
    as ``(batch * n, seq_len, d_model)``. This lets each stream pass through
    the block independently while the width/depth connections handle mixing.

    Boundary operations ``expand_streams`` / ``reduce_streams`` are called
    exactly once: expand before the first mHC layer, reduce after the last.
    """

    def __init__(self, args: DSArgs, block: nn.Module):
        super().__init__()
        self.n = args.expansion_rate
        self.d_model = args.d_model
        self.block = block
        self.use_checkpoint = args.use_checkpoint

        self.flatten_dim = args.d_model * self.n

        self.phi_pre = model.Linear(
            self.flatten_dim, self.n, device=args.device, dtype=args.dtype
        )
        self.phi_post = model.Linear(
            self.flatten_dim, self.n, device=args.device, dtype=args.dtype
        )
        self.phi_res = model.Linear(
            self.flatten_dim,
            self.n**2,
            device=args.device,
            dtype=args.dtype,
        )

        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))

        self.bias_pre = nn.Parameter(torch.zeros(self.n))
        self.bias_post = nn.Parameter(torch.zeros(self.n))
        self.bias_res = nn.Parameter(torch.zeros(self.n, self.n))

        self.norm = model.RMSNorm(
            args.d_model * self.n, device=args.device, dtype=args.dtype
        )

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def expand_streams(x: torch.Tensor, n: int) -> torch.Tensor:
        """(batch, seq_len, d) -> (batch * n, seq_len, d)

        Creates n identical copies of each sequence. The copies diverge
        after the first mHC layer's depth connection applies h_post.
        """
        batch, seq_len, d = x.shape
        return (
            x.unsqueeze(1)
            .expand(batch, n, seq_len, d)
            .reshape(batch * n, seq_len, d)
        )

    @staticmethod
    def reduce_streams(x: torch.Tensor, n: int) -> torch.Tensor:
        """(batch * n, seq_len, d) -> (batch, seq_len, d)

        Averages the n (now genuinely different) streams back into one.
        Called once after the final mHC layer.
        """
        bn, seq_len, d = x.shape
        return x.view(bn // n, n, seq_len, d).mean(dim=1)

    # ------------------------------------------------------------------
    # Sinkhorn-Knopp (log-space, column-first per the paper)
    # ------------------------------------------------------------------

    def sinkhorn_knopp(self, logits: torch.Tensor, iters: int = 20) -> torch.Tensor:
        """Log-space Sinkhorn-Knopp normalization.

        Matches Eq. (9): T_c (column norm) first, then T_r (row norm).
        Stays in log-space for numerical stability.
        Functional (not in-place) to preserve the autograd graph.
        """
        for _ in range(iters):
            logits = logits - logits.logsumexp(dim=-2, keepdim=True)  # T_c: columns
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)  # T_r: rows
        return logits.exp()

    # ------------------------------------------------------------------
    # Kernel computation (the lightweight, recomputable part)
    # ------------------------------------------------------------------

    def _compute_kernels(
        self, x_normed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute dynamic mHC kernels from RMSNorm-ed flattened input.

        Args:
            x_normed: (batch, seq_len, n * d_model)

        Returns:
            h_pre:  (batch, seq_len, n)          — sigmoid,  in (0, 1)
            h_post: (batch, seq_len, n)          — 2*sigmoid, in (0, 2)
            h_res:  (batch, seq_len, n, n)       — doubly stochastic
        """
        h_pre_tilde = self.alpha_pre * self.phi_pre(x_normed) + self.bias_pre
        h_post_tilde = self.alpha_post * self.phi_post(x_normed) + self.bias_post

        batch, seq_len, _ = x_normed.shape
        h_res_tilde = (
            self.alpha_res
            * self.phi_res(x_normed).view(batch, seq_len, self.n, self.n)
            + self.bias_res
        )

        h_pre = torch.sigmoid(h_pre_tilde)
        h_post = 2 * torch.sigmoid(h_post_tilde)
        h_res = self.sinkhorn_knopp(h_res_tilde)

        return h_pre, h_post, h_res

    # ------------------------------------------------------------------
    # Width connection – mixes streams BEFORE the block
    # ------------------------------------------------------------------

    def _width_connection(
        self,
        x: torch.Tensor,
        h_pre: torch.Tensor,
        h_res: torch.Tensor,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix the n residual streams into a single branch input + updated residuals.

        Args:
            x:      (batch * n, seq_len, d_model)
            h_pre:  (batch, seq_len, n)
            h_res:  (batch, seq_len, n, n)
            batch:  original batch size (without stream multiplier)

        Returns:
            branch_input:    (batch, seq_len, d_model) – weighted combination
            mixed_residuals: (batch * n, seq_len, d_model) – h_res-mixed streams
        """
        seq_len = x.shape[1]

        # (b*n, s, d) -> (b, s, n, d)
        x_by_seq = x.view(batch, self.n, seq_len, self.d_model).permute(0, 2, 1, 3)

        # Pre-gate: weighted sum over n streams -> single block input
        branch_input = torch.einsum("bsn,bsnd->bsd", h_pre, x_by_seq)

        # Residual mix via doubly stochastic matrix
        mixed_by_seq = torch.einsum("bsij,bsjd->bisd", h_res, x_by_seq)
        mixed_residuals = mixed_by_seq.reshape(
            batch * self.n, seq_len, self.d_model
        )

        return branch_input, mixed_residuals

    # ------------------------------------------------------------------
    # Depth connection – distributes block output AFTER the block
    # ------------------------------------------------------------------

    def _depth_connection(
        self,
        branch_output: torch.Tensor,
        mixed_residuals: torch.Tensor,
        h_post: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """Distribute block output across n streams and add to mixed residuals.

        Args:
            branch_output:   (batch, seq_len, d_model)
            mixed_residuals: (batch * n, seq_len, d_model)
            h_post:          (batch, seq_len, n)
            batch:           original batch size

        Returns:
            (batch * n, seq_len, d_model) – updated multi-stream representation
        """
        seq_len = branch_output.shape[1]

        # Distribute: einsum outputs (b, n, s, d) → reshape to (b*n, s, d)
        post_block = torch.einsum("bsd,bsn->bnsd", branch_output, h_post)
        post_block = post_block.reshape(
            batch * self.n, seq_len, self.d_model
        )

        return mixed_residuals + post_block

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Multi-stream mHC forward.

        Args:
            x: (batch * n, seq_len, d_model)

        Returns:
            (batch * n, seq_len, d_model) — streams preserved, no collapse.
        """
        bn, seq_len, _ = x.shape
        batch = bn // self.n

        # Flatten streams for kernel computation: (b*n, s, d) -> (b, s, n*d)
        x_flat = x.view(batch, seq_len, self.flatten_dim)
        x_normed = self.norm(x_flat)

        if self.training and self.use_checkpoint:
            h_pre, h_post, h_res = torch.utils.checkpoint.checkpoint(
                self._compute_kernels, x_normed, use_reentrant=False
            )
        else:
            h_pre, h_post, h_res = self._compute_kernels(x_normed)

        branch_input, mixed_residuals = self._width_connection(
            x, h_pre, h_res, batch
        )
        branch_output = self.block(branch_input, *args, **kwargs)
        return self._depth_connection(
            branch_output, mixed_residuals, h_post, batch
        )
