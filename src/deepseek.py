"""
Implement model blocks for deepseek

DeepSeek似乎很多地方都要做混合精度，需要注意不能直接全都bf16或者fp32进去
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Optional, Literal

import torch

torch.set_default_device("cuda:0")
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
    max_batch_len: int = 4
    max_seq_len: int = 2048
    vocab_size: int = 32000
    d_model: int = 2048
    d_ff: int = 2048
    n_layer: int = 7
    n_mtp_layer: int = 1
    n_hash_layer: int = 1

    # MoE
    n_experts: int = 8
    n_shared_experts: int = 1
    topk_experts: int = 2
    d_moe_ff: int = 2048
    route_scale: float = 1.0

    # mHC
    expansion_rate: int = 4
    use_checkpoint: bool = True

    # Attention
    n_heads: int = 16
    head_dim: int = 256
    rope_head_dim: int = 64
    index_head_dim: int = 128
    index_num: int = 16
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    attn_rank: int = 512
    index_topk: int = 256
    window_size: int = 128
    output_group: int = 4
    output_lora: int = 512

    # rope
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1

    eps: float = 1e-6


# MoE的部分是在DeepSeekMoE代码的基础上按照V4的改动做的


class Gate(nn.Module):
    """
    Compared with V3, DeepSeekV4 applies Sqrt(Softplus) instead of sigmoid.

    bias 用于负载均衡：overloaded expert 降低 bias，减少被选中概率；
    underloaded expert 提高 bias，增加被选中概率。
    bias 仅影响 expert 选择（topk），不影响路由权重。
    """

    def __init__(self, args: DSArgs):
        super().__init__()
        self.topk_experts = args.topk_experts
        self.route_scale = args.route_scale
        self.n_experts = args.n_experts

        self.route = model.Linear(
            args.d_model, args.n_experts, device=args.device, dtype=torch.float32
        )
        self.bias = nn.Parameter(torch.zeros(args.n_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        scores = self.route(x.float())
        scores = torch.sqrt(F.softplus(scores))
        original_scores = scores

        # bias 仅用于 expert 选择，不影响权重
        scores = scores + self.bias

        _, indices = torch.topk(scores, self.topk_experts, dim=-1)

        # 权重取自无 bias 的原始分数
        weights = original_scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights.type_as(x), indices


class HashGate(nn.Module):
    def __init__(self, args: DSArgs):
        super().__init__()
        self.n_experts = args.n_experts
        self.topk = args.topk_experts
        self.route_scale = args.route_scale

    def forward(self, token_ids: torch.Tensor):
        """
        前面几层直接用 HashGate 来负载均衡。

        用确定性 hash 给每个 token 映射 topk 个不同的 expert，
        等价于维护了一张 (vocab_size, topk) 的随机映射表。
        """
        T = token_ids.numel()
        # 对每个 rank 用不同的质数乘子做 hash，保证 topk 个 expert 互不相同
        indices = torch.stack(
            [
                (token_ids * (157 + i * 199) + i * 1063) % self.n_experts
                for i in range(self.topk)
            ],
            dim=-1,
        )  # (T, topk)
        # 均匀权重，归一化后总和为 route_scale
        weight = self.route_scale / self.topk
        weights = torch.full(
            (T, self.topk), weight, dtype=torch.float32, device=indices.device
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # 似乎tokenwise的处理（或者说，每个token独立的操作）
        # 最好一开始就flatten到tokenwise上
        # 最后再变回去原来的shape
        token_ids: Optional[torch.Tensor] = kwargs.get("token_ids", None)
        shape = x.size()
        x = x.view(-1, self.d_model)
        y = torch.zeros_like(x)

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

        self.hc_norm_eps = args.eps
        self.block_norm = model.RMSNorm(
            args.d_model, args.eps, device=args.device, dtype=args.dtype
        )

    def sinkhorn_knopp(
        self, logits: torch.Tensor, iters: int = 20, eps: float = 1e-6
    ) -> torch.Tensor:
        """Log-space Sinkhorn-Knopp normalization.

        Matches Eq. (9): T_c (column norm) first, then T_r (row norm).
        Stays in log-space for numerical stability.
        Functional (not in-place) to preserve the autograd graph.
        """
        for _ in range(iters):
            logits = logits - logits.logsumexp(dim=-2, keepdim=True)  # T_c: columns
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)  # T_r: rows
        return logits.exp() + eps

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
            self.alpha_res * self.phi_res(x_normed).view(batch, seq_len, self.n, self.n)
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
            x:      (batch, seq_len, n, d_model)
            h_pre:  (batch, seq_len, n)
            h_res:  (batch, seq_len, n, n)
            batch:  original batch size (without stream multiplier)

        Returns:
            branch_input:    (batch, seq_len, d_model) – weighted combination
            mixed_residuals: (batch, seq_len, n, d_model) – h_res-mixed streams
        """
        seq_len = x.shape[1]

        # Pre-gate: weighted sum over n streams -> single block input
        branch_input = torch.einsum("bsn,bsnd->bsd", h_pre, x)

        # Residual mix via doubly stochastic matrix
        mixed_by_seq = torch.einsum("bsij,bsjd->bsid", h_res, x)

        return branch_input, mixed_by_seq

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
            mixed_residuals: (batch, seq_len, n, d_model)
            h_post:          (batch, seq_len, n)
            batch:           original batch size

        Returns:
            (batch, seq_len, n, d_model) – updated multi-stream representation
        """

        post_block = torch.einsum("bsd,bsn->bsnd", branch_output, h_post)

        return mixed_residuals + post_block

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Multi-stream mHC forward.

        Args:
            x: (batch, seq_len, n, d_model)

        Returns:
            (batch, seq_len, n, d_model) — streams preserved, no collapse.
        """
        batch, seq_len, n, d = x.shape

        # Flatten streams for kernel computation: (b, s, n, d) -> (b, s, n*d)
        x_flat = x.view(batch, seq_len, self.flatten_dim)
        # 与官方一致：用 rsqrt 做归一化，无可学习参数
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.hc_norm_eps)
        x_normed = x_flat * rsqrt

        if self.training and self.use_checkpoint:
            h_pre, h_post, h_res = torch.utils.checkpoint.checkpoint(
                self._compute_kernels, x_normed, use_reentrant=False
            )
        else:
            h_pre, h_post, h_res = self._compute_kernels(x_normed)

        branch_input, mixed_residuals = self._width_connection(x, h_pre, h_res, batch)
        branch_input = self.block_norm(branch_input)
        branch_output = self.block(branch_input, *args, **kwargs)
        return self._depth_connection(branch_output, mixed_residuals, h_post, batch)


class Compressor(nn.Module):
    """
    Implementation of KV compressor

    Code基本上使用了官方的实现来处理交叉块和decode时的缓存
    """

    def __init__(
        self,
        args: DSArgs,
        head_dim: int,
        compress_ratio: int,
        rope: model.RotaryPositionalEmbedding,
    ):
        super().__init__()
        self.d_model = args.d_model
        self.head_dim = head_dim  # 文章Section 2.3.1的head dim c
        self.compress_ratio = compress_ratio  # 文章Section 2.3.1的m
        self.rope_head_dim = args.rope_head_dim
        self.rope = rope
        self.overlap = compress_ratio == 4
        self.eps = args.eps
        coff = 1 + self.overlap  # 2 when overlap, 1 otherwise

        # 要同时维护Ca, Cb, Za, Zb，所以是两倍的head_dim（仅overlap时）
        self.weight_kv = model.Linear(
            self.d_model, coff * self.head_dim, dtype=torch.float32, device=args.device
        )
        self.weight_z = model.Linear(
            self.d_model, coff * self.head_dim, dtype=torch.float32, device=args.device
        )
        self.bias = nn.Parameter(
            torch.zeros(
                self.compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
                device=args.device,
            )
        )
        self.norm = model.RMSNorm(
            self.head_dim, self.eps, device=args.device, dtype=args.dtype
        )

        # 解码阶段的状态缓存
        # assigned lazily from Attention.kv_cache
        self.kv_cache: torch.Tensor = None
        # 解码阶段用到的状态缓存，分别存储未压缩的 KV 和 score
        # 在CSA中会用交替两块的缓存构造compress kv
        # 但HCA直接做了非常狠的压缩，没有做交替，所以只有在CSA中才保留双份的
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_len,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
                device=args.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (args.max_batch_len, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
                device=args.device,
            ),
            persistent=False,
        )

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b, s, compress_ratio, 2 * head_dim]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim

        # new_tensor: [b, s, 2 * compress_ratio, head_dim]
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        # 当前块的后半部分（原 b 部分）放在新张量的 [r:2r] 位置
        # 事实上这一步把所有块的后半部分都先放上去了，等待下面a部分覆盖一遍即可
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        # 前一块的前半部分（原 a 部分）放在当前块的 [0:r] 位置
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        """
        压缩input hidden states，对应paper中Eq. (9)-(12)

        args:
            x: [batch, seqlen, dim]
            start_pos: 0 = prefill 阶段, >0 = decode 阶段
        """
        assert self.kv_cache is not None
        batch, seq_len, _ = x.shape
        ratio, overlap, d, rd = (
            self.compress_ratio,
            self.overlap,
            self.head_dim,
            self.rope_head_dim,
        )

        dtype = x.dtype
        x = x.float()  # 压缩需要 fp32
        kv = self.weight_kv(x)  # c_ab
        score = self.weight_z(x)  # Z_ab

        if start_pos == 0:
            # prefill or training stage
            should_compress = seq_len >= ratio

            # Step 1: 处理decode时候产生的小尾巴
            remainder = seq_len % ratio
            cut_off = seq_len - remainder  # 能完整压缩的部分

            if overlap:
                # ---- overlap path (ratio == 4) ----
                if cut_off >= ratio:
                    # 存下最后一个完整块，供decode时消耗
                    self.kv_state[:batch, :ratio] = kv[:, cut_off - ratio : cut_off]
                    self.score_state[:batch, :ratio] = (
                        score[:, cut_off - ratio : cut_off] + self.bias
                    )

                if remainder > 0:
                    kv, self.kv_state[:batch, ratio : ratio + remainder] = kv.split(
                        [cut_off, remainder], dim=1
                    )
                    self.score_state[:batch, ratio : ratio + remainder] = (
                        score[:, cut_off:] + self.bias[:remainder]
                    )
                    score = score[:, :cut_off]

                # Step 2: implement Eq. (9)--(12)
                kv = kv.unflatten(1, (-1, ratio))
                score = score.unflatten(1, (-1, ratio)) + self.bias

                # 构造重叠窗口
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))

                # softmax over 2*ratio elements, then weighted sum
                kv = (kv * score.softmax(dim=2)).sum(dim=2)
            else:
                # ---- non-overlap path (ratio == 128) ----
                if remainder > 0:
                    self.kv_state[:batch, :remainder] = kv[:, cut_off:]
                    self.score_state[:batch, :remainder] = (
                        score[:, cut_off:] + self.bias[:remainder]
                    )
                    kv = kv[:, :cut_off]
                    score = score[:, :cut_off]

                kv = kv.unflatten(1, (-1, ratio))
                score = score.unflatten(1, (-1, ratio)) + self.bias

                # softmax over ratio elements, then weighted sum
                kv = (kv * score.softmax(dim=2)).sum(dim=2)

        else:
            # decode部分
            should_compress = (start_pos + 1) % ratio == 0
            index = start_pos % ratio

            if overlap:
                # ---- overlap path (ratio == 4) ----
                score += self.bias[index]

                self.kv_state[:batch, ratio + index] = kv.squeeze(1)
                self.score_state[:batch, ratio + index] = score.squeeze(1)

                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:batch, :ratio, :d],
                            self.kv_state[:batch, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:batch, :ratio, :d],
                            self.score_state[:batch, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )

                    self.kv_state[:batch, :ratio] = self.kv_state[:batch, ratio:]
                    self.score_state[:batch, :ratio] = self.score_state[:batch, ratio:]
            else:
                # ---- non-overlap path (ratio == 128) ----
                self.kv_state[:batch, index] = kv.squeeze(1)
                self.score_state[:batch, index] = score.squeeze(1) + self.bias[index]

                if should_compress:
                    kv = (
                        self.kv_state[:batch] * self.score_state[:batch].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

        if not should_compress:
            # 没凑够ratio个token，不压缩
            return None

        kv = self.norm(kv.to(dtype))

        if start_pos == 0:
            # 预填充：每隔 ratio 个 token 取 start of each block 作为代表位置
            positions = torch.arange(0, cut_off, ratio, device=kv.device)
        else:
            # 解码：当前压缩结果的代表位置
            positions = torch.tensor(
                [start_pos + 1 - self.compress_ratio], device=kv.device
            )

        kv[..., -self.rope_head_dim :] = self.rope(
            kv[..., -self.rope_head_dim :], positions
        )

        if start_pos == 0:
            self.kv_cache[:batch, : seq_len // ratio] = kv
        else:
            self.kv_cache[:batch, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(nn.Module):
    def __init__(
        self,
        args: DSArgs,
        rope: model.RotaryPositionalEmbedding,
        compress_ratio: int = 4,
    ):
        super().__init__()
        self.d_model = args.d_model
        self.compress_ratio = compress_ratio
        self.rope_head_dim = args.rope_head_dim
        self.index_head_dim = args.index_head_dim
        self.index_num = args.index_num
        self.attn_rank = args.attn_rank
        self.index_topk = args.index_topk

        self.rope = rope

        self.weight_iuq = model.Linear(
            self.attn_rank,
            self.index_num * self.index_head_dim,
            device=args.device,
            dtype=args.dtype,
        )
        self.weight_h = model.Linear(
            self.d_model, self.index_num, device=args.device, dtype=args.dtype
        )

        self.compressor = Compressor(
            args, self.index_head_dim, self.compress_ratio, rope
        )

        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_len,
                args.max_seq_len // self.compress_ratio,
                self.index_head_dim,
                device=args.device,
            ),
            persistent=False,
        )

        # 本质上Indexer部分实现了一个mini的Attention
        # 在Eq.16有一堆求和，除上一个scale保持数值稳定
        self.attention_scale = (self.index_head_dim * self.index_num) ** -0.5

    def forward(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        start_pos: int,
        offset: int,
    ):
        """
        原文中Eq. (13)输入hidden state生成compressed latent vector for query
        但这个向量c_t^Q是整个Attention共享的 (Shared KV MQA)
        所以这里要同时输入原始的x和Attention部分创建好的query c_t^Q

        Returns:
            (topk_idxs, index_score) — topk_idxs for sparse attention gather,
            index_score (before topk) for KL auxiliary loss.
        """
        # Detach inputs: Indexer is trained separately via KL loss.
        # This detaches the Indexer subgraph from the main model,
        # so LM loss backward and KL loss backward see disjoint graphs.
        hidden_state = hidden_state.detach()
        query = query.detach()

        batch, seq_len, _ = hidden_state.size()
        m = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seq_len

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache

        # Eq.14
        # index_query: (batch, len, n_index, d_index)
        index_query = self.weight_iuq(query)
        index_query = index_query.unflatten(-1, (self.index_num, self.index_head_dim))

        # Obtain K_s^{IComp}
        # K is stored in kv_cache
        self.compressor(hidden_state, start_pos)

        # Compressor里面的hidden state被旋转了一次，这里同样旋转
        # Apply RoPE
        positions = torch.arange(
            start_pos, start_pos + seq_len, device=index_query.device
        )
        index_query = torch.cat([index_query[..., :-rd], self.rope(index_query[..., -rd:], positions)], dim=-1)

        # Eq.15
        # Eq15~16这里类似于做了一个小的attention
        # 为了数值稳定性，除掉一个\sqrt{d * n}
        # weights: (b, s, n_index)
        weights = self.weight_h(hidden_state) * self.attention_scale

        # q_{th}^I K_s^{IComp}
        index_score = torch.einsum(
            "bsnd,bld->bsnl", index_query, self.kv_cache[:batch, : end_pos // m]
        )

        # I: (batch, seq, block_num)
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=-2)

        if start_pos == 0:
            # Eq.17: 训练and prefill 选topk时不能选未来的，因果掩码
            # Implement s < floor(t / m)

            # left tensor: (seq_len, n_block)
            # _left[i] = [0, ..., n_block - 1] for each i
            _left = torch.arange(seq_len // m, device=index_score.device).repeat(
                seq_len, 1
            )

            # right tensor: (seq_len, 1)
            # _right[..., 0] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, ..., n_block]
            _right = (
                torch.arange(1, seq_len + 1, device=index_score.device).unsqueeze(1)
                // m
            )

            mask = _left >= _right
            index_score = index_score + torch.where(mask, float("-inf"), 0)

        # Detach before topk: topk_idxs are only used as gather indices
        # in sparse attention, no gradient flows through this path.
        topk_idxs = index_score.detach().topk(min(self.index_topk, end_pos // m), dim=-1)[1]

        if start_pos == 0:
            mask = (
                topk_idxs
                >= torch.arange(1, seq_len + 1, device=index_score.device).unsqueeze(1)
                // m
            )
            local_topk_idxs = topk_idxs.clone()
            topk_idxs = torch.where(mask, -1, local_topk_idxs + offset)
            local_topk_idxs = torch.where(mask, -1, local_topk_idxs)
        else:
            local_topk_idxs = topk_idxs.clone()
            topk_idxs += offset
        return topk_idxs, local_topk_idxs, index_score


class Attention(nn.Module):
    def __init__(
        self, args: DSArgs, rope: model.RotaryPositionalEmbedding, layer_id: int
    ):
        super().__init__()

        self.compress_ratio = args.compress_ratios[layer_id]
        self.index_num = args.index_num
        self.n_heads = args.n_heads
        self.eps = args.eps

        self.d_model = args.d_model  # initial dim for hidden state, d
        self.head_dim = args.head_dim  # dimension for compressor dim, i.e., c
        self.index_head_dim = args.index_head_dim  # dimension for index dim, c^I
        self.rope_head_dim = args.rope_head_dim  # RoPE applies for last rd dimensions
        self.attn_rank = args.attn_rank  # hidden state is projected onto d_c dim
        self.softmax_scale = self.head_dim**-0.5
        self.output_group = args.output_group
        self.output_lora = args.output_lora

        self.window_size = args.window_size  # 根据论文，滑动窗口保留前n_win个token
        # 于是kv_cache总共有两部分，前win个是最新的win个token，后面的压缩（如果有）
        self.kv_cache_size = self.window_size + (
            args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_len,
                self.kv_cache_size,
                self.head_dim,
                device=args.device,
            ),
            persistent=False,
        )

        self.proj_lora = model.Linear(
            self.d_model, self.attn_rank, device=args.device, dtype=args.dtype
        )
        self.rms_norm_q = model.RMSNorm(
            self.attn_rank, self.eps, device=args.device, dtype=args.dtype
        )
        self.w_uq = model.Linear(
            self.attn_rank,
            self.n_heads * self.head_dim,
            device=args.device,
            dtype=args.dtype,
        )

        self.weight_kv = model.Linear(self.d_model, self.head_dim)
        self.rms_norm_kv = model.RMSNorm(
            self.head_dim, self.eps, device=args.device, dtype=args.dtype
        )

        self.rope = rope
        self.attn_sink = nn.Parameter(
            torch.zeros(self.n_heads, device=args.device)
        )

        self.o_down = model.Linear(
            self.n_heads * self.head_dim // self.output_group,
            self.output_group * self.output_lora,
            device=args.device,
            dtype=args.dtype,
        )  # 这里事实上只是用了Linear的参数初始化，实际中没有使用forward
        self.o_up = model.Linear(
            self.output_group * self.output_lora,
            self.d_model,
            device=args.device,
            dtype=args.dtype,
        )

        self.compressor = (
            Compressor(args, self.head_dim, self.compress_ratio, rope)
            if self.compress_ratio
            else None
        )
        self.indexer = (
            Indexer(args, rope, self.compress_ratio)
            if self.compress_ratio == 4
            else None
        )

    def _get_window_topk_id(self, batch: int, seq_len: int, start_pos: int):
        """
        Output: Tensor of shape (batch_size, seq_len, window_size)
        """
        dev = self.kv_cache.device
        if start_pos >= self.window_size - 1:
            start_pos %= self.window_size
            matrix = torch.cat(
                [
                    torch.arange(start_pos + 1, self.window_size, device=dev),
                    torch.arange(0, start_pos + 1, device=dev),
                ],
                dim=0,
            )
        elif start_pos > 0:
            matrix = F.pad(
                torch.arange(start_pos + 1, device=dev),
                (0, self.window_size - start_pos - 1),
                value=-1,
            )
        else:
            base = torch.arange(seq_len, device=dev).unsqueeze(1)
            start = (base - self.window_size + 1).clamp(0)
            matrix = start + torch.arange(min(seq_len, self.window_size), device=dev)
            matrix = torch.where(matrix > base, -1, matrix)

        return matrix.unsqueeze(0).expand(batch, -1, -1)

    def _get_compress_topk_id(
        self, batch: int, seq_len: int, start_pos: int, offset: int
    ):
        """
        Select every compress_ratio-th compressed block uniformly.

        与Indexer不同，这个方法是直接按照ratio均匀选取压缩块，
        不做学习式的稀疏选择。用于ratio=128等没有Indexer的场景。

        Output: Tensor of shape (batch_size, seq_len, n_compressed_blocks)

        示例 (ratio=4, seq_len=12, start_pos=0, offset=0):
        [[ 0, -1, -1],
         [ 0, -1, -1],
         [ 0, -1, -1],
         [ 0, -1, -1],
         [ 1, -1, -1],
         [ 1, -1, -1],
         [ 1, -1, -1],
         [ 1, -1, -1],
         [ 1,  2, -1],
         [ 1,  2, -1],
         [ 1,  2, -1],
         [ 1,  2, -1]]
        第t行能看到所有 block_index < floor(t/ratio) 的压缩块。
        """
        dev = self.kv_cache.device
        ratio = self.compress_ratio
        if start_pos > 0:
            matrix = torch.arange(0, (start_pos + 1) // ratio, device=dev) + offset
        else:
            matrix = torch.arange(seq_len // ratio, device=dev).repeat(seq_len, 1)
            mask = matrix >= torch.arange(1, seq_len + 1, device=dev).unsqueeze(1) // ratio
            matrix = torch.where(mask, -1, matrix + offset)

        return matrix.unsqueeze(0).expand(batch, -1, -1)

    def _sparse_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
        attn_sink: torch.Tensor,
        softmax_scale: float,
        num_compress: int = 0,
    ) -> torch.Tensor:
        """
        PyTorch version 的 sparse attention 实现
        后续用triton重写

        Inputs:
        - q: query (batch, len, n_head, head_dim)
        - kv: shared kv (batch, len // m, head_dim)
        - topk_idxs: (batch, len, topk)
        - attn_sink: addtitional bias for each head, (h,)
        - num_compress: number of compressed-block entries at the end of topk_idxs

        Returns:
        - o: (batch, len, n_head, head_dim)
        - weight_compress: (batch, len, n_head, num_compress) or None if num_compress==0
        """
        b, m, h, d = q.shape
        topk = topk_idxs.shape[-1]

        # Step 1: 根据topk的索引取出kv
        # kv: (b, kv_len, d), topk_idxs: (b, m, topk) — 每个query位置有独立的topk索引
        # mask 掉无效索引（-1），clamp 到 0 避免 gather 报错，之后用 mask 置 -inf
        mask = topk_idxs == -1
        idx = topk_idxs.clamp(0).unsqueeze(-1).expand(-1, -1, -1, d)  # (b, m, topk, d)
        kv_expanded = kv.unsqueeze(1).expand(-1, m, -1, -1)  # (b, m, kv_len, d)
        k = v = kv_expanded.gather(2, idx)  # (b, m, topk, d)

        # Step 3: Attention
        score = torch.einsum("bmhd,bmkd->bmhk", q, k) * softmax_scale
        score = score.masked_fill(mask.unsqueeze(2), float("-inf"))

        # Step 4: Add attention sink
        # (h,) -> (1, 1, h, 1) -> (b, m, h, 1)
        sink = attn_sink.view(1, 1, h, 1).expand(b, m, -1, 1)
        score = torch.cat([score, sink], dim=-1)  # (b,m,h,k+1)

        # Step 5: softmax
        weight = torch.softmax(score, dim=-1)[..., :topk]

        output = torch.einsum("bmhk,bmkd->bmhd", weight, v)

        if num_compress > 0:
            # detach: KL loss trains indexer, not main model
            weight_compress = weight[:, :, :, -num_compress:].detach()
        else:
            weight_compress = None
        return output, weight_compress

    def forward(self, x: torch.Tensor, **kwargs):
        start_pos = kwargs.get("start_pos", 0)
        batch, seq_len, _ = x.size()
        m = self.compress_ratio
        rd = self.rope_head_dim
        win = self.window_size

        # Initialization
        if self.compress_ratio and self.compressor.kv_cache is None:
            # 约定前win个放最近的token，后面的放压缩
            self.compressor.kv_cache = self.kv_cache[:, win:]

        qr = self.rms_norm_q(self.proj_lora(x))  # Eq.13
        q = self.w_uq(qr).unflatten(-1, (self.n_heads, self.head_dim))  # Eq.18

        # 归一化
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

        # RoPE for query
        positions = torch.arange(start_pos, start_pos + seq_len, device=q.device)
        q = torch.cat([q[..., :-rd], self.rope(q[..., -rd:], positions)], dim=-1)

        # kv
        kv = self.rms_norm_kv(self.weight_kv(x))
        kv = torch.cat([kv[..., :-rd], self.rope(kv[..., -rd:], positions)], dim=-1)

        # select topk indices
        topk_win = self._get_window_topk_id(batch, seq_len, start_pos)
        # 预填充时kv = cat([kv, kv_compress])，需要offset seq_len个
        offset = seq_len if start_pos == 0 else self.window_size

        if m == 4:
            compress_topk_idxs, local_topk_idxs, index_score = self.indexer(x, qr, start_pos, offset)
            self._index_score = index_score
            self._compress_topk_idxs = local_topk_idxs  # raw block indices for KL loss gather
        elif m > 0:
            compress_topk_idxs = self._get_compress_topk_id(
                batch, seq_len, start_pos, offset
            ).to(kv.device)
            self._index_score = None
            self._compress_topk_idxs = None
        else:
            compress_topk_idxs = None
            self._index_score = None
            self._compress_topk_idxs = None

        num_compress = compress_topk_idxs.shape[-1] if compress_topk_idxs is not None else 0

        if compress_topk_idxs is not None:
            topk_idxs = torch.cat([topk_win, compress_topk_idxs], dim=-1)
        else:
            topk_idxs = topk_win

        if start_pos == 0:
            # prefill or training
            # 先写滑动窗口的，放到前win个
            if seq_len <= win:
                # 直接存
                self.kv_cache[:batch, :seq_len] = kv
            else:
                # 在decode时按照 start_pos % win 保存的话
                # 相当于前win个写满后，要接着回到一开始，覆盖旧的数据
                # 所以prefill的时候，假设win_size = 5, len = 12
                # 那么最后5个token (7, 8, 9, 10, 11) 写到kv_cache的顺序是
                # (10, 11, 7, 8, 9)
                cutoff = seq_len % win
                self.kv_cache[:batch, cutoff:win], self.kv_cache[:batch, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)

            # 之后是压缩部分的
            if m:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)

            # Sparse Attention
            o, weight_compress = self._sparse_attention(
                q, kv, topk_idxs, self.attn_sink, self.softmax_scale, num_compress
            )  # (batch, seq_len, n_head, head_dim)

        else:
            self.kv_cache[:batch, start_pos % win] = kv.squeeze(1)
            if m:
                self.compressor(x, start_pos)

            # Sparse Attention
            o, weight_compress = self._sparse_attention(
                q, self.kv_cache[:batch], topk_idxs, self.attn_sink, self.softmax_scale, num_compress
            )  # (batch, seq_len, n_head, head_dim)

        self._weight_compress = weight_compress

        o_rope = self.rope(o[..., -rd:], positions, inverse=True)
        o = torch.cat([o[..., :-rd], o_rope], dim=-1)
        # Grouped Ouput Projection
        o = o.view(batch, seq_len, self.output_group, -1)
        w_o_down = self.o_down.weight.view(self.output_group, self.output_lora, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, w_o_down)
        return self.o_up(o.flatten(2))


class TransformerBlock(nn.Module):
    def __init__(
        self, args: DSArgs, rope: model.RotaryPositionalEmbedding, layer_id: int
    ):
        super().__init__()

        self.rope = rope
        self.use_hash_routing = layer_id < args.n_hash_layer
        self.attention = Attention(args, self.rope, layer_id)
        self.ffn = DSMoE(args, use_hash_routing=self.use_hash_routing)

        self.attn_hc = ManifoldHyperConnections(args, self.attention)
        self.ffn_hc = ManifoldHyperConnections(args, self.ffn)

    def forward(
        self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        """
        Input:
        - x: torch.Tensor of shape (batch, seq, hc_num, d_model)

        Returns:
        - x: torch.Tensor of shape (batch, seq, hc_num, d_model)
        - indexer_data: (index_score, weight_compress, compress_topk_idxs) or None
        """
        x = self.attn_hc(x, start_pos=start_pos)
        indexer_data = (
            self.attention._index_score,
            self.attention._weight_compress,
            self.attention._compress_topk_idxs,
        )
        x = self.ffn_hc(x, token_ids=input_ids)
        return x, indexer_data


class PredictionHead(nn.Module):
    def __init__(self, args: DSArgs, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()

        self.d_model = args.d_model
        self.eps = args.eps
        self.hc_num = args.expansion_rate
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.vocab_size = args.vocab_size

        self.weight = model.Linear(
            self.d_model * self.hc_num,
            self.hc_num,
            device=args.device,
            dtype=torch.float32,
        )
        self.logit = model.Linear(
            self.d_model, self.vocab_size, dtype=torch.float32, device=args.device
        )
        self.norm = model.RMSNorm(
            self.d_model, self.eps, device=args.device, dtype=args.dtype
        )

    def forward(self, x: torch.Tensor):
        """
        Input:

        x: (batch, seq, hc_num, d_model)
        """

        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()  # (b, s, h*d)
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)

        weight = torch.sigmoid(self.weight(x) * rsqrt) + self.hc_eps  # (b, s, h)
        x = torch.sum(weight.unsqueeze(-1) * x.view(shape), dim=2).to(
            dtype
        )  # (b, s, d)
        x = self.norm(x)
        logits = self.logit(x.float())
        return logits


class MTPBlock(TransformerBlock):
    def __init__(
        self,
        args: DSArgs,
        rope: model.RotaryPositionalEmbedding,
        embedding: model.Embedding,
        prediction_head: PredictionHead,
        layer_id: int,
    ):
        super().__init__(args, rope, layer_id)

        self.eps = args.eps

        self.embedding = embedding
        self.prediction_head = prediction_head

        self.embed_proj = model.Linear(args.d_model, args.d_model)
        self.x_proj = model.Linear(args.d_model, args.d_model)

        self.embed_norm = model.RMSNorm(
            args.d_model, self.eps, device=args.device, dtype=args.dtype
        )
        self.x_norm = model.RMSNorm(
            args.d_model, self.eps, device=args.device, dtype=args.dtype
        )

    def forward(self, x: torch.Tensor, start_pos: int, token_ids: torch.Tensor):
        # x: [b,s,hc,d]
        embedding = self.embedding(token_ids)
        embedding = self.embed_norm(embedding)
        x = self.x_norm(x)

        x = self.embed_proj(embedding).unsqueeze(2) + self.x_proj(x)
        x, indexer_data = super().forward(x, start_pos, token_ids)

        logits = self.prediction_head(x)

        return logits, x, indexer_data


class DeepSeekV4(nn.Module):
    def __init__(self, args: DSArgs):
        super().__init__()

        self.max_seq_len = args.max_seq_len
        self.hc_num = args.expansion_rate
        self.eps = args.eps

        self.vocab_size = args.vocab_size

        self.d_model = args.d_model

        self.rope = model.RotaryPositionalEmbedding(
            args.rope_theta,
            args.rope_head_dim,
            args.max_seq_len,
            factor=args.rope_factor,
            beta_fast=args.beta_fast,
            beta_slow=args.beta_slow,
            device=args.device,
        )

        self.embedding = model.Embedding(
            self.vocab_size, self.d_model, device=args.device, dtype=args.dtype
        )
        self.layers = nn.ModuleList()

        for layer_id in range(args.n_layer):
            self.layers.append(TransformerBlock(args, self.rope, layer_id))

        self.prediction = PredictionHead(args, self.eps, self.eps)

        self.mtp_layers = nn.ModuleList()
        for layer_id in range(args.n_mtp_layer):
            self.mtp_layers.append(
                MTPBlock(
                    args,
                    self.rope,
                    self.embedding,
                    self.prediction,
                    layer_id + args.n_layer,
                )
            )

    def forward(self, token_ids: torch.Tensor, start_pos: int = 0):
        embed = self.embedding(token_ids)  # (b, s, d)
        h = embed.unsqueeze(2).repeat(1, 1, self.hc_num, 1)  # (b, s, h, d)

        indexer_data_list = []
        for layer in self.layers:
            h, idx_data = layer(h, start_pos, token_ids)
            indexer_data_list.append(idx_data)

        # Next token prediction
        ntp = self.prediction(h)

        # Multi token prediction
        mtp_res = []
        for layer in self.mtp_layers:
            mtp, h, idx_data = layer(h, start_pos, token_ids)
            mtp_res.append(mtp)
            indexer_data_list.append(idx_data)

        return ntp, mtp_res, indexer_data_list


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = DSArgs(n_hash_layer=0)
    x = torch.randint(0, args.vocab_size, (2, 128))
    dpsk = DeepSeekV4(args)

    ntp, mtp_list, _ = dpsk(x)
    print("prefill ntp:", ntp.size(), "mtp:", [m.size() for m in mtp_list])
    for i in range(128, 131):
        ntp, mtp_list, _ = dpsk(x[:, 0:1], i)
        print(f"decode pos {i}: ntp {ntp.size()}, mtp {[m.size() for m in mtp_list]}")

    h = torch.randn(2, 128, args.expansion_rate, args.d_model)
    logits, _, _ = dpsk.mtp_layers[0](h, 0, x)
    print("mtp prefill:", logits.size())
    logits, _, _ = dpsk.mtp_layers[0](h[:, 0:1], 1, x[:, 0:1])
    print("mtp decode:", logits.size())
