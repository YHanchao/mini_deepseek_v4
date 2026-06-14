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

    # Attention
    rope_head_dim: int = 64
    index_head_dim: int = 512
    index_num: int = 4
    compress_ratio: int = 4
    attn_rank: int = 64
    index_topk: int = 4


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
            x.unsqueeze(1).expand(batch, n, seq_len, d).reshape(batch * n, seq_len, d)
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
        mixed_residuals = mixed_by_seq.reshape(batch * self.n, seq_len, self.d_model)

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
        post_block = post_block.reshape(batch * self.n, seq_len, self.d_model)

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

        branch_input, mixed_residuals = self._width_connection(x, h_pre, h_res, batch)
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

        # 要同时维护Ca, Cb, Za, Zb，所以是两倍的head_dim
        self.weight_kv = model.Linear(
            self.d_model, 2 * self.head_dim, dtype=torch.float32, device=args.device
        )
        self.weight_z = model.Linear(
            self.d_model, 2 * self.head_dim, dtype=torch.float32, device=args.device
        )
        self.bias = nn.Parameter(
            torch.empty(
                self.compress_ratio,
                2 * self.head_dim,
                dtype=torch.float32,
                device=args.device,
            )
        )
        self.norm = model.RMSNorm(self.head_dim)

        # 解码阶段的状态缓存
        # 形状: [max_batch, 2 * compress_ratio, 2 * head_dim]
        # assigned lazily from Attention.kv_cache
        self.kv_cache: torch.Tensor = None
        # 解码阶段用到的状态缓存，分别存储未压缩的 KV 和 score
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_len,
                2 * compress_ratio,
                2 * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (args.max_batch_len, 2 * compress_ratio, 2 * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
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
        m = self.compress_ratio

        # (batch, seq_len, 2 * head_dim)
        dtype = x.dtype
        x = x.float()  # 压缩需要 fp32
        kv = self.weight_kv(x)  # c_ab
        score = self.weight_z(x)  # Z_ab

        if start_pos == 0:
            # prefill or training stage
            should_compress = seq_len >= m

            # Step 1: 处理decode时候产生的小尾巴
            remainder = seq_len % m
            cut_off = seq_len - remainder  # 能完整压缩的部分

            if cut_off >= m:
                # 存下最后一个完整块，供decode时消耗
                self.kv_state[:batch, :m] = kv[:, cut_off - m : cut_off]
                self.score_state[:batch, :m] = (
                    score[:, cut_off - m : cut_off] + self.bias
                )

            # 余数部分暂时保存在 state 中，不参与本次压缩
            if remainder > 0:
                kv, self.kv_state[:batch, m : m + remainder] = kv.split(
                    [cut_off, remainder], dim=1
                )
                self.score_state[:batch, m : m + remainder] = (
                    score[:, cut_off:] + self.bias[:remainder]
                )
                score = score[:, :cut_off]

            # Step 2: implement Eq. (9)--(12)
            # 将序列按 ratio 分块，形状: [b, n/ratio, ratio, coff*d]
            kv = kv.unflatten(1, (-1, m))
            score = score.unflatten(1, (-1, m)) + self.bias

            # 构造重叠窗口，得到 [b, n/ratio, 2*ratio, d]
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))

            # softmax 沿每个压缩块的 2*ratio (或 ratio) 个元素，对 score 归一化
            # Eq (11) softmax，然后Eq(12) 加权求和
            kv = (kv * score.softmax(dim=2)).sum(dim=2)  # 压缩后形状 [b, n/ratio, d]

        else:
            # decode部分
            # 利用在prefill阶段已经维护好的历史信息
            should_compress = (start_pos + 1) % m == 0
            index = start_pos % m
            score += self.bias[index]

            # 将当前 token 存入 kv_state 和 score_state 的“b 部分”（ratio + 偏移）
            self.kv_state[:batch, m + index] = kv.squeeze(1)
            self.score_state[:batch, m + index] = score.squeeze(1)

            if should_compress:
                # 取出需要压缩的两个窗口：前 r 个来自前一组的 b 部分，后 r 个来自当前组的 a 部分
                kv_state = torch.cat(
                    [
                        self.kv_state[:batch, :m, : self.head_dim],
                        self.kv_state[:batch, m:, self.head_dim :],
                    ],
                    dim=1,
                )
                score_state = torch.cat(
                    [
                        self.score_state[:batch, :m, : self.head_dim],
                        self.score_state[:batch, m:, self.head_dim :],
                    ],
                    dim=1,
                )
                # 加权求和压缩
                kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)

                # 将 b 部分整体复制到 a 部分，为下一次重叠做准备
                self.kv_state[:batch, :m] = self.kv_state[:batch, m:]
                self.score_state[:batch, :m] = self.score_state[:batch, m:]

        if not should_compress:
            # 没凑够m个token，不压缩
            return None

        kv = self.norm(kv)

        if start_pos == 0:
            # 预填充：每隔 ratio 个 token 取最后一个位置作为压缩结果的代表
            positions = torch.arange(0, cut_off, m, device=kv.device)
        else:
            # 解码：当前压缩结果的代表位置
            positions = torch.tensor(
                [start_pos + 1 - self.compress_ratio], device=kv.device
            )

        kv[..., -self.rope_head_dim :] = self.rope(
            kv[..., -self.rope_head_dim :], positions
        )

        if start_pos == 0:
            self.kv_cache[:batch, : seq_len // m] = kv
        else:
            self.kv_cache[:batch, start_pos // m] = kv.squeeze(1)
        return kv


class Indexer(nn.Module):
    def __init__(self, args: DSArgs, rope: model.RotaryPositionalEmbedding):
        super().__init__()
        self.d_model = args.d_model
        self.compress_ratio = args.compress_ratio
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
                args.max_seq_len // args.compress_ratio,
                self.index_head_dim,
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
        """
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
        index_query[..., -rd:] = self.rope(index_query[..., -rd:], positions)

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
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=-2)

        if start_pos == 0:
            # Eq.17: 训练and prefill 选topk时不能选未来的，因果掩码
            # Implement s < floor(t / m)

            # left tensor: (seq_len, n_block)
            # _left[i] = [0, ..., n_block - 1] for each i
            _left = torch.arange(seq_len // m).repeat(seq_len, 1)

            # right tensor: (seq_len, 1)
            # _right[..., 0] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, ..., n_block]
            _right = torch.arange(1, seq_len + 1).unsqueeze(1) // m

            mask = _left >= _right
            index_score += torch.where(mask, float("-inf"), 0)

        topk_idxs = index_score.topk(min(self.index_topk, end_pos // m), dim=-1)[1]

        if start_pos == 0:
            # 在序列一开始时，上一层会全都fill为-inf
            # 那么返回的topk是没有意义的
            # 所以进一步筛选，保证topk_idx取值合理
            mask = topk_idxs >= torch.arange(1, seq_len + 1).unsqueeze(1) // m
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs
