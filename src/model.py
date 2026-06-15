import math

import torch
torch.set_default_device("cuda:0")
import torch.nn as nn
import torch.nn.functional as F

# Some basic NN blocks


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weight.to(torch.float32)
        return x.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.linear_1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear_2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear_3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        w1 = self.silu(self.linear_1(x))
        w3 = self.linear_3(x)
        return self.linear_2(w1 * w3)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        original_seq_len: int = 0,
        factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        device=None,
    ):
        super().__init__()
        freqs = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        )  # 原始RoPE

        if original_seq_len > 0:
            low, high = self._find_correction_range(
                beta_fast, beta_slow, d_k, theta, original_seq_len
            )
            smooth = 1 - self._linear_ramp_factor(low, high, d_k // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = pos[:, None] * freqs[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("freqs_cis", cos + 1j * sin)

    @staticmethod
    def _find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        给定预期旋转次数，反推对应的index是多少
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    @staticmethod
    def _find_correction_range(low_rot, high_rot, dim, base, max_seq_len):

        low = math.floor(
            RotaryPositionalEmbedding._find_correction_dim(
                low_rot, dim, base, max_seq_len
            )
        )
        high = math.ceil(
            RotaryPositionalEmbedding._find_correction_dim(
                high_rot, dim, base, max_seq_len
            )
        )
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
            max_val - min_val
        )
        return torch.clamp(linear_func, 0, 1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor, inverse: bool = False):
        freqs = self.freqs_cis[token_positions]
        if inverse:
            freqs = freqs.conj()

        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)

        # 显式 reshape freqs 以适配多头输入，与官方 apply_rotary_emb 一致
        if x_complex.ndim == 3:
            freqs = freqs.view(1, x_complex.size(1), x_complex.size(-1))
        else:
            freqs = freqs.view(1, x_complex.size(1), 1, x_complex.size(-1))

        x_rot = torch.view_as_real(x_complex * freqs).flatten(-2)
        return x_rot.reshape(*x.shape)


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
):
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, value)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope: RotaryPositionalEmbedding | None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_k = self.d_v = d_model // num_heads

        self.rope = rope

        self.QKV = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.O = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        qkv = self.QKV(x)  # (batch, seq, 3 * d_model)
        wq_x, wk_x, wv_x = [
            t.view(*x.shape[:-1], self.num_heads, self.d_k).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        ]

        # rope
        if self.rope is not None:
            wq_x = self.rope(wq_x, token_positions)
            wk_x = self.rope(wk_x, token_positions)

        output = F.scaled_dot_product_attention(wq_x, wk_x, wv_x, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(*x.shape[:-1], self.d_model)

        return self.O(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope: RotaryPositionalEmbedding | None,
        *,
        eps=1e-5,
        use_checkpoint: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope = rope
        self.use_checkpoint = use_checkpoint

        self.rms_norm_1 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.attention = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, rope=self.rope, device=device, dtype=dtype
        )
        self.rms_norm_2 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.fnn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def _forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        x1 = self.rms_norm_1(x)
        x1 = self.attention(x1, token_positions)

        x = x + x1
        x2 = self.rms_norm_2(x)
        x2 = self.fnn(x2)

        return x + x2

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, token_positions, use_reentrant=False
            )
        return self._forward(x, token_positions)


class MiniLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        rope_theta: float,
        *,
        use_checkpoint: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads

        self.token_embedding = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.rope = RotaryPositionalEmbedding(
            rope_theta, d_k, context_length, device=device
        )
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    rope=self.rope,
                    eps=1e-5,
                    use_checkpoint=use_checkpoint,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids, token_positions=None):
        if token_positions is None:
            token_positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        emb = self.token_embedding(token_ids)

        for transformer in self.transformers:
            emb = transformer(emb, token_positions)

        emb = self.rms_norm(emb)
        return self.output_linear(emb)
