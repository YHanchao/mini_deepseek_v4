import math

import torch
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
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freqs = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        )
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = pos[:, None] * freqs[None, :]  # (max_seq_len, d_k/2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("freqs_cis", cos + 1j * sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        freqs = self.freqs_cis[token_positions]  # (seq_len, d_k/2) complex

        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # (..., seq_len, d_k/2, 2)
        x_complex = torch.view_as_complex(x_reshaped)
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
