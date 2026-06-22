# Model Memory Optimization Analysis

对 [src/model.py](../src/model.py) 中 AdamW 训练时激活内存占用分析及优化建议。

> 相关脚本：[archives/memory_analysis.py](../archives/memory_analysis.py) — 量化各架构下 AdamW 峰值内存的分解。

## 内存公式回顾

AdamW 训练峰值内存 = **Parameters + Gradients + Optimizer State + Activations**：

| 组件 | 元素数 (× 4 bytes for fp32) |
|---|---|
| Parameters | `num_layers × (2d + 4d² + 3d·d_ff) + d + d·V` |
| Gradients | = Parameters |
| Optimizer State | = 2 × Parameters |
| Activations | 见下 |

其中激活的元素数分解（per block）为：

```
8·bs·ctx·d + bs·heads·ctx² + 3·bs·ctx·d_ff
```

加上最后的 RMSNorm（`bs·ctx·d`）、output embedding（`bs·ctx·V`）、cross-entropy（`bs·ctx·V`）。

---

## 可优化的内存热点（按影响从大到小）

### P0 — Output Logits 张量

**位置**: [MiniLLM.forward L251](../src/model.py#L251)

```python
return self.output_linear(emb)  # (bs, seq, vocab_size)
```

logits 形状为 `(bs, seq, vocab_size)`。以 GPT-2-XL-Long 为例：

> 1 × 16384 × 50257 × 4B ≈ **3.3 GB**

若外部 loss 函数再 materialize 一份 logits（如 `F.cross_entropy` 内部），直接翻倍到 **6.6 GB**。

**优化方向**: 将 `output_linear` 与 `cross_entropy` 融合为单一算子，或在 loss 中以 chunked 方式逐段计算 logits + loss，不使用完整 logits 张量。

---

### P0 — TransformerBlock 级 Activation Checkpointing

每个 block 的中间激活必须在反向时可用。不用 checkpoint 时：
- Attention 子层保存 `scores` 和 `attn`（各 `bs·heads·ctx²`）
- SwiGLU 子层保存 `gate_input`, `w1`, `w3`, `w1*w3`（共 `bs·ctx·d + 3·bs·ctx·d_ff`）

以 GPT-2-XL-Long（bs=1, ctx=16384, heads=25）为例，单个 block 的激活约为 **570 MB**，48 层 = **~27 GB**。

**优化方向**: 对 `TransformerBlock.forward` 使用 `torch.utils.checkpoint.checkpoint`，反向时重新计算中间激活而非保存。代价是额外一次 forward 计算（~33% 训练时间），但激活内存几乎归零。

---

### P1 — Q、K、V 作为三个独立 Linear

**位置**: [CausalMultiHeadSelfAttention L126-L129](../src/model.py#L126-L129)

```python
self.Q = Linear(d_model, d_model, ...)
self.K = Linear(d_model, d_model, ...)
self.V = Linear(d_model, d_model, ...)
```

每个 Linear 独立执行 `x @ W.T`，产生三个 `(bs, seq, d_model)` 输出张量同时存活。

**优化方向**: 合并为 Fused QKV：

```python
self.QKV = Linear(d_model, 3 * d_model, ...)

# forward:
qkv = self.QKV(x)                     # (bs, seq, 3*d_model)
wq_x, wk_x, wv_x = qkv.chunk(3, dim=-1)
```

一次 matmul 代替三次（更快），且只需一个 Linear 的反向上下文。

---

### P1 — RotaryPositionalEmbedding 中的临时张量

**位置**: [RotaryPositionalEmbedding.forward L73-L85](../src/model.py#L73-L85)

```python
x_rot_even = x_even * cos - x_odd * sin   # 新张量
x_rot_odd = x_even * sin + x_odd * cos    # 新张量
x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # 第三个新张量
```

峰值时 `x_rot_even`、`x_rot_odd`、`x_rot` 三个张量同时存活（加上原始输入 `x`），额外内存 ≈ `1.5 × (Q 或 K 的大小)`。

**优化方向**: 用复数旋转消除中间变量：

```python
x_complex = torch.view_as_complex(x_reshaped.float())
freqs = cos + 1j * sin  # 预计算
x_rot = torch.view_as_real(x_complex * freqs).flatten(-2)
```

只产生一个临时张量（`x_complex * freqs`），减少约一半临时内存。

---

### P2 — causal_mask 每次 forward 做切片

**位置**: [CausalMultiHeadSelfAttention.forward L155-L156](../src/model.py#L155-L156)

```python
self.causal_mask[:seq_len, :seq_len]
```

每层 forward 创建一个新的 `(seq_len, seq_len)` 张量。单个虽小（1024² × 4B = 4MB），48 层累积约 ~200MB。

**优化方向**: 切片可在 block 外做一次后传入，或在 `attention` 内部直接用原始 mask + 索引（不影响正确性，仅减少 Python 层面冗余分配）。

---

## 小结

| 优先级 | 优化项 | 预计节省 | 代价 |
|---|---|---|---|
| **P0** | output logits + cross_entropy 融合 | 数 GB（长序列/大词表） | 实现复杂度中等 |
| **P0** | TransformerBlock checkpoint | 省掉全部 block 内激活（~50% 激活内存） | +33% 训练时间 |
| **P1** | QKV 融合为一个 Linear | overhead 减小，代码更干净 | 无 |
| **P1** | RoPE 用复数实现 | ~0.5×(Q 或 K) 临时内存 | 无 |
| **P2** | causal_mask 切片外提 | 百 MB 级 | 无 |

**最高性价比方案**: P0 checkpoint + P1 QKV 融合 + P1 RoPE 复数化，三者实现成本低且收益大。
