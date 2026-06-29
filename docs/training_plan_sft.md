# DeepSeekV4 SFT 工作计划

**最后更新**: 2026-06-28

## 最终决定

| 项 | 值 |
|---|---|
| 配置 | small（d_model=1024, n_layer=8, n_experts=6） |
| 基座模型 | 预训练 checkpoint，只加载权重，优化器重新初始化 |
| 数据 | Alpaca-GPT4 (52k) + Dolly 15k (15k) + OASST 英文高质量 (~3.5k trees) |
| Train/Val | Alpaca+Dolly 90/10 随机；OASST 用自带 tree-level split |
| 预训练数据混合 | 不混预训练数据 |
| MTP + Indexer KL | 保留 |
| Epochs | 2-3，盯 val loss |
| 序列长度 | 1024 |
| 并行策略 | 4×4090 DDP（同预训练） |

## 数据概览

| 数据集 | 原始条数 | 处理后 |
|---|---|---|
| Alpaca-GPT4 | 52,002 | 52,002 单轮（instruction+input → user, output → assistant） |
| Dolly 15k | 15,011 | 15,011 单轮（instruction+context → user, response → assistant） |
| OASST train | 37,783 msg / 3,495 trees（过滤后） | 3,471 条多轮路径 |
| OASST valid | 1,936 msg / 188 trees（过滤后） | 187 条多轮路径 |

**总计**：Train 63,782 条 / Valid 6,889 条，其中多轮 1,334 条。平均长度 ~196 tokens，截断率 ~0.6%。

## Chat Template

```
单轮：
<|system|>
You are a helpful assistant.
<|user|>
{query}
<|assistant|>
{response}<|endoftext|>

多轮（OASST）：
<|system|>
You are a helpful assistant.
<|user|>
{第一轮问题}
<|assistant|>
{第一轮回复}<|endoftext|>
<|user|>
{追问}
<|assistant|>
{追问回复}<|endoftext|>
```

System 只出现一次在开头。每个 assistant 回复后紧接 `<|endoftext|>`。

## Loss Mask 设计

### 预处理阶段

逐段 tokenize，分别标注 `assistant_mask`（保存在 `.pt` 文件中）：

| 段 | assistant_mask |
|---|---|
| `<\|system\|>\n{system_prompt}\n` | 全 0 |
| `<\|user\|>\n{query}\n` | 全 0 |
| `<\|assistant\|>\n{response}` | **全 1** |
| `<\|endoftext\|>\n` | 全 0 |

语义：`assistant_mask[i] = True` 表示 `input_ids[i]` 来自 assistant 回复（含 `<|assistant|>` 标签本身）。

### Dataset 阶段（SFTDataset.__getitem__）

```python
input_ids  = stored_ids[:-1]       # 去掉最后一位
target_ids = stored_ids[1:]        # 右移一位
loss_mask  = assistant_mask[:-1]   # 同 input_ids 对齐
```

`loss_mask[t] = True` 表示「当前位置的 input token 属于 assistant 段」。这带来的效果是：

- 从 `<|assistant|>` 开始算 loss → 模型学会一旦看到 `<|assistant|>` 就开始生成回复
- 一直算到 `resp[-1]` → 包含预测 `<|eos|>`，模型学会适时停止
- **不算**「预测 `<|assistant|>` 标签」的 loss（推理时标签由 ChatEngine 手动拼接）

### Trainer 阶段

**NTP**：`cross_entropy_masked(target_ids, ntp, loss_mask)`

**MTP**：第 i 层 MTP 预测 `target_ids[:, i+1:]`，输入侧截断：

```python
cross_entropy_masked(
    target_ids[:, i + 1 :],    # 第 i 层 MTP 的目标
    m[:, : -(i + 1)],           # MTP 输出（右截断以对齐 target）
    loss_mask[:, : -(i + 1)]   # mask 也同步右截断（input-oriented）
)
```

MTP mask 从右截断的本质：MTP 头在位置 t 拿到的 hidden state 编码的是 `input[0..t]`，判断该位置是否该算 loss 只需看 `loss_mask[t]`（当前位置的 input 是否属于 assistant 段），与预测目标的位置无关。

长度处理：超长左截断（保留最后的 assistant 回复），不足右 pad `<|endoftext|>`（id=256, mask=0）。

## 预处理输出格式

```python
# data/llm/sft/train.pt 和 valid.pt
{
    "input_ids": tensor(N, 1024),        # int64
    "assistant_mask": tensor(N, 1024),    # bool
}
```

## TODO 列表

- [x] 实现：`scripts/preprocess_sft.py` — 数据预处理
- [x] 实现：`src/loss.py` — 新增 `cross_entropy_masked`
- [x] 实现：`src/dataset.py` — `SFTDataset` 类
- [x] 实现：`src/trainer.py` — `SFTTrainer` + `SFTTrainerArgs`
- [x] 验证：预处理后抽查 mask 正确性
- [x] 实现：`scripts/sft.py` + `scripts/sft.sh` — 训练入口
- [ ] 验证：单 GPU 跑 100 步确认 loss 下降、无 NaN
- [ ] 验证：加载 SFT checkpoint 用 `scripts/inference.py --chat` 测试对话效果
