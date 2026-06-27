# DeepSeekV4 SFT 工作计划

**最后更新**: 2026-06-27

## SFT 是什么？它和预训练有什么不同？

预训练的目标是让模型「学会语言」——语法、事实知识、推理能力。你在 3B tokens 上做的就是这个。

SFT（Supervised Fine-Tuning，监督微调）的目标完全不同：**让模型学会「对话格式」和「遵循指令」**。你不再给它喂一大坨连续文本让它续写，而是给它看「用户问 → 助手答」的例子，让它学会这种交互模式。

核心区别：

| | 预训练 | SFT |
|---|---|---|
| 数据格式 | 连续文本，无结构 | 有结构的对话（system/user/assistant 轮流） |
| 数据量 | 越大越好（3B tokens） | 少量高质量即可（~50k 条对话，约 50-100M tokens） |
| Loss 计算 | 所有 token 都参与 | **只在 assistant 回复上计算 loss** |
| 学习率 | 较高（1e-4） | 较低（1e-5 ~ 5e-5） |
| 训练轮数 | 1 epoch 或更少 | 通常 2-4 epochs |
| 目标 | 学习世界知识 | 学习行为模式（格式对齐） |

---

## 你的数据情况

| 数据集 | 条数 | 格式 | 特点 |
|---|---|---|---|
| Alpaca-GPT4 | 52,002 | parquet（instruction/input/output） | GPT-4 生成的指令数据，质量高 |
| Dolly 15k | 15,010 | jsonl（instruction/context/response） | 人工标注，涵盖分类/摘要/QA 等 |
| OASST | 84,437 条消息 | parquet（对话树） | 多轮对话，但一半是非英文，需要过滤 |

**重要**：OASST 的 84,437 是「消息数」，不是「对话数」。这些消息通过 `parent_id` 构成对话树。过滤英文+高质量后大约剩 39,000 条消息，组成约 10,000-15,000 个对话树。

过滤后三个数据集加起来大约 **7-8 万条对话**。

---

## 你需要做的决策

以下每个决策我都会解释 trade-off，给出建议，但最终由你来选。

### 决策 1：Train/Val 怎么划分？

SFT 的验证集主要用来**监控 loss**——一旦 val loss 不降反升，说明开始过拟合了，该停了。不需要像预训练那样留 20%。

**建议**：90/10 随机划分。但 OASST 需要按对话树划分（同一个树的所有路径要么全在 train 要么全在 val，否则会数据泄漏——同一个树的多个回复高度相关）。

### 决策 2：训练几个 Epoch？

这是你最担心的问题——预训练时第二个 epoch 就过拟合了，SFT 会吗？

**SFT 和预训练的过拟合是不同的概念**：

- **预训练过拟合** = 模型在背训练文本。预训练数据是「知识载体」，背下来没用，丧失了泛化到新文本的能力。
- **SFT「格式过拟合」** = 模型牢牢记住 `<|user|>...<|assistant|>...` 的交互格式。这**恰恰是你想要的**。
- **SFT「内容过拟合」** = 模型逐字背诵训练数据里的回复，丧失了多样性。这才是需要警惕的。

对于 0.3B 模型 + ~70k 数据：
- **2-3 个 epoch 是合理范围**。第一个 epoch 学格式，第二第三个 epoch 精炼回复质量。
- 实际做法：**盯着 val loss，它不再下降后就再多跑 0.5 个 epoch 然后停**。
- 小模型（0.3B）比大模型更容易内容过拟合，3 个 epoch 差不多是上限。

### 决策 3：要不要混入预训练数据？

**这是 SFT 领域的一个重要话题，目前没有标准答案。**

两种主流观点：

**观点 A（不混）**—— LIMA 论文（Meta, 2023）发现，只用 1000 条高质量指令数据微调 LLaMA-65B，就能达到接近 GPT-4 的水平。SFT 只是「激活」模型已有的能力，不需要预训练数据来「提醒」模型。

**观点 B（混 5-10%）**—— 一些实践者发现，纯 SFT 后模型会出现「alignment tax」——通用能力（续写、常识推理）下降。混少量预训练数据相当于「正则化」，告诉模型「别忘了你还会干别的」。

**我的建议**：先不混，跑一版纯 SFT。用 `scripts/inference.py` 测试，感受一下效果。如果发现模型只会机械回复、丧失了多样性，再加 5-10% 预训练数据跑第二版。

如果你决定混入，最简单的方式：从 `train.bin` 随机采样 chunk，loss_mask 全设为 1（所有 token 都参与 loss，和预训练一样），不需要包装成对话格式。

### 决策 4：MTP 和 Indexer KL Loss 要不要保留？

预训练时总 loss 是 `ntp_loss + 0.3 * mtp_loss + 0.5 * kl_loss`。SFT 时这些还要吗？

**MTP（Multi-Token Prediction）**：
- MTP 的作用是提升训练效率 + 推理时的投机解码加速。
- 对于 SFT（数据量小、训练步数少），MTP 的额外训练收益有限。
- 但保留它也没有坏处——如果 SFT 后你想用投机解码加速推理，SFT 阶段继续训练 MTP head 是必要的。

**Indexer KL Loss**：
- Indexer 负责选择注意力要关注的压缩块。SFT 阶段数据分布从连续文本变成对话，indexer 也需要微调以适应新的注意力模式。
- KL loss 计算量很小，保留成本几乎为零。

**建议**：**全保留**。代码改动最小，且每个组件都应该有机会适应新的数据分布。

**注意**：MTP loss 需要和 NTP loss 一样做 loss masking。MTP 第 i 层预测的是 i+1 步之后的 token：

```
MTP 第 0 层：预测 target[:, 1:]   → mask 对应 loss_mask[:, 1:]
MTP 第 1 层：预测 target[:, 2:]   → mask 对应 loss_mask[:, 2:]
...
```

### 决策 5：多轮对话怎么处理？

OASST 的对话树结构比较复杂——一个 prompter 的问题可能有多个 assistant 回复，每个回复后面又有 follow-up 问题。

**选项 A（推荐）：选最优路径**。从根开始，在 assistant 分支选 rank 最高 / review_result 最好的那条，继续往下走。每个对话树产出一条多轮对话样本。

**选项 B：拆成多个单轮**。把树里每个 (prompter, assistant) 对当作独立样本。缺点：丢失了多轮上下文。

**选项 C：保留所有路径**。DFS 遍历所有可能的对话路径。缺点：会产生大量低质量/重复路径。

**建议选 A**。多轮对话数据很宝贵，保留多轮结构能让模型学会「根据对话历史回答追问」。

---

## 你需要做的工作（分步指南）

### 步骤 1：数据预处理

**目标**：将 Alpaca、Dolly、OASST 三个数据集统一转换成 chat template 格式并 tokenize。

**Chat Template 格式**（三段式）：

```
<|system|>
{system prompt，可为空}
<|user|>
{用户内容}
<|assistant|>
{助手回复}<|endoftext|>
```

对于你现有的三个数据集，都没有显式的 system prompt，所以 `<|system|>\n` 后面直接跟 `<|user|>` 即可（或者放一句通用的如 "You are a helpful assistant."）。

多轮对话（OASST）：
```
<|system|>
{system prompt}
<|user|>
{第一轮问题}
<|assistant|>
{第一轮回复}<|endoftext|>
<|user|>
{追问}
<|assistant|>
{追问回复}<|endoftext|>
```

多轮中 system 只出现一次（在开头），后续轮次只有 user/assistant 交替。

#### Loss Mask 的构建

这是 SFT 最核心的概念。你要构建一个和 input_ids 等长的 mask 数组，标记哪些 token 是「模型需要学会预测的」（即 assistant 的回复内容），哪些是「不需要学的」（即 system prompt、user 提问和特殊标签）。

**方法**：逐段 tokenize，分别记录 mask：

1. tokenize `<|system|>\n{system_prompt}\n` → mask 全 0
2. tokenize `<|user|>\n{用户内容}\n` → mask 全 0
3. tokenize `<|assistant|>\n{助手回复}` → **mask 全 1**
4. tokenize `<|endoftext|>` → mask 全 0

然后把各段的 token_ids 拼起来，mask 拼起来，就得到了 `(input_ids, assistant_mask)`。

对于多轮对话，重复步骤 2-4 即可（system 只在开头出现一次）。

#### 长度处理

- 超过 `max_seq_len`（1024）→ 从**左边**截断。因为截断的是早期的对话轮次/system prompt，保留了最后的 assistant 回复（最重要的部分）。
- 不足 `max_seq_len` → 右侧 pad `<|endoftext|>`（token id=256），mask=0。

#### 输出格式

保存为 `.pt` 文件：

```python
{
    "input_ids": tensor(N, max_seq_len),       # int64
    "assistant_mask": tensor(N, max_seq_len),   # bool, True 表示该位置的 INPUT token 是 assistant 说的
}
```

注意：`assistant_mask[i] = True` 表示 `input_ids[i]` 这个 token 是 assistant 回复的一部分。后面做 loss 时需要错位对齐（见步骤 2）。

#### OASST 多轮提取算法

1. 过滤：`lang == 'en'` AND `review_result == True`
2. 按 `message_tree_id` 分组，每组构建一棵树
3. 找到根节点（`parent_id` 为空，`role='prompter'`）
4. 从根开始 DFS：
   - 在 prompter 节点：看所有 assistant 子节点，选 rank 最高（或 review_count 最多）的那个
   - 在 assistant 节点：看所有 prompter 子节点（follow-up 问题），选一个继续往下
   - 限制深度 ≤ 6 条消息（3 轮对话）
5. 收集路径：`[prompter, best_assistant, followup_prompter, best_assistant, ...]`

#### Train/Val 划分

- 90/10 随机划分
- OASST 按 conversation tree 级别划分（同一个 tree 的所有路径必须在同一集合，防止数据泄漏）

---

### 步骤 2：SFT Dataset 类

**目标**：创建 `SFTDataset`（建议放在 `src/sft_dataset.py`），加载步骤 1 预处理好的 `.pt` 文件，返回 `(input_ids, target_ids, loss_mask)`。

**参考**：你现有的 `PretrainDataset`（`src/dataset.py`）返回 `(x, y)` 其中 `y = x[1:]`。SFT 版本多加一个 loss_mask。

**关键**：`target_ids` 的构造和 loss_mask 的错位对齐。

```
input_ids:      [tok0, tok1, tok2, ..., tok_{S-1}]
target_ids:     [tok1, tok2, tok3, ..., 0          ]   ← 右移一位，最后填 0（会被 mask 掉）
assistant_mask: [True,  False, True,  ..., False    ]   ← 来自预处理

loss_mask[t] 应该表示 target_ids[t] 是不是 assistant token。
由于 target_ids[t] = input_ids[t+1]，所以：
    loss_mask[t] = assistant_mask[t+1]
    loss_mask[-1] = 0  （target_ids[-1] 是 dummy）
```

**预训练数据混合**（如果你决定做）：
- 以概率 `pretrain_mix_ratio` 随机从 `PretrainDataset` 采样
- 此时 loss_mask 全 1（所有 token 都参与计算，和预训练一致）

---

### 步骤 3：Masked Cross-Entropy Loss

**目标**：在 `src/loss.py` 中新增一个支持 mask 的 cross-entropy 函数。

**为什么需要新函数？**你现有的 `cross_entropy` 对所有位置一视同仁地取平均。SFT 需要只对 assistant token 取平均：

```python
# 现有（预训练）：所有位置平均
loss = cross_entropy(target_ids, logits)  # mean over ALL positions

# SFT 需要：只对 assistant 位置平均
loss = (cross_entropy_per_position * loss_mask).sum() / loss_mask.sum()
```

**实现要点**：
1. 正常计算 `log_softmax`
2. 取出每个位置对应 target token 的 log probability → 得到 `nll`（shape: `(B, S)`）
3. `masked_nll = nll * loss_mask`（0 的位置贡献 0，1 的位置保留原值）
4. `return masked_nll.sum() / (loss_mask.sum() + 1e-8)`（防止除零）

---

### 步骤 4：SFT Trainer

**目标**：创建 `SFTTrainer`（建议放在 `src/sft_trainer.py`），继承 `PretrainTrainer`，覆写需要改的方法。

**需要覆写的方法**：

#### `build_model_and_optimizers()`

1. 调用 `super().build_model_and_optimizers()` 创建模型 + 优化器（随机初始化）
2. **然后加载预训练权重**：从 `--pretrained-ckpt` 读取 `model_state_dict` 并加载
3. 优化器**不加载**预训练的状态——SFT 用更低 LR，需要全新的优化器状态

```python
state = torch.load(pretrained_ckpt, ...)
model.load_state_dict(state["model_state_dict"])
# optimizers 保持新建状态，不做 load_state_dict
```

#### `build_dataloaders()`

- 用 `SFTDataset` 加载预处理好的 `.pt` 文件
- 根据 epochs 和数据集大小自动算 `total_steps`：

```
steps_per_epoch = ceil(len(train_dataset) / (batch_size × world_size × grad_accum))
total_steps = steps_per_epoch × epochs
```

- `DistributedSampler` + `DataLoader`（照抄 `PretrainTrainer` 的做法）

#### `train_step(batch, is_last_micro)`

batch 现在是三元组 `(input_ids, target_ids, loss_mask)`：

```python
input_ids, target_ids, loss_mask = batch

ntp, mtp_list, idx_data = self.model(input_ids)

# NTP loss（masked）
ntp_loss = cross_entropy_masked(target_ids, ntp, loss_mask)

# MTP loss（masked，注意 loss_mask 的偏移对齐）
mtp_loss = sum(
    cross_entropy_masked(
        target_ids[:, i+1:],      # MTP 第 i 层预测的目标
        m[:, :-(i+1)],             # MTP 第 i 层的输出
        loss_mask[:, i+1:]         # 对应的 mask（偏移 i+1）
    )
    for i, m in enumerate(mtp_list)
)

# KL loss 不变（它不依赖 loss mask）
kl_loss = sum(indexer_kl_loss(iscore, idx, wc.detach()) for ...)

lm_loss = ntp_loss + 0.3 * mtp_loss
total_loss = (lm_loss + 0.5 * kl_loss) / self.grad_accum
```

#### `validate()`

和 `PretrainTrainer.validate()` 结构相同，只是用 `cross_entropy_masked` 替代 `cross_entropy`。

#### `get_lr(step)`

用和预训练相同的 cosine annealing schedule（`src/optimizer.py` 里的 `cosine_annealing_lr_schedule`），只是 LR 参数不同。

---

### 步骤 5：训练入口和启动脚本

**参考**：`scripts/pretrain.py` 和 `scripts/pretrain.sh`，照葫芦画瓢。

**关键超参数**：

| 参数 | 推荐值 | 为什么 |
|---|---|---|
| LR | 2e-5 | 预训练 LR（1e-4）的 1/5。SFT 改动小，LR 太高会破坏预训练学到的知识 |
| LR min | 2e-6 | Cosine 衰减底线 |
| Warmup | 200 steps | 快速 ramp-up，不需要预训练那种 2000 步的 warmup |
| Epochs | 2-3 | 见决策 2 |
| Batch size | 4/GPU | 和预训练一致 |
| Grad accum | 1 | 和预训练一致 |
| Max grad norm | 1.0 | 和预训练一致 |

**训练时长估算**（~70k 训练样本，4 GPU × batch 4 × accum 1）：

- steps_per_epoch ≈ 70,000 / 16 ≈ 4,375
- 3 epochs ≈ 13,125 steps
- 单步 ~1.1s → 总共约 **4 小时**

---

## 总结

| 文件 | 做什么 | 参考 |
|---|---|---|
| `scripts/preprocess_sft.py` | 加载三个数据集 → 统一 chat template → tokenize → 构建 assistant_mask → 保存 .pt | 参考 `scripts/pre_tokenization.py` 的 tokenize 流程 |
| `src/loss.py`（修改） | 新增 `cross_entropy_masked` | 参考现有 `cross_entropy`（第 4-7 行） |
| `src/sft_dataset.py` | `SFTDataset`，返回 `(input_ids, target_ids, loss_mask)` | 参考 `src/dataset.py` 的 `PretrainDataset` |
| `src/sft_trainer.py` | `SFTTrainer(PreTrainer)`，覆写 train_step / build_dataloaders / validate | 参考 `src/trainer.py` 的 `PretrainTrainer` |
| `scripts/sft.py` | CLI 入口 | 参考 `scripts/pretrain.py` |
| `scripts/sft.sh` | Shell 启动脚本（torchrun） | 参考 `scripts/pretrain.sh` |

## TODO 列表

- [ ] 决策：确认 MTP + Indexer KL 是否保留
- [ ] 决策：确认是否混入预训练数据
- [ ] 决策：确认 epochs 数
- [ ] 实现：`scripts/preprocess_sft.py` — 数据预处理
- [ ] 实现：`src/loss.py` — 新增 `cross_entropy_masked`
- [ ] 实现：`src/sft_dataset.py` — SFT Dataset 类
- [ ] 实现：`src/sft_trainer.py` — SFT Trainer
- [ ] 实现：`scripts/sft.py` + `scripts/sft.sh` — 训练入口
- [ ] 验证：预处理后抽查几条数据的 mask 是否正确
- [ ] 验证：单 GPU 跑 100 步确认 loss 下降、无 NaN
- [ ] 验证：加载 SFT checkpoint 用 `scripts/inference.py --chat` 测试对话效果
