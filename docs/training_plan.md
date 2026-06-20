# DeepSeekV4 预训练工作计划

**最后更新**: 2026-06-20（依据 benchmark 实测数据更新）

## 硬件实测数据（DGX Spark, bf16, 含 forward+backward+3×optimizer）

| Config | Params | d_model | n_layer | n_experts | tok/s (bs=4,seq=128) | Peak Mem |
|--------|--------|---------|---------|-----------|----------------------|----------|
| tiny | 106M | 768 | 4 | 4 | 2,166 | 1,549 MB |
| small | 305M | 1024 | 8 | 6 | 788 | 3,042 MB |
| medium | 1.04B | 1536 | 12 | 8 | 233 | 7,679 MB |
| default | 1.14B | 2048 | 7 | 8 | 194 | 8,567 MB |

### tiny 的 Batch/Seq Sweep

| Batch | Seq | tok/s | Mem | Tokens/step |
|-------|-----|-------|-----|-------------|
| 2 | 128 | 1,158 | 1,354 MB | 256 |
| 4 | 128 | 2,172 | 1,532 MB | 512 |
| 8 | 128 | 3,652 | 2,282 MB | 1,024 |
| 4 | 256 | 3,562 | 2,305 MB | 1,024 |
| 8 | 256 | 5,289 | 3,919 MB | 2,048 |
| 4 | 512 | 5,198 | 3,919 MB | 2,048 |

> 关键发现：**相同 tokens/step 的吞吐几乎相同**（bs=8,seq=128 和 bs=4,seq=256 都是 2048 tokens/step，通量 3,652 vs 3,562，差 < 3%）。这意味着训练时选更大的 seq_len 更好（同样通量，但更长的上下文）。

---

## DGX Spark：推荐用 tiny（106M）

medium 和 default 在 DGX Spark 上每秒不到 250 tokens，完全不可行。**只有 tiny 是实用选项**。

### 预估训练吞吐（seq=1024）

| Batch | 预估 tok/s | 预估显存 | 等效 tokens/step |
|-------|-----------|---------|-----------------|
| 2 | ~4,000 | ~4 GB | 2,048 |
| 4 | ~5,500 | ~7 GB | 4,096 |

> 基于 sweep 规律外推：同等 tokens/step 吞吐接近。seq=1024 时 bs=4 约等于 sweep 中 bs=8,seq=256（都是 4,096 tokens/step）。

### 训练时间估计

| 数据量 | bs=2,seq=1024 (~4K tok/s) | bs=4,seq=1024 (~5.5K tok/s) |
|--------|--------------------------|---------------------------|
| 100M tokens | ~7 h | ~5 h |
| 500M tokens（TinyStories 1 epoch） | ~35 h | ~25 h |
| 1B tokens | ~70 h (~3 d) | ~50 h (~2 d) |
| 2B tokens | ~5.8 d | ~4.2 d |

**推荐**：`bs=4, seq=1024`，TinyStories 训 2-3 个 epoch（~1-1.5B tokens），在 DGX Spark 上一周内可以完成。

### 可选：tiny+（约 160M）

如果 tiny 训得太快想挑战更大模型：`d_model=768, n_layer=6, n_experts=6`，预估 ~3,500 tok/s，一周能训 ~2B tokens。

---

## 4× RTX 4090：推荐用 small（305M）+ DDP

### 为什么 DDP 而不是 FSDP？

small（305M）在单卡上的显存占用：
- 模型权重（bf16）：~610 MB
- 梯度（bf16）：~610 MB
- AdamW 状态（fp32 m+v）：~2.4 GB
- Muon 动量缓冲（bf16）：~610 MB
- 激活值（bs=2,seq=1024）：~3-5 GB
- **单卡总计：~8-10 GB** ← 远在 24 GB 以内

**DDP 完全够用，不需要 FSDP 的复杂度。** 如果用 medium（1.04B）单卡 ~17-20 GB 就紧张了，且吞吐太低（一周训不到 2B tokens）。

### DDP 需要改什么

集中在训练脚本，模型代码零改动：

1. **启动方式**：`torchrun --nproc_per_node=4 scripts/train.py`
2. **模型 wrap**：`model = DDP(model, device_ids=[local_rank])`
3. **数据采样**：`DistributedSampler(dataset)` 替代默认 shuffle
4. **Checkpoint**：只在 rank 0 保存，加载时先 map 到 CPU

约 30 行改动。

### 4×4090 预估吞吐

单卡 4090 带宽约 1 TB/s，是 DGX Spark（273 GB/s）的 3.6 倍。small 在 DGX Spark 上 788 tok/s，单卡 4090 预估 ~2,800-3,500 tok/s（@seq=1024）。

4 卡 DDP：~10,000-13,000 tok/s（扣掉 ~10% all-reduce 通信开销）。

### 训练时间估计（small, 4×4090 DDP）

| 数据量 | 预估耗时 |
|--------|---------|
| 1B tokens | ~22-28 h |
| 3B tokens（TinyStories 6 epoch） | ~3-4 d |
| 5B tokens（TinyStories 10 epoch / OWT ~1.5 epoch）| ~4-6 d |
| 10B tokens（OWT ~3 epoch） | ~9-12 d（超出一周） |

**推荐**：训 3-5B tokens，一周内轻松完成。TinyStories 多个 epoch 或 OWT 一个 epoch 都可行。

---

## 最终推荐对照表

| 硬件 | 配置 | 参数量 | 数据 | 预估耗时 |
|------|------|--------|------|---------|
| DGX Spark | tiny, bs=4, seq=1024 | 106M | TinyStories 2 epoch (~1B tok) | ~2 d |
| DGX Spark | tiny+, bs=4, seq=1024 | ~160M | TinyStories 2 epoch (~1B tok) | ~3.3 d |
| 4×4090 DDP | small, bs=2×4, seq=1024 | 305M | TinyStories 6 epoch / OWT 1 epoch (~3-5B tok) | ~3-5 d |

---

## TODO 列表（与之前一致，补充了 Muon Nesterov fix）

- [x] DeepSeekV4 模型代码
- [x] BPE Tokenizer
- [x] CE Loss + Indexer KL Loss
- [x] Muon 优化器（**待修复：Nesterov 动量**）
- [x] AdamW 优化器
- [x] `group_params` / `get_indexer_params`
- [x] Benchmark 脚本 + 结果
- [ ] **TODO-1**: 修复 Muon Nesterov（`state["m"]` 计算完后，对 `momentum * state["m"] + p.grad` 做 NS，而非直接对 `state["m"]` 做）
- [ ] **TODO-2**: 用随机数据跑通完整训练循环（150 步），包含 LM loss + KL loss + grad_clip + 双优化器 step
- [ ] **TODO-3**: 数据预处理管线（dataset.py + preprocess_data.py）
- [ ] **TODO-4**: 正式训练脚本（train.py），支持 DDP 开关
- [ ] **TODO-5**: DGX Spark 上训 tiny × TinyStories
- [ ] **TODO-6**（可选）: 4×4090 上训 small × OWT

### 文件总览

| 文件 | 操作 | 所属 TODO |
|------|------|----------|
| src/optimizer.py | 修复 Muon Nesterov | TODO-1 |
| scripts/verify_training.py | **新建** | TODO-2 |
| src/dataset.py | **新建** | TODO-3 |
| scripts/preprocess_data.py | **新建** | TODO-3 |
| scripts/train.py | **新建** | TODO-4 |
