# DeepSeekV4 预训练工作计划

**最后更新**: 2026-06-20（依据 4×4090 DDP + DGX Spark 实测数据）

## 最终决定：small（305M）× 4×RTX 4090 DDP

| 项 | 值 |
|---|-----|
| 配置 | small（d_model=1024, n_layer=8, n_experts=6） |
| 参数量 | 305M |
| 序列长度 | 1024 |
| 并行策略 | 4×4090 DDP（非 FSDP） |
| 吞吐（实测） | **14,578 tok/s**（3,645 tok/s/GPU） |
| 单步耗时 | 1.1 s（effective batch = 16,384 tokens） |
| 单卡显存 | rank0 ~19 GB, rank1-3 ~15.5 GB（nvidia-smi） |

## 训练时间估计（实测通量）

| 训练量 | 耗时 | 说明 |
|--------|------|------|
| 100M tokens | ~1.9 h | 冒烟测试 |
| 500M tokens（TinyStories ~1 epoch） | ~9.5 h | 验证 loss 收敛 |
| 1B tokens | ~19 h | 初见模型能力 |
| 3B tokens（TinyStories ~6 epoch） | ~2.4 d | 充分训练 |
| 5B tokens | ~4 d | TinyStories 10 epoch 或 OWT ~1.5 epoch |
| **8B tokens** | **~6.3 d** | **推荐：Chinchilla 最优（305M×~20=6B）** |

> 推荐训 5-8B tokens，一周内完成。此时数据量恰好匹配 305M 参数的 Chinchilla 最优（~6B tokens），不会欠拟合。

## 硬件实测数据

### DGX Spark（单卡，用于本地开发验证）

| Config | Params | seq | bs | tok/s | Step | Mem |
|--------|--------|-----|----|-------|------|-----|
| tiny | 106M | 512 | 4 | 5,370 | 381 ms | 3.9 GB |
| small | 305M | 1024 | 4 | 2,982 | 1.4 s | 12.3 GB |
| prefer | 684M | 2048 | 4 | 1,948 | 4.2 s | 34.2 GB |
| medium | 1.04B | 2048 | 4 | 1,332 | 6.2 s | 45.0 GB |

### 4×RTX 4090 DDP（正式训练）

| Config | Params | seq | tok/s/G | total tok/s | Step | Status |
|--------|--------|-----|---------|-------------|------|--------|
| tiny | 106M | 512 | 3,660 | 14,641 | 559 ms | ✓ |
| **small** | **305M** | **1024** | **3,645** | **14,578** | **1.1 s** | **✓** |
| prefer | 684M | 2048 | - | - | - | ✗ OOM |
| medium | 1.04B | 2048 | - | - | - | ✗ OOM |
| default | 1.14B | 2048 | - | - | - | ✗ OOM |

> 显存用的是 nvidia-smi（gpustat）读数，包含 CUDA context、NCCL buffers、cuBLAS workspace，比 PyTorch `max_memory_allocated()` 高 2-4 GB。

## 为什么选 small 而非 prefer / medium

prefer（684M）和 medium（1.04B）在 4090 DDP 下全部 OOM——模型+优化器状态占 ~9GB，剩余 15GB 不够装 seq=2048 的激活值。要跑它们必须上 FSDP，复杂度远超学习收益。而 small 的 305M 配 8B tokens 恰好 Chinchilla 最优，训练更"科学"。

## 执行计划

### 阶段 0：本地验证（DGX Spark, tiny）

用 DGX Spark 验证管线正确性，再搬到 4090 正式训。

| 步骤 | 配置 | 数据 | 步数 | 预计 | 目的 |
|------|------|------|------|------|------|
| 0a | tiny, bs=4, seq=512 | 随机 | 150 | ~1 min | 验证 forward+backward+optimizer |
| 0b | tiny, bs=4, seq=512 | TinyStories | 100 | ~2 min | 验证真实数据管线 |
| 0c | small, bs=4, seq=1024 | TinyStories | 500 | ~4 min | 确认 small 配置在真实数据上正常 |

### 阶段 1：正式预训练（4×4090 DDP）

```
torchrun --nproc_per_node=4 scripts/train.py \
  --config small \
  --data-train data/tinystories_train.bin \
  --data-val data/tinystories_valid.bin \
  --batch-size 4 \
  --grad-accum 1 \
  --max-seq-len 1024 \
  --total-steps 500000 \
  --lr 3e-4 \
  --warmup-steps 2000 \
  --output-dir checkpoints/run_small_tinystories
```

> effective batch = 4 GPUs × 4 batch × 1024 seq = 16,384 tokens/step
> 500K steps × 16,384 = 8.2B tokens → ~6.3 天

## TODO 列表

- [x] DeepSeekV4 模型代码
- [x] BPE Tokenizer
- [x] CE Loss + Indexer KL Loss
- [x] Muon 优化器（**待修复：Nesterov 动量**）
- [x] AdamW 优化器
- [x] `group_params` / `get_indexer_params`
- [x] Benchmark 脚本 + 实测数据
- [x] `.detach()` fix（Compressor buffer 写入需 detach，PyTorch 2.12 要求）
- [x] **TODO-1**: 修复 Muon Nesterov（对 `momentum * state["m"] + p.grad` 做 NS）
- [x] **TODO-2**: `scripts/verify_training.py` — 随机数据 150 步全流程验证
- [x] **TODO-3**: `src/dataset.py` + `scripts/preprocess_data.py` — 数据预处理管线
- [x] **TODO-4**: `scripts/train.py` — 正式训练脚本，支持 `--distributed` 开关
- [x] **TODO-5**: DGX Spark 上用 tiny 冒烟验证（阶段 0）
- [x] **TODO-6**: 4×4090 上用 small 正式训练（阶段 1）
