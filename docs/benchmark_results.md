# Training Throughput Benchmark

**Date**: 2026-06-20
**Scripts**: `scripts/benchmark.py` (single GPU), `scripts/benchmark_ddp.py` (multi-GPU DDP)

---

## Config Definitions

| Name | d_model | n_layer | n_experts | d_ff | n_heads | head_dim | max_seq_len | Params |
|------|---------|---------|-----------|------|---------|----------|-------------|--------|
| tiny | 768 | 4 | 4 | 768 | 12 | 128 | 512 | 106M |
| small | 1024 | 8 | 6 | 1024 | 16 | 128 | 1024 | 305M |
| prefer | 1536 | 7 | 8 | 1536 | 16 | 192 | 2048 | 684M |
| medium | 1536 | 12 | 8 | 1536 | 16 | 192 | 2048 | 1.04B |
| default | 2048 | 7 | 8 | 2048 | 16 | 256 | 2048 | 1.14B |

All configs: `n_mtp_layer=1`, `expansion_rate=4`, `vocab_size=32000`, `bf16`.
Benchmark includes forward + LM loss + KL loss + backward + grad_clip + 3× optimizer step.

---

## DGX Spark (NVIDIA GB10, single GPU)

**Environment**: CUDA 13.0, PyTorch 2.12.0, bf16, 128 GB unified memory, 273 GB/s bandwidth

### Each config at its native max_seq_len (bs=4)

| Config | Params | seq | tok/s | Step | Peak Mem |
|--------|--------|-----|-------|------|----------|
| tiny | 106M | 512 | 5,370 | 381 ms | 3.9 GB |
| small | 305M | 1024 | 2,982 | 1.4 s | 12.3 GB |
| prefer | 684M | 2048 | 1,948 | 4.2 s | 34.2 GB |
| medium | 1.04B | 2048 | 1,332 | 6.2 s | 45.0 GB |

> **Note**: required a one-line fix in `src/deepseek.py` — `.detach()` on `kv` before writing to `kv_cache` buffer
> (PyTorch 2.12 rejects in-place buffer writes from tensors with `grad_fn`). Does not affect gradient flow.

### tiny Config Batch/Seq Sweep

| bs | seq | tok/s | Step |
|----|-----|-------|------|
| 2 | 128 | 1,158 | 221 ms |
| 4 | 128 | 2,172 | 236 ms |
| 8 | 128 | 3,652 | 280 ms |
| 4 | 256 | 3,562 | 288 ms |
| 8 | 256 | 5,289 | 387 ms |
| 4 | 512 | 5,370 | 381 ms |

---

## 4× RTX 4090 (DDP)

**Environment**: CUDA 12.1, PyTorch 2.4.0, bf16, 24 GB per GPU

### Each config at its native max_seq_len (bs=4/GPU)

| Config | Params | seq | tok/s/G | total tok/s | Step | Status | Rank 0 Mem | Rank 1-3 Mem |
|--------|--------|-----|---------|-------------|------|--------|------------|-------------|
| tiny | 106M | 512 | 3,660 | 14,641 | 559 ms | ✓ | ~6.5 GB | ~5 GB |
| small | 305M | 1024 | 3,645 | 14,578 | 1.1 s | ✓ | ~19 GB | ~15.5 GB |
| prefer | 684M | 2048 | - | - | - | ✗ OOM | ~24 GB | ~20-23 GB |
| medium | 1.04B | 2048 | - | - | - | ✗ OOM | - | - |
| default | 1.14B | 2048 | - | - | - | ✗ | - | - |

> Memory numbers are **gpustat** (nvidia-smi) readings, including CUDA context, NCCL buffers,
> cuBLAS workspaces. PyTorch `max_memory_allocated()` reports ~4 GB lower.

### tiny Config: push batch size (seq=128, 4 GPUs)

| bs/G | total bs | tok/s/G | total tok/s | Step |
|------|----------|---------|-------------|------|
| 4 | 16 | 770 | 3,080 | 666 ms |
| 8 | 32 | 1,526 | 6,104 | 670 ms |
| 16 | 64 | 3,079 | 12,314 | 666 ms |
| 32 | 128 | - | - | OOM |

### Memory: PyTorch vs gpustat

PyTorch `max_memory_allocated()` measures tensor allocations only.
`gpustat` (nvidia-smi) additionally includes: CUDA context (~1 GB),
NCCL ring buffers for all-reduce (~1-2 GB), cuBLAS/cuDNN workspaces,
PyTorch CUDA caching allocator reserved memory. Rank 0 DDP master
uses 2-4 GB more than other ranks.

---

## Key Takeaways

| Platform | Runs comfortably | Runs tight | OOM |
|----------|-----------------|------------|-----|
| DGX Spark (128 GB) | tiny, small, prefer, medium | - | - |
| 4×4090 DDP (24 GB) | tiny (seq=512) | small (seq=1024, 19 GB) | prefer/medium/default (seq=2048) |

1. **DGX Spark**: All configs run at native seq_len. Bottleneck is compute (273 GB/s), not memory.
2. **4×4090**: tiny + small work with DDP. prefer/medium/default need FSDP or smaller batch.
3. **`.detach()` fix**: Compressor buffer writes need `.detach()` on PyTorch 2.12. Safe — does not break gradient flow.
4. **DDP vs FSDP**: prefer/medium/default on 4090 should use FSDP instead of DDP to fit seq=2048.
