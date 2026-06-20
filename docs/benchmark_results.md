# DGX Spark Training Throughput Benchmark

**Date**: 2026-06-20
**Device**: NVIDIA GB10 (DGX Spark), 128 GB unified memory, 273 GB/s bandwidth
**Environment**: CUDA 13.0, PyTorch 2.12.0, bfloat16
**Script**: `scripts/benchmark.py`

---

## Config Definitions

| Name | d_model | n_layer | n_experts | d_ff | n_heads | head_dim | Params |
|------|---------|---------|-----------|------|---------|----------|--------|
| tiny | 768 | 4 | 4 | 768 | 12 | 128 | 106M |
| small | 1024 | 8 | 6 | 1024 | 16 | 128 | 305M |
| medium | 1536 | 12 | 8 | 1536 | 16 | 192 | 1.04B |
| default | 2048 | 7 | 8 | 2048 | 16 | 256 | 1.14B |

All configs use `n_mtp_layer=1`, `expansion_rate=4`, `vocab_size=32000`.
Benchmarks include forward + LM loss + KL loss + backward + grad_clip + 3× optimizer step.

---

## Phase 1: Config Comparison (bs=4, seq=128)

| Config | tok/s | Step (ms) | 100 steps | 2000 steps | Peak Mem |
|--------|-------|-----------|-----------|------------|----------|
| tiny | 2,166 | 236 | 24 s | 7.9 min | 1,549 MB |
| small | 788 | 650 | 65 s | 21.7 min | 3,042 MB |
| medium | 233 | 2,200 | 3.7 min | 1.2 h | 7,679 MB |
| default | 194 | 2,642 | 4.4 min | 1.5 h | 8,567 MB |

## Phase 2: tiny Config — Batch/Seq Sweep

| Batch | Seq | tok/s | Step (ms) | Peak Mem | Tokens/step |
|-------|-----|-------|-----------|----------|-------------|
| 2 | 128 | 1,158 | 221 | 1,354 MB | 256 |
| 4 | 128 | 2,172 | 236 | 1,532 MB | 512 |
| 8 | 128 | 3,652 | 280 | 2,282 MB | 1,024 |
| 4 | 256 | 3,562 | 288 | 2,305 MB | 1,024 |
| 8 | 256 | 5,289 | 387 | 3,919 MB | 2,048 |
| 4 | 512 | 5,198 | 394 | 3,919 MB | 2,048 |

---

## Time-to-Token Estimates

| Config | tok/s | 100K tok | 1M tok | 10M tok | 100M tok | 1B tok |
|--------|-------|----------|--------|---------|----------|--------|
| tiny | 2,172 | 46 s | 7.7 min | 1.3 h | 12.8 h | 128 h |
| tiny (bs=8,seq=256) | 5,289 | 19 s | 3.2 min | 0.5 h | 5.3 h | 52 h |
| small | 788 | 2.1 min | 21 min | 3.5 h | 35 h | 353 h |
| medium | 233 | 7.2 min | 72 min | 12 h | 119 h | 1192 h |
| default | 194 | 8.6 min | 86 min | 14 h | 143 h | 1432 h |

---

## Training Plan Alignment

| TODO | Config | Steps | Plan Estimate | Measured | Status |
|------|--------|-------|---------------|----------|--------|
| 5a 冒烟 | tiny (bs=4) | 100 | ~5 min | ~24 s | OK |
| 5b 小规模 | small (bs=4) | 2000 | ~20 min | ~22 min | OK |
| 5c 中等 | medium (bs=4) | 30000 | 4-12 h | ~18.3 h | 偏高，建议降到 small |
| 5d 大规模 | default | 50000+ | 1-3 d | ~37 h+ | 不推荐，硬件不足 |

---

## 4× RTX 4090 Feasibility

### Memory (per GPU, with FSDP)

| Config | Peak Mem (1 GPU) | FSDP est. (4 GPUs) | 24 GB budget |
|--------|------------------|---------------------|--------------|
| tiny | 1.5 GB | ~0.9 GB | OK |
| small | 3.0 GB | ~1.8 GB | OK |
| medium | 7.7 GB | ~4.5 GB | OK |
| default | 8.6 GB | ~5.0 GB | OK |

All configs fit comfortably on 24 GB per card with FSDP sharding.
With 4× higher effective bandwidth, throughput should scale ~3-3.5×.

### Code changes needed

Changes are concentrated in the training script, model code is untouched:

1. **Launch**: `torchrun --nproc_per_node=4 scripts/train.py`
2. **FSDP wrap**: one `FullyShardedDataParallel` call per `TransformerBlock`
3. **Sampler**: `DistributedSampler` for the dataset
4. **Checkpoint**: save/load with `FULL_STATE_DICT` for portability

Estimated effort: ~50 lines added to training script.
