"""Benchmark training throughput with DDP on multi-GPU (e.g., 4× RTX 4090).

Usage:
    torchrun --nproc_per_node=4 scripts/benchmark_ddp.py

Each GPU holds the full model; DDP syncs gradients via all-reduce.
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import gc

from src.deepseek import DeepSeekV4, DSArgs
from src.loss import cross_entropy, indexer_kl_loss
from src.optimizer import Muon, AdamW, group_params, get_indexer_params, grad_clip

# ---------------------------------------------------------------------------
# Config definitions (same as single-GPU benchmark)
# ---------------------------------------------------------------------------
CONFIGS = {
    "tiny": dict(
        d_model=768,
        d_ff=768,
        n_layer=4,
        n_experts=4,
        d_moe_ff=768,
        n_heads=12,
        head_dim=128,
        attn_rank=256,
        output_lora=256,
        max_seq_len=512,
        compress_ratios=(0, 4, 128, 0, 0),
        description="d=768, L=4, E=4",
    ),
    "small": dict(
        d_model=1024,
        d_ff=1024,
        n_layer=8,
        n_experts=6,
        d_moe_ff=1024,
        n_heads=16,
        head_dim=128,
        attn_rank=384,
        output_lora=384,
        max_seq_len=2048,
        compress_ratios=(0, 0, 4, 128, 0, 0, 4, 128, 0, 0),
        description="d=1024, L=8, E=6",
    ),
    "medium": dict(
        d_model=1536,
        d_ff=1536,
        n_layer=12,
        n_experts=8,
        d_moe_ff=1536,
        n_heads=16,
        head_dim=192,
        attn_rank=512,
        output_lora=512,
        max_seq_len=2048,
        compress_ratios=(0, 0, 4, 128, 0, 0, 4, 128, 0, 0, 4, 128, 0, 0),
        description="d=1536, L=12, E=8",
    ),
    "prefer": dict(
        d_model=1536,
        d_ff=1536,
        n_layer=7,
        n_experts=8,
        d_moe_ff=1536,
        n_heads=16,
        head_dim=192,
        attn_rank=512,
        output_lora=512,
        max_seq_len=2048,
        compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
        description="d=1536, L=7, E=8",
    ),
    "default": dict(
        d_model=2048,
        d_ff=2048,
        n_layer=7,
        n_experts=8,
        d_moe_ff=2048,
        n_heads=16,
        head_dim=256,
        attn_rank=512,
        output_lora=512,
        max_seq_len=2048,
        compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
        description="d=2048, L=7, E=8 (DSArgs default)",
    ),
}


def format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def setup_distributed():
    """Initialize NCCL process group. Called on every rank."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_model_and_opt(args_dict, local_rank):
    """Build model (unwrapped), then wrap with DDP. Return DDP model + optimizers."""
    base = DSArgs.__dataclass_fields__
    args = DSArgs(**{k: v for k, v in args_dict.items() if k in base})
    args.max_batch_len = max(args.max_batch_len, 16)  # enough for DDP batch

    needed = args.n_layer + args.n_mtp_layer
    if len(args.compress_ratios) < needed:
        args.compress_ratios = args.compress_ratios + tuple(
            [0] * (needed - len(args.compress_ratios))
        )

    model = DeepSeekV4(args)
    model.train()
    model = model.to(local_rank)

    # Build optimizers BEFORE wrapping with DDP (params are shared by reference)
    muon_p, adamw_p = group_params(model)
    idx_p = get_indexer_params(model)
    muon_opt = Muon(muon_p, lr=3e-4, momentum=0.95, weight_decay=0.1)
    adamw_opt = AdamW(adamw_p, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    idx_opt = AdamW(idx_p, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    return ddp_model, muon_opt, adamw_opt, idx_opt, idx_p, args


def train_step(
    ddp_model, args, muon_opt, adamw_opt, idx_opt, idx_p, batch_size, seq_len
):
    """Single training step: forward → loss → backward → optimizer."""
    ids = torch.randint(
        0, args.vocab_size, (batch_size, seq_len), device=torch.cuda.current_device()
    )
    ntp, mtp_list, idx_data = ddp_model(ids)

    ntp_loss = cross_entropy(ids[:, 1:], ntp[:, :-1])
    mtp_loss = sum(cross_entropy(ids[:, 1:], m[:, :-1]) for m in mtp_list)
    kl_loss = sum(indexer_kl_loss(iscore, idx, wc) for (iscore, wc, idx) in idx_data)
    lm_loss = ntp_loss + 0.3 * mtp_loss

    # KL loss: manual grad on indexer params — bypasses DDP hooks.
    idx_grads = torch.autograd.grad(
        0.5 * kl_loss, idx_p, retain_graph=True, allow_unused=True
    )
    for p, g in zip(idx_p, idx_grads):
        if g is not None:
            p.grad = g if p.grad is None else p.grad.add_(g)
    # LM loss: DDP backward, all-reduces gradients across GPUs.
    lm_loss.backward()

    grad_clip(ddp_model.parameters(), max_norm=1.0)
    muon_opt.step()
    adamw_opt.step()
    idx_opt.step()
    ddp_model.zero_grad()


def benchmark_config(
    name,
    cfg_overrides,
    rank,
    world_size,
    local_rank,
    batch_size=4,
    seq_len=128,
    warmup=5,
    profile=15,
):
    """Benchmark one config. Only rank 0 prints results."""
    desc = cfg_overrides.get("description", name)

    torch.cuda.empty_cache()
    gc.collect()

    oom = torch.tensor([0], device=torch.cuda.current_device())
    try:
        ddp_model, muon_opt, adamw_opt, idx_opt, idx_p, args = build_model_and_opt(
            cfg_overrides, local_rank
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        oom[0] = 1
    dist.all_reduce(oom, op=dist.ReduceOp.MAX)
    if oom.item() > 0:
        if rank == 0:
            print(f"  OOM during model build — skipping {name}")
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        return None

    total_params = sum(p.numel() for p in ddp_model.parameters())
    trainable = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
    mem_mb = torch.cuda.memory_allocated() / 1024**2

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"Benchmark: {name} ({desc})")
        print(f"  GPUs: {world_size} × {torch.cuda.get_device_name(local_rank)}")
        print(
            f"  batch_per_gpu={batch_size}, seq_len={seq_len}, "
            f"total_batch={world_size * batch_size}"
        )
        print(f"{'='*65}")
        print(
            f"  Params: {format_params(total_params)} total, "
            f"{format_params(trainable)} trainable"
        )
        print(f"  GPU memory after build: {mem_mb:.0f} MB")

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    oom = torch.tensor([0], device=torch.cuda.current_device())
    try:
        for _ in range(warmup):
            train_step(
                ddp_model,
                args,
                muon_opt,
                adamw_opt,
                idx_opt,
                idx_p,
                batch_size,
                seq_len,
            )
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        oom[0] = 1
    torch.cuda.synchronize()
    dist.all_reduce(oom, op=dist.ReduceOp.MAX)
    if oom.item() > 0:
        if rank == 0:
            print(f"  OOM during warmup — skipping")
        del ddp_model, muon_opt, adamw_opt, idx_opt
        gc.collect()
        torch.cuda.empty_cache()
        return None

    # Profile
    torch.cuda.reset_peak_memory_stats()
    if rank == 0:
        print(f"  Profiling ({profile} steps)...")

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    try:
        start_evt.record()
        for _ in range(profile):
            train_step(
                ddp_model,
                args,
                muon_opt,
                adamw_opt,
                idx_opt,
                idx_p,
                batch_size,
                seq_len,
            )
        end_evt.record()
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        oom[0] = 1
    torch.cuda.synchronize()
    dist.all_reduce(oom, op=dist.ReduceOp.MAX)
    if oom.item() > 0:
        if rank == 0:
            print(f"  OOM during profile — skipping")
        del ddp_model, muon_opt, adamw_opt, idx_opt
        gc.collect()
        torch.cuda.empty_cache()
        return None

    elapsed_ms = start_evt.elapsed_time(end_evt)
    per_gpu_tokens = profile * batch_size * seq_len
    total_tokens = per_gpu_tokens * world_size
    per_gpu_tok_s = per_gpu_tokens / (elapsed_ms / 1000)
    total_tok_s = total_tokens / (elapsed_ms / 1000)
    step_time_ms = elapsed_ms / profile

    # Memory: report allocated (live tensors) and reserved (cached)
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    # All-reduce for max across ranks
    elapsed_t = torch.tensor([elapsed_ms], device=torch.cuda.current_device())
    alloc_t = torch.tensor([peak_alloc], device=torch.cuda.current_device())
    reserved_t = torch.tensor([peak_reserved], device=torch.cuda.current_device())
    dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)
    dist.all_reduce(alloc_t, op=dist.ReduceOp.MAX)
    dist.all_reduce(reserved_t, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"  Results (per GPU):")
        print(f"    Tokens/sec (1 GPU): {per_gpu_tok_s:,.0f}")
        print(f"    Step time (max):    {elapsed_t.item() / profile:.0f} ms")
        print(f"    Peak alloc (max):   {alloc_t.item():.0f} MB")
        print(f"    Peak reserved(max): {reserved_t.item():.0f} MB")
        print(f"    (nvidia-smi shows ~2-3GB more: CUDA context + NCCL buffers)")
        print(f"  Results (aggregate, {world_size} GPUs):")
        print(f"    Tokens/sec (total): {total_tok_s:,.0f}")

    del ddp_model, muon_opt, adamw_opt, idx_opt
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()  # ensure all ranks freed before next build

    if rank == 0:
        return {
            "name": name,
            "params": total_params,
            "tok_per_sec_per_gpu": per_gpu_tok_s,
            "tok_per_sec_total": total_tok_s,
            "step_time_ms": elapsed_t.item() / profile,
            "peak_mem_per_gpu": alloc_t.item(),
            "batch_per_gpu": batch_size,
            "seq_len": seq_len,
            "world_size": world_size,
            "total_batch": world_size * batch_size,
        }
    return None


def main():
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(42)
        print("DDP Training Throughput Benchmark")
        print(f"  World size: {world_size}")
        print(f"  Device:     {torch.cuda.get_device_name(local_rank)}")
        print(f"  CUDA:       {torch.version.cuda}")
        print(f"  PyTorch:    {torch.__version__}")
    else:
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(42 + rank)

    results = []

    # Phase 1: each config at its native max_seq_len
    if rank == 0:
        print("\n\n--- Phase 1: Config comparison (bs=4/GPU, native seq_len) ---")
    for name, cfg in CONFIGS.items():
        sl = cfg.get("max_seq_len", 128)
        r = benchmark_config(
            name, cfg, rank, world_size, local_rank, batch_size=4, seq_len=sl
        )
        if r:
            results.append(r)

    # Phase 2: tiny config — push batch size to fill GPU memory
    if rank == 0:
        print("\n\n--- Phase 2: tiny config, push batch size ---")
    for bs in [8, 16, 32, 48]:
        r = benchmark_config(
            f"tiny_bs{bs}",
            CONFIGS["tiny"],
            rank,
            world_size,
            local_rank,
            batch_size=bs,
            seq_len=128,
        )
        if r:
            results.append(r)
        else:
            break

    # Phase 3: longer sequences for key configs
    if rank == 0:
        print("\n\n--- Phase 3: longer sequences (bs=4/GPU) ---")
    for cfg_name in ["tiny", "prefer"]:
        for sl in [256, 512, 1024, 2048]:
            r = benchmark_config(
                f"{cfg_name}_seq{sl}",
                CONFIGS[cfg_name],
                rank,
                world_size,
                local_rank,
                batch_size=4,
                seq_len=sl,
            )
            if r:
                results.append(r)
            else:
                break  # OOM, stop pushing this config

    # Summary
    if rank == 0:
        print("\n\n" + "=" * 75)
        print("SUMMARY")
        print("=" * 75)
        print(
            f"{'Config':<16s} {'GPUs':>4s} {'bs/G':>5s} {'seq':>5s} "
            f"{'Total bs':>8s} {'tok/s/G':>10s} {'tok/s total':>12s} "
            f"{'Step(ms)':>9s} {'Mem/G':>8s}"
        )
        print("-" * 75)
        for r in results:
            print(
                f"{r['name']:<16s} {r['world_size']:>4d} "
                f"{r['batch_per_gpu']:>5d} {r['seq_len']:>5d} "
                f"{r['total_batch']:>8d} "
                f"{r['tok_per_sec_per_gpu']:>10,.0f} "
                f"{r['tok_per_sec_total']:>12,.0f} "
                f"{r['step_time_ms']:>9.0f} "
                f"{r['peak_mem_per_gpu']:>7.0f}MB"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
