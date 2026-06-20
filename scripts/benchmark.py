"""Benchmark training throughput for different model configs on DGX Spark."""

import torch
import time
import gc
from src.deepseek import DeepSeekV4, DSArgs
from src.loss import cross_entropy, indexer_kl_loss
from src.optimizer import Muon, AdamW, group_params, get_indexer_params, grad_clip

# ---------------------------------------------------------------------------
# Config definitions
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
        max_seq_len=1024,
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


def benchmark_config(
    name, cfg_overrides, batch_size=4, seq_len=128, warmup_steps=5, profile_steps=15
):
    print(f"\n{'='*65}")
    print(f"Benchmark: {name} ({cfg_overrides['description']})")
    print(f"  batch={batch_size}, seq_len={seq_len}")
    print(f"{'='*65}")

    torch.cuda.empty_cache()
    gc.collect()

    # Build args
    base = DSArgs.__dataclass_fields__
    args = DSArgs(**{k: cfg_overrides[k] for k in cfg_overrides if k in base},
                  use_checkpoint=False)
    # Fix max_batch_len for the batch size
    args.max_batch_len = max(args.max_batch_len, batch_size)

    # Ensure compress_ratios covers all layers + MTP
    needed = args.n_layer + args.n_mtp_layer
    if len(args.compress_ratios) < needed:
        pad = tuple([0] * (needed - len(args.compress_ratios)))
        args.compress_ratios = args.compress_ratios + pad

    # Build model
    try:
        model = DeepSeekV4(args)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM during model build — skipping {name}")
        return None
    except RuntimeError as e:
        print(f"  Error building model: {e}")
        return None

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Params: {format_params(total_params)} total, "
        f"{format_params(trainable)} trainable"
    )

    # Build optimizers
    muon_p, adamw_p = group_params(model)
    idx_p = get_indexer_params(model)
    muon_opt = Muon(muon_p, lr=1e-3, momentum=0.95, weight_decay=0.1)
    adamw_opt = AdamW(adamw_p, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    idx_opt = AdamW(idx_p, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPU memory after build: {mem_mb:.0f} MB")

    # Warmup
    print(f"  Warming up ({warmup_steps} steps)...")
    for step in range(warmup_steps):
        ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        ntp, mtp_list, idx_data = model(ids)
        ntp_loss = cross_entropy(ids[:, 1:], ntp[:, :-1])
        mtp_loss = sum(cross_entropy(ids[:, 1:], m[:, :-1]) for m in mtp_list)
        kl_loss = sum(
            indexer_kl_loss(iscore, idx, wc) for (iscore, wc, idx) in idx_data
        )
        lm_loss = ntp_loss + 0.3 * mtp_loss
        (0.5 * kl_loss).backward(retain_graph=True)
        lm_loss.backward()
        grad_clip(model.parameters(), max_norm=1.0)
        muon_opt.step()
        adamw_opt.step()
        idx_opt.step()
        model.zero_grad()

    torch.cuda.synchronize()

    # Profile
    print(f"  Profiling ({profile_steps} steps)...")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for step in range(profile_steps):
        ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        ntp, mtp_list, idx_data = model(ids)
        ntp_loss = cross_entropy(ids[:, 1:], ntp[:, :-1])
        mtp_loss = sum(cross_entropy(ids[:, 1:], m[:, :-1]) for m in mtp_list)
        kl_loss = sum(
            indexer_kl_loss(iscore, idx, wc) for (iscore, wc, idx) in idx_data
        )
        lm_loss = ntp_loss + 0.3 * mtp_loss
        (0.5 * kl_loss).backward(retain_graph=True)
        lm_loss.backward()
        grad_clip(model.parameters(), max_norm=1.0)
        muon_opt.step()
        adamw_opt.step()
        idx_opt.step()
        model.zero_grad()
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_ms = start_evt.elapsed_time(end_evt)
    total_tokens = profile_steps * batch_size * seq_len
    tok_per_sec = total_tokens / (elapsed_ms / 1000)
    step_time_ms = elapsed_ms / profile_steps

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Results:")
    print(f"    Tokens/sec:    {tok_per_sec:,.0f}")
    print(f"    Step time:     {step_time_ms:.0f} ms")
    print(f"    Peak GPU mem:  {peak_mem:.0f} MB")

    # Cleanup
    del model, muon_opt, adamw_opt, idx_opt
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "params": total_params,
        "trainable": trainable,
        "tok_per_sec": tok_per_sec,
        "step_time_ms": step_time_ms,
        "peak_mem_mb": peak_mem,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_tokens": total_tokens,
    }


def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(42)

    print("DGX Spark Training Throughput Benchmark")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA:   {torch.version.cuda}")
    print(f"  PyTorch:{torch.__version__}")

    results = []

    # Phase 1: each config at its native max_seq_len
    print("\n\n--- Phase 1: native seq_len, batch_size=4 ---")
    for name in ["tiny", "small", "prefer", "medium"]:
        sl = CONFIGS[name].get("max_seq_len", 128)
        r = benchmark_config(name, CONFIGS[name], batch_size=4, seq_len=sl)
        if r:
            results.append(r)

    # Phase 2: test larger configs with smaller batch if needed
    print("\n\n--- Phase 2: default config, sweep batch sizes ---")
    for bs in [4, 2, 1]:
        r = benchmark_config(
            "default",
            CONFIGS["default"],
            batch_size=bs,
            seq_len=128,
            warmup_steps=3,
            profile_steps=10,
        )
        if r:
            results.append(r)
            break  # Take the first one that fits

    # Summary
    print("\n\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(
        f"{'Config':<12s} {'Params':>10s} {'Batch':>6s} {'Seq':>5s} "
        f"{'tok/s':>10s} {'Step(ms)':>9s} {'PeakMem':>8s}"
    )
    print("-" * 65)
    for r in results:
        print(
            f"{r['name']:<12s} {format_params(r['params']):>10s} "
            f"{r['batch_size']:>6d} {r['seq_len']:>5d} "
            f"{r['tok_per_sec']:>10,.0f} {r['step_time_ms']:>9.0f} "
            f"{r['peak_mem_mb']:>7.0f}MB"
        )

    # Recommend
    print("\n--- Recommendations ---")
    for r in results:
        if r["tok_per_sec"] > 1000:
            steps_per_hour = 3600 / (r["step_time_ms"] / 1000)
            print(
                f"  {r['name']}: ~{steps_per_hour:,.0f} steps/hour "
                f"@ batch={r['batch_size']}, seq={r['seq_len']}"
            )


if __name__ == "__main__":
    main()
