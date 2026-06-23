"""Verify training loop with random data -- TODO-2.

Uses tiny config (d_model=768, n_layer=4, n_experts=4) to validate
forward → loss → backward → optimizer.step() for 150 steps.

Training objective:
  LM loss = NTP + 0.3 * MTP  (main model)
  KL loss = Σ KL(p_attn || softmax(I))  (indexer, Eq. 3-4)
"""

import torch
from src.deepseek import DeepSeekV4, DSArgs
from src.loss import cross_entropy, indexer_kl_loss
from src.optimizer import Muon, AdamW, group_params, get_indexer_params, grad_clip


def format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def print_model_info(model, args: DSArgs, batch_size, seq_len):
    """Print per-component parameter counts and memory estimates."""
    total = 0
    lines = []
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        total += n
        lines.append((name, n))

    print("=" * 65)
    print(
        f"Model config: d_model={args.d_model}, n_layer={args.n_layer}, "
        f"n_experts={args.n_experts}, hc_num={args.expansion_rate}"
    )
    print(f"compress_ratios={args.compress_ratios[:args.n_layer]}")
    print("-" * 65)
    for name, n in lines:
        pct = n / total * 100
        print(f"  {name:<20s} {format_params(n):>8s} params  ({pct:5.1f}%)")
    print("-" * 65)
    print(f"  {'total':20s} {format_params(total):>8s} params")
    print(
        f"  {'trainable':20s} {format_params(sum(p.numel() for p in model.parameters() if p.requires_grad)):>8s} params"
    )

    # Memory estimates
    bf16_bytes = total * 2
    fp32_bytes = total * 4
    print(f"\n  Param memory (bf16):  {bf16_bytes / 1e6:.1f} MB")
    print(f"  Param memory (fp32):  {fp32_bytes / 1e6:.1f} MB")

    muon_p, adamw_p = group_params(model)
    muon_params = sum(p.numel() for p in muon_p)
    adamw_params = sum(p.numel() for p in adamw_p)
    opt_mem = muon_params * 4 + adamw_params * 12
    print(f"  Optimizer state (est): {opt_mem / 1e6:.1f} MB")

    # Activation memory estimate (per microbatch, rough)
    # Hidden states: (batch, seq, hc_num, d_model)
    hidden = batch_size * seq_len * args.expansion_rate * args.d_model * 2  # bf16
    # MoE intermediate: roughly (batch*seq, d_moe_ff) per expert
    moe = batch_size * seq_len * args.d_moe_ff * 2  # bf16
    # Logits: (batch, seq, vocab_size)
    logits = batch_size * seq_len * args.vocab_size * 2  # bf16
    act_mem = hidden + moe + logits
    print(
        f"  Activation (est):     {act_mem / 1e6:.1f} MB "
        f"(batch={batch_size}, seq={seq_len})"
    )

    # Total
    total_mem = fp32_bytes + opt_mem + act_mem
    print(f"  Total (est):          {total_mem / 1e6:.1f} MB")
    print("=" * 65)

    print(f"\nOptimizer param groups: Muon={len(muon_p)}, AdamW={len(adamw_p)}")
    print(f"  Muon params:  {format_params(muon_params)}")
    print(f"  AdamW params: {format_params(adamw_params)}")
    print()


def verify_training():
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(42)

    args = DSArgs(
        d_model=768,
        d_ff=768,
        n_layer=4,
        n_mtp_layer=1,
        n_experts=4,
        n_shared_experts=1,
        d_moe_ff=768,
        n_heads=12,
        head_dim=128,
        attn_rank=256,
        output_lora=256,
        max_batch_len=8,
        max_seq_len=512,
        vocab_size=32000,
        # Layer 0: no compression, Layer 1: overlap(4)+indexer, Layer 2: non-overlap(128), Layer 3: no compression, MTP: no compression
        compress_ratios=(0, 4, 128, 0, 0),
    )

    model = DeepSeekV4(args)
    model.train()

    batch_size = 4
    seq_len = 128
    print_model_info(model, args, batch_size, seq_len)

    muon_p, adamw_p = group_params(model)  # main model params (excludes indexer)
    idx_p = get_indexer_params(model)       # indexer params (trained via KL loss)

    muon_opt = Muon(muon_p, lr=1e-3, momentum=0.95, weight_decay=0.1)
    adamw_opt = AdamW(adamw_p, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    idx_opt = AdamW(idx_p, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    print(f"Indexer params: {len(idx_p)}, total: {sum(p.numel() for p in idx_p):,}")
    print()

    total_steps = 150
    base_mem = torch.cuda.memory_allocated() / 1024**2
    for step in range(total_steps):
        input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))

        ntp_logits, mtp_list, indexer_data_list = model(input_ids)

        ntp_loss = cross_entropy(input_ids[:, 1:], ntp_logits[:, :-1])
        mtp_loss = sum(
            cross_entropy(input_ids[:, i + 2 :], mtp[:, : -(i + 2)])
            for i, mtp in enumerate(mtp_list)
        )

        # Indexer KL loss (Eq. 3-4): separate optimization
        kl_loss = sum(
            indexer_kl_loss(index_score, compress_topk_idxs, weight_compress)
            for (index_score, weight_compress, compress_topk_idxs) in indexer_data_list
        )

        total_loss = ntp_loss + 0.3 * mtp_loss + 0.5 * kl_loss

        model.zero_grad()
        total_loss.backward()
        grad_clip(model.parameters(), max_norm=1.0)
        muon_opt.step()
        adamw_opt.step()
        idx_opt.step()

        cur_mem = torch.cuda.memory_allocated() / 1024**2

        if step % 10 == 0 or step == total_steps - 1:
            grad_norm = torch.sqrt(
                sum(
                    torch.norm(p.grad.data) ** 2
                    for p in model.parameters()
                    if p.grad is not None
                )
            ).item()
            print(
                f"step {step:4d} | loss: {total_loss.item():.4f} | "
                f"ntp: {ntp_loss.item():.4f} | mtp: {mtp_loss.item():.4f} | "
                f"kl: {kl_loss.item():.4f} | "
                f"grad_norm: {grad_norm:.4f} | "
                f"mem: {cur_mem:.0f}MB"
            )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"FAILED at step {step}: NaN or Inf detected!")
            return

    final_loss = total_loss.item()
    final_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"\nFinal loss: {final_loss:.4f}  |  mem: {base_mem:.0f} → {final_mem:.0f}MB  (Δ+{final_mem - base_mem:.0f}MB)")
    if final_loss < 8.0:
        print("PASSED: loss decreased below 8.0")
    else:
        print(f"NOTE: loss stays near random baseline (~{args.vocab_size} vocab → ln(V) ≈ {torch.tensor(args.vocab_size).float().log().item():.2f})")

    print("150 steps completed without errors.")


if __name__ == "__main__":
    verify_training()
