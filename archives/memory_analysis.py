def approx_memory(batch_size, seq_len, d_model, num_heads, d_ff, num_layers, vocab_size, bytes_per_elem=4):
    """
    Calculate approximate peak memory usage (in bytes) for training with AdamW.

    Decomposes memory into: parameters, gradients, optimizer state (AdamW), and activations.

    bytes_per_elem: 4 for fp32, 2 for bf16.
      - Parameters, gradients, and optimizer states are always stored in fp32 (4 bytes).
      - Activations use the specified bytes_per_elem.
    """

    # --- Parameters (element count) ---
    # Per transformer block:
    #   RMSNorms: 2 * d_model
    #   Q, K, V, O projections: 4 * d_model^2
    #   SwiGLU FFN (W1, W2, W3): 3 * d_model * d_ff
    params_per_block = 2 * d_model + 4 * d_model**2 + 3 * d_model * d_ff
    params_all_blocks = params_per_block * num_layers

    # Final RMSNorm + output embedding
    params_final_norm = d_model
    params_output = d_model * vocab_size

    total_params_elements = params_all_blocks + params_final_norm + params_output

    # Memory (fp32): 4 bytes per param
    param_mem = 4 * total_params_elements

    # Gradients: same as parameters
    grad_mem = param_mem

    # AdamW optimizer state: momentum + variance = 2 * parameters (fp32)
    optim_mem = 2 * param_mem

    # --- Activations (element count) ---
    # Per transformer block:
    #   Input: bs * seq_len * d_model
    #   QKV: 3 * bs * seq_len * d_model
    #   softmax(QK^T): bs * num_heads * seq_len^2
    #   Attention output: bs * seq_len * d_model
    #   RMSNorms (2 per block): 2 * bs * seq_len * d_model
    #   SiLU gate input (saved X): bs * seq_len * d_model
    #   FFN intermediates (W1X, W3X, W2 output): 3 * bs * seq_len * d_ff
    act_per_block = (
        8 * batch_size * seq_len * d_model
        + batch_size * num_heads * seq_len**2
        + 3 * batch_size * seq_len * d_ff
    )
    act_all_blocks = act_per_block * num_layers

    # Final RMSNorm: bs * seq_len * d_model
    act_final_norm = batch_size * seq_len * d_model

    # Output embedding + cross-entropy logits: 2 * bs * seq_len * vocab_size
    act_output = 2 * batch_size * seq_len * vocab_size

    total_act_elements = act_all_blocks + act_final_norm + act_output
    act_mem = bytes_per_elem * total_act_elements

    # --- Peak Memory ---
    peak_mem = param_mem + grad_mem + optim_mem + act_mem

    steps = {
        "Parameters": param_mem,
        "Gradients": grad_mem,
        "AdamW Optimizer State (m + v)": optim_mem,
        "Activations": act_mem,
        "--- Peak Total ---": peak_mem,
    }

    # Per-component activation breakdown (single block)
    act_breakdown = {
        "  Input": bytes_per_elem * batch_size * seq_len * d_model,
        "  QKV projections (3 tensors)": bytes_per_elem * 3 * batch_size * seq_len * d_model,
        "  softmax(QK^T)": bytes_per_elem * batch_size * num_heads * seq_len**2,
        "  Attention output": bytes_per_elem * batch_size * seq_len * d_model,
        "  RMSNorms (2 per block)": bytes_per_elem * 2 * batch_size * seq_len * d_model,
        "  SiLU gate input": bytes_per_elem * batch_size * seq_len * d_model,
        "  FFN intermediates (W1X, W3X, W2)": bytes_per_elem * 3 * batch_size * seq_len * d_ff,
    }

    # Find largest activation component
    max_act = max(act_breakdown, key=act_breakdown.get)

    return steps, max_act, total_params_elements, total_act_elements


def fmt_bytes(b):
    """Format bytes to human-readable string."""
    if b >= 1e12:
        return f"{b / 1e12:.2f} TB"
    elif b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    else:
        return f"{b:.0f} B"


if __name__ == "__main__":
    import sys

    # Parse batch_size from command line, default to 1
    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])
    else:
        batch_size = 1

    gpt2_xl = dict(
        context_length=1024,
        d_model=1600,
        num_heads=25,
        d_ff=4288,
        num_layers=48,
        vocab_size=50257,
    )

    gpt2_xl_long = dict(
        context_length=16384,
        d_model=1600,
        num_heads=25,
        d_ff=4288,
        num_layers=48,
        vocab_size=50257,
    )

    gpt2_small = dict(
        context_length=1024,
        d_model=768,
        num_heads=12,
        d_ff=2048,
        num_layers=12,
        vocab_size=50257,
    )

    gpt2_median = dict(
        context_length=1024,
        d_model=1024,
        num_heads=16,
        d_ff=2752,
        num_layers=24,
        vocab_size=50257,
    )

    gpt2_large = dict(
        context_length=1024,
        d_model=1280,
        num_heads=20,
        d_ff=3392,
        num_layers=36,
        vocab_size=50257,
    )

    print(f"Memory Analysis for AdamW Training (batch_size={batch_size}, activations=fp32)\n")

    for arch, spec in zip(
        ["GPT-2-XL", "GPT-2-Small", "GPT-2-Median", "GPT-2-Large", "GPT-2-XL-Long"],
        [gpt2_xl, gpt2_small, gpt2_median, gpt2_large, gpt2_xl_long],
    ):
        print(f"{'='*60}")
        print(f"  {arch}  |  d={spec['d_model']}, heads={spec['num_heads']}, "
              f"layers={spec['num_layers']}, ctx={spec['context_length']}")
        print(f"{'='*60}")

        steps, max_act, total_params, total_act = approx_memory(
            batch_size=batch_size,
            seq_len=spec["context_length"],
            d_model=spec["d_model"],
            num_heads=spec["num_heads"],
            d_ff=spec["d_ff"],
            num_layers=spec["num_layers"],
            vocab_size=spec["vocab_size"],
            bytes_per_elem=4,
        )

        for name, mem_bytes in steps.items():
            print(f"  {name:<38} {mem_bytes:>15,} B  ({fmt_bytes(mem_bytes)})")

        print(f"\n  Model parameters: {total_params:,.0f} elements  "
              f"({total_params * 4:,.0f} B = {fmt_bytes(total_params * 4)})")
        print(f"  Activation elements (peak): {total_act:,.0f}  "
              f"({total_act * 4:,.0f} B = {fmt_bytes(total_act * 4)})")
        print(f"  Largest per-block activation component: {max_act.strip()}")
        print()
