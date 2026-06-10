def approx_flops(seq_len, d_model, num_heads, d_ff, num_layers, vocab_size):
    """
    Calculate approximate FLOPs for each step of a GPT-2 XL-shaped model
    forward pass with batch_size=1.

    Each matmul (M,N) @ (N,K) costs 2*M*N*K FLOPs.
    """
    d_k = d_model // num_heads

    # --- Per Transformer Block ---

    # Q, K, V, O projections: 4 matmuls of shape (seq_len, d_model) @ (d_model, d_model)
    qkvo_flops = 4 * 2 * seq_len * d_model * d_model

    # Q @ K^T: (num_heads, seq_len, d_k) @ (num_heads, d_k, seq_len)
    qk_matmul_flops = num_heads * 2 * seq_len * seq_len * d_k

    # softmax(QK^T) @ V: (num_heads, seq_len, seq_len) @ (num_heads, seq_len, d_k)
    attn_v_flops = num_heads * 2 * seq_len * seq_len * d_k

    # SwiGLU FFN: linear_1, linear_2, linear_3 — 3 matmuls
    # linear_1, linear_3: (seq_len, d_model) @ (d_model, d_ff)
    # linear_2: (seq_len, d_ff) @ (d_ff, d_model)
    swiglu_flops = 3 * 2 * seq_len * d_model * d_ff

    per_block_flops = qkvo_flops + qk_matmul_flops + attn_v_flops + swiglu_flops
    all_blocks_flops = per_block_flops * num_layers

    # --- Output ---
    # Output linear: (seq_len, d_model) @ (d_model, vocab_size)
    output_flops = 2 * seq_len * d_model * vocab_size

    # --- Total ---
    total_flops = all_blocks_flops + output_flops

    steps = {
        "Q, K, V, O projections (per block)": qkvo_flops,
        "Q @ K^T (per block)": qk_matmul_flops,
        "Attention @ V (per block)": attn_v_flops,
        "SwiGLU FFN (per block)": swiglu_flops,
        "Per Transformer Block Total": per_block_flops,
        f"All {num_layers} Transformer Blocks": all_blocks_flops,
        "Output Linear": output_flops,
        "Total": total_flops,
    }

    # The single most expensive matmul/operation (exclude aggregate steps)
    component_flops = {
        "Q, K, V, O projections": qkvo_flops,
        "Q @ K^T": qk_matmul_flops,
        "Attention @ V": attn_v_flops,
        "SwiGLU FFN": swiglu_flops,
        "Output Linear": output_flops,
    }
    max_step = max(component_flops, key=component_flops.get)

    return steps, max_step


if __name__ == "__main__":
    # GPT-2 XL parameters
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

    for arch, spec in zip(
        ["GPT-2-XL", "GPT-2-Small", "GPT-2-Median", "GPT-2-Large", "GPT-2-XL-Long"],
        [gpt2_xl, gpt2_small, gpt2_median, gpt2_large, gpt2_xl_long],
    ):
        print(f"======== {arch} ========")

        steps, max_step = approx_flops(
            seq_len=spec["context_length"],
            d_model=spec["d_model"],
            num_heads=spec["num_heads"],
            d_ff=spec["d_ff"],
            num_layers=spec["num_layers"],
            vocab_size=spec["vocab_size"],
        )

        for name, flops in steps.items():
            print(f"{name}: {flops} FLOPs ({flops / 1e12:.4f} TFLOPs)")

        print(f"\nMost FLOPs-consuming step: {max_step}")
