SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|beginoftext|>" "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
]

MODEL_CONFIGS = {
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
}

CP_VOCABS = "checkpoints/gpt2_vocab.json"
CP_MERGES = "checkpoints/gpt2_merges.txt"
