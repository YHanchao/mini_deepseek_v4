#!/usr/bin/env python3
"""验证 tokenized .bin 文件与 tokenizer 的一致性。

用法:
    python scripts/verify_tokenization.py \
        --bin data/train.bin \
        --vocab checkpoints/tokenizer_vocab.json \
        --merges checkpoints/tokenizer_merges.txt
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402
from array import array  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="验证 tokenized .bin 文件")
    parser.add_argument("--bin", required=True, help=".bin 文件路径")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--samples", type=int, default=5, help="采样点数，默认 5")
    parser.add_argument("--seq-len", type=int, default=500, help="每段 token 数，默认 500")
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_file(
        args.vocab, args.merges, special_tokens=config.SPECIAL_TOKENS
    )

    file_size = os.path.getsize(args.bin)
    total_tokens = file_size // 2
    print(f"文件: {args.bin}")
    print(f"大小: {file_size / 1024**2:.1f} MB, tokens: {total_tokens:,}")
    print(f"采样 {args.samples} 段, 每段 {args.seq_len} tokens\n")

    arr = array("H")
    with open(args.bin, "rb") as f:
        arr.fromfile(f, total_tokens)

    all_pass = True
    for i in range(args.samples):
        start = random.randint(0, total_tokens - args.seq_len - 1)
        ids = arr[start : start + args.seq_len]

        # decode → encode round-trip
        decoded = tokenizer.decode(ids.tolist())
        re_encoded = tokenizer.encode(decoded)

        if ids.tolist() == re_encoded:
            print(f"  [{i+1}] pos={start:>12,}  ✓  round-trip 一致")
        else:
            all_pass = False
            # 找第一个不一致的位置
            mismatch = None
            for j, (a, b) in enumerate(zip(ids, re_encoded)):
                if a != b:
                    mismatch = j
                    break
            if mismatch is None:
                mismatch = min(len(ids), len(re_encoded))
            print(
                f"  [{i+1}] pos={start:>12,}  ✗  位置 {mismatch} 不一致: "
                f"bin={ids[mismatch]}, re-encoded={re_encoded[mismatch] if mismatch < len(re_encoded) else 'EOF'}"
            )

    if all_pass:
        print(f"\n全部 {args.samples} 段 round-trip 验证通过 ✓")
    else:
        print(f"\n存在不一致 ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
