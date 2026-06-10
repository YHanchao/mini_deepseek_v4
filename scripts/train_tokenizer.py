"""命令行训练 BPE tokenizer，并将结果保存到文件。"""

import argparse
import os
import sys

# 确保项目根目录在 sys.path 中，方便从任意位置运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import BPETokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="训练 BPE tokenizer")
    parser.add_argument("--input", required=True, help="训练语料文件路径")
    parser.add_argument("--vocab-size", type=int, required=True, help="目标词表大小（含 special tokens）")
    parser.add_argument("--output-dir", required=True, help="保存 tokenizer 的目录")
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"],
                        help="Special token 列表，空格分隔")
    parser.add_argument("--max-chunk-size", type=int, default=64 * 1024 * 1024,
                        help="分块大小（字节），默认 64MB")
    parser.add_argument("--num-processes", type=int, default=1,
                        help="并行进程数，默认 1")
    parser.add_argument("--name", default="tokenizer",
                        help="输出文件名前缀，默认 tokenizer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BPETokenizer(vocab={}, merges=[], special_tokens=args.special_tokens)
    vocab, merges = tokenizer.train(
        args.input,
        vocab_size=args.vocab_size,
        max_chunk_size=args.max_chunk_size,
        num_processes=args.num_processes,
    )

    vocab_path = os.path.join(args.output_dir, f"{args.name}_vocab.json")
    merges_path = os.path.join(args.output_dir, f"{args.name}_merges.txt")
    tokenizer.to_file(vocab_path, merges_path)

    print(f"训练完成，vocab: {len(vocab)} tokens，merges: {len(merges)} 条")
    print(f"  vocab  → {vocab_path}")
    print(f"  merges → {merges_path}")


if __name__ == "__main__":
    main()
