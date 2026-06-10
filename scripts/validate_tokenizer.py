"""加载训练好的 tokenizer，在验证集上做统计。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import BPETokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="验证 BPE tokenizer")
    parser.add_argument("--vocab", required=True, help="vocab JSON 文件路径")
    parser.add_argument("--merges", required=True, help="merges TXT 文件路径")
    parser.add_argument("--input", required=True, help="验证集文本文件")
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"],
                        help="训练时用的 special tokens")
    args = parser.parse_args()

    print(f"[validate] 加载 tokenizer: {args.vocab}")
    tokenizer = BPETokenizer.from_file(args.vocab, args.merges, special_tokens=args.special_tokens)
    print(f"[validate] vocab: {len(tokenizer.vocabs)} tokens, merges: {len(tokenizer.merges)} 条")

    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    original_bytes = len(text.encode("utf-8"))
    all_ids = tokenizer.encode(text)
    token_count = len(all_ids)

    print(f"[validate] 验证集统计")
    print(f"  原始 UTF-8 字节:  {original_bytes:>12,}")
    print(f"  token 数量:       {token_count:>12,}")
    print(f"  每 token 字节数:  {original_bytes / token_count:.2f}")
    print(f"  压缩率:           {original_bytes / token_count / tokenizer.vocab_size:.4%}")


if __name__ == "__main__":
    main()
