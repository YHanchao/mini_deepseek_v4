"""
预训练数据 Tokenization —— 将原始文本语料编码为 uint16 token ID 二进制文件。

用法:
    python scripts/pre_tokenization.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output data/tokenized.bin \
        --chunk-size 1G \
        --num-workers 8
"""

import argparse
import os
import re
import sys
import time
from array import array
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.tokenizer import BPETokenizer, find_chunk_boundaries  # noqa: E402

_tokenizer: BPETokenizer | None = None


def _parse_size(s: str) -> int:
    """将 1G / 512M / 1048576 等格式解析为字节数。"""
    s = s.strip().upper()
    m = re.match(r"^([\d.]+)\s*(G|GB)?$", s)
    if m:
        return int(float(m.group(1)) * 1024**3)
    m = re.match(r"^([\d.]+)\s*(M|MB)$", s)
    if m:
        return int(float(m.group(1)) * 1024**2)
    m = re.match(r"^([\d.]+)\s*(K|KB)$", s)
    if m:
        return int(float(m.group(1)) * 1024)
    m = re.match(r"^(\d+)$", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"无法解析的大小格式: {s!r}，支持 1G / 512M / 1048576 等")


def _init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    global _tokenizer
    _tokenizer = BPETokenizer.from_file(
        vocab_path, merges_path, special_tokens=special_tokens
    )


def _encode_chunk(args: tuple[int, str, int, int]) -> tuple[int, list[int]]:
    idx, file_path, chunk_start, chunk_end = args
    with open(file_path, "rb") as f:
        f.seek(chunk_start)
        data = f.read(chunk_end - chunk_start)
    text = data.decode("utf-8", errors="ignore")
    token_ids = _tokenizer.encode(text)
    return idx, token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="预训练数据 Tokenization")
    parser.add_argument("--input", required=True, help="输入文本文件路径")
    parser.add_argument("--output", required=True, help="输出 .bin 文件路径")
    parser.add_argument(
        "--vocab",
        default="checkpoints/tokenizer_vocab.json",
        help="vocab JSON 路径",
    )
    parser.add_argument(
        "--merges",
        default="checkpoints/tokenizer_merges.txt",
        help="merges TXT 路径",
    )
    parser.add_argument(
        "--chunk-size",
        default="1G",
        help="每个 chunk 的目标大小，支持 1G / 512M / 1048576 等格式，默认 1G",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="并行进程数，默认 8",
    )
    args = parser.parse_args()

    chunk_size_bytes = _parse_size(args.chunk_size)

    t0 = time.time()

    # 1. 加载 tokenizer（主进程，用于打印信息）
    print(f"[pre-tokenize] 加载 tokenizer: vocab={args.vocab}, merges={args.merges}")
    tokenizer = BPETokenizer.from_file(
        args.vocab, args.merges, special_tokens=config.SPECIAL_TOKENS
    )
    print(
        f"[pre-tokenize] vocab: {tokenizer.vocab_size} tokens, "
        f"merges: {len(tokenizer.merges)} 条, "
        f"special_tokens: {len(tokenizer.special_tokens)}"
    )

    # 2. 切分 chunk 边界
    file_size = os.path.getsize(args.input)
    num_chunks = max(1, file_size // chunk_size_bytes)
    print(
        f"[pre-tokenize] 输入文件: {args.input} "
        f"({file_size / 1024 ** 3:.2f} GB), "
        f"chunk_size={args.chunk_size}, "
        f"{num_chunks} chunks"
    )

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    print(f"[pre-tokenize] 实际切分: {len(chunks)} chunks")

    # 3. 多进程并行编码
    num_workers = min(args.num_workers, len(chunks))
    tasks = [(i, args.input, start, end) for i, (start, end) in enumerate(chunks)]

    print(f"[pre-tokenize] 启动 {num_workers} 个 worker 进程编码...")

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(args.vocab, args.merges, config.SPECIAL_TOKENS),
    ) as pool:
        results = pool.map(_encode_chunk, tasks)

    # 按 chunk 顺序排列
    results.sort(key=lambda x: x[0])
    all_ids: list[int] = []
    for _, token_ids in results:
        all_ids.extend(token_ids)

    encode_elapsed = time.time() - t0

    # 4. 写入 .bin 文件
    print(
        f"[pre-tokenize] 编码完成, {len(all_ids):,} tokens, "
        f"耗时 {encode_elapsed:.1f}s, "
        f"写入 → {args.output}"
    )
    arr = array("H", all_ids)
    with open(args.output, "wb") as f:
        arr.tofile(f)

    total_elapsed = time.time() - t0
    output_mb = os.path.getsize(args.output) / 1024**2

    print(
        f"[pre-tokenize] 完成! "
        f"总 tokens: {len(all_ids):,}, "
        f"输出大小: {output_mb:.1f} MB, "
        f"总耗时: {total_elapsed:.1f}s, "
        f"吞吐: {file_size / 1024 ** 2 / total_elapsed:.1f} MB/s 输入, "
        f"压缩比: {len(all_ids) / (file_size / 1024 ** 2):.0f} tokens/MB 输入"
    )


if __name__ == "__main__":
    main()
