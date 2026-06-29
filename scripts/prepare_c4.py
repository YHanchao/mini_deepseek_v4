"""
从 C4 数据集下载文本，存为与 owt_train.txt 相同格式的纯文本文件。

文档间用 <|endoftext|> 分隔，可直接喂给 pre_tokenization.py。

用法:
    python scripts/prepare_c4.py --output data/c4_train.txt --target-gb 12
"""

import argparse
import json
import os
import time
import zlib

import requests


def _h(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def main() -> None:
    parser = argparse.ArgumentParser(description="C4 文本下载")
    parser.add_argument("--output", required=True, help="输出 .txt 文件路径")
    parser.add_argument("--target-gb", type=float, default=12.0, help="目标文件大小 (GB)")
    args = parser.parse_args()

    MIRROR = "https://hf-mirror.com/datasets/allenai/c4/resolve/main/en"
    TOTAL_SHARDS = 1024
    TARGET_BYTES = int(args.target_gb * 1024**3)

    print(f"[prepare_c4] target: {args.target_gb} GB text → {args.output}")
    print(f"[prepare_c4] source: hf-mirror.com, up to {TOTAL_SHARDS} shards")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t_start = time.time()
    total_bytes = 0
    total_docs = 0

    with open(args.output, "wb") as f_out:
        for shard_idx in range(TOTAL_SHARDS):
            if total_bytes >= TARGET_BYTES:
                break

            url = f"{MIRROR}/c4-train.{shard_idx:05d}-of-01024.json.gz"
            print(
                f"[prepare_c4] shard {shard_idx + 1}/{TOTAL_SHARDS} "
                f"({total_bytes / 1024**3:.1f} GB so far)..."
            )

            # 流式下载 + gzip 解压
            try:
                resp = requests.get(url, stream=True, timeout=120)
                resp.raise_for_status()
            except Exception as e:
                print(f"[prepare_c4] download failed: {e}, stopping")
                break

            decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
            buf = b""
            shard_docs = 0

            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                buf += decompressor.decompress(chunk)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    if not text or not text.strip():
                        continue
                    # 文档间 <|endoftext|> 分隔（第一个文档前不加）
                    if shard_docs > 0:
                        f_out.write(b"<|endoftext|>")
                    f_out.write(text.encode("utf-8"))
                    total_bytes = f_out.tell()
                    shard_docs += 1

            total_docs += shard_docs
            elapsed = time.time() - t_start
            print(
                f"[prepare_c4]   {shard_docs} docs, "
                f"total {total_bytes / 1024**3:.2f} GB, "
                f"{total_bytes / 1024**2 / elapsed:.1f} MB/s"
            )

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"[prepare_c4] 完成!")
    print(f"  output:  {total_bytes / 1024**3:.2f} GB")
    print(f"  docs:    {_h(total_docs)}")
    print(f"  time:    {elapsed / 60:.1f} min")
    print(f"  speed:   {total_bytes / 1024**2 / elapsed:.1f} MB/s")
    print()
    print(f"  Tokenize 命令:")
    print(f"    python scripts/pre_tokenization.py \\")
    print(f"        --input {args.output} \\")
    print(f"        --output data/c4_train.bin")
    print("=" * 60)


if __name__ == "__main__":
    main()
