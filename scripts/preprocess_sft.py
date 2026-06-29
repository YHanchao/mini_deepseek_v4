"""
SFT 数据预处理：加载 Alpaca-GPT4 / Dolly 15k / OASST，转换为 chat template 格式并 tokenize。

用法:
    python scripts/preprocess_sft.py

输出:
    data/llm/sft/train.pt  — {"input_ids": tensor(N, 1024), "assistant_mask": tensor(N, 1024)}
    data/llm/sft/valid.pt  — 同上
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_SEQ_LEN = 1024
EOS_ID = 256
PAD_ID = 256  # <|endoftext|> 同时用作 padding token
TRAIN_RATIO = 0.9
RANDOM_SEED = 42
MAX_OASST_DEPTH = 6  # 最多 6 条消息 = 3 轮对话


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_alpaca(path: str) -> list[list[tuple[str, str]]]:
    """加载 Alpaca-GPT4，每条返回 [(prompter, query), (assistant, response)]"""
    df = pd.read_parquet(path)
    paths: list[list[tuple[str, str]]] = []
    for _, row in df.iterrows():
        instruction = str(row["instruction"])
        inp = row.get("input")
        if pd.isna(inp) or str(inp).strip() == "":
            user_msg = instruction
        else:
            user_msg = f"{instruction}\n\n{str(inp).strip()}"
        paths.append([("prompter", user_msg), ("assistant", str(row["output"]))])
    return paths


def load_dolly(path: str) -> list[list[tuple[str, str]]]:
    """加载 Dolly 15k，每条返回 [(prompter, query), (assistant, response)]"""
    df = pd.read_json(path, lines=True)
    paths: list[list[tuple[str, str]]] = []
    for _, row in df.iterrows():
        instruction = str(row["instruction"])
        context = row.get("context")
        if pd.isna(context) or str(context).strip() == "":
            user_msg = instruction
        else:
            user_msg = f"{instruction}\n\nContext: {str(context).strip()}"
        paths.append([("prompter", user_msg), ("assistant", str(row["response"]))])
    return paths


def extract_oasst_paths(df: pd.DataFrame) -> list[list[tuple[str, str]]]:
    """从 OASST 对话树中提取最优路径。

    每个 message_tree_id 构成一棵树。从根 prompter 开始贪心选择：
    - 在 prompter 节点：选 rank 最低（最好）的 assistant 子节点
    - 在 assistant 节点：选第一个 prompter 子节点继续
    - 跳过 rank 为 NaN 的 assistant（未被评价）
    - 限制深度 ≤ MAX_OASST_DEPTH（6 条消息 = 3 轮）
    """
    trees: dict[str, dict] = {}
    for _, row in df.iterrows():
        tree_id = row["message_tree_id"]
        if tree_id not in trees:
            trees[tree_id] = {}
        trees[tree_id][row["message_id"]] = {
            "parent_id": row["parent_id"],
            "role": row["role"],
            "text": str(row["text"]),
            "rank": row["rank"],
        }

    paths: list[list[tuple[str, str]]] = []
    for tree_id, nodes in trees.items():
        # 构建邻接表：parent_id → 子节点 msg_id 列表
        children: dict[str, list[str]] = {}
        for msg_id, node in nodes.items():
            pid = node["parent_id"]
            if pd.notna(pid):
                children.setdefault(pid, []).append(msg_id)

        # 找根节点（无 parent_id 的 prompter）
        roots = [
            mid
            for mid, n in nodes.items()
            if pd.isna(n["parent_id"]) and n["role"] == "prompter"
        ]
        if not roots:
            continue

        current_id = roots[0]
        path = [(nodes[current_id]["role"], nodes[current_id]["text"])]

        while len(path) < MAX_OASST_DEPTH:
            child_ids = children.get(current_id, [])
            if not child_ids:
                break

            if nodes[current_id]["role"] == "prompter":
                # 选 rank 最低的 assistant（0 最好）
                best_id = None
                best_rank = float("inf")
                for cid in child_ids:
                    cnode = nodes[cid]
                    if cnode["role"] != "assistant":
                        continue
                    rank = cnode["rank"]
                    if pd.isna(rank):
                        continue
                    if rank < best_rank:
                        best_rank = rank
                        best_id = cid
                if best_id is None:
                    break
                current_id = best_id
            else:
                # assistant → 选第一个 prompter 子节点
                next_ids = [
                    cid for cid in child_ids if nodes[cid]["role"] == "prompter"
                ]
                if not next_ids:
                    break
                current_id = next_ids[0]

            path.append((nodes[current_id]["role"], nodes[current_id]["text"]))

        # 至少包含一轮完整对话（prompter + assistant）
        if len(path) >= 2:
            paths.append(path)

    return paths


# ── Chat Template ────────────────────────────────────────────────────────────


def build_segments(
    path: list[tuple[str, str]], system_prompt: str = SYSTEM_PROMPT
) -> list[tuple[str, bool]]:
    """将对话路径转换为 (文本段, 是否assistant) 的列表。

    System 段只出现一次在开头。每条 assistant 回复后跟一个 <|endoftext|> 段。
    """
    segments: list[tuple[str, bool]] = []
    segments.append((f"<|system|>\n{system_prompt}\n", False))

    for role, text in path:
        if role == "prompter":
            segments.append((f"<|user|>\n{text}\n", False))
        else:
            segments.append((f"<|assistant|>\n{text}", True))
            segments.append(("<|endoftext|>\n", False))

    return segments


# ── Tokenization ─────────────────────────────────────────────────────────────


def tokenize_segments(
    segments: list[tuple[str, bool]],
    tokenizer: BPETokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐段 tokenize，构建 input_ids 和 assistant_mask。

    返回形状为 (max_seq_len,) 的两个 tensor。
    超长从左侧截断（保留最后的 assistant 回复），不足右侧 pad <|endoftext|>。
    """
    all_ids: list[int] = []
    all_mask: list[bool] = []

    for text, is_assistant in segments:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        all_mask.extend([is_assistant] * len(ids))

    # 左侧截断
    if len(all_ids) > max_seq_len:
        all_ids = all_ids[-max_seq_len:]
        all_mask = all_mask[-max_seq_len:]

    # 右侧 padding（用 <|endoftext|> id=256, mask=0）
    if len(all_ids) < max_seq_len:
        pad_len = max_seq_len - len(all_ids)
        all_ids.extend([PAD_ID] * pad_len)
        all_mask.extend([False] * pad_len)

    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(all_mask, dtype=torch.bool),
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 数据预处理")
    parser.add_argument(
        "--data-dir", default="data/llm/sft", help="SFT 原始数据目录"
    )
    parser.add_argument("--output-dir", default="data/llm/sft", help="输出目录")
    parser.add_argument(
        "--vocab", default="checkpoints/tokenizer_vocab.json", help="vocab JSON 路径"
    )
    parser.add_argument(
        "--merges",
        default="checkpoints/tokenizer_merges.txt",
        help="merges TXT 路径",
    )
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--no-verify", action="store_true", help="跳过末尾的抽样验证"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. 加载 tokenizer ──
    print("[preprocess_sft] Loading tokenizer...")
    tokenizer = BPETokenizer.from_file(
        args.vocab, args.merges, special_tokens=config.SPECIAL_TOKENS
    )
    print(
        f"[preprocess_sft] vocab: {tokenizer.vocab_size} tokens, "
        f"merges: {len(tokenizer.merges)} 条"
    )

    # ── 2. 加载原始数据集 ──
    print("[preprocess_sft] Loading Alpaca-GPT4...")
    alpaca_paths = load_alpaca(os.path.join(args.data_dir, "alpaca-gpt4.parquet"))
    print(f"  → {len(alpaca_paths)} samples")

    print("[preprocess_sft] Loading Dolly 15k...")
    dolly_paths = load_dolly(os.path.join(args.data_dir, "databricks-dolly-15k.jsonl"))
    print(f"  → {len(dolly_paths)} samples")

    print("[preprocess_sft] Loading OASST train...")
    oasst_train_raw = pd.read_parquet(
        os.path.join(args.data_dir, "oasst_train.parquet")
    )
    oasst_train_df = oasst_train_raw[
        (oasst_train_raw["lang"] == "en")
        & (oasst_train_raw["review_result"] == True)  # noqa: E712
        & (oasst_train_raw["deleted"] == False)  # noqa: E712
    ]
    print(
        f"  → {len(oasst_train_df)} messages "
        f"({oasst_train_df['message_tree_id'].nunique()} trees) after filter"
    )
    oasst_train_paths = extract_oasst_paths(oasst_train_df)
    print(f"  → {len(oasst_train_paths)} conversation paths extracted")

    print("[preprocess_sft] Loading OASST valid...")
    oasst_valid_raw = pd.read_parquet(
        os.path.join(args.data_dir, "oasst_valid.parquet")
    )
    oasst_valid_df = oasst_valid_raw[
        (oasst_valid_raw["lang"] == "en")
        & (oasst_valid_raw["review_result"] == True)  # noqa: E712
        & (oasst_valid_raw["deleted"] == False)  # noqa: E712
    ]
    print(
        f"  → {len(oasst_valid_df)} messages "
        f"({oasst_valid_df['message_tree_id'].nunique()} trees) after filter"
    )
    oasst_valid_paths = extract_oasst_paths(oasst_valid_df)
    print(f"  → {len(oasst_valid_paths)} conversation paths extracted")

    # ── 3. Alpaca + Dolly 随机 90/10 划分 ──
    single_turn = alpaca_paths + dolly_paths
    random.shuffle(single_turn)
    split = int(len(single_turn) * TRAIN_RATIO)
    single_train = single_turn[:split]
    single_val = single_turn[split:]

    # OASST 使用自带 train/valid split，直接合并
    train_paths = single_train + oasst_train_paths
    val_paths = single_val + oasst_valid_paths

    random.shuffle(train_paths)
    random.shuffle(val_paths)

    print(f"\n[preprocess_sft] Train: {len(train_paths)} samples")
    print(f"[preprocess_sft] Valid: {len(val_paths)} samples")

    # ── 4. 逐条 tokenize ──
    def process_split(paths, split_name):
        input_ids_list = []
        mask_list = []
        lengths = []
        truncated = 0

        for i, path in enumerate(paths):
            segments = build_segments(path)
            ids, mask = tokenize_segments(segments, tokenizer, args.max_seq_len)

            # 记录原始长度（不含 padding）
            actual_len = 0
            for text, _ in segments:
                actual_len += len(tokenizer.encode(text))

            lengths.append(min(actual_len, args.max_seq_len))
            if actual_len > args.max_seq_len:
                truncated += 1

            input_ids_list.append(ids)
            mask_list.append(mask)

            if (i + 1) % 10000 == 0:
                print(f"  {split_name}: {i + 1}/{len(paths)} samples processed...")

        input_ids = torch.stack(input_ids_list)
        assistant_mask = torch.stack(mask_list)
        return input_ids, assistant_mask, lengths, truncated

    print("\n[preprocess_sft] Tokenizing train split...")
    train_ids, train_mask, train_lengths, train_trunc = process_split(
        train_paths, "train"
    )

    print("\n[preprocess_sft] Tokenizing valid split...")
    val_ids, val_mask, val_lengths, val_trunc = process_split(val_paths, "valid")

    # ── 5. 统计 ──
    print(f"\n[preprocess_sft] === Statistics ===")
    print(
        f"Train: {len(train_paths)} samples, "
        f"avg_len={np.mean(train_lengths):.0f}, "
        f"truncated={train_trunc} ({100 * train_trunc / len(train_paths):.1f}%)"
    )
    print(
        f"Valid: {len(val_paths)} samples, "
        f"avg_len={np.mean(val_lengths):.0f}, "
        f"truncated={val_trunc} ({100 * val_trunc / len(val_paths):.1f}%)"
    )

    # ── 6. Shuffle 并保存 ──
    print("\n[preprocess_sft] Shuffling and saving...")

    train_idx = torch.randperm(len(train_ids))
    train_ids = train_ids[train_idx]
    train_mask = train_mask[train_idx]

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.pt")
    torch.save(
        {"input_ids": train_ids, "assistant_mask": train_mask}, train_path
    )
    print(
        f"  → {train_path} "
        f"({os.path.getsize(train_path) / 1024 ** 2:.1f} MB)"
    )

    val_path = os.path.join(args.output_dir, "valid.pt")
    torch.save(
        {"input_ids": val_ids, "assistant_mask": val_mask}, val_path
    )
    print(
        f"  → {val_path} "
        f"({os.path.getsize(val_path) / 1024 ** 2:.1f} MB)"
    )

    # ── 7. 抽样验证 ──
    if args.no_verify:
        print("\n[preprocess_sft] Done!")
        return

    print("\n[preprocess_sft] === Verification Samples ===\n")
    num_show = min(5, len(train_paths))
    for i in range(num_show):
        idx = train_idx[i].item()
        ids = train_ids[idx]
        mask = train_mask[idx]

        # 找实际内容长度：最后一个 mask=1 的位置 + 余量（末尾 <|endoftext|>\n）
        last_assistant_pos = 0
        for j in range(len(ids)):
            if mask[j]:
                last_assistant_pos = j
        # margin 包含末尾的 <|endoftext|>\n（通常 2-5 个 token）
        actual_len = min(last_assistant_pos + 10, len(ids))

        # 找到 mask 变化点
        transitions: list[str] = []
        prev = False
        for j in range(actual_len):
            cur = bool(mask[j].item())
            if cur != prev:
                label = "█ ASSISTANT" if cur else "· other"
                transitions.append(f"{label}@pos{j}")
                prev = cur

        text = tokenizer.decode(ids[:actual_len].tolist())

        print(
            f"── Sample {i + 1} "
            f"(assistant_tokens={mask.sum().item()}) ──"
        )
        print(f"Transitions: {', '.join(transitions[:12])}"
              f"{'...' if len(transitions) > 12 else ''}")
        # 打印前 700 字符
        print(text[:700] + ("..." if len(text) > 700 else ""))
        print()

    print("[preprocess_sft] Done!")


if __name__ == "__main__":
    main()
