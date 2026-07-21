"""
Preprocess roast_dataset.jsonl for Weighted SFT.

Output format — one group (4 candidates + scores) per item, same as the old
GRPO off-policy format.  Train/val split is 90/10 with seed=42 so the
validation set matches the previous ``val/grpo.pt``.

Usage:
    python scripts/preprocess_roast.py

Output:
    data/llm/roast/weighted_sft/train.pt
    data/llm/roast/weighted_sft/valid.pt
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a witty roast comedian.\n"
    "Write concise observational roasts.\n"
    "Avoid explaining the joke."
)
MAX_SEQ_LEN = 1024
EOS_ID = 256
PAD_ID = 256
RANDOM_SEED = 42

TRAIN_RATIO = 0.90
VAL_RATIO = 0.10


# ── Segment Builders ─────────────────────────────────────────────────────────


def build_full_segments(
    user_input: str, response: str
) -> list[tuple[str, bool]]:
    """Build segments for a complete sequence (prompt + response + EOS)."""
    return [
        (f"<|system|>\n{SYSTEM_PROMPT}\n", False),
        (f"<|user|>\n{user_input}\n", False),
        (f"<|assistant|>\n{response}", True),
        ("<|endoftext|>\n", False),
    ]


# ── Tokenization ─────────────────────────────────────────────────────────────


def tokenize_segments(
    segments: list[tuple[str, bool]],
    tokenizer: BPETokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize segments into (input_ids, assistant_mask) of shape (max_seq_len,).

    Left-truncates, right-pads with <|endoftext|> (id=256, mask=False).
    """
    all_ids: list[int] = []
    all_mask: list[bool] = []

    for text, is_assistant in segments:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        all_mask.extend([is_assistant] * len(ids))

    if len(all_ids) > max_seq_len:
        all_ids = all_ids[-max_seq_len:]
        all_mask = all_mask[-max_seq_len:]

    if len(all_ids) < max_seq_len:
        pad_len = max_seq_len - len(all_ids)
        all_ids.extend([PAD_ID] * pad_len)
        all_mask.extend([False] * pad_len)

    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(all_mask, dtype=torch.bool),
    )


# ── Processing ───────────────────────────────────────────────────────────────


def process_grpo_offpolicy(
    records: list[dict], tokenizer: BPETokenizer, max_seq_len: int
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]
]:
    """Process records: each record → 4 rows (one per candidate).

    Returns (input_ids, completion_mask, scores, group_ids, is_winner, lengths).
    """
    input_ids_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    scores_list: list[list[float]] = []
    group_ids_list: list[int] = []
    is_winner_list: list[bool] = []
    lengths: list[int] = []

    for group_id, rec in enumerate(records):
        winner_idx = rec["editor"]["winner_index"]
        for cand in rec["candidates"]:
            segments = build_full_segments(rec["user_input"], cand["response"])
            ids, mask = tokenize_segments(segments, tokenizer, max_seq_len)

            actual_len = sum(len(tokenizer.encode(text)) for text, _ in segments)
            lengths.append(min(actual_len, max_seq_len))

            cs = rec["editor"]["scores"][str(cand["index"])]
            score_vec = [
                cs["observation"],
                cs["punchline"],
                cs["originality"],
                cs["economy"],
                cs["overall"],
            ]

            input_ids_list.append(ids)
            mask_list.append(mask)
            scores_list.append(score_vec)
            group_ids_list.append(group_id)
            is_winner_list.append(cand["index"] == winner_idx)

    return (
        torch.stack(input_ids_list),
        torch.stack(mask_list),
        torch.tensor(scores_list, dtype=torch.float),
        torch.tensor(group_ids_list, dtype=torch.long),
        torch.tensor(is_winner_list, dtype=torch.bool),
        lengths,
    )


def drop_incomplete_groups(data: dict) -> dict:
    """Remove groups that don't have exactly 4 rows, then remap group_ids contiguously."""
    gids = data["group_ids"]
    cnt = Counter(gids.tolist())
    bad = {k for k, v in cnt.items() if v != 4}
    if not bad:
        return data

    keep = [i for i, g in enumerate(gids.tolist()) if g not in bad]
    for key in data:
        data[key] = data[key][keep]

    old2new = {}
    new_gid = 0
    for old in data["group_ids"].tolist():
        if old not in old2new:
            old2new[old] = new_gid
            new_gid += 1
    data["group_ids"] = torch.tensor(
        [old2new[g.item()] for g in data["group_ids"]], dtype=torch.long
    )
    return data


def save_pt(path: str, data: dict) -> None:
    torch.save(data, path)
    n_groups = len(set(data["group_ids"].tolist()))
    print(
        f"  → {path} ({os.path.getsize(path) / 1024**2:.0f} MB) "
        f"— {len(data['input_ids'])} rows, {n_groups} groups"
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Roast 数据预处理 (Weighted SFT)")
    parser.add_argument(
        "--data-file",
        default="data/llm/roast_dataset_v4.jsonl",
        help="原始 JSONL 路径",
    )
    parser.add_argument(
        "--output-dir",
        default="data/llm/roast/weighted_sft",
        help="输出目录",
    )
    parser.add_argument(
        "--vocab", default="checkpoints/tokenizer_vocab.json", help="vocab JSON"
    )
    parser.add_argument(
        "--merges", default="checkpoints/tokenizer_merges.txt", help="merges TXT"
    )
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-verify", action="store_true", help="跳过验证")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Load tokenizer ──
    print("[preprocess_roast] Loading tokenizer...")
    tokenizer = BPETokenizer.from_file(
        args.vocab, args.merges, special_tokens=config.SPECIAL_TOKENS
    )
    print(f"  vocab: {tokenizer.vocab_size} tokens")

    # ── 2. Load & filter clean records ──
    print(f"[preprocess_roast] Loading {args.data_file}...")
    all_records: list[dict] = []
    with open(args.data_file) as f:
        for line in f:
            all_records.append(json.loads(line))
    print(f"  → {len(all_records)} records")

    def _has_all_scores(rec: dict) -> bool:
        for c in rec["candidates"]:
            k = str(c["index"])
            if k not in rec["editor"]["scores"]:
                return False
            s = rec["editor"]["scores"][k]
            if len(s) == 0:
                return False
            if any(v is None for v in s.values()):
                return False
        return True

    clean = [r for r in all_records if _has_all_scores(r)]
    print(
        f"  → {len(clean)} clean, "
        f"{len(all_records) - len(clean)} dirty (dropped)"
    )

    # ── 3. Shuffle & split 90/10 ──
    random.shuffle(clean)
    n_val = int(len(clean) * VAL_RATIO)
    val_records = clean[:n_val]
    train_records = clean[n_val:]
    print(
        f"\n[preprocess_roast] Split: "
        f"train={len(train_records)}, val={len(val_records)}"
    )

    # ── 4. Process ──
    print("\n[preprocess_roast] Processing train...")
    (
        train_ids,
        train_mask,
        train_scores,
        train_gids,
        train_winner,
        train_lens,
    ) = process_grpo_offpolicy(train_records, tokenizer, args.max_seq_len)

    print("[preprocess_roast] Processing val...")
    (
        val_ids,
        val_mask,
        val_scores,
        val_gids,
        val_winner,
        val_lens,
    ) = process_grpo_offpolicy(val_records, tokenizer, args.max_seq_len)

    # ── 5. Drop incomplete groups ──
    train_data = {
        "input_ids": train_ids,
        "completion_mask": train_mask,
        "scores": train_scores,
        "group_ids": train_gids,
        "is_winner": train_winner,
    }
    val_data = {
        "input_ids": val_ids,
        "completion_mask": val_mask,
        "scores": val_scores,
        "group_ids": val_gids,
        "is_winner": val_winner,
    }

    train_data = drop_incomplete_groups(train_data)
    val_data = drop_incomplete_groups(val_data)

    # ── 6. Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n[preprocess_roast] Saving...")
    save_pt(os.path.join(args.output_dir, "train.pt"), train_data)
    save_pt(os.path.join(args.output_dir, "valid.pt"), val_data)

    # ── 7. Statistics ──
    print(f"\n[preprocess_roast] === Statistics ===")
    print(
        f"Train: {len(set(train_data['group_ids'].tolist()))} groups, "
        f"avg_len={np.mean(train_lens):.0f}"
    )
    print(
        f"Valid: {len(set(val_data['group_ids'].tolist()))} groups, "
        f"avg_len={np.mean(val_lens):.0f}"
    )

    # ── 8. Verify val consistency with previous val/grpo.pt (if exists) ──
    if args.no_verify:
        print("\n[preprocess_roast] Done!")
        return

    prev_val_path = "data/llm/roast/val/grpo.pt"
    if os.path.exists(prev_val_path):
        print("\n[preprocess_roast] === Verification ===")
        prev = torch.load(prev_val_path)
        assert len(val_data["input_ids"]) == len(
            prev["input_ids"]
        ), "Val size differs from previous val/grpo.pt"
        assert torch.equal(
            val_data["scores"], prev["scores"]
        ), "Val scores differ from previous val/grpo.pt"
        print("  Val matches previous val/grpo.pt ✓")

    print("\n[preprocess_roast] Done!")


if __name__ == "__main__":
    main()
