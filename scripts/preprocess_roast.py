"""
Preprocess roast_dataset.jsonl for SFT, Off-policy GRPO, On-policy GRPO, and Test.

Usage:
    python scripts/preprocess_roast.py

Output:
    data/llm/roast/sft/train.pt
    data/llm/roast/test/sft.pt
    data/llm/roast/test/grpo.pt
    data/llm/roast/grpo_offpolicy/train.pt
    data/llm/roast/grpo_onpolicy/train.pt
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a witty roast comedian.\nWrite concise observational roasts.\nAvoid explaining the joke."
MAX_SEQ_LEN = 1024
EOS_ID = 256
PAD_ID = 256
RANDOM_SEED = 42

TEST_RATIO = 0.10
SFT_RATIO = 0.50
OFFPOLICY_RATIO = 0.30
ONPOLICY_RATIO = 0.10


# ── Segment Builders ─────────────────────────────────────────────────────────


def build_sft_segments(
    user_input: str, response: str
) -> list[tuple[str, bool]]:
    """Build segments for a complete SFT sequence (prompt + winner + EOS)."""
    return [
        (f"<|system|>\n{SYSTEM_PROMPT}\n", False),
        (f"<|user|>\n{user_input}\n", False),
        (f"<|assistant|>\n{response}", True),
        ("<|endoftext|>\n", False),
    ]


def build_grpo_prompt_segments(user_input: str) -> list[tuple[str, bool]]:
    """Build segments for on-policy GRPO prompt (no response, ends with assistant marker)."""
    return [
        (f"<|system|>\n{SYSTEM_PROMPT}\n", False),
        (f"<|user|>\n{user_input}\n", False),
        ("<|assistant|>\n", False),
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


def tokenize_prompt(
    segments: list[tuple[str, bool]],
    tokenizer: BPETokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompt-only segments into (input_ids, prompt_mask).

    prompt_mask is True for all non-padding tokens.
    """
    all_ids: list[int] = []
    for text, _ in segments:
        all_ids.extend(tokenizer.encode(text))

    if len(all_ids) > max_seq_len:
        all_ids = all_ids[-max_seq_len:]

    prompt_mask = [True] * len(all_ids)

    if len(all_ids) < max_seq_len:
        pad_len = max_seq_len - len(all_ids)
        all_ids.extend([PAD_ID] * pad_len)
        prompt_mask.extend([False] * pad_len)

    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(prompt_mask, dtype=torch.bool),
    )


# ── Split Processors ─────────────────────────────────────────────────────────


def process_sft(
    records: list[dict], tokenizer: BPETokenizer, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Process records into SFT format. Returns (input_ids, assistant_mask, lengths)."""
    input_ids_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    lengths: list[int] = []

    for rec in records:
        segments = build_sft_segments(rec["user_input"], rec["winner_response"])
        ids, mask = tokenize_segments(segments, tokenizer, max_seq_len)

        actual_len = sum(len(tokenizer.encode(text)) for text, _ in segments)
        lengths.append(min(actual_len, max_seq_len))

        input_ids_list.append(ids)
        mask_list.append(mask)

    return torch.stack(input_ids_list), torch.stack(mask_list), lengths


def process_grpo_offpolicy(
    records: list[dict], tokenizer: BPETokenizer, max_seq_len: int
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]
]:
    """Process records into off-policy GRPO format.

    Each record is flattened into 4 rows (one per candidate).
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
            segments = build_sft_segments(rec["user_input"], cand["response"])
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


def process_grpo_onpolicy(
    records: list[dict], tokenizer: BPETokenizer, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Process records into on-policy GRPO format. Returns (input_ids, prompt_mask, lengths)."""
    input_ids_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    lengths: list[int] = []

    for rec in records:
        segments = build_grpo_prompt_segments(rec["user_input"])
        ids, mask = tokenize_prompt(segments, tokenizer, max_seq_len)

        actual_len = sum(len(tokenizer.encode(text)) for text, _ in segments)
        lengths.append(min(actual_len, max_seq_len))

        input_ids_list.append(ids)
        mask_list.append(mask)

    return torch.stack(input_ids_list), torch.stack(mask_list), lengths


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Roast 数据预处理")
    parser.add_argument(
        "--data-file", default="data/llm/roast_dataset.jsonl", help="原始 JSONL 路径"
    )
    parser.add_argument("--output-dir", default="data/llm/roast", help="输出目录")
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
    parser.add_argument("--no-verify", action="store_true", help="跳过验证")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Load tokenizer ──
    print("[preprocess_roast] Loading tokenizer...")
    tokenizer = BPETokenizer.from_file(
        args.vocab, args.merges, special_tokens=config.SPECIAL_TOKENS
    )
    print(
        f"[preprocess_roast] vocab: {tokenizer.vocab_size} tokens, "
        f"merges: {len(tokenizer.merges)} 条"
    )

    # ── 2. Load data ──
    print(f"[preprocess_roast] Loading {args.data_file}...")
    records: list[dict] = []
    with open(args.data_file) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  → {len(records)} records")

    # ── 2.5. Filter records missing scores for any candidate ──
    before = len(records)
    records = [
        r
        for r in records
        if all(
            str(c["index"]) in r["editor"]["scores"] for c in r["candidates"]
        )
    ]
    print(
        f"  → {len(records)} records after dropping "
        f"{before - len(records)} with missing scores ({100 * (before - len(records)) / before:.1f}%)"
    )

    # ── 3. Shuffle & split ──
    random.shuffle(records)

    n = len(records)
    n_test = int(n * TEST_RATIO)
    n_sft = int(n * SFT_RATIO)
    n_offpolicy = int(n * OFFPOLICY_RATIO)

    test_records = records[:n_test]
    sft_records = records[n_test : n_test + n_sft]
    offpolicy_records = records[n_test + n_sft : n_test + n_sft + n_offpolicy]
    onpolicy_records = records[n_test + n_sft + n_offpolicy :]

    print(
        f"\n[preprocess_roast] Split: "
        f"test={len(test_records)}, "
        f"sft={len(sft_records)}, "
        f"offpolicy={len(offpolicy_records)}, "
        f"onpolicy={len(onpolicy_records)}"
    )

    # ── 4. Process each split ──

    print("\n[preprocess_roast] Processing test (SFT format)...")
    test_sft_ids, test_sft_mask, test_sft_lens = process_sft(
        test_records, tokenizer, args.max_seq_len
    )

    print("[preprocess_roast] Processing test (GRPO format)...")
    (
        test_grpo_ids,
        test_grpo_cmask,
        test_grpo_scores,
        test_grpo_gids,
        test_grpo_winner,
        test_grpo_lens,
    ) = process_grpo_offpolicy(test_records, tokenizer, args.max_seq_len)

    print("[preprocess_roast] Processing SFT train...")
    sft_ids, sft_mask, sft_lens = process_sft(
        sft_records, tokenizer, args.max_seq_len
    )

    print("[preprocess_roast] Processing off-policy GRPO train...")
    (
        off_ids,
        off_cmask,
        off_scores,
        off_gids,
        off_winner,
        off_lens,
    ) = process_grpo_offpolicy(offpolicy_records, tokenizer, args.max_seq_len)

    print("[preprocess_roast] Processing on-policy GRPO train...")
    on_ids, on_pmask, on_lens = process_grpo_onpolicy(
        onpolicy_records, tokenizer, args.max_seq_len
    )

    # ── 5. Shuffle SFT train ──
    train_idx = torch.randperm(len(sft_ids))
    sft_ids = sft_ids[train_idx]
    sft_mask = sft_mask[train_idx]

    # ── 6. Save ──
    os.makedirs(os.path.join(args.output_dir, "sft"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "grpo_offpolicy"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "grpo_onpolicy"), exist_ok=True)

    print("\n[preprocess_roast] Saving...")

    def save_pt(dirname: str, filename: str, data: dict) -> None:
        path = os.path.join(args.output_dir, dirname, filename)
        torch.save(data, path)
        print(f"  → {path} ({os.path.getsize(path) / 1024**2:.1f} MB)")

    save_pt("sft", "train.pt", {"input_ids": sft_ids, "assistant_mask": sft_mask})
    save_pt("test", "sft.pt", {"input_ids": test_sft_ids, "assistant_mask": test_sft_mask})
    save_pt(
        "test",
        "grpo.pt",
        {
            "input_ids": test_grpo_ids,
            "completion_mask": test_grpo_cmask,
            "scores": test_grpo_scores,
            "group_ids": test_grpo_gids,
            "is_winner": test_grpo_winner,
        },
    )
    save_pt(
        "grpo_offpolicy",
        "train.pt",
        {
            "input_ids": off_ids,
            "completion_mask": off_cmask,
            "scores": off_scores,
            "group_ids": off_gids,
            "is_winner": off_winner,
        },
    )
    save_pt(
        "grpo_onpolicy",
        "train.pt",
        {"input_ids": on_ids, "prompt_mask": on_pmask},
    )

    # ── 7. Statistics ──
    print(f"\n[preprocess_roast] === Statistics ===")
    print(
        f"Test (SFT):     {len(test_sft_ids):>5} samples, "
        f"avg_len={np.mean(test_sft_lens):.0f}"
    )
    print(
        f"Test (GRPO):    {len(test_grpo_ids):>5} rows "
        f"({len(test_records)} groups), avg_len={np.mean(test_grpo_lens):.0f}"
    )
    print(
        f"SFT train:      {len(sft_ids):>5} samples, "
        f"avg_len={np.mean(sft_lens):.0f}"
    )
    print(
        f"Off-policy:     {len(off_ids):>5} rows "
        f"({len(offpolicy_records)} groups), avg_len={np.mean(off_lens):.0f}"
    )
    print(
        f"On-policy:      {len(on_ids):>5} samples, "
        f"avg_len={np.mean(on_lens):.0f}"
    )

    # ── 8. Verify ──
    if args.no_verify:
        print("\n[preprocess_roast] Done!")
        return

    print("\n[preprocess_roast] === Verification ===")

    assert len(test_sft_ids) == len(
        test_records
    ), f"Test SFT: {len(test_sft_ids)} != {len(test_records)}"
    assert len(test_grpo_ids) == 4 * len(
        test_records
    ), f"Test GRPO rows: {len(test_grpo_ids)} != 4*{len(test_records)}"
    assert test_grpo_winner.sum().item() == len(
        test_records
    ), f"Test winners: {test_grpo_winner.sum()} != {len(test_records)}"

    # Verify each group has exactly 1 winner
    for gid in range(len(test_records)):
        group_winners = test_grpo_winner[test_grpo_gids == gid]
        assert (
            group_winners.sum().item() == 1
        ), f"Group {gid} has {group_winners.sum()} winners"

    print("  All assertions passed!")

    # Decode sample
    print("\n  --- Sample decode (first test SFT record) ---")
    ids = test_sft_ids[0]
    mask = test_sft_mask[0]
    # Find non-pad length
    non_pad = (ids != PAD_ID).sum().item()
    text = tokenizer.decode(ids[:non_pad].tolist())
    # Find assistant region
    asst_start = 0
    for j in range(len(mask)):
        if mask[j] and not (j > 0 and mask[j - 1]):
            asst_start = j
            break
    print(f"  [assistant_mask True starts at token pos {asst_start}]")
    print(f"  {text[:600]}")
    print()

    print("[preprocess_roast] Done!")


if __name__ == "__main__":
    main()
