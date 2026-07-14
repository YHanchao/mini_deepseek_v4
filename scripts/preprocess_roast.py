"""
Preprocess roast_dataset.jsonl for SFT, Off-policy GRPO, On-policy GRPO, and Validation.

Usage:
    python scripts/preprocess_roast.py

Output:
    data/llm/roast/sft/train.pt
    data/llm/roast/val/sft.pt
    data/llm/roast/val/grpo.pt
    data/llm/roast/grpo_offpolicy/train.pt
    data/llm/roast/grpo_onpolicy/train.pt

Records with missing candidate scores (filtered low-quality candidates) are excluded
from GRPO splits but their winners are still used for SFT training.
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

VAL_RATIO = 0.10
SFT_RATIO = 0.60
ONPOLICY_RATIO = 0.10
OFFPOLICY_RATIO = 0.20


# ── Response Cleaning ────────────────────────────────────────────────────────

# Prefix artifacts from LLM synthesis — strip aggressively
_PREFIX_ARTIFACTS = [
    "GOOD: ",
    "GOOD:\n\n",
    "GOOD:\n",
    "BAD: ",
    "ROAST: ",
    "OK: ",
    "MEH: ",
    "(Observing the specific absurdity)\n\n",
    "(Observing the specific absurdity)\n",
]


def clean_response(text: str) -> str:
    """Strip known LLM synthesis artifacts from a candidate response."""
    for prefix in _PREFIX_ARTIFACTS:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break

    # Strip bracketed stage directions: [DEADPAN]\n\n, [Deadpan] , etc.
    import re

    text = re.sub(
        r"^\[(?:DEADPAN|Deadpan|deadpan|Flat tone|flatly[^]]*|SOUNDLEVEL[^]]*|WILDLIFE DOCUMENTARY|A host[^]]*|A man[^]]*)\]\s*\n*",
        "",
        text,
    )

    # Strip markdown document headers at start of line
    # e.g. "**ABSTRACT:** ", "**Abstract**\n", "**Performance Review – ...**\n"
    text = re.sub(r"^\*\*[^*]+\*\*[:\s]*\n*", "", text)
    # Second pass: some have multiple headers
    text = re.sub(r"^\*\*[^*]+\*\*[:\s]*\n*", "", text)

    # Strip "**OBSERVATION:** ... **ROAST:** ..." structured format
    text = re.sub(r"^\*\*OBSERVATION:?\*\*[^*]*\*\*ROAST:?\*\*\s*", "", text)
    text = re.sub(r"^\*\*OBSERVE:?\*\*\s*[^*]+\n", "", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


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
        winner_text = clean_response(rec["winner_response"])
        segments = build_sft_segments(rec["user_input"], winner_text)
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
            cleaned_resp = clean_response(cand["response"])
            segments = build_sft_segments(rec["user_input"], cleaned_resp)
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
        "--data-file", default="data/llm/roast_dataset_v4.jsonl", help="原始 JSONL 路径"
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
    all_records: list[dict] = []
    with open(args.data_file) as f:
        for line in f:
            all_records.append(json.loads(line))
    print(f"  → {len(all_records)} records")

    # ── 2.5. Separate clean (all 4 candidates scored) vs dirty (some missing) ──
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

    # Normalise known typos in score keys
    for rec in all_records:
        for c in rec["candidates"]:
            k = str(c["index"])
            if k in rec["editor"]["scores"]:
                s = rec["editor"]["scores"][k]
                if "ecomony" in s:
                    s["economy"] = s.pop("ecomony")

    clean_records = [r for r in all_records if _has_all_scores(r)]
    dirty_records = [r for r in all_records if not _has_all_scores(r)]
    print(
        f"  → {len(clean_records)} clean (all candidates scored), "
        f"{len(dirty_records)} dirty (partial scores → winners → SFT only)"
    )

    # ── 3. Shuffle & split clean records ──
    random.shuffle(clean_records)

    n_clean = len(clean_records)
    n_val = int(n_clean * VAL_RATIO)
    n_sft_clean = int(n_clean * SFT_RATIO)
    n_onpolicy = int(n_clean * ONPOLICY_RATIO)

    val_records = clean_records[:n_val]
    sft_clean_records = clean_records[n_val : n_val + n_sft_clean]
    onpolicy_records = clean_records[
        n_val + n_sft_clean : n_val + n_sft_clean + n_onpolicy
    ]
    offpolicy_records = clean_records[n_val + n_sft_clean + n_onpolicy :]

    # Dirty records → SFT train only (winners always have scores)
    sft_records = sft_clean_records + dirty_records
    random.shuffle(sft_records)

    print(
        f"\n[preprocess_roast] Split: "
        f"val={len(val_records)}, "
        f"sft={len(sft_records)} ({len(sft_clean_records)} clean + {len(dirty_records)} dirty), "
        f"offpolicy={len(offpolicy_records)}, "
        f"onpolicy={len(onpolicy_records)}"
    )

    # ── 4. Process each split ──

    print("\n[preprocess_roast] Processing val (SFT format)...")
    val_sft_ids, val_sft_mask, val_sft_lens = process_sft(
        val_records, tokenizer, args.max_seq_len
    )

    print("[preprocess_roast] Processing val (GRPO format)...")
    (
        val_grpo_ids,
        val_grpo_cmask,
        val_grpo_scores,
        val_grpo_gids,
        val_grpo_winner,
        val_grpo_lens,
    ) = process_grpo_offpolicy(val_records, tokenizer, args.max_seq_len)

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
    os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "grpo_offpolicy"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "grpo_onpolicy"), exist_ok=True)

    print("\n[preprocess_roast] Saving...")

    def save_pt(dirname: str, filename: str, data: dict) -> None:
        path = os.path.join(args.output_dir, dirname, filename)
        torch.save(data, path)
        print(f"  → {path} ({os.path.getsize(path) / 1024**2:.1f} MB)")

    save_pt("sft", "train.pt", {"input_ids": sft_ids, "assistant_mask": sft_mask})
    save_pt("val", "sft.pt", {"input_ids": val_sft_ids, "assistant_mask": val_sft_mask})
    save_pt(
        "val",
        "grpo.pt",
        {
            "input_ids": val_grpo_ids,
            "completion_mask": val_grpo_cmask,
            "scores": val_grpo_scores,
            "group_ids": val_grpo_gids,
            "is_winner": val_grpo_winner,
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
        f"Val (SFT):      {len(val_sft_ids):>5} samples, "
        f"avg_len={np.mean(val_sft_lens):.0f}"
    )
    print(
        f"Val (GRPO):     {len(val_grpo_ids):>5} rows "
        f"({len(val_records)} groups), avg_len={np.mean(val_grpo_lens):.0f}"
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
    print(
        f"Total prompts:  {len(val_records) + len(sft_records) + len(offpolicy_records) + len(onpolicy_records)}"
    )

    # ── 8. Verify ──
    if args.no_verify:
        print("\n[preprocess_roast] Done!")
        return

    print("\n[preprocess_roast] === Verification ===")

    assert len(val_sft_ids) == len(
        val_records
    ), f"Val SFT: {len(val_sft_ids)} != {len(val_records)}"
    assert len(val_grpo_ids) == 4 * len(
        val_records
    ), f"Val GRPO rows: {len(val_grpo_ids)} != 4*{len(val_records)}"
    assert val_grpo_winner.sum().item() == len(
        val_records
    ), f"Val winners: {val_grpo_winner.sum()} != {len(val_records)}"

    # Verify each val group has exactly 1 winner
    for gid in range(len(val_records)):
        group_winners = val_grpo_winner[val_grpo_gids == gid]
        assert (
            group_winners.sum().item() == 1
        ), f"Group {gid} has {group_winners.sum()} winners"

    print("  All assertions passed!")

    # Decode sample
    print("\n  --- Sample decode (first val SFT record) ---")
    ids = val_sft_ids[0]
    mask = val_sft_mask[0]
    non_pad = (ids != PAD_ID).sum().item()
    text = tokenizer.decode(ids[:non_pad].tolist())
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
