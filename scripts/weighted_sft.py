"""Weighted SFT training entry point for DeepSeekV4 mini."""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import WeightedSFTTrainer, WeightedSFTTrainerArgs


def get_args():
    p = argparse.ArgumentParser(description="Weighted SFT DeepSeekV4 mini")
    p.add_argument("--config-name", type=str, default="small")
    p.add_argument(
        "--data-train", type=str, default="data/llm/roast/grpo_offpolicy/train.pt"
    )
    p.add_argument("--data-val", type=str, default="data/llm/roast/val/grpo.pt")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2.7e-5)
    p.add_argument("--lr-min", type=float, default=2.7e-6)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default="checkpoints/weighted_sft")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--keep-last-ckpt", type=int, default=2)
    p.add_argument("--wandb-project", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--resume-step", type=bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--uniform", action="store_true", default=False)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--base-model-path", type=str, default="checkpoints/roast_sft/ckpt_best.pt"
    )
    return p.parse_args()


def main():
    cli = get_args()
    train_args = WeightedSFTTrainerArgs(**vars(cli))
    trainer = WeightedSFTTrainer(train_args)
    trainer.train()


if __name__ == "__main__":
    main()
