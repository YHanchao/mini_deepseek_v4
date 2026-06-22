"""Pretraining entry point for DeepSeekV4 mini.

Usage:
    # Single GPU (DGX Spark debugging)
    python scripts/pretrain.py --config tiny --data-train /path/to/data.bin --total-steps 100

    # 4-GPU DDP
    torchrun --nproc_per_node=4 scripts/pretrain.py \
        --data-train /mnt/nfs/data/train.bin \
        --data-val /mnt/nfs/data/valid.bin
"""

import argparse

from src.trainer import PretrainTrainer, PretrainTrainerArgs


def get_args():
    p = argparse.ArgumentParser(description="Pretrain DeepSeekV4 mini")

    # 从命令行覆盖的关键字段（其他字段用 PretrainTrainerArgs 默认值）
    p.add_argument("--config", type=str, default="small")
    p.add_argument("--data-train", type=str, default="data/train.bin")
    p.add_argument("--data-val", type=str, default="data/valid.bin")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--total-steps", type=int, default=401000)
    p.add_argument("--lr", type=float, default=2.7e-4)
    p.add_argument("--lr-min", type=float, default=2.7e-5)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--output-dir", type=str, default="checkpoints/pretrain")
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--keep-last-ckpt", type=int, default=1)
    p.add_argument("--wandb-project", type=str, default="mini_deepseek_v4")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    return p.parse_args()


def main():
    cli = get_args()
    train_args = PretrainTrainerArgs(**vars(cli))
    trainer = PretrainTrainer(train_args)
    trainer.train()


if __name__ == "__main__":
    main()
