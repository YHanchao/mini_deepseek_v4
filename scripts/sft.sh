#!/usr/bin/env bash
set -euo pipefail

# SFT 超参说明：
#   有效 batch size = 5 GPUs × 4 = 20 samples/step
#   1 epoch = 63782 / 20 ≈ 3189 steps
#   total_steps=6378 正好跑 2 个 epoch
#   lr=1e-4 比预训练低 ~2.7x，避免破坏预训练表征
#   warmup=300 约占一个 epoch 的 10%

torchrun --nproc_per_node=5 scripts/sft.py \
    --data-train /mnt/MiniDSv4/data/sft/train.pt \
    --data-val /mnt/MiniDSv4/data/sft/valid.pt \
    --wandb-project mini_deepseek_v4 \
    --base-model-path checkpoints/pretrain/ckpt_0375000.pt \
    --total-steps 6378 \
    --lr 1e-4 \
    --lr-min 1e-5 \
    --warmup-steps 300 \
    --val-every 300 \
    --ckpt-every 2000 \
    --output-dir checkpoints/sft
