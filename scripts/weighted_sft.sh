#!/usr/bin/env bash
set -euo pipefail

# Weighted SFT 训练（5 卡 DDP）
#   train set: 4973 groups
#   effective batch = 5 GPUs × 1 = 5 pairs/step
#   1 epoch ≈ 4973 / 5 ≈ 994 steps
#   total_steps=1988 跑 2 个 epoch
#
#   LR 沿用 SFT 的设置：2.7e-4（SimPO 无需 ref model，不需要压低 LR）
#   beta=1.0, gamma=0.1 是 SimPO 论文的默认值

torchrun --nproc_per_node=5 scripts/simpo.py \
    --data-train data/roast/grpo_offpolicy/train.pt \
    --data-val data/roast/val/grpo.pt \
    --wandb-project mini_deepseek_v4 \
    --base-model-path checkpoints/roast_sft_long/ckpt_best.pt \
    --resume checkpoints/roast_sft_long/ckpt_best.pt \
    --batch-size 1 \
    --total-steps 4000 \
    --lr 1e-5 \
    --lr-min 1e-6 \
    --warmup-steps 30 \
    --val-every 100 \
    --ckpt-every 500 \
    --log-every 10 \
    --keep-last-ckpt 8 \
    --grad-accum 1 \
    --max-grad-norm 5.0 \
    --output-dir checkpoints/weighted_sft
