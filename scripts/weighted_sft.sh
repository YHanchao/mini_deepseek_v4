#!/usr/bin/env bash
set -euo pipefail

# WeightedSFT + SimPO（5 卡 DDP）
#   train set: 4973 groups
#   effective batch = 5 GPUs × 1 = 5 pairs/step
#   1 epoch ≈ 4973 / 5 ≈ 994 steps
#   total_steps=4000 跑 ~4 个 epoch
#
#   SimPO: winner vs worst loser, NTP only, beta=2, gamma=0

# ---- lambda_simpo = 0.1 ----
torchrun --nproc_per_node=5 scripts/weighted_sft.py \
    --data-train data/roast/grpo_offpolicy/train.pt \
    --data-val data/roast/val/grpo.pt \
    --wandb-project mini_deepseek_v4 \
    --base-model-path checkpoints/roast_sft_long/ckpt_best.pt \
    --resume checkpoints/roast_sft_long/ckpt_best.pt \
    --resume-step false \
    --batch-size 1 \
    --total-steps 4000 \
    --lr 1e-5 \
    --lr-min 1e-6 \
    --warmup-steps 30 \
    --val-every 100 \
    --ckpt-every 1000 \
    --log-every 10 \
    --keep-last-ckpt 4 \
    --grad-accum 1 \
    --max-grad-norm 5.0 \
    --beta 2.0 \
    --gamma 0.0 \
    --lambda-simpo 0.1 \
    --output-dir checkpoints/weighted_sft_simpo01

# ---- lambda_simpo = 1.0 ----
torchrun --nproc_per_node=5 scripts/weighted_sft.py \
    --data-train data/roast/grpo_offpolicy/train.pt \
    --data-val data/roast/val/grpo.pt \
    --wandb-project mini_deepseek_v4 \
    --base-model-path checkpoints/roast_sft_long/ckpt_best.pt \
    --resume checkpoints/roast_sft_long/ckpt_best.pt \
    --resume-step false \
    --batch-size 1 \
    --total-steps 4000 \
    --lr 1e-5 \
    --lr-min 1e-6 \
    --warmup-steps 30 \
    --val-every 100 \
    --ckpt-every 1000 \
    --log-every 10 \
    --keep-last-ckpt 4 \
    --grad-accum 1 \
    --max-grad-norm 5.0 \
    --beta 2.0 \
    --gamma 0.0 \
    --lambda-simpo 1.0 \
    --output-dir checkpoints/weighted_sft_simpo1
