#!/usr/bin/env bash
set -euo pipefail

# Weighted SFT 正式实验（5 卡 DDP）
#   数据: data/llm/roast/weighted_sft/train.pt (22371 groups)
#   1 epoch = 22371 / 5 ≈ 4474 steps
#
#   weight_min schedule: 从 0.7 线性衰减到 0.2
#   winner 权重恒为 1.0

torchrun --nproc_per_node=5 scripts/weighted_sft.py \
    --data-train data/weighted_sft/train.pt \
    --data-val data/weighted_sft/valid.pt \
    --wandb-project mini_deepseek_v4 \
    --base-model-path checkpoints/sft/ckpt_final.pt \
    --resume checkpoints/sft/ckpt_final.pt \
    --resume-step false \
    --batch-size 1 \
    --total-steps 12000 \
    --lr 1e-5 \
    --lr-min 1e-6 \
    --warmup-steps 200 \
    --val-every 200 \
    --ckpt-every 2000 \
    --log-every 10 \
    --keep-last-ckpt 8 \
    --grad-accum 1 \
    --max-grad-norm 5.0 \
    --weight-min-start 0.7 \
    --weight-min-end 0.2 \
    --output-dir checkpoints/weighted_sft_long

