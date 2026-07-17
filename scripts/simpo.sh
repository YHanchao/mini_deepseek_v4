#!/usr/bin/env bash
set -euo pipefail

# SimPO + SFT 训练（5 卡 DDP）
#   train set: 4973 groups (winner + lowest-score loser)
#   effective batch = 5 GPUs × 1 = 5 pairs/step
#   1 epoch ≈ 4973 / 5 ≈ 994 steps
#   total_steps=1988 跑 2 个 epoch
#
#   LR 沿用 SFT 的设置：2.7e-4（SimPO 无需 ref model，不需要压低 LR）
#   beta=1.0, gamma=0.1 是 SimPO 论文的默认值

torchrun --nproc_per_node=5 scripts/simpo.py \
    --data-train data/llm/roast/grpo_offpolicy/train.pt \
    --data-val data/llm/roast/val/grpo.pt \
    --wandb-project mini_deepseek_v4 \
    --base-model-path checkpoints/roast_sft_v4/ckpt_final.pt \
    --total-steps 1988 \
    --lr 2.7e-4 \
    --lr-min 2.7e-5 \
    --warmup-steps 200 \
    --beta 1.0 \
    --gamma 0.1 \
    --lambda-simpo 1.0 \
    --val-every 200 \
    --ckpt-every 500 \
    --log-every 10 \
    --output-dir checkpoints/simpo
