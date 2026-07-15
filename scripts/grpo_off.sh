#!/usr/bin/env bash
set -euo pipefail

# GRPO Off-Policy 训练（单卡）
#   train set: 4973 groups
#   effective batch = 1 × group_size=4 × grad_accum=8 = 8 groups/step
#   1 epoch ≈ 4973 / 8 ≈ 621 steps
#   total_steps=1242 跑 2 个 epoch
#
# 超参选择：
#   lr=1e-5        RL 对 LR 敏感，比 SFT 低一个数量级
#   kl_penalty=0.05 防止 policy 偏离 ref 过多
#   clip_eps=0.2    PPO clip，标准值
#   max_grad_norm=5 GRPO 梯度幅度比 CE loss 大，放宽裁剪
#   warmup=60       约占一个 epoch 的 10%

python scripts/grpo_off.py \
    --base-model-path checkpoints/roast_sft_v4/ckpt_final.pt \
    --data-train data/llm/roast/grpo_offpolicy/train.pt \
    --data-val data/llm/roast/val/grpo.pt \
    --batch-size 1 \
    --group-size 4 \
    --grad-accum 8 \
    --total-steps 1242 \
    --lr 1e-5 \
    --lr-min 1e-6 \
    --warmup-steps 60 \
    --kl-penalty 0.05 \
    --clip-eps 0.2 \
    --max-grad-norm 5.0 \
    --val-every 100 \
    --ckpt-every 500 \
    --log-every 10 \
    --output-dir checkpoints/grpo_off
