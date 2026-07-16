#!/usr/bin/env bash
set -euo pipefail

# GRPO Off-Policy 训练（5 卡 DDP）
#   train set: 4973 groups
#   effective batch = 5 GPUs × 1 × 1 = 5 groups/step
#   1 epoch ≈ 4973 / 5 ≈ 994 steps
#   total_steps=1988 跑 2 个 epoch
#
# 超参选择：
#   lr=1e-5        RL 对 LR 敏感，比 SFT 低一个数量级
#   kl_penalty=0.05 防止 policy 偏离 ref 过多
#   clip_eps=0.2    PPO clip，标准值
#   max_grad_norm=5 GRPO 梯度幅度比 CE loss 大，放宽裁剪
#   warmup=100      约占一个 epoch 的 10%

torchrun --nproc_per_node=5 scripts/grpo_off.py \
    --data-train data/llm/roast/grpo_offpolicy/train.pt \
    --data-val data/llm/roast/val/grpo.pt \
    --wandb-project mini_deepseek_v4 \
    --base-model-path checkpoints/roast_sft_v4/ckpt_final.pt \
    --total-steps 4000 \
    --lr 1e-5 \
    --lr-min 1e-6 \
    --warmup-steps 100 \
    --kl-penalty 0.02 \
    --clip-eps 0.2 \
    --max-grad-norm 5.0 \
    --val-every 200 \
    --ckpt-every 500 \
    --log-every 10 \
    --keep-last-ckpt 8 \
    --output-dir checkpoints/grpo_off
