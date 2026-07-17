#!/usr/bin/env bash
set -euo pipefail

# DPO 训练（5 卡 DDP）
#   train set: 4973 groups
#   effective batch = 5 GPUs × 1 = 5 groups/step (30 pairs)
#   1 epoch ≈ 4973 / 5 ≈ 995 steps
#   total_steps=2000 跑 ~2 个 epoch
#
# 超参选择：
#   lr=1e-5         RL 对 LR 敏感，比 SFT 低一个数量级
#   beta=0.1        Bradley-Terry temperature，控制 policy 偏离 ref 的程度
#                   当前用 mean log_ratio，初始时 diff 很小，需要足够 steps 积累信号
#   warmup=50       约 5% 的一个 epoch
#   max_grad_norm=1.0  DPO 梯度量级与 CE loss 接近，标准 clip 即可

torchrun --nproc_per_node=5 scripts/dpo.py \
    --data-train data/roast/grpo_offpolicy/train.pt \
    --data-val data/roast/val/grpo.pt \
    --wandb-project mini_deepseek_v4 \
    --base-model-path checkpoints/roast_sft/ckpt_best.pt \
    --total-steps 4000 \
    --lr 1e-5 \
    --lr-min 1e-6 \
    --warmup-steps 50 \
    --beta 0.1 \
    --max-grad-norm 1.0 \
    --val-every 100 \
    --ckpt-every 500 \
    --log-every 10 \
    --keep-last-ckpt 4 \
    --output-dir checkpoints/dpo
