"""Pretraining entry point for DeepSeekV4 mini.

Usage (single GPU):
    python scripts/pretrain.py --config tiny --data-train data/train.bin --total-steps 100

Usage (4-GPU DDP):
    torchrun --nproc_per_node=4 scripts/pretrain.py \
        --config small --data-train data/train.bin --data-val data/valid.bin
"""

from src.trainer import PretrainTrainer, PretrainTrainerArgs

train_args = PretrainTrainerArgs(
    config_name="small",
    total_steps=401000,
    data_train="data/train.bin",
    data_val="data/valid.bin",
    batch_size=4,
    max_seq_len=1024,
    # 学习率直接copy文章的设置
    lr=2.7e-4,
    lr_min=2.7e-5,
    warmup_steps=2000,
    # Output 相关
    output_dir="checkpoints/pretrain",
    ckpt_every=5000,
    val_every=500,
    log_every=100,
    keep_last_ckpt=5,
    # Wandb
    wandb_project="mini_deepseek_v4",
)


def main():
    trainer = PretrainTrainer(train_args)
    trainer.train()


if __name__ == "__main__":
    main()
