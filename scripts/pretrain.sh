torchrun --nproc_per_node=4 scripts/pretrain.py \
    --data-train /data/train.bin \
    --data-val  /data/valid.bin \
    --total-steps 401000 \
    --lr 1e-4 \
    --lr-min 1e-5 \
    --warmup-steps 2000 \
    --log-every 10