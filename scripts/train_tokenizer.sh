#!/bin/bash
# 训练 BPE tokenizer 的快捷脚本
# 用法: bash scripts/train_tokenizer.sh

set -euo pipefail

# 默认参数，可按需修改
INPUT="${INPUT:-data/owt_train.txt}"
VOCAB_SIZE="${VOCAB_SIZE:-50257}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
TOKENIZER_NAME="${TOKENIZER_NAME:-gpt2}"
CHUNK_SIZE="${CHUNK_SIZE:-536870912}"

echo "========================================="
echo " Training BPE Tokenizer"
echo "========================================="
echo " Input:        ${INPUT}"
echo " Vocab size:   ${VOCAB_SIZE}"
echo " Output dir:   ${OUTPUT_DIR}"
echo " Tokenizer:    ${TOKENIZER_NAME}"
echo " Max chunk size:    ${CHUNK_SIZE}"
echo "========================================="

python scripts/train_tokenizer.py \
    --input "${INPUT}" \
    --vocab-size "${VOCAB_SIZE}" \
    --output-dir "${OUTPUT_DIR}" \
    --name "${TOKENIZER_NAME}" \
    --max-chunk-size "${CHUNK_SIZE}" \
    "$@"
