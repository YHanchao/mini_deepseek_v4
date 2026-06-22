#!/bin/bash
# 预训练数据 Tokenization
#   OWT + TinyStories → train.bin / valid.bin
#
# 用法: bash scripts/pre_tokenization.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
VOCAB="${VOCAB:-checkpoints/tokenizer_vocab.json}"
MERGES="${MERGES:-checkpoints/tokenizer_merges.txt}"
CHUNK_SIZE="${CHUNK_SIZE:-1G}"

# ---- 输入文件 ----
OWT_TRAIN="${DATA_DIR}/owt_train.txt"                       # ~12 GB
OWT_VALID="${DATA_DIR}/owt_valid.txt"                       # ~277 MB
TS_TRAIN="${DATA_DIR}/TinyStoriesV2-GPT4-train.txt"         # ~2.1 GB
TS_VALID="${DATA_DIR}/TinyStoriesV2-GPT4-valid.txt"         # ~22 MB

# ---- 中间输出 ----
OWT_TRAIN_BIN="${DATA_DIR}/owt_train.bin"
OWT_VALID_BIN="${DATA_DIR}/owt_valid.bin"
TS_TRAIN_BIN="${DATA_DIR}/TinyStoriesV2_train.bin"
TS_VALID_BIN="${DATA_DIR}/TinyStoriesV2_valid.bin"

# ---- 最终输出 ----
TRAIN_BIN="${OUTPUT_DIR}/train.bin"
VALID_BIN="${OUTPUT_DIR}/valid.bin"

echo "========================================="
echo " Pre-tokenization → train.bin / valid.bin"
echo "========================================="
echo " Tokenizer:  ${VOCAB}"
echo " Chunk size: ${CHUNK_SIZE}"
echo ""
echo " 训练数据:"
echo "   ${OWT_TRAIN}   (12 GB, 12 workers)"
echo "   ${TS_TRAIN}    (2.1 GB, 3 workers)"
echo ""
echo " 验证数据:"
echo "   ${OWT_VALID}   (277 MB, 1 worker)"
echo "   ${TS_VALID}    (22 MB, 1 worker)"
echo "========================================="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dpsk

# ---- 四路并行 tokenize ----
python scripts/pre_tokenization.py \
    --input "${OWT_TRAIN}" --output "${OWT_TRAIN_BIN}" \
    --vocab "${VOCAB}" --merges "${MERGES}" \
    --chunk-size "${CHUNK_SIZE}" --num-workers 12 &
PID1=$!

python scripts/pre_tokenization.py \
    --input "${TS_TRAIN}" --output "${TS_TRAIN_BIN}" \
    --vocab "${VOCAB}" --merges "${MERGES}" \
    --chunk-size "${CHUNK_SIZE}" --num-workers 3 &
PID2=$!

python scripts/pre_tokenization.py \
    --input "${OWT_VALID}" --output "${OWT_VALID_BIN}" \
    --vocab "${VOCAB}" --merges "${MERGES}" \
    --chunk-size "${CHUNK_SIZE}" --num-workers 1 &
PID3=$!

python scripts/pre_tokenization.py \
    --input "${TS_VALID}" --output "${TS_VALID_BIN}" \
    --vocab "${VOCAB}" --merges "${MERGES}" \
    --chunk-size "${CHUNK_SIZE}" --num-workers 1 &
PID4=$!

echo ""
echo "PID  OWT train:       ${PID1}"
echo "PID  TinyStories train: ${PID2}"
echo "PID  OWT valid:       ${PID3}"
echo "PID  TinyStories valid: ${PID4}"
echo ""
echo "等待 tokenization 完成 (17/20 cores)..."

wait ${PID1} ${PID2} ${PID3} ${PID4}

# ---- 拼接 ----
echo ""
echo "拼接训练集: ${OWT_TRAIN_BIN} + ${TS_TRAIN_BIN} → ${TRAIN_BIN}"
cat "${OWT_TRAIN_BIN}" "${TS_TRAIN_BIN}" > "${TRAIN_BIN}"

echo "拼接验证集: ${OWT_VALID_BIN} + ${TS_VALID_BIN} → ${VALID_BIN}"
cat "${OWT_VALID_BIN}" "${TS_VALID_BIN}" > "${VALID_BIN}"

# ---- 统计 ----
TRAIN_TOKENS=$(stat --format=%s "${TRAIN_BIN}")
VALID_TOKENS=$(stat --format=%s "${VALID_BIN}")

echo ""
echo "========================================="
echo " 完成!"
echo "========================================="
echo " train.bin : $(du -h "${TRAIN_BIN}" | cut -f1)  (~$(( TRAIN_TOKENS / 2 )) tokens)"
echo " valid.bin : $(du -h "${VALID_BIN}" | cut -f1)  (~$(( VALID_TOKENS / 2 )) tokens)"
echo " 总计      : ~$(( (TRAIN_TOKENS + VALID_TOKENS) / 2 )) tokens"
