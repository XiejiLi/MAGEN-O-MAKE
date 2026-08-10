#!/bin/bash
# Cross-modal retrieval (image <-> text) on SkinCAP.
# Run from the repository root:  bash script/retrieval_eval.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT=${CHECKPOINT:-'checkpoints/O-MAKE_epoch_15.pt'}
GPU=${GPU:-0}
DATA=${DATA:-'data/downstream/skin_cap/skin_cap_meta.csv'}
CAPTION_KEY=${CAPTION_KEY:-'caption_zh_polish_en'}

CUDA_VISIBLE_DEVICES=${GPU} python src/retrieval.py \
    --model 'ViT-B-16' \
    --resume "${CHECKPOINT}" \
    --batch-size 256 \
    --workers 8 \
    --retrieval-data "${DATA}" \
    --csv-img-key filename \
    --csv-caption-key "${CAPTION_KEY}"
