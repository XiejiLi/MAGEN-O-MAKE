#!/bin/bash
# Zero-shot disease classification across the eight benchmarks.
# Run from the repository root:  bash script/zeroshot_eval.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# By default the weights are pulled straight from the Hub. To score a local
# checkpoint instead:  MODEL='ViT-B-16' CHECKPOINT=checkpoints/O-MAKE_epoch_15.pt
MODEL=${MODEL:-'hf-hub:Xieji-Li/MAGEN-O-MAKE'}
CHECKPOINT=${CHECKPOINT-''}   # note: '-' not ':-', so CHECKPOINT='' means "no local checkpoint"
GPU=${GPU:-0}

args=()
[ -n "${CHECKPOINT}" ] && args+=(--resume "${CHECKPOINT}")

CUDA_VISIBLE_DEVICES=${GPU} python src/test.py \
    --model "${MODEL}" \
    "${args[@]+"${args[@]}"}" \
    --batch-size 512 \
    --workers 8 \
    --csv-img-key filename \
    --csv-label-key label \
    --eval-pad       data/downstream/PAD/MAKE_PAD.csv \
    --eval-f17k      data/downstream/F17K/MAKE_F17K.csv \
    --eval-snu       data/downstream/SNU/MAKE_SNU.csv \
    --eval-sd128     data/downstream/SD-128/MAKE_SD-128.csv \
    --eval-daffodil  data/downstream/Daffodil/MAKE_Daffodil.csv \
    --eval-sd-tails  data/downstream/SD-198/SD-tails-70.csv \
    --eval-snu-tails data/downstream/SNU/MAKE_SNU_tails.csv
