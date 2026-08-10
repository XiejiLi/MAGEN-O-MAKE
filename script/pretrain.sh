#!/bin/bash
# O-MAKE pretraining on the MAGEN-augmented Derm1M corpus.
# Reproduces the released checkpoint (ViT-B-16, 15 epochs, batch 2048).
# Run from the repository root:  bash script/pretrain.sh
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_DATA=${TRAIN_DATA:-'data/Derm1M-AgentAug/MAGEN_train.csv'}
VAL_DATA=${VAL_DATA:-'data/Derm1M-AgentAug/MAGEN_valid.csv'}
LOGS=${LOGS:-'logs/pretrain'}

python src/main.py \
    --train-data="${TRAIN_DATA}" \
    --val-data="${VAL_DATA}" \
    --csv-img-key filename \
    --csv-caption-key truncated_caption \
    --csv-label-key label \
    --logs "${LOGS}" \
    --model ViT-B-16 \
    --pretrained OPENAI \
    --batch-size 2048 \
    --lr=1e-4 \
    --wd=0.1 \
    --warmup 1500 \
    --epochs=15 \
    --workers=32 \
    --local-loss \
    --grad-checkpointing \
    --dataset-resampled \
    --aug-cfg scale="(0.4, 1.0)" color_jitter="(0.32, 0.32, 0.32, 0.08)" color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --MKCL \
    --subcaptions \
    --num_subcaptions 8 \
    --use_disease_specific_weight \
    --lambda_m 1.0 \
    --lambda_s 0.7 \
    --OHCL \
    --OHCL_temp 0.07 \
    --OHCL_beta 0.5 \
    --loss_type 'KL' \
    --zeroshot-frequency 1 \
    --zeroshot-eval3=data/F17K/MAKE_F17K.csv \
    --zeroshot-eval9=data/SNU/MAKE_SNU.csv \
    --zeroshot-eval12=data/SD-128/MAKE_SD-128.csv \
    --save-frequency 15 \
    --copy-codebase \
    --report-to wandb \
    --wandb-project-name O-MAKE
