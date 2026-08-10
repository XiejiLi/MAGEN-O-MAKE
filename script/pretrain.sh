#!/bin/bash
# O-MAKE pretraining on the MAGEN-augmented Derm1M corpus.
# Reproduces the released checkpoint (ViT-B-16, 15 epochs, batch 2048).
# Run from the repository root:  bash script/pretrain.sh
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_DATA=${TRAIN_DATA:-'data/pretrain/MAGEN_train.csv'}
VAL_DATA=${VAL_DATA:-'data/pretrain/MAGEN_valid.csv'}
LOGS=${LOGS:-'logs/pretrain'}

ARGS=(
    --train-data="${TRAIN_DATA}"
    --val-data="${VAL_DATA}"
    --csv-img-key filename
    --csv-caption-key truncated_caption
    --csv-label-key label
    --logs "${LOGS}"

    --model ViT-B-16
    --pretrained OPENAI                 # initialise from OpenAI CLIP weights
    --batch-size 2048
    --lr=1e-4
    --wd=0.1
    --warmup 1500
    --epochs=15
    --workers=32
    --local-loss                        # compute contrastive loss per GPU shard
    --grad-checkpointing
    --dataset-resampled
    --aug-cfg scale="(0.4, 1.0)" color_jitter="(0.32, 0.32, 0.32, 0.08)" color_jitter_prob=0.8 gray_scale_prob=0.2

    # ---- O-MAKE objective ----
    --MKCL                              # multi-knowledge contrastive learning: align the image against several text views at once
    --subcaptions                       # include the per-aspect subcaptions among those views
    --num_subcaptions 8                 # how many subcaption columns to read per row
    --use_disease_specific_weight       # weight each subcaption by its similarity to the sample's ontology caption
    --lambda_m 1.0                      # weight of the MKCL term
    --lambda_s 0.7                      # weight of the subcaption-local region alignment (SLRA) term
    --OHCL                              # ontology-guided hierarchical contrastive learning: grade negatives by disease-tree distance
    --OHCL_temp 0.07                    # softmax temperature turning tree distances into soft targets; lower = peakier
    --OHCL_beta 0.5                     # target = (1 - beta) * one-hot + beta * ontology soft label; 0 = plain contrastive
    --loss_type 'cross entropy'         # how the soft target is scored: 'cross entropy' or 'KL'

    # ---- zero-shot monitoring during training ----
    --zeroshot-frequency 1
    --zeroshot-eval3=data/downstream/F17K/MAKE_F17K.csv     # slot 3 -> F17K_DISEASE_113_CLASSES
    --zeroshot-eval9=data/downstream/SNU/MAKE_SNU.csv       # slot 9 -> SNU_134_CLASSNAMES
    --zeroshot-eval12=data/downstream/SD-128/MAKE_SD-128.csv  # slot 12 -> SD_128_CLASSNAMES

    --save-frequency 15
    --copy-codebase                     # snapshot src/ next to the checkpoints
    --report-to wandb
    --wandb-project-name O-MAKE
)

python src/main.py "${ARGS[@]}"
