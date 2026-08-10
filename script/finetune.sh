#!/bin/bash
# End-to-end fine-tuning of the O-MAKE vision tower for skin disease
# classification. Run from the repository root:
#
#   bash script/finetune.sh                # all datasets
#   DATASETS='PAD' bash script/finetune.sh # just one
set -euo pipefail

MODEL=${MODEL:-'open_clip_vit_base_16'}
CHECKPOINT=${CHECKPOINT:-'checkpoints/O-MAKE_epoch_15.pt'}
GPU=${GPU:-0}
SEED=${SEED:-122}
EPOCHS=${EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-5e-5}
OUTPUT_ROOT=${OUTPUT_ROOT:-'logs/finetune'}
DATASETS=${DATASETS:-'PAD F17K SNU SD-128 Daffodil ISIC2018'}

# Number of classes per dataset, matching meta/downstream/<name>-LP.csv.
declare -A NUM_CLASSES=(
  [PAD]=6
  [F17K]=114
  [SNU]=134
  [SD-128]=128
  [Daffodil]=5
  [ISIC2018]=7
)

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

for dataset in ${DATASETS}; do
  csv_path="${REPO_ROOT}/meta/downstream/${dataset}-LP.csv"
  nb_classes=${NUM_CLASSES[$dataset]:-}
  if [ ! -f "$csv_path" ]; then
    echo "Metadata CSV not found: $csv_path"
    continue
  fi
  if [ -z "$nb_classes" ]; then
    echo "No class count registered for dataset: $dataset"
    continue
  fi
  echo "=== finetune | model=${MODEL} dataset=${dataset} classes=${nb_classes} ==="

  CUDA_VISIBLE_DEVICES=${GPU} python finetune/run_class_finetuning.py \
    --model "${MODEL}" \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --nb_classes "${nb_classes}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --update_freq 1 \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --epochs "${EPOCHS}" \
    --layer_decay 0.65 \
    --drop_path 0.2 \
    --weights \
    --weight_decay 0.05 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --monitor acc \
    --sin_pos_emb \
    --no_auto_resume \
    --imagenet_default_mean_and_std \
    --exp_name "O-MAKE FT - ${dataset}" \
    --output_dir "${REPO_ROOT}/${OUTPUT_ROOT}/${dataset}/" \
    --csv_path "${csv_path}" \
    --root_path "${REPO_ROOT}/" \
    --image_key 'image_path' \
    --seed "${SEED}"
done
