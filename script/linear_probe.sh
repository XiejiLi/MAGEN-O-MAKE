#!/bin/bash
# Linear probing benchmark: freeze the image encoder, fit logistic regression
# on the extracted features. Run from the repository root.
#
#   bash script/linear_probe.sh
#
# To probe a baseline instead of O-MAKE, set MODEL to an open_clip name and
# leave CHECKPOINT empty, e.g.
#   MODEL='open_clip_hf-hub:redlessone/DermLIP_ViT-B-16' CHECKPOINT='' bash script/linear_probe.sh
set -euo pipefail

MODEL=${MODEL:-'open_clip_ViT-B-16'}
CHECKPOINT=${CHECKPOINT-'checkpoints/O-MAKE_epoch_15.pt'}  # note: '-' not ':-', so CHECKPOINT='' means "no checkpoint"
PERCENT_DATA=${PERCENT_DATA:-0.5}
GPU=${GPU:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-'logs/linear_probe'}

DATASETS=${DATASETS:-'PAD F17K SNU SD-128 Daffodil'}

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

for dataset in ${DATASETS}; do
  csv_path="${REPO_ROOT}/meta/downstream/${dataset}-LP.csv"
  if [ ! -f "$csv_path" ]; then
    echo "Metadata CSV not found: $csv_path"
    continue
  fi
  echo "=== linear probe | model=${MODEL} dataset=${dataset} ==="

  args=(
    --batch_size 256
    --model "${MODEL}"
    --root_path "${REPO_ROOT}/"
    --csv_path "${csv_path}"
    --csv_filename "${MODEL}-${dataset}_results.csv"
    --output_dir "${REPO_ROOT}/${OUTPUT_ROOT}/${MODEL}-${dataset}"
    --image_key 'image_path'
    --label_key 'label'
    --percent_data "${PERCENT_DATA}"
  )
  if [ -n "${CHECKPOINT}" ]; then
    args+=(--checkpoint "${CHECKPOINT}")
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python linear_probe/linear_eval.py "${args[@]}"
done
