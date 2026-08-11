#!/bin/bash
# MAGEN Agent 2 — Captioning Agent (image + Top-5 disease prior -> caption).
#
# This agent is a fine-tuned DermLLaVA (Qwen3-14B backbone, trained "w-top5-diag")
# run through LLaVA's batch VQA inference. It must run inside the LLaVA repo
# (checkpoint + llava/ package), so this is a launcher, not a standalone script.
#
#   model : checkpoints/Dermllava-Qwen3-14b-w-top5-diag/   (conv-mode qwen3)
#   input : 1_captioning_input.jsonl   (step 1 output; {question_id,image,text})
#           `text` = "Top 5 possible diagnoses: <top5>\n\nAnalyze the image ..."
#   output: 1_captioning_output.jsonl  ({question_id, prompt, text, ...})
#   core  : llava/eval/model_vqa_batch.py   (a copy sits in captioning/)
#
# Usage (from the LLaVA repo root):
#   bash 2_captioning_agent.sh <captioning_input.jsonl> <captioning_output.jsonl> [image_folder]

set -e
LLAVA_ROOT="${LLAVA_ROOT:?set LLAVA_ROOT to your LLaVA checkout}"
CKPT="${CKPT:-checkpoints/Dermllava-Qwen3-14b-w-top5-diag/}"
QUESTION_FILE="${1:?usage: 2_captioning_agent.sh <input.jsonl> <output.jsonl> [image_folder]}"
ANSWERS_FILE="${2:?missing output.jsonl}"
IMAGE_FOLDER="${3:-}"

cd "$LLAVA_ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python llava/eval/model_vqa_batch.py \
    --max_new_tokens 128 \
    --batch-size 1 \
    --model-path "$CKPT" \
    --conv-mode qwen3 \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$ANSWERS_FILE"
