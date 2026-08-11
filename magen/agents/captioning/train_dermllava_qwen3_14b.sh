#!/bin/bash
# Fine-tunes the MAGEN captioning agent: Qwen3-14B + the DermFM-Zero vision tower,
# on Derm1M instruction data with the Top-5 disease prior in the prompt.
#
# Run from the LLaVA repository root. The vision tower (redlessone/DermFM-Zero) is
# not publicly available, so this command is provided for reference: it documents
# exactly how the released captions were produced, but cannot be re-run without it.

# Global batchsize: 16*4*2=128
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path Qwen/Qwen3-14B \
    --version qwen3 \
    --data_path data/Derm-LLaVA-S2-with-top5-diag.json \
    --image_folder /data/Derm1M-Instruct/ \
    --vision_tower hf-hub:redlessone/DermFM-Zero \
    --pretrain_mm_mlp_adapter checkpoints/Dermllava-Qwen3-14b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Dermllava-Qwen3-14b-w-top5-diag \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 64 \
    --lazy_preprocess True \
    --report_to wandb

# ---------------------------------------------------------------------------
# Recaptioning inference: run the fine-tuned agent over the step-1 prompts.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python llava/eval/model_vqa_batch.py \
    --max_new_tokens 128 \
    --batch-size 64 \
    --model-path checkpoints/Dermllava-Qwen3-14b-w-top5-diag/ \
    --conv-mode qwen3 \
    --question-file magen/work/1_captioning_input.jsonl \
    --image-folder '' \
    --answers-file magen/work/1_captioning_output.jsonl