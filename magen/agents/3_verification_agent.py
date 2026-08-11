#!/usr/bin/env python
"""MAGEN Agent 3 — Verification Agent.

Verifies / revises each captioning-agent caption against the Top-5 disease priors
and matching DiseaseCards, emitting a JSON `vl_output`
({"diagnosis": ..., "corrected_caption_paragraph": ...}) per image.

  input : verification_agent_input CSV  (needs `filename` [image path] + `mllm_prompt`)
  output: verification_agent_output CSV (adds `vl_output`)
  model : Qwen/Qwen2.5-VL-72B-Instruct

I/O and model are CLI args; incremental save + resume are preserved.
Env: transformers==4.49.0, torch, accelerate, qwen_vl_utils.

Usage:
  python 3_verification_agent.py \
      --input  ../work/3_verification_input.csv \
      --output ../work/3_verification_output.csv
"""
import os
import argparse
import pandas as pd
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with `filename` + `mllm_prompt`")
    ap.add_argument("--output", required=True, help="CSV to write, adds `vl_output`")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--cache-dir", default=None, help="HF cache dir for the weights")
    ap.add_argument("--image-root", default="", help="prefix prepended to each `filename`")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # resume: skip already-processed rows
    if os.path.exists(args.output):
        done = pd.read_csv(args.output)
        df = df[~df.index.isin(set(done.index))]
        print(f"[resume] {len(done)} done; {len(df)} remaining")

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"[load] {args.model}")
    kw = dict(torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
    if args.cache_dir:
        kw["cache_dir"] = args.cache_dir
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kw)
    processor = AutoProcessor.from_pretrained(args.model)

    for start in tqdm(range(0, len(df), args.batch_size), desc="verify"):
        chunk = df.iloc[start:start + args.batch_size]
        batch_msgs, idxs = [], []
        for idx, row in chunk.iterrows():
            batch_msgs.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{args.image_root}{row['filename']}"},
                    {"type": "text", "text": row["mllm_prompt"]},
                ],
            }])
            idxs.append(idx)
        try:
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in batch_msgs]
            img_in, vid_in = process_vision_info(batch_msgs)
            inputs = processor(text=texts, images=img_in, videos=vid_in,
                               padding=True, return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
            resp = processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
            for idx, r in zip(idxs, resp):
                df.loc[idx, "vl_output"] = r
            save = df.loc[idxs].copy()
            append = os.path.exists(args.output) and start > 0
            save.to_csv(args.output, mode="a" if append else "w", header=not append, index=True)
        except Exception as e:
            print(f"[err] batch @ {start}: {e}")
            continue
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
