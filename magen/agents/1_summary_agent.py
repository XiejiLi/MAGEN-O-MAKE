#!/usr/bin/env python
"""MAGEN Agent 1 — Summary Agent (Disease-Card generation).

Compresses each disease's free-text knowledge into a concise, morphology-only
"DiseaseCard" (NAME / POS / SITES / MINSET) used by the Verification Agent.

  input : disease_knowledge_base.json  ({"diseases": {id: {name, description}}})
  output: disease_cards_output.csv      (columns: disease_name, description, disease_card)
  model : Qwen/Qwen2.5-72B-Instruct     (text LLM)

Cleaned from Qwen/Qwen2.5B/disease_summarize/diseas_summarize.py — paths/model
are now CLI args and generation is resumable.

Usage:
  python 1_summary_agent.py --kb disease_knowledge_base.json --out disease_cards_output.csv
"""
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a dermatologist. Compress the disease knowledge below into a concise "
    "DiseaseCard. Keep morphology only: shape, border, surface, color/pattern, "
    "distribution, dermoscopy, pathognomonic clues. No epidemiology or treatment. "
    "<=120 tokens. Use short noun phrases; semicolon-separated.\n"
    "NAME: <disease name>\n"
    "POS: <3-8 hallmark positive cues; short phrases>\n"
    "SITES: <key anatomical sites/patterns>\n"
    "MINSET: <2-4 minimal sufficient cues>"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="disease_knowledge_base.json",
                    help="disease knowledge base JSON ({'diseases': {id: {name, description}}})")
    ap.add_argument("--out", default="disease_cards_output.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    with open(args.kb) as f:
        data = json.load(f)
    df = pd.DataFrame([{"disease_name": d["name"], "description": d["description"]}
                       for d in data["diseases"].values()])

    # resume
    if os.path.exists(args.out):
        done = pd.read_csv(args.out)
        df = df[~df.index.isin(set(done.index))]
        print(f"[resume] {len(done)} already done; {len(df)} remaining")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    for start in tqdm(range(0, len(df), args.batch_size)):
        chunk = df.iloc[start:start + args.batch_size]
        prompts = []
        for _, row in chunk.iterrows():
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"NAME: {row['disease_name']}\n{row['description']}"}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, max_length=1024, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens)
        resp = tok.batch_decode([o[len(i):] for i, o in zip(enc.input_ids, out)],
                                skip_special_tokens=True)
        chunk = chunk.copy()
        chunk["disease_card"] = resp
        chunk.to_csv(args.out, mode="a" if start > 0 or os.path.exists(args.out) else "w",
                     header=not (start > 0 or os.path.exists(args.out)), index=True)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
