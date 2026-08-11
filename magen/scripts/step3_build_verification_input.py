#!/usr/bin/env python3
"""Step 3 — build the verification agent's prompt.

Combines the draft caption with (a) the Top-K disease priors kept above
`--prob-threshold` and (b) the matching disease knowledge cards from Agent 1,
producing one `mllm_prompt` per image. The verification agent then confirms or
corrects the diagnosis while preserving the morphology description.

Usage:
    python step3_build_verification_input.py \
        --recaptioned work/2_recaptioned.csv \
        --probs       work/derm1m_zs_probs.csv \
        --labels      derm1m_classnames.txt \
        --cards       work/disease_cards.csv \
        --out         work/3_verification_input.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from step1_build_captioning_input import load_label_names

# Kept verbatim from the paper run; changing the wording changes the outputs.
PROMPT_TEMPLATE = """
You are a multimodal dermatology revision agent.

Caption:
{caption}

Top-5 Model Predictions (with probabilities):
{top_preds}

Candidate DiseaseCards:
{candidates_text}

Task:
Verify and, if necessary, correct the diagnosis in the caption so that it matches the best-fitting DiseaseCard.

Important principle:
The morphology description in the caption must be preserved as much as possible.
Only adjust the diagnosis if it clearly conflicts with the DiseaseCards or visual evidence.

Procedure (follow in order):

1) Identify dermatologic morphology and anatomical site from the image and caption.

Examples include papule, nodule, plaque, pustule, scale, crust, ulcer, pigmentation, telangiectasia, etc.
Visual morphology and site are the PRIMARY evidence.

2) Treat the Top-5 prediction list as a probabilistic prior.
Higher probability diseases should be considered first, but they must NOT override morphology or site evidence.

3) Compare the extracted findings with each DiseaseCard using POS, MINSET, and SITES.
MINSET features are the most important criteria.

4) Select the diagnosis with the highest overall consistency based on:
(a) morphology and site
(b) POS/MINSET/SITES from the DiseaseCard
(c) support from the Top-5 probability list.

5) If multiple diagnoses remain plausible, prefer the one with higher probability.

6) If no DiseaseCard clearly fits or confidence is low, keep the original diagnosis.

7) Revise the caption only if necessary:
- Preserve morphology description
- Preserve anatomical site
- Only update the disease name
- Do not add new facts

Output JSON ONLY:

{{
"diagnosis": "<verified_or_corrected_diagnosis>",
"corrected_caption_paragraph": "<single paragraph of 3-5 sentences preserving morphology description>"
}}
"""


def prepare_card_lookup(cards):
    """Normalised disease-name variant -> (original name, card text)."""
    lookup = {}
    for _, row in cards.iterrows():
        name, card = str(row["disease_name"]), str(row["disease_card"])
        for variant in (v.strip().lower() for v in name.split(",") if v.strip()):
            lookup.setdefault(variant, (name, card))
    return lookup


def get_disease_card(name, lookup):
    if pd.isna(name):
        return None
    key = str(name).lower().strip()
    if not key:
        return None
    if key in lookup:
        return lookup[key]
    for k, v in lookup.items():          # partial-match fallback
        if k in key or key in k:
            return v
    return None


def build_prompt(row, lookup, label_names):
    parts = []
    idxs = row.get("top_disease_indices") or []
    if not idxs:
        parts.append("[No DiseaseCard candidates after probability filtering]")
    else:
        for i, cidx in enumerate(idxs, 1):
            dname = label_names.get(int(cidx), f"unknown_class_{cidx}")
            parts.append(f"--- CARD {i} ---")
            card = get_disease_card(dname, lookup)
            parts.append(card[1] if card else f"NAME: {dname}\n[Card not available]")
    return PROMPT_TEMPLATE.format(caption=row.get("caption", ""),
                                  top_preds=row.get("top_diseases", ""),
                                  candidates_text="\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recaptioned", default=config.RECAPTIONED_CSV)
    ap.add_argument("--probs", default=config.ZS_PROBS_CSV)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cards", default=config.DISEASE_CARDS,
                    help="Agent 1 output: disease_name, disease_card")
    ap.add_argument("--out", default=config.VERIFICATION_INPUT)
    ap.add_argument("--top-k", type=int, default=config.TOP_K_DISEASES)
    ap.add_argument("--prob-threshold", type=float, default=0.3,
                    help="only show cards whose prior probability reaches this")
    args = ap.parse_args()

    config.ensure_work_dir()
    df = pd.read_csv(args.recaptioned)
    label_names = load_label_names(args.labels)

    probs = pd.read_csv(args.probs)
    if "filename" not in probs.columns:
        probs["filename"] = probs["image_path"].astype(str)

    prob_cols = [c for c in probs.columns if c.startswith("probability_class_")]
    P = probs[prob_cols].to_numpy(dtype=float)
    top = np.argsort(-P, axis=1)[:, :args.top_k]
    idx_list, name_list = [], []
    for r, idxs in enumerate(top):
        valid = [int(j) for j in idxs if P[r, j] >= args.prob_threshold]
        idx_list.append(valid)
        name_list.append(
            " | ".join(f"{label_names.get(j, f'class_{j}')} ({P[r, j]:.4f})" for j in valid)
            if valid else "None above threshold")
    probs = probs.assign(top_disease_indices=idx_list, top_diseases=name_list)

    df = df.merge(probs[["filename", "top_disease_indices", "top_diseases"]],
                  on="filename", how="left")

    lookup = prepare_card_lookup(pd.read_csv(args.cards))
    df["mllm_prompt"] = df.apply(lambda r: build_prompt(r, lookup, label_names), axis=1)

    df.to_csv(args.out, index=False)
    n_cards = int(df["top_disease_indices"].apply(lambda x: bool(x) if isinstance(x, list) else False).sum())
    print(f"{len(df)} prompts -> {args.out}")
    print(f"with at least one DiseaseCard above {args.prob_threshold}: {n_cards}")


if __name__ == "__main__":
    main()
