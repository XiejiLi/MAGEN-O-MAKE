#!/usr/bin/env python3
"""Step 1 — build the captioning agent's input, grounded in Top-K disease priors.

Takes zero-shot disease probabilities for each image, renormalises them after
dropping the "no definitive diagnosis" class, and prepends the Top-K candidates
to the captioning prompt. Grounding the agent in a shortlist is what keeps it
from inventing diagnoses.

Usage:
    python step1_build_captioning_input.py \
        --pairs work/0_low_quality_pairs.csv \
        --probs work/derm1m_zs_probs.csv \
        --out   work/1_captioning_input.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

PROMPT = ("Describe this skin lesion image in a single paragraph of 3-5 sentences. "
          "Describe the morphology, colour, distribution and anatomical site.")

# Dropped before renormalising: it is a placeholder class, not a diagnosis.
NO_DIAGNOSIS_CLASS = "no definitive diagnosis"


def load_label_names(path):
    """Class index -> disease name, from a one-name-per-line file or a JSON list."""
    if path.endswith(".json"):
        names = json.load(open(path))
    else:
        names = [l.strip() for l in open(path) if l.strip()]
    return {i: n for i, n in enumerate(names)}


def top_k_diseases(probs_df, label_names, k, drop_class=None):
    prob_cols = [c for c in probs_df.columns if c.startswith("probability_class_")]
    if not prob_cols:
        sys.exit("no probability_class_* columns found in the probabilities CSV")

    matrix = probs_df[prob_cols].astype(float).values
    if drop_class is not None and 0 <= drop_class < matrix.shape[1]:
        matrix[:, drop_class] = 0.0
    # Renormalise so the shortlist reflects the remaining classes only.
    sums = matrix.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    matrix = matrix / sums

    top = np.argsort(matrix, axis=1)[:, -k:][:, ::-1]
    return [", ".join(label_names.get(i, f"class_{i}") for i in row) for row in top]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=config.LOW_QUALITY_CSV,
                    help="pairs to recaption (step 0 output)")
    ap.add_argument("--probs", default=config.ZS_PROBS_CSV,
                    help="zero-shot disease probabilities per image")
    ap.add_argument("--labels", required=True,
                    help="class-index -> disease-name list (.txt or .json)")
    ap.add_argument("--out", default=config.CAPTIONING_INPUT)
    ap.add_argument("--top-k", type=int, default=config.TOP_K_DISEASES)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    config.ensure_work_dir()
    pairs = pd.read_csv(args.pairs)
    probs = pd.read_csv(args.probs)
    label_names = load_label_names(args.labels)

    drop_idx = next((i for i, n in label_names.items()
                     if n.strip().lower() == NO_DIAGNOSIS_CLASS), None)
    if drop_idx is not None:
        print(f"zeroing placeholder class {drop_idx} ({NO_DIAGNOSIS_CLASS!r}) before renormalising")

    probs["top_diseases"] = top_k_diseases(probs, label_names, args.top_k, drop_idx)
    prior = dict(zip(probs["image_path"].astype(str), probs["top_diseases"]))

    written = matched = 0
    with open(args.out, "w") as fh:
        for i, row in enumerate(pairs.itertuples(index=False)):
            filename = str(getattr(row, "filename"))
            text = args.prompt
            if filename in prior:
                matched += 1
                text = f"Top {args.top_k} possible diagnoses: {prior[filename]}\n\n{text}"
            fh.write(json.dumps({
                "question_id": i,
                "image": filename,
                "text": text,
                "category": "recaption",
            }) + "\n")
            written += 1

    print(f"{written} prompts -> {args.out}")
    print(f"with disease priors: {matched} ({100 * matched / max(written, 1):.1f}%)")
    if matched < written:
        print("  (rows without a prior fall back to the bare captioning prompt)")


if __name__ == "__main__":
    main()
