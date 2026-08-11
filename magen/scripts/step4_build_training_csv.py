#!/usr/bin/env python3
"""Step 4 — turn verified captions into the O-MAKE pretraining schema.

This is where a caption becomes the multi-aspect record O-MAKE trains on:

  verification output ──> diagnosis + corrected caption
                     ├──> ontology_caption      (diagnosis placed on the Derm1M tree)
                     ├──> visual_concept_caption (findings matched from the concept list)
                     ├──> subcaption_1..N        (caption split by sentence)
                     ├──> sub_caption_mask       (which subcaptions are present)
                     └──> knowledge_masks        (which of the 3 knowledge views exist)

The output columns are exactly those of the released Derm1M-AgentAug dataset,
and are what `script/pretrain.sh` reads via --MKCL / --subcaptions.

Usage:
    python step4_build_training_csv.py \
        --recaptioned  work/2_recaptioned.csv \
        --verification work/3_verification_output.csv \
        --out          work/4_magen_training.csv
"""
import argparse
import json
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

NO_DIAGNOSIS = "no definitive diagnosis"
KNOWLEDGE_FIELDS = ["truncated_caption", "ontology_final", "visual_concept_caption"]


# --------------------------------------------------------------------------- #
# verification-agent output
# --------------------------------------------------------------------------- #
def parse_vl_output(cell, key):
    """The agent answers with a JSON object, sometimes fenced in ```json."""
    if pd.isna(cell):
        return None
    try:
        s = str(cell).replace("```json", "").replace("```", "").strip()
        return json.loads(s).get(key)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# ontology caption
# --------------------------------------------------------------------------- #
def get_disease_path(ontology: dict, disease_name: str) -> list:
    """Path of keys leading to `disease_name`, or [] when it is not in the tree.

    Node keys may list synonyms separated by commas, so each is checked.
    """
    stack = [(ontology, [])]
    while stack:
        current, path = stack.pop()
        for key, value in current.items():
            synonyms = [x.strip().lower() for x in key.split(",")]
            if disease_name.lower() in synonyms:
                return path + [key]
            if isinstance(value, dict):
                stack.append((value, path + [key]))
    return []


def build_ontology_caption(ontology, disease_name):
    if pd.isna(disease_name) or disease_name == "":
        return np.nan
    if str(disease_name).lower() == NO_DIAGNOSIS:
        return np.nan
    path = get_disease_path(ontology, str(disease_name))
    if not path:
        return f"This is a skin photo diagnosed as {disease_name}."
    # Keep the ancestors, then name the disease as given rather than by its
    # (possibly synonym-laden) node key.
    full_path = path[:-1] + [disease_name]
    return f"This is a skin photo diagnosed as {', '.join(full_path)}."


def extract_ontology_final(text):
    """The bare ontology path, with the sentence scaffolding stripped off."""
    if pd.isna(text):
        return ""
    text = str(text)
    if "This is a skin photo diagnosed as {" in text and "}" in text:
        match = re.search(r"This is a skin photo diagnosed as \{(.+?)\}", text)
        if match:
            return match.group(1).strip()
    if text.startswith("This is a skin photo diagnosed as "):
        text = text.replace("This is a skin photo diagnosed as ", "", 1)
    stripped = text.rstrip(". ")
    return stripped if stripped else np.nan


# --------------------------------------------------------------------------- #
# visual concepts
# --------------------------------------------------------------------------- #
def find_skin_concepts(caption, concepts_sorted):
    """Longest-first substring match, so 'erythematous plaque' wins over 'plaque'.

    Concepts are emitted longest-first. The original notebook joined a `set()`
    here, so the released CSV lists the same concepts in an arbitrary order;
    the set of concepts is identical, only the ordering is deterministic now.
    """
    if pd.isna(caption):
        return ""
    caption_lower = str(caption).lower()
    found = []
    for concept in concepts_sorted:
        if concept in caption_lower:
            if all(concept not in f or len(concept) > len(f) for f in found):
                found.append(concept)
    return ", ".join(found)


def visual_concept_caption(concepts):
    if concepts is None or concepts in ("", " "):
        return np.nan
    return f"This skin photo shows {concepts}."


# --------------------------------------------------------------------------- #
# subcaptions and masks
# --------------------------------------------------------------------------- #
def split_caption_into_subcaptions(caption: str, max_subcaptions: int) -> Tuple[List[str], List[int]]:
    if pd.isna(caption) or caption == "":
        return [""] * max_subcaptions, [0] * max_subcaptions
    sentences = [s.strip() for s in re.split(r"[.!?]+", str(caption)) if s.strip()]
    sentences = sentences[:max_subcaptions]
    subcaptions = sentences + [""] * (max_subcaptions - len(sentences))
    mask = [1 if s else 0 for s in sentences] + [0] * (max_subcaptions - len(sentences))
    return subcaptions, mask


def create_knowledge_mask(row, fields):
    mask = []
    for field in fields:
        value = row.get(field)
        mask.append(0 if pd.isna(value) or value is None or str(value).strip() == "" else 1)
    return mask


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recaptioned", default=config.RECAPTIONED_CSV,
                    help="step 2 output (drafts merged into the metadata)")
    ap.add_argument("--verification", nargs="*", default=[config.VERIFICATION_OUTPUT],
                    help="verification agent output shard(s); omit to skip Agent 3")
    ap.add_argument("--out", default=config.TRAINING_CSV)
    ap.add_argument("--ontology", default=config.ONTOLOGY_TREE)
    ap.add_argument("--concepts", default=config.SKIN_CONCEPTS)
    ap.add_argument("--num-subcaptions", type=int, default=config.NUM_SUBCAPTIONS)
    ap.add_argument("--vl-output-col", default="vl_output")
    args = ap.parse_args()

    config.ensure_work_dir()
    df = pd.read_csv(args.recaptioned)
    print(f"{len(df)} rows from {args.recaptioned}")

    # ---- apply the verification agent's corrections ---------------------------
    shards = [pd.read_csv(p) for p in args.verification if os.path.exists(p)]
    if shards:
        ver = pd.concat(shards, ignore_index=True)
        if args.vl_output_col not in ver.columns:
            sys.exit(f"'{args.vl_output_col}' not in the verification output; "
                     f"got {list(ver.columns)}")
        ver["_diagnosis"] = ver[args.vl_output_col].apply(parse_vl_output, key="diagnosis")
        ver["_caption"] = ver[args.vl_output_col].apply(
            parse_vl_output, key="corrected_caption_paragraph")
        ver = ver.dropna(subset=["filename"]).drop_duplicates("filename", keep="last")

        diag = df["filename"].map(dict(zip(ver["filename"], ver["_diagnosis"])))
        capt = df["filename"].map(dict(zip(ver["filename"], ver["_caption"])))
        # Rows the agent did not return are backfilled with the draft caption.
        df["LLM_output"] = diag
        corrected = capt.notna()
        df.loc[corrected, "caption"] = capt[corrected]
        df.loc[corrected, "truncated_caption"] = capt[corrected]
        print(f"verified: {int(corrected.sum())} captions corrected, "
              f"{int(diag.notna().sum())} diagnoses parsed")
    else:
        print("no verification output given -- building from the draft captions only")
        df["LLM_output"] = df.get("standardized_disease")

    if "truncated_caption" not in df.columns:
        df["truncated_caption"] = df["caption"]

    # ---- ontology view -------------------------------------------------------
    ontology = json.load(open(args.ontology))
    df["ontology_caption"] = df["LLM_output"].apply(
        lambda d: build_ontology_caption(ontology, d))
    df["ontology_final"] = df["ontology_caption"].apply(extract_ontology_final)
    print(f"ontology captions: {int(df['ontology_caption'].notna().sum())}")

    # ---- visual-concept view -------------------------------------------------
    concepts = sorted({l.strip() for l in open(args.concepts) if l.strip()},
                      key=len, reverse=True)
    df["skin_concept_standarized"] = df["truncated_caption"].apply(
        lambda c: find_skin_concepts(c, concepts))
    df["visual_concept_caption"] = df["skin_concept_standarized"].apply(visual_concept_caption)
    print(f"visual-concept captions: {int(df['visual_concept_caption'].notna().sum())} "
          f"(vocabulary: {len(concepts)} concepts)")

    # ---- subcaptions ---------------------------------------------------------
    n = args.num_subcaptions
    split = df["caption"].apply(lambda c: split_caption_into_subcaptions(c, n))
    for i in range(n):
        df[f"subcaption_{i + 1}"] = [s[0][i] if s[0][i] else np.nan for s in split]
    df["sub_caption_mask"] = [str(s[1]) for s in split]
    filled = np.mean([sum(s[1]) for s in split])
    print(f"subcaptions: {filled:.2f} of {n} filled on average")

    # ---- knowledge mask ------------------------------------------------------
    df["knowledge_masks"] = df.apply(
        lambda r: create_knowledge_mask(r, KNOWLEDGE_FIELDS), axis=1)

    # Ontology class id; -1 marks "not mapped", matching the released dataset.
    if "ontology_label" not in df.columns:
        df["ontology_label"] = -1

    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}  ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
