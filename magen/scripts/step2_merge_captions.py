#!/usr/bin/env python3
"""Step 2 — fold the captioning agent's drafts back into the metadata table.

The agent answers by `question_id`, so this maps those ids back to filenames via
the step-1 input and overwrites `caption` / `truncated_caption` for the rows it
rewrote. Rows the agent did not touch keep their original Derm1M text.

Usage:
    python step2_merge_captions.py \
        --pairs   work/0_low_quality_pairs.csv \
        --input   work/1_captioning_input.jsonl \
        --output  work/1_captioning_output.jsonl \
        --out     work/2_recaptioned.csv
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def read_jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=config.LOW_QUALITY_CSV)
    ap.add_argument("--input", default=config.CAPTIONING_INPUT,
                    help="step 1 output; provides question_id -> filename")
    ap.add_argument("--output", nargs="+", default=[config.CAPTIONING_OUTPUT],
                    help="captioning agent output shard(s)")
    ap.add_argument("--out", default=config.RECAPTIONED_CSV)
    args = ap.parse_args()

    config.ensure_work_dir()
    qid_to_file = {r["question_id"]: r["image"] for r in read_jsonl(args.input)}
    print(f"{len(qid_to_file)} question_id -> filename mappings")

    new_caption, orphans = {}, 0
    for shard in args.output:
        for r in read_jsonl(shard):
            qid = r["question_id"]
            if qid in qid_to_file:
                new_caption[qid_to_file[qid]] = r["text"]
            else:
                orphans += 1
    print(f"{len(new_caption)} drafts parsed" + (f", {orphans} with unknown question_id" if orphans else ""))

    df = pd.read_csv(args.pairs)
    if "truncated_caption" not in df.columns:
        df["truncated_caption"] = df["caption"]

    drafts = df["filename"].astype(str).map(new_caption)
    mask = drafts.notna()
    df.loc[mask, "caption"] = drafts[mask]
    df.loc[mask, "truncated_caption"] = drafts[mask]
    # Records which rows the agent actually rewrote; carried through to the
    # released dataset's `agent_generated` column.
    df["agent_generated"] = mask

    print(f"rewritten: {int(mask.sum())} / {len(df)}")
    missing = len(df) - int(mask.sum())
    if missing:
        print(f"  {missing} rows had no agent output and keep their original caption")

    df.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
