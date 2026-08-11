#!/usr/bin/env python3
"""Prepare the Derm1M-AgentAug caption CSV for release.

Keeps the columns the pretraining code reads plus the provenance fields, and
rewrites the absolute training-server paths to be relative to the repository
root:

    /VL_Data/<rest>  ->  data/pretrain/images/<rest>

The sub-path is preserved rather than flattened: 403,563 rows share only
401,947 distinct basenames, so a flat images/ directory would drop files.

Usage:
    python script/build_pretrain_csv.py --source <raw csv> --out data/pretrain/MAGEN_train.csv
"""
import argparse
import os
import sys

import pandas as pd

# Read by open_clip_train/data.py (MultiPositiveCsvDataset) and the O-MAKE loss.
REQUIRED = [
    'filename',                 # --csv-img-key
    'truncated_caption',        # --csv-caption-key
    'ontology_caption',
    'visual_concept_caption',
    *[f'subcaption_{i}' for i in range(1, 9)],
    'sub_caption_mask',
    'knowledge_masks',
    'ontology_label',
]
# Not used for training, kept so others can analyse the corpus by origin.
PROVENANCE = ['source', 'source_type', 'agent_generated']

OLD_ROOT = '/VL_Data/'
NEW_ROOT = 'data/pretrain/images/'
# Internal column: where the image actually sits under --image-root. Consumed by
# build_pretrain_parquet.py and stripped from anything released.
SOURCE_PATH = '_source_path'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, help='raw MAGEN caption CSV')
    ap.add_argument('--out', required=True, help='where to write the release CSV')
    ap.add_argument('--image-root', default=None,
                    help='if given, verify a sample of images resolves under this prefix')
    ap.add_argument('--original-csv', nargs='+', required=True,
                    help='the pre-MAGEN Derm1M caption CSV(s); a row is marked '
                         'agent_generated when its caption differs from the original')
    args = ap.parse_args()

    df = pd.read_csv(args.source)
    print(f'read {len(df)} rows, {len(df.columns)} columns')

    # MAGEN rewrites only part of the corpus, so mark which captions it produced.
    original = {}
    for path in args.original_csv:
        o = pd.read_csv(path, usecols=['filename', 'caption'])
        original.update(dict(zip(o['filename'].astype(str), o['caption'].astype(str))))
    print(f'loaded {len(original)} original captions')

    unmatched = [f for f in df['filename'].astype(str) if f not in original]
    if unmatched:
        sys.exit(f'{len(unmatched)} rows have no original caption to compare against, '
                 f'e.g. {unmatched[0]!r}; pass the right --original-csv')
    df['agent_generated'] = [
        str(cap).strip() != original[str(fn)].strip()
        for fn, cap in zip(df['filename'], df['caption'])
    ]
    n = int(df['agent_generated'].sum())
    print(f'agent_generated: {n} True / {len(df) - n} False ({100 * n / len(df):.1f}% rewritten)')

    keep = REQUIRED + PROVENANCE
    missing = [c for c in keep if c not in df.columns]
    if missing:
        sys.exit(f'source CSV is missing required columns: {missing}')
    dropped = [c for c in df.columns if c not in keep]
    df = df[keep]
    print(f'dropped {len(dropped)}: {dropped}')

    bad = ~df['filename'].astype(str).str.startswith(OLD_ROOT)
    if bad.any():
        sys.exit(f'{bad.sum()} paths do not start with {OLD_ROOT}, e.g. '
                 f'{df.loc[bad, "filename"].iloc[0]!r}')

    # Original location, relative to --image-root, kept for the parquet builder.
    df[SOURCE_PATH] = df['filename'].astype(str).str.slice(len(OLD_ROOT))

    # Released layout: one directory per corpus, named by the `source` column
    # rather than the internal working directory names. Verified collision-free.
    df['filename'] = (NEW_ROOT + df['source'].astype(str) + '/'
                      + df[SOURCE_PATH].map(os.path.basename))
    dupes = df['filename'].duplicated().sum()
    if dupes:
        sys.exit(f'{dupes} duplicate filenames after normalisation; refusing to '
                 f'write a CSV whose images would overwrite each other')

    if args.image_root:
        import random
        random.seed(0)
        sample = random.sample(df[SOURCE_PATH].tolist(), min(300, len(df)))
        missing_imgs = [p for p in sample
                        if not os.path.exists(os.path.join(args.image_root, p))]
        print(f'sampled {len(sample)} images under {args.image_root}: '
              f'{len(missing_imgs)} missing')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nwrote {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB, '
          f'{len(df)} rows, {len(df.columns)} columns incl. {SOURCE_PATH})')
    print(f'example filename: {df["filename"].iloc[0]}')


if __name__ == '__main__':
    main()
