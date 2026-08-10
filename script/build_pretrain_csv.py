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
PROVENANCE = ['source', 'source_type']

OLD_ROOT = '/VL_Data/'
NEW_ROOT = 'data/pretrain/images/'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, help='raw MAGEN caption CSV')
    ap.add_argument('--out', required=True, help='where to write the release CSV')
    ap.add_argument('--image-root', default=None,
                    help='if given, verify a sample of images resolves under this prefix')
    args = ap.parse_args()

    df = pd.read_csv(args.source)
    print(f'read {len(df)} rows, {len(df.columns)} columns')

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
    df['filename'] = df['filename'].astype(str).str.replace(OLD_ROOT, NEW_ROOT, n=1, regex=False)

    if args.image_root:
        import random
        random.seed(0)
        sample = random.sample(df['filename'].tolist(), min(300, len(df)))
        missing_imgs = [p for p in sample
                        if not os.path.exists(os.path.join(
                            args.image_root, p[len(NEW_ROOT):]))]
        print(f'sampled {len(sample)} images under {args.image_root}: '
              f'{len(missing_imgs)} missing')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nwrote {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB, '
          f'{len(df)} rows, {len(df.columns)} columns)')
    print(f'example filename: {df["filename"].iloc[0]}')


if __name__ == '__main__':
    main()
