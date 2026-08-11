#!/usr/bin/env python3
"""Pack the Derm1M-AgentAug images and captions into parquet shards for Hugging Face.

Each row carries the image bytes alongside its MAGEN captions, so the dataset
previews in the HF viewer and `load_dataset` returns decoded images. The
`filename` column keeps the repository-relative path, which is what
`script/pretrain.sh` reads once the images are materialised on disk.

403,563 loose files would exceed the file count Hugging Face recommends per
repository, hence the packing.

Usage:
    python script/build_pretrain_parquet.py \
        --csv data/pretrain/MAGEN_train.csv \
        --image-root /path/to/derm1m/images \
        --out /path/to/parquet_shards
"""
import argparse
import os
import sys

import pandas as pd
from datasets import Dataset, Features, Image, Value

PATH_PREFIX = 'data/pretrain/images/'
SOURCE_PATH = '_source_path'   # written by build_pretrain_csv.py, never released


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True, help='release caption CSV')
    ap.add_argument('--image-root', required=True, help='directory holding the images')
    ap.add_argument('--out', required=True, help='directory to write parquet shards into')
    ap.add_argument('--shard-bytes', type=int, default=450 * 1024 * 1024,
                    help='approximate uncompressed image bytes per shard (default 450MB)')
    ap.add_argument('--split', default='train', help='split name used in the shard filenames')
    ap.add_argument('--limit', type=int, default=None, help='only process the first N rows')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    os.makedirs(args.out, exist_ok=True)
    print(f'{len(df)} rows, {len(df.columns)} columns -> {args.out}')

    if SOURCE_PATH not in df.columns:
        sys.exit(f'{args.csv} has no {SOURCE_PATH} column; rebuild it with '
                 f'script/build_pretrain_csv.py')
    # Everything except the image path itself and the internal locator column.
    text_cols = [c for c in df.columns if c not in ('filename', SOURCE_PATH)]
    # Declaring `image` as datasets.Image() writes HuggingFace feature metadata
    # into the parquet schema, which is what makes the dataset viewer render the
    # column as a picture and load_dataset return a decoded PIL image.
    features = Features({
        'image': Image(),
        'filename': Value('string'),
        **{c: Value('string') for c in text_cols},
    })

    rows, shard_bytes, shard_idx, total_rows, total_bytes = [], 0, 0, 0, 0

    def flush():
        nonlocal rows, shard_bytes, shard_idx, total_rows
        if not rows:
            return
        path = os.path.join(args.out, f'{args.split}-{shard_idx:05d}.parquet')
        Dataset.from_list(rows, features=features).to_parquet(path)
        print(f'  {os.path.basename(path)}  {len(rows):6d} rows  '
              f'{os.path.getsize(path) / 1e6:7.1f} MB', flush=True)
        total_rows += len(rows)
        rows, shard_bytes = [], 0
        shard_idx += 1

    for record in df.to_dict('records'):
        rel = str(record['filename'])                       # released path
        src = os.path.join(args.image_root, str(record[SOURCE_PATH]))   # on-disk path
        try:
            with open(src, 'rb') as fh:
                data = fh.read()
        except OSError as exc:
            sys.exit(f'cannot read {src}: {exc}')

        row = {'image': {'bytes': data, 'path': rel},
               'filename': rel}
        for c in text_cols:
            v = record[c]
            row[c] = None if pd.isna(v) else str(v)
        rows.append(row)
        shard_bytes += len(data)
        total_bytes += len(data)
        if shard_bytes >= args.shard_bytes:
            flush()
    flush()

    print(f'\n{total_rows} rows in {shard_idx} shards, '
          f'{total_bytes / 1e9:.2f} GB of images')


if __name__ == '__main__':
    main()
