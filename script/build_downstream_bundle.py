#!/usr/bin/env python3
"""Repackage the downstream benchmark datasets for release.

Copies only the images referenced by the metadata CSVs into the layout the
repository expects, rewriting every path to be relative to the repository root:

    data/downstream/<dataset>/images/<file>
    data/downstream/<dataset>/<metadata>.csv

Usage:
    python script/build_downstream_bundle.py --source <old data dir> --out <staging dir>
    python script/build_downstream_bundle.py ... --tar data/downstream.tar.gz
"""
import argparse
import csv
import os
import shutil
import sys
import tarfile

import pandas as pd

# dataset -> metadata CSVs to ship with it. The first path column of each CSV is
# rewritten; every image it references is copied.
DATASETS = {
    'PAD':       ['MAKE_PAD.csv'],
    'F17K':      ['MAKE_F17K.csv'],
    'SNU':       ['MAKE_SNU.csv', 'MAKE_SNU_tails.csv', 'MAKE_SNU_LP.csv'],
    'SD-128':    ['MAKE_SD-128.csv', 'MAKE_SD-128_LP.csv'],
    'SD-198':    ['SD-tails-70.csv'],
    'Daffodil':  ['MAKE_Daffodil.csv'],
    'ISIC2018':  ['meta_v2.csv'],
    'skin_cap':  ['skin_cap_meta.csv'],
}

PATH_COLUMNS = ('filename', 'image_path')


def rewrite(path, dataset):
    """data/<ds>/images/x.png (or an absolute path) -> data/downstream/<ds>/images/x.png"""
    return f'data/downstream/{dataset}/images/{os.path.basename(str(path))}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, help='existing data/ directory to read from')
    ap.add_argument('--out', required=True, help='staging directory to build into')
    ap.add_argument('--tar', default=None, help='also write a .tar.gz archive here')
    ap.add_argument('--dry-run', action='store_true', help='report sizes without copying')
    args = ap.parse_args()

    root = os.path.join(args.out, 'data', 'downstream')
    total_bytes = total_images = 0
    missing = []

    for dataset, csv_names in DATASETS.items():
        wanted = {}   # source path -> destination path
        for csv_name in csv_names:
            src_csv = os.path.join(args.source, dataset, csv_name)
            if not os.path.exists(src_csv):
                sys.exit(f'metadata CSV not found: {src_csv}')
            df = pd.read_csv(src_csv)

            path_cols = [c for c in PATH_COLUMNS if c in df.columns]
            if not path_cols:
                sys.exit(f'{src_csv} has no path column (looked for {PATH_COLUMNS})')

            for col in path_cols:
                for original in df[col].astype(str):
                    src_img = os.path.join(args.source, dataset, 'images',
                                           os.path.basename(original))
                    wanted[src_img] = os.path.join(root, dataset, 'images',
                                                   os.path.basename(original))
                df[col] = df[col].map(lambda p: rewrite(p, dataset))

            if not args.dry_run:
                os.makedirs(os.path.join(root, dataset), exist_ok=True)
                df.to_csv(os.path.join(root, dataset, csv_name), index=False,
                          quoting=csv.QUOTE_MINIMAL)

        copied = 0
        for src_img, dst_img in wanted.items():
            if not os.path.exists(src_img):
                missing.append(src_img)
                continue
            total_bytes += os.path.getsize(src_img)
            copied += 1
            if not args.dry_run and not os.path.exists(dst_img):
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                shutil.copy2(src_img, dst_img)
        total_images += copied
        print(f'{dataset:12s} {copied:6d} images  ({len(csv_names)} metadata CSV)')

    print(f'\ntotal: {total_images} images, {total_bytes / 1e9:.2f} GB')
    if missing:
        print(f'MISSING {len(missing)} images, e.g. {missing[:3]}')

    if args.tar and not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.tar)), exist_ok=True)
        print(f'\nwriting {args.tar} ...')
        with tarfile.open(args.tar, 'w:gz') as tf:
            tf.add(os.path.join(args.out, 'data'), arcname='data')
        print(f'archive: {os.path.getsize(args.tar) / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
