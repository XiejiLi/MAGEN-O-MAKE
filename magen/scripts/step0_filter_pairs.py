#!/usr/bin/env python3
"""Step 0 — find the Derm1M pairs whose caption does not match the image.

Scores every image-text pair with a dermatology CLIP model and keeps the ones
below `SIMILARITY_THRESHOLD`. Those are the pairs MAGEN recaptions; well-matched
pairs keep their original text, which is why only part of the released corpus is
marked `agent_generated`.

Usage:
    python step0_filter_pairs.py [--batch-size 256] [--threshold 0.7]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class PairDataset(Dataset):
    def __init__(self, df, preprocess, image_root):
        self.paths = df["filename"].astype(str).tolist()
        self.captions = df["caption"].astype(str).tolist()
        self.preprocess = preprocess
        self.image_root = image_root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.image_root, self.paths[idx])
        try:
            image = self.preprocess(Image.open(path).convert("RGB"))
            ok = True
        except (OSError, ValueError):
            # Unreadable image: emit a blank tensor and mark it, so one bad file
            # does not abort a run over hundreds of thousands of pairs.
            image = torch.zeros(3, 224, 224)
            ok = False
        return image, self.captions[idx], ok, idx


def collate(batch):
    images, captions, oks, idxs = zip(*batch)
    return torch.stack(images), list(captions), list(oks), list(idxs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derm1m-csv", default=config.DERM1M_CSV)
    ap.add_argument("--image-root", default=config.IMAGE_ROOT)
    ap.add_argument("--out", default=config.LOW_QUALITY_CSV)
    ap.add_argument("--model", default=config.FILTER_MODEL)
    ap.add_argument("--threshold", type=float, default=config.SIMILARITY_THRESHOLD)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scores-out", default=None,
                    help="also write every pair with its similarity score")
    args = ap.parse_args()

    import open_clip

    config.ensure_work_dir()
    df = pd.read_csv(args.derm1m_csv)
    for col in ("filename", "caption"):
        if col not in df.columns:
            sys.exit(f"{args.derm1m_csv} needs a '{col}' column; got {list(df.columns)}")
    print(f"{len(df)} pairs from {args.derm1m_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval().to(device)

    loader = DataLoader(PairDataset(df, preprocess, args.image_root),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    scores = np.full(len(df), np.nan, dtype=np.float32)
    with torch.no_grad():
        for images, captions, oks, idxs in tqdm(loader, desc="scoring"):
            images = images.to(device)
            text = tokenizer(captions).to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # cosine similarity of each image with *its own* caption
            sim = (image_features * text_features).sum(dim=-1).float().cpu().numpy()
            for s, ok, i in zip(sim, oks, idxs):
                if ok:
                    scores[i] = s

    df["similarity_score"] = scores
    unreadable = int(np.isnan(scores).sum())
    low = df[df["similarity_score"] < args.threshold]
    print(f"\nunreadable images   : {unreadable}")
    print(f"below {args.threshold}          : {len(low)} / {len(df)} "
          f"({100 * len(low) / len(df):.1f}%) -> recaption these")

    if args.scores_out:
        df.to_csv(args.scores_out, index=False)
        print(f"all scores  -> {args.scores_out}")
    low.to_csv(args.out, index=False)
    print(f"low quality -> {args.out}")


if __name__ == "__main__":
    main()
