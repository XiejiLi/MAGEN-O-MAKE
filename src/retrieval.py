"""Cross-modal retrieval evaluation for O-MAKE.

Encodes every image and its paired caption, then ranks each image against all
captions (and vice versa) and reports Recall@K plus mean/median rank.

Example
-------
    python src/retrieval.py \
        --model ViT-B-16 \
        --resume checkpoints/O-MAKE_epoch_15.pt \
        --batch-size 256 \
        --retrieval-data data/downstream/SkinCAP/skincap_retrieval.csv \
        --csv-img-key filename \
        --csv-caption-key caption

The CSV needs one row per image-caption pair: an image path relative to the
repository root, and its caption.
"""

import warnings

warnings.filterwarnings("ignore")

import logging
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from open_clip import create_model_and_transforms, get_input_dtype, get_tokenizer
from open_clip_train.distributed import init_distributed_device
from open_clip_train.params import parse_args
from open_clip_train.precision import get_autocast
from test import load_checkpoint, random_seed

RECALL_KS = (1, 5, 10, 50, 100)


class RetrievalCsvDataset(Dataset):
    """One image-caption pair per row."""

    def __init__(self, input_filename, transforms, img_key, caption_key, tokenizer, parent_path=''):
        df = pd.read_csv(input_filename)
        for key in (img_key, caption_key):
            if key not in df.columns:
                raise SystemExit(
                    f"Column '{key}' not in {input_filename}; available: {list(df.columns)}")
        df = df[[img_key, caption_key]].dropna()
        self.images = df[img_key].astype(str).tolist()
        self.captions = df[caption_key].astype(str).tolist()
        self.transforms = transforms
        self.tokenize = tokenizer
        self.parent_path = parent_path
        logging.info('Loaded %d image-caption pairs from %s', len(self.images), input_filename)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.transforms(Image.open(self.parent_path + self.images[idx]).convert('RGB'))
        text = self.tokenize([self.captions[idx]])[0]
        return image, text


def retrieval_metrics(image_features, text_features):
    """Recall@K and rank statistics in both directions.

    Row i of image_features and row i of text_features are the ground-truth
    pair, so the correct match always lies on the diagonal.
    """
    metrics = {}
    logits_per_image = (image_features @ text_features.t()).detach().cpu()
    logits = {
        'image_to_text': logits_per_image,
        'text_to_image': logits_per_image.t(),
    }
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1].detach().cpu().numpy()
        for k in RECALL_KS:
            metrics[f'{name}_R@{k}'] = float(np.mean(preds < k))
        metrics[f'{name}_mean_rank'] = float(preds.mean() + 1)
        metrics[f'{name}_median_rank'] = float(np.floor(np.median(preds)) + 1)
    return metrics


def main(args):
    args = parse_args(args)

    if not args.retrieval_data:
        raise SystemExit('Pass --retrieval-data <csv> (see --help).')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        output_dict=True,
        cache_dir=args.cache_dir,
    )
    model.eval()

    if args.resume:
        load_checkpoint(model, args.resume, args.distributed)
    elif not args.pretrained:
        logging.warning('Neither --resume nor --pretrained was given: '
                        'evaluating a randomly initialised model.')

    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    dataset = RetrievalCsvDataset(
        args.retrieval_data,
        preprocess_val,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=tokenizer,
        parent_path=args.parent_path,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,          # row i of both feature matrices must stay a pair
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    all_image_features, all_text_features = [], []
    with torch.inference_mode():
        for images, texts in tqdm(dataloader, desc='encoding'):
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            with autocast():
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                text_features = model.encode_text(texts, normalize=True)
            all_image_features.append(image_features.float().cpu())
            all_text_features.append(text_features.float().cpu())

    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)
    logging.info('Ranking %d images against %d captions', len(image_features), len(text_features))

    metrics = retrieval_metrics(image_features, text_features)
    print()
    for k, v in metrics.items():
        print(f'{k} : {v:.4f}')


if __name__ == '__main__':
    main(sys.argv[1:])
