"""Zero-shot disease classification evaluation for O-MAKE.

Example
-------
    python src/test.py \
        --model ViT-B-16 \
        --resume checkpoints/O-MAKE_epoch_15.pt \
        --batch-size 512 \
        --csv-img-key filename --csv-label-key label \
        --eval-pad      data/PAD/MAKE_PAD.csv \
        --eval-f17k     data/F17K/MAKE_F17K.csv \
        --eval-snu      data/SNU/MAKE_SNU.csv \
        --eval-sd128    data/SD-128/MAKE_SD-128.csv

Run `python src/test.py --help` for the full list of `--eval-<dataset>` flags.
"""

import warnings

warnings.filterwarnings("ignore")

import logging
import random
import sys

import numpy as np
import torch

from open_clip import create_model_and_transforms, get_tokenizer
from open_clip_train.data import get_zeroshot_eval_data
from open_clip_train.distributed import init_distributed_device
from open_clip_train.file_utils import pt_load
from open_clip_train.params import parse_args
from open_clip_train.zero_shot_eval import zero_shot_eval, requested_tasks


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def load_checkpoint(model, path, distributed):
    """Load either a training checkpoint (epoch_*.pt) or a bare state_dict."""
    checkpoint = pt_load(path, map_location='cpu')
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if not distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    logging.info('Loaded checkpoint %s', path)


def main(args):
    args = parse_args(args)

    if not requested_tasks(args):
        raise SystemExit(
            'No benchmark selected. Pass at least one --eval-<dataset> flag, '
            'e.g. --eval-pad data/PAD/MAKE_PAD.csv (see --help).')

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
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
    )
    model.eval()

    random_seed(args.seed, args.rank)
    args.save_logs = None
    args.wandb = None

    if args.resume:
        load_checkpoint(model, args.resume, args.distributed)
    elif not args.pretrained:
        logging.warning('Neither --resume nor --pretrained was given: '
                        'evaluating a randomly initialised model.')

    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    data = get_zeroshot_eval_data(args, preprocess_val, tokenizer=tokenizer)

    metrics = zero_shot_eval(model, data, args, tokenizer=tokenizer)
    print()
    for k, v in metrics.items():
        print(f'{k} : {v:.4f}')


if __name__ == '__main__':
    main(sys.argv[1:])
