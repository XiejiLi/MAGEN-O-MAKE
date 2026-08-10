import sys
import os

# Use the O-MAKE open_clip that lives in this repository's src/ directory, so the
# linear probe sees exactly the architecture the checkpoints were trained with.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'linear_probe'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

import timm
import torch
from torchvision import transforms
from open_clip import create_model_and_transforms

def get_norm_constants(which_img_norm: str = 'imagenet'):
    print('normalization method: ',which_img_norm)
    constants_zoo = {
        'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.228, 0.224, 0.225)},
        'openai_clip':{'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)},
        'uniform': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
    }

    constants = constants_zoo[which_img_norm]
    return constants.get('mean'), constants.get('std')

def get_eval_transforms(
        which_img_norm: str = 'imagenet',
        img_resize: int = 224,
        center_crop: bool = False
):
    r"""
    Gets the image transformation for normalizing images before feature extraction.

    Args:
        - which_img_norm (str): transformation type

    Return:
        - eval_transform (torchvision.Transform): PyTorch transformation function for images.
    """
    mean, std = get_norm_constants(which_img_norm)
    eval_trans = [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=std)]
    eval_transform = transforms.Compose(eval_trans)
    return eval_transform


def get_encoder(model_name, which_img_norm='imagenet'):
    """Build the frozen image encoder to probe.

    Supported names:
      open_clip_<arch>              e.g. open_clip_ViT-B-16 -- combine with
                                    --checkpoint to probe an O-MAKE checkpoint
      open_clip_hf-hub:<repo>       e.g. open_clip_hf-hub:redlessone/DermLIP_ViT-B-16
      vit-base-16                   timm OpenAI CLIP ViT-B/16 baseline
      dinov2                        timm DINOv2 ViT-L/14 baseline
    """
    print('loading model checkpoint')

    if model_name == 'vit-base-16':
        model = timm.create_model("hf_hub:timm/vit_base_patch16_clip_224.openai", pretrained=True)
    elif model_name == 'open_clip_vit_base_16':
        model, _, _ = create_model_and_transforms('ViT-B-16', pretrained='openai')
        model.eval()
    elif model_name.startswith('open_clip_'):
        model, _, _ = create_model_and_transforms(model_name.replace('open_clip_', ''))
        model.eval()
    elif model_name == 'dinov2':
        model = timm.create_model("vit_large_patch14_dinov2.lvd142m",
                                  num_classes=0,
                                  dynamic_img_size=True,
                                  pretrained=True)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))


    print(model)
    eval_transform = get_eval_transforms(
        which_img_norm=which_img_norm,
        img_resize=256,
        center_crop=True
    )

    return model, eval_transform
