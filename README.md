# MAGEN-O-MAKE

Multi-Aspect Knowledge-Enhanced Medical Vision-Language Pretraining with Multi-Agent Data Generation

<p align="center">
  <a href="https://arxiv.org/abs/2512.03445"><img src="https://img.shields.io/badge/arXiv-2512.03445-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/Xieji-Li/MAGEN-O-MAKE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-MAGEN--O--MAKE-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/datasets/Xieji-Li/Derm1M-AgentAug"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Derm1M--AgentAug-yellow" alt="Hugging Face Dataset"></a>
  <img src="https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg" alt="License">
</p>

## Abstract

We propose a novel medical VLP framework combining **MAGEN** (Multi-Agent data GENeration) and **O-MAKE** (Ontology-based Multi-Aspect Knowledge-Enhanced pretraining). MAGEN synthesizes knowledge-enriched image descriptions via a foundation model-assisted captioning and retrieval-based verification pipeline. O-MAKE decomposes long clinical texts into distinct knowledge aspects, enabling fine-grained alignment at both global and patch levels with ontology-guided modeling. Validated on dermatology, our approach achieves state-of-the-art zero-shot performance on disease classification and cross-modal retrieval across eight datasets.

<p align="center">
    <img src="assets/Overview.png" width="100%"> <br>
</p>

## Repository layout

```
├── src/                        O-MAKE pretraining + evaluation
├── linear_probe/               frozen-encoder linear probing
├── finetune/                   end-to-end fine-tuning
├── meta/downstream/            train/val/test splits for probing & fine-tuning
├── script/                     one command per experiment
└── data/                       images (downloaded separately)
    ├── pretrain/               Derm1M-AgentAug
    └── downstream/             zero-shot benchmark datasets
```

## Environment

```bash
conda create -n omake python=3.10
conda activate omake
git clone https://github.com/XiejiLi/MAGEN-O-MAKE.git
cd MAGEN-O-MAKE
pip install -r requirements.txt
```

## Pretrained model

| Model | Architecture | Weights |
|---|---|---|
| O-MAKE | CLIP ViT-B/16 | [🤗 Xieji-Li/MAGEN-O-MAKE](https://huggingface.co/Xieji-Li/MAGEN-O-MAKE) |

<details>
<summary><b>Quick start — zero-shot skin cancer classification</b></summary>

```python
import torch
from PIL import Image
import open_clip   # pip install open_clip_torch

model, preprocess = open_clip.create_model_from_pretrained('hf-hub:Xieji-Li/MAGEN-O-MAKE')
tokenizer = open_clip.get_tokenizer('hf-hub:Xieji-Li/MAGEN-O-MAKE')
model.eval()

# The six PAD-UFES-20 classes.
labels = ['nevus', 'basal cell carcinoma', 'actinic keratosis',
          'seborrheic keratosis', 'squamous cell carcinoma', 'melanoma']

# A confirmed melanoma from the PAD-UFES-20 test split.
image = preprocess(Image.open('data/downstream/PAD/images/PAT_611_1158_156.png').convert('RGB')).unsqueeze(0)
text = tokenizer([f'This is a skin image of {c}' for c in labels])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

for label, p in sorted(zip(labels, probs.tolist()), key=lambda kv: -kv[1]):
    print(f'{label:25s} {p:.4f}')
```

To reproduce the numbers below instead, download `O-MAKE_epoch_15.pt` from the same Hugging Face
repository into `checkpoints/` and use the scripts in `script/`:

```bash
mkdir -p checkpoints
hf download Xieji-Li/MAGEN-O-MAKE O-MAKE_epoch_15.pt --local-dir checkpoints
```

</details>

## Data preparation

Both parts unpack at the repository root; every image path in the metadata is relative to it, so run
all commands from there.

<details>
<summary><b>Training data — Derm1M-AgentAug (403,563 image-text pairs)</b></summary>

[🤗 Xieji-Li/Derm1M-AgentAug](https://huggingface.co/datasets/Xieji-Li/Derm1M-AgentAug) holds the
MAGEN-augmented captions together with their images.

> The images and the original captions come from **[Derm1M](https://github.com/SiyuanYan1/Derm1M)**.
> Derm1M-AgentAug is a derivative that adds MAGEN-generated captions and knowledge-aspect
> decompositions; MAGEN rewrote 186,069 of the 403,563 training captions (46.1%), and the
> `agent_generated` column marks which. Please cite Derm1M alongside this work.

```python
from datasets import load_dataset

ds = load_dataset('Xieji-Li/Derm1M-AgentAug', split='train')
```

`script/pretrain.sh` reads images from disk, so materialise them once at `data/pretrain/images/`
using each row's `filename`:

```python
import os
for row in ds:
    path = row['filename']            # data/pretrain/images/<source>/<file>
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row['image'].save(path)
```

| Column | Description |
|---|---|
| `filename` | image path relative to the repository root |
| `truncated_caption` | MAGEN caption, truncated to the text encoder's context length |
| `ontology_caption` | names the diagnosis along its Derm1M ontology path |
| `visual_concept_caption` | lists the visual findings observed in the image |
| `subcaption_1` … `subcaption_8` | the caption split into knowledge aspects |
| `sub_caption_mask` | 8-element mask marking which subcaptions are present |
| `knowledge_masks` | 3-element mask over (caption, ontology, visual-concept) |
| `ontology_label` | index into the disease hierarchy, `-1` when unmapped |
| `agent_generated` | `True` if MAGEN rewrote this caption, `False` if it is the original Derm1M text |
| `source`, `source_type` | corpus of origin |

</details>

<details>
<summary><b>Evaluation data — downstream benchmarks (7.0 GB)</b></summary>

Download `MAGEN-O-MAKE_downstream.tar.gz` from
[Google Drive](https://drive.google.com/drive/folders/1OnLA1gBFg0To7TplSVEE7FzO6vmmLvQ4) and unpack it:

```bash
tar -xzf MAGEN-O-MAKE_downstream.tar.gz    # creates data/downstream/ and meta/downstream/
```

```
data/downstream/
├── Daffodil/   ├── ISIC2018/   ├── SD-128/    ├── SNU/
├── F17K/       ├── PAD/        ├── SD-198/    └── skin_cap/
```

Each dataset directory holds `images/` plus its metadata CSVs. `meta/downstream/` carries the
train/val/test splits used by zero-shot, linear probing and fine-tuning.

</details>

## Zero-shot disease classification

```bash
bash script/zeroshot_eval.sh
```

which runs all seven benchmarks:

```bash
python src/test.py \
    --model 'hf-hub:Xieji-Li/MAGEN-O-MAKE' \
    --batch-size 512 \
    --workers 8 \
    --csv-img-key filename \
    --csv-label-key label \
    --eval-pad       data/downstream/PAD/MAKE_PAD.csv \
    --eval-f17k      data/downstream/F17K/MAKE_F17K.csv \
    --eval-snu       data/downstream/SNU/MAKE_SNU.csv \
    --eval-sd128     data/downstream/SD-128/MAKE_SD-128.csv \
    --eval-daffodil  data/downstream/Daffodil/MAKE_Daffodil.csv \
    --eval-sd-tails  data/downstream/SD-198/SD-tails-70.csv \
    --eval-snu-tails data/downstream/SNU/MAKE_SNU_tails.csv
```

### Results

Top-1 accuracy against open-source dermatology and general-domain VLMs.

| Model | PAD | F17K | SD-128 | SNU-134 | Daffodil | **Avg.** | SD-Tails | SNU-Tails | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|
| *Classes* | *6* | *113* | *128* | *134* | *5* | | *70* | *85* | |
| [CLIP-OPENAI](https://github.com/openai/CLIP) | 0.433 | 0.063 | 0.073 | 0.073 | 0.454 | 0.219 | 0.148 | 0.118 | 0.133 |
| [BMC-CLIP](https://huggingface.co/BIOMEDICA/BMC_CLIP_CF) | 0.526 | 0.107 | 0.137 | 0.140 | 0.682 | 0.318 | 0.205 | 0.175 | 0.190 |
| [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | 0.430 | 0.089 | 0.132 | 0.097 | 0.589 | 0.267 | 0.192 | 0.136 | 0.164 |
| [MONET](https://github.com/suinleelab/MONET) | 0.474 | 0.150 | 0.217 | 0.150 | 0.758 | 0.350 | 0.311 | 0.179 | 0.245 |
| [DermLIP-PanDerm](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256) | 0.615 | 0.319 | 0.403 | 0.322 | 0.799 | 0.492 | 0.513 | 0.419 | 0.466 |
| [DermLIP-ViTB16](https://huggingface.co/redlessone/DermLIP_ViT-B-16) | 0.627 | 0.229 | 0.287 | 0.253 | 0.733 | 0.426 | 0.424 | 0.312 | 0.368 |
| [**O-MAKE (ours)**](https://huggingface.co/Xieji-Li/MAGEN-O-MAKE) | **0.667** | **0.371** | **0.460** | **0.390** | **0.832** | **0.544** | **0.558** | **0.457** | **0.508** |

See the paper for the full comparison, which also covers models re-pretrained on the same corpus
(SigLIP, CoCa, CLIP, KEP) and the MICCAI'25 MAKE baseline.

## Cross-modal retrieval

Ranks every SkinCAP image against all captions and vice versa, reporting Recall@{10,50,100} and
mean/median rank in both directions:

```bash
bash script/retrieval_eval.sh
```

## Linear probing

Freezes the image encoder, extracts features, and fits logistic regression on top:

```bash
bash script/linear_probe.sh
python linear_probe/sort_script.py logs/linear_probe     # summary table
```

## Fine-tuning

End-to-end fine-tuning of the vision tower with a linear classification head:

```bash
bash script/finetune.sh                  # PAD, F17K, SNU, SD-128, Daffodil, ISIC2018
DATASETS='PAD' bash script/finetune.sh   # a single dataset
```

## Pretraining

```bash
bash script/pretrain.sh
```

This reproduces the released checkpoint: CLIP ViT-B/16 initialised from OpenAI weights, 15 epochs at
batch size 2048, with multi-aspect knowledge contrastive learning (`--MKCL --subcaptions
--num_subcaptions 8`) and Ontology-Based Multi-Knowledge Contrastive Learning (`--OHCL --OHCL_temp 0.07
--OHCL_beta 0.5`).

It reads the MAGEN-augmented corpus from `data/pretrain/` — see
[Data preparation](#data-preparation) for the schema and how to materialise the images.

The ontology asset used by O-MAKE ships with the code:
`src/open_clip_train/ontology/ontology_distance.npy` holds the precomputed pairwise distances in the
Derm1M disease hierarchy, and drives both the hierarchical contrastive loss and the ontology sampler.

## License

Released under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) for
non-commercial research use.

## Acknowledgements

Built on [open_clip](https://github.com/mlfoundations/open_clip). The pretraining corpus derives from
[Derm1M](https://github.com/SiyuanYan1/Derm1M); this work extends [MAKE](https://github.com/SiyuanYan1/MAKE) (MICCAI'25).

## Citation

```bibtex
@article{li2025magenomake,
  title   = {Multi-Aspect Knowledge-Enhanced Medical Vision-Language Pretraining with Multi-Agent Data Generation},
  author  = {Li, Xieji and Yan, Siyuan and Liu, Yingsheng and Soyer, H. Peter and Janda, Monika and Mar, Victoria and Ge, Zongyuan},
  journal = {arXiv preprint arXiv:2512.03445},
  year    = {2025}
}
```
