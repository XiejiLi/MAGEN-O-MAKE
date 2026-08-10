# MAGEN-O-MAKE

Multi-Aspect Knowledge-Enhanced Medical Vision-Language Pretraining with Multi-Agent Data Generation

<p align="center">
  <a href="https://arxiv.org/abs/2512.03445"><img src="https://img.shields.io/badge/arXiv-2512.03445-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/Xieji-Li/MAGEN-O-MAKE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Xieji--Li/MAGEN--O--MAKE-yellow" alt="Hugging Face"></a>
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

### Quick start(Zero-Shot Skin Cancer Classification)

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

## Data preparation

Download the downstream benchmark bundle from
[Google Drive](TODO: paste the share link after uploading MAGEN-O-MAKE_downstream.tar.gz) and unpack
it at the repository root:

```bash
tar -xzf MAGEN-O-MAKE_downstream.tar.gz    # creates data/downstream/ and meta/downstream/
```

```
data
├── pretrain/                 Derm1M-AgentAug (pretrained dataset)
└── downstream/
    ├── Daffodil/   ├── ISIC2018/   ├── SD-128/    ├── SNU/
    ├── F17K/       ├── PAD/        ├── SD-198/    └── skin_cap/
```

Each dataset directory holds `images/` plus its metadata CSVs, and `meta/downstream/` carries the
train/val/test splits used by linear probing and fine-tuning. Every image path is relative to the
repository root, so run all commands from there.

## Zero-shot disease classification

```bash
bash script/zeroshot_eval.sh
```

which runs all seven benchmarks:

```bash
python src/test.py \
    --model 'ViT-B-16' \
    --resume checkpoints/O-MAKE_epoch_15.pt \
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

Drop any `--eval-<dataset>` flag to skip that benchmark.

### Results

Top-1 accuracy against open-source dermatology and general-domain VLMs, as reported in the paper,
using the 8-template prompt ensemble (`OPENAI_SKIN_TEMPLATES`). The last two columns are long-tail
splits covering rare conditions the model was never given a label for.

| Model | PAD | F17K | SD-128 | SNU-134 | Daffodil | **Avg.** | SD-Tails | SNU-Tails | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|
| *Classes* | *6* | *113* | *128* | *134* | *5* | | *70* | *85* | |
| CLIP-OPENAI | 0.433 | 0.063 | 0.073 | 0.073 | 0.454 | 0.219 | 0.148 | 0.118 | 0.133 |
| BMC-CLIP | 0.526 | 0.107 | 0.137 | 0.140 | 0.682 | 0.318 | 0.205 | 0.175 | 0.190 |
| BioMedCLIP | 0.430 | 0.089 | 0.132 | 0.097 | 0.589 | 0.267 | 0.192 | 0.136 | 0.164 |
| MONET | 0.474 | 0.150 | 0.217 | 0.150 | 0.758 | 0.350 | 0.311 | 0.179 | 0.245 |
| DermLIP-PanDerm | 0.615 | 0.319 | 0.403 | 0.322 | 0.799 | 0.492 | 0.513 | 0.419 | 0.466 |
| DermLIP-ViTB16 | 0.627 | 0.229 | 0.287 | 0.253 | 0.733 | 0.426 | 0.424 | 0.312 | 0.368 |
| **O-MAKE (ours)** | **0.667** | **0.371** | **0.460** | **0.390** | **0.832** | **0.544** | **0.558** | **0.457** | **0.508** |

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
--OHCL_beta 0.5 --loss_type 'cross entropy'`).

The MAGEN-augmented pretraining corpus (**Derm1M-AgentAug**) will be released on Hugging Face upon
acceptance; unpack it into `data/pretrain/`. The script expects a CSV with one row per image-text
pair and these columns:

| Column | Description |
|---|---|
| `filename` | image path relative to the repository root, i.e. `data/pretrain/images/<file>` |
| `truncated_caption` | MAGEN-generated caption, split into `--num_subcaptions` knowledge aspects |
| `label` | disease label, used for the disease-specific aspect weighting |

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
@article{magenomake2025,
  title   = {Multi-Aspect Knowledge-Enhanced Medical Vision-Language Pretraining with Multi-Agent Data Generation},
  author  = {TODO: fill in the author list},
  journal = {arXiv preprint arXiv:2512.03445},
  year    = {2025}
}
```
