# MAGEN — Multi-Agent Data GENeration

The pipeline that recaptioned Derm1M into
[Derm1M-AgentAug](https://huggingface.co/datasets/Xieji-Li/Derm1M-AgentAug), the corpus O-MAKE is
pretrained on.

Derm1M pairs a lot of images with captions that do not describe them. MAGEN finds those pairs,
rewrites them with a dermatology captioning agent grounded in a disease shortlist, has a second agent
verify the diagnosis against structured disease knowledge, and finally decomposes each caption into
the knowledge aspects O-MAKE aligns against.

```
Derm1M image-text pairs
        │  step 0   image-text similarity  →  the pairs worth rewriting
        ▼
low-quality pairs
        │  step 1   + Top-5 disease priors from a zero-shot classifier
        ▼
captioning prompts ──────────► Agent 2: Captioning ──────────► draft captions
        │  step 2   merge drafts back into the metadata
        ▼
recaptioned metadata
        │  step 3   + disease cards from Agent 1 (Summary)
        ▼
verification prompts ────────► Agent 3: Verification ────────► verified diagnosis + caption
        │  step 4   ontology caption · visual concepts · subcaptions · masks
        ▼
Derm1M-AgentAug  (the schema script/pretrain.sh consumes)
```

## The captioning agent is not redistributable

Agent 2 is a LLaVA-style model: **Qwen3-14B** (public) with **DermFM-Zero** as its vision tower.
DermFM-Zero is not publicly available, so the captioning agent **cannot be rebuilt from scratch, and
its weights are not released**.

What is provided, under `agents/captioning/`:

| File | |
|---|---|
| `train_dermllava_qwen3_14b.sh` | the exact fine-tuning command used |
| `model_vqa_batch.py` | the batch inference code the recaptioning runs on |

Every other component — the Summary Agent, the Verification Agent, and all four pipeline steps —
runs on publicly available models.

## Agents

| # | Agent | Runner | Model | Role |
|---|---|---|---|---|
| 1 | Summary | `agents/1_summary_agent.py` | Qwen2.5-72B-Instruct | disease knowledge base → disease cards |
| 2 | Captioning | `agents/2_captioning_agent.sh` | DermLLaVA-Qwen3-14B *(not released)* | image + Top-5 priors → draft caption |
| 3 | Verification | `agents/3_verification_agent.py` | Qwen2.5-VL-72B-Instruct | draft + disease cards → verified diagnosis + caption |

Agents 1 and 3 are 72B models — expect multi-GPU. The exact prompt for each is in
[Prompts](#prompts) below.

## Running the agents

```bash
# Agent 1 — disease cards
python agents/1_summary_agent.py --kb disease_knowledge_base.json --out work/disease_cards.csv

# Agent 2 — captioning (inside the LLaVA repo; needs the checkpoint)
LLAVA_ROOT=/path/to/LLaVA bash agents/2_captioning_agent.sh \
    work/1_captioning_input.jsonl work/1_captioning_output.jsonl

# Agent 3 — verification
python agents/3_verification_agent.py \
    --input  work/3_verification_input.csv \
    --output work/3_verification_output.csv \
    --image-root /path/to/derm1m/images
```

## Prompts

The exact prompts used for the released corpus. Wording matters — changing it changes what the
agents produce, so these are reproduced verbatim from the code.

### Agent 1 — Summary (system prompt)

Source: `agents/1_summary_agent.py`. The user turn is `NAME: {disease_name}\n{description}`, one disease per
call. The output is one DiseaseCard per disease.

```text
You are a dermatologist. Compress the disease knowledge below into a concise DiseaseCard.
Keep morphology only: shape, border, surface, color/pattern, distribution, dermoscopy,
pathognomonic clues. No epidemiology or treatment. <=120 tokens. Use short noun phrases;
semicolon-separated.
NAME: <disease name>
POS: <3-8 hallmark positive cues; short phrases>
SITES: <key anatomical sites/patterns>
MINSET: <2-4 minimal sufficient cues>
```

The `POS` / `SITES` / `MINSET` fields are what Agent 3 reasons over: hallmark cues, typical
anatomical sites, and the minimal set of findings sufficient for the diagnosis.

### Agent 2 — Captioning

Source: `scripts/step1_build_captioning_input.py`. Step 1 prepends the disease shortlist, so each
image arrives as:

```text
Top 5 possible diagnoses: {comma-separated diseases, highest probability first}

Describe this skin lesion image in a single paragraph of 3-5 sentences. Describe the
morphology, colour, distribution and anatomical site.
```

Grounding the agent in a shortlist rather than letting it name any disease is what keeps it from
inventing diagnoses.

### Agent 3 — Verification

Source: `scripts/step3_build_verification_input.py`. `{caption}` is Agent 2's draft,
`{top_preds}` the priors above the probability threshold, `{candidates_text}` their DiseaseCards
from Agent 1.

```text
You are a multimodal dermatology revision agent.

Caption:
{caption}

Top-5 Model Predictions (with probabilities):
{top_preds}

Candidate DiseaseCards:
{candidates_text}

Task:
Verify and, if necessary, correct the diagnosis in the caption so that it matches the
best-fitting DiseaseCard.

Important principle:
The morphology description in the caption must be preserved as much as possible.
Only adjust the diagnosis if it clearly conflicts with the DiseaseCards or visual evidence.

Procedure (follow in order):

1) Identify dermatologic morphology and anatomical site from the image and caption.

Examples include papule, nodule, plaque, pustule, scale, crust, ulcer, pigmentation,
telangiectasia, etc.
Visual morphology and site are the PRIMARY evidence.

2) Treat the Top-5 prediction list as a probabilistic prior.
Higher probability diseases should be considered first, but they must NOT override
morphology or site evidence.

3) Compare the extracted findings with each DiseaseCard using POS, MINSET, and SITES.
MINSET features are the most important criteria.

4) Select the diagnosis with the highest overall consistency based on:
(a) morphology and site
(b) POS/MINSET/SITES from the DiseaseCard
(c) support from the Top-5 probability list.

5) If multiple diagnoses remain plausible, prefer the one with higher probability.

6) If no DiseaseCard clearly fits or confidence is low, keep the original diagnosis.

7) Revise the caption only if necessary:
- Preserve morphology description
- Preserve anatomical site
- Only update the disease name
- Do not add new facts

Output JSON ONLY:

{
"diagnosis": "<verified_or_corrected_diagnosis>",
"corrected_caption_paragraph": "<single paragraph of 3-5 sentences preserving morphology description>"
}
```

Two design points worth carrying over if you adapt this: the agent is told to treat the classifier's
ranking as a *prior* that must not override visual evidence, and to keep the original diagnosis when
nothing fits — both push it toward leaving captions alone rather than rewriting them confidently.
That is why only part of the corpus ends up `agent_generated`.

Each `vl_output` is parsed by `scripts/step4_build_training_csv.py`, which tolerates the
` ```json ` fences the model sometimes adds.

## Pipeline steps

All paths live in `scripts/config.py`, overridable by environment variable.

Run these from `magen/`:

```bash
# 0. score every pair and keep the mismatched ones
python scripts/step0_filter_pairs.py --derm1m-csv <derm1m.csv> --image-root <images/>

# 1. ground the captioning agent in a Top-5 disease shortlist
python scripts/step1_build_captioning_input.py --probs <zs_probs.csv> --labels <classnames.txt>

#    → run Agent 2 (see "Running the agents")

# 2. fold the drafts back into the metadata
python scripts/step2_merge_captions.py

# 3. build the verification prompt from disease cards + priors
python scripts/step3_build_verification_input.py --labels <classnames.txt> --cards <disease_cards.csv>

#    → run Agent 3

# 4. produce the O-MAKE training schema
python scripts/step4_build_training_csv.py
```

### What you need to supply

| | |
|---|---|
| Derm1M metadata + images | from [Derm1M](https://github.com/SiyuanYan1/Derm1M) |
| Zero-shot disease probabilities | any classifier over the Derm1M label space; the paper uses DermFM-Zero |
| Disease knowledge base | input to Agent 1 |

Shipped in `assets/`: the nested Derm1M disease tree (`derm1m_ontology_tree.json`, walked to build the
ontology caption) and the visual-concept vocabulary (`skin_concept_list.txt`, 2,371 terms).

## Step 4 output columns

These are exactly the Derm1M-AgentAug columns:

| Column | Built from |
|---|---|
| `truncated_caption` | the verified caption |
| `ontology_caption` | the diagnosis placed on its path through the disease tree |
| `ontology_final` | that path with the sentence scaffolding stripped |
| `visual_concept_caption` | concepts matched in the caption, longest-first |
| `subcaption_1` … `subcaption_8` | the caption split on sentence boundaries |
| `sub_caption_mask` | which subcaption slots are filled |
| `knowledge_masks` | which of (caption, ontology, visual-concept) exist |
| `agent_generated` | whether the captioning agent rewrote this row |

Verified against the released dataset over 2,000 rows: `sub_caption_mask` and `knowledge_masks`
reproduce exactly, subcaption splitting and the visual-concept set both reproduce on 1,997/2,000.
The remainder are artifacts in the source rows, not differences in logic — 163 of the 403,563
released captions (0.04%, all non-agent-generated) carry a literal `"nan"` from an upstream merge.

One deliberate difference: the original joined the concepts through a Python `set()`, so the released
`visual_concept_caption` lists them in an arbitrary order. Here they are emitted longest-first, which
is deterministic. The set of concepts is identical.

## Notes

- Agents 1 and 3 are 72B models — expect multi-GPU (`device_map="auto"`, flash-attention).
- Agent 2 needs the LLaVA repo plus the fine-tuned checkpoint, so it is launched rather than
  reimplemented here. Its vision tower (DermFM-Zero) is not publicly available, so the agent cannot
  be rebuilt and its weights are not distributed; `agents/captioning/` holds the training command and the
  batch inference code for reference.
