"""Central path configuration for the MAGEN Derm1M recaptioning pipeline.

Every step imports its paths from here, so pointing the pipeline at your own
copy of Derm1M only touches this file (or the matching environment variables).
"""
import os

# magen/ root (parent of scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
ASSETS = os.path.join(ROOT, "assets")
WORK = os.environ.get("MAGEN_WORK", os.path.join(ROOT, "work"))

# ---- Inputs you provide ------------------------------------------------------
# Derm1M metadata: one row per image-text pair, with at least `filename` and
# `caption`. Obtain it from https://github.com/SiyuanYan1/Derm1M
DERM1M_CSV = os.environ.get(
    "DERM1M_CSV", os.path.join(REPO_ROOT, "data", "pretrain", "derm1m.csv"))

# Root the `filename` column is relative to.
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", REPO_ROOT)

# Per-image disease probabilities over the Derm1M label space, produced by a
# zero-shot classifier (the paper uses DermFM-Zero). Needs an
# `image_path` column plus `probability_class_0 … probability_class_N`.
ZS_PROBS_CSV = os.environ.get("ZS_PROBS_CSV", os.path.join(WORK, "derm1m_zs_probs.csv"))

# ---- Stage outputs -----------------------------------------------------------
LOW_QUALITY_CSV = os.path.join(WORK, "0_low_quality_pairs.csv")      # step 0
CAPTIONING_INPUT = os.path.join(WORK, "1_captioning_input.jsonl")    # step 1
CAPTIONING_OUTPUT = os.path.join(WORK, "1_captioning_output.jsonl")  # agent 2
RECAPTIONED_CSV = os.path.join(WORK, "2_recaptioned.csv")            # step 2
VERIFICATION_INPUT = os.path.join(WORK, "3_verification_input.csv")  # step 3
VERIFICATION_OUTPUT = os.path.join(WORK, "3_verification_output.csv")  # agent 3
TRAINING_CSV = os.path.join(WORK, "4_magen_training.csv")            # step 4

# ---- Shipped assets ----------------------------------------------------------
# Nested Derm1M disease hierarchy; walked to build the ontology caption.
ONTOLOGY_TREE = os.path.join(ASSETS, "derm1m_ontology_tree.json")
# Visual-concept vocabulary matched against captions.
SKIN_CONCEPTS = os.path.join(ASSETS, "skin_concept_list.txt")
# Agent 1 output; disease knowledge cards used to build the verification prompt.
DISEASE_CARDS = os.environ.get("DISEASE_CARDS", os.path.join(WORK, "disease_cards.csv"))

# ---- Models ------------------------------------------------------------------
# Image-text scorer used to find low-quality pairs (step 0).
FILTER_MODEL = os.environ.get(
    "FILTER_MODEL", "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256")
# Pairs scoring below this are sent for recaptioning.
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.7"))
# How many disease priors to show the captioning agent.
TOP_K_DISEASES = int(os.environ.get("TOP_K_DISEASES", "5"))
# Number of knowledge aspects the caption is split into.
NUM_SUBCAPTIONS = int(os.environ.get("NUM_SUBCAPTIONS", "8"))


def ensure_work_dir():
    os.makedirs(WORK, exist_ok=True)
