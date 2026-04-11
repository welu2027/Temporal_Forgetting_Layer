"""
config.py
---------
Central configuration for the "Where does a math solution disappear?" experiment.

Architecture: Qwen2.5-7B  (28 layers, hidden=3584, heads=28, kv_heads=4)
Checkpoints: UWNSL/Qwen2.5-7B-deepscaler_4k_step_{32,64,...,256}
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
SAMPLING_ZIP = REPO_ROOT / "sampling_64_responses.zip"
SAMPLING_DIR = REPO_ROOT / "sampling_64_responses"   # extracted
OUTPUT_DIR   = REPO_ROOT / "mechanistic_forgetting" / "results"
FIGS_DIR     = REPO_ROOT / "mechanistic_forgetting" / "figures"

for _d in [OUTPUT_DIR, FIGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Model / training ─────────────────────────────────────────────────────────
CHECKPOINT_STEPS = [32, 64, 96, 128, 160, 192, 224, 256]
BASE_MODEL_ID    = "UWNSL/Qwen2.5-7B-deepscaler_4k_step_{step}"
BASE_QWEN_ID     = "Qwen/Qwen2.5-7B"           # pretrained (step 0 reference)

# Convenience: map step → HF model id
def ckpt_id(step: int) -> str:
    return BASE_MODEL_ID.format(step=step)

def all_ckpt_ids():
    return [ckpt_id(s) for s in CHECKPOINT_STEPS]

# ─── Qwen2.5-7B architecture constants ───────────────────────────────────────
NUM_LAYERS      = 28
HIDDEN_DIM      = 3584
NUM_HEADS       = 28
NUM_KV_HEADS    = 4
HEAD_DIM        = HIDDEN_DIM // NUM_HEADS   # 128
INTERMEDIATE    = 18944
VOCAB_SIZE      = 152064

# ─── Experiment settings ──────────────────────────────────────────────────────
# Which tasks to analyse
TASKS = ["AIME", "AIME25", "AMC"]

# Pairs of checkpoints to compare (earlier → later where forgetting is detected)
# These are (checkpoint_A, checkpoint_B) pairs where A solves, B forgets.
# You can override this or let identify_forgotten.py discover them automatically.
ANALYSIS_PAIRS = [
    (32,  64),
    (64,  96),
    (96,  128),
    (128, 160),
    (160, 192),
    (192, 224),
    (224, 256),
]

# For mechanistic experiments we pick ONE representative pair with the most
# forgotten problems.  Set to None to auto-detect.
PRIMARY_PAIR: tuple | None = None   # e.g. (96, 128)

# Number of forgotten problems to run activation analysis on
# (GPU memory bound — 8 is plenty for stable statistics)
MAX_PROBLEMS_FOR_ACTIVATION = 8

# Generation / inference settings
MAX_NEW_TOKENS = 1024          # shorter for activation extraction (save memory)
BATCH_SIZE     = 1             # activation patching requires single-example batches
DEVICE         = "cuda"        # "cpu" for debugging
DTYPE          = "bfloat16"

# ─── Logit-lens settings ──────────────────────────────────────────────────────
# We track the probability of the *correct* answer digit/token at each layer.
# For AIME problems answers are integers; we extract the first token of the answer.
LOGIT_LENS_LAYERS = list(range(NUM_LAYERS + 1))  # 0 = embed, 1-28 = after each layer

# ─── Activation patching settings ────────────────────────────────────────────
# Which components to patch: "residual", "attention_out", "mlp_out"
PATCH_TARGETS = ["residual", "attention_out", "mlp_out"]

# ─── Representation analysis settings ────────────────────────────────────────
# Number of random problems (from the CORRECT set) used as a reference distribution
# for CKA baselines
CKA_BASELINE_SAMPLES = 32
