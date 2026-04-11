# Mechanistic Forgetting Analysis

> **Research question:** *Where in the transformer network does a math solution "disappear" when a model forgets it between training checkpoints?*

This directory contains the full mechanistic interpretability pipeline for the follow-up analysis to the Temporal Forgetting paper.

---

## Overview

We take two checkpoints of Qwen2.5-7B trained via RL:
- **Checkpoint A** (e.g. step 96): correctly solves a problem
- **Checkpoint B** (e.g. step 128): completely forgets the same problem

We then apply four complementary mechanistic analyses to find *which transformer layer* causes the forgetting:

| Analysis | Method | Question |
|---|---|---|
| **Logit lens** | Project residual stream at each layer through unembedding | At which layer does A first "commit" to the correct answer, and does B diverge here? |
| **Activation patching** | Swap model B's activations at layer l with model A's | Which layer, when patched from A→B, *restores* the correct answer? |
| **Representation geometry** | CKA, cosine similarity, PCA, effective dimensionality | At which layer do A and B's internal representations diverge most? |
| **Attention analysis** | JSD of attention distributions, head-level divergence | Which attention heads change the most? Does B attend less to relevant numbers? |

---

## File Structure

```
mechanistic_forgetting/
  config.py                  # all settings (checkpoints, paths, constants)
  identify_forgotten.py      # mine the pre-existing sampling data to find forgotten problems
  hooks.py                   # PyTorch forward-hook utilities (capture & patch activations)
  logit_lens.py              # logit-lens computation per layer
  activation_patching.py     # causal intervention: patch A's activations into B layer by layer
  representation_analysis.py # CKA, cosine sim, effective dim, PCA trajectory, weight drift
  attention_analysis.py      # head-level JSD, entropy, induction scores, numeric attention
  run_experiment.py          # main orchestrator (runs all analyses, saves JSON)
  visualize.py               # publication-quality figures from saved results
  results/                   # JSON output files
  figures/                   # generated figures
```

---

## Prerequisites

```bash
conda create -n mech python=3.10
conda activate mech
pip install torch transformers accelerate matplotlib numpy scikit-learn
```

---

## Step 1: Identify Forgotten Problems (no GPU needed)

First, verify the sampling data is available:

```bash
cd Temporal_Forgetting
unzip sampling_64_responses.zip   # if not already done
```

Then mine the forgotten problems:

```bash
cd mechanistic_forgetting
python identify_forgotten.py --save
```

Expected output:
```
Total forgotten problems : 39
By pair:  96->128: 8 problems  (this becomes the primary pair)
```

---

## Step 2: Run All Mechanistic Analyses (GPU required)

```bash
# Full run — all 4 analyses, primary pair auto-detected
python run_experiment.py

# Specify the checkpoint pair explicitly
python run_experiment.py --step-a 96 --step-b 128

# Also capture attention weights (adds ~2x memory but enables head-level analysis)
python run_experiment.py --capture-attn

# Lighter run for testing
python run_experiment.py --device cpu --max-problems 2 --analyses logit_lens
```

This will:
1. Load both checkpoints from HuggingFace (`UWNSL/Qwen2.5-7B-deepscaler_4k_step_{96,128}`)
2. Run all analyses on the forgotten problems
3. Save results to `results/logit_lens.json`, `results/activation_patching.json`, etc.

---

## Step 3: Generate Paper Figures

```bash
python visualize.py --format pdf
```

Figures generated:

| File | Description |
|---|---|
| `fig1_forgotten_overview.pdf` | Bar chart: forgotten problems per checkpoint pair |
| `fig2_logit_lens_prob.pdf` | P(correct answer) vs layer (A vs B) |
| `fig3_logit_lens_rank.pdf` | Rank of correct answer token vs layer |
| `fig4_logit_lens_entropy.pdf` | Distribution entropy vs layer |
| `fig4b_divergence_layer_hist.pdf` | Histogram of per-problem divergence layers |
| `fig5_patching_delta_p.pdf` | Activation patching: mean Δp per layer |
| `fig5b_patching_normalised.pdf` | Normalised patching effect (0=no effect, 1=full recovery) |
| `fig6_patching_peak_votes_*.pdf` | Peak patching layer distribution |
| `fig7_cka_cosine.pdf` | CKA + cosine similarity vs layer |
| `fig7b_component_cosine.pdf` | Cosine sim of attention vs MLP outputs |
| `fig8_effective_dim.pdf` | Effective dimensionality (participation ratio) |
| `fig9_pca_centroid_dist.pdf` | PCA centroid distance A vs B |
| `fig10_weight_drift.pdf` | Frobenius weight drift per layer |
| `fig11_attn_jsd_layer.pdf` | Layer-wise attention pattern JSD |
| `fig12_attn_head_heatmap.pdf` | Head × layer JSD heatmap |
| `fig13_numeric_attention.pdf` | Attention to numeric tokens |
| **`fig14_summary_composite.pdf`** | **Main paper figure: 4-panel summary** |

---

## Key Expected Results

Based on the mechanistic interpretability literature and initial data exploration, expect:

- **Logit lens**: The correct answer token first appears in the residual stream at layers ~18-22 for model A but never stabilises for model B
- **Activation patching**: Peak causal layer around layers 20-25 (later layers encode the "committed" answer)
- **CKA**: Drops sharply at middle layers (the "processing" layers diverge before output)
- **Attention**: A specific head in layers 18-22 may lose its "induction-like" focus on relevant numbers

---

## Interpreting Results for Paper Narrative

The convergence of evidence from all four analyses points to a **critical layer range** where the solution disappears. This is the main finding for the paper:

> "The math solution is lost at layer [X] — below this, both checkpoints process the problem similarly; above this, they commit to different answers."

This supports a mechanistic understanding of temporal forgetting as a *localized*, not *distributed*, phenomenon in the transformer — potentially pointing to specific weight matrices that could be monitored or regularized to prevent forgetting.
