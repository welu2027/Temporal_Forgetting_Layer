"""
visualize.py
------------
Publication-quality figures for the mechanistic forgetting paper.

Figures generated
-----------------
Fig 1.  Overview: forgotten problem count per checkpoint pair  (bar chart)
Fig 2.  Logit lens: P(correct answer) vs layer for A and B  (line plot)
Fig 3.  Logit lens: rank of correct answer vs layer  (line plot)
Fig 4.  Logit lens: distribution entropy vs layer  (line plot)
Fig 5.  Activation patching: mean Δp per layer × component  (grouped bar / heatmap)
Fig 6.  Activation patching heatmap: [problem × layer] Δp matrix
Fig 7.  Representation: CKA and cosine sim vs layer  (dual-axis line)
Fig 8.  Representation: effective dimensionality A vs B  (line plot)
Fig 9.  Representation: PCA centroid distance vs layer
Fig 10. Weight drift: Frobenius norm per layer (attn vs MLP)
Fig 11. Attention: layer-wise mean JSD  (line)
Fig 12. Attention: head-level JSD heatmap [layer × head]
Fig 13. Attention: fraction attending to numeric tokens A vs B
Fig 14. Summary: "where does the solution disappear?" composite figure
        (multi-panel with logit-lens + patching peak distribution)

Usage
-----
    python visualize.py                    # uses all JSON files in OUTPUT_DIR
    python visualize.py --tag my_tag       # loads *my_tag.json files
    python visualize.py --figs 2,5,14      # only generate specific figures
    python visualize.py --format pdf       # save as PDF instead of PNG
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# make local imports work
sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, FIGS_DIR, NUM_LAYERS


# ─── Style ────────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "figure.facecolor":  "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.6,
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
    })

COLOR_A   = "#1f77b4"   # blue   = model A (correct)
COLOR_B   = "#d62728"   # red    = model B (forgotten)
COLOR_GAP = "#2ca02c"   # green  = difference


def savefig(fig, name: str, fmt: str = "pdf", tight: bool = True):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    path = FIGS_DIR / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"  Saved -> {path}")
    plt.close(fig)


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_json(name: str, tag: str = "") -> dict | None:
    suffix = f"_{tag}" if tag else ""
    path   = OUTPUT_DIR / f"{name}{suffix}.json"
    if not path.exists():
        # try without tag suffix
        path = OUTPUT_DIR / f"{name}.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ─── Fig 1: Forgotten problem count ──────────────────────────────────────────

def fig_forgotten_overview(tag: str = "", fmt: str = "pdf"):
    """Bar chart: number of forgotten problems per checkpoint pair."""
    data = load_json("forgotten_problems", tag)
    if data is None:
        return

    from collections import defaultdict
    pair_counts: dict[str, int] = defaultdict(int)
    for fp in data:
        k = f"{fp['step_A']}->{fp['step_B']}"
        pair_counts[k] += 1

    pairs  = sorted(pair_counts.keys(), key=lambda s: int(s.split("->")[0]))
    counts = [pair_counts[p] for p in pairs]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(range(len(pairs)), counts, color=COLOR_B, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=25, ha="right")
    ax.set_xlabel("Checkpoint pair (A -> B)")
    ax.set_ylabel("# forgotten problems")
    ax.set_title("Forgotten Problems per Training Step Transition")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                str(c), ha="center", va="bottom", fontsize=9)
    savefig(fig, "fig1_forgotten_overview", fmt)


# ─── Fig 2-4: Logit lens ──────────────────────────────────────────────────────

def fig_logit_lens(tag: str = "", fmt: str = "pdf"):
    data = load_json("logit_lens", tag)
    if data is None:
        return

    layers = data["layers"]
    pA     = data["mean_prob_A"]
    pB     = data["mean_prob_B"]
    gap    = data["prob_gap"]
    rA     = data["mean_rank_A"]
    rB     = data["mean_rank_B"]
    eA     = data["mean_entropy_A"]
    eB     = data["mean_entropy_B"]

    # ── Fig 2: probability ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(layers, pA, color=COLOR_A, label="Checkpoint A (correct)", marker="o", markersize=3)
    ax.plot(layers, pB, color=COLOR_B, label="Checkpoint B (forgotten)", marker="s", markersize=3)
    ax.fill_between(layers, pB, pA, where=[a > b for a, b in zip(pA, pB)],
                    alpha=0.15, color=COLOR_GAP, label="Probability gap")
    ax.set_xlabel("Layer (0 = embedding, N = final)")
    ax.set_ylabel("P(correct answer token)")
    ax.set_title(f"Logit Lens: Correct-Answer Probability Across Layers\n"
                 f"(n={data.get('n_problems', '?')} forgotten problems, "
                 f"mean over {data.get('n_problems', '?')} examples)")
    ax.set_xlim(0, max(layers))
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    # Mark peak gap
    peak = int(np.argmax(gap))
    ax.axvline(peak, color=COLOR_GAP, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(peak + 0.3, max(pA) * 0.5, f"Peak gap\nlayer {peak}",
            color=COLOR_GAP, fontsize=8)
    savefig(fig, "fig2_logit_lens_prob", fmt)

    # ── Fig 3: rank ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(layers, rA, color=COLOR_A, label="Checkpoint A (correct)", marker="o", markersize=3)
    ax.plot(layers, rB, color=COLOR_B, label="Checkpoint B (forgotten)", marker="s", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank of correct answer token\n(lower = better)")
    ax.set_title("Logit Lens: Rank of Correct Answer Token")
    ax.invert_yaxis()
    ax.legend()
    savefig(fig, "fig3_logit_lens_rank", fmt)

    # ── Fig 4: entropy ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(layers, eA, color=COLOR_A, label="Checkpoint A (correct)", marker="o", markersize=3)
    ax.plot(layers, eB, color=COLOR_B, label="Checkpoint B (forgotten)", marker="s", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy of logit distribution")
    ax.set_title("Logit Lens: Distribution Entropy Across Layers")
    ax.legend()
    savefig(fig, "fig4_logit_lens_entropy", fmt)

    # ── Histogram: divergence layers ────────────────────────────────────────
    div_layers = data.get("divergence_layers", [])
    if div_layers:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(div_layers, bins=range(max(layers) + 2), color=COLOR_B, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Number of problems")
        ax.set_title("Distribution of Divergence Layers\n(first layer where A predicts correct, B doesn't)")
        savefig(fig, "fig4b_divergence_layer_hist", fmt)


# ─── Fig 5-6: Activation patching ─────────────────────────────────────────────

def fig_activation_patching(tag: str = "", fmt: str = "pdf"):
    data = load_json("activation_patching", tag)
    if data is None:
        return

    targets      = [t for t in ["residual", "attention_out", "mlp_out"] if t in data]
    target_labels = {"residual": "Residual stream", "attention_out": "Attention output",
                     "mlp_out": "MLP output"}
    target_colors = {"residual": "#1f77b4", "attention_out": "#ff7f0e", "mlp_out": "#2ca02c"}

    # ── Fig 5: Δp curves by component ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for target in targets:
        d       = data[target]
        layers  = d["layers"]
        mean_dp = d["mean_delta_p"]
        std_dp  = d.get("std_delta_p", [0]*len(layers))
        ax.plot(layers, mean_dp,
                label=target_labels.get(target, target),
                color=target_colors.get(target, "grey"),
                marker="o", markersize=3)
        ax.fill_between(layers,
                        [m - s for m, s in zip(mean_dp, std_dp)],
                        [m + s for m, s in zip(mean_dp, std_dp)],
                        alpha=0.15, color=target_colors.get(target, "grey"))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer patched (A -> B)")
    ax.set_ylabel("Mean Δp (patched − baseline)")
    ax.set_title("Activation Patching: Mean Recovery of Correct-Answer Probability\n"
                 "(positive = patching layer l from A into B restores correct answer)")
    ax.legend()
    savefig(fig, "fig5_patching_delta_p", fmt)

    # ── Fig 5b: normalised Δp ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for target in targets:
        d = data[target]
        ax.plot(d["layers"], d["mean_delta_p_norm"],
                label=target_labels.get(target, target),
                color=target_colors.get(target, "grey"),
                marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="grey",  linewidth=0.8, linestyle=":")
    ax.set_xlabel("Layer patched")
    ax.set_ylabel("Normalised Δp  (0=no effect, 1=full recovery)")
    ax.set_title("Activation Patching: Normalised Causal Effect per Layer")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    savefig(fig, "fig5b_patching_normalised", fmt)

    # ── Fig 6: peak-layer vote histogram ────────────────────────────────────
    for target in targets:
        d = data[target]
        votes = d.get("peak_layer_votes", {})
        if not votes:
            continue
        layers = sorted(int(k) for k in votes)
        counts = [votes[str(l)] for l in layers]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(layers, counts, color=target_colors.get(target, "grey"),
               alpha=0.8, edgecolor="white")
        ax.set_xlabel("Layer")
        ax.set_ylabel("# problems")
        ax.set_title(f"Peak Patching Layer Distribution\n({target_labels.get(target, target)})")
        savefig(fig, f"fig6_patching_peak_votes_{target}", fmt)


# ─── Fig 7-10: Representation ────────────────────────────────────────────────

def fig_representation(tag: str = "", fmt: str = "pdf"):
    data = load_json("representation_analysis", tag)
    if data is None:
        return

    layers = list(range(len(data.get("cka", []))))
    if not layers:
        return

    # ── Fig 7: CKA + cosine ──────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    cka = data.get("cka", [])
    cos = data.get("cosine_sim", [])
    ax1.plot(layers, cka, color=COLOR_A, label="Linear CKA", marker="o", markersize=3)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Linear CKA  (1 = identical)", color=COLOR_A)
    ax1.tick_params(axis="y", labelcolor=COLOR_A)
    ax1.set_ylim(0, 1.05)

    if cos:
        ax2 = ax1.twinx()
        ax2.plot(layers, cos, color=COLOR_B, label="Cosine similarity",
                 linestyle="--", marker="s", markersize=3)
        ax2.set_ylabel("Mean cosine similarity", color=COLOR_B)
        ax2.tick_params(axis="y", labelcolor=COLOR_B)
        ax2.set_ylim(-0.05, 1.05)
        lines = [plt.Line2D([0], [0], color=COLOR_A, label="Linear CKA"),
                 plt.Line2D([0], [0], color=COLOR_B, linestyle="--", label="Cosine sim")]
        ax1.legend(handles=lines, loc="lower left")

    ax1.set_title("Representational Similarity Between Checkpoints A and B")
    # mark the minimum-CKA layer
    if cka:
        min_layer = int(np.argmin(cka))
        ax1.axvline(min_layer, color="grey", linestyle=":", linewidth=1)
        ax1.text(min_layer + 0.3, 0.05, f"min CKA\nlayer {min_layer}",
                 fontsize=8, color="grey")
    savefig(fig, "fig7_cka_cosine", fmt)

    # ── Fig 7b: component cosine ─────────────────────────────────────────────
    comp = data.get("component_cos", {})
    if comp.get("attn") and comp.get("mlp"):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(range(len(comp["attn"])), comp["attn"],
                color="#ff7f0e", label="Attention output", marker="^", markersize=3)
        ax.plot(range(len(comp["mlp"])),  comp["mlp"],
                color="#2ca02c", label="MLP output", marker="v", markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title("Cosine Similarity of Attention vs MLP Outputs (A vs B)")
        ax.legend()
        savefig(fig, "fig7b_component_cosine", fmt)

    # ── Fig 8: effective dimensionality ─────────────────────────────────────
    edA = data.get("eff_dim_A", [])
    edB = data.get("eff_dim_B", [])
    if edA and edB:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(layers, edA, color=COLOR_A, label="Checkpoint A", marker="o", markersize=3)
        ax.plot(layers, edB, color=COLOR_B, label="Checkpoint B", marker="s", markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Participation ratio (effective dimensionality)")
        ax.set_title("Effective Dimensionality of Residual Stream\n"
                     "(lower = more collapsed representation)")
        ax.legend()
        savefig(fig, "fig8_effective_dim", fmt)

    # ── Fig 9: PCA centroid distance ─────────────────────────────────────────
    pca = data.get("pca", {})
    cd  = pca.get("centroid_dist", [])
    if cd:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(range(len(cd)), cd, color=COLOR_B, marker="o", markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("L2 distance between PCA centroids (A vs B)")
        ax.set_title("PCA Centroid Distance: How Far Apart Are A and B?\n"
                     "(PCA fitted on model A's final-layer activations)")
        peak = int(np.argmax(cd))
        ax.axvline(peak, color="grey", linestyle=":", linewidth=1)
        ax.text(peak + 0.3, max(cd) * 0.5, f"layer {peak}", fontsize=8, color="grey")
        savefig(fig, "fig9_pca_centroid_dist", fmt)

    # ── Fig 10: weight drift ─────────────────────────────────────────────────
    wd = data.get("weight_drift", {})
    if wd.get("total_attn") and wd.get("total_mlp"):
        n = len(wd["layers"])
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(wd["layers"], wd["total_attn"], color="#ff7f0e",
                label="Attention (Q+K+V+O Frobenius norm diff)", marker="^", markersize=3)
        ax.plot(wd["layers"], wd["total_mlp"],  color="#2ca02c",
                label="MLP (gate+up+down Frobenius norm diff)",  marker="v", markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("||W_A − W_B||_F  (weight drift)")
        ax.set_title("Weight-Space Drift Between Checkpoints A and B")
        ax.legend()
        savefig(fig, "fig10_weight_drift", fmt)


# ─── Fig 11-13: Attention ─────────────────────────────────────────────────────

def fig_attention(tag: str = "", fmt: str = "pdf"):
    data = load_json("attention_analysis", tag)
    if data is None:
        return

    # ── Fig 11: layer-wise JSD ───────────────────────────────────────────────
    layer_jsd = data.get("layer_mean_jsd", [])
    if layer_jsd:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(range(len(layer_jsd)), layer_jsd,
                color=COLOR_B, marker="o", markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Jensen-Shannon divergence of attention weights")
        ax.set_title("Layer-wise Attention Pattern Divergence (A vs B)")
        peak = int(np.argmax(layer_jsd))
        ax.axvline(peak, color="grey", linestyle=":", linewidth=1)
        ax.text(peak + 0.3, max(layer_jsd) * 0.5, f"layer {peak}", fontsize=8)
        savefig(fig, "fig11_attn_jsd_layer", fmt)

    # ── Fig 12: head-level JSD heatmap ──────────────────────────────────────
    head_matrix = data.get("head_jsd_matrix", [])
    if head_matrix:
        matrix = np.array(head_matrix)   # [n_layers, n_heads]
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(matrix.T, aspect="auto", cmap="Reds", origin="lower",
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Mean JSD")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        ax.set_title("Attention Head Divergence Heatmap\n"
                     "Jensen-Shannon divergence per (layer, head)")
        savefig(fig, "fig12_attn_head_heatmap", fmt)

    # ── Fig 13: number attention ─────────────────────────────────────────────
    num_attn = data.get("num_attention", {})
    if num_attn.get("layers"):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(num_attn["layers"], num_attn.get("num_attn_A", []),
                color=COLOR_A, label="Checkpoint A", marker="o", markersize=3)
        ax.plot(num_attn["layers"], num_attn.get("num_attn_B", []),
                color=COLOR_B, label="Checkpoint B", marker="s", markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Fraction of attention on numeric tokens")
        ax.set_title("Attention to Numeric Tokens in the Prompt")
        ax.legend()
        savefig(fig, "fig13_numeric_attention", fmt)


# ─── Fig 14: Summary / composite ─────────────────────────────────────────────

def fig_summary(tag: str = "", fmt: str = "pdf"):
    """
    Multi-panel summary figure answering: "Where does the solution disappear?"
    Panel A: logit-lens probability gap (peak = where A and B first diverge)
    Panel B: activation patching Δp (residual stream, normalised)
    Panel C: CKA similarity
    Panel D: histogram of divergence layers
    """
    lens_data  = load_json("logit_lens", tag)
    patch_data = load_json("activation_patching", tag)
    repr_data  = load_json("representation_analysis", tag)

    n_panels = sum([
        lens_data is not None,
        patch_data is not None and "residual" in patch_data,
        repr_data  is not None and "cka" in repr_data,
        lens_data  is not None and bool(lens_data.get("divergence_layers")),
    ])

    if n_panels == 0:
        print("  No data available for summary figure.")
        return

    fig = plt.figure(figsize=(14, 5))
    cols  = n_panels
    spec  = gridspec.GridSpec(1, cols, wspace=0.38)
    panel = 0

    if lens_data is not None:
        ax = fig.add_subplot(spec[0, panel])
        layers = lens_data["layers"]
        pA     = lens_data["mean_prob_A"]
        pB     = lens_data["mean_prob_B"]
        ax.plot(layers, pA, color=COLOR_A, label="Checkpoint A", linewidth=2)
        ax.plot(layers, pB, color=COLOR_B, label="Checkpoint B", linewidth=2)
        ax.fill_between(layers, pB, pA, where=[a > b for a, b in zip(pA, pB)],
                        alpha=0.15, color=COLOR_GAP)
        peak = int(np.argmax(lens_data["prob_gap"]))
        ax.axvline(peak, color=COLOR_GAP, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("P(correct answer)")
        ax.set_title("(A) Logit Lens")
        ax.legend(fontsize=8)
        panel += 1

    if patch_data is not None and "residual" in patch_data:
        ax    = fig.add_subplot(spec[0, panel])
        d     = patch_data["residual"]
        ax.bar(d["layers"], d["mean_delta_p_norm"],
               color=COLOR_A, alpha=0.8, edgecolor="white", width=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        peak = int(np.argmax(d["mean_delta_p_norm"]))
        ax.axvline(peak, color=COLOR_GAP, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Layer patched")
        ax.set_ylabel("Normalised Δp")
        ax.set_title("(B) Activation Patching\n(residual stream)")
        panel += 1

    if repr_data is not None and repr_data.get("cka"):
        ax  = fig.add_subplot(spec[0, panel])
        cka = repr_data["cka"]
        ax.plot(range(len(cka)), cka, color="#9467bd", linewidth=2, marker="o", markersize=3)
        min_l = int(np.argmin(cka))
        ax.axvline(min_l, color="grey", linestyle=":", linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Linear CKA")
        ax.set_title("(C) Representational\nSimilarity")
        panel += 1

    if lens_data is not None and lens_data.get("divergence_layers"):
        ax   = fig.add_subplot(spec[0, panel])
        divs = lens_data["divergence_layers"]
        ax.hist(divs, bins=range(max(lens_data["layers"]) + 2),
                color=COLOR_B, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Layer")
        ax.set_ylabel("# problems")
        ax.set_title("(D) Divergence Layer\nHistogram")
        panel += 1

    fig.suptitle(
        "Where Does the Math Solution Disappear?\n"
        "Mechanistic Analysis Across Transformer Layers",
        fontsize=13, fontweight="bold", y=1.02,
    )
    savefig(fig, "fig14_summary_composite", fmt, tight=False)
    fig.savefig(FIGS_DIR / f"fig14_summary_composite.{fmt}",
                bbox_inches="tight", dpi=200)
    print(f"  Saved -> {FIGS_DIR}/fig14_summary_composite.{fmt}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",    default="",    help="File tag suffix")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--figs",   default="all",
                        help="Comma-separated fig numbers (1,2,5,14) or 'all'")
    args = parser.parse_args()

    set_style()

    fig_ids = None if args.figs == "all" else set(args.figs.split(","))

    def should_run(n):
        return fig_ids is None or str(n) in fig_ids

    print(f"\nGenerating figures -> {FIGS_DIR}\n")

    if should_run(1):
        fig_forgotten_overview(args.tag, args.format)
    if should_run("2") or should_run("3") or should_run("4"):
        fig_logit_lens(args.tag, args.format)
    if should_run("5") or should_run("6"):
        fig_activation_patching(args.tag, args.format)
    if any(should_run(n) for n in ["7", "8", "9", "10"]):
        fig_representation(args.tag, args.format)
    if any(should_run(n) for n in ["11", "12", "13"]):
        fig_attention(args.tag, args.format)
    if should_run("14"):
        fig_summary(args.tag, args.format)

    print("\nDone.")


if __name__ == "__main__":
    main()
