"""
representation_analysis.py
---------------------------
Geometric analysis of how representations differ between checkpoints.

Methods implemented
-------------------
1.  Centered Kernel Alignment (CKA)
    Measures representational similarity between two sets of activations
    (e.g. model_A vs model_B at the same layer, or same model at different layers).
    CKA = 1.0  ⟹  identical geometry;  CKA = 0.0  ⟹  maximally dissimilar.
    Uses the linear (HSIC) variant: fast and unbiased.
    Reference: Kornblith et al. 2019 "Similarity of Neural Network Representations
               Revisited".

2.  Layer-wise cosine similarity
    Mean cosine sim between activation vectors of model_A and model_B at each layer.
    Cheaper than CKA; good for a quick sanity check.

3.  Effective dimensionality
    Participation ratio of the singular values of the activation matrix.
    PR = (Σσ_i)² / Σ(σ_i²)
    Measures how "distributed" the representation is across directions.
    High PR → spread across many directions; low PR → low-rank / collapsed.

4.  Residual stream decomposition
    For each layer l, decompose the total change in residual stream into:
        Δ_attn(l) = mean cosine sim of attn_out_A vs attn_out_B
        Δ_mlp(l)  = mean cosine sim of mlp_out_A  vs mlp_out_B
    This tells us whether the divergence is driven by attention or MLP layers.

5.  PCA trajectory
    Project residuals from both models into 2D using PCA on model_A's activations.
    Tracks whether model_B's trajectory through representation space diverges
    from model_A's at a specific layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hooks import ActivationStore
from config import NUM_LAYERS, OUTPUT_DIR


# ─── CKA ─────────────────────────────────────────────────────────────────────

def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Gretton et al.)."""
    n = K.shape[0]
    # Centre the kernel matrices
    H = np.eye(n) - np.ones((n, n)) / n
    KH = K @ H
    LH = L @ H
    return float(np.trace(KH @ LH) / ((n - 1) ** 2))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two activation matrices.
    X, Y: shape [n_examples, n_features]
    """
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = _hsic(K, L)
    hsic_xx = _hsic(K, K)
    hsic_yy = _hsic(L, L)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def layer_cka(stores_A: list[ActivationStore], stores_B: list[ActivationStore]) -> list[float]:
    """
    Compute CKA(A_layer_l, B_layer_l) for each layer l.

    Parameters
    ----------
    stores_A / stores_B : one ActivationStore per problem (n_examples stores)

    Returns
    -------
    list of CKA values, length = num_layers + 1
    """
    if not stores_A or not stores_B:
        return []

    num_layers = len(stores_A[0].residuals)
    cka_values = []

    for l in range(num_layers):
        # Collect [last-token] representation for each example
        X = np.array([s.residuals[l][-1].numpy() for s in stores_A])   # [n, H]
        Y = np.array([s.residuals[l][-1].numpy() for s in stores_B])   # [n, H]
        cka_values.append(linear_cka(X, Y))

    return cka_values


# ─── Cosine similarity ────────────────────────────────────────────────────────

def layer_cosine_sim(stores_A: list[ActivationStore], stores_B: list[ActivationStore]) -> list[float]:
    """
    Mean pairwise cosine similarity between model_A and model_B activations at
    each layer (last token position).
    """
    import torch.nn.functional as F

    if not stores_A:
        return []
    num_layers = len(stores_A[0].residuals)
    cos_sims = []

    for l in range(num_layers):
        sims = []
        for sA, sB in zip(stores_A, stores_B):
            vA = sA.residuals[l][-1].float()
            vB = sB.residuals[l][-1].float()
            sim = F.cosine_similarity(vA.unsqueeze(0), vB.unsqueeze(0)).item()
            sims.append(sim)
        cos_sims.append(float(np.mean(sims)))

    return cos_sims


def component_cosine_divergence(
    stores_A: list[ActivationStore],
    stores_B: list[ActivationStore],
) -> dict[str, list[float]]:
    """
    Compute cosine similarity of attention outputs and MLP outputs per layer.
    Returns {"attn": [...], "mlp": [...]}, length = NUM_LAYERS each.
    """
    import torch.nn.functional as F

    if not stores_A:
        return {}
    n_layers = len(stores_A[0].attn_outs)

    attn_sims = []
    mlp_sims  = []

    for l in range(n_layers):
        a_sims = []
        m_sims = []
        for sA, sB in zip(stores_A, stores_B):
            if l < len(sA.attn_outs) and l < len(sB.attn_outs):
                vA = sA.attn_outs[l][-1].float()
                vB = sB.attn_outs[l][-1].float()
                a_sims.append(F.cosine_similarity(vA.unsqueeze(0), vB.unsqueeze(0)).item())
            if l < len(sA.mlp_outs) and l < len(sB.mlp_outs):
                vA = sA.mlp_outs[l][-1].float()
                vB = sB.mlp_outs[l][-1].float()
                m_sims.append(F.cosine_similarity(vA.unsqueeze(0), vB.unsqueeze(0)).item())
        attn_sims.append(float(np.mean(a_sims)) if a_sims else 0.0)
        mlp_sims.append(float(np.mean(m_sims))  if m_sims else 0.0)

    return {"attn": attn_sims, "mlp": mlp_sims}


# ─── Effective dimensionality ─────────────────────────────────────────────────

def participation_ratio(X: np.ndarray) -> float:
    """
    Participation ratio = (Σσ_i)² / Σ(σ_i²)
    where σ_i are the singular values of X (centred).
    High PR → activations span many directions.
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X_c, full_matrices=False)
    s2 = s ** 2
    denom = (s2 ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(s2.sum() ** 2 / denom)


def layer_effective_dim(stores: list[ActivationStore]) -> list[float]:
    """Participation ratio at each layer (using last-token representations)."""
    if not stores:
        return []
    num_layers = len(stores[0].residuals)
    prs = []
    for l in range(num_layers):
        X = np.array([s.residuals[l][-1].numpy() for s in stores])
        prs.append(participation_ratio(X))
    return prs


# ─── PCA trajectory ───────────────────────────────────────────────────────────

def pca_trajectory(
    stores_A: list[ActivationStore],
    stores_B: list[ActivationStore],
    pca_fit_layer: int = -1,
    n_components: int = 2,
) -> dict:
    """
    Fit PCA on model_A's activations at `pca_fit_layer` (default: final layer),
    then project all layers of both models into that 2D space.

    Returns
    -------
    {
        "layers":         [0, 1, ..., N],
        "variance_A":     [...],   # fraction of variance explained by PC1+PC2 at each layer
        "trajectory_A":   { layer: [[pc1, pc2], ...] },   # per-problem per-layer coords
        "trajectory_B":   { layer: [[pc1, pc2], ...] },
        "centroid_dist":  [...],   # L2 distance between centroids at each layer
    }
    """
    from sklearn.decomposition import PCA

    if not stores_A or not stores_B:
        return {}

    num_layers = len(stores_A[0].residuals)
    fit_layer  = pca_fit_layer if pca_fit_layer >= 0 else num_layers - 1

    X_fit = np.array([s.residuals[fit_layer][-1].numpy() for s in stores_A])
    pca   = PCA(n_components=n_components)
    pca.fit(X_fit)

    traj_A: dict[int, list] = {}
    traj_B: dict[int, list] = {}
    centroid_dists = []
    variances_A    = []

    for l in range(num_layers):
        XA = np.array([s.residuals[l][-1].numpy() for s in stores_A])
        XB = np.array([s.residuals[l][-1].numpy() for s in stores_B])

        cA = pca.transform(XA)   # [n, 2]
        cB = pca.transform(XB)   # [n, 2]

        traj_A[l] = cA.tolist()
        traj_B[l] = cB.tolist()

        centroid_dists.append(float(np.linalg.norm(cA.mean(0) - cB.mean(0))))

        # Variance explained by these 2 PCs when applied to layer l
        total_var = np.var(XA, axis=0).sum() + 1e-9
        proj_var  = np.var(cA, axis=0).sum()
        variances_A.append(float(proj_var / total_var))

    return {
        "layers":        list(range(num_layers)),
        "variance_A":    variances_A,
        "trajectory_A":  {str(k): v for k, v in traj_A.items()},
        "trajectory_B":  {str(k): v for k, v in traj_B.items()},
        "centroid_dist": centroid_dists,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


# ─── Weight-space analysis: parameter drift ──────────────────────────────────

def weight_drift_per_layer(
    model_A,
    model_B,
) -> dict:
    """
    Compute the Frobenius-norm difference of weights for each layer's
    attention projection matrices and MLP matrices.

    Returns
    -------
    {
        "layers":      [0, ..., N-1],
        "q_proj":      [...],
        "k_proj":      [...],
        "v_proj":      [...],
        "o_proj":      [...],
        "gate_proj":   [...],
        "up_proj":     [...],
        "down_proj":   [...],
        "total_attn":  [...],   # sum of q/k/v/o
        "total_mlp":   [...],   # sum of gate/up/down
    }
    """
    state_A = {n: p.detach().float() for n, p in model_A.named_parameters()}
    state_B = {n: p.detach().float() for n, p in model_B.named_parameters()}

    layers = list(range(len(model_A.model.layers)))
    metrics = {k: [] for k in
               ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "total_attn", "total_mlp"]}
    metrics["layers"] = layers

    for l in layers:
        attn_total = 0.0
        mlp_total  = 0.0
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            key = f"model.layers.{l}.self_attn.{proj}.weight"
            if key in state_A and key in state_B:
                diff = (state_A[key] - state_B[key]).norm().item()
            else:
                diff = 0.0
            metrics[proj].append(diff)
            attn_total += diff

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            key = f"model.layers.{l}.mlp.{proj}.weight"
            if key in state_A and key in state_B:
                diff = (state_A[key] - state_B[key]).norm().item()
            else:
                diff = 0.0
            metrics[proj].append(diff)
            mlp_total += diff

        metrics["total_attn"].append(attn_total)
        metrics["total_mlp"].append(mlp_total)

    return metrics


# ─── Aggregate and save ───────────────────────────────────────────────────────

def run_full_representation_analysis(
    stores_A: list[ActivationStore],
    stores_B: list[ActivationStore],
    model_A=None,
    model_B=None,
) -> dict:
    """Compute all representation metrics and return as a single dict."""
    result = {}

    print("  Computing layer CKA …")
    result["cka"]          = layer_cka(stores_A, stores_B)

    print("  Computing cosine similarities …")
    result["cosine_sim"]   = layer_cosine_sim(stores_A, stores_B)
    result["component_cos"] = component_cosine_divergence(stores_A, stores_B)

    print("  Computing effective dimensionality …")
    result["eff_dim_A"]    = layer_effective_dim(stores_A)
    result["eff_dim_B"]    = layer_effective_dim(stores_B)

    print("  Computing PCA trajectories …")
    try:
        result["pca"] = pca_trajectory(stores_A, stores_B)
    except ImportError:
        print("  (sklearn not available, skipping PCA)")
        result["pca"] = {}

    if model_A is not None and model_B is not None:
        print("  Computing weight drift …")
        result["weight_drift"] = weight_drift_per_layer(model_A, model_B)

    result["n_problems"] = len(stores_A)
    return result


def save_representation_results(data: dict, tag: str = "") -> Path:
    out = OUTPUT_DIR / f"representation_analysis{('_'+tag) if tag else ''}.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved representation analysis → {out}")
    return out
