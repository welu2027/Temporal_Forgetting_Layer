"""
attention_analysis.py
---------------------
Detailed analysis of attention patterns across checkpoints.

Questions answered
------------------
1.  Which layer's attention patterns change most between checkpoint A and B?
2.  Which heads change most?  (Head-level JS divergence)
3.  Does model B "attend less" to the relevant numbers / operators in the prompt?
4.  What is the effective rank (attention entropy) per head per layer?
5.  Induction-head signatures: do any heads that support in-context problem
    solving disappear between A and B?

Methods
-------
- Jensen-Shannon divergence between attention distributions (per head, per layer)
- Attention entropy comparison (high entropy = diffuse; low entropy = sharp/induction)
- "Attention to numbers" metric: what fraction of total attention falls on
  numeric tokens in the prompt?
- Attention distance: average distance between attended position and query position
  (local vs. global attention pattern)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from hooks import ActivationStore
from config import NUM_LAYERS, NUM_HEADS, OUTPUT_DIR


# ─── Utilities ────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / (e.sum(-1, keepdims=True) + 1e-9)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Jensen-Shannon divergence between two distributions p and q."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * np.log(p / m)).sum()
    kl_qm = (q * np.log(q / m)).sum()
    return float(0.5 * (kl_pm + kl_qm))


def attention_entropy(attn_weights: np.ndarray) -> np.ndarray:
    """
    Compute entropy of attention distribution.
    attn_weights: [..., seq_q, seq_k]
    Returns: [..., seq_q]
    """
    p = attn_weights + 1e-9
    p = p / p.sum(-1, keepdims=True)
    return -(p * np.log(p)).sum(-1)


# ─── Per-layer, per-head attention divergence ─────────────────────────────────

@dataclass
class HeadDivergenceResult:
    """JSD between model A and B attention patterns for one head at one layer."""
    layer:          int
    head:           int
    mean_jsd:       float    # averaged over all query positions and examples
    last_pos_jsd:   float    # at the last query position (most relevant for generation)
    entropy_A:      float    # mean entropy of A's distribution
    entropy_B:      float    # mean entropy of B's distribution
    delta_entropy:  float    # entropy_B - entropy_A  (positive = B is more diffuse)


def compute_head_divergences(
    stores_A: list[ActivationStore],
    stores_B: list[ActivationStore],
) -> list[list[HeadDivergenceResult]]:
    """
    Compute head-level divergence for every (layer, head) pair.

    Returns
    -------
    results[layer][head] = HeadDivergenceResult
    """
    if not stores_A or not stores_A[0].attn_weights:
        return []

    n_layers = len(stores_A[0].attn_weights)
    n_heads  = stores_A[0].attn_weights[0].shape[0] if stores_A[0].attn_weights[0] is not None else NUM_HEADS

    layer_results = []
    for l in range(n_layers):
        head_results = []
        for h in range(n_heads):
            jsds   = []
            ents_A = []
            ents_B = []
            last_jsds = []

            for sA, sB in zip(stores_A, stores_B):
                wA = sA.attn_weights[l]   # [heads, seq, seq] or None
                wB = sB.attn_weights[l]
                if wA is None or wB is None:
                    continue
                if h >= wA.shape[0] or h >= wB.shape[0]:
                    continue

                pA = wA[h].numpy()   # [seq, seq]
                pB = wB[h].numpy()

                # JSD over all query positions
                for q in range(pA.shape[0]):
                    jsds.append(js_divergence(pA[q], pB[q]))
                last_jsds.append(js_divergence(pA[-1], pB[-1]))

                # Entropy
                ents_A.extend(attention_entropy(pA).tolist())
                ents_B.extend(attention_entropy(pB).tolist())

            mean_jsd  = float(np.mean(jsds))   if jsds   else 0.0
            last_jsd  = float(np.mean(last_jsds)) if last_jsds else 0.0
            ent_A     = float(np.mean(ents_A)) if ents_A else 0.0
            ent_B     = float(np.mean(ents_B)) if ents_B else 0.0

            head_results.append(HeadDivergenceResult(
                layer=l, head=h,
                mean_jsd=mean_jsd, last_pos_jsd=last_jsd,
                entropy_A=ent_A, entropy_B=ent_B,
                delta_entropy=ent_B - ent_A,
            ))
        layer_results.append(head_results)
    return layer_results


def top_divergent_heads(
    head_divs: list[list[HeadDivergenceResult]],
    top_k: int = 10,
) -> list[HeadDivergenceResult]:
    """Return the top-k most divergent (layer, head) pairs by mean JSD."""
    all_heads = [hd for layer in head_divs for hd in layer]
    return sorted(all_heads, key=lambda h: h.mean_jsd, reverse=True)[:top_k]


# ─── "Attention to numbers" metric ────────────────────────────────────────────

def _find_number_positions(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Return token positions that are numeric tokens."""
    number_pattern = re.compile(r"^\s*\d+\.?\d*\s*$")
    positions = []
    for i, tok_id in enumerate(input_ids[0].tolist()):
        decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
        if number_pattern.match(decoded):
            positions.append(i)
    return positions


def attention_to_numbers(
    stores_A: list[ActivationStore],
    stores_B: list[ActivationStore],
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """
    For each layer, compute what fraction of total attention (from the last
    query position) is directed towards numeric tokens in the prompt.

    Returns: {"layers": [...], "num_attn_A": [...], "num_attn_B": [...], "delta": [...]}
    """
    if not stores_A or not stores_A[0].attn_weights:
        return {}

    n_layers = len(stores_A[0].attn_weights)

    num_attn_A = [[] for _ in range(n_layers)]
    num_attn_B = [[] for _ in range(n_layers)]

    for sA, sB in zip(stores_A, stores_B):
        if sA.input_ids is None:
            continue
        num_positions = _find_number_positions(sA.input_ids, tokenizer)
        if not num_positions:
            continue

        for l in range(n_layers):
            wA = sA.attn_weights[l]
            wB = sB.attn_weights[l]
            if wA is None or wB is None:
                continue

            # Average over all heads at last query position
            # wA: [heads, seq, seq]
            pA = wA[:, -1, :].mean(0).numpy()   # [seq]
            pB = wB[:, -1, :].mean(0).numpy()

            pA = pA / (pA.sum() + 1e-9)
            pB = pB / (pB.sum() + 1e-9)

            frac_A = sum(pA[pos] for pos in num_positions if pos < len(pA))
            frac_B = sum(pB[pos] for pos in num_positions if pos < len(pB))

            num_attn_A[l].append(frac_A)
            num_attn_B[l].append(frac_B)

    layers  = list(range(n_layers))
    mean_A  = [float(np.mean(v)) if v else 0.0 for v in num_attn_A]
    mean_B  = [float(np.mean(v)) if v else 0.0 for v in num_attn_B]
    delta   = [a - b for a, b in zip(mean_A, mean_B)]

    return {
        "layers":        layers,
        "num_attn_A":    mean_A,
        "num_attn_B":    mean_B,
        "delta":         delta,
        "peak_layer_A":  int(np.argmax(mean_A)),
        "peak_layer_B":  int(np.argmax(mean_B)),
    }


# ─── Attention distance ───────────────────────────────────────────────────────

def mean_attention_distance(
    stores: list[ActivationStore],
) -> list[float]:
    """
    Average distance (in token positions) between query and attended position.
    Low distance → local attention (induction-head-like).
    High distance → global attention.
    """
    if not stores or not stores[0].attn_weights:
        return []

    n_layers = len(stores[0].attn_weights)
    dists = [[] for _ in range(n_layers)]

    for s in stores:
        for l in range(n_layers):
            w = s.attn_weights[l]
            if w is None:
                continue
            seq = w.shape[1]
            pos = np.arange(seq)
            # w: [heads, seq, seq]
            for h in range(w.shape[0]):
                attn = w[h].numpy()   # [seq, seq]
                attn = attn / (attn.sum(-1, keepdims=True) + 1e-9)
                for q in range(seq):
                    dist = np.abs(pos - q)
                    dists[l].append(float((attn[q] * dist).sum()))

    return [float(np.mean(d)) if d else 0.0 for d in dists]


# ─── Induction head detection ─────────────────────────────────────────────────

def induction_head_score(
    stores: list[ActivationStore],
) -> list[list[float]]:
    """
    Induction score for each (layer, head): the average attention weight on
    the token that follows the previous occurrence of the current token.
    (Olsson et al. 2022 "In-context Learning and Induction Heads")

    Returns scores[layer][head] ∈ [0, 1].
    """
    if not stores or not stores[0].attn_weights:
        return []

    n_layers = len(stores[0].attn_weights)
    n_heads  = stores[0].attn_weights[0].shape[0] if stores[0].attn_weights[0] is not None else NUM_HEADS

    scores = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    for s in stores:
        if s.input_ids is None:
            continue
        ids = s.input_ids[0].tolist()   # [seq]
        seq = len(ids)

        for l in range(n_layers):
            w = s.attn_weights[l]
            if w is None:
                continue
            for h in range(min(n_heads, w.shape[0])):
                attn = w[h].numpy()   # [seq, seq]
                # For each position q, find previous occurrence of ids[q]
                for q in range(1, seq):
                    target_tok = ids[q]
                    # look for previous occurrences at position k < q-1
                    # induction target = k+1 (the token *after* previous occurrence)
                    prev_occurrences = [k for k in range(q - 1) if ids[k] == target_tok]
                    if not prev_occurrences:
                        continue
                    # highest-weight induction target
                    induction_targets = [k + 1 for k in prev_occurrences if k + 1 < seq]
                    if not induction_targets:
                        continue
                    induction_weight = max(attn[q, t] for t in induction_targets)
                    scores[l][h].append(float(induction_weight))

    return [[float(np.mean(scores[l][h])) if scores[l][h] else 0.0
             for h in range(n_heads)]
            for l in range(n_layers)]


# ─── Full attention analysis ──────────────────────────────────────────────────

def run_full_attention_analysis(
    stores_A: list[ActivationStore],
    stores_B: list[ActivationStore],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> dict:
    result = {}

    if stores_A and stores_A[0].attn_weights and any(w is not None for w in stores_A[0].attn_weights):
        print("  Computing head divergences …")
        head_divs = compute_head_divergences(stores_A, stores_B)
        top_heads = top_divergent_heads(head_divs, top_k=10)

        # Serialise
        result["layer_mean_jsd"] = [
            float(np.mean([hd.mean_jsd for hd in layer])) if layer else 0.0
            for layer in head_divs
        ]
        result["top_divergent_heads"] = [
            {"layer": h.layer, "head": h.head, "mean_jsd": h.mean_jsd,
             "entropy_A": h.entropy_A, "entropy_B": h.entropy_B,
             "delta_entropy": h.delta_entropy}
            for h in top_heads
        ]
        result["head_jsd_matrix"] = [
            [hd.mean_jsd for hd in layer] for layer in head_divs
        ]

        print("  Computing attention distances …")
        result["attn_dist_A"] = mean_attention_distance(stores_A)
        result["attn_dist_B"] = mean_attention_distance(stores_B)

        print("  Computing induction head scores …")
        result["induction_A"] = induction_head_score(stores_A)
        result["induction_B"] = induction_head_score(stores_B)

    if tokenizer is not None:
        print("  Computing number-attention fractions …")
        result["num_attention"] = attention_to_numbers(stores_A, stores_B, tokenizer)

    return result


def save_attention_results(data: dict, tag: str = "") -> Path:
    out = OUTPUT_DIR / f"attention_analysis{('_'+tag) if tag else ''}.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved attention analysis → {out}")
    return out
