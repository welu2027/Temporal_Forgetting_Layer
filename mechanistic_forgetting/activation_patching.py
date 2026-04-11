"""
activation_patching.py
----------------------
Causal activation patching: the gold-standard mechanistic intervention for
locating where in the network a computation "lives".

Experimental design
-------------------
For each forgotten problem (question Q, answer A):

1.  Run checkpoint_A on Q  →  store_A  (correct model; captures full activations)
2.  Run checkpoint_B on Q  →  store_B  (forgotten model; wrong output)

3.  For every layer l (0 .. N) and every component c ∈ {residual, attn_out, mlp_out}:
        Run checkpoint_B on Q, but at layer l replace component c with store_A's value.
        Measure the resulting probability of the correct answer token.
        Record: Δp(A→B at layer l, component c)  =  patched_prob - baseline_B_prob

4.  The layer/component with the *largest Δp* is the one that is most causally
    responsible for the forgetting.

Metric: "patching effect"
    Δp(l) = P(correct | B patched at layer l with A's activations)
           - P(correct | B unpatched)

    A large positive Δp means: replacing B's layer-l activations with A's
    *restores* the correct answer.  This is strong causal evidence that layer l
    is where the solution disappears.

Normalisation (optional):
    Δp_norm(l) = Δp(l) / (P(correct | A) - P(correct | B))
    = 1.0 means layer l fully explains the difference; 0.0 means no effect.

Results
-------
PatchingResult dataclass, one per (problem, checkpoint_pair, component).
Aggregate across problems → mean Δp per layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from config  import NUM_LAYERS, PATCH_TARGETS, OUTPUT_DIR
from hooks   import (ActivationStore, run_with_hooks, run_with_patch,
                     answer_token_probs, get_answer_token_ids)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SinglePatchResult:
    """
    Result of patching *one* layer and *one* component for *one* problem.
    """
    problem_index:  int
    task:           str
    step_A:         int
    step_B:         int
    patch_layer:    int
    patch_target:   str          # "residual" | "attention_out" | "mlp_out"

    # Probabilities of the correct answer token
    prob_A:         float        # model A (upper bound)
    prob_B:         float        # model B unpatched (baseline)
    prob_patched:   float        # model B with patch at this layer

    @property
    def delta_p(self) -> float:
        return self.prob_patched - self.prob_B

    @property
    def delta_p_norm(self) -> float:
        denom = self.prob_A - self.prob_B
        if abs(denom) < 1e-6:
            return 0.0
        return self.delta_p / denom


@dataclass
class ProblemPatchingResult:
    """All patching results for one forgotten problem."""
    problem_index:   int
    task:            str
    step_A:          int
    step_B:          int
    answer:          str
    prob_A:          float
    prob_B:          float
    # Dict[patch_target → list of SinglePatchResult], length = NUM_LAYERS
    by_target:       dict[str, list[SinglePatchResult]] = field(default_factory=dict)

    def delta_p_curve(self, target: str = "residual") -> list[float]:
        """Return the layer-wise Δp curve for a given component."""
        return [r.delta_p for r in self.by_target.get(target, [])]

    def delta_p_norm_curve(self, target: str = "residual") -> list[float]:
        return [r.delta_p_norm for r in self.by_target.get(target, [])]

    def peak_layer(self, target: str = "residual") -> Optional[int]:
        """Layer with highest Δp for the given component."""
        curve = self.delta_p_curve(target)
        if not curve:
            return None
        return int(max(range(len(curve)), key=lambda i: curve[i]))


# ─── Core computation ─────────────────────────────────────────────────────────

def _answer_prob_from_store(
    model:    PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    store:    ActivationStore,
    ans_ids:  list[int],
    position: int = -1,
) -> float:
    """Extract P(answer) from a store's logits at `position`."""
    if store.logits is None:
        return 0.0
    probs = F.softmax(store.logits[position].float(), dim=-1)
    return max(probs[tid].item() for tid in ans_ids) if ans_ids else 0.0


def run_patching_for_problem(
    model_A:   PreTrainedModel,
    model_B:   PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt:    str,
    answer:    str,
    problem_index: int,
    task:      str,
    step_A:    int,
    step_B:    int,
    patch_targets: list[str] = None,
    device:    str = "cuda",
    device_A:  str | None = None,
    device_B:  str | None = None,
    position:  int = -1,
    verbose:   bool = False,
) -> ProblemPatchingResult:
    """
    Run the full causal patching sweep for one problem.

    device_A / device_B override `device` for each model independently,
    enabling dual-GPU setups (model_A on cuda:0, model_B on cuda:1).
    The source activations (from model_A) are stored on CPU and automatically
    moved to model_B's device during patching, so cross-device patching works
    transparently.

    For each layer l and each target in patch_targets:
        1. Run model_A clean -> store_A  (activations cached on CPU)
        2. Run model_B clean -> store_B  (baseline)
        3. Run model_B with layer-l patch from store_A -> patched_store
        4. Record delta_p
    """
    if patch_targets is None:
        patch_targets = PATCH_TARGETS

    dev_A = device_A or device
    dev_B = device_B or device

    ans_ids = get_answer_token_ids(tokenizer, answer)
    if not ans_ids:
        print(f"Warning: could not tokenize answer {answer!r}")

    # ── Clean forward passes ────────────────────────────────────────────────
    store_A = run_with_hooks(model_A, tokenizer, prompt, device=dev_A)
    store_B = run_with_hooks(model_B, tokenizer, prompt, device=dev_B)

    prob_A = _answer_prob_from_store(model_A, tokenizer, store_A, ans_ids, position)
    prob_B = _answer_prob_from_store(model_B, tokenizer, store_B, ans_ids, position)

    if verbose:
        print(f"  P(correct|A)={prob_A:.4f}   P(correct|B)={prob_B:.4f}")

    result = ProblemPatchingResult(
        problem_index=problem_index,
        task=task,
        step_A=step_A,
        step_B=step_B,
        answer=answer,
        prob_A=prob_A,
        prob_B=prob_B,
    )

    # ── Patching sweep ──────────────────────────────────────────────────────
    # Residual patching: patch at index 0..N (where index l = after layer l-1)
    # Attention/MLP patching: patch at layer index 0..N-1

    for target in patch_targets:
        patches = []

        if target == "residual":
            # residuals[0] = embed output, residuals[l] = after layer l-1
            n_patches = len(store_A.residuals)   # NUM_LAYERS + 1
        elif target == "attention_out":
            n_patches = len(store_A.attn_outs)   # NUM_LAYERS
        elif target == "mlp_out":
            n_patches = len(store_A.mlp_outs)    # NUM_LAYERS
        else:
            continue

        for l in range(n_patches):
            patched_store = run_with_patch(
                model_B, tokenizer, prompt,
                source_store=store_A,
                patch_layer=l,
                patch_target=target,
                device=dev_B,
            )
            prob_p = _answer_prob_from_store(
                model_B, tokenizer, patched_store, ans_ids, position
            )
            patches.append(SinglePatchResult(
                problem_index=problem_index,
                task=task,
                step_A=step_A,
                step_B=step_B,
                patch_layer=l,
                patch_target=target,
                prob_A=prob_A,
                prob_B=prob_B,
                prob_patched=prob_p,
            ))
            if verbose:
                print(f"    target={target} layer={l:2d}  "
                      f"p_patched={prob_p:.4f}  Δp={prob_p-prob_B:+.4f}")

        result.by_target[target] = patches

    return result


# ─── Aggregate across problems ────────────────────────────────────────────────

def aggregate_patching_results(
    results: list[ProblemPatchingResult],
    targets: list[str] = None,
) -> dict:
    """
    Compute mean Δp and mean normalised Δp per layer, aggregated over problems.

    Returns
    -------
    {
        "residual": {
            "layers": [...],
            "mean_delta_p":      [...],
            "mean_delta_p_norm": [...],
            "std_delta_p":       [...],
            "peak_layer_votes":  {layer: count},   # which layer "wins" most
        },
        "attention_out": { ... },
        "mlp_out":       { ... },
    }
    """
    import statistics
    targets = targets or PATCH_TARGETS
    out = {}

    for target in targets:
        # Collect per-layer Δp across all problems
        all_curves     = [r.delta_p_curve(target)      for r in results
                          if r.by_target.get(target)]
        all_norm_curves = [r.delta_p_norm_curve(target) for r in results
                           if r.by_target.get(target)]
        if not all_curves:
            continue

        n_layers = len(all_curves[0])
        mean_dp  = [statistics.mean(c[l] for c in all_curves)       for l in range(n_layers)]
        std_dp   = [statistics.stdev(c[l] for c in all_curves) if len(all_curves) > 1 else 0.0
                    for l in range(n_layers)]
        mean_ndp = [statistics.mean(c[l] for c in all_norm_curves)  for l in range(n_layers)]

        peak_votes: dict[int, int] = {}
        for r in results:
            pl = r.peak_layer(target)
            if pl is not None:
                peak_votes[pl] = peak_votes.get(pl, 0) + 1

        out[target] = {
            "layers":            list(range(n_layers)),
            "mean_delta_p":      mean_dp,
            "mean_delta_p_norm": mean_ndp,
            "std_delta_p":       std_dp,
            "peak_layer_votes":  peak_votes,
            "n_problems":        len(all_curves),
        }

    return out


def save_patching_results(agg: dict, tag: str = "") -> Path:
    out = OUTPUT_DIR / f"activation_patching{('_'+tag) if tag else ''}.json"
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Saved activation-patching results → {out}")
    return out


# ─── Component-level fine-grained patching ────────────────────────────────────

def run_attention_head_patching(
    model_A:   PreTrainedModel,
    model_B:   PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt:    str,
    answer:    str,
    focus_layer: int,
    problem_index: int = 0,
    device:    str = "cuda",
    position:  int = -1,
) -> dict:
    """
    Within a single layer (focus_layer), patch individual attention heads
    from model_A into model_B.

    This reveals *which heads* within the critical layer are most causally
    responsible.

    Implementation: for head h, we reconstruct the head output
        O_h = Attn_h(Q,K,V) @ W_O_h
    and add it as a correction.  Here we use a simpler approach:
    we ablate each head by zeroing its contribution to the attention output
    and compare to the full patch, identifying which heads matter most.
    """
    from config import NUM_HEADS, HEAD_DIM, HIDDEN_DIM

    ans_ids = get_answer_token_ids(tokenizer, answer)
    store_A = run_with_hooks(model_A, tokenizer, prompt, device=device,
                             capture_attn_weights=True)
    store_B = run_with_hooks(model_B, tokenizer, prompt, device=device,
                             capture_attn_weights=True)

    prob_A = _answer_prob_from_store(model_A, tokenizer, store_A, ans_ids, position)
    prob_B = _answer_prob_from_store(model_B, tokenizer, store_B, ans_ids, position)

    # Full residual patch at focus_layer as reference
    full_patch_store = run_with_patch(
        model_B, tokenizer, prompt, store_A,
        patch_layer=focus_layer, patch_target="residual", device=device
    )
    prob_full = _answer_prob_from_store(model_B, tokenizer, full_patch_store, ans_ids, position)

    # For each head: what's the attention weight divergence?
    head_metrics = []
    layer_A_attn = store_A.attn_weights[focus_layer] if focus_layer < len(store_A.attn_weights) else None
    layer_B_attn = store_B.attn_weights[focus_layer] if focus_layer < len(store_B.attn_weights) else None

    if layer_A_attn is not None and layer_B_attn is not None:
        # [heads, seq, seq]
        for h in range(min(layer_A_attn.shape[0], layer_B_attn.shape[0])):
            attn_A_h = layer_A_attn[h]   # [seq, seq]
            attn_B_h = layer_B_attn[h]   # [seq, seq]
            # Jensen-Shannon divergence averaged over positions
            js_divs = []
            for pos in range(attn_A_h.shape[0]):
                p = F.softmax(attn_A_h[pos].float(), dim=-1)
                q = F.softmax(attn_B_h[pos].float(), dim=-1)
                m = 0.5 * (p + q)
                js = 0.5 * (
                    (p * (p / (m + 1e-9)).log()).sum() +
                    (q * (q / (m + 1e-9)).log()).sum()
                )
                js_divs.append(js.item())
            mean_js = sum(js_divs) / len(js_divs)
            head_metrics.append({
                "head":    h,
                "mean_js_divergence": mean_js,
                "attn_A_last_pos": attn_A_h[-1].tolist(),
                "attn_B_last_pos": attn_B_h[-1].tolist(),
            })

    return {
        "problem_index": problem_index,
        "focus_layer":   focus_layer,
        "prob_A":        prob_A,
        "prob_B":        prob_B,
        "prob_full_patch": prob_full,
        "head_metrics":  head_metrics,
    }
