"""
logit_lens.py
-------------
Logit-lens analysis: project the residual stream at every layer through the
unembedding matrix to see *when* (in terms of network depth) the correct answer
token is being predicted.

Core question answered
----------------------
"At which layer does checkpoint A first 'commit' to the correct answer, and does
 checkpoint B commit to the wrong answer at an earlier layer or a later one?"

Method (Nostalgebraist 2020, Belrose et al. 2023 "Eliciting Latent Predictions")
---------------------------------------------------------------------------
  logit_lens(h_l) = LayerNorm_final(h_l) @ W_U

where  h_l  is the residual stream after layer l,  LayerNorm_final  is the model's
final layer norm, and  W_U  is the unembedding (lm_head) weight matrix.

We then compute  softmax(logit_lens(h_l))  and extract:
  - P(correct_answer_token  | layer l)
  - rank of the correct answer token at layer l
  - entropy of the distribution at layer l
  - top-5 predicted tokens at each layer

Results
-------
Returns a LayerwiseLensResult dataclass with per-problem, per-layer statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from hooks import ActivationStore, get_answer_token_ids, run_with_hooks
from config import NUM_LAYERS, OUTPUT_DIR


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SingleLensResult:
    """Logit-lens results for one (problem, model) pair."""
    problem_index:      int
    model_label:        str          # e.g. "step_96"
    answer:             str
    answer_token_ids:   list[int]

    # per-layer metrics — length = NUM_LAYERS + 1  (0=embed, 1..N=after layer)
    answer_prob:        list[float]  = field(default_factory=list)   # P(answer tok)
    answer_rank:        list[int]    = field(default_factory=list)   # rank of answer tok
    entropy:            list[float]  = field(default_factory=list)   # H of distribution
    top1_token:         list[str]    = field(default_factory=list)   # top-1 decoded token
    top1_prob:          list[float]  = field(default_factory=list)   # top-1 prob

    # position in prompt where we measure the lens
    # (default = last token of prompt = position -1)
    measure_position:   int = -1


@dataclass
class PairLensResult:
    """Logit-lens comparison between two checkpoints on one forgotten problem."""
    problem_index:  int
    task:           str
    answer:         str
    step_A:         int
    step_B:         int
    lens_A:         Optional[SingleLensResult] = None
    lens_B:         Optional[SingleLensResult] = None

    @property
    def divergence_layer(self) -> Optional[int]:
        """
        The first layer where the two models' predictions diverge substantially:
        i.e., where model A has top-1 = correct AND model B's answer_prob drops
        below 0.05.
        """
        if self.lens_A is None or self.lens_B is None:
            return None
        for l in range(len(self.lens_A.answer_prob)):
            if (self.lens_A.answer_prob[l] > 0.10 and
                    self.lens_B.answer_prob[l] < 0.05):
                return l
        return None

    @property
    def crossover_layer(self) -> Optional[int]:
        """
        The layer where model A first exceeds model B in answer probability
        (and stays above for the remaining layers).
        """
        if self.lens_A is None or self.lens_B is None:
            return None
        pA = self.lens_A.answer_prob
        pB = self.lens_B.answer_prob
        for l in range(len(pA)):
            if pA[l] > pB[l] and all(pA[ll] >= pB[ll] for ll in range(l, len(pA))):
                return l
        return None


# ─── Core computation ─────────────────────────────────────────────────────────

def _apply_lens_at_position(
    model:         PreTrainedModel,
    tokenizer:     PreTrainedTokenizerBase,
    store:         ActivationStore,
    answer_token_ids: list[int],
    position:      int = -1,
) -> tuple[list[float], list[int], list[float], list[str], list[float]]:
    """
    Apply logit-lens at every layer, returning per-layer metrics.

    Returns
    -------
    answer_probs, answer_ranks, entropies, top1_tokens, top1_probs
    """
    final_ln   = model.model.norm          # Qwen2 / LLaMA-family final LayerNorm
    lm_head    = model.lm_head             # Linear(hidden_dim, vocab)

    answer_probs = []
    answer_ranks = []
    entropies    = []
    top1_tokens  = []
    top1_probs   = []

    for l, h in enumerate(store.residuals):
        # h: [seq, hidden]  (already CPU float)
        h_pos = h[position].unsqueeze(0).unsqueeze(0)   # [1, 1, H]

        # Apply the model's final layer-norm
        h_pos_gpu = h_pos.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            h_normed = final_ln(h_pos_gpu)              # [1, 1, H]
            logits   = lm_head(h_normed).squeeze()      # [vocab]
            probs    = F.softmax(logits.float(), dim=-1)

        probs_cpu = probs.cpu()

        # Answer probability (max over all answer token ids)
        ans_p = max(probs_cpu[tid].item() for tid in answer_token_ids) if answer_token_ids else 0.0
        answer_probs.append(ans_p)

        # Answer rank
        sorted_ids = torch.argsort(probs_cpu, descending=True)
        min_rank = min(
            (sorted_ids == tid).nonzero(as_tuple=True)[0].item()
            for tid in answer_token_ids
        ) if answer_token_ids else -1
        answer_ranks.append(int(min_rank))

        # Entropy
        ent = (-probs_cpu * torch.log(probs_cpu + 1e-9)).sum().item()
        entropies.append(float(ent))

        # Top-1
        top1_id   = int(probs_cpu.argmax())
        top1_str  = tokenizer.decode([top1_id], skip_special_tokens=True).strip()
        top1_prob = probs_cpu[top1_id].item()
        top1_tokens.append(top1_str)
        top1_probs.append(float(top1_prob))

    return answer_probs, answer_ranks, entropies, top1_tokens, top1_probs


def compute_lens(
    model:      PreTrainedModel,
    tokenizer:  PreTrainedTokenizerBase,
    prompt:     str,
    answer:     str,
    model_label: str,
    problem_index: int,
    device:     str = "cuda",
    position:   int = -1,
) -> SingleLensResult:
    """
    Run the model on `prompt`, extract residual stream at every layer, and
    apply the logit lens to get per-layer answer probabilities.
    """
    ans_ids = get_answer_token_ids(tokenizer, answer)
    store   = run_with_hooks(model, tokenizer, prompt, device=device,
                             capture_attn_weights=False)

    probs, ranks, ents, top1_toks, top1_ps = _apply_lens_at_position(
        model, tokenizer, store, ans_ids, position=position
    )

    return SingleLensResult(
        problem_index    = problem_index,
        model_label      = model_label,
        answer           = answer,
        answer_token_ids = ans_ids,
        answer_prob      = probs,
        answer_rank      = ranks,
        entropy          = ents,
        top1_token       = top1_toks,
        top1_prob        = top1_ps,
        measure_position = position,
    )


def compute_pair_lens(
    model_A:     PreTrainedModel,
    model_B:     PreTrainedModel,
    tokenizer:   PreTrainedTokenizerBase,
    prompt:      str,
    answer:      str,
    problem_index: int,
    task:        str,
    step_A:      int,
    step_B:      int,
    device:      str = "cuda",
    device_A:    str | None = None,
    device_B:    str | None = None,
    position:    int = -1,
) -> PairLensResult:
    """Run logit-lens on both models and return a PairLensResult.

    device_A / device_B override `device` for each model independently,
    enabling dual-GPU setups (e.g. model_A on cuda:0, model_B on cuda:1).
    """
    dev_A = device_A or device
    dev_B = device_B or device
    lens_A = compute_lens(model_A, tokenizer, prompt, answer,
                          model_label=f"step_{step_A}",
                          problem_index=problem_index,
                          device=dev_A, position=position)
    lens_B = compute_lens(model_B, tokenizer, prompt, answer,
                          model_label=f"step_{step_B}",
                          problem_index=problem_index,
                          device=dev_B, position=position)
    return PairLensResult(
        problem_index=problem_index,
        task=task,
        answer=answer,
        step_A=step_A,
        step_B=step_B,
        lens_A=lens_A,
        lens_B=lens_B,
    )


# ─── Aggregate across problems ────────────────────────────────────────────────

def aggregate_lens_results(results: list[PairLensResult]) -> dict:
    """
    Compute layer-wise averages over all forgotten problems.

    Returns a dict:
    {
        "layers": [0, 1, ..., 28],
        "mean_prob_A":  [...],
        "mean_prob_B":  [...],
        "mean_rank_A":  [...],
        "mean_rank_B":  [...],
        "mean_entropy_A": [...],
        "mean_entropy_B": [...],
        "prob_gap":     [...],   # mean_prob_A - mean_prob_B
        "divergence_layers": [4, 7, ...],   # per-problem divergence layer
        "crossover_layers":  [...],
    }
    """
    if not results:
        return {}

    num_layers = len(results[0].lens_A.answer_prob)
    import statistics

    def mean_list(vals):
        return statistics.mean(vals) if vals else 0.0

    prob_A_by_layer   = [[] for _ in range(num_layers)]
    prob_B_by_layer   = [[] for _ in range(num_layers)]
    rank_A_by_layer   = [[] for _ in range(num_layers)]
    rank_B_by_layer   = [[] for _ in range(num_layers)]
    ent_A_by_layer    = [[] for _ in range(num_layers)]
    ent_B_by_layer    = [[] for _ in range(num_layers)]

    div_layers = []
    cross_layers = []

    for res in results:
        if res.lens_A is None or res.lens_B is None:
            continue
        for l in range(num_layers):
            prob_A_by_layer[l].append(res.lens_A.answer_prob[l])
            prob_B_by_layer[l].append(res.lens_B.answer_prob[l])
            rank_A_by_layer[l].append(res.lens_A.answer_rank[l])
            rank_B_by_layer[l].append(res.lens_B.answer_rank[l])
            ent_A_by_layer[l].append(res.lens_A.entropy[l])
            ent_B_by_layer[l].append(res.lens_B.entropy[l])
        dl = res.divergence_layer
        if dl is not None:
            div_layers.append(dl)
        cl = res.crossover_layer
        if cl is not None:
            cross_layers.append(cl)

    layers         = list(range(num_layers))
    mean_prob_A    = [mean_list(prob_A_by_layer[l]) for l in layers]
    mean_prob_B    = [mean_list(prob_B_by_layer[l]) for l in layers]
    mean_rank_A    = [mean_list(rank_A_by_layer[l]) for l in layers]
    mean_rank_B    = [mean_list(rank_B_by_layer[l]) for l in layers]
    mean_ent_A     = [mean_list(ent_A_by_layer[l]) for l in layers]
    mean_ent_B     = [mean_list(ent_B_by_layer[l]) for l in layers]
    prob_gap       = [mean_prob_A[l] - mean_prob_B[l] for l in layers]

    return {
        "layers":             layers,
        "mean_prob_A":        mean_prob_A,
        "mean_prob_B":        mean_prob_B,
        "mean_rank_A":        mean_rank_A,
        "mean_rank_B":        mean_rank_B,
        "mean_entropy_A":     mean_ent_A,
        "mean_entropy_B":     mean_ent_B,
        "prob_gap":           prob_gap,
        "divergence_layers":  div_layers,
        "crossover_layers":   cross_layers,
        "n_problems":         len(results),
    }


def save_lens_results(agg: dict, tag: str = "") -> Path:
    out = OUTPUT_DIR / f"logit_lens{('_'+tag) if tag else ''}.json"
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Saved logit-lens results → {out}")
    return out
