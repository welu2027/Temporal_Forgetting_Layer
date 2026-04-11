"""
hooks.py
--------
PyTorch forward-hook utilities for extracting and patching internal activations
of a Qwen2.5 (or any decoder-only transformer) model.

Key abstractions
----------------
ActivationStore
    Collects hidden states, attention outputs, and MLP outputs at every layer
    during a single forward pass.

PatchedForwardContext
    Context manager that, during a forward pass for model_B, replaces specified
    layer activations with pre-recorded activations from model_A.

Usage
-----
    store_A = run_with_hooks(model_A, tokenizer, prompt, device)
    # store_A.residuals[l]  → (seq_len, hidden_dim) tensor, layer l residual stream
    # store_A.attn_outs[l]  → attention output at layer l
    # store_A.mlp_outs[l]   → MLP output at layer l

    # Patching:  run model_B but replace layer 14's residual with model_A's
    patched = run_with_patch(model_B, tokenizer, prompt, store_A, patch_layer=14,
                             patch_target="residual", device=device)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ─── Activation store ─────────────────────────────────────────────────────────

@dataclass
class ActivationStore:
    """
    Stores per-layer activations captured during a single forward pass.

    Indexing convention
    -------------------
    Layer 0  = after the embedding lookup (residual before any transformer block)
    Layer l  = after transformer block l (1-indexed in most papers, but here
               we store index 0 = embed, index k = after block k-1 for k >= 1)

    Concretely for a 28-layer model:
        residuals[0]   = word embedding output   (shape: [seq, H])
        residuals[1]   = output of layer 0       (shape: [seq, H])
        ...
        residuals[28]  = output of layer 27 = final hidden state before LM head
        attn_outs[k]   = attention module output at layer k  (0-indexed)
        mlp_outs[k]    = MLP module output at layer k        (0-indexed)
        hidden_states  = same as residuals (alias for clarity)
    """
    # residual stream snapshots AFTER each layer (and at embed = before layer 0)
    residuals:   list[torch.Tensor] = field(default_factory=list)   # [num_layers+1]
    # attention sub-layer output (the *add* before the residual is added back)
    attn_outs:   list[torch.Tensor] = field(default_factory=list)   # [num_layers]
    # MLP sub-layer output
    mlp_outs:    list[torch.Tensor] = field(default_factory=list)   # [num_layers]
    # per-layer per-head attention weights  [num_layers, heads, seq, seq]
    attn_weights: list[torch.Tensor] = field(default_factory=list)  # [num_layers]

    # metadata
    input_ids:   Optional[torch.Tensor] = None
    prompt_text: str = ""
    logits:      Optional[torch.Tensor] = None   # final logits [seq, vocab]

    @property
    def num_layers(self) -> int:
        return len(self.attn_outs)

    def clear(self):
        self.residuals.clear()
        self.attn_outs.clear()
        self.mlp_outs.clear()
        self.attn_weights.clear()
        self.input_ids = None
        self.logits    = None


# ─── Hook installation helpers ────────────────────────────────────────────────

def _detach_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().cpu()


def install_capture_hooks(
    model: PreTrainedModel,
    store: ActivationStore,
    capture_attn_weights: bool = True,
) -> list:
    """
    Register forward hooks that populate `store` during the next forward pass.
    Returns the list of hook handles (call handle.remove() to clean up).

    Supports Qwen2 / LLaMA / Mistral-family architectures where:
        model.model.embed_tokens   → embedding
        model.model.layers[i]      → transformer block i
        model.model.layers[i].self_attn   → attention sub-module
        model.model.layers[i].mlp         → MLP sub-module
    """
    handles = []

    # --- embedding output  (residuals[0]) ---
    def embed_hook(module, inp, out):
        store.residuals.append(_detach_cpu(out[0] if isinstance(out, tuple) else out))

    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    for layer_idx, layer in enumerate(model.model.layers):

        # --- attention output  (attn_outs[layer_idx]) ---
        def attn_hook(module, inp, out, _li=layer_idx):
            # Qwen / LLaMA attention returns (hidden_states, attn_weights, past_kv) or
            # (hidden_states, past_kv) depending on output_attentions flag.
            if isinstance(out, tuple):
                hidden = out[0]
                if capture_attn_weights and len(out) > 1 and isinstance(out[1], torch.Tensor):
                    store.attn_weights.append(_detach_cpu(out[1]))
                elif capture_attn_weights and len(store.attn_weights) < _li + 1:
                    store.attn_weights.append(None)
            else:
                hidden = out
                if capture_attn_weights and len(store.attn_weights) < _li + 1:
                    store.attn_weights.append(None)
            store.attn_outs.append(_detach_cpu(hidden))

        handles.append(layer.self_attn.register_forward_hook(attn_hook))

        # --- MLP output  (mlp_outs[layer_idx]) ---
        def mlp_hook(module, inp, out, _li=layer_idx):
            store.mlp_outs.append(_detach_cpu(out[0] if isinstance(out, tuple) else out))

        handles.append(layer.mlp.register_forward_hook(mlp_hook))

        # --- residual stream after full layer block  (residuals[layer_idx+1]) ---
        def layer_hook(module, inp, out, _li=layer_idx):
            h = out[0] if isinstance(out, tuple) else out
            store.residuals.append(_detach_cpu(h))

        handles.append(layer.register_forward_hook(layer_hook))

    return handles


# ─── Full forward pass with activation capture ────────────────────────────────

@torch.no_grad()
def run_with_hooks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 512,
    capture_attn_weights: bool = False,
) -> ActivationStore:
    """
    Run a greedy forward pass and capture all layer activations.

    Note: we run the full prompt as a *prefill* pass (no autoregressive generation)
    so that the activations reflect the model's internal processing of the entire
    prompt.  For activation patching on the answer-generation step, call this
    separately with the prompt+answer tokens.
    """
    model.eval()
    store = ActivationStore(prompt_text=prompt)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    store.input_ids = enc["input_ids"].cpu()

    handles = install_capture_hooks(model, store, capture_attn_weights)
    try:
        out = model(**enc, output_attentions=capture_attn_weights)
        store.logits = out.logits[0].detach().float().cpu()   # [seq, vocab]
    finally:
        for h in handles:
            h.remove()

    return store


@torch.no_grad()
def run_with_generation_hooks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 512,
) -> tuple[ActivationStore, str]:
    """
    Run autoregressive generation and capture the prefill activations.
    Returns (store for prefill tokens, generated text).
    Useful for checking *what the model was about to generate*.
    """
    model.eval()
    store = ActivationStore(prompt_text=prompt)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    store.input_ids = enc["input_ids"].cpu()

    handles = install_capture_hooks(model, store)
    try:
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    finally:
        for h in handles:
            h.remove()

    generated = tokenizer.decode(
        gen_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return store, generated


# ─── Activation patching ──────────────────────────────────────────────────────

@torch.no_grad()
def run_with_patch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    source_store: ActivationStore,
    patch_layer: int,
    patch_target: str = "residual",
    device: str = "cuda",
    capture_attn_weights: bool = False,
) -> ActivationStore:
    """
    Run a forward pass for `model` on `prompt`, but at `patch_layer` replace
    the specified activation with the one from `source_store` (captured from a
    different forward pass, e.g. from model_A).

    Parameters
    ----------
    patch_layer   : which layer to patch (0-indexed for attention/mlp,
                    1-indexed for residual → residuals[patch_layer] replaces
                    the output of transformer block patch_layer-1)
    patch_target  : "residual"      → replace residual stream after the layer
                    "attention_out" → replace attention sub-layer output
                    "mlp_out"       → replace MLP sub-layer output

    Returns
    -------
    An ActivationStore with the activations of the patched forward pass.
    """
    model.eval()
    store = ActivationStore(prompt_text=prompt)
    enc   = tokenizer(prompt, return_tensors="pt").to(device)
    store.input_ids = enc["input_ids"].cpu()

    # Determine source tensor to inject (move to device for patching)
    if patch_target == "residual":
        src = source_store.residuals[patch_layer].to(device=device, dtype=model.dtype)
    elif patch_target == "attention_out":
        src = source_store.attn_outs[patch_layer].to(device=device, dtype=model.dtype)
    elif patch_target == "mlp_out":
        src = source_store.mlp_outs[patch_layer].to(device=device, dtype=model.dtype)
    else:
        raise ValueError(f"Unknown patch_target: {patch_target!r}")

    # Patch hooks
    patch_handles = []

    if patch_target == "residual":
        layer_module = model.model.layers[patch_layer - 1] if patch_layer > 0 else model.model.embed_tokens

        def _patch_residual(module, inp, out):
            h   = out[0] if isinstance(out, tuple) else out
            seq = h.shape[1]
            src_trimmed = src[:, :seq, :].to(h.dtype)
            h   = src_trimmed.expand_as(h)
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h
        patch_handles.append(layer_module.register_forward_hook(_patch_residual))

    elif patch_target == "attention_out":
        attn_module = model.model.layers[patch_layer].self_attn

        def _patch_attn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            seq = h.shape[1]
            src_trimmed = src[:, :seq, :].to(h.dtype)
            h = src_trimmed.expand_as(h)
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h
        patch_handles.append(attn_module.register_forward_hook(_patch_attn))

    elif patch_target == "mlp_out":
        mlp_module = model.model.layers[patch_layer].mlp

        def _patch_mlp(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            seq = h.shape[1]
            src_trimmed = src[:, :seq, :].to(h.dtype)
            h = src_trimmed.expand_as(h)
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h
        patch_handles.append(mlp_module.register_forward_hook(_patch_mlp))

    # Capture hooks for the patched run
    capture_handles = install_capture_hooks(model, store, capture_attn_weights)

    try:
        out = model(**enc, output_attentions=capture_attn_weights)
        store.logits = out.logits[0].detach().float().cpu()
    finally:
        for h in patch_handles + capture_handles:
            h.remove()

    return store


# ─── Utility: token probability extractor ─────────────────────────────────────

def answer_token_probs(
    logits: torch.Tensor,       # [seq, vocab]
    answer_token_ids: list[int],
    position: int = -1,         # which sequence position to look at
) -> dict[int, float]:
    """
    Given logits at a sequence position, return softmax probabilities for each
    token in `answer_token_ids`.
    """
    import torch.nn.functional as F
    probs = F.softmax(logits[position], dim=-1)
    return {tid: probs[tid].item() for tid in answer_token_ids}


def get_answer_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    answer: str,
    add_space: bool = True,
) -> list[int]:
    """
    Tokenize the answer string and return all token IDs that represent it.
    Handles common prefixes: "204", " 204", "\\boxed{204}", etc.
    """
    candidates = [answer]
    if add_space:
        candidates.append(" " + answer)
    candidates.append(f"\\boxed{{{answer}}}")
    candidates.append(f"\\boxed {{ {answer} }}")

    ids = set()
    for cand in candidates:
        toks = tokenizer.encode(cand, add_special_tokens=False)
        if toks:
            ids.add(toks[0])   # first token of the answer
    return list(ids)
