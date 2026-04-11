"""
run_experiment.py
-----------------
Main orchestrator for the mechanistic forgetting analysis.

Pipeline
--------
Step 1   Identify forgotten problems (from the provided sampling data)
Step 2   Select a primary checkpoint pair  (A = correct, B = forgotten)
Step 3   Load both models and tokenizer
Step 4   Build prompts for the forgotten problems
Step 5   Run all analyses:
            a.  Logit lens          (logit_lens.py)
            b.  Activation patching (activation_patching.py)
            c.  Representation     (representation_analysis.py)
            d.  Attention patterns  (attention_analysis.py)
Step 6   Save results and print a concise summary table

Usage
-----
    # Full run on GPU (recommended):
    python run_experiment.py

    # Quick test on CPU with 2 problems, logit-lens only:
    python run_experiment.py --device cpu --max-problems 2 --analyses logit_lens

    # Override the primary pair:
    python run_experiment.py --step-a 96 --step-b 128

    # Skip to visualization (results already saved):
    python run_experiment.py --skip-compute

Flags
-----
    --device        cuda | cpu  (default: cuda)
    --max-problems  int         (default: config.MAX_PROBLEMS_FOR_ACTIVATION)
    --step-a        int         (override primary pair A)
    --step-b        int         (override primary pair B)
    --task          str         (AIME | AIME25 | AMC | all)  default: all
    --analyses      comma-list  (logit_lens,patching,representation,attention,weight_drift)
    --skip-compute  flag        skip model loading; reuse saved JSON results
    --capture-attn  flag        also capture attention weights (slower, ~2x memory)
    --tag           str         suffix for output files
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── make local imports work regardless of CWD ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_STEPS, OUTPUT_DIR, FIGS_DIR,
    MAX_PROBLEMS_FOR_ACTIVATION, DEVICE, TASKS,
    ckpt_id, PRIMARY_PAIR,
)
from identify_forgotten import (
    find_all_forgotten, attach_sample_texts, best_pair, summarise,
    ForgottenProblem,
)
from logit_lens import (
    compute_pair_lens, aggregate_lens_results, save_lens_results,
    PairLensResult,
)
from activation_patching import (
    run_patching_for_problem, aggregate_patching_results, save_patching_results,
    ProblemPatchingResult,
)
from representation_analysis import (
    run_full_representation_analysis, save_representation_results,
)
from attention_analysis import (
    run_full_attention_analysis, save_attention_results,
)
from hooks import run_with_hooks


# ─── Prompt builder ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

def build_prompt(problem_text: str, tokenizer, apply_chat_template: bool = True) -> str:
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": problem_text},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{SYSTEM_PROMPT}\n\n{problem_text}\n"


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str, device: str = "cuda", dtype: str = "bfloat16"):
    """Load a HuggingFace model and tokenizer, returning (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.bfloat16)

    print(f"  Loading tokenizer from {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model from {model_id} …")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mechanistic forgetting analysis")
    parser.add_argument("--device",        default=DEVICE)
    parser.add_argument("--max-problems",  type=int, default=MAX_PROBLEMS_FOR_ACTIVATION)
    parser.add_argument("--step-a",        type=int, default=None)
    parser.add_argument("--step-b",        type=int, default=None)
    parser.add_argument("--task",          default="all")
    parser.add_argument("--analyses",      default="logit_lens,patching,representation,attention,weight_drift")
    parser.add_argument("--skip-compute",  action="store_true")
    parser.add_argument("--capture-attn",  action="store_true")
    parser.add_argument("--tag",           default="")
    args = parser.parse_args()

    active = set(args.analyses.split(","))
    tag    = args.tag

    # ── Step 1: identify forgotten problems ────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 1  Identifying forgotten problems")
    print("="*65)

    tasks_to_use = TASKS if args.task == "all" else [args.task.upper()]
    forgotten    = find_all_forgotten(tasks=tasks_to_use)
    attach_sample_texts(forgotten)
    summary = summarise(forgotten)
    print(json.dumps(summary, indent=2))

    # ── Step 2: select primary pair ────────────────────────────────────────────
    if args.step_a and args.step_b:
        primary = (args.step_a, args.step_b)
    elif PRIMARY_PAIR is not None:
        primary = PRIMARY_PAIR
    else:
        primary = best_pair(forgotten)
    print(f"\n→ Primary pair: step {primary[0]} → step {primary[1]}")

    # Filter to primary pair
    fp_list: list[ForgottenProblem] = [
        fp for fp in forgotten if (fp.step_A, fp.step_B) == primary
    ]

    # Optionally filter by task
    if args.task != "all":
        fp_list = [fp for fp in fp_list if fp.task == args.task.upper()]

    if not fp_list:
        print("No forgotten problems found for the selected pair/task. Exiting.")
        sys.exit(1)

    # Limit to max_problems
    fp_list = fp_list[: args.max_problems]
    print(f"  Using {len(fp_list)} forgotten problem(s) for activation analysis")

    if args.skip_compute:
        print("\n--skip-compute flag set; skipping model loading and activation analysis.")
        return

    # ── Step 3: load models ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 3  Loading models")
    print("="*65)

    id_A = ckpt_id(primary[0])
    id_B = ckpt_id(primary[1])

    model_A, tokenizer = load_model_and_tokenizer(id_A, args.device)
    model_B, _         = load_model_and_tokenizer(id_B, args.device)

    # ── Step 4: build prompts ──────────────────────────────────────────────────
    prompts = [build_prompt(fp.problem_text, tokenizer) for fp in fp_list]

    # ── Step 5a: Logit lens ────────────────────────────────────────────────────
    if "logit_lens" in active:
        print("\n" + "="*65)
        print("  STEP 5a  Logit lens analysis")
        print("="*65)

        lens_results: list[PairLensResult] = []
        for i, (fp, prompt) in enumerate(zip(fp_list, prompts)):
            print(f"  Problem {i+1}/{len(fp_list)}: [{fp.task}] Q{fp.problem_index}")
            pair_lens = compute_pair_lens(
                model_A, model_B, tokenizer,
                prompt=prompt,
                answer=fp.answer,
                problem_index=fp.problem_index,
                task=fp.task,
                step_A=primary[0],
                step_B=primary[1],
                device=args.device,
            )
            lens_results.append(pair_lens)
            # Print divergence layer
            dl = pair_lens.divergence_layer
            print(f"    Divergence layer: {dl}  "
                  f"P(ans|A@last): {pair_lens.lens_A.answer_prob[-1]:.4f}  "
                  f"P(ans|B@last): {pair_lens.lens_B.answer_prob[-1]:.4f}")

        agg_lens = aggregate_lens_results(lens_results)
        save_lens_results(agg_lens, tag)

        # Print summary
        peak_layer = int(max(range(len(agg_lens["prob_gap"])),
                             key=lambda i: agg_lens["prob_gap"][i]))
        print(f"\n  → Max probability gap at layer {peak_layer}")
        if agg_lens["divergence_layers"]:
            import statistics
            med_div = statistics.median(agg_lens["divergence_layers"])
            print(f"  → Median divergence layer: {med_div}")

    # ── Step 5b: Activation patching ──────────────────────────────────────────
    if "patching" in active:
        print("\n" + "="*65)
        print("  STEP 5b  Activation patching")
        print("="*65)

        patch_results: list[ProblemPatchingResult] = []
        for i, (fp, prompt) in enumerate(zip(fp_list, prompts)):
            print(f"  Problem {i+1}/{len(fp_list)}: [{fp.task}] Q{fp.problem_index}")
            pr = run_patching_for_problem(
                model_A, model_B, tokenizer,
                prompt=prompt,
                answer=fp.answer,
                problem_index=fp.problem_index,
                task=fp.task,
                step_A=primary[0],
                step_B=primary[1],
                device=args.device,
                verbose=True,
            )
            patch_results.append(pr)
            for target in pr.by_target:
                pk = pr.peak_layer(target)
                dp = pr.delta_p_curve(target)
                if pk is not None:
                    print(f"    Peak layer ({target}): {pk}  Δp={dp[pk]:+.4f}")

        agg_patch = aggregate_patching_results(patch_results)
        save_patching_results(agg_patch, tag)

        # Summary
        for target, dat in agg_patch.items():
            pk = int(max(dat["layers"], key=lambda l: dat["mean_delta_p"][l]))
            print(f"\n  → {target}: most critical layer = {pk}  "
                  f"mean Δp = {dat['mean_delta_p'][pk]:+.4f}")

    # ── Step 5c: Representation analysis ──────────────────────────────────────
    if "representation" in active:
        print("\n" + "="*65)
        print("  STEP 5c  Representation analysis")
        print("="*65)

        # Collect stores for both models across all problems
        stores_A = []
        stores_B = []
        for i, (fp, prompt) in enumerate(zip(fp_list, prompts)):
            print(f"  Collecting activations for problem {i+1}/{len(fp_list)} …")
            sA = run_with_hooks(model_A, tokenizer, prompt, device=args.device)
            sB = run_with_hooks(model_B, tokenizer, prompt, device=args.device)
            stores_A.append(sA)
            stores_B.append(sB)

        include_weight = "weight_drift" in active
        repr_data = run_full_representation_analysis(
            stores_A, stores_B,
            model_A=model_A if include_weight else None,
            model_B=model_B if include_weight else None,
        )
        save_representation_results(repr_data, tag)

        # Quick summary
        cos = repr_data.get("cosine_sim", [])
        if cos:
            min_cos_layer = int(min(range(len(cos)), key=lambda i: cos[i]))
            print(f"\n  → Lowest cosine sim at layer {min_cos_layer}: {cos[min_cos_layer]:.4f}")
        cka = repr_data.get("cka", [])
        if cka:
            min_cka_layer = int(min(range(len(cka)), key=lambda i: cka[i]))
            print(f"  → Lowest CKA at layer {min_cka_layer}: {cka[min_cka_layer]:.4f}")

    # ── Step 5d: Attention analysis ────────────────────────────────────────────
    if "attention" in active and args.capture_attn:
        print("\n" + "="*65)
        print("  STEP 5d  Attention pattern analysis")
        print("="*65)

        attn_stores_A = []
        attn_stores_B = []
        for i, (fp, prompt) in enumerate(zip(fp_list, prompts)):
            print(f"  Collecting attention for problem {i+1}/{len(fp_list)} …")
            sA = run_with_hooks(model_A, tokenizer, prompt, device=args.device,
                                capture_attn_weights=True)
            sB = run_with_hooks(model_B, tokenizer, prompt, device=args.device,
                                capture_attn_weights=True)
            attn_stores_A.append(sA)
            attn_stores_B.append(sB)

        attn_data = run_full_attention_analysis(attn_stores_A, attn_stores_B, tokenizer)
        save_attention_results(attn_data, tag)

        if "top_divergent_heads" in attn_data:
            print("\n  Top 5 most divergent heads:")
            for h in attn_data["top_divergent_heads"][:5]:
                print(f"    Layer {h['layer']:2d}, Head {h['head']:2d}  "
                      f"JSD={h['mean_jsd']:.4f}  "
                      f"ΔEntropy={h['delta_entropy']:+.3f}")
    elif "attention" in active and not args.capture_attn:
        print("\n  [attention analysis skipped — rerun with --capture-attn]")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  DONE  All results saved to:")
    print(f"    {OUTPUT_DIR}")
    print(f"\n  To generate paper figures, run:")
    print(f"    python visualize.py{(' --tag '+tag) if tag else ''}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
