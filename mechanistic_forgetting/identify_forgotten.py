"""
identify_forgotten.py
---------------------
Mine the pre-existing sampling data to find "forgotten" problems:
    problem p is forgotten at checkpoint pair (A, B) if:
        - at least one of the 64 responses from checkpoint A is correct, AND
        - ALL 64 responses from checkpoint B are incorrect.

Produces a structured dataset:
    ForgottenProblem(
        problem_text,
        answer,
        task,
        step_A,        # checkpoint that solved it
        step_B,        # checkpoint that forgot it
        correct_resp_A,  # one correct response from A
        wrong_resp_B,    # one (representative) wrong response from B
    )

Usage
-----
    python identify_forgotten.py                  # prints summary table
    python identify_forgotten.py --save           # also saves JSON to OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from config import (
    SAMPLING_ZIP, SAMPLING_DIR, OUTPUT_DIR,
    CHECKPOINT_STEPS, TASKS, ANALYSIS_PAIRS,
)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CheckpointResult:
    """All 64 responses for one problem at one checkpoint."""
    step: int
    task: str
    problem_index: int
    problem_text: str
    answer: str
    # list of (resp_idx, predicted_answer, is_correct)
    responses: list[dict]

    @property
    def any_correct(self) -> bool:
        return any(r["llm_check_result"] == 1 for r in self.responses)

    @property
    def all_wrong(self) -> bool:
        return all(r["llm_check_result"] != 1 for r in self.responses)

    @property
    def pass_at_1(self) -> float:
        """Fraction of first responses that are correct."""
        return sum(1 for r in self.responses if r["llm_check_result"] == 1) / len(self.responses)

    def first_correct_idx(self) -> Optional[int]:
        for r in self.responses:
            if r["llm_check_result"] == 1:
                return r["resp_idx"]
        return None


@dataclass
class ForgottenProblem:
    """A problem forgotten between checkpoint A and B."""
    problem_index: int
    problem_text: str
    answer: str
    task: str
    step_A: int
    step_B: int
    pass_at_1_A: float
    pass_at_1_B: float
    # Full response list (for loading the actual text later)
    responses_A: list[dict] = field(default_factory=list)
    responses_B: list[dict] = field(default_factory=list)
    # Sample texts (loaded separately from JSONL)
    sample_texts_A: list[str] = field(default_factory=list)
    sample_texts_B: list[str] = field(default_factory=list)


# ─── Data loading ─────────────────────────────────────────────────────────────

def _ckpt_folder_name(step: int) -> str:
    return f"UWNSL__Qwen2.5-7B-deepscaler_4k_step_{step}"


def _load_model_final_answer(step: int, task: str, zf: zipfile.ZipFile) -> Optional[list[dict]]:
    folder = _ckpt_folder_name(step)
    fname  = f"sampling_64_responses/{folder}/model_final_answer_{task}.json"
    try:
        return json.loads(zf.read(fname))
    except KeyError:
        return None


def _load_samples_jsonl(step: int, task: str, zf: zipfile.ZipFile) -> Optional[list[dict]]:
    """Load the JSONL file containing full response texts."""
    folder = _ckpt_folder_name(step)
    prefix = f"sampling_64_responses/{folder}/samples_{task}_"
    matches = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".jsonl")]
    if not matches:
        return None
    raw = zf.read(matches[0]).decode("utf-8", errors="replace")
    rows = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_checkpoint_results(step: int, task: str, zf: zipfile.ZipFile) -> Optional[list[CheckpointResult]]:
    data = _load_model_final_answer(step, task, zf)
    if data is None:
        return None
    results = []
    for item in data:
        results.append(CheckpointResult(
            step=step,
            task=task,
            problem_index=item["index"],
            problem_text=item["problem"],
            answer=str(item["answer"]),
            responses=item["responses"],
        ))
    return results


# ─── Core mining logic ────────────────────────────────────────────────────────

def find_forgotten_for_pair(
    step_A: int,
    step_B: int,
    task: str,
    zf: zipfile.ZipFile,
    require_pass1_A: float = 0.0,  # A must have pass@1 > this
) -> list[ForgottenProblem]:
    """Return problems where A solved at least once and B never solved."""
    res_A = load_checkpoint_results(step_A, task, zf)
    res_B = load_checkpoint_results(step_B, task, zf)
    if res_A is None or res_B is None:
        return []

    # Index by problem index
    idx_B = {r.problem_index: r for r in res_B}

    forgotten = []
    for rA in res_A:
        rB = idx_B.get(rA.problem_index)
        if rB is None:
            continue
        if rA.any_correct and rB.all_wrong:
            fp = ForgottenProblem(
                problem_index=rA.problem_index,
                problem_text=rA.problem_text,
                answer=rA.answer,
                task=task,
                step_A=step_A,
                step_B=step_B,
                pass_at_1_A=rA.pass_at_1,
                pass_at_1_B=rB.pass_at_1,
                responses_A=rA.responses,
                responses_B=rB.responses,
            )
            forgotten.append(fp)
    return forgotten


def find_all_forgotten(
    pairs: list[tuple[int, int]] | None = None,
    tasks: list[str] | None = None,
) -> list[ForgottenProblem]:
    """Mine all forgotten problems across all (step_A, step_B, task) combinations."""
    pairs = pairs or ANALYSIS_PAIRS
    tasks = tasks or TASKS

    # Ensure ZIP is extracted (or use ZipFile directly)
    zf = zipfile.ZipFile(SAMPLING_ZIP, "r")

    all_forgotten: list[ForgottenProblem] = []
    for step_A, step_B in pairs:
        for task in tasks:
            fp_list = find_forgotten_for_pair(step_A, step_B, task, zf)
            all_forgotten.extend(fp_list)

    zf.close()
    return all_forgotten


def attach_sample_texts(forgotten: list[ForgottenProblem]) -> None:
    """
    Attach the actual generated response texts from the JSONL files.
    Each ForgottenProblem.sample_texts_A / _B will be filled with
    a list of response strings (one per sample).
    """
    zf = zipfile.ZipFile(SAMPLING_ZIP, "r")

    # Cache: (step, task) → list of JSONL rows
    cache: dict[tuple, list] = {}

    for fp in forgotten:
        for step, attr in [(fp.step_A, "sample_texts_A"), (fp.step_B, "sample_texts_B")]:
            key = (step, fp.task)
            if key not in cache:
                rows = _load_samples_jsonl(step, fp.task, zf)
                cache[key] = rows or []

            rows = cache[key]
            # rows are ordered by doc_id; find the one matching problem_index
            matching = [r for r in rows if r.get("doc_id") == fp.problem_index
                        or r.get("doc", {}).get("id") == fp.problem_index]
            if not matching:
                # fallback: use positional index
                if fp.problem_index < len(rows):
                    matching = [rows[fp.problem_index]]

            if matching:
                row = matching[0]
                texts = row.get("resps", [[]])[0]   # list of 64 response strings
                setattr(fp, attr, texts)

    zf.close()


# ─── Summary helpers ──────────────────────────────────────────────────────────

def summarise(forgotten: list[ForgottenProblem]) -> dict:
    """Compute summary statistics for printing / saving."""
    from collections import defaultdict
    by_pair: dict[tuple, list] = defaultdict(list)
    by_task: dict[str, list]   = defaultdict(list)
    for fp in forgotten:
        by_pair[(fp.step_A, fp.step_B)].append(fp)
        by_task[fp.task].append(fp)

    return {
        "total_forgotten": len(forgotten),
        "by_pair": {f"{a}->{b}": len(v) for (a, b), v in sorted(by_pair.items())},
        "by_task": {t: len(v) for t, v in sorted(by_task.items())},
    }


def best_pair(forgotten: list[ForgottenProblem]) -> tuple[int, int]:
    """Return the (step_A, step_B) pair with the most forgotten problems."""
    from collections import Counter
    counts = Counter((fp.step_A, fp.step_B) for fp in forgotten)
    return counts.most_common(1)[0][0]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--attach-texts", action="store_true",
                        help="Also load and attach full response texts (slow)")
    args = parser.parse_args()

    print("Mining forgotten problems …")
    forgotten = find_all_forgotten()

    if args.attach_texts:
        print("Attaching response texts …")
        attach_sample_texts(forgotten)

    summary = summarise(forgotten)
    print(f"\n{'='*60}")
    print(f"  Total forgotten problems : {summary['total_forgotten']}")
    print(f"\n  By checkpoint pair:")
    for pair, cnt in summary["by_pair"].items():
        print(f"    {pair:>12}  ->  {cnt:>3} problems")
    print(f"\n  By task:")
    for task, cnt in summary["by_task"].items():
        print(f"    {task:<12}  ->  {cnt:>3} problems")
    print(f"{'='*60}\n")

    bp = best_pair(forgotten)
    print(f"Best pair for activation analysis: step {bp[0]} → step {bp[1]}")
    best_fp = [fp for fp in forgotten if (fp.step_A, fp.step_B) == bp]
    for fp in best_fp[:5]:
        print(f"  [{fp.task}] Q{fp.problem_index}: ans={fp.answer!r}  "
              f"pass@1_A={fp.pass_at_1_A:.2f}  pass@1_B={fp.pass_at_1_B:.2f}")

    if args.save:
        out_path = OUTPUT_DIR / "forgotten_problems.json"
        with open(out_path, "w") as f:
            json.dump([asdict(fp) for fp in forgotten], f, indent=2)
        print(f"\nSaved {len(forgotten)} records → {out_path}")


if __name__ == "__main__":
    main()
