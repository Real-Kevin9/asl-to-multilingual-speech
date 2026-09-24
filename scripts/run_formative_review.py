"""Record a formative usability review (heuristics + developer walkthrough).

Use when a summative SUS study with ≥10 participants is not feasible. This does
**not** produce a SUS score and must not be reported as meeting the SUS ≥ 80 target.

    ./.venv/bin/python scripts/run_formative_review.py
    ./.venv/bin/python scripts/run_formative_review.py --non-interactive \\
        --heuristic-scores 0,1,2,0,1,2,1,0,2,1 \\
        --task-success 1,1,1,0,1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.evaluation.formative import (  # noqa: E402
    HEURISTICS,
    SEVERITY_LABELS,
    WALKTHROUGH_TASKS,
    build_formative_report,
)

DEFAULT_OUTPUT = "logs/evaluation/formative_usability.json"
DEFAULT_REASON = (
    "Summative SUS study with ≥10 participants was planned but participant "
    "recruitment was not feasible within the project timeline."
)


def ask_int(prompt: str, low: int, high: int) -> int:
    while True:
        raw = input(f"{prompt} [{low}-{high}]: ").strip()
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        print(f"  Enter a whole number between {low} and {high}.")


def ask_yes_no(prompt: str) -> bool:
    while True:
        raw = input(f"{prompt} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def run_walkthrough() -> list:
    results = []
    print("\n--- Developer walkthrough (5 tasks) ---")
    print("Run each task on the live system or via scripts/demo.py, then answer.\n")
    print("Tip: run ./.venv/bin/python scripts/evaluate_scenarios.py first for objective checks.\n")

    for i, task in enumerate(WALKTHROUGH_TASKS, start=1):
        print(f"Task {i}/{len(WALKTHROUGH_TASKS)} [{task['objective']}]")
        print(f"  {task['prompt']}")
        input("  Press Enter when you have tried the task...")
        start = time.perf_counter()
        success = ask_yes_no("  Did it work without workarounds?")
        note = input("  What happened? (optional): ").strip()
        results.append({
            "id": task["id"],
            "objective": task["objective"],
            "success": success,
            "duration_s": round(time.perf_counter() - start, 1),
            "note": note or None,
        })
        print()
    return results


def run_heuristics() -> tuple:
    print("--- Nielsen heuristic evaluation ---")
    print("Rate each heuristic by the **worst** issue you found:")
    for level, label in SEVERITY_LABELS.items():
        print(f"  {level} = {label}")
    print()

    scores = []
    notes = []
    for i, statement in enumerate(HEURISTICS, start=1):
        print(f"{i}. {statement}")
        scores.append(ask_int("   Worst severity found", 0, 4))
        note = input("   Note (optional): ").strip()
        notes.append(note)
    return scores, notes


def parse_csv_ints(raw: str, expected: int, name: str) -> list:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != expected:
        raise ValueError(f"{name} needs {expected} comma-separated values, got {len(parts)}.")
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Formative usability review (no participants)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--reason", default=DEFAULT_REASON)
    parser.add_argument("--evaluator", default="developer")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts; requires --heuristic-scores and --task-success.",
    )
    parser.add_argument(
        "--heuristic-scores",
        help=f"{len(HEURISTICS)} comma-separated severities (0-4).",
    )
    parser.add_argument(
        "--task-success",
        help=f"{len(WALKTHROUGH_TASKS)} comma-separated 0/1 task outcomes.",
    )
    args = parser.parse_args()

    if args.non_interactive:
        if not args.heuristic_scores or not args.task_success:
            print("Non-interactive mode needs --heuristic-scores and --task-success.")
            sys.exit(1)
        scores = parse_csv_ints(args.heuristic_scores, len(HEURISTICS), "heuristic-scores")
        successes = parse_csv_ints(args.task_success, len(WALKTHROUGH_TASKS), "task-success")
        walkthrough = [
            {
                "id": WALKTHROUGH_TASKS[i]["id"],
                "objective": WALKTHROUGH_TASKS[i]["objective"],
                "success": bool(successes[i]),
                "duration_s": None,
                "note": "non-interactive entry",
            }
            for i in range(len(WALKTHROUGH_TASKS))
        ]
        notes = [""] * len(HEURISTICS)
    else:
        print("\nFormative usability review (substitute for SUS participant study)")
        print("=" * 60)
        print("This records YOUR assessment. It is not a user study.\n")
        walkthrough = run_walkthrough()
        scores, notes = run_heuristics()

    report = build_formative_report(
        scores,
        notes,
        walkthrough,
        reason=args.reason,
        evaluator=args.evaluator,
    )
    report["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    h = report["heuristics"]
    w = report["walkthrough"]
    print("\n" + "=" * 60)
    print(f"Heuristics  : mean severity {h['mean_severity']}, "
          f"{h['actionable_issues']} actionable (≥2)")
    print(f"Walkthrough : {w['successes']}/{w['n_tasks']} tasks succeeded")
    print(f"Saved       : {out_path}")
    print("\nNext:")
    print("  ./.venv/bin/python scripts/evaluate_scenarios.py")
    print("  ./.venv/bin/python scripts/evaluate_all.py --aggregate-only")


if __name__ == "__main__":
    main()
