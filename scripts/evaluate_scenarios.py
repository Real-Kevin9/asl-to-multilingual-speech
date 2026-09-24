"""Run automated end-to-end scenario checks.

Objective pass/fail tests on fixed inputs — not a usability study.

    ./.venv/bin/python scripts/evaluate_scenarios.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.evaluation.scenarios import run_scenarios  # noqa: E402

DEFAULT_OUTPUT = "logs/evaluation/scenario_evaluation.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated pipeline scenario evaluation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", default="auto", help="Recognition backend.")
    parser.add_argument("--nlp", default="auto", help="NLP mode (auto/on/off).")
    args = parser.parse_args()

    print("Running automated scenarios (this loads TensorFlow + models)...")
    report = run_scenarios(recognition_backend=args.backend, nlp_mode=args.nlp)
    report["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nScenarios : {report['successes']}/{report['n_scenarios']} passed "
          f"({report['success_rate']})")
    for scenario in report["scenarios"]:
        mark = "ok" if scenario["success"] else "FAIL"
        print(f"  [{mark}] {scenario['id']:<16} ({scenario['objective']})")
        if scenario.get("error"):
            print(f"         error: {scenario['error']}")
        elif not scenario["success"] and scenario.get("details"):
            print(f"         {scenario['details']}")

    print(f"\nWrote {out_path}")
    print("Aggregate: ./.venv/bin/python scripts/evaluate_all.py --aggregate-only")


if __name__ == "__main__":
    main()
