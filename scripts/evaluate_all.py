"""Aggregate module metrics and optional latency into one unified report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.evaluation.unified import (  # noqa: E402
    MODULE_REPORT_PATHS,
    build_unified_report,
    load_module_reports,
    run_unified_evaluation,
    write_unified_report,
)

MODULE_SCRIPTS = {
    "nlp": ["scripts/evaluate_nlp.py"],
    "emotion": ["scripts/evaluate_emotion.py"],
    "recognition": ["scripts/evaluate_recognizers.py"],
    "user_recognition": ["scripts/evaluate_on_user_data.py"],
}


def _run_script(rel_path: str) -> bool:
    cmd = [sys.executable, str(repo_root / rel_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation across all modules")
    parser.add_argument("--log-dir", default="logs/evaluation")
    parser.add_argument(
        "--run-module-evals",
        action="store_true",
        help="Run missing per-module evaluate_* scripts before aggregating.",
    )
    parser.add_argument(
        "--run-latency",
        action="store_true",
        help="Run latency benchmark (TTS + gloss pipeline) and include in report.",
    )
    parser.add_argument(
        "--use-nlp-model",
        action="store_true",
        help="Use fine-tuned T5 when running latency benchmark.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only merge existing JSON reports (no benchmarks, no subprocess).",
    )
    parser.add_argument(
        "--run-scenarios",
        action="store_true",
        help="Run automated end-to-end scenario checks before aggregating.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.run_module_evals and not args.aggregate_only:
        bundle = load_module_reports(str(log_dir))
        for name in bundle["missing"]:
            if name in ("latency", "formative_usability", "scenario_evaluation", "usability"):
                continue
            for script in MODULE_SCRIPTS.get(name, []):
                ok = _run_script(script)
                if not ok:
                    print(f"Warning: {script} exited non-zero.")

    if args.run_scenarios and not args.aggregate_only:
        _run_script("scripts/evaluate_scenarios.py")

    if args.aggregate_only:
        report = build_unified_report(log_dir=str(log_dir))
        write_unified_report(report, log_dir=str(log_dir))
    else:
        report = run_unified_evaluation(
            log_dir=str(log_dir),
            run_latency=args.run_latency,
            use_nlp_model=args.use_nlp_model,
        )

    print(f"Unified report: {report.get('report_path', log_dir / 'unified_report.json')}")
    if report.get("missing_reports"):
        print("Still missing:", ", ".join(report["missing_reports"]))
    summary = report.get("module_summary") or {}
    if summary:
        print(json.dumps(summary, indent=2))
    for note in report.get("notes") or []:
        print(f"- {note}")


if __name__ == "__main__":
    main()
