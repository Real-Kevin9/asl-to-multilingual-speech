"""Export an aggregate-only SUS summary for dissertation writing.

Reads ``logs/evaluation/usability_sus.json`` (built from local session files)
and writes ``logs/evaluation/usability_public_summary.json`` with no participant
codes and no raw response vectors.

This file is a **writing aid** for your report. It is still stored under ``logs/``
(gitignored) and must not be emailed, uploaded, or shared as a file. Copy the
statistics into the dissertation text only.

    ./.venv/bin/python scripts/export_usability_summary.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.evaluation.privacy import build_public_summary  # noqa: E402

DEFAULT_INPUT = "logs/evaluation/usability_sus.json"
DEFAULT_OUTPUT = "logs/evaluation/usability_public_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export aggregate-only SUS summary (no participant identifiers)"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"No aggregated report at {in_path}.")
        print("Run scripts/evaluate_usability.py after recording sessions first.")
        sys.exit(1)

    full = json.loads(in_path.read_text(encoding="utf-8"))
    public = build_public_summary(full)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(public, indent=2), encoding="utf-8")

    overall = public["overall"]
    print(f"Participants : {public.get('participants')} (target >= 10)")
    print(f"SUS mean     : {overall.get('sus_mean')} ({overall.get('adjective')})")
    print(f"Target >= 80 : {'met' if overall.get('target_met') else 'NOT met'}")
    print(f"\nWrote aggregate summary: {out_path}")
    print("Use these numbers in the dissertation. Do not share this file externally.")


if __name__ == "__main__":
    main()
