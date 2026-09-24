"""Aggregate usability sessions into a single evaluation report.

Reads every ``session_*.json`` written by ``run_usability_session.py`` and
produces ``logs/evaluation/usability_sus.json``, which the unified report picks
up alongside the recognition, NLP, emotion and latency metrics.

Reports SUS per participant group as well as overall, because a mean that clears
80 can still hide a group the system does not serve.

    ./.venv/bin/python scripts/evaluate_usability.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.evaluation.sus import score_responses, summarise_by_group  # noqa: E402

DEFAULT_SESSION_DIR = "logs/usability"
DEFAULT_OUTPUT = "logs/evaluation/usability_sus.json"


def load_sessions(session_dir: Path) -> list:
    sessions = []
    for path in sorted(session_dir.glob("session_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  skipping unreadable file: {path.name}")
            continue

        # Recompute rather than trusting the stored score, so a hand-edited file
        # cannot quietly change the headline number.
        responses = data.get("sus_responses")
        if responses:
            try:
                data["sus_score"] = score_responses(responses)
            except ValueError as exc:
                print(f"  skipping {path.name}: {exc}")
                continue
        elif data.get("sus_score") is None:
            print(f"  skipping {path.name}: no SUS responses")
            continue

        sessions.append(data)
    return sessions


def task_breakdown(sessions: list) -> dict:
    """Per-task success rate, so failures point at a specific pipeline stage."""
    attempts: Counter = Counter()
    successes: Counter = Counter()
    objectives = {}
    for session in sessions:
        for task in session.get("tasks") or []:
            task_id = task.get("id")
            if not task_id:
                continue
            attempts[task_id] += 1
            if task.get("success"):
                successes[task_id] += 1
            objectives.setdefault(task_id, task.get("objective"))

    return {
        task_id: {
            "objective": objectives.get(task_id),
            "attempts": n,
            "successes": successes[task_id],
            "success_rate": round(successes[task_id] / n, 3),
        }
        for task_id, n in sorted(attempts.items())
    }


def collect_comments(sessions: list) -> list:
    comments = []
    for session in sessions:
        entries = session.get("comments") or {}
        if any(entries.values()):
            comments.append({
                "participant": session.get("participant"),
                "group": session.get("group"),
                **{k: v for k, v in entries.items() if v},
            })
    return comments


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate usability sessions")
    parser.add_argument("--session-dir", default=DEFAULT_SESSION_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"No session directory at {session_dir}.")
        print("Run scripts/run_usability_session.py for at least one participant first.")
        sys.exit(1)

    sessions = load_sessions(session_dir)
    if not sessions:
        print(f"No usable sessions found in {session_dir}.")
        sys.exit(1)

    report = summarise_by_group(sessions)
    report["tasks"] = task_breakdown(sessions)
    report["qualitative"] = collect_comments(sessions)
    report["session_dir"] = str(session_dir)
    report["_data_governance"] = {
        "raw_sessions": "logs/usability/ — local only, never commit or share as files",
        "external_sharing": "prohibited — aggregate statistics only in dissertation",
        "public_export_script": "scripts/export_usability_summary.py",
        "policy": "study/DATA_HANDLING.md",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["overall"]
    print(f"\nParticipants : {report['participants']} (target >= 10)")
    print(f"SUS mean     : {overall['mean']} ({overall['adjective']})")
    print(f"SUS range    : {overall['min']} - {overall['max']}, sd {overall['stdev']}")
    print(f"Target >= 80 : {'met' if overall['target_met'] else 'NOT met'}")

    if report["by_group"]:
        print("\nBy group:")
        for group, stats in report["by_group"].items():
            print(f"  {group:<12} n={stats['n']:<3} mean={stats['mean']}")
    if report["groups_missing"]:
        print(f"\nGroups not yet recruited: {', '.join(report['groups_missing'])}")
    if report.get("tasks"):
        print("\nTask success:")
        for task_id, stats in report["tasks"].items():
            print(f"  {task_id:<12} {stats['successes']}/{stats['attempts']} "
                  f"({stats['objective']})")

    print(f"\nWrote {out_path}")
    print("Export aggregate-only summary (for dissertation drafting):")
    print("  ./.venv/bin/python scripts/export_usability_summary.py")


if __name__ == "__main__":
    main()
