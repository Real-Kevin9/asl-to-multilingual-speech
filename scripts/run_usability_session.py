"""Run one usability session and record the result.

Guides the facilitator through the consent check, the task list, the System
Usability Scale and the free-text questions, then writes a single JSON file per
participant to ``logs/usability/``.

Nothing identifying is stored. The participant is referenced by a pseudonym
(``P01``), and the file records only the group, task outcomes, ratings and any
comments the participant agreed to have written down. ``logs/`` is gitignored,
so session files stay on this machine and must **not** be shared externally —
only aggregate statistics go in the dissertation. See ``study/DATA_HANDLING.md``.

    ./.venv/bin/python scripts/run_usability_session.py --participant P01 --group deaf_hoh
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

from modules.evaluation.sus import (  # noqa: E402
    PARTICIPANT_GROUPS,
    SUS_STATEMENTS,
    adjective,
    score_responses,
)

DEFAULT_OUTPUT_DIR = "logs/usability"

# Each task exercises one proposal objective so a low SUS score can be traced to
# a stage rather than to the system as a whole.
TASKS = (
    {
        "id": "fingerspell",
        "objective": "Recognition",
        "prompt": "Fingerspell a three-letter word to the camera and confirm the "
                  "letters appear correctly on screen.",
    },
    {
        "id": "sentence",
        "objective": "NLP",
        "prompt": "Sign or type a short gloss, then check the English sentence the "
                  "system produces reads naturally.",
    },
    {
        "id": "speech_en",
        "objective": "TTS (English)",
        "prompt": "Have the system speak the sentence in English and confirm it is "
                  "intelligible.",
    },
    {
        "id": "speech_ne",
        "objective": "TTS (Nepali)",
        "prompt": "Switch to Nepali output and confirm speech is produced.",
    },
    {
        "id": "emotion",
        "objective": "Emotion",
        "prompt": "Repeat an utterance with a clear facial expression and note "
                  "whether the speech delivery changes.",
    },
)

LIKERT_HELP = "1 = strongly disagree, 5 = strongly agree"


def ask_int(prompt: str, low: int, high: int) -> int:
    while True:
        raw = input(f"{prompt} [{low}-{high}]: ").strip()
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        print(f"  Please enter a whole number between {low} and {high}.")


def ask_yes_no(prompt: str) -> bool:
    while True:
        raw = input(f"{prompt} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def run_tasks() -> list:
    results = []
    print("\n--- Tasks ---")
    print("Read each task aloud, then stay quiet and let the participant try it.\n")

    for i, task in enumerate(TASKS, start=1):
        print(f"Task {i}/{len(TASKS)} [{task['objective']}]")
        print(f"  {task['prompt']}")
        input("  Press Enter when the participant starts...")
        start = time.perf_counter()
        success = ask_yes_no("  Completed without facilitator help?")
        duration = time.perf_counter() - start
        note = input("  Observation (optional): ").strip()
        results.append({
            "id": task["id"],
            "objective": task["objective"],
            "success": success,
            "duration_s": round(duration, 1),
            "note": note or None,
        })
        print()
    return results


def run_sus() -> list:
    print("--- System Usability Scale ---")
    print(f"{LIKERT_HELP}\n")
    responses = []
    for i, statement in enumerate(SUS_STATEMENTS, start=1):
        print(f"{i}. {statement}")
        responses.append(ask_int("   Response", 1, 5))
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one usability session")
    parser.add_argument("--participant", required=True,
                        help="Pseudonym only, e.g. P01. Never a real name.")
    parser.add_argument("--group", required=True, choices=PARTICIPANT_GROUPS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--notes", default=None, help="Optional facilitator note.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"session_{args.participant}.json"
    if out_path.exists():
        print(f"{out_path} already exists. Use a different participant id.")
        sys.exit(1)

    print(f"\nUsability session — participant {args.participant} ({args.group})")
    print("=" * 60)
    print("Before starting, confirm the participant has read the information "
          "sheet and signed the consent form (study/consent_form.md).")
    if not ask_yes_no("Informed consent given and recorded?"):
        print("Cannot proceed without consent. Nothing was saved.")
        sys.exit(1)

    tasks = run_tasks()
    responses = run_sus()
    sus_score = score_responses(responses)

    print("\n--- Closing questions ---")
    fluency = ask_int(
        "How natural was the English the system produced? "
        "(1 = unnatural, 5 = fluent)", 1, 5
    )
    liked = input("What worked well? ").strip()
    disliked = input("What was frustrating? ").strip()
    suggestion = input("What would you change? ").strip()

    session = {
        "participant": args.participant,
        "group": args.group,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "consent_confirmed": True,
        "tasks": tasks,
        "sus_responses": responses,
        "sus_score": sus_score,
        "sus_adjective": adjective(sus_score),
        "fluency_rating": fluency,
        "comments": {
            "liked": liked or None,
            "disliked": disliked or None,
            "suggestion": suggestion or None,
        },
        "facilitator_notes": args.notes,
        "data_governance": {
            "storage": "local_only",
            "share_externally": False,
            "gitignored": True,
            "policy": "study/DATA_HANDLING.md",
        },
    }
    out_path.write_text(json.dumps(session, indent=2), encoding="utf-8")

    completed = sum(1 for t in tasks if t["success"])
    print("\n" + "=" * 60)
    print(f"SUS score : {sus_score:.1f} ({adjective(sus_score)})")
    print(f"Tasks     : {completed}/{len(tasks)} completed unaided")
    print(f"Saved     : {out_path} (local only — do not share this file)")
    print("Next:")
    print("  ./.venv/bin/python scripts/evaluate_usability.py")
    print("  ./.venv/bin/python scripts/export_usability_summary.py")


if __name__ == "__main__":
    main()
