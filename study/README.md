# Usability Evaluation — SUS Study

The interim report commits to a **formal user experience study**: System
Usability Scale (SUS) with **≥ 10 participants** from three groups (deaf/HoH
signers, interpreters, hearing non-signers), target **SUS ≥ 80**, alongside
quantitative pipeline metrics (accuracy, BLEU, latency, confusion matrices).

This folder holds the participant-facing documents and the procedure. Technical
scoring lives in `modules/evaluation/sus.py`; session recording in
`scripts/run_usability_session.py`.

**Privacy:** raw session data stay on your machine, are gitignored, and are **not
shared with anyone else**. Only aggregate statistics appear in the dissertation.
See [`DATA_HANDLING.md`](DATA_HANDLING.md).

## Before session one

1. Fill in `[PLACEHOLDER]` fields in the information sheet and consent form.
2. Obtain ethics approval from your university.
3. Warm the models: `./.venv/bin/python scripts/demo.py --show-models`
4. Optional pre-check: `./.venv/bin/python scripts/evaluate_scenarios.py`

## Running sessions

```bash
# One session per participant — pseudonyms only (P01, P02, …), never real names
./.venv/bin/python scripts/run_usability_session.py --participant P01 --group deaf_hoh
./.venv/bin/python scripts/run_usability_session.py --participant P02 --group interpreter
./.venv/bin/python scripts/run_usability_session.py --participant P03 --group hearing
```

Groups: `deaf_hoh`, `interpreter`, `hearing`.

Follow [`facilitator_script.md`](facilitator_script.md) for the full procedure.

## After sessions

```bash
# Aggregate all sessions → usability_sus.json
./.venv/bin/python scripts/evaluate_usability.py

# Aggregate-only export for dissertation drafting (no participant codes)
./.venv/bin/python scripts/export_usability_summary.py

# Merge with recognition, NLP, emotion, latency metrics
./.venv/bin/python scripts/evaluate_all.py --aggregate-only
```

## Recruitment targets

| Group | Code | Minimum |
|-------|------|---------|
| Deaf / hard-of-hearing signers | `deaf_hoh` | 4 |
| Interpreters | `interpreter` | 3 |
| Hearing non-signers | `hearing` | 3 |
| **Total** | | **≥ 10** |

Report SUS **per group** as well as overall. A mean above 80 that hides a low
deaf/HoH score is a finding, not a success.

## Study documents

| File | Purpose |
|------|---------|
| `participant_information_sheet.md` | Given before consent |
| `consent_form.md` | Signed on paper, stored **outside** this repo |
| `facilitator_script.md` | Session procedure |
| `questionnaire.md` | Paper fallback if the script cannot be used |
| `DATA_HANDLING.md` | What stays local vs what goes in the report |

## What goes in the dissertation

- Overall and per-group SUS means, n, target comparison (≥ 80)
- Task success rates (aggregate)
- Anonymous themes from comments (only where consent item 8 was signed)
- Statement that raw data are retained locally and not shared externally

Do **not** attach individual session JSON files or consent forms.

## Supplementary checks (not a substitute for SUS)

`scripts/evaluate_scenarios.py` runs automated pass/fail checks on fixed pipeline
inputs. Use it to verify the system before sessions. It does **not** replace the
SUS study or satisfy the interim report's user-study requirement.

The formative heuristic review (`scripts/run_formative_review.py`) is optional
developer documentation only.

## Data protection summary

- No video, audio or images of participants are recorded.
- Pseudonyms only in digital files; names exist only on paper consent forms.
- `logs/usability/` and all session JSON are gitignored.
- Do not email, upload, or publish raw session files.
