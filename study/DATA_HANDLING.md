# Participant data handling

This study collects **pseudonymous** usability data for the dissertation. Raw
records stay on your machine. **Do not share session files, consent forms, or
identifying notes with anyone outside the research team** (including posting online,
sending to classmates, or uploading to cloud drives shared broadly).

The interim report commits to a formal evaluation including **SUS ≥ 80 with ≥ 10
participants** from deaf/HoH signers, interpreters, and hearing non-signers.
That is the **primary** usability method. Automated scenario checks
(`scripts/evaluate_scenarios.py`) are supplementary pipeline verification only.

## What is stored where

| Data | Location | In git? | Share externally? |
|------|----------|---------|-------------------|
| Raw session JSON (`session_P01.json`, …) | `logs/usability/` | **No** (gitignored) | **No** |
| Signed consent forms | Paper, locked away from laptop | **No** | **No** |
| Aggregated SUS report | `logs/evaluation/usability_sus.json` | **No** (under `logs/`) | **No** — cite numbers in dissertation only |
| Public summary (aggregates only) | `logs/evaluation/usability_public_summary.json` | **No** | **No** — writing aid for your report |
| Unified evaluation report | `logs/evaluation/unified_report.json` | **No** | **No** |

Nothing in `logs/` is committed. Even aggregate JSON files stay local unless you
manually copy **statistics** (means, counts) into the dissertation text.

## What never gets recorded

- No video of participants
- No audio recordings of participants
- No photographs of participants
- No names in digital files (pseudonyms `P01`, `P02`, … only)

The webcam feed is processed live and discarded.

## What goes in the dissertation

**Allowed (aggregate only):**

- Overall SUS mean, median, range, standard deviation
- Per-group SUS means and sample sizes (`deaf_hoh`, `interpreter`, `hearing`)
- Task success rates across all participants
- Anonymous paraphrased themes from comments (only where consent item 8 was signed)
- Statement that raw data are retained locally and not shared

**Not allowed in appendices or supplementary materials:**

- Individual `session_*.json` files
- Raw SUS response vectors tied to a participant code
- Consent forms
- Any material that could identify a participant

## Workflow

```bash
# Per participant (after consent on paper)
./.venv/bin/python scripts/run_usability_session.py --participant P01 --group deaf_hoh

# After each session or when all are done
./.venv/bin/python scripts/evaluate_usability.py
./.venv/bin/python scripts/export_usability_summary.py
./.venv/bin/python scripts/evaluate_all.py --aggregate-only
```

`export_usability_summary.py` builds an aggregate-only file for drafting the
evaluation chapter. It still lives under `logs/` and must not be uploaded or emailed.

## Recruitment targets

| Group | Code | Target |
|-------|------|--------|
| Deaf / hard-of-hearing signers | `deaf_hoh` | 4+ |
| Interpreters | `interpreter` | 3+ |
| Hearing non-signers | `hearing` | 3+ |
| **Total** | | **≥ 10** |

Proposal target: **SUS mean ≥ 80** overall (report per-group as well).

## Retention and destruction

- Raw session files: delete by **[DATE — after project is marked]**
- Consent forms: destroy separately by the same date
- Dissertation: retains aggregate statistics only; no raw data appendix

## If a participant withdraws

Email quote with their participant code within the withdrawal window stated on
the consent form. Delete `logs/usability/session_<code>.json`, re-run
`evaluate_usability.py` and `export_usability_summary.py`.

## Ethics

Fill in `participant_information_sheet.md` and `consent_form.md`, submit with
your institutional ethics application before session one. The information sheet
states that raw data are not shared outside the research project.
