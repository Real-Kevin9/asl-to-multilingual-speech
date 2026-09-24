# Post-Session Questionnaire

Paper fallback for when the session script cannot be used (remote sessions, or a
participant who prefers to fill it in themselves). Transfer the responses into
`scripts/run_usability_session.py` afterwards so they reach the analysis.

Participant code: `P___`  Group: deaf/HoH · interpreter · hearing (circle one)

## Part 1 — System Usability Scale

Circle one number per statement. **1 = strongly disagree, 5 = strongly agree.**
Answer with your first reaction. If you really cannot decide, circle 3.

| # | Statement | | | | | |
|---|---|---|---|---|---|---|
| 1 | I think that I would like to use this system frequently. | 1 | 2 | 3 | 4 | 5 |
| 2 | I found the system unnecessarily complex. | 1 | 2 | 3 | 4 | 5 |
| 3 | I thought the system was easy to use. | 1 | 2 | 3 | 4 | 5 |
| 4 | I think that I would need the support of a technical person to be able to use this system. | 1 | 2 | 3 | 4 | 5 |
| 5 | I found the various functions in this system were well integrated. | 1 | 2 | 3 | 4 | 5 |
| 6 | I thought there was too much inconsistency in this system. | 1 | 2 | 3 | 4 | 5 |
| 7 | I would imagine that most people would learn to use this system very quickly. | 1 | 2 | 3 | 4 | 5 |
| 8 | I found the system very cumbersome to use. | 1 | 2 | 3 | 4 | 5 |
| 9 | I felt very confident using the system. | 1 | 2 | 3 | 4 | 5 |
| 10 | I needed to learn a lot of things before I could get going with this system. | 1 | 2 | 3 | 4 | 5 |

Statements alternate between positive and negative wording on purpose, so the
answers should not all point the same way.

## Part 2 — Output quality

How natural was the English the system produced?

| Unnatural | | | | Fluent |
|---|---|---|---|---|
| 1 | 2 | 3 | 4 | 5 |

## Part 3 — In your own words

Anything you write here may be quoted anonymously, but only if you initialled
item 8 on the consent form.

**What worked well?**

<br><br>

**What was frustrating?**

<br><br>

**What would you change?**

<br><br>

---

*Do not write your name on this sheet.*

## Scoring (researcher only)

Do not score by hand. Enter the ten responses into
`scripts/run_usability_session.py`, which applies the standard rule: odd-numbered
statements score `response − 1`, even-numbered score `5 − response`, and the total
is multiplied by 2.5 to give 0–100. A SUS score is not a percentage; 68 is the
published average and 80 is this project's target.
