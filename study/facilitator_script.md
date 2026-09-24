# Facilitator Script

Run every session the same way, so the scores are comparable. Total time is
20–30 minutes. Text in *italics* is spoken to the participant; text in brackets
is an instruction to you.

## Before the participant arrives

- [ ] Interpreter booked if the participant asked for one.
- [ ] Room lit from the front, plain background, webcam at chest height.
- [ ] Speakers on and at a comfortable volume; a hearing participant needs to
      judge the audio, a deaf participant may want to feel or see the output.
- [ ] Warm the models up so the participant does not sit through a cold start:

```bash
./.venv/bin/python scripts/demo.py --show-models
./.venv/bin/python scripts/demo.py --text "hello" --lang en
```

- [ ] Printed information sheet and consent form on the table.
- [ ] Next participant code decided (`P01`, `P02`, ...). Never use a real name.

## 1. Consent (5 minutes)

*Thanks for coming. Before anything else, please read this sheet. It explains
what the session involves and what happens to your answers. Take your time, and
ask me anything.*

[Wait. Then answer questions.]

*The main things: I am testing the software, not you. Nothing is video recorded.
You can stop at any point and you do not have to give a reason. Are you happy to
sign?*

[Collect the signed form. Put it away separately from your laptop. Only now start
the session script:]

```bash
./.venv/bin/python scripts/run_usability_session.py --participant P01 --group deaf_hoh
```

Groups are `deaf_hoh`, `interpreter`, or `hearing`.

## 2. Demonstration (3 minutes)

*Here is what the system does. I sign a letter to the camera, it recognises the
handshape, builds a sentence, and speaks it out loud in English or Nepali.*

[Fingerspell `C-A-T` yourself. Let the system speak. Show the Nepali toggle.]

*It only knows the manual alphabet and a small vocabulary, and it does get things
wrong. That is what I want to find out about.*

## 3. Tasks (10 minutes)

The script prompts you through five tasks, one per pipeline stage: fingerspelling,
sentence quality, English speech, Nepali speech, and expression.

For each one:

- Read the task aloud, then **stop talking**.
- Do not help, hint or point, even when it is uncomfortable. Silence is data.
- Only step in if the participant asks directly, or is stuck for more than about
  a minute. If you do step in, answer **n** to "completed without help".
- Note what they tried, not just whether it worked.

If the participant says something revealing, write it in the observation field
in their words, not your paraphrase.

## 4. Questionnaire (5 minutes)

*Now ten statements about the system. For each one, tell me how strongly you
agree, from 1, strongly disagree, to 5, strongly agree. Answer quickly with your
first reaction rather than thinking hard about it.*

[Read each statement as written. Do not explain or reword them — the scale only
works if everyone answers the same questions. If asked what a statement means,
say: *whatever it means to you.*]

Watch for straight-lining (all 4s or all 5s). Statements alternate between
positive and negative wording, so a participant agreeing with everything is
probably not reading them.

## 5. Closing questions (5 minutes)

The script asks for a fluency rating and three open questions. Let silence do the
work; people fill it with the most useful material.

Useful follow-ups:

- *You paused there — what were you expecting to happen?*
- *If a friend asked whether they should use this, what would you tell them?*
- *What would have to change before this was useful to you?*

## 6. After the participant leaves

- [ ] Session JSON written to `logs/usability/session_P01.json`.
- [ ] Consent form filed separately from the laptop.
- [ ] Anything the script did not capture noted while it is fresh.
- [ ] Re-aggregate:

```bash
./.venv/bin/python scripts/evaluate_usability.py
./.venv/bin/python scripts/evaluate_all.py --aggregate-only
```

## Recruitment targets

At least 10 participants overall, with all three groups represented:

| Group | Code | Target | Where to recruit |
|---|---|---|---|
| Deaf / hard-of-hearing signers | `deaf_hoh` | 4+ | Local deaf association, university disability service |
| Interpreters | `interpreter` | 3+ | Interpreter agencies, sign language tutors |
| Hearing non-signers | `hearing` | 3+ | Course peers, university noticeboards |

The deaf/HoH group is the hardest to recruit and the most important to the
findings. Start there, and allow several weeks.
