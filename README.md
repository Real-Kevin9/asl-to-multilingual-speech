# Real-Time Context-Aware Sign Language → Multilingual Speech

Final-year project (CMP6200). Converts American Sign Language (ASL) gestures from a
webcam into grammatically correct, emotion-aware speech in **English and Nepali**.

This repository is a **complete end-to-end outline**: every stage of the pipeline is
implemented and wired together so the whole system runs start-to-finish. Individual
components are deliberately simple (a lightweight recognizer, rule-based grammar
fallback, off-the-shelf TTS) and are meant to be upgraded toward the proposal's
targets. Accuracy is *not* the goal yet — a working skeleton is.

## Pipeline

```
Webcam / images
      │
      ▼
1. Recognition      MediaPipe hand landmarks (63-d) → MLP classifier → sign labels
      │
      ▼
2. Gloss assembly   labels → text (handles space / del / nothing, debounces frames)
      │
      ▼
3. NLP correction   gloss → grammatical English (T5 if available, else rule-based)
      │
      ▼
4. Emotion          face image → {happy, sad, angry, neutral} → TTS prosody
      │
      ▼
5. Multilingual TTS English sentence → English + Nepali audio (gTTS + translation)
```

Each stage lives in its own module and is swappable:

| Stage | Module | Status |
|-------|--------|--------|
| Recognition | `modules/recognition/` | Working (MediaPipe + sklearn MLP baseline; upgrade path: CNN+LSTM) |
| Gloss assembly | `pipeline/pipeline.py` (`assemble_text`) | Working |
| NLP correction | `modules/nlp/correction.py` | T5 backend + rule-based fallback |
| Emotion detection | `modules/emotion/detector.py` | Face detection + classifier hook (fallback: neutral) |
| Multilingual TTS | `modules/tts/synthesizer.py` | gTTS (en/ne) + translation, offline-safe fallback |
| Orchestration | `pipeline/pipeline.py` (`ASLPipeline`) | Working, with per-stage latency logging |

## Setup

```bash
# Use the project virtual environment
./.venv/bin/python --version   # Python 3.12.x

# (Optional) fetch the MediaPipe hand model if it isn't present
./.venv/bin/python scripts/preprocess_asl.py --download-model --per-class-limit 1
```

The MediaPipe **hand_landmarker.task** bundle lives at
`models/recognition/hand_landmarker.task`. It is required for real hand-landmark
extraction; without it, preprocessing falls back to zero vectors.

## Running the prototype (all phases)

The **prototype runner** executes every project phase in order and prints phase-by-phase
input/output, even when accuracy is low or a stage is skipped.

```bash
# Text only — skips recognition (NLP → Emotion → TTS → Evaluation)
./.venv/bin/python scripts/prototype.py --text "hello world"

# Gloss only — skips recognition/preprocessing
./.venv/bin/python scripts/prototype.py --gloss "I GO STORE"

# Spell a word from bundled test images (full pipeline)
./.venv/bin/python scripts/prototype.py --spell HELLO

# Your own sign-frame images
./.venv/bin/python scripts/prototype.py --images frame1.jpg frame2.jpg

# Webcam capture
./.venv/bin/python scripts/prototype.py --webcam --frames 5
```

Phases executed:

| # | Phase | Module |
|---|--------|--------|
| 1 | Input capture | `pipeline/input.py` |
| 2 | Preprocessing (MediaPipe landmarks) | `modules/recognition/preprocess.py` |
| 3 | Recognition (sign labels) | `modules/recognition/model.py` |
| 4 | Gloss assembly + NLP correction | `pipeline/pipeline.py`, `modules/nlp/` |
| 5 | Emotion detection → prosody | `modules/emotion/detector.py` |
| 6 | Multilingual TTS (English + Nepali) | `modules/tts/synthesizer.py` |
| 7 | Evaluation & logging (stub) | `modules/evaluation/runner.py` |

Quick demo of all input modes:

```bash
./.venv/bin/python run_pipeline.py
```

## Running

### Full pipeline on bundled test images

```bash
# Spell a word using the bundled *_test.jpg images
./.venv/bin/python scripts/run_end_to_end.py --spell CAB

# Or pass your own ordered sign frames + a face image for emotion
./.venv/bin/python scripts/run_end_to_end.py \
    --images frame1.jpg frame2.jpg frame3.jpg \
    --face-image face.jpg
```

### Quick demo

```bash
./.venv/bin/python run_pipeline.py
```

### Real-time webcam demo

```bash
./.venv/bin/python scripts/realtime_demo.py
# [space] finish sentence · [c] clear · [q] quit
```

> Note: TTS (`gTTS`) and Nepali translation (`deep-translator`) need internet
> access. Offline, the pipeline still runs and writes the spoken text to
> `logs/tts/*.txt` instead of `.mp3`.

## Training the recognizer

```bash
# 1. Preprocess a subset of the ASL alphabet into hand-landmark features
./.venv/bin/python scripts/preprocess_asl.py --per-class-limit 100

# 2. Train the classifier (saves model + scaler + label encoder)
./.venv/bin/python scripts/train_recognition.py

# 3. Predict a single image
./.venv/bin/python scripts/infer_recognition.py data/raw/asl_alphabet/asl_alphabet_test/A_test.jpg
```

The bundled model was trained on 100 images/class (~2.9k samples) and reaches
~93% held-out accuracy on that subset. Increase `--per-class-limit` (or drop it
entirely) to train on the full corpus.

## Tests

```bash
./.venv/bin/python -m pytest -q
```

## Roadmap toward the proposal targets

- Recognition: replace the MLP baseline with the proposed **CNN + BiLSTM**, add
  dynamic-sign support (WLASL) beyond the static alphabet.
- NLP: fine-tune **T5** on ASLG-PC12 gloss→sentence pairs; evaluate with BLEU.
- Emotion: train the facial-expression CNN (FER-style) to hit ≥80% accuracy.
- TTS: add caching + async calls to keep end-to-end latency < 1.5 s.
- Evaluation: accuracy / F1 / BLEU / latency benchmarks + SUS user study.
