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
1. Recognition      MediaPipe hand crop → MobileNetV2 → BiLSTM ×2 → sign labels
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
| Recognition | `modules/recognition/` | Working — **MobileNetV2 + BiLSTM** (default) with the landmark MLP as fallback |
| Gloss assembly | `pipeline/pipeline.py` (`assemble_text`) | Working |
| NLP correction | `modules/nlp/correction.py` | Fine-tuned **t5-small** gloss→English (ASLG-PC12) + rule-based fallback |
| Emotion detection | `modules/emotion/detector.py` | Haar face + **trained CNN** (happy/sad/angry/neutral) → TTS prosody |
| Multilingual TTS | `modules/tts/synthesizer.py` | gTTS (en/ne) + translation, **disk cache**, parallel synthesis |
| Orchestration | `pipeline/pipeline.py` (`ASLPipeline`) | Working, with per-stage latency logging |
| Evaluation | `modules/evaluation/` | Per-run logs + **unified report** (`evaluate_all.py`) |
| Usability study | `study/`, `modules/evaluation/sus.py` | SUS study (≥10 users, ≥80); local-only data policy |

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

## One command, best models

`scripts/demo.py` resolves the strongest artefact for every stage from disk, so
there is no need to remember which flags select which model. It prints what it
loaded before running.

```bash
# Live webcam with the best recogniser, fine-tuned T5, and the emotion CNN
./.venv/bin/python scripts/demo.py

# Same models, other input modes
./.venv/bin/python scripts/demo.py --spell HELLO
./.venv/bin/python scripts/demo.py --gloss "I GO STORE"
./.venv/bin/python scripts/demo.py --images frame1.jpg frame2.jpg --face-image face.jpg

# Just report which artefacts would be used
./.venv/bin/python scripts/demo.py --show-models
```

```
Models in use:
  ok recognition  hybrid_user    val_accuracy=94.2%, user_holdout_accuracy=93.7%
  ok nlp          t5_gloss       bleu=57.62
  ok emotion      cnn            val_accuracy=65.3%, test_accuracy=65.5%
  ok tts          gtts_cached    (languages: en, ne)
```

Selection is re-checked on every run, so retraining a stage is picked up
automatically. Override any stage when needed:

## Web dashboard (React + FastAPI)

Browser UI with the camera in the centre, recognition/TTS panels on the sides,
and a **Speech → Sign** tab that shows a coming-soon screen.

```bash
./.venv/bin/pip install -r app/requirements-web.txt
cd app/web && npm install && cd ../../
./.venv/bin/python scripts/run_web_app.py --dev   # UI http://127.0.0.1:5173
# or
./.venv/bin/python scripts/run_web_app.py         # single server http://127.0.0.1:8000
```

See `app/README.md` for layout and API details.

| Flag | Default | Effect |
|------|---------|--------|
| `--backend` | `auto` | Force a recogniser (`hybrid_user`, `hybrid`, `cnn_lstm`, `landmark_mlp`) |
| `--nlp` | `auto` | `auto` = local T5 if present, `on` = force T5, `off` = rule-based |
| `--languages` | `en ne` | TTS output languages |

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
| 7 | Evaluation & logging | `modules/evaluation/` |

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

(`scripts/demo.py` runs the same loop with the best models pre-selected; use
`realtime_demo.py` directly when tuning the dwell-gate flags below.)

A sign is accepted only after it is **held steadily for ~0.9 s**, shown by the
progress bar at the bottom of the window. This deliberate dwell stops frames
captured mid-movement from being written out as letters. Continuing to hold a
sign produces a double letter (`LL`) rather than a stream of repeats.

| Flag | Default | Effect |
|------|---------|--------|
| `--hold-time` | `0.9` | Seconds a sign must be held before it is accepted |
| `--min-confidence` | `0.6` | Mean confidence required to accept a held sign |
| `--agreement` | `0.65` | Fraction of the hold window that must agree |
| `--cooldown` | `0.5` | Pause after a commit (enables double letters) |
| `--predict-interval` | `0.12` | Seconds between predictions |

```bash
# Slower and stricter, if letters are still being committed too eagerly
./.venv/bin/python scripts/realtime_demo.py --hold-time 1.2 --min-confidence 0.75

# Compare against the old landmark model
./.venv/bin/python scripts/realtime_demo.py --backend landmark_mlp
```

> Note: TTS (`gTTS`) and Nepali translation (`deep-translator`) need internet
> access. Offline, the pipeline still runs and writes the spoken text to
> `logs/tts/*.txt` instead of `.mp3`.

## Recognition backends

Four recognisers are available and share one interface. Select with `--backend`:

| Backend | Architecture | Input | Notes |
|---------|--------------|-------|-------|
| `hybrid_user` | `hybrid`, fine-tuned on your own webcam | hand crop + landmarks | Best on your camera; needs recorded samples |
| `hybrid` | **MobileNetV2 → 2-layer BiLSTM + geometry stream** | hand crop + landmarks | Most robust across camera framings |
| `cnn_lstm` | MobileNetV2 → 2-layer BiLSTM | hand crop (160×160) | Strong on close-ups, weak on webcam framing |
| `landmark_mlp` | MLP over 63-d landmarks | MediaPipe landmarks | Original baseline; kept permanently |
| `auto` | `hybrid_user` → `hybrid` → `cnn_lstm` → `landmark_mlp` | — | Default |

**Adapt the model to your camera.** Every model loses roughly 35 accuracy points
when moved to a webcam absent from its training data — the corpus `hybrid` scores
97.8 % on held-out corpus frames but only 62.4 % on newly recorded ones. Recording
a few hundred of your own samples and fine-tuning recovers most of that
(62.4 % → 93.7 % on held-out user samples):

```bash
./.venv/bin/python scripts/capture_samples.py --split test --per-letter 20
./.venv/bin/python scripts/finetune_on_user_data.py
./.venv/bin/python scripts/realtime_demo.py --backend hybrid_user
```

**Why a hybrid.** Measured on held-out images, the two representations fail in
opposite directions — pixels are excellent on sharp close-ups but degrade badly
on low-resolution webcam crops, while landmark geometry is invariant to
resolution and lighting but discards appearance detail. The hybrid keeps the
proposal's MobileNetV2 → BiLSTM path and concatenates the normalised landmark
vector before the classifier, so the network can lean on whichever stream is
reliable for a given frame. Run `scripts/evaluate_recognizers.py` to reproduce
the comparison.

The CNN branch does **not** see the raw frame. MediaPipe first locates the hand
and the frame is cropped to a padded square around it. This matters because the
dataset mixes close-up hand photos with full webcam scenes, and live webcam
frames are full scenes — cropping normalises all of them into one hand-centred
distribution so the CNN learns handshapes rather than backgrounds.

### Train the CNN+LSTM recogniser

```bash
# 1. Cache MediaPipe hand crops (runs MediaPipe once, not once per epoch)
./.venv/bin/python scripts/prepare_hand_crops.py --per-class-limit 150

# 2. Train MobileNetV2 + BiLSTM
./.venv/bin/python scripts/train_cnn_lstm.py --epochs 14

# Optional: fine-tune the top of the backbone (slower)
./.venv/bin/python scripts/train_cnn_lstm.py --epochs 14 --fine-tune-epochs 5
```

Artefacts: `models/recognition/cnn_lstm_model.keras`,
`cnn_lstm_metadata.json` (class names, metrics, history), and
`cnn_lstm_confusion_matrix.json`.

### Train the hybrid recogniser (recommended)

```bash
# 1. Cache hand crops + landmarks, balanced across both image groups
./.venv/bin/python scripts/prepare_hand_crops.py \
    --cache-dir data/processed/hand_crops_v2 --scene-limit 400 --closeup-limit 300

# 2. Train the two-stream model
./.venv/bin/python scripts/train_hybrid.py --epochs 12
```

Artefacts: `models/recognition/hybrid_model.keras`, `hybrid_metadata.json`,
`hybrid_confusion_matrix.json`.

### Measure accuracy honestly

```bash
# Held-out comparison of every trained backend, split by image group
./.venv/bin/python scripts/evaluate_recognizers.py --per-group 12
```

Accuracy on the public corpus only tells you how well a model fits somebody
else's recordings. To measure *your* setup, record your own samples:

```bash
./.venv/bin/python scripts/capture_samples.py --split test --per-letter 20
./.venv/bin/python scripts/evaluate_on_user_data.py
```

Captures land in `data/raw/user_samples/<split>/<LETTER>/`, the same layout as
the main dataset. Recording with `--split train` lets you fold your own data
into training so the model adapts to your camera, lighting and hand.

**This scores only the images the fine-tune held back.** `hybrid_user` trains on
60% of that directory, so measuring it over the whole folder would compare a
model against its own training data. The held-out portion is reconstructed from
the crop-cache manifest by `modules/recognition/user_split.py`, which both the
fine-tune and the evaluation import so the split cannot drift. `--all-samples`
overrides this and labels the affected backend in the output.

Measured on 205 held-out images across 26 classes:

| Backend | Your samples | Corpus holdout |
|---------|--------------|----------------|
| `hybrid_user` | **93.7%** | 99.1% |
| `hybrid` | 62.4% | 97.8% |
| `landmark_mlp` | 55.1% | 66.7% |
| `cnn_lstm` | 22.4% | 71.6% |

The 31-point gap on your own camera is the whole argument for fine-tuning. The
corpus column is much flatter, and the difference between `hybrid_user` and
`hybrid` there is four images out of 324 — noise, not a gain.

### Train the landmark MLP baseline

```bash
./.venv/bin/python scripts/preprocess_asl.py --per-class-limit 100
./.venv/bin/python scripts/train_recognition.py
./.venv/bin/python scripts/infer_recognition.py data/raw/asl_alphabet/asl_alphabet_test/A_test.jpg
```

## Tests

```bash
./.venv/bin/python -m pytest -q
```

### NLP grammar correction (gloss → English)

Recognition can be paused: NLP only needs a gloss string. Fine-tune **t5-small**
on ASLG-PC12, then compare BLEU against the rule-based baseline:

```bash
./.venv/bin/python scripts/download_aslg_pc12.py
./.venv/bin/python scripts/train_nlp_t5.py --max-train 20000 --epochs 2
./.venv/bin/python scripts/evaluate_nlp.py
./.venv/bin/python scripts/prototype.py --gloss "I GO STORE" --use-nlp-model
```

When `models/nlp/t5_gloss_en/` exists, `--use-nlp-model` loads it automatically.
Without it, the offline rule-based corrector is used.

### Emotion detection (face → prosody)

Trains a CNN on FER2013 mapped to the four proposal classes
(happy / sad / angry / neutral). When `models/emotion/emotion_model.keras`
exists it loads automatically; no face → neutral.

```bash
./.venv/bin/python scripts/download_fer2013.py
./.venv/bin/python scripts/train_emotion.py --arch v2 --epochs 50
./.venv/bin/python scripts/evaluate_emotion.py
./.venv/bin/python scripts/prototype.py --text "hello" --face-image path/to/face.jpg
```

`v2` is a VGG-style network with augmentation built into the model, so the layers
are inert at inference and the detector needs no changes. It scores **65.5%** on
the 3,937-image test split against **63.7%** for the original three-block `v1`
(McNemar exact p = 0.023; `--arch v1` reproduces it). Both fall well short of the
80% target — see `PROJECT_STATUS.md` §4.1, which also explains why the tempting
"matches human accuracy on FER2013" defence does not apply to a 4-class subset.

Training takes roughly 45 minutes on 8 CPU cores. It writes to `--output-dir`, so
train somewhere else and promote only if the result wins.

### Latency + unified evaluation

TTS caches repeated utterances under `logs/tts/cache/`. Benchmark the gloss-pipeline
tail (NLP → emotion → TTS) and merge all module metrics:

```bash
# TTS cold/warm + gloss-pipeline latency
./.venv/bin/python scripts/benchmark_latency.py

# Aggregate existing nlp/emotion/recognition JSON + optional latency run
./.venv/bin/python scripts/evaluate_all.py --run-latency

# Run missing per-module scripts, then aggregate + benchmark
./.venv/bin/python scripts/evaluate_all.py --run-module-evals --run-latency
```

Output: `logs/evaluation/unified_report.json` (headline BLEU, accuracy, p95 latency).

Latency is reported in three separate buckets so the numbers mean something:

| Bucket | Measured | Typical |
|--------|----------|---------|
| Startup | First utterance — imports TensorFlow, loads T5 + CNN | ~3.6 s (one-off) |
| Gloss tail | NLP → emotion → TTS per utterance | p95 ~0.8 s (target 1.5 s) |
| Repeated utterance | Same text served from the TTS cache | ~1 ms |

The cache stores both the audio and the Nepali translation, so a repeated
sentence needs no network round-trip at all.

### Usability study (SUS ≥ 80, ≥ 10 participants)

The interim report commits to a formal user study across deaf/HoH signers,
interpreters, and hearing non-signers. Materials and procedure: [`study/`](study/README.md).
**Raw participant data stay on your machine and are never shared externally** —
see [`study/DATA_HANDLING.md`](study/DATA_HANDLING.md).

```bash
# One session per participant — pseudonyms only (P01, P02, …)
./.venv/bin/python scripts/run_usability_session.py --participant P01 --group deaf_hoh

./.venv/bin/python scripts/evaluate_usability.py
./.venv/bin/python scripts/export_usability_summary.py   # aggregates for dissertation
./.venv/bin/python scripts/evaluate_all.py --aggregate-only
```

Optional pre-check before sessions: `./.venv/bin/python scripts/evaluate_scenarios.py`

Quantitative metrics (accuracy, BLEU, latency, confusion matrices) are aggregated
by `scripts/evaluate_all.py` alongside SUS.

## Roadmap toward the proposal targets

- Recognition: alphabet path met on accuracy; **WLASL word-level (≥50 glosses)** is
  implemented via `scripts/download_wlasl.py` + `scripts/train_wlasl.py`
  (`wlasl_video` backend). Evaluate with `scripts/evaluate_wlasl.py`.
- Emotion: 65.5% on the 4-class test set against a ≥80% target. Further gains need a
  pretrained backbone at higher resolution, which needs a GPU to be practical.
- Evaluation: run SUS sessions (`study/`, `run_usability_session.py`); cite aggregates
  only in the dissertation; quantitative metrics via `evaluate_all.py`.

## WLASL word-level recognition

```bash
# Download top-50 glosses from Voxel51/WLASL (C-UDA academic licence) + cache frames
./.venv/bin/python scripts/download_wlasl.py --num-classes 50

# Train MobileNetV2 (TimeDistributed) + BiLSTM
./.venv/bin/python scripts/train_wlasl.py

# Held-out val accuracy → logs/evaluation/wlasl_accuracy.json
./.venv/bin/python scripts/evaluate_wlasl.py

# Classify a clip then run NLP → TTS
./.venv/bin/python scripts/demo.py --wlasl-video path/to/clip.mp4 --nlp off
```

Alphabet webcam recognition (`hybrid_user`) and WLASL word-clip recognition are
separate backends. ``auto`` still prefers the alphabet models for live fingerspelling.

### Live word signs (camera)

```bash
# OpenCV: press [r] to record a sign, [r] again to detect, [space] to speak
./.venv/bin/python scripts/demo.py --wlasl --nlp off

# Or the web dashboard: switch Input → Words, then Record sign → Stop & detect
./.venv/bin/python scripts/run_web_app.py --dev
```

You can retrain `wlasl_video` later; the live record/detect UI stays the same.