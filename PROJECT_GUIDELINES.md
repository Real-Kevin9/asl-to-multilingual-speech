# Real-Time Context-Aware Sign Language → Multilingual Speech

## Overview
- Goal: Build a real-time pipeline that converts ASL gestures (webcam) into grammatically-correct spoken English and Nepali, with emotion-aware speech modulation.
- Key gaps addressed: grammatical correction, Nepali output, emotion-aware speech.

## Objectives
- Sign recognition: CNN+LSTM (or MobileNetV2+BiLSTM) for ≥50 signs, target ≥85% accuracy.
- NLP correction: Transformer-based module (T5 or similar) to map glosses → grammatically correct sentences; +20% fluency over baseline.
- Multilingual TTS: English + Nepali output with end-to-end latency < 1.5s.
- Emotion detection: facial-expression classifier for at least {happy, sad, angry, neutral} with ≥80% accuracy to modulate TTS (pitch/rate).
- Evaluation: quantitative metrics (accuracy, F1, BLEU, latency) and user study (≥10 participants, SUS ≥ 80).

## Scope (In-Scope / Out-of-Scope)
- In-scope: ASL only, webcam input, grammar correction, English & Nepali audio, real-time desktop deployment, ethical user studies.
- Out-of-scope: other sign languages, mobile/embedded targets, full non-manual linguistic features beyond facial emotion, automated user-specific adaptation without retraining.

## High-Level Architecture
1. Video capture (webcam) → MediaPipe skeletal/keypoint extraction
2. Recognition module (frame-level CNN → sequence model LSTM/Transformer)
3. NLP correction module (glosses → sentence)
4. Emotion detection (face branch) → emotion label
5. Post-processing & language translation (if needed)
6. Multilingual TTS with emotion-based prosody modulation
7. Playback + logging

## Data & Datasets
- Use: ASL Alphabet (static images), WLASL (video), ASLG-PC12 (gloss→sentence). 
- Nepali: no NSL dataset available; synthesize speech via TTS after English correction + translation.
- Preprocessing: MediaPipe landmarks extraction, normalization, sequence padding/truncation, augmentation for robustness.
- Storage: encrypted, access-controlled, anonymized video/audio for user studies.

## Module-level Guidelines
- Recognition
  - Prefer skeleton/keypoints (MediaPipe) over raw frames to reduce domain variance.
  - Baseline: MobileNetV2 backbone → BiLSTM (2 layers). Evaluate Transformer encoder alternatives later.
  - Metrics: class-wise precision/recall, confusion matrix.
- NLP Correction
  - Fine-tune T5-small / T5-base on gloss→sentence pairs; include beam search and sampling experiments.
  - Evaluate with BLEU and human fluency checks.
- Emotion Detection
  - Facial expression classifier (ResNet or lightweight CNN) trained on emotion datasets; ensure domain adaptation if possible.
  - Map emotion to TTS prosody parameters (e.g., pitch shift, speaking rate multipliers).
- Multilingual TTS
  - Use commercial APIs (Google Cloud TTS) for Nepali fallback; keep `gTTS` or `pyttsx3` as offline fallback.
  - Keep network calls asynchronous and cache repeated phrases to reduce latency.
- Integration
  - Design modules with clear interfaces (JSON messages or in-memory objects).
  - Keep per-module unit tests and a lightweight integration harness to simulate pipeline flow.

## Implementation Practices
- Language: Python 3.9+.
- Frameworks: PyTorch or TensorFlow (pick one), Hugging Face Transformers for NLP, MediaPipe for landmarks.
- Code style: follow PEP8; use black and isort for formatting.
- Version control: feature branches, PRs, concise commit messages.
- Containerization: provide `requirements.txt` and optional `Dockerfile` for reproducible runs.

## Testing & Validation
- Unit tests for data pipelines, model IO, and TTS modules.
- Small integration test: recorded sample videos → end-to-end audio output (automated on CI when possible).
- Evaluation suite: holdout test set (≥100 gestures), latency benchmark script, SUS & user feedback forms.

## Ethical, Privacy & Safety Guidelines
- Obtain informed consent for any participant recordings; provide withdrawal options.
- Anonymize and encrypt all recorded data; redact PII from logs.
- Report demographic performance disparities; include known limitations in user-facing docs.
- Do not recommend system for high-stakes translations (legal/medical) without human oversight.

## Evaluation Plan & Metrics
- Recognition: accuracy, precision, recall, F1, confusion matrix per-class.
- NLP: BLEU, human fluency rating (Likert scale), qualitative error analysis.
- Emotion detection: accuracy and confusion across emotion classes.
- End-to-end: latency (frame→audio), user SUS score, qualitative user feedback.

## Timeline (High-level milestones)
- Weeks 1–4: Data collection & preprocessing, MediaPipe pipeline.
- Weeks 5–10: Train recognition baseline; iterate to target accuracy.
- Weeks 11–14: Fine-tune NLP correction model + evaluation.
- Weeks 15–18: Integrate TTS & emotion mapping; latency optimization.
- Weeks 19–24: End-to-end evaluation, user studies, final reporting.

## Deliverables
- `modules/recognition/` model and training scripts
- `modules/nlp/` fine-tuning and inference code
- `modules/tts/` integration and prosody mapping
- Evaluation scripts, sample datasets, user-study results, final report
- `PROJECT_GUIDELINES.md` (this file)

---
