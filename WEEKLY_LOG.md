# Weekly Project Log

**Project:** Real-Time Context-Aware Sign Language to Multilingual Speech Conversion System with Emotion Detection
**Student:** Kevin Raj Karki (ID: 23189623)
**Course:** BSc (Hons) Computer and Data Science — CMP6200 Individual Project
**Supervisor:** Manoj Gautam

This log distributes completed work across the phases defined in the interim
report Gantt chart (Figure 4.2, 34-week schedule). Weeks 1–28 record work
completed to date; Weeks 29–34 record planned/remaining work.

---

## Phase 1 — Literature Review & Proposal (Weeks 1–6)

### Week 1
- Selected the project topic and framed the problem statement: communication
  barriers for the 430M+ people with disabling hearing loss, and the lack of
  affordable, real-time, grammatically-correct sign-to-speech tools.
- Set up the working environment and Git repository.

### Week 2
- Reviewed sign-language recognition literature (Mitra & Acharya 2007; Pigou
  et al. 2018; Camgöz et al. 2020) and hand-pose estimation with MediaPipe
  (Zhang et al. 2020).
- Noted the shift from raw-pixel to skeletal/keypoint representations.

### Week 3
- Reviewed NLP grammar-correction (seq2seq, T5 — Raffel et al. 2020) and the
  ASL-vs-English grammar mismatch (topic-comment vs subject-verb-object).
- Reviewed TTS literature (WaveNet, Tacotron 2, FastSpeech) and the
  low-resource situation for Nepali speech synthesis.

### Week 4
- Consolidated the three research gaps: (1) grammatical inadequacy,
  (2) exclusion of Nepali, (3) absence of emotion-aware speech.
- Drafted research questions (RQ1–RQ3) and the five SMART objectives.

### Week 5
- Designed the high-level system architecture: a four-layer modular pipeline
  (recognition → NLP correction → emotion → multilingual TTS).
- Justified method choices (CNN-LSTM recognizer, MediaPipe landmarks, T5
  correction, cloud/gTTS synthesis).

### Week 6
- Finalized the project proposal and `PROJECT_GUIDELINES.md`.
- Scaffolded the repository (`modules/`, `pipeline/`, `scripts/`, `tests/`,
  `data/`, `models/`) with placeholder module interfaces.
- Documented risk register and ethical considerations (consent,
  anonymization, algorithmic fairness).

---

## Phase 2 — Dataset Collection & Pre-processing (Weeks 5–10)

### Week 7
- Selected and justified datasets: ASL Alphabet (Kaggle, 87k+ images) for
  static signs, WLASL for dynamic signs, ASLG-PC12 for gloss→sentence pairs.
- Downloaded the ASL Alphabet corpus and organized it into 29 classes
  (A–Z, `del`, `nothing`, `space`); **204,924 images** total.

### Week 8
- Built the Python 3.12 virtual environment and installed dependencies
  (mediapipe, opencv, scikit-learn, tensorflow, torch, transformers, gTTS,
  deep-translator).
- Ran dataset analysis — class distribution and counts
  (`evaluation/class_counts.csv`, `class_distribution.png`,
  `dataset_summary.txt`).

### Week 9
- Investigated the MediaPipe API change (Solutions vs Tasks). Documented that
  0.10.x only ships the **Tasks API**, which needs an external model bundle
  (`API_ANALYSIS.md`, `MEDIAPIPE_SETUP.md`).
- Implemented `modules/recognition/preprocess.py` — `HandLandmarker` backend
  with a safe zero-vector fallback.

### Week 10
- Obtained the `hand_landmarker.task` bundle and wired an auto-download path
  into `scripts/preprocess_asl.py` (plus a `--per-class-limit` subset option).
- Extracted 21×3 = **63-dim landmark features**; verified real (non-zero)
  output (**92.3% non-zero** on the working subset vs 100% zero previously).
- Added `dataset.py` loader and per-sample wrist-centering normalization
  (`modules/recognition/features.py`).

---

## Phase 3 — ASL Recognition Model (Weeks 9–20)

### Week 11
- Implemented `modules/recognition/trainer.py`: landmark normalization →
  `StandardScaler` → classifier, with an 80/20 stratified split.
- Established the training/evaluation loop (accuracy + per-class report).

### Week 12
- Trained the recognition **baseline** (MLP over normalized landmarks) on the
  processed subset (~2,900 samples).
- Achieved **~93% held-out accuracy** on the subset — above the 85% target for
  the static-alphabet task.

### Week 13
- Built `RecognitionPredictor` (`model.py`) bundling model + scaler + encoder.
- **Fixed a consistency bug**: inference previously skipped the
  wrist-centering + scaling used in training; unified both via a shared
  `normalize_landmark_features` helper.

### Week 14
- Added the single-image inference CLI (`scripts/infer_recognition.py`) and
  validated predictions end-to-end on the bundled test images.
- Saved model artifacts (`recognition_model.joblib`, `scaler.joblib`,
  `label_encoder.joblib`).

### Weeks 15–16
- Analysed per-class errors from the classification report; confirmed
  confusable classes and the effect of missing hand detections.
- *(Ongoing)* Preparing to scale up preprocessing to the full corpus and
  begin the **CNN+LSTM** upgrade over the current MLP baseline.

### Weeks 17–20 *(in progress / planned)*
- Implement and train the MobileNetV2 + BiLSTM recognizer.
- Introduce dynamic-sign support (WLASL) beyond the static alphabet.
- Target: ≥50 signs at ≥85% accuracy with confusion-matrix analysis.

---

## Phase 4 — NLP Context-Correction Module (Weeks 18–24)

### Weeks 18–19
- Implemented `modules/nlp/correction.py` with a `GrammarCorrector` class.
- Added the **gloss-assembly** stage (`assemble_text`) converting per-frame
  sign labels into text (handling `space`/`del`/`nothing` + frame debouncing).

### Weeks 20–21
- Implemented the **rule-based fallback** (gloss-marker stripping,
  capitalization, punctuation) so the stage runs fully offline.
- Wired the **T5** backend hook (`vennify/t5-base-grammar-correction`) with
  graceful degradation when weights/network are unavailable.

### Weeks 22–24 *(in progress / planned)*
- Fine-tune T5 on ASLG-PC12 gloss→sentence pairs.
- Evaluate with BLEU vs the word-by-word baseline (target ≥20% improvement).

---

## Phase 5 — Multilingual TTS Integration (Weeks 22–26)

### Weeks 22–23
- Implemented `modules/tts/synthesizer.py` (`MultilingualSynthesizer`):
  **English + Nepali** output via gTTS, with English→Nepali translation
  (deep-translator) and an offline-safe fallback that writes the spoken text.

### Weeks 24–25
- Implemented the **emotion detection** module (`modules/emotion/detector.py`)
  addressing proposal Objective 4: OpenCV face detection + a classifier hook,
  with `EMOTION_PROSODY` mapping {happy, sad, angry, neutral} → rate/pitch.
- Connected emotion output to TTS prosody modulation.

### Week 26 *(in progress / planned)*
- Add phrase caching and asynchronous calls to push end-to-end latency
  toward the < 1.5 s target; train the emotion CNN to ≥80% accuracy.

---

## Phase 6 — Pipeline Integration & Testing (Weeks 25–28)

### Weeks 25–26
- Built `ASLPipeline` (`pipeline/pipeline.py`) orchestrating all five stages
  end-to-end with **per-stage latency logging**.
- Added runnable entrypoints: `run_pipeline.py`, `scripts/run_end_to_end.py`
  (`--spell`/`--images`), and `scripts/realtime_demo.py` (webcam).

### Week 27
- Wrote the automated test suite (gloss assembly, NLP, TTS, emotion,
  preprocessing, dataset) — **10 tests passing** under pytest.
- Cleaned up MediaPipe teardown handling (`close()` / context manager).

### Week 28
- Verified full end-to-end runs (sign frames → landmarks → gloss → English →
  emotion → English/Nepali speech).
- Authored project `README.md` documenting the architecture, run commands, and
  the roadmap toward the proposal targets.
- **Milestone:** complete working outline of the whole system.

---

## Phase 7 — Usability Testing & Evaluation (Weeks 28–32) *(planned)*

### Weeks 29–30
- Build the quantitative evaluation suite: recognition accuracy/F1/confusion
  matrix, NLP BLEU, end-to-end latency benchmark.

### Weeks 31–32
- Conduct the usability study (≥10 participants: deaf/HoH signers,
  interpreters, hearing non-signers) using the System Usability Scale
  (target SUS ≥ 80); collect qualitative feedback.

---

## Phase 8 — Final Report & Submission (Weeks 30–34) *(planned)*

### Weeks 33–34
- Analyse evaluation results, report demographic performance disparities and
  limitations, and finalize the dissertation and demo for submission.

---

## Summary of progress to date

- **Completed:** literature review & proposal; dataset collection &
  preprocessing (real MediaPipe landmarks); recognition **baseline** at ~93%
  on subset; NLP correction (rule-based + T5 hook); multilingual TTS
  (English + Nepali) with emotion-aware prosody; full pipeline integration,
  entrypoints, tests, and documentation.
- **In progress:** CNN+LSTM recognizer, T5 fine-tuning on ASLG-PC12, emotion
  CNN training, latency optimization.
- **Remaining:** formal evaluation and usability study, final report.
