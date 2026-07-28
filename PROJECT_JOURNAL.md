# Project Management Journal (Mahara)

**Student:** Kevin Raj Karki (ID: 23189623)
**Course:** BSc (Hons) Computer and Data Science — CMP6200 Individual Undergraduate Project
**Supervisor:** Manoj Gautam
**Project:** Real-Time Context-Aware Sign Language to Multilingual Speech Conversion System with Emotion Detection

> Each entry follows the required journal format: the **date of entry** at the start,
> notes on **supervisor meetings**, **key issues**, **key events / action plans**, and a
> **conclusion** describing the actions to be done next. Week numbers follow the interim
> report Gantt chart (Figure 4.2); Week 9 corresponds to the interim report submission
> (27 April 2026). This journal covers **Weeks 1–17**.

---

## Week 1, Monday 2 March 2026

**Date of entry:** 2 March 2026

**Key events / activities:**

- Project allocation confirmed and topic finalised: a real-time system converting American
Sign Language into grammatically correct, emotion-aware English and Nepali speech.
- Read background material on the scale of the problem (WHO: 430M+ people with disabling
hearing loss; shortage of interpreters, including 40,000+ deaf people in Nepal).
- Set up the working environment: created the Git repository, project folder structure, and
this Mahara project-management journal.

**Supervisor meeting:**

- Introductory meeting. Discussed the overall aim, feasibility, and marking expectations.
- Agreed to constrain scope to **ASL input** with **English + Nepali speech output** to keep
the project achievable within the timeframe.

**Key issues:**

- The e pintopic is broad; scope and datasets need to bned down early.
- Need to confirm that suitable, openly-licensed datasets exist.

**Conclusion / actions for next week:**

- Begin the literature review on sign-language recognition and hand-pose estimation.

---

## Week 2, Monday 9 March 2026

**Date of entry:** 9 March 2026

**Key events / activities:**

- Reviewed sign-language recognition (SLR) literature: Mitra & Acharya (2007), Pigou et al.
(2018), Camgöz et al. (2020), and Koller's (2020) survey.
- Studied MediaPipe hand tracking (Zhang et al., 2020) and the advantage of skeletal/keypoint
representations over raw pixels (robust to lighting, background, skin tone).

**Key issues:**

- Early architecture question: CNN-LSTM vs Transformer for the recogniser. Leaning toward
CNN-LSTM for a better accuracy-to-latency trade-off on a ~50-sign vocabulary.

**Conclusion / actions for next week:**

- Continue the review, focusing on NLP grammar correction and text-to-speech synthesis.

---

## Week 3, Monday 16 March 2026

**Date of entry:** 16 March 2026

**Key events / activities:**

- Reviewed NLP grammar-correction literature: sequence-to-sequence translation (Stoll et al.,
  1. and pre-trained transformers, especially **T5** (Raffel et al., 2020).
- Noted the ASL↔English grammar mismatch (topic-comment vs subject-verb-object) that makes
word-by-word gloss translation unusable.
- Reviewed TTS literature (WaveNet, Tacotron 2, FastSpeech) and the low-resource situation for
Nepali speech (Sitaula & Ghimire, 2021); identified gTTS / Google Cloud TTS as pragmatic.

**Supervisor meeting:**

- Reviewed reading progress. Supervisor advised framing the review around clearly-defined
research gaps and using BLEU for the NLP evaluation.

**Key issues:**

- No labelled Nepali Sign Language dataset exists — Nepali must be handled at the synthesis
stage (translate corrected English → Nepali TTS).

**Conclusion / actions for next week:**

- Consolidate the identified gaps and draft the research questions and objectives.

---

## Week 4, Monday 23 March 2026

**Date of entry:** 23 March 2026

**Key events / activities:**

- Synthesised the three research gaps: (1) grammatical inadequacy of gloss output,
(2) exclusion of Nepali, (3) absence of emotion-aware speech.
- Drafted the three research questions (RQ1–RQ3) and the five SMART objectives (recognition
≥85%, NLP +20% fluency, latency <1.5s, emotion ≥80%, evaluation with SUS ≥80).

**Supervisor meeting:**

- Reviewed the draft objectives; agreed they are measurable and appropriately scoped.
- Confirmed emotion detection stays as a stretch objective feeding TTS prosody.

**Key issues:**

- Keeping the objectives ambitious but achievable; the emotion objective carries the most risk.

**Conclusion / actions for next week:**

- Design the system architecture and justify the chosen methods.

---

## Week 5, Monday 30 March 2026

**Date of entry:** 30 March 2026

**Key events / activities:**

- Designed the high-level architecture as a **four-layer modular pipeline**: recognition →
NLP correction → emotion → multilingual TTS, so each stage can be built and tested in
isolation.
- Documented method justifications (CNN-LSTM recogniser, MediaPipe landmarks, T5 correction,
cloud/gTTS synthesis).
- Shortlisted datasets: ASL Alphabet (Kaggle, static), WLASL (dynamic), ASLG-PC12 (gloss→sentence).

**Key issues:**

- Confirming dataset licences are suitable for academic use.

**Conclusion / actions for next week:**

- Finalise the project proposal and scaffold the code repository.

---

## Week 6, Monday 6 April 2026

**Date of entry:** 6 April 2026

**Key events / activities:**

- Finalised the project proposal and wrote `PROJECT_GUIDELINES.md`.
- Scaffolded the repository: `modules/` (recognition, nlp, tts), `pipeline/`, `scripts/`,
`tests/`, `data/`, `models/`, with placeholder module interfaces.
- Drafted the risk register and ethical considerations (informed consent, anonymisation,
algorithmic fairness).

**Supervisor meeting:**

- Proposal reviewed and signed off. Agreed to start the dataset and preprocessing phase.

**Key issues:**

- None significant; proposal approved.

**Conclusion / actions for next week:**

- Set up the development environment and download the ASL dataset.

---

## Week 7, Monday 13 April 2026

**Date of entry:** 13 April 2026

**Key events / activities:**

- Built the Python 3.12 virtual environment and installed dependencies (mediapipe, opencv,
scikit-learn, tensorflow, torch, transformers, gTTS, deep-translator).
- Downloaded the ASL Alphabet corpus: **204,924 images across 29 classes** (A–Z, `del`,
`nothing`, `space`) and organised it under `data/raw/`.

**Key issues:**

- Large dataset size — preprocessing the full corpus will be time-consuming.
- Need to confirm dependency version compatibility (notably MediaPipe with Python 3.12).

**Conclusion / actions for next week:**

- Analyse the dataset and design the preprocessing (landmark-extraction) pipeline.

---

## Week 8, Monday 20 April 2026

**Date of entry:** 20 April 2026

**Key events / activities:**

- Ran dataset analysis — per-class counts and distribution
(`evaluation/class_counts.csv`, `class_distribution.png`, `dataset_summary.txt`).
- Investigated the MediaPipe API and found the legacy **Solutions API is removed** in 0.10.x;
only the **Tasks API** remains, which needs an external model bundle. Documented findings in
`API_ANALYSIS.md` and `MEDIAPIPE_SETUP.md`.
- Implemented `modules/recognition/preprocess.py` using `HandLandmarker` with a safe
zero-vector fallback.

**Supervisor meeting:**

- Discussed the MediaPipe blocker; agreed to proceed with the Tasks API and obtain the model
file rather than downgrading the package.

**Key issues:**

- **Blocker:** the MediaPipe Tasks `HandLandmarker` needs an external model file that is not
bundled and could not initially be fetched — preprocessing currently returns zero vectors.

**Conclusion / actions for next week:**

- Compile and submit the interim report; work on resolving the MediaPipe model blocker.

---

## Week 9, Monday 27 April 2026 — INTERIM REPORT SUBMITTED

**Date of entry:** 27 April 2026

**Key events / activities:**

- Compiled and **submitted the Interim Report**: introduction, literature review, scope and
objectives, system design and methods, project timeline (Gantt), feasibility, risk analysis,
and ethical considerations.
- Continued troubleshooting the preprocessing blocker (still producing zero vectors without the
hand-landmark model).

**Supervisor meeting:**

- Reviewed the interim report before submission; confirmed the plan and timeline are sound.

**Key issues:**

- Recognition work cannot meaningfully start until real hand landmarks are extracted.

**Conclusion / actions for next week:**

- Resolve the MediaPipe model issue and regenerate real landmark features.

---

## Week 10, Monday 4 May 2026

**Date of entry:** 4 May 2026

**Key events / activities:**

- **Resolved the blocker:** obtained the `hand_landmarker.task` bundle and wired an
auto-download path into `scripts/preprocess_asl.py`; added a `--per-class-limit` option to
build a quick working subset instead of processing all 200k+ images.
- Extracted **63-dimensional landmark features** (21 points × x,y,z) and verified real output —
**92.3% non-zero values** on the subset, versus 100% zeros previously.
- Added the `dataset.py` loader and per-sample wrist-centring + scaling normalisation
(`modules/recognition/features.py`).

**Key issues:**

- Full-corpus preprocessing is slow (hours); chose a per-class subset to build the pipeline
first and scale up later.

**Conclusion / actions for next week:**

- Implement the recognition training pipeline.

---

## Week 11, Monday 11 May 2026

**Date of entry:** 11 May 2026

**Key events / activities:**

- Implemented `modules/recognition/trainer.py`: landmark normalisation → `StandardScaler` →
classifier, using an 80/20 stratified train/test split.
- Established the evaluation loop reporting accuracy and a per-class precision/recall/F1 report.

**Key issues:**

- Deciding the baseline model; chose a compact MLP over the landmark features as a fast,
reliable baseline before attempting the full CNN+LSTM.

**Conclusion / actions for next week:**

- Train the baseline recogniser and evaluate the results.

---

## Week 12, Monday 18 May 2026

**Date of entry:** 18 May 2026

**Key events / activities:**

- Trained the recognition **baseline** (MLP over normalised landmarks) on the processed subset
(~2,900 samples).
- Achieved **~93% held-out accuracy** — above the 85% target for the static-alphabet task.
Reviewed the per-class report to spot weaker classes.

**Supervisor meeting:**

- Presented the baseline results. Supervisor was satisfied with the accuracy and advised
documenting it as the reference baseline before moving to CNN+LSTM.

**Key issues:**

- The baseline is an MLP, not the proposed CNN+LSTM, and is trained on a subset only — both
noted as planned upgrades.

**Conclusion / actions for next week:**

- Build the inference/prediction component and ensure it matches the training pipeline.

---

## Week 13, Monday 25 May 2026

**Date of entry:** 25 May 2026

**Key events / activities:**

- Implemented `RecognitionPredictor` (`model.py`) bundling the model, scaler, and label encoder.
- **Found and fixed a consistency bug:** inference was not applying the wrist-centring and
feature scaling used during training. Unified both paths through a shared
`normalize_landmark_features` helper so training and inference are identical.

**Key issues:**

- The bug would have silently degraded real-world accuracy; important to catch before integration.

**Conclusion / actions for next week:**

- Add the inference CLI and validate predictions on the held-out test images.

---

## Week 14, Monday 1 June 2026

**Date of entry:** 1 June 2026

**Key events / activities:**

- Added `scripts/infer_recognition.py` for single-image prediction and validated it end-to-end.
- Saved model artefacts (`recognition_model.joblib`, `scaler.joblib`, `label_encoder.joblib`).

**Supervisor meeting:**

- Reviewed inference results. Discussed that some bundled test images are misclassified because
MediaPipe fails to detect a hand (returning a zero vector → `nothing`).

**Key issues:**

- Domain gap between training images and the separate test images; hand-detection failures on
some inputs.

**Conclusion / actions for next week:**

- Perform per-class error analysis and plan the CNN+LSTM upgrade.

---

## Week 15, Monday 8 June 2026

**Date of entry:** 8 June 2026

**Key events / activities:**

- Carried out per-class error and confusion analysis from the classification report; identified
the most confusable classes and the failure cases caused by missing landmarks.
- Planned full-corpus preprocessing and data augmentation to improve robustness.

**Key issues:**

- Some visually-similar handshapes are confused; more data and a temporal model should help.

**Conclusion / actions for next week:**

- Design the CNN+LSTM (MobileNetV2 + BiLSTM) architecture for the recogniser upgrade.

---

## Week 16, Monday 15 June 2026

**Date of entry:** 15 June 2026

**Key events / activities:**

- Designed the **CNN+LSTM** recogniser architecture (MobileNetV2 spatial features → 2-layer
BiLSTM) and outlined the plan for dynamic signs using the WLASL video corpus.
- Researched sequence modelling and padding/truncation strategies for variable-length signs.

**Supervisor meeting:**

- Agreed to keep the working MLP baseline as a fallback while developing the CNN+LSTM, so the
end-to-end pipeline always has a functioning recogniser.

**Key issues:**

- WLASL video processing is heavier and will likely need GPU time (Google Colab).

**Conclusion / actions for next week:**

- Consolidate and document the recognition module; prepare to transition into the NLP phase.

---

## Week 17, Monday 22 June 2026

**Date of entry:** 22 June 2026

**Key events / activities:**

- Consolidated the recognition module, added documentation, and froze the MLP baseline as the
reference model.
- Reviewed progress against the Gantt chart — the recognition phase (Weeks 9–20) is on track,
with the baseline complete and the CNN+LSTM upgrade in progress.

**Supervisor meeting:**

- Reviewed overall progress. Confirmed readiness to begin the **NLP context-correction phase
(Week 18)** while continuing CNN+LSTM development in parallel.

**Key issues:**

- Need to balance recognition refinement against starting the next phases to stay on schedule.

**Conclusion / actions for next week (Week 18 onward):**

- Begin the **NLP grammar-correction module** (gloss assembly + T5 with a rule-based fallback),
then proceed to multilingual TTS, emotion-aware prosody, and full pipeline integration per the
Gantt chart.

---

### Note on scope of this journal

This journal records project-management activity for **Weeks 1–17** (Literature Review &
Proposal → Dataset & Preprocessing → ASL Recognition, per the Gantt chart). The subsequent
phases — NLP correction (Weeks 18–24), multilingual TTS (22–26), emotion detection, and pipeline
integration & testing (25–28) — will be recorded in later weekly entries.