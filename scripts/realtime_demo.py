from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import cv2  # noqa: E402

from modules.emotion.detector import EmotionDetector  # noqa: E402
from modules.nlp.correction import GrammarCorrector  # noqa: E402
from modules.recognition.hand_crop import landmark_bbox  # noqa: E402
from modules.recognition.preprocess import ASLPreprocessor  # noqa: E402
from modules.recognition.registry import BACKENDS, create_recognizer  # noqa: E402
from modules.recognition.stabilizer import SignStabilizer  # noqa: E402
from modules.recognition.wlasl_live import ClipRecorder, assemble_word_gloss  # noqa: E402
from modules.tts.synthesizer import MultilingualSynthesizer  # noqa: E402
from pipeline.model_selection import format_model_report, resolve_models  # noqa: E402
from pipeline.pipeline import assemble_text  # noqa: E402

HAND_MODEL = str(repo_root / "models" / "recognition" / "hand_landmarker.task")
WLASL_MODEL = repo_root / "models" / "recognition" / "wlasl_video_model.keras"


def draw_hold_bar(frame, progress: float, label: str) -> None:
    """Draw the dwell progress bar so the user can see when a sign will commit."""
    h, w = frame.shape[:2]
    bar_w, bar_h = 260, 16
    x, y = 10, h - 40

    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * max(0.0, min(1.0, progress)))
    colour = (0, 255, 0) if progress >= 1.0 else (0, 180, 255)
    cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), colour, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (220, 220, 220), 1)

    if label:
        cv2.putText(frame, f"holding {label}", (x + bar_w + 12, y + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)


def run_webcam_demo(
    camera: int = 0,
    backend: str = "auto",
    hold_time: float = 0.9,
    min_confidence: float = 0.6,
    agreement: float = 0.65,
    cooldown: float = 0.5,
    predict_interval: float = 0.12,
    use_nlp_model: bool = False,
    languages: tuple = ("en", "ne"),
    show_model_report: bool = True,
) -> None:
    """Live webcam loop: dwell-gated recognition, then NLP → emotion → TTS.

    Shared by ``scripts/realtime_demo.py`` and the unified ``scripts/demo.py`` so
    both entry points behave identically.
    """
    if show_model_report:
        print(format_model_report(resolve_models(
            recognition_backend=backend,
            nlp_mode="on" if use_nlp_model else "off",
            languages=tuple(languages),
        )))

    recognizer = create_recognizer(backend=backend)
    print(f"Recognition backend : {recognizer.backend_name}")
    print(f"Hold to accept      : {hold_time:.2f}s "
          f"(min conf {min_confidence}, agreement {agreement})")

    preprocessor = ASLPreprocessor(
        model_asset_path=HAND_MODEL if Path(HAND_MODEL).exists() else None
    )
    emotion = EmotionDetector()
    corrector = GrammarCorrector(use_model=use_nlp_model)
    synth = MultilingualSynthesizer()
    print(f"NLP backend         : {corrector.backend}")
    print(f"Emotion backend     : {emotion.backend}")

    stabilizer = SignStabilizer(
        hold_time=hold_time,
        min_confidence=min_confidence,
        agreement=agreement,
        cooldown=cooldown,
    )

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    labels: list[str] = []
    last_label = ""
    last_conf = 0.0
    last_box = None
    next_predict_at = 0.0
    print("Controls:  [space]=finish sentence   [c]=clear   [q]=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            if now >= next_predict_at:
                next_predict_at = now + predict_interval

                landmarks = preprocessor.extract_landmarks_from_array(frame)
                pred = recognizer.predict_frame(bgr_image=frame, landmarks=landmarks)
                last_label = pred["label"]
                last_conf = pred.get("confidence") or 0.0
                last_box = landmark_bbox(landmarks, frame.shape[1], frame.shape[0])

                # The stabilizer decides whether this becomes a committed letter.
                committed = stabilizer.update(last_label, last_conf, now=now)
                if committed is not None:
                    labels.append(committed)
                    print(f"  + {committed}")

            if last_box:
                x1, y1, x2, y2 = last_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

            # debounce=False: the stabilizer already de-duplicates over time, so
            # repeated labels here are deliberate double letters.
            current = assemble_text(labels, debounce=False)
            cv2.putText(frame, current[-40:], (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"{last_label} ({last_conf:.2f}) [{recognizer.backend_name}]",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            candidate, progress = stabilizer.status(now=time.monotonic())
            draw_hold_bar(frame, progress, candidate or "")

            cv2.imshow("ASL -> Speech (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                labels = []
                stabilizer.reset()
            if key == ord(" "):
                gloss = assemble_text(labels, debounce=False)
                english = corrector.correct(gloss)
                emo = emotion.detect(frame)
                speech, timings = synth.synthesize(
                    english,
                    prosody=emo["prosody"],
                    languages=tuple(languages),
                    return_timings=True,
                )
                print(f"\ngloss   : {gloss!r}")
                print(f"english : {english!r}")
                print(f"emotion : {emo['emotion']}")
                for lang, info in speech.items():
                    cached = " (cached)" if info.get("cached") else ""
                    print(f"  [{lang}] {info['text']!r} -> {info['audio_path']}{cached}")
                print(f"tts     : {timings['total_ms']:.0f} ms total")
                labels = []
                stabilizer.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        preprocessor.close()


def run_wlasl_webcam_demo(
    camera: int = 0,
    use_nlp_model: bool = False,
    languages: tuple = ("en", "ne"),
    show_model_report: bool = True,
    max_seconds: float = 4.0,
) -> None:
    """Live webcam loop for word-level WLASL signs.

    Press ``r`` to start/stop recording one sign. On stop, the buffered clip is
    classified and the predicted gloss is appended. Space finishes the sentence
    through NLP → emotion → TTS. The model can be retrained later without
    changing this interaction.
    """
    if not WLASL_MODEL.exists():
        print(
            "WLASL model missing. Train it first:\n"
            "  ./.venv/bin/python scripts/download_wlasl.py\n"
            "  ./.venv/bin/python scripts/train_wlasl.py"
        )
        sys.exit(1)

    if show_model_report:
        print(format_model_report(resolve_models(
            recognition_backend="wlasl_video",
            nlp_mode="on" if use_nlp_model else "off",
            languages=tuple(languages),
        )))

    from modules.recognition.wlasl_predictor import WLASLPredictor

    predictor = WLASLPredictor(min_confidence=0.12)
    emotion = EmotionDetector()
    corrector = GrammarCorrector(use_model=use_nlp_model)
    synth = MultilingualSynthesizer()
    recorder = ClipRecorder(target_fps=12.0, max_seconds=max_seconds)

    print(f"Recognition backend : {predictor.backend_name}")
    print(f"NLP backend         : {corrector.backend}")
    print(f"Classes             : {len(predictor.class_names)}")
    print(
        "Controls:  [r]=record/stop sign   [space]=finish sentence   "
        "[c]=clear   [x]=cancel recording   [q]=quit"
    )

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    words: list[str] = []
    last_pred = ""
    last_conf = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            if recorder.recording:
                still = recorder.add_frame(frame, now=now)
                if not still:
                    # Auto-stop at max duration.
                    frames = recorder.stop()
                    if len(frames) >= 4:
                        result = predictor.predict_clip(frames_bgr=frames)
                        last_pred = result["label"]
                        last_conf = float(result.get("confidence") or 0.0)
                        if result.get("rejected") or last_pred == "unknown":
                            print(
                                f"  ? low confidence {last_conf:.2f} "
                                f"(raw={result.get('raw_label')}) — try again"
                            )
                        else:
                            words.append(last_pred)
                            print(f"  + {last_pred} ({last_conf:.2f})  [auto-stop]")
                            if result.get("top5"):
                                top = ", ".join(
                                    f"{t['label']}:{t['confidence']:.2f}"
                                    for t in result["top5"][:3]
                                )
                                print(f"    top3: {top}")
                    else:
                        print("  (recording too short — ignored)")

            gloss = assemble_word_gloss(words)
            cv2.putText(
                frame, gloss[-50:] or "(no words yet)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )

            if recorder.recording:
                dur = recorder.duration(now)
                cv2.putText(
                    frame,
                    f"RECORDING {dur:.1f}s  frames={recorder.frame_count}  [r]=stop",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
                # Red border while recording.
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (4, 4), (w - 5, h - 5), (0, 0, 255), 3)
            else:
                hint = last_pred and f"last: {last_pred} ({last_conf:.2f})" or "press [r] to record a sign"
                cv2.putText(
                    frame, hint, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2,
                )

            cv2.imshow("ASL WLASL words -> Speech (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                words = []
                last_pred = ""
                last_conf = 0.0
                recorder.cancel()
            if key == ord("x"):
                recorder.cancel()
                print("  (recording cancelled)")
            if key == ord("r"):
                if not recorder.recording:
                    recorder.start(now=time.monotonic())
                    print("  recording…")
                else:
                    frames = recorder.stop()
                    if len(frames) < 4:
                        print("  (recording too short — hold [r] longer / sign longer)")
                    else:
                        result = predictor.predict_clip(frames_bgr=frames)
                        last_pred = result["label"]
                        last_conf = float(result.get("confidence") or 0.0)
                        if result.get("rejected") or last_pred == "unknown":
                            print(
                                f"  ? low confidence {last_conf:.2f} "
                                f"(raw={result.get('raw_label')}) — try again"
                            )
                        else:
                            words.append(last_pred)
                            print(f"  + {last_pred} ({last_conf:.2f})")
                            if result.get("top5"):
                                top = ", ".join(
                                    f"{t['label']}:{t['confidence']:.2f}"
                                    for t in result["top5"][:3]
                                )
                                print(f"    top3: {top}")
            if key == ord(" "):
                gloss = assemble_word_gloss(words)
                if not gloss:
                    print("  (nothing to speak — record a sign first)")
                    continue
                english = corrector.correct(gloss)
                emo = emotion.detect(frame)
                speech, timings = synth.synthesize(
                    english,
                    prosody=emo["prosody"],
                    languages=tuple(languages),
                    return_timings=True,
                )
                print(f"\ngloss   : {gloss!r}")
                print(f"english : {english!r}")
                print(f"emotion : {emo['emotion']}")
                for lang, info in speech.items():
                    cached = " (cached)" if info.get("cached") else ""
                    print(f"  [{lang}] {info['text']!r} -> {info['audio_path']}{cached}")
                print(f"tts     : {timings['total_ms']:.0f} ms total")
                words = []
                last_pred = ""
                last_conf = 0.0
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time webcam ASL demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--backend", choices=BACKENDS, default="auto",
                        help="Recognition backend (default: auto).")
    parser.add_argument("--hold-time", type=float, default=0.9,
                        help="Seconds a sign must be held before it is accepted.")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Mean confidence required to accept a held sign.")
    parser.add_argument("--agreement", type=float, default=0.65,
                        help="Fraction of the hold window that must agree on the label.")
    parser.add_argument("--cooldown", type=float, default=0.5,
                        help="Pause after accepting a sign (allows double letters).")
    parser.add_argument("--predict-interval", type=float, default=0.12,
                        help="Seconds between predictions. Time-based, so the hold "
                             "duration is unaffected by camera frame rate.")
    parser.add_argument("--languages", nargs="+", default=["en", "ne"])
    parser.add_argument("--use-nlp-model", action="store_true")
    parser.add_argument(
        "--wlasl",
        action="store_true",
        help="Word-level live mode: record signs with [r], classify on stop.",
    )
    args = parser.parse_args()

    if args.wlasl:
        run_wlasl_webcam_demo(
            camera=args.camera,
            use_nlp_model=args.use_nlp_model,
            languages=tuple(args.languages),
        )
        return

    run_webcam_demo(
        camera=args.camera,
        backend=args.backend,
        hold_time=args.hold_time,
        min_confidence=args.min_confidence,
        agreement=args.agreement,
        cooldown=args.cooldown,
        predict_interval=args.predict_interval,
        use_nlp_model=args.use_nlp_model,
        languages=tuple(args.languages),
    )


if __name__ == "__main__":
    main()
