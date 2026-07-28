from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import cv2  # noqa: E402

from modules.emotion.detector import EmotionDetector  # noqa: E402
from modules.nlp.correction import GrammarCorrector  # noqa: E402
from modules.recognition.model import RecognitionPredictor  # noqa: E402
from modules.recognition.preprocess import ASLPreprocessor  # noqa: E402
from modules.tts.synthesizer import MultilingualSynthesizer  # noqa: E402
from pipeline.pipeline import assemble_text  # noqa: E402

HAND_MODEL = str(repo_root / "models" / "recognition" / "hand_landmarker.task")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time webcam ASL demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--capture-every", type=int, default=15,
                        help="Predict a sign every N frames.")
    parser.add_argument("--use-nlp-model", action="store_true")
    args = parser.parse_args()

    predictor = RecognitionPredictor()
    preprocessor = ASLPreprocessor(model_asset_path=HAND_MODEL if Path(HAND_MODEL).exists() else None)
    emotion = EmotionDetector()
    corrector = GrammarCorrector(use_model=args.use_nlp_model)
    synth = MultilingualSynthesizer()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    labels: list[str] = []
    frame_idx = 0
    print("Controls:  [space]=finish sentence   [c]=clear   [q]=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % args.capture_every == 0:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                landmarks = preprocessor.extract_landmarks(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)
            pred = predictor.predict(landmarks)
            labels.append(pred["label"])

        current = assemble_text(labels)
        cv2.putText(frame, current[-40:], (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.imshow("ASL -> Speech (press q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            labels = []
        if key == ord(" "):
            gloss = assemble_text(labels)
            english = corrector.correct(gloss)
            emo = emotion.detect(frame)
            speech = synth.synthesize(english, prosody=emo["prosody"])
            print(f"\ngloss   : {gloss!r}")
            print(f"english : {english!r}")
            print(f"emotion : {emo['emotion']}")
            for lang, info in speech.items():
                print(f"  [{lang}] {info['text']!r} -> {info['audio_path']}")
            labels = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
