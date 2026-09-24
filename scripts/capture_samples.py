"""Record your own ASL alphabet samples from the webcam.

Why this exists: the corpus was recorded by other people, in other rooms, with
other cameras. Held-out testing shows the recognisers transfer poorly to
webcam-style framing, and no amount of retraining on the existing images fixes
a domain the data does not contain — *your* camera, lighting, hand and
background.

Samples recorded here can be used two ways:

* ``--split test``  (do this first) — an honest measure of live accuracy;
* ``--split train`` — fold into training so the model adapts to your setup.

Images are written to ``data/raw/user_samples/<split>/<LETTER>/`` in the same
one-folder-per-class layout as the main dataset, so every existing script works
on them unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import cv2  # noqa: E402

from modules.recognition.hand_crop import landmark_bbox  # noqa: E402
from modules.recognition.preprocess import ASLPreprocessor  # noqa: E402

HAND_MODEL = str(repo_root / "models" / "recognition" / "hand_landmarker.task")
DEFAULT_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["space", "del", "nothing"]

# Classes that are *defined* by the absence of a hand. Requiring a hand detection
# for these would mean no frame ever qualifies and the class could never be
# recorded, so the hand requirement is waived for them.
NO_HAND_CLASSES = {"nothing"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture your own ASL samples from the webcam")
    parser.add_argument("--letters", nargs="+", default=DEFAULT_LETTERS,
                        help="Classes to record (default: A-Z plus space/del/nothing).")
    parser.add_argument("--per-letter", type=int, default=20,
                        help="Frames to record per letter.")
    parser.add_argument("--split", choices=["train", "test"], default="test",
                        help="'test' measures real accuracy; 'train' adapts the model to you.")
    parser.add_argument("--output-dir", default="data/raw/user_samples")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.25,
                        help="Seconds between captured frames while recording.")
    parser.add_argument("--countdown", type=float, default=1.5,
                        help="Grace period after pressing space, so you can form the sign "
                             "before capture starts.")
    parser.add_argument("--require-hand", action=argparse.BooleanOptionalAction, default=True,
                        help="Only save frames where MediaPipe detects a hand "
                             "(use --no-require-hand to save every frame).")
    args = parser.parse_args()

    out_root = Path(args.output_dir) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    preprocessor = ASLPreprocessor(
        model_asset_path=HAND_MODEL if Path(HAND_MODEL).exists() else None
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    print(f"Saving to : {out_root}")
    print(f"Letters   : {len(args.letters)}  x {args.per_letter} frames each")
    print("\nControls:  [space]=start/stop recording current letter")
    print("           [n]=next letter   [b]=back   [s]=skip   [q]=quit\n")
    print("Tip: move your hand slightly between frames (angle, distance, position)."
          "\n     Varied samples are far more useful than 20 identical ones.\n")

    index = 0
    recording = False
    next_capture = 0.0
    capture_from = 0.0
    session_saved = 0

    try:
        while index < len(args.letters):
            letter = args.letters[index]
            target_dir = out_root / letter
            target_dir.mkdir(parents=True, exist_ok=True)
            saved = len(list(target_dir.glob("*.jpg")))

            # "nothing" means an empty frame, so it cannot require a detection.
            require_hand = args.require_hand and letter not in NO_HAND_CLASSES

            ok, frame = cap.read()
            if not ok:
                break

            landmarks = preprocessor.extract_landmarks_from_array(frame)
            has_hand = bool(landmarks.any())
            box = landmark_bbox(landmarks, frame.shape[1], frame.shape[0]) if has_hand else None

            now = time.monotonic()
            counting_down = recording and now < capture_from

            if recording and not counting_down and saved < args.per_letter and now >= next_capture:
                if has_hand or not require_hand:
                    # ``user_`` prefix marks these as webcam-scene framing; see
                    # modules.recognition.crop_dataset.image_group.
                    path = target_dir / f"user_{letter}_{saved:03d}_{int(now*1000)%100000}.jpg"
                    cv2.imwrite(str(path), frame)
                    saved += 1
                    session_saved += 1
                    next_capture = now + args.interval

            # Only advance on a session that actually recorded something. Without
            # this, pressing space on an already-full letter silently skips ahead.
            if saved >= args.per_letter and recording and session_saved:
                recording = False
                print(f"  {letter}: done ({saved} frames)")
                index += 1
                continue

            display = frame.copy()
            if box:
                cv2.rectangle(display, box[:2], box[2:], (0, 200, 255), 2)

            if counting_down:
                status = f"get ready... {capture_from - now:.1f}s"
                colour = (0, 200, 255)
            elif recording:
                status = "RECORDING"
                colour = (0, 0, 255)
            else:
                status = "paused"
                colour = (200, 200, 200)

            cv2.putText(display, f"Sign: {letter}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
            cv2.putText(display, f"{saved}/{args.per_letter}  {status}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.putText(display, f"({index + 1}/{len(args.letters)} classes)",
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            if not has_hand and require_hand:
                cv2.putText(display, "no hand detected - not saving", (10, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, "[space] record  [n] next  [b] back  [s] skip  [q] quit",
                        (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

            cv2.imshow("Capture ASL samples", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
                session_saved = 0
                next_capture = 0.0
                # Give the user time to form the sign; frames captured while the
                # hand is still moving into position are the transitional poses
                # that hurt accuracy.
                capture_from = time.monotonic() + args.countdown if recording else 0.0
            if key == ord("n") or key == ord("s"):
                recording = False
                session_saved = 0
                index += 1
            if key == ord("b"):
                recording = False
                session_saved = 0
                index = max(0, index - 1)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        preprocessor.close()

    total = sum(len(list(p.glob("*.jpg"))) for p in out_root.iterdir() if p.is_dir())
    print(f"\nCaptured {total} images in {out_root}")
    if args.split == "test":
        print("\nMeasure real accuracy with:")
        print(f"  ./.venv/bin/python scripts/evaluate_on_user_data.py --data-dir {out_root}")


if __name__ == "__main__":
    main()
