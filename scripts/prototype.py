from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.registry import BACKENDS
from pipeline.input import PipelineInput
from pipeline.prototype import PrototypePipeline

_TEST_DIR = repo_root / "data" / "raw" / "asl_alphabet" / "asl_alphabet_test"


def _capture_webcam_frames(camera: int, count: int) -> list[str]:
    """Grab N frames from the webcam and save them as temp JPEGs."""
    import cv2

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera}")

    paths: list[str] = []
    saved = 0
    print(f"Capturing {count} frame(s) — press [q] to stop early...")

    while saved < count:
        ok, frame = cap.read()
        if not ok:
            break
        if saved == 0 or saved % max(1, count // 5) == 0:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, prefix="asl_frame_")
            cv2.imwrite(tmp.name, frame)
            paths.append(tmp.name)
            saved += 1
        cv2.imshow("Prototype capture (q=stop)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return paths


def _print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("  ASL → Multilingual Speech  |  PROTOTYPE OUTLINE")
    print("=" * 60)

    for phase in result.get("phases", []):
        name = phase["phase"].upper()
        status = phase["status"]
        ms = phase["duration_ms"]
        print(f"\n--- Phase: {name} [{status}] ({ms:.1f} ms) ---")
        if phase.get("input") is not None:
            print(f"  IN : {json.dumps(phase['input'], default=str)[:200]}")
        print(f"  OUT: {json.dumps(phase['output'], default=str)[:400]}")

    print("\n--- SUMMARY ---")
    print(f"  Mode     : {result.get('input_mode')}")
    print(f"  Backend  : {result.get('recognition_backend')}")
    print(f"  Gloss    : {result.get('gloss')!r}")
    print(f"  English  : {result.get('english')!r}")
    print(f"  Emotion  : {result.get('emotion')}")
    if result.get("speech"):
        for lang, info in result["speech"].items():
            print(f"  Speech[{lang}]: {info.get('text')!r}  audio={info.get('audio_path')}")
    print(f"  Total    : {result['timings']['total_ms']:.1f} ms")
    if result.get("evaluation", {}).get("log_path"):
        print(f"  Eval log : {result['evaluation']['log_path']}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full project prototype (all 7 phases, any input mode)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", nargs="+", help="Ordered sign image paths.")
    group.add_argument("--spell", type=str, help="Spell a word using bundled test images.")
    group.add_argument("--gloss", type=str, help="Skip recognition; provide gloss text directly.")
    group.add_argument("--text", type=str, help="Skip recognition; provide plain text (NLP→TTS demo).")
    group.add_argument("--webcam", action="store_true", help="Capture frames from webcam.")

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=5, help="Webcam frames to capture.")
    parser.add_argument("--face-image", type=str, default=None)
    parser.add_argument("--languages", nargs="+", default=["en", "ne"])
    parser.add_argument("--use-nlp-model", action="store_true")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="Recognition backend: auto prefers the hybrid model, then CNN+LSTM, then the MLP.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted output.")
    args = parser.parse_args()

    if args.images:
        inp = PipelineInput.from_images(args.images, face_image=args.face_image)
    elif args.spell:
        inp = PipelineInput.from_spell(args.spell, _TEST_DIR, face_image=args.face_image)
        if not inp.image_paths:
            print(f"No test images found for {args.spell!r} in {_TEST_DIR}")
            sys.exit(1)
    elif args.gloss:
        inp = PipelineInput.from_gloss(args.gloss, face_image=args.face_image)
    elif args.text:
        inp = PipelineInput.from_text(args.text, face_image=args.face_image)
    else:
        paths = _capture_webcam_frames(args.camera, args.frames)
        if not paths:
            print("No frames captured.")
            sys.exit(1)
        inp = PipelineInput.from_images(paths, face_image=args.face_image)

    with PrototypePipeline(
        use_nlp_model=args.use_nlp_model, recognition_backend=args.backend
    ) as proto:
        result = proto.run(inp, languages=tuple(args.languages))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
