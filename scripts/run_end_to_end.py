from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.registry import BACKENDS
from pipeline.pipeline import ASLPipeline

_TEST_DIR = repo_root / "data" / "raw" / "asl_alphabet" / "asl_alphabet_test"


def _spell_to_paths(word: str):
    """Map a word to per-letter test images (e.g. 'CAB' -> [C_test.jpg, ...])."""
    paths = []
    for ch in word.upper():
        candidate = _TEST_DIR / f"{ch}_test.jpg"
        if candidate.exists():
            paths.append(str(candidate))
        else:
            print(f"Warning: no test image for '{ch}' at {candidate}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ASL -> multilingual speech pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", nargs="+", help="Ordered sign image paths (one per frame).")
    group.add_argument("--spell", type=str, help="Spell a word using bundled *_test.jpg images.")

    parser.add_argument("--face-image", type=str, default=None,
                        help="Optional image for emotion detection.")
    parser.add_argument("--languages", nargs="+", default=["en", "ne"],
                        help="Output languages for TTS.")
    parser.add_argument("--use-nlp-model", action="store_true",
                        help="Attempt the T5 grammar model (needs weights/network).")
    parser.add_argument("--backend", choices=BACKENDS, default="auto",
                        help="Recognition backend (default: auto).")
    args = parser.parse_args()

    image_paths = args.images if args.images else _spell_to_paths(args.spell)
    if not image_paths:
        print("No input images resolved; nothing to do.")
        sys.exit(1)

    pipeline = ASLPipeline(use_nlp_model=args.use_nlp_model, recognition_backend=args.backend)
    try:
        result = pipeline.process(
            image_paths,
            face_image=args.face_image,
            languages=tuple(args.languages),
        )
    finally:
        pipeline.close()

    print("\n=== ASL -> Multilingual Speech ===")
    print(f"Recognition backend: {pipeline.recognition_backend}")
    print(f"Frames processed : {len(image_paths)}")
    print(f"Recognized labels: {result['recognized_labels']}")
    print(f"Assembled gloss  : {result['gloss']!r}")
    print(f"English sentence : {result['english']!r}")
    print(f"Emotion          : {result['emotion']} (conf={result['emotion_confidence']:.2f})")
    print("Speech outputs   :")
    for lang, info in result["speech"].items():
        print(f"  [{lang}] text={info['text']!r} audio={info['audio_path']}")
    print("Timings (ms)     :", json.dumps(result["timings"], indent=2))


if __name__ == "__main__":
    main()
