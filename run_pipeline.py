from __future__ import annotations

from pathlib import Path

from pipeline.input import PipelineInput
from pipeline.prototype import PrototypePipeline

_TEST_DIR = Path("data/raw/asl_alphabet/asl_alphabet_test")


def main() -> None:
    print("=== ASL Prototype — all phases ===\n")

    # Mode 1: text-only (no models needed)
    print("[1/3] Text input (NLP → Emotion → TTS):")
    with PrototypePipeline() as proto:
        r = proto.run(PipelineInput.from_text("hello world"))
    print(f"      gloss={r['gloss']!r}  english={r['english']!r}  emotion={r['emotion']}\n")

    model_ok = Path("models/recognition/recognition_model.joblib").exists()
    hand_ok = Path("models/recognition/hand_landmarker.task").exists()
    if not (model_ok and hand_ok and _TEST_DIR.exists()):
        print("[2/3] Skipped image/spell demos (train recognizer + MediaPipe model first).")
        print("[3/3] Skipped.")
        return

    # Mode 2: spell using test images
    print("[2/3] Spell input (Recognition → NLP → TTS):")
    inp = PipelineInput.from_spell("HELLO", _TEST_DIR)
    with PrototypePipeline() as proto:
        r = proto.run(inp)
    print(f"      labels={r['recognized_labels']}  gloss={r['gloss']!r}  english={r['english']!r}\n")

    # Mode 3: gloss-only (skip recognition)
    print("[3/3] Gloss input (skip recognition):")
    with PrototypePipeline() as proto:
        r = proto.run(PipelineInput.from_gloss("I GO STORE"))
    print(f"      english={r['english']!r}  phases={len(r['phases'])}")


if __name__ == "__main__":
    main()
