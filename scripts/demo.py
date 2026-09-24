"""Single demo command that runs the best trained model at every stage.

Equivalent to picking ``--backend hybrid_user``, ``--use-nlp-model`` and the
trained emotion CNN by hand, except the artefacts are resolved from disk so the
command keeps working after any stage is retrained.

    scripts/demo.py                     # webcam, best models
    scripts/demo.py --spell HELLO       # bundled test images
    scripts/demo.py --gloss "I GO STORE"
    scripts/demo.py --show-models       # report artefacts and exit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.recognition.registry import BACKENDS  # noqa: E402
from pipeline.input import PipelineInput  # noqa: E402
from pipeline.model_selection import (  # noqa: E402
    NLP_MODES,
    format_model_report,
    resolve_models,
    resolve_nlp_use_model,
)
from pipeline.prototype import PrototypePipeline  # noqa: E402

_TEST_DIR = repo_root / "data" / "raw" / "asl_alphabet" / "asl_alphabet_test"


def _print_summary(result: dict) -> None:
    timings = result.get("timings") or {}
    print("\n" + "=" * 60)
    print("  ASL -> Multilingual Speech")
    print("=" * 60)
    print(f"  Mode     : {result.get('input_mode')}")
    print(f"  Backend  : {result.get('recognition_backend')}")
    print(f"  Gloss    : {result.get('gloss')!r}")
    print(f"  English  : {result.get('english')!r}")
    print(f"  Emotion  : {result.get('emotion')} "
          f"({result.get('emotion_confidence') or 0:.2f})")
    for lang, info in (result.get("speech") or {}).items():
        cached = " (cached)" if info.get("cached") else ""
        print(f"  Speech[{lang}]: {info.get('text')!r} -> {info.get('audio_path')}{cached}")
    print(
        "  Latency  : "
        f"total {timings.get('total_ms', 0):.0f} ms "
        f"(nlp {timings.get('nlp_ms', 0):.0f}, "
        f"emotion {timings.get('emotion_ms', 0):.0f}, "
        f"tts {timings.get('tts_ms', 0):.0f})"
    )
    evaluation = result.get("evaluation") or {}
    if evaluation.get("log_path"):
        print(f"  Eval log : {evaluation['log_path']}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline with the best available models",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--webcam", action="store_true",
                       help="Live webcam demo (default when no mode is given).")
    group.add_argument("--images", nargs="+", help="Ordered sign image paths.")
    group.add_argument("--spell", type=str, help="Spell a word using bundled test images.")
    group.add_argument("--gloss", type=str, help="Skip recognition; supply gloss text.")
    group.add_argument("--text", type=str, help="Skip recognition; supply plain text.")
    group.add_argument(
        "--wlasl-video",
        type=str,
        help="Word-level recognition: classify a saved video clip, then NLP+TTS.",
    )
    group.add_argument(
        "--wlasl",
        action="store_true",
        help="Live webcam word-level mode: press [r] to record each WLASL sign.",
    )

    parser.add_argument("--backend", choices=BACKENDS, default="auto",
                        help="Recognition backend (default: auto = best on disk).")
    parser.add_argument("--nlp", choices=NLP_MODES, default="auto",
                        help="auto = fine-tuned T5 when present, on = force, off = rules.")
    parser.add_argument("--languages", nargs="+", default=["en", "ne"])
    parser.add_argument("--face-image", type=str, default=None,
                        help="Image used for emotion in non-webcam modes.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hold-time", type=float, default=0.9,
                        help="Webcam only: seconds a sign must be held.")
    parser.add_argument("--show-models", action="store_true",
                        help="Print the resolved artefacts and exit.")
    parser.add_argument("--json", action="store_true",
                        help="Print the raw result dict (non-webcam modes).")
    args = parser.parse_args()

    languages = tuple(args.languages)
    selection = resolve_models(
        recognition_backend=args.backend,
        nlp_mode=args.nlp,
        languages=languages,
    )
    print(format_model_report(selection))

    if args.show_models:
        if args.json:
            print(json.dumps(selection.as_dict(), indent=2))
        return

    use_nlp_model = resolve_nlp_use_model(args.nlp)
    webcam = args.webcam or not any([
        args.images, args.spell, args.gloss, args.text,
        args.wlasl_video, args.wlasl,
    ])

    if args.wlasl:
        from scripts.realtime_demo import run_wlasl_webcam_demo

        run_wlasl_webcam_demo(
            camera=args.camera,
            use_nlp_model=use_nlp_model,
            languages=languages,
            show_model_report=False,
        )
        return

    if webcam:
        from scripts.realtime_demo import run_webcam_demo

        run_webcam_demo(
            camera=args.camera,
            backend=args.backend,
            hold_time=args.hold_time,
            use_nlp_model=use_nlp_model,
            languages=languages,
            show_model_report=False,
        )
        return

    if args.wlasl_video:
        from modules.recognition.wlasl_predictor import WLASLPredictor

        predictor = WLASLPredictor()
        pred = predictor.predict_clip(video_path=args.wlasl_video)
        print(
            f"WLASL prediction: {pred['label']!r} "
            f"({pred['confidence']:.2f}) backend={pred['backend']}"
        )
        if pred.get("error"):
            print(f"Error: {pred['error']}")
            sys.exit(1)
        # Feed the recognised gloss word into the NLP → TTS tail.
        inp = PipelineInput.from_gloss(pred["label"].upper(), face_image=args.face_image)
        print(
            "Running pipeline (loading NLP/emotion models, then TTS; "
            "cached audio skips Google network calls)...",
            flush=True,
        )
        with PrototypePipeline(
            use_nlp_model=use_nlp_model,
            recognition_backend="auto",
        ) as proto:
            result = proto.run(inp, languages=languages)
        result["wlasl_prediction"] = pred
        result["recognition_backend"] = "wlasl_video"
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            _print_summary(result)
        return

    if args.images:
        inp = PipelineInput.from_images(args.images, face_image=args.face_image)
    elif args.spell:
        inp = PipelineInput.from_spell(args.spell, _TEST_DIR, face_image=args.face_image)
        if not inp.image_paths:
            print(f"No test images found for {args.spell!r} in {_TEST_DIR}")
            sys.exit(1)
    elif args.gloss:
        inp = PipelineInput.from_gloss(args.gloss, face_image=args.face_image)
    else:
        inp = PipelineInput.from_text(args.text, face_image=args.face_image)

    print(
        "Running pipeline (loading NLP/emotion models, then TTS; "
        "cached audio skips Google network calls)...",
        flush=True,
    )
    with PrototypePipeline(
        use_nlp_model=use_nlp_model,
        recognition_backend=args.backend,
    ) as proto:
        result = proto.run(inp, languages=languages)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_summary(result)


if __name__ == "__main__":
    main()
