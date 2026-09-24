"""End-to-end latency benchmark (TTS + gloss-pipeline tail)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from modules.evaluation.benchmark import run_latency_benchmark  # noqa: E402
from modules.tts.synthesizer import LATENCY_TARGET_MS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TTS and gloss-pipeline latency")
    parser.add_argument(
        "--output",
        default="logs/evaluation/latency_benchmark.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--use-nlp-model",
        action="store_true",
        help="Use fine-tuned T5 during gloss-pipeline timing (slower load, better text).",
    )
    args = parser.parse_args()

    report = run_latency_benchmark(use_nlp_model=args.use_nlp_model)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    tts = report["tts"]
    gloss = report["gloss_pipeline"]
    print(f"Wrote {out}")
    print(f"Latency target: {LATENCY_TARGET_MS} ms")
    print(
        f"TTS cold  mean/p95: {tts['cold']['mean_ms']:.1f} / {tts['cold']['p95_ms']:.1f} ms"
    )
    print(
        f"TTS miss  mean/p95: {tts['warm_first']['mean_ms']:.1f} / "
        f"{tts['warm_first']['p95_ms']:.1f} ms"
    )
    print(
        f"TTS cache mean/p95: {tts['warm_repeat']['mean_ms']:.1f} / "
        f"{tts['warm_repeat']['p95_ms']:.1f} ms "
        f"({tts['warm_cache_hits']} hits)"
    )
    if gloss.get("warmup_ms") is not None:
        print(f"Startup (first utterance): {gloss['warmup_ms']:.0f} ms")
    print(
        f"Gloss tail mean/p95: {gloss['total']['mean_ms']:.1f} / "
        f"{gloss['total']['p95_ms']:.1f} ms "
        f"(ok={gloss['latency_ok_p95']})"
    )


if __name__ == "__main__":
    main()
