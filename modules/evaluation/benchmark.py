from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

from modules.tts.synthesizer import LATENCY_TARGET_MS, MultilingualSynthesizer

DEFAULT_LATENCY_PHRASES: Sequence[str] = (
    "Hello world.",
    "I go to the store.",
    "How are you today?",
)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def _summarize_ms(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "n": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def benchmark_tts(
    phrases: Sequence[str] = DEFAULT_LATENCY_PHRASES,
    output_dir: str = "logs/tts/benchmark",
    prosody: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Measure TTS latency with cold (no cache) and warm (cached) passes."""
    prosody = prosody or {"rate": 1.0, "pitch": 0.0}
    cold_synth = MultilingualSynthesizer(
        output_dir=output_dir,
        cache_dir=f"{output_dir}/cache_cold",
        use_cache=False,
        parallel=True,
    )
    warm_synth = MultilingualSynthesizer(
        output_dir=output_dir,
        cache_dir=f"{output_dir}/cache_warm",
        use_cache=True,
        parallel=True,
    )

    cold_totals: List[float] = []
    first_totals: List[float] = []
    repeat_totals: List[float] = []
    repeat_hits = 0

    for text in phrases:
        _, cold_timings = cold_synth.synthesize(
            text, prosody=prosody, return_timings=True,
        )
        cold_totals.append(float(cold_timings["total_ms"]))

        # First cached call still pays for synthesis; it only fills the cache.
        _, first_timings = warm_synth.synthesize(
            text, prosody=prosody, return_timings=True,
        )
        first_totals.append(float(first_timings["total_ms"]))

        # Second call is the number that matters for a repeated utterance.
        _, repeat_timings = warm_synth.synthesize(
            text, prosody=prosody, return_timings=True,
        )
        repeat_totals.append(float(repeat_timings["total_ms"]))
        repeat_hits += int(repeat_timings.get("cache_hits") or 0)

    return {
        "phrases": list(phrases),
        "cold": _summarize_ms(cold_totals),
        "warm_first": _summarize_ms(first_totals),
        "warm_repeat": _summarize_ms(repeat_totals),
        "warm_cache_hits": repeat_hits,
        "note": (
            "cold = caching disabled; warm_first = cache miss that fills the cache; "
            "warm_repeat = same phrase served from cache."
        ),
        "latency_target_ms": LATENCY_TARGET_MS,
    }


def benchmark_gloss_pipeline(
    glosses: Sequence[str] = ("HELLO", "I GO STORE", "HOW ARE YOU"),
    use_nlp_model: bool = False,
    languages: Sequence[str] = ("en", "ne"),
    warmup: bool = True,
) -> Dict[str, Any]:
    """Benchmark NLP → emotion → TTS tail latency (recognition skipped).

    A warm-up run is timed separately: the first call pays for importing
    TensorFlow and loading the T5/CNN weights, which is a one-off startup cost
    rather than per-utterance latency.
    """
    from pipeline.input import PipelineInput
    from pipeline.prototype import PrototypePipeline

    totals: List[float] = []
    nlp_ms: List[float] = []
    emotion_ms: List[float] = []
    tts_ms: List[float] = []
    per_run: List[Dict[str, Any]] = []
    warmup_ms: Optional[float] = None

    with PrototypePipeline(use_nlp_model=use_nlp_model) as proto:
        if warmup:
            t_warm = time.perf_counter()
            proto.run(PipelineInput.from_gloss("WARM UP"), languages=languages)
            warmup_ms = round((time.perf_counter() - t_warm) * 1000, 2)

        for gloss in glosses:
            t0 = time.perf_counter()
            result = proto.run(PipelineInput.from_gloss(gloss), languages=languages)
            wall = (time.perf_counter() - t0) * 1000
            timings = result.get("timings") or {}
            totals.append(wall)
            nlp_ms.append(float(timings.get("nlp_ms") or 0.0))
            emotion_ms.append(float(timings.get("emotion_ms") or 0.0))
            tts_ms.append(float(timings.get("tts_ms") or 0.0))
            per_run.append({
                "gloss": gloss,
                "english": result.get("english"),
                "total_ms": round(wall, 2),
                "nlp_ms": timings.get("nlp_ms"),
                "emotion_ms": timings.get("emotion_ms"),
                "tts_ms": timings.get("tts_ms"),
                "tts_cache_hits": timings.get("tts_cache_hits"),
            })

    summary = {
        "mode": "gloss",
        "use_nlp_model": use_nlp_model,
        "glosses": list(glosses),
        "warmup_ms": warmup_ms,
        "total": _summarize_ms(totals),
        "nlp": _summarize_ms(nlp_ms),
        "emotion": _summarize_ms(emotion_ms),
        "tts": _summarize_ms(tts_ms),
        "latency_target_ms": LATENCY_TARGET_MS,
        "latency_ok_p95": _percentile(totals, 95) < LATENCY_TARGET_MS,
        "runs": per_run,
    }
    return summary


def run_latency_benchmark(
    use_nlp_model: bool = False,
    tts_phrases: Sequence[str] = DEFAULT_LATENCY_PHRASES,
    glosses: Sequence[str] = ("HELLO", "I GO STORE", "HOW ARE YOU"),
) -> Dict[str, Any]:
    """Run TTS-only and gloss-pipeline latency benchmarks."""
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tts": benchmark_tts(phrases=tts_phrases),
        "gloss_pipeline": benchmark_gloss_pipeline(
            glosses=glosses,
            use_nlp_model=use_nlp_model,
        ),
        "latency_target_ms": LATENCY_TARGET_MS,
    }
