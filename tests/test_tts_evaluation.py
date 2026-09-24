import json
from pathlib import Path

from modules.evaluation.unified import build_unified_report, load_module_reports
from modules.tts.synthesizer import MultilingualSynthesizer


def test_tts_uses_legacy_output_without_network(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "hello_en.mp3").write_bytes(b"fake-en-audio")
    (out_dir / "hello_ne.mp3").write_bytes(b"fake-ne-audio")

    synth = MultilingualSynthesizer(
        output_dir=str(out_dir),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
        parallel=False,
        network_timeout=0.001,
    )
    result, timings = synth.synthesize("hello", return_timings=True)

    assert result["en"]["cached"] is True
    assert result["ne"]["cached"] is True
    assert float(timings["translation_ms"]) == 0.0
    assert int(timings["cache_hits"]) == 2


def test_tts_skips_later_network_calls_after_timeout(tmp_path, monkeypatch):
    synth = MultilingualSynthesizer(
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
        parallel=False,
        network_timeout=0.05,
    )
    calls = {"n": 0}

    class CountingTranslator:
        def translate(self, text):
            calls["n"] += 1
            import time

            time.sleep(1.0)
            return "slow"

    monkeypatch.setattr(synth, "_get_translator", lambda: CountingTranslator())
    assert synth.translate_to_nepali("first unseen phrase") == "first unseen phrase"
    assert synth.translate_to_nepali("second unseen phrase") == "second unseen phrase"
    assert calls["n"] == 1


def test_wlasl_glossary_translation():
    from modules.tts.wlasl_nepali_glossary import translate_wlasl_phrase

    ne = translate_wlasl_phrase("Help go.")
    assert ne is not None
    assert "मद्दत" in ne
    assert "जानु" in ne


def test_tts_wlasl_glossary_avoids_network(tmp_path, monkeypatch):
    synth = MultilingualSynthesizer(
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
        network_timeout=0.001,
    )

    def fail():
        raise AssertionError("network should not be used for WLASL glossary hits")

    monkeypatch.setattr(synth, "_get_translator", fail)
    assert "मद्दत" in synth.translate_to_nepali("Help go.")


def test_tts_offline_glossary_avoids_network(tmp_path, monkeypatch):
    synth = MultilingualSynthesizer(
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
        network_timeout=0.001,
    )

    def fail():
        raise AssertionError("network should not be used for glossary hits")

    monkeypatch.setattr(synth, "_get_translator", fail)
    assert synth.translate_to_nepali("I go store.") == "म पसल जान्छु।"
    assert synth.translate_to_nepali("hello") == "नमस्कार"


def test_tts_espeak_fallback_writes_wav(tmp_path, monkeypatch):
    synth = MultilingualSynthesizer(
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
        parallel=False,
        network_timeout=0.001,
    )
    synth._network_ok = False

    def fake_espeak(text, lang, out_path, slow):
        out_path.write_bytes(b"RIFF-fake-wav")
        return True

    monkeypatch.setattr(synth, "_espeak", fake_espeak)
    result = synth.synthesize("brand new phrase", languages=("en",))
    assert str(result["en"]["audio_path"]).endswith(".wav")
    assert Path(result["en"]["audio_path"]).read_bytes() == b"RIFF-fake-wav"


def test_tts_translate_times_out(tmp_path, monkeypatch):
    synth = MultilingualSynthesizer(
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
        parallel=False,
        network_timeout=0.05,
    )

    class SlowTranslator:
        def translate(self, text):
            import time

            time.sleep(1.0)
            return "slow"

    monkeypatch.setattr(synth, "_get_translator", lambda: SlowTranslator())
    t0 = __import__("time").perf_counter()
    out = synth.translate_to_nepali("timeout probe")
    elapsed = __import__("time").perf_counter() - t0

    assert out == "timeout probe"
    assert elapsed < 0.5


def test_tts_cache_hit_when_audio_cached(tmp_path):
    synth = MultilingualSynthesizer(
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
        parallel=False,
    )
    prosody = {"rate": 1.0, "pitch": 0.0}
    text = "Cache test."

    for lang in ("en", "ne"):
        cache_path = synth._cache_path(text, lang, slow=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"fake-mp3")

    result, timings = synth.synthesize(text, prosody=prosody, return_timings=True)

    assert result["en"]["text"] == text
    assert result["en"]["cached"] is True
    assert result["ne"]["cached"] is True
    assert int(timings["cache_hits"]) == 2
    assert float(timings["en_ms"]) == 0.0


def test_tts_cache_path_is_stable(tmp_path):
    synth = MultilingualSynthesizer(
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
    )
    p1 = synth._cache_path("Hello", "en", slow=False)
    p2 = synth._cache_path("Hello", "en", slow=False)
    assert p1 == p2
    assert p1.name.startswith("en_")


def test_unified_report_aggregates_existing_json(tmp_path):
    log_dir = tmp_path / "eval"
    log_dir.mkdir()
    (log_dir / "nlp_bleu.json").write_text(
        '{"n": 10, "rule_based": {"bleu": 19.0}, "t5_finetuned": {"bleu": 57.0}}',
        encoding="utf-8",
    )
    (log_dir / "latency_benchmark.json").write_text(
        '{"gloss_pipeline": {"total": {"p95_ms": 900}, "latency_ok_p95": true}}',
        encoding="utf-8",
    )

    bundle = load_module_reports(str(log_dir))
    assert "nlp" in bundle["reports"]
    assert "emotion" in bundle["missing"]

    report = build_unified_report(log_dir=str(log_dir))
    assert report["module_summary"]["nlp"]["t5_bleu"] == 57.0
    assert report["module_summary"]["latency"]["gloss_latency_ok_p95"] is True


def test_recognition_reports_are_keyed_by_backend(tmp_path):
    log_dir = tmp_path / "eval"
    log_dir.mkdir()
    # evaluate_recognizers.py writes accuracy per image group.
    (log_dir / "backend_comparison.json").write_text(
        json.dumps({
            "hybrid": {"accuracy": {"all": 0.92, "scene": 0.90, "closeup": 0.94},
                       "counts": {"all": 393}},
            "landmark_mlp": {"accuracy": {"all": 0.65, "scene": 0.91}, "counts": {"all": 393}},
        }),
        encoding="utf-8",
    )
    # evaluate_on_user_data.py writes a single float per backend.
    (log_dir / "user_data_accuracy.json").write_text(
        json.dumps({"hybrid": {"accuracy": 0.607, "total": 534}}),
        encoding="utf-8",
    )

    summary = build_unified_report(log_dir=str(log_dir))["module_summary"]

    assert summary["recognition"]["best_backend"] == "hybrid"
    assert summary["recognition"]["backends"]["hybrid"]["closeup_accuracy"] == 0.94
    assert summary["user_recognition"]["backends"]["hybrid"]["overall_accuracy"] == 0.607
    assert summary["user_recognition"]["backends"]["hybrid"]["n_samples"] == 534
