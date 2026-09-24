from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, TypeVar

T = TypeVar("T")

# Google Translate / gTTS can block for many minutes on slow or filtered networks.
DEFAULT_NETWORK_TIMEOUT_SEC = float(os.environ.get("TTS_NETWORK_TIMEOUT", "15"))

# Used when Google Translate is unreachable so Nepali TTS still has real text.
_OFFLINE_NEPALI: Dict[str, str] = {
    "hello": "नमस्कार",
    "hello.": "नमस्कार।",
    "hello world.": "नमस्ते संसार।",
    "game": "खेल",
    "game.": "खेल।",
    "how are you": "तपाईलाई कस्तो छ",
    "how are you.": "तपाईलाई कस्तो छ।",
    "how are you?": "तपाईलाई कस्तो छ?",
    "hello, how are you?": "नमस्ते, तपाईलाई कस्तो छ?",
    "i go store.": "म पसल जान्छु।",
    "i go to the store.": "म पसल जान्छु।",
    "i go toilet.": "म शौचालय जान्छु।",
    "i go to toilet.": "म शौचालय जान्छु।",
    "i go to the toilet.": "म शौचालय जान्छु।",
    "warm up.": "वार्म अप गर्नुहोस्।",
}

# Proposal latency target for end-to-end pipeline (ms).
LATENCY_TARGET_MS = 1500


class MultilingualSynthesizer:
    """Text-to-speech for English and Nepali with emotion-aware prosody.

    Pipeline per objective 3 of the proposal:

        corrected English  --(translate)-->  Nepali text
                 |                                  |
              gTTS(en)                          gTTS(ne)
                 |                                  |
             en.mp3                             ne.mp3

    Performance notes:
    - **Disk cache** avoids repeat gTTS/network calls for identical text, and
      caches Nepali translations so a repeated utterance needs no network at all.
    - **Parallel synthesis** runs English and Nepali gTTS concurrently.
    - A reused ``GoogleTranslator`` instance cuts per-call setup cost.

    Emotion prosody: ``rate``/``pitch`` multipliers come from the emotion
    module. gTTS only exposes a coarse ``slow`` flag, so ``rate < 0.9`` maps to
    slow speech.
    """

    def __init__(
        self,
        output_dir: str = "logs/tts",
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        parallel: bool = True,
        network_timeout: float = DEFAULT_NETWORK_TIMEOUT_SEC,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir or self.output_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.parallel = parallel
        self.network_timeout = max(0.0, float(network_timeout))
        self._translator = None
        self._network_ok = True
        self._espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
        self._translation_cache_path = self.cache_dir / "translations.json"
        self._translations: Dict[str, str] = self._load_translations()

    @staticmethod
    def _slug(text: str, limit: int = 30) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
        return (slug[:limit] or "utterance")

    @staticmethod
    def _slow_flag(prosody: Optional[Dict[str, float]]) -> bool:
        return bool(prosody and prosody.get("rate", 1.0) < 0.9)

    def _cache_path(self, text: str, lang: str, slow: bool) -> Path:
        key = hashlib.sha256(f"{lang}|{int(slow)}|{text}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{lang}_{key}.mp3"

    @staticmethod
    def _valid_audio(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    def _legacy_output_path(self, english_text: str, lang: str) -> Path:
        return self.output_dir / f"{self._slug(english_text)}_{lang}.mp3"

    def _call_with_timeout(self, fn: Callable[[], T], label: str) -> Optional[T]:
        if not self._network_ok:
            return None
        if self.network_timeout <= 0:
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[tts] {label} unavailable ({exc}).")
                self._network_ok = False
                return None

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn)
        try:
            return future.result(timeout=self.network_timeout)
        except FuturesTimeoutError:
            print(
                f"[tts] {label} timed out after {self.network_timeout:.0f}s; "
                "using offline fallback."
            )
            self._network_ok = False
            return None
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[tts] {label} unavailable ({exc}).")
            self._network_ok = False
            return None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _lookup_translation(self, text: str) -> Optional[str]:
        if text in self._translations:
            return self._translations[text]
        key = text.strip().lower()
        for stored, translated in self._translations.items():
            if stored.strip().lower() == key:
                return translated
        return _OFFLINE_NEPALI.get(key)

    def _get_translator(self):
        if self._translator is None:
            from deep_translator import GoogleTranslator  # type: ignore

            self._translator = GoogleTranslator(source="en", target="ne")
        return self._translator

    def _load_translations(self) -> Dict[str, str]:
        if not self.use_cache or not self._translation_cache_path.exists():
            return {}
        try:
            data = json.loads(self._translation_cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _store_translation(self, text: str, translated: str) -> None:
        if not self.use_cache:
            return
        self._translations[text] = translated
        try:
            self._translation_cache_path.write_text(
                json.dumps(self._translations, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:  # pragma: no cover - disk dependent
            pass

    def translate_to_nepali(self, text: str) -> str:
        """Translate English text to Nepali; returns original text on failure.

        Translations are cached alongside the audio: repeating an utterance
        should not pay for a second network round-trip.
        """
        cached = self._lookup_translation(text) if self.use_cache else _OFFLINE_NEPALI.get(text.strip().lower())
        if cached:
            return cached

        from modules.tts.wlasl_nepali_glossary import translate_wlasl_phrase

        gloss_ne = translate_wlasl_phrase(text)
        if gloss_ne:
            self._store_translation(text, gloss_ne)
            return gloss_ne

        def _translate() -> str:
            return self._get_translator().translate(text)

        translated = self._call_with_timeout(_translate, "Nepali translation")
        if not translated:
            return text
        self._store_translation(text, translated)
        return translated

    def _espeak(
        self,
        text: str,
        lang: str,
        out_path: Path,
        slow: bool,
    ) -> bool:
        if not self._espeak_bin:
            return False
        voice = "ne" if lang == "ne" else "en"
        speed = "120" if slow else "175"
        try:
            completed = subprocess.run(
                [self._espeak_bin, "-v", voice, "-s", speed, "-w", str(out_path), text],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
            return False
        if completed.returncode != 0 or not self._valid_audio(out_path):
            return False
        print(f"[tts] {lang}: used local {Path(self._espeak_bin).name} (Google TTS unavailable).")
        return True

    def _gtts(
        self,
        text: str,
        lang: str,
        prosody: Optional[Dict[str, float]],
        out_path: Path,
    ) -> Tuple[Optional[str], bool, float]:
        """Return (audio_path, cached, duration_ms)."""
        slow = self._slow_flag(prosody)
        cache_path = self._cache_path(text, lang, slow)

        if self.use_cache and self._valid_audio(cache_path):
            if out_path != cache_path:
                out_path.write_bytes(cache_path.read_bytes())
            return str(out_path), True, 0.0

        if self.use_cache and out_path != cache_path and self._valid_audio(out_path):
            if not self._valid_audio(cache_path):
                cache_path.write_bytes(out_path.read_bytes())
            return str(out_path), True, 0.0

        t0 = time.perf_counter()

        def _synthesize() -> None:
            from gtts import gTTS  # type: ignore

            gTTS(text=text, lang=lang, slow=slow).save(str(cache_path))

        self._call_with_timeout(_synthesize, f"gTTS ({lang})")
        if self._valid_audio(cache_path):
            if out_path != cache_path:
                out_path.write_bytes(cache_path.read_bytes())
            ms = (time.perf_counter() - t0) * 1000
            return str(out_path), False, ms

        wav_path = out_path.with_suffix(".wav")
        if self._espeak(text, lang, wav_path, slow):
            ms = (time.perf_counter() - t0) * 1000
            return str(wav_path), False, ms

        txt_path = out_path.with_suffix(".txt")
        txt_path.write_text(text, encoding="utf-8")
        return str(txt_path), False, (time.perf_counter() - t0) * 1000

    def _synthesize_language(
        self,
        text: str,
        lang: str,
        prosody: Optional[Dict[str, float]],
        out_path: Path,
    ) -> Dict[str, object]:
        audio_path, cached, duration_ms = self._gtts(text, lang, prosody, out_path)
        audio_format = Path(audio_path).suffix.lower().lstrip(".") if audio_path else ""
        return {
            "text": text,
            "audio_path": audio_path,
            "audio_format": audio_format,
            "prosody": prosody or {"rate": 1.0, "pitch": 0.0},
            "cached": cached,
            "duration_ms": round(duration_ms, 2),
        }

    def synthesize(
        self,
        english_text: str,
        prosody: Optional[Dict[str, float]] = None,
        languages=("en", "ne"),
        return_timings: bool = False,
    ):
        """Synthesize speech for the given text in the requested languages.

        Returns a mapping like::

            {
              "en": {"text": ..., "audio_path": ..., "prosody": {...},
                     "cached": bool, "duration_ms": float},
              "ne": {...},
            }

        When ``return_timings=True``, returns ``(results, timings)`` where
        ``timings`` includes ``translation_ms``, ``en_ms``, ``ne_ms``,
        ``total_ms``, and ``cache_hits``.
        """
        prosody = prosody or {"rate": 1.0, "pitch": 0.0}
        # Allow a fresh network attempt for each utterance (CPU-heavy WLASL runs
        # can cause spurious gTTS timeouts that must not block the next finish).
        self._network_ok = True
        base = self._slug(english_text)
        results: Dict[str, Dict[str, object]] = {}
        timings: Dict[str, object] = {
            "translation_ms": 0.0,
            "en_ms": 0.0,
            "ne_ms": 0.0,
            "total_ms": 0.0,
            "cache_hits": 0,
        }
        t_total = time.perf_counter()

        nepali_text = None
        if "ne" in languages:
            if self.use_cache and self._valid_audio(self._legacy_output_path(english_text, "ne")):
                nepali_text = self._translations.get(english_text, english_text)
                timings["translation_ms"] = 0.0
            else:
                t_tr = time.perf_counter()
                nepali_text = self.translate_to_nepali(english_text)
                timings["translation_ms"] = round((time.perf_counter() - t_tr) * 1000, 2)

        def _en():
            path = self.output_dir / f"{base}_en.mp3"
            return "en", self._synthesize_language(english_text, "en", prosody, path)

        def _ne():
            path = self.output_dir / f"{base}_ne.mp3"
            return "ne", self._synthesize_language(nepali_text or english_text, "ne", prosody, path)

        jobs = []
        if "en" in languages:
            jobs.append(_en)
        if "ne" in languages:
            jobs.append(_ne)

        if self.parallel and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=2) as pool:
                for lang, info in pool.map(lambda fn: fn(), jobs):
                    results[lang] = info
        else:
            for fn in jobs:
                lang, info = fn()
                results[lang] = info

        for lang in results:
            if results[lang].get("cached"):
                timings["cache_hits"] = int(timings["cache_hits"]) + 1
            ms = float(results[lang].get("duration_ms") or 0.0)
            timings[f"{lang}_ms"] = ms

        timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)

        if return_timings:
            return results, timings
        return results


_default_synth: Optional[MultilingualSynthesizer] = None


def synthesize(text: str, prosody: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, object]]:
    """Module-level convenience wrapper returning multilingual TTS results."""
    global _default_synth
    if _default_synth is None:
        _default_synth = MultilingualSynthesizer()
    return _default_synth.synthesize(text, prosody=prosody)
