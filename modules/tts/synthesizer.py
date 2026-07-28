from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


class MultilingualSynthesizer:
    """Text-to-speech for English and Nepali with emotion-aware prosody.

    Pipeline per objective 3 of the proposal:

        corrected English  --(translate)-->  Nepali text
                 |                                  |
              gTTS(en)                          gTTS(ne)
                 |                                  |
             en.mp3                             ne.mp3

    All external calls (translation, gTTS) are optional and degrade
    gracefully: if the network or a dependency is unavailable the synthesizer
    still returns structured metadata (and writes the text to disk) so the
    end-to-end pipeline never breaks.

    Emotion prosody: ``rate``/``pitch`` multipliers come from the emotion
    module. gTTS only exposes a coarse ``slow`` flag, so ``rate < 0.9`` maps to
    slow speech. ``pitch`` is carried through for backends (or post-processing)
    that support it.
    """

    def __init__(self, output_dir: str = "logs/tts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(text: str, limit: int = 30) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
        return (slug[:limit] or "utterance")

    def translate_to_nepali(self, text: str) -> str:
        """Translate English text to Nepali; returns original text on failure."""
        try:
            from deep_translator import GoogleTranslator  # type: ignore

            return GoogleTranslator(source="en", target="ne").translate(text)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[tts] Nepali translation unavailable ({exc}); keeping English text.")
            return text

    def _gtts(self, text: str, lang: str, prosody: Optional[Dict[str, float]],
              out_path: Path) -> Optional[str]:
        try:
            from gtts import gTTS  # type: ignore

            slow = bool(prosody and prosody.get("rate", 1.0) < 0.9)
            gTTS(text=text, lang=lang, slow=slow).save(str(out_path))
            return str(out_path)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[tts] gTTS ({lang}) unavailable ({exc}); wrote text only.")
            txt_path = out_path.with_suffix(".txt")
            txt_path.write_text(text, encoding="utf-8")
            return None

    def synthesize(
        self,
        english_text: str,
        prosody: Optional[Dict[str, float]] = None,
        languages=("en", "ne"),
    ) -> Dict[str, Dict[str, object]]:
        """Synthesize speech for the given text in the requested languages.

        Returns a mapping like::

            {
              "en": {"text": ..., "audio_path": ..., "prosody": {...}},
              "ne": {"text": ..., "audio_path": ..., "prosody": {...}},
            }
        """
        prosody = prosody or {"rate": 1.0, "pitch": 0.0}
        base = self._slug(english_text)
        results: Dict[str, Dict[str, object]] = {}

        if "en" in languages:
            en_path = self.output_dir / f"{base}_en.mp3"
            results["en"] = {
                "text": english_text,
                "audio_path": self._gtts(english_text, "en", prosody, en_path),
                "prosody": prosody,
            }

        if "ne" in languages:
            nepali_text = self.translate_to_nepali(english_text)
            ne_path = self.output_dir / f"{base}_ne.mp3"
            results["ne"] = {
                "text": nepali_text,
                "audio_path": self._gtts(nepali_text, "ne", prosody, ne_path),
                "prosody": prosody,
            }

        return results


_default_synth: Optional[MultilingualSynthesizer] = None


def synthesize(text: str, prosody: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, object]]:
    """Module-level convenience wrapper returning multilingual TTS results."""
    global _default_synth
    if _default_synth is None:
        _default_synth = MultilingualSynthesizer()
    return _default_synth.synthesize(text, prosody=prosody)
