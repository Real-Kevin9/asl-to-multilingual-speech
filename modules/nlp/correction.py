from __future__ import annotations

import re
from typing import List, Optional

# ASL gloss markers that should be stripped before building an English sentence.
# (e.g. "IX" = index/pointing, "DESC-" / "X-" = classifier/description prefixes)
_GLOSS_MARKERS = {"IX", "DESC", "CL", "FS", "POSS", "NEG"}

# Very small function-word list used by the rule-based fallback to glue glosses
# into something closer to English. This is intentionally simple; the T5 model
# does the real work when it is available.
_ARTICLES_BEFORE = {"boy", "girl", "man", "woman", "dog", "cat", "book", "car", "house"}


class GrammarCorrector:
    """Turns raw ASL gloss / spelled text into a grammatical English sentence.

    Two backends:

    1. **Transformer (preferred):** a T5 text-to-text model fine-tuned for
       grammar correction. Loaded lazily; if ``transformers`` or the model
       weights are unavailable (e.g. offline), it degrades gracefully.
    2. **Rule-based fallback:** lowercases, strips ASL gloss markers, collapses
       whitespace, capitalizes, and adds terminal punctuation.
    """

    def __init__(self, model_name: Optional[str] = "vennify/t5-base-grammar-correction",
                 use_model: bool = True):
        self.model_name = model_name
        self._pipeline = None
        self.backend = "rule_based"
        if use_model and model_name:
            self._try_load_model()

    def _try_load_model(self) -> None:
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline("text2text-generation", model=self.model_name)
            self.backend = "t5"
        except Exception as exc:  # pragma: no cover - network/dependency dependent
            print(f"[nlp] T5 grammar model unavailable ({exc}); using rule-based fallback.")
            self._pipeline = None
            self.backend = "rule_based"

    def _rule_based(self, text: str) -> str:
        tokens = re.split(r"\s+", text.strip())
        cleaned: List[str] = []
        for tok in tokens:
            if not tok:
                continue
            base = tok.split("-")[0].upper()
            if base in _GLOSS_MARKERS:
                continue
            cleaned.append(tok.lower())

        if not cleaned:
            return ""

        # Naive article insertion for a slightly more natural sentence.
        expanded: List[str] = []
        for word in cleaned:
            if word in _ARTICLES_BEFORE:
                expanded.append("the")
            expanded.append(word)

        sentence = " ".join(expanded)
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".?!":
            sentence += "."
        return sentence

    def correct(self, text: str) -> str:
        """Correct a single gloss/text string into an English sentence."""
        if text is None or not text.strip():
            return ""

        if self._pipeline is not None:
            try:
                prompt = f"grammar: {text}"
                result = self._pipeline(prompt, max_length=64, num_beams=4)
                return result[0]["generated_text"].strip()
            except Exception as exc:  # pragma: no cover
                print(f"[nlp] T5 inference failed ({exc}); using rule-based fallback.")

        return self._rule_based(text)


_default_corrector: Optional[GrammarCorrector] = None


def correct_text(text: str, use_model: bool = False) -> str:
    """Module-level convenience wrapper.

    By default this uses the offline rule-based backend so it is fast and has no
    network dependency. Pass ``use_model=True`` to attempt the T5 backend.
    """
    global _default_corrector
    if _default_corrector is None or (use_model and _default_corrector.backend == "rule_based"):
        _default_corrector = GrammarCorrector(use_model=use_model)
    return _default_corrector.correct(text)
