from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

# ASL gloss markers that should be stripped before building an English sentence.
# (e.g. "IX" = index/pointing, "DESC-" / "X-" = classifier/description prefixes)
_GLOSS_MARKERS = {"IX", "DESC", "CL", "FS", "POSS", "NEG"}

# Very small function-word list used by the rule-based fallback to glue glosses
# into something closer to English. This is intentionally simple; the T5 model
# does the real work when it is available.
_ARTICLES_BEFORE = {"boy", "girl", "man", "woman", "dog", "cat", "book", "car", "house"}

DEFAULT_LOCAL_MODEL = "models/nlp/t5_gloss_en"
DEFAULT_PRETRAINED = "vennify/t5-base-grammar-correction"


class GrammarCorrector:
    """Turns raw ASL gloss / spelled text into a grammatical English sentence.

    Backend priority when ``use_model=True``:

    1. **Local fine-tuned T5** at ``local_model_dir`` (ASLG-PC12 gloss→English),
       if the directory exists.
    2. **Pretrained grammar T5** (``model_name``), if downloadable.
    3. **Rule-based fallback:** lowercases, strips ASL gloss markers, collapses
       whitespace, capitalizes, and adds terminal punctuation.

    When ``use_model=False`` the rule-based path is always used so demos and
    tests stay offline and fast.
    """

    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_PRETRAINED,
        use_model: bool = True,
        local_model_dir: Optional[str] = DEFAULT_LOCAL_MODEL,
        max_length: int = 64,
    ):
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.max_length = max_length
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self.backend = "rule_based"
        if use_model:
            self._try_load_model()

    def _try_load_local(self) -> bool:
        if not self.local_model_dir:
            return False
        path = Path(self.local_model_dir)
        if not path.exists() or not (path / "config.json").exists():
            return False
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(path))
            self._model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
            self._model.eval()
            self.backend = "t5_gloss"
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[nlp] Local gloss T5 unavailable ({exc}); trying next backend.")
            self._tokenizer = None
            self._model = None
            return False

    def _try_load_pretrained(self) -> bool:
        if not self.model_name:
            return False
        # A local path may be passed as model_name after fine-tuning.
        candidate = Path(self.model_name)
        if candidate.exists() and (candidate / "config.json").exists():
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(str(candidate))
                self._model = AutoModelForSeq2SeqLM.from_pretrained(str(candidate))
                self._model.eval()
                self.backend = "t5_gloss"
                return True
            except Exception as exc:  # pragma: no cover
                print(f"[nlp] Local model at {candidate} failed ({exc}).")
                return False
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline("text2text-generation", model=self.model_name)
            self.backend = "t5"
            return True
        except Exception as exc:  # pragma: no cover - network/dependency dependent
            print(f"[nlp] T5 grammar model unavailable ({exc}); using rule-based fallback.")
            self._pipeline = None
            return False

    def _try_load_model(self) -> None:
        if self._try_load_local():
            return
        self._try_load_pretrained()

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

    def _generate_local(self, text: str) -> str:
        import torch

        from modules.nlp.data import format_source

        prompt = format_source(text)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def correct(self, text: str) -> str:
        """Correct a single gloss/text string into an English sentence."""
        if text is None or not text.strip():
            return ""

        if self._model is not None and self._tokenizer is not None:
            try:
                return self._generate_local(text)
            except Exception as exc:  # pragma: no cover
                print(f"[nlp] Local T5 inference failed ({exc}); using rule-based fallback.")

        if self._pipeline is not None:
            try:
                prompt = f"grammar: {text}"
                result = self._pipeline(prompt, max_length=self.max_length, num_beams=4)
                return result[0]["generated_text"].strip()
            except Exception as exc:  # pragma: no cover
                print(f"[nlp] T5 inference failed ({exc}); using rule-based fallback.")

        return self._rule_based(text)

    def correct_rule_based(self, text: str) -> str:
        """Always use the offline rule-based path (for baselines / BLEU)."""
        if text is None or not text.strip():
            return ""
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
