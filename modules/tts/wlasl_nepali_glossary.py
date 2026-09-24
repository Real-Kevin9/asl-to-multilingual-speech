"""Offline Nepali gloss strings for the live WLASL 50-class vocabulary.

Used when Google Translate is rate-limited so word-mode ``Finish & speak`` can
still produce Nepali audio without network access.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# English gloss (lowercase) -> Nepali text for TTS.
WLASL_WORD_NEPALI: Dict[str, str] = {
    "before": "अघि",
    "cool": "राम्रो",
    "thin": "पातलो",
    "drink": "पिउनु",
    "go": "जानु",
    "computer": "कम्प्युटर",
    "cousin": "कजिन",
    "help": "मद्दत",
    "who": "को",
    "accident": "दुर्घटना",
    "bed": "ओछ्यान",
    "bowling": "बोलिङ",
    "candy": "क्यान्डी",
    "short": "छोटो",
    "tall": "अग्लो",
    "thanksgiving": "थ्याङ्क्सगिभिङ",
    "trade": "व्यापार",
    "basketball": "बास्केटबल",
    "call": "फोन गर्नु",
    "change": "परिवर्तन",
    "cold": "चिसो",
    "corn": "मकै",
    "dark": "अँध्यारो",
    "last": "अन्तिम",
    "later": "पछि",
    "man": "मान्छे",
    "pizza": "पिज्जा",
    "shirt": "शर्ट",
    "what": "के",
    "yes": "हो",
    "apple": "स्याउ",
    "bar": "बार",
    "brother": "दाजु",
    "champion": "च्याम्पियन",
    "check": "जाँच",
    "deaf": "बधिर",
    "delay": "ढिलाइ",
    "dog": "कुकुर",
    "environment": "वातावरण",
    "example": "उदाहरण",
    "family": "परिवार",
    "far": "टाढा",
    "laugh": "हाँसो",
    "leave": "छोड्नु",
    "letter": "पत्र",
    "mother": "आमा",
    "no": "होइन",
    "play": "खेल्नु",
    "room": "कोठा",
    "score": "स्कोर",
}

_WORD = re.compile(r"[A-Za-z']+")


def translate_wlasl_phrase(english: str) -> Optional[str]:
    """Map a short English sentence built from WLASL glosses to Nepali."""
    words = [w.lower() for w in _WORD.findall(english or "")]
    if not words:
        return None
    parts = []
    for w in words:
        ne = WLASL_WORD_NEPALI.get(w)
        if not ne:
            return None
        parts.append(ne)
    text = " ".join(parts)
    if not text.endswith("।"):
        text += "।"
    return text
