from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class PipelineInput:
    """Normalized input for the prototype pipeline.

    Supports four modes aligned with project inputs:
      * ``images`` — ordered sign-frame image paths (webcam capture or files)
      * ``spell``  — spell a word using bundled test images
      * ``gloss``  — skip recognition; start from raw gloss text
      * ``text``   — skip recognition + gloss; start from plain text (NLP→TTS demo)
    """

    mode: str
    image_paths: List[str] = field(default_factory=list)
    gloss: Optional[str] = None
    text: Optional[str] = None
    face_image: Optional[str] = None

    @classmethod
    def from_images(cls, paths: Sequence[str], face_image: Optional[str] = None) -> "PipelineInput":
        return cls(mode="images", image_paths=list(paths), face_image=face_image)

    @classmethod
    def from_spell(cls, word: str, test_dir: Path, face_image: Optional[str] = None) -> "PipelineInput":
        paths: List[str] = []
        for ch in word.upper():
            candidate = test_dir / f"{ch}_test.jpg"
            if candidate.exists():
                paths.append(str(candidate))
        return cls(mode="spell", image_paths=paths, gloss=word.upper(), face_image=face_image)

    @classmethod
    def from_gloss(cls, gloss: str, face_image: Optional[str] = None) -> "PipelineInput":
        return cls(mode="gloss", gloss=gloss.strip(), face_image=face_image)

    @classmethod
    def from_text(cls, text: str, face_image: Optional[str] = None) -> "PipelineInput":
        return cls(mode="text", text=text.strip(), face_image=face_image)
