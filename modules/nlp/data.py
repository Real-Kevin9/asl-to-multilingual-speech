"""ASLG-PC12 gloss→English data loading and cleaning.

The Hugging Face subset ``achrafothman/aslg_pc12`` provides ~87k parallel pairs.
Raw strings often carry a BOM and a trailing newline; those are stripped here so
training and evaluation see the same cleaned text the corrector will produce.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HF_DATASET_ID = "achrafothman/aslg_pc12"
DEFAULT_RAW_DIR = "data/raw/aslg_pc12"
DEFAULT_PROCESSED_DIR = "data/processed/aslg_pc12"
TASK_PREFIX = "translate Gloss to English: "

_WHITESPACE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    """Normalise a single gloss or English string from ASLG-PC12."""
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\r", " ").replace("\n", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def clean_pair(gloss: str, text: str) -> Optional[Tuple[str, str]]:
    """Return a cleaned (gloss, english) pair, or None if either side is empty."""
    g = clean_text(gloss)
    t = clean_text(text)
    if not g or not t:
        return None
    return g, t


def download_aslg_pc12(
    raw_dir: str = DEFAULT_RAW_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    dataset_id: str = HF_DATASET_ID,
) -> Dict[str, object]:
    """Fetch ASLG-PC12 from Hugging Face and write cleaned JSONL splits.

    Returns a small stats dict for logging.
    """
    from datasets import load_dataset

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_id, split="train")
    pairs: List[Dict[str, str]] = []
    skipped = 0
    for row in ds:
        cleaned = clean_pair(row.get("gloss", ""), row.get("text", ""))
        if cleaned is None:
            skipped += 1
            continue
        gloss, text = cleaned
        pairs.append({"gloss": gloss, "text": text})

    # Persist the full cleaned corpus once; train/val are sliced later.
    all_path = processed_path / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Keep a tiny raw sample for inspection / offline smoke tests.
    sample_path = raw_path / "sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as fh:
        for pair in pairs[:20]:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    meta = {
        "dataset_id": dataset_id,
        "total_raw": len(ds),
        "total_cleaned": len(pairs),
        "skipped": skipped,
        "all_jsonl": str(all_path),
        "sample_jsonl": str(sample_path),
    }
    (processed_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_jsonl(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_aslg_pc12.py first."
        )
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_pairs(
    pairs: Sequence[Dict[str, str]],
    val_fraction: float = 0.1,
    seed: int = 42,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Deterministic train/val split with optional size caps for CPU training."""
    import random

    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)

    n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
    val = items[:n_val]
    train = items[n_val:]

    if max_train is not None:
        train = train[:max_train]
    if max_val is not None:
        val = val[:max_val]
    return train, val


def write_split_jsonl(
    train: Sequence[Dict[str, str]],
    val: Sequence[Dict[str, str]],
    processed_dir: str = DEFAULT_PROCESSED_DIR,
) -> Tuple[Path, Path]:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    train_path = processed_path / "train.jsonl"
    val_path = processed_path / "val.jsonl"
    for path, rows in ((train_path, train), (val_path, val)):
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_path, val_path


def prepare_splits(
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    val_fraction: float = 0.1,
    seed: int = 42,
    max_train: Optional[int] = 20000,
    max_val: Optional[int] = 2000,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Path, Path]:
    """Load cleaned corpus, slice train/val, and write split JSONL files."""
    all_path = Path(processed_dir) / "all.jsonl"
    pairs = load_jsonl(all_path)
    train, val = split_pairs(
        pairs,
        val_fraction=val_fraction,
        seed=seed,
        max_train=max_train,
        max_val=max_val,
    )
    train_path, val_path = write_split_jsonl(train, val, processed_dir=processed_dir)
    return train, val, train_path, val_path


def format_source(gloss: str) -> str:
    """T5 text-to-text source string for a gloss."""
    return f"{TASK_PREFIX}{clean_text(gloss)}"
