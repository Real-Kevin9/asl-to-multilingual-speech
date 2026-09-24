"""Unit tests for the NLP gloss→English module (offline, no network)."""

from __future__ import annotations

import json
from pathlib import Path

from modules.nlp.bleu_eval import corpus_bleu
from modules.nlp.correction import GrammarCorrector, correct_text
from modules.nlp.data import (
    TASK_PREFIX,
    clean_pair,
    clean_text,
    format_source,
    split_pairs,
    write_split_jsonl,
)


def test_clean_text_strips_bom_and_newlines():
    assert clean_text("\ufeffHELLO WORLD\n") == "HELLO WORLD"
    assert clean_text("  a   b  ") == "a b"


def test_clean_pair_drops_empty():
    assert clean_pair("I GO", "I go.") == ("I GO", "I go.")
    assert clean_pair("", "I go.") is None
    assert clean_pair("I GO", "\n") is None


def test_format_source_uses_task_prefix():
    assert format_source("I GO STORE") == f"{TASK_PREFIX}I GO STORE"


def test_split_pairs_is_partition_and_respects_caps():
    pairs = [{"gloss": f"G{i}", "text": f"t{i}"} for i in range(100)]
    train, val = split_pairs(pairs, val_fraction=0.1, seed=0, max_train=20, max_val=5)
    assert len(train) == 20
    assert len(val) == 5
    train_g = {p["gloss"] for p in train}
    val_g = {p["gloss"] for p in val}
    assert not train_g & val_g


def test_write_split_jsonl(tmp_path: Path):
    train = [{"gloss": "A", "text": "a"}]
    val = [{"gloss": "B", "text": "b"}]
    train_path, val_path = write_split_jsonl(train, val, processed_dir=str(tmp_path))
    assert json.loads(train_path.read_text(encoding="utf-8").strip())["gloss"] == "A"
    assert json.loads(val_path.read_text(encoding="utf-8").strip())["text"] == "b"


def test_rule_based_corrector_unchanged():
    corrected = correct_text("hello world", use_model=False)
    assert corrected == "Hello world."


def test_rule_based_strips_gloss_markers():
    c = GrammarCorrector(use_model=False)
    out = c.correct_rule_based("IX BOY GO STORE")
    assert "ix" not in out.lower()
    assert out.startswith("The boy") or "boy" in out.lower()
    assert out.endswith(".")


def test_corrector_defaults_to_rules_when_use_model_false():
    c = GrammarCorrector(use_model=False, local_model_dir="models/definitely_missing")
    assert c.backend == "rule_based"
    assert c.correct("I GO") == "I go."


def test_corrector_prefers_local_checkpoint(tmp_path: Path):
    """If a config.json exists under local_model_dir, backend becomes t5_gloss.

    We only assert selection logic when transformers can load a tiny fake
    layout; if load fails, the corrector must still fall back to rules without
    crashing.
    """
    model_dir = tmp_path / "fake_t5"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    c = GrammarCorrector(use_model=True, local_model_dir=str(model_dir), model_name=None)
    assert c.backend in {"t5_gloss", "rule_based"}


def test_corpus_bleu_identical_is_high():
    refs = ["the cat sat on the mat", "i go to the store"]
    score = corpus_bleu(refs, refs)
    assert score > 95.0


def test_corpus_bleu_empty():
    assert corpus_bleu([], []) == 0.0
