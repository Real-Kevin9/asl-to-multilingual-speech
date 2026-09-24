from modules.nlp.correction import correct_text
from modules.tts.synthesizer import MultilingualSynthesizer


def test_grammar_correction_rule_based():
    # Rule-based backend: cleans, capitalizes, and punctuates.
    corrected = correct_text("hello world", use_model=False)
    assert corrected == "Hello world."


def test_grammar_correction_empty():
    assert correct_text("", use_model=False) == ""


def test_tts_returns_structured_result(tmp_path):
    synth = MultilingualSynthesizer(output_dir=str(tmp_path))
    result = synth.synthesize("Hello world.", prosody={"rate": 1.0, "pitch": 0.0})
    # Both languages should be present with text + prosody, even offline.
    assert set(result.keys()) == {"en", "ne"}
    assert result["en"]["text"] == "Hello world."
    assert "prosody" in result["en"]
    assert "cached" in result["en"]
    assert "duration_ms" in result["en"]
