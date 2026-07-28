import unittest

from pipeline.input import PipelineInput
from pipeline.prototype import PrototypePipeline


class PrototypePipelineTest(unittest.TestCase):
    def test_text_mode_runs_all_phases(self):
        with PrototypePipeline() as proto:
            result = proto.run(PipelineInput.from_text("hello"))

        phase_names = [p["phase"] for p in result["phases"]]
        self.assertEqual(
            phase_names,
            ["input", "preprocessing", "recognition", "nlp", "emotion", "tts", "evaluation"],
        )
        self.assertEqual(result["input_mode"], "text")
        self.assertIn("english", result)
        self.assertIn("evaluation", result)

    def test_gloss_mode_skips_recognition(self):
        with PrototypePipeline() as proto:
            result = proto.run(PipelineInput.from_gloss("HELLO"))

        rec_phase = next(p for p in result["phases"] if p["phase"] == "recognition")
        self.assertEqual(rec_phase["status"], "skipped")
        self.assertEqual(result["gloss"], "HELLO")


if __name__ == "__main__":
    unittest.main()
