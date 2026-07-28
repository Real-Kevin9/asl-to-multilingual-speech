import unittest

from pipeline.pipeline import assemble_text, run_demo


class PipelineSmokeTest(unittest.TestCase):
    def test_run_demo_returns_expected_summary(self):
        result = run_demo("hello")
        self.assertIn("hello", result["input_text"])
        self.assertEqual(result["status"], "ready")
        self.assertIn("recognition", result["steps"])
        self.assertIn("tts", result["steps"])
        self.assertIn("english", result)
        self.assertIn("speech", result)

    def test_assemble_text_letters(self):
        labels = ["H", "H", "E", "L", "L", "O"]
        # Debounce collapses the repeated H and the double L into single chars.
        self.assertEqual(assemble_text(labels), "HELO")

    def test_assemble_text_control_tokens(self):
        labels = ["C", "A", "T", "space", "D", "O", "G"]
        self.assertEqual(assemble_text(labels), "CAT DOG")

    def test_assemble_text_delete(self):
        labels = ["C", "A", "B", "del", "T"]
        self.assertEqual(assemble_text(labels), "CAT")


if __name__ == "__main__":
    unittest.main()
