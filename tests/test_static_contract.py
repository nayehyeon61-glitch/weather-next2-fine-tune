"""Lightweight checks that do not download WeatherNext weights."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StaticContractTest(unittest.TestCase):

    def test_fine_tune_script_parses(self):
        source = (ROOT / "fine_tune.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_required_weight_name_is_present(self):
        source = (ROOT / "fine_tune.py").read_text(encoding="utf-8")
        self.assertIn('Path("weather-me-fine_tune_weight.npz")', source)

    def test_weight_is_gitignored(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("weather-me-fine_tune_weight*.npz", ignored.splitlines())

    def test_model_metadata_is_saved_for_inference_adapter(self):
        source = (ROOT / "fine_tune.py").read_text(encoding="utf-8")
        self.assertIn('args.output.with_suffix(".metadata.json")', source)
        self.assertIn('"model_name": args.model_name', source)
        self.assertIn('"weathernext_release": "v0.3.0"', source)


if __name__ == "__main__":
    unittest.main()
